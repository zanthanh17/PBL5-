import time, threading, json, random, logging
import queue
import struct
import zlib
import base64
from pathlib import Path
import cv2, numpy as np, requests
from picamera2 import Picamera2
from collections import deque, Counter
from typing import Optional, Dict, List, Tuple
from tts_speaker import TTSSpeaker
from sensor_controller import SensorController

# ============= Recognition State Management =============
class RecognitionState:
    """Quản lý state của recognition workflow với sensor"""
    
    IDLE = "idle"
    PERSON_DETECTED = "person_detected"
    RECOGNIZING = "recognizing"
    RECOGNIZED = "recognized"
    RECOGNITION_FAILED = "recognition_failed"
    PERSON_LEFT = "person_left"
    COOLDOWN = "cooldown"  # Cooldown sau khi nhận diện thành công
    
    def __init__(self, max_retries: int = 3, timeout_sec: float = 15.0, cooldown_sec: float = 5.0):
        self.current_state = self.IDLE
        self.recognition_start_time = None
        self.retry_count = 0
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec
        self.cooldown_sec = cooldown_sec
        self.cooldown_start_time = None
        self.last_recognized_name = None
        self.has_spoken = False  # Flag để track đã nói TTS chưa
        self._lock = threading.Lock()
    
    def set_state(self, new_state: str):
        """Set state (thread-safe)"""
        with self._lock:
            self.current_state = new_state
            if new_state == self.RECOGNIZING:
                self.recognition_start_time = time.time()
            elif new_state == self.RECOGNIZED:
                self.retry_count = 0
                # Bắt đầu cooldown ngay khi nhận diện thành công
                self.cooldown_start_time = time.time()
                # Tự động chuyển sang COOLDOWN state
                self.current_state = self.COOLDOWN
                # Reset flag để cho phép nói TTS (chỉ 1 lần)
                self.has_spoken = False
            elif new_state == self.RECOGNITION_FAILED:
                self.retry_count += 1
            elif new_state == self.COOLDOWN:
                if self.cooldown_start_time is None:
                    self.cooldown_start_time = time.time()
    
    def get_state(self) -> str:
        """Get current state (thread-safe)"""
        with self._lock:
            return self.current_state
    
    def should_retry(self) -> bool:
        """Check if should retry recognition"""
        with self._lock:
            return self.retry_count < self.max_retries
    
    def is_timeout(self) -> bool:
        """Check if recognition timeout"""
        with self._lock:
            if self.recognition_start_time is None:
                return False
            return (time.time() - self.recognition_start_time) > self.timeout_sec
    
    def is_in_cooldown(self) -> bool:
        """
        Check if đang trong cooldown period
        
        Returns:
            True nếu đang trong cooldown, False nếu không
            Tự động chuyển về IDLE khi cooldown hết
        """
        with self._lock:
            if self.current_state == self.COOLDOWN:
                if self.cooldown_start_time is None:
                    # Không có start time → không trong cooldown
                    self.current_state = self.IDLE
                    return False
                
                elapsed = time.time() - self.cooldown_start_time
                if elapsed >= self.cooldown_sec:
                    # Cooldown hết → chuyển về IDLE và quay lại hoạt động bình thường
                    self.current_state = self.IDLE
                    self.cooldown_start_time = None
                    self.last_recognized_name = None  # Clear tên đã nhận diện
                    self.has_spoken = False  # Reset flag TTS
                    logger.info("Cooldown finished - returning to normal operation")
                    return False
                return True
            return False
    
    def get_cooldown_remaining(self) -> float:
        """
        Get thời gian còn lại của cooldown (giây)
        
        Returns:
            Thời gian còn lại (giây), hoặc 0 nếu không trong cooldown
        """
        with self._lock:
            if self.current_state == self.COOLDOWN and self.cooldown_start_time is not None:
                elapsed = time.time() - self.cooldown_start_time
                remaining = self.cooldown_sec - elapsed
                return max(0.0, remaining)
            return 0.0
    
    def reset(self):
        """Reset state"""
        with self._lock:
            self.current_state = self.IDLE
            self.recognition_start_time = None
            self.retry_count = 0
            self.cooldown_start_time = None
            self.last_recognized_name = None
            self.has_spoken = False
    
    def mark_spoken(self):
        """Đánh dấu đã nói TTS (chỉ gọi 1 lần)"""
        with self._lock:
            self.has_spoken = True
    
    def should_speak(self) -> bool:
        """Kiểm tra có nên nói TTS không (chưa nói và đang trong RECOGNIZED/COOLDOWN)"""
        with self._lock:
            if self.has_spoken:
                return False
            return self.current_state == self.RECOGNIZED or self.current_state == self.COOLDOWN

# Try import psutil for adaptive frame skip
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ============= Logging Setup =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============= Config & Utils =============
def get_project_root():
    """Lấy đường dẫn gốc của project (face-client/)"""
    # File này ở: face-client/src/client.py
    # Project root: face-client/
    current_file = Path(__file__).resolve()
    # current_file = /path/to/face-client/src/client.py
    # project_root = /path/to/face-client/
    project_root = current_file.parent.parent
    return project_root

def load_config(path=None):
    """
    Load config từ file YAML
    
    Args:
        path: Đường dẫn đến config file. Nếu None, dùng mặc định config/client.yaml
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError("Missing PyYAML. Install: pip install pyyaml")
    
    if path is None:
        # Dùng đường dẫn mặc định: config/client.yaml từ project root
        project_root = get_project_root()
        path = project_root / "config" / "client.yaml"
    else:
        # Nếu path là string, convert sang Path và resolve
        path = Path(path)
        if not path.is_absolute():
            # Nếu là đường dẫn tương đối, resolve từ project root
            project_root = get_project_root()
            path = project_root / path
    
    path = path.expanduser()  # Expand ~ trong đường dẫn
    with open(path, "r") as f:
        return yaml.safe_load(f)

def find_haarcascade():
    for p in [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ]:
        try:
            with open(p, "r"):
                return p
        except Exception:
            pass
    raise FileNotFoundError("Missing haarcascade_frontalface_default.xml (sudo apt install -y opencv-data)")

def calc_quality(face_bgr: np.ndarray) -> float:
    """Tính chất lượng ảnh (brightness + sharpness) - OPTIMIZED"""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    
    # OPTIMIZATION: Dùng Sobel thay vì Laplacian (nhanh hơn 2x)
    # Sobel gradient magnitude ≈ sharpness
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = float(np.mean(sobel_x**2 + sobel_y**2))
    
    b_norm = np.clip((brightness - 50) / (190 - 50), 0, 1)
    s_norm = np.clip((sharpness - 100) / (500 - 100), 0, 1)  # Adjusted threshold for Sobel
    return float(0.5 * b_norm + 0.5 * s_norm)

def calc_quality_fast(face_gray: np.ndarray) -> float:
    """Fast quality calculation từ gray image (tránh convert lại)"""
    brightness = float(np.mean(face_gray))
    
    # Sobel gradient
    sobel_x = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = float(np.mean(sobel_x**2 + sobel_y**2))
    
    b_norm = np.clip((brightness - 50) / (190 - 50), 0, 1)
    s_norm = np.clip((sharpness - 100) / (500 - 100), 0, 1)
    return float(0.5 * b_norm + 0.5 * s_norm)

def enhance_face_quality(face_bgr: np.ndarray) -> np.ndarray:
    """Enhance face image quality - FAST version cho Pi 3B+"""
    # Convert to LAB color space
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel (nhanh)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # BỎ denoising vì quá chậm trên Pi 3B+!
    # enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
    
    # Thay bằng Gaussian blur nhẹ (nhanh hơn nhiều)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return enhanced

def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    vec[np.isnan(vec)] = 0.0
    vec[np.isinf(vec)] = 0.0
    n = np.linalg.norm(vec) + 1e-9
    return vec / n

def make_embedding_fixed():
    # 128-d để test; sẽ pad lên 192-d khi cần
    return [0.02, 0.03, 0.04, 0.05, 0.06] + [0.0]*123

# ============= Multi-frame Voting =============
class VotingBuffer:
    """Buffer để vote results từ nhiều frames - tăng accuracy"""
    
    def __init__(self, window_size: int = 5, vote_threshold: int = 3):
        self.window_size = window_size
        self.vote_threshold = vote_threshold
        self.buffer = deque(maxlen=window_size)
        self.last_result = None
    
    def add(self, emp_id: Optional[str], score: float):
        """Thêm kết quả mới"""
        self.buffer.append((emp_id, score))
    
    def get_voted_result(self) -> Optional[tuple]:
        """
        Lấy kết quả sau khi vote
        Returns: (emp_id, avg_score) hoặc None
        """
        if len(self.buffer) < self.vote_threshold:
            return None
        
        # Đếm votes
        emp_ids = [emp_id for emp_id, _ in self.buffer if emp_id is not None]
        if not emp_ids:
            return None
        
        counter = Counter(emp_ids)
        most_common_emp, count = counter.most_common(1)[0]
        
        # Cần đủ votes
        if count < self.vote_threshold:
            return None
        
        # Tính average score của emp_id được vote nhiều nhất
        scores = [score for emp_id, score in self.buffer if emp_id == most_common_emp]
        avg_score = sum(scores) / len(scores)
        
        return (most_common_emp, avg_score)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()

# ============= Phase 1 Optimizations =============

# ============= Phase 2 Optimizations =============

class FaceTracker:
    """Track face position để giảm detection area (ROI Tracking)"""
    
    def __init__(self, decay: float = 0.9, min_confidence: float = 0.3, expand: float = 1.5):
        """
        Args:
            decay: Confidence decay rate khi không detect được face
            min_confidence: Confidence tối thiểu để dùng ROI
            expand: Hệ số mở rộng ROI (1.5 = 150% kích thước face)
        """
        self.last_bbox = None  # (x, y, w, h)
        self.confidence = 0.0
        self.decay = decay
        self.min_confidence = min_confidence
        self.expand = expand
        self.miss_count = 0  # Đếm số lần miss liên tiếp
    
    def update(self, bbox: Optional[Tuple[int, int, int, int]]):
        """Update tracked bbox"""
        if bbox:
            x, y, w, h = bbox
            self.last_bbox = (x, y, w, h)
            self.confidence = 1.0
            self.miss_count = 0
        else:
            # Không detect được → giảm confidence
            self.confidence *= self.decay
            self.miss_count += 1
    
    def get_roi(self, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get ROI để detect (chỉ detect trong vùng này)
        
        Args:
            frame_shape: (height, width) của frame
            
        Returns:
            (x, y, w, h) ROI hoặc None nếu không có track
        """
        if self.last_bbox is None:
            return None
        
        if self.confidence < self.min_confidence:
            return None
        
        # Nếu miss quá nhiều lần → reset
        if self.miss_count > 10:
            self.last_bbox = None
            self.confidence = 0.0
            return None
        
        x, y, w, h = self.last_bbox
        frame_h, frame_w = frame_shape
        
        # Expand ROI từ center
        cx, cy = x + w // 2, y + h // 2
        new_w = int(w * self.expand)
        new_h = int(h * self.expand)
        
        # Clamp to frame bounds
        x1 = max(0, cx - new_w // 2)
        y1 = max(0, cy - new_h // 2)
        x2 = min(frame_w, cx + new_w // 2)
        y2 = min(frame_h, cy + new_h // 2)
        
        return (x1, y1, x2 - x1, y2 - y1)
    
    def reset(self):
        """Reset tracker"""
        self.last_bbox = None
        self.confidence = 0.0
        self.miss_count = 0

# PHASE 2 OPTIMIZATION: Network Compression
def compress_embedding(emb: np.ndarray) -> bytes:
    """
    Compress embedding với quantization + zlib
    
    Args:
        emb: Embedding vector (192d, float32, normalized)
        
    Returns:
        Compressed bytes
    """
    # Quantize từ float32 -> int16 (giảm 50% size)
    # Scale: [-1, 1] -> [-32767, 32767]
    emb_int16 = (emb * 32767).astype(np.int16)
    
    # Pack binary
    packed = struct.pack(f'{len(emb_int16)}h', *emb_int16)
    
    # Compress với zlib (level=1 nhanh, level=6 cân bằng)
    compressed = zlib.compress(packed, level=1)
    
    return compressed

def decompress_embedding(data: bytes) -> np.ndarray:
    """
    Decompress embedding
    
    Args:
        data: Compressed bytes
        
    Returns:
        Embedding vector (192d, float32)
    """
    # Decompress
    unpacked = zlib.decompress(data)
    
    # Unpack binary
    emb_int16 = struct.unpack(f'{len(unpacked)//2}h', unpacked)
    
    # Convert back to float32
    emb = np.array(emb_int16, dtype=np.float32) / 32767.0
    
    return emb

class AdaptiveFrameSkip:
    """Tự động điều chỉnh frame skip dựa trên CPU load"""
    
    def __init__(self, base_skip: int = 3, max_skip: int = 5, min_skip: int = 1, 
                 check_interval: float = 2.0):
        self.base_skip = base_skip
        self.max_skip = max_skip
        self.min_skip = min_skip
        self.current_skip = base_skip
        self.check_interval = check_interval
        self.last_check = time.time()
        self.enabled = PSUTIL_AVAILABLE
    
    def get_skip(self) -> int:
        """Get current frame skip value"""
        if not self.enabled:
            return self.current_skip
        
        # Check CPU mỗi check_interval giây
        now = time.time()
        if now - self.last_check >= self.check_interval:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                if cpu_percent > 80:
                    # CPU cao → tăng skip
                    self.current_skip = min(self.max_skip, self.current_skip + 1)
                    logger.debug(f"CPU high ({cpu_percent:.1f}%) → skip={self.current_skip}")
                elif cpu_percent < 50:
                    # CPU thấp → giảm skip
                    self.current_skip = max(self.min_skip, self.current_skip - 1)
                    logger.debug(f"CPU low ({cpu_percent:.1f}%) → skip={self.current_skip}")
            except Exception as e:
                logger.warning(f"Error checking CPU: {e}")
            
            self.last_check = now
        
        return self.current_skip

class RecognitionWorker:
    """Background worker để xử lý recognition (non-blocking)"""
    
    def __init__(self, api_client, embedder, device_id, max_queue_size: int = 2):
        self.api_client = api_client
        self.embedder = embedder
        self.device_id = device_id
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.stats = {"processed": 0, "dropped": 0}
    
    def start(self):
        """Start worker thread"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("Recognition worker started")
    
    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Recognition worker stopped")
    
    def add_frame(self, face_crop: np.ndarray, quality: float) -> bool:
        """
        Add frame để xử lý (non-blocking)
        Returns: True nếu đã thêm vào queue, False nếu queue đầy
        """
        try:
            self.queue.put_nowait((face_crop.copy(), quality))
            return True
        except queue.Full:
            self.stats["dropped"] += 1
            return False
    
    def get_result(self) -> Optional[Dict]:
        """Get result (non-blocking)"""
        try:
            result = self.result_queue.get_nowait()
            return result
        except queue.Empty:
            return None
    
    def clear_queues(self):
        """Clear tất cả queues (dùng khi vào cooldown)"""
        # Clear input queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.stats["dropped"] += 1
            except queue.Empty:
                break
        
        # Clear result queue
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def _worker(self):
        """Worker thread loop"""
        while self.running:
            try:
                face_crop, quality = self.queue.get(timeout=0.1)
                
                # Compute embedding
                emb = normalize_embedding(self.embedder(face_crop))
                
                # API call
                resp = self.api_client.recognize(
                    self.device_id, emb.tolist(),
                    liveness=0.9, quality=quality
                )
                
                # Put result với timestamp
                self.result_queue.put_nowait({
                    "response": resp,
                    "timestamp": time.time()
                })
                
                self.stats["processed"] += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Recognition worker error: {e}", exc_info=True)
                # Put error result
                try:
                    self.result_queue.put_nowait({
                        "response": {"status": "error", "reason": str(e)},
                        "timestamp": time.time()
                    })
                except:
                    pass

# ============= API calls với Retry Logic =============
class APIClient:
    """API client với retry logic và timeout handling"""
    
    def __init__(self, base_url: str, timeout: int = 3, max_retries: int = 3, use_compression: bool = True):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_compression = use_compression  # PHASE 2: Network compression
        self.session = requests.Session()
        # Connection pooling để tăng tốc
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0  # Tự xử lý retry
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _retry_request(self, method, url, **kwargs):
        """Retry logic với exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
            except requests.exceptions.ConnectionError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1.0 * (2 ** attempt))
            except requests.exceptions.HTTPError as e:
                # Không retry cho 4xx errors
                if 400 <= e.response.status_code < 500:
                    raise
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))
    
    def recognize(self, device_id: str, emb: List[float], liveness: float, quality: float) -> Dict:
        """POST /v1/recognize"""
        # PHASE 2 OPTIMIZATION: Network compression
        if self.use_compression:
            emb_array = np.array(emb, dtype=np.float32)
            compressed = compress_embedding(emb_array)
            # Encode base64 để gửi qua JSON
            compressed_b64 = base64.b64encode(compressed).decode('ascii')
            
            payload = {
                "device_id": device_id,
                "embedding_compressed": compressed_b64,
                "liveness": float(liveness),
                "quality": float(quality),
                "options": {"save_event_face": False, "mode": "auto"},
            }
        else:
            # Fallback: uncompressed
            payload = {
                "device_id": device_id,
                "embedding": list(map(float, emb)),
                "liveness": float(liveness),
                "quality": float(quality),
                "options": {"save_event_face": False, "mode": "auto"},
            }
        
        response = self._retry_request("POST", f"{self.base_url}/v1/recognize", json=payload)
        return response.json()

def api_recognize(base_url, device_id, emb, liveness, quality, timeout):
    payload = {
        "device_id": device_id,
        "embedding": list(map(float, emb)),
        "liveness": float(liveness),
        "quality": float(quality),
        "options": {"save_event_face": False, "mode": "auto"},
    }
    r = requests.post(f"{base_url}/v1/recognize", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_poll_next_job(base_url, device_id, timeout=3):
    r = requests.get(f"{base_url}/v1/enroll_jobs/next",
                     params={"device_id": device_id}, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None

def api_job_start(base_url, job_id, timeout=3):
    try:
        requests.post(f"{base_url}/v1/enroll_jobs/{job_id}/start", timeout=timeout)
    except Exception:
        pass

def api_job_done(base_url, job_id, ok=True, notes="", timeout=5):
    try:
        requests.post(f"{base_url}/v1/enroll_jobs/{job_id}/done",
                      params={"ok": ok, "notes": notes}, timeout=timeout)
    except Exception:
        pass

def api_enroll(base_url, emp_id, embeddings_192d, timeout=15):
    payload = {"model": "mobilefacenet-192d", "embeddings": embeddings_192d}
    r = requests.post(f"{base_url}/v1/enroll/{emp_id}",
                      headers={"Content-Type": "application/json"},
                      data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()
    return r.json()

# ============= Main =============
def main():
    cfg = load_config()
    base_url  = cfg["server"]["base_url"]
    timeout   = cfg["server"].get("timeout_sec", 3)
    mode_cfg  = cfg["recognition"]["mode"]         # "real" | "fixed_test"
    model_path= cfg["recognition"].get("model_path")
    # Tối ưu cho Pi 3B+ - giảm resolution nếu quá lớn
    W, H = int(cfg["camera"]["width"]), int(cfg["camera"]["height"])
    if W > 480:  # Auto reduce cho Pi 3B+
        W, H = 480, 360
        logger.info(f"Reduced resolution to {W}x{H} for Pi 3B+ optimization")
    window    = cfg.get("ui", {}).get("window_name", "Face Client (Pi3B+)")
    show_fps  = bool(cfg.get("ui", {}).get("show_fps", True))
    device_id = cfg.get("device", {}).get("id", "pi-entrance-01")

    jobs_cfg       = cfg.get("jobs", {}) if isinstance(cfg, dict) else {}
    poll_min       = float(jobs_cfg.get("poll_min_sec", 1.0))
    poll_max       = float(jobs_cfg.get("poll_max_sec", 15.0))
    enroll_timeout = float(jobs_cfg.get("enroll_timeout_sec", 30.0))  # timeout không hoạt động

    # Initialize components
    use_compression = cfg.get("recognition", {}).get("network_compression", True)
    api_client = APIClient(base_url, timeout, use_compression=use_compression)
    cascade_path = find_haarcascade()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    embedder = None
    recognition_worker = None
    
    if mode_cfg == "real":
        from embedding_tflite import MobileFaceNetEmbedder
        tta_flip = cfg.get("recognition", {}).get("tta_flip", False)
        
        # Resolve model_path: nếu là đường dẫn tương đối, resolve từ project root
        if model_path:
            model_path = Path(model_path).expanduser()  # Expand ~
            if not model_path.is_absolute():
                project_root = get_project_root()
                model_path = project_root / model_path
            model_path = str(model_path.resolve())
        
        embedder = MobileFaceNetEmbedder(model_path, use_tta=tta_flip)   # 192d
        logger.info(f"Loaded MobileFaceNet embedder (TTA: {tta_flip}) from {model_path}")
        
        # PHASE 1 OPTIMIZATION: Initialize async recognition worker
        use_async_recognition = cfg.get("recognition", {}).get("async_recognition", True)
        if use_async_recognition:
            recognition_worker = RecognitionWorker(api_client, embedder, device_id, max_queue_size=2)
            recognition_worker.start()
            logger.info("Async recognition worker enabled")
    
    # Initialize TTS Speaker
    tts_cfg = cfg.get("tts", {})
    tts_enabled = tts_cfg.get("enabled", True)
    tts_volume = tts_cfg.get("volume", 100)
    tts_speed = tts_cfg.get("speed", 150)
    tts_cooldown = tts_cfg.get("cooldown", 3.0)
    tts = TTSSpeaker(enabled=tts_enabled, volume=tts_volume, speed=tts_speed, cooldown=tts_cooldown)
    logger.info(f"TTS Speaker initialized (enabled: {tts_enabled})")
    
    # Initialize Recognition State (tích hợp sensor + recognition)
    recog_cfg = cfg.get("recognition", {})
    max_retries = recog_cfg.get("max_retries", 3)
    recog_timeout = recog_cfg.get("recognition_timeout", 15.0)
    cooldown_sec = recog_cfg.get("cooldown_after_success", 5.0)  # Cooldown sau khi nhận diện thành công
    recognition_state = RecognitionState(max_retries=max_retries, timeout_sec=recog_timeout, cooldown_sec=cooldown_sec)
    
    # Initialize Sensor Controller (HC-SR04 + LED)
    sensor_cfg = cfg.get("sensor", {})
    sensor_enabled = sensor_cfg.get("enabled", False)
    sensor_trig_pin = sensor_cfg.get("trig_pin", 23)
    sensor_echo_pin = sensor_cfg.get("echo_pin", 24)
    sensor_led_pin = sensor_cfg.get("led_pin", 18)
    sensor_trigger_distance = sensor_cfg.get("trigger_distance", 100.0)
    sensor_led_duration = sensor_cfg.get("led_on_duration", 10.0)
    sensor_check_interval = sensor_cfg.get("check_interval", 0.2)
    turn_off_on_success = sensor_cfg.get("turn_off_on_success", True)
    
    sensor = None
    if sensor_enabled:
        sensor = SensorController(
            trig_pin=sensor_trig_pin,
            echo_pin=sensor_echo_pin,
            led_pin=sensor_led_pin,
            trigger_distance=sensor_trigger_distance,
            led_on_duration=sensor_led_duration,
            check_interval=sensor_check_interval
        )
        
        # Callbacks tích hợp với recognition workflow
        def on_person_detected(distance):
            # Kiểm tra cooldown: Nếu đang trong cooldown → không làm gì
            if recognition_state.is_in_cooldown():
                logger.info(f"Person detected but in cooldown - ignoring")
                return
            
            logger.info(f"Person detected at {distance}cm - LED ON")
            recognition_state.set_state(RecognitionState.PERSON_DETECTED)
            # Thông báo ngắn gọn
            tts.speak_custom("Xin chào")
        
        def on_person_left():
            logger.info("Person left - LED OFF")
            current_state = recognition_state.get_state()
            if current_state == RecognitionState.RECOGNIZED:
                # Đã nhận diện thành công → cảm ơn
                tts.speak_custom("Cảm ơn")
            recognition_state.set_state(RecognitionState.PERSON_LEFT)
            recognition_state.reset()
        
        sensor.set_on_person_detected(on_person_detected)
        sensor.set_on_person_left(on_person_left)
        sensor.start()
        logger.info(f"Sensor Controller started (trigger={sensor_trigger_distance}cm)")
    else:
        logger.info("Sensor Controller disabled")

    # ---- Camera: tối ưu cho Pi 3B+ ----
    picam2 = Picamera2()
    cfg_preview = picam2.create_preview_configuration(
        main={"size": (W, H), "format": "RGB888"}, 
        buffer_count=2  # Giảm từ 4 xuống 2 để tiết kiệm memory
    )
    picam2.configure(cfg_preview)
    picam2.start()
    print("[CAM] preview started")

    cam_lock = threading.Lock()

    # ---- State machine ----
    state_lock = threading.Lock()
    state = {
        "mode": "recognize",          # "recognize" | "enrolling"
        "job": None,                  # {"id":..., "emp_id":...}
        "samples": [],                # embeddings 192d
        "target_samples": 15,
        "last_cap": 0.0,
        "enroll_last_activity": 0.0,  # mốc hoạt động gần nhất trong chế độ enroll
    }

    # ---- Poll thread with BACKOFF ----
    def poll_jobs_loop():
        interval = poll_min
        while True:
            try:
                with state_lock:
                    busy = (state["mode"] != "recognize")
                if busy:
                    # đang enroll -> không nhận job mới
                    time.sleep(max(5.0, interval))
                    continue

                job = api_poll_next_job(base_url, device_id, timeout=3)
                if job and job.get("id"):
                    with state_lock:
                        state["mode"] = "enrolling"
                        state["job"] = {"id": job["id"], "emp_id": job["emp_id"]}
                        state["samples"] = []
                        state["last_cap"] = 0.0
                        state["enroll_last_activity"] = time.time()
                    # PHASE 2: Reset tracker khi chuyển sang enroll mode
                    if face_tracker:
                        face_tracker.reset()
                    print(f"[Enroll] new job -> id={job['id']} emp_id={job['emp_id']}")
                    api_job_start(base_url, job["id"], timeout=3)
                    interval = poll_min
                else:
                    interval = min(poll_max, max(poll_min, interval * 1.5))
            except Exception:
                interval = min(poll_max, max(poll_min, interval * 2))
            time.sleep(interval + random.uniform(0, 0.3))

    threading.Thread(target=poll_jobs_loop, daemon=True).start()
    print("[Enroll] poll thread started")

    # ---- Main loop: preview + xử lý theo mode ----
    prev=time.time(); frames=0; fps=0.0
    last_msg="Waiting face..."
    last_color=(0,255,0)
    last_time=0.0
    
    # PHASE 1 OPTIMIZATION: Adaptive frame skip
    base_frame_skip = cfg.get("recognition", {}).get("frame_skip", 3)
    use_adaptive_skip = cfg.get("recognition", {}).get("adaptive_frame_skip", True) and PSUTIL_AVAILABLE
    if use_adaptive_skip:
        adaptive_skip = AdaptiveFrameSkip(
            base_skip=base_frame_skip,
            max_skip=cfg.get("recognition", {}).get("max_frame_skip", 5),
            min_skip=cfg.get("recognition", {}).get("min_frame_skip", 1),
            check_interval=cfg.get("recognition", {}).get("cpu_check_interval", 2.0)
        )
    else:
        # Fixed frame skip
        adaptive_skip = type('obj', (object,), {'get_skip': lambda: base_frame_skip, 'enabled': False})()
    frame_counter = 0
    
    # Multi-frame voting để tăng accuracy
    use_voting = cfg.get("recognition", {}).get("multi_frame_voting", True)
    vote_window = cfg.get("recognition", {}).get("vote_window", 5)
    vote_threshold = cfg.get("recognition", {}).get("vote_threshold", 3)
    voting_buffer = VotingBuffer(window_size=vote_window, vote_threshold=vote_threshold) if use_voting else None
    
    # Throttle recognition (đọc từ config) - chỉ dùng khi không có async worker
    throttle_recog = cfg.get("recognition", {}).get("throttle_recognition", 0.8) if not recognition_worker else 0.0
    
    # PHASE 1 OPTIMIZATION: Memory buffers để reuse
    gray_buffer = None  # Reuse gray buffer
    
    # PHASE 2 OPTIMIZATION: ROI Tracking
    use_roi_tracking = cfg.get("recognition", {}).get("roi_tracking", True)
    face_tracker = None
    if use_roi_tracking:
        face_tracker = FaceTracker(
            decay=cfg.get("recognition", {}).get("roi_decay", 0.9),
            min_confidence=cfg.get("recognition", {}).get("roi_min_confidence", 0.3),
            expand=cfg.get("recognition", {}).get("roi_expand", 1.5)
        )
        logger.info("ROI Tracking enabled")
    
    logger.info(f"Performance settings: adaptive_skip={adaptive_skip.enabled}, base_skip={base_frame_skip}, throttle={throttle_recog}s")
    logger.info(f"Accuracy settings: voting={use_voting}, vote_window={vote_window}, vote_threshold={vote_threshold}")
    logger.info(f"Optimizations: async_recognition={recognition_worker is not None}, fast_quality=True, roi_tracking={use_roi_tracking}")
    if not PSUTIL_AVAILABLE:
        logger.warning("psutil not available - adaptive frame skip disabled (install: pip install psutil)")

    try:
        while True:
            frame_counter += 1
            
            # Capture frame
            with cam_lock:
                rgb = picam2.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # PHASE 1 OPTIMIZATION: Adaptive frame skip
            current_skip = adaptive_skip.get_skip()
            if frame_counter % current_skip != 0:
                # Vẫn hiển thị nhưng không xử lý
                cv2.putText(bgr, last_msg, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_color, 2, cv2.LINE_AA)
                cv2.imshow(window, bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            
            # Kiểm tra cooldown: Skip face detection và recognition nếu đang trong cooldown
            if sensor_enabled:
                # Lưu state trước khi check (vì is_in_cooldown() có thể thay đổi state)
                prev_state = recognition_state.get_state()
                is_cooldown = recognition_state.is_in_cooldown()
                current_state = recognition_state.get_state()
                
                if is_cooldown:
                    # Đang trong cooldown → không detect face, không nhận diện
                    # LED đã tắt từ khi nhận diện thành công
                    remaining = recognition_state.get_cooldown_remaining()
                    cv2.putText(bgr, f"Cooldown... {remaining:.1f}s", (10, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window, bgr)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue
                elif prev_state == RecognitionState.COOLDOWN and current_state == RecognitionState.IDLE:
                    # Cooldown vừa hết (chuyển từ COOLDOWN → IDLE) → reset message và voting buffer
                    last_msg = "Waiting face..."
                    last_color = (0, 255, 0)
                    if use_voting and voting_buffer:
                        voting_buffer.clear()
                    logger.info("Cooldown finished - cleared old results and reset UI")
            
            # PHASE 1 OPTIMIZATION: Reuse gray buffer
            if gray_buffer is None or gray_buffer.shape != bgr.shape[:2]:
                gray_buffer = np.zeros(bgr.shape[:2], dtype=np.uint8)
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY, dst=gray_buffer)
            gray = gray_buffer
            
            # PHASE 2 OPTIMIZATION: ROI Tracking - chỉ detect trong ROI
            faces = []
            roi = None
            if face_tracker:
                roi = face_tracker.get_roi(gray.shape)
            
            if roi:
                # Chỉ detect trong ROI (nhanh hơn 3-5x)
                x_roi, y_roi, w_roi, h_roi = roi
                roi_gray = gray[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
                faces_roi = face_cascade.detectMultiScale(roi_gray, 1.3, 4, minSize=(60,60))
                # Offset về tọa độ gốc
                faces = [(x+x_roi, y+y_roi, w, h) for (x,y,w,h) in faces_roi]
                # Vẽ ROI rectangle để debug (optional)
                if cfg.get("recognition", {}).get("show_roi", False):
                    cv2.rectangle(bgr, (x_roi, y_roi), (x_roi+w_roi, y_roi+h_roi), (255, 0, 255), 1)
            else:
                # Full frame detection (chỉ khi mất track hoặc lần đầu)
                faces = face_cascade.detectMultiScale(gray, 1.3, 4, minSize=(60,60))

            target=None; area_max=0
            for (x,y,w,h) in faces:
                if w*h > area_max:
                    area_max=w*h; target=(x,y,w,h)
            
            # PHASE 2 OPTIMIZATION: Update tracker
            if face_tracker:
                face_tracker.update(target)

            with state_lock:
                cur_mode = state["mode"]
                job_info = state["job"]
                samples  = state["samples"]
                last_cap = state["last_cap"]
                target_samples = state["target_samples"]
                last_activity  = state["enroll_last_activity"]

            if cur_mode == "recognize":
                # Kiểm tra timeout nếu đang nhận diện
                if sensor_enabled and recognition_state.get_state() == RecognitionState.RECOGNIZING:
                    if recognition_state.is_timeout():
                        recognition_state.set_state(RecognitionState.RECOGNITION_FAILED)
                        if sensor and turn_off_on_success:
                            sensor.turn_led_off_immediate()
                        tts.speak_custom("Hết thời gian")
                        recognition_state.reset()
                
                # PHASE 1 OPTIMIZATION: Check results từ async worker
                # QUAN TRỌNG: Chỉ xử lý results khi KHÔNG trong cooldown
                if recognition_worker:
                    # Kiểm tra cooldown TRƯỚC: Nếu đang trong cooldown → ignore tất cả results
                    if sensor_enabled and recognition_state.is_in_cooldown():
                        # Clear queue liên tục để tránh xử lý kết quả cũ
                        while True:
                            try:
                                recognition_worker.get_result()  # Pop result nếu có
                            except:
                                break
                    else:
                        # Chỉ xử lý result khi KHÔNG trong cooldown
                        result = recognition_worker.get_result()
                        if result:
                            # Double check: Nếu đã vào cooldown trong lúc xử lý → skip
                            if sensor_enabled and recognition_state.is_in_cooldown():
                                continue
                            
                            resp = result["response"]
                            if resp.get("status") == "ok":
                                res = resp.get("result", {}) or {}
                                emp = res.get("emp_id")
                                name = res.get("full_name")
                                score = res.get("score")
                                
                                if emp and name and score is not None:
                                    if use_voting and voting_buffer:
                                        voting_buffer.add(emp, score)
                                        voted = voting_buffer.get_voted_result()
                                        
                                        if voted:
                                            voted_emp, voted_score = voted
                                            last_msg = f"✓ {voted_emp} | {name} | {voted_score:.2f}"
                                            last_color = (0, 200, 0)
                                            
                                            # QUAN TRỌNG: Kiểm tra lại cooldown trước khi xử lý
                                            # (tránh xử lý nhiều results cùng lúc)
                                            if sensor_enabled and recognition_state.is_in_cooldown():
                                                # Đã vào cooldown từ result trước → skip result này
                                                continue
                                            
                                            # Tích hợp sensor: Tắt LED và thông báo
                                            recognition_state.set_state(RecognitionState.RECOGNIZED)
                                            recognition_state.last_recognized_name = name
                                            # QUAN TRỌNG: Luôn tắt LED khi nhận diện thành công (không phụ thuộc config)
                                            if sensor_enabled and sensor:
                                                sensor.turn_led_off_immediate()
                                                logger.info("LED turned off after successful recognition")
                                            
                                            # Clear async worker queues NGAY LẬP TỨC để tránh xử lý results cũ
                                            if recognition_worker:
                                                recognition_worker.clear_queues()
                                            
                                            # Thông báo ngắn gọn (chỉ 1 lần) - kiểm tra flag
                                            if recognition_state.should_speak():
                                                tts.speak_welcome(name)
                                                recognition_state.mark_spoken()
                                            # Cooldown đã được bắt đầu tự động trong set_state(RECOGNIZED)
                                            logger.info(f"Recognition successful - entering {cooldown_sec}s cooldown")
                                            
                                            # QUAN TRỌNG: Đã nhận diện thành công → skip phần còn lại của frame
                                            # Không cần break vì đã vào cooldown, các checks sau sẽ skip
                                            pass
                                        else:
                                            last_msg = f"Verifying... {emp} ({len(voting_buffer.buffer)}/{vote_threshold})"
                                            last_color = (0, 150, 150)
                                    else:
                                        last_msg = f"ACCEPTED: {emp} | {name} | score={score:.2f}"
                                        last_color = (0, 200, 0)
                                        
                                        # QUAN TRỌNG: Kiểm tra lại cooldown trước khi xử lý
                                        if sensor_enabled and recognition_state.is_in_cooldown():
                                            # Đã vào cooldown từ result trước → skip
                                            pass
                                        else:
                                            # Tích hợp sensor: Tắt LED và thông báo
                                            recognition_state.set_state(RecognitionState.RECOGNIZED)
                                            recognition_state.last_recognized_name = name
                                            # QUAN TRỌNG: Luôn tắt LED khi nhận diện thành công (không phụ thuộc config)
                                            if sensor_enabled and sensor:
                                                sensor.turn_led_off_immediate()
                                                logger.info("LED turned off after successful recognition")
                                            
                                            # Clear async worker queues NGAY LẬP TỨC để tránh xử lý results cũ
                                            if recognition_worker:
                                                recognition_worker.clear_queues()
                                            
                                            # Thông báo ngắn gọn (chỉ 1 lần) - kiểm tra flag
                                            if recognition_state.should_speak():
                                                tts.speak_welcome(name)
                                                recognition_state.mark_spoken()
                                            # Cooldown đã được bắt đầu tự động trong set_state(RECOGNIZED)
                                            logger.info(f"Recognition successful - entering {cooldown_sec}s cooldown")
                                else:
                                    last_msg = "OK but missing result"
                                    last_color = (0,150,150)
                            else:
                                if use_voting and voting_buffer:
                                    voting_buffer.add(None, 0.0)
                                last_msg = f"REJECT: {resp.get('reason')}"
                                last_color = (0,0,200)
                                
                                # Tích hợp sensor: Thông báo thất bại
                                if sensor_enabled:
                                    recognition_state.set_state(RecognitionState.RECOGNITION_FAILED)
                                    if recognition_state.should_retry():
                                        # Thông báo ngắn gọn
                                        tts.speak_custom("Không nhận diện được")
                                    else:
                                        # Hết số lần thử → tắt LED
                                        if sensor and turn_off_on_success:
                                            sensor.turn_led_off_immediate()
                                        tts.speak_custom("Hết thời gian")
                                        recognition_state.reset()
                
                if target:
                    x,y,w,h = target
                    cv2.rectangle(bgr, (x,y), (x+w,y+h), (0,255,0), 2)
                    
                    # PHASE 1 OPTIMIZATION: Use view instead of copy when possible
                    face_crop = bgr[y:y+h, x:x+w]  # View first
                    face_gray_roi = gray[y:y+h, x:x+w]  # Reuse gray
                    
                    # PHASE 1 OPTIMIZATION: Fast quality calculation từ gray
                    quality = calc_quality_fast(face_gray_roi)
                    
                    # QUAN TRỌNG: Kiểm tra cooldown TRƯỚC khi xử lý face
                    # Nếu đang trong cooldown → skip toàn bộ phần recognition
                    if sensor_enabled and recognition_state.is_in_cooldown():
                        # Đang trong cooldown → không nhận diện
                        continue
                    
                    # Tích hợp sensor: Set state RECOGNIZING khi bắt đầu nhận diện
                    if sensor_enabled:
                        current_state = recognition_state.get_state()
                        if current_state == RecognitionState.PERSON_DETECTED or current_state == RecognitionState.RECOGNITION_FAILED:
                            recognition_state.set_state(RecognitionState.RECOGNIZING)
                    
                    # PHASE 1 OPTIMIZATION: Async recognition (non-blocking)
                    if recognition_worker:
                        # Kiểm tra cooldown trước khi add frame vào queue
                        if sensor_enabled and recognition_state.is_in_cooldown():
                            continue
                        
                        # Add to worker queue (non-blocking)
                        if quality >= cfg.get("recognition", {}).get("quality_min", 0.45):
                            # Copy only when needed
                            face_crop_copy = face_crop.copy()
                            
                            # Image enhancement nếu quality thấp
                            enhance_low_quality = cfg.get("recognition", {}).get("enhance_low_quality", True)
                            if enhance_low_quality and quality < 0.5:
                                face_crop_copy = enhance_face_quality(face_crop_copy)
                            
                            # Add to worker (non-blocking)
                            recognition_worker.add_frame(face_crop_copy, quality)
                    else:
                        # Synchronous recognition (fallback)
                        # Kiểm tra cooldown trước khi nhận diện
                        if sensor_enabled and recognition_state.is_in_cooldown():
                            continue
                        
                        if quality >= cfg.get("recognition", {}).get("quality_min", 0.45):
                            if time.time() - last_time > throttle_recog:
                                try:
                                    # Image enhancement nếu quality thấp
                                    enhance_low_quality = cfg.get("recognition", {}).get("enhance_low_quality", True)
                                    if enhance_low_quality and quality < 0.5:
                                        face_crop = enhance_face_quality(face_crop)
                                    
                                    # Handle fixed_test mode
                                    if mode_cfg == "fixed_test":
                                        emb128 = np.array(make_embedding_fixed(), np.float32)
                                        emb = np.zeros(192, np.float32)
                                        emb[:128] = emb128
                                    else:
                                        emb = normalize_embedding(embedder(face_crop))
                                    
                                    resp = api_client.recognize(device_id, emb.tolist(), liveness=0.9, quality=quality)
                                    last_time = time.time()
                                    
                                    if resp.get("status") == "ok":
                                        res = resp.get("result", {}) or {}
                                        emp = res.get("emp_id")
                                        name = res.get("full_name")
                                        score = res.get("score")
                                        
                                        if emp and name and score is not None:
                                            if use_voting and voting_buffer:
                                                voting_buffer.add(emp, score)
                                                voted = voting_buffer.get_voted_result()
                                                
                                                if voted:
                                                    voted_emp, voted_score = voted
                                                    last_msg = f"✓ {voted_emp} | {name} | {voted_score:.2f}"
                                                    last_color = (0, 200, 0)
                                                    
                                                    # QUAN TRỌNG: Kiểm tra lại cooldown trước khi xử lý
                                                    if sensor_enabled and recognition_state.is_in_cooldown():
                                                        continue
                                                    
                                                    # Tích hợp sensor
                                                    recognition_state.set_state(RecognitionState.RECOGNIZED)
                                                    recognition_state.last_recognized_name = name
                                                    # QUAN TRỌNG: Luôn tắt LED khi nhận diện thành công (không phụ thuộc config)
                                                    if sensor_enabled and sensor:
                                                        sensor.turn_led_off_immediate()
                                                        logger.info("LED turned off after successful recognition")
                                                    
                                                    # Clear async worker queues NGAY LẬP TỨC
                                                    if recognition_worker:
                                                        recognition_worker.clear_queues()
                                                    
                                                    # Thông báo ngắn gọn (chỉ 1 lần) - kiểm tra flag
                                                    if recognition_state.should_speak():
                                                        tts.speak_welcome(name)
                                                        recognition_state.mark_spoken()
                                                    # Cooldown đã được bắt đầu tự động trong set_state(RECOGNIZED)
                                                    logger.info(f"Recognition successful - entering {cooldown_sec}s cooldown")
                                                    
                                                    # Break để không xử lý thêm
                                                    break
                                                else:
                                                    last_msg = f"Verifying... {emp} ({len(voting_buffer.buffer)}/{vote_threshold})"
                                                    last_color = (0, 150, 150)
                                            else:
                                                last_msg = f"ACCEPTED: {emp} | {name} | score={score:.2f}"
                                                last_color = (0, 200, 0)
                                                
                                                # QUAN TRỌNG: Kiểm tra lại cooldown trước khi xử lý
                                                if sensor_enabled and recognition_state.is_in_cooldown():
                                                    continue
                                                
                                                # Tích hợp sensor
                                                recognition_state.set_state(RecognitionState.RECOGNIZED)
                                                recognition_state.last_recognized_name = name
                                                # QUAN TRỌNG: Luôn tắt LED khi nhận diện thành công (không phụ thuộc config)
                                                if sensor_enabled and sensor:
                                                    sensor.turn_led_off_immediate()
                                                    logger.info("LED turned off after successful recognition")
                                                
                                                # Clear async worker queues NGAY LẬP TỨC
                                                if recognition_worker:
                                                    recognition_worker.clear_queues()
                                                
                                                # Thông báo ngắn gọn (chỉ 1 lần) - kiểm tra flag
                                                if recognition_state.should_speak():
                                                    tts.speak_welcome(name)
                                                    recognition_state.mark_spoken()
                                                # Cooldown đã được bắt đầu tự động trong set_state(RECOGNIZED)
                                                logger.info(f"Recognition successful - entering {cooldown_sec}s cooldown")
                                                
                                                # Break để không xử lý thêm
                                                break
                                    else:
                                        if use_voting and voting_buffer:
                                            voting_buffer.add(None, 0.0)
                                        last_msg = f"REJECT: {resp.get('reason')}"
                                        last_color = (0,0,200)
                                        
                                        # Tích hợp sensor
                                        if sensor_enabled:
                                            recognition_state.set_state(RecognitionState.RECOGNITION_FAILED)
                                            if recognition_state.should_retry():
                                                tts.speak_custom("Không nhận diện được")
                                            else:
                                                if sensor and turn_off_on_success:
                                                    sensor.turn_led_off_immediate()
                                                tts.speak_custom("Hết thời gian")
                                                recognition_state.reset()
                                except Exception as e:
                                    logger.error(f"Recognition error: {e}")
                                    last_msg = f"ERROR: {str(e)[:30]}"
                                    last_color = (0,0,255)
                else:
                    if use_voting and voting_buffer:
                        voting_buffer.clear()
                    last_msg="No face. Move closer."
                    last_color=(0,255,255)

            else:
                # ---- ENROLL MODE ----
                cv2.putText(bgr, "ENROLLING... look at camera",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)

                # 1) TIMEOUT: không có hoạt động quá enroll_timeout giây -> huỷ job & quay lại nhận diện
                if (time.time() - last_activity) > enroll_timeout:
                    if job_info is not None:
                        try:
                            api_job_done(base_url, job_info["id"], ok=False, notes="timeout_no_activity", timeout=5)
                        except Exception:
                            pass
                    last_msg = "ENROLL TIMEOUT -> back to recognize"
                    last_color = (0, 0, 255)
                    with state_lock:
                        state["mode"] = "recognize"
                        state["job"] = None
                        state["samples"] = []
                        state["last_cap"] = 0.0
                        state["enroll_last_activity"] = 0.0
                    # PHASE 2: Reset tracker khi quay lại recognize mode
                    if face_tracker:
                        face_tracker.reset()

                else:
                    # 2) Thu mẫu bình thường
                    if target and job_info is not None:
                        x,y,w,h = target
                        cv2.rectangle(bgr, (x,y), (x+w,y+h), (255,255,0), 2)
                        face_crop = bgr[y:y+h, x:x+w].copy()
                        q = calc_quality(face_crop)

                        # có hoạt động (thấy mặt) -> cập nhật mốc
                        with state_lock:
                            state["enroll_last_activity"] = time.time()

                        if (time.time()-last_cap) > 0.35 and q >= 0.5:
                            try:
                                if mode_cfg == "fixed_test":
                                    emb = np.zeros(192, np.float32)
                                    emb[:128] = np.array(make_embedding_fixed(), np.float32)
                                else:
                                    emb = normalize_embedding(embedder(face_crop))
                                    if emb.shape[0] != 192:
                                        tmp = np.zeros(192, np.float32)
                                        n = min(192, emb.shape[0]); tmp[:n] = emb[:n]
                                        emb = tmp
                                samples.append(emb.tolist())
                                last_cap = time.time()
                                last_msg = f"ENROLL {len(samples)}/{target_samples} (q={q:.2f})"
                                last_color = (0,200,255)

                                # vừa thêm mẫu -> cập nhật mốc
                                with state_lock:
                                    state["enroll_last_activity"] = last_cap

                            except Exception as e:
                                last_msg = f"ENROLL ERR: {e}"
                                last_color = (0,0,255)

                        if len(samples) >= target_samples:
                            try:
                                emp_id = job_info["emp_id"]; jid = job_info["id"]
                                resp = api_enroll(base_url, emp_id, samples, timeout=15)
                                ok = resp.get("status") == "ok"
                                notes = f"inserted={resp.get('inserted',0)}"
                                api_job_done(base_url, jid, ok=ok, notes=notes, timeout=5)
                                last_msg = f"ENROLL OK: {notes}" if ok else "ENROLL FAILED"
                                last_color = (0,200,0) if ok else (0,0,255)
                            except Exception as e:
                                try:
                                    api_job_done(base_url, job_info["id"], ok=False, notes=str(e)[:180], timeout=5)
                                except Exception:
                                    pass
                                last_msg = f"ENROLL ERR: {e}"
                                last_color = (0,0,255)
                            with state_lock:
                                state["mode"]="recognize"; state["job"]=None; state["samples"]=[]; state["last_cap"]=0.0
                                state["enroll_last_activity"] = 0.0
                            # PHASE 2: Reset tracker khi enroll xong
                            if face_tracker:
                                face_tracker.reset()
                        else:
                            with state_lock:
                                state["samples"]=samples
                                state["last_cap"]=last_cap
                    else:
                        # không thấy mặt -> hiển thị, để timeout xử lý nếu kéo dài
                        last_msg="ENROLLING... no face"
                        last_color=(0,200,255)

            # FPS & OSD
            frames += 1
            now=time.time()
            if now - prev >= 1.0:
                fps = frames / (now - prev)
                frames = 0; prev = now
            if show_fps:
                cv2.putText(bgr, f"FPS: {fps:.1f}", (10,24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(bgr, last_msg, (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_color, 2, cv2.LINE_AA)

            cv2.imshow(window, bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        try:
            # PHASE 1 OPTIMIZATION: Stop recognition worker
            if recognition_worker:
                recognition_worker.stop()
            if sensor:
                sensor.cleanup()
            tts.stop()
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
