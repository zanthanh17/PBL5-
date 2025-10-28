import time, threading, json, random, logging
import cv2, numpy as np, requests
from picamera2 import Picamera2
from collections import deque, Counter
from typing import Optional, Dict, List
from tts_speaker import TTSSpeaker
from sensor_controller import SensorController

# ============= Logging Setup =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============= Config & Utils =============
def load_config(path="config/client.yaml"):
    try:
        import yaml
    except ImportError:
        raise RuntimeError("Missing PyYAML. Install: pip install pyyaml")
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
    """Tính chất lượng ảnh (brightness + sharpness)"""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    b_norm = np.clip((brightness - 50) / (190 - 50), 0, 1)
    f_norm = np.clip((focus - 80) / (400 - 80), 0, 1)
    return float(0.5 * b_norm + 0.5 * f_norm)

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

# ============= API calls với Retry Logic =============
class APIClient:
    """API client với retry logic và timeout handling"""
    
    def __init__(self, base_url: str, timeout: int = 3, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
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
    api_client = APIClient(base_url, timeout)
    cascade_path = find_haarcascade()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    embedder = None
    if mode_cfg == "real":
        from embedding_tflite import MobileFaceNetEmbedder
        tta_flip = cfg.get("recognition", {}).get("tta_flip", False)
        embedder = MobileFaceNetEmbedder(model_path, use_tta=tta_flip)   # 192d
        logger.info(f"Loaded MobileFaceNet embedder (TTA: {tta_flip})")
    
    # Initialize TTS Speaker
    tts_cfg = cfg.get("tts", {})
    tts_enabled = tts_cfg.get("enabled", True)
    tts_volume = tts_cfg.get("volume", 100)
    tts_speed = tts_cfg.get("speed", 150)
    tts_cooldown = tts_cfg.get("cooldown", 3.0)
    tts = TTSSpeaker(enabled=tts_enabled, volume=tts_volume, speed=tts_speed, cooldown=tts_cooldown)
    logger.info(f"TTS Speaker initialized (enabled: {tts_enabled})")
    
    # Initialize Sensor Controller (HC-SR04 + LED)
    sensor_cfg = cfg.get("sensor", {})
    sensor_enabled = sensor_cfg.get("enabled", False)
    sensor_trig_pin = sensor_cfg.get("trig_pin", 23)
    sensor_echo_pin = sensor_cfg.get("echo_pin", 24)
    sensor_led_pin = sensor_cfg.get("led_pin", 18)
    sensor_trigger_distance = sensor_cfg.get("trigger_distance", 100.0)
    sensor_led_duration = sensor_cfg.get("led_on_duration", 10.0)
    
    sensor = None
    if sensor_enabled:
        sensor = SensorController(
            trig_pin=sensor_trig_pin,
            echo_pin=sensor_echo_pin,
            led_pin=sensor_led_pin,
            trigger_distance=sensor_trigger_distance,
            led_on_duration=sensor_led_duration
        )
        
        # Callbacks khi phát hiện người
        def on_person_detected(distance):
            logger.info(f"Person detected at {distance}cm - LED ON")
            tts.speak_custom("Xin chào")
        
        def on_person_left():
            logger.info("Person left - LED OFF")
        
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
    
    # Frame skip để giảm CPU cho Pi 3B+ (đọc từ config)
    frame_skip = cfg.get("recognition", {}).get("frame_skip", 3)
    frame_counter = 0
    
    # Multi-frame voting để tăng accuracy
    use_voting = cfg.get("recognition", {}).get("multi_frame_voting", True)
    vote_window = cfg.get("recognition", {}).get("vote_window", 5)
    vote_threshold = cfg.get("recognition", {}).get("vote_threshold", 3)
    voting_buffer = VotingBuffer(window_size=vote_window, vote_threshold=vote_threshold) if use_voting else None
    
    # Throttle recognition (đọc từ config)
    throttle_recog = cfg.get("recognition", {}).get("throttle_recognition", 0.8)
    
    logger.info(f"Performance settings: frame_skip={frame_skip}, throttle={throttle_recog}s")
    logger.info(f"Accuracy settings: voting={use_voting}, vote_window={vote_window}, vote_threshold={vote_threshold}")

    try:
        while True:
            frame_counter += 1
            
            # Capture frame
            with cam_lock:
                rgb = picam2.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Frame skip để giảm CPU
            if frame_counter % frame_skip != 0:
                # Vẫn hiển thị nhưng không xử lý
                cv2.putText(bgr, last_msg, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_color, 2, cv2.LINE_AA)
                cv2.imshow(window, bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Tối ưu face detection cho Pi 3B+
            # scaleFactor=1.3 (tăng từ 1.2) = nhanh hơn nhưng có thể miss một số faces
            # minNeighbors=4 (giảm từ 5) = detect nhanh hơn
            # minSize giảm xuống (60,60) để detect xa hơn
            faces = face_cascade.detectMultiScale(gray, 1.3, 4, minSize=(60,60))

            target=None; area_max=0
            for (x,y,w,h) in faces:
                if w*h > area_max:
                    area_max=w*h; target=(x,y,w,h)

            with state_lock:
                cur_mode = state["mode"]
                job_info = state["job"]
                samples  = state["samples"]
                last_cap = state["last_cap"]
                target_samples = state["target_samples"]
                last_activity  = state["enroll_last_activity"]

            if cur_mode == "recognize":
                if target:
                    x,y,w,h = target
                    cv2.rectangle(bgr, (x,y), (x+w,y+h), (0,255,0), 2)
                    face_crop = bgr[y:y+h, x:x+w].copy()
                    quality = calc_quality(face_crop)
                    if time.time()-last_time > throttle_recog:
                        try:
                            # Image enhancement nếu quality thấp (chỉ khi RẤT thấp để tránh lag)
                            enhance_low_quality = cfg.get("recognition", {}).get("enhance_low_quality", True)
                            if enhance_low_quality and quality < 0.5:  # Giảm từ 0.6 xuống 0.5
                                face_crop = enhance_face_quality(face_crop)
                            
                            if mode_cfg == "fixed_test":
                                # pad lên 192d để khớp server
                                emb128 = np.array(make_embedding_fixed(), np.float32)
                                emb = np.zeros(192, np.float32); emb[:128] = emb128
                            else:
                                emb = normalize_embedding(embedder(face_crop))
                            
                            # Sử dụng APIClient với retry logic
                            resp = api_client.recognize(device_id, emb.tolist(),
                                                       liveness=0.9, quality=float(quality))
                            
                            if resp.get("status") == "ok":
                                res = resp.get("result", {}) or {}
                                emp = res.get("emp_id")
                                name = res.get("full_name")
                                score = res.get("score")
                                
                                if emp and name and score is not None:
                                    # Multi-frame voting để tăng accuracy
                                    if use_voting and voting_buffer:
                                        voting_buffer.add(emp, score)
                                        voted = voting_buffer.get_voted_result()
                                        
                                        if voted:
                                            voted_emp, voted_score = voted
                                            last_msg = f"✓ {voted_emp} | {name} | {voted_score:.2f}"
                                            last_color = (0, 200, 0)
                                            # TTS: Phát âm tên user
                                            tts.speak_welcome(name)
                                        else:
                                            last_msg = f"Verifying... {emp} ({len(voting_buffer.buffer)}/{vote_threshold})"
                                            last_color = (0, 150, 150)
                                    else:
                                        # Không dùng voting
                                        last_msg = f"ACCEPTED: {emp} | {name} | score={score:.2f}"
                                        last_color = (0, 200, 0)
                                        # TTS: Phát âm tên user
                                        tts.speak_welcome(name)
                                else:
                                    last_msg = "OK but missing result"
                                    last_color = (0,150,150)
                            else:
                                if use_voting and voting_buffer:
                                    voting_buffer.add(None, 0.0)
                                last_msg = f"REJECT: {resp.get('reason')}"
                                last_color = (0,0,200)
                        except Exception as e:
                            last_msg = f"ERR: {e}"
                            last_color = (0,0,255)
                        last_time = time.time()
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
            if sensor:
                sensor.cleanup()
            tts.stop()
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
