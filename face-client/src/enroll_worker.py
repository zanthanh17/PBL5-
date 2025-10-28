import time, json, cv2, numpy as np, requests

def _quality(face_bgr: np.ndarray) -> float:
    g = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    b = float(np.mean(g)); f = float(cv2.Laplacian(g, cv2.CV_64F).var())
    b = np.clip((b - 50) / (190 - 50), 0, 1)
    f = np.clip((f - 80) / (400 - 80), 0, 1)
    return float(0.5*b + 0.5*f)

class EnrollWorker:
    """
    Dùng CHUNG camera với client:
    - switch_to_enroll(): client tắt preview, configure/start camera ở chế độ enroll
    - capture_once_rgb(): chụp 1 khung RGB từ camera chung
    - switch_to_preview(): chuyển camera về preview nhận diện
    """
    def __init__(
        self,
        base_url: str,
        device_id: str,
        model_path: str,
        batch: int = 15,
        on_status=None,
        switch_to_enroll=None,
        switch_to_preview=None,
        capture_once_rgb=None,
        face_cascade=None,
        embedder=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.device_id = device_id
        self.batch = batch
        self.on_status = on_status
        self.switch_to_enroll = switch_to_enroll
        self.switch_to_preview = switch_to_preview
        self.capture_once_rgb = capture_once_rgb
        self.cascade = face_cascade or cv2.CascadeClassifier(
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        )
        self.embedder = embedder  # MobileFaceNetEmbedder, được tạo sẵn ở client

    # ----- server API -----
    def _poll_next_job(self):
        try:
            r = requests.get(
                f"{self.base_url}/v1/enroll_jobs/next",
                params={"device_id": self.device_id},
                timeout=3,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    def _mark(self, job_id: int, phase: str, ok: bool | None = None, notes: str = ""):
        try:
            if phase == "start":
                requests.post(f"{self.base_url}/v1/enroll_jobs/{job_id}/start", timeout=3)
            elif phase == "done":
                requests.post(
                    f"{self.base_url}/v1/enroll_jobs/{job_id}/done",
                    params={"ok": ok, "notes": notes},
                    timeout=5,
                )
        except Exception:
            pass
        if self.on_status:
            self.on_status(phase, job_id, ok)

    # ----- capture embeddings dùng camera chung -----
    def _collect_embeddings(self):
        samples = []
        last_cap = 0.0
        warmup = 3  # bỏ qua vài frame đầu sau khi start()
        while len(samples) < self.batch:
            rgb = self.capture_once_rgb()
            
            if rgb is None:
                time.sleep(0.05)
                continue

            if warmup > 0:
                warmup -= 1
                time.sleep(0.01)
                continue

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
            if len(faces):
                x, y, w, h = max(faces, key=lambda t: t[2] * t[3])
                face = bgr[y : y + h, x : x + w].copy()
                if _quality(face) >= 0.5 and (time.time() - last_cap) > 0.3:
                    emb = self.embedder(face).astype(np.float32)
                    emb[np.isnan(emb)] = 0.0
                    emb[np.isinf(emb)] = 0.0
                    n = float(np.linalg.norm(emb) + 1e-9)
                    emb = emb / n
                    samples.append(emb.tolist())
                    last_cap = time.time()
            else:
                time.sleep(0.02)
        return samples

    # ----- 1 chu kỳ job -----
    def run_once(self):
        job = self._poll_next_job()
        if not job:
            return False

        jid, emp_id = job["id"], job["emp_id"]

        # chuyển camera sang chế độ enroll
        if self.switch_to_enroll:
            self.switch_to_enroll()

        self._mark(jid, "start")
        try:
            embs = self._collect_embeddings()
            payload = {"model": "mobilefacenet-192d", "embeddings": embs}
            r = requests.post(
                f"{self.base_url}/v1/enroll/{emp_id}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=20,
            )
            ok = r.status_code == 200 and (r.json().get("status") == "ok")
            notes = f"inserted={r.json().get('inserted', 0) if ok else 0}"
            self._mark(jid, "done", ok=ok, notes=notes)
        except Exception as e:
            self._mark(jid, "done", ok=False, notes=str(e)[:200])
        finally:
            # chuyển lại preview nhận diện
            if self.switch_to_preview:
                self.switch_to_preview()
        return True
