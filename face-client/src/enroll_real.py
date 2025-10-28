import time, json, sys
import numpy as np
import cv2, requests
from pathlib import Path
from picamera2 import Picamera2

# ---- embedder (dùng cùng file như client) ----
from embedding_tflite import MobileFaceNetEmbedder

SERVER_BASE = "http://192.168.110.63:8000"   # ĐỔI IP SERVER
MODEL_PATH  = "/home/pi/face-client/models/mobilefacenet.tflite"
DEVICE_ID   = "pi-entrance-01"

def calc_quality(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    b_norm = np.clip((brightness - 50) / (190 - 50), 0, 1)
    f_norm = np.clip((focus - 80) / (400 - 80), 0, 1)
    return float(0.5*b_norm + 0.5*f_norm), brightness, focus

def load_cascade():
    for p in ["/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
              "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"]:
        try:
            with open(p,"r"): return cv2.CascadeClassifier(p)
        except Exception: pass
    raise FileNotFoundError("Thiếu haarcascade_frontalface_default.xml (sudo apt install opencv-data).")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 enroll_real.py <EMP_ID> [FULL_NAME]"); sys.exit(1)
    emp_id = sys.argv[1]
    full_name = sys.argv[2] if len(sys.argv) >= 3 else None

    # đảm bảo employee tồn tại (tạo nếu chưa có)
    if full_name:
        try:
            requests.post(f"{SERVER_BASE}/v1/enroll/{emp_id}", json={"embeddings":[]}, timeout=2)
        except Exception:
            pass
        try:
            # Insert employee nếu chưa có
            import subprocess
            cmd = f"docker exec -i fa_db psql -U fa_user -d fa_db -c \"INSERT INTO employees(emp_id, full_name) VALUES ('{emp_id}','{full_name}') ON CONFLICT (emp_id) DO NOTHING;\""
            subprocess.run(cmd, shell=True)
        except Exception:
            pass

    face_cascade = load_cascade()
    embedder = MobileFaceNetEmbedder(MODEL_PATH)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640,480), "format":"RGB888"}, buffer_count=4)
    picam2.configure(config); picam2.start()

    win = "Enroll"
    collected = []
    needed = 15
    last_q = 0.0

    try:
        while True:
            rgb = picam2.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80,80))

            target=None; area=0
            for (x,y,w,h) in faces:
                if w*h>area: area=w*h; target=(x,y,w,h)

            if target:
                x,y,w,h = target
                cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,255,0),2)
                face = bgr[y:y+h, x:x+w].copy()

                q, br, fc = calc_quality(face)
                last_q = q
                # lấy mẫu khi chất lượng đủ tốt
                if q >= 0.6 and len(collected) < needed and (w*h) > 120*120:
                    emb = embedder(face).astype(np.float32)
                    emb[np.isnan(emb)] = 0.0
                    emb[np.isinf(emb)] = 0.0
                    collected.append(emb.tolist())
                    cv2.putText(bgr, f"Captured {len(collected)}/{needed}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(bgr, f"Q={last_q:.2f}  Hold steady & blink", (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.imshow(win,bgr)
            key = cv2.waitKey(1) & 0xFF
            if key==27: break
            if len(collected)>=needed:
                break
    finally:
        picam2.stop(); cv2.destroyAllWindows()

    if not collected:
        print("Không thu được mẫu đủ chất lượng."); sys.exit(1)

    print(f"Sending {len(collected)} embeddings to server for emp_id={emp_id} ...")
    payload = {"model":"mobilefacenet-192d", "embeddings": collected}
    r = requests.post(f"{SERVER_BASE}/v1/enroll/{emp_id}",
                      headers={"Content-Type":"application/json"},
                      data=json.dumps(payload))
    print("Server:", r.status_code, r.text)

if __name__=="__main__":
    main()
