# Face Attendance System - Raspberry Pi 3B+ + Server

Hệ thống chấm công nhận diện khuôn mặt với:
- **Client**: Raspberry Pi 3B+ với camera (nhận diện real-time)
- **Server**: FastAPI + PostgreSQL + pgvector (quản lý và xử lý)

## 🎯 Tính năng chính

### Client (Raspberry Pi)
- ✅ Nhận diện khuôn mặt real-time với MobileFaceNet
- ✅ **Tối ưu cho Pi 3B+**: Giảm CPU, memory usage
- ✅ **Offline mode**: Lưu queue khi mất kết nối
- ✅ **Multi-frame voting**: Tăng độ chính xác
- ✅ **TTA (Test Time Augmentation)**: Flip horizontal để tăng accuracy
- ✅ **Retry logic**: Xử lý network failures
- ✅ **Monitoring**: CPU, RAM, temperature tracking
- ✅ **Remote enrollment**: Enroll từ xa qua server

### Server
- ✅ FastAPI với OpenAPI docs
- ✅ PostgreSQL + pgvector (HNSW index)
- ✅ Centroid-based fast recognition
- ✅ KNN fallback với voting
- ✅ Web dashboard để quản lý
- ✅ Attendance tracking (checkin/checkout)
- ✅ CSV export
- ✅ Employee management

## 📦 Cấu trúc dự án

```
PBL5/
├── face-client/              # Raspberry Pi client
│   ├── config/
│   │   └── client.yaml       # Cấu hình client
│   ├── models/
│   │   └── mobilefacenet.tflite
│   ├── src/
│   │   ├── client.py         # Client gốc
│   │   ├── client_optimized.py  # ⭐ Client tối ưu mới
│   │   ├── embedding_tflite.py
│   │   ├── embedding_tflite_optimized.py  # ⭐ Với TTA
│   │   ├── offline_queue.py  # ⭐ Offline support
│   │   ├── monitoring.py     # ⭐ Metrics & monitoring
│   │   ├── enroll_real.py
│   │   └── enroll_worker.py
│   ├── setup_pi.sh           # ⭐ Setup script
│   ├── setup_autostart.sh    # ⭐ Autostart config
│   └── requirements.txt
│
└── face-attendance-server/   # Server
    ├── app/
    │   ├── main.py
    │   ├── models.py
    │   ├── schemas.py
    │   ├── db.py             # ✅ Đã thêm declarative_base
    │   ├── routers/
    │   │   ├── recognize.py
    │   │   ├── enroll.py
    │   │   ├── enroll_jobs.py
    │   │   ├── attendance.py
    │   │   ├── employees.py
    │   │   └── ui_dashboard.py
    │   ├── utils/
    │   │   ├── match.py
    │   │   ├── centroid_cache.py
    │   │   └── att_logic.py
    │   ├── static/
    │   └── templates/
    ├── db_init.sql           # ✅ Đã fix: 192d, thêm bảng
    ├── setup_server.sh       # ⭐ Setup script
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md
```

## 🚀 Cài đặt & Chạy

### 1. Setup Server (trên máy chủ)

```bash
cd face-attendance-server

# Chạy setup script
bash setup_server.sh

# Hoặc thủ công:
docker-compose up -d db
sleep 10
docker exec -i fa_db psql -U fa_user -d fa_db < db_init.sql
pip install -r requirements.txt

# Chạy server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập:
- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8000/

### 2. Setup Raspberry Pi Client

```bash
cd face-client

# Chạy setup script (sẽ cài tất cả dependencies)
bash setup_pi.sh

# Cấu hình
nano config/client.yaml  # Sửa IP server và device_id

# Copy source files
cp ../path/to/source/src/*.py src/

# Copy model file
cp mobilefacenet.tflite models/

# Kích hoạt venv và chạy
source venv/bin/activate
python src/client_optimized.py  # ⭐ Sử dụng version tối ưu
```

### 3. Setup Autostart (tùy chọn)

```bash
bash setup_autostart.sh

# Quản lý service
sudo systemctl start face-client
sudo systemctl status face-client
sudo systemctl stop face-client
sudo journalctl -u face-client -f  # Xem logs
```

## ⚙️ Cấu hình

### Client Config (config/client.yaml)

```yaml
server:
  base_url: "http://192.168.1.100:8000"  # IP server của bạn
  timeout_sec: 3

device:
  id: "pi-entrance-01"  # ID thiết bị (unique)

camera:
  width: 480   # Giảm từ 640 cho Pi 3B+
  height: 360  # Giảm từ 480 cho Pi 3B+

recognition:
  mode: "real"  # hoặc "fixed_test" để test
  model_path: "/home/pi/face-client/models/mobilefacenet.tflite"
  tta_flip: false  # true = chậm hơn nhưng chính xác hơn
  vote_window: 5   # Số frames để vote
  vote_need: 3     # Số votes tối thiểu

jobs:
  poll_min_sec: 2
  poll_max_sec: 30
  enroll_timeout_sec: 45
```

### Server Environment

```bash
# .env file (tùy chọn)
DATABASE_URL=postgresql+psycopg2://fa_user:fa_pass@127.0.0.1:5433/fa_db
MODEL_NAME=mobilefacenet-192d
CENTROID_THRESHOLD=0.65
KNN_THRESHOLD=0.60
DEBOUNCE_SEC=30
```

## 📊 API Endpoints

### Recognition
```bash
POST /v1/recognize
{
  "device_id": "pi-entrance-01",
  "embedding": [0.01, 0.02, ..., 192 floats],
  "liveness": 0.9,
  "quality": 0.75
}
```

### Enrollment
```bash
POST /v1/enroll/{emp_id}
{
  "model": "mobilefacenet-192d",
  "embeddings": [[...], [...], ...]  # 10-20 samples
}
```

### Enroll Jobs (Remote)
```bash
# Tạo job enroll từ xa
POST /v1/enroll_jobs?emp_id=E001&device_id=pi-entrance-01

# Pi sẽ tự động poll và thực hiện enroll
```

## 🔧 Troubleshooting

### Pi 3B+ chạy chậm?
1. Giảm resolution: `width: 320, height: 240`
2. Tăng frame skip trong code: `frame_skip = 3`
3. Tắt TTA: `tta_flip: false`
4. Giảm vote_window: `vote_window: 3`

### Lỗi camera?
```bash
# Check camera
libcamera-hello

# Nếu lỗi, reboot
sudo reboot
```

### Lỗi database?
```bash
# Recreate database
docker-compose down -v
docker-compose up -d db
sleep 10
docker exec -i fa_db psql -U fa_user -d fa_db < db_init.sql
```

### Check logs
```bash
# Client logs
sudo journalctl -u face-client -f

# Server logs
docker logs -f fa_db
```

## 📈 Performance Tuning

### Raspberry Pi 3B+ Optimization
- ✅ Resolution: 480x360 (giảm từ 640x480)
- ✅ Buffer count: 2 (giảm từ 4)
- ✅ Frame skip: 1/3 frames
- ✅ TFLite threads: 2
- ✅ Connection pooling
- ✅ Retry với exponential backoff

### Server Optimization
- ✅ pgvector HNSW index (fast approximate search)
- ✅ Centroid-first recognition (O(n) → O(1))
- ✅ Connection pooling: 10-20 connections
- ✅ Cache centroid in memory

## 📝 Cải tiến so với version cũ

| Tính năng | Cũ | Mới |
|-----------|-----|-----|
| Resolution | 640x480 | 480x360 |
| Camera buffers | 4 | 2 |
| Frame processing | Mọi frame | 1/3 frames |
| Offline support | ❌ | ✅ |
| Retry logic | ❌ | ✅ với backoff |
| Multi-frame voting | ❌ | ✅ |
| TTA | ❌ | ✅ (optional) |
| Monitoring | ❌ | ✅ CPU/RAM/Temp |
| Logging | Print | ✅ Structured logging |
| Database schema | 128d, thiếu bảng | ✅ 192d, đầy đủ |
| Setup scripts | ❌ | ✅ |

## 🎓 Workflow

### 1. Thêm nhân viên mới
```bash
# Via dashboard
http://localhost:8000/employees/new

# Hoặc API
curl -X POST http://localhost:8000/v1/employees \
  -d "emp_id=E001&full_name=Nguyen Van A"
```

### 2. Enroll khuôn mặt

**Cách 1: Remote từ dashboard**
```
1. Vào http://localhost:8000/employees/E001
2. Click "Enroll trên Pi"
3. Chọn device_id
4. Pi sẽ tự động bắt đầu enroll
```

**Cách 2: Local trên Pi**
```bash
python src/enroll_real.py E001 "Nguyen Van A"
```

### 3. Recognition
- Client tự động nhận diện khi thấy khuôn mặt
- Kết quả hiển thị trên màn hình và gửi về server
- Server lưu attendance record

### 4. Xem báo cáo
```
http://localhost:8000/
http://localhost:8000/employees/{emp_id}
http://localhost:8000/employees/{emp_id}/export.csv
```

## 🔐 Security Notes

- Thay đổi mật khẩu database trong docker-compose.yml
- Sử dụng HTTPS trong production
- Thêm authentication cho API endpoints
- Giới hạn rate limiting

## 📞 Support

- Issues: Tạo issue trên GitHub
- Email: your-email@example.com

## 📄 License

MIT License

---

**Lưu ý**: Đây là hệ thống tối ưu cho Raspberry Pi 3B+. Trên Pi 4 có thể tăng resolution và giảm frame skip để có performance tốt hơn.

