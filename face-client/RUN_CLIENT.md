# 🚀 Chạy Client Với Tất Cả Features

## ✅ Features Đã Tích Hợp

Client.py bao gồm **TẤT CẢ** tính năng:

1. **Face Recognition** 📷
   - MobileFaceNet 192D embeddings
   - Multi-frame voting (5 frames, 3 votes)
   - Image enhancement cho ảnh chất lượng thấp
   - Frame skip để tối ưu CPU

2. **Text-to-Speech** 🔊
   - Phát âm tên user khi nhận diện thành công
   - Cooldown 3s để tránh phát trùng
   - Volume 100, speed 150 wpm

3. **HC-SR04 Sensor + LED** 💡
   - Phát hiện khoảng cách
   - LED ON khi distance > 10cm
   - LED OFF khi distance ≤ 10cm
   - Auto LED control

## 🎯 Workflow Hoàn Chỉnh

```
1. Sensor phát hiện khoảng cách > 10cm
   ↓
2. LED bật ⚡
   ↓
3. TTS: "Xin chào" 🔊
   ↓
4. Camera bắt đầu nhận diện 📷
   ↓
5. Voting 5 frames (cần 3 votes)
   ↓
6. Nhận diện thành công
   ↓
7. TTS: "Xin chào, [Tên User]" 🔊
   ↓
8. Lưu attendance record
   ↓
9. Khoảng cách ≤ 10cm hoặc timeout 10s
   ↓
10. LED tắt 💡
```

## 🚀 Cách Chạy

### Trên Pi:

```bash
cd ~/face-client

# 1. Kiểm tra config
nano config/client.yaml

# Đảm bảo:
# - server.base_url đúng IP server
# - tts.enabled = true
# - sensor.enabled = true

# 2. Chạy client
python3 src/client.py
```

### Output Mong Đợi:

```
2025-10-28 10:30:00 [INFO] Loaded MobileFaceNet embedder (TTA: False)
2025-10-28 10:30:01 [INFO] TTS Speaker initialized (enabled: True)
2025-10-28 10:30:01 [INFO] HC-SR04 initialized (TRIG=23, ECHO=24)
2025-10-28 10:30:01 [INFO] LED Controller initialized (PIN=18)
2025-10-28 10:30:01 [INFO] Sensor Controller initialized (trigger=10.0cm)
2025-10-28 10:30:01 [INFO] Sensor Controller started (trigger=10.0cm)
2025-10-28 10:30:02 [INFO] Performance settings: frame_skip=3, throttle=0.8s
2025-10-28 10:30:02 [INFO] Accuracy settings: voting=True, vote_window=5, vote_threshold=3
[CAM] preview started

# Khi có người lại gần (>10cm):
2025-10-28 10:30:10 [INFO] Person detected at 45.23cm - LED ON
# TTS phát: "Xin chào"

# Khi nhận diện thành công:
✓ E001 | Nguyễn Văn A | 0.85
# TTS phát: "Xin chào, Nguyễn Văn A"

# Khi người rời đi:
2025-10-28 10:30:25 [INFO] Person left - LED OFF
```

## ⚙️ Config Quan Trọng

### File: `config/client.yaml`

```yaml
server:
  base_url: "http://172.20.10.2:8001"  # ⚠️ Sửa IP server

camera:
  width: 320   # Tối ưu cho Pi 3B+
  height: 240

recognition:
  frame_skip: 3              # Xử lý 1/4 frames
  throttle_recognition: 0.8  # Delay giữa các lần recognize
  multi_frame_voting: true   # Voting để tăng accuracy
  vote_window: 5             # 5 frames
  vote_threshold: 3          # Cần 3 votes
  enhance_low_quality: false # TẮT để tránh lag

tts:
  enabled: true    # Bật TTS
  volume: 100      # Âm lượng
  speed: 150       # Tốc độ nói
  cooldown: 3.0    # Tránh phát trùng

sensor:
  enabled: true           # ⚠️ Bật sensor
  trigger_distance: 10.0  # Ngưỡng 10cm
  led_on_duration: 10.0   # LED sáng 10s
  check_interval: 0.2     # Check mỗi 0.2s
```

## 🔧 Troubleshooting

### 1. TTS Không Hoạt Động

```bash
# Test TTS
python3 src/tts_speaker.py

# Nếu lỗi, cài espeak
sudo apt-get install espeak
```

### 2. Sensor Không Hoạt Động

```bash
# Test sensor
python3 src/sensor_controller.py sensor

# Test LED
python3 src/sensor_controller.py led

# Kiểm tra wiring theo WIRING_DIAGRAM.md
```

### 3. Camera Lag

```bash
# Tắt image enhancement
nano config/client.yaml
# recognition:
#   enhance_low_quality: false

# Tăng frame skip
# recognition:
#   frame_skip: 4
```

### 4. Recognition Không Chính Xác

```bash
# Tăng voting threshold
# recognition:
#   vote_threshold: 4

# Enroll lại users với nhiều samples hơn
```

### 5. LED Không Tắt

```bash
# Kiểm tra trigger_distance
# sensor:
#   trigger_distance: 10.0  # Tăng nếu cần
```

## 📊 Performance

### Trên Pi 3B+:

| Feature | CPU % | Impact |
|---------|-------|--------|
| Camera | 15% | Base |
| Face Recognition | 20% | Processing |
| Frame Skip | -10% | Optimization |
| Multi-frame Voting | +2% | Accuracy |
| TTS | ~5% | Non-blocking |
| Sensor + LED | ~3% | Background |
| **Total** | **35-40%** | ✅ Good |

### FPS:

- Display: 15-20 FPS (mượt)
- Processing: 4-5 FPS (frame_skip=3)
- Recognition: ~1.25 recognition/s (throttle=0.8s)

## 💡 Tips

### 1. Tắt Features Không Cần

**Tắt TTS:**
```yaml
tts:
  enabled: false
```

**Tắt Sensor:**
```yaml
sensor:
  enabled: false
```

### 2. Tối Ưu Cho Accuracy

```yaml
recognition:
  frame_skip: 2           # Xử lý nhiều hơn
  vote_threshold: 4       # Voting nghiêm ngặt hơn
  tta_flip: true          # Bật TTA (chậm hơn nhưng chính xác)
```

### 3. Tối Ưu Cho Performance

```yaml
recognition:
  frame_skip: 4           # Xử lý ít hơn
  multi_frame_voting: false  # Tắt voting
  enhance_low_quality: false # Tắt enhancement
```

### 4. Debug Mode

```python
# Trong client.py, thay đổi logging level
logging.basicConfig(level=logging.DEBUG)
```

## 🎓 Advanced

### Tùy Chỉnh Sensor Callback

Edit `src/client.py`:

```python
def on_person_detected(distance):
    logger.info(f"Person detected at {distance}cm - LED ON")
    
    # Custom actions
    if distance < 20:
        tts.speak_custom("Xin vui lòng đứng xa hơn")
    else:
        tts.speak_custom("Xin chào")
```

### Tùy Chỉnh Recognition Message

```python
# Trong phần nhận diện thành công
if voted:
    voted_emp, voted_score = voted
    
    # Thêm logic tùy chỉnh
    if voted_score > 0.9:
        tts.speak_custom(f"Chào mừng {name}")
    else:
        tts.speak_custom(f"Xin chào {name}")
```

## 📝 Checklist

Trước khi chạy client:

- [ ] Server đang chạy (`http://IP:8001`)
- [ ] Database đã setup
- [ ] Đã enroll users
- [ ] Config `base_url` đúng IP server
- [ ] TTS espeak đã cài (`./setup_tts.sh`)
- [ ] Sensor đã đấu nối đúng (xem `WIRING_DIAGRAM.md`)
- [ ] Camera hoạt động
- [ ] Volume speaker đủ lớn
- [ ] LED hoạt động

## 🚀 Quick Start

```bash
# 1. Setup (chỉ làm 1 lần)
cd face-client
./setup_tts.sh
./setup_sensor.sh

# 2. Config
nano config/client.yaml
# Sửa server IP

# 3. Test từng phần
python3 src/tts_speaker.py           # Test TTS
python3 src/sensor_controller.py 5   # Test sensor

# 4. Chạy full client
python3 src/client.py

# → Tất cả features sẽ hoạt động cùng lúc!
```

---

**Lưu ý**: Tất cả features đã được tích hợp sẵn trong `client.py`. Bạn chỉ cần chạy 1 file duy nhất!

## 🎉 Kết Quả

Khi chạy `python3 src/client.py`, bạn sẽ có:

✅ Camera nhận diện real-time
✅ Sensor tự động bật/tắt LED
✅ TTS phát âm tên user
✅ Multi-frame voting cho accuracy cao
✅ Offline queue khi mất mạng
✅ Auto retry khi API fail

**All-in-one solution!** 🚀

