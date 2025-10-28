# 🔊 Text-to-Speech Guide - Phát âm thanh khi nhận diện

## Tính năng

Khi nhận diện khuôn mặt thành công, hệ thống sẽ phát âm thanh qua loa 3.5mm:
- **"Xin chào, [Tên User]"** - Khi nhận diện thành công

## 📋 Yêu cầu

### Hardware:
- ✅ Raspberry Pi 3B+ (hoặc cao hơn)
- ✅ Loa 3.5mm cắm vào Pi
- ✅ Loa có nguồn (nếu cần)

### Software:
- ✅ `espeak` - Text-to-Speech engine (lightweight)
- ✅ Python 3

## 🚀 Setup

### 1. Cài đặt TTS trên Pi

```bash
cd face-client
./setup_tts.sh
```

Script này sẽ:
- ✅ Cài đặt `espeak`
- ✅ Force audio output ra 3.5mm jack (không phải HDMI)
- ✅ Set volume 100%
- ✅ Test TTS

### 2. Kiểm tra âm thanh

```bash
# Test espeak
espeak -v vi "Xin chào"

# Hoặc dùng English nếu không có tiếng Việt
espeak -v en "Hello"

# Test Python TTS module
python3 src/tts_speaker.py
```

### 3. Chạy client với TTS

```bash
python3 src/client.py
```

## ⚙️ Cấu hình

### File: `config/client.yaml`

```yaml
tts:
  enabled: true  # Bật/tắt TTS
  volume: 100    # Âm lượng (0-200)
  speed: 150     # Tốc độ nói (80-450 words per minute)
```

### Các thông số:

| Thông số | Giá trị | Mô tả |
|----------|---------|-------|
| `enabled` | true/false | Bật/tắt TTS |
| `volume` | 0-200 | Âm lượng (100 = mặc định) |
| `speed` | 80-450 | Tốc độ nói (150 = vừa phải) |

### Ví dụ cấu hình:

**Volume cao, nói nhanh:**
```yaml
tts:
  enabled: true
  volume: 150
  speed: 200
```

**Volume thấp, nói chậm:**
```yaml
tts:
  enabled: true
  volume: 80
  speed: 120
```

**Tắt TTS:**
```yaml
tts:
  enabled: false
```

## 🎯 Các Message

### 1. Welcome (mặc định)
```python
tts.speak_welcome("Nguyễn Văn A")
# Output: "Xin chào, Nguyễn Văn A"
```

### 2. Check-in
```python
tts.speak_checkin("Nguyễn Văn A")
# Output: "Check in thành công, Nguyễn Văn A"
```

### 3. Check-out
```python
tts.speak_checkout("Nguyễn Văn A")
# Output: "Check out thành công, Nguyễn Văn A"
```

### 4. Rejected
```python
tts.speak_rejected()
# Output: "Không nhận diện được khuôn mặt"
```

### 5. Custom Message
```python
tts.speak_custom("Hệ thống chấm công")
# Output: "Hệ thống chấm công"
```

## 🔧 Troubleshooting

### Vấn đề 1: Không nghe thấy âm thanh

**Kiểm tra:**
```bash
# 1. Check loa đã cắm đúng jack 3.5mm chưa
# 2. Check loa đã bật nguồn chưa
# 3. Check audio output
aplay -l

# 4. Force audio to 3.5mm jack
sudo raspi-config nonint do_audio 1

# 5. Tăng volume
amixer sset PCM 100%

# 6. Test
speaker-test -t wav -c 2
```

### Vấn đề 2: espeak không cài được

```bash
# Update apt
sudo apt-get update

# Install espeak
sudo apt-get install -y espeak espeak-data

# Check version
espeak --version
```

### Vấn đề 3: Không có tiếng Việt

Nếu không có voice tiếng Việt, hệ thống sẽ tự động dùng English:
```bash
# Thử cài espeak-ng (có nhiều voice hơn)
sudo apt-get install -y espeak-ng espeak-ng-data

# List voices
espeak --voices
```

### Vấn đề 4: Âm thanh bị delay/lag

```bash
# Giảm tốc độ nói
nano config/client.yaml
# tts:
#   speed: 120  # Giảm từ 150
```

### Vấn đề 5: TTS làm camera lag

```bash
# TTS chạy trong thread riêng, không block camera
# Nếu vẫn lag, tắt TTS:
nano config/client.yaml
# tts:
#   enabled: false
```

## 🎓 Advanced Usage

### Tùy chỉnh Message trong Code

Edit `src/client.py`:

```python
# Thay đổi message khi nhận diện thành công
# Tìm dòng:
tts.speak_welcome(name)

# Thay bằng:
tts.speak_custom(f"Chào mừng {name} đến công ty")

# Hoặc check-in/check-out
event_type = res.get("type", "checkin")
if event_type == "checkin":
    tts.speak_checkin(name)
else:
    tts.speak_checkout(name)
```

### Thêm Sound Effects

```bash
# Cài sox để play sound effects
sudo apt-get install -y sox

# Play beep trước khi TTS
aplay /usr/share/sounds/alsa/Front_Center.wav
```

### Tích hợp với Multiple Speakers

```python
# Trong tts_speaker.py, thêm:
def speak_with_device(self, text: str, device: str = "default"):
    cmd = f'AUDIODEV={device} espeak -v vi "{text}"'
    os.system(cmd)
```

## 📊 Performance Impact

| Config | CPU Impact | Khi nào dùng |
|--------|-----------|--------------|
| TTS OFF | 0% | Không cần âm thanh |
| TTS ON | ~5% | ✅ Khuyến nghị |

TTS chạy trong thread riêng, không ảnh hưởng đến camera/recognition.

## 💡 Tips & Tricks

### 1. Volume System-wide
```bash
# Điều chỉnh volume của Pi
alsamixer

# Hoặc command line
amixer sset PCM 80%
```

### 2. Auto-start TTS
TTS tự động start khi client start, không cần làm gì thêm.

### 3. Test TTS riêng
```bash
# Test module TTS
python3 src/tts_speaker.py

# Test espeak trực tiếp
espeak -v vi "Xin chào Nguyễn Văn A"
```

### 4. Giảm độ trễ
```bash
# Trong config, giảm throttle_recognition
recognition:
  throttle_recognition: 0.5  # Giảm từ 0.8
```

### 5. Multiple Languages
```bash
# List available voices
espeak --voices

# Test English
espeak -v en "Hello"

# Test Vietnamese
espeak -v vi "Xin chào"
```

## 📝 Checklist

Sau khi setup:
- [ ] espeak đã cài đặt
- [ ] Loa đã cắm vào jack 3.5mm
- [ ] Audio output đã force ra 3.5mm
- [ ] Volume đã set đủ lớn
- [ ] Test espeak thành công
- [ ] Test Python TTS module thành công
- [ ] Config TTS trong client.yaml
- [ ] Chạy client và test nhận diện

## 🚀 Quick Start

```bash
# 1. Setup
cd face-client
./setup_tts.sh

# 2. Test
python3 src/tts_speaker.py

# 3. Chạy client
python3 src/client.py

# 4. Test nhận diện
# → Nên nghe "Xin chào, [Tên]" khi nhận diện thành công
```

---

**Lưu ý**: TTS hoạt động tốt nhất với:
- Loa có nguồn (powered speaker)
- Volume Pi set ở 80-100%
- Tốc độ nói 120-180 wpm

