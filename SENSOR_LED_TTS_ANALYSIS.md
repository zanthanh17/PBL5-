# 📊 Phân Tích Logic Bật/Tắt LED & TTS - Đề Xuất Cải Tiến

## 🔍 Phân Tích Logic Hiện Tại

### 1. **Sensor Controller Logic**

**Workflow hiện tại:**
```
1. Đo khoảng cách liên tục (khi LED không bật)
2. Nếu khoảng cách < trigger_distance (50cm) → Bật LED
3. LED bật trong led_on_duration (15 giây)
4. Trong khi LED bật → TẠM DỪNG đo khoảng cách
5. Sau 15 giây → LED tự động tắt
6. Tiếp tục đo khoảng cách
```

**Callbacks:**
- `on_person_detected(distance)`: Chỉ nói "Xin chào"
- `on_person_left()`: Không có thông báo

### 2. **TTS Speaker Logic**

**Các hàm có sẵn:**
- `speak_welcome(name)`: "Xin chào, {name}"
- `speak_checkin(name)`: "Check in thành công, {name}"
- `speak_checkout(name)`: "Check out thành công, {name}"
- `speak_rejected()`: "Không nhận diện được khuôn mặt"
- `speak_custom(message)`: Message tùy chỉnh

**Cooldown:** 3 giây (tránh phát trùng)

### 3. **Face Recognition Logic**

**Workflow:**
- Detect face → Compute embedding → API call → Response
- Nếu accepted → `tts.speak_welcome(name)`
- Nếu rejected → Không có thông báo

### 4. **Vấn Đề Hiện Tại**

❌ **Tách rời giữa Sensor và Face Recognition:**
- Sensor chỉ nói "Xin chào" khi phát hiện người
- Face recognition nói "Xin chào, {name}" khi nhận diện thành công
- Không có sự phối hợp giữa 2 hệ thống

❌ **LED không tích hợp với recognition:**
- LED bật 15 giây cố định, không liên quan đến kết quả nhận diện
- Có thể LED tắt trước khi nhận diện xong

❌ **Thiếu thông báo:**
- Không có thông báo khi người rời đi
- Không có thông báo khi nhận diện thất bại
- Không có hướng dẫn người dùng

---

## ✅ Đề Xuất Logic Cải Tiến

### **Workflow Mới (Tích Hợp Sensor + LED + TTS + Recognition)**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SENSOR PHÁT HIỆN NGƯỜI (< 50cm)                         │
│    → LED BẬT                                                 │
│    → TTS: "Xin chào, vui lòng nhìn vào camera"              │
│    → Bắt đầu face recognition                                │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. ĐANG NHẬN DIỆN (LED vẫn bật)                            │
│    → LED có thể nhấp nháy (optional)                        │
│    → TTS: "Đang nhận diện..." (nếu > 3 giây)               │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3A. NHẬN DIỆN THÀNH CÔNG                                   │
│     → LED TẮT NGAY                                            │
│     → TTS: "Xin chào, {name}" hoặc                         │
│            "Check in thành công, {name}"                     │
│     → Reset sensor (để phát hiện người tiếp theo)           │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ 3B. NHẬN DIỆN THẤT BẠI                                      │
│     → LED vẫn bật (cho phép thử lại)                        │
│     → TTS: "Không nhận diện được, vui lòng thử lại"         │
│     → Tiếp tục nhận diện (trong thời gian LED bật)         │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. NGƯỜI RỜI ĐI (sensor > 50cm)                            │
│    → LED TẮT                                                  │
│    → TTS: "Cảm ơn bạn" (nếu đã nhận diện thành công)        │
│    → Reset state                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Chi Tiết Đề Xuất

### **1. Tích Hợp Sensor với Face Recognition**

**State Machine:**
```python
States:
- IDLE: Không có người, LED tắt
- PERSON_DETECTED: Phát hiện người, LED bật, đang chờ nhận diện
- RECOGNIZING: Đang nhận diện (LED bật)
- RECOGNIZED: Nhận diện thành công (LED tắt ngay)
- RECOGNITION_FAILED: Nhận diện thất bại (LED vẫn bật, cho phép thử lại)
- PERSON_LEFT: Người rời đi (LED tắt)
```

### **2. LED Control Logic**

**Các chế độ LED:**
- **OFF**: Không có người
- **ON (solid)**: Phát hiện người, đang chờ/đang nhận diện
- **ON (blink)**: Đang nhận diện (optional, để feedback)
- **OFF (immediate)**: Nhận diện thành công → tắt ngay

**Thời gian LED:**
- **Không cố định 15 giây**
- **Tắt ngay khi nhận diện thành công**
- **Tắt sau 15 giây nếu không nhận diện được**
- **Tắt khi người rời đi (sensor > 50cm)**

### **3. TTS Messages**

**Message Flow:**
1. **Phát hiện người:**
   - "Xin chào, vui lòng nhìn vào camera"

2. **Đang nhận diện (nếu > 3 giây):**
   - "Đang nhận diện, vui lòng đợi..."

3. **Nhận diện thành công:**
   - "Xin chào, {name}" (nếu checkin)
   - "Check out thành công, {name}" (nếu checkout)

4. **Nhận diện thất bại:**
   - "Không nhận diện được, vui lòng thử lại"
   - "Vui lòng đứng gần hơn và nhìn thẳng vào camera"

5. **Người rời đi:**
   - "Cảm ơn bạn" (nếu đã nhận diện thành công)
   - Không nói gì (nếu chưa nhận diện)

### **4. Timeout & Retry Logic**

**Timeout:**
- Nếu sau 15 giây vẫn không nhận diện được → LED tắt, thông báo "Hết thời gian"
- Reset state, chờ người tiếp theo

**Retry:**
- Cho phép nhận diện lại trong thời gian LED bật
- Tối đa 3 lần thử (có thể config)

---

## 💻 Implementation Plan

### **Phase 1: State Management**

1. Tạo `RecognitionState` class để quản lý state
2. Tích hợp với SensorController
3. Callback từ sensor → update state

### **Phase 2: LED Control**

1. Thêm method `turn_led_off_immediate()` trong SensorController
2. Thêm method `blink_led()` (optional)
3. Update logic: LED tắt ngay khi nhận diện thành công

### **Phase 3: TTS Integration**

1. Update callbacks trong client.py
2. Thêm messages mới vào TTS
3. Tích hợp với recognition workflow

### **Phase 4: Timeout & Retry**

1. Thêm timeout logic
2. Thêm retry counter
3. Reset state khi timeout

---

## 📝 Code Structure Đề Xuất

```python
class RecognitionState:
    """Quản lý state của recognition workflow"""
    IDLE = "idle"
    PERSON_DETECTED = "person_detected"
    RECOGNIZING = "recognizing"
    RECOGNIZED = "recognized"
    RECOGNITION_FAILED = "recognition_failed"
    PERSON_LEFT = "person_left"
    
    def __init__(self):
        self.current_state = self.IDLE
        self.recognition_start_time = None
        self.retry_count = 0
        self.max_retries = 3
        self.timeout_sec = 15.0

# Trong client.py:
recognition_state = RecognitionState()

def on_person_detected(distance):
    recognition_state.set_state(RecognitionState.PERSON_DETECTED)
    sensor.turn_led_on()
    tts.speak_custom("Xin chào, vui lòng nhìn vào camera")

def on_recognition_start():
    recognition_state.set_state(RecognitionState.RECOGNIZING)
    recognition_state.recognition_start_time = time.time()

def on_recognition_success(name, att_type):
    recognition_state.set_state(RecognitionState.RECOGNIZED)
    sensor.turn_led_off_immediate()  # Tắt ngay
    if att_type == "checkin":
        tts.speak_checkin(name)
    elif att_type == "checkout":
        tts.speak_checkout(name)
    else:
        tts.speak_welcome(name)

def on_recognition_failed(reason):
    recognition_state.set_state(RecognitionState.RECOGNITION_FAILED)
    recognition_state.retry_count += 1
    
    if recognition_state.retry_count < recognition_state.max_retries:
        tts.speak_custom("Không nhận diện được, vui lòng thử lại")
        # Tiếp tục nhận diện
    else:
        tts.speak_custom("Hết thời gian, vui lòng thử lại sau")
        sensor.turn_led_off_immediate()
        recognition_state.reset()

def on_person_left():
    if recognition_state.current_state == RecognitionState.RECOGNIZED:
        tts.speak_custom("Cảm ơn bạn")
    recognition_state.set_state(RecognitionState.PERSON_LEFT)
    sensor.turn_led_off()
    recognition_state.reset()
```

---

## ⚙️ Configuration Options

```yaml
sensor:
  enabled: true
  trig_pin: 23
  echo_pin: 24
  led_pin: 18
  trigger_distance: 50.0  # cm
  led_on_duration: 15.0  # giây (max, sẽ tắt sớm hơn nếu nhận diện thành công)
  check_interval: 0.2
  led_blink_on_recognition: false  # Nhấp nháy khi đang nhận diện
  turn_off_on_success: true  # Tắt LED ngay khi nhận diện thành công

recognition:
  max_retries: 3  # Số lần thử lại khi thất bại
  recognition_timeout: 15.0  # Timeout (giây)

tts:
  enabled: true
  volume: 100
  speed: 150
  cooldown: 3.0
  messages:
    person_detected: "Xin chào, vui lòng nhìn vào camera"
    recognizing: "Đang nhận diện, vui lòng đợi..."
    recognition_failed: "Không nhận diện được, vui lòng thử lại"
    recognition_timeout: "Hết thời gian, vui lòng thử lại sau"
    person_left: "Cảm ơn bạn"
```

---

## 🎯 Lợi Ích

✅ **Tích hợp tốt hơn:**
- Sensor, LED, TTS và Recognition hoạt động đồng bộ
- User experience mượt mà hơn

✅ **Feedback rõ ràng:**
- User biết hệ thống đang làm gì
- Hướng dẫn rõ ràng khi cần

✅ **Hiệu quả hơn:**
- LED tắt ngay khi nhận diện xong (tiết kiệm điện)
- Không chờ đợi không cần thiết

✅ **Linh hoạt:**
- Có thể config các thông số
- Dễ dàng tùy chỉnh messages

---

## 🚀 Next Steps

1. **Review đề xuất** - Xác nhận logic phù hợp
2. **Implement Phase 1** - State management
3. **Implement Phase 2** - LED control
4. **Implement Phase 3** - TTS integration
5. **Test & Tune** - Điều chỉnh messages và timing

