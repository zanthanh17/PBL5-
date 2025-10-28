# 🌟 HC-SR04 Sensor + LED Guide

## Tính năng

Hệ thống tự động phát hiện người lại gần và bật đèn LED:
- **Cảm biến HC-SR04**: Đo khoảng cách người với Pi (2-400cm)
- **LED qua MOSFET**: Bật đèn khi phát hiện người
- **TTS**: Phát âm "Xin chào" khi phát hiện người
- **Auto off**: Tắt đèn sau 10 giây (configurable)

## 📋 Yêu cầu

### Hardware:
- ✅ Raspberry Pi 3B+ với GPIO
- ✅ HC-SR04 Ultrasonic Sensor
- ✅ MOSFET (IRLZ44N hoặc tương tự)
- ✅ LED strip/bulb 12V hoặc 5V
- ✅ Resistors: 1kΩ và 2kΩ (cho voltage divider)
- ✅ Resistor cho LED (nếu cần)
- ✅ Breadboard và jumper wires

### Software:
- ✅ `RPi.GPIO` - GPIO control library
- ✅ Python 3

## 🔌 Sơ Đồ Đấu Nối

### HC-SR04 Ultrasonic Sensor

```
┌─────────────┐
│ HC-SR04     │
│             │
│ VCC   TRIG  │
│ ECHO  GND   │
└─────────────┘
     │
     │  VCC  ──────────────► Pi 5V (Pin 2 or 4)
     │  GND  ──────────────► Pi GND (Pin 6, 9, 14, ...)
     │  TRIG ──────────────► GPIO 23 (Pin 16)
     │  ECHO ──┬── 1kΩ ────► GPIO 24 (Pin 18)
     │         └── 2kΩ ────► GND
     │
     └─ ⚠️ Voltage Divider cho ECHO!
```

**⚠️ QUAN TRỌNG**: 
- HC-SR04 ECHO output = 5V
- Pi GPIO chỉ chịu được 3.3V
- **BẮT BUỘC phải dùng voltage divider!**

### Voltage Divider cho ECHO

```
HC-SR04 ECHO ──┬── 1kΩ ──┬── GPIO 24
               │          │
               │          └── 2kΩ ──► GND
               │
            5V output   3.3V to Pi
```

**Công thức**: Vout = Vin × R2 / (R1 + R2)
- Vout = 5V × 2kΩ / (1kΩ + 2kΩ) = 3.33V ✅

### LED Control via MOSFET

```
Pi GPIO 18 ──► MOSFET Gate (G)
                   │
                   ├── Drain (D) ──► LED Cathode (-)
                   │
                   └── Source (S) ──► GND

Power (+) ───┬── Resistor ──► LED Anode (+)
             │
        12V/5V supply
```

**MOSFET khuyến nghị**: 
- IRLZ44N (logic-level, Vgs = 3.3V)
- IRF540 (nếu không có IRLZ44N)
- Bất kỳ N-channel MOSFET logic-level nào

### Sơ Đồ Hoàn Chỉnh

```
┌─────────────────────────────────────────┐
│  Raspberry Pi 3B+                        │
│                                          │
│  Pin 2 (5V) ────────────► HC-SR04 VCC   │
│  Pin 6 (GND) ───────────► HC-SR04 GND   │
│  Pin 16 (GPIO 23) ──────► HC-SR04 TRIG  │
│  Pin 18 (GPIO 24) ◄──┬── HC-SR04 ECHO   │
│                       │   (via divider)  │
│  Pin 12 (GPIO 18) ───┼──► MOSFET Gate   │
│                       │                  │
└───────────────────────┼──────────────────┘
                        │
                        │  Voltage Divider
                        │  1kΩ + 2kΩ
                        │
                   ┌────┴─────┐
                   │ MOSFET   │
                   │  IRLZ44N │
                   └────┬─────┘
                        │
                   ┌────┴─────┐
                   │   LED    │
                   │  Strip   │
                   └──────────┘
```

## 🚀 Setup

### 1. Cài đặt dependencies

```bash
cd face-client
./setup_sensor.sh
```

### 2. Test Hardware

**Test HC-SR04 sensor:**
```bash
python3 src/sensor_controller.py sensor
# Nên thấy khoảng cách hiện lên
```

**Test LED:**
```bash
python3 src/sensor_controller.py led
# LED nên nháy 5 lần
```

**Test full system:**
```bash
python3 src/sensor_controller.py
# Di chuyển tay trước sensor
# LED nên sáng khi < 100cm
```

### 3. Enable trong Client

Edit `config/client.yaml`:
```yaml
sensor:
  enabled: true  # Bật sensor
  trig_pin: 23
  echo_pin: 24
  led_pin: 18
  trigger_distance: 100.0  # cm
  led_on_duration: 10.0    # seconds
```

### 4. Chạy Client

```bash
python3 src/client.py
```

## ⚙️ Cấu hình

### GPIO Pins (BCM numbering)

| Pin | Function | Description |
|-----|----------|-------------|
| 23 | HC-SR04 TRIG | Trigger pulse |
| 24 | HC-SR04 ECHO | Echo return (via voltage divider!) |
| 18 | MOSFET Gate | LED control |

**Thay đổi pins**:
```yaml
sensor:
  trig_pin: 17  # Thay GPIO 23 -> 17
  echo_pin: 27  # Thay GPIO 24 -> 27
  led_pin: 22   # Thay GPIO 18 -> 22
```

### Distance Threshold

```yaml
sensor:
  trigger_distance: 150.0  # Tăng lên 1.5m
```

| Distance | Khi nào dùng |
|----------|--------------|
| 50cm | Cảm biến gần, chỉ bật khi rất gần |
| 100cm | ✅ **Khuyến nghị** - 1 mét, vừa phải |
| 150cm | Cảm biến xa hơn, bật sớm |
| 200cm+ | Quá xa, dễ false trigger |

### LED Duration

```yaml
sensor:
  led_on_duration: 15.0  # Giữ LED sáng 15 giây
```

## 🎯 Workflow

1. **Người lại gần** (< trigger_distance):
   - Sensor phát hiện
   - LED bật ⚡
   - TTS phát "Xin chào" 🔊
   - Log: "Person detected at Xcm"

2. **Người đứng yên**:
   - LED vẫn sáng
   - Timer reset

3. **Người rời đi** (> trigger_distance):
   - Đếm ngược led_on_duration giây
   - LED tắt 💡
   - Log: "Person left"

## 🔧 Troubleshooting

### Vấn đề 1: Sensor không hoạt động

**Kiểm tra:**
```bash
# 1. Check GPIO permissions
groups | grep gpio
# Nếu không có: sudo usermod -a -G gpio $USER

# 2. Check wiring
# VCC -> 5V
# GND -> GND
# TRIG -> GPIO 23
# ECHO -> GPIO 24 (via voltage divider!)

# 3. Test sensor
python3 src/sensor_controller.py sensor
```

**Lỗi thường gặp:**
- Không có voltage divider → **Pi có thể hỏng!**
- ECHO nối trực tiếp vào GPIO → ❌ NGUY HIỂM!
- Sử dụng 1kΩ + 2kΩ voltage divider → ✅ AN TOÀN

### Vấn đề 2: LED không sáng

**Kiểm tra:**
```bash
# 1. Test LED trực tiếp
python3 src/sensor_controller.py led

# 2. Check MOSFET
# - Gate -> GPIO 18
# - Source -> GND
# - Drain -> LED cathode

# 3. Check MOSFET type
# Phải là logic-level MOSFET (Vgs ≤ 3.3V)
# IRLZ44N: ✅
# IRF540: ⚠️ (cần Vgs cao hơn, có thể không đủ)

# 4. Check LED polarity
# Anode (+) -> Power
# Cathode (-) -> MOSFET Drain
```

### Vấn đề 3: LED nhấp nháy liên tục

**Nguyên nhân**: Sensor không ổn định

**Giải pháp:**
```yaml
sensor:
  trigger_distance: 120.0  # Tăng lên
  check_interval: 0.3      # Chậm hơn (default 0.2)
```

### Vấn đề 4: False triggers

**Nguyên nhân**: Sensor phát hiện vật khác

**Giải pháp:**
- Đặt sensor xa tường/vật cản
- Giảm trigger_distance
- Đặt sensor ở góc tốt hơn

### Vấn đề 5: RPi.GPIO import error

```bash
# Cài đặt RPi.GPIO
pip3 install --user RPi.GPIO

# Hoặc
sudo apt-get install python3-rpi.gpio
```

## 💡 Tips & Tricks

### 1. Tối Ưu Placement

**Sensor placement:**
- Chiều cao: 1-1.5m từ mặt đất
- Góc: Hướng thẳng, không nghiêng
- Vị trí: Xa tường/vật cản > 50cm

**LED placement:**
- Đủ sáng để nhìn thấy từ xa
- Không chiếu trực tiếp vào camera

### 2. Power Management

**12V LED:**
```
12V PSU (+) ─┬─ 330Ω ─► LED (+)
             │
         External
          Supply
```

**5V LED:**
```
Pi 5V ──┬─ 100Ω ─► LED (+)
        │
    (nếu LED < 1W)
```

### 3. Multiple LEDs

```python
# Trong sensor_controller.py, thêm nhiều LED pins
led_pins = [18, 22, 27]
leds = [LEDController(pin) for pin in led_pins]
```

### 4. Adjust Sensitivity

```yaml
sensor:
  trigger_distance: 80.0   # Giảm sensitivity
  led_on_duration: 5.0     # Tắt nhanh hơn
  check_interval: 0.3      # Check chậm hơn
```

### 5. Debug Mode

```python
# Trong sensor_controller.py
logging.basicConfig(level=logging.DEBUG)
# Sẽ thấy distance readings real-time
```

## 📊 Performance

| Config | CPU Impact | Response Time |
|--------|-----------|---------------|
| Sensor OFF | 0% | N/A |
| Sensor ON | ~2-3% | ~200ms |

Sensor chạy trong thread riêng, không ảnh hưởng camera/recognition.

## 🎓 Advanced

### Custom Actions on Detection

Edit `src/client.py`:
```python
def on_person_detected(distance):
    logger.info(f"Person at {distance}cm")
    
    # Custom actions
    if distance < 50:
        tts.speak_custom("Xin vui lòng đứng xa một chút")
    else:
        tts.speak_custom("Xin chào")
    
    # Blink LED
    sensor.led.blink(times=2, delay=0.1)
```

### Integration with Recognition

```python
# Chỉ nhận diện khi có người gần
if sensor and sensor.led.is_on:
    # Recognition logic
    pass
else:
    # Skip recognition, save CPU
    continue
```

### Distance-based Actions

```python
def on_person_detected(distance):
    if distance < 30:
        tts.speak_custom("Quá gần")
        sensor.led.blink(3, 0.1)
    elif distance < 80:
        tts.speak_welcome("user")
    else:
        sensor.led.on()
```

## 📝 Checklist

Sau khi setup:
- [ ] RPi.GPIO đã cài đặt
- [ ] HC-SR04 đã đấu nối (với voltage divider!)
- [ ] MOSFET đã đấu nối đúng
- [ ] LED hoạt động
- [ ] Test sensor thành công
- [ ] Test LED thành công
- [ ] Config sensor enabled=true
- [ ] Chạy client và test

## ⚠️ Safety Notes

1. **Voltage Divider là BẮT BUỘC** cho HC-SR04 ECHO!
2. Không nối ECHO trực tiếp vào Pi GPIO (5V sẽ hỏng Pi!)
3. MOSFET phải là logic-level (Vgs ≤ 3.3V)
4. LED power riêng nếu > 1W (không dùng Pi 5V)
5. Kiểm tra polarity LED trước khi cấp nguồn

## 🚀 Quick Start

```bash
# 1. Setup hardware theo sơ đồ trên
# 2. Install
cd face-client
./setup_sensor.sh

# 3. Test
python3 src/sensor_controller.py

# 4. Enable
nano config/client.yaml
# sensor: enabled: true

# 5. Run
python3 src/client.py

# → LED sẽ sáng khi có người < 100cm
```

---

**Lưu ý**: Đây là hệ thống tự động, không cần can thiệp thủ công!

