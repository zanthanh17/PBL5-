# 🔌 Sơ Đồ Đấu Nối HC-SR04 + LED

## 📋 Cần Thiết

### Hardware:
- ✅ Raspberry Pi 3B+
- ✅ HC-SR04 Ultrasonic Sensor
- ✅ MOSFET (IRLZ44N hoặc tương tự)
- ✅ LED 12V/5V + Resistor
- ✅ 2 Resistors: 1kΩ và 2kΩ (cho voltage divider)
- ✅ Breadboard + jumper wires

## 🔌 Sơ Đồ Đấu Nối

### HC-SR04 Ultrasonic Sensor

```
HC-SR04 Sensor:
┌─────────────┐
│ VCC   TRIG  │
│ ECHO  GND   │
└─────────────┘
     │
     │  VCC  ──────────────► Pi 5V (Pin 2 hoặc 4)
     │  GND  ──────────────► Pi GND (Pin 6, 9, 14, 20, 25, 30, 34, 39)
     │  TRIG ──────────────► GPIO 23 (Pin 16)
     │  ECHO ──┬── 1kΩ ────► GPIO 24 (Pin 18)
     │         └── 2kΩ ────► GND
     │
     └─ ⚠️ QUAN TRỌNG: Voltage Divider cho ECHO!
```

### Voltage Divider (BẮT BUỘC!)

```
HC-SR04 ECHO (5V) ──┬── 1kΩ ──┬── GPIO 24 (3.3V)
                    │          │
                    │          └── 2kΩ ──► GND
                    │
                 5V output   3.3V to Pi
```

**Tại sao cần voltage divider?**
- HC-SR04 ECHO output = 5V
- Pi GPIO chỉ chịu được 3.3V
- **Không có voltage divider → Pi sẽ hỏng!**

### LED Control via MOSFET

```
Pi GPIO 18 (Pin 12) ──► MOSFET Gate (G)
                           │
                           ├── Drain (D) ──► LED Cathode (-)
                           │
                           └── Source (S) ──► GND

Power Supply (+) ───┬── Resistor ──► LED Anode (+)
                    │
                12V/5V
```

### Sơ Đồ Hoàn Chỉnh

```
                    Raspberry Pi 3B+
                    ┌─────────────────┐
                    │  Pin 2  (5V)    │ ────► HC-SR04 VCC
                    │  Pin 6  (GND)   │ ────► HC-SR04 GND
                    │  Pin 16 (GPIO23)│ ────► HC-SR04 TRIG
                    │  Pin 18 (GPIO24)│ ◄──── HC-SR04 ECHO (via divider)
                    │  Pin 12 (GPIO18)│ ────► MOSFET Gate
                    │  Pin 14 (GND)   │ ────► MOSFET Source
                    └─────────────────┘
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

## 📌 Chi Tiết Từng Chân

### Raspberry Pi GPIO Pins (BCM numbering)

| Pin | Function | Connection |
|-----|----------|------------|
| **Pin 2** | 5V | HC-SR04 VCC |
| **Pin 6** | GND | HC-SR04 GND, MOSFET Source, Voltage Divider |
| **Pin 12** | GPIO 18 | MOSFET Gate |
| **Pin 14** | GND | MOSFET Source |
| **Pin 16** | GPIO 23 | HC-SR04 TRIG |
| **Pin 18** | GPIO 24 | HC-SR04 ECHO (via voltage divider) |

### HC-SR04 Pinout

| Pin | Name | Connection |
|-----|------|------------|
| **VCC** | Power | Pi 5V (Pin 2) |
| **TRIG** | Trigger | Pi GPIO 23 (Pin 16) |
| **ECHO** | Echo | Pi GPIO 24 (Pin 18) **via voltage divider** |
| **GND** | Ground | Pi GND (Pin 6) |

### MOSFET Pinout (IRLZ44N)

| Pin | Name | Connection |
|-----|------|------------|
| **Gate (G)** | Control | Pi GPIO 18 (Pin 12) |
| **Drain (D)** | Load | LED Cathode (-) |
| **Source (S)** | Ground | Pi GND (Pin 14) |

## 🔧 Cách Đấu Nối

### Bước 1: Đấu HC-SR04

1. **VCC → Pi 5V** (Pin 2)
2. **GND → Pi GND** (Pin 6)
3. **TRIG → Pi GPIO 23** (Pin 16)

### Bước 2: Đấu Voltage Divider cho ECHO

**QUAN TRỌNG**: ECHO không được nối trực tiếp!

```
HC-SR04 ECHO ──┬── 1kΩ ──┬── Pi GPIO 24 (Pin 18)
               │          │
               │          └── 2kΩ ──► Pi GND
               │
            5V output   3.3V to Pi
```

**Cách đấu:**
1. Lấy 1 dây từ HC-SR04 ECHO
2. Nối vào 1 đầu của resistor 1kΩ
3. Đầu kia của 1kΩ nối vào Pi GPIO 24
4. Từ điểm nối GPIO 24, nối 1 dây qua resistor 2kΩ xuống GND

### Bước 3: Đấu MOSFET + LED

1. **MOSFET Gate → Pi GPIO 18** (Pin 12)
2. **MOSFET Source → Pi GND** (Pin 14)
3. **MOSFET Drain → LED Cathode (-)**
4. **LED Anode (+) → Power Supply (+) qua Resistor**

### Bước 4: Power cho LED

**Nếu LED 5V:**
```
Pi 5V ──┬── 100Ω ──► LED Anode (+)
        │
     (nếu LED < 1W)
```

**Nếu LED 12V:**
```
12V PSU (+) ──┬── 330Ω ──► LED Anode (+)
              │
          External Supply
```

## ⚠️ Lưu Ý Quan Trọng

### 1. Voltage Divider là BẮT BUỘC
- **KHÔNG** nối ECHO trực tiếp vào GPIO!
- Phải dùng 1kΩ + 2kΩ voltage divider
- Nếu không có → Pi sẽ hỏng!

### 2. MOSFET Type
- Dùng **logic-level MOSFET** (Vgs ≤ 3.3V)
- **IRLZ44N**: ✅ Tốt nhất
- **IRF540**: ⚠️ Cần Vgs cao hơn, có thể không đủ

### 3. LED Polarity
- **Anode (+)**: Dương, nối với nguồn
- **Cathode (-)**: Âm, nối với MOSFET Drain

### 4. Power Supply
- LED < 1W: Có thể dùng Pi 5V
- LED > 1W: Dùng nguồn riêng 12V

## 🧪 Test Sau Khi Đấu Nối

### 1. Test HC-SR04
```bash
python3 src/sensor_controller.py sensor
# Nên thấy khoảng cách hiện lên
```

### 2. Test LED
```bash
python3 src/sensor_controller.py led
# LED nên nháy 5 lần
```

### 3. Test Full System
```bash
python3 src/sensor_controller.py 5
# Di chuyển tay trước sensor
# LED nên sáng khi < 100cm
```

## 🔍 Troubleshooting

### Vấn đề: Sensor không hoạt động
- ✅ Kiểm tra VCC → 5V
- ✅ Kiểm tra GND → GND
- ✅ Kiểm tra TRIG → GPIO 23
- ✅ Kiểm tra ECHO có voltage divider chưa

### Vấn đề: LED không sáng
- ✅ Kiểm tra MOSFET Gate → GPIO 18
- ✅ Kiểm tra MOSFET Source → GND
- ✅ Kiểm tra LED polarity
- ✅ Kiểm tra MOSFET type (logic-level)

### Vấn đề: Pi bị hỏng
- ❌ ECHO nối trực tiếp vào GPIO (không có voltage divider)
- ❌ Dùng nguồn 12V cho GPIO

## 📋 Checklist

- [ ] HC-SR04 VCC → Pi 5V
- [ ] HC-SR04 GND → Pi GND
- [ ] HC-SR04 TRIG → Pi GPIO 23
- [ ] HC-SR04 ECHO → Pi GPIO 24 **via voltage divider**
- [ ] MOSFET Gate → Pi GPIO 18
- [ ] MOSFET Source → Pi GND
- [ ] MOSFET Drain → LED Cathode (-)
- [ ] LED Anode (+) → Power qua Resistor
- [ ] Test sensor: `python3 src/sensor_controller.py sensor`
- [ ] Test LED: `python3 src/sensor_controller.py led`
- [ ] Test full: `python3 src/sensor_controller.py 5`

---

**⚠️ QUAN TRỌNG NHẤT**: ECHO phải qua voltage divider 1kΩ + 2kΩ, không được nối trực tiếp!
