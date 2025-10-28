#!/bin/bash
# Script setup HC-SR04 Ultrasonic Sensor + LED via MOSFET

set -e

echo "=========================================="
echo "Setup HC-SR04 Sensor + LED Controller"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "\n${YELLOW}Step 1: Cài đặt RPi.GPIO${NC}"
if python3 -c "import RPi.GPIO" 2>/dev/null; then
    echo -e "${GREEN}✓ RPi.GPIO đã được cài đặt${NC}"
else
    echo "Installing RPi.GPIO..."
    pip3 install --user RPi.GPIO
    echo -e "${GREEN}✓ RPi.GPIO installed${NC}"
fi

echo -e "\n${YELLOW}Step 2: Kiểm tra GPIO permissions${NC}"
if groups | grep -q gpio; then
    echo -e "${GREEN}✓ User có quyền GPIO${NC}"
else
    echo -e "${RED}Warning: User không có quyền GPIO${NC}"
    echo "Run: sudo usermod -a -G gpio $USER"
    echo "Then logout and login again"
fi

echo -e "\n${GREEN}=========================================="
echo "Wiring Instructions"
echo "==========================================${NC}"
echo ""
echo "HC-SR04 Ultrasonic Sensor:"
echo "  ┌─────────────┐"
echo "  │ VCC  TRIG   │"
echo "  │ ECHO GND    │"
echo "  └─────────────┘"
echo ""
echo "  VCC  -> Pi 5V (Pin 2 or 4)"
echo "  TRIG -> GPIO 23 (Pin 16)"
echo "  ECHO -> GPIO 24 (Pin 18) via voltage divider!"
echo "  GND  -> Pi GND (Pin 6, 9, 14, 20, 25, 30, 34, 39)"
echo ""
echo -e "${RED}⚠️  IMPORTANT: ECHO cần voltage divider 5V -> 3.3V!${NC}"
echo "  Dùng 2 resistors: 1kΩ và 2kΩ"
echo "  ECHO -> 1kΩ -> GPIO 24"
echo "               -> 2kΩ -> GND"
echo ""
echo "LED Control via MOSFET:"
echo "  ┌─────────────┐"
echo "  │ MOSFET      │"
echo "  │  G D S      │"
echo "  └─────────────┘"
echo ""
echo "  Gate (G)   -> GPIO 18 (Pin 12)"
echo "  Drain (D)  -> LED cathode (-)"
echo "  Source (S) -> GND"
echo "  LED anode (+) -> 12V/5V (via resistor)"
echo ""
echo -e "${YELLOW}Recommended MOSFET: IRLZ44N or similar logic-level${NC}"
echo ""

echo -e "\n${YELLOW}Step 3: GPIO Pin Summary (BCM numbering)${NC}"
echo "  GPIO 23 (Pin 16) -> HC-SR04 TRIG"
echo "  GPIO 24 (Pin 18) -> HC-SR04 ECHO (with voltage divider!)"
echo "  GPIO 18 (Pin 12) -> MOSFET Gate"
echo ""

echo -e "\n${YELLOW}Step 4: Test Sensor${NC}"
if [ -f "src/sensor_controller.py" ]; then
    echo "Test HC-SR04 sensor..."
    echo "Press Ctrl+C to stop"
    python3 src/sensor_controller.py sensor || echo "Sensor test failed (normal if hardware not connected)"
else
    echo -e "${RED}Warning: src/sensor_controller.py not found${NC}"
fi

echo -e "\n${YELLOW}Step 5: Test LED${NC}"
if [ -f "src/sensor_controller.py" ]; then
    echo "Test LED controller..."
    python3 src/sensor_controller.py led || echo "LED test failed (normal if hardware not connected)"
else
    echo -e "${RED}Warning: src/sensor_controller.py not found${NC}"
fi

echo -e "\n${GREEN}=========================================="
echo "Sensor Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Đấu nối phần cứng theo sơ đồ trên"
echo "  2. Test sensor:"
echo "     python3 src/sensor_controller.py"
echo ""
echo "  3. Enable trong config:"
echo "     nano config/client.yaml"
echo "     sensor:"
echo "       enabled: true"
echo ""
echo "  4. Chạy client:"
echo "     python3 src/client.py"
echo ""
echo "Tips:"
echo "  - HC-SR04 hoạt động tốt ở khoảng cách 2-400cm"
echo "  - Đặt trigger_distance = 100cm (1 mét) là hợp lý"
echo "  - LED sẽ sáng khi có người < trigger_distance"
echo "  - LED tự tắt sau led_on_duration giây"

