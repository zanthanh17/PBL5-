#!/bin/bash
# Script setup Text-to-Speech cho Raspberry Pi với loa 3.5mm

set -e

echo "=========================================="
echo "Setup Text-to-Speech for Raspberry Pi"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "\n${YELLOW}Step 1: Cài đặt espeak${NC}"
if command -v espeak &> /dev/null; then
    echo -e "${GREEN}✓ espeak đã được cài đặt${NC}"
else
    echo "Installing espeak..."
    sudo apt-get update
    sudo apt-get install -y espeak espeak-data
    echo -e "${GREEN}✓ espeak installed${NC}"
fi

echo -e "\n${YELLOW}Step 2: Cài đặt gói tiếng Việt (optional)${NC}"
# Thử cài tiếng Việt, nếu không có thì skip
sudo apt-get install -y espeak-ng-espeak espeak-ng-data || echo "Vietnamese voice not available, will use English"

echo -e "\n${YELLOW}Step 3: Kiểm tra audio output${NC}"
echo "Available audio devices:"
aplay -l

echo -e "\n${YELLOW}Step 4: Force audio ra 3.5mm jack${NC}"
# Force audio output to 3.5mm jack (not HDMI)
sudo raspi-config nonint do_audio 1
echo -e "${GREEN}✓ Audio forced to 3.5mm jack${NC}"

echo -e "\n${YELLOW}Step 5: Set audio volume${NC}"
# Set volume to 100%
amixer sset PCM 100%
echo -e "${GREEN}✓ Volume set to 100%${NC}"

echo -e "\n${YELLOW}Step 6: Test TTS${NC}"
echo "Testing espeak with Vietnamese..."
espeak -v vi "Xin chào" 2>/dev/null || espeak -v en "Hello" 2>/dev/null || echo "Warning: espeak test failed"

echo -e "\n${YELLOW}Step 7: Test Python TTS module${NC}"
if [ -f "src/tts_speaker.py" ]; then
    echo "Testing Python TTS module..."
    python3 src/tts_speaker.py
else
    echo -e "${RED}Warning: src/tts_speaker.py not found${NC}"
fi

echo -e "\n${GREEN}=========================================="
echo "TTS Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Tips:"
echo "  - Nếu không nghe thấy âm thanh, check:"
echo "    1. Loa đã cắm đúng jack 3.5mm"
echo "    2. Loa đã bật nguồn"
echo "    3. Volume đủ lớn: amixer sset PCM 100%"
echo ""
echo "  - Điều chỉnh volume trong config:"
echo "    nano config/client.yaml"
echo "    tts:"
echo "      volume: 100  # 0-200"
echo ""
echo "  - Test TTS:"
echo "    python3 src/tts_speaker.py"
echo ""
echo "  - Tắt TTS nếu không cần:"
echo "    config/client.yaml -> tts: enabled: false"

