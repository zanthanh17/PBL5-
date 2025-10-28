#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test LED Always ON
Test LED luôn sáng để kiểm tra phần cứng
"""

import time
import logging

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("RPi.GPIO not available - LED test disabled")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def test_led_always_on(led_pin=18, duration=30):
    """
    Test LED luôn sáng
    
    Args:
        led_pin: GPIO pin cho LED (default 18)
        duration: Thời gian test (giây, default 30)
    """
    if not GPIO_AVAILABLE:
        print("❌ RPi.GPIO not available!")
        print("Install: pip3 install RPi.GPIO")
        return
    
    print("🔌 LED Always ON Test")
    print("=" * 40)
    print(f"LED Pin: GPIO {led_pin}")
    print(f"Duration: {duration} seconds")
    print("WIRING: GPIO 18 → MOSFET Gate")
    print("        MOSFET Drain → LED Cathode (-)")
    print("        MOSFET Source → GND")
    print("        LED Anode (+) → Power via Resistor")
    print("")
    print("Press Ctrl+C to stop early")
    print("=" * 40)
    
    try:
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(led_pin, GPIO.OUT)
        
        print(f"✅ GPIO {led_pin} configured as OUTPUT")
        print("")
        
        # Turn LED ON
        GPIO.output(led_pin, GPIO.HIGH)
        print("🔆 LED ON - Should be bright now!")
        print("")
        
        # Countdown
        for i in range(duration, 0, -1):
            print(f"\r⏱️  LED will turn OFF in {i:2d} seconds...", end="", flush=True)
            time.sleep(1)
        
        print("\n")
        
        # Turn LED OFF
        GPIO.output(led_pin, GPIO.LOW)
        print("💡 LED OFF")
        
    except KeyboardInterrupt:
        print("\n")
        print("⏹️  Stopped by user")
        GPIO.output(led_pin, GPIO.LOW)
        print("💡 LED OFF")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
    finally:
        # Cleanup
        GPIO.cleanup()
        print("🧹 GPIO cleanup completed")

def test_led_blink(led_pin=18, times=10, delay=0.5):
    """
    Test LED nháy
    
    Args:
        led_pin: GPIO pin cho LED
        times: Số lần nháy
        delay: Thời gian delay (giây)
    """
    if not GPIO_AVAILABLE:
        print("❌ RPi.GPIO not available!")
        return
    
    print("✨ LED Blink Test")
    print("=" * 30)
    print(f"LED Pin: GPIO {led_pin}")
    print(f"Blink {times} times with {delay}s delay")
    print("")
    
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(led_pin, GPIO.OUT)
        
        for i in range(times):
            print(f"Blink {i+1}/{times}...", end="", flush=True)
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(delay)
            GPIO.output(led_pin, GPIO.LOW)
            time.sleep(delay)
            print(" ✓")
        
        print("✅ Blink test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    import sys
    
    print("LED Hardware Test")
    print("=" * 20)
    print("1. Always ON test (30s)")
    print("2. Blink test (10 times)")
    print("3. Custom duration")
    print("")
    
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
            test_led_always_on(duration=duration)
        except ValueError:
            print("Usage: python3 test_led_always_on.py [duration_seconds]")
    else:
        # Interactive mode
        choice = input("Choose test (1/2/3): ").strip()
        
        if choice == "1":
            test_led_always_on()
        elif choice == "2":
            test_led_blink()
        elif choice == "3":
            try:
                duration = int(input("Enter duration (seconds): "))
                test_led_always_on(duration=duration)
            except ValueError:
                print("Invalid duration!")
        else:
            print("Invalid choice!")
