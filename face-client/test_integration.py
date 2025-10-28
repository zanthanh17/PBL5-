#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra tích hợp sensor + TTS + face recognition
Chạy trên máy desktop (không cần camera thật)
"""

import time
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tts_speaker import TTSSpeaker
from sensor_controller import SensorController

# ============= Logging Setup =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_integration():
    """Test tích hợp TTS + Sensor"""
    print("🧪 Testing TTS + Sensor Integration")
    print("=" * 50)
    
    # Initialize TTS
    print("1. Initializing TTS Speaker...")
    tts = TTSSpeaker(enabled=True, volume=100, speed=150, cooldown=2.0)
    
    # Initialize Sensor (dummy mode)
    print("2. Initializing Sensor Controller...")
    sensor = SensorController(
        trig_pin=23,
        echo_pin=24, 
        led_pin=18,
        trigger_distance=10.0,
        led_on_duration=5.0,
        check_interval=1.0
    )
    
    # Callbacks
    def on_person_detected(distance):
        print(f"🔆 LED ON - Distance {distance:.2f}cm > 10cm")
        tts.speak_custom("Xin chào")
    
    def on_person_left():
        print(f"💡 LED OFF - Distance <= 10cm")
    
    sensor.set_on_person_detected(on_person_detected)
    sensor.set_on_person_left(on_person_left)
    
    # Start sensor
    print("3. Starting sensor monitoring...")
    sensor.start()
    
    print("\n📋 Test Instructions:")
    print("- Simulate person at distance > 10cm (LED should turn ON)")
    print("- Simulate person at distance <= 10cm (LED should turn OFF)")
    print("- Press Ctrl+C to stop")
    print("\n🎯 Expected behavior:")
    print("- LED ON when distance > 10cm → TTS says 'Xin chào'")
    print("- LED OFF when distance <= 10cm")
    print("- LED stays ON for 5 seconds after detection")
    print("")
    
    try:
        # Simulate sensor readings
        test_distances = [50.0, 8.0, 25.0, 5.0, 30.0, 2.0, 40.0]
        
        for i, distance in enumerate(test_distances):
            print(f"\n--- Test {i+1}: Distance = {distance}cm ---")
            
            # Simulate sensor reading
            if distance > 10.0:
                print("🔆 Simulating: Person far away")
                if sensor.led:
                    sensor.led.on()
                if sensor.on_person_detected:
                    sensor.on_person_detected(distance)
            else:
                print("💡 Simulating: Person close")
                if sensor.led:
                    sensor.led.off()
                if sensor.on_person_left:
                    sensor.on_person_left()
            
            time.sleep(3)  # Wait between tests
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test stopped by user")
    
    finally:
        print("\n🧹 Cleaning up...")
        sensor.stop()
        sensor.cleanup()
        tts.stop()
        print("✅ Test completed!")

if __name__ == "__main__":
    test_integration()
