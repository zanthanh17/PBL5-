#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor Controller for HC-SR04 Ultrasonic + LED Control via MOSFET
Phát hiện người lại gần và bật đèn LED
"""

import time
import logging
from threading import Thread, Event
from typing import Callable, Optional

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO not available - Sensor controller disabled")

logger = logging.getLogger(__name__)


class HCSR04Sensor:
    """HC-SR04 Ultrasonic Distance Sensor"""
    
    def __init__(self, trig_pin: int, echo_pin: int, max_distance: float = 4.0):
        """
        Initialize HC-SR04 Sensor
        
        Args:
            trig_pin: GPIO pin cho TRIG (BCM numbering)
            echo_pin: GPIO pin cho ECHO (BCM numbering)
            max_distance: Khoảng cách tối đa đo được (m), default 4m
        """
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.max_distance = max_distance
        self.timeout = max_distance / 343.0 * 2  # Timeout based on max distance
        
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trig_pin, GPIO.LOW)
            time.sleep(0.1)  # Sensor settle time
            logger.info(f"HC-SR04 initialized (TRIG={trig_pin}, ECHO={echo_pin})")
    
    def get_distance(self) -> Optional[float]:
        """
        Đo khoảng cách (cm)
        
        Returns:
            Khoảng cách (cm) hoặc None nếu lỗi
        """
        if not GPIO_AVAILABLE:
            return None
        
        try:
            # Send trigger pulse
            GPIO.output(self.trig_pin, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.trig_pin, GPIO.LOW)
            
            # Wait for echo start
            pulse_start = time.time()
            timeout_start = time.time()
            while GPIO.input(self.echo_pin) == GPIO.LOW:
                pulse_start = time.time()
                if pulse_start - timeout_start > self.timeout:
                    return None
            
            # Wait for echo end
            pulse_end = time.time()
            timeout_start = time.time()
            while GPIO.input(self.echo_pin) == GPIO.HIGH:
                pulse_end = time.time()
                if pulse_end - timeout_start > self.timeout:
                    return None
            
            # Calculate distance
            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150  # Speed of sound = 343m/s
            distance = round(distance, 2)
            
            # Validate distance
            if 2 <= distance <= self.max_distance * 100:
                return distance
            return None
            
        except Exception as e:
            logger.error(f"HC-SR04 error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup GPIO"""
        # Note: GPIO.cleanup() sẽ được gọi bởi SensorController


class LEDController:
    """LED Controller via MOSFET"""
    
    def __init__(self, led_pin: int):
        """
        Initialize LED Controller
        
        Args:
            led_pin: GPIO pin cho MOSFET gate (BCM numbering)
        """
        self.led_pin = led_pin
        self.is_on = False
        
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.output(self.led_pin, GPIO.LOW)
            logger.info(f"LED Controller initialized (PIN={led_pin})")
    
    def on(self):
        """Bật LED"""
        if GPIO_AVAILABLE and not self.is_on:
            GPIO.output(self.led_pin, GPIO.HIGH)
            self.is_on = True
            logger.info("LED ON")
    
    def off(self):
        """Tắt LED"""
        if GPIO_AVAILABLE and self.is_on:
            GPIO.output(self.led_pin, GPIO.LOW)
            self.is_on = False
            logger.info("LED OFF")
    
    def toggle(self):
        """Toggle LED"""
        if self.is_on:
            self.off()
        else:
            self.on()
    
    def blink(self, times: int = 3, delay: float = 0.2):
        """
        Nháy LED
        
        Args:
            times: Số lần nháy
            delay: Thời gian delay giữa các lần (giây)
        """
        for _ in range(times):
            self.on()
            time.sleep(delay)
            self.off()
            time.sleep(delay)
    
    def cleanup(self):
        """Cleanup và tắt LED"""
        self.off()


class SensorController:
    """Controller cho HC-SR04 + LED"""
    
    def __init__(self, 
                 trig_pin: int = 23,
                 echo_pin: int = 24,
                 led_pin: int = 18,
                 trigger_distance: float = 100.0,
                 led_on_duration: float = 10.0,
                 check_interval: float = 0.2):
        """
        Initialize Sensor Controller
        
        Args:
            trig_pin: GPIO pin cho HC-SR04 TRIG (default 23)
            echo_pin: GPIO pin cho HC-SR04 ECHO (default 24)
            led_pin: GPIO pin cho LED MOSFET (default 18)
            trigger_distance: Khoảng cách kích hoạt LED (cm), default 100cm
            led_on_duration: Thời gian bật LED (giây), default 10s
            check_interval: Thời gian giữa các lần đo (giây), default 0.2s
        """
        self.trigger_distance = trigger_distance
        self.led_on_duration = led_on_duration
        self.check_interval = check_interval
        
        # Initialize components
        self.sensor = HCSR04Sensor(trig_pin, echo_pin) if GPIO_AVAILABLE else None
        self.led = LEDController(led_pin) if GPIO_AVAILABLE else None
        
        # State
        self.enabled = GPIO_AVAILABLE
        self.running = False
        self.stop_event = Event()
        self.worker_thread = None
        self.led_off_timer = None
        
        # Callbacks
        self.on_person_detected = None
        self.on_person_left = None
        
        if self.enabled:
            logger.info(f"Sensor Controller initialized (trigger={trigger_distance}cm)")
        else:
            logger.warning("Sensor Controller disabled (GPIO not available)")
    
    def set_on_person_detected(self, callback: Callable):
        """
        Set callback khi phát hiện người
        
        Args:
            callback: Function(distance: float) -> None
        """
        self.on_person_detected = callback
    
    def set_on_person_left(self, callback: Callable):
        """
        Set callback khi người rời đi
        
        Args:
            callback: Function() -> None
        """
        self.on_person_left = callback
    
    def start(self):
        """Start sensor monitoring"""
        if not self.enabled:
            return
        
        if self.running:
            logger.warning("Sensor Controller already running")
            return
        
        self.running = True
        self.stop_event.clear()
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("Sensor Controller started")
    
    def stop(self):
        """Stop sensor monitoring"""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        
        if self.led:
            self.led.off()
        
        logger.info("Sensor Controller stopped")
    
    def _worker(self):
        """Worker thread để monitor sensor"""
        person_detected = False
        last_detection_time = 0
        check_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Đo khoảng cách
                distance = self.sensor.get_distance()
                check_count += 1
                
                if distance is not None:
                    # Log distance để user biết sensor đang hoạt động
                    if self.check_interval >= 1.0:  # Chỉ log khi interval >= 1s
                        logger.info(f"[Check #{check_count}] Distance: {distance:.2f}cm")
                    else:
                        logger.debug(f"Distance: {distance}cm")
                    
                    if distance <= self.trigger_distance:
                        # Phát hiện người
                        if not person_detected:
                            person_detected = True
                            last_detection_time = time.time()
                            logger.info(f">>> Person detected at {distance}cm - LED ON")
                            
                            # Bật LED
                            if self.led:
                                self.led.on()
                            
                            # Callback
                            if self.on_person_detected:
                                try:
                                    self.on_person_detected(distance)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")
                        else:
                            # Vẫn còn người, update timer
                            last_detection_time = time.time()
                            logger.debug(f"Person still present at {distance}cm")
                    
                    else:
                        # Không phát hiện người
                        if person_detected:
                            # Kiểm tra timeout
                            elapsed = time.time() - last_detection_time
                            remaining = self.led_on_duration - elapsed
                            
                            if elapsed > self.led_on_duration:
                                person_detected = False
                                logger.info(">>> Person left - LED OFF")
                                
                                # Tắt LED
                                if self.led:
                                    self.led.off()
                                
                                # Callback
                                if self.on_person_left:
                                    try:
                                        self.on_person_left()
                                    except Exception as e:
                                        logger.error(f"Callback error: {e}")
                            else:
                                logger.debug(f"Waiting for person to leave (LED OFF in {remaining:.1f}s)")
                else:
                    logger.warning("Sensor reading failed (out of range?)")
                
                # Sleep
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Sensor worker error: {e}")
                time.sleep(1)
    
    def cleanup(self):
        """Cleanup GPIO"""
        self.stop()
        if GPIO_AVAILABLE:
            GPIO.cleanup()
            logger.info("GPIO cleanup completed")


# ============= Test Functions =============

def test_sensor():
    """Test HC-SR04 sensor"""
    print("Testing HC-SR04 Sensor...")
    print("WIRING:")
    print("  TRIG -> GPIO 23")
    print("  ECHO -> GPIO 24")
    print("")
    
    sensor = HCSR04Sensor(trig_pin=23, echo_pin=24)
    
    try:
        for i in range(10):
            distance = sensor.get_distance()
            if distance:
                print(f"Distance: {distance} cm")
            else:
                print("Out of range")
            time.sleep(0.5)
    finally:
        GPIO.cleanup()


def test_led():
    """Test LED controller"""
    print("Testing LED Controller...")
    print("WIRING:")
    print("  MOSFET Gate -> GPIO 18")
    print("")
    
    led = LEDController(led_pin=18)
    
    try:
        print("LED ON")
        led.on()
        time.sleep(2)
        
        print("LED OFF")
        led.off()
        time.sleep(1)
        
        print("LED Blink 5 times")
        led.blink(times=5, delay=0.2)
        
    finally:
        led.cleanup()
        GPIO.cleanup()


def test_controller(check_interval: float = 5.0):
    """
    Test full sensor controller
    
    Args:
        check_interval: Thời gian giữa các lần đo (giây), default 5.0s
    """
    print("Testing Sensor Controller...")
    print("WIRING:")
    print("  HC-SR04 TRIG -> GPIO 23")
    print("  HC-SR04 ECHO -> GPIO 24")
    print("  MOSFET Gate  -> GPIO 18")
    print("")
    print(f"Check interval: {check_interval}s")
    print("Move your hand in front of sensor (< 100cm)...")
    print("LED should turn on when person detected")
    print("Press Ctrl+C to stop")
    print("")
    
    def on_detected(distance):
        print(f">>> [{time.strftime('%H:%M:%S')}] Person detected at {distance}cm - LED ON")
    
    def on_left():
        print(f">>> [{time.strftime('%H:%M:%S')}] Person left - LED OFF")
    
    controller = SensorController(
        trig_pin=23,
        echo_pin=24,
        led_pin=18,
        trigger_distance=100.0,
        led_on_duration=5.0,
        check_interval=check_interval
    )
    
    controller.set_on_person_detected(on_detected)
    controller.set_on_person_left(on_left)
    controller.start()
    
    try:
        print(f"Monitoring started (checking every {check_interval}s)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "sensor":
            test_sensor()
        elif sys.argv[1] == "led":
            test_led()
        elif sys.argv[1] == "controller":
            # Check if interval is provided
            interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
            test_controller(check_interval=interval)
        else:
            # Try to parse as interval for backward compatibility
            try:
                interval = float(sys.argv[1])
                test_controller(check_interval=interval)
            except ValueError:
                print("Usage:")
                print("  python3 sensor_controller.py                    # Test với interval 5s")
                print("  python3 sensor_controller.py 5                  # Test với interval 5s")
                print("  python3 sensor_controller.py controller 3       # Test với interval 3s")
                print("  python3 sensor_controller.py sensor             # Test chỉ sensor")
                print("  python3 sensor_controller.py led                # Test chỉ LED")
    else:
        # Default: test với interval 5s
        test_controller(check_interval=5.0)

