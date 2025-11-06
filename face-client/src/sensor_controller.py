#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensor Controller for HC-SR04 Ultrasonic Sensor + LED
- Đo khoảng cách bằng HC-SR04
- Nếu khoảng cách < 50cm → bật LED
- LED bật 15 giây, trong thời gian này KHÔNG đo khoảng cách
- Sau 15 giây, LED tắt và tiếp tục đo
"""

import time
import threading
import logging

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO not available - Sensor Controller will not work")

logger = logging.getLogger(__name__)


class SensorController:
    """
    Controller cho HC-SR04 Ultrasonic Sensor + LED
    
    Logic:
    - Đo khoảng cách liên tục (khi LED không bật)
    - Nếu khoảng cách < trigger_distance → bật LED
    - LED bật trong led_on_duration giây
    - Trong khi LED bật, TẠM DỪNG đo khoảng cách
    - Sau khi LED tắt, tiếp tục đo khoảng cách
    """
    
    def __init__(
        self,
        trig_pin: int = 23,
        echo_pin: int = 24,
        led_pin: int = 18,
        trigger_distance: float = 50.0,  # cm
        led_on_duration: float = 15.0,   # giây
        check_interval: float = 0.2,      # giây giữa các lần đo
    ):
        """
        Args:
            trig_pin: GPIO pin cho HC-SR04 TRIG (BCM)
            echo_pin: GPIO pin cho HC-SR04 ECHO (BCM)
            led_pin: GPIO pin cho LED/MOSFET Gate (BCM)
            trigger_distance: Khoảng cách ngưỡng (cm) - LED bật khi < ngưỡng
            led_on_duration: Thời gian LED bật (giây)
            check_interval: Thời gian giữa các lần đo (giây)
        """
        if not GPIO_AVAILABLE:
            raise RuntimeError("RPi.GPIO not available. Install: pip install RPi.GPIO")
        
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.led_pin = led_pin
        self.trigger_distance = trigger_distance
        self.led_on_duration = led_on_duration
        self.check_interval = check_interval
        
        # State
        self._running = False
        self._led_on = False
        self._led_off_time = 0.0  # Thời điểm LED sẽ tắt
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_person_detected = None
        self._on_person_left = None
        
        # Thread
        self._thread = None
        
        # Setup GPIO
        self._setup_gpio()
        
        logger.info(f"SensorController initialized: trig={trig_pin}, echo={echo_pin}, led={led_pin}")
        logger.info(f"Trigger distance: {trigger_distance}cm, LED duration: {led_on_duration}s")
    
    def _setup_gpio(self):
        """Setup GPIO pins"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
        GPIO.setup(self.led_pin, GPIO.OUT)
        
        # Initial state
        GPIO.output(self.trig_pin, GPIO.LOW)
        GPIO.output(self.led_pin, GPIO.LOW)
        
        # Wait for sensor to settle
        time.sleep(0.1)
    
    def _measure_distance(self) -> float:
        """
        Đo khoảng cách bằng HC-SR04 (cm)
        
        Returns:
            Khoảng cách (cm), hoặc -1 nếu lỗi/timeout
        """
        try:
            # Send trigger pulse
            GPIO.output(self.trig_pin, GPIO.HIGH)
            time.sleep(0.00001)  # 10 microseconds
            GPIO.output(self.trig_pin, GPIO.LOW)
            
            # Wait for echo to start
            timeout = 0.03  # 30ms timeout
            start_time = time.time()
            while GPIO.input(self.echo_pin) == GPIO.LOW:
                if time.time() - start_time > timeout:
                    return -1
                time.sleep(0.0001)
            
            # Measure echo duration
            pulse_start = time.time()
            while GPIO.input(self.echo_pin) == GPIO.HIGH:
                if time.time() - pulse_start > timeout:
                    return -1
                time.sleep(0.0001)
            
            pulse_end = time.time()
            pulse_duration = pulse_end - pulse_start
            
            # Calculate distance (speed of sound = 34300 cm/s)
            # Distance = (pulse_duration * speed) / 2
            distance = (pulse_duration * 34300) / 2
            
            # Limit to reasonable range (2cm - 400cm)
            if distance < 2 or distance > 400:
                return -1
            
            return distance
            
        except Exception as e:
            logger.error(f"Error measuring distance: {e}")
            return -1
    
    def _turn_led_on(self):
        """Bật LED"""
        with self._lock:
            if not self._led_on:
                GPIO.output(self.led_pin, GPIO.HIGH)
                self._led_on = True
                self._led_off_time = time.time() + self.led_on_duration
                logger.info(f"LED ON (will turn OFF in {self.led_on_duration}s)")
    
    def _turn_led_off(self):
        """Tắt LED"""
        with self._lock:
            if self._led_on:
                GPIO.output(self.led_pin, GPIO.LOW)
                self._led_on = False
                self._led_off_time = 0.0
                logger.info("LED OFF")
    
    def _check_led_timeout(self):
        """Kiểm tra và tắt LED nếu hết thời gian"""
        should_turn_off = False
        with self._lock:
            if self._led_on and time.time() >= self._led_off_time:
                should_turn_off = True
        
        if should_turn_off:
            self._turn_led_off()
            if self._on_person_left:
                try:
                    self._on_person_left()
                except Exception as e:
                    logger.error(f"Error in on_person_left callback: {e}")
    
    def _main_loop(self):
        """Main loop trong thread riêng"""
        logger.info("Sensor Controller thread started")
        
        while self._running:
            try:
                # Kiểm tra LED timeout trước
                self._check_led_timeout()
                
                # Nếu LED đang bật, KHÔNG đo khoảng cách (tạm dừng)
                with self._lock:
                    led_on = self._led_on
                
                if led_on:
                    # LED đang bật → chờ đến khi tắt
                    time.sleep(0.1)
                    continue
                
                # LED không bật → đo khoảng cách
                distance = self._measure_distance()
                
                if distance > 0:
                    # Nếu khoảng cách < trigger_distance → bật LED
                    if distance < self.trigger_distance:
                        self._turn_led_on()
                        if self._on_person_detected:
                            try:
                                self._on_person_detected(distance)
                            except Exception as e:
                                logger.error(f"Error in on_person_detected callback: {e}")
                
                # Chờ check_interval trước khi đo lần tiếp theo
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in sensor loop: {e}")
                time.sleep(0.5)
        
        logger.info("Sensor Controller thread stopped")
    
    def set_on_person_detected(self, callback):
        """
        Set callback khi phát hiện người (khoảng cách < trigger_distance)
        
        Args:
            callback: Function nhận 1 argument (distance: float)
        """
        self._on_person_detected = callback
    
    def set_on_person_left(self, callback):
        """
        Set callback khi LED tắt (người rời đi)
        
        Args:
            callback: Function không có argument
        """
        self._on_person_left = callback
    
    def start(self):
        """Bắt đầu đo khoảng cách trong thread riêng"""
        if self._running:
            logger.warning("Sensor Controller already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        logger.info("Sensor Controller started")
    
    def stop(self):
        """Dừng đo khoảng cách"""
        if not self._running:
            return
        
        self._running = False
        self._turn_led_off()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        logger.info("Sensor Controller stopped")
    
    def cleanup(self):
        """Cleanup GPIO và dừng thread"""
        self.stop()
        
        try:
            GPIO.output(self.led_pin, GPIO.LOW)
            GPIO.cleanup([self.trig_pin, self.echo_pin, self.led_pin])
            logger.info("GPIO cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up GPIO: {e}")
    
    def is_led_on(self) -> bool:
        """Kiểm tra LED có đang bật không"""
        with self._lock:
            return self._led_on
    
    def get_last_distance(self) -> float:
        """
        Lấy khoảng cách gần nhất (không implement cache, chỉ để tương lai)
        """
        # Có thể implement cache nếu cần
        return -1


# ============= Test Functions =============

def test_sensor():
    """Test sensor controller"""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    print("HC-SR04 + LED Test")
    print("=" * 40)
    print("Wiring:")
    print("  HC-SR04 TRIG → GPIO 23")
    print("  HC-SR04 ECHO → GPIO 24")
    print("  LED/MOSFET Gate → GPIO 18")
    print("")
    print("Logic:")
    print("  - Đo khoảng cách liên tục")
    print("  - Nếu < 50cm → bật LED 15s")
    print("  - Trong khi LED bật, KHÔNG đo khoảng cách")
    print("  - Sau 15s, LED tắt và tiếp tục đo")
    print("")
    print("Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        sensor = SensorController(
            trig_pin=23,
            echo_pin=24,
            led_pin=18,
            trigger_distance=50.0,
            led_on_duration=15.0,
            check_interval=0.2
        )
        
        def on_detected(distance):
            print(f"👤 Person detected at {distance:.1f}cm - LED ON")
        
        def on_left():
            print("👋 Person left - LED OFF, resuming measurement")
        
        sensor.set_on_person_detected(on_detected)
        sensor.set_on_person_left(on_left)
        sensor.start()
        
        # Keep running
        while True:
            time.sleep(1)
            if sensor.is_led_on():
                remaining = sensor._led_off_time - time.time()
                print(f"\r🔆 LED ON (turns OFF in {remaining:.1f}s)    ", end="", flush=True)
            else:
                print(f"\r📏 Measuring distance...    ", end="", flush=True)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if 'sensor' in locals():
            sensor.cleanup()


if __name__ == "__main__":
    test_sensor()
