#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text-to-Speech Speaker for Face Recognition
Phát âm thanh khi nhận diện thành công qua loa 3.5mm
"""

import os
import logging
from pathlib import Path
from threading import Thread
from queue import Queue
import time

logger = logging.getLogger(__name__)

class TTSSpeaker:
    """Text-to-Speech speaker using espeak (lightweight for Pi)"""
    
    def __init__(self, enabled: bool = True, volume: int = 100, speed: int = 150, 
                 cooldown: float = 3.0):
        """
        Initialize TTS Speaker
        
        Args:
            enabled: Bật/tắt TTS
            volume: Âm lượng (0-200), default 100
            speed: Tốc độ nói (80-450 wpm), default 150
            cooldown: Thời gian chờ giữa 2 lần phát cùng message (giây)
        """
        self.enabled = enabled
        self.volume = volume
        self.speed = speed
        self.cooldown = cooldown
        self.queue = Queue()
        self.worker_thread = None
        self.last_spoken = {}  # Cache để tránh phát trùng
        
        if self.enabled:
            self._check_espeak()
            self._start_worker()
    
    def _check_espeak(self):
        """Kiểm tra espeak đã cài đặt chưa"""
        result = os.system("which espeak > /dev/null 2>&1")
        if result != 0:
            logger.warning("espeak not found! Installing...")
            logger.warning("Please run: sudo apt-get install -y espeak")
            self.enabled = False
        else:
            logger.info("espeak found, TTS enabled")
    
    def _start_worker(self):
        """Start worker thread để xử lý queue"""
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        logger.info("TTS worker thread started")
    
    def _worker(self):
        """Worker thread xử lý TTS queue"""
        while True:
            try:
                text = self.queue.get(timeout=1)
                if text is None:  # Stop signal
                    break
                self._speak_blocking(text)
                self.queue.task_done()
            except:
                continue
    
    def _speak_blocking(self, text: str):
        """
        Phát âm text (blocking) - OPTIMIZED
        
        Args:
            text: Văn bản cần phát âm
        """
        try:
            # Tối ưu: Dùng subprocess.run thay vì os.system (nhanh hơn)
            import subprocess
            
            # espeak command với các tối ưu:
            # -v vi+f3: Tiếng Việt female voice 3 (rõ ràng hơn)
            # -a: amplitude (volume) 0-200
            # -s: speed (words per minute) - tăng để nhanh hơn
            # -g: gap between words (ms) - giảm để nhanh hơn
            # --stdout: Output to stdout, pipe to aplay (giảm latency)
            
            cmd = [
                'espeak',
                '-v', 'vi+f3',  # Vietnamese female voice 3
                '-a', str(self.volume),
                '-s', str(self.speed + 20),  # Tăng tốc độ một chút
                '-g', '3',  # Giảm gap
                text
            ]
            
            # Run với timeout để tránh hang
            subprocess.run(cmd, timeout=5, stderr=subprocess.DEVNULL, check=False)
                
        except subprocess.TimeoutExpired:
            logger.warning("TTS timeout")
        except Exception as e:
            logger.error(f"TTS error: {e}")
    
    def speak(self, text: str):
        """
        Phát âm text (non-blocking) với cooldown
        
        Args:
            text: Văn bản cần phát âm
        """
        if not self.enabled:
            return
        
        # Kiểm tra cooldown để tránh phát trùng
        import time
        now = time.time()
        if text in self.last_spoken:
            if now - self.last_spoken[text] < self.cooldown:
                logger.debug(f"TTS cooldown: {text}")
                return
        
        # Update cache và thêm vào queue
        self.last_spoken[text] = now
        self.queue.put(text)
    
    def speak_welcome(self, name: str):
        """
        Chào mừng user
        
        Args:
            name: Tên user
        """
        message = f"Xin chào, {name}"
        self.speak(message)
        logger.info(f"TTS: {message}")
    
    def speak_checkin(self, name: str):
        """
        Thông báo check-in
        
        Args:
            name: Tên user
        """
        message = f"Check in thành công, {name}"
        self.speak(message)
        logger.info(f"TTS: {message}")
    
    def speak_checkout(self, name: str):
        """
        Thông báo check-out
        
        Args:
            name: Tên user
        """
        message = f"Check out thành công, {name}"
        self.speak(message)
        logger.info(f"TTS: {message}")
    
    def speak_rejected(self):
        """Thông báo từ chối"""
        message = "Không nhận diện được khuôn mặt"
        self.speak(message)
        logger.info(f"TTS: {message}")
    
    def speak_custom(self, message: str):
        """
        Phát âm message tùy chỉnh
        
        Args:
            message: Message cần phát
        """
        self.speak(message)
        logger.info(f"TTS: {message}")
    
    def stop(self):
        """Dừng TTS worker"""
        if self.worker_thread and self.worker_thread.is_alive():
            self.queue.put(None)  # Stop signal
            self.worker_thread.join(timeout=2)
            logger.info("TTS worker stopped")
    
    def set_volume(self, volume: int):
        """
        Điều chỉnh âm lượng
        
        Args:
            volume: Âm lượng (0-200)
        """
        self.volume = max(0, min(200, volume))
        logger.info(f"TTS volume set to {self.volume}")
    
    def set_speed(self, speed: int):
        """
        Điều chỉnh tốc độ nói
        
        Args:
            speed: Tốc độ (80-450 wpm)
        """
        self.speed = max(80, min(450, speed))
        logger.info(f"TTS speed set to {self.speed}")


# ============= Test Functions =============

def test_tts():
    """Test TTS functionality"""
    print("Testing TTS Speaker...")
    
    tts = TTSSpeaker(enabled=True, volume=100, speed=150)
    
    print("Test 1: Welcome message")
    tts.speak_welcome("Nguyễn Văn A")
    time.sleep(3)
    
    print("Test 2: Check-in message")
    tts.speak_checkin("Trần Thị B")
    time.sleep(3)
    
    print("Test 3: Check-out message")
    tts.speak_checkout("Lê Văn C")
    time.sleep(3)
    
    print("Test 4: Rejected message")
    tts.speak_rejected()
    time.sleep(3)
    
    print("Test 5: Custom message")
    tts.speak_custom("Hệ thống chấm công nhận diện khuôn mặt")
    time.sleep(3)
    
    tts.stop()
    print("Test completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    test_tts()

