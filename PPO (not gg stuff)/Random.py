
import subprocess
import sys
import os
import time
import random
import threading
from typing import Optional

def install_pynput():
    python_exe = sys.executable
    print("Installing pynput into Blender's Python...")
    try:
        subprocess.check_call([
            python_exe, "-m", "pip", "install", "pynput", "--quiet"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("pynput installed!")
    except Exception as e:
        print(f"Failed: {e}")
        print("Run Blender as Admin or try manually.")
        return False
    return True

try:
    from pynput.keyboard import Controller, Key
except ImportError:
    print("pynput not found. Installing...")
    if install_pynput():
        from pynput.keyboard import Controller, Key
    else:
        raise

print("pynput ready!")



class RandomKeyPresser:
    def __init__(self):
        self.keyboard = Controller()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Key mappings
        self.keys = {
            'W': 'W',   # Accelerate
            'S': 'S',   # Brake / Reverse
            'A': 'A',   # Left
            'D': 'D',   # Right
        }

    def press_and_hold(self, key_char: str, duration: float):
        """Press and hold a key for given seconds"""
        print(f"Pressing {key_char} for {duration:.1f}s")
        self.keyboard.press(key_char)
        time.sleep(duration)
        self.keyboard.release(key_char)

    def random_action(self):
        """Pick random key + random hold time (1–6 sec)"""
        key = random.choice(list(self.keys.keys()))
        duration = random.uniform(1.0, 6.0)
        self.press_and_hold(key, duration)

    def start(self):
        if self.running:
            return
        self.running = True
        print("\nRANDOM KEY PRESSER STARTED!")
        print("W = Accelerate | S = Brake/Reverse | A/D = Steering")
        print("Stop with Ctrl+C in Blender Console\n")

        def loop():
            try:
                while self.running:
                    self.random_action()
                    # Short pause between actions (0.1–1 sec)
                    time.sleep(random.uniform(0.1, 1.0))
            except KeyboardInterrupt:
                print("\nStopping key presser...")
                self.stop()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("Key presser stopped.")

presser = RandomKeyPresser()
presser.start()

# Keep script alive
print("Script running in background. Use Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    presser.stop()