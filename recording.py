import threading
import wave
import numpy as np
from datetime import datetime

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.sample_rate = None
        self.lock = threading.Lock()

    def start(self, sample_rate):
        with self.lock:
            if self.recording:
                return False

            self.recording = True
            self.frames = []
            self.sample_rate = sample_rate

        return True

    def add_frame(self, audio):
        with self.lock:
            if self.recording:
                self.frames.append(audio.copy())

    def stop(self, filename=f"recording-{datetime.now()}.wav"):
        for char in ["<", ">", ":", "\"", "/", "\\", "|", "?", "*"]:
            filename = filename.replace(char, "-")
        with self.lock:
            if not self.recording:
                return None

            self.recording = False

            audio = np.concatenate(self.frames)
            self.frames = []

        audio = audio.astype(np.int16)

        with wave.open(filename, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(audio.tobytes())

        return filename


recorder = AudioRecorder()