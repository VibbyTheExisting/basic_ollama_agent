from callbacks import Callbacks, ListenerCallbacks
import numpy as np
import sounddevice as sd
import subprocess
import json
import threading
import queue
import os
import sys
import time
from run import run_agent
from listener import start_listener, clear_history

from dotenv import load_dotenv
load_dotenv()

VOICE = os.getenv("VOICE_PATH")
# If PIPER_PATH isn't set, assume the system-wide 'piper' command (Linux python implementation)
PIPER = os.getenv("PIPER_PATH", "piper") 

USER_AUDIO = True
STOP_SPEAKING = threading.Event()
CANCEL_RESPONSE = threading.Event()

AGENT_AUDIO = True if VOICE and PIPER else False
AGENT_SPEAKING = False

user_speech_queue = queue.Queue()
agent_speech_queue = queue.Queue()

def stop_audio():
    global AGENT_SPEAKING
    CANCEL_RESPONSE.set()
    STOP_SPEAKING.set()
    AGENT_SPEAKING = False

def check_stop_command(text):
    global AGENT_SPEAKING

    stop_words = [
        "stop",
        "cancel",
        "quiet",
        "shut up",
        "shush"
    ]

    text = text.lower()

    if any(word in text for word in stop_words):
        print("Stopping...")
        stop_audio()
        return True

    return False

def get_sample_rate(voice_path=VOICE):
    config_path = voice_path + ".json"
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["audio"]["sample_rate"]

def audio_worker(sample_rate):
    global AGENT_SPEAKING

    while True:
        try:
            audio = agent_speech_queue.get()
            if audio is None:
                AGENT_SPEAKING = False
                continue
            data = np.frombuffer(audio, dtype=np.int16)
            sd.play(data, samplerate=sample_rate, blocking=False)

            while True:
                if STOP_SPEAKING.is_set():
                    print("Stopping playback")
                    sd.stop()
                    while not agent_speech_queue.empty():
                        try:
                            agent_speech_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    break

                # Check if playback is finished
                stream = sd.get_stream()
                if stream is None or not stream.active:
                    break

                time.sleep(0.05)

            sd.wait()
            clear_history()
        except Exception as e:
            print(e)

def play_audio_async(audio):
    agent_speech_queue.put(audio)

def get_audio_data(text: str, voice_path=VOICE):
    # LINUX PORT FIX: Ensures subprocess uses the binary path or system command strings correctly
    process = subprocess.Popen(
        [PIPER, "--quiet", "--model", voice_path, "--output_raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    audio, _ = process.communicate(text.encode("utf-8"))
    return audio

class mainListenerCallbacks(ListenerCallbacks):
    def __init__(self):
        self.interrupt_mode = False

    def start(self):
        self.interrupt_mode = AGENT_SPEAKING

    def on_text(self, text):
        if self.interrupt_mode:
            check_stop_command(text)
        else:
            user_speech_queue.put(text)


class mainCallbacks(Callbacks):
    def __init__(self, messages: list = None, speaking=True):
        self.messages = messages or []
        self.buffer = ""
        self.speaking = speaking
        self.thread = None
    
    def on_token(self, token: str):
        if self.speaking:
            if CANCEL_RESPONSE.is_set():
                return

            self.buffer += token

            if len(self.buffer) > 100 and any(
                self.buffer.endswith(x)
                for x in [".", "!", "?", ";"]
            ):
                self.speak()

    def on_tool_call_start(self, name, args):
        print(f"Calling tool {name} with {args}")
    
    def on_message(self, msg):
        self.messages.append(msg)

    def speak(self):
        if self.buffer and not CANCEL_RESPONSE.is_set():
            play_audio_async(get_audio_data(self.buffer))

        self.buffer = ""
    
    def on_complete(self):
        if self.speaking:
            self.speak()
            agent_speech_queue.put(None)
    
    def on_start(self):
        if self.speaking:
            global AGENT_SPEAKING, STOP_SPEAKING, CANCEL_RESPONSE
            AGENT_SPEAKING = True
            STOP_SPEAKING.clear()
            CANCEL_RESPONSE.clear()
            self.buffer = ""
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=audio_worker, args=(get_sample_rate(),), daemon=True)
                self.thread.start()

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else ""

    callbacks = mainCallbacks(speaking=AGENT_AUDIO)
    if USER_AUDIO:
        listen_callback = mainListenerCallbacks()
        start_listener(listen_callback)
        while True:
            try:
                text = user_speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            print("You:", text, flush=True)
            run_agent(text, callbacks.messages, callbacks, model_name=model_name, cancel_event=CANCEL_RESPONSE)
    else:
        while (inp:=input("> ")):
            run_agent(inp, callbacks.messages, callbacks, model_name=model_name)
            print()