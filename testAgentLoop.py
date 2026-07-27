import ollama
from tools import tools
from system_prompt import SYSTEM_PROMPT
from callbacks import Callbacks
import numpy as np
import sounddevice as sd
import subprocess
import json
import threading
import queue
import os
import sys
import speech_recognition as sr
import io
import scipy.io.wavfile as wav
import time

from dotenv import load_dotenv
load_dotenv()

TARGET_RATE = 16000
DEFAULT_MODEL = "qwen2.5"

VOICE = os.getenv("VOICE_PATH")
# If PIPER_PATH isn't set, assume the system-wide 'piper' command (Linux)
PIPER = os.getenv("PIPER_PATH", "piper") 

USER_AUDIO = True
try:
    rec = sr.Recognizer()
except Exception as e:
    print(e)
    USER_AUDIO = False

AGENT_AUDIO = True if VOICE and PIPER else False
AGENT_SPEAKING = False

user_speech_queue = queue.Queue()
agent_speech_queue = queue.Queue()

def get_sample_rate(voice_path=VOICE):
    config_path = voice_path + ".json"
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["audio"]["sample_rate"]

def audio_worker(sample_rate):
    global AGENT_SPEAKING

    while True:
        audio = agent_speech_queue.get()
        if audio is None:
            AGENT_SPEAKING = False
            agent_speech_queue.task_done()
            continue
        data = np.frombuffer(audio, dtype=np.int16)
        sd.play(data, samplerate=sample_rate)
        sd.wait()

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

def listen():
    global AGENT_SPEAKING

    SAMPLE_RATE = 16000
    BLOCK_SIZE = 1024
    THRESHOLD = 200          # Increase if it triggers too easily
    SILENCE_TIME = 0.8       # Seconds of silence before sending

    silent_blocks_required = int(SILENCE_TIME * SAMPLE_RATE / BLOCK_SIZE)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SIZE
    )

    stream.start()

    recording = False
    frames = []
    silent_blocks = 0

    print("Listening...")

    while True:
        if AGENT_SPEAKING:
            recording = False
            frames.clear()
            silent_blocks = 0
            time.sleep(0.1)
            continue

        audio, overflowed = stream.read(BLOCK_SIZE)

        if overflowed:
            print("Audio overflow!")

        audio = audio.flatten()

        volume = np.abs(audio).mean()
        # print(volume)   # Uncomment for threshold tuning

        if not recording:
            if volume > THRESHOLD:
                print("Recording...")
                recording = True
                frames = [audio.copy()]
                silent_blocks = 0
            continue

        # Already recording
        frames.append(audio.copy())

        if volume > THRESHOLD:
            silent_blocks = 0
        else:
            silent_blocks += 1

        if silent_blocks < silent_blocks_required:
            continue

        print("Processing...")

        recording = False
        silent_blocks = 0

        recording_audio = np.concatenate(frames)
        frames.clear()

        buffer = io.BytesIO()
        wav.write(buffer, SAMPLE_RATE, recording_audio)
        buffer.seek(0)

        try:
            with sr.AudioFile(buffer) as source:
                audio_data = rec.record(source)

            text = rec.recognize_google(
                audio_data,
                language="en-US"
            ).strip()

            if text:
                user_speech_queue.put(text)

        except sr.UnknownValueError:
            pass

        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")

        except Exception as e:
            print(e)

def start_listener():
    threading.Thread(target=listen, daemon=True).start()

class testCallbacks(Callbacks):
    def __init__(self, messages: list = None, speaking=True):
        self.messages = messages or []
        self.buffer = ""
        self.speaking = speaking
        self.thread = None
    
    def on_token(self, token: str):
        if self.speaking:
            self.buffer += token
            if len(self.buffer) > 100 and any(self.buffer.endswith(x) for x in [".", "!", "?", ";"]):
                self.speak()
        else:
            print(token, end="", flush=True)

    def on_tool_call_start(self, name, args):
        print(f"Calling tool {name} with {args}")
    
    def on_message(self, msg):
        self.messages.append(msg)

    def speak(self):
        if self.buffer:
            play_audio_async(get_audio_data(self.buffer))
            self.buffer = ""
    
    def on_complete(self):
        if self.speaking:
            self.speak()
            agent_speech_queue.put(None)
    
    def on_start(self):
        if self.speaking:
            global AGENT_SPEAKING
            AGENT_SPEAKING = True
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=audio_worker, args=(get_sample_rate(),), daemon=True)
                self.thread.start()

def run_agent(
    user_message: str,
    conversation_history: list,
    callbacks,
    tools: dict = tools,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: str = "",
):
    callbacks.on_start()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    callbacks.on_message({"role": "user", "content": user_message})

    if (not model_name):
        model_name = DEFAULT_MODEL

    full_response = ""

    ollama_tools = []
    for _, tool in tools.items():
        ollama_tools.append({
            "type": "function",
            "function": tool["schema"]
        })
    while True:
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            tools=ollama_tools,
            stream=True
        )

        tool_calls = []
        current_text = ""
        stream_error = None

        try:
            for chunk in stream:
                msg = chunk["message"]

                if "content" in msg and msg["content"]:
                    token = msg["content"]
                    current_text += token
                    callbacks.on_token(token)

                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_calls.append(tc)
                        args = tc["function"]["arguments"]
                        callbacks.on_tool_call_start(tc["function"]["name"], args)

        except Exception as e:
            stream_error = e
            if not current_text:
                raise e

        callbacks.on_message({"role": "assistant", "content": current_text})
        full_response += current_text

        if stream_error and not current_text:
            full_response = "I couldn't generate a response. Try rephrasing your message."
            break

        if not tool_calls:
            messages.append({"role": "assistant", "content": current_text})
            break

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            args = tc["function"]["arguments"]

            tool_def = tools.get(tool_name)
            if not tool_def:
                continue

            if tool_def.get("needs_approval"):
                approved = callbacks.on_tool_approval(tool_name, args)
                if not approved:
                    return full_response

            try:
                result = tool_def["fn"](**args)
            except Exception as e:
                result = str(e)
            callbacks.on_tool_call_end(tool_name, result)

            messages.append({
                "role": "assistant",
                "tool_calls": [tc]
            })
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": result
            })
            callbacks.on_message({
                "role": "assistant",
                "tool_calls": [tc]
            })
            callbacks.on_message({
                "role": "tool",
                "name": tool_name,
                "content": result
            })

    callbacks.on_complete()
    return full_response

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else ""

    callbacks = testCallbacks(speaking=AGENT_AUDIO)
    if USER_AUDIO:
        start_listener()
        while True:
            try:
                text = user_speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            print("You:", text, flush=True)
            run_agent(text, callbacks.messages, callbacks, model_name=model_name)
    else:
        while (inp:=input("> ")):
            run_agent(inp, callbacks.messages, callbacks, model_name=model_name)
            print()