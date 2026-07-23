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
from scipy.signal import resample_poly
from math import gcd
import wave
import tempfile
from faster_whisper import WhisperModel

from dotenv import load_dotenv
load_dotenv()

TARGET_RATE = 16000
DEFAULT_MODEL = "qwen2.5"

VOICE = os.getenv("VOICE_PATH")
# If PIPER_PATH isn't set, assume the system-wide 'piper' command (Linux)
PIPER = os.getenv("PIPER_PATH", "piper")

USER_AUDIO = bool(os.getenv("USER_AUDIO", "True"))

model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8"
)

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

def transcribe(audio, samplerate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name

    with wave.open(filename, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # int16
        wav.setframerate(samplerate)
        wav.writeframes(audio.tobytes())

    segments, _ = model.transcribe(filename)

    text = "".join(
        segment.text for segment in segments
    )

    os.remove(filename)

    return text.strip()

def start_listener():

    input_device = sd.query_devices(kind="input")
    sample_rate = int(input_device["default_samplerate"])

    silence_threshold = 10
    silence_time = 2

    recording = False
    silence = 0
    frames = []

    def callback(indata, frames_count, time, status):
        nonlocal recording, silence, frames

        if AGENT_SPEAKING:
            return
        
        g = gcd(sample_rate, TARGET_RATE)
        up = TARGET_RATE // g
        down = sample_rate // g

        audio = np.frombuffer(indata, dtype=np.int16).copy()

        if sample_rate != TARGET_RATE:
            audio = resample_poly(audio, up, down)
            audio = audio.astype(np.int16)

        volume = np.abs(audio).mean()

        if recording:

            frames.append(audio)

            if volume < silence_threshold:
                silence += len(audio)
            else:
                silence = 0

            if silence > sample_rate * silence_time:

                recording = False

                speech = np.concatenate(frames)

                frames = []
                silence = 0

                threading.Thread(
                    target=transcribe_worker,
                    args=(speech, sample_rate),
                    daemon=True
                ).start()

        else:

            if volume > silence_threshold:
                recording = True
                frames = [audio]
                silence = 0

    def listen():
        stop = threading.Event()

        with sd.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
            blocksize=2048,
        ):
            while not stop.is_set():
                stop.wait(0.2)

    threading.Thread(target=listen, daemon=True).start()

def transcribe_worker(audio, samplerate):

    text = transcribe(audio, samplerate)

    if text:
        user_speech_queue.put(text)

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
        print("Ready.")
        while True:
            try:
                text = user_speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            print("You: ", text, flush=True)
            run_agent(text, callbacks.messages, callbacks, model_name=model_name)
    else:
        while (inp:=input("> ")):
            run_agent(inp, callbacks.messages, callbacks, model_name=model_name)
            print()