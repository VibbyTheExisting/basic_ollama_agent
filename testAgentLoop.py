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
from vosk import Model, KaldiRecognizer
import os

VOICE = os.getenv("VOICE_PATH")
PIPER = os.getenv("PIPER_PATH")
# VOSK = os.getenv("VOSK_PATH")

# model = Model(VOSK)
# recognizer = KaldiRecognizer(model, 16000)

# def callback(indata, frames, time, status):
#     if recognizer.AcceptWaveform(bytes(indata)):
#         result = json.loads(recognizer.Result())
#         print("Final:", result.get("text", ""))
#     else:
#         partial = json.loads(recognizer.PartialResult())
#         print("Partial:", partial.get("partial", ""))

# with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
#                        channels=1, callback=callback):
#     print("Listening...")
#     while True:
#         pass

# exit()

AGENT_SPEAKING = True if VOICE and PIPER else False

user_speech_queue = queue.Queue()
agent_speech_queue = queue.Queue()

def get_sample_rate(voice_path=VOICE):
    config_path = voice_path + ".json"
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["audio"]["sample_rate"]

def play_audio(raw_audio, sample_rate):
    audio = np.frombuffer(raw_audio, dtype=np.int16)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

def audio_worker(sample_rate):
    while (audio := agent_speech_queue.get()) or AGENT_SPEAKING:
        if audio is None:
            continue
        data = np.frombuffer(audio, dtype=np.int16)
        sd.play(data, samplerate=sample_rate)
        sd.wait()
        agent_speech_queue.task_done()

def play_audio_async(audio):
    agent_speech_queue.put(audio)

def get_audio_data(text: str, voice_path=VOICE):
    process = subprocess.Popen(
        [PIPER, "--quiet", "--model", voice_path, "--output_raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    audio, _ = process.communicate(text.encode("utf-8"))
    return audio

def start_listener(model_path):
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    def callback(indata, frames, time, status):
        volume = np.abs(indata).mean()

        if volume < 50 or AGENT_SPEAKING:  # tune this number
            return

        if recognizer.AcceptWaveform(bytes(indata)):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text:
                user_speech_queue.put(text)
        else:
            # Partial results
            pass

    def listen():
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=4000,
            dtype='int16',
            channels=1,
            callback=callback
        ):
            while True:
                pass

    threading.Thread(target=listen, daemon=True).start()

class testCallbacks(Callbacks):
    def __init__(self, messages: list = None, speaking=True):
        self.messages = messages or []
        self.buffer = ""
        self.speaking = speaking
    
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
        # print(msg)
        # if (msg.get("role") == "assistant" and msg.get("content")):
        #     play_audio(get_audio_data(msg["content"]), get_sample_rate(VOICE))

    def speak(self):
        if self.buffer:
            # play_audio(get_audio_data(self.buffer), get_sample_rate())
            play_audio_async(get_audio_data(self.buffer))
            self.buffer = ""
    
    def on_complete(self):
        if self.speaking:
            self.speak()
            global AGENT_SPEAKING
            AGENT_SPEAKING = False
    
    def on_start(self):
        if self.speaking:
            global AGENT_SPEAKING
            AGENT_SPEAKING = True
            threading.Thread(target=audio_worker, args=(get_sample_rate(),), daemon=True).start()

def run_agent(
    user_message: str,
    conversation_history: list,
    callbacks,
    tools: dict = tools,
    system_prompt: str = SYSTEM_PROMPT,
    model_name: str = "qwen2.5",
):
    # Build working message list
    callbacks.on_start()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})
    callbacks.on_message({"role": "user", "content": user_message})

    full_response = ""

    while True:
        # Convert tool definitions to Ollama schema
        ollama_tools = []
        for name, tool in tools.items():
            ollama_tools.append({
                "type": "function",
                "function": tool["schema"]
            })

        # Start streaming
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

                # Text delta
                if "content" in msg and msg["content"]:
                    token = msg["content"]
                    current_text += token
                    callbacks.on_token(token)

                # Tool call
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        tool_calls.append(tc)
                        args = tc["function"]["arguments"]
                        callbacks.on_tool_call_start(tc["function"]["name"], args)

        except Exception as e:
            stream_error = e
            if not current_text:
                raise e

        # Emit assistant message
        callbacks.on_message({"role": "assistant", "content": current_text})
        full_response += current_text

        # Recovery if no output
        if stream_error and not current_text:
            full_response = "I couldn't generate a response. Try rephrasing your message."
            break

        # If no tool calls, conversation ends
        if not tool_calls:
            messages.append({"role": "assistant", "content": current_text})
            break

        # Handle tool calls
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            args = tc["function"]["arguments"]

            tool_def = tools.get(tool_name)
            if not tool_def:
                continue

            # Optional approval
            if tool_def.get("needs_approval"):
                approved = callbacks.on_tool_approval(tool_name, args)
                if not approved:
                    return full_response

            # Execute tool
            try:
                result = tool_def["fn"](**args)
            except Exception as e:
                result = str(e)
            callbacks.on_tool_call_end(tool_name, result)

            # Insert tool result into messages
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
    # start_listener(VOSK)
    callbacks = testCallbacks(speaking=False)
    while (inp:=input("> ")):
        run_agent(inp, callbacks.messages, callbacks)
        print()
    # while True:
    #     text = user_speech_queue.get()
    #     run_agent(text, callbacks.messages, callbacks)