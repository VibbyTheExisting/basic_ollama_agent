import io
import scipy.io.wavfile as wav
from collections import deque
import sounddevice as sd
import numpy as np
import threading
import speech_recognition as sr

rec = sr.Recognizer()
history = None
history_enabled = True

def clear_history():
    global history_enabled

    history_enabled = False
    if history is not None:
        history.clear()

    threading.Timer(0.3, enable_history).start()

def enable_history():
    global history_enabled
    history_enabled = True

def listen(callback: callable = lambda x: None):
    global history

    input_device = sd.query_devices(kind="input")
    SAMPLE_RATE = int(input_device["default_samplerate"])

    BLOCK_SIZE = 4096
    THRESHOLD = 150          # Needs to be tuned
    SILENCE_TIME = 0.8       # Silence duration before processing
    MIN_SECONDS = 0.3        # Ignore very short sounds
    PRE_ROLL_BLOCKS = 5      # Around 100-300ms

    silent_blocks_required = int(
        SILENCE_TIME * SAMPLE_RATE / BLOCK_SIZE
    )

    history = deque(maxlen=PRE_ROLL_BLOCKS)

    print(f"Listening at {SAMPLE_RATE}Hz...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SIZE
    ) as stream:

        recording = False
        frames = []
        silent_blocks = 0

        while True:

            audio, overflowed = stream.read(BLOCK_SIZE)

            if overflowed:
                print("Audio overflow")

            audio = audio.flatten()
            if history_enabled and not recording: history.append(audio.copy())

            volume = np.abs(audio).mean()
            # print(volume)  # Uncomment to tune THRESHOLD

            if not recording:
                if volume > THRESHOLD:
                    print("Recording...")
                    recording = True

                    # Include audio immediately before speech started
                    frames = list(history)

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
            history.clear()

            # Ignore accidental taps/clicks/noise
            if len(recording_audio) < SAMPLE_RATE * MIN_SECONDS:
                continue

            recording_audio = recording_audio.astype(np.int16)

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
                    callback(text)

            except sr.UnknownValueError:
                # Speech detected but not understood
                pass

            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")

            except Exception as e:
                print(f"Recognition error: {e}")

def start_listener(callback: callable):
    threading.Thread(target=listen, args=(callback,), daemon=True).start()