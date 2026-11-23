import os
import subprocess
import time
import wave

from google import genai
from google.genai import types
import numpy as np
import openwakeword
import openwakeword.utils
from openwakeword.model import Model
import requests
from faster_whisper import WhisperModel
from gpiozero import Motor

# ---------- settings you may need to change ----------
MODEL = "gemini-3.8-flash"   # fast Gemini model for spoken replies
WAKE_WORD = "hey_jarvis"     # built-in openWakeWord options: hey_jarvis, hey_mycroft, alexa
WAKE_THRESHOLD = 0.5         # raise to 0.6 if it triggers accidentally
ELEVENLABS_VOICE_ID = "pqHfZKP75CvOlQylNhV4"  # Bill (premade voice; works on free tier)
ELEVENLABS_MODEL = "eleven_flash_v2_5"        # ultra-low latency
SILENCE_LEVEL = 500          # raise this if it never stops listening
MOUTH_LEVEL = 800            # lower this if the mouth stays shut
MOUTH_SPEED = 0.7            # 0 to 1
MIC_DEVICE = "plughw:Device,0"  # ALSA capture device from `arecord -l`
SYSTEM = (
    "You are Billy, a talking singing fish mounted on a wall plaque. "
    "Be warm, funny and brief. Answer in one or two short spoken sentences. "
    "Never use lists, markdown or emojis, because your words are spoken aloud."
)
# ------------------------------------------------------

mouth = Motor(forward=5, backward=6, enable=12)
body = Motor(forward=19, backward=26, enable=13)
stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")
if hasattr(openwakeword.utils, "download_models"):
    openwakeword.utils.download_models([WAKE_WORD])
oww = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
ai = genai.Client()          # reads GEMINI_API_KEY from the environment
history = []


def open_mic_stream():
    return subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE,
    )


def read_frame(proc, samples):
    raw = proc.stdout.read(samples * 2)
    if len(raw) < samples * 2:
        raise RuntimeError("Microphone stream ended unexpectedly.")
    return np.frombuffer(raw, dtype=np.int16)


def wait_for_wake_word():
    oww.reset()
    proc = open_mic_stream()
    try:
        while True:
            audio = read_frame(proc, 1280)
            scores = oww.predict(audio)
            if any(score > WAKE_THRESHOLD for score in scores.values()):
                return
    finally:
        proc.terminate()
        proc.wait()


def listen():
    """Record until about 1.3 seconds of quiet follows speech (10 second limit)."""
    proc = open_mic_stream()
    frames, quiet_frames, heard_speech = [], 0, False
    started = time.time()
    try:
        while time.time() - started < 10:
            frame = read_frame(proc, 512)
            frames.append(frame)
            loudness = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
            if loudness > SILENCE_LEVEL:
                heard_speech, quiet_frames = True, 0
            elif heard_speech:
                quiet_frames += 1
            if heard_speech and quiet_frames > 40:
                break
    finally:
        proc.terminate()
        proc.wait()
    return np.concatenate(frames).astype(np.float32) / 32768.0


def transcribe(audio):
    segments, _ = stt.transcribe(audio, language="en")
    return " ".join(s.text for s in segments).strip()


def ask(text):
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
    del history[:-10]  # keep only the recent conversation
    reply = ai.models.generate_content(
        model=MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=1024,
        ),
    )
    answer = (reply.text or "").strip()
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=answer)]))
    return answer


def speak(text):
    wav_path = "/tmp/billy_reply.wav"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=pcm_16000"
    response = requests.post(
        url,
        headers={
            "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
            "Content-Type": "application/json",
        },
        json={"text": text, "model_id": ELEVENLABS_MODEL},
        timeout=30,
    )
    response.raise_for_status()

    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(response.content)

    with wave.open(wav_path) as w:
        rate = w.getframerate()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    player = subprocess.Popen(["aplay", "-q", wav_path])
    step = int(rate * 0.08)  # look at the voice in 80 millisecond slices
    start = time.time()
    for i in range(0, len(samples), step):
        chunk = samples[i : i + step].astype(np.float32)
        if np.sqrt(np.mean(chunk ** 2)) > MOUTH_LEVEL:
            mouth.forward(MOUTH_SPEED)   # use backward() if your mouth opens that way
        else:
            mouth.stop()
        time.sleep(max(0, start + (i + step) / rate - time.time()))
    mouth.stop()
    player.wait()


def nod():
    body.forward(0.6)
    time.sleep(0.3)
    body.stop()


if __name__ == "__main__":
    print(f"Billy is ready (mic: {MIC_DEVICE}). Say 'Hey Jarvis'!")
    while True:
        try:
            wait_for_wake_word()
            print("Wake word heard! Listening...")
            nod()                  # shows you it heard the wake word
            text = transcribe(listen())
            if not text:
                print("(Didn't catch any speech.)")
                continue
            print("You:", text)
            answer = ask(text)
            print("Billy:", answer)
            speak(answer)
        except KeyboardInterrupt:
            break
        except Exception as error: # keep the fish alive if one step fails
            print("Problem:", error)
            mouth.stop()
            body.stop()
