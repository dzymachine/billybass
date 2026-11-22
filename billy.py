import os
import subprocess
import time
import wave

from google import genai
from google.genai import types
import numpy as np
import pvporcupine
from faster_whisper import WhisperModel
from gpiozero import Motor
from pvrecorder import PvRecorder

# ---------- settings you may need to change ----------
MODEL = "gemini-3.8-flash"   # fast Gemini model for spoken replies
WAKE_WORD_FILE = os.path.expanduser("~/Hey-Billy.ppn")
VOICE = "en_US-danny-low"
SILENCE_LEVEL = 500          # raise this if it never stops listening
MOUTH_LEVEL = 800            # lower this if the mouth stays shut
MOUTH_SPEED = 0.7            # 0 to 1
MIC_INDEX = -1               # -1 = default mic; set to your USB mic's number from the printed list
SYSTEM = (
    "You are Billy, a talking singing fish mounted on a wall plaque. "
    "Be warm, funny and brief. Answer in one or two short spoken sentences. "
    "Never use lists, markdown or emojis, because your words are spoken aloud."
)
# ------------------------------------------------------

mouth = Motor(forward=5, backward=6, enable=12)
body = Motor(forward=19, backward=26, enable=13)
stt = WhisperModel("tiny.en", device="cpu", compute_type="int8")
ai = genai.Client()          # reads GEMINI_API_KEY from the environment
history = []


def wait_for_wake_word():
    porcupine = pvporcupine.create(
        access_key=os.environ["PICOVOICE_ACCESS_KEY"],
        keyword_paths=[WAKE_WORD_FILE],
    )
    rec = PvRecorder(frame_length=porcupine.frame_length, device_index=MIC_INDEX)
    rec.start()
    try:
        while True:
            if porcupine.process(rec.read()) >= 0:
                return
    finally:
        rec.stop()
        rec.delete()
        porcupine.delete()


def listen():
    """Record until about 1.3 seconds of quiet follows speech (10 second limit)."""
    rec = PvRecorder(frame_length=512, device_index=MIC_INDEX)
    rec.start()
    frames, quiet_frames, heard_speech = [], 0, False
    started = time.time()
    try:
        while time.time() - started < 10:
            frame = np.array(rec.read(), dtype=np.int16)
            frames.append(frame)
            loudness = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
            if loudness > SILENCE_LEVEL:
                heard_speech, quiet_frames = True, 0
            elif heard_speech:
                quiet_frames += 1
            if heard_speech and quiet_frames > 40:
                break
    finally:
        rec.stop()
        rec.delete()
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
            max_output_tokens=512,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        ),
    )
    answer = (reply.text or "").strip()
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=answer)]))
    return answer


def speak(text):
    wav_path = "/tmp/billy_reply.wav"
    subprocess.run(
        ["python3", "-m", "piper", "-m", VOICE, "-f", wav_path, "--", text],
        check=True,
    )
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
    print("Microphones:", PvRecorder.get_available_devices())
    print("Billy is ready. Say the wake word.")
    while True:
        try:
            wait_for_wake_word()
            nod()                  # shows you it heard the wake word
            text = transcribe(listen())
            if not text:
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
