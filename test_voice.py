import os
import wave

import requests
from google import genai
from google.genai import types

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George; swap with any Voice ID from elevenlabs.io

ai = genai.Client()
reply = ai.models.generate_content(
    model="gemini-3.8-flash",
    contents="Introduce yourself in one funny sentence.",
    config=types.GenerateContentConfig(
        system_instruction=(
            "You are Billy, a talking singing fish mounted on a wall plaque. "
            "Be warm, funny and brief."
        ),
        max_output_tokens=1024,
    ),
)

answer = (reply.text or "").strip()
print("Billy:", answer)

url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=pcm_16000"
response = requests.post(
    url,
    headers={
        "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
        "Content-Type": "application/json",
    },
    json={"text": answer, "model_id": "eleven_flash_v2_5"},
    timeout=30,
)
response.raise_for_status()

wav_path = "/tmp/test_billy.wav"
with wave.open(wav_path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(response.content)

print(f"Saved {wav_path}")
