import subprocess
from google import genai
from google.genai import types

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

subprocess.run(
    [
        "python3",
        "-m",
        "piper",
        "-m",
        "en_US-danny-low",
        "-f",
        "/tmp/test_billy.wav",
        "--",
        answer,
    ],
    check=True,
)
print("Saved /tmp/test_billy.wav")
