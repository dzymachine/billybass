# AI Billy Bass

Voice-activated Big Mouth Billy Bass powered by a Raspberry Pi 3, `openWakeWord`, Google Gemini (`google-genai`, native audio input), ElevenLabs TTS, and a TB6612FNG dual motor driver.

## Files
- `billy.py` — main wake-word, speech, Gemini AI, TTS, and mouth-sync loop
- `motor_test.py` — standalone GPIO motor direction & speed test
- `test_voice.py` — standalone Gemini + ElevenLabs voice test
- `requirements.txt` — Python dependencies for the Raspberry Pi virtual environment

## Environment Variables (on the Pi)
- `GEMINI_API_KEY` — Google AI Studio / Gemini API key
- `ELEVENLABS_API_KEY` — ElevenLabs API key for text-to-speech
