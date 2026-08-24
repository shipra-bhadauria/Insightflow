import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Speech to Text using OpenAI Whisper."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    suffix = "." + filename.split(".")[-1] if "." in filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
            )
        return transcript.text
    finally:
        os.unlink(tmp_path)


def synthesize_speech(text: str, voice: str = "alloy") -> bytes:
    """Text to Speech using OpenAI TTS."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text[:4096],
    )
    return response.content
