"""CinematicSRT v2.0 - Central Configuration"""
import math
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables if present
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

def _parse_ffmpeg_timeout(raw_val) -> float:
    if not isinstance(raw_val, str):
        return 300.0
    try:
        value = float(raw_val.strip())
    except (ValueError, TypeError, OverflowError):
        return 300.0
    if not math.isfinite(value) or not 0.0 < value <= 86400.0:
        return 300.0
    return value

class Settings:
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    TEMP_DIR: Path = BASE_DIR / "temp"

    # Whisper Settings (Default: CUDA on GTX 1650 with int8_float32 hardware acceleration)
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "medium")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float32")

    # Transcription Engine: "local" (faster-whisper) or "groq" (Cloud API ~5s)
    TRANSCRIPTION_PROVIDER: str = os.getenv("TRANSCRIPTION_PROVIDER", "local")

    # Translation Engine: "local" (NLLB-200), "gemini" (Google Gemini API), "groq" (Llama 3.3), "qwen" (Local LLM)
    TRANSLATION_PROVIDER: str = os.getenv("TRANSLATION_PROVIDER", "local")

    # API Keys for Cloud Providers
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LOCAL_QWEN_URL: str = os.getenv("LOCAL_QWEN_URL", "http://127.0.0.1:8080/v1")

    # Subtitle Display Constraints
    MAX_CHARS_PER_LINE: int = 42
    MAX_LINES_PER_SEGMENT: int = 2
    MIN_SEGMENT_DURATION: float = 0.8
    MAX_SEGMENT_DURATION: float = 7.0
    MIN_GAP_BETWEEN_SEGMENTS: float = 0.05

    # File Upload Limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "4000"))
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

    FFMPEG_TIMEOUT_SECONDS: float = _parse_ffmpeg_timeout(os.getenv("FFMPEG_TIMEOUT_SECONDS"))

    @property
    def FFMPEG_BINARY(self) -> str:
        """Resolves FFmpeg executable in order: local bin/ffmpeg.exe -> PATH -> 'ffmpeg'."""
        portable = self.BASE_DIR / "bin" / "ffmpeg.exe"
        if portable.exists():
            return str(portable)
        system_bin = shutil.which("ffmpeg")
        if system_bin:
            return system_bin
        return "ffmpeg"

    @classmethod
    def save_env_key(cls, key: str, value: str):
        """Persist key into .env file with input sanitization (prevent newline injection)."""
        clean_key = key.strip().replace("\n", "").replace("\r", "")
        clean_val = value.strip().replace("\n", "").replace("\r", "")
        os.environ[clean_key] = clean_val
        setattr(cls, clean_key, clean_val)

        lines = []
        if ENV_PATH.exists():
            lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

        found = False
        new_lines = []
        for line in lines:
            if line.startswith(f"{clean_key}="):
                new_lines.append(f"{clean_key}={clean_val}")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"{clean_key}={clean_val}")

        ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

settings = Settings()
