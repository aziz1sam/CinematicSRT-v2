"""Audio Processing Service (FFmpeg 16kHz Mono Normalization)"""
import asyncio
import logging
import math
import subprocess
from pathlib import Path
from typing import Optional
from core.config import settings

logger = logging.getLogger(__name__)

async def extract_and_enhance_audio(
    video_path: Path,
    output_wav_path: Path,
    enhance: bool = True,
    timeout: Optional[float] = None
) -> Path:
    """Extracts audio from video file to 16kHz mono WAV with acoustic normalization."""
    if timeout is None:
        timeout = settings.FFMPEG_TIMEOUT_SECONDS
    if type(timeout) not in (int, float):
        raise TypeError("FFmpeg timeout must be a built-in int or float.")
    try:
        timeout_val = float(timeout)
    except (ValueError, OverflowError) as exc:
        raise ValueError("FFmpeg timeout must be finite and in (0, 86400].") from exc
    if not math.isfinite(timeout_val) or not 0.0 < timeout_val <= 86400.0:
        raise ValueError("FFmpeg timeout must be finite and in (0, 86400].")

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = settings.FFMPEG_BINARY
    
    # Filter: Highpass/Lowpass + Dynamic Compressor + Loudnorm to bring out quiet dialogue
    if enhance:
        audio_filter = (
            "highpass=f=80,"
            "lowpass=f=8000,"
            "acompressor=threshold=-24dB:ratio=3:attack=5:release=120:makeup=6,"
            "loudnorm=I=-16:LRA=11:TP=-1.5"
        )
    else:
        audio_filter = "anull"

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", audio_filter,
        str(output_wav_path)
    ]

    logger.info("Extracting audio via %s: %s -> %s", ffmpeg_bin, video_path.name, output_wav_path.name)
    
    def _run_ffmpeg():
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
            timeout=timeout_val
        )

    try:
        result = await asyncio.to_thread(_run_ffmpeg)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Audio extraction timed out after {timeout_val}s.") from exc

    if result.returncode != 0 or not output_wav_path.exists():
        logger.warning("FFmpeg enhancement failed, attempting plain 16k extraction...")
        # Fallback without audio filters
        fallback_cmd = [
            ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            str(output_wav_path)
        ]
        try:
            result_fb = await asyncio.to_thread(
                lambda: subprocess.run(
                    fallback_cmd,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=timeout_val
                )
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Audio extraction timed out after {timeout_val}s.") from exc

        if result_fb.returncode != 0:
            err_msg = result_fb.stderr.decode("utf-8", errors="ignore")[-400:] if result_fb.stderr else "Unknown error"
            raise RuntimeError(f"Audio extraction failed with {ffmpeg_bin}: {err_msg}")

    return output_wav_path
