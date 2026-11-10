"""Speech-to-Text Transcriber (Local faster-whisper CPU/CUDA + Groq Cloud API)"""
import asyncio
import gc
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple
import httpx
import torch

from core.config import settings

logger = logging.getLogger(__name__)

TRANSCRIPTION_PRESETS = {
    "normal": {"beam_size": 1, "best_of": 1, "vad_filter": False},
    "balanced": {"beam_size": 3, "best_of": 3, "vad_filter": True},
    "low_volume": {"beam_size": 5, "best_of": 5, "vad_filter": False},
}


class UnifiedTranscriber:
    """Manages audio transcription with dynamic compute device & strict VRAM lifecycle."""

    def __init__(self):
        self.default_device = settings.WHISPER_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_device_and_compute(self, device: Optional[str] = None, compute_type: Optional[str] = None) -> Tuple[str, str]:
        target_device = (device or settings.WHISPER_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
        if target_device == "cuda" and not torch.cuda.is_available():
            target_device = "cpu"

        if target_device == "cuda":
            import ctranslate2
            supported = set(ctranslate2.get_supported_compute_types("cuda"))
            if compute_type and compute_type in supported:
                target_compute = compute_type
            else:
                target_compute = next(
                    (kind for kind in ("int8_float32", "int8", "float32", "float16") if kind in supported),
                    "float16"
                )
        else:
            target_compute = "int8"

        return target_device, target_compute

    async def transcribe(
        self,
        audio_path: Path,
        provider: str = "local",
        language: Optional[str] = None,
        progress_cb=None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        whisper_model: Optional[str] = None,
        transcription_mode: str = "balanced",
    ) -> Tuple[List[SimpleNamespace], str, float]:
        """Transcribes audio using specified provider ('local' or 'groq')."""
        if provider == "groq" and settings.GROQ_API_KEY:
            return await self._transcribe_groq(audio_path, language)
        return await self._transcribe_local(
            audio_path=audio_path,
            language=language,
            progress_cb=progress_cb,
            device=device,
            compute_type=compute_type,
            whisper_model=whisper_model,
            transcription_mode=transcription_mode,
        )

    async def _transcribe_local(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        progress_cb=None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        whisper_model: Optional[str] = None,
        transcription_mode: str = "balanced",
    ) -> Tuple[List[SimpleNamespace], str, float]:
        """Local transcription with faster-whisper, live stage progress, and guaranteed VRAM cleanup."""
        from faster_whisper import WhisperModel

        target_model = (whisper_model or settings.WHISPER_MODEL or "medium").strip()
        target_device, target_compute = self._resolve_device_and_compute(device, compute_type)
        preset = TRANSCRIPTION_PRESETS.get(transcription_mode, TRANSCRIPTION_PRESETS["balanced"])

        logger.info(
            "[Whisper Local] Loading model=%s on %s (%s), mode=%s...",
            target_model, target_device, target_compute, transcription_mode
        )
        start_time = time.time()
        model = None
        loop = asyncio.get_running_loop()

        try:
            model = WhisperModel(target_model, device=target_device, compute_type=target_compute, local_files_only=True)

            def _run_transcribe():
                kwargs = dict(
                    beam_size=preset["beam_size"],
                    best_of=preset["best_of"],
                    temperature=0.0,
                    vad_filter=preset["vad_filter"],
                    word_timestamps=True,
                    condition_on_previous_text=False
                )
                if language:
                    kwargs["language"] = language
                segments_gen, info = model.transcribe(str(audio_path), **kwargs)
                duration_val = float(getattr(info, "duration", 0.0) or 0.0)
                collected = []
                last_emit_ts = 0.0

                for s in segments_gen:
                    collected.append(s)
                    now = time.time()
                    if progress_cb and duration_val > 0 and (now - last_emit_ts >= 0.45):
                        last_emit_ts = now
                        ratio = min(1.0, max(0.0, float(getattr(s, "end", 0.0) or 0.0) / duration_val))
                        pct = min(99, int(ratio * 100))
                        try:
                            loop.call_soon_threadsafe(progress_cb, pct, float(getattr(s, "end", 0.0) or 0.0), duration_val)
                        except Exception:
                            pass

                if progress_cb and duration_val > 0:
                    try:
                        loop.call_soon_threadsafe(progress_cb, 100, duration_val, duration_val)
                    except Exception:
                        pass

                return collected, info

            segments_raw, info = await asyncio.to_thread(_run_transcribe)
            detected_lang = getattr(info, "language", language or "en")
            duration = float(getattr(info, "duration", 0.0) or 0.0)

            segments = []
            for i, s in enumerate(segments_raw):
                text = (getattr(s, "text", "") or "").strip()
                if text:
                    words_meta = getattr(s, "words", None)
                    segments.append(SimpleNamespace(
                        index=i + 1,
                        start=float(s.start),
                        end=float(s.end),
                        text=text,
                        words=words_meta
                    ))

            logger.info("[Whisper Local] Transcription complete (%.2fs, %d segments).", time.time() - start_time, len(segments))
            return segments, detected_lang, duration

        finally:
            if model is not None:
                del model
            gc.collect()
            if target_device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def _transcribe_groq(
        self,
        audio_path: Path,
        language: Optional[str] = None
    ) -> Tuple[List[SimpleNamespace], str, float]:
        """Ultra-fast Cloud Whisper transcription using Groq API (~5 to 10s for full movie)."""
        logger.info("[Whisper Groq] Submitting audio to Groq Whisper-large-v3...")
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}

        upload_path = audio_path
        temp_mp3 = None
        if audio_path.stat().st_size > 24 * 1024 * 1024:
            temp_mp3 = audio_path.with_suffix(".groq.mp3")
            ffmpeg_bin = str(settings.FFMPEG_BINARY or "ffmpeg")
            cmd = [
                ffmpeg_bin, "-y",
                "-i", str(audio_path),
                "-vn",
                "-c:a", "libmp3lame",
                "-b:a", "64k",
                str(temp_mp3)
            ]
            logger.info("[Whisper Groq] Audio file size (%d bytes) exceeds 24MB limit. Compressing to MP3 for Groq upload...", audio_path.stat().st_size)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            if temp_mp3.exists() and temp_mp3.stat().st_size > 0:
                upload_path = temp_mp3

        try:
            mime_type = "audio/mpeg" if upload_path.suffix.lower() == ".mp3" else "audio/wav"
            async with httpx.AsyncClient(timeout=180.0) as client:
                with open(upload_path, "rb") as f_audio:
                    files = {"file": (upload_path.name, f_audio, mime_type)}
                    data = {
                        "model": "whisper-large-v3",
                        "response_format": "verbose_json",
                        "temperature": "0.0"
                    }
                    if language:
                        data["language"] = language

                    resp = await client.post(url, headers=headers, files=files, data=data)
                    if resp.status_code != 200:
                        raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")

                    res_json = resp.json()
                    detected_lang = res_json.get("language", language or "en")
                    duration = float(res_json.get("duration", 0.0))

                    segments = []
                    for i, seg in enumerate(res_json.get("segments", [])):
                        text = seg.get("text", "").strip()
                        if text:
                            segments.append(SimpleNamespace(
                                index=i + 1,
                                start=float(seg["start"]),
                                end=float(seg["end"]),
                                text=text,
                                words=seg.get("words", None)
                            ))

                    return segments, detected_lang, duration
        finally:
            if temp_mp3 is not None and temp_mp3.exists():
                temp_mp3.unlink(missing_ok=True)


transcriber = UnifiedTranscriber()
