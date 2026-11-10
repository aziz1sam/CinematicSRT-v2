"""System Diagnostic, Prerequisite Checker, and Automated Model/FFmpeg Downloader"""
import asyncio
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any
import httpx
import torch

from core.config import settings

logger = logging.getLogger(__name__)

download_progress = {
    "status": "idle",
    "percent": 0,
    "current_task": ""
}

def get_system_status() -> Dict[str, Any]:
    """Diagnoses local hardware, dependencies, and installed models."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "غير متوفر (سيتم استخدام المعالج CPU)"
    vram_mb = 0
    if cuda_available:
        try:
            vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        except Exception:
            vram_mb = 4096

    # Check FFmpeg
    ffmpeg_bin = shutil.which("ffmpeg") or str(settings.BASE_DIR / "bin" / "ffmpeg.exe")
    ffmpeg_installed = bool(shutil.which("ffmpeg")) or (settings.BASE_DIR / "bin" / "ffmpeg.exe").exists()

    # Check Local Models in HuggingFace cache
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    nllb_cached = False
    whisper_cached = False

    if hf_hub.exists():
        entries = [p.name for p in hf_hub.iterdir() if p.is_dir()]
        nllb_cached = any("nllb-200-distilled-600M" in e for e in entries)
        whisper_cached = any("faster-whisper" in e for e in entries)

    # Check API keys
    gemini_ready = bool(settings.GEMINI_API_KEY)
    groq_ready = bool(settings.GROQ_API_KEY)

    # Determine if system is ready to operate
    has_translation = gemini_ready or groq_ready or nllb_cached
    has_transcription = groq_ready or whisper_cached or True # Whisper can auto-download on first transcribe
    is_ready = ffmpeg_installed and (has_translation or gemini_ready or nllb_cached)

    return {
        "is_ready": is_ready,
        "first_run_needed": not is_ready,
        "hardware": {
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "vram_mb": vram_mb,
            "cpu_cores": os.cpu_count() or 4,
        },
        "dependencies": {
            "ffmpeg_installed": ffmpeg_installed,
            "ffmpeg_path": ffmpeg_bin if ffmpeg_installed else None,
        },
        "models": {
            "whisper_local_cached": whisper_cached,
            "nllb_local_cached": nllb_cached,
        },
        "cloud_apis": {
            "gemini_configured": gemini_ready,
            "groq_configured": groq_ready,
        },
        "download_progress": download_progress
    }

async def download_local_model(model_type: str):
    """Downloads Whisper or NLLB model to local HuggingFace cache."""
    global download_progress
    download_progress["status"] = "downloading"
    download_progress["percent"] = 10

    try:
        if model_type == "whisper":
            download_progress["current_task"] = "تحميل نموذج Whisper (أصوات إلى نص)..."
            from faster_whisper import WhisperModel
            def _dl_w():
                WhisperModel("medium", device="cpu", compute_type="int8")
            await asyncio.to_thread(_dl_w)
            download_progress["percent"] = 100

        elif model_type == "nllb":
            download_progress["current_task"] = "تحميل نموذج NLLB-200 للترجمة العربية..."
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            def _dl_n():
                m_name = "facebook/nllb-200-distilled-600M"
                AutoTokenizer.from_pretrained(m_name)
                AutoModelForSeq2SeqLM.from_pretrained(m_name)
            await asyncio.to_thread(_dl_n)
            download_progress["percent"] = 100

        download_progress["status"] = "completed"
        download_progress["current_task"] = "اكتمل التحميل بنجاح!"
    except Exception as e:
        logger.exception("Model download failed: %s", e)
        download_progress["status"] = "failed"
        download_progress["current_task"] = f"فشل التحميل: {str(e)}"

async def install_portable_ffmpeg():
    """Downloads standalone portable FFmpeg for Windows if missing."""
    global download_progress
    download_progress["status"] = "downloading"
    download_progress["current_task"] = "تحميل أداة FFmpeg المحمولة لويندوز..."
    download_progress["percent"] = 20

    bin_dir = settings.BASE_DIR / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = bin_dir / "ffmpeg.zip"

    # Static Windows FFmpeg build URL (gyan.dev essentials)
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-essentials_build.zip"

    try:
        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                zip_dest.write_bytes(resp.content)
                download_progress["percent"] = 70
                download_progress["current_task"] = "فك ضغط FFmpeg..."
                
                # Extract only ffmpeg.exe into bin/
                with zipfile.ZipFile(zip_dest, "r") as z:
                    for name in z.namelist():
                        if name.endswith("ffmpeg.exe"):
                            with z.open(name) as src, open(bin_dir / "ffmpeg.exe", "wb") as dst:
                                shutil.copyfileobj(src, dst)
                                break
                
                zip_dest.unlink(missing_ok=True)
                download_progress["percent"] = 100
                download_progress["status"] = "completed"
                download_progress["current_task"] = "تم تثبيت FFmpeg بنجاح!"
            else:
                raise RuntimeError(f"HTTP {resp.status_code}")
    except Exception as e:
        logger.exception("FFmpeg download failed: %s", e)
        download_progress["status"] = "failed"
        download_progress["current_task"] = "فشل تحميل FFmpeg تلقائياً. يرجى تثبيته يدوياً."
