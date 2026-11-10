"""Deterministic runtime preferences, format definitions, and 5-stage progress snapshots."""
from functools import wraps
import torch

COMPUTE_MODES = ("cpu", "hybrid", "gpu")
TRANSCRIPTION_MODES = ("normal", "balanced", "low_volume")
STAGES = ("prepare", "transcribe", "segment", "translate", "export")

STATUS_STAGE = {
    "pending": "prepare",
    "extracting_audio": "prepare",
    "transcribing": "transcribe",
    "segmenting": "segment",
    "translating": "translate",
    "generating_srt": "export",
    "exporting": "export",
}

SUPPORTED_VIDEO_FORMATS = [
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".ts", ".mts", ".m2ts", ".vob", ".ogv", ".3gp", ".3g2"
]

SUPPORTED_AUDIO_FORMATS = [
    ".mp3", ".m4a", ".m4b", ".wav", ".flac", ".ogg", ".opus", ".aac",
    ".wma", ".aiff", ".aif", ".amr"
]

SUPPORTED_SUBTITLE_FORMATS = [".srt", ".ass", ".vtt"]

FORMAT_NOTES = {
    "video": "جميع حاويات الفيديو القياسية والاحترافية؛ يستخرج المحرك الصوت تلقائياً بترميز 16kHz Mono.",
    "audio": "صيغ الصوت المضغوطة وغير المضغوطة (Lossless). ملفات Audible المحمية (.aa/.aax) تحتاج فك الحماية خارجياً.",
    "text": "ملفات ترجمة SRT وASS وVTT مع الحفاظ التام على التوقيت والأنماط؛ لا يشمل مستندات TXT أو Word."
}


def device_profile(mode=None, cuda_available=None):
    """Resolve CPU / Hybrid (CPU+GPU) / GPU device allocation safely."""
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    mode = (mode or ("hybrid" if cuda_available else "cpu")).strip().lower()
    if mode not in COMPUTE_MODES:
        raise ValueError("خيار المعالجة غير صحيح. القيم المتاحة: cpu, hybrid, gpu")
    if mode != "cpu" and not cuda_available:
        raise ValueError("هذا الخيار يحتاج كرت NVIDIA متوافقًا مع CUDA. اختر المعالج فقط.")
    return {
        "mode": mode,
        "whisper_device": "cuda" if mode == "gpu" else "cpu",
        "translation_device": "cpu" if mode == "cpu" else "cuda",
    }


def describe_compute_mode(mode=None):
    """Validate compute mode against live CUDA & CTranslate2 capability without loading weights."""
    profile = device_profile(mode, torch.cuda.is_available())
    if profile["whisper_device"] == "cuda":
        try:
            import ctranslate2
            supported = set(ctranslate2.get_supported_compute_types("cuda"))
            if not supported.intersection({"int8_float32", "int8", "float32", "float16"}):
                raise ValueError("لا توجد دقة Whisper متوافقة مع الكرت")
        except RuntimeError as error:
            raise ValueError("تعذّر تشغيل Whisper على الكرت. اختر المعالج أو الوضع المشترك.") from error
    return profile


def stage_snapshot(status, percent=None, basis="indeterminate", subtitle=False, from_cache=False, active_stage=None):
    """Produce structured 5-stage progress array for the UI stepper."""
    status_val = getattr(status, "value", status)
    current = active_stage if active_stage in STAGES else STATUS_STAGE.get(status_val)
    is_finished = status_val in ("completed", "completed_with_warnings")
    current_index = STAGES.index(current) if current else (len(STAGES) if is_finished else -1)

    rows = []
    for index, key in enumerate(STAGES):
        skipped = subtitle and key in ("transcribe", "segment")
        done = index < current_index
        value = 100 if done else percent if key == current else None
        state = (
            "cached" if from_cache
            else "skipped" if skipped
            else "done" if done
            else "active" if key == current
            else "pending"
        )
        rows.append({
            "id": key,
            "percent": None if skipped or from_cache else value,
            "state": state,
            "basis": basis if key == current else ("completed" if done else "indeterminate"),
        })
    return rows
