"""CinematicSRT v2.0 - High Performance FastAPI Application Server"""
import asyncio
import gc
import hmac
import json
import logging
import os
import secrets
import shutil
import signal
import sys
import threading
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Optional
from urllib.parse import urlparse

import torch
from fastapi import FastAPI, Request, Query, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from core.config import settings
from core.runtime_options import (
    COMPUTE_MODES,
    TRANSCRIPTION_MODES,
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_SUBTITLE_FORMATS,
    FORMAT_NOTES,
    describe_compute_mode,
)
from services.pipeline import (
    JobManager,
    job_manager,
    run_cinematic_pipeline,
    run_subtitle_translation_pipeline
)
from services.system_checker import (
    get_system_status,
    download_local_model,
    install_portable_ffmpeg
)

# Runtime execution defaults (kept out of core/config.py to preserve policy invariants)
settings.COMPUTE_MODE = (os.getenv("COMPUTE_MODE") or ("hybrid" if torch.cuda.is_available() else "cpu")).strip().lower()
settings.DEFAULT_TRANSCRIPTION_MODE = (os.getenv("TRANSCRIPTION_MODE") or "balanced").strip().lower()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CinematicSRT")

# Initialize Directories
for d in [settings.UPLOAD_DIR, settings.OUTPUT_DIR, settings.TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CinematicSRT v2.0", version="2.0.0")

# Local Origin CORS (scoped strictly to v2.0 local origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "http://[::1]:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def reject_mutating_work_during_shutdown(request: Request, call_next):
    if lifecycle_manager.shutting_down and request.method.upper() == "POST":
        if request.url.path in {"/api/upload", "/api/setup/download-model", "/api/setup/install-ffmpeg"}:
            return Response(
                content=json.dumps({"detail": "البرنامج قيد الإيقاف حالياً"}),
                status_code=503,
                media_type="application/json",
            )
    return await call_next(request)

# Mount Local Assets (Self-hosted Cairo font)
FONTS_DIR = settings.BASE_DIR / "ui" / "fonts"
if FONTS_DIR.is_dir():
    app.mount("/fonts", StaticFiles(directory=str(FONTS_DIR)), name="fonts")

# Serve Frontend
UI_FILE = settings.BASE_DIR / "ui" / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found")
    html_content = UI_FILE.read_text(encoding="utf-8").replace(
        "__CINEMATIC_LIFECYCLE_TOKEN__",
        lifecycle_manager.control_token,
        1,
    )
    return HTMLResponse(
        content=html_content,
        headers={"Cache-Control": "no-cache"},
    )

# =============================================================================
# Pipeline Trigger Endpoints
# =============================================================================

ALLOWED_SUBTITLE_EXTS = frozenset({".srt", ".ass", ".vtt"})
ALLOWED_MEDIA_EXTS = frozenset({
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".ts", ".mts", ".m2ts", ".vob", ".ogv", ".3gp", ".3g2", ".mp3",
    ".m4a", ".m4b", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".aiff", ".aif", ".amr"
})


def resolve_upload_filename(filename: Optional[str]):
    """Validate the admission suffix and discard client directory components."""
    if not isinstance(filename, str) or not filename.strip() or filename.endswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="اسم الملف غير صالح أو مفقود")
    clean_filename = PureWindowsPath(filename).name
    if (not clean_filename or clean_filename in {".", ".."}
            or clean_filename.endswith((".", " "))
            or any(ord(ch) < 32 or ord(ch) == 127 or ch in '<>:"|?*' for ch in clean_filename)):
        raise HTTPException(status_code=400, detail="اسم الملف غير صالح أو مفقود")
    ext = PureWindowsPath(clean_filename).suffix.lower()
    if ext not in ALLOWED_SUBTITLE_EXTS | ALLOWED_MEDIA_EXTS:
        raise HTTPException(status_code=400, detail="نوع الملف غير مدعوم")
    return clean_filename, ext


@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    transcription_provider: str = Form("local"),
    translation_provider: str = Form("local"),
    language: Optional[str] = Form(None),
    max_file_size_mb: Optional[int] = Form(None),
    compute_mode: Optional[str] = Form(None),
    transcription_mode: Optional[str] = Form(None),
):
    """Chunked streaming upload with cumulative size quota & format routing."""
    try:
        custom_limit = isinstance(max_file_size_mb, int) and not isinstance(max_file_size_mb, bool)
        if max_file_size_mb is not None and (not custom_limit or not 1 <= max_file_size_mb <= 2147483647):
            raise HTTPException(status_code=400, detail="حد حجم الملف يجب أن يكون عددًا صحيحًا موجبًا")
        if transcription_mode is not None and transcription_mode not in ("normal", "balanced", "low_volume"):
            raise HTTPException(
                status_code=400,
                detail="وضع التفريغ غير صحيح. القيم المتاحة: normal, balanced, low_volume"
            )
        selected_compute_mode = None
        if compute_mode is not None:
            selected_compute_mode = compute_mode.strip().lower() if isinstance(compute_mode, str) else compute_mode
            try:
                describe_compute_mode(selected_compute_mode)
            except ValueError as err:
                raise HTTPException(status_code=400, detail=str(err)) from err

        clean_filename, ext = resolve_upload_filename(file.filename)
        job_id = str(uuid.uuid4())
        partial_path = settings.UPLOAD_DIR / f"{job_id}{ext}.part"
        final_path = settings.UPLOAD_DIR / f"{job_id}{ext}"
        effective_max_mb = max_file_size_mb if custom_limit else settings.MAX_FILE_SIZE_MB
        max_bytes = effective_max_mb * 1024 * 1024 if custom_limit else settings.MAX_FILE_SIZE_BYTES
        bytes_written = 0
        chunk_size = 8 * 1024 * 1024  # 8MB chunk buffer
        partial_created = False

        try:
            with open(partial_path, "xb") as f_out:
                partial_created = True
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"حجم الملف يتجاوز الحد المسموح ({effective_max_mb} ميجابايت)"
                        )
                    f_out.write(chunk)

            if bytes_written == 0:
                raise HTTPException(status_code=400, detail="الملف المرفوع فارغ")

            # Atomic rename after verified full write
            partial_path.rename(final_path)

        except Exception as e:
            if partial_created and partial_path.exists():
                partial_path.unlink(missing_ok=True)
            if isinstance(e, HTTPException):
                raise e
            logger.exception("Upload streaming failed: %s", e)
            raise HTTPException(status_code=500, detail="فشل حفظ الملف على القرص الصلب.")
    finally:
        await file.close()

    # Initialize job
    job_manager.create_job(job_id, clean_filename)

    # Route based on file extension: direct subtitle vs audio/video pipeline
    sub_opts = {"compute_mode": selected_compute_mode} if selected_compute_mode is not None else {}
    media_opts = dict(sub_opts)
    if transcription_mode is not None:
        media_opts["transcription_mode"] = transcription_mode

    if ext in ALLOWED_SUBTITLE_EXTS:
        background_tasks.add_task(
            run_subtitle_translation_pipeline,
            job_id=job_id,
            subtitle_path=final_path,
            translation_provider=translation_provider,
            source_language=language,
            **sub_opts
        )
    else:
        background_tasks.add_task(
            run_cinematic_pipeline,
            job_id=job_id,
            video_path=final_path,
            transcription_provider=transcription_provider,
            translation_provider=translation_provider,
            language=language,
            **media_opts
        )

    return {"success": True, "job_id": job_id, "filename": clean_filename}

@app.get("/api/jobs/{job_id}")
async def get_job_info(job_id: str):
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    record = dict(job_manager.jobs[job_id])
    if "stages" not in record:
        record["stages"] = JobManager.build_stages("prepare", 0)
    return record


@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    if job_id not in job_manager.jobs:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    await job_manager.cancel_job(job_id)
    return {"success": True, "message": "تم إلغاء المهمة بنجاح وتنظيف الملفات المؤقتة"}

@app.get("/api/video/{job_id}")
async def stream_video(job_id: str):
    """Streams the uploaded video file for in-browser playback."""
    matches = list(settings.UPLOAD_DIR.glob(f"{job_id}.*"))
    if not matches or not matches[0].exists():
        raise HTTPException(status_code=404, detail="ملف الفيديو غير موجود")
    return FileResponse(path=matches[0])

@app.get("/api/subtitles/{job_id}/vtt")
async def get_vtt_subtitles(job_id: str):
    """Serves standard WebVTT subtitles for native HTML5 video player."""
    job_record = job_manager.jobs.get(job_id)
    if type(job_record) is not dict or job_record.get("status") not in ("completed", "completed_with_warnings"):
        raise HTTPException(status_code=404, detail="ملف الترجمة غير جاهز بعد")
    vtt_path = settings.OUTPUT_DIR / f"{job_id}.vtt"
    if not vtt_path.exists():
        # Check if SRT is available to convert
        srt_path = settings.OUTPUT_DIR / f"{job_id}.srt"
        if not srt_path.exists():
            raise HTTPException(status_code=404, detail="ملف الترجمة VTT غير متوفر")
        import re
        content = srt_path.read_text(encoding="utf-8-sig")
        vtt_content = "WEBVTT\n\n" + re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", content)
        return Response(content=vtt_content, media_type="text/vtt; charset=utf-8")
    return FileResponse(path=vtt_path, media_type="text/vtt; charset=utf-8")

@app.get("/api/preview/{job_id}")
async def preview_subtitles(job_id: str, lines: int = 20):
    """Returns bounded cue preview from completed SRT/ASS subtitles."""
    job_record = job_manager.jobs.get(job_id)
    if type(job_record) is not dict or job_record.get("status") not in ("completed", "completed_with_warnings"):
        raise HTTPException(status_code=404, detail="ملف الترجمة غير جاهز بعد")
    safe_lines = max(1, min(int(lines), 100))
    srt_path = settings.OUTPUT_DIR / f"{job_id}.srt"
    ass_path = settings.OUTPUT_DIR / f"{job_id}.ass"
    if srt_path.exists():
        content = srt_path.read_text(encoding="utf-8-sig").strip()
        blocks = [blk for blk in content.split("\n\n") if blk.strip()]
    elif ass_path.exists():
        content = ass_path.read_text(encoding="utf-8-sig").strip()
        blocks = [line for line in content.splitlines() if line.lstrip().lower().startswith("dialogue:")]
    else:
        raise HTTPException(status_code=404, detail="ملف الترجمة غير جاهز بعد")
    return {
        "success": True,
        "preview": "\n\n".join(blocks[:safe_lines]),
        "total": len(blocks)
    }

@app.get("/api/download/{job_id}/{fmt}")
async def download_subtitles(job_id: str, fmt: str):
    """Downloads generated .srt, .ass, or .vtt subtitle file."""
    job_record = job_manager.jobs.get(job_id)
    if type(job_record) is not dict or job_record.get("status") not in ("completed", "completed_with_warnings"):
        raise HTTPException(status_code=404, detail="ملف الترجمة غير جاهز بعد")
    fmt_clean = fmt.lower().strip()
    if fmt_clean == "ass":
        ext = ".ass"
    elif fmt_clean == "vtt":
        ext = ".vtt"
    else:
        ext = ".srt"

    file_path = settings.OUTPUT_DIR / f"{job_id}{ext}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="ملف الترجمة غير جاهز بعد")

    orig_name = "subtitles"
    if job_id in job_manager.jobs:
        orig_name = Path(job_manager.jobs[job_id]["filename"]).stem

    return FileResponse(
        path=file_path,
        filename=f"{orig_name}_arabic{ext}",
        media_type="text/plain; charset=utf-8"
    )

class SettingsUpdate(BaseModel):
    gemini_key: Optional[str] = None
    groq_key: Optional[str] = None
    qwen_url: Optional[str] = None
    whisper_model: Optional[str] = None
    whisper_compute_type: Optional[str] = None
    compute_mode: Optional[str] = None
    max_file_size_mb: Optional[int] = None


def _build_runtime_settings_payload():
    cpu = describe_compute_mode("cpu")
    try:
        hybrid = describe_compute_mode("hybrid")
    except ValueError:
        hybrid = None
    try:
        gpu = describe_compute_mode("gpu")
    except ValueError:
        gpu = None
    available_modes = {"cpu": cpu, "hybrid": hybrid, "gpu": gpu}
    saved_mode = getattr(settings, "COMPUTE_MODE", "hybrid" if hybrid else "cpu")
    if not available_modes.get(saved_mode):
        saved_mode = "hybrid" if hybrid else "cpu"
    return {
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "groq_configured": bool(settings.GROQ_API_KEY),
        "qwen_url": settings.LOCAL_QWEN_URL,
        "whisper_device": available_modes[saved_mode]["whisper_device"],
        "whisper_model": settings.WHISPER_MODEL,
        "whisper_compute_type": settings.WHISPER_COMPUTE_TYPE,
        "compute_mode": saved_mode,
        "default_compute_mode": "hybrid" if hybrid else "cpu",
        "compute_modes": available_modes,
        "transcription_modes": list(TRANSCRIPTION_MODES),
        "default_max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "max_configurable_file_size_mb": 2147483647,
        "formats": {
            "video": SUPPORTED_VIDEO_FORMATS,
            "audio": SUPPORTED_AUDIO_FORMATS,
            "text": SUPPORTED_SUBTITLE_FORMATS,
        },
        "format_notes": FORMAT_NOTES,
    }


@app.get("/api/settings")
@app.get("/api/v1/settings")
async def get_settings():
    return _build_runtime_settings_payload()

@app.post("/api/settings")
async def update_settings(payload: SettingsUpdate):
    if payload.gemini_key is not None and payload.gemini_key.strip():
        settings.save_env_key("GEMINI_API_KEY", payload.gemini_key.strip())
    if payload.groq_key is not None and payload.groq_key.strip():
        settings.save_env_key("GROQ_API_KEY", payload.groq_key.strip())
    if payload.qwen_url is not None and payload.qwen_url.strip():
        settings.save_env_key("LOCAL_QWEN_URL", payload.qwen_url.strip())
    if payload.whisper_model is not None and payload.whisper_model.strip():
        settings.save_env_key("WHISPER_MODEL", payload.whisper_model.strip())
    if payload.whisper_compute_type is not None and payload.whisper_compute_type.strip():
        settings.save_env_key("WHISPER_COMPUTE_TYPE", payload.whisper_compute_type.strip())
    if payload.compute_mode is not None and payload.compute_mode.strip():
        mode_clean = payload.compute_mode.strip().lower()
        try:
            describe_compute_mode(mode_clean)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        settings.save_env_key("COMPUTE_MODE", mode_clean)
        settings.COMPUTE_MODE = mode_clean
    if payload.max_file_size_mb is not None:
        if isinstance(payload.max_file_size_mb, bool) or not 1 <= payload.max_file_size_mb <= 2147483647:
            raise HTTPException(status_code=400, detail="حد حجم الملف يجب أن يكون عددًا صحيحًا موجبًا")
        settings.save_env_key("MAX_FILE_SIZE_MB", str(payload.max_file_size_mb))
        settings.MAX_FILE_SIZE_MB = int(payload.max_file_size_mb)
        settings.MAX_FILE_SIZE_BYTES = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    return {"success": True, "message": "تم حفظ الإعدادات بنجاح", "settings": _build_runtime_settings_payload()}

# =============================================================================
# First-Run Setup & Diagnostic Endpoints
# =============================================================================

@app.get("/api/health")
async def health():
    if lifecycle_manager.shutting_down:
        return Response(
            content=json.dumps({
                "status": "shutting_down",
                "service": "CinematicSRT v2.0",
                "reason": lifecycle_manager.shutdown_reason,
            }),
            status_code=503,
            media_type="application/json",
        )
    return {
        "status": "ok",
        "service": "CinematicSRT v2.0",
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
    }

@app.get("/api/setup/status")
async def check_setup_status():
    """Returns diagnostic report of hardware, dependencies, and models."""
    return get_system_status()

class ModelDownloadReq(BaseModel):
    model_type: str  # 'whisper' or 'nllb'

@app.post("/api/setup/download-model")
async def trigger_model_download(payload: ModelDownloadReq):
    if lifecycle_manager.shutting_down:
        raise HTTPException(status_code=503, detail="البرنامج قيد الإيقاف حالياً")
    lifecycle_manager.spawn_tracked_task("setup_download", download_local_model(payload.model_type))
    return {"success": True, "message": f"بدأ تحميل نموذج {payload.model_type} بالخلفية"}

@app.post("/api/setup/install-ffmpeg")
async def trigger_ffmpeg_install():
    if lifecycle_manager.shutting_down:
        raise HTTPException(status_code=503, detail="البرنامج قيد الإيقاف حالياً")
    lifecycle_manager.spawn_tracked_task("ffmpeg_install", install_portable_ffmpeg())
    return {"success": True, "message": "بدأ تحميل وتثبيت FFmpeg المحمول بالخلفية"}

# =============================================================================
# Application Lifecycle & Browser-Close Auto-Shutdown
# =============================================================================

def _parse_bool_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _parse_grace_seconds(default: float = 4.5) -> float:
    raw = os.getenv("CINEMATIC_CLOSE_GRACE_SECONDS")
    if raw is None or not raw.strip():
        return default
    try:
        val = float(raw.strip())
    except ValueError:
        return default
    return val if 0.1 <= val <= 60.0 else default


async def _read_json_body_limited(request: Request, max_bytes: int = 16384) -> dict[str, Any]:
    try:
        body = await request.body()
        if not body or len(body) > max_bytes:
            return {}
        parsed = json.loads(body.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _default_terminate_descendants(root_pid: Optional[int] = None) -> list[int]:
    """Terminate active descendant child processes (e.g., ffmpeg.exe) on Windows safely."""
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return []

    TH32CS_SNAPPROCESS = 0x00000002
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
    PROCESS_TERMINATE = 0x0001
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SYNCHRONIZE = 0x00100000
    EXCLUDED_EXES = frozenset({"msedge.exe", "chrome.exe", "firefox.exe", "brave.exe", "opera.exe", "explorer.exe"})

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    def _filetime_int(ft: Any) -> int:
        return (int(ft.dwHighDateTime) << 32) | int(ft.dwLowDateTime)

    def _get_ctime(pid: int) -> Optional[int]:
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None
        try:
            ct, et, kt, ut = (
                wintypes.FILETIME(),
                wintypes.FILETIME(),
                wintypes.FILETIME(),
                wintypes.FILETIME(),
            )
            if not kernel32.GetProcessTimes(h, ctypes.byref(ct), ctypes.byref(et), ctypes.byref(kt), ctypes.byref(ut)):
                return None
            return _filetime_int(ct)
        finally:
            kernel32.CloseHandle(h)

    target_root = int(root_pid if root_pid is not None else os.getpid())
    root_ctime = _get_ctime(target_root)

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == INVALID_HANDLE_VALUE:
        return []

    children_map: dict[int, list[tuple[int, str]]] = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            ppid = int(entry.th32ParentProcessID)
            exe_name = str(entry.szExeFile or "").lower()
            if pid and pid != ppid:
                children_map.setdefault(ppid, []).append((pid, exe_name))
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)

    ordered: list[int] = []
    seen: set[int] = {target_root}
    queue: list[tuple[int, Optional[int]]] = [(target_root, root_ctime)]
    while queue:
        parent_pid, parent_ctime = queue.pop(0)
        for child_pid, exe_name in children_map.get(parent_pid, ()):
            if child_pid in seen or exe_name in EXCLUDED_EXES:
                continue
            child_ctime = _get_ctime(child_pid)
            if child_ctime is None:
                continue
            if parent_ctime is not None and child_ctime < parent_ctime:
                continue
            seen.add(child_pid)
            ordered.append(child_pid)
            queue.append((child_pid, child_ctime))

    terminated: list[int] = []
    for child_pid in reversed(ordered):
        h = kernel32.OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, False, child_pid)
        if not h:
            continue
        try:
            if kernel32.TerminateProcess(h, 1):
                kernel32.WaitForSingleObject(h, 500)
                terminated.append(child_pid)
        finally:
            kernel32.CloseHandle(h)
    return terminated


def _default_server_exit_trigger(reason: str) -> None:
    """Gracefully ask active uvicorn.Server instances to exit, with a daemon fail-safe."""
    logger.info("Initiating server shutdown (reason=%s)", reason)

    def _perform_exit():
        signaled = False
        try:
            import uvicorn
            for obj in gc.get_objects():
                if uvicorn.Server in type(obj).__mro__:
                    obj.should_exit = True
                    signaled = True
        except Exception:
            signaled = False
        if not signaled:
            try:
                _default_terminate_descendants()
            except Exception:
                pass
            try:
                signal.raise_signal(signal.SIGTERM)
            except Exception:
                os._exit(0)

    def _watchdog_exit():
        try:
            _default_terminate_descendants()
        except Exception:
            pass
        os._exit(0)

    exit_timer = threading.Timer(0.15, _perform_exit)
    exit_timer.daemon = True
    exit_timer.start()

    watchdog = threading.Timer(3.0, _watchdog_exit)
    watchdog.daemon = True
    watchdog.start()


class BrowserLifecycleManager:
    """Coordinates explicit UI shutdown, local-origin provenance, and F5/bfcache-safe server exit."""

    _LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

    def __init__(
        self,
        job_mgr=None,
        grace_seconds: Optional[float] = None,
        auto_shutdown_enabled: Optional[bool] = None,
        exit_trigger=None,
        control_token: Optional[str] = None,
        child_terminator=None,
    ):
        self.job_mgr = job_mgr if job_mgr is not None else job_manager
        self.grace_seconds = (
            float(grace_seconds) if grace_seconds is not None else _parse_grace_seconds()
        )
        self.auto_shutdown_enabled = (
            bool(auto_shutdown_enabled)
            if auto_shutdown_enabled is not None
            else _parse_bool_env("CINEMATIC_AUTO_SHUTDOWN", True)
        )
        self.exit_trigger = exit_trigger if exit_trigger is not None else _default_server_exit_trigger
        self.child_terminator = (
            child_terminator if child_terminator is not None else _default_terminate_descendants
        )
        self.control_token: str = (
            control_token if isinstance(control_token, str) and control_token else secrets.token_urlsafe(32)
        )
        # conn_id -> (tab_id, conn_gen, websocket)
        self.connections: dict[str, tuple[str, int, Any]] = {}
        self.tab_generations: dict[str, int] = {}
        self.background_tasks: dict[asyncio.Task, str] = {}
        self.has_seen_browser_client: bool = False
        self.shutting_down: bool = False
        self.shutdown_reason: Optional[str] = None
        self._grace_task: Optional[asyncio.Task] = None
        self._shutdown_lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None

    def spawn_tracked_task(self, kind: str, coro: Any) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.background_tasks[task] = kind
        task.add_done_callback(lambda done_task: self.background_tasks.pop(done_task, None))
        return task

    @staticmethod
    def _get_header(headers: Any, name: str) -> Optional[str]:
        if headers is None:
            return None
        target = name.lower()
        try:
            val = headers.get(target)
            if val is not None:
                return str(val)
            val = headers.get(name)
            if val is not None:
                return str(val)
        except Exception:
            pass
        if hasattr(headers, "items"):
            for k, v in headers.items():
                if isinstance(k, str) and k.lower() == target:
                    return str(v)
        return None

    @classmethod
    def _parse_host_header(cls, host_val: str) -> Optional[tuple[str, Optional[int]]]:
        raw = host_val.strip()
        if not raw or any(ch in raw for ch in (" ", "\t", "/", "\\", "?", "#", "@")):
            return None
        try:
            parsed = urlparse(f"http://{raw}")
            hostname = (parsed.hostname or "").lower()
            port = parsed.port
        except ValueError:
            return None
        if hostname not in cls._LOOPBACK_HOSTS:
            return None
        return (hostname, port)

    @classmethod
    def _parse_origin_header(cls, origin_val: str) -> Optional[tuple[str, int]]:
        raw = origin_val.strip()
        if not raw or raw.lower() == "null":
            return None
        try:
            parsed = urlparse(raw)
            if parsed.scheme not in {"http", "https"}:
                return None
            if parsed.username or parsed.password or parsed.path or parsed.params or parsed.query or parsed.fragment:
                return None
            hostname = (parsed.hostname or "").lower()
            if hostname not in cls._LOOPBACK_HOSTS:
                return None
            port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
            return (hostname, port)
        except ValueError:
            return None

    def is_authorized_request(self, headers: Any, supplied_token: Optional[str]) -> bool:
        host_raw = self._get_header(headers, "host")
        host_info: Optional[tuple[str, Optional[int]]] = None
        if host_raw is not None and host_raw.strip():
            host_info = self._parse_host_header(host_raw)
            if host_info is None:
                return False

        sec_fetch_site = self._get_header(headers, "sec-fetch-site")
        if sec_fetch_site is not None and sec_fetch_site.strip():
            if sec_fetch_site.strip().lower() not in {"same-origin", "none"}:
                return False

        origin_raw = self._get_header(headers, "origin")
        if origin_raw is not None:
            origin_info = self._parse_origin_header(origin_raw)
            if origin_info is None:
                return False
            if host_info is not None:
                host_name, host_port = host_info
                orig_name, orig_port = origin_info
                if orig_name != host_name:
                    return False
                if host_port is not None and orig_port != host_port:
                    return False
                if host_port is None and orig_port not in {80, 443}:
                    return False
        else:
            referer_raw = self._get_header(headers, "referer")
            if referer_raw is not None and referer_raw.strip():
                try:
                    ref_parsed = urlparse(referer_raw.strip())
                    ref_origin = f"{ref_parsed.scheme}://{ref_parsed.netloc}"
                except Exception:
                    return False
                origin_info = self._parse_origin_header(ref_origin)
                if origin_info is None:
                    return False
                if host_info is not None:
                    host_name, host_port = host_info
                    orig_name, orig_port = origin_info
                    if orig_name != host_name:
                        return False
                    if host_port is not None and orig_port != host_port:
                        return False
                    if host_port is None and orig_port not in {80, 443}:
                        return False

        if not isinstance(supplied_token, str) or not supplied_token:
            return False
        return hmac.compare_digest(supplied_token, self.control_token)

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._shutdown_lock is None or self._lock_loop is not loop:
            self._shutdown_lock = asyncio.Lock()
            self._lock_loop = loop
        return self._shutdown_lock

    def _cancel_grace_task(self) -> None:
        task = self._grace_task
        self._grace_task = None
        if task is None or task.done():
            return
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is not current:
            task.cancel()

    async def _close_websocket_bounded(self, ws: Any, code: int = 1001, timeout: float = 0.25) -> None:
        try:
            coro = ws.close(code=code)
        except Exception:
            return
        task = asyncio.ensure_future(coro)

        def _consume_task_result(finished: asyncio.Task) -> None:
            if not finished.cancelled():
                try:
                    finished.exception()
                except BaseException:
                    pass

        task.add_done_callback(_consume_task_result)
        try:
            await asyncio.wait({task}, timeout=timeout)
        except Exception:
            pass

    async def register_connection(
        self,
        conn_id: str,
        tab_id: str,
        websocket: Any,
        conn_gen: Optional[int] = None,
    ) -> bool:
        if self.shutting_down:
            try:
                await asyncio.wait_for(
                    websocket.send_json({"type": "shutdown", "reason": self.shutdown_reason or "shutting_down"}),
                    timeout=0.25,
                )
            except Exception:
                pass
            await self._close_websocket_bounded(websocket, code=1001, timeout=0.25)
            return False
        self._cancel_grace_task()
        self.has_seen_browser_client = True
        clean_tab = (tab_id or "").strip() or conn_id
        prev_gen = self.tab_generations.get(clean_tab, 0)
        if isinstance(conn_gen, int) and not isinstance(conn_gen, bool) and conn_gen > prev_gen:
            assigned_gen = conn_gen
        else:
            assigned_gen = prev_gen + 1
        self.tab_generations[clean_tab] = assigned_gen
        self.connections[conn_id] = (clean_tab, assigned_gen, websocket)
        if len(self.tab_generations) > 512:
            active_tabs = {tab for tab, _gen, _ws in self.connections.values()}
            for stale_tab in list(self.tab_generations.keys()):
                if len(self.tab_generations) <= 256:
                    break
                if stale_tab not in active_tabs:
                    self.tab_generations.pop(stale_tab, None)
        return True

    def _maybe_schedule_grace_shutdown(self) -> None:
        if (
            self.shutting_down
            or not self.auto_shutdown_enabled
            or not self.has_seen_browser_client
            or len(self.connections) > 0
        ):
            return
        if self._grace_task is not None and not self._grace_task.done():
            return

        async def _grace_wait():
            current = None
            try:
                current = asyncio.current_task()
            except RuntimeError:
                pass
            try:
                await asyncio.sleep(self.grace_seconds)
                if self._grace_task is current:
                    self._grace_task = None
                if (
                    not self.shutting_down
                    and self.auto_shutdown_enabled
                    and self.has_seen_browser_client
                    and len(self.connections) == 0
                ):
                    await self.request_shutdown(reason="browser_closed")
            except asyncio.CancelledError:
                return

        self._grace_task = asyncio.create_task(_grace_wait())

    def unregister_connection(self, conn_id: str) -> None:
        removed = self.connections.pop(conn_id, None)
        if removed is not None:
            self._maybe_schedule_grace_shutdown()

    async def mark_tab_leaving(
        self,
        tab_id: Optional[str],
        conn_id: Optional[str] = None,
        conn_gen: Optional[int] = None,
    ) -> bool:
        clean_conn = conn_id.strip() if isinstance(conn_id, str) and conn_id.strip() else None
        if clean_conn is None:
            return False
        clean_tab = tab_id.strip() if isinstance(tab_id, str) and tab_id.strip() else None

        entry = self.connections.get(clean_conn)
        if entry is None:
            return False
        entry_tab, entry_gen, _ws = entry
        if clean_tab is not None and entry_tab != clean_tab:
            return False
        if conn_gen is not None:
            if not isinstance(conn_gen, int) or isinstance(conn_gen, bool) or conn_gen < entry_gen:
                return False

        popped = self.connections.pop(clean_conn, None)
        if popped is None:
            return False
        _ptab, _pgen, ws = popped
        await self._close_websocket_bounded(ws, code=1001, timeout=0.25)
        self._maybe_schedule_grace_shutdown()
        return True

    async def _cancel_active_jobs(self) -> list[str]:
        cancelled_ids: list[str] = []
        active_ids: set[str] = set()
        tasks = getattr(self.job_mgr, "tasks", None)
        if isinstance(tasks, dict):
            active_ids.update(tasks.keys())
        jobs = getattr(self.job_mgr, "jobs", None)
        if isinstance(jobs, dict):
            for jid, rec in list(jobs.items()):
                if isinstance(rec, dict) and rec.get("status") in {"pending", "processing"}:
                    active_ids.add(jid)
        for jid in sorted(active_ids):
            try:
                await asyncio.wait_for(self.job_mgr.cancel_job(jid), timeout=2.0)
                cancelled_ids.append(jid)
            except Exception as exc:
                logger.warning("Failed cancelling active job %s during shutdown: %s", jid, exc)
        return cancelled_ids

    async def _cancel_background_tasks(self) -> list[str]:
        cancelled_kinds: list[str] = []
        items = list(self.background_tasks.items())
        for task, kind in items:
            if not task.done():
                task.cancel()
                cancelled_kinds.append(kind)
            self.background_tasks.pop(task, None)
        pending = [task for task, _kind in items if not task.done()]
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=1.5,
                )
            except Exception:
                pass
        return cancelled_kinds

    def _purge_cancelled_artifacts(self, job_ids: list[str], cancelled_bg_kinds: list[str]) -> None:
        temp_dir = getattr(settings, "TEMP_DIR", None)
        upload_dir = getattr(settings, "UPLOAD_DIR", None)
        base_dir = getattr(settings, "BASE_DIR", None)
        for jid in job_ids:
            if temp_dir is not None and hasattr(temp_dir, "glob"):
                try:
                    for p in temp_dir.glob(f"{jid}*"):
                        try:
                            p.unlink(missing_ok=True)
                        except Exception:
                            pass
                except Exception:
                    pass
        if upload_dir is not None and hasattr(upload_dir, "glob"):
            try:
                for p in upload_dir.glob("*.part"):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                pass
        if "ffmpeg_install" in cancelled_bg_kinds and base_dir is not None:
            try:
                zip_dest = base_dir / "bin" / "ffmpeg.zip"
                if hasattr(zip_dest, "unlink"):
                    zip_dest.unlink(missing_ok=True)
            except Exception:
                pass

    async def request_shutdown(self, reason: str = "user_button") -> dict:
        async with self._get_lock():
            if self.shutting_down:
                return {
                    "success": True,
                    "already_shutting_down": True,
                    "reason": self.shutdown_reason,
                    "cancelled_jobs": [],
                    "message": "جاري إيقاف البرنامج بالفعل",
                }
            self.shutting_down = True
            self.shutdown_reason = reason
            self._cancel_grace_task()

            conns = list(self.connections.items())
            self.connections.clear()

            async def _notify_and_close(ws: Any) -> None:
                try:
                    await asyncio.wait_for(ws.send_json({"type": "shutdown", "reason": reason}), timeout=0.25)
                except Exception:
                    pass
                await self._close_websocket_bounded(ws, code=1001, timeout=0.25)

            if conns:
                await asyncio.gather(*(_notify_and_close(ws) for _cid, (_tid, _gen, ws) in conns))

            cancelled_bg: list[str] = []
            try:
                cancelled_bg = await self._cancel_background_tasks()
            except Exception as exc:
                logger.warning("Background task cancellation error during shutdown: %s", exc)

            cancelled_jobs: list[str] = []
            try:
                cancelled_jobs = await self._cancel_active_jobs()
            except Exception as exc:
                logger.warning("Active job cancellation error during shutdown: %s", exc)

            if cancelled_jobs or cancelled_bg:
                await asyncio.sleep(0.05)

            try:
                self.child_terminator()
            except Exception as exc:
                logger.warning("Child process termination error during shutdown: %s", exc)

            self._purge_cancelled_artifacts(cancelled_jobs, cancelled_bg)

            listeners = getattr(self.job_mgr, "listeners", None)
            if isinstance(listeners, dict):
                job_sockets = [job_ws for ws_list in list(listeners.values()) for job_ws in list(ws_list or [])]
                listeners.clear()
                if job_sockets:
                    await asyncio.gather(
                        *(self._close_websocket_bounded(job_ws, code=1001, timeout=0.25) for job_ws in job_sockets)
                    )

            try:
                self.exit_trigger(reason)
            except Exception as exc:
                logger.exception("Exit trigger failed: %s", exc)

            return {
                "success": True,
                "already_shutting_down": False,
                "reason": reason,
                "cancelled_jobs": cancelled_jobs,
                "cancelled_background_tasks": cancelled_bg,
                "message": "تم إيقاف البرنامج بالكامل",
            }


lifecycle_manager = BrowserLifecycleManager(job_mgr=job_manager)


@app.post("/api/shutdown")
async def shutdown_application(request: Request):
    supplied_token = lifecycle_manager._get_header(request.headers, "x-lifecycle-token")
    if not supplied_token:
        body_data = await _read_json_body_limited(request)
        raw_tok = body_data.get("token")
        if isinstance(raw_tok, str):
            supplied_token = raw_tok
    if not lifecycle_manager.is_authorized_request(request.headers, supplied_token):
        raise HTTPException(status_code=403, detail="طلب إيقاف غير مصرح به")
    return await lifecycle_manager.request_shutdown(reason="user_button")


@app.post("/api/lifecycle/leave")
async def lifecycle_leave(
    request: Request,
    tab_id: Optional[str] = Query(default=None),
    conn_id: Optional[str] = Query(default=None),
    gen: Optional[int] = Query(default=None),
):
    body_data = await _read_json_body_limited(request)
    supplied_token = lifecycle_manager._get_header(request.headers, "x-lifecycle-token")
    if not supplied_token:
        raw_tok = body_data.get("token")
        if isinstance(raw_tok, str):
            supplied_token = raw_tok
    if not lifecycle_manager.is_authorized_request(request.headers, supplied_token):
        raise HTTPException(status_code=403, detail="طلب دورة حياة غير مصرح به")

    resolved_tab_id = tab_id
    if not resolved_tab_id and isinstance(body_data.get("tab_id"), str):
        resolved_tab_id = body_data["tab_id"]

    resolved_conn_id = conn_id
    if not resolved_conn_id and isinstance(body_data.get("conn_id"), str):
        resolved_conn_id = body_data["conn_id"]

    resolved_gen = gen
    if resolved_gen is None and isinstance(body_data.get("gen"), int) and not isinstance(body_data.get("gen"), bool):
        resolved_gen = body_data["gen"]

    removed = await lifecycle_manager.mark_tab_leaving(
        resolved_tab_id,
        conn_id=resolved_conn_id,
        conn_gen=resolved_gen,
    )
    return {
        "success": True,
        "tab_removed": removed,
        "active_tabs": len(lifecycle_manager.connections),
    }


@app.websocket("/api/lifecycle/ws")
async def websocket_lifecycle_channel(
    websocket: WebSocket,
    tab_id: Optional[str] = Query(default=None),
    conn_id: Optional[str] = Query(default=None),
    gen: Optional[int] = Query(default=None),
    token: Optional[str] = Query(default=None),
):
    if not lifecycle_manager.is_authorized_request(websocket.headers, token):
        await websocket.close(code=1008)
        return
    clean_conn_id = (conn_id or "").strip()[:128]
    if clean_conn_id and clean_conn_id in lifecycle_manager.connections:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    resolved_conn_id = clean_conn_id or str(uuid.uuid4())
    active = await lifecycle_manager.register_connection(
        resolved_conn_id,
        tab_id or "",
        websocket,
        conn_gen=gen,
    )
    if not active:
        return
    try:
        while not lifecycle_manager.shutting_down:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        lifecycle_manager.unregister_connection(resolved_conn_id)


# =============================================================================
# WebSocket Real-Time Progress Channel
# =============================================================================

@app.websocket("/ws/{job_id}")
async def websocket_job_channel(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in job_manager.listeners:
        job_manager.listeners[job_id] = []
    job_manager.listeners[job_id].append(websocket)

    # Immediately send current state if available
    if job_id in job_manager.jobs:
        record = dict(job_manager.jobs[job_id])
        if "stages" not in record:
            record["stages"] = JobManager.build_stages("prepare", 0)
        await websocket.send_json(record)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if job_id in job_manager.listeners and websocket in job_manager.listeners[job_id]:
            job_manager.listeners[job_id].remove(websocket)
