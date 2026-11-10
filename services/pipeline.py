"""CinematicSRT v2.0 - Core Pipeline Orchestrator with Concurrency Locking & Subtitle Fidelity"""
import asyncio
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from types import SimpleNamespace
from typing import Dict, Any, Optional
import uuid

from core.config import settings
from services.audio import extract_and_enhance_audio
from services.transcriber import transcriber
from services.translator import translator
from services.subtitles import (
    sanitize_timing_without_drift,
    wrap_arabic_text,
    build_srt_content,
    build_ass_content,
    build_vtt_content,
    parse_srt_content,
    parse_ass_content,
    parse_vtt_content,
    validate_imported_cues,
    make_subtitle_document
)

logger = logging.getLogger(__name__)

# Global GPU concurrency lock: prevents simultaneous GPU memory allocations across jobs
gpu_lock = asyncio.Lock()

async def run_exclusive_worker(coro):
    """Executes a worker coroutine guaranteeing GPU lock exclusivity even under repeated cancellation."""
    task = asyncio.create_task(coro)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError()
    return task.result()

class JobManager:
    """Tracks active and historical jobs with WebSocket broadcast support."""
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.listeners: Dict[str, list] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    @staticmethod
    def build_stages(stage_key: str, stage_percent: Optional[int] = None, basis: str = "indeterminate", subtitle: bool = False, completed: bool = False):
        keys = ("prepare", "transcribe", "segment", "translate", "export")
        cur_idx = len(keys) if completed else (keys.index(stage_key) if stage_key in keys else -1)
        rows = []
        for idx, key in enumerate(keys):
            skipped = bool(subtitle and key in ("transcribe", "segment"))
            done = idx < cur_idx
            pct = 100 if done else (stage_percent if key == stage_key else None)
            state = (
                "skipped" if skipped
                else "done" if done
                else "active" if key == stage_key
                else "pending"
            )
            rows.append({
                "id": key,
                "percent": None if skipped else pct,
                "state": state,
                "basis": basis if key == stage_key else ("completed" if done else "indeterminate"),
            })
        return rows

    async def cancel_job(self, job_id: str):
        """Cancels an active job task and purges temporary files."""
        task = self.tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()
        if job_id in self.jobs:
            await self.update_job(
                job_id,
                status="cancelled",
                progress=0,
                stage="تم إلغاء المعالجة وتنظيف الملفات المؤقتة",
                error="تم الإلغاء بواسطة المستخدم"
            )
        # Purge temporary files
        for p in settings.TEMP_DIR.glob(f"{job_id}*"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        for p in settings.UPLOAD_DIR.glob(f"{job_id}*.part"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def create_job(self, job_id: str, filename: str) -> Dict[str, Any]:
        job_data = {
            "id": job_id,
            "filename": filename,
            "status": "pending",
            "progress": 0,
            "stage": "جاهز للبدء",
            "error": None,
            "srt_path": None,
            "ass_path": None,
            "vtt_path": None,
            "whisper_time": 0.0,
            "translation_time": 0.0,
            "total_time": 0.0,
            "segment_count": 0,
            "warnings_count": 0
        }
        self.jobs[job_id] = job_data
        return job_data

    async def update_job(self, job_id: str, status: str, progress: int, stage: str, **kwargs):
        if job_id in self.jobs:
            subtitle_flag = bool(kwargs.pop("subtitle_mode", False))
            stage_key = kwargs.pop("stage_key", None)
            stage_percent = kwargs.pop("stage_percent", None)
            stage_basis = kwargs.pop("stage_basis", "indeterminate")
            if status in ("completed", "completed_with_warnings") and "stages" not in kwargs:
                kwargs["stages"] = self.build_stages("export", 100, basis="completed", subtitle=subtitle_flag, completed=True)
            elif stage_key and "stages" not in kwargs:
                kwargs["stages"] = self.build_stages(stage_key, stage_percent, basis=stage_basis, subtitle=subtitle_flag, completed=False)
            elif status == "processing" and "stages" not in kwargs:
                inferred = (
                    "prepare" if progress < 25 else
                    "transcribe" if progress < 50 else
                    "segment" if progress < 55 else
                    "translate" if progress < 90 else
                    "export"
                )
                inferred_pct = stage_percent if stage_percent is not None else (progress if inferred in ("prepare", "export") else None)
                kwargs["stages"] = self.build_stages(inferred, inferred_pct, basis=stage_basis, subtitle=subtitle_flag, completed=False)
            self.jobs[job_id].update({
                "status": status,
                "progress": progress,
                "stage": stage,
                **kwargs
            })
            # Broadcast to connected WebSockets
            if job_id in self.listeners:
                msg = self.jobs[job_id]
                for ws in self.listeners[job_id]:
                    try:
                        await ws.send_json(msg)
                    except Exception:
                        pass

job_manager = JobManager()

def _is_reparse_or_link(path: Any) -> bool:
    """Returns True if the path is a symlink or Windows reparse point (junction/mount), even if dangling."""
    try:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return True
        reparse_attr = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if hasattr(st, "st_file_attributes") and bool(st.st_file_attributes & reparse_attr):
            return True
    except FileNotFoundError:
        return False
    except Exception:
        return True
    return False

def _validate_qc_output_path(output_dir: Any, valid_job_id: str) -> Optional[Path]:
    """Inspects lexical output directory, ancestors, and target file with non-following lstat before resolve."""
    try:
        raw_dir = Path(output_dir)
        if not raw_dir.is_absolute():
            raw_dir = raw_dir.absolute()

        # Check existing lexical ancestors and directory itself for symlinks/reparse points
        current = raw_dir
        chain = []
        while True:
            chain.append(current)
            parent = current.parent
            if parent == current:
                break
            current = parent
        chain.reverse()

        for comp in chain:
            if comp.exists() and _is_reparse_or_link(comp):
                logger.warning("QC output directory component is a symlink or reparse point")
                return None

        if not raw_dir.is_dir() or _is_reparse_or_link(raw_dir):
            logger.warning("QC output directory is invalid or reparse point")
            return None

        # Intended lexical target file
        target_file = raw_dir / f"{valid_job_id}.qc.json"
        if target_file.name != f"{valid_job_id}.qc.json" or target_file.parent != raw_dir:
            logger.warning("QC target file lexical path mismatch")
            return None

        # Non-following lstat check on target file (detects dangling links, symlinks, junctions)
        try:
            st = os.lstat(target_file)
            if stat.S_ISLNK(st.st_mode):
                logger.warning("QC target file is a symlink")
                return None
            reparse_attr = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            if hasattr(st, "st_file_attributes") and bool(st.st_file_attributes & reparse_attr):
                logger.warning("QC target file is a reparse point")
                return None
            if not stat.S_ISREG(st.st_mode):
                logger.warning("QC target file exists and is not a regular file")
                return None
        except FileNotFoundError:
            # Normal absence: target file does not yet exist
            pass
        except Exception:
            logger.warning("QC target file inspection failed")
            return None

        # Verify resolved directory matches and is not a reparse point
        resolved_dir = raw_dir.resolve()
        if not resolved_dir.is_dir() or _is_reparse_or_link(resolved_dir):
            logger.warning("QC resolved output directory is invalid or reparse point")
            return None

        return target_file
    except Exception:
        logger.warning("QC path validation failed")
        return None

def _validate_qc_job_id(job_id: str) -> str:
    """Validates that job_id is strictly a canonical lowercase hyphenated UUID string."""
    if not isinstance(job_id, str):
        raise TypeError(f"job_id must be a str, got {type(job_id).__name__}")
    try:
        parsed = uuid.UUID(job_id)
    except Exception as exc:
        raise ValueError(f"job_id is not a valid UUID: {job_id!r}") from exc
    canonical = str(parsed).lower()
    if job_id != canonical:
        raise ValueError(f"job_id must be canonical lowercase hyphenated UUID: expected {canonical!r}, got {job_id!r}")
    return canonical

def _clean_qc_filename(filename: str) -> str:
    """Validates filename has no NUL byte and extracts non-empty basename recognizing both / and \\."""
    if not isinstance(filename, str):
        raise TypeError(f"filename must be a str, got {type(filename).__name__}")
    if "\0" in filename:
        raise ValueError("filename contains NUL byte")
    cleaned = filename.replace("\\", "/").rstrip("/")
    basename = cleaned.split("/")[-1] if cleaned else ""
    if not basename:
        raise ValueError("filename must have a non-empty basename")
    return basename

def _validate_qc_created_at(created_at: str) -> str:
    """Validates that created_at is an ISO-8601 UTC string ending in 'Z'."""
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise ValueError(f"created_at must be an ISO-8601 string ending in 'Z', got {created_at!r}")
    try:
        dt = datetime.fromisoformat(created_at[:-1] + "+00:00")
        if dt.tzinfo != timezone.utc:
            raise ValueError("created_at does not denote UTC")
    except Exception as exc:
        raise ValueError(f"Invalid created_at ISO-8601 UTC timestamp: {created_at!r}") from exc
    return created_at

def _copy_qc_rejected_blocks(rejected_blocks) -> list:
    """Validates and deep-copies only allowed rejected block keys: block_index, line_number, raw_text, reason."""
    if rejected_blocks is None:
        return []
    if not isinstance(rejected_blocks, (list, tuple)):
        raise TypeError(f"rejected_blocks must be a list or tuple, got {type(rejected_blocks).__name__}")
    
    allowed_keys = {"block_index", "line_number", "raw_text", "reason"}
    copied = []
    for item in rejected_blocks:
        if not isinstance(item, dict):
            raise TypeError(f"rejected_blocks item must be a dict, got {type(item).__name__}")
        extra = set(item.keys()) - allowed_keys
        if extra:
            raise ValueError(f"rejected_blocks item contains disallowed keys: {extra}")
        entry = {}
        if "block_index" in item:
            val = item["block_index"]
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise ValueError(f"block_index must be a positive int, got {val!r}")
            entry["block_index"] = val
        if "line_number" in item:
            val = item["line_number"]
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise ValueError(f"line_number must be a positive int, got {val!r}")
            entry["line_number"] = val
        if "raw_text" in item:
            val = item["raw_text"]
            if not isinstance(val, str):
                raise TypeError(f"raw_text must be a str, got {type(val).__name__}")
            entry["raw_text"] = val
        if "reason" in item:
            val = item["reason"]
            if not isinstance(val, str):
                raise TypeError(f"reason must be a str, got {type(val).__name__}")
            entry["reason"] = val
        copied.append(entry)
    return copied

def build_qc_report(
    job_id: str,
    filename: str,
    pipeline_type: str,
    segments: Any,
    rejected_blocks: Any,
    created_at: str
) -> dict:
    """Pure, deterministic builder for the durable per-job Quality Control (QC) sidecar report."""
    valid_job_id = _validate_qc_job_id(job_id)
    clean_fn = _clean_qc_filename(filename)
    if pipeline_type not in ("subtitle_translation", "cinematic"):
        raise ValueError(f"pipeline_type must be 'subtitle_translation' or 'cinematic', got {pipeline_type!r}")
    valid_created_at = _validate_qc_created_at(created_at)

    if not isinstance(segments, (list, tuple)):
        raise TypeError(f"segments must be a list or tuple, got {type(segments).__name__}")

    copied_rejected_blocks = _copy_qc_rejected_blocks(rejected_blocks)

    issue_ledger = []
    all_warnings = []

    for ordinal, cue in enumerate(segments, start=1):
        # Extract source_index
        if isinstance(cue, dict):
            raw_idx = cue.get("index") if "index" in cue else cue.get("source_index", None)
        else:
            raw_idx = getattr(cue, "index", getattr(cue, "source_index", None))

        if raw_idx is None:
            source_index = None
        elif isinstance(raw_idx, bool):
            raise TypeError(f"cue index cannot be bool: {raw_idx!r}")
        elif isinstance(raw_idx, (int, str)):
            source_index = raw_idx
        else:
            raise TypeError(f"cue index must be int, str, or None, got {type(raw_idx).__name__}")

        # Extract start and end
        if isinstance(cue, dict):
            start = cue.get("start") if "start" in cue else cue.get("start_seconds", None)
            end = cue.get("end") if "end" in cue else cue.get("end_seconds", None)
        else:
            start = getattr(cue, "start", getattr(cue, "start_seconds", None))
            end = getattr(cue, "end", getattr(cue, "end_seconds", None))

        if start is None or end is None:
            raise ValueError(f"cue {ordinal} must specify start and end times")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise TypeError(f"start_seconds must be numeric (int/float), got {type(start).__name__}")
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            raise TypeError(f"end_seconds must be numeric (int/float), got {type(end).__name__}")
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"start_seconds and end_seconds must be finite numbers in cue {ordinal}")
        if start < 0:
            raise ValueError(f"start_seconds cannot be negative in cue {ordinal}: {start}")
        if end <= start:
            raise ValueError(f"end_seconds must be strictly greater than start_seconds in cue {ordinal}: start={start}, end={end}")

        duration = end - start
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"computed duration_seconds must be positive and finite in cue {ordinal}")

        # Extract warnings: missing attribute/key defaults to empty tuple; explicit non-sequence fails
        _MISSING = object()
        if isinstance(cue, dict):
            raw_warnings = cue.get("warnings", _MISSING)
        else:
            raw_warnings = getattr(cue, "warnings", _MISSING)

        if raw_warnings is _MISSING:
            raw_warnings = ()
        elif not isinstance(raw_warnings, (list, tuple)):
            raise TypeError(f"cue {ordinal} warnings must be list or tuple, got {type(raw_warnings).__name__}")

        cue_warnings = []
        for w in raw_warnings:
            if not isinstance(w, str):
                raise TypeError(f"cue {ordinal} warning must be str, got {type(w).__name__}")
            cue_warnings.append(w)

        all_warnings.append(cue_warnings)

        if cue_warnings:
            diagnostic_flags = []
            if "SOURCE_RETAINED_FOR_REVIEW" in cue_warnings:
                diagnostic_flags.append("SOURCE_RETENTION_REVIEW")
            if "ASS_SOURCE_EVENT_RETAINED_FOR_REVIEW" in cue_warnings:
                diagnostic_flags.append("ASS_SOURCE_RETENTION_REVIEW")
            if "UNSUPPORTED_LANGUAGE_FALLBACK" in cue_warnings:
                diagnostic_flags.append("LANGUAGE_FALLBACK_REVIEW")
            if "ASS_INLINE_SCOPE_TRANSFORMATION_REVIEW" in cue_warnings:
                diagnostic_flags.append("INLINE_SCOPE_REVIEW")

            issue_ledger.append({
                "ordinal": ordinal,
                "source_index": source_index,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": duration,
                "warnings": cue_warnings,
                "diagnostic_flags": diagnostic_flags
            })

    total_final_cues = len(segments)
    warned_cues = len(issue_ledger)
    clean_cues = total_final_cues - warned_cues
    warning_occurrences = sum(len(cw) for cw in all_warnings)
    source_retention_flagged_cues = sum(1 for cw in all_warnings if "SOURCE_RETAINED_FOR_REVIEW" in cw)
    ass_source_retention_flagged_cues = sum(1 for cw in all_warnings if "ASS_SOURCE_EVENT_RETAINED_FOR_REVIEW" in cw)
    any_retention_flagged_cues = sum(1 for cw in all_warnings if ("SOURCE_RETAINED_FOR_REVIEW" in cw or "ASS_SOURCE_EVENT_RETAINED_FOR_REVIEW" in cw))
    rejected_blocks_count = len(copied_rejected_blocks)
    review_status = "REVIEW_REQUIRED" if (warned_cues > 0 or rejected_blocks_count > 0) else "NO_RECORDED_ISSUES"

    return {
        "qc_version": "1.0",
        "job_id": valid_job_id,
        "filename": clean_fn,
        "pipeline_type": pipeline_type,
        "created_at": valid_created_at,
        "population": "final_export_cues",
        "summary": {
            "total_final_cues": total_final_cues,
            "clean_cues": clean_cues,
            "warned_cues": warned_cues,
            "warning_occurrences": warning_occurrences,
            "source_retention_flagged_cues": source_retention_flagged_cues,
            "ass_source_retention_flagged_cues": ass_source_retention_flagged_cues,
            "any_retention_flagged_cues": any_retention_flagged_cues,
            "rejected_blocks_count": rejected_blocks_count,
            "review_status": review_status
        },
        "issue_ledger": issue_ledger,
        "rejected_blocks": copied_rejected_blocks
    }

def persist_qc_report(output_dir: Any, job_id: str, report_data: dict) -> str:
    """Safely serializes and durably persists the QC report as a per-job sidecar JSON file."""
    try:
        valid_job_id = _validate_qc_job_id(job_id)
        if not isinstance(report_data, dict):
            logger.warning("QC report data must be a dict")
            return "failed"
        if report_data.get("job_id") != valid_job_id:
            logger.warning("QC report job_id does not match target job_id")
            return "failed"
        
        # Serialize before touching the filesystem; reject NaN/infinities strictly
        serialized = json.dumps(report_data, ensure_ascii=False, indent=2, allow_nan=False)

        # Lexical path inspection before resolve
        dest_file = _validate_qc_output_path(output_dir, valid_job_id)
        if dest_file is None:
            return "failed"

        # Create temporary file exclusively in the validated output directory
        raw_fd, temp_path = tempfile.mkstemp(dir=str(dest_file.parent), prefix=f".{valid_job_id}.qc.", suffix=".tmp")
    except Exception as exc:
        logger.warning("QC report pre-persistence failed: %s", type(exc).__name__)
        return "failed"

    stream = None
    try:
        # Wrap fd in text stream; transfer descriptor ownership immediately
        stream = open(raw_fd, "w", encoding="utf-8", closefd=True)
        raw_fd = None  # Stream now owns the descriptor exclusively

        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        stream = None  # Successfully closed

        os.replace(temp_path, dest_file)
        temp_path = None  # Consumed by atomic replacement
        return "written"
    except Exception as exc:
        logger.warning("QC report write/replace failed: %s", type(exc).__name__)
        if stream is not None:
            try:
                stream.close()
            except Exception:
                logger.warning("QC report stream cleanup failed")
            stream = None
        if raw_fd is not None:
            try:
                os.close(raw_fd)
            except Exception:
                logger.warning("QC report descriptor cleanup failed")
            raw_fd = None
        if temp_path is not None:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                logger.warning("QC report temporary file cleanup failed")
        return "failed"

class PublicationError(Exception):
    """Raised when subtitle output staging, verification, or replacement fails."""
    pass


def _validate_subtitle_output_paths(output_dir: Any, job_id: str) -> dict:
    """Validates job_id and output directory paths fail-closed without following symlinks."""
    try:
        valid_job_id = _validate_qc_job_id(job_id)
    except Exception as val_err:
        raise PublicationError(f"Invalid canonical job_id rejected: {job_id!r}") from val_err

    try:
        if isinstance(output_dir, Path):
            raw_dir = output_dir
        else:
            raw_dir = Path(str(output_dir))

        lexical_str = os.path.abspath(str(raw_dir))
        lexical_dir = Path(lexical_str)
    except Exception as path_err:
        raise PublicationError(f"Invalid output directory path: {path_err}") from path_err

    # Inspect lexical ancestors from root to leaf with non-following checks before follow-based checks
    try:
        ancestors = list(lexical_dir.parents)[::-1] + [lexical_dir]
        for ancestor in ancestors:
            if _is_reparse_or_link(ancestor):
                raise PublicationError(f"Symlink or reparse point rejected in directory chain: {ancestor}")

        if not lexical_dir.exists() or not lexical_dir.is_dir():
            raise PublicationError(f"Output directory does not exist or is not a directory: {lexical_dir}")
    except PublicationError:
        raise
    except Exception as stat_err:
        raise PublicationError(f"Failed inspecting output directory ancestors: {stat_err}") from stat_err

    destinations = {
        "srt": lexical_dir / f"{valid_job_id}.srt",
        "ass": lexical_dir / f"{valid_job_id}.ass",
        "vtt": lexical_dir / f"{valid_job_id}.vtt",
    }

    try:
        for fmt, dest in destinations.items():
            if dest.is_symlink() or _is_reparse_or_link(dest):
                raise PublicationError(f"Symlink or reparse point rejected at destination: {dest}")
            if dest.exists() and not dest.is_file():
                raise PublicationError(f"Destination exists but is not a regular file: {dest}")
    except PublicationError:
        raise
    except Exception as dest_err:
        raise PublicationError(f"Failed inspecting destination paths: {dest_err}") from dest_err

    return destinations


def publish_subtitle_triplet(
    output_dir: Any,
    job_id: str,
    srt_content: str,
    ass_content: str,
    vtt_content: str
) -> dict:
    """Synchronously stages and sequentially replaces subtitle triplet files in binary mode per file."""
    destinations = _validate_subtitle_output_paths(output_dir, job_id)
    out_dir = destinations["srt"].parent

    # Pre-encode all three complete payloads in memory
    try:
        payloads = {
            "srt": (srt_content.encode("utf-8-sig"), destinations["srt"]),
            "ass": (ass_content.encode("utf-8-sig"), destinations["ass"]),
            "vtt": (vtt_content.encode("utf-8"), destinations["vtt"]),
        }
    except Exception as enc_err:
        raise PublicationError(f"Failed pre-encoding subtitle payloads: {enc_err}") from enc_err

    staged = {}
    for fmt in ("srt", "ass", "vtt"):
        payload_bytes, dest_path = payloads[fmt]
        raw_fd = None
        stream = None
        staged_ok = False
        primary_exc = None
        try:
            try:
                raw_fd, tmp_path_str = tempfile.mkstemp(
                    dir=str(out_dir),
                    prefix=f".{job_id}.",
                    suffix=f".{fmt}.tmp"
                )
            except Exception as mk_err:
                raise PublicationError(f"Failed creating stage file for {fmt}: {mk_err}") from mk_err

            tmp_path = Path(tmp_path_str)

            # Wrap raw fd; transfer descriptor ownership immediately
            try:
                stream = open(raw_fd, "wb", closefd=True)
                raw_fd = None  # Transferred to stream
            except Exception as wrap_err:
                raise PublicationError(f"Failed wrapping descriptor for {fmt}: {wrap_err}") from wrap_err

            try:
                written = stream.write(payload_bytes)
                if written != len(payload_bytes):
                    raise PublicationError(f"Short write on {fmt}: wrote {written} of {len(payload_bytes)} bytes")

                stream.flush()
                os.fsync(stream.fileno())

                actual_size = os.fstat(stream.fileno()).st_size
                if actual_size != len(payload_bytes):
                    raise PublicationError(
                        f"Open descriptor size mismatch for {fmt}: expected {len(payload_bytes)}, got {actual_size}"
                    )

                # Detach stream reference before calling close so close errors are not retried in finally
                to_close = stream
                stream = None
                try:
                    to_close.close()
                except Exception as close_err:
                    raise PublicationError(f"Failed closing staging stream for {fmt}: {close_err}") from close_err

                staged[fmt] = (tmp_path, dest_path)
                staged_ok = True
            except PublicationError:
                raise
            except Exception as op_err:
                raise PublicationError(f"Staging write/verify failed for {fmt}: {op_err}") from op_err
        except BaseException as exc:
            primary_exc = exc
            raise
        finally:
            # Cleanup runs on both ordinary exceptions and BaseException (e.g. CancelledError)
            if not staged_ok:
                if stream is not None:
                    to_close = stream
                    stream = None
                    try:
                        to_close.close()
                    except BaseException as sec_err:
                        if primary_exc is not None:
                            if hasattr(primary_exc, "add_note"):
                                try:
                                    primary_exc.add_note(
                                        f"Secondary cleanup close error: {type(sec_err).__name__}: {sec_err}"
                                    )
                                except Exception:
                                    pass
                        else:
                            raise
                if raw_fd is not None:
                    to_close_fd = raw_fd
                    raw_fd = None
                    try:
                        os.close(to_close_fd)
                    except BaseException as sec_err:
                        if primary_exc is not None:
                            if hasattr(primary_exc, "add_note"):
                                try:
                                    primary_exc.add_note(
                                        f"Secondary cleanup raw close error: {type(sec_err).__name__}: {sec_err}"
                                    )
                                except Exception:
                                    pass
                        else:
                            raise

    # Sequential promotion via os.replace in fixed order: srt, ass, vtt
    for fmt in ("srt", "ass", "vtt"):
        tmp_path, dest_path = staged[fmt]
        try:
            os.replace(tmp_path, dest_path)
        except Exception as rep_err:
            raise PublicationError(
                f"Sequential replacement failed for {fmt} -> {dest_path.name}: {rep_err}"
            ) from rep_err

    return {
        "srt": destinations["srt"],
        "ass": destinations["ass"],
        "vtt": destinations["vtt"]
    }

async def run_cinematic_pipeline(
    job_id: str,
    video_path: Path,
    transcription_provider: str = "local",
    translation_provider: str = "local",
    language: Optional[str] = None,
    compute_mode: Optional[str] = None,
    transcription_mode: Optional[str] = None,
    whisper_model: Optional[str] = None
):
    """Executes the full cinema-grade speech-to-subtitle pipeline under shielded GPU lock."""
    start_total = time.time()
    logger.info("🎬 Starting CinematicSRT v2.0 pipeline for job=%s (%s)", job_id, video_path.name)
    temp_wav = settings.TEMP_DIR / f"{job_id}_16k.wav"

    try:
        job_manager.tasks[job_id] = asyncio.current_task()
    except Exception:
        pass

    effective_compute = compute_mode if compute_mode is not None else getattr(settings, "COMPUTE_MODE", None)
    effective_tmode = transcription_mode if transcription_mode is not None else getattr(settings, "DEFAULT_TRANSCRIPTION_MODE", None)
    effective_wmodel = whisper_model
    has_live_options = any(v is not None for v in (effective_compute, effective_tmode, effective_wmodel))
    enhance_audio = (effective_tmode != "normal") if effective_tmode is not None else True
    whisper_dev = ("cuda" if str(effective_compute).lower() == "gpu" else "cpu") if effective_compute is not None else None
    trans_dev = ("cpu" if str(effective_compute).lower() == "cpu" else "cuda") if effective_compute is not None else None

    def _transcribe_prog(pct, cur_sec=0.0, dur_sec=0.0):
        rec = job_manager.jobs.get(job_id)
        if type(rec) is dict and rec.get("status") != "processing":
            return
        overall = 25 + int(max(0, min(100, pct)) * 0.25)
        asyncio.create_task(
            job_manager.update_job(
                job_id,
                "processing",
                overall,
                f"تفريغ الصوت... {pct}%",
                stage_key="transcribe",
                stage_percent=pct,
                stage_basis="media_duration",
                duration_seconds=dur_sec
            )
        )

    transcribe_extra = {}
    translate_extra = {}
    if has_live_options:
        if effective_tmode is not None:
            transcribe_extra["transcription_mode"] = effective_tmode
        if whisper_dev is not None:
            transcribe_extra["device"] = whisper_dev
        if effective_wmodel is not None:
            transcribe_extra["whisper_model"] = effective_wmodel
        transcribe_extra["progress_cb"] = _transcribe_prog
        if trans_dev is not None:
            translate_extra["device"] = trans_dev

    async with gpu_lock:
        try:
            # Step 1: Audio Extraction & Preprocessing (16kHz mono WAV)
            await job_manager.update_job(job_id, "processing", 10, "استخراج وهندسة الصوت...", stage_key="prepare", stage_percent=35, stage_basis="extract")
            await extract_and_enhance_audio(video_path, temp_wav, enhance=enhance_audio)

            # Step 2: Speech-to-Text Transcription (Exclusive worker)
            await job_manager.update_job(job_id, "processing", 25, f"التفريغ الصوتي ({transcription_provider})...", stage_key="transcribe", stage_percent=0, stage_basis="media_duration")
            t_whisper_start = time.time()

            raw_segments, detected_lang, duration = await run_exclusive_worker(
                transcriber.transcribe(
                    temp_wav,
                    provider=transcription_provider,
                    language=language,
                    **transcribe_extra
                )
            )
            whisper_duration = time.time() - t_whisper_start

            if not raw_segments:
                raise RuntimeError("لم يتم العثور على حوار واضح في الملف الصوتي")

            if has_live_options:
                await job_manager.update_job(
                    job_id,
                    "processing",
                    52,
                    f"تجهيز مقاطع الترجمة ({len(raw_segments)} مقطع)...",
                    detected_language=detected_lang,
                    duration_seconds=duration,
                    stage_key="segment",
                    stage_percent=100,
                    stage_basis="cues"
                )

            # Step 3: Neural Translation (Exclusive worker)
            await job_manager.update_job(job_id, "processing", 55, f"الترجمة إلى العربية ({translation_provider})...", stage_key="translate", stage_percent=0, stage_basis="segments", detected_language=detected_lang)
            t_trans_start = time.time()

            def _trans_prog(pct):
                asyncio.create_task(job_manager.update_job(job_id, "processing", 55 + int(pct * 0.3), f"جارٍ الترجمة... {pct}%", stage_key="translate", stage_percent=pct, stage_basis="segments"))

            translated_segments = await run_exclusive_worker(
                translator.translate(
                    raw_segments,
                    source_language=detected_lang,
                    provider=translation_provider,
                    progress_cb=_trans_prog,
                    **translate_extra
                )
            )
            translation_duration = time.time() - t_trans_start

            # Step 4: Timing Sanitization (Drift-Free) & Subtitle Files Generation
            await job_manager.update_job(job_id, "processing", 90, "ضبط التوقيت وتوليد ملفات الترجمة...", stage_key="export", stage_percent=50, stage_basis="subtitles")
            sanitized_segments = sanitize_timing_without_drift(translated_segments)

            srt_content = build_srt_content(sanitized_segments)
            ass_content = build_ass_content(sanitized_segments, title=video_path.stem)
            vtt_content = build_vtt_content(sanitized_segments)

            publish_subtitle_triplet(
                output_dir=settings.OUTPUT_DIR,
                job_id=job_id,
                srt_content=srt_content,
                ass_content=ass_content,
                vtt_content=vtt_content
            )

            # Clean temporary audio buffer
            if temp_wav.exists():
                temp_wav.unlink(missing_ok=True)
            job_manager.tasks.pop(job_id, None)

            total_duration = time.time() - start_total
            total_warnings = sum(len(getattr(s, "warnings", ())) for s in sanitized_segments)
            final_status = "completed_with_warnings" if total_warnings > 0 else "completed"

            # Synchronous QC report persistence
            qc_report_status = "failed"
            qc_report_file = None
            try:
                utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                report_dict = build_qc_report(
                    job_id=job_id,
                    filename=video_path.name,
                    pipeline_type="cinematic",
                    segments=sanitized_segments,
                    rejected_blocks=[],
                    created_at=utc_now
                )
                outcome = persist_qc_report(settings.OUTPUT_DIR, job_id, report_dict)
                qc_report_status = outcome
                if outcome == "written":
                    qc_report_file = f"{job_id}.qc.json"
            except Exception as qc_exc:
                logger.warning("Cinematic QC persistence failed: %s", type(qc_exc).__name__)
                qc_report_status = "failed"
                qc_report_file = None

            await job_manager.update_job(
                job_id,
                status=final_status,
                progress=100,
                stage="اكتملت الترجمة بنجاح!" if final_status == "completed" else f"اكتملت مع {total_warnings} تنبيه تدقيق",
                srt_path=f"/api/download/{job_id}/srt",
                ass_path=f"/api/download/{job_id}/ass",
                vtt_path=f"/api/subtitles/{job_id}/vtt",
                video_url=f"/api/video/{job_id}",
                whisper_time=whisper_duration,
                translation_time=translation_duration,
                total_time=total_duration,
                segment_count=len(sanitized_segments),
                warnings_count=total_warnings,
                rejected_blocks=[],
                qc_report_status=qc_report_status,
                qc_report_file=qc_report_file,
                detected_language=detected_lang,
                duration_seconds=duration
            )

        except asyncio.CancelledError as original_cancel:
            try:
                job_record = job_manager.jobs.get(job_id)
                if type(job_record) is dict:
                    job_record.update({
                        "status": "failed",
                        "progress": 0,
                        "stage": "تم إلغاء المعالجة",
                        "error": "تم إلغاء العملية أثناء التشغيل",
                    })
            except Exception:
                pass
            raise original_cancel

        except Exception as e:
            logger.exception("Pipeline execution failed: %s", e)
            if temp_wav.exists():
                temp_wav.unlink(missing_ok=True)
            job_manager.tasks.pop(job_id, None)
            rej = getattr(locals().get("raw_segments", None), "rejected_blocks", [])
            await job_manager.update_job(
                job_id,
                status="failed",
                progress=0,
                stage=f"حدث خطأ: {str(e)}",
                error=str(e),
                rejected_blocks=rej
            )

async def run_subtitle_translation_pipeline(
    job_id: str,
    subtitle_path: Path,
    translation_provider: str = "local",
    source_language: Optional[str] = None,
    compute_mode: Optional[str] = None
):
    """Direct subtitle translation pipeline preserving 100% timing fidelity of imported files."""
    start_total = time.time()
    logger.info("📄 Direct subtitle translation for job=%s (%s)", job_id, subtitle_path.name)

    try:
        job_manager.tasks[job_id] = asyncio.current_task()
    except Exception:
        pass

    effective_compute = compute_mode if compute_mode is not None else getattr(settings, "COMPUTE_MODE", None)
    trans_dev = ("cpu" if str(effective_compute).lower() == "cpu" else "cuda") if effective_compute is not None else None
    translate_extra = {}
    if trans_dev is not None:
        translate_extra["device"] = trans_dev

    async with gpu_lock:
        try:
            await job_manager.update_job(job_id, "processing", 15, "قراءة وتحليل ملف الترجمة...", stage_key="prepare", stage_percent=50, stage_basis="parse", subtitle_mode=True)

            # Read file with encoding fallbacks
            content = ""
            for enc in ["utf-8-sig", "utf-8", "cp1256", "latin-1"]:
                try:
                    content = subtitle_path.read_text(encoding=enc)
                    break
                except Exception:
                    continue

            if not content:
                raise RuntimeError("فشل قراءة ملف الترجمة، يرجى التأكد من تشفير الملف.")

            # Parse segments based on extension
            ext = subtitle_path.suffix.lower()
            if ext == ".ass":
                raw_segments = parse_ass_content(content)
            elif ext == ".vtt":
                raw_segments = parse_vtt_content(content)
            else:
                raw_segments = parse_srt_content(content)

            if not raw_segments:
                raise RuntimeError("لم يتم العثور على أسطر ترجمة صالحة داخل الملف.")

            # Validate imported cues (rejects reversed/invalid intervals without retiming)
            raw_segments = validate_imported_cues(raw_segments)

            # Translate segments (Exclusive worker)
            await job_manager.update_job(job_id, "processing", 40, f"الترجمة إلى العربية ({translation_provider})...", stage_key="translate", stage_percent=0, stage_basis="segments", subtitle_mode=True)
            t_trans_start = time.time()

            def _trans_prog(pct):
                asyncio.create_task(job_manager.update_job(job_id, "processing", 40 + int(pct * 0.45), f"جارٍ الترجمة... {pct}%", stage_key="translate", stage_percent=pct, stage_basis="segments", subtitle_mode=True))

            translated_segments = await run_exclusive_worker(
                translator.translate(
                    raw_segments,
                    source_language=source_language or "en",
                    provider=translation_provider,
                    progress_cb=_trans_prog,
                    **translate_extra
                )
            )
            translation_duration = time.time() - t_trans_start

            # Preserve exact source timestamps; wrap text and evaluate layout overflow QC
            await job_manager.update_job(job_id, "processing", 90, "توليد ملفات الترجمة المعربة...", stage_key="export", stage_percent=50, stage_basis="subtitles", subtitle_mode=True)
            final_segments = []
            for s in translated_segments:
                s.lines = wrap_arabic_text(s.text, max_chars=42)
                warnings = list(getattr(s, "warnings", ()))
                if len(s.lines) > 2:
                    warnings.append("LAYOUT_OVERFLOW_WARNING")
                s.warnings = tuple(warnings)
                final_segments.append(s)

            final_doc = make_subtitle_document(
                segments=final_segments,
                rejected_blocks=getattr(raw_segments, "rejected_blocks", []),
                custom_styles=getattr(raw_segments, "custom_styles", []),
                raw_header=getattr(raw_segments, "raw_header", "")
            )

            srt_content = build_srt_content(final_doc)
            ass_content = build_ass_content(final_doc, title=subtitle_path.stem)
            vtt_content = build_vtt_content(final_doc)

            publish_subtitle_triplet(
                output_dir=settings.OUTPUT_DIR,
                job_id=job_id,
                srt_content=srt_content,
                ass_content=ass_content,
                vtt_content=vtt_content
            )
            job_manager.tasks.pop(job_id, None)

            total_duration = time.time() - start_total
            total_warnings = sum(len(getattr(s, "warnings", ())) for s in final_segments)
            final_status = "completed_with_warnings" if total_warnings > 0 else "completed"

            # Synchronous QC report persistence
            qc_report_status = "failed"
            qc_report_file = None
            try:
                utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                report_dict = build_qc_report(
                    job_id=job_id,
                    filename=subtitle_path.name,
                    pipeline_type="subtitle_translation",
                    segments=final_doc,
                    rejected_blocks=getattr(raw_segments, "rejected_blocks", []),
                    created_at=utc_now
                )
                outcome = persist_qc_report(settings.OUTPUT_DIR, job_id, report_dict)
                qc_report_status = outcome
                if outcome == "written":
                    qc_report_file = f"{job_id}.qc.json"
            except Exception as qc_exc:
                logger.warning("Subtitle QC persistence failed: %s", type(qc_exc).__name__)
                qc_report_status = "failed"
                qc_report_file = None

            await job_manager.update_job(
                job_id,
                status=final_status,
                progress=100,
                stage="اكتملت ترجمة الملف بنجاح!" if final_status == "completed" else f"اكتملت مع {total_warnings} تنبيه",
                srt_path=f"/api/download/{job_id}/srt",
                ass_path=f"/api/download/{job_id}/ass",
                vtt_path=f"/api/subtitles/{job_id}/vtt",
                video_url="",
                whisper_time=0.0,
                translation_time=translation_duration,
                total_time=total_duration,
                segment_count=len(final_segments),
                warnings_count=total_warnings,
                rejected_blocks=getattr(raw_segments, "rejected_blocks", []),
                qc_report_status=qc_report_status,
                qc_report_file=qc_report_file,
                subtitle_mode=True
            )

        except asyncio.CancelledError as original_cancel:
            try:
                job_record = job_manager.jobs.get(job_id)
                if type(job_record) is dict:
                    job_record.update({
                        "status": "failed",
                        "progress": 0,
                        "stage": "تم إلغاء المعالجة",
                        "error": "تم إلغاء العملية أثناء التشغيل",
                    })
            except Exception:
                pass
            raise original_cancel

        except Exception as e:
            logger.exception("Direct subtitle translation failed: %s", e)
            job_manager.tasks.pop(job_id, None)
            rej = getattr(locals().get("raw_segments", None), "rejected_blocks", [])
            await job_manager.update_job(
                job_id,
                status="failed",
                progress=0,
                stage=f"حدث خطأ: {str(e)}",
                error=str(e),
                rejected_blocks=rej
            )
