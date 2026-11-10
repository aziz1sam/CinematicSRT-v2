"""CinematicSRT v2.0 - Acceptance Test Suite for AIX-QC-PERSIST-001 v3-R1-T2

Completely isolated stdlib-only acceptance suite using AST extraction:
- Zero imports of production services (no config, audio, transcriber, translator, model, torch, or whisper).
- Lexical non-following lstat and Windows reparse point (0x400) guards before resolution or creation.
- Strict single-ownership descriptor lifecycle, stage-by-stage fault injection, and observable cleanup logging.
- Pure builder schema, flags, counters, duplicate index/warning preservation, and explicit invalid warnings rejection.
- Complete integration matrix for both coroutines including actual late ASS warning side effects and exporter write failures.
- Direct integration with real stdlib subtitle parsers and exporters via AST without third-party dependencies.
"""
import ast
import asyncio
from datetime import datetime, timezone
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict, Any, Optional, List, Tuple
import unittest
import uuid
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent

# --- AST Extraction of Subtitles Functions (Zero production/third-party imports) ---
_subtitles_source = (ROOT / "services" / "subtitles.py").read_text(encoding="utf-8")
_subtitles_tree = ast.parse(_subtitles_source)
_subtitles_nodes = [
    node for node in _subtitles_tree.body
    if not isinstance(node, (ast.Import, ast.ImportFrom))
]
_subtitles_ns = {
    "re": re,
    "SimpleNamespace": SimpleNamespace,
    "List": List,
    "Optional": Optional,
    "Tuple": Tuple,
    "Dict": Dict,
    "Any": Any,
}
_subtitles_code = compile(ast.Module(body=_subtitles_nodes, type_ignores=[]), "subtitles_extracted", "exec")
exec(_subtitles_code, _subtitles_ns)

# --- AST Extraction of Production Pipeline Functions and Classes ---
_pipeline_source = (ROOT / "services" / "pipeline.py").read_text(encoding="utf-8")
_pipeline_tree = ast.parse(_pipeline_source)

# Extract only non-import AST nodes (excluding production import statements and global module singletons)
_extracted_nodes = [
    node for node in _pipeline_tree.body
    if not isinstance(node, (ast.Import, ast.ImportFrom))
    and not (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("logger", "gpu_lock", "job_manager"))
]

_ast_logger = logging.getLogger("test_qc_ast")
_ast_logger.addHandler(logging.NullHandler())

# Synthetic isolated namespace providing stdlib types and safe mock dependencies
_pipeline_ns = {
    "asyncio": asyncio,
    "logging": logging,
    "time": SimpleNamespace(time=lambda: 1000.0),
    "Path": Path,
    "SimpleNamespace": SimpleNamespace,
    "Dict": Dict,
    "Any": Any,
    "Optional": Optional,
    "List": List,
    "datetime": datetime,
    "timezone": timezone,
    "json": json,
    "math": math,
    "os": os,
    "stat": stat,
    "tempfile": tempfile,
    "uuid": uuid,
    "logger": _ast_logger,
    "gpu_lock": asyncio.Lock(),
    "settings": SimpleNamespace(OUTPUT_DIR=Path("."), TEMP_DIR=Path(".")),
    # Pipeline dependency stubs & real extracted subtitle adapters
    "extract_and_enhance_audio": None,
    "transcriber": SimpleNamespace(),
    "translator": SimpleNamespace(),
    "sanitize_timing_without_drift": _subtitles_ns["sanitize_timing_without_drift"],
    "wrap_arabic_text": _subtitles_ns["wrap_arabic_text"],
    "build_srt_content": _subtitles_ns["build_srt_content"],
    "build_ass_content": _subtitles_ns["build_ass_content"],
    "build_vtt_content": _subtitles_ns["build_vtt_content"],
    "parse_srt_content": _subtitles_ns["parse_srt_content"],
    "parse_ass_content": _subtitles_ns["parse_ass_content"],
    "parse_vtt_content": _subtitles_ns["parse_vtt_content"],
    "validate_imported_cues": _subtitles_ns["validate_imported_cues"],
    "make_subtitle_document": _subtitles_ns["make_subtitle_document"]
}

_compiled_code = compile(ast.Module(body=_extracted_nodes, type_ignores=[]), "pipeline_extracted", "exec")
exec(_compiled_code, _pipeline_ns)

# Bind extracted functions into module scope
_is_reparse_or_link = _pipeline_ns["_is_reparse_or_link"]
_validate_qc_job_id = _pipeline_ns["_validate_qc_job_id"]
_clean_qc_filename = _pipeline_ns["_clean_qc_filename"]
_validate_qc_created_at = _pipeline_ns["_validate_qc_created_at"]
_copy_qc_rejected_blocks = _pipeline_ns["_copy_qc_rejected_blocks"]
_validate_qc_output_path = _pipeline_ns["_validate_qc_output_path"]
build_qc_report = _pipeline_ns["build_qc_report"]
persist_qc_report = _pipeline_ns["persist_qc_report"]
JobManager = _pipeline_ns["JobManager"]
run_exclusive_worker = _pipeline_ns["run_exclusive_worker"]
run_cinematic_pipeline = _pipeline_ns["run_cinematic_pipeline"]
run_subtitle_translation_pipeline = _pipeline_ns["run_subtitle_translation_pipeline"]


class TestNoProductionImportsProof(unittest.TestCase):
    """Explicitly verifies that no production heavyweight or ML modules are imported by this test file."""

    def test_this_module_imports_no_production_services(self):
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in ("core", "services", "torch", "transformers", "whisper"):
                    self.assertFalse(node.module.startswith(forbidden), f"Direct import forbidden: {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in ("core", "services", "torch", "transformers", "whisper"):
                        self.assertFalse(alias.name.startswith(forbidden), f"Direct import forbidden: {alias.name}")

    def test_isolated_process_loads_no_production_stack(self):
        import subprocess
        script = (
            "import sys, unittest\n"
            "loader = unittest.TestLoader()\n"
            "suite = loader.loadTestsFromName('test_qc_persistence')\n"
            "failed = []\n"
            "def check(s):\n"
            "    for i in s:\n"
            "        if isinstance(i, unittest.TestSuite):\n"
            "            check(i)\n"
            "        elif isinstance(i, unittest.loader._FailedTest):\n"
            "            failed.append(str(i))\n"
            "check(suite)\n"
            "assert not failed, f'Loading failed: {failed}'\n"
            "assert suite.countTestCases() >= 40, f'Expected >= 40 tests, got {suite.countTestCases()}'\n"
            "forbidden = ['core.config', 'services.audio', 'services.transcriber', 'services.translator', 'torch', 'transformers', 'whisper']\n"
            "loaded = [m for m in forbidden if m in sys.modules]\n"
            "assert not loaded, f'Loaded forbidden modules: {loaded}'\n"
            "print('ISOLATED_OK')\n"
        )
        cmd = [sys.executable, "-B", "-c", script]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        res = subprocess.run(cmd, cwd=str(ROOT / "tests"), env=env, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Subprocess failed:\n{res.stderr}")
        self.assertIn("ISOLATED_OK", res.stdout)


class TestQCJobIdValidation(unittest.TestCase):
    """Verifies strict lowercase canonical hyphenated UUID validation."""

    def test_canonical_uuid_success(self):
        canonical = str(uuid.uuid4()).lower()
        self.assertEqual(_validate_qc_job_id(canonical), canonical)

    def test_non_canonical_variants_rejected(self):
        valid = str(uuid.uuid4()).lower()
        with self.assertRaises(ValueError):
            _validate_qc_job_id(valid.upper())
        with self.assertRaises(ValueError):
            _validate_qc_job_id(f"{{{valid}}}")
        with self.assertRaises(ValueError):
            _validate_qc_job_id(valid.replace("-", ""))
        with self.assertRaises(ValueError):
            _validate_qc_job_id(f"  {valid}  ")

    def test_path_traversal_and_invalid_strings_rejected(self):
        with self.assertRaises(ValueError):
            _validate_qc_job_id("..\\..\\malicious")
        with self.assertRaises(ValueError):
            _validate_qc_job_id("job:alternate_stream")
        with self.assertRaises(ValueError):
            _validate_qc_job_id("abc\0def")
        with self.assertRaises(ValueError):
            _validate_qc_job_id("")

    def test_non_string_types_rejected(self):
        with self.assertRaises(TypeError):
            _validate_qc_job_id(12345)
        with self.assertRaises(TypeError):
            _validate_qc_job_id(None)
        with self.assertRaises(TypeError):
            _validate_qc_job_id(True)


class TestQCFilenameValidation(unittest.TestCase):
    """Verifies filename sanitization and basename extraction."""

    def test_valid_basenames(self):
        self.assertEqual(_clean_qc_filename("movie.mp4"), "movie.mp4")
        self.assertEqual(_clean_qc_filename("C:\\data\\media\\feature.mkv"), "feature.mkv")
        self.assertEqual(_clean_qc_filename("/var/subtitles/arabic.srt"), "arabic.srt")
        self.assertEqual(_clean_qc_filename("nested/path\\mixed/clip.ts"), "clip.ts")

    def test_empty_or_nul_filenames_rejected(self):
        with self.assertRaises(ValueError):
            _clean_qc_filename("")
        with self.assertRaises(ValueError):
            _clean_qc_filename("///")
        with self.assertRaises(ValueError):
            _clean_qc_filename("bad\0file.mp4")

    def test_non_string_rejected(self):
        with self.assertRaises(TypeError):
            _clean_qc_filename(None)
        with self.assertRaises(TypeError):
            _clean_qc_filename(100)


class TestQCCreatedAtValidation(unittest.TestCase):
    """Verifies ISO-8601 UTC timestamp requirements."""

    def test_valid_utc_timestamps(self):
        ts = "2026-11-05T07:30:00Z"
        self.assertEqual(_validate_qc_created_at(ts), ts)
        ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.assertEqual(_validate_qc_created_at(ts_now), ts_now)

    def test_non_z_or_local_offsets_rejected(self):
        with self.assertRaises(ValueError):
            _validate_qc_created_at("2026-11-05T07:30:00+03:00")
        with self.assertRaises(ValueError):
            _validate_qc_created_at("2026-11-05 07:30:00")
        with self.assertRaises(ValueError):
            _validate_qc_created_at("invalid_timestamp")
        with self.assertRaises(ValueError):
            _validate_qc_created_at(None)


class TestQCRejectedBlocks(unittest.TestCase):
    """Verifies parser rejection records deep-copy across SRT/ASS/VTT formats."""

    def test_empty_or_none(self):
        self.assertEqual(_copy_qc_rejected_blocks(None), [])
        self.assertEqual(_copy_qc_rejected_blocks([]), [])

    def test_srt_rejection_block_with_physical_line(self):
        blocks = [{
            "block_index": 3,
            "line_number": 12,
            "raw_text": "corrupted text",
            "reason": "INVALID_TIMING"
        }]
        copied = _copy_qc_rejected_blocks(blocks)
        self.assertEqual(copied, blocks)
        self.assertIsNot(copied[0], blocks[0])
        self.assertEqual(copied[0]["line_number"], 12)

    def test_ass_rejection_block_missing_block_index(self):
        blocks = [{
            "line_number": 45,
            "raw_text": "Dialogue: 0,bad_timing",
            "reason": "MALFORMED_EVENT"
        }]
        copied = _copy_qc_rejected_blocks(blocks)
        self.assertEqual(copied, blocks)
        self.assertNotIn("block_index", copied[0])

    def test_vtt_rejection_block_missing_line_number(self):
        blocks = [{
            "block_index": 7,
            "raw_text": "NOTE Comment",
            "reason": "NOTE_BLOCK_IGNORED"
        }]
        copied = _copy_qc_rejected_blocks(blocks)
        self.assertEqual(copied, blocks)
        self.assertNotIn("line_number", copied[0])

    def test_disallowed_keys_rejected(self):
        blocks = [{
            "line_number": 1,
            "raw_text": "foo",
            "reason": "bad",
            "secret_key": "unauthorized"
        }]
        with self.assertRaises(ValueError):
            _copy_qc_rejected_blocks(blocks)

    def test_invalid_types_rejected(self):
        with self.assertRaises(ValueError):
            _copy_qc_rejected_blocks([{"block_index": -1, "raw_text": "x", "reason": "y"}])
        with self.assertRaises(ValueError):
            _copy_qc_rejected_blocks([{"block_index": True, "raw_text": "x", "reason": "y"}])
        with self.assertRaises(TypeError):
            _copy_qc_rejected_blocks([{"block_index": 1, "raw_text": 123, "reason": "y"}])

    def test_real_srt_parser_rejected_block_physical_line_in_qc(self):
        # Actual stdlib SRT parser produces rejected_blocks with physical line numbers flowing into QC
        malformed_srt = (
            "1\n00:00:01,000 --> 00:00:02,000\nValid line\n\n"
            "NOT_A_NUMBER\n00:00:03,000 --> 00:00:04,000\nMalformed block\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\nAnother valid\n"
        )
        doc = _subtitles_ns["parse_srt_content"](malformed_srt)
        rejected = getattr(doc, "rejected_blocks", [])
        self.assertGreater(len(rejected), 0)
        self.assertEqual(rejected[0]["line_number"], 5)
        report = build_qc_report(
            job_id=str(uuid.uuid4()).lower(),
            filename="sample.srt",
            pipeline_type="subtitle_translation",
            segments=list(doc),
            rejected_blocks=rejected,
            created_at="2026-11-05T07:30:00Z"
        )
        self.assertEqual(len(report["rejected_blocks"]), len(rejected))
        self.assertEqual(report["rejected_blocks"][0]["line_number"], 5)
        self.assertEqual(report["rejected_blocks"][0]["block_index"], 2)
        self.assertEqual(report["summary"]["rejected_blocks_count"], len(rejected))
        self.assertEqual(report["summary"]["review_status"], "REVIEW_REQUIRED")

    def test_real_ass_parser_rejected_block_in_qc(self):
        # Actual stdlib ASS parser produces rejected_blocks with physical line numbers flowing into QC
        malformed_ass = (
            "[Script Info]\nTitle: Test\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Valid\n"
            "Dialogue: INVALID_PARTS\n"
        )
        doc = _subtitles_ns["parse_ass_content"](malformed_ass)
        rejected = getattr(doc, "rejected_blocks", [])
        self.assertGreater(len(rejected), 0)
        report = build_qc_report(
            job_id=str(uuid.uuid4()).lower(),
            filename="sample.ass",
            pipeline_type="subtitle_translation",
            segments=list(doc),
            rejected_blocks=rejected,
            created_at="2026-11-05T07:30:00Z"
        )
        self.assertEqual(len(report["rejected_blocks"]), len(rejected))
        self.assertEqual(report["rejected_blocks"][0]["line_number"], 6)

    def test_real_vtt_parser_rejected_block_in_qc(self):
        # Actual stdlib VTT parser produces rejected_blocks with block index flowing into QC
        malformed_vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:02.000\nValid\n\n"
            "BAD_TIMESTAMP --> INVALID\nMalformed\n"
        )
        doc = _subtitles_ns["parse_vtt_content"](malformed_vtt)
        rejected = getattr(doc, "rejected_blocks", [])
        self.assertGreater(len(rejected), 0)
        report = build_qc_report(
            job_id=str(uuid.uuid4()).lower(),
            filename="sample.vtt",
            pipeline_type="subtitle_translation",
            segments=list(doc),
            rejected_blocks=rejected,
            created_at="2026-11-05T07:30:00Z"
        )
        self.assertEqual(len(report["rejected_blocks"]), len(rejected))
        self.assertEqual(report["rejected_blocks"][0]["block_index"], 3)


class TestQCReportBuilder(unittest.TestCase):
    """Verifies pure deterministic report builder contract, schema, and diagnostics."""

    def setUp(self):
        self.job_id = str(uuid.uuid4()).lower()
        self.created_at = "2026-11-05T07:30:00Z"

    def test_all_clean_cues_and_missing_attribute_default(self):
        cues = [
            SimpleNamespace(index=1, start=1.0, end=3.5, text="مرحبا"),
            SimpleNamespace(index=2, start=4.0, end=6.0, text="كيف حالك", warnings=[])
        ]
        report = build_qc_report(
            job_id=self.job_id,
            filename="sample.mp4",
            pipeline_type="cinematic",
            segments=cues,
            rejected_blocks=[],
            created_at=self.created_at
        )

        self.assertEqual(report["qc_version"], "1.0")
        self.assertEqual(report["job_id"], self.job_id)
        self.assertEqual(report["filename"], "sample.mp4")
        self.assertEqual(report["pipeline_type"], "cinematic")
        self.assertEqual(report["created_at"], self.created_at)
        self.assertEqual(report["population"], "final_export_cues")

        summary = report["summary"]
        self.assertEqual(summary["total_final_cues"], 2)
        self.assertEqual(summary["clean_cues"], 2)
        self.assertEqual(summary["warned_cues"], 0)
        self.assertEqual(summary["warning_occurrences"], 0)
        self.assertEqual(summary["source_retention_flagged_cues"], 0)
        self.assertEqual(summary["ass_source_retention_flagged_cues"], 0)
        self.assertEqual(summary["any_retention_flagged_cues"], 0)
        self.assertEqual(summary["rejected_blocks_count"], 0)
        self.assertEqual(summary["review_status"], "NO_RECORDED_ISSUES")
        self.assertEqual(report["issue_ledger"], [])

    def test_explicit_none_or_non_sequence_warnings_rejected(self):
        bad_cues_none = [SimpleNamespace(index=1, start=1.0, end=2.0, warnings=None)]
        with self.assertRaises(TypeError):
            build_qc_report(self.job_id, "v.mp4", "cinematic", bad_cues_none, [], self.created_at)

        bad_cues_int = [SimpleNamespace(index=1, start=1.0, end=2.0, warnings=123)]
        with self.assertRaises(TypeError):
            build_qc_report(self.job_id, "v.mp4", "cinematic", bad_cues_int, [], self.created_at)

        bad_cues_str = [SimpleNamespace(index=1, start=1.0, end=2.0, warnings="WARNING_STR")]
        with self.assertRaises(TypeError):
            build_qc_report(self.job_id, "v.mp4", "cinematic", bad_cues_str, [], self.created_at)

    def test_warned_cues_and_diagnostic_flags_ordering(self):
        cues = [
            SimpleNamespace(
                index="cue_1",
                start=0.5,
                end=2.0,
                warnings=[
                    "UNSUPPORTED_LANGUAGE_FALLBACK",
                    "SOURCE_RETAINED_FOR_REVIEW",
                    "UNKNOWN_CUSTOM_WARN",
                    "SOURCE_RETAINED_FOR_REVIEW"
                ]
            ),
            SimpleNamespace(
                index="cue_2",
                start=2.5,
                end=4.0,
                warnings=[
                    "ASS_INLINE_SCOPE_TRANSFORMATION_REVIEW",
                    "ASS_SOURCE_EVENT_RETAINED_FOR_REVIEW"
                ]
            ),
            SimpleNamespace(
                index="cue_3",
                start=5.0,
                end=7.0,
                warnings=()
            )
        ]

        report = build_qc_report(
            job_id=self.job_id,
            filename="C:\\path\\video.mkv",
            pipeline_type="subtitle_translation",
            segments=cues,
            rejected_blocks=[{"line_number": 5, "raw_text": "err", "reason": "SYNTAX"}],
            created_at=self.created_at
        )

        summary = report["summary"]
        self.assertEqual(summary["total_final_cues"], 3)
        self.assertEqual(summary["warned_cues"], 2)
        self.assertEqual(summary["clean_cues"], 1)
        self.assertEqual(summary["clean_cues"] + summary["warned_cues"], summary["total_final_cues"])
        self.assertEqual(summary["warning_occurrences"], 6)
        self.assertEqual(summary["source_retention_flagged_cues"], 1)
        self.assertEqual(summary["ass_source_retention_flagged_cues"], 1)
        self.assertEqual(summary["any_retention_flagged_cues"], 2)
        self.assertEqual(summary["rejected_blocks_count"], 1)
        self.assertEqual(summary["review_status"], "REVIEW_REQUIRED")

        ledger = report["issue_ledger"]
        self.assertEqual(len(ledger), 2)

        item1 = ledger[0]
        self.assertEqual(item1["ordinal"], 1)
        self.assertEqual(item1["source_index"], "cue_1")
        self.assertEqual(item1["duration_seconds"], 1.5)
        self.assertEqual(item1["diagnostic_flags"], ["SOURCE_RETENTION_REVIEW", "LANGUAGE_FALLBACK_REVIEW"])
        self.assertEqual(len(item1["warnings"]), 4)

        item2 = ledger[1]
        self.assertEqual(item2["ordinal"], 2)
        self.assertEqual(item2["source_index"], "cue_2")
        self.assertEqual(item2["diagnostic_flags"], ["ASS_SOURCE_RETENTION_REVIEW", "INLINE_SCOPE_REVIEW"])

    def test_duplicate_source_indexes_preserved(self):
        cues = [
            SimpleNamespace(index=1, start=1.0, end=2.0, warnings=["WARN1"]),
            SimpleNamespace(index=1, start=3.0, end=4.0, warnings=["WARN2"])
        ]
        report = build_qc_report(self.job_id, "v.mp4", "cinematic", cues, [], self.created_at)
        ledger = report["issue_ledger"]
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger[0]["ordinal"], 1)
        self.assertEqual(ledger[0]["source_index"], 1)
        self.assertEqual(ledger[1]["ordinal"], 2)
        self.assertEqual(ledger[1]["source_index"], 1)

    def test_invalid_pipeline_type_rejected(self):
        cues = [SimpleNamespace(index=1, start=1.0, end=2.0)]
        with self.assertRaises(ValueError):
            build_qc_report(self.job_id, "v.mp4", "unsupported_pipeline", cues, [], self.created_at)

    def test_non_finite_and_invalid_timestamps_rejected(self):
        bad_nan = [SimpleNamespace(index=1, start=float("nan"), end=2.0)]
        with self.assertRaises(ValueError):
            build_qc_report(self.job_id, "v.mp4", "cinematic", bad_nan, [], self.created_at)

        bad_inf = [SimpleNamespace(index=1, start=1.0, end=float("inf"))]
        with self.assertRaises(ValueError):
            build_qc_report(self.job_id, "v.mp4", "cinematic", bad_inf, [], self.created_at)

        negative_start = [SimpleNamespace(index=1, start=-0.5, end=2.0)]
        with self.assertRaises(ValueError):
            build_qc_report(self.job_id, "v.mp4", "cinematic", negative_start, [], self.created_at)

        reversed_interval = [SimpleNamespace(index=1, start=3.0, end=2.0)]
        with self.assertRaises(ValueError):
            build_qc_report(self.job_id, "v.mp4", "cinematic", reversed_interval, [], self.created_at)

        bool_timestamp = [SimpleNamespace(index=1, start=False, end=2.0)]
        with self.assertRaises(TypeError):
            build_qc_report(self.job_id, "v.mp4", "cinematic", bool_timestamp, [], self.created_at)

    def test_caller_immutability(self):
        cue = SimpleNamespace(index=1, start=1.0, end=2.0, warnings=["WARN1"])
        rejected = [{"block_index": 1, "raw_text": "t", "reason": "r"}]
        report = build_qc_report(self.job_id, "v.mp4", "cinematic", [cue], rejected, self.created_at)
        self.assertEqual(cue.warnings, ["WARN1"])
        self.assertEqual(rejected, [{"block_index": 1, "raw_text": "t", "reason": "r"}])

        # Mutate caller containers after build and verify report remains unchanged
        cue.warnings.append("MUTATED_WARN")
        rejected.append({"block_index": 2, "raw_text": "t2", "reason": "r2"})
        self.assertEqual(len(report["issue_ledger"]), 1)
        self.assertEqual(report["issue_ledger"][0]["warnings"], ["WARN1"])
        self.assertEqual(len(report["rejected_blocks"]), 1)
        self.assertEqual(report["summary"]["rejected_blocks_count"], 1)

        # Mutate report containers and verify caller containers are not affected
        report["issue_ledger"][0]["warnings"].append("REPORT_MUTATION")
        report["rejected_blocks"].append({"block_index": 3, "raw_text": "t3", "reason": "r3"})
        self.assertEqual(cue.warnings, ["WARN1", "MUTATED_WARN"])
        self.assertEqual(len(rejected), 2)

    def test_exact_schema_and_unicode_serialization(self):
        # Strict verification of exact schema keys, pure builder determinism, and UTF-8 preservation
        cues = [
            SimpleNamespace(index="فصل_1", start=1.5, end=3.5, warnings=["تحذير_مخصص_ARABIC"]),
            SimpleNamespace(index=2, start=4.0, end=5.0, warnings=())
        ]
        rejected = [{
            "block_index": 1,
            "line_number": 10,
            "raw_text": "حوار: مرحبا بالعالم",
            "reason": "MALFORMED_TIMING"
        }]
        report = build_qc_report(
            job_id=self.job_id,
            filename="فيديو_تجريبي.mp4",
            pipeline_type="cinematic",
            segments=cues,
            rejected_blocks=rejected,
            created_at=self.created_at
        )
        required_root_keys = {"qc_version", "job_id", "filename", "pipeline_type", "created_at", "population", "summary", "issue_ledger", "rejected_blocks"}
        self.assertEqual(set(report.keys()), required_root_keys)

        # Verify exact nested schema keys for summary
        required_summary_keys = {
            "total_final_cues", "clean_cues", "warned_cues", "warning_occurrences",
            "source_retention_flagged_cues", "ass_source_retention_flagged_cues",
            "any_retention_flagged_cues", "rejected_blocks_count", "review_status"
        }
        self.assertEqual(set(report["summary"].keys()), required_summary_keys)

        # Verify exact nested schema keys for issue_ledger items
        required_issue_keys = {
            "ordinal", "source_index", "start_seconds", "end_seconds",
            "duration_seconds", "warnings", "diagnostic_flags"
        }
        self.assertEqual(set(report["issue_ledger"][0].keys()), required_issue_keys)

        # Verify exact nested schema keys for rejected_blocks items
        required_rejected_keys = {"block_index", "line_number", "raw_text", "reason"}
        self.assertEqual(set(report["rejected_blocks"][0].keys()), required_rejected_keys)

        self.assertEqual(report["filename"], "فيديو_تجريبي.mp4")
        self.assertEqual(report["issue_ledger"][0]["source_index"], "فصل_1")
        self.assertEqual(report["issue_ledger"][0]["warnings"], ["تحذير_مخصص_ARABIC"])
        self.assertEqual(report["rejected_blocks"][0]["raw_text"], "حوار: مرحبا بالعالم")

        # Serialized JSON roundtrip through json.dumps
        raw_json = json.dumps(report, ensure_ascii=False)
        self.assertIn("فيديو_تجريبي.mp4", raw_json)
        self.assertIn("تحذير_مخصص_ARABIC", raw_json)
        deserialized = json.loads(raw_json)
        self.assertEqual(deserialized, report)

        # Persist and parse through actual persist_qc_report in an owned temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            res = persist_qc_report(tmpdir, self.job_id, report)
            self.assertEqual(res, "written")
            written_path = Path(tmpdir) / f"{self.job_id}.qc.json"
            self.assertTrue(written_path.exists())
            loaded_data = json.loads(written_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_data, report)
            self.assertEqual(loaded_data["filename"], "فيديو_تجريبي.mp4")
            self.assertEqual(loaded_data["issue_ledger"][0]["warnings"], ["تحذير_مخصص_ARABIC"])


class TestQCLexicalPathGuards(unittest.TestCase):
    """Verifies non-following lstat and Windows reparse point checks reject aliases before mkstemp."""

    def setUp(self):
        self.job_id = str(uuid.uuid4()).lower()
        self.report_data = {
            "qc_version": "1.0",
            "job_id": self.job_id,
            "filename": "video.mp4",
            "pipeline_type": "cinematic",
            "created_at": "2026-11-05T07:30:00Z",
            "population": "final_export_cues",
            "summary": {"review_status": "NO_RECORDED_ISSUES"},
            "issue_ledger": [],
            "rejected_blocks": []
        }

    def test_directory_reparse_point_rejected_before_mkstemp(self):
        real_lstat = os.lstat
        with tempfile.TemporaryDirectory() as tmpdir:
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("protected", encoding="utf-8")
            dir_path = Path(tmpdir)

            def fake_lstat(path, *args, **kwargs):
                p = Path(path)
                if p == dir_path:
                    mock_stat = MagicMock()
                    mock_stat.st_mode = stat.S_IFDIR | 0o755
                    mock_stat.st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                    return mock_stat
                return real_lstat(path, *args, **kwargs)

            mkstemp_called = []
            def fake_mkstemp(*args, **kwargs):
                mkstemp_called.append(True)
                raise AssertionError("mkstemp must not be reached when directory is a reparse point")

            captured_logs = []
            class LogCap(logging.Handler):
                def emit(self, record): captured_logs.append(record.getMessage())
            handler = LogCap()
            _ast_logger.addHandler(handler)

            try:
                with patch("os.lstat", side_effect=fake_lstat), \
                     patch("tempfile.mkstemp", side_effect=fake_mkstemp):
                    res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                    self.assertEqual(res, "failed")
                    self.assertEqual(mkstemp_called, [])
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "protected")
                    self.assertTrue(any("symlink or reparse point" in m for m in captured_logs))
            finally:
                _ast_logger.removeHandler(handler)

    def test_ancestor_directory_reparse_point_rejected(self):
        real_lstat = os.lstat
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_dir = Path(tmpdir) / "sub1" / "sub2"
            sub_dir.mkdir(parents=True)
            ancestor = Path(tmpdir) / "sub1"

            def fake_lstat(path, *args, **kwargs):
                p = Path(path)
                if p == ancestor:
                    mock_stat = MagicMock()
                    mock_stat.st_mode = stat.S_IFDIR | 0o755
                    mock_stat.st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                    return mock_stat
                return real_lstat(path, *args, **kwargs)

            captured_logs = []
            class LogCap(logging.Handler):
                def emit(self, record): captured_logs.append(record.getMessage())
            handler = LogCap()
            _ast_logger.addHandler(handler)

            try:
                with patch("os.lstat", side_effect=fake_lstat):
                    res = persist_qc_report(sub_dir, self.job_id, self.report_data)
                    self.assertEqual(res, "failed")
                    self.assertTrue(any("symlink or reparse point" in m for m in captured_logs))
            finally:
                _ast_logger.removeHandler(handler)

    def test_target_symlink_indicator_rejected_reaches_target_branch(self):
        # Save real lstat before patching to avoid recursive fake calls on ancestor components
        real_lstat = os.lstat
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / f"{self.job_id}.qc.json"
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("protected", encoding="utf-8")

            def fake_lstat(path, *args, **kwargs):
                p = Path(path)
                if p.name == target.name:
                    mock_stat = MagicMock()
                    mock_stat.st_mode = stat.S_IFLNK | 0o777
                    mock_stat.st_file_attributes = 0
                    return mock_stat
                return real_lstat(path, *args, **kwargs)

            mkstemp_called = []
            def fake_mkstemp(*args, **kwargs):
                mkstemp_called.append(True)
                raise AssertionError("mkstemp must not be reached when target is a link")

            captured_logs = []
            class LogCap(logging.Handler):
                def emit(self, record): captured_logs.append(record.getMessage())
            handler = LogCap()
            _ast_logger.addHandler(handler)

            try:
                with patch("os.lstat", side_effect=fake_lstat), \
                     patch("tempfile.mkstemp", side_effect=fake_mkstemp):
                    res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                    self.assertEqual(res, "failed")
                    self.assertEqual(mkstemp_called, [])
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "protected")
                    # Assert the target branch specifically was reached
                    self.assertTrue(any("QC target file is a symlink" in m for m in captured_logs))
            finally:
                _ast_logger.removeHandler(handler)

    def test_target_reparse_point_rejected_before_mkstemp(self):
        real_lstat = os.lstat
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / f"{self.job_id}.qc.json"
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("protected", encoding="utf-8")

            def fake_lstat(path, *args, **kwargs):
                p = Path(path)
                if p.name == target.name:
                    mock_stat = MagicMock()
                    mock_stat.st_mode = stat.S_IFREG | 0o644
                    mock_stat.st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                    return mock_stat
                return real_lstat(path, *args, **kwargs)

            mkstemp_called = []
            def fake_mkstemp(*args, **kwargs):
                mkstemp_called.append(True)
                raise AssertionError("mkstemp must not be reached when target has reparse attribute")

            captured_logs = []
            class LogCap(logging.Handler):
                def emit(self, record): captured_logs.append(record.getMessage())
            handler = LogCap()
            _ast_logger.addHandler(handler)

            try:
                with patch("os.lstat", side_effect=fake_lstat), \
                     patch("tempfile.mkstemp", side_effect=fake_mkstemp):
                    res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                    self.assertEqual(res, "failed")
                    self.assertEqual(mkstemp_called, [])
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "protected")
                    self.assertTrue(any("QC target file is a reparse point" in m for m in captured_logs))
            finally:
                _ast_logger.removeHandler(handler)

    def test_target_dangling_symlink_rejected_before_mkstemp(self):
        real_lstat = os.lstat
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / f"{self.job_id}.qc.json"

            def fake_lstat(path, *args, **kwargs):
                p = Path(path)
                if p.name == target.name:
                    mock_stat = MagicMock()
                    mock_stat.st_mode = stat.S_IFLNK | 0o777
                    mock_stat.st_file_attributes = 0
                    return mock_stat
                return real_lstat(path, *args, **kwargs)

            captured_logs = []
            class LogCap(logging.Handler):
                def emit(self, record): captured_logs.append(record.getMessage())
            handler = LogCap()
            _ast_logger.addHandler(handler)

            try:
                with patch("os.lstat", side_effect=fake_lstat):
                    res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                    self.assertEqual(res, "failed")
                    self.assertTrue(any("QC target file is a symlink" in m for m in captured_logs))
            finally:
                _ast_logger.removeHandler(handler)

    def test_target_existing_as_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_target = Path(tmpdir) / f"{self.job_id}.qc.json"
            bad_target.mkdir()

            res = persist_qc_report(tmpdir, self.job_id, self.report_data)
            self.assertEqual(res, "failed")
            self.assertTrue(bad_target.is_dir())

    def test_nonexistent_output_directory_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            absent_dir = Path(tmpdir) / "absent_child_directory"
            res = persist_qc_report(absent_dir, self.job_id, self.report_data)
            self.assertEqual(res, "failed")


class TestQCPersistenceWriter(unittest.TestCase):
    """Verifies atomic temp-to-replace sidecar serialization and idempotence."""

    def setUp(self):
        self.job_id = str(uuid.uuid4()).lower()
        self.created_at = "2026-11-05T07:30:00Z"
        self.report_data = {
            "qc_version": "1.0",
            "job_id": self.job_id,
            "filename": "video.mp4",
            "pipeline_type": "cinematic",
            "created_at": self.created_at,
            "population": "final_export_cues",
            "summary": {
                "total_final_cues": 1,
                "clean_cues": 1,
                "warned_cues": 0,
                "warning_occurrences": 0,
                "source_retention_flagged_cues": 0,
                "ass_source_retention_flagged_cues": 0,
                "any_retention_flagged_cues": 0,
                "rejected_blocks_count": 0,
                "review_status": "NO_RECORDED_ISSUES"
            },
            "issue_ledger": [],
            "rejected_blocks": []
        }

    def test_successful_sidecar_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = persist_qc_report(tmpdir, self.job_id, self.report_data)
            self.assertEqual(res, "written")

            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            self.assertTrue(dest.exists())
            self.assertTrue(dest.is_file())

            loaded = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(loaded["job_id"], self.job_id)
            self.assertEqual(loaded["summary"]["review_status"], "NO_RECORDED_ISSUES")

    def test_two_distinct_jobs_no_collision(self):
        job1 = str(uuid.uuid4()).lower()
        job2 = str(uuid.uuid4()).lower()

        data1 = dict(self.report_data, job_id=job1)
        data2 = dict(self.report_data, job_id=job2)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(persist_qc_report(tmpdir, job1, data1), "written")
            self.assertEqual(persist_qc_report(tmpdir, job2, data2), "written")

            dest1 = Path(tmpdir) / f"{job1}.qc.json"
            dest2 = Path(tmpdir) / f"{job2}.qc.json"
            self.assertTrue(dest1.exists())
            self.assertTrue(dest2.exists())
            self.assertNotEqual(dest1.name, dest2.name)

    def test_same_job_replacement_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(persist_qc_report(tmpdir, self.job_id, self.report_data), "written")
            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            self.assertTrue(dest.exists())

            updated_data = dict(self.report_data)
            updated_data["summary"] = dict(self.report_data["summary"], review_status="REVIEW_REQUIRED")
            self.assertEqual(persist_qc_report(tmpdir, self.job_id, updated_data), "written")

            loaded = json.loads(dest.read_text(encoding="utf-8"))
            self.assertEqual(loaded["summary"]["review_status"], "REVIEW_REQUIRED")

    def test_strict_json_serialization_rejects_nan(self):
        nan_data = dict(self.report_data, bad_num=float("nan"))
        with tempfile.TemporaryDirectory() as tmpdir:
            res = persist_qc_report(tmpdir, self.job_id, nan_data)
            self.assertEqual(res, "failed")
            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            self.assertFalse(dest.exists())

    def test_job_id_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            other_id = str(uuid.uuid4()).lower()
            res = persist_qc_report(tmpdir, self.job_id, dict(self.report_data, job_id=other_id))
            self.assertEqual(res, "failed")


class TestQCFaultInjection(unittest.TestCase):
    """Verifies descriptor ownership, stream lifecycle, and stage-by-stage error recovery."""

    def setUp(self):
        self.job_id = str(uuid.uuid4()).lower()
        self.report_data = {
            "qc_version": "1.0",
            "job_id": self.job_id,
            "filename": "f.srt",
            "pipeline_type": "subtitle_translation",
            "created_at": "2026-11-05T07:30:00Z",
            "population": "final_export_cues",
            "summary": {"review_status": "NO_RECORDED_ISSUES"},
            "issue_ledger": [],
            "rejected_blocks": []
        }

    def test_mkstemp_failure_does_not_unlink(self):
        unlink_calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            dest.write_text("prior_complete_report_bytes", encoding="utf-8")
            sub = Path(tmpdir) / f"{self.job_id}.srt"
            sub.write_text("prior_complete_subtitles", encoding="utf-8")
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("sentinel_data", encoding="utf-8")

            with patch("tempfile.mkstemp", side_effect=OSError("mkstemp failure")), \
                 patch("os.unlink", side_effect=lambda p: unlink_calls.append(p)):
                res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                self.assertEqual(res, "failed")
                self.assertEqual(unlink_calls, [])
                self.assertEqual(dest.read_text(encoding="utf-8"), "prior_complete_report_bytes")
                self.assertEqual(sub.read_text(encoding="utf-8"), "prior_complete_subtitles")
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel_data")

    def test_open_failure_closes_raw_descriptor(self):
        raw_closed = []
        orig_close = os.close
        def spy_close(fd):
            raw_closed.append(fd)
            orig_close(fd)

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            dest.write_text("prior_complete_report_bytes", encoding="utf-8")
            sub = Path(tmpdir) / f"{self.job_id}.srt"
            sub.write_text("prior_complete_subtitles", encoding="utf-8")
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("sentinel_data", encoding="utf-8")

            with patch("builtins.open", side_effect=OSError("open failure")), \
                 patch("os.close", side_effect=spy_close):
                res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                self.assertEqual(res, "failed")
                # Raw fd must have been closed because ownership was not yet transferred
                self.assertEqual(len(raw_closed), 1)
                self.assertEqual(dest.read_text(encoding="utf-8"), "prior_complete_report_bytes")
                self.assertEqual(sub.read_text(encoding="utf-8"), "prior_complete_subtitles")
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel_data")
                temp_files = list(Path(tmpdir).glob(f".{self.job_id}.qc.*.tmp"))
                self.assertEqual(temp_files, [])

    def test_stream_write_failure_transfers_ownership_and_prevents_raw_os_close(self):
        raw_os_close_calls = []
        orig_os_close = os.close
        def spy_os_close(fd):
            raw_os_close_calls.append(fd)
            orig_os_close(fd)

        orig_open = open
        def fake_open(*args, **kwargs):
            stream = orig_open(*args, **kwargs)
            def failing_write(data):
                raise OSError("stream write error")
            stream.write = failing_write
            return stream

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            dest.write_text("prior_complete_report_bytes", encoding="utf-8")
            sub = Path(tmpdir) / f"{self.job_id}.srt"
            sub.write_text("prior_complete_subtitles", encoding="utf-8")
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("sentinel_data", encoding="utf-8")

            with patch("builtins.open", side_effect=fake_open), \
                 patch("os.close", side_effect=spy_os_close):
                res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                self.assertEqual(res, "failed")
                self.assertEqual(raw_os_close_calls, [])
                self.assertEqual(dest.read_text(encoding="utf-8"), "prior_complete_report_bytes")
                self.assertEqual(sub.read_text(encoding="utf-8"), "prior_complete_subtitles")
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel_data")
                temp_files = list(Path(tmpdir).glob(f".{self.job_id}.qc.*.tmp"))
                self.assertEqual(temp_files, [])

    def test_flush_failure_transfers_ownership(self):
        raw_os_close_calls = []
        orig_os_close = os.close
        def spy_os_close(fd):
            raw_os_close_calls.append(fd)
            orig_os_close(fd)

        orig_open = open
        def fake_open(*args, **kwargs):
            stream = orig_open(*args, **kwargs)
            def failing_flush():
                raise OSError("flush failure")
            stream.flush = failing_flush
            return stream

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            dest.write_text("prior_complete_report_bytes", encoding="utf-8")
            sub = Path(tmpdir) / f"{self.job_id}.srt"
            sub.write_text("prior_complete_subtitles", encoding="utf-8")
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("sentinel_data", encoding="utf-8")

            with patch("builtins.open", side_effect=fake_open), \
                 patch("os.close", side_effect=spy_os_close):
                res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                self.assertEqual(res, "failed")
                self.assertEqual(raw_os_close_calls, [])
                self.assertEqual(dest.read_text(encoding="utf-8"), "prior_complete_report_bytes")
                self.assertEqual(sub.read_text(encoding="utf-8"), "prior_complete_subtitles")
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel_data")
                temp_files = list(Path(tmpdir).glob(f".{self.job_id}.qc.*.tmp"))
                self.assertEqual(temp_files, [])

    def test_fsync_failure_transfers_ownership_and_prevents_raw_os_close(self):
        raw_os_close_calls = []
        orig_os_close = os.close
        def spy_os_close(fd):
            raw_os_close_calls.append(fd)
            orig_os_close(fd)

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            dest.write_text("prior_complete_report_bytes", encoding="utf-8")
            sub = Path(tmpdir) / f"{self.job_id}.srt"
            sub.write_text("prior_complete_subtitles", encoding="utf-8")
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("sentinel_data", encoding="utf-8")

            with patch("os.fsync", side_effect=OSError("Injected fsync error")), \
                 patch("os.close", side_effect=spy_os_close):
                res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                self.assertEqual(res, "failed")
                # Raw os.close must NOT be called on the transferred descriptor
                self.assertEqual(raw_os_close_calls, [])
                self.assertEqual(dest.read_text(encoding="utf-8"), "prior_complete_report_bytes")
                self.assertEqual(sub.read_text(encoding="utf-8"), "prior_complete_subtitles")
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel_data")
                temp_files = list(Path(tmpdir).glob(f".{self.job_id}.qc.*.tmp"))
                self.assertEqual(temp_files, [])

    def test_replace_failure_unlinks_temp_and_preserves_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / f"{self.job_id}.qc.json"
            dest.write_text("prior_complete_report_bytes", encoding="utf-8")
            sub = Path(tmpdir) / f"{self.job_id}.srt"
            sub.write_text("prior_complete_subtitles", encoding="utf-8")
            sentinel = Path(tmpdir) / "sentinel.txt"
            sentinel.write_text("sentinel_data", encoding="utf-8")

            with patch("os.replace", side_effect=OSError("Injected replace error")):
                res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                self.assertEqual(res, "failed")

            self.assertEqual(dest.read_text(encoding="utf-8"), "prior_complete_report_bytes")
            self.assertEqual(sub.read_text(encoding="utf-8"), "prior_complete_subtitles")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel_data")
            temp_files = list(Path(tmpdir).glob(f".{self.job_id}.qc.*.tmp"))
            self.assertEqual(temp_files, [])

    def test_stream_close_failure_logs_safe_category_and_does_not_raise(self):
        log_records = []
        class LogCapturer(logging.Handler):
            def emit(self, record): log_records.append(record.getMessage())
        handler = LogCapturer()
        _ast_logger.addHandler(handler)

        orig_open = open
        def fake_open(*args, **kwargs):
            stream = orig_open(*args, **kwargs)
            orig_close = stream.close
            def failing_close():
                orig_close()
                raise OSError("SECRET stream close error")
            stream.close = failing_close
            return stream

        raw_close_calls = []
        orig_os_close = os.close
        def spy_os_close(fd):
            raw_close_calls.append(fd)
            return orig_os_close(fd)

        replace_calls = []
        orig_replace = os.replace
        def spy_replace(src, dst):
            replace_calls.append((src, dst))
            return orig_replace(src, dst)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / f"{self.job_id}.qc.json"
                dest.write_text("prior_complete_report_bytes", encoding="utf-8")
                sub = Path(tmpdir) / f"{self.job_id}.srt"
                sub.write_text("prior_complete_subtitles", encoding="utf-8")
                sentinel = Path(tmpdir) / "sentinel.txt"
                sentinel.write_text("sentinel_data", encoding="utf-8")

                with patch("builtins.open", side_effect=fake_open), \
                     patch("os.close", side_effect=spy_os_close), \
                     patch("os.replace", side_effect=spy_replace):
                    res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                    self.assertEqual(res, "failed")
                    self.assertTrue(any("stream cleanup failed" in msg for msg in log_records))
                    # Safe category log must not leak injected text or raw details
                    self.assertFalse(any("SECRET" in msg for msg in log_records))
                    # Replace must NOT be called on failure
                    self.assertEqual(replace_calls, [])
                    # Stream took ownership, raw_fd was set to None, so no raw os.close call after transfer
                    self.assertEqual(raw_close_calls, [])

                # Prior complete bytes survive completely untouched
                self.assertEqual(dest.read_text(encoding="utf-8"), "prior_complete_report_bytes")
                self.assertEqual(sub.read_text(encoding="utf-8"), "prior_complete_subtitles")
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "sentinel_data")
                temp_files = list(Path(tmpdir).glob(f".{self.job_id}.qc.*.tmp"))
                self.assertEqual(temp_files, [])
        finally:
            _ast_logger.removeHandler(handler)

    def test_raw_descriptor_cleanup_failure_category(self):
        # Fully synthetic fd/stream API: verifies safe category logging without sensitive leak and unlinks only owned temp
        log_records = []
        class LogCapturer(logging.Handler):
            def emit(self, record): log_records.append(record.getMessage())
        handler = LogCapturer()
        _ast_logger.addHandler(handler)

        events = []
        fake_os = SimpleNamespace(
            close=lambda fd: (_ for _ in ()).throw(OSError("SECRET injected raw descriptor detail")),
            path=SimpleNamespace(exists=lambda p: True),
            unlink=lambda p: events.append(("unlink", p))
        )
        try:
            with patch.dict(_pipeline_ns, {
                "logger": _ast_logger,
                "os": fake_os,
                "tempfile": SimpleNamespace(mkstemp=lambda **kw: (777, "owned_synthetic.tmp")),
                "_validate_qc_output_path": lambda *a: Path("X:/synthetic/report.qc.json"),
                "open": lambda *a, **k: (_ for _ in ()).throw(OSError("SECRET opening error"))
            }):
                res = persist_qc_report("synthetic", self.job_id, self.report_data)
            self.assertEqual(res, "failed")
            self.assertTrue(any("descriptor cleanup failed" in msg for msg in log_records))
            self.assertFalse(any("SECRET" in msg for msg in log_records))
            self.assertEqual(events, [("unlink", "owned_synthetic.tmp")])
        finally:
            _ast_logger.removeHandler(handler)

    def test_cleanup_unlink_failure_logs_safe_category_and_does_not_raise(self):
        log_records = []
        class LogCapturer(logging.Handler):
            def emit(self, record):
                log_records.append(record.getMessage())

        handler = LogCapturer()
        _ast_logger.addHandler(handler)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with patch("os.replace", side_effect=OSError("Replace error")), \
                     patch("os.unlink", side_effect=OSError("Unlink error")):
                    res = persist_qc_report(tmpdir, self.job_id, self.report_data)
                    self.assertEqual(res, "failed")
                    self.assertTrue(any("temporary file cleanup failed" in msg for msg in log_records))
        finally:
            _ast_logger.removeHandler(handler)

    def test_success_lifecycle_ordering(self):
        events = []
        real_replace = os.replace
        real_fsync = os.fsync
        orig_open = open

        def spy_replace(src, dst):
            events.append("replace")
            return real_replace(src, dst)

        def spy_fsync(fd):
            events.append("fsync")
            return real_fsync(fd)

        def fake_open(*args, **kwargs):
            stream = orig_open(*args, **kwargs)
            orig_write = stream.write
            orig_flush = stream.flush
            orig_close = stream.close
            def spy_write(data):
                events.append("write")
                return orig_write(data)
            def spy_flush():
                events.append("flush")
                return orig_flush()
            def spy_close():
                events.append("close")
                stream.flush = orig_flush
                return orig_close()
            stream.write = spy_write
            stream.flush = spy_flush
            stream.close = spy_close
            return stream

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("builtins.open", side_effect=fake_open), \
             patch("os.fsync", side_effect=spy_fsync), \
             patch("os.replace", side_effect=spy_replace):
            res = persist_qc_report(tmpdir, self.job_id, self.report_data)
            self.assertEqual(res, "written")
            # Complete write -> flush -> fsync -> close -> replace order
            self.assertEqual(events, ["write", "flush", "fsync", "close", "replace"])
            # Assert written bytes are non-empty and valid json (detects in-memory empty-write mutant)
            qc_file = Path(tmpdir) / f"{self.job_id}.qc.json"
            content = qc_file.read_text(encoding="utf-8")
            self.assertGreater(len(content), 0)
            parsed = json.loads(content)
            self.assertEqual(parsed["job_id"], self.job_id)


class TestQCPipelineIntegrationMatrix(unittest.TestCase):
    """Verifies complete integration matrix across both coroutines under fakes."""

    def setUp(self):
        self.job_id = str(uuid.uuid4()).lower()
        self.jm = JobManager()

    def test_cinematic_pipeline_clean_written(self):
        video_path = Path("video.mp4")
        mock_cues = [SimpleNamespace(index=1, start=0.0, end=2.0, text="hello", warnings=())]

        with tempfile.TemporaryDirectory() as out_dir:
            fake_settings = SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir))
            fake_transcriber = SimpleNamespace(
                transcribe=lambda *a, **k: asyncio.sleep(0, result=(mock_cues, "en", 2.0))
            )
            fake_translator = SimpleNamespace(
                translate=lambda *a, **k: asyncio.sleep(0, result=mock_cues)
            )

            with patch.dict(_pipeline_ns, {
                "settings": fake_settings,
                "job_manager": self.jm,
                "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                "transcriber": fake_transcriber,
                "translator": fake_translator
            }):
                self.jm.create_job(self.job_id, video_path.name)
                asyncio.run(run_cinematic_pipeline(self.job_id, video_path))

                job = self.jm.jobs[self.job_id]
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["qc_report_status"], "written")
                self.assertEqual(job["qc_report_file"], f"{self.job_id}.qc.json")
                self.assertNotIn("qc_report_path", job)
                self.assertTrue((Path(out_dir) / f"{self.job_id}.qc.json").exists())

    def test_cinematic_pipeline_warned_written(self):
        video_path = Path("video.mp4")
        mock_cues = [SimpleNamespace(index=1, start=0.0, end=2.0, text="hello", warnings=("LAYOUT_OVERFLOW_WARNING",))]

        with tempfile.TemporaryDirectory() as out_dir:
            fake_settings = SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir))
            fake_transcriber = SimpleNamespace(
                transcribe=lambda *a, **k: asyncio.sleep(0, result=(mock_cues, "en", 2.0))
            )
            fake_translator = SimpleNamespace(
                translate=lambda *a, **k: asyncio.sleep(0, result=mock_cues)
            )

            with patch.dict(_pipeline_ns, {
                "settings": fake_settings,
                "job_manager": self.jm,
                "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                "transcriber": fake_transcriber,
                "translator": fake_translator
            }):
                self.jm.create_job(self.job_id, video_path.name)
                asyncio.run(run_cinematic_pipeline(self.job_id, video_path))

                job = self.jm.jobs[self.job_id]
                self.assertEqual(job["status"], "completed_with_warnings")
                self.assertEqual(job["qc_report_status"], "written")
                self.assertEqual(job["qc_report_file"], f"{self.job_id}.qc.json")

    def test_cinematic_pipeline_qc_writer_returned_failed_decoupled(self):
        for warned in (False, True):
            with self.subTest(warned=warned), tempfile.TemporaryDirectory() as out_dir:
                video_path = Path(out_dir) / "video.mp4"
                cues = [
                    SimpleNamespace(index=1, start=1.25, end=3.75, text="translated",
                                    clean_text="clean", warnings=("REVIEW",) if warned else ())
                ]
                # Baseline run with successful QC to capture subtitle bytes
                baseline_bytes = {}
                base_job = str(uuid.uuid4()).lower()
                self.jm.create_job(base_job, video_path.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                    "transcriber": SimpleNamespace(transcribe=lambda *a, **k: asyncio.sleep(0, result=(cues, "en", 4.0))),
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": persist_qc_report
                }):
                    asyncio.run(run_cinematic_pipeline(base_job, video_path))
                    for ext in ("srt", "ass", "vtt"):
                        baseline_bytes[ext] = (Path(out_dir) / f"{base_job}.{ext}").read_bytes()

                hits = []
                def failing_writer(*a, **k):
                    hits.append("writer")
                    return "failed"

                test_job = str(uuid.uuid4()).lower()
                self.jm.create_job(test_job, video_path.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                    "transcriber": SimpleNamespace(transcribe=lambda *a, **k: asyncio.sleep(0, result=(cues, "en", 4.0))),
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": failing_writer
                }):
                    asyncio.run(run_cinematic_pipeline(test_job, video_path))

                    job = self.jm.jobs[test_job]
                    self.assertEqual(job["status"], "completed_with_warnings" if warned else "completed")
                    self.assertEqual(job["progress"], 100)
                    self.assertEqual(job["segment_count"], 1)
                    self.assertEqual(job["warnings_count"], 1 if warned else 0)
                    self.assertEqual(job["qc_report_status"], "failed")
                    self.assertIsNone(job["qc_report_file"])
                    self.assertEqual(hits, ["writer"])

                    # Subtitle files must match byte-for-byte with the baseline run
                    for ext in ("srt", "ass", "vtt"):
                        out_path = Path(out_dir) / f"{test_job}.{ext}"
                        self.assertTrue(out_path.exists())
                        self.assertEqual(out_path.read_bytes(), baseline_bytes[ext])

                    # Verify exact cue timing via stdlib parse_srt_content
                    srt_content = (Path(out_dir) / f"{test_job}.srt").read_text(encoding="utf-8-sig")
                    parsed = _subtitles_ns["parse_srt_content"](srt_content)
                    self.assertEqual([(c.start, c.end, c.text) for c in parsed], [(1.25, 3.75, "translated")])

    def test_cinematic_pipeline_qc_builder_exception_decoupled(self):
        for warned in (False, True):
            with self.subTest(warned=warned), tempfile.TemporaryDirectory() as out_dir:
                video_path = Path(out_dir) / "video.mp4"
                cues = [
                    SimpleNamespace(index=1, start=1.25, end=3.75, text="translated",
                                    clean_text="clean", warnings=("REVIEW",) if warned else ())
                ]
                baseline_bytes = {}
                base_job = str(uuid.uuid4()).lower()
                self.jm.create_job(base_job, video_path.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                    "transcriber": SimpleNamespace(transcribe=lambda *a, **k: asyncio.sleep(0, result=(cues, "en", 4.0))),
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": persist_qc_report
                }):
                    asyncio.run(run_cinematic_pipeline(base_job, video_path))
                    for ext in ("srt", "ass", "vtt"):
                        baseline_bytes[ext] = (Path(out_dir) / f"{base_job}.{ext}").read_bytes()

                hits = []
                def failing_builder(*a, **k):
                    hits.append("builder")
                    raise ValueError("Injected QC builder error")

                test_job = str(uuid.uuid4()).lower()
                self.jm.create_job(test_job, video_path.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                    "transcriber": SimpleNamespace(transcribe=lambda *a, **k: asyncio.sleep(0, result=(cues, "en", 4.0))),
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": failing_builder,
                    "persist_qc_report": persist_qc_report
                }):
                    asyncio.run(run_cinematic_pipeline(test_job, video_path))

                    job = self.jm.jobs[test_job]
                    self.assertEqual(job["status"], "completed_with_warnings" if warned else "completed")
                    self.assertEqual(job["progress"], 100)
                    self.assertEqual(job["segment_count"], 1)
                    self.assertEqual(job["warnings_count"], 1 if warned else 0)
                    self.assertEqual(job["qc_report_status"], "failed")
                    self.assertIsNone(job["qc_report_file"])
                    self.assertEqual(hits, ["builder"])

                    for ext in ("srt", "ass", "vtt"):
                        out_path = Path(out_dir) / f"{test_job}.{ext}"
                        self.assertTrue(out_path.exists())
                        self.assertEqual(out_path.read_bytes(), baseline_bytes[ext])

                    srt_content = (Path(out_dir) / f"{test_job}.srt").read_text(encoding="utf-8-sig")
                    parsed = _subtitles_ns["parse_srt_content"](srt_content)
                    self.assertEqual([(c.start, c.end, c.text) for c in parsed], [(1.25, 3.75, "translated")])

    def test_cinematic_pipeline_qc_writer_exception_decoupled(self):
        for warned in (False, True):
            with self.subTest(warned=warned), tempfile.TemporaryDirectory() as out_dir:
                video_path = Path(out_dir) / "video.mp4"
                cues = [
                    SimpleNamespace(index=1, start=1.25, end=3.75, text="translated",
                                    clean_text="clean", warnings=("REVIEW",) if warned else ())
                ]
                baseline_bytes = {}
                base_job = str(uuid.uuid4()).lower()
                self.jm.create_job(base_job, video_path.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                    "transcriber": SimpleNamespace(transcribe=lambda *a, **k: asyncio.sleep(0, result=(cues, "en", 4.0))),
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": persist_qc_report
                }):
                    asyncio.run(run_cinematic_pipeline(base_job, video_path))
                    for ext in ("srt", "ass", "vtt"):
                        baseline_bytes[ext] = (Path(out_dir) / f"{base_job}.{ext}").read_bytes()

                hits = []
                def failing_writer(*a, **k):
                    hits.append("writer")
                    raise RuntimeError("Injected QC writer error")

                test_job = str(uuid.uuid4()).lower()
                self.jm.create_job(test_job, video_path.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                    "transcriber": SimpleNamespace(transcribe=lambda *a, **k: asyncio.sleep(0, result=(cues, "en", 4.0))),
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": failing_writer
                }):
                    asyncio.run(run_cinematic_pipeline(test_job, video_path))

                    job = self.jm.jobs[test_job]
                    self.assertEqual(job["status"], "completed_with_warnings" if warned else "completed")
                    self.assertEqual(job["progress"], 100)
                    self.assertEqual(job["segment_count"], 1)
                    self.assertEqual(job["warnings_count"], 1 if warned else 0)
                    self.assertEqual(job["qc_report_status"], "failed")
                    self.assertIsNone(job["qc_report_file"])
                    self.assertEqual(hits, ["writer"])

                    for ext in ("srt", "ass", "vtt"):
                        out_path = Path(out_dir) / f"{test_job}.{ext}"
                        self.assertTrue(out_path.exists())
                        self.assertEqual(out_path.read_bytes(), baseline_bytes[ext])

                    srt_content = (Path(out_dir) / f"{test_job}.srt").read_text(encoding="utf-8-sig")
                    parsed = _subtitles_ns["parse_srt_content"](srt_content)
                    self.assertEqual([(c.start, c.end, c.text) for c in parsed], [(1.25, 3.75, "translated")])

    def test_cinematic_pipeline_export_write_failure_before_qc_prevents_qc_report(self):
        video_path = Path("video.mp4")
        mock_cues = [SimpleNamespace(index=1, start=0.0, end=2.0, text="hello", warnings=())]

        with tempfile.TemporaryDirectory() as out_dir:
            fake_settings = SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir))
            fake_transcriber = SimpleNamespace(
                transcribe=lambda *a, **k: asyncio.sleep(0, result=(mock_cues, "en", 2.0))
            )
            fake_translator = SimpleNamespace(
                translate=lambda *a, **k: asyncio.sleep(0, result=mock_cues)
            )

            qc_builder_calls = []
            qc_writer_calls = []

            def spy_builder(*a, **k):
                qc_builder_calls.append(True)
                return build_qc_report(*a, **k)

            def spy_writer(*a, **k):
                qc_writer_calls.append(True)
                return persist_qc_report(*a, **k)

            def fail_publication(*a, **k):
                raise _pipeline_ns["PublicationError"]("Disk full")

            # Injected publication error at subtitle export time
            with patch.dict(_pipeline_ns, {
                "settings": fake_settings,
                "job_manager": self.jm,
                "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                "transcriber": fake_transcriber,
                "translator": fake_translator,
                "build_qc_report": spy_builder,
                "persist_qc_report": spy_writer,
                "publish_subtitle_triplet": fail_publication
            }):
                self.jm.create_job(self.job_id, video_path.name)
                asyncio.run(run_cinematic_pipeline(self.job_id, video_path))

                job = self.jm.jobs[self.job_id]
                self.assertEqual(job["status"], "failed")
                self.assertIn("Disk full", job["stage"])
                # QC builder and writer must NOT be called when export fails
                self.assertEqual(qc_builder_calls, [])
                self.assertEqual(qc_writer_calls, [])
                self.assertFalse((Path(out_dir) / f"{self.job_id}.qc.json").exists())

    def test_cinematic_pipeline_cancellation_at_qc_boundary_propagates(self):
        video_path = Path("video.mp4")
        mock_cues = [SimpleNamespace(index=1, start=0.0, end=2.0, text="hello", warnings=())]

        with tempfile.TemporaryDirectory() as out_dir:
            fake_settings = SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir))
            fake_transcriber = SimpleNamespace(
                transcribe=lambda *a, **k: asyncio.sleep(0, result=(mock_cues, "en", 2.0))
            )
            fake_translator = SimpleNamespace(
                translate=lambda *a, **k: asyncio.sleep(0, result=mock_cues)
            )

            def cancel_at_qc(*a, **k):
                raise asyncio.CancelledError("Cancelled at cinematic QC boundary")

            with patch.dict(_pipeline_ns, {
                "settings": fake_settings,
                "job_manager": self.jm,
                "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                "transcriber": fake_transcriber,
                "translator": fake_translator,
                "build_qc_report": cancel_at_qc
            }):
                self.jm.create_job(self.job_id, video_path.name)
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(run_cinematic_pipeline(self.job_id, video_path))

                job = self.jm.jobs[self.job_id]
                self.assertNotEqual(job["status"], "completed")
                self.assertFalse((Path(out_dir) / f"{self.job_id}.qc.json").exists())

    def test_subtitle_translation_pipeline_clean_written(self):
        with tempfile.TemporaryDirectory() as out_dir:
            sub_file = Path(out_dir) / "test.srt"
            sub_file.write_text("1\n00:00:01,000 --> 00:00:03,000\nhello\n", encoding="utf-8")
            mock_cues = [SimpleNamespace(index=1, start=1.0, end=3.0, text="hello", lines=["hello"], warnings=())]

            fake_settings = SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir))
            fake_translator = SimpleNamespace(
                translate=lambda *a, **k: asyncio.sleep(0, result=mock_cues)
            )

            with patch.dict(_pipeline_ns, {
                "settings": fake_settings,
                "job_manager": self.jm,
                "translator": fake_translator
            }):
                self.jm.create_job(self.job_id, sub_file.name)
                asyncio.run(run_subtitle_translation_pipeline(self.job_id, sub_file))

                job = self.jm.jobs[self.job_id]
                self.assertEqual(job["status"], "completed")
                self.assertEqual(job["qc_report_status"], "written")
                self.assertEqual(job["qc_report_file"], f"{self.job_id}.qc.json")
                self.assertTrue((Path(out_dir) / f"{self.job_id}.qc.json").exists())

    def test_subtitle_translation_pipeline_warned_written_with_real_ass_exporter(self):
        # Full end-to-end integration with actual stdlib subtitles exporter and parser via AST
        with tempfile.TemporaryDirectory() as out_dir:
            sub_file = Path(out_dir) / "test.ass"
            sub_file.write_text(
                "[Script Info]\nTitle: Test\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                "Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,{\\i1}Original English{\\i0}\n",
                encoding="utf-8"
            )

            # Translator translates incoming parsed cues to distinct Arabic text
            def fake_translate(segs, *a, **k):
                for s in segs:
                    s.text = "مرحبا بالعالم"
                return asyncio.sleep(0, result=segs)

            fake_settings = SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir))
            fake_translator = SimpleNamespace(translate=fake_translate)

            with patch.dict(_pipeline_ns, {
                "settings": fake_settings,
                "job_manager": self.jm,
                "translator": fake_translator
            }):
                self.jm.create_job(self.job_id, sub_file.name)
                asyncio.run(run_subtitle_translation_pipeline(self.job_id, sub_file))

                job = self.jm.jobs[self.job_id]
                self.assertEqual(job["status"], "completed_with_warnings")
                self.assertEqual(job["qc_report_status"], "written")

                # ASS file preserves original formatted source event
                ass_path = Path(out_dir) / f"{self.job_id}.ass"
                self.assertTrue(ass_path.exists())
                ass_content = ass_path.read_text(encoding="utf-8")
                self.assertIn("{\\i1}Original English{\\i0}", ass_content)

                # SRT file preserves distinct Arabic translation
                srt_path = Path(out_dir) / f"{self.job_id}.srt"
                self.assertTrue(srt_path.exists())
                srt_content = srt_path.read_text(encoding="utf-8")
                self.assertIn("مرحبا بالعالم", srt_content)

                # VTT file preserves distinct Arabic translation
                vtt_path = Path(out_dir) / f"{self.job_id}.vtt"
                self.assertTrue(vtt_path.exists())
                vtt_content = vtt_path.read_text(encoding="utf-8")
                self.assertIn("مرحبا بالعالم", vtt_content)

                # QC sidecar captured the late exporter retention warning
                qc_path = Path(out_dir) / f"{self.job_id}.qc.json"
                self.assertTrue(qc_path.exists())
                qc_data = json.loads(qc_path.read_text(encoding="utf-8"))
                self.assertEqual(qc_data["summary"]["ass_source_retention_flagged_cues"], 1)
                self.assertIn("ASS_SOURCE_RETENTION_REVIEW", qc_data["issue_ledger"][0]["diagnostic_flags"])

    def test_subtitle_translation_pipeline_qc_failure_decoupled(self):
        for warned in (False, True):
            with self.subTest(warned=warned), tempfile.TemporaryDirectory() as out_dir:
                sub_file = Path(out_dir) / "test.srt"
                sub_file.write_text("1\n00:00:01,250 --> 00:00:03,750\nsource\n", encoding="utf-8")
                cues = [
                    SimpleNamespace(index=1, start=1.25, end=3.75, text="translated",
                                    clean_text="clean", warnings=("REVIEW",) if warned else ())
                ]
                baseline_bytes = {}
                base_job = str(uuid.uuid4()).lower()
                self.jm.create_job(base_job, sub_file.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": persist_qc_report
                }):
                    asyncio.run(run_subtitle_translation_pipeline(base_job, sub_file))
                    for ext in ("srt", "ass", "vtt"):
                        baseline_bytes[ext] = (Path(out_dir) / f"{base_job}.{ext}").read_bytes()

                hits = []
                def failing_writer(*a, **k):
                    hits.append("writer")
                    return "failed"

                test_job = str(uuid.uuid4()).lower()
                self.jm.create_job(test_job, sub_file.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": failing_writer
                }):
                    asyncio.run(run_subtitle_translation_pipeline(test_job, sub_file))

                    job = self.jm.jobs[test_job]
                    self.assertEqual(job["status"], "completed_with_warnings" if warned else "completed")
                    self.assertEqual(job["progress"], 100)
                    self.assertEqual(job["segment_count"], 1)
                    self.assertEqual(job["warnings_count"], 1 if warned else 0)
                    self.assertEqual(job["qc_report_status"], "failed")
                    self.assertIsNone(job["qc_report_file"])
                    self.assertEqual(hits, ["writer"])

                    for ext in ("srt", "ass", "vtt"):
                        out_file = Path(out_dir) / f"{test_job}.{ext}"
                        self.assertTrue(out_file.exists())
                        self.assertEqual(out_file.read_bytes(), baseline_bytes[ext])

                    srt_content = (Path(out_dir) / f"{test_job}.srt").read_text(encoding="utf-8-sig")
                    parsed = _subtitles_ns["parse_srt_content"](srt_content)
                    self.assertEqual([(c.start, c.end, c.text) for c in parsed], [(1.25, 3.75, "translated")])

    def test_subtitle_translation_pipeline_qc_builder_exception_decoupled(self):
        for warned in (False, True):
            with self.subTest(warned=warned), tempfile.TemporaryDirectory() as out_dir:
                sub_file = Path(out_dir) / "test.srt"
                sub_file.write_text("1\n00:00:01,250 --> 00:00:03,750\nsource\n", encoding="utf-8")
                cues = [
                    SimpleNamespace(index=1, start=1.25, end=3.75, text="translated",
                                    clean_text="clean", warnings=("REVIEW",) if warned else ())
                ]
                baseline_bytes = {}
                base_job = str(uuid.uuid4()).lower()
                self.jm.create_job(base_job, sub_file.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": persist_qc_report
                }):
                    asyncio.run(run_subtitle_translation_pipeline(base_job, sub_file))
                    for ext in ("srt", "ass", "vtt"):
                        baseline_bytes[ext] = (Path(out_dir) / f"{base_job}.{ext}").read_bytes()

                hits = []
                def failing_builder(*a, **k):
                    hits.append("builder")
                    raise ValueError("Injected QC builder error in subtitle pipeline")

                test_job = str(uuid.uuid4()).lower()
                self.jm.create_job(test_job, sub_file.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": failing_builder,
                    "persist_qc_report": persist_qc_report
                }):
                    asyncio.run(run_subtitle_translation_pipeline(test_job, sub_file))

                    job = self.jm.jobs[test_job]
                    self.assertEqual(job["status"], "completed_with_warnings" if warned else "completed")
                    self.assertEqual(job["progress"], 100)
                    self.assertEqual(job["segment_count"], 1)
                    self.assertEqual(job["warnings_count"], 1 if warned else 0)
                    self.assertEqual(job["qc_report_status"], "failed")
                    self.assertIsNone(job["qc_report_file"])
                    self.assertEqual(hits, ["builder"])

                    for ext in ("srt", "ass", "vtt"):
                        out_file = Path(out_dir) / f"{test_job}.{ext}"
                        self.assertTrue(out_file.exists())
                        self.assertEqual(out_file.read_bytes(), baseline_bytes[ext])

                    srt_content = (Path(out_dir) / f"{test_job}.srt").read_text(encoding="utf-8-sig")
                    parsed = _subtitles_ns["parse_srt_content"](srt_content)
                    self.assertEqual([(c.start, c.end, c.text) for c in parsed], [(1.25, 3.75, "translated")])

    def test_subtitle_translation_pipeline_qc_writer_exception_decoupled(self):
        for warned in (False, True):
            with self.subTest(warned=warned), tempfile.TemporaryDirectory() as out_dir:
                sub_file = Path(out_dir) / "test.srt"
                sub_file.write_text("1\n00:00:01,250 --> 00:00:03,750\nsource\n", encoding="utf-8")
                cues = [
                    SimpleNamespace(index=1, start=1.25, end=3.75, text="translated",
                                    clean_text="clean", warnings=("REVIEW",) if warned else ())
                ]
                baseline_bytes = {}
                base_job = str(uuid.uuid4()).lower()
                self.jm.create_job(base_job, sub_file.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": persist_qc_report
                }):
                    asyncio.run(run_subtitle_translation_pipeline(base_job, sub_file))
                    for ext in ("srt", "ass", "vtt"):
                        baseline_bytes[ext] = (Path(out_dir) / f"{base_job}.{ext}").read_bytes()

                hits = []
                def failing_writer(*a, **k):
                    hits.append("writer")
                    raise RuntimeError("Injected QC writer error in subtitle pipeline")

                test_job = str(uuid.uuid4()).lower()
                self.jm.create_job(test_job, sub_file.name)
                with patch.dict(_pipeline_ns, {
                    "settings": SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir)),
                    "job_manager": self.jm,
                    "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                    "build_qc_report": build_qc_report,
                    "persist_qc_report": failing_writer
                }):
                    asyncio.run(run_subtitle_translation_pipeline(test_job, sub_file))

                    job = self.jm.jobs[test_job]
                    self.assertEqual(job["status"], "completed_with_warnings" if warned else "completed")
                    self.assertEqual(job["progress"], 100)
                    self.assertEqual(job["segment_count"], 1)
                    self.assertEqual(job["warnings_count"], 1 if warned else 0)
                    self.assertEqual(job["qc_report_status"], "failed")
                    self.assertIsNone(job["qc_report_file"])
                    self.assertEqual(hits, ["writer"])

                    for ext in ("srt", "ass", "vtt"):
                        out_file = Path(out_dir) / f"{test_job}.{ext}"
                        self.assertTrue(out_file.exists())
                        self.assertEqual(out_file.read_bytes(), baseline_bytes[ext])

                    srt_content = (Path(out_dir) / f"{test_job}.srt").read_text(encoding="utf-8-sig")
                    parsed = _subtitles_ns["parse_srt_content"](srt_content)
                    self.assertEqual([(c.start, c.end, c.text) for c in parsed], [(1.25, 3.75, "translated")])

    def test_subtitle_translation_pipeline_export_write_failure_before_qc_prevents_qc_report(self):
        with tempfile.TemporaryDirectory() as out_dir:
            sub_file = Path(out_dir) / "test.srt"
            sub_file.write_text("1\n00:00:01,000 --> 00:00:03,000\nhello\n", encoding="utf-8")
            mock_cues = [SimpleNamespace(index=1, start=1.0, end=3.0, text="hello", lines=["hello"], warnings=())]

            fake_settings = SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir))
            fake_translator = SimpleNamespace(
                translate=lambda *a, **k: asyncio.sleep(0, result=mock_cues)
            )

            qc_builder_calls = []
            qc_writer_calls = []

            def spy_builder(*a, **k):
                qc_builder_calls.append(True)
                return build_qc_report(*a, **k)

            def spy_writer(*a, **k):
                qc_writer_calls.append(True)
                return persist_qc_report(*a, **k)

            def fail_publication_sub(*a, **k):
                raise _pipeline_ns["PublicationError"]("Disk full during subtitle export")

            with patch.dict(_pipeline_ns, {
                "settings": fake_settings,
                "job_manager": self.jm,
                "translator": fake_translator,
                "build_qc_report": spy_builder,
                "persist_qc_report": spy_writer,
                "publish_subtitle_triplet": fail_publication_sub
            }):
                self.jm.create_job(self.job_id, sub_file.name)
                asyncio.run(run_subtitle_translation_pipeline(self.job_id, sub_file))

                job = self.jm.jobs[self.job_id]
                self.assertEqual(job["status"], "failed")
                self.assertIn("Disk full", job["stage"])
                self.assertEqual(qc_builder_calls, [])
                self.assertEqual(qc_writer_calls, [])
                self.assertFalse((Path(out_dir) / f"{self.job_id}.qc.json").exists())

    def test_subtitle_translation_pipeline_cancellation_at_qc_boundary_propagates(self):
        with tempfile.TemporaryDirectory() as out_dir:
            sub_file = Path(out_dir) / "test.srt"
            sub_file.write_text("1\n00:00:01,000 --> 00:00:03,000\nhello\n", encoding="utf-8")
            mock_cues = [SimpleNamespace(index=1, start=1.0, end=3.0, text="hello", lines=["hello"], warnings=())]

            fake_settings = SimpleNamespace(OUTPUT_DIR=Path(out_dir), TEMP_DIR=Path(out_dir))
            fake_translator = SimpleNamespace(
                translate=lambda *a, **k: asyncio.sleep(0, result=mock_cues)
            )

            def cancel_at_qc(*a, **k):
                raise asyncio.CancelledError("Cancelled at subtitle translation QC boundary")

            with patch.dict(_pipeline_ns, {
                "settings": fake_settings,
                "job_manager": self.jm,
                "translator": fake_translator,
                "build_qc_report": cancel_at_qc
            }):
                self.jm.create_job(self.job_id, sub_file.name)
                with self.assertRaises(asyncio.CancelledError):
                    asyncio.run(run_subtitle_translation_pipeline(self.job_id, sub_file))

                job = self.jm.jobs[self.job_id]
                self.assertNotEqual(job["status"], "completed")
                self.assertFalse((Path(out_dir) / f"{self.job_id}.qc.json").exists())

    def test_worker_cancellation_propagates(self):
        # Explicit worker queue cancellation propagation test under honest name
        async def cancel_coro():
            await asyncio.sleep(0)
            raise asyncio.CancelledError("Worker cancelled")

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run_exclusive_worker(cancel_coro()))

    def test_clean_warned_actual_coroutine_matrix_outputs_and_times(self):
        # Comprehensive 20-case parameterized actual coroutine matrix covering all combinations:
        # (cinematic, subtitle_translation) x (clean, warned) x (written, returned_failed, builder_exception, writer_exception, cancel)
        for kind in ("cinematic", "subtitle_translation"):
            for warned in (False, True):
                baseline_bytes = None
                for stage in ("written", "returned_failed", "builder_exception", "writer_exception", "cancel"):
                    with self.subTest(kind=kind, warned=warned, stage=stage), tempfile.TemporaryDirectory() as temp:
                        folder = Path(temp)
                        path = folder / ("input.mp4" if kind == "cinematic" else "input.srt")
                        if kind != "cinematic":
                            path.write_text("1\n00:00:01,250 --> 00:00:03,750\nsource\n", encoding="utf-8")
                        cues = [
                            SimpleNamespace(index=1, start=1.25, end=3.75, text="translated",
                                            clean_text="clean", warnings=("REVIEW",) if warned else ())
                        ]
                        case_job = str(uuid.uuid4()).lower()
                        manager = JobManager()
                        manager.create_job(case_job, path.name)
                        hits = []
                        real_builder = build_qc_report
                        real_writer = persist_qc_report

                        def builder(*a, **kw):
                            hits.append("builder")
                            if stage == "builder_exception":
                                raise ValueError("injected builder")
                            return real_builder(*a, **kw)

                        def writer(*a, **kw):
                            hits.append("writer")
                            if stage == "cancel":
                                raise asyncio.CancelledError()
                            if stage == "writer_exception":
                                raise OSError("injected writer")
                            if stage == "returned_failed":
                                return "failed"
                            return real_writer(*a, **kw)

                        with patch.dict(_pipeline_ns, {
                            "settings": SimpleNamespace(OUTPUT_DIR=folder, TEMP_DIR=folder),
                            "gpu_lock": asyncio.Lock(),
                            "job_manager": manager,
                            "extract_and_enhance_audio": lambda *a, **k: asyncio.sleep(0),
                            "transcriber": SimpleNamespace(transcribe=lambda *a, **k: asyncio.sleep(0, result=(cues, "en", 4.0))),
                            "translator": SimpleNamespace(translate=lambda *a, **k: asyncio.sleep(0, result=cues)),
                            "build_qc_report": builder,
                            "persist_qc_report": writer
                        }):
                            coro = run_cinematic_pipeline if kind == "cinematic" else run_subtitle_translation_pipeline
                            if stage == "cancel":
                                with self.assertRaises(asyncio.CancelledError):
                                    asyncio.run(coro(case_job, path))
                                self.assertNotIn(manager.jobs[case_job]["status"], ("completed", "completed_with_warnings"))
                            else:
                                asyncio.run(coro(case_job, path))
                                job = manager.jobs[case_job]
                                self.assertEqual(job["status"], "completed_with_warnings" if warned else "completed")
                                self.assertEqual(job["progress"], 100)
                                self.assertEqual(job["segment_count"], 1)
                                self.assertEqual(job["warnings_count"], int(warned))
                                self.assertEqual(job["qc_report_status"], "written" if stage == "written" else "failed")
                                self.assertEqual(job["qc_report_file"], f"{case_job}.qc.json" if stage == "written" else None)

                        if stage == "cancel":
                            continue
                        self.assertEqual(hits, ["builder"] if stage == "builder_exception" else ["builder", "writer"])
                        outputs = {s: (folder / f"{case_job}.{s}").read_bytes() for s in ("srt", "ass", "vtt")}
                        if baseline_bytes is None:
                            baseline_bytes = outputs
                        self.assertEqual(outputs, baseline_bytes)
                        parsed = _subtitles_ns["parse_srt_content"](outputs["srt"].decode("utf-8-sig"))
                        self.assertEqual([(c.start, c.end, c.text) for c in parsed], [(1.25, 3.75, "translated")])
                        self.assertEqual([(c.start, c.end) for c in cues], [(1.25, 3.75)])


if __name__ == "__main__":
    unittest.main()
