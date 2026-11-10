"""Actual installed publication AST with deterministic memory I/O and shared rejecting oracles.

No temporary directories, media, live application/config, external services or real
descriptors are used. The only host file read is the installed Python source.
"""
import ast
import asyncio
import builtins
from contextlib import ExitStack
import copy
import io
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import threading
from types import SimpleNamespace
from typing import Any, Dict, Optional
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "services" / "pipeline.py"
JOB = "12345678-1234-4234-8234-123456789abc"
FORMATS = ("srt", "ass", "vtt")
ENCODINGS = {"srt": "utf-8-sig", "ass": "utf-8-sig", "vtt": "utf-8"}
PAYLOADS = {"srt": "SRT مرحبا\n", "ass": "ASS مرحبا\r\n", "vtt": "WEBVTT\n\nمرحبا\n"}


def denied(*args, **kwargs):
    raise AssertionError("ISOLATION_ORACLE: forbidden external operation")


def source_text(path=None):
    selected = PIPELINE_PATH if path is None else path
    if selected.resolve() != (ROOT / "services" / "pipeline.py").resolve():
        raise AssertionError("SOURCE_ANCHOR_ORACLE: installed source required")
    return selected.read_text(encoding="utf-8-sig")


class MemoryPath:
    owner = None

    def __init__(self, value):
        self.fs = type(self).owner
        text = str(value)
        self.value = str(PurePosixPath(text if text.startswith("/") else "/" + text))

    def __str__(self): return self.value
    def __fspath__(self): return self.value
    def __hash__(self): return hash(self.value)
    def __eq__(self, other): return isinstance(other, MemoryPath) and self.value == other.value
    def __truediv__(self, other): return type(self)(str(PurePosixPath(self.value) / str(other)))
    @property
    def parent(self): return type(self)(str(PurePosixPath(self.value).parent))
    @property
    def parents(self): return tuple(type(self)(str(p)) for p in PurePosixPath(self.value).parents)
    @property
    def name(self): return PurePosixPath(self.value).name
    @property
    def stem(self): return PurePosixPath(self.value).stem
    @property
    def suffix(self): return PurePosixPath(self.value).suffix
    def exists(self): return self.fs.follow(self, "exists")
    def is_dir(self): return self.fs.follow(self, "dir")
    def is_file(self): return self.fs.follow(self, "file")
    def is_symlink(self):
        self.fs.events.append(("symlink-check", self.value))
        return self.fs.nodes.get(self.value, {}).get("kind") == "link"
    def resolve(self): raise AssertionError("LEXICAL_INSPECTION_ORACLE: resolve not needed")
    def read_text(self, encoding="utf-8", **kwargs):
        self.fs.events.append(("source-read", self.value))
        return self.fs.nodes[self.value]["data"].decode(encoding)
    def unlink(self, **kwargs): self.fs.unlink(self)


class MemoryFS:
    def __init__(self, faults=None, preexisting=False):
        self.events, self.counts, self.fds = [], {}, {}
        self.faults = faults or {}
        self.short_at = self.size_at = None
        self.observe_replaces = False
        self.nodes = {p: {"kind": "dir", "data": b""} for p in ("/", "/out", "/tmp", "/input")}
        self.Path = type("BoundMemoryPath", (MemoryPath,), {"owner": self})
        self.old = {fmt: ("OLD-" + fmt).encode("ascii") for fmt in FORMATS}
        if preexisting:
            for fmt in FORMATS: self.nodes[self.dest(fmt)] = {"kind": "file", "data": self.old[fmt]}
        self.os = SimpleNamespace(path=SimpleNamespace(abspath=lambda p: str(self.Path(p))),
            lstat=self.lstat, fstat=self.fstat, fsync=self.fsync, replace=self.replace,
            close=self.raw_close, unlink=self.unlink)

    def dest(self, fmt): return "/out/" + JOB + "." + fmt
    def tick(self, operation):
        number = self.counts.get(operation, 0) + 1
        self.counts[operation] = number
        fault = self.faults.get((operation, number))
        if fault is not None: raise fault
        return number

    def follow(self, path, kind):
        self.events.append(("follow-" + kind, str(path)))
        self.tick("follow-" + kind)
        node = self.nodes.get(str(path))
        if node is None: return False
        if kind == "exists": return node["kind"] != "link"
        return node["kind"] == ("dir" if kind == "dir" else "file")

    def lstat(self, path):
        key = str(path)
        self.events.append(("lstat", key)); self.tick("lstat")
        if key not in self.nodes: raise FileNotFoundError(key)
        node = self.nodes[key]
        mode = {"dir": stat.S_IFDIR, "file": stat.S_IFREG, "link": stat.S_IFLNK, "other": stat.S_IFIFO}[node["kind"]]
        return SimpleNamespace(st_mode=mode, st_file_attributes=0x400 if node.get("reparse") else 0)

    def mkstemp(self, dir, prefix, suffix):
        self.events.append(("create", str(dir))); index = self.tick("create")
        fd = 100 + index
        path = str(self.Path(dir) / (prefix + str(index) + suffix))
        self.nodes[path] = {"kind": "file", "data": b""}
        self.fds[fd] = {"path": path, "closed": False, "close_attempts": 0, "fmt": FORMATS[(index - 1) % 3]}
        return fd, path

    def open(self, fd, mode, closefd=True):
        self.events.append(("wrap", fd)); self.tick("wrap")
        if mode != "wb" or closefd is not True: raise AssertionError("BINARY_STREAM_ORACLE")
        return MemoryStream(self, fd)

    def fstat(self, fd):
        self.events.append(("fstat", fd)); number = self.tick("fstat")
        if self.fds[fd]["closed"]: raise AssertionError("OPEN_DESCRIPTOR_ORACLE")
        size = len(self.nodes[self.fds[fd]["path"]]["data"])
        return SimpleNamespace(st_size=size + 1 if number == self.size_at else size)

    def fsync(self, fd): self.events.append(("fsync", fd)); self.tick("fsync")

    def close(self, fd, raw):
        operation = "raw-close" if raw else "stream-close"
        self.events.append((operation, fd))
        self.fds[fd]["close_attempts"] += 1
        if self.fds[fd]["close_attempts"] > 1: raise AssertionError("SINGLE_CLOSE_ORACLE")
        self.tick(operation)
        self.fds[fd]["closed"] = True

    def raw_close(self, fd): self.close(fd, True)

    def replace(self, source, dest):
        self.events.append(("replace", str(source), str(dest))); self.tick("replace")
        if self.observe_replaces: return
        if str(source) not in self.nodes: raise FileNotFoundError(str(source))
        self.nodes[str(dest)] = self.nodes.pop(str(source))

    def unlink(self, path, *args, **kwargs):
        self.events.append(("unlink", str(path)))
        self.nodes.pop(str(path), None)

    def canonical(self): return {f: self.nodes.get(self.dest(f), {}).get("data") for f in FORMATS}
    def staged(self): return {p: n["data"] for p, n in self.nodes.items() if p.endswith(".tmp")}


class MemoryStream:
    def __init__(self, fs, fd): self.fs, self.fd = fs, fd
    def write(self, payload):
        self.fs.events.append(("write", self.fd)); number = self.fs.tick("write")
        count = len(payload) - 1 if number == self.fs.short_at else len(payload)
        self.fs.nodes[self.fs.fds[self.fd]["path"]]["data"] = payload[:count]
        return count
    def flush(self): self.fs.events.append(("flush", self.fd)); self.fs.tick("flush")
    def fileno(self): return self.fd
    def close(self): self.fs.close(self.fd, False)


def subject(fs, custom_source=None):
    text = source_text() if custom_source is None else custom_source
    names = {"PublicationError", "_validate_qc_job_id", "_is_reparse_or_link",
        "_validate_subtitle_output_paths", "publish_subtitle_triplet", "JobManager",
        "run_exclusive_worker", "run_cinematic_pipeline", "run_subtitle_translation_pipeline"}
    nodes = [copy.deepcopy(n) for n in ast.parse(text).body if getattr(n, "name", None) in names]
    if {n.name for n in nodes} != names: raise AssertionError("ACTUAL_SYMBOL_EXTRACTION_ORACLE")
    ns = dict(__name__="isolated_installed_publication", Any=Any, Dict=Dict, Optional=Optional,
        Path=fs.Path, os=fs.os, stat=stat, uuid=uuid, tempfile=SimpleNamespace(mkstemp=fs.mkstemp),
        open=fs.open, asyncio=asyncio,
        __builtins__=dict(vars(builtins), __import__=denied, open=denied))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PIPELINE_PATH), "exec"), ns)
    return ns


def invoke(fs, ns, payloads=None):
    values = PAYLOADS if payloads is None else payloads
    return ns["publish_subtitle_triplet"](fs.Path("/out"), JOB, *(values[f] for f in FORMATS))


def mutated_source(kind):
    tree = ast.parse(source_text())
    fn = next(n for n in tree.body if getattr(n, "name", None) == "publish_subtitle_triplet")
    class Mutate(ast.NodeTransformer):
        def visit_Expr(self, n):
            if kind == "no-fsync" and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "os.fsync": return None
            return self.generic_visit(n)
        def visit_If(self, n):
            if kind == "no-short-count" and ast.unparse(n.test).startswith("written !="): n.test = ast.Constant(False)
            if kind == "no-size-check" and ast.unparse(n.test).startswith("actual_size !="): n.test = ast.Constant(False)
            if kind == "no-base-cleanup" and ast.unparse(n.test) == "not staged_ok": n.test = ast.Constant(False)
            return self.generic_visit(n)
        def visit_ExceptHandler(self, n):
            if kind == "mask-primary" and isinstance(n.name, str) and n.name == "sec_err": n.body = [ast.Raise()]
            return self.generic_visit(n)
    Mutate().visit(fn)
    if kind in {"early-promotion", "double-close", "unlink"}:
        class Inject(ast.NodeTransformer):
            def visit_Expr(self, n):
                value = ast.unparse(n.value)
                if kind == "double-close" and value == "to_close.close()": return [n, copy.deepcopy(n)]
                return self.generic_visit(n)
            def visit_Assign(self, n):
                target = ast.unparse(n.targets[0])
                if kind == "early-promotion" and target == "staged[fmt]": return [ast.parse("os.replace(tmp_path, dest_path)").body[0], n]
                if kind == "unlink" and target == "tmp_path": return [n, ast.parse("os.unlink(tmp_path)").body[0]]
                return self.generic_visit(n)
        Inject().visit(fn)
    if kind == "wrong-caller-data":
        caller = next(n for n in tree.body if getattr(n, "name", None) == "run_cinematic_pipeline")
        call = next(n for n in ast.walk(caller) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "publish_subtitle_triplet")
        next(k for k in call.keywords if k.arg == "srt_content").value = ast.Constant("WRONG")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class TestOutputPublicationPolicy(unittest.TestCase):
    def setUp(self):
        stack = ExitStack(); self.addCleanup(stack.close)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"app", "core", "services", "torch", "transformers", "whisper", "faster_whisper", "requests", "httpx", "socket", "groq", "google", "dotenv"}: denied()
            return original_import(name, *args, **kwargs)
        stack.enter_context(patch.object(builtins, "__import__", guarded_import))
        for obj, attr in ((os, "getenv"), (subprocess, "run"), (subprocess, "Popen"), (threading.Thread, "start"), (asyncio, "to_thread")):
            stack.enter_context(patch.object(obj, attr, denied))
        for obj in (builtins, io):
            original = obj.open
            def guarded_open(path, *a, _original=original, **kw):
                if isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).replace("\\", "/").split("/")[-1].lower().startswith(".env"): denied()
                return _original(path, *a, **kw)
            stack.enter_context(patch.object(obj, "open", guarded_open))

    def assert_closed_once(self, fs):
        self.assertTrue(all(d["close_attempts"] == 1 for d in fs.fds.values()), "SINGLE_CLOSE_ORACLE")

    def assert_no_promotion(self, fs):
        self.assertEqual([e for e in fs.events if e[0] == "replace"], [], "NO_PROMOTION_ON_STAGE_FAILURE_ORACLE")

    def assert_no_unlink(self, fs):
        self.assertEqual([e for e in fs.events if e[0] == "unlink"], [], "NO_UNLINK_ORACLE")

    def assert_success(self, payloads=None, custom_source=None):
        fs = MemoryFS(); ns = subject(fs, custom_source); values = PAYLOADS if payloads is None else payloads
        result = invoke(fs, ns, values)
        self.assertEqual(result, {f: fs.Path(fs.dest(f)) for f in FORMATS}, "DESTINATION_PATH_ORACLE")
        self.assertEqual(fs.canonical(), {f: values[f].encode(ENCODINGS[f]) for f in FORMATS}, "PAYLOAD_BYTE_ORACLE")
        self.assertEqual(fs.staged(), {}, "SUCCESS_CONSUMES_STAGE_ORACLE")
        self.assert_closed_once(fs); self.assert_no_unlink(fs)
        return fs

    def assert_order(self, custom_source=None):
        fs = MemoryFS(); fs.observe_replaces = True
        try: invoke(fs, subject(fs, custom_source))
        except Exception:
            if any(d["close_attempts"] > 1 for d in fs.fds.values()): self.fail("SINGLE_CLOSE_ORACLE")
            raise
        self.assertEqual(len([e for e in fs.events if e[0] == "fsync"]), 3, "MANDATORY_FSYNC_ORACLE")
        for fd in fs.fds:
            operations = [e[0] for e in fs.events if len(e) == 2 and e[1] == fd]
            self.assertEqual(operations, ["wrap", "write", "flush", "fsync", "fstat", "stream-close"], "STAGE_OPERATION_ORDER_ORACLE")
        closes = [i for i, e in enumerate(fs.events) if e[0] == "stream-close"]
        replacements = [i for i, e in enumerate(fs.events) if e[0] == "replace"]
        self.assertLess(max(closes), min(replacements), "STAGE_COMPLETION_BEFORE_PROMOTION_ORACLE")
        self.assertEqual([fs.events[i][2] for i in replacements], [fs.dest(f) for f in FORMATS], "PROMOTION_ORDER_ORACLE")

    def assert_short(self, custom_source=None, stage=1):
        fs = MemoryFS(); fs.short_at = stage; ns = subject(fs, custom_source)
        with self.assertRaisesRegex(ns["PublicationError"], "Short write", msg="SHORT_COUNT_ORACLE"): invoke(fs, ns)
        self.assert_no_promotion(fs); self.assert_closed_once(fs); self.assert_no_unlink(fs)

    def assert_size(self, custom_source=None, stage=1):
        fs = MemoryFS(); fs.size_at = stage; ns = subject(fs, custom_source)
        with self.assertRaisesRegex(ns["PublicationError"], "size mismatch", msg="DESCRIPTOR_SIZE_ORACLE"): invoke(fs, ns)
        self.assert_no_promotion(fs); self.assert_closed_once(fs)

    def assert_primary(self, custom_source=None, primary=None, secondary=None, phase="write"):
        primary = asyncio.CancelledError("primary write cancellation") if primary is None else primary
        close_op = "raw-close" if phase == "wrap" else "stream-close"
        faults = {(phase, 1): primary}
        if secondary is not None: faults[(close_op, 1)] = secondary
        fs = MemoryFS(faults); ns = subject(fs, custom_source)
        try: invoke(fs, ns)
        except BaseException as caught:
            if isinstance(primary, Exception): self.assertIs(caught.__cause__, primary, "PRIMARY_CAUSE_ORACLE")
            else: self.assertIs(caught, primary, "PRIMARY_IDENTITY_ORACLE")
        else: self.fail("PRIMARY_PROPAGATION_ORACLE")
        self.assertTrue(fs.fds, "FAULT_DESCRIPTOR_ORACLE")
        self.assert_closed_once(fs); self.assert_no_promotion(fs); self.assert_no_unlink(fs)
        self.assertEqual(fs.fds[101]["closed"], secondary is None, "NO_FALSE_CLOSE_SUCCESS_ORACLE")

    def assert_replacement(self, step, existing):
        failure = OSError("replacement fault " + str(step))
        fs = MemoryFS({("replace", step): failure}, preexisting=existing); ns = subject(fs)
        with self.assertRaises(ns["PublicationError"]) as caught: invoke(fs, ns)
        self.assertIs(caught.exception.__cause__, failure, "REPLACEMENT_CAUSE_ORACLE")
        expected = {f: PAYLOADS[f].encode(ENCODINGS[f]) if i < step - 1 else (fs.old[f] if existing else None) for i, f in enumerate(FORMATS)}
        self.assertEqual(fs.canonical(), expected, "PARTIAL_CANONICAL_STATE_ORACLE")
        self.assertEqual(len(fs.staged()), 4 - step, "RETAINED_STAGE_ORACLE")
        self.assert_closed_once(fs); self.assert_no_unlink(fs)

    def assert_caller(self, media, failure=False, custom_source=None):
        fs = MemoryFS(); ns = subject(fs, custom_source); calls, events = [], []
        clock = iter([100, 110, 115, 120, 127, 140]); ns["time"] = SimpleNamespace(time=lambda: next(clock))
        ns["logger"] = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None)
        ns["settings"] = SimpleNamespace(OUTPUT_DIR=fs.Path("/out"), TEMP_DIR=fs.Path("/tmp"))
        ns["gpu_lock"] = asyncio.Lock(); manager = ns["JobManager"](); manager.create_job(JOB, "movie.mp4" if media else "input.srt"); ns["job_manager"] = manager
        original_update = manager.update_job
        async def update(*a, **kw):
            events.append(("job", kw.get("status", a[1] if len(a) > 1 else None)))
            return await original_update(*a, **kw)
        manager.update_job = update
        cue = SimpleNamespace(index=1, start=1.0, end=2.0, text="مرحبا", lines=["مرحبا"], warnings=())
        async def extract(*a, **kw): fs.nodes["/tmp/" + JOB + "_16k.wav"] = {"kind": "file", "data": b"memory-audio"}
        async def transcribe(*a, **kw): return [cue], "en", 3.5
        async def translate(*a, **kw): return [cue]
        def publish(*a, **kw):
            calls.append((a, kw)); events.append(("publish",))
            if failure: raise ns["PublicationError"]("synthetic publication failure")
            return {f: fs.Path(fs.dest(f)) for f in FORMATS}
        def qc_build(**kw): events.append(("qc-builder",)); return {"job_id": JOB}
        def qc_write(*a): events.append(("qc-writer",)); return "written"
        ns.update(extract_and_enhance_audio=extract, transcriber=SimpleNamespace(transcribe=transcribe),
            translator=SimpleNamespace(translate=translate), sanitize_timing_without_drift=lambda cues: cues,
            parse_srt_content=lambda text: [cue], validate_imported_cues=lambda cues: cues,
            wrap_arabic_text=lambda text, **kw: [text], make_subtitle_document=lambda **kw: kw["segments"],
            build_srt_content=lambda *a, **kw: PAYLOADS["srt"], build_ass_content=lambda *a, **kw: PAYLOADS["ass"], build_vtt_content=lambda *a, **kw: PAYLOADS["vtt"],
            publish_subtitle_triplet=publish, build_qc_report=qc_build, persist_qc_report=qc_write,
            datetime=SimpleNamespace(now=lambda tz: SimpleNamespace(strftime=lambda fmt: "2026-11-06T00:00:00Z")), timezone=SimpleNamespace(utc=None))
        path = fs.Path("/input/movie.mp4" if media else "/input/input.srt")
        fs.nodes[str(path)] = {"kind": "file", "data": b"synthetic source"}
        fn = ns["run_cinematic_pipeline"] if media else ns["run_subtitle_translation_pipeline"]
        asyncio.run(fn(JOB, path))
        self.assertEqual(calls, [((), dict(output_dir=ns["settings"].OUTPUT_DIR, job_id=JOB, srt_content=PAYLOADS["srt"], ass_content=PAYLOADS["ass"], vtt_content=PAYLOADS["vtt"]))], "CALLER_FORWARDING_ORACLE")
        job = manager.jobs[JOB]
        self.assertEqual(job["status"], "failed" if failure else "completed", "CALLER_TERMINAL_STATE_ORACLE")
        if failure:
            self.assertIn("synthetic publication failure", job["error"])
            self.assertNotIn(("qc-builder",), events, "NO_QC_AFTER_FAILED_PUBLICATION_ORACLE")
            self.assertNotIn(("job", "completed"), events)
        else:
            self.assertLess(events.index(("publish",)), events.index(("qc-builder",)))
            self.assertLess(events.index(("qc-writer",)), events.index(("job", "completed")))
            self.assertEqual(job["qc_report_status"], "written")
            self.assertEqual(job["srt_path"], "/api/download/" + JOB + "/srt")
            self.assertEqual(job["ass_path"], "/api/download/" + JOB + "/ass")
            self.assertEqual(job["vtt_path"], "/api/subtitles/" + JOB + "/vtt")
            self.assertEqual(job["progress"], 100)
            self.assertEqual((job["whisper_time"], job["translation_time"], job["total_time"]), (5, 7, 40) if media else (0, 5, 20), "CALLER_TIMING_ORACLE")
        output_unlinks = [e for e in fs.events if e[0] == "unlink" and e[1].startswith("/out/")]
        self.assertEqual(output_unlinks, [], "NO_OUTPUT_CLEANUP_ORACLE")

    def test_clean_triplet_staging_and_promotion(self): self.assert_success()
    def test_newline_neutrality_crlf_and_lf(self):
        for text in ("a\nb\n", "a\r\nb\r\n", "مرحبا\r\nworld\n"):
            with self.subTest(text=text): self.assert_success({f: text for f in FORMATS})
    def test_empty_content_fidelity(self): self.assert_success({f: "" for f in FORMATS})
    def test_exact_byte_and_descriptor_counts(self): self.assert_success(); self.assert_order()
    def test_path_traversal_job_id_rejected(self):
        for bad in (None, 42, True, "../job", "job\\id", "D:job", "a\0b", JOB.upper(), JOB.replace("-", ""), " " + JOB):
            fs = MemoryFS(); ns = subject(fs)
            with self.subTest(bad=bad), self.assertRaises(ns["PublicationError"]): ns["publish_subtitle_triplet"](fs.Path("/out"), bad, "s", "a", "v")
            self.assertEqual(fs.fds, {}, "VALIDATION_BEFORE_STAGE_ORACLE")
    def test_missing_or_non_dir_output_rejected(self):
        for kind in (None, "file"):
            fs = MemoryFS(); ns = subject(fs)
            if kind is None: fs.nodes.pop("/out")
            else: fs.nodes["/out"]["kind"] = kind
            with self.subTest(kind=kind), self.assertRaises(ns["PublicationError"]): invoke(fs, ns)
            self.assertEqual(fs.fds, {})
    def test_reparse_ancestor_rejected_fail_closed(self):
        for key in ("/", "/out"):
            fs = MemoryFS(); fs.nodes[key]["reparse"] = True; ns = subject(fs)
            with self.subTest(path=key), self.assertRaises(ns["PublicationError"]): invoke(fs, ns)
            self.assertEqual(fs.fds, {})
            self.assertFalse(any(e[0].startswith("follow-") for e in fs.events), "NONFOLLOWING_ANCESTOR_FIRST_ORACLE")
    def test_reparse_destination_rejected_fail_closed(self):
        for kind, reparse in (("link", False), ("file", True)):
            fs = MemoryFS(); fs.nodes[fs.dest("srt")] = {"kind": kind, "reparse": reparse, "data": b"old"}; ns = subject(fs)
            with self.subTest(kind=kind), self.assertRaises(ns["PublicationError"]): invoke(fs, ns)
            self.assertEqual(fs.fds, {})
    def test_nonregular_destination_rejected(self):
        for kind in ("dir", "other"):
            fs = MemoryFS(); fs.nodes[fs.dest("ass")] = {"kind": kind, "data": b""}; ns = subject(fs)
            with self.subTest(kind=kind), self.assertRaises(ns["PublicationError"]): invoke(fs, ns)
            self.assertEqual(fs.fds, {})
    def test_validation_root_to_leaf_and_lstat_metadata(self):
        fs = MemoryFS(); result = subject(fs)["_validate_subtitle_output_paths"](fs.Path("/out"), JOB)
        self.assertEqual([e[1] for e in fs.events if e[0] == "lstat"][:2], ["/", "/out"], "ROOT_TO_LEAF_ORACLE")
        self.assertLess(fs.events.index(("lstat", "/out")), next(i for i, e in enumerate(fs.events) if e[0].startswith("follow-")), "NONFOLLOWING_BEFORE_FOLLOW_ORACLE")
        for fmt, dest in result.items(): self.assertEqual((str(dest.parent), dest.name), ("/out", JOB + "." + fmt), "LEXICAL_DESTINATION_ORACLE")
        failure = OSError("inspection fault"); fs = MemoryFS({("follow-exists", 1): failure}); ns = subject(fs)
        with self.assertRaises(ns["PublicationError"]) as caught: invoke(fs, ns)
        self.assertIs(caught.exception.__cause__, failure); self.assertEqual(fs.fds, {})
        fs = MemoryFS({("lstat", 1): OSError("lstat denied")}); ns = subject(fs)
        with self.assertRaises(ns["PublicationError"]): invoke(fs, ns)
        self.assertEqual(fs.fds, {}, "LSTAT_FAIL_CLOSED_ORACLE")
    def test_wrap_failure_closes_raw_descriptor_without_double_close(self):
        for secondary in (None, OSError("raw close failed"), KeyboardInterrupt("raw close interrupted")):
            with self.subTest(secondary=type(secondary).__name__): self.assert_primary(primary=OSError("wrap failed"), secondary=secondary, phase="wrap")
    def test_stream_close_failure_prevented_from_double_close(self):
        failure = OSError("close failed"); fs = MemoryFS({("stream-close", 1): failure}); ns = subject(fs)
        with self.assertRaises(ns["PublicationError"]) as caught: invoke(fs, ns)
        self.assertIs(caught.exception.__cause__, failure); self.assert_closed_once(fs); self.assert_no_promotion(fs)
    def test_base_exception_cleanup_closes_stream_and_propagates(self): self.assert_primary()
    def test_short_write_triggers_publication_error(self):
        for stage in (1, 2, 3):
            with self.subTest(stage=stage): self.assert_short(stage=stage)
    def test_fstat_size_mismatch_triggers_publication_error(self):
        for stage in (1, 2, 3):
            with self.subTest(stage=stage): self.assert_size(stage=stage)
    def test_stage_ordering_and_mandatory_fsync_oracle(self): self.assert_order()
    def test_ordinary_fault_matrix_deterministic(self):
        for op in ("create", "wrap", "write", "flush", "fsync", "fstat", "stream-close"):
            for stage in (1, 2, 3):
                failure = OSError(op + " fault"); fs = MemoryFS({(op, stage): failure}, preexisting=True); ns = subject(fs)
                with self.subTest(op=op, stage=stage), self.assertRaises(ns["PublicationError"]) as caught: invoke(fs, ns)
                self.assertIs(caught.exception.__cause__, failure, "STAGE_ERROR_CAUSE_ORACLE")
                self.assertEqual(fs.canonical(), fs.old, "PREPROMOTION_CANONICAL_PRESERVATION_ORACLE")
                self.assert_closed_once(fs); self.assert_no_promotion(fs); self.assert_no_unlink(fs)
                self.assertEqual(len(fs.staged()), stage - 1 if op == "create" else stage)
    def test_base_exception_primary_identity_and_secondary_close_handling(self):
        for phase in ("write", "wrap", "fstat"):
            for primary in (asyncio.CancelledError("cancel"), KeyboardInterrupt("interrupt"), OSError("ordinary")):
                for secondary in (None, OSError("close error"), KeyboardInterrupt("close interrupt"), SystemExit("close exit")):
                    with self.subTest(phase=phase, primary=type(primary).__name__, secondary=type(secondary).__name__): self.assert_primary(primary=primary, secondary=secondary, phase=phase)
    def test_replacement_failure_matrix_absent_and_existing(self):
        for existing in (False, True):
            for step in (1, 2, 3):
                with self.subTest(existing=existing, step=step): self.assert_replacement(step, existing)
    def test_replacement_failure_with_preexisting_outputs(self): self.assert_replacement(2, True)
    def test_run_subtitle_translation_pipeline_success_and_failure(self): self.assert_caller(False); self.assert_caller(False, True)
    def test_run_cinematic_pipeline_caller_failure_handling(self): self.assert_caller(True, True)
    def test_run_cinematic_pipeline_caller_success_and_failure(self): self.assert_caller(True); self.assert_caller(True, True)
    def test_genuine_in_memory_mutants(self):
        controls = [("no-fsync", self.assert_order, "MANDATORY_FSYNC_ORACLE"),
            ("no-short-count", self.assert_short, "SHORT_COUNT_ORACLE|Short write"),
            ("no-size-check", self.assert_size, "DESCRIPTOR_SIZE_ORACLE"),
            ("early-promotion", self.assert_order, "STAGE_COMPLETION_BEFORE_PROMOTION_ORACLE"),
            ("no-base-cleanup", self.assert_primary, "SINGLE_CLOSE_ORACLE"),
            ("double-close", self.assert_order, "SINGLE_CLOSE_ORACLE"),
            ("mask-primary", lambda custom_source: self.assert_primary(custom_source, secondary=KeyboardInterrupt("secondary")), "PRIMARY_IDENTITY_ORACLE"),
            ("wrong-caller-data", lambda custom_source: self.assert_caller(True, custom_source=custom_source), "CALLER_FORWARDING_ORACLE")]
        for kind, oracle, reason in controls:
            with self.subTest(mutant=kind), self.assertRaisesRegex(AssertionError, reason): oracle(custom_source=mutated_source(kind))
        fs = MemoryFS(); ns = subject(fs, mutated_source("unlink"))
        try: invoke(fs, ns)
        except ns["PublicationError"]: pass
        with self.assertRaisesRegex(AssertionError, "NO_UNLINK_ORACLE"): self.assert_no_unlink(fs)
        with patch.dict(globals(), PIPELINE_PATH=ROOT / "archive.py"):
            with self.assertRaisesRegex(AssertionError, "SOURCE_ANCHOR_ORACLE"): source_text()
    def test_positive_isolation_controls(self):
        for name in ("app", "core.config", "services.pipeline", "torch", "transformers", "httpx", "requests", "socket", "groq", "dotenv"):
            with self.subTest(import_name=name), self.assertRaisesRegex(AssertionError, "ISOLATION_ORACLE"): __import__(name)
        ns = subject(MemoryFS())
        for attempt in (lambda: ns["__builtins__"]["open"](".env"), lambda: Path(".env").read_text(),
            lambda: os.getenv("DO_NOT_READ"), lambda: subprocess.run(["DO_NOT_RUN"]), lambda: subprocess.Popen(["DO_NOT_RUN"]),
            lambda: threading.Thread(target=lambda: None).start(), lambda: asyncio.to_thread(lambda: None)):
            with self.assertRaisesRegex(AssertionError, "ISOLATION_ORACLE"): attempt()


if __name__ == "__main__": unittest.main()
