"""Actual upload helper/handler AST, stdlib-only synthetic storage/streams.

Never imports the production app, FastAPI/config/model stack or writes app
uploads. No HTTP server, multipart parsing, decoding or background work runs.
Synthetic failure ordering is not real filesystem durability/security proof.
"""
import ast
import asyncio
import logging
import unittest
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SUBTITLES = {".srt", ".ass", ".vtt"}
MEDIA = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
         ".ts", ".mts", ".m2ts", ".vob", ".ogv", ".3gp", ".3g2", ".mp3",
         ".m4a", ".m4b", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma",
         ".aiff", ".aif", ".amr"}


class SyntheticHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class UploadFixture:
    def __init__(self, filename="source.srt", chunks=(b"hello",), maximum=10,
                 fault=None, preexisting=None):
        self.filename, self.chunks = filename, list(chunks)
        self.store, self.events, self.jobs, self.tasks = dict(preexisting or {}), [], [], []
        self.active, self.close_calls, self.uuid_calls = set(), 0, 0
        self.result = self.failure = None
        fixture = self

        class MemoryPath:
            def __init__(self, name):
                self.name = name
            def __truediv__(self, name):
                return MemoryPath(self.name + "/" + name)
            def exists(self):
                return self.name in fixture.store
            def unlink(self, **kwargs):
                fixture.events.append(("unlink", self.name))
                if self.name in fixture.active:
                    raise AssertionError("Cannot remove an open writer on Windows")
                if fault == "unlink":
                    raise PermissionError("synthetic cleanup failure")
                fixture.store.pop(self.name, None)
            def rename(self, target):
                fixture.events.append(("rename", self.name, target.name))
                if self.name in fixture.active:
                    raise AssertionError("Writer must close before rename")
                if fault == "rename":
                    raise OSError("synthetic rename failure")
                fixture.store[target.name] = fixture.store.pop(self.name)

        class Writer:
            def __init__(self, path, mode):
                fixture.events.append(("open", path.name, mode))
                if mode != "xb":
                    raise AssertionError("Exclusive partial creation required")
                if path.name in fixture.store:
                    raise FileExistsError("synthetic partial collision")
                if fault == "open":
                    raise OSError("synthetic open failure")
                self.path = path
                fixture.store[path.name] = b""
            def __enter__(self):
                fixture.active.add(self.path.name)
                return self
            def __exit__(self, *args):
                fixture.active.remove(self.path.name)
                fixture.events.append(("writer_closed", self.path.name))
            def write(self, data):
                fixture.events.append(("write", self.path.name, bytes(data)))
                if fault == "write":
                    raise OSError("synthetic write failure")
                fixture.store[self.path.name] += data

        class Upload:
            def __init__(self):
                self.filename = filename
            async def read(self, size):
                fixture.events.append(("read", size))
                if fault == "read":
                    raise OSError("synthetic read failure")
                return fixture.chunks.pop(0) if fixture.chunks else b""
            async def close(self):
                fixture.close_calls += 1
                fixture.events.append(("upload_closed",))
                if fault == "close":
                    raise OSError("synthetic close failure")

        def new_uuid():
            fixture.uuid_calls += 1
            return "FAKE_UUID"
        def media(*args, **kwargs):
            raise AssertionError("No background pipeline may execute")
        def subtitles(*args, **kwargs):
            raise AssertionError("No background pipeline may execute")
        def add_task(function, **kwargs):
            fixture.tasks.append((function, kwargs))
        logger = logging.Logger("synthetic-upload-policy")
        logger.addHandler(logging.NullHandler())
        self.namespace = dict(Path=Path, PureWindowsPath=PureWindowsPath, Optional=Optional,
                              HTTPException=SyntheticHTTPException, BackgroundTasks=object,
                              UploadFile=object, File=lambda x: x, Form=lambda x: x,
                              uuid=SimpleNamespace(uuid4=new_uuid), logger=logger, open=Writer,
                              settings=SimpleNamespace(UPLOAD_DIR=MemoryPath("memory_uploads"),
                                                       MAX_FILE_SIZE_BYTES=maximum, MAX_FILE_SIZE_MB=1),
                              job_manager=SimpleNamespace(create_job=lambda *args: fixture.jobs.append(args)),
                              run_cinematic_pipeline=media, run_subtitle_translation_pipeline=subtitles)
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8-sig"))
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id in {"ALLOWED_SUBTITLE_EXTS", "ALLOWED_MEDIA_EXTS"} for t in node.targets):
                nodes.append(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                    "resolve_upload_filename", "upload_file"}:
                node.decorator_list = []
                nodes.append(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     "reviewed_upload_ast_synthetic_io", "exec"), self.namespace)
        self.upload = Upload()
        self.background = SimpleNamespace(add_task=add_task)
        self.media, self.subtitles = media, subtitles

    def run(self):
        try:
            self.result = asyncio.run(self.namespace["upload_file"](
                self.background, self.upload, "transcribe-spy", "translate-spy", "fr"))
        except Exception as exc:
            self.failure = exc
        return self


class TestUploadValidation(unittest.TestCase):
    def assert_rejected_before_storage(self, filename):
        case = UploadFixture(filename).run()
        self.assertIsInstance(case.failure, SyntheticHTTPException)
        self.assertEqual(case.failure.status_code, 400)
        self.assertEqual(case.close_calls, 1)
        self.assertEqual(case.uuid_calls, 0)
        self.assertEqual(case.events, [("upload_closed",)])
        self.assertEqual((case.store, case.jobs, case.tasks), ({}, [], []))

    def test_missing_blank_and_nonstring_names(self):
        for name in [None, "", " \t\n", 12, b"file.srt", ".", "..", "/", "\\", "C:"]:
            with self.subTest(name=name):
                self.assert_rejected_before_storage(name)

    def test_trailing_separator_invalid_characters_and_controls(self):
        names = ["folder.srt/", "folder.ass\\", "file.srt.", "file.srt "]
        names += ["bad" + ch + "name.srt" for ch in '<>:"|?*']
        names += ["bad" + chr(code) + "name.srt" for code in [*range(32), 127]]
        for name in names:
            with self.subTest(name=repr(name)):
                self.assert_rejected_before_storage(name)

    def test_extension_rejections_happen_without_read_or_open(self):
        for name in ["file.exe", "file.bat", "file.sh", "file.py", "file.txt", "file.aa",
                     "file.aax", "file", "file.srt.exe", ".srt"]:
            with self.subTest(name=name):
                self.assert_rejected_before_storage(name)

    def test_exact_allowlists_and_all_case_insensitive_routes(self):
        policy = UploadFixture().namespace
        self.assertEqual(policy["ALLOWED_SUBTITLE_EXTS"], SUBTITLES)
        self.assertEqual(policy["ALLOWED_MEDIA_EXTS"], MEDIA)
        self.assertEqual((len(SUBTITLES), len(MEDIA)), (3, 27))
        for ext in sorted(SUBTITLES | MEDIA):
            for suffix in [ext, ext.upper()]:
                with self.subTest(suffix=suffix):
                    case = UploadFixture("source" + suffix).run()
                    self.assertIsNone(case.failure)
                    self.assertEqual(case.close_calls, 1)
                    self.assertEqual(case.uuid_calls, 1)
                    self.assertEqual(case.jobs, [("FAKE_UUID", "source" + suffix)])
                    self.assertEqual(len(case.tasks), 1)
                    function, kwargs = case.tasks[0]
                    self.assertIs(function, case.subtitles if ext in SUBTITLES else case.media)
                    self.assertEqual(kwargs["job_id"], "FAKE_UUID")
                    self.assertEqual(kwargs["translation_provider"], "translate-spy")
                    path_key = "subtitle_path" if ext in SUBTITLES else "video_path"
                    self.assertEqual(kwargs[path_key].name, "memory_uploads/FAKE_UUID" + ext)
                    if ext in SUBTITLES:
                        self.assertEqual(set(kwargs), {"job_id", "subtitle_path", "translation_provider", "source_language"})
                        self.assertEqual(kwargs["source_language"], "fr")
                    else:
                        self.assertEqual(set(kwargs), {"job_id", "video_path", "transcription_provider", "translation_provider", "language"})
                        self.assertEqual(kwargs["language"], "fr")
                        self.assertEqual(kwargs["transcription_provider"], "transcribe-spy")

    def test_windows_posix_unc_and_unicode_names_use_basename_only(self):
        for original, basename in [("../outside.SRT", "outside.SRT"),
                                   (r"..\outside.ass", "outside.ass"),
                                   (r"C:\client\اسم الفيلم.srt", "اسم الفيلم.srt"),
                                   (r"\\server\share\movie.vtt", "movie.vtt"),
                                   ("/client/日本語.mp4", "日本語.mp4")]:
            with self.subTest(original=original):
                case = UploadFixture(original).run()
                self.assertIsNone(case.failure)
                self.assertEqual(case.result["filename"], basename)
                self.assertEqual(case.jobs, [("FAKE_UUID", basename)])
                ext = PureWindowsPath(basename).suffix.lower()
                self.assertEqual(case.store, {"memory_uploads/FAKE_UUID" + ext: b"hello"})
                self.assertEqual(case.filename, original)

    def test_zero_bytes_close_writer_then_cleanup_exactly_once(self):
        case = UploadFixture(chunks=()).run()
        self.assertEqual(case.failure.status_code, 400)
        self.assertEqual((case.store, case.jobs, case.tasks), ({}, [], []))
        operations = [event[0] for event in case.events]
        self.assertLess(operations.index("writer_closed"), operations.index("unlink"))
        self.assertEqual(operations.count("unlink"), 1)
        self.assertNotIn("rename", operations)
        self.assertEqual(case.close_calls, 1)

    def test_success_stream_bytes_response_and_order_are_preserved(self):
        case = UploadFixture(chunks=(b"ab", b"cd", b"ef"), maximum=6).run()
        self.assertIsNone(case.failure)
        self.assertEqual(case.store, {"memory_uploads/FAKE_UUID.srt": b"abcdef"})
        self.assertEqual(case.result, {"success": True, "job_id": "FAKE_UUID", "filename": "source.srt"})
        self.assertEqual([e[1] for e in case.events if e[0] == "read"], [8 * 1024 * 1024] * 4)
        operations = [e[0] for e in case.events]
        self.assertEqual(operations.count("rename"), 1)
        self.assertLess(operations.index("writer_closed"), operations.index("rename"))
        self.assertLess(operations.index("rename"), operations.index("upload_closed"))
        self.assertNotIn("unlink", operations)

    def test_quota_single_and_multiple_chunk_boundary(self):
        for chunks, written in [((b"123456",), []), ((b"123", b"456"), [b"123"])]:
            case = UploadFixture(chunks=chunks, maximum=5).run()
            self.assertEqual(case.failure.status_code, 413)
            self.assertEqual((case.store, case.jobs, case.tasks), ({}, [], []))
            self.assertEqual([e[2] for e in case.events if e[0] == "write"], written)
            self.assertEqual(case.close_calls, 1)
            self.assertEqual(sum(e[0] == "unlink" for e in case.events), 1)
            self.assertFalse(any(e[0] == "rename" for e in case.events))
        self.assertIsNone(UploadFixture(chunks=(b"12345",), maximum=5).run().failure)

    def test_read_write_rename_failures_cleanup_own_partial(self):
        for fault in ["read", "write", "rename"]:
            with self.subTest(fault=fault):
                case = UploadFixture(fault=fault).run()
                self.assertEqual(case.failure.status_code, 500)
                self.assertEqual((case.store, case.jobs, case.tasks), ({}, [], []))
                self.assertEqual(case.close_calls, 1)
                operations = [e[0] for e in case.events]
                self.assertLess(operations.index("writer_closed"), operations.index("unlink"))
                self.assertEqual(operations.count("unlink"), 1)

    def test_open_failure_does_not_attempt_unowned_cleanup(self):
        case = UploadFixture(fault="open").run()
        self.assertEqual(case.failure.status_code, 500)
        self.assertEqual(case.close_calls, 1)
        self.assertEqual((case.store, case.jobs, case.tasks), ({}, [], []))
        self.assertFalse(any(e[0] == "unlink" for e in case.events))

    def test_partial_collision_preserves_existing_file_and_closes_once(self):
        existing = {"memory_uploads/FAKE_UUID.srt.part": b"DO NOT TOUCH"}
        case = UploadFixture(preexisting=existing).run()
        self.assertEqual(case.failure.status_code, 500)
        self.assertEqual(case.store, existing)
        self.assertEqual((case.jobs, case.tasks), ([], []))
        self.assertEqual(case.close_calls, 1)
        self.assertEqual([e[0] for e in case.events], ["open", "upload_closed"])

    def test_cleanup_failure_is_surfaced_without_retry_or_broader_delete(self):
        case = UploadFixture(chunks=(), fault="unlink").run()
        self.assertIsInstance(case.failure, PermissionError)
        self.assertEqual(case.close_calls, 1)
        self.assertEqual(case.store, {"memory_uploads/FAKE_UUID.srt.part": b""})
        self.assertEqual((case.jobs, case.tasks), ([], []))
        self.assertEqual(sum(e[0] == "unlink" for e in case.events), 1)

    def test_close_failure_surfaces_without_retries_or_final_file_deletion(self):
        case = UploadFixture(fault="close").run()
        self.assertIsInstance(case.failure, OSError)
        self.assertEqual(case.close_calls, 1)
        self.assertEqual(case.store, {"memory_uploads/FAKE_UUID.srt": b"hello"})
        self.assertEqual((case.jobs, case.tasks), ([], []))
        self.assertFalse(any(e[0] == "unlink" for e in case.events))


if __name__ == "__main__":
    unittest.main()
