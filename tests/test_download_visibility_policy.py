"""Source-backed /api/download eligibility only; no FastAPI/server/config/media I/O."""
import ast
import asyncio
import builtins
from contextlib import ExitStack
import copy
import io
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'
APP_SOURCE = APP_PATH.read_text(encoding='utf-8-sig')

def denied(*args, **kwargs):
    raise AssertionError('ISOLATION: forbidden external operation')

def selected_handler(source=APP_SOURCE):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'download_subtitles']
    if len(nodes) != 1:
        raise AssertionError('exactly one actual download handler required')
    node = copy.deepcopy(nodes[0])
    node.decorator_list = []
    return node

class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail
        super().__init__(detail)

class MemoryPath:
    def __init__(self, text, fixture):
        self.text, self.fixture = str(text), fixture
    def __truediv__(self, name):
        return MemoryPath(self.text + '/' + name, self.fixture)
    def __str__(self):
        return self.text
    @property
    def stem(self):
        return self.text.replace('\\', '/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
    def exists(self):
        self.fixture.exists_calls.append(self.text)
        return self.text in self.fixture.files

class Fixture:
    def __init__(self, status='completed', missing=False, record=None, handler=None):
        self.job_id = 'synthetic-job'
        self.files = {'OUTPUT/synthetic-job' + ext: 'SYNTHETIC_NOT_MEDIA' for ext in ('.srt', '.ass', '.vtt')}
        self.exists_calls, self.responses = [], []
        self.jobs = {'unrelated-job': {'status': 'processing', 'marker': {'preserve': [1, 2]}}}
        if not missing:
            self.jobs[self.job_id] = record if record is not None else {'status': status, 'filename': 'فيلم.v1.mp4'}
        def file_response(**kwargs):
            self.responses.append(kwargs)
            return kwargs
        self.ns = dict(__name__='isolated_actual_download_handler',
            job_manager=SimpleNamespace(jobs=self.jobs),
            settings=SimpleNamespace(OUTPUT_DIR=MemoryPath('OUTPUT', self)),
            HTTPException=FakeHTTPException, FileResponse=file_response,
            Path=lambda text: MemoryPath(text, self),
            __builtins__=dict(vars(builtins), __import__=denied, open=denied))
        tree = ast.Module(body=[copy.deepcopy(handler if handler is not None else selected_handler())], type_ignores=[])
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(APP_PATH), 'exec'), self.ns)
    def request(self, fmt):
        return asyncio.run(self.ns['download_subtitles'](self.job_id, fmt))

class TestDownloadVisibilityPolicy(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'app', 'core', 'services', 'fastapi', 'starlette', 'dotenv', 'torch', 'transformers', 'requests', 'httpx', 'socket'}:
                denied()
            return original_import(name, *args, **kwargs)
        stack.enter_context(patch.object(builtins, '__import__', guarded_import))
        for obj, key in ((os, 'getenv'), (subprocess, 'run'), (subprocess, 'Popen'), (threading.Thread, 'start'), (asyncio, 'to_thread')):
            stack.enter_context(patch.object(obj, key, denied))
        for obj, key in ((builtins, 'open'), (io, 'open')):
            original = getattr(obj, key)
            def guarded_open(path, *args, _original=original, **kwargs):
                if isinstance(path, (str, bytes, os.PathLike)):
                    name = os.fsdecode(path).replace('\\', '/').lower().split('/')[-1]
                    if name == '.env' or name.startswith('.env.') or name in {'credentials', 'secrets.json'}:
                        denied()
                return _original(path, *args, **kwargs)
            stack.enter_context(patch.object(obj, key, guarded_open))

    def assert_blocked(self, f, fmt='srt'):
        before_jobs, before_files = copy.deepcopy(f.jobs), copy.deepcopy(f.files)
        try:
            f.request(fmt)
        except FakeHTTPException as exc:
            self.assertEqual(exc.status_code, 404, 'visibility gate must return HTTP404')
            self.assertEqual(exc.detail, 'ملف الترجمة غير جاهز بعد', 'visibility gate response detail')
        else:
            self.fail('visibility gate must reject ineligible job')
        self.assertEqual(f.exists_calls, [], 'ineligible job must be rejected before filesystem access')
        self.assertEqual(f.responses, [], 'ineligible job must not construct FileResponse')
        self.assertEqual(f.jobs, before_jobs, 'request must preserve every job record')
        self.assertEqual(f.files, before_files, 'request must preserve every output artifact')

    def assert_download(self, f, fmt, ext):
        before_jobs, before_files = copy.deepcopy(f.jobs), copy.deepcopy(f.files)
        try:
            response = f.request(fmt)
        except FakeHTTPException:
            self.fail('eligible completed job must remain downloadable')
        self.assertEqual(set(response), {'path', 'filename', 'media_type'}, 'FileResponse arguments preserved')
        self.assertEqual(str(response['path']), 'OUTPUT/synthetic-job' + ext, 'requested format path')
        self.assertEqual(response['filename'], 'فيلم.v1_arabic' + ext, 'download filename preserved')
        self.assertEqual(response['media_type'], 'text/plain; charset=utf-8', 'download media type preserved')
        self.assertEqual(f.exists_calls, ['OUTPUT/synthetic-job' + ext], 'eligible file existence check')
        self.assertEqual(len(f.responses), 1, 'exactly one FileResponse')
        self.assertEqual(f.jobs, before_jobs, 'eligible request preserves jobs')
        self.assertEqual(f.files, before_files, 'eligible request preserves files')

    def test_01_actual_handler_extraction_and_positive_isolation_controls(self):
        self.assertEqual(selected_handler().name, 'download_subtitles')
        for name in ('app', 'core.config', 'services.pipeline', 'fastapi', 'dotenv', 'torch', 'httpx'):
            with self.assertRaisesRegex(AssertionError, 'ISOLATION:'):
                __import__(name)
        for attempt in (lambda: os.getenv('NEVER_READ'), lambda: Path('.env').read_text(),
                        lambda: subprocess.run(['NEVER_EXECUTE']), lambda: subprocess.Popen(['NEVER_EXECUTE']),
                        lambda: threading.Thread(target=lambda: None).start(), lambda: asyncio.to_thread(lambda: None)):
            with self.assertRaisesRegex(AssertionError, 'ISOLATION:'):
                attempt()
        f = Fixture()
        with self.assertRaisesRegex(AssertionError, 'ISOLATION:'):
            f.ns['__builtins__']['__import__']('app')
        with self.assertRaisesRegex(AssertionError, 'ISOLATION:'):
            f.ns['__builtins__']['open']('.env')
        self.assert_download(f, 'srt', '.srt')

    def test_02_both_completed_states_keep_all_three_formats(self):
        for status in ('completed', 'completed_with_warnings'):
            for fmt, ext in (('srt', '.srt'), (' ASS ', '.ass'), ('VTT', '.vtt')):
                with self.subTest(status=status, fmt=fmt):
                    self.assert_download(Fixture(status=status), fmt, ext)

    def test_03_ineligible_states_deny_existing_artifacts_without_path_access(self):
        for status in ('pending', 'processing', 'failed', 'cancelled', 'completed_partial', None, True):
            for fmt in ('srt', 'ass', 'vtt'):
                with self.subTest(status=status, fmt=fmt):
                    self.assert_blocked(Fixture(status=status), fmt)

    def test_04_missing_or_nonplain_records_deny_existing_artifacts(self):
        self.assert_blocked(Fixture(missing=True))
        null_record = Fixture()
        null_record.jobs[null_record.job_id] = None
        self.assert_blocked(null_record)
        for record in ([], 'invalid-record', 42):
            with self.subTest(record=record):
                self.assert_blocked(Fixture(record=record))
        class HostileRecord(dict):
            def get(self, *args, **kwargs):
                raise AssertionError('nonplain record getter must never execute')
        self.assert_blocked(Fixture(record=HostileRecord(status='completed', filename='x.mp4')))

    def test_05_completed_job_with_missing_artifact_keeps_existing_404(self):
        f = Fixture()
        f.files.clear()
        with self.assertRaises(FakeHTTPException) as caught:
            f.request('ass')
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.detail, 'ملف الترجمة غير جاهز بعد')
        self.assertEqual(f.exists_calls, ['OUTPUT/synthetic-job.ass'])
        self.assertEqual(f.responses, [])

    def test_06_legacy_unknown_format_srt_fallback_is_preserved(self):
        for fmt in ('', 'json', 'unknown-format'):
            with self.subTest(fmt=fmt):
                self.assert_download(Fixture(), fmt, '.srt')

    def test_07_status_transition_changes_next_request_without_mutation(self):
        f = Fixture()
        self.assert_download(f, 'vtt', '.vtt')
        f.exists_calls.clear()
        f.responses.clear()
        f.jobs[f.job_id]['status'] = 'processing'
        self.assert_blocked(f, 'vtt')
        f.jobs[f.job_id]['status'] = 'failed'
        self.assert_blocked(f, 'vtt')

    def test_08_genuine_handler_mutants_fail_the_intended_shared_oracles(self):
        for kind, reason in (('remove-gate', 'visibility gate must reject ineligible job'),
                             ('allow-processing', 'visibility gate must reject ineligible job'),
                             ('reject-completed', 'eligible completed job must remain downloadable'),
                             ('touch-file-first', 'ineligible job must be rejected before filesystem access')):
            node = selected_handler()
            guard = next(n for n in node.body if isinstance(n, ast.If) and any(isinstance(v, ast.Name) and v.id == 'job_record' for v in ast.walk(n.test)))
            if kind == 'remove-gate':
                node.body = [n for n in node.body if n is not guard and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'job_record' for t in n.targets))]
            elif kind in ('allow-processing', 'reject-completed'):
                for n in ast.walk(guard.test):
                    if isinstance(n, ast.Constant) and n.value == 'completed':
                        n.value = 'processing' if kind == 'allow-processing' else 'never_completed'
            else:
                node.body.insert(1, ast.parse('(settings.OUTPUT_DIR / f"{job_id}.srt").exists()').body[0])
            f = Fixture(status='completed' if kind == 'reject-completed' else 'processing', handler=node)
            with self.subTest(mutation=kind), self.assertRaisesRegex(AssertionError, reason):
                if kind == 'reject-completed':
                    self.assert_download(f, 'srt', '.srt')
                else:
                    self.assert_blocked(f)

if __name__ == '__main__':
    unittest.main()
