"""Source-backed /api/subtitles/{job_id}/vtt preview eligibility only; no FastAPI/server/config/media I/O."""
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
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'get_vtt_subtitles']
    if len(nodes) != 1:
        raise AssertionError('exactly one actual preview handler required')
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
    def read_text(self, encoding='utf-8'):
        self.fixture.read_calls.append(self.text)
        if self.text not in self.fixture.files:
            raise FileNotFoundError(self.text)
        return self.fixture.files[self.text]

class Fixture:
    def __init__(self, status='completed', missing=False, record=None, vtt_present=True, srt_present=True, handler=None):
        self.job_id = 'synthetic-job'
        self.files = {}
        if vtt_present:
            self.files['OUTPUT/synthetic-job.vtt'] = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nمرحبا بك"
        if srt_present:
            self.files['OUTPUT/synthetic-job.srt'] = "1\n00:00:01,000 --> 00:00:04,000\nمرحبا بك"
        self.exists_calls, self.read_calls, self.responses = [], [], []
        self.jobs = {'unrelated-job': {'status': 'processing', 'marker': {'preserve': [1, 2]}}}
        if not missing:
            self.jobs[self.job_id] = record if record is not None else {'status': status, 'filename': 'فيلم.v1.mp4'}
        def file_response(**kwargs):
            res = dict(kind='FileResponse', **kwargs)
            self.responses.append(res)
            return res
        def plain_response(**kwargs):
            res = dict(kind='Response', **kwargs)
            self.responses.append(res)
            return res
        original_import = builtins.__import__
        def fixture_import(name, *args, **kwargs):
            if name == 're':
                return original_import(name, *args, **kwargs)
            denied()
        self.ns = dict(__name__='isolated_actual_preview_handler',
            job_manager=SimpleNamespace(jobs=self.jobs),
            settings=SimpleNamespace(OUTPUT_DIR=MemoryPath('OUTPUT', self)),
            HTTPException=FakeHTTPException, FileResponse=file_response, Response=plain_response,
            Path=lambda text: MemoryPath(text, self),
            __builtins__=dict(vars(builtins), __import__=fixture_import, open=denied))
        tree = ast.Module(body=[copy.deepcopy(handler if handler is not None else selected_handler())], type_ignores=[])
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(APP_PATH), 'exec'), self.ns)
    def request(self):
        return asyncio.run(self.ns['get_vtt_subtitles'](self.job_id))

class TestPreviewVisibilityPolicy(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 're':
                return original_import(name, *args, **kwargs)
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

    def assert_blocked(self, f):
        before_jobs, before_files = copy.deepcopy(f.jobs), copy.deepcopy(f.files)
        try:
            f.request()
        except FakeHTTPException as exc:
            self.assertEqual(exc.status_code, 404, 'visibility gate must return HTTP404')
            self.assertEqual(exc.detail, 'ملف الترجمة غير جاهز بعد', 'visibility gate response detail')
        else:
            self.fail('visibility gate must reject ineligible job')
        self.assertEqual(f.exists_calls, [], 'ineligible job must be rejected before filesystem access')
        self.assertEqual(f.read_calls, [], 'ineligible job must not read artifact files')
        self.assertEqual(f.responses, [], 'ineligible job must not construct Response or FileResponse')
        self.assertEqual(f.jobs, before_jobs, 'request must preserve every job record')
        self.assertEqual(f.files, before_files, 'request must preserve every output artifact')

    def assert_direct_vtt(self, f):
        before_jobs, before_files = copy.deepcopy(f.jobs), copy.deepcopy(f.files)
        try:
            response = f.request()
        except FakeHTTPException:
            self.fail('eligible completed job with direct VTT must be served')
        self.assertEqual(response['kind'], 'FileResponse')
        self.assertEqual(str(response['path']), 'OUTPUT/synthetic-job.vtt')
        self.assertEqual(response['media_type'], 'text/vtt; charset=utf-8')
        self.assertEqual(f.exists_calls, ['OUTPUT/synthetic-job.vtt'])
        self.assertEqual(f.read_calls, [])
        self.assertEqual(len(f.responses), 1)
        self.assertEqual(f.jobs, before_jobs)
        self.assertEqual(f.files, before_files)

    def assert_fallback_srt(self, f):
        before_jobs, before_files = copy.deepcopy(f.jobs), copy.deepcopy(f.files)
        try:
            response = f.request()
        except FakeHTTPException:
            self.fail('eligible completed job with fallback SRT must be converted and served')
        self.assertEqual(response['kind'], 'Response')
        self.assertTrue(response['content'].startswith('WEBVTT\n\n'))
        self.assertIn('00:00:01.000 --> 00:00:04.000', response['content'])
        self.assertEqual(response['media_type'], 'text/vtt; charset=utf-8')
        self.assertEqual(f.exists_calls, ['OUTPUT/synthetic-job.vtt', 'OUTPUT/synthetic-job.srt'])
        self.assertEqual(f.read_calls, ['OUTPUT/synthetic-job.srt'])
        self.assertEqual(len(f.responses), 1)
        self.assertEqual(f.jobs, before_jobs)
        self.assertEqual(f.files, before_files)

    def test_01_actual_handler_extraction_and_positive_isolation_controls(self):
        self.assertEqual(selected_handler().name, 'get_vtt_subtitles')
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
        self.assert_direct_vtt(f)

    def test_02_both_completed_states_serve_direct_vtt_artifact(self):
        for status in ('completed', 'completed_with_warnings'):
            with self.subTest(status=status):
                self.assert_direct_vtt(Fixture(status=status, vtt_present=True, srt_present=True))

    def test_03_both_completed_states_convert_fallback_srt_artifact(self):
        for status in ('completed', 'completed_with_warnings'):
            with self.subTest(status=status):
                self.assert_fallback_srt(Fixture(status=status, vtt_present=False, srt_present=True))

    def test_04_ineligible_statuses_deny_both_direct_and_fallback_artifacts_without_path_access(self):
        for status in ('pending', 'processing', 'failed', 'cancelled', 'completed_partial', None, True):
            for vtt_p, srt_p in ((True, True), (False, True), (True, False), (False, False)):
                with self.subTest(status=status, vtt=vtt_p, srt=srt_p):
                    self.assert_blocked(Fixture(status=status, vtt_present=vtt_p, srt_present=srt_p))

    def test_05_missing_or_nonplain_records_deny_artifacts_before_path_access(self):
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

    def test_06_completed_job_with_missing_artifacts_preserves_unready_vtt_404(self):
        f = Fixture(vtt_present=False, srt_present=False)
        with self.assertRaises(FakeHTTPException) as caught:
            f.request()
        self.assertEqual(caught.exception.status_code, 404)
        self.assertEqual(caught.exception.detail, 'ملف الترجمة VTT غير متوفر')
        self.assertEqual(f.exists_calls, ['OUTPUT/synthetic-job.vtt', 'OUTPUT/synthetic-job.srt'])
        self.assertEqual(f.read_calls, [])
        self.assertEqual(f.responses, [])

    def test_07_status_transitions_and_job_isolation(self):
        f = Fixture()
        self.assert_direct_vtt(f)
        f.exists_calls.clear()
        f.read_calls.clear()
        f.responses.clear()
        f.jobs[f.job_id]['status'] = 'processing'
        self.assert_blocked(f)
        f.jobs[f.job_id]['status'] = 'failed'
        self.assert_blocked(f)

    def test_08_genuine_handler_mutants_fail_intended_shared_oracles(self):
        for kind, reason in (('remove-gate', 'visibility gate must reject ineligible job'),
                             ('allow-processing', 'visibility gate must reject ineligible job'),
                             ('reject-completed', 'eligible completed job with direct VTT must be served'),
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
                node.body.insert(1, ast.parse('(settings.OUTPUT_DIR / f"{job_id}.vtt").exists()').body[0])
            f = Fixture(status='completed' if kind == 'reject-completed' else 'processing', handler=node)
            with self.subTest(mutation=kind), self.assertRaisesRegex(AssertionError, reason):
                if kind == 'reject-completed':
                    self.assert_direct_vtt(f)
                else:
                    self.assert_blocked(f)

if __name__ == '__main__':
    unittest.main()
