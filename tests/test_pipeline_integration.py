"""AIX-CANCEL-STATE-001 v6 candidate: source-backed orchestration boundary tests.

Guarded JobManager/run_exclusive_worker ASTs with synchronous local terminal-state CancelledError handler;
all service/formatter/QC/path/clock boundaries are explicit in-memory spies.
Not HTTP, real-media, live-model or whole-application integration evidence.
Local terminal state on cancellation is implemented; external notification/resource cleanup remain deferred.
"""
import ast
import re
import asyncio
import builtins
from contextlib import ExitStack
import copy
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any, Dict, Optional
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / 'services/pipeline.py').is_file():
    raise FileNotFoundError(f'Expected repository root with services/pipeline.py at {ROOT}')
PIPELINE_PATH = ROOT / 'services/pipeline.py'
PIPELINE_SOURCE = PIPELINE_PATH.read_text(encoding='utf-8-sig')
SYMBOLS = {'JobManager', 'run_exclusive_worker', 'run_cinematic_pipeline'}


def denied(*args, **kwargs):
    raise AssertionError('ISOLATION: forbidden external operation')


def selected_nodes(source):
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in SYMBOLS]
    if len(nodes) != 3 or {n.name for n in nodes} != SYMBOLS:
        raise AssertionError('actual symbol extraction mismatch')
    return nodes


def _handler_type_name(h):
    t = getattr(h, 'type', None)
    if isinstance(t, ast.Name): return t.id
    if isinstance(t, ast.Attribute): return t.attr
    return None


def check_cancellation_handler_structural(cancel_handler):
    for node in ast.walk(cancel_handler):
        if isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom)):
            raise AssertionError(f'forbidden async/generator control flow: {type(node).__name__}')
    
    allowed_calls = {'job_manager.jobs.get', 'type', 'job_record.update'}
    for node in ast.walk(cancel_handler):
        if isinstance(node, ast.Call):
            call_name = ast.unparse(node.func).strip()
            if call_name not in allowed_calls:
                raise AssertionError(f'forbidden call in cancellation handler: {call_name}')
            if call_name == 'type':
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name) or node.args[0].id != 'job_record':
                    raise AssertionError('type() call must be strictly type(job_record)')
            elif call_name == 'job_manager.jobs.get':
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Name) or node.args[0].id != 'job_id':
                    raise AssertionError('job_manager.jobs.get call must be strictly (job_id)')
            elif call_name == 'job_record.update':
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Dict):
                    raise AssertionError('job_record.update call must be strictly ({...})')


def isolated_exec(nodes, namespace):
    namespace['__builtins__'] = dict(vars(builtins), __import__=denied, open=denied)
    exec(compile(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]),
                 str(PIPELINE_PATH), 'exec'), namespace)


class MemoryPath:
    """Opaque path token: every write/unlink is a dict operation, never host I/O."""
    def __init__(self, text, fixture):
        self.text, self.fixture = text, fixture

    def __truediv__(self, name):
        return MemoryPath(self.text + '/' + name, self.fixture)

    def __str__(self):
        return self.text

    @property
    def name(self):
        return self.text.rsplit('/', 1)[-1]

    @property
    def stem(self):
        return self.name.rsplit('.', 1)[0]

    def exists(self):
        answer = self.text in self.fixture.files
        self.fixture.events.append(('exists', self.text, answer))
        return answer

    def unlink(self, **kwargs):
        self.fixture.events.append(('unlink', self.text, kwargs))
        self.fixture.files.pop(self.text, None)

    def write_text(self, text, **kwargs):
        self.fixture.events.append(('write', self.text, text, kwargs))
        self.fixture.files[self.text] = text
        return len(text)


class Fixture:
    """Actual orchestrator plus distinguishable deterministic boundary spies."""
    def __init__(self, source=PIPELINE_SOURCE, job_id='job-a', warned=False,
                 failure=None, error=None, partial=True, empty=False, progress=False):
        self.events, self.files, self.calls = [], {}, {}
        self.worker_cancellations = []
        self.job_id, self.warned = job_id, warned
        self.failure, self.error, self.partial = failure, error, partial
        self.empty, self.progress = empty, progress
        self.requested, self.detected = 'fr', 'en'
        self.stt_provider, self.translation_provider = 'synthetic-stt', 'synthetic-translation'
        self.video = MemoryPath('INPUT/' + job_id + '.mp4', self)
        self.temp = MemoryPath('TEMP/' + job_id, self)
        self.output = MemoryPath('OUTPUT/' + job_id, self)
        self.wav = str(self.temp / (job_id + '_16k.wav'))
        self.raw = [] if empty else [SimpleNamespace(index=11, text='RAW', warnings=())]
        self.translated = [SimpleNamespace(index=22, text='TRANSLATED', warnings=())]
        self.sanitized = [SimpleNamespace(index=33, text='SANITIZED', warnings=('REVIEW',) if warned else ())]
        self.report = {'synthetic_qc': job_id}
        self.clock_values = iter([100, 110, 115, 120, 127, 140])
        logger = logging.Logger('isolated-orchestration')
        logger.addHandler(logging.NullHandler())
        self.ns = dict(__name__='isolated_orchestration_subject', asyncio=asyncio,
            Dict=Dict, Any=Any, Optional=Optional, Path=MemoryPath,
            logger=logger, time=SimpleNamespace(time=lambda: next(self.clock_values)),
            settings=SimpleNamespace(TEMP_DIR=self.temp, OUTPUT_DIR=self.output),
            gpu_lock=asyncio.Lock(), extract_and_enhance_audio=self.extract,
            transcriber=SimpleNamespace(transcribe=self.transcribe),
            translator=SimpleNamespace(translate=self.translate),
            sanitize_timing_without_drift=self.sanitize,
            build_srt_content=lambda items: self.build('srt', items),
            build_ass_content=lambda items, **kw: self.build('ass', items, **kw),
            build_vtt_content=lambda items: self.build('vtt', items),
            build_qc_report=self.qc_builder, persist_qc_report=self.qc_writer, publish_subtitle_triplet=self.publish_subtitles,
            datetime=SimpleNamespace(now=lambda tz: SimpleNamespace(strftime=lambda fmt:'2026-11-06T00:00:00Z')),
            timezone=SimpleNamespace(utc=None))
        isolated_exec(selected_nodes(source), self.ns)
        self.manager = self.ns['JobManager']()
        self.manager.create_job(job_id, self.video.name)
        self.ns['job_manager'] = self.manager
        actual_update = self.manager.update_job
        async def update(*args, **kwargs):
            status = kwargs.get('status', args[1] if len(args) > 1 else None)
            pct = kwargs.get('progress', args[2] if len(args) > 2 else None)
            self.events.append(('job', status, pct))
            self.call('update', args, kwargs)
            await actual_update(*args, **kwargs)
        self.manager.update_job = update
        actual_worker = self.ns['run_exclusive_worker']
        self.actual_worker = actual_worker
        async def observed_actual_worker(coro):
            self.events.append(('worker-enter',))
            try:
                result = await actual_worker(coro)
            except asyncio.CancelledError as exc:
                # Observe and re-raise unchanged at the actual worker boundary.
                # The worker may intentionally emit a new cancellation after
                # draining its shielded task; do not normalize it in this spy.
                self.worker_cancellations.append(exc)
                raise
            self.events.append(('worker-exit',))
            return result
        self.ns['run_exclusive_worker'] = observed_actual_worker

    def publish_subtitles(self, *args, **kwargs):
        self.call('publish_subtitle_triplet', args, kwargs)
        output_dir = kwargs.get('output_dir', args[0] if len(args) > 0 else self.output)
        write_dir = output_dir if output_dir is not None else self.output
        job_id = kwargs.get('job_id', args[1] if len(args) > 1 else self.job_id)
        srt_content = kwargs.get('srt_content', args[2] if len(args) > 2 else '')
        ass_content = kwargs.get('ass_content', args[3] if len(args) > 3 else '')
        vtt_content = kwargs.get('vtt_content', args[4] if len(args) > 4 else '')
        (write_dir / f"{job_id}.srt").write_text(srt_content, encoding="utf-8-sig")
        (write_dir / f"{job_id}.ass").write_text(ass_content, encoding="utf-8-sig")
        (write_dir / f"{job_id}.vtt").write_text(vtt_content, encoding="utf-8")
        return {
            'srt': output_dir / f"{job_id}.srt",
            'ass': output_dir / f"{job_id}.ass",
            'vtt': output_dir / f"{job_id}.vtt"
        }

    def call(self, name, args, kwargs):
        self.calls.setdefault(name, []).append((args, kwargs))

    def fault(self, name):
        if self.failure == name:
            raise self.error

    async def extract(self, *args, **kwargs):
        self.call('extract', args, kwargs)
        self.events.append(('extract',))
        if self.partial:
            self.files[str(args[1])] = 'SYNTHETIC_WAV_MARKER_NOT_MEDIA'
        self.fault('extract')

    async def transcribe(self, *args, **kwargs):
        self.call('transcribe', args, kwargs)
        self.events.append(('transcribe',))
        self.fault('transcribe')
        return self.raw, self.detected, 3.5

    async def translate(self, *args, **kwargs):
        self.call('translate', args, kwargs)
        self.events.append(('translate',))
        self.fault('translate')
        if self.progress:
            for pct in (0, 40, 100):
                kwargs['progress_cb'](pct)
                # Controlled event-loop yields; no wall-clock delay/thread.
                await asyncio.sleep(0)
        return self.translated

    def sanitize(self, *args, **kwargs):
        self.call('sanitize', args, kwargs)
        self.events.append(('sanitize',))
        return self.sanitized

    def build(self, kind, *args, **kwargs):
        self.call('build-' + kind, args, kwargs)
        self.events.append(('build-' + kind,))
        return kind.upper() + '-CONTENT-' + self.job_id

    def qc_builder(self, *args, **kwargs):
        self.call('qc-builder', args, kwargs)
        self.events.append(('qc-builder',))
        return self.report

    def qc_writer(self, *args, **kwargs):
        self.call('qc-writer', args, kwargs)
        self.events.append(('qc-writer',))
        return 'written'

    async def invoke(self):
        self.live_orphan_tasks = []
        try:
            await self.ns['run_cinematic_pipeline'](self.job_id, self.video,
                transcription_provider=self.stt_provider,
                translation_provider=self.translation_provider, language=self.requested)
            # Drain fixture-controlled callback tasks before terminal assertions.
            await asyncio.sleep(0)
        finally:
            current = asyncio.current_task()
            controller_tasks = getattr(self, 'known_controller_tasks', set())
            self.live_orphan_tasks = [t for t in asyncio.all_tasks() if t is not current and t not in controller_tasks and not t.done()]
            if self.live_orphan_tasks:
                raise AssertionError('zero orphan tasks before loop shutdown')

    def run(self):
        asyncio.run(self.invoke(), debug=False)
        return self

    @property
    def job(self):
        return self.manager.jobs[self.job_id]


class TestPipelineIntegration(unittest.TestCase):
    """Twelve incremental orchestrator contracts; local terminal state implemented under cancellation."""
    def setUp(self):
        # 1. Structural preflight check on CancelledError handler before any fixture executes:
        nodes = selected_nodes(PIPELINE_SOURCE)
        fn = next(n for n in nodes if n.name == 'run_cinematic_pipeline')
        cancel_handler = next(h for h in fn.body[-1].body[0].handlers if _handler_type_name(h) == 'CancelledError')
        check_cancellation_handler_structural(cancel_handler)

        stack = ExitStack()
        self.addCleanup(stack.close)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'core','services','dotenv','torch','transformers',
                    'whisper','faster_whisper','requests','httpx','urllib','socket','google','groq'}:
                denied()
            return original_import(name, *args, **kwargs)
        stack.enter_context(patch.object(builtins,'__import__',guarded_import))
        for obj, attr in [(os,'getenv'),(subprocess,'run'),(subprocess,'Popen'),
                          (threading.Thread,'start'),(asyncio,'to_thread')]:
            stack.enter_context(patch.object(obj,attr,denied))
        for obj, attr in [(builtins,'open'),(io,'open')]:
            original = getattr(obj,attr)
            def guarded_open(path,*args,_original=original,**kwargs):
                if isinstance(path,(str,bytes,os.PathLike)):
                    name=os.fsdecode(path).replace('\\','/').lower().split('/')[-1]
                    if name=='.env' or name.startswith('.env.') or name in {'credentials','secrets.json'}:
                        denied()
                return _original(path,*args,**kwargs)
            stack.enter_context(patch.object(obj,attr,guarded_open))

    def assert_called_once(self, f, name):
        self.assertEqual(len(f.calls.get(name,[])),1,'call count '+name)
        return f.calls[name][0]

    def assert_forwarding(self, f):
        args,kw=self.assert_called_once(f,'extract')
        self.assertEqual(len(args),2,'extraction positional contract')
        self.assertIs(args[0],f.video)
        self.assertEqual(str(args[1]),f.wav)
        self.assertEqual(kw,{'enhance':True})
        args,kw=self.assert_called_once(f,'transcribe')
        self.assertEqual([str(p) for p in args],[f.wav])
        self.assertEqual(kw,{'provider':f.stt_provider,'language':f.requested},'transcriber forwarding')
        args,kw=self.assert_called_once(f,'translate')
        self.assertEqual(len(args),1)
        self.assertIs(args[0],f.raw,'translation input identity')
        self.assertEqual(set(kw),{'source_language','provider','progress_cb'})
        self.assertEqual(kw['source_language'],f.detected,'detected language forwarding')
        self.assertEqual(kw['provider'],f.translation_provider,'translation provider forwarding')
        self.assertTrue(callable(kw['progress_cb']))
        self.assertEqual([e[0] for e in f.events].count('worker-enter'),2)
        self.assertEqual([e[0] for e in f.events].count('worker-exit'),2)

    def assert_cleanup(self, f, present=True):
        self.assertEqual([e for e in f.events if e[0]=='exists'],[('exists',f.wav,present)],'cleanup existence guard')
        self.assertEqual([e for e in f.events if e[0]=='unlink'],
            [('unlink',f.wav,{'missing_ok':True})] if present else [],'ordinary-error cleanup contract')
        self.assertNotIn(f.wav,f.files,'ordinary-error cleanup contract')

    def assert_failed(self, f, text, reached):
        self.assertEqual(f.job['status'],'failed','failure terminal status')
        self.assertEqual(f.job['progress'],0)
        self.assertEqual(f.job['error'],text)
        self.assertEqual(f.job['stage'],'حدث خطأ: '+text)
        self.assertEqual(f.job['rejected_blocks'],[])
        self.assertEqual({name for name in f.calls if name!='update'},set(reached),'downstream gating')
        self.assertEqual([e for e in f.events if e[0]=='job'][-1],('job','failed',0))
        self.assertFalse(any(e[0]=='job' and e[1].startswith('completed') for e in f.events))
        self.assertFalse(any(p.startswith(str(f.output)+'/') for p in f.files))
        for name in reached:self.assert_called_once(f,name)

    def assert_cancellation_contract(self, f, caught_exception, expected_exception,
                                     initial_metadata_snapshot=None, boundary_type='orchestrator',
                                     label='cancellation contract'):
        if boundary_type == 'worker':
            self.assertEqual(len(f.worker_cancellations), 1, f"{label}: worker cancellations must be recorded")
            self.assertIs(caught_exception, f.worker_cancellations[0], f"{label}: caught exception must match worker cancellation")
        else:
            self.assertIs(caught_exception, expected_exception, f"{label}: caught exception must match expected cancellation")
        self.assertEqual(f.job['status'], 'failed', f"{label}: status must be failed")
        self.assertEqual(f.job['progress'], 0, f"{label}: progress must be 0")
        self.assertEqual(f.job['stage'], 'تم إلغاء المعالجة', f"{label}: stage must match terminal string")
        self.assertEqual(f.job['error'], 'تم إلغاء العملية أثناء التشغيل', f"{label}: error must match terminal string")
        if initial_metadata_snapshot is not None:
            for k, v in initial_metadata_snapshot.items():
                self.assertIn(k, f.job, f"{label}: preexisting key {k} must be preserved")
                self.assertEqual(f.job[k], v, f"{label}: preexisting value for {k} must be preserved")
        self.assertFalse(f.ns['gpu_lock'].locked(), f"{label}: gpu_lock must be released")
        self.assertFalse(any(p.startswith(str(f.output) + '/') for p in f.files), f"{label}: no output files written")
        self.assertEqual([e for e in f.events if e[0] == 'unlink'], [], f"{label}: zero unlink operations")
        self.assertIn(f.wav, f.files, f"{label}: temporary WAV must be retained")
        if hasattr(f, 'live_orphan_tasks'):
            self.assertEqual(f.live_orphan_tasks, [], f"{label}: zero orphan tasks before loop shutdown")

    def capture_stored_mapping_at_extraction(self, f):
        """Snapshot underlying storage after allowed processing updates, before cancellation."""
        captured = {}
        original_extract = f.ns['extract_and_enhance_audio']
        async def capture_then_extract(*args, **kwargs):
            captured['mapping'] = copy.deepcopy(dict(f.manager.jobs))
            return await original_extract(*args, **kwargs)
        f.ns['extract_and_enhance_audio'] = capture_then_extract
        return captured

    def assert_success(self, f):
        self.assert_forwarding(f)
        for pct,stage in [(10,'extract'),(25,'transcribe'),(55,'translate'),(90,'sanitize')]:
            self.assertLess(f.events.index(('job','processing',pct)),
                            f.events.index((stage,)),'orchestration stage order')
        self.assertIs(self.assert_called_once(f,'sanitize')[0][0],f.translated,'sanitizer input identity')
        for kind in ('srt','ass','vtt'):
            args,kw=self.assert_called_once(f,'build-'+kind)
            self.assertEqual(len(args),1)
            self.assertIs(args[0],f.sanitized,'export input identity')
            self.assertEqual(kw,{'title':f.video.stem} if kind=='ass' else {})
        pub_args, pub_kw = self.assert_called_once(f, 'publish_subtitle_triplet')
        self.assertEqual(pub_args, (), 'publication forwarding')
        self.assertEqual(pub_kw, {
            'output_dir': f.output,
            'job_id': f.job_id,
            'srt_content': 'SRT-CONTENT-' + f.job_id,
            'ass_content': 'ASS-CONTENT-' + f.job_id,
            'vtt_content': 'VTT-CONTENT-' + f.job_id,
        }, 'publication forwarding')
        sequence=[e[0] for e in f.events if e[0] not in {'job','exists'}]
        self.assertEqual(sequence,['extract','worker-enter','transcribe','worker-exit',
            'worker-enter','translate','worker-exit','sanitize','build-srt','build-ass','build-vtt',
            'write','write','write','unlink','qc-builder','qc-writer'],'orchestration event order')
        self.assertEqual([e for e in f.events if e[0]=='write'],[
            ('write',str(f.output/(f.job_id+'.'+kind)),kind.upper()+'-CONTENT-'+f.job_id,
             {'encoding':'utf-8' if kind=='vtt' else 'utf-8-sig'}) for kind in ('srt','ass','vtt')])
        self.assert_cleanup(f)
        args,kw=self.assert_called_once(f,'qc-builder')
        self.assertEqual(args,())
        self.assertEqual(set(kw),{'job_id','filename','pipeline_type','segments','rejected_blocks','created_at'})
        self.assertEqual({k:v for k,v in kw.items() if k!='segments'},dict(job_id=f.job_id,
            filename=f.video.name,pipeline_type='cinematic',rejected_blocks=[],created_at='2026-11-06T00:00:00Z'))
        self.assertIs(kw['segments'],f.sanitized)
        args,kw=self.assert_called_once(f,'qc-writer')
        self.assertEqual(kw,{})
        self.assertIs(args[0],f.output);self.assertEqual(args[1],f.job_id);self.assertIs(args[2],f.report)
        self.assertEqual(f.job['status'],'completed_with_warnings' if f.warned else 'completed','success terminal status')
        self.assertEqual(f.job['progress'],100)
        self.assertEqual(f.job['segment_count'],len(f.sanitized))
        self.assertEqual(f.job['warnings_count'],int(f.warned))
        self.assertEqual(f.job['qc_report_status'],'written')
        self.assertEqual(f.job['qc_report_file'],f.job_id+'.qc.json')
        self.assertEqual(f.job['rejected_blocks'],[])
        self.assertEqual([f.job[k] for k in ('whisper_time','translation_time','total_time')],[5,7,40])
        self.assertEqual(f.job['srt_path'],'/api/download/'+f.job_id+'/srt')
        self.assertEqual(f.job['ass_path'],'/api/download/'+f.job_id+'/ass')
        self.assertEqual(f.job['vtt_path'],'/api/subtitles/'+f.job_id+'/vtt')
        self.assertEqual(f.job['video_url'],'/api/video/'+f.job_id)
        self.assertEqual(f.job['id'],f.job_id)
        self.assertEqual(f.job['filename'],f.video.name)
        self.assertIsNone(f.job['error'])
        self.assertEqual(f.job['stage'],'اكتملت مع 1 تنبيه تدقيق' if f.warned else 'اكتملت الترجمة بنجاح!')
        self.assertNotIn('qc_report_path',f.job)
        self.assertEqual(f.events[-1],('job',f.job['status'],100),'completion must follow writes and QC')

    def test_01_restricted_loading_and_positive_isolation_controls(self):
        self.assertEqual({n.name for n in selected_nodes(PIPELINE_SOURCE)},SYMBOLS)
        for name in ('core.config','services.pipeline','dotenv','torch','transformers','requests'):
            with self.assertRaisesRegex(AssertionError,'ISOLATION:'):__import__(name)
        attempts=[lambda:os.getenv('NEVER_READ'),lambda:open('.env'),lambda:Path('.env').read_text(),
            lambda:subprocess.run(['NEVER_EXECUTE']),lambda:subprocess.Popen(['NEVER_EXECUTE']),
            lambda:threading.Thread(target=lambda:None).start(),lambda:asyncio.to_thread(lambda:None)]
        for attempt in attempts:
            with self.assertRaisesRegex(AssertionError,'ISOLATION:'):attempt()
        for code in ('import os','import core.config','open(".env")','load_dotenv()'):
            with self.assertRaises((AssertionError,NameError)):isolated_exec(ast.parse(code).body,{})
        f=Fixture().run()
        self.assert_success(f)
        self.assertNotIn('os',f.ns)
        self.assertNotIn('load_dotenv',f.ns)

    def test_02_success_exact_trace_arguments_and_writes(self):
        f=Fixture(job_id='clean-a').run()
        self.assert_success(f)
        self.assertEqual([e for e in f.events if e[0]=='job'],[
            ('job','processing',10),('job','processing',25),('job','processing',55),
            ('job','processing',90),('job','completed',100)])

    def test_03_warned_completion_distinct_cue_data_flow(self):
        f=Fixture(job_id='warn-b',warned=True).run()
        self.assertIsNot(f.raw,f.translated);self.assertIsNot(f.translated,f.sanitized)
        self.assertEqual([x[0].text for x in (f.raw,f.translated,f.sanitized)],['RAW','TRANSLATED','SANITIZED'])
        self.assert_success(f)
        self.assertNotIn(('job','completed',100),f.events)

    def test_04_nondefault_providers_and_requested_detected_language(self):
        f=Fixture(job_id='providers-c')
        f.stt_provider,f.translation_provider='alternate-stt','alternate-translate'
        f.requested,f.detected='de','ja'
        f.run();self.assert_success(f)

    def test_05_extraction_timeout_boundary_failure(self):
        cause=subprocess.TimeoutExpired('SYNTHETIC',45)
        error=RuntimeError('Audio extraction timed out after 45.0s.')
        error.__cause__=cause
        f=Fixture(failure='extract',error=error).run()
        self.assert_failed(f,str(error),{'extract'});self.assert_cleanup(f)
        self.assertIs(error.__cause__,cause)

    def test_06_extraction_oserror_present_and_absent_fake_partial(self):
        for present in (True,False):
            with self.subTest(present=present):
                f=Fixture(failure='extract',error=OSError('synthetic binary failure'),partial=present).run()
                self.assert_failed(f,'synthetic binary failure',{'extract'})
                self.assert_cleanup(f,present)

    def test_07_transcriber_error_gates_downstream_and_cleans_fake_audio(self):
        f=Fixture(failure='transcribe',error=RuntimeError('synthetic STT failure')).run()
        self.assert_failed(f,'synthetic STT failure',{'extract','transcribe'});self.assert_cleanup(f)

    def test_08_empty_transcript_is_failure_not_completion(self):
        f=Fixture(empty=True).run()
        self.assert_failed(f,'لم يتم العثور على حوار واضح في الملف الصوتي',{'extract','transcribe'})
        self.assert_cleanup(f)

    def test_09_translator_error_preserves_raw_and_gates_export(self):
        f=Fixture(failure='translate',error=RuntimeError('synthetic translation failure')).run()
        self.assert_failed(f,'synthetic translation failure',{'extract','transcribe','translate'})
        self.assert_cleanup(f)
        self.assertIs(f.calls['translate'][0][0][0],f.raw)
        self.assertEqual(f.raw[0].text,'RAW')

    def test_10_cancellation_propagates_local_terminal_state_without_notification_or_deletion(self):
        # 1. Base phase boundaries: extract, transcribe, translate with deep independent snapshots
        for boundary, reached in [('extract', {'extract'}),
                                  ('transcribe', {'extract', 'transcribe'}),
                                  ('translate', {'extract', 'transcribe', 'translate'})]:
            with self.subTest(boundary=boundary):
                error = asyncio.CancelledError('synthetic ' + boundary)
                f = Fixture(failure=boundary, error=error)
                f.job['preserved_custom_key'] = {'synthetic': {'nested_deep': 'keep'}}
                f.job['extra_id'] = 9999
                f.job['user_flag'] = True
                initial_meta = copy.deepcopy({k: v for k, v in f.job.items() if k not in {'status', 'progress', 'stage', 'error'}})
                with self.assertRaises(asyncio.CancelledError) as caught:
                    f.run()
                b_type = 'orchestrator' if boundary == 'extract' else 'worker'
                self.assert_cancellation_contract(f, caught.exception, error,
                                                 initial_metadata_snapshot=initial_meta,
                                                 boundary_type=b_type,
                                                 label=f'base boundary {boundary}')
                self.assertEqual({n for n in f.calls if n != 'update'}, reached)

        # 2. Broken async updater injection (assert not invoked on cancellation path)
        with self.subTest(scenario='broken_async_updater'):
            f = Fixture(failure='extract', error=asyncio.CancelledError('cancel-updater'))
            f.job['preserved_meta'] = {'nested': 'updater-test'}
            initial_meta = copy.deepcopy({k: v for k, v in f.job.items() if k not in {'status', 'progress', 'stage', 'error'}})
            attempted_failed_updates = []
            previous_update = f.manager.update_job
            async def recording_update(*args, **kw):
                if kw.get('status') == 'failed':
                    attempted_failed_updates.append((args, kw))
                    raise AssertionError('forbidden updater call on cancellation path')
                return await previous_update(*args, **kw)
            f.manager.update_job = recording_update
            with self.assertRaises(asyncio.CancelledError) as caught:
                f.run()
            self.assertEqual(attempted_failed_updates, [], 'cancellation path must never attempt updater calls')
            self.assert_cancellation_contract(f, caught.exception, f.error,
                                             initial_metadata_snapshot=initial_meta,
                                             label='broken updater scenario')

        # 3. Broken logging injection (assert not invoked on cancellation path)
        with self.subTest(scenario='broken_logging'):
            f = Fixture(failure='extract', error=asyncio.CancelledError('cancel-log'))
            f.job['preserved_meta'] = {'nested': 'logging-test'}
            initial_meta = copy.deepcopy({k: v for k, v in f.job.items() if k not in {'status', 'progress', 'stage', 'error'}})
            attempted_cancellation_logs = []
            def record_and_raise_log(msg, *args, **kw):
                attempted_cancellation_logs.append(msg)
                raise AssertionError('forbidden logger call on cancellation path')
            f.ns['logger'].warning = record_and_raise_log
            f.ns['logger'].error = record_and_raise_log
            with self.assertRaises(asyncio.CancelledError) as caught:
                f.run()
            self.assertEqual(attempted_cancellation_logs, [], 'cancellation path must never invoke logger')
            self.assert_cancellation_contract(f, caught.exception, f.error,
                                             initial_metadata_snapshot=initial_meta,
                                             label='broken logging scenario')

        # 4. Uncooperative failed listener (assert no notification attempted and active loop release)
        with self.subTest(scenario='uncooperative_listener'):
            f = Fixture(failure='extract', error=asyncio.CancelledError('cancel-listener'))
            f.job['preserved_meta'] = {'nested': 'listener-test'}
            initial_meta = copy.deepcopy({k: v for k, v in f.job.items() if k not in {'status', 'progress', 'stage', 'error'}})
            class RecordingListener:
                def __init__(self):
                    self.failed_sends = []
                    self.release_event = asyncio.Event()
                    self.failed_send_entered = asyncio.Event()
                async def send_json(self, msg):
                    if msg.get('status') == 'failed':
                        self.failed_sends.append(msg)
                        self.failed_send_entered.set()
                        await self.release_event.wait()
            listener = RecordingListener()
            f.manager.listeners[f.job_id] = [listener]
            async def run_listener_control():
                entered_task = asyncio.create_task(listener.failed_send_entered.wait())
                f.known_controller_tasks = {asyncio.current_task(), entered_task}
                pipeline_task = asyncio.create_task(f.invoke())
                try:
                    # Detect an unintended send without a timeout or waiting for a never-entered callback.
                    await asyncio.wait({pipeline_task, entered_task}, return_when=asyncio.FIRST_COMPLETED)
                    self.assertEqual(listener.failed_sends, [], 'cancellation path must never dispatch failed notification to listener')
                    with self.assertRaises(asyncio.CancelledError) as caught:
                        await pipeline_task
                    self.assert_cancellation_contract(f, caught.exception, f.error,
                                                     initial_metadata_snapshot=initial_meta,
                                                     label='uncooperative listener scenario')
                finally:
                    listener.release_event.set()
                    entered_task.cancel()
                    await asyncio.gather(pipeline_task, entered_task, return_exceptions=True)
                self.assertTrue(pipeline_task.done() and entered_task.done(), 'listener control tasks joined in active loop')
            asyncio.run(run_listener_control())

        # 5. Missing job, non-dict record, storage mapping get failure, other-job isolation, and raw_segments rejected_blocks
        with self.subTest(scenario='missing_and_corrupt_records'):
            # 5a. Missing job:
            f_miss = Fixture(failure='extract', error=asyncio.CancelledError('cancel-missing'))
            other_job_1 = f_miss.manager.create_job('other-job-123', 'other.mp4')
            other_snapshot_1 = copy.deepcopy(other_job_1)
            f_miss.manager.jobs.pop(f_miss.job_id)
            stored_before_missing = self.capture_stored_mapping_at_extraction(f_miss)
            with self.assertRaises(asyncio.CancelledError) as caught_miss:
                f_miss.run()
            self.assertIs(caught_miss.exception, f_miss.error)
            self.assertNotIn(f_miss.job_id, f_miss.manager.jobs, 'missing job must not be invented')
            self.assertEqual(f_miss.manager.jobs['other-job-123'], other_snapshot_1, 'other jobs must not be mutated')
            self.assertFalse(f_miss.ns['gpu_lock'].locked())
            self.assertEqual(dict(f_miss.manager.jobs), stored_before_missing['mapping'], 'missing-job underlying storage unchanged after boundary')

            # 5b. Non-plain/malformed record:
            f_corrupt = Fixture(failure='extract', error=asyncio.CancelledError('cancel-corrupt'))
            other_job_2 = f_corrupt.manager.create_job('other-job-456', 'other456.mp4')
            other_snapshot_2 = copy.deepcopy(other_job_2)
            class JobsMapping(dict):
                def get(self, k, default=None):
                    if k == f_corrupt.job_id:
                        return ['not-a-dict']
                    return super().get(k, default)
            f_corrupt.manager.jobs = JobsMapping(f_corrupt.manager.jobs)
            stored_before_corrupt = self.capture_stored_mapping_at_extraction(f_corrupt)
            with self.assertRaises(asyncio.CancelledError) as caught_corrupt:
                f_corrupt.run()
            self.assertIs(caught_corrupt.exception, f_corrupt.error)
            self.assertIn(f_corrupt.job_id, f_corrupt.manager.jobs, 'target record must remain in stored mapping')
            self.assertEqual(f_corrupt.manager.jobs['other-job-456'], other_snapshot_2, 'other jobs must not be mutated')
            self.assertFalse(f_corrupt.ns['gpu_lock'].locked())
            self.assertEqual(dict(f_corrupt.manager.jobs), stored_before_corrupt['mapping'], 'malformed-get underlying storage unchanged after boundary')

            # 5c. Storage mapping.get failure with other job isolation:
            f_err = Fixture(failure='extract', error=asyncio.CancelledError('cancel-err'))
            other_job_3 = f_err.manager.create_job('other-job-789', 'other789.mp4')
            other_snapshot_3 = copy.deepcopy(other_job_3)
            class FailingJobs(dict):
                def get(self, k, default=None):
                    raise RuntimeError('synthetic storage failure')
            f_err.manager.jobs = FailingJobs(f_err.manager.jobs)
            stored_before_error = self.capture_stored_mapping_at_extraction(f_err)
            with self.assertRaises(asyncio.CancelledError) as caught_err:
                f_err.run()
            self.assertIs(caught_err.exception, f_err.error)
            self.assertEqual(f_err.manager.jobs['other-job-789'], other_snapshot_3, 'other jobs must not be mutated on storage get failure')
            self.assertFalse(f_err.ns['gpu_lock'].locked())
            self.assertEqual(dict(f_err.manager.jobs), stored_before_error['mapping'], 'get-failure underlying storage unchanged after boundary')

            # 5d. Hostile raw_segments rejected_blocks getter:
            f_hostile = Fixture(failure='translate', error=asyncio.CancelledError('cancel-translate-hostile'))
            class HostileRawSegments(list):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.getter_invoked = False
                @property
                def rejected_blocks(self):
                    self.getter_invoked = True
                    raise AssertionError('hostile rejected_blocks getter must not be invoked on cancellation path')
            hostile_raw = HostileRawSegments([SimpleNamespace(index=1, text='RAW', warnings=())])
            f_hostile.raw = hostile_raw
            initial_meta_hostile = copy.deepcopy({k: v for k, v in f_hostile.job.items() if k not in {'status', 'progress', 'stage', 'error'}})
            with self.assertRaises(asyncio.CancelledError) as caught_hostile:
                f_hostile.run()
            self.assertFalse(hostile_raw.getter_invoked, 'rejected_blocks getter must never be accessed on cancellation path')
            self.assert_cancellation_contract(f_hostile, caught_hostile.exception, f_hostile.error,
                                             initial_metadata_snapshot=initial_meta_hostile,
                                             boundary_type='worker',
                                             label='hostile raw segments scenario')

            # 5e. Baseline missing-job update_job compatibility:
            async def run_update_compat():
                await f_miss.manager.update_job('nonexistent-id', status='processing', progress=10, stage='test')
            asyncio.run(run_update_compat())

        # 6. Repeated cancellation while actual exclusive worker drains, with guaranteed cleanup on failure
        with self.subTest(scenario='repeated_worker_cancellation'):
            async def run_repeated_cancellation(inject_failure=False):
                shield_state = {'count': 0}
                child_entered = asyncio.Event()
                child_release = asyncio.Event()
                first_drain_reentry = asyncio.Event()
                second_drain_reentry = asyncio.Event()

                orig_shield = asyncio.shield
                def recording_shield(arg):
                    shield_state['count'] += 1
                    res = orig_shield(arg)
                    if shield_state['count'] == 2:
                        first_drain_reentry.set()
                    elif shield_state['count'] == 3:
                        second_drain_reentry.set()
                    return res

                f = Fixture(job_id='rep-worker')
                f.known_controller_tasks = {asyncio.current_task()}
                f.job['preserved_meta'] = {'nested': 'repeated-cancel-worker'}
                initial_meta = copy.deepcopy({k: v for k, v in f.job.items() if k not in {'status', 'progress', 'stage', 'error'}})

                async def controlled_transcribe(*args, **kwargs):
                    child_entered.set()
                    await child_release.wait()
                    return f.raw, f.detected, 3.5

                f.ns['transcriber'].transcribe = controlled_transcribe
                pipeline_task = None

                with patch('asyncio.shield', side_effect=recording_shield):
                    try:
                        pipeline_task = asyncio.create_task(f.invoke())
                        await child_entered.wait()
                        
                        if inject_failure:
                            self.assertEqual(1, 2, "injected assertion failure after child entry")

                        self.assertTrue(f.ns['gpu_lock'].locked(), 'gpu_lock must be held by running worker')
                        self.assertFalse(pipeline_task.done(), 'worker task must remain pending')
                        
                        # 1. Issue first cancellation
                        pipeline_task.cancel()
                        await first_drain_reentry.wait()
                        
                        self.assertTrue(f.ns['gpu_lock'].locked(), 'gpu_lock must remain held during worker drain')
                        self.assertFalse(pipeline_task.done(), 'worker task must remain pending during drain')
                        
                        # 2. Issue second cancellation at distinct event loop boundary
                        pipeline_task.cancel()
                        await second_drain_reentry.wait()
                        
                        self.assertTrue(f.ns['gpu_lock'].locked(), 'gpu_lock must remain held during second drain')
                        self.assertFalse(pipeline_task.done(), 'worker task must remain pending during second drain')
                        
                        # 3. Release child
                        child_release.set()
                        
                        with self.assertRaises(asyncio.CancelledError) as caught:
                            await pipeline_task
                        
                        self.assertEqual(len(f.worker_cancellations), 1)
                        self.assertIs(caught.exception, f.worker_cancellations[0])
                        self.assertFalse(f.ns['gpu_lock'].locked(), 'gpu_lock must be released after completion')
                        self.assert_cancellation_contract(f, caught.exception, f.worker_cancellations[0],
                                                         initial_metadata_snapshot=initial_meta,
                                                         boundary_type='worker',
                                                         label='repeated cancellation worker drain')
                    finally:
                        child_release.set()
                        if pipeline_task and not pipeline_task.done():
                            try:
                                await pipeline_task
                            except (asyncio.CancelledError, Exception):
                                pass

            # Positive run:
            asyncio.run(run_repeated_cancellation(inject_failure=False))

            # Failure path run proving finally releases tasks without swallowing failure:
            with self.assertRaisesRegex(AssertionError, 'injected assertion failure after child entry'):
                asyncio.run(run_repeated_cancellation(inject_failure=True))

        # 7. Nested metadata mutation rejecting control
        with self.subTest(scenario='nested_metadata_mutation_rejecting_control'):
            f_mut = Fixture(failure='extract', error=asyncio.CancelledError('cancel-nested'))
            f_mut.job['preserved_custom_key'] = {'nested': {'deep': 'value'}}
            initial_meta = copy.deepcopy({k: v for k, v in f_mut.job.items() if k not in {'status', 'progress', 'stage', 'error'}})
            with self.assertRaises(asyncio.CancelledError) as caught:
                f_mut.run()
            # Mutate nested metadata in place:
            f_mut.job['preserved_custom_key']['nested']['deep'] = 'MUTATED'
            with self.assertRaisesRegex(AssertionError, 'preexisting value for preserved_custom_key must be preserved'):
                self.assert_cancellation_contract(f_mut, caught.exception, f_mut.error,
                                                 initial_metadata_snapshot=initial_meta,
                                                 label='nested mutation control')

        # 8. Orphan task detection rejecting control
        with self.subTest(scenario='orphan_task_detection_rejecting_control'):
            unreleased_event = asyncio.Event()
            f_orphan = Fixture(failure='extract', error=asyncio.CancelledError('cancel-orphan'))
            synthetic_task = None
            orig_extract = f_orphan.extract
            async def inject_orphan_during_extract(*args, **kwargs):
                nonlocal synthetic_task
                synthetic_task = asyncio.create_task(unreleased_event.wait())
                return await orig_extract(*args, **kwargs)
            f_orphan.ns['extract_and_enhance_audio'] = inject_orphan_during_extract
            async def run_orphan_rejecting_control():
                try:
                    with self.assertRaisesRegex(AssertionError, '^zero orphan tasks before loop shutdown$'):
                        await f_orphan.invoke()
                    self.assertIn(synthetic_task, f_orphan.live_orphan_tasks)
                    self.assertFalse(synthetic_task.done(), 'orphan detected while still pending in active loop')
                finally:
                    unreleased_event.set()
                    if synthetic_task:
                        await asyncio.gather(synthetic_task, return_exceptions=True)
                self.assertTrue(synthetic_task.done(), 'orphan control task joined before loop shutdown')
            asyncio.run(run_orphan_rejecting_control())

    def test_11_progress_order_deterministic_timings_and_job_isolation(self):
        fixtures=[Fixture(job_id='progress-d',progress=True).run(),Fixture(job_id='separate-e').run()]
        for f in fixtures:self.assert_success(f)
        f=fixtures[0]
        self.assertEqual([e[2] for e in f.events if e[0]=='job'],[10,25,55,55,67,85,90,100])
        export=f.events.index(('job','processing',90))
        self.assertTrue(all(f.events.index(('job','processing',p))<export for p in (67,85)))
        self.assertTrue(set(fixtures[0].files).isdisjoint(fixtures[1].files))
        self.assertIsNot(fixtures[0].manager.jobs,fixtures[1].manager.jobs)

    def test_12_actual_source_mutants_fail_intended_contract_assertions(self):
        self.assert_success(Fixture().run())
        mutations = [
            ('source_language=detected_lang', 'source_language=language', 'detected language forwarding'),
            ('provider=transcription_provider', 'provider=translation_provider', 'transcriber forwarding'),
            ('provider=translation_provider', 'provider=transcription_provider', 'translation provider forwarding'),
            ('sanitize_timing_without_drift(translated_segments)', 'sanitize_timing_without_drift(raw_segments)', 'sanitizer input identity'),
            ('                    raw_segments,', '                    [],', 'translation input identity'),
            ('"processing", 25,', '"processing", 26,', 'processing'),
            ('output_dir=settings.OUTPUT_DIR,', 'output_dir=None,', 'publication forwarding'),
        ]
        for old, new, label in mutations:
            self.assertIn(old, PIPELINE_SOURCE)
            with self.subTest(mutation=label):
                f = Fixture(source=PIPELINE_SOURCE.replace(old, new, 1)).run()
                with self.assertRaisesRegex(AssertionError, label):
                    if label == 'processing':
                        self.assertEqual([e for e in f.events if e[0] == 'job'], [
                            ('job', 'processing', 10), ('job', 'processing', 25), ('job', 'processing', 55),
                            ('job', 'processing', 90), ('job', 'completed', 100)], 'processing stage ordering')
                    else:
                        self.assert_success(f)

        # 1. Structural preflight check on CancelledError handler in production source:
        nodes = selected_nodes(PIPELINE_SOURCE)
        fn = next(n for n in nodes if n.name == 'run_cinematic_pipeline')
        cancel_handler = next(h for h in fn.body[-1].body[0].handlers if _handler_type_name(h) == 'CancelledError')
        check_cancellation_handler_structural(cancel_handler)

        # 2. Structural preflight rejects added await, task creation, and scheduling mutants before runtime:
        structural_mutants = [
            ('await asyncio.sleep(0)', 'forbidden async/generator control flow: Await'),
            ('yield None', 'forbidden async/generator control flow: Yield'),
            ('asyncio.create_task(asyncio.sleep(0))', 'forbidden call in cancellation handler: asyncio.create_task'),
            ('asyncio.ensure_future(asyncio.sleep(0))', 'forbidden call in cancellation handler: asyncio.ensure_future'),
            ('loop.call_soon(lambda: None)', 'forbidden call in cancellation handler: loop.call_soon'),
            ('loop.call_later(0, lambda: None)', 'forbidden call in cancellation handler: loop.call_later'),
            ('logger.warning("synthetic forbidden cancellation logging")', 'forbidden call in cancellation handler: logger.warning'),
        ]
        for bad_stmt, expected_err in structural_mutants:
            with self.subTest(structural_check=bad_stmt):
                m_nodes = selected_nodes(PIPELINE_SOURCE)
                m_fn = next(n for n in m_nodes if n.name == 'run_cinematic_pipeline')
                m_ch = next(h for h in m_fn.body[-1].body[0].handlers if _handler_type_name(h) == 'CancelledError')
                m_ch.body[0].body.insert(0, ast.parse(bad_stmt).body[0])
                with self.assertRaisesRegex(AssertionError, re.escape(expected_err)):
                    check_cancellation_handler_structural(m_ch)

        # 2b. Sentinel control: prove actual-source forbidden task/await is rejected with zero fixture runtime entry:
        fixture_entered = False
        def sentinel_fixture(*args, **kwargs):
            nonlocal fixture_entered
            fixture_entered = True
            raise AssertionError("sentinel fixture should never be reached if preflight rejects")
        bad_source = PIPELINE_SOURCE.replace('            except Exception:\n                pass\n            raise original_cancel',
            '                asyncio.create_task(asyncio.sleep(0))\n            except Exception:\n                pass\n            raise original_cancel')
        bad_nodes = selected_nodes(bad_source)
        bad_fn = next(n for n in bad_nodes if n.name == 'run_cinematic_pipeline')
        bad_ch = next(h for h in bad_fn.body[-1].body[0].handlers if _handler_type_name(h) == 'CancelledError')
        with self.assertRaisesRegex(AssertionError, 'forbidden call in cancellation handler: asyncio.create_task'):
            check_cancellation_handler_structural(bad_ch)
            sentinel_fixture()
        self.assertFalse(fixture_entered, "preflight must reject before fixture entry")

        # 3. Real AST error-path variants; validation code above is shared unchanged:
        for kind, label in [('early-success', 'failure terminal status'),
                            ('cleanup-omitted', 'ordinary-error cleanup contract'),
                            ('downstream-after-error', 'downstream gating'),
                            ('stage-order', 'orchestration stage order')]:
            nodes = selected_nodes(PIPELINE_SOURCE)
            fn = next(n for n in nodes if n.name == 'run_cinematic_pipeline')
            handler = next(h for h in fn.body[-1].body[0].handlers if _handler_type_name(h) == 'Exception')
            if kind == 'early-success':
                for n in ast.walk(handler):
                    if isinstance(n, ast.keyword) and n.arg == 'status':
                        n.value = ast.Constant('completed')
            elif kind == 'cleanup-omitted':
                cleanup = next(n for n in handler.body if isinstance(n, ast.If))
                cleanup.body = [ast.Pass()]
            elif kind == 'downstream-after-error':
                handler.body.insert(0, ast.parse('sanitize_timing_without_drift([])').body[0])
            else:
                statements = fn.body[-1].body[0].body
                statements[0], statements[1] = statements[1], statements[0]
            mutant_tree = ast.Module(body=nodes, type_ignores=[])
            ast.fix_missing_locations(mutant_tree)
            f = Fixture(source=ast.unparse(mutant_tree), failure=None if kind == 'stage-order' else 'extract',
                        error=RuntimeError('synthetic error')).run()
            with self.subTest(mutation=kind), self.assertRaisesRegex(AssertionError, label):
                if kind == 'stage-order':
                    self.assert_success(f)
                else:
                    self.assert_failed(f, 'synthetic error', {'extract'})
                    self.assert_cleanup(f)

        # 4. 5 Genuine Cancellation Behavioral mutants matching specific assertion reason:
        cancel_mutations = [
            ('cancel-omitted-commit', r'status must be failed'),
            ('cancel-wrong-status', r'status must be failed'),
            ('cancel-omitted-reraise', r'handler must re-raise original_cancel'),
            ('cancel-wrong-identity', r'caught exception must match expected cancellation'),
            ('cancel-added-unlink', r'zero unlink operations'),
        ]
        for kind, reason_pattern in cancel_mutations:
            nodes = selected_nodes(PIPELINE_SOURCE)
            fn = next(n for n in nodes if n.name == 'run_cinematic_pipeline')
            ch = next(h for h in fn.body[-1].body[0].handlers if _handler_type_name(h) == 'CancelledError')
            if kind == 'cancel-omitted-commit':
                ch.body[0] = ast.Pass()
            elif kind == 'cancel-wrong-status':
                for n in ast.walk(ch):
                    if isinstance(n, ast.Constant) and n.value == 'failed':
                        n.value = 'completed'
            elif kind == 'cancel-omitted-reraise':
                ch.body = [n for n in ch.body if not isinstance(n, ast.Raise)]
            elif kind == 'cancel-wrong-identity':
                ch.body[-1] = ast.Raise(exc=ast.Call(
                    func=ast.Attribute(value=ast.Name(id='asyncio', ctx=ast.Load()), attr='CancelledError', ctx=ast.Load()),
                    args=[ast.Constant('WRONG_IDENTITY')], keywords=[]
                ))
            elif kind == 'cancel-added-unlink':
                ch.body.insert(0, ast.Expr(value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id='temp_wav', ctx=ast.Load()), attr='unlink', ctx=ast.Load()),
                    args=[], keywords=[]
                )))
            mutant_tree = ast.Module(body=nodes, type_ignores=[])
            ast.fix_missing_locations(mutant_tree)
            mutant_source = ast.unparse(mutant_tree)
            error = asyncio.CancelledError('synthetic-extract-cancel')
            f = Fixture(source=mutant_source, failure='extract', error=error)
            with self.subTest(cancel_mutation=kind):
                if kind == 'cancel-omitted-reraise':
                    with self.assertRaisesRegex(AssertionError, reason_pattern):
                        try:
                            f.run()
                            raised = False
                        except asyncio.CancelledError:
                            raised = True
                        self.assertTrue(raised, 'cancellation propagation: handler must re-raise original_cancel')
                else:
                    with self.assertRaises(asyncio.CancelledError) as caught:
                        f.run()
                    with self.assertRaisesRegex(AssertionError, reason_pattern):
                        self.assert_cancellation_contract(f, caught.exception, error, label='cancellation test')

        # 5. Wrong-reason control: verify identity failure cannot pass status assertion
        with self.subTest(control='wrong_reason_rejection'):
            fake_f = Fixture()
            fake_f.job['status'] = 'failed'
            fake_f.job['progress'] = 0
            fake_f.job['stage'] = 'تم إلغاء المعالجة'
            fake_f.job['error'] = 'تم إلغاء العملية أثناء التشغيل'
            fake_err = asyncio.CancelledError('wrong-id')
            expected_err = asyncio.CancelledError('expected-id')
            with self.assertRaises(AssertionError) as caught_assert:
                self.assert_cancellation_contract(fake_f, fake_err, expected_err, label='cancelled job terminal status')
            self.assertIn('caught exception must match expected cancellation', str(caught_assert.exception))
            self.assertNotIn('status must be failed', str(caught_assert.exception))


if __name__ == '__main__':
    unittest.main()
