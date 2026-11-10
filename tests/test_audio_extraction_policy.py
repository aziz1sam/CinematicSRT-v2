"""AIX-AUDIO-RELIABILITY-001 v3-T2: Isolated, Source-Backed Acceptance Tests.

Pure AST extraction:
- No runtime execution of core/config.py (blocks dotenv, real .env, real os.environ).
- No runtime execution of services/audio.py through importlib.
- Pure AST extraction of _parse_ffmpeg_timeout, Settings.FFMPEG_TIMEOUT_SECONDS,
  and extract_and_enhance_audio with injected fake environment and fake settings.
- Real AST mutations in test_17 proving precise failure on timeout/stdin omissions.
- Fixed portable baseline AST oracle and exact-addition verification in test_18.
"""
import ast
import asyncio
import copy
import builtins
from contextlib import ExitStack
import io
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Optional
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "core" / "config.py").exists():
    raise FileNotFoundError(f"Expected repository root with core/config.py at {ROOT}")

CONFIG_PATH = ROOT / "core" / "config.py"
AUDIO_PATH = ROOT / "services" / "audio.py"

def denied_operation(*args, **kwargs):
    raise AssertionError("ISOLATION: forbidden external operation")


def isolated_exec(mod, filename, namespace):
    # No module import, file open, real environment or config loader is supplied.
    safe_builtins = dict(vars(builtins), __import__=denied_operation,
                         open=denied_operation)
    namespace["__builtins__"] = safe_builtins
    exec(compile(mod, filename, "exec"), namespace)

# ---------------------------------------------------------------------------
# AST Extraction of config helper and Settings.FFMPEG_TIMEOUT_SECONDS
# ---------------------------------------------------------------------------
config_source = CONFIG_PATH.read_text(encoding="utf-8-sig")
config_tree = ast.parse(config_source, filename=str(CONFIG_PATH))

# Find _parse_ffmpeg_timeout function node
helper_node = next(
    n for n in config_tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_parse_ffmpeg_timeout"
)

# Find Settings.FFMPEG_TIMEOUT_SECONDS assignment node
settings_node = next(
    c for c in config_tree.body
    if isinstance(c, ast.ClassDef) and c.name == "Settings"
)
setting_node = next(
    n for n in settings_node.body
    if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "FFMPEG_TIMEOUT_SECONDS"
)

# Extract and compile helper into isolated namespace
fake_environ = {}
fake_os = SimpleNamespace(
    environ=fake_environ,
    getenv=lambda k, default=None: fake_environ.get(k, default)
)
helper_namespace = {"math": math, "os": fake_os}
isolated_exec(ast.Module(body=[helper_node], type_ignores=[]), str(CONFIG_PATH), helper_namespace)
_parse_ffmpeg_timeout = helper_namespace["_parse_ffmpeg_timeout"]

# Helper to evaluate Settings.FFMPEG_TIMEOUT_SECONDS with custom fake env
def evaluate_settings_timeout(env_dict):
    ns_os = SimpleNamespace(environ=env_dict, getenv=lambda k, default=None: env_dict.get(k, default))
    ns = {"math": math, "os": ns_os, "_parse_ffmpeg_timeout": _parse_ffmpeg_timeout}
    class_def = ast.ClassDef(
        name="TestSettings",
        bases=[],
        keywords=[],
        body=[copy.deepcopy(setting_node)],
        decorator_list=[]
    )
    mod = ast.Module(body=[class_def], type_ignores=[])
    ast.fix_missing_locations(mod)
    isolated_exec(mod, str(CONFIG_PATH), ns)
    return getattr(ns["TestSettings"], "FFMPEG_TIMEOUT_SECONDS")

# ---------------------------------------------------------------------------
# AST Extraction of services/audio.py: extract_and_enhance_audio
# ---------------------------------------------------------------------------
audio_source = AUDIO_PATH.read_text(encoding="utf-8-sig")
audio_tree = ast.parse(audio_source, filename=str(AUDIO_PATH))

audio_func_node = next(
    n for n in audio_tree.body
    if isinstance(n, ast.AsyncFunctionDef) and n.name == "extract_and_enhance_audio"
)

# Default fake settings injected into audio namespace
default_settings = SimpleNamespace(
    FFMPEG_BINARY="ffmpeg",
    FFMPEG_TIMEOUT_SECONDS=300.0
)

def compile_audio_function(func_node, custom_settings=None):
    st = custom_settings or default_settings
    ns = {
        "asyncio": asyncio,
        "logging": logging,
        "logger": logging.getLogger("test.audio"),
        "math": math,
        "subprocess": subprocess,
        "Path": Path,
        "Optional": Optional,
        "settings": st,
    }
    mod = ast.Module(body=[copy.deepcopy(func_node)], type_ignores=[])
    ast.fix_missing_locations(mod)
    isolated_exec(mod, str(AUDIO_PATH), ns)
    return ns["extract_and_enhance_audio"]

extract_and_enhance_audio = compile_audio_function(audio_func_node)


# ---------------------------------------------------------------------------
# Test Suite Class
# ---------------------------------------------------------------------------
# Fixed v3 preimages, verified from audio.py.bak (90A884B5...) and
# config.py.bak (D1BF016F...). Parsed by this runtime, never read from AIX.
LEGACY_AUDIO_SOURCE = r'''
"""Audio Processing Service (FFmpeg 16kHz Mono Normalization)"""
import asyncio
import logging
import subprocess
from pathlib import Path
from core.config import settings

logger = logging.getLogger(__name__)

async def extract_and_enhance_audio(video_path: Path, output_wav_path: Path, enhance: bool = True) -> Path:
    """Extracts audio from video file to 16kHz mono WAV with acoustic normalization."""
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = settings.FFMPEG_BINARY
    
    # Filter: Highpass/Lowpass + Dynamic Compressor + Loudnorm to bring out quiet dialogue
    if enhance:
        audio_filter = (
            "highpass=f=80,"
            "lowpass=f=8000,"
            "acompressor=threshold=-24dB:ratio=3:attack=5:release=120:makeup=6,"
            "loudnorm=I=-16:LRA=11:TP=-1.5"
        )
    else:
        audio_filter = "anull"

    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-af", audio_filter,
        str(output_wav_path)
    ]

    logger.info("Extracting audio via %s: %s -> %s", ffmpeg_bin, video_path.name, output_wav_path.name)
    
    def _run_ffmpeg():
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore"
        )

    result = await asyncio.to_thread(_run_ffmpeg)
    if result.returncode != 0 or not output_wav_path.exists():
        logger.warning("FFmpeg enhancement failed, attempting plain 16k extraction...")
        # Fallback without audio filters
        fallback_cmd = [
            ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            str(output_wav_path)
        ]
        result_fb = await asyncio.to_thread(lambda: subprocess.run(fallback_cmd, capture_output=True))
        if result_fb.returncode != 0:
            err_msg = result_fb.stderr.decode("utf-8", errors="ignore")[-400:] if result_fb.stderr else "Unknown error"
            raise RuntimeError(f"Audio extraction failed with {ffmpeg_bin}: {err_msg}")

    return output_wav_path

'''
LEGACY_CONFIG_SOURCE = r'''
"""CinematicSRT v2.0 - Central Configuration"""
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load local environment variables if present
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

class Settings:
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    TEMP_DIR: Path = BASE_DIR / "temp"

    # Whisper Settings (Default: CUDA on GTX 1650 with float16)
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "medium")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

    # Transcription Engine: "local" (faster-whisper) or "groq" (Cloud API ~5s)
    TRANSCRIPTION_PROVIDER: str = os.getenv("TRANSCRIPTION_PROVIDER", "local")

    # Translation Engine: "local" (NLLB-200), "gemini" (Google Gemini API), "groq" (Llama 3.3), "qwen" (Local LLM)
    TRANSLATION_PROVIDER: str = os.getenv("TRANSLATION_PROVIDER", "local")

    # API Keys for Cloud Providers
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LOCAL_QWEN_URL: str = os.getenv("LOCAL_QWEN_URL", "http://127.0.0.1:8080/v1")

    # Subtitle Display Constraints
    MAX_CHARS_PER_LINE: int = 42
    MAX_LINES_PER_SEGMENT: int = 2
    MIN_SEGMENT_DURATION: float = 0.8
    MAX_SEGMENT_DURATION: float = 7.0
    MIN_GAP_BETWEEN_SEGMENTS: float = 0.05

    # File Upload Limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "4000"))
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def FFMPEG_BINARY(self) -> str:
        """Resolves FFmpeg executable in order: local bin/ffmpeg.exe -> PATH -> 'ffmpeg'."""
        portable = self.BASE_DIR / "bin" / "ffmpeg.exe"
        if portable.exists():
            return str(portable)
        system_bin = shutil.which("ffmpeg")
        if system_bin:
            return system_bin
        return "ffmpeg"

    @classmethod
    def save_env_key(cls, key: str, value: str):
        """Persist key into .env file with input sanitization (prevent newline injection)."""
        clean_key = key.strip().replace("\n", "").replace("\r", "")
        clean_val = value.strip().replace("\n", "").replace("\r", "")
        os.environ[clean_key] = clean_val
        setattr(cls, clean_key, clean_val)

        lines = []
        if ENV_PATH.exists():
            lines = ENV_PATH.read_text(encoding="utf-8").splitlines()

        found = False
        new_lines = []
        for line in lines:
            if line.startswith(f"{clean_key}="):
                new_lines.append(f"{clean_key}={clean_val}")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"{clean_key}={clean_val}")

        ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

settings = Settings()

'''


APPROVED_PARSER = """
def _parse_ffmpeg_timeout(raw_val) -> float:
    if not isinstance(raw_val, str):
        return 300.0
    try:
        value = float(raw_val.strip())
    except (ValueError, TypeError, OverflowError):
        return 300.0
    if not math.isfinite(value) or not 0.0 < value <= 86400.0:
        return 300.0
    return value
"""
APPROVED_SETTING = 'FFMPEG_TIMEOUT_SECONDS: float = _parse_ffmpeg_timeout(os.getenv("FFMPEG_TIMEOUT_SECONDS"))'
APPROVED_VALIDATION = """
if timeout is None:
    timeout = settings.FFMPEG_TIMEOUT_SECONDS
if type(timeout) not in (int, float):
    raise TypeError("FFmpeg timeout must be a built-in int or float.")
try:
    timeout_val = float(timeout)
except (ValueError, OverflowError) as exc:
    raise ValueError("FFmpeg timeout must be finite and in (0, 86400].") from exc
if not math.isfinite(timeout_val) or not 0.0 < timeout_val <= 86400.0:
    raise ValueError("FFmpeg timeout must be finite and in (0, 86400].")
"""
APPROVED_HANDLER = """
try:
    pass
except subprocess.TimeoutExpired as exc:
    raise RuntimeError(f"Audio extraction timed out after {timeout_val}s.") from exc
"""


def ast_dump(value):
    if isinstance(value, list):
        return [ast_dump(n) for n in value]
    return ast.dump(value, include_attributes=False)


def check_invariance(tc, config, audio):
    """Assert exact permitted additions FIRST, then compare fixed legacy ASTs."""
    config = copy.deepcopy(config)
    audio = copy.deepcopy(audio)

    def equal(actual, expected, label):
        tc.assertEqual(ast_dump(actual), ast_dump(expected), label)

    def strip_import(tree, statement, label):
        expected = ast.parse(statement).body[0]
        matches = [n for n in tree.body if ast_dump(n) == ast_dump(expected)]
        tc.assertEqual(len(matches), 1, label)
        tree.body.remove(matches[0])

    strip_import(config, "import math", "config math import")
    parsers = [n for n in config.body if isinstance(n, ast.FunctionDef)
               and n.name == "_parse_ffmpeg_timeout"]
    tc.assertEqual(len(parsers), 1, "parser count")
    equal(parsers[0], ast.parse(APPROVED_PARSER).body[0], "approved parser")
    config.body.remove(parsers[0])
    settings = next(n for n in config.body if isinstance(n, ast.ClassDef)
                    and n.name == "Settings")
    assignments = [n for n in settings.body if isinstance(n, ast.AnnAssign)
                   and getattr(n.target, "id", None) == "FFMPEG_TIMEOUT_SECONDS"]
    tc.assertEqual(len(assignments), 1, "setting count")
    equal(assignments[0], ast.parse(APPROVED_SETTING).body[0], "approved setting")
    settings.body.remove(assignments[0])
    equal(config, ast.parse(LEGACY_CONFIG_SOURCE), "legacy config")

    strip_import(audio, "import math", "audio math import")
    strip_import(audio, "from typing import Optional", "audio Optional import")
    fn = next(n for n in audio.body if isinstance(n, ast.AsyncFunctionDef)
              and n.name == "extract_and_enhance_audio")
    legacy_fn = next(n for n in ast.parse(LEGACY_AUDIO_SOURCE).body
                     if isinstance(n, ast.AsyncFunctionDef))
    expected_args = copy.deepcopy(legacy_fn.args)
    expected_args.args.append(ast.parse("def f(timeout: Optional[float] = None): pass").body[0].args.args[0])
    expected_args.defaults.append(ast.Constant(None))
    equal(fn.args, expected_args, "approved timeout signature")
    equal(fn.body[1:5], ast.parse(APPROVED_VALIDATION).body, "approved validation")
    fn.args.args.pop()
    fn.args.defaults.pop()
    del fn.body[1:5]

    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "subprocess" and n.func.attr == "run"]
    tc.assertEqual(len(calls), 2, "process call count")
    expected_new = ast.parse("run(stdin=subprocess.DEVNULL, timeout=timeout_val)").body[0].value.keywords
    for index, call in enumerate(calls):
        added = [k for k in call.keywords if k.arg in {"stdin", "timeout"}]
        equal(added, expected_new, "approved process keywords " + str(index))
        call.keywords = [k for k in call.keywords if k.arg not in {"stdin", "timeout"}]

    wrappers = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    tc.assertEqual(len(wrappers), 2, "timeout wrapper count")
    handler = ast.parse(APPROVED_HANDLER).body[0].handlers[0]
    for index, wrapper in enumerate(wrappers):
        equal(wrapper.handlers, [handler], "approved timeout handler " + str(index))
        tc.assertEqual(len(wrapper.body), 1, "timeout body count")
        tc.assertEqual(wrapper.orelse, [], "timeout else")
        tc.assertEqual(wrapper.finalbody, [], "timeout finally")

    class Unwrap(ast.NodeTransformer):
        def visit_Try(self, node):
            self.generic_visit(node)
            return node.body
    Unwrap().visit(fn)
    equal(audio, ast.parse(LEGACY_AUDIO_SOURCE), "legacy audio")


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


def primary_kwargs(timeout):
    return dict(stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, errors="ignore", timeout=float(timeout))


def fallback_kwargs(timeout):
    return dict(stdin=subprocess.DEVNULL, capture_output=True, timeout=float(timeout))


def expected_argv(enhance=True, fallback=False):
    cmd = ["ffmpeg", "-y", "-i", "video.mp4", "-vn", "-ac", "1", "-ar", "16000"]
    if not fallback:
        cmd += ["-af", ("highpass=f=80,lowpass=f=8000,acompressor=threshold=-24dB:ratio=3:attack=5:release=120:makeup=6,loudnorm=I=-16:LRA=11:TP=-1.5"
                       if enhance else "anull")]
    return cmd + ["audio.wav"]


class TestAudioExtractionPolicy(unittest.TestCase):
    """18 top-level isolated acceptance unit tests for AIX-AUDIO-RELIABILITY-001 v3-T2."""

    def setUp(self):
        self.worker_calls = 0
        guards = ExitStack()
        self.addCleanup(guards.close)
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'core', 'services', 'dotenv', 'torch',
                    'transformers', 'whisper', 'faster_whisper', 'requests',
                    'httpx', 'urllib', 'socket'}:
                denied_operation()
            return original_import(name, *args, **kwargs)
        guards.enter_context(patch.object(builtins, '__import__', guarded_import))
        for obj, attr in [(os, 'getenv'), (subprocess, 'run'),
                          (subprocess, 'Popen'), (asyncio, 'to_thread'),
                          (threading.Thread, 'start')]:
            guards.enter_context(patch.object(obj, attr, denied_operation))
        for obj, attr in [(builtins, 'open'), (io, 'open')]:
            original = getattr(obj, attr)
            def guarded_open(path, *args, _original=original, **kwargs):
                if isinstance(path, (str, bytes, os.PathLike)):
                    name = os.fsdecode(path).replace('\\', '/').lower().split('/')[-1]
                    if name == '.env' or name.startswith('.env.') or name in {'credentials', 'secrets.json'}:
                        denied_operation()
                return _original(path, *args, **kwargs)
            guards.enter_context(patch.object(obj, attr, guarded_open))

    def assert_invalid(self, value, error, runtime_setting=False, fn_node=None):
        self.worker_calls = 0
        path, out = MagicMock(spec=Path), MagicMock(spec=Path)
        settings = SimpleNamespace(FFMPEG_BINARY='ffmpeg', FFMPEG_TIMEOUT_SECONDS=value)
        fn = compile_audio_function(fn_node or audio_func_node, settings)
        with patch.object(asyncio, 'to_thread', self.inline_worker), patch.object(subprocess, 'run') as run:
            run.return_value = subprocess.CompletedProcess(['fake'], 0, stdout='', stderr='')
            with self.assertRaises(error) as caught:
                asyncio.run(fn(path, out, timeout=None if runtime_setting else value))
            expected = ('FFmpeg timeout must be a built-in int or float.' if error is TypeError
                        else 'FFmpeg timeout must be finite and in (0, 86400].')
            self.assertEqual(str(caught.exception), expected)
            out.parent.mkdir.assert_not_called()
            out.unlink.assert_not_called()
            run.assert_not_called()
            self.assertEqual(self.worker_calls, 0)

    def assert_case(self, outcomes, enhance=True, timeout=60, exists=True,
                    expected_error=None, message=None, cause=None, fn=None):
        path, out = MagicMock(spec=Path), MagicMock(spec=Path)
        path.__str__.return_value, out.__str__.return_value = 'video.mp4', 'audio.wav'
        out.exists.return_value = exists
        calls = []
        self.worker_calls = 0
        def run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            self.assertLessEqual(len(calls), len(outcomes), 'unexpected retry')
            outcome = outcomes[len(calls)-1]
            if isinstance(outcome, BaseException):
                raise outcome
            return subprocess.CompletedProcess(cmd, outcome[0], stdout=b'', stderr=outcome[1])
        with patch.object(asyncio, 'to_thread', self.inline_worker), patch.object(subprocess, 'run', side_effect=run):
            if expected_error is not None:
                with self.assertRaises(type(expected_error)) as caught:
                    asyncio.run((fn or extract_and_enhance_audio)(path, out, enhance=enhance, timeout=timeout))
                if isinstance(expected_error, (OSError, asyncio.CancelledError)):
                    self.assertIs(caught.exception, expected_error)
                if message is not None:
                    self.assertEqual(str(caught.exception), message)
                if cause is not None:
                    self.assertIs(caught.exception.__cause__, cause)
            else:
                self.assertIs(asyncio.run((fn or extract_and_enhance_audio)(path, out, enhance=enhance, timeout=timeout)), out)
        normalized = (fn or extract_and_enhance_audio).__globals__['settings'].FFMPEG_TIMEOUT_SECONDS if timeout is None else float(timeout)
        expected = [(expected_argv(enhance), primary_kwargs(normalized))]
        if len(outcomes) == 2:
            expected.append((expected_argv(fallback=True), fallback_kwargs(normalized)))
        self.assertEqual(calls, expected)
        self.assertEqual(self.worker_calls, len(outcomes))
        out.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        for method in ('unlink', 'rmdir', 'rename', 'replace'):
            getattr(out, method).assert_not_called()
        return calls

    async def inline_worker(self, fn, *args, **kwargs):
        self.worker_calls += 1
        return fn(*args, **kwargs)

    def test_01_parser_valid_strings_bounds_and_86400(self):
        """1. Parser valid strings/whitespace, positive bounds and 86400."""
        self.assertEqual(_parse_ffmpeg_timeout("300"), 300.0)
        self.assertEqual(_parse_ffmpeg_timeout("  120.5  "), 120.5)
        self.assertEqual(_parse_ffmpeg_timeout("0.1"), 0.1)
        self.assertEqual(_parse_ffmpeg_timeout("1"), 1.0)
        self.assertEqual(_parse_ffmpeg_timeout("86400"), 86400.0)
        self.assertEqual(_parse_ffmpeg_timeout("86400.0"), 86400.0)

    def test_02_parser_fallback_grid_and_settings_assignment(self):
        """2. Parser missing/blank/malformed/zero/negative/NaN/infinities/overflow fallback grid & direct Settings assignment."""
        fallbacks = [
            None, "", "   ", "\t\n", "abc", "300s",
            "0", "0.0", "-1", "-300.0",
            "nan", "NaN", "inf", "+inf", "-inf",
            "1e400", "86400.0001", "100000",
            123, 300.0, [], {},
        ]
        for val in fallbacks:
            self.assertEqual(_parse_ffmpeg_timeout(val), 300.0, f"Failed fallback for: {val!r}")

        # Direct verification of Settings.FFMPEG_TIMEOUT_SECONDS assignment
        # Case A: Default when env key is absent
        self.assertEqual(evaluate_settings_timeout({}), 300.0)
        # Case B: Valid non-default string
        self.assertEqual(evaluate_settings_timeout({"FFMPEG_TIMEOUT_SECONDS": "450.5"}), 450.5)
        # Case C: Invalid non-numeric string falls back to default
        self.assertEqual(evaluate_settings_timeout({"FFMPEG_TIMEOUT_SECONDS": "not-a-number"}), 300.0)
        # Case D: Out of bounds negative number falls back to default
        self.assertEqual(evaluate_settings_timeout({"FFMPEG_TIMEOUT_SECONDS": "-60"}), 300.0)
        # Positive controls prove guards reject real access rather than trusting
        # sys.modules absence (other discovery modules may legitimately be loaded).
        for attempt in [lambda: __import__('core.config'), lambda: __import__('services.audio'),
                        lambda: __import__('dotenv'), lambda: __import__('requests'),
                        lambda: os.getenv('AIX_NEVER_READ'),
                        lambda: open('.env'), lambda: Path('.env').read_text(),
                        lambda: subprocess.run(['NEVER_EXECUTE']),
                        lambda: subprocess.Popen(['NEVER_EXECUTE']),
                        lambda: threading.Thread(target=lambda: None).start(),
                        lambda: asyncio.to_thread(lambda: None)]:
            with self.assertRaisesRegex(AssertionError, 'ISOLATION:'):
                attempt()
        for source in ['import os', 'import core.config', 'open(".env")', 'load_dotenv()']:
            with self.assertRaises((AssertionError, NameError)):
                isolated_exec(ast.parse(source), '<guard-positive-control>', {})
        self.assertNotIn('os', extract_and_enhance_audio.__globals__)

    def test_03_default_timeout_none_resolves_setting_and_passes_to_process(self):
        """3. Default timeout=None resolves setting and passes normalized float to process."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

        fake_path = MagicMock(spec=Path)
        fake_path.name = "test.mp4"
        fake_path.__str__.return_value = "test.mp4"
        fake_out = MagicMock(spec=Path)
        fake_out.name = "test.wav"
        fake_out.__str__.return_value = "test.wav"
        fake_out.parent.mkdir = MagicMock()
        fake_out.exists.return_value = True

        # Test default setting (300.0)
        with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run", side_effect=fake_run):
            result = asyncio.run(extract_and_enhance_audio(fake_path, fake_out, timeout=None))
            self.assertEqual(result, fake_out)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["timeout"], 300.0)
            self.assertIsInstance(calls[0][1]["timeout"], float)

        # Test isolated custom non-default setting (150.0)
        calls.clear()
        custom_st = SimpleNamespace(FFMPEG_BINARY="ffmpeg", FFMPEG_TIMEOUT_SECONDS=150.0)
        fn_custom = compile_audio_function(audio_func_node, custom_st)
        with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run", side_effect=fake_run):
            result = asyncio.run(fn_custom(fake_path, fake_out, timeout=None))
            self.assertEqual(result, fake_out)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1]["timeout"], 150.0)
        self.assert_case([(1, b'primary'), (0, b'')], timeout=None)
        self.assert_case([(1, b'primary'), (0, b'')], timeout=None, fn=fn_custom)

    def test_04_explicit_valid_timeout_floats_and_integers(self):
        """4. Explicit valid timeout arguments: positive float/int pass exact float to process."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

        fake_path = MagicMock(spec=Path)
        fake_path.name = "test.mp4"
        fake_path.__str__.return_value = "test.mp4"
        fake_out = MagicMock(spec=Path)
        fake_out.name = "test.wav"
        fake_out.__str__.return_value = "test.wav"
        fake_out.parent.mkdir = MagicMock()
        fake_out.exists.return_value = True

        for explicit_val, expected_float in [(60, 60.0), (45.5, 45.5), (86400, 86400.0), (0.5, 0.5)]:
            calls.clear()
            with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run", side_effect=fake_run):
                asyncio.run(extract_and_enhance_audio(fake_path, fake_out, timeout=explicit_val))
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][1]["timeout"], expected_float)
                self.assertIsInstance(calls[0][1]["timeout"], float)
            self.assert_case([(1, b'primary'), (0, b'')], timeout=explicit_val)

    def test_05_explicit_zero_and_negative_timeout_raises_value_error(self):
        """5. Explicit timeout <= 0 raises ValueError before mkdir, worker, or process execution."""
        fake_path = MagicMock(spec=Path)
        fake_out = MagicMock(spec=Path)
        fake_out.parent.mkdir = MagicMock()

        for invalid_val in [0, 0.0, -1, -0.001, -300]:
            self.worker_calls = 0
            fake_out.parent.mkdir.reset_mock()
            with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run") as mock_run:
                with self.assertRaises(ValueError) as ctx:
                    asyncio.run(extract_and_enhance_audio(fake_path, fake_out, timeout=invalid_val))
                self.assertIn("(0, 86400]", str(ctx.exception))
                fake_out.parent.mkdir.assert_not_called()
                mock_run.assert_not_called()
                self.assertEqual(self.worker_calls, 0, "Worker must not be called on validation failure")

    def test_06_explicit_nan_and_infinities_raise_value_error(self):
        """6. Explicit NaN and infinities raise ValueError before mkdir, worker, or process execution."""
        fake_path = MagicMock(spec=Path)
        fake_out = MagicMock(spec=Path)
        fake_out.parent.mkdir = MagicMock()

        for invalid_val in [float("nan"), float("inf"), float("-inf")]:
            self.worker_calls = 0
            fake_out.parent.mkdir.reset_mock()
            with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run") as mock_run:
                with self.assertRaises(ValueError) as ctx:
                    asyncio.run(extract_and_enhance_audio(fake_path, fake_out, timeout=invalid_val))
                self.assertIn("finite and in (0, 86400]", str(ctx.exception))
                fake_out.parent.mkdir.assert_not_called()
                mock_run.assert_not_called()
                self.assertEqual(self.worker_calls, 0)

    def test_07_explicit_over_max_and_non_numeric_raise_value_or_type_error(self):
        """7. Explicit > 86400 raises ValueError, non-numeric raises TypeError before mkdir, worker, or process execution."""
        fake_path = MagicMock(spec=Path)
        fake_out = MagicMock(spec=Path)
        fake_out.parent.mkdir = MagicMock()

        # > 86400 -> ValueError
        self.worker_calls = 0
        fake_out.parent.mkdir.reset_mock()
        with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run") as mock_run:
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(extract_and_enhance_audio(fake_path, fake_out, timeout=86400.1))
            self.assertIn("(0, 86400]", str(ctx.exception))
            fake_out.parent.mkdir.assert_not_called()
            mock_run.assert_not_called()
            self.assertEqual(self.worker_calls, 0)

        # non-numeric -> TypeError
        for non_num in ["300", [], {}, True, False]:
            self.worker_calls = 0
            fake_out.parent.mkdir.reset_mock()
            with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run") as mock_run:
                with self.assertRaises(TypeError) as ctx:
                    asyncio.run(extract_and_enhance_audio(fake_path, fake_out, timeout=non_num))
                self.assertIn("built-in int or float", str(ctx.exception))
                fake_out.parent.mkdir.assert_not_called()
                mock_run.assert_not_called()
                self.assertEqual(self.worker_calls, 0)
        type_cases = [IntSubclass(60), FloatSubclass(45.5), b'300', object(), (300,), complex(3, 0)]
        value_cases = [86400.1, 10**400]
        for value in type_cases:
            self.assert_invalid(value, TypeError)
        for value in value_cases:
            self.assert_invalid(value, ValueError)
        # Separate None -> runtime-settings validation path, never the parser.
        for value in [True, False, '300', IntSubclass(60), FloatSubclass(45.5), b'300']:
            self.assert_invalid(value, TypeError, runtime_setting=True)
        for value in [0, -1, float('nan'), float('inf'), float('-inf'), 86400.1, 10**400]:
            self.assert_invalid(value, ValueError, runtime_setting=True)

    def test_08_primary_process_stdin_and_timeout_propagation(self):
        """8. Primary process invocation propagates exact command list, stdin=DEVNULL and timeout float."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

        fake_path = MagicMock(spec=Path)
        fake_path.__str__.return_value = "video.mp4"
        fake_out = MagicMock(spec=Path)
        fake_out.__str__.return_value = "audio.wav"
        fake_out.parent.mkdir = MagicMock()
        fake_out.exists.return_value = True

        with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run", side_effect=fake_run):
            asyncio.run(extract_and_enhance_audio(fake_path, fake_out, enhance=True, timeout=120.0))

        self.assertEqual(len(calls), 1)
        cmd, kwargs = calls[0]
        self.assertEqual(kwargs, primary_kwargs(120))
        self.assertEqual(kwargs.get("stdin"), subprocess.DEVNULL)
        self.assertEqual(kwargs.get("timeout"), 120.0)
        self.assertEqual(kwargs.get("stdout"), subprocess.PIPE)
        self.assertEqual(kwargs.get("stderr"), subprocess.PIPE)
        self.assertIs(kwargs.get("text"), True)

        expected_filter = "highpass=f=80,lowpass=f=8000,acompressor=threshold=-24dB:ratio=3:attack=5:release=120:makeup=6,loudnorm=I=-16:LRA=11:TP=-1.5"
        expected_cmd = [
            "ffmpeg", "-y",
            "-i", "video.mp4",
            "-vn", "-ac", "1", "-ar", "16000",
            "-af", expected_filter,
            "audio.wav"
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_09_primary_enhanced_filtergraph_exact_parameters(self):
        """9. Primary enhanced mode exact filtergraph parameter validation."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

        fake_path = MagicMock(spec=Path)
        fake_path.__str__.return_value = "in.mp4"
        fake_out = MagicMock(spec=Path)
        fake_out.__str__.return_value = "out.wav"
        fake_out.parent.mkdir = MagicMock()
        fake_out.exists.return_value = True

        with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run", side_effect=fake_run):
            asyncio.run(extract_and_enhance_audio(fake_path, fake_out, enhance=True, timeout=50))

        cmd, _ = calls[0]
        af_idx = cmd.index("-af")
        filter_str = cmd[af_idx + 1]
        expected_filter = "highpass=f=80,lowpass=f=8000,acompressor=threshold=-24dB:ratio=3:attack=5:release=120:makeup=6,loudnorm=I=-16:LRA=11:TP=-1.5"
        self.assertEqual(filter_str, expected_filter)

    def test_10_plain_mode_direct_arguments_stdin_and_timeout(self):
        """10. Plain mode direct invocation exact command list, stdin=DEVNULL and timeout float."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout="out", stderr="")

        fake_path = MagicMock(spec=Path)
        fake_path.__str__.return_value = "video.mp4"
        fake_out = MagicMock(spec=Path)
        fake_out.__str__.return_value = "audio.wav"
        fake_out.parent.mkdir = MagicMock()
        fake_out.exists.return_value = True

        with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run", side_effect=fake_run):
            asyncio.run(extract_and_enhance_audio(fake_path, fake_out, enhance=False, timeout=75.0))

        self.assertEqual(len(calls), 1)
        cmd, kwargs = calls[0]
        self.assertEqual(kwargs, primary_kwargs(75))
        self.assertEqual(kwargs.get("stdin"), subprocess.DEVNULL)
        self.assertEqual(kwargs.get("timeout"), 75.0)
        self.assertEqual(kwargs.get("stdout"), subprocess.PIPE)
        self.assertEqual(kwargs.get("stderr"), subprocess.PIPE)
        self.assertIs(kwargs.get("text"), True)

        expected_cmd = [
            "ffmpeg", "-y",
            "-i", "video.mp4",
            "-vn", "-ac", "1", "-ar", "16000",
            "-af", "anull",
            "audio.wav"
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_11_fallback_triggers_on_failure_and_missing_output(self):
        """11. Both enhanced causes and plain nonzero-primary legacy retry."""
        for enhance, code, exists in [(True, 1, True), (True, 0, False), (False, 1, True)]:
            with self.subTest(enhance=enhance, code=code, exists=exists):
                self.assert_case([(code, b'primary'), (0, b'')], enhance, 90, exists)

    def test_12_both_attempts_fail_preserves_runtime_error_and_last_400_stderr(self):
        """12. Exact bytes decode/last400 and empty stderr fallback, two calls."""
        for stderr in [b'x'*600 + b'\xffIMPORTANT_ERROR_TAIL', b'']:
            tail = stderr.decode('utf-8', errors='ignore')[-400:] if stderr else 'Unknown error'
            self.assert_case([(1, b'primary'), (1, stderr)],
                expected_error=RuntimeError(),
                message='Audio extraction failed with ffmpeg: ' + tail)

    def test_13_os_error_not_caught_as_timeout_expired(self):
        """13. Both OSError boundaries propagate identity with no extra retry."""
        for index in [0, 1]:
            error = FileNotFoundError('ffmpeg not found')
            outcomes = ([(1, b'primary')] if index else []) + [error]
            self.assert_case(outcomes, expected_error=error)

    def test_14_timeout_expired_on_primary_raises_chained_runtime_error(self):
        """14. Enhanced/plain primary timeout: exact message/cause, no retry/cleanup."""
        for enhance in [True, False]:
            error = subprocess.TimeoutExpired('fake-primary', 45)
            self.assert_case([error], enhance=enhance, timeout=45,
                expected_error=RuntimeError(), cause=error,
                message='Audio extraction timed out after 45.0s.')

    def test_15_timeout_expired_on_fallback_raises_chained_runtime_error(self):
        """15. Fallback timeout: exact chain, two calls, zero cleanup."""
        error = subprocess.TimeoutExpired('fake-fallback', 30)
        self.assert_case([(1, b'primary'), error], timeout=30,
            expected_error=RuntimeError(), cause=error,
            message='Audio extraction timed out after 30.0s.')

    def test_16_asyncio_cancelled_error_propagation(self):
        """16. Both cancellation boundaries preserve identity and zero cleanup."""
        for index in [0, 1]:
            error = asyncio.CancelledError('cancelled')
            self.assert_case(([(1, b'primary')] if index else []) + [error],
                             expected_error=error)

    def test_17_real_ast_mutants_verify_timeout_and_stdin_assertions(self):
        """17. Real AST mutations of extract_and_enhance_audio verify that omitting primary/fallback timeout/stdin fails tests."""
        # Verification harness to run against a function implementation
        def run_keyword_verification(fn):
            calls = []
            def fake_run(cmd, **kwargs):
                calls.append((cmd, kwargs))
                if len(calls) == 1:
                    return subprocess.CompletedProcess(cmd, 1, stdout="out", stderr="fail")
                return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

            fake_path = MagicMock(spec=Path)
            fake_out = MagicMock(spec=Path)
            fake_out.parent.mkdir = MagicMock()
            fake_out.exists.return_value = True

            with patch.object(asyncio, "to_thread", self.inline_worker), patch.object(subprocess, "run", side_effect=fake_run):
                asyncio.run(fn(fake_path, fake_out, enhance=True, timeout=99.0))

            if len(calls) != 2:
                raise AssertionError(f"Expected 2 calls, got {len(calls)}")

            # Assert Primary call (call 0)
            if calls[0][1].get("timeout") != 99.0:
                raise AssertionError("Primary call missing timeout=99.0")
            if calls[0][1].get("stdin") != subprocess.DEVNULL:
                raise AssertionError("Primary call missing stdin=subprocess.DEVNULL")

            # Assert Fallback call (call 1)
            if calls[1][1].get("timeout") != 99.0:
                raise AssertionError("Fallback call missing timeout=99.0")
            if calls[1][1].get("stdin") != subprocess.DEVNULL:
                raise AssertionError("Fallback call missing stdin=subprocess.DEVNULL")

        # Step 0: Unmodified extracted function passes cleanly
        run_keyword_verification(extract_and_enhance_audio)

        # Helper to create AST mutant by stripping specified keyword on specified call index
        class RunCallMutator(ast.NodeTransformer):
            def __init__(self, target_call_index, omit_keyword):
                self.target_call_index = target_call_index
                self.omit_keyword = omit_keyword
                self.current_call_index = 0

            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and getattr(node.func.value, "id", None) == "subprocess" and node.func.attr == "run":
                    if self.current_call_index == self.target_call_index:
                        node.keywords = [k for k in node.keywords if k.arg != self.omit_keyword]
                    self.current_call_index += 1
                return node

        def create_ast_mutant(call_idx, keyword):
            mutated_node = copy.deepcopy(audio_func_node)
            transformer = RunCallMutator(call_idx, keyword)
            transformer.visit(mutated_node)
            ast.fix_missing_locations(mutated_node)
            return compile_audio_function(mutated_node)

        # Mutant 1: Primary call (index 0) omits timeout
        fn_mutant_1 = create_ast_mutant(0, "timeout")
        with self.assertRaises(AssertionError) as ctx1:
            run_keyword_verification(fn_mutant_1)
        self.assertIn("Primary call missing timeout=99.0", str(ctx1.exception))

        # Mutant 2: Primary call (index 0) omits stdin
        fn_mutant_2 = create_ast_mutant(0, "stdin")
        with self.assertRaises(AssertionError) as ctx2:
            run_keyword_verification(fn_mutant_2)
        self.assertIn("Primary call missing stdin=subprocess.DEVNULL", str(ctx2.exception))

        # Mutant 3: Fallback call (index 1) omits timeout
        fn_mutant_3 = create_ast_mutant(1, "timeout")
        with self.assertRaises(AssertionError) as ctx3:
            run_keyword_verification(fn_mutant_3)
        self.assertIn("Fallback call missing timeout=99.0", str(ctx3.exception))

        # Mutant 4: Fallback call (index 1) omits stdin
        fn_mutant_4 = create_ast_mutant(1, "stdin")
        with self.assertRaises(AssertionError) as ctx4:
            run_keyword_verification(fn_mutant_4)
        self.assertIn("Fallback call missing stdin=subprocess.DEVNULL", str(ctx4.exception))

    def test_18_source_and_ast_invariance_checks(self):
        """18. Exact additions + fixed legacy oracle; genuine rejecting controls."""
        check_invariance(self, config_tree, audio_tree)

        def reject(config, audio, label):
            with self.assertRaisesRegex(AssertionError, label):
                check_invariance(self, config, audio)

        # Actual source AST variants, never a detached digest inequality.
        for old, new, label in [
            ('if type(timeout) not in (int, float):',
             'if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):',
             'approved validation'),
            ('timeout: Optional[float] = None', 'timeout: Optional[float] = 300', 'approved timeout signature'),
            ('timeout=timeout_val', 'timeout=123', 'approved process keywords'),
            ('loudnorm=I=-16:LRA=11:TP=-1.5', 'loudnorm=I=-16:LRA=99:TP=-1.5', 'legacy audio'),
            ('"acompressor=threshold=-24dB:ratio=3:attack=5:release=120:makeup=6,"', '""', 'legacy audio'),
            ('"-ar", "16000"', '"-ar", "22050"', 'legacy audio'),
            ('return output_wav_path', 'return video_path', 'legacy audio'),
        ]:
            self.assertIn(old, audio_source)
            reject(config_tree, ast.parse(audio_source.replace(old, new)), label)

        for old, new, label in [
            ('return 300.0', 'return 301.0', 'approved parser'),
            ('_parse_ffmpeg_timeout(os.getenv("FFMPEG_TIMEOUT_SECONDS"))',
             '_parse_ffmpeg_timeout("500")', 'approved setting'),
            ('MAX_CHARS_PER_LINE: int = 42', 'MAX_CHARS_PER_LINE: int = 43', 'legacy config'),
        ]:
            self.assertIn(old, config_source)
            reject(ast.parse(config_source.replace(old, new)), audio_tree, label)

        for index in [0, 1]:
            mutant = copy.deepcopy(audio_tree)
            wrappers = [n for n in ast.walk(mutant) if isinstance(n, ast.Try)
                        and any(isinstance(h.type, ast.Attribute)
                                and h.type.attr == 'TimeoutExpired' for h in n.handlers)]
            self.assertEqual(len(wrappers), 2)
            wrappers[index].handlers[0].body.insert(0, ast.parse(
                'output_wav_path.unlink(missing_ok=True)').body[0])
            reject(config_tree, mutant, 'approved timeout handler ' + str(index))

        # This same behavioral assertion also rejects the type-relaxing mutant,
        # independently of the source checker, while bool rejection is retained.
        relaxed = audio_source.replace('if type(timeout) not in (int, float):',
            'if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):')
        relaxed_fn = next(n for n in ast.parse(relaxed).body if isinstance(n, ast.AsyncFunctionDef))
        with self.assertRaisesRegex(AssertionError, 'TypeError not raised'):
            self.assert_invalid(IntSubclass(60), TypeError, fn_node=relaxed_fn)


if __name__ == "__main__":
    unittest.main()
