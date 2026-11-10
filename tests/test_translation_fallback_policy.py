"""Language fallback policy: actual AST methods, wholly synthetic backends.

Stdlib only. Does not import production translator/config/torch/transformers,
download models, contact HTTP services, or write application output files.
Fake token counts and generated text are not inference/quality benchmarks.
"""
import ast
import asyncio
import contextlib
import copy
import json
import logging
import os
import tempfile
import weakref
import re
import sys
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
WARNING = "UNSUPPORTED_LANGUAGE_FALLBACK"


def execute(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "reviewed_ast_fake_dependencies", "exec"), namespace)


class SyntheticBackend:
    """Factories and HTTP clients are spies returning Python fakes only."""
    def __init__(self, text="ترجمة اختبار", status=200):
        self.text, self.status = text, status
        self.calls, self.tokenizers = [], []
        owner = self

        class Inputs(dict):
            def to(self, device):
                owner.calls.append(("inputs_to", device))
                return self

        class Tokenizer:
            src_lang = None
            eos_token_id = 2
            pad_token_id = 1
            decoder_start_token_id = 2

            def convert_tokens_to_ids(self, value):
                return 1

            def __call__(self, texts, **kwargs):
                owner.calls.append(("tokenize", kwargs, list(texts)))
                return {"input_ids": [[1] for _ in texts], "attention_mask": [[1] for _ in texts]}

            def pad(self, encoded, **kwargs):
                owner.calls.append(("pad", kwargs, copy.deepcopy(encoded)))
                return Inputs(**copy.deepcopy(encoded))

            def decode(self, value, **kwargs):
                return owner.text

        class TokenizerFactory:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                owner.calls.append(("fake_tokenizer_factory", args, kwargs))
                tokenizer = Tokenizer()
                owner.tokenizers.append(tokenizer)
                return tokenizer

        class Model:
            generation_config = SimpleNamespace(
                eos_token_id=2, pad_token_id=1, decoder_start_token_id=2
            )

            def to(self, device):
                return self

            def eval(self):
                pass

            def generate(self, **kwargs):
                owner.calls.append(("generate", kwargs))
                arabic_id = kwargs.get("forced_bos_token_id", 1)
                return [[2, arabic_id, 77, 2] for _ in kwargs["input_ids"]]

        class ModelFactory:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                owner.calls.append(("fake_model_factory", args, kwargs))
                return Model()

        class Client:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def post(self, url, **kwargs):
                owner.calls.append(("fake_http", url))
                payload = json.dumps([{"index": 1, "arabic": "نجاح سحابي وهمي"}])
                data = ({"candidates": [{"content": {"parts": [{"text": payload}]}}]}
                        if "googleapis" in url else
                        {"choices": [{"message": {"content": payload}}]})
                return SimpleNamespace(status_code=owner.status, text="fake failure",
                                       json=lambda: data)

        self.transformers = types.ModuleType("transformers")
        self.transformers.AutoTokenizer = TokenizerFactory
        self.transformers.AutoModelForSeq2SeqLM = ModelFactory
        self.main_config = SimpleNamespace(_commit_hash="main-commit")
        self.transformers.AutoConfig = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: self.main_config
        )
        self.transformers.GenerationConfig = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(source="main"),
            from_model_config=lambda config: SimpleNamespace(source=config._commit_hash),
        )
        directory = tempfile.TemporaryDirectory()
        weakref.finalize(self, directory.cleanup)
        self.weight_fixture = Path(directory.name) / "model.safetensors"
        self.weight_fixture.touch()
        self.cache_files = {("main", "model.safetensors"): self.weight_fixture}
        self.hub = types.ModuleType("huggingface_hub")
        self.hub.try_to_load_from_cache = (
            lambda repo, filename, *, revision: self.cache_files.get((revision, filename))
        )
        tree = ast.parse((ROOT / "services/translator.py").read_text(encoding="utf-8-sig"))
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedTranslator")
        methods = [copy.deepcopy(n) for n in klass.body if isinstance(n, ast.AsyncFunctionDef)]
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                   {"resolve_nllb_source_language", "validate_provider_response", "reattach_segment"}]
        mapping = next(n for n in tree.body if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == "NLLB_LANG_MAP" for t in n.targets))
        self.namespace = dict(asyncio=asyncio, copy=copy, json=json, os=os, re=re,
                              SimpleNamespace=SimpleNamespace, List=List, Dict=Dict,
                              Optional=Optional, Any=Any, Tuple=Tuple,
                              NLLB_LANG_MAP=ast.literal_eval(mapping.value),
                              logger=logging.Logger("synthetic-language-policy"),
                              gc=SimpleNamespace(collect=lambda: None),
                              torch=SimpleNamespace(no_grad=contextlib.nullcontext,
                                  cuda=SimpleNamespace(is_available=lambda: False)),
                              httpx=SimpleNamespace(AsyncClient=Client),
                              settings=SimpleNamespace(GEMINI_API_KEY="FAKE_KEY", GROQ_API_KEY="FAKE_KEY",
                                                       LOCAL_QWEN_URL="http://not-contacted.invalid/v1"),
                              CINEMATIC_PROMPT="SYNTHETIC TEST ONLY")
        self.namespace["logger"].addHandler(logging.NullHandler())
        shell = ast.ClassDef(name="IsolatedTranslator", bases=[], keywords=[],
                             body=methods, decorator_list=[])
        execute([*helpers, shell], self.namespace)
        self.translator = self.namespace["IsolatedTranslator"]()
        self.translator.device, self.translator.dtype = "cpu", "FAKE_DTYPE"

    def run(self, awaitable):
        # Even nested `from transformers import ...` resolves only to our spies.
        with patch.dict(sys.modules, {"transformers": self.transformers, "huggingface_hub": self.hub}):
            return asyncio.run(awaitable)

    def local(self, cues, language, **kwargs):
        return self.run(self.translator._translate_nllb_local(cues, language, **kwargs))


def cue(**kwargs):
    fields = dict(index=1, start=1.25, end=2.75, text="Original", style="Blue",
                  speaker="Alice", layer=3, margin_l="10", effect="K0", warnings=())
    fields.update(kwargs)
    return SimpleNamespace(**fields)


class TestTranslationFallbackPolicy(unittest.TestCase):
    def test_resolver_preserves_complete_mapping_and_variants(self):
        fake = SyntheticBackend()
        resolve = fake.namespace["resolve_nllb_source_language"]
        for language, code in fake.namespace["NLLB_LANG_MAP"].items():
            for variant in [language, language.upper() + "-XX", language.upper() + "_xx"]:
                with self.subTest(variant=variant):
                    self.assertEqual(resolve(variant), (code, ()))

    def test_resolver_classifies_unspecified_and_application_unmapped(self):
        resolve = SyntheticBackend().namespace["resolve_nllb_source_language"]
        for language in [None, "", " \t\n"]:
            self.assertEqual(resolve(language), ("eng_Latn", ()))
        for language in ["sw", "XX", "sw-KE", "xx_YY", " fr "]:
            self.assertEqual(resolve(language), ("eng_Latn", (WARNING,)))

    def test_local_effective_code_and_warning(self):
        for language, code, warnings in [("sw", "eng_Latn", (WARNING,)),
                                         ("FR_fr", "fra_Latn", ()),
                                         (None, "eng_Latn", ()), (" \t", "eng_Latn", ())]:
            fake = SyntheticBackend()
            output = fake.local([cue()], language)
            self.assertEqual(fake.tokenizers[-1].src_lang, code)
            self.assertEqual(output[0].warnings, warnings)

    def test_metadata_input_immutability_and_warning_dedup(self):
        fake = SyntheticBackend()
        for warnings in [("UPSTREAM",), ("UPSTREAM", WARNING)]:
            source = cue(warnings=warnings, raw_ass_text=r"{\i1}Original", custom={"preserve": True})
            before = copy.deepcopy(vars(source))
            output = fake.local([source], "xx")[0]
            self.assertIsNot(source, output)
            self.assertEqual(vars(source), before)
            self.assertEqual(output.warnings, ("UPSTREAM", WARNING))
            self.assertEqual({k: v for k, v in vars(output).items() if k not in {"text", "warnings"}},
                             {k: v for k, v in before.items() if k not in {"text", "warnings"}})

    def test_clean_no_warning_attribute_and_empty_decode_are_unchanged(self):
        source = cue()
        del source.warnings
        fake = SyntheticBackend(text="   ")
        clean = fake.local([source], "en-US")[0]
        self.assertEqual(clean.text, "Original")
        self.assertEqual(clean.warnings, ("EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW"))
        warned = fake.local([source], "sw")[0]
        self.assertEqual(warned.text, "Original")
        self.assertEqual(warned.warnings, ("EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW"))
        self.assertEqual(source.text, "Original")
        self.assertFalse(hasattr(source, "warnings"))

    def test_batches_progress_and_existing_generation_limits(self):
        fake = SyntheticBackend()
        progress = []
        output = fake.local([cue(index=i) for i in range(1, 6)], "xx", progress_cb=progress.append)
        self.assertEqual([s.index for s in output], list(range(1, 6)))
        self.assertTrue(all(s.warnings == (WARNING,) for s in output))
        self.assertEqual(progress, [40, 80, 100])
        token_calls = [c for c in fake.calls if c[0] == "tokenize"]
        self.assertEqual([len(c[2]) for c in token_calls], [1, 1, 1, 1, 1])
        self.assertTrue(all(c[1] == {"add_special_tokens": True, "padding": False, "truncation": False,
                                   "return_attention_mask": True, "return_token_type_ids": False} for c in token_calls))
        pad_calls = [c for c in fake.calls if c[0] == "pad"]
        self.assertEqual([len(c[2]["input_ids"]) for c in pad_calls], [2, 2, 1])
        self.assertTrue(all(c[1] == {"padding": True, "return_tensors": "pt"} for c in pad_calls))
        self.assertTrue(all(c[1]["max_new_tokens"] == 140 for c in fake.calls if c[0] == "generate"))

    def test_translate_arabic_bypass_empty_and_local_routing(self):
        fake = SyntheticBackend()
        async def forbidden(*args, **kwargs):
            raise AssertionError("Arabic/empty input must not dispatch")
        for name in ["_translate_gemini", "_translate_groq", "_translate_local_qwen", "_translate_nllb_local"]:
            setattr(fake.translator, name, forbidden)
        source = [cue()]
        for language in ["ar", "ar-EG", "AR_eg"]:
            self.assertIs(fake.run(fake.translator.translate(source, language, "local")), source)
        self.assertEqual(fake.run(fake.translator.translate([], "sw", "gemini")), [])
        self.assertEqual(fake.calls, [])
        local = SyntheticBackend()
        self.assertEqual(local.run(local.translator.translate(source, "sw", "local"))[0].warnings, (WARNING,))

    def test_all_provider_success_routes_have_no_false_local_warning(self):
        for provider in ["gemini", "groq", "qwen"]:
            fake = SyntheticBackend()
            async def forbidden(*args, **kwargs):
                raise AssertionError("Successful provider must not invoke NLLB")
            fake.translator._translate_nllb_local = forbidden
            out = fake.run(fake.translator.translate([cue()], "sw", provider))[0]
            self.assertEqual(out.text, "نجاح سحابي وهمي")
            self.assertNotIn(WARNING, out.warnings)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(fake.calls[0][0], "fake_http")

    def test_all_provider_failure_routes_carry_actual_local_warning(self):
        for provider in ["gemini", "groq", "qwen"]:
            for language, expected in [("sw", (WARNING,)), ("fr", ())]:
                fake = SyntheticBackend(status=503)
                out = fake.run(fake.translator.translate([cue()], language, provider))[0]
                self.assertEqual(out.text, fake.text)
                self.assertEqual(out.warnings, expected)
                self.assertEqual(len(fake.tokenizers), 1)

    def test_missing_cloud_key_keeps_existing_local_recovery(self):
        for provider, key in [("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")]:
            fake = SyntheticBackend()
            setattr(fake.namespace["settings"], key, "")
            out = fake.run(fake.translator.translate([cue()], "sw", provider))[0]
            self.assertEqual(out.warnings, (WARNING,))
            self.assertFalse(any(c[0] == "fake_http" for c in fake.calls))

    def test_actual_memory_pipeline_propagates_local_warning(self):
        for language, status, count in [("sw", "completed_with_warnings", 1), ("fr-FR", "completed", 0)]:
            fake = SyntheticBackend()
            sub_tree = ast.parse((ROOT / "services/subtitles.py").read_text(encoding="utf-8-sig"))
            for node in sub_tree.body:
                if isinstance(node, ast.Import):
                    self.assertTrue(all(a.name == "re" for a in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, {"types", "typing"})
                else:
                    self.assertIsInstance(node, (ast.Expr, ast.FunctionDef))
            namespace = dict(fake.namespace, Path=Path, time=time, gpu_lock=asyncio.Lock())
            execute(sub_tree.body, namespace)
            pt = ast.parse((ROOT / "services/pipeline.py").read_text(encoding="utf-8-sig"))
            execute([n for n in pt.body if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef)) and
                     n.name in {"JobManager", "run_exclusive_worker", "run_subtitle_translation_pipeline"}], namespace)
            writes = {}
            class MemoryOutput:
                def __init__(self, name=""):
                    self.name = name
                def __truediv__(self, name):
                    return MemoryOutput(name)
                def write_text(self, text, **kwargs):
                    writes[self.name] = text
            pub_calls = []
            def publish_subtitle_triplet(output_dir, job_id, srt_content, ass_content, vtt_content):
                pub_calls.append((output_dir, job_id, srt_content, ass_content, vtt_content))
                (output_dir / f"{job_id}.srt").write_text(srt_content, encoding="utf-8-sig")
                (output_dir / f"{job_id}.ass").write_text(ass_content, encoding="utf-8-sig")
                (output_dir / f"{job_id}.vtt").write_text(vtt_content, encoding="utf-8")
                return {"srt": output_dir / f"{job_id}.srt", "ass": output_dir / f"{job_id}.ass", "vtt": output_dir / f"{job_id}.vtt"}
            class Source:
                name, suffix, stem = "memory.srt", ".srt", "memory"
                def read_text(self, **kwargs):
                    return "1\n00:00:01,250 --> 00:00:02,750\nOriginal\n"
            async def translate_without_background_progress(segments, **kwargs):
                kwargs.pop("progress_cb", None)
                return await fake.translator.translate(segments, **kwargs)
            captured_formats = {}
            orig_build_srt = namespace["build_srt_content"]
            orig_build_ass = namespace["build_ass_content"]
            orig_build_vtt = namespace["build_vtt_content"]

            def spy_build_srt(doc):
                res = orig_build_srt(doc)
                captured_formats["srt"] = res
                return res

            def spy_build_ass(doc, **kwargs):
                res = orig_build_ass(doc, **kwargs)
                captured_formats["ass"] = res
                return res

            def spy_build_vtt(doc):
                res = orig_build_vtt(doc)
                captured_formats["vtt"] = res
                return res

            namespace["build_srt_content"] = spy_build_srt
            namespace["build_ass_content"] = spy_build_ass
            namespace["build_vtt_content"] = spy_build_vtt

            namespace.update(settings=SimpleNamespace(OUTPUT_DIR=MemoryOutput()),
                             job_manager=namespace["JobManager"](),
                             translator=SimpleNamespace(translate=translate_without_background_progress),
                             publish_subtitle_triplet=publish_subtitle_triplet)
            namespace["job_manager"].create_job("memory", "memory.srt")
            fake.run(namespace["run_subtitle_translation_pipeline"]("memory", Source(), source_language=language))
            self.assertEqual(len(pub_calls), 1)
            self.assertEqual(pub_calls[0][1], "memory")
            self.assertIs(pub_calls[0][0], namespace["settings"].OUTPUT_DIR)
            self.assertEqual(pub_calls[0][2], captured_formats["srt"])
            self.assertEqual(pub_calls[0][3], captured_formats["ass"])
            self.assertEqual(pub_calls[0][4], captured_formats["vtt"])
            job = namespace["job_manager"].jobs["memory"]
            self.assertEqual(job["status"], status)
            self.assertEqual(job["warnings_count"], count)
            self.assertEqual(set(writes), {"memory.srt", "memory.ass", "memory.vtt"})
            self.assertIn("00:00:01,250 --> 00:00:02,750", writes["memory.srt"])


if __name__ == "__main__":
    unittest.main()
