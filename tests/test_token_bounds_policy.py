"""Input admission: actual AST methods with synthetic tokens/model/HTTP only.

Stdlib-only fixture; no production app/config/transformers/torch imports, real
tokenizer counts, GPU, downloads, network, uploads or output disk writes.
Counts are declared test data, never claims about actual multilingual tokens.
"""
import asyncio
import contextlib
import copy
import runpy
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LANG = runpy.run_path(str(ROOT / "tests/test_translation_fallback_policy.py"),
                      run_name="isolated_language_fixture_for_bounds")
cue = LANG["cue"]
BOUNDS = ("INPUT_TOKEN_BUDGET_EXCEEDED", "SOURCE_RETAINED_FOR_REVIEW")
LANGUAGE_WARNING = LANG["WARNING"]


class BoundsBackend(LANG["SyntheticBackend"]):
    """Actual extracted translator, declared fake content/special token counts."""
    def __init__(self, content_counts=None, specials=2, side="right", fault=None,
                 empty_decode=(), status=200):
        super().__init__(status=status)
        owner = self
        self.counts = content_counts or {}
        self.events, self.prepared, self.last_admitted = [], {}, []
        self.text_ids, self.id_texts = {}, {}
        self.collections = 0
        self.main_thread = threading.get_ident()

        def event(kind, **data):
            owner.events.append(dict(kind=kind, thread=threading.get_ident(), **data))

        class Rows(list):
            def tolist(self):
                return copy.deepcopy(list(self))

        class Inputs(dict):
            def to(self, device):
                event("transfer", device=device)
                return self

        class Tokenizer:
            src_lang = None
            eos_token_id = 2
            pad_token_id = 1
            decoder_start_token_id = 2
            def convert_tokens_to_ids(self, value):
                return 1
            def __call__(self, texts, **kwargs):
                if len(texts) != 1 or self.src_lang is None:
                    raise AssertionError("Single-cue preparation must follow source-language selection")
                text = texts[0]
                token_id = owner.text_ids.setdefault(text, 100 + len(owner.text_ids))
                owner.id_texts[token_id] = text
                ids = list(range(900, 900 + specials)) + [token_id] * owner.counts.get(text, 2)
                mask = [1] * len(ids)
                owner.prepared[tuple(ids)] = text
                event("prepare", text=text, language=self.src_lang, kwargs=kwargs, ids=list(ids))
                if fault == "preflight_count":
                    return {"input_ids": [], "attention_mask": [mask]}
                if fault == "preflight_mask":
                    mask[-1] = 0
                if fault == "preflight_type":
                    ids[-1] = True
                return {"input_ids": [ids], "attention_mask": [mask]}
            def pad(self, encoded, **kwargs):
                event("pad", encoded=copy.deepcopy(encoded), kwargs=kwargs)
                if fault == "pad_mutate":
                    encoded["input_ids"][0].pop()
                    encoded["attention_mask"][0].pop()
                owner.last_admitted = [owner.prepared.get(tuple(row), "CORRUPT")
                                       for row in encoded["input_ids"]]
                width = max(map(len, encoded["input_ids"]))
                ids, masks = [], []
                for row in encoded["input_ids"]:
                    zeros, ones = [0] * (width - len(row)), [1] * len(row)
                    pads = [1] * len(zeros)
                    ids.append(list(row) + pads if side == "right" else pads + list(row))
                    masks.append(ones + zeros if side == "right" else zeros + ones)
                if fault == "pad_corrupt":
                    ids[0][0] = 99999
                if fault == "pad_mask":
                    masks[0][-1] = 2
                if fault == "pad_cardinality":
                    ids, masks = ids[:-1], masks[:-1]
                if fault == "pad_width":
                    for row, mask in zip(ids, masks):
                        extra = 257 - len(row)
                        row.extend([1] * extra)
                        mask.extend([0] * extra)
                if fault == "pad_ragged" and len(ids) > 1:
                    ids[1].append(1)
                return Inputs(input_ids=Rows(ids), attention_mask=Rows(masks))
            def decode(self, value, **kwargs):
                event("decode", value=list(value), kwargs=kwargs)
                if fault == "decode":
                    raise ValueError("synthetic decode failure")
                val_list = list(value)
                matching = next((tok for tok in val_list if tok in owner.id_texts), None)
                text = owner.id_texts.get(matching, "unknown") if matching is not None else owner.id_texts.get(val_list[0], "")
                return "   " if text in empty_decode else "translated:" + text

        class TokenizerFactory:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                event("fake_tokenizer_factory")
                tokenizer = Tokenizer()
                owner.tokenizers.append(tokenizer)
                return tokenizer

        class Model:
            generation_config = SimpleNamespace(
                eos_token_id=2, pad_token_id=1, decoder_start_token_id=2
            )
            def to(self, device):
                event("fake_model_to", device=device)
                return self
            def eval(self):
                pass
            def generate(self, **kwargs):
                event("generate", kwargs=copy.deepcopy(kwargs), texts=list(owner.last_admitted))
                arabic_id = kwargs.get("forced_bos_token_id", 1)
                result = [[2, arabic_id, owner.text_ids[text], 2] for text in owner.last_admitted]
                if fault == "generated_count":
                    return result[:-1]
                if fault == "generated_iter_short":
                    class ShortIterable(list):
                        def __iter__(self):
                            return iter(list.__getitem__(self, slice(0, -1)))
                    return ShortIterable(result)
                return result

        class ModelFactory:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                event("fake_model_factory")
                return Model()

        def no_grad():
            event("no_grad")
            return contextlib.nullcontext()
        def collect():
            owner.collections += 1
        self.namespace.update(torch=SimpleNamespace(no_grad=no_grad,
                              cuda=SimpleNamespace(is_available=lambda: False)),
                              gc=SimpleNamespace(collect=collect))
        self.transformers.AutoTokenizer = TokenizerFactory
        self.transformers.AutoModelForSeq2SeqLM = ModelFactory

    def events_of(self, kind):
        return [event for event in self.events if event["kind"] == kind]


class TestTokenBoundsPolicy(unittest.TestCase):
    def test_prepared_255_256_257_boundary_and_language_state(self):
        fake = BoundsBackend({"a": 253, "b": 254, "c": 255})
        output = fake.local([cue(index=i + 1, text=t) for i, t in enumerate("abc")], "FR_fr")
        self.assertEqual([s.text for s in output], ["translated:a", "translated:b", "c"])
        self.assertEqual([s.warnings for s in output], [(), (), BOUNDS])
        self.assertEqual([len(e["ids"]) for e in fake.events_of("prepare")], [255, 256, 257])
        self.assertTrue(all(e["language"] == "fra_Latn" for e in fake.events_of("prepare")))

    def test_variable_specials_and_no_character_heuristic(self):
        for specials, retained in [(2, False), (3, True)]:
            fake = BoundsBackend({"x": 254}, specials=specials)
            out = fake.local([cue(text="x")], "en")[0]
            self.assertEqual(out.text, "x" if retained else "translated:x")
        long_text = "界" * 2000
        fake = BoundsBackend({long_text: 1, "短": 300})
        out = fake.local([cue(text=long_text), cue(index=2, text="短")], "en")
        self.assertEqual(out[0].text, "translated:" + long_text)
        self.assertEqual(out[1].text, "短")

    def test_unicode_whitespace_and_empty_source_are_tokenizer_driven(self):
        texts = ["مرحبا", "日本語", " \t ", ""]
        fake = BoundsBackend({text: 0 for text in texts})
        out = fake.local([cue(index=i, text=t) for i, t in enumerate(texts)], None)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(not s.warnings for s in out))
        self.assertEqual([e["text"] for e in fake.events_of("prepare")], texts)

    def test_mixed_batch_reinterleaves_and_preserves_metadata(self):
        fake = BoundsBackend({"a": 2, "over": 300, "b": 3})
        fake.translator.device = "cuda"  # synthetic to()/torch only; never real CUDA
        source = [cue(index=17, text="a"), cue(index=3, text="over", custom={"x": 1}), cue(index=91, text="b")]
        before = copy.deepcopy([vars(s) for s in source])
        out = fake.local(source, "en")
        self.assertEqual([s.text for s in out], ["translated:a", "over", "translated:b"])
        self.assertEqual([s.index for s in out], [17, 3, 91])
        self.assertEqual([vars(s) for s in source], before)
        for original, result in zip(source, out):
            self.assertIsNot(original, result)
            self.assertEqual({k: v for k, v in vars(original).items() if k not in {"text", "warnings"}},
                             {k: v for k, v in vars(result).items() if k not in {"text", "warnings"}})
        self.assertEqual(fake.events_of("generate")[0]["texts"], ["a", "b"])

    def test_all_over_budget_forbids_pad_transfer_generate_decode(self):
        fake = BoundsBackend({"over": 255})
        progress = []
        out = fake.local([cue(index=i, text="over") for i in range(3)], "sw", progress_cb=progress.append)
        self.assertTrue(all(s.text == "over" and s.warnings == BOUNDS for s in out))
        for kind in ["pad", "transfer", "no_grad", "generate", "decode"]:
            self.assertEqual(fake.events_of(kind), [])
        self.assertEqual(progress, [66, 100])
        self.assertEqual(len(fake.events_of("fake_model_factory")), 1)  # existing load is NOT bypassed
        self.assertEqual(fake.collections, 1)

    def test_multiple_cpu_batches_keep_progress_and_count(self):
        fake = BoundsBackend({"over": 300})
        progress = []
        out = fake.local([cue(index=i, text="over" if i % 2 else "a") for i in range(5)],
                         "en", progress_cb=progress.append)
        self.assertEqual([s.index for s in out], list(range(5)))
        self.assertEqual(progress, [40, 80, 100])
        self.assertEqual([len(e["texts"]) for e in fake.events_of("generate")], [1, 1, 1])

    def test_simulated_cuda_branch_keeps_six_cue_batching(self):
        fake = BoundsBackend()
        fake.translator.device = "cuda"
        progress = []
        out = fake.local([cue(index=i, text="a") for i in range(7)], "en", progress_cb=progress.append)
        self.assertEqual(len(out), 7)
        self.assertEqual(progress, [85, 100])
        self.assertEqual([len(e["texts"]) for e in fake.events_of("generate")], [6, 1])

    def test_exact_prepared_ids_left_and_right_padding_and_generation_kwargs(self):
        for side in ["left", "right"]:
            fake = BoundsBackend({"a": 2, "b": 7}, side=side)
            fake.local([cue(text="a"), cue(index=2, text="b")], "en")
            prepared = [e["ids"] for e in fake.events_of("prepare")]
            generation = fake.events_of("generate")[0]["kwargs"]
            recovered = [[token for token, bit in zip(row, mask) if bit]
                         for row, mask in zip(generation["input_ids"], generation["attention_mask"])]
            self.assertEqual(recovered, prepared)
            self.assertEqual(len(fake.events_of("prepare")), 2)
            self.assertEqual({k: generation[k] for k in ["forced_bos_token_id", "max_new_tokens", "num_beams", "do_sample"]},
                             dict(forced_bos_token_id=1, max_new_tokens=140, num_beams=2, do_sample=False))
            self.assertTrue(all(len(row) <= 256 for row in generation["input_ids"]))

    def test_warning_dedup_and_no_new_language_warning_on_retained_cue(self):
        fake = BoundsBackend({"over": 300})
        source = cue(text="over", warnings=("UPSTREAM", BOUNDS[0], LANGUAGE_WARNING))
        out = fake.local([source, cue(index=2, text="a")], "sw")
        self.assertEqual(out[0].warnings, ("UPSTREAM", BOUNDS[0], LANGUAGE_WARNING, BOUNDS[1]))
        self.assertEqual(source.warnings, ("UPSTREAM", BOUNDS[0], LANGUAGE_WARNING))
        self.assertEqual(out[1].warnings, (LANGUAGE_WARNING,))
        fresh = fake.local([cue(text="over")], "sw")[0]
        self.assertEqual(fresh.warnings, BOUNDS)

    def test_empty_decode_and_clean_missing_warning_attribute_stay_unchanged(self):
        fake = BoundsBackend(empty_decode={"empty"})
        source = cue(text="empty")
        del source.warnings
        out = fake.local([source], "en")[0]
        self.assertEqual(out.text, "empty")
        self.assertEqual(out.warnings, ("EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW"))
        self.assertEqual(source.text, "empty")
        self.assertFalse(hasattr(source, "warnings"))

    def test_malformed_preflight_fails_before_padding_and_cleans_up(self):
        for fault in ["preflight_count", "preflight_mask", "preflight_type"]:
            fake = BoundsBackend(fault=fault)
            original = cue()
            before = copy.deepcopy(vars(original))
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "NLLB preflight"):
                fake.local([original], "en")
            self.assertEqual(fake.events_of("pad"), [])
            self.assertEqual(fake.events_of("generate"), [])
            self.assertEqual(fake.collections, 1)
            self.assertEqual(vars(original), before)

    def test_corrupted_or_mutated_padding_fails_before_device_transfer(self):
        for fault in ["pad_corrupt", "pad_mutate", "pad_mask", "pad_cardinality", "pad_width", "pad_ragged"]:
            fake = BoundsBackend({"a": 3, "b": 4}, fault=fault)
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "NLLB pad"):
                fake.local([cue(text="a"), cue(index=2, text="b")], "en")
            self.assertEqual(fake.events_of("transfer"), [])
            self.assertEqual(fake.events_of("generate"), [])
            self.assertEqual(fake.collections, 1)

    def test_generated_or_decoded_cardinality_failure_is_not_silent_zip_loss(self):
        for fault in ["generated_count", "generated_iter_short", "decode"]:
            fake = BoundsBackend(fault=fault)
            source = [cue(text="a"), cue(index=2, text="b")]
            before = copy.deepcopy([vars(s) for s in source])
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                fake.local(source, "en")
            self.assertEqual(fake.collections, 1)
            self.assertEqual([vars(s) for s in source], before)

    def test_outer_worker_cardinality_failure_is_explicit(self):
        fake = BoundsBackend()
        async def shortened_worker(*args, **kwargs):
            return []
        fake.namespace["asyncio"] = SimpleNamespace(to_thread=shortened_worker)
        with self.assertRaisesRegex(ValueError, "original batch cardinality"):
            fake.local([cue()], "en")
        self.assertEqual(fake.collections, 1)

    def test_cloud_success_and_arabic_empty_bypass_do_not_run_preflight(self):
        for provider in ["gemini", "groq", "qwen"]:
            fake = BoundsBackend({"over": 300})
            async def forbidden(*args, **kwargs):
                raise AssertionError("NLLB forbidden on successful provider")
            fake.translator._translate_nllb_local = forbidden
            out = fake.run(fake.translator.translate([cue(text="over")], "sw", provider))
            self.assertNotIn(BOUNDS[0], out[0].warnings)
            self.assertEqual(fake.events, [])
        fake = BoundsBackend()
        source = [cue()]
        self.assertIs(fake.run(fake.translator.translate(source, "ar-EG")), source)
        self.assertEqual(fake.run(fake.translator.translate([], "en")), [])
        self.assertEqual(fake.events, [])

    def test_actual_cloud_failures_recover_admitted_and_retained_cues(self):
        for provider in ["gemini", "groq", "qwen"]:
            for text, expected in [("over", BOUNDS), ("a", (LANGUAGE_WARNING,))]:
                fake = BoundsBackend({"over": 300}, status=503)
                out = fake.run(fake.translator.translate([cue(text=text)], "sw", provider))[0]
                self.assertEqual(out.warnings, expected)
                self.assertEqual(out.text, "over" if text == "over" else "translated:a")

    def test_actual_memory_pipeline_warns_on_retained_source_and_clean_control(self):
        for source_text, expected_status, warning_count in [("over", "completed_with_warnings", 2),
                                                            ("a", "completed", 0)]:
            fake = BoundsBackend({"over": 300})
            # This stdlib AST facade executes the actual existing import pipeline.
            import ast
            import time
            namespace = dict(fake.namespace, Path=Path, time=time, gpu_lock=asyncio.Lock())
            tree = ast.parse((ROOT / "services/subtitles.py").read_text(encoding="utf-8-sig"))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name == "re" for alias in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, {"types", "typing"})
                else:
                    self.assertIsInstance(node, (ast.Expr, ast.FunctionDef))
            LANG["execute"](tree.body, namespace)
            pt = ast.parse((ROOT / "services/pipeline.py").read_text(encoding="utf-8-sig"))
            LANG["execute"]([node for node in pt.body if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef))
                             and node.name in {"JobManager", "run_exclusive_worker", "run_subtitle_translation_pipeline"}], namespace)
            writes = {}
            class Output:
                def __init__(self, name=""):
                    self.name = name
                def __truediv__(self, name):
                    return Output(name)
                def write_text(self, value, **kwargs):
                    writes[self.name] = value
            pub_calls = []
            def publish_subtitle_triplet(output_dir, job_id, srt_content, ass_content, vtt_content):
                pub_calls.append((output_dir, job_id, srt_content, ass_content, vtt_content))
                (output_dir / f"{job_id}.srt").write_text(srt_content, encoding="utf-8-sig")
                (output_dir / f"{job_id}.ass").write_text(ass_content, encoding="utf-8-sig")
                (output_dir / f"{job_id}.vtt").write_text(vtt_content, encoding="utf-8")
                return {"srt": output_dir / f"{job_id}.srt", "ass": output_dir / f"{job_id}.ass", "vtt": output_dir / f"{job_id}.vtt"}
            class Source:
                name, stem, suffix = "memory.srt", "memory", ".srt"
                def read_text(self, **kwargs):
                    return "1\n00:00:01,250 --> 00:00:02,750\n" + source_text + "\n"
            async def translate(segments, **kwargs):
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

            namespace.update(settings=SimpleNamespace(OUTPUT_DIR=Output()),
                             translator=SimpleNamespace(translate=translate), job_manager=namespace["JobManager"](),
                             publish_subtitle_triplet=publish_subtitle_triplet)
            namespace["job_manager"].create_job("memory", "memory.srt")
            fake.run(namespace["run_subtitle_translation_pipeline"]("memory", Source(), source_language="fr"))
            self.assertEqual(len(pub_calls), 1)
            self.assertEqual(pub_calls[0][1], "memory")
            self.assertIs(pub_calls[0][0], namespace["settings"].OUTPUT_DIR)
            self.assertEqual(pub_calls[0][2], captured_formats["srt"])
            self.assertEqual(pub_calls[0][3], captured_formats["ass"])
            self.assertEqual(pub_calls[0][4], captured_formats["vtt"])
            job = namespace["job_manager"].jobs["memory"]
            self.assertEqual(job["status"], expected_status)
            self.assertEqual(job["warnings_count"], warning_count)
            self.assertIn("00:00:01,250 --> 00:00:02,750", writes["memory.srt"])
            self.assertIn("over" if source_text == "over" else "translated:a", writes["memory.srt"])

    def test_prepare_pad_transfer_generate_decode_execute_in_worker_thread(self):
        fake = BoundsBackend()
        fake.local([cue()], "en")
        for kind in ["prepare", "pad", "transfer", "no_grad", "generate", "decode"]:
            self.assertTrue(fake.events_of(kind))
            self.assertTrue(all(e["thread"] != fake.main_thread for e in fake.events_of(kind)))


if __name__ == "__main__":
    unittest.main()
