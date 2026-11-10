"""Output completion policy: AST-extracted translator with synthetic tokens, model, and generation config.

Stdlib-only fixture; no production transformers/torch imports, real model downloads,
network/API calls, or GPU allocations.
"""
import ast
import asyncio
import copy
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LANG = runpy.run_path(str(ROOT / "tests/test_translation_fallback_policy.py"),
                      run_name="isolated_language_fixture_for_output_completion")
cue = LANG["cue"]
LANGUAGE_WARNING = LANG["WARNING"]

# Standard warning tuples
WARN_BUDGET = ("OUTPUT_TOKEN_BUDGET_EXCEEDED", "SOURCE_RETAINED_FOR_REVIEW")
WARN_INCOMPLETE = ("OUTPUT_GENERATION_INCOMPLETE", "SOURCE_RETAINED_FOR_REVIEW")
WARN_EMPTY = ("EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW")
WARN_BUDGET_EMPTY = ("OUTPUT_TOKEN_BUDGET_EXCEEDED", "EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW")
WARN_INCOMPLETE_EMPTY = ("OUTPUT_GENERATION_INCOMPLETE", "EMPTY_TRANSLATION_OUTPUT", "SOURCE_RETAINED_FOR_REVIEW")
WARN_INPUT_BOUNDS = ("INPUT_TOKEN_BUDGET_EXCEEDED", "SOURCE_RETAINED_FOR_REVIEW")


class IntChild(int):
    """Exact int subclass for scalar validation rejection tests."""
    pass


CUE_TOKEN_ENCODINGS = {
    "Clean 1": [1010, 1011, 1012, 1013],                          # 4 tokens
    "No-EOS Cap 2": [1020, 1021, 1022, 1023, 1024, 1025],         # 6 tokens
    "Incomplete 3": [1030, 1031, 1032],                           # 3 tokens
    "Empty EOS 4": [1040, 1041, 1042, 1043, 1044],                # 5 tokens
    "Clean 5": [1050, 1051, 1052, 1053],                          # 4 tokens
    "Cap EOS 7": [1070, 1071, 1072, 1073, 1074, 1075],           # 6 tokens
    "Incomplete Empty 8": [1080, 1081, 1082],                     # 3 tokens
    "Clean 9": [1090, 1091, 1092, 1093, 1094],                    # 5 tokens
    "Special Empty 10": [1100, 1101],                             # 2 tokens
    "Clean 11": [1110, 1111, 1112, 1113],                         # 4 tokens
}


class OutputCompletionBackend(LANG["SyntheticBackend"]):
    """Output completion test backend with configurable generated rows and config."""
    def __init__(self, generated_rows=None, decode_texts=None, eos_token_id=2, pad_token_id=1,
                 decoder_start_token_id=2, config_level="generation_config", fault=None,
                 device="cpu", gen_config_dict=None, model_config_dict=None,
                 tokenizer_config=None, return_plain_list=False):
        super().__init__()
        owner = self
        self.generated_rows = generated_rows or {}
        self.decode_texts = decode_texts or {}
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.config_level = config_level
        self.gen_config_dict = gen_config_dict
        self.model_config_dict = model_config_dict
        self.tokenizer_config = tokenizer_config
        self.fault = fault
        self.return_plain_list = return_plain_list
        self.collections = 0
        self.cuda_clears = 0
        self.events = []
        self.cue_gen_idx = 0
        self.translator.device = device

        def event(kind, **data):
            owner.events.append(dict(kind=kind, **data))

        class Rows(list):
            def tolist(self):
                return copy.deepcopy(list(self))

        class Inputs(dict):
            def to(self, device):
                event("transfer", device=device)
                return self

        class Tokenizer:
            src_lang = None
            if owner.tokenizer_config is not None:
                eos_token_id = owner.tokenizer_config.get("eos_token_id")
                pad_token_id = owner.tokenizer_config.get("pad_token_id")
                decoder_start_token_id = owner.tokenizer_config.get("decoder_start_token_id")
            elif owner.config_level == "generation_config":
                eos_token_id = 999 if owner.eos_token_id is not None else None
                pad_token_id = 999 if owner.pad_token_id is not None else None
                decoder_start_token_id = 999 if owner.decoder_start_token_id is not None else None
            elif owner.config_level == "config":
                eos_token_id = 999 if owner.eos_token_id is not None else None
                pad_token_id = 999 if owner.pad_token_id is not None else None
                decoder_start_token_id = 999 if owner.decoder_start_token_id is not None else None
            else:  # tokenizer level fallback
                eos_token_id = owner.eos_token_id
                pad_token_id = owner.pad_token_id
                decoder_start_token_id = owner.decoder_start_token_id

            def convert_tokens_to_ids(self, value):
                return 256047 if value == "arb_Arab" else 1

            def __call__(self, texts, **kwargs):
                event("tokenize", texts=list(texts), kwargs=copy.deepcopy(kwargs))
                ids_list, mask_list = [], []
                for text in texts:
                    if "OVERSIZE" in text:
                        ids = [900] + [100] * 299
                    elif text in CUE_TOKEN_ENCODINGS:
                        ids = list(CUE_TOKEN_ENCODINGS[text])
                    else:
                        ids = [2000 + i for i in range(len(text.split()) + 2)]
                    mask = [1] * len(ids)
                    ids_list.append(ids)
                    mask_list.append(mask)
                return {"input_ids": ids_list, "attention_mask": mask_list}

            def pad(self, encoded, **kwargs):
                event("pad", encoded=copy.deepcopy(encoded))
                width = max(map(len, encoded["input_ids"]))
                ids, masks = [], []
                for row in encoded["input_ids"]:
                    zeros, ones = [0] * (width - len(row)), [1] * len(row)
                    pads = [1] * len(zeros)
                    ids.append(list(row) + pads)
                    masks.append(ones + zeros)
                return Inputs(input_ids=Rows(ids), attention_mask=Rows(masks))

            def decode(self, value, **kwargs):
                val_list = value.tolist() if hasattr(value, "tolist") else list(value)
                event("decode", value=list(val_list))
                if owner.fault == "decode_raises":
                    raise ValueError("synthetic decode exception")
                if owner.fault == "decode_non_string":
                    return 12345
                key = tuple(val_list)
                if key in owner.decode_texts:
                    return owner.decode_texts[key]
                return "مرحبا ترجمة ناجحة"

        class TokenizerFactory:
            @staticmethod
            def from_pretrained(name, **kwargs):
                event("load_tokenizer", name=name)
                return Tokenizer()

        class Model:
            def __init__(self):
                if owner.gen_config_dict is not None or owner.model_config_dict is not None:
                    self.generation_config = owner.gen_config_dict
                    self.config = owner.model_config_dict
                elif owner.config_level == "generation_config":
                    self.generation_config = SimpleNamespace(
                        decoder_start_token_id=owner.decoder_start_token_id,
                        eos_token_id=owner.eos_token_id,
                        pad_token_id=owner.pad_token_id,
                    )
                    self.config = SimpleNamespace(
                        decoder_start_token_id=None,
                        eos_token_id=None,
                        pad_token_id=None,
                    )
                elif owner.config_level == "config":
                    self.generation_config = None
                    self.config = SimpleNamespace(
                        decoder_start_token_id=owner.decoder_start_token_id,
                        eos_token_id=owner.eos_token_id,
                        pad_token_id=owner.pad_token_id,
                    )
                else:  # tokenizer level fallback
                    self.generation_config = None
                    self.config = None

            def to(self, device):
                event("model_to", device=device)
                return self

            def eval(self):
                event("model_eval")
                return self

            def generate(self, **kwargs):
                batch_inputs = kwargs.get("input_ids", kwargs.get("inputs", [1]))
                batch_masks = kwargs.get("attention_mask", None)
                event(
                    "generate",
                    kwargs={k: v for k, v in kwargs.items() if k not in ("input_ids", "inputs", "attention_mask")},
                    input_ids=copy.deepcopy(batch_inputs.tolist() if hasattr(batch_inputs, "tolist") else list(batch_inputs)),
                    attention_mask=copy.deepcopy(batch_masks.tolist() if hasattr(batch_masks, "tolist") else list(batch_masks)) if batch_masks is not None else None,
                )
                if owner.fault == "generate_raises":
                    raise RuntimeError("synthetic generation error")

                batch_len = len(batch_inputs)
                out_rows = []
                for i in range(batch_len):
                    idx = owner.cue_gen_idx + i
                    if owner.generated_rows:
                        if idx in owner.generated_rows:
                            row = list(owner.generated_rows[idx])
                        else:
                            raise KeyError(f"Missing explicit generated row for admitted cue generation index {idx}")
                    else:
                        row = [2, 256047, 7, 2]
                    out_rows.append(row)
                owner.cue_gen_idx += batch_len
                return out_rows if owner.return_plain_list else Rows(out_rows)

        class ModelFactory:
            @staticmethod
            def from_pretrained(name, **kwargs):
                event("load_model", name=name)
                return Model()

        self.namespace["AutoTokenizer"] = TokenizerFactory
        self.namespace["AutoModelForSeq2SeqLM"] = ModelFactory
        self.transformers.AutoTokenizer = TokenizerFactory
        self.transformers.AutoModelForSeq2SeqLM = ModelFactory

        def collect():
            owner.collections += 1
            event("gc_collect")

        def empty_cache():
            owner.cuda_clears += 1
            event("cuda_empty_cache")

        class NoGrad:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        self.namespace["gc"] = SimpleNamespace(collect=collect)
        self.namespace["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, empty_cache=empty_cache),
            float16="float16",
            float32="float32",
            no_grad=lambda: NoGrad(),
        )

    def local(self, cues, source_lang="en", progress_cb=None):
        return self.run(self.translator._translate_nllb_local(cues, source_lang, progress_cb=progress_cb))


class TestOutputCompletionPolicy(unittest.TestCase):
    """Exhaustive test suite for NLLB output completion policy."""

    def test_01_the_five_canonical_boundary_examples(self):
        """1. No-EOS 140, early EOS padded, EOS at 140, short no-EOS, pad==EOS early termination, EOS at 139."""
        examples = [
            # Case A: 140 tokens without EOS -> budget exhausted, source retained
            ([2, 256047] + [7] * 139, WARN_BUDGET, True),
            # Case B: Early EOS at step 3, padded out -> natural completion, translation kept
            ([2, 256047, 7, 2, 1, 1], (), False),
            # Case C: EOS at step 140 ceiling -> budget exhausted, source retained
            ([2, 256047] + [7] * 138 + [2], WARN_BUDGET, True),
            # Case D: Short no-EOS without reaching ceiling -> generation incomplete, source retained
            ([2, 256047, 7, 7], WARN_INCOMPLETE, True),
            # Case E: Pad == EOS with early termination -> natural completion, translation kept
            ([2, 256047, 7, 2, 2, 2], (), False),
            # Case F: EOS at step 139 (strictly < 140) -> natural completion, translation kept
            ([2, 256047] + [7] * 137 + [2], (), False),
        ]

        for i, (row, expected_warnings, expect_source_retained) in enumerate(examples):
            with self.subTest(example_index=i):
                pad_id = 2 if i == 4 else 1
                backend = OutputCompletionBackend(
                    generated_rows={0: row},
                    eos_token_id=2,
                    pad_token_id=pad_id,
                )
                original_cue = cue(index=i + 1, text=f"Source Dialogue {i}", warnings=())
                results = backend.local([original_cue], "en")
                self.assertEqual(len(results), 1)
                res = results[0]
                if expect_source_retained:
                    self.assertEqual(res.text, original_cue.text)
                else:
                    self.assertEqual(res.text, "مرحبا ترجمة ناجحة")
                self.assertEqual(res.warnings, expected_warnings)

    def test_02_effective_config_precedence_and_validation(self):
        """2. Precedence: generation_config > config > tokenizer; token 0 valid; invalid/Boolean IDs rejected."""
        # Precedence: generation_config overrides config and tokenizer
        b_gen = OutputCompletionBackend(eos_token_id=2, config_level="generation_config")
        res = b_gen.local([cue(text="Hello", warnings=())], "en")
        self.assertEqual(res[0].warnings, ())

        # Precedence: config used when generation_config absent
        b_cfg = OutputCompletionBackend(eos_token_id=2, config_level="config")
        res_cfg = b_cfg.local([cue(text="Hello", warnings=())], "en")
        self.assertEqual(res_cfg[0].warnings, ())

        # Token 0 is valid ID for eos, start, and pad
        b_zero = OutputCompletionBackend(eos_token_id=0, decoder_start_token_id=0, pad_token_id=0,
                                         generated_rows={0: [0, 256047, 7, 0]})
        res_zero = b_zero.local([cue(text="Zero", warnings=())], "en")
        self.assertEqual(res_zero[0].warnings, ())

        # Invalid/Boolean/Missing IDs must raise ValueError with specific messages
        for invalid in [True, False, -1, "bad_id"]:
            with self.subTest(invalid_eos=invalid):
                b_inv = OutputCompletionBackend(eos_token_id=invalid)
                with self.assertRaises(ValueError) as cm:
                    b_inv.local([cue(text="Hello")], "en")
                self.assertIn("invalid eos_token_id", str(cm.exception))

        # Missing decoder_start_token_id
        b_no_start = OutputCompletionBackend(decoder_start_token_id=None, config_level="config")
        with self.assertRaises(ValueError) as cm:
            b_no_start.local([cue(text="Hello")], "en")
        self.assertIn("decoder_start_token_id", str(cm.exception))

        # Invalid start IDs (boolean, negative, string)
        for bad_start in [True, False, -1, "invalid"]:
            with self.subTest(bad_start=bad_start):
                b_bad = OutputCompletionBackend(decoder_start_token_id=bad_start)
                with self.assertRaises(ValueError) as cm:
                    b_bad.local([cue(text="Hello")], "en")
                self.assertIn("decoder_start_token_id", str(cm.exception))

    def test_03_interior_pad_vs_trailing_pad_and_ceiling_enforcement(self):
        """3. Interior pad counted in 140 budget boundary; trailing pad stops; over-ceiling active count rejected."""
        # 140-step no-EOS boundary containing interior pad token: 1 forced BOS + 1 interior pad + 138 content = 140 positions -> BUDGET
        row_interior_140 = [2, 256047, 1] + [7] * 138
        b_140 = OutputCompletionBackend(generated_rows={0: row_interior_140})
        res_140 = b_140.local([cue(text="Interior pad 140 boundary", warnings=())], "en")
        self.assertEqual(res_140[0].text, "Interior pad 140 boundary")
        self.assertEqual(res_140[0].warnings, WARN_BUDGET)

        # No-EOS row with contiguous trailing padding: active generation stopped early -> INCOMPLETE
        row_trailing_pad = [2, 256047, 7] + [1] * 137
        b_trail = OutputCompletionBackend(generated_rows={0: row_trailing_pad})
        res_trail = b_trail.local([cue(text="Trailing pad incomplete", warnings=())], "en")
        self.assertEqual(res_trail[0].text, "Trailing pad incomplete")
        self.assertEqual(res_trail[0].warnings, WARN_INCOMPLETE)

        # Over-ceiling active tokens without EOS (> 140) must raise ValueError
        row_over_ceiling = [2, 256047] + [7] * 141
        backend_over = OutputCompletionBackend(generated_rows={0: row_over_ceiling})
        with self.assertRaises(ValueError):
            backend_over.local([cue(text="Over ceiling")], "en")

        # Wide batch tensor allocation where trailing padding extends beyond 140, but active content stops
        row_wide = [2, 256047, 7, 2] + [1] * 200
        backend_wide = OutputCompletionBackend(generated_rows={0: row_wide})
        res_wide = backend_wide.local([cue(text="Wide batch", warnings=())], "en")
        self.assertEqual(res_wide[0].text, "مرحبا ترجمة ناجحة")
        self.assertEqual(res_wide[0].warnings, ())

    def test_04_empty_decode_and_coinciding_warning_ordering(self):
        """4. Empty/whitespace decode with early EOS, budget exhaustion, and incomplete generation."""
        cases = [
            # A: Early EOS but decoded text is empty -> retains source, [EMPTY_TRANSLATION_OUTPUT, SOURCE_RETAINED_FOR_REVIEW]
            ([2, 256047, 7, 2], "", WARN_EMPTY),
            # B: Early EOS but decoded text is whitespace -> retains source, [EMPTY_TRANSLATION_OUTPUT, SOURCE_RETAINED_FOR_REVIEW]
            ([2, 256047, 7, 2], "   \t\n  ", WARN_EMPTY),
            # C: Budget exhausted AND decoded text is empty -> retains source, [OUTPUT_TOKEN_BUDGET_EXCEEDED, EMPTY_TRANSLATION_OUTPUT, SOURCE_RETAINED_FOR_REVIEW]
            ([2, 256047] + [7] * 139, "", WARN_BUDGET_EMPTY),
            # D: Incomplete generation AND decoded text is empty -> retains source, [OUTPUT_GENERATION_INCOMPLETE, EMPTY_TRANSLATION_OUTPUT, SOURCE_RETAINED_FOR_REVIEW]
            ([2, 256047, 7], "", WARN_INCOMPLETE_EMPTY),
        ]

        for i, (row, decoded_val, expected_warnings) in enumerate(cases):
            with self.subTest(empty_case_index=i):
                backend = OutputCompletionBackend(
                    generated_rows={0: row},
                    decode_texts={tuple(row): decoded_val},
                )
                original_cue = cue(index=i + 1, text=f"Dialogue Line {i}", warnings=())
                res = backend.local([original_cue], "en")
                self.assertEqual(res[0].text, original_cue.text)
                self.assertEqual(res[0].warnings, expected_warnings)

    def _run_matrix_check(self, dev="cpu"):
        cue_specs = [
            ("OVERSIZE " + ("word " * 150), "oversize_0"),
            ("Clean 1", "clean_1"),
            ("No-EOS Cap 2", "cap_2"),
            ("Incomplete 3", "inc_3"),
            ("Empty EOS 4", "empty_4"),
            ("Clean 5", "clean_5"),
            ("OVERSIZE " + ("word " * 150), "oversize_6"),
            ("Cap EOS 7", "cap_7"),
            ("Incomplete Empty 8", "inc_empty_8"),
            ("Clean 9", "clean_9"),
            ("Special Empty 10", "special_10"),
            ("Clean 11", "clean_11"),
            ("OVERSIZE " + ("word " * 150), "oversize_12"),
        ]
        gen_map = {
            0: [2, 256047, 701, 2],
            1: [2, 256047] + [702] * 139,
            2: [2, 256047, 703],
            3: [2, 256047, 704, 2],
            4: [2, 256047, 705, 2],
            5: [2, 256047] + [707] * 138 + [2],
            6: [2, 256047, 708],
            7: [2, 256047, 709, 2],
            8: [2, 256047, 2],
            9: [2, 256047, 711, 2],
        }
        dec_map = {
            tuple([2, 256047, 701, 2]): "ترجمة نظيفة 1",
            tuple([2, 256047] + [702] * 139): "ترجمة كاب 2",
            tuple([2, 256047, 703]): "ترجمة غير مكتملة 3",
            tuple([2, 256047, 704, 2]): "",
            tuple([2, 256047, 705, 2]): "ترجمة نظيفة 5",
            tuple([2, 256047] + [707] * 138 + [2]): "ترجمة كاب 7",
            tuple([2, 256047, 708]): "   ",
            tuple([2, 256047, 709, 2]): "ترجمة نظيفة 9",
            tuple([2, 256047, 2]): "",
            tuple([2, 256047, 711, 2]): "ترجمة نظيفة 11",
        }
        cpu_expected_batches = [
            ([[1010, 1011, 1012, 1013]], [[1, 1, 1, 1]]),
            ([[1020, 1021, 1022, 1023, 1024, 1025], [1030, 1031, 1032, 1, 1, 1]],
             [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]),
            ([[1040, 1041, 1042, 1043, 1044], [1050, 1051, 1052, 1053, 1]],
             [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
            ([[1070, 1071, 1072, 1073, 1074, 1075]], [[1, 1, 1, 1, 1, 1]]),
            ([[1080, 1081, 1082, 1, 1], [1090, 1091, 1092, 1093, 1094]],
             [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
            ([[1100, 1101, 1, 1], [1110, 1111, 1112, 1113]],
             [[1, 1, 0, 0], [1, 1, 1, 1]]),
        ]
        cuda_expected_batches = [
            ([
                [1010, 1011, 1012, 1013, 1, 1],
                [1020, 1021, 1022, 1023, 1024, 1025],
                [1030, 1031, 1032, 1, 1, 1],
                [1040, 1041, 1042, 1043, 1044, 1],
                [1050, 1051, 1052, 1053, 1, 1],
            ], [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0, 0],
            ]),
            ([
                [1070, 1071, 1072, 1073, 1074, 1075],
                [1080, 1081, 1082, 1, 1, 1],
                [1090, 1091, 1092, 1093, 1094, 1],
                [1100, 1101, 1, 1, 1, 1],
                [1110, 1111, 1112, 1113, 1, 1],
            ], [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
            ]),
        ]
        expected_batches = cpu_expected_batches if dev == "cpu" else cuda_expected_batches
        source_cues = [
            cue(index=i if i % 2 == 0 else i - 1,
                start=float(i), end=float(i + 1),
                text=text,
                warnings=("UPSTREAM", "UPSTREAM") if i == 0 else (),
                custom={"orig_id": i, "meta": f"custom_{i}"})
            for i, (text, _) in enumerate(cue_specs)
        ]
        before_manifest = [copy.deepcopy(vars(s)) for s in source_cues]
        progress = []
        b = OutputCompletionBackend(
            generated_rows=copy.deepcopy(gen_map),
            decode_texts=copy.deepcopy(dec_map),
            device=dev,
        )
        res = b.local(source_cues, "sw", progress_cb=progress.append)
        self.assertEqual(len(res), 13)
        expected_progress = ([15, 30, 46, 61, 76, 92, 100]
                             if dev == "cpu" else [46, 92, 100])
        self.assertEqual(progress, expected_progress, "matrix progress")

        # Restore T2 result assertions without replacing the T3 input assertions.
        expected_results = [
            (before_manifest[0]["text"], ("UPSTREAM",) + WARN_INPUT_BOUNDS),
            ("ترجمة نظيفة 1", (LANGUAGE_WARNING,)),
            (before_manifest[2]["text"], WARN_BUDGET),
            (before_manifest[3]["text"], WARN_INCOMPLETE),
            (before_manifest[4]["text"], WARN_EMPTY),
            ("ترجمة نظيفة 5", (LANGUAGE_WARNING,)),
            (before_manifest[6]["text"], WARN_INPUT_BOUNDS),
            (before_manifest[7]["text"], WARN_BUDGET),
            (before_manifest[8]["text"], WARN_INCOMPLETE_EMPTY),
            ("ترجمة نظيفة 9", (LANGUAGE_WARNING,)),
            (before_manifest[10]["text"], WARN_EMPTY),
            ("ترجمة نظيفة 11", (LANGUAGE_WARNING,)),
            (before_manifest[12]["text"], WARN_INPUT_BOUNDS),
        ]
        for idx, (text, warnings) in enumerate(expected_results):
            self.assertEqual(res[idx].text, text, f"cue {idx} text")
            self.assertEqual(res[idx].warnings, warnings, f"cue {idx} warnings")
            for field in ("index", "start", "end", "custom"):
                self.assertEqual(getattr(res[idx], field), before_manifest[idx][field],
                                 f"cue {idx} {field}")
        self.assertEqual([vars(s) for s in source_cues], before_manifest,
                         "source cues changed")

        preflight_events = [e for e in b.events if e["kind"] == "tokenize"]
        self.assertEqual([e["texts"] for e in preflight_events],
                         [[text] for text, _ in cue_specs])
        expected_flags = dict(add_special_tokens=True, padding=False, truncation=False,
                              return_attention_mask=True, return_token_type_ids=False)
        for event in preflight_events:
            self.assertEqual(event["kwargs"], expected_flags)

        # Expected admission derives from declared fixtures, never observed tensors.
        batch_size = 2 if dev == "cpu" else 6
        expected_pad_encodings = []
        for start in range(0, len(cue_specs), batch_size):
            admitted = [list(CUE_TOKEN_ENCODINGS[text])
                        for text, _ in cue_specs[start:start + batch_size]
                        if not text.startswith("OVERSIZE")]
            if admitted:
                expected_pad_encodings.append(dict(
                    input_ids=admitted,
                    attention_mask=[[1] * len(row) for row in admitted]))
        self.assertEqual([e["encoded"] for e in b.events if e["kind"] == "pad"],
                         expected_pad_encodings)
        self.assertEqual([e["device"] for e in b.events if e["kind"] == "transfer"],
                         [dev] * len(expected_batches))
        self.assertEqual(len([e for e in b.events if e["kind"] == "decode"]), 10)
        final_preflight = max(i for i, e in enumerate(b.events) if e["kind"] == "tokenize")
        self.assertFalse(any(e["kind"] in {"pad", "transfer", "generate", "decode"}
                             for e in b.events[final_preflight + 1:]),
                         "all-oversized final batch must not enter generation")

        # Verify exact input_ids and attention_mask
        generate_events = [e for e in b.events if e["kind"] == "generate"]
        self.assertEqual(len(generate_events), len(expected_batches))
        for gen_event, (exp_ids, exp_masks) in zip(generate_events, expected_batches):
            self.assertEqual(gen_event["input_ids"], exp_ids)
            self.assertEqual(gen_event["attention_mask"], exp_masks)
            self.assertEqual(gen_event["kwargs"], dict(
                forced_bos_token_id=256047, max_new_tokens=140,
                num_beams=2, do_sample=False))

    def test_05_mixed_batches_cardinality_and_warning_immutability(self):
        """5. 13-cue mixed anomaly sequence on CPU2 and fake CUDA6, verifying progress, ordering, and immutability."""
        for dev in ["cpu", "cuda"]:
            with self.subTest(device=dev):
                self._run_matrix_check(dev=dev)

        # All-oversized control: 0 admitted cues, 0 generate calls
        b_all_over = OutputCompletionBackend()
        res_all_over = b_all_over.local([
            cue(text="OVERSIZE " + ("word " * 150)),
            cue(text="OVERSIZE " + ("word " * 150))
        ], "en")
        self.assertEqual(len(res_all_over), 2)
        self.assertEqual(res_all_over[0].warnings, WARN_INPUT_BOUNDS)
        self.assertEqual(res_all_over[1].warnings, WARN_INPUT_BOUNDS)
        self.assertFalse(any(e["kind"] in {"pad", "transfer", "generate", "decode"}
                             for e in b_all_over.events))

    def test_06_sanitizer_defense_and_timing_preservation(self):
        """6. Retained nonempty source cues survive sanitize_timing_without_drift."""
        from services.subtitles import sanitize_timing_without_drift

        backend = OutputCompletionBackend(
            generated_rows={0: [2, 256047] + [7] * 139},
        )
        cues = [
            cue(index=1, start=1.0, end=3.0, text="Important Dialogue Line", warnings=()),
        ]
        translated = backend.local(cues, "en")
        self.assertEqual(translated[0].text, "Important Dialogue Line")
        self.assertEqual(translated[0].warnings, WARN_BUDGET)

        # Pass through actual sanitizer
        sanitized = sanitize_timing_without_drift(translated)
        self.assertEqual(len(sanitized), 1)
        self.assertEqual(sanitized[0].text, "Important Dialogue Line")
        self.assertEqual(sanitized[0].start, 1.0)
        self.assertEqual(sanitized[0].end, 3.0)

    def test_07_resource_lifecycle_and_cleanup(self):
        """7. Verify gc.collect and cuda.empty_cache called under success, error, and cancellation."""
        # Success path
        backend = OutputCompletionBackend()
        backend.local([cue(text="Hello")], "en")
        self.assertGreaterEqual(backend.collections, 1)
        self.assertGreaterEqual(backend.cuda_clears, 1)

        # Error path
        backend_err = OutputCompletionBackend(fault="decode_raises")
        with self.assertRaises(ValueError):
            backend_err.local([cue(text="Hello")], "en")
        self.assertGreaterEqual(backend_err.collections, 1)
        self.assertGreaterEqual(backend_err.cuda_clears, 1)

        # Synthetic cancellation path at awaited worker boundary
        backend_cancel = OutputCompletionBackend()
        async def cancelled(*args):
            raise asyncio.CancelledError()
        backend_cancel.namespace["asyncio"] = SimpleNamespace(to_thread=cancelled)
        with self.assertRaises(asyncio.CancelledError):
            backend_cancel.local([cue(text="Hello")], "en")
        self.assertGreaterEqual(backend_cancel.collections, 1)
        self.assertGreaterEqual(backend_cancel.cuda_clears, 1)

    def test_08_in_memory_mutation_controls(self):
        """8. Strict negative-control harness with 8 in-memory AST mutants and unexpected error rejection."""
        source = (ROOT / "services/translator.py").read_text(encoding="utf-8-sig")
        mutants = {
            "ignore_bos": [
                ("gen_step = first_eos_idx + 1", "gen_step = first_eos_idx"),
                ("active_count = len(active_tokens)", "active_count = len(active_tokens) - 1"),
            ],
            "accept_cap_eos": [
                ("if gen_step == 140:", "if False:"),
            ],
            "remove_empty_guard": [
                ("is_empty = (len(decoded_text) == 0)", "is_empty = False"),
            ],
            "erase_pad_equal_eos": [
                ("gen_tokens = raw_row[1:]", "gen_tokens = [t for t in raw_row[1:] if t != pad_id]"),
            ],
            "skip_invalid_bool_config": [
                ("if val is not None:", "if val is not None and type(val) is not bool:"),
            ],
            "wrong_ordinal": [
                ("ordinal = ordinals[ordinal_idx]", "ordinal = ordinal_idx"),
            ],
            "filter_all_pads": [
                ("active_tokens = list(gen_tokens)", "active_tokens = [t for t in gen_tokens if t != pad_id]"),
            ],
            "allow_start_pad_subclass": [
                ("type(decoder_start_id) is not int", "not isinstance(decoder_start_id, int)"),
                ("type(pad_id) is not int", "not isinstance(pad_id, int)"),
            ],
        }

        # Verify mutation targets exist in source
        for mutant_name, edits in mutants.items():
            for old, _ in edits:
                self.assertIn(old, source, f"Mutation target missing: {old}")

        orig_init = OutputCompletionBackend.__init__

        def run_semantic_assertion(m_name, backend_cls):
            if m_name == "ignore_bos":
                b = backend_cls(generated_rows={0: [2, 256047] + [7] * 139})
                r = b.local([cue(text="Src")], "en")[0]
                self.assertEqual(r.warnings, WARN_BUDGET)
            elif m_name == "accept_cap_eos":
                b = backend_cls(generated_rows={0: [2, 256047] + [7] * 138 + [2]})
                r = b.local([cue(text="Cap")], "en")[0]
                self.assertEqual(r.warnings, WARN_BUDGET)
            elif m_name == "remove_empty_guard":
                b = backend_cls(
                    generated_rows={0: [2, 256047, 7, 2]},
                    decode_texts={tuple([2, 256047, 7, 2]): ""},
                )
                r = b.local([cue(text="Src")], "en")[0]
                self.assertEqual(r.warnings, WARN_EMPTY)
            elif m_name == "erase_pad_equal_eos":
                b = backend_cls(
                    generated_rows={0: [2, 256047, 7, 2, 2, 2]},
                    eos_token_id=2, pad_token_id=2,
                )
                r = b.local([cue(text="PadEOS")], "en")[0]
                self.assertEqual(r.warnings, ())
            elif m_name == "skip_invalid_bool_config":
                b = backend_cls(
                    gen_config_dict={"eos_token_id": True},
                    model_config_dict={"eos_token_id": 2, "decoder_start_token_id": 2, "pad_token_id": 1},
                )
                with self.assertRaises(ValueError) as cm:
                    b.local([cue(text="Bool")], "en")
                self.assertIn("invalid eos_token_id", str(cm.exception))
            elif m_name == "wrong_ordinal":
                b = backend_cls(generated_rows={0: [2, 256047, 7, 2]})
                cues = [cue(text="OVERSIZE " + ("word " * 150)), cue(text="Normal")]
                r = b.local(cues, "en")
                self.assertEqual(r[1].text, "مرحبا ترجمة ناجحة")
            elif m_name == "filter_all_pads":
                b = backend_cls(generated_rows={0: [2, 256047, 1] + [7] * 138})
                r = b.local([cue(text="PadBoundary")], "en")[0]
                self.assertEqual(r.warnings, WARN_BUDGET)
            elif m_name == "allow_start_pad_subclass":
                # Must reject IntChild subclasses for start and pad
                b_start = backend_cls(
                    gen_config_dict={"decoder_start_token_id": IntChild(2)},
                    model_config_dict={"eos_token_id": 2, "decoder_start_token_id": 2, "pad_token_id": 1},
                )
                with self.assertRaises(ValueError) as cm1:
                    b_start.local([cue(text="SubStart")], "en")
                self.assertIn("decoder_start_token_id", str(cm1.exception))

                b_pad = backend_cls(
                    gen_config_dict={"pad_token_id": IntChild(1)},
                    model_config_dict={"eos_token_id": 2, "decoder_start_token_id": 2, "pad_token_id": 1},
                )
                with self.assertRaises(ValueError) as cm2:
                    b_pad.local([cue(text="SubPad")], "en")
                self.assertIn("invalid pad_token_id", str(cm2.exception))

        # Baseline: all 8 semantic assertions pass against unmutated production code
        for mutant_name in mutants:
            with self.subTest(unmutated_baseline=mutant_name):
                run_semantic_assertion(mutant_name, OutputCompletionBackend)

        def run_mutant_check(m_name, m_edits, inject_error=None):
            mutated = source
            for old, new in m_edits:
                mutated = mutated.replace(old, new)
            parsed_tree = ast.parse(mutated)
            method_ast = next(
                n for n in next(
                    n for n in parsed_tree.body
                    if isinstance(n, ast.ClassDef) and n.name == "UnifiedTranslator"
                ).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "_translate_nllb_local"
            )

            def mutating_init(inst, *args, _method=method_ast, **kwargs):
                orig_init(inst, *args, **kwargs)
                LANG["execute"]([copy.deepcopy(_method)], inst.namespace)
                inst.translator._translate_nllb_local = inst.namespace["_translate_nllb_local"].__get__(inst.translator)
                if inject_error is not None:
                    async def faulty_translate(*a, **kw):
                        raise inject_error
                    inst.translator._translate_nllb_local = faulty_translate

            OutputCompletionBackend.__init__ = mutating_init
            try:
                run_semantic_assertion(m_name, OutputCompletionBackend)
            finally:
                OutputCompletionBackend.__init__ = orig_init

        # Verify all 8 mutants fail with an AssertionError (strict semantic failure)
        for mutant_name, edits in mutants.items():
            with self.subTest(mutant_detection=mutant_name):
                with self.assertRaises(AssertionError, msg=f"Mutant {mutant_name} failed to trigger AssertionError"):
                    run_mutant_check(mutant_name, edits)

        # Runner self-test: verify that injecting RuntimeError into run_mutant_check propagates across all 8 checks
        for mutant_name, edits in mutants.items():
            with self.subTest(runner_runtime_error_propagation=mutant_name):
                with self.assertRaises(RuntimeError):
                    run_mutant_check(mutant_name, edits, inject_error=RuntimeError("injected runner failure"))

    def test_09_upstream_duplicate_warning_deduplication_and_shape_preservation(self):
        """9. Upstream duplicate warnings deduplicated by first occurrence across retained and clean cues."""
        # A: Retained cue with duplicates and new warnings
        src_retained = cue(warnings=("UP", "UP", "SOURCE_RETAINED_FOR_REVIEW"))
        b_ret = OutputCompletionBackend(generated_rows={0: [2, 256047] + [7] * 139})
        out_ret = b_ret.local([src_retained], "sw")[0]
        self.assertEqual(out_ret.warnings, ("UP", "SOURCE_RETAINED_FOR_REVIEW", "OUTPUT_TOKEN_BUDGET_EXCEEDED"))
        self.assertEqual(src_retained.warnings, ("UP", "UP", "SOURCE_RETAINED_FOR_REVIEW"))

        # B: Clean translated cue with duplicates, no new warnings
        src_clean = cue(warnings=("UP", "UP"))
        out_clean = OutputCompletionBackend().local([src_clean], "en")[0]
        self.assertEqual(out_clean.warnings, ("UP",))

        # C: Clean translated cue with absent warnings attribute
        raw_clean = SimpleNamespace(text="Clean without warnings attribute")
        out_raw = OutputCompletionBackend().local([raw_clean], "en")[0]
        self.assertFalse(hasattr(out_raw, "warnings"))

        # D: Clean translated cue with empty tuple warnings attribute
        empty_clean = SimpleNamespace(text="Clean with empty tuple", warnings=())
        out_empty = OutputCompletionBackend().local([empty_clean], "en")[0]
        self.assertEqual(out_empty.warnings, ())

    def test_10_exact_integer_scalar_eos_rejection(self):
        """10. Exact-int validation rejects int subclass on scalar EOS, start ID, and pad ID."""
        # EOS IntChild rejection
        b_subclass_eos = OutputCompletionBackend(eos_token_id=IntChild(2))
        with self.assertRaises(ValueError) as cm_eos:
            b_subclass_eos.local([cue(text="Test")], "en")
        self.assertIn("invalid eos_token_id", str(cm_eos.exception))

        # decoder_start_token_id IntChild rejection
        b_subclass_start = OutputCompletionBackend(
            decoder_start_token_id=IntChild(2),
            config_level="config"
        )
        with self.assertRaises(ValueError) as cm_start:
            b_subclass_start.local([cue(text="Test")], "en")
        self.assertIn("decoder_start_token_id", str(cm_start.exception))

        # pad_token_id IntChild rejection
        b_subclass_pad = OutputCompletionBackend(
            pad_token_id=IntChild(1),
            config_level="config"
        )
        with self.assertRaises(ValueError) as cm_pad:
            b_subclass_pad.local([cue(text="Test")], "en")
        self.assertIn("invalid pad_token_id", str(cm_pad.exception))

    def test_11_worker_result_dictionary_contract(self):
        """11. Internal worker result dictionary covers all 5 outcomes with exact keys, types, and warnings."""
        captured_results = []
        async def worker_spy(fn, *args):
            results = fn(*args)
            captured_results.extend(copy.deepcopy(results))
            return results

        # 5 cues: translated, input rejected, capped, incomplete, empty
        b = OutputCompletionBackend(
            generated_rows={
                0: [2, 256047, 701, 2],                 # Admitted 0 (Cue 0): Translated
                1: [2, 256047] + [7] * 139,             # Admitted 1 (Cue 2): Retained budget
                2: [2, 256047, 7],                       # Admitted 2 (Cue 3): Retained incomplete
                3: [2, 256047, 777, 2],                 # Admitted 3 (Cue 4): Retained empty
            },
            decode_texts={
                tuple([2, 256047, 701, 2]): "ترجمة نظيفة",
                tuple([2, 256047, 777, 2]): "",
            },
            device="cuda"  # batch size 6 -> single batch for all 5 cues
        )
        b.namespace["asyncio"] = SimpleNamespace(to_thread=worker_spy)

        cues = [
            cue(text="C1", warnings=("UPSTREAM",)),             # Translated
            cue(text="OVERSIZE " + ("word " * 150)),            # Input bounds rejected
            cue(text="C3"),                                     # Retained budget
            cue(text="C4"),                                     # Retained incomplete
            cue(text="C5"),                                     # Retained empty
        ]
        out = b.local(cues, "sw")
        self.assertEqual(len(out), 5)

        # Worker inspection: exactly 5 items
        self.assertEqual(len(captured_results), 5)
        for r in captured_results:
            self.assertEqual(set(r.keys()), {"action", "text", "warnings"})
            self.assertIsInstance(r["warnings"], tuple)

        # Row 0: Translated -> action TRANSLATED, receives language warning only
        self.assertEqual(captured_results[0]["action"], "TRANSLATED")
        self.assertEqual(captured_results[0]["text"], "ترجمة نظيفة")
        self.assertEqual(captured_results[0]["warnings"], (LANGUAGE_WARNING,))

        # Row 1: Input rejected -> action RETAIN_SOURCE, text None, input bounds warnings
        self.assertEqual(captured_results[1]["action"], "RETAIN_SOURCE")
        self.assertIsNone(captured_results[1]["text"])
        self.assertEqual(captured_results[1]["warnings"], WARN_INPUT_BOUNDS)

        # Row 2: Retained budget -> action RETAIN_SOURCE, text None, budget warnings
        self.assertEqual(captured_results[2]["action"], "RETAIN_SOURCE")
        self.assertIsNone(captured_results[2]["text"])
        self.assertEqual(captured_results[2]["warnings"], WARN_BUDGET)

        # Row 3: Retained incomplete -> action RETAIN_SOURCE, text None, incomplete warnings
        self.assertEqual(captured_results[3]["action"], "RETAIN_SOURCE")
        self.assertIsNone(captured_results[3]["text"])
        self.assertEqual(captured_results[3]["warnings"], WARN_INCOMPLETE)

        # Row 4: Retained empty -> action RETAIN_SOURCE, text None, empty warnings
        self.assertEqual(captured_results[4]["action"], "RETAIN_SOURCE")
        self.assertIsNone(captured_results[4]["text"])
        self.assertEqual(captured_results[4]["warnings"], WARN_EMPTY)

        # In outer returned cues: upstream warning deduplicated with worker result
        self.assertEqual(out[0].warnings, ("UPSTREAM", LANGUAGE_WARNING))
        self.assertEqual(out[1].warnings, WARN_INPUT_BOUNDS)

    def test_12_additional_config_and_malformed_row_cases(self):
        """12. Comprehensive table-driven matrix for config resolution, pad handling, and malformed rows."""
        # A: Dictionary sources and explicit None fallthrough
        b_dict = OutputCompletionBackend(
            gen_config_dict={"eos_token_id": None, "decoder_start_token_id": None},
            model_config_dict={"eos_token_id": 2, "decoder_start_token_id": 2, "pad_token_id": 1},
        )
        self.assertEqual(b_dict.local([cue(text="Dict")], "en")[0].warnings, ())

        # B: Tokenizer-only fallback when both configs absent
        b_tok = OutputCompletionBackend(config_level="tokenizer", eos_token_id=2, decoder_start_token_id=2, pad_token_id=1)
        self.assertEqual(b_tok.local([cue(text="Tok")], "en")[0].warnings, ())

        # C: Valid EOS collection representations (list, tuple, set)
        for eos_coll in [[2], (2,), {2}]:
            with self.subTest(eos_coll=type(eos_coll)):
                b_coll = OutputCompletionBackend(eos_token_id=eos_coll)
                self.assertEqual(b_coll.local([cue(text="Coll")], "en")[0].warnings, ())

        # Multi-ID EOS collection [2, 3] with generation terminating on token 3
        b_multi_eos = OutputCompletionBackend(
            eos_token_id=[2, 3],
            pad_token_id=1,
            decoder_start_token_id=2,
            generated_rows={0: [2, 256047, 7, 3, 1, 1]}
        )
        res_multi = b_multi_eos.local([cue(text="MultiEOS")], "en")
        self.assertEqual(res_multi[0].warnings, ())

        # D: Pad None allows clean completion when output ends at EOS without extra suffix
        b_pad_none = OutputCompletionBackend(pad_token_id=None, generated_rows={0: [2, 256047, 7, 2]})
        self.assertEqual(b_pad_none.local([cue(text="PadNone")], "en")[0].warnings, ())

        # Pad None raises if extra suffix follows EOS
        b_pad_none_suffix = OutputCompletionBackend(pad_token_id=None, generated_rows={0: [2, 256047, 7, 2, 88]})
        with self.assertRaises(ValueError) as cm:
            b_pad_none_suffix.local([cue(text="PadNoneBad")], "en")
        self.assertIn("pad token ID", str(cm.exception))

        # E: Invalid effective values raise even if lower-priority valid value exists
        valid_lower = {"eos_token_id": 2, "decoder_start_token_id": 2, "pad_token_id": 1}
        # Paired lower-only success:
        b_lower_only = OutputCompletionBackend(gen_config_dict={}, model_config_dict=copy.deepcopy(valid_lower))
        self.assertEqual(b_lower_only.local([cue(text="LowerOnly")], "en")[0].warnings, ())

        invalid_precedences = [
            ({"eos_token_id": True}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": False}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": -1}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": -99}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": "invalid"}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": []}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": ()}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": set()}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": [2, -1]}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": [2, True]}, valid_lower, "invalid eos_token_id"),
            ({"eos_token_id": IntChild(2)}, valid_lower, "invalid eos_token_id"),
            ({"decoder_start_token_id": True}, valid_lower, "decoder_start_token_id"),
            ({"decoder_start_token_id": False}, valid_lower, "decoder_start_token_id"),
            ({"decoder_start_token_id": -1}, valid_lower, "decoder_start_token_id"),
            ({"decoder_start_token_id": "bad"}, valid_lower, "decoder_start_token_id"),
            ({"decoder_start_token_id": IntChild(2)}, valid_lower, "decoder_start_token_id"),
            ({"pad_token_id": True}, valid_lower, "invalid pad_token_id"),
            ({"pad_token_id": False}, valid_lower, "invalid pad_token_id"),
            ({"pad_token_id": -1}, valid_lower, "invalid pad_token_id"),
            ({"pad_token_id": "bad"}, valid_lower, "invalid pad_token_id"),
            ({"pad_token_id": IntChild(1)}, valid_lower, "invalid pad_token_id"),
        ]
        for g_dict, m_dict, expected_err in invalid_precedences:
            with self.subTest(invalid_higher=g_dict, expected=expected_err):
                b_inv = OutputCompletionBackend(gen_config_dict=g_dict, model_config_dict=m_dict)
                with self.assertRaises(ValueError) as cm:
                    b_inv.local([cue(text="Invalid")], "en")
                self.assertIn(expected_err, str(cm.exception))
                self.assertFalse(any(e["kind"] == "generate" for e in b_inv.events))

        # F: Malformed rows
        malformed_rows = [
            ([], "empty row"),
            ([2, True, 2], "boolean in row"),
            ([2, -1, 2], "negative in row"),
            ([2, [7], 2], "nested list in row"),
            ([2, 256047, 2, 88], "non-padding following EOS"),
        ]
        for row, desc in malformed_rows:
            with self.subTest(malformed_row=desc):
                b_m = OutputCompletionBackend(generated_rows={0: row})
                with self.assertRaises(ValueError):
                    b_m.local([cue()], "en")

        # G: Plain list output (no .tolist() method on generated container)
        b_plain = OutputCompletionBackend(generated_rows={0: [2, 256047, 7, 2]}, return_plain_list=True)
        res_plain = b_plain.local([cue(text="Plain")], "en")
        self.assertEqual(res_plain[0].warnings, ())
        self.assertEqual(res_plain[0].text, "مرحبا ترجمة ناجحة")

        # H: Decode fault
        b_fault = OutputCompletionBackend(fault="decode_non_string")
        with self.assertRaises(ValueError):
            b_fault.local([cue()], "en")

    def test_13_provider_failure_fallback_to_local_output_anomalies(self):
        """13. Real provider failure fallback across 9 cases (3 providers x 3 output anomalies)."""
        providers = ["gemini", "groq", "qwen"]
        anomalies = [
            ([2, 256047] + [7] * 139, WARN_BUDGET),
            ([2, 256047, 7], WARN_INCOMPLETE),
            ([2, 256047, 7, 2], WARN_EMPTY),
        ]

        for provider in providers:
            for row, expected_warns in anomalies:
                with self.subTest(provider=provider, warns=expected_warns):
                    dec = {tuple([2, 256047, 7, 2]): ""} if expected_warns == WARN_EMPTY else {}
                    b = OutputCompletionBackend(generated_rows={0: row}, decode_texts=dec)
                    b.status = 503
                    out = b.run(b.translator.translate([cue(text="source dialogue")], "sw", provider))[0]
                    self.assertEqual(out.text, "source dialogue")
                    self.assertEqual(out.warnings, expected_warns)
                    self.assertTrue(any(c[0] == "fake_http" for c in b.calls))

    def test_14_retained_output_pipeline_export_and_qc_integration(self):
        """14. Full pipeline export, SRT/ASS/VTT timestamps, and exact QC issue ledger assertions."""
        qc = runpy.run_path(str(ROOT / "tests/test_qc_persistence.py"), run_name="review_qc_facade")
        ns = qc["_pipeline_ns"]

        anomaly_cases = [
            ([2, 256047] + [7] * 139, "translation", WARN_BUDGET),
            ([2, 256047, 7], "translation", WARN_INCOMPLETE),
            ([2, 256047, 7, 2], " ", WARN_EMPTY),
        ]

        for row, decoded, warnings in anomaly_cases:
            with self.subTest(warnings=warnings):
                b = OutputCompletionBackend(generated_rows={0: row}, decode_texts={tuple(row): decoded})
                with tempfile.TemporaryDirectory(prefix="aix-qc-test-") as temp:
                    outdir = Path(temp)
                    source_srt = outdir / "source.srt"
                    source_srt.write_text("1\n00:00:01,250 --> 00:00:02,750\nOriginal Text\n", encoding="utf-8")

                    async def translate_mock(segments, **kwargs):
                        kwargs.pop("progress_cb", None)
                        out = await b.translator.translate(segments, **kwargs)
                        return ns["sanitize_timing_without_drift"](out)

                    ns.update(
                        settings=SimpleNamespace(OUTPUT_DIR=outdir, TEMP_DIR=outdir),
                        translator=SimpleNamespace(translate=translate_mock),
                        job_manager=ns["JobManager"](),
                        gpu_lock=asyncio.Lock(),
                    )
                    job_id = "12345678-1234-4234-8234-123456789032"
                    ns["job_manager"].create_job(job_id, "source.srt")
                    b.run(ns["run_subtitle_translation_pipeline"](job_id, source_srt))

                    job = ns["job_manager"].jobs[job_id]
                    self.assertEqual(job["status"], "completed_with_warnings")
                    self.assertEqual(job["qc_report_status"], "written")
                    self.assertEqual(job["warnings_count"], len(warnings))

                    # Exact timestamps and text in SRT, ASS, VTT
                    srt_content = (outdir / f"{job_id}.srt").read_text(encoding="utf-8-sig")
                    self.assertIn("00:00:01,250 --> 00:00:02,750", srt_content)
                    self.assertIn("Original Text", srt_content)

                    vtt_content = (outdir / f"{job_id}.vtt").read_text(encoding="utf-8-sig")
                    self.assertIn("00:00:01.250 --> 00:00:02.750", vtt_content)
                    self.assertIn("Original Text", vtt_content)

                    ass_content = (outdir / f"{job_id}.ass").read_text(encoding="utf-8-sig")
                    self.assertIn("0:00:01.25,0:00:02.75", ass_content)
                    self.assertIn("Original Text", ass_content)

                    # Exact QC issue-ledger assertions
                    report = json.loads((outdir / f"{job_id}.qc.json").read_text(encoding="utf-8"))
                    self.assertIn("issue_ledger", report)
                    self.assertEqual(len(report["issue_ledger"]), 1)
                    first_issue = report["issue_ledger"][0]
                    self.assertEqual(first_issue["ordinal"], 1)
                    self.assertEqual(first_issue["source_index"], 1)
                    self.assertEqual(first_issue["start_seconds"], 1.25)
                    self.assertEqual(first_issue["end_seconds"], 2.75)
                    self.assertEqual(first_issue["duration_seconds"], 1.5)
                    self.assertEqual(first_issue["warnings"], list(warnings))
                    self.assertEqual(first_issue["diagnostic_flags"], ["SOURCE_RETENTION_REVIEW"])

                    # Exact QC summary assertions against schema
                    expected_summary = {
                        "total_final_cues": 1,
                        "clean_cues": 0,
                        "warned_cues": 1,
                        "warning_occurrences": len(warnings),
                        "source_retention_flagged_cues": 1,
                        "ass_source_retention_flagged_cues": 0,
                        "any_retention_flagged_cues": 1,
                        "rejected_blocks_count": 0,
                        "review_status": "REVIEW_REQUIRED"
                    }
                    self.assertEqual(report["summary"], expected_summary)

    def test_15_multi_batch_final_one_admitted_cue_control(self):
        """15. Multi-batch sequence ending with a single admitted cue verifies exact ordinal lookup and no silent row reuse."""
        r0 = [2, 256047, 701, 2]
        r1 = [2, 256047, 702, 2]
        r2 = [2, 256047, 703, 2]
        decodes = {
            tuple(r0): "ترجمة دفعة 1 cue 0",
            tuple(r1): "ترجمة دفعة 1 cue 1",
            tuple(r2): "ترجمة دفعة 2 cue 2",
        }
        b = OutputCompletionBackend(
            generated_rows={0: r0, 1: r1, 2: r2},
            decode_texts=decodes,
            device="cpu"
        )
        cues = [cue(text="C0"), cue(text="C1"), cue(text="C2")]
        res = b.local(cues, "en")
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0].text, "ترجمة دفعة 1 cue 0")
        self.assertEqual(res[1].text, "ترجمة دفعة 1 cue 1")
        # Final one-item batch MUST receive row 2, NOT row 0:
        self.assertEqual(res[2].text, "ترجمة دفعة 2 cue 2")

        # Negative control: missing row 2 must raise KeyError (never silently reuse row 0)
        b_missing = OutputCompletionBackend(
            generated_rows={0: r0, 1: r1},  # missing key 2
            decode_texts=decodes,
            device="cpu"
        )
        with self.assertRaises(KeyError):
            b_missing.local(cues, "en")

    def test_16_generation_input_reversal_and_mask_corruption_controls(self):
        """16. Strict negative controls detect in-memory input reversal and mask corruption after validation."""
        source = (ROOT / "services/translator.py").read_text(encoding="utf-8-sig")

        # Baseline: matrix check passes unmutated
        self._run_matrix_check(dev="cpu")
        self._run_matrix_check(dev="cuda")

        # Mutant A: Input reversal after validation (inputs = inputs.to(self.device))
        reversal_mutant = [
            (
                "inputs = inputs.to(self.device)",
                "inputs = inputs.to(self.device)\n                inputs['input_ids'] = list(reversed(inputs['input_ids']))\n                inputs['attention_mask'] = list(reversed(inputs['attention_mask']))"
            )
        ]

        # Mutant B: Attention mask corruption after validation
        mask_corruption_mutant = [
            (
                "inputs = inputs.to(self.device)",
                "inputs = inputs.to(self.device)\n                if len(inputs['attention_mask']) > 1:\n                    inputs['attention_mask'][1] = [0] * len(inputs['attention_mask'][1])"
            )
        ]

        orig_init = OutputCompletionBackend.__init__

        def run_with_mutant(mutant_edits):
            mutated = source
            for old, new in mutant_edits:
                mutated = mutated.replace(old, new)
            parsed_tree = ast.parse(mutated)
            method_ast = next(
                n for n in next(
                    n for n in parsed_tree.body
                    if isinstance(n, ast.ClassDef) and n.name == "UnifiedTranslator"
                ).body
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "_translate_nllb_local"
            )

            def mutating_init(inst, *args, _method=method_ast, **kwargs):
                orig_init(inst, *args, **kwargs)
                LANG["execute"]([copy.deepcopy(_method)], inst.namespace)
                inst.translator._translate_nllb_local = inst.namespace["_translate_nllb_local"].__get__(inst.translator)

            OutputCompletionBackend.__init__ = mutating_init
            try:
                self._run_matrix_check(dev="cpu")
            finally:
                OutputCompletionBackend.__init__ = orig_init

        # Verify Mutant A (Input Reversal) triggers AssertionError
        with self.assertRaises(AssertionError, msg="Input reversal mutant escaped detection!"):
            run_with_mutant(reversal_mutant)

        # Verify Mutant B (Mask Corruption) triggers AssertionError
        with self.assertRaises(AssertionError, msg="Mask corruption mutant escaped detection!"):
            run_with_mutant(mask_corruption_mutant)

    def test_17_full_three_tier_config_precedence_and_none_fallthrough(self):
        """17. Conflicting 3-tier precedence (gen_config > model.config > tokenizer) and explicit-None fallthrough."""
        triples = [
            dict(eos_token_id=20, decoder_start_token_id=2, pad_token_id=10),
            dict(eos_token_id=30, decoder_start_token_id=3, pad_token_id=15),
            dict(eos_token_id=40, decoder_start_token_id=4, pad_token_id=20),
        ]
        for representation in ("dictionary", "attributes"):
            for winning_tier in range(3):
                with self.subTest(representation=representation, winning_tier=winning_tier):
                    configs = [dict.fromkeys(triples[i], None) if i < winning_tier
                               else copy.deepcopy(triples[i]) for i in range(3)]
                    gen, model, tokenizer = configs
                    if representation == "attributes":
                        gen, model = SimpleNamespace(**gen), SimpleNamespace(**model)
                    winner = triples[winning_tier]
                    row = [winner["decoder_start_token_id"], 256047, 7,
                           winner["eos_token_id"], winner["pad_token_id"], winner["pad_token_id"]]
                    b = OutputCompletionBackend(gen_config_dict=gen, model_config_dict=model,
                                                tokenizer_config=tokenizer, generated_rows={0: row})
                    result = b.local([cue(text="Tier winner")], "en")[0]
                    self.assertEqual(result.warnings, ())
                    self.assertEqual(result.text, "مرحبا ترجمة ناجحة")

    def test_18_matrix_output_metadata_progress_and_immutability_controls(self):
        """18. Each restored assertion rejects its own isolated fake-boundary corruption."""
        original_local = OutputCompletionBackend.local
        cases = [
            ("text", "cue 4 text"), ("warnings", "cue 8 warnings"),
            ("index", "cue 9 index"), ("start", "cue 9 start"),
            ("end", "cue 9 end"), ("custom", "cue 9 custom"),
            ("progress", "matrix progress"), ("source", "source cues changed"),
        ]
        for dev in ("cpu", "cuda"):
            self._run_matrix_check(dev)
            for category, assertion_message in cases:
                def corrupt(backend, cues, *args, _category=category, **kwargs):
                    if _category == "progress":
                        kwargs.pop("progress_cb", None)
                    output = original_local(backend, cues, *args, **kwargs)
                    if _category == "text":
                        output[4].text = "CORRUPTED"
                    elif _category == "warnings":
                        output[8].warnings = ()
                    elif _category in {"index", "start", "end", "custom"}:
                        setattr(output[9], _category,
                                {"corrupt": True} if _category == "custom" else -99)
                    elif _category == "source":
                        cues[0].text = "MUTATED_SOURCE"
                    return output
                with self.subTest(device=dev, corruption=category):
                    with patch.object(OutputCompletionBackend, "local", corrupt):
                        with self.assertRaisesRegex(AssertionError, assertion_message):
                            self._run_matrix_check(dev)


if __name__ == "__main__":
    unittest.main()
