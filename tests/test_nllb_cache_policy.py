"""Offline cache selection and loader argument policy for NLLB."""
import runpy
import unittest
from pathlib import Path
from unittest.mock import Mock

SHARED = runpy.run_path(str(Path(__file__).with_name("test_translation_fallback_policy.py")))
PIN = "a3e77be725cf30383f1faeb8d2f0b0ea98ac554e"
REPO = "facebook/nllb-200-distilled-600M"


class TestNllbCachePolicy(unittest.TestCase):
    def setUp(self):
        self.backend = SHARED["SyntheticBackend"]()
        hub = self.backend.hub
        hub.try_to_load_from_cache = Mock(wraps=hub.try_to_load_from_cache)
        self.probe = hub.try_to_load_from_cache
        self.files = self.backend.cache_files
        self.weight = self.backend.weight_fixture
        modules = self.backend.transformers
        for factory in (modules.AutoConfig, modules.GenerationConfig, modules.AutoModelForSeq2SeqLM):
            factory.from_pretrained = Mock(wraps=factory.from_pretrained)
        self.config = modules.AutoConfig.from_pretrained
        self.model = modules.AutoModelForSeq2SeqLM.from_pretrained
        self.generation = modules.GenerationConfig
        self.generation.from_model_config = Mock(wraps=self.generation.from_model_config)

    def translate(self):
        return self.backend.local([SHARED["cue"]()], "en")

    def model_args(self):
        self.model.assert_called_once()
        self.assertEqual(self.model.call_args.args, (REPO,))
        kwargs = self.model.call_args.kwargs
        self.assertIs(kwargs["local_files_only"], True)
        self.assertIs(kwargs["use_safetensors"], True)
        self.assertEqual(kwargs["torch_dtype"], "FAKE_DTYPE")
        return kwargs

    def test_main_single_and_sharded(self):
        for name in ("model.safetensors", "model.safetensors.index.json"):
            with self.subTest(name=name):
                self.files.clear()
                self.files["main", name] = self.weight
                self.probe.reset_mock()
                self.model.reset_mock()
                self.translate()
                self.assertNotIn("revision", self.model_args())
                self.assertNotIn("generation_config", self.model_args())
                self.assertTrue(all(c.kwargs == {"revision": "main"} for c in self.probe.call_args_list))
        self.config.assert_not_called()

    def use_pin(self):
        self.files["main", "model.safetensors"] = object()
        self.files["main", "model.safetensors.index.json"] = self.weight.with_name("stale")
        self.files[PIN, "model.safetensors"] = self.weight

    def test_pin_both_generation_sources(self):
        self.use_pin()
        for has_file in (False, True):
            with self.subTest(has_file=has_file):
                if has_file:
                    self.files["main", "generation_config.json"] = self.weight
                self.config.reset_mock()
                self.model.reset_mock()
                result = self.translate()
                self.assertEqual((result[0].start, result[0].end), (1.25, 2.75))
                self.config.assert_called_once_with(REPO, local_files_only=True)
                kwargs = self.model_args()
                self.assertEqual(kwargs["revision"], PIN)
                self.assertIsNot(kwargs["config"], self.backend.main_config)
                self.assertEqual(kwargs["config"]._commit_hash, PIN)
                self.assertEqual(self.backend.main_config._commit_hash, "main-commit")
                self.assertEqual(kwargs["generation_config"].source, "main" if has_file else PIN)
        self.generation.from_pretrained.assert_called_once_with(REPO, local_files_only=True)
        self.generation.from_model_config.assert_called_once()

    def test_missing_restores_state_and_runs_cleanup(self):
        self.files.clear()
        cleanup = Mock()
        self.backend.namespace["gc"].collect = cleanup
        self.backend.translator.device = "cuda"
        with self.assertRaisesRegex(RuntimeError, "Missing local NLLB safetensors"):
            self.backend.local([SHARED["cue"]()], "en", device="cpu")
        self.assertEqual((self.backend.translator.device, self.backend.translator.dtype), ("cuda", "FAKE_DTYPE"))
        cleanup.assert_called_once()
        self.model.assert_not_called()

    def test_corrupt_generation_config_is_not_replaced(self):
        self.use_pin()
        self.files["main", "generation_config.json"] = self.weight
        error = OSError("bad generation JSON")
        self.generation.from_pretrained.side_effect = error
        with self.assertRaises(OSError) as caught:
            self.translate()
        self.assertIs(caught.exception, error)
        self.model.assert_not_called()
        self.generation.from_model_config.assert_not_called()

    def test_selected_model_failure_never_retries(self):
        self.files[PIN, "model.safetensors"] = self.weight
        error = OSError("bad checkpoint")
        self.model.side_effect = error
        with self.assertRaises(OSError) as caught:
            self.translate()
        self.assertIs(caught.exception, error)
        self.model_args()
        self.probe.assert_called_once_with(REPO, "model.safetensors", revision="main")
