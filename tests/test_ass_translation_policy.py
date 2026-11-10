"""AIX-ASS-SAFE-001 v1: stdlib-only ASS safety policy regressions.

Loads only services/subtitles.py, never the application/model configuration.
Run directly with python -B tests/test_ass_translation_policy.py.
"""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "services" / "subtitles.py"
SPEC = importlib.util.spec_from_file_location("subtitle_policy_under_test", SOURCE)
subtitles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subtitles)

WARNING = "ASS_SOURCE_EVENT_RETAINED_FOR_REVIEW"
COMPAT_WARNING = "ASS_INLINE_SCOPE_TRANSFORMATION_REVIEW"
RAW = r"{\i1}Hello{\i0} plain\Nnext{\b1}bold{\b0}"
TRANSLATED = "ترجمة عربية تجريبية"


def document(raw=RAW):
    content = (
        "[Script Info]\nPlayResX: 640\nPlayResY: 360\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0," + raw + "\n"
    )
    return subtitles.parse_ass_content(content)


def event_text(output):
    event = next(line for line in output.splitlines() if line.startswith("Dialogue:"))
    return event.split(",", 9)[9]


class TestAssTranslationPolicy(unittest.TestCase):
    def test_unchanged_raw_event_is_exact_and_clean(self):
        doc = document()
        self.assertEqual(event_text(subtitles.build_ass_content(doc)), RAW)
        self.assertEqual(getattr(doc[0], "warnings", ()), ())

    def test_missing_clean_reference_retains_previous_behavior(self):
        doc = document()
        del doc[0].clean_text
        doc[0].text = TRANSLATED
        self.assertEqual(event_text(subtitles.build_ass_content(doc)), RAW)
        self.assertNotIn(WARNING, getattr(doc[0], "warnings", ()))

    def test_changed_plain_event_is_translated_without_policy_warning(self):
        doc = document("Hello plain world")
        doc[0].text = TRANSLATED
        self.assertEqual(event_text(subtitles.build_ass_content(doc)), TRANSLATED)
        self.assertEqual(getattr(doc[0], "warnings", ()), ())

    def test_changed_protected_events_use_exact_source_fallback(self):
        raw_events = (
            r"{\i1}Hello{\i0} plain",
            RAW,
            r"{\i1}Hello\Nworld",
            r"{\i1}Hello{\i0}{\u1} plain{\u0}\Nnext{\b1}bold{\b0}\Nlast",
            r"Hello\Nworld",
            r"Hello\nworld",
            r"{\pos(10,20)}Hello",
        )
        for raw in raw_events:
            for translation in (TRANSLATED, "موافقة"):
                with self.subTest(raw=raw, translation=translation):
                    doc = document(raw)
                    doc[0].text = translation
                    self.assertEqual(event_text(subtitles.build_ass_content(doc)), raw)
                    self.assertEqual(doc[0].text, translation)
                    self.assertIn(WARNING, doc[0].warnings)
                    self.assertIn(COMPAT_WARNING, doc[0].warnings)

    def test_text_metadata_and_existing_warnings_are_preserved(self):
        doc = document()
        cue = doc[0]
        cue.text = TRANSLATED
        cue.warnings = ("EXISTING_PROVIDER_WARNING",)
        cue.lines = [TRANSLATED]
        before = copy.deepcopy(vars(cue))
        output = subtitles.build_ass_content(doc)
        for key, value in before.items():
            if key != "warnings":
                self.assertEqual(getattr(cue, key), value, key)
        self.assertEqual(cue.warnings, ("EXISTING_PROVIDER_WARNING", WARNING, COMPAT_WARNING))
        self.assertIn("PlayResX: 640", output)
        self.assertIn("Dialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0,", output)

    def test_repeated_export_does_not_duplicate_policy_warnings(self):
        doc = document()
        doc[0].text = TRANSLATED
        doc[0].warnings = ("EXISTING", COMPAT_WARNING)
        first = subtitles.build_ass_content(doc)
        second = subtitles.build_ass_content(doc)
        self.assertEqual(first, second)
        self.assertEqual(doc[0].warnings.count(WARNING), 1)
        self.assertEqual(doc[0].warnings.count(COMPAT_WARNING), 1)
        self.assertIn("EXISTING", doc[0].warnings)

    def test_srt_vtt_stay_translated_when_ass_keeps_source(self):
        doc = document()
        doc[0].text = TRANSLATED
        self.assertEqual(event_text(subtitles.build_ass_content(doc)), RAW)
        self.assertIn(TRANSLATED, subtitles.build_srt_content(doc))
        self.assertIn(TRANSLATED, subtitles.build_vtt_content(doc))
        self.assertEqual(doc[0].text, TRANSLATED)

    def test_generated_event_without_raw_ass_keeps_existing_prefix(self):
        cue = SimpleNamespace(index=1, start=1, end=2, text=TRANSLATED, ass_tags=r"{\i1}")
        output = subtitles.build_ass_content([cue])
        self.assertEqual(event_text(output), r"{\i1}" + TRANSLATED)
        self.assertEqual(getattr(cue, "warnings", ()), ())


if __name__ == "__main__":
    unittest.main()
