"""CinematicSRT v2.0 - Core Unit and Regression Test Suite

Verifies:
1. Canonical divmod timestamping (eliminating the 60.000s rollover defect)
2. Start-time invariance and bounded gap satisfaction
3. Zero artificial overlap creation on small disjoint cues (0-10ms and 20-30ms)
4. True source overlap retention with speaker_id and SOURCE_OVERLAP_REVIEW warning
5. Overlong speech retention with REQUIRES_SOURCE_AWARE_SPLIT warning (independent of overlap)
6. Empty cue isolation from lookahead calculations
7. Strict Arabic line wrapping with 100% CONTENT CONSERVATION (zero words or tokens dropped)
8. Nonpositive max_chars validation
9. Full NLLB language code mapping (FLORES-200)
10. Subtitle parsing (SRT, ASS, and WebVTT with MM:SS.mmm)
11. WebVTT export format compliance
12. Upstream provider warning propagation (PROVIDER_MISSING_TRANSLATION preserved)
13. FFmpeg binary path resolution
"""
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config import settings
from services.subtitles import (
    format_timestamp_srt,
    format_timestamp_ass,
    format_timestamp_vtt,
    wrap_arabic_text,
    sanitize_timing_without_drift,
    build_srt_content,
    build_ass_content,
    build_vtt_content,
    parse_srt_content,
    parse_ass_content,
    parse_vtt_content,
    validate_imported_cues
)
from services.pipeline import run_exclusive_worker
from services.translator import NLLB_LANG_MAP, validate_provider_response, reattach_segment

def cue(start, end, text="Yes", index=1, **meta):
    return SimpleNamespace(index=index, start=start, end=end, text=text, **meta)

class TestCinematicSRTV2(unittest.TestCase):

    def test_01_timestamp_rollover(self):
        """Timestamp 59.9996 rolls over properly to 1 minute across SRT, ASS, and VTT."""
        self.assertEqual(format_timestamp_srt(59.9996), "00:01:00,000")
        self.assertEqual(format_timestamp_ass(59.9996), "0:01:00.00")
        self.assertEqual(format_timestamp_vtt(59.9996), "00:01:00.000")

    def test_02_start_invariance_and_gap_satisfaction(self):
        """Rapid 200ms cues (100ms duration) preserve start invariance and satisfy 50ms min gap."""
        items = [cue(i * 0.2, i * 0.2 + 0.1, index=i + 1) for i in range(10)]
        out = sanitize_timing_without_drift(items, min_gap=0.05)
        self.assertEqual([s.start for s in out], [s.start for s in items])
        gaps = [b.start - a.end for a, b in zip(out, out[1:])]
        self.assertTrue(all(g >= 0.0499 for g in gaps), f"Gaps violated: {gaps}")

    def test_03_zero_artificial_overlap_small_intervals(self):
        """Disjoint small cues (0-10ms and 20-30ms) do not create artificial overlap."""
        items = [cue(0.0, 0.010, index=1), cue(0.020, 0.030, index=2)]
        out = sanitize_timing_without_drift(items)
        self.assertFalse(out[0].is_overlap)
        self.assertFalse(out[1].is_overlap)
        self.assertLessEqual(out[0].end, out[1].start)

    def test_04_true_overlap_retention(self):
        """True source overlap preserves duration, speaker_id, and flags SOURCE_OVERLAP_REVIEW."""
        out = sanitize_timing_without_drift([
            cue(0.0, 5.0, speaker_id="A"),
            cue(1.0, 1.4, index=2, speaker_id="B")
        ])
        self.assertEqual(out[0].end, 5.0)
        self.assertEqual(out[0].speaker_id, "A")
        self.assertTrue(out[0].is_overlap)
        self.assertIn("SOURCE_OVERLAP_REVIEW", out[0].warnings)
        self.assertEqual(out[1].speaker_id, "B")
        self.assertTrue(out[1].is_overlap)

    def test_05_unsplit_long_speech_flagged(self):
        """Unsplit 12-second speech retains duration and flags REQUIRES_SOURCE_AWARE_SPLIT."""
        out = sanitize_timing_without_drift([cue(0.0, 12.0), cue(25.0, 26.0, index=2)], max_duration=7.0)
        self.assertEqual(out[0].end, 12.0)
        self.assertIn("REQUIRES_SOURCE_AWARE_SPLIT", out[0].warnings)

    def test_06_empty_cue_isolation(self):
        """Empty cue is discarded without corrupting lookahead boundary of previous cue."""
        out = sanitize_timing_without_drift([
            cue(0.0, 0.1, "First"),
            cue(0.2, 0.3, "", 2),
            cue(3.0, 3.2, "Next", 3)
        ])
        self.assertGreaterEqual(out[0].end, 0.8)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1].text, "Next")

    def test_07_content_conservation_wrapping(self):
        """Wrapping preserves 100% of words and characters; zero text dropped."""
        # 20 words fixture (179 chars): all 20 words must be present in lines
        raw_words = ["abcdefgh"] * 20
        text = " ".join(raw_words)
        lines = wrap_arabic_text(text, max_chars=42)
        rejoined = " ".join(lines)
        self.assertEqual(rejoined.split(), raw_words)
        for line in lines:
            self.assertLessEqual(len(line), 42)

        # 90-character unbroken token: splits into 42+42+6 = 90 chars total (zero chars lost)
        unbroken = wrap_arabic_text("a" * 90, max_chars=42)
        self.assertEqual(sum(len(l) for l in unbroken), 90)
        self.assertEqual(len(unbroken), 3)

        # Nonpositive max_chars raises ValueError
        with self.assertRaises(ValueError):
            wrap_arabic_text(text, max_chars=0)

    def test_08_upstream_warnings_propagated(self):
        """Upstream warnings like PROVIDER_MISSING_TRANSLATION are retained after timing sanitization."""
        c = cue(0.0, 1.0, text="Fallback text", warnings=("PROVIDER_MISSING_TRANSLATION",))
        out = sanitize_timing_without_drift([c])
        self.assertIn("PROVIDER_MISSING_TRANSLATION", out[0].warnings)

    def test_09_nllb_language_mapping(self):
        """NLLB language codes map accurately for all major source languages."""
        expected = {
            "en": "eng_Latn", "fr": "fra_Latn", "ja": "jpn_Jpan", "ar": "arb_Arab",
            "es": "spa_Latn", "de": "deu_Latn", "it": "ita_Latn", "ko": "kor_Hang",
            "zh": "zho_Hans", "ru": "rus_Cyrl", "tr": "tur_Latn"
        }
        for code, flores_id in expected.items():
            self.assertEqual(NLLB_LANG_MAP.get(code), flores_id)

    def test_10_parsers_srt_ass_vtt(self):
        """SRT, ASS, and WebVTT parsers parse correctly."""
        # SRT
        srt_raw = "1\n00:00:01,000 --> 00:00:04,500\nHello world\n"
        p_srt = parse_srt_content(srt_raw)
        self.assertEqual(p_srt[0].start, 1.0)
        self.assertEqual(p_srt[0].end, 4.5)

        # ASS
        ass_raw = "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 0,0:00:01.00,0:00:04.50,Cinematic,John,0,0,0,,Hello ASS\n"
        p_ass = parse_ass_content(ass_raw)
        self.assertEqual(p_ass[0].speaker_id, "John")

        # WebVTT without hours (MM:SS.mmm)
        vtt_raw = "WEBVTT\n\n00:01.000 --> 00:02.000\nWebVTT cue text\n"
        p_vtt = parse_vtt_content(vtt_raw)
        self.assertEqual(len(p_vtt), 1)
        self.assertEqual(p_vtt[0].start, 1.0)
        self.assertEqual(p_vtt[0].end, 2.0)
        self.assertEqual(p_vtt[0].text, "WebVTT cue text")

    def test_11_media_end_validation(self):
        """Cue start exceeding media_end raises ValueError, and clipping flags CUE_TRUNCATED_AT_MEDIA_END."""
        with self.assertRaises(ValueError):
            sanitize_timing_without_drift([cue(10.0, 11.0)], media_end=5.0)

        # Cue crossing media end gets truncated with warning
        out = sanitize_timing_without_drift([cue(1.0, 10.0)], media_end=5.0)
        self.assertEqual(out[0].end, 5.0)
        self.assertIn("CUE_TRUNCATED_AT_MEDIA_END", out[0].warnings)

    def test_12_vtt_note_identifier_preserved(self):
        """WebVTT cues with identifier like NOTE123 are parsed and not discarded as comments."""
        vtt_with_id = "WEBVTT\n\nNOTE123\n00:01.000 --> 00:02.000\nValid dialog text\n"
        parsed = parse_vtt_content(vtt_with_id)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].text, "Valid dialog text")

    def test_13_nonfinite_timestamps_rejected(self):
        """Non-finite timestamps (NaN, Inf) raise ValueError."""
        import math
        with self.assertRaises(ValueError):
            sanitize_timing_without_drift([cue(math.nan, 2.0)])
        with self.assertRaises(ValueError):
            sanitize_timing_without_drift([cue(1.0, math.inf)])

    def test_14_ffmpeg_resolution(self):
        """FFmpeg binary resolution yields a valid string path."""
        self.assertTrue(len(settings.FFMPEG_BINARY) > 0)

    def test_15_validate_imported_cues_rejects_reversed_interval(self):
        """Reversed interval like 00:00:04,000 --> 00:00:01,000 is rejected with ValueError."""
        reversed_cue = cue(4.0, 1.0, text="Reversed")
        with self.assertRaises(ValueError):
            validate_imported_cues([reversed_cue])

        # Valid interval passes untouched without retiming
        valid = cue(1.0, 1.1, text="Valid")
        validated = validate_imported_cues([valid])
        self.assertEqual(validated[0].start, 1.0)
        self.assertEqual(validated[0].end, 1.1)

    def test_16_ass_document_preserves_custom_styles_and_layers(self):
        """ASS parser and exporter preserve custom layer, style, speaker, margins, and effect."""
        ass_raw = "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0,Hello ASS\n"
        parsed = parse_ass_content(ass_raw)
        self.assertEqual(parsed[0].layer, 3)
        self.assertEqual(parsed[0].style, "Blue")
        self.assertEqual(parsed[0].speaker_id, "Alice")
        self.assertEqual(parsed[0].margin_l, 10)
        self.assertEqual(parsed[0].margin_r, 20)
        self.assertEqual(parsed[0].margin_v, 30)

        # Export and verify preservation
        exported = build_ass_content(parsed)
        self.assertIn("Dialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0,Hello ASS", exported)

    def test_17_exclusive_worker_repeated_cancellation(self):
        """run_exclusive_worker holds GPU lock through repeated cancellations until worker completes."""
        import asyncio

        async def _test():
            finish_event = asyncio.Event()
            lock = asyncio.Lock()
            worker_finished = False

            async def fake_worker():
                nonlocal worker_finished
                await finish_event.wait()
                worker_finished = True
                return "DONE"

            async def pipeline():
                async with lock:
                    await run_exclusive_worker(fake_worker())

            task = asyncio.create_task(pipeline())
            await asyncio.sleep(0.01)
            self.assertTrue(lock.locked())

            # 1st Cancellation
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertTrue(lock.locked(), "Lock MUST remain held after 1st cancellation")

            # 2nd Cancellation
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertTrue(lock.locked(), "Lock MUST remain held after 2nd cancellation")

            # Finish worker
            finish_event.set()
            try:
                await task
            except asyncio.CancelledError:
                pass

            self.assertFalse(lock.locked(), "Lock released only AFTER worker finished")
            self.assertTrue(worker_finished)

        asyncio.run(_test())

    def test_18_ass_metadata_preserved_across_translation_adapter(self):
        """ASS event attributes (layer, style, speaker, margins, effect, tags) are 100% preserved through translation reattachment."""
        source = cue(1.0, 4.5, text="Hello world", index=1, layer=3, style="Blue", speaker_id="Alice", margin_l=10, margin_r=20, margin_v=30, effect="K0", ass_tags=r"{\i1}")
        translated = reattach_segment(source, "مرحبا بالعالم")
        self.assertEqual(translated.layer, 3)
        self.assertEqual(translated.style, "Blue")
        self.assertEqual(translated.speaker_id, "Alice")
        self.assertEqual(translated.margin_l, 10)
        self.assertEqual(translated.margin_r, 20)
        self.assertEqual(translated.margin_v, 30)
        self.assertEqual(translated.effect, "K0")
        self.assertEqual(translated.ass_tags, r"{\i1}")

        # Check export
        exported = build_ass_content([translated])
        self.assertIn(r"Dialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0,{\i1}", exported)

    def test_19_partial_malformed_subtitle_generates_warnings(self):
        """Mixed valid/malformed SRT retains valid cues, records rejected blocks, and attaches warnings so job is never reported clean."""
        raw_srt = "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n2\nBROKEN TIMESTAMP\nSecond dialogue must not silently vanish\n"
        doc = parse_srt_content(raw_srt)
        self.assertEqual(len(doc), 1)
        self.assertEqual(len(doc.rejected_blocks), 1)
        self.assertEqual(doc.rejected_blocks[0]["reason"], "MALFORMED_TIMESTAMP_FORMAT")

        validated = validate_imported_cues(doc)
        self.assertGreaterEqual(len(validated[0].warnings), 1)
        self.assertIn("MALFORMED_TIMESTAMP_FORMAT", validated[0].warnings)

    def test_20_provider_identity_contract_rejects_duplicates_and_unexpected(self):
        """Strict provider validation detects duplicates and unexpected IDs, preventing silent overwrites and tagging warnings."""
        c = cue(0.0, 1.0, text="Source", index=1)
        payload = [
            {"index": 1, "arabic": "First"},
            {"index": 1, "arabic": "Last"},
            {"index": 999, "arabic": "Unexpected"}
        ]
        ans_map, cue_warns, batch_warns = validate_provider_response([c], payload)
        self.assertEqual(ans_map[1], "First", "Must NOT silently accept duplicate last translation")
        self.assertIn("DUPLICATE_PROVIDER_ID", cue_warns[1])
        self.assertIn("UNEXPECTED_PROVIDER_ID", batch_warns)
        self.assertNotIn(999, ans_map)

    def test_21_ass_style_definitions_and_overrides_preserved(self):
        """ASS roundtrip retains original style definitions in [V4+ Styles] and inline overrides in [Events]."""
        raw_ass = """[Script Info]
Title: Original
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour
Style: Blue,Arial,30,&H0000FF00
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0,{\\i1}Hello\\Nworld
"""
        parsed = parse_ass_content(raw_ass)
        exported = build_ass_content(parsed)
        self.assertIn("Style: Blue,Arial,30,&H0000FF00", exported)
        self.assertIn(r"Dialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0,{\i1}", exported)

    def test_22_ass_document_resolution_and_inline_scopes_preserved(self):
        """ASS script resolution (PlayResX/Y), inline tag reset ({\\i0}), and explicit hard breaks (\\N) are preserved."""
        raw_ass = """[Script Info]
Title: Original
PlayResX: 640
PlayResY: 360
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour
Style: Blue,Arial,30,&H0000FF00
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0,{\\i1}Hello{\\i0} plain\\Nnext
"""
        doc = parse_ass_content(raw_ass)
        out = build_ass_content(doc)
        self.assertIn("PlayResX: 640", doc.raw_header)
        self.assertIn("PlayResX: 640", out)
        self.assertNotIn("PlayResX: 1920", out)
        self.assertIn(r"{\i0}", out)
        dialogue_line = next(l for l in out.splitlines() if l.startswith("Dialogue:"))
        self.assertIn(r"\N", dialogue_line)

    def test_23_provider_strict_type_and_empty_payload_validation(self):
        """Provider validation rejects boolean index, list index without TypeError, non-string text, and handles empty payload."""
        c = cue(0.0, 1.0, text="Source", index=1)

        # 1. Boolean True index rejected
        mapping, cue_w, batch_w = validate_provider_response([c], [{"index": True, "arabic": "Boolean"}])
        self.assertNotIn(1, mapping)
        self.assertIn("MALFORMED_PROVIDER_ID_TYPE", batch_w)
        self.assertIn("PROVIDER_MISSING_TRANSLATION", cue_w[1])

        # 2. List index [] rejected without TypeError
        mapping, cue_w, batch_w = validate_provider_response([c], [{"index": [], "arabic": "List"}])
        self.assertIn("MALFORMED_PROVIDER_ID_TYPE", batch_w)

        # 3. Non-string text flagged as INVALID_TRANSLATION_TYPE
        mapping, cue_w, batch_w = validate_provider_response([c], [{"index": 1, "arabic": 42}])
        self.assertNotIn(1, mapping)
        self.assertIn("INVALID_TRANSLATION_TYPE", cue_w[1])

        # 4. Empty payload flags PROVIDER_MISSING_TRANSLATION
        mapping, cue_w, batch_w = validate_provider_response([c], [])
        self.assertIn("PROVIDER_MISSING_TRANSLATION", cue_w[1])

    def test_24_all_adapters_call_shared_validator(self):
        """Source inspection proves _translate_gemini, _translate_groq, and _translate_local_qwen all call validate_provider_response."""
        import ast
        tt = ast.parse((ROOT / "services/translator.py").read_text(encoding="utf-8-sig"))
        tc = next(n for n in tt.body if isinstance(n, ast.ClassDef) and n.name == "UnifiedTranslator")
        adapters = [n for n in tc.body if isinstance(n, ast.AsyncFunctionDef) and n.name in {"_translate_gemini", "_translate_groq", "_translate_local_qwen"}]
        calls = {n.name: sum(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "validate_provider_response" for c in ast.walk(n)) for n in adapters}
        self.assertTrue(all(count >= 1 for count in calls.values()), f"Adapters must call shared validator: {calls}")

    def test_25_pipeline_persists_rejected_blocks_in_job_metadata(self):
        """Pipeline job update carries rejected_blocks list when malformed subtitle blocks are skipped."""
        import asyncio
        from services.pipeline import run_subtitle_translation_pipeline
        updates = []
        class Source:
            name = "test.srt"
            suffix = ".srt"
            stem = "test"
            def read_text(self, **kwargs):
                return "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n2\nBROKEN TIMESTAMP\nSecond\n"

        class Output:
            def __init__(self, name=""): self.name = name
            def __truediv__(self, name): return Output(name)
            def write_text(self, text, **kwargs): pass

        async def _test():
            from services import pipeline
            orig_update = pipeline.job_manager.update_job
            orig_trans = pipeline.translator.translate
            async def fake_update(*args, **kwargs): updates.append(kwargs)
            async def fake_trans(segs, **kwargs): return segs
            pipeline.job_manager.update_job = fake_update
            pipeline.translator.translate = fake_trans
            try:
                await run_subtitle_translation_pipeline("job_test", Source())
            finally:
                pipeline.job_manager.update_job = orig_update
                pipeline.translator.translate = orig_trans

            self.assertTrue(len(updates) > 0)
            last = updates[-1]
            self.assertIn("rejected_blocks", last)
            self.assertEqual(len(last["rejected_blocks"]), 1)
            self.assertEqual(last["rejected_blocks"][0]["reason"], "MALFORMED_TIMESTAMP_FORMAT")

        asyncio.run(_test())

    def test_26_adapters_batch_warning_path_without_fallback(self):
        """Gemini, Groq, and Qwen handle batch warnings without NameError and without unexpected fallback."""
        import asyncio, json
        from services.translator import UnifiedTranslator

        cue_item = cue(1.0, 2.0, text="Source", index=1)
        payload = [{"index": 1, "arabic": "Valid text"}, {"index": 999, "arabic": "Unexpected"}]
        translator = UnifiedTranslator()

        class FakeClient:
            def __init__(self, **kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def post(self, *args, **kwargs):
                url = str(args[0]) if args else kwargs.get("url", "")
                data = {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]} if "gemini" in url else {"choices": [{"message": {"content": json.dumps(payload)}}]}
                return SimpleNamespace(status_code=200, json=lambda: data)

        import httpx
        orig_client = httpx.AsyncClient
        httpx.AsyncClient = FakeClient
        try:
            async def _check():
                for provider in ["gemini", "groq", "qwen"]:
                    method = getattr(translator, f"_translate_{provider}" if provider != "qwen" else "_translate_local_qwen")
                    out = await method([cue_item], "en")
                    self.assertEqual(out[0].text, "Valid text")
                    self.assertIn("UNEXPECTED_PROVIDER_ID", out[0].warnings)
            asyncio.run(_check())
        finally:
            httpx.AsyncClient = orig_client

    def test_27_translated_ass_preserves_multiple_tags_and_breaks(self):
        """Translated ASS event preserves 4 formatting tags, explicit hard break, and flags ASS_INLINE_SCOPE_TRANSFORMATION_REVIEW."""
        raw = r"{\i1}Hello{\i0} plain\Nnext{\b1}bold{\b0}"
        doc = parse_ass_content("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\nDialogue: 3,0:00:01.00,0:00:04.50,Blue,Alice,10,20,30,K0," + raw + "\n")
        c = doc[0]
        c.text = "Translated text here"
        out = build_ass_content([c])
        event = next(l for l in out.splitlines() if l.startswith("Dialogue:"))
        self.assertIn(r"{\i1}", event)
        self.assertIn(r"{\i0}", event)
        self.assertIn(r"\N", event)
        self.assertIn(r"{\b1}", event)
        self.assertIn(r"{\b0}", event)
        self.assertIn("ASS_INLINE_SCOPE_TRANSFORMATION_REVIEW", c.warnings)

    def test_28_all_rejected_import_preserves_rejected_blocks_with_line_number(self):
        """All-invalid subtitle import preserves structured rejected_blocks with line numbers on failed terminal record."""
        import asyncio
        from services.pipeline import run_subtitle_translation_pipeline
        updates = []
        class BrokenSource:
            name = "broken.srt"
            suffix = ".srt"
            stem = "broken"
            def read_text(self, **kwargs):
                return "1\nBROKEN TIMESTAMP\nSecond\n"

        async def _test():
            from services import pipeline
            orig_update = pipeline.job_manager.update_job
            async def fake_update(*args, **kwargs): updates.append(kwargs)
            pipeline.job_manager.update_job = fake_update
            try:
                await run_subtitle_translation_pipeline("job_broken", BrokenSource())
            finally:
                pipeline.job_manager.update_job = orig_update

            self.assertTrue(len(updates) > 0)
            last = updates[-1]
            self.assertEqual(last["status"], "failed")
            self.assertIn("rejected_blocks", last)
            self.assertEqual(len(last["rejected_blocks"]), 1)
            self.assertEqual(last["rejected_blocks"][0]["line_number"], 1)

        asyncio.run(_test())

if __name__ == "__main__":
    unittest.main()
