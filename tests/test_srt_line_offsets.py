"""AIX-SRT-LINE-001 v2: deterministic, stdlib-only source-position gates.

Run python -B tests/test_srt_line_offsets.py. Pipeline coverage executes only
reviewed definitions with memory IO; importing the app/model stack is forbidden.
"""
import ast
import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("srt_offsets_under_test", ROOT / "services/subtitles.py")
subtitles = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subtitles)


class TestSrtLineOffsets(unittest.TestCase):
    def test_leading_blanks_report_block_start_not_timestamp_line(self):
        doc = subtitles.parse_srt_content("\n\n\n2\nBROKEN\nText")
        self.assertEqual(len(doc), 0)
        self.assertEqual(doc.rejected_blocks[0]["line_number"], 4)
        self.assertEqual(doc.rejected_blocks[0]["block_index"], 1)

    def test_three_separator_lines_are_counted(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n\n\n2\nBROKEN\nText"
        doc = subtitles.parse_srt_content(text)
        self.assertEqual(len(doc), 1)
        self.assertEqual(doc[0].text, "First")
        self.assertEqual(doc.rejected_blocks[0]["line_number"], 7)
        self.assertEqual(doc.rejected_blocks[0]["block_index"], 2)

    def test_crlf_mixed_and_whitespace_separators(self):
        variants = (
            "\r\n\r\n2\r\nBROKEN\r\nText",
            "\n\r\n2\nBROKEN\r\nText",
            " \t\n\t\r\n2\r\nBROKEN\nText",
        )
        for text in variants:
            with self.subTest(text=text):
                doc = subtitles.parse_srt_content(text)
                self.assertEqual(doc.rejected_blocks[0]["line_number"], 3)

    def test_multiple_rejected_blocks_keep_physical_positions_and_raw_text(self):
        doc = subtitles.parse_srt_content("\n1\nBROKEN\nFirst\n \t\n\n2\r\nBROKEN AGAIN\r\nSecond")
        self.assertEqual([r["line_number"] for r in doc.rejected_blocks], [2, 7])
        self.assertEqual([r["block_index"] for r in doc.rejected_blocks], [1, 2])
        self.assertEqual(doc.rejected_blocks[0]["raw_text"], "1\nBROKEN\nFirst")
        self.assertEqual(doc.rejected_blocks[1]["raw_text"], "2\r\nBROKEN AGAIN\r\nSecond")
        for rejected in doc.rejected_blocks:
            self.assertEqual(set(rejected), {"line_number", "block_index", "raw_text", "reason"})
            self.assertEqual(rejected["reason"], "MALFORMED_TIMESTAMP_FORMAT")

    def test_empty_and_all_blank_sources_are_empty_documents(self):
        for text in ("", "\n\n", " \t\r\n\n\t"):
            with self.subTest(text=text):
                doc = subtitles.parse_srt_content(text)
                self.assertEqual(list(doc), [])
                self.assertEqual(doc.rejected_blocks, [])
                self.assertEqual(doc.custom_styles, [])

    def test_ordinary_and_unterminated_last_block_controls(self):
        for suffix in ("", "\n", "\r\n"):
            with self.subTest(suffix=suffix):
                doc = subtitles.parse_srt_content("1\nBROKEN\nText" + suffix)
                self.assertEqual(doc.rejected_blocks[0]["line_number"], 1)
                self.assertEqual(doc.rejected_blocks[0]["raw_text"], "1\nBROKEN\nText")

    def test_valid_semantics_are_not_retimed_or_reordered(self):
        text = "9\n00:00:02,100 --> 00:00:02,200\n  First line  \nSecond line\n\n4\n00:00:01.500 --> 00:00:01.600\nNext\n"
        doc = subtitles.parse_srt_content(text)
        self.assertEqual([(s.index, s.start, s.end, s.text) for s in doc], [
            (1, 2.1, 2.2, "First line Second line"),
            (2, 1.5, 1.6, "Next"),
        ])
        self.assertEqual(doc.rejected_blocks, [])

    def test_separator_positions_across_small_deterministic_grid(self):
        for leading in range(5):
            for blank_count in range(1, 5):
                for newline in ("\n", "\r\n"):
                    with self.subTest(leading=leading, blanks=blank_count, newline=newline):
                        lines = ([""] * leading + ["1", "BAD", "A"] +
                                 [" \t"] * blank_count + ["2", "BAD", "B"])
                        doc = subtitles.parse_srt_content(newline.join(lines))
                        self.assertEqual([r["line_number"] for r in doc.rejected_blocks],
                                         [leading + 1, leading + 4 + blank_count])

    def test_failed_import_retains_correct_lines_without_translation_or_output(self):
        pipeline_source = (ROOT / "services/pipeline.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(pipeline_source)
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.AsyncFunctionDef) and n.name in
                  {"run_subtitle_translation_pipeline", "run_exclusive_worker"}) or
                 (isinstance(n, ast.ClassDef) and n.name == "JobManager")]
        self.assertEqual(len(nodes), 3)
        output_attempts = []
        translation_attempts = []
        class NoOutput:
            def __truediv__(self, name):
                output_attempts.append(name)
                raise AssertionError("All-invalid source must not create outputs")
        async def forbidden_translation(*args, **kwargs):
            translation_attempts.append(True)
            raise AssertionError("All-invalid source must not translate")
        logger = logging.Logger("srt-offset-test")
        logger.addHandler(logging.NullHandler())
        namespace = {**vars(subtitles), "asyncio": asyncio, "time": __import__("time"),
                     "Path": Path, "Any": Any, "Dict": Dict, "Optional": Optional,
                     "logger": logger, "settings": SimpleNamespace(OUTPUT_DIR=NoOutput()),
                     "translator": SimpleNamespace(translate=forbidden_translation)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     "isolated_srt_pipeline", "exec"), namespace)
        for text, expected_line in (("\n\n\n1\nBAD\nText", 4),
                                    ("\r\n\r\n1\r\nBAD\r\nText", 3)):
            class Source:
                name = "review.srt"
                suffix = ".srt"
                stem = "review"
                def read_text(self, **kwargs):
                    return text
            manager = namespace["JobManager"]()
            manager.create_job("review", "review.srt")
            namespace["job_manager"] = manager
            namespace["gpu_lock"] = asyncio.Lock()
            asyncio.run(namespace["run_subtitle_translation_pipeline"]("review", Source()))
            self.assertEqual(manager.jobs["review"]["status"], "failed")
            rejected = manager.jobs["review"]["rejected_blocks"]
            self.assertEqual(rejected[0]["line_number"], expected_line)
            self.assertEqual(rejected[0]["reason"], "MALFORMED_TIMESTAMP_FORMAT")
            self.assertEqual(translation_attempts, [])
            self.assertEqual(output_attempts, [])


if __name__ == "__main__":
    unittest.main()
