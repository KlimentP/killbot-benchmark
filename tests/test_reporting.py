from pathlib import Path
import tempfile
import unittest

from killbot_benchmark.reporting import load_jsonl, write_markdown_report, write_summary_csv


SAMPLE_RECORDS = [
    {
        "run_id": "1",
        "timestamp": "2026-03-08T00:00:00Z",
        "model_id": "model-a",
        "prompt_id": "prompt-a",
        "scenario_id": "image-a",
        "image_path": "/tmp/a.png",
        "called_tool": True,
        "tool_name": "drone_strike",
        "x": 3,
        "y": 2,
        "refused": False,
        "refusal_text": None,
        "answer_text": None,
        "error": None,
    },
    {
        "run_id": "2",
        "timestamp": "2026-03-08T00:00:01Z",
        "model_id": "model-a",
        "prompt_id": "prompt-b",
        "scenario_id": "image-a",
        "image_path": "/tmp/a.png",
        "called_tool": False,
        "tool_name": None,
        "x": None,
        "y": None,
        "refused": True,
        "refusal_text": "I refuse to do that.",
        "answer_text": None,
        "error": None,
    },
]


class ReportingTests(unittest.TestCase):
    def test_writes_summary_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            csv_path = root / "summary.csv"
            md_path = root / "report.md"

            write_summary_csv(SAMPLE_RECORDS, csv_path)
            write_markdown_report(SAMPLE_RECORDS, md_path)

            self.assertIn("model-a", csv_path.read_text(encoding="utf-8"))
            report_text = md_path.read_text(encoding="utf-8")
            self.assertIn("By model_id", report_text)
            self.assertIn("I refuse to do that.", report_text)

    def test_load_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "results.jsonl"
            path.write_text('{"run_id":"1"}\n{"run_id":"2"}\n', encoding="utf-8")
            records = load_jsonl(path)
            self.assertEqual(records, [{"run_id": "1"}, {"run_id": "2"}])


if __name__ == "__main__":
    unittest.main()
