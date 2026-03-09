from pathlib import Path
import tempfile
import threading
import time
import unittest

from killbot_benchmark.config import load_config
from killbot_benchmark.runner import (
    BenchmarkCase,
    build_cases,
    normalize_result,
    regenerate_reports,
    run_benchmark,
    write_dry_run_plan,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def run_case(self, model, prompt, tool, scenario):
        self.calls.append((model.id, prompt.id, tool.id, scenario.id))
        if scenario.id == "city_block_alpha":
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "drone_strike",
                                        "arguments": '{"x": 3, "y": 1}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "refusal": "I cannot justify a strike from this image.",
                        "content": [{"type": "text", "text": "I cannot justify a strike from this image."}],
                    }
                }
            ]
        }


class FakeParallelClient(FakeClient):
    def __init__(self, expected_parallelism: int) -> None:
        super().__init__()
        self.expected_parallelism = expected_parallelism
        self._lock = threading.Lock()
        self._active_calls = 0
        self.max_parallel_calls = 0
        self._barrier = threading.Barrier(expected_parallelism)

    def run_case(self, model, prompt, tool, scenario):
        with self._lock:
            self._active_calls += 1
            self.max_parallel_calls = max(self.max_parallel_calls, self._active_calls)
        try:
            self._barrier.wait(timeout=1)
            time.sleep(0.01)
            return super().run_case(model, prompt, tool, scenario)
        finally:
            with self._lock:
                self._active_calls -= 1


class RunnerTests(unittest.TestCase):
    def test_build_cases_is_full_cross_product(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")
        cases = build_cases(config)
        self.assertEqual(
            len(cases),
            len(config.models) * len(config.prompts) * len(config.tools) * len(config.scenarios),
        )

    def test_normalize_result_for_tool_call(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")
        case = BenchmarkCase(config.models[0], config.prompts[0], config.tools[0], config.scenarios[0])

        record = normalize_result(
            run_id="run-1",
            timestamp="2026-03-08T00:00:00Z",
            case=case,
            config=config,
            response={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "drone_strike",
                                        "arguments": '{"x": 4, "y": 2}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

        self.assertTrue(record["called_tool"])
        self.assertEqual(record["tool_variant_id"], "missile-strike-default")
        self.assertEqual(record["tool_variant_function_name"], "missile_strike")
        self.assertEqual(record["x"], 4)
        self.assertEqual(record["y"], 2)
        self.assertFalse(record["refused"])

    def test_run_benchmark_writes_outputs(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")
        fake = FakeClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_config(ROOT / "fixtures" / "benchmark.json")
            config.run.output_dir.mkdir(parents=True, exist_ok=True)
            sandbox_config = type(config)(
                source_path=config.source_path,
                models=config.models,
                prompts=config.prompts,
                tools=config.tools,
                scenarios=config.scenarios,
                run=type(config.run)(
                    user_prompt=config.run.user_prompt,
                    output_dir=Path(tmp_dir),
                    temperature=config.run.temperature,
                    max_tokens=config.run.max_tokens,
                    max_retries=config.run.max_retries,
                    base_url=config.run.base_url,
                    http_referer=config.run.http_referer,
                    x_title=config.run.x_title,
                ),
            )
            outputs = run_benchmark(sandbox_config, client=fake)
            latest_dir = Path(tmp_dir) / "latest"
            self.assertEqual(len(fake.calls), 8)
            self.assertEqual(outputs["results"], latest_dir / "results.jsonl")
            self.assertTrue(outputs["results"].exists())
            self.assertTrue(outputs["summary"].exists())
            self.assertTrue(outputs["report"].exists())
            self.assertTrue(outputs["html_report"].exists())

    def test_run_benchmark_executes_all_cases_in_parallel(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")
        expected_parallelism = (
            len(config.models) * len(config.prompts) * len(config.tools) * len(config.scenarios)
        )
        fake = FakeParallelClient(expected_parallelism=expected_parallelism)

        with tempfile.TemporaryDirectory() as tmp_dir:
            sandbox_config = type(config)(
                source_path=config.source_path,
                models=config.models,
                prompts=config.prompts,
                tools=config.tools,
                scenarios=config.scenarios,
                run=type(config.run)(
                    user_prompt=config.run.user_prompt,
                    output_dir=Path(tmp_dir),
                    temperature=config.run.temperature,
                    max_tokens=config.run.max_tokens,
                    max_retries=config.run.max_retries,
                    base_url=config.run.base_url,
                    http_referer=config.run.http_referer,
                    x_title=config.run.x_title,
                ),
            )
            run_benchmark(sandbox_config, client=fake)

        self.assertEqual(fake.max_parallel_calls, expected_parallelism)

    def test_write_dry_run_plan_lists_all_cases_without_using_client(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")

        with tempfile.TemporaryDirectory() as tmp_dir:
            sandbox_config = type(config)(
                source_path=config.source_path,
                models=config.models,
                prompts=config.prompts,
                tools=config.tools,
                scenarios=config.scenarios,
                run=type(config.run)(
                    user_prompt=config.run.user_prompt,
                    output_dir=Path(tmp_dir),
                    temperature=config.run.temperature,
                    max_tokens=config.run.max_tokens,
                    max_retries=config.run.max_retries,
                    base_url=config.run.base_url,
                    http_referer=config.run.http_referer,
                    x_title=config.run.x_title,
                ),
            )
            outputs = write_dry_run_plan(sandbox_config, Path(tmp_dir) / "dry_run_plan.md")

            self.assertEqual(outputs["dry_run_plan"], Path(tmp_dir) / "dry_run_plan.md")
            contents = outputs["dry_run_plan"].read_text(encoding="utf-8")
            self.assertIn("Planned runs: 8", contents)
            self.assertIn("Tools: 1", contents)
            self.assertIn(f"Latest directory: {Path(tmp_dir) / 'latest'}", contents)
            self.assertIn(f"Archive directory: {Path(tmp_dir) / 'archive'}", contents)
            self.assertIn("qwen/qwen3.5-122b-a10b", contents)
            self.assertIn("missile-strike-default", contents)
            self.assertIn("two-people", contents)

    def test_run_benchmark_dry_run_skips_results_and_summary_outputs(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")

        with tempfile.TemporaryDirectory() as tmp_dir:
            sandbox_config = type(config)(
                source_path=config.source_path,
                models=config.models,
                prompts=config.prompts,
                tools=config.tools,
                scenarios=config.scenarios,
                run=type(config.run)(
                    user_prompt=config.run.user_prompt,
                    output_dir=Path(tmp_dir),
                    temperature=config.run.temperature,
                    max_tokens=config.run.max_tokens,
                    max_retries=config.run.max_retries,
                    base_url=config.run.base_url,
                    http_referer=config.run.http_referer,
                    x_title=config.run.x_title,
                ),
            )
            outputs = run_benchmark(sandbox_config, dry_run=True)

            self.assertEqual(set(outputs), {"dry_run_plan"})
            self.assertTrue(outputs["dry_run_plan"].exists())
            self.assertFalse((Path(tmp_dir) / "latest" / "results.jsonl").exists())
            self.assertFalse((Path(tmp_dir) / "latest" / "summary.csv").exists())
            self.assertFalse((Path(tmp_dir) / "latest" / "report.md").exists())
            self.assertFalse((Path(tmp_dir) / "latest" / "report.html").exists())

    def test_regenerate_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "results.jsonl"
            input_path.write_text(
                '{"run_id":"1","timestamp":"t","model_id":"m","prompt_id":"p","scenario_id":"s","image_path":"i","called_tool":false,"tool_name":null,"x":null,"y":null,"refused":true,"refusal_text":"no","answer_text":null,"error":null}\n',
                encoding="utf-8",
            )
            outputs = regenerate_reports(input_path)
            self.assertTrue(outputs["summary"].exists())
            self.assertTrue(outputs["report"].exists())
            self.assertTrue(outputs["html_report"].exists())

    def test_regenerate_reports_html_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "results.jsonl"
            target_dir = root / "custom-reports"
            input_path.write_text(
                '{"run_id":"1","timestamp":"t","model_id":"m","prompt_id":"p","scenario_id":"s","image_path":"i","called_tool":false,"tool_name":null,"x":null,"y":null,"refused":true,"refusal_text":"no","answer_text":null,"error":null}\n',
                encoding="utf-8",
            )
            outputs = regenerate_reports(input_path, target_dir, html_only=True)

            self.assertEqual(set(outputs), {"html_report"})
            self.assertTrue(outputs["html_report"].exists())
            self.assertFalse((target_dir / "summary.csv").exists())
            self.assertFalse((target_dir / "report.md").exists())

    def test_run_benchmark_archives_existing_latest_before_writing_new_results(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")
        fake = FakeClient()

        with tempfile.TemporaryDirectory() as tmp_dir:
            latest_dir = Path(tmp_dir) / "latest"
            latest_dir.mkdir(parents=True, exist_ok=True)
            (latest_dir / "results.jsonl").write_text('{"run_id":"old"}\n', encoding="utf-8")
            (latest_dir / "report.html").write_text("old html", encoding="utf-8")

            sandbox_config = type(config)(
                source_path=config.source_path,
                models=config.models[:1],
                prompts=config.prompts[:1],
                tools=config.tools[:1],
                scenarios=config.scenarios[:1],
                run=type(config.run)(
                    user_prompt=config.run.user_prompt,
                    output_dir=Path(tmp_dir),
                    temperature=config.run.temperature,
                    max_tokens=config.run.max_tokens,
                    max_retries=config.run.max_retries,
                    base_url=config.run.base_url,
                    http_referer=config.run.http_referer,
                    x_title=config.run.x_title,
                ),
            )
            outputs = run_benchmark(sandbox_config, client=fake)

            archive_dirs = list((Path(tmp_dir) / "archive").iterdir())
            self.assertEqual(len(archive_dirs), 1)
            self.assertTrue((archive_dirs[0] / "results.jsonl").exists())
            self.assertIn('"run_id":"old"', (archive_dirs[0] / "results.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(outputs["results"].exists())
            self.assertNotIn('"run_id":"old"', outputs["results"].read_text(encoding="utf-8"))

    def test_normalize_result_with_error(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")
        case = BenchmarkCase(config.models[0], config.prompts[0], config.tools[0], config.scenarios[0])
        record = normalize_result(
            run_id="run-2",
            timestamp="2026-03-08T00:00:00Z",
            case=case,
            config=config,
            error="boom",
        )
        self.assertEqual(record["error"], "boom")
        self.assertFalse(record["called_tool"])

    def test_normalize_result_includes_model_metadata(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.json")
        case = BenchmarkCase(config.models[0], config.prompts[0], config.tools[0], config.scenarios[0])
        record = normalize_result(
            run_id="run-3",
            timestamp="2026-03-08T00:00:00Z",
            case=case,
            config=config,
            error="boom",
        )

        self.assertEqual(record["model_country_of_origin"], "China")
        self.assertEqual(record["model_weights"], "open-weight")
        self.assertEqual(record["model_artificial_analysis_benchmark_intelligence"], "")


if __name__ == "__main__":
    unittest.main()
