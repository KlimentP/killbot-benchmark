from pathlib import Path
import tempfile
import unittest

from killbot_benchmark.config import load_config
from killbot_benchmark.runner import BenchmarkCase, build_cases, normalize_result, regenerate_reports, run_benchmark


ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def run_case(self, model, prompt, scenario):
        self.calls.append((model.id, prompt.id, scenario.id))
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


class RunnerTests(unittest.TestCase):
    def test_build_cases_is_full_cross_product(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.toml")
        cases = build_cases(config)
        self.assertEqual(len(cases), len(config.models) * len(config.prompts) * len(config.scenarios))

    def test_normalize_result_for_tool_call(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.toml")
        case = BenchmarkCase(config.models[0], config.prompts[0], config.scenarios[0])

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
        self.assertEqual(record["x"], 4)
        self.assertEqual(record["y"], 2)
        self.assertFalse(record["refused"])

    def test_run_benchmark_writes_outputs(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.toml")
        fake = FakeClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = load_config(ROOT / "fixtures" / "benchmark.toml")
            config.run.output_dir.mkdir(parents=True, exist_ok=True)
            sandbox_config = type(config)(
                source_path=config.source_path,
                models=config.models,
                prompts=config.prompts,
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
            self.assertEqual(len(fake.calls), 8)
            self.assertTrue(outputs["results"].exists())
            self.assertTrue(outputs["summary"].exists())
            self.assertTrue(outputs["report"].exists())

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

    def test_normalize_result_with_error(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.toml")
        case = BenchmarkCase(config.models[0], config.prompts[0], config.scenarios[0])
        record = normalize_result(
            run_id="run-2",
            timestamp="2026-03-08T00:00:00Z",
            case=case,
            config=config,
            error="boom",
        )
        self.assertEqual(record["error"], "boom")
        self.assertFalse(record["called_tool"])


if __name__ == "__main__":
    unittest.main()
