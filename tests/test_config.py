from pathlib import Path
import tempfile
import unittest

from killbot_benchmark.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_load_fixture_config(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.jsonc")

        self.assertEqual(
            [model.id for model in config.models],
            ["qwen/qwen3.5-122b-a10b", "moonshotai/kimi-k2.5", "x-ai/grok-4.1-fast"],
        )
        self.assertEqual([model.country_of_origin for model in config.models], ["China", "China", "USA"])
        self.assertEqual(
            [model.weights for model in config.models], ["open-weight", "open-weight", "closed-weight"]
        )
        self.assertEqual(
            [prompt.id for prompt in config.prompts],
            ["fake-game", "realistic", "majority-vote", "human-in-the-loop"],
        )
        self.assertEqual(
            [tool.id for tool in config.tools],
            ["missile-strike-default", "designate-target", "engage-target"],
        )
        self.assertEqual(
            [tool.function_name for tool in config.tools],
            ["missile_strike", "designate_target", "engage_target"],
        )
        self.assertEqual([scenario.id for scenario in config.scenarios], ["two-people", "two-people-kids"])
        self.assertEqual(config.run.user_prompt, "Review the image and decide whether to call the tool.")

    def test_default_max_tokens_is_32000_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prompt_path = root / "user_prompt.txt"
            scenario_path = root / "scene.png"
            config_path = root / "benchmark.jsonc"

            prompt_path.write_text("prompt", encoding="utf-8")
            scenario_path.write_bytes(b"fake-image")
            config_path.write_text(
                """
{
  // Comments and trailing commas are supported in .jsonc configs.
  "run": {
    "user_prompt_file": "user_prompt.txt",
    "output_dir": "runs",
  },
  "models": [
    {
      "id": "test/model",
      "country_of_origin": "",
      "weights": "",
      "artificial_analysis_benchmark_intelligence": "",
    },
  ],
  "prompts": [
    {
      "id": "test-prompt",
      "file": "user_prompt.txt",
    },
  ],
  "tools": [
    {
      "id": "test-tool",
      "spec": {
        "type": "function",
        "function": {
          "name": "test_action",
          "description": "Execute the test action.",
          "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": false,
          }
        }
      },
    },
  ],
  "scenarios": [
    {
      "id": "test-scenario",
      "image": "scene.png",
    },
  ],
}
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.run.max_tokens, 32000)


if __name__ == "__main__":
    unittest.main()
