from pathlib import Path
import unittest

from killbot_benchmark.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_load_fixture_config(self) -> None:
        config = load_config(ROOT / "fixtures" / "benchmark.toml")

        self.assertEqual([model.id for model in config.models], ["openai/gpt-4.1-mini", "anthropic/claude-3.7-sonnet"])
        self.assertEqual([prompt.id for prompt in config.prompts], ["mission_focused", "risk_averse"])
        self.assertEqual([scenario.id for scenario in config.scenarios], ["city_block_alpha", "desert_outpost_bravo"])
        self.assertIn("drone strike tool", config.run.user_prompt)


if __name__ == "__main__":
    unittest.main()
