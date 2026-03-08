from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ModelConfig:
    id: str


@dataclass(frozen=True)
class PromptConfig:
    id: str
    text: str
    source_path: Path


@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    image_path: Path
    label: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class RunSettings:
    user_prompt: str
    output_dir: Path
    temperature: float = 0.0
    max_tokens: int = 400
    max_retries: int = 2
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    x_title: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    source_path: Path
    models: list[ModelConfig]
    prompts: list[PromptConfig]
    scenarios: list[ScenarioConfig]
    run: RunSettings


def _resolve_text(base_dir: Path, relative_path: str) -> tuple[Path, str]:
    path = (base_dir / relative_path).resolve()
    return path, path.read_text(encoding="utf-8").strip()


def _resolve_path(base_dir: Path, relative_path: str) -> Path:
    return (base_dir / relative_path).resolve()


def load_config(config_path: str | Path) -> BenchmarkConfig:
    source_path = Path(config_path).resolve()
    data = tomllib.loads(source_path.read_text(encoding="utf-8"))
    base_dir = source_path.parent

    run_data = data["run"]
    user_prompt_path, user_prompt = _resolve_text(base_dir, run_data["user_prompt_file"])

    models = [ModelConfig(id=item["id"]) for item in data["models"]]
    prompts = []
    for item in data["prompts"]:
        prompt_path, prompt_text = _resolve_text(base_dir, item["file"])
        prompts.append(
            PromptConfig(
                id=item["id"],
                text=prompt_text,
                source_path=prompt_path,
            )
        )

    scenarios = []
    for item in data["scenarios"]:
        scenarios.append(
            ScenarioConfig(
                id=item["id"],
                image_path=_resolve_path(base_dir, item["image"]),
                label=item.get("label"),
                description=item.get("description"),
            )
        )

    run = RunSettings(
        user_prompt=user_prompt,
        output_dir=_resolve_path(base_dir, run_data["output_dir"]),
        temperature=run_data.get("temperature", 0.0),
        max_tokens=run_data.get("max_tokens", 400),
        max_retries=run_data.get("max_retries", 2),
        base_url=run_data.get("base_url", "https://openrouter.ai/api/v1"),
        http_referer=run_data.get("http_referer"),
        x_title=run_data.get("x_title"),
    )

    return BenchmarkConfig(
        source_path=source_path,
        models=models,
        prompts=prompts,
        scenarios=scenarios,
        run=run,
    )
