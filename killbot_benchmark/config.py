from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    id: str
    country_of_origin: str = ""
    weights: str = ""
    artificial_analysis_benchmark_intelligence: str = ""


@dataclass(frozen=True)
class PromptConfig:
    id: str
    text: str
    source_path: Path


@dataclass(frozen=True)
class ToolConfig:
    id: str
    spec: dict

    @property
    def function_name(self) -> str:
        function = self.spec.get("function", {})
        return str(function.get("name", ""))


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
    max_tokens: int = 32000
    max_retries: int = 2
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    x_title: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    source_path: Path
    models: list[ModelConfig]
    prompts: list[PromptConfig]
    tools: list[ToolConfig]
    scenarios: list[ScenarioConfig]
    run: RunSettings


def _resolve_text(base_dir: Path, relative_path: str) -> tuple[Path, str]:
    path = (base_dir / relative_path).resolve()
    return path, path.read_text(encoding="utf-8").strip()


def _resolve_path(base_dir: Path, relative_path: str) -> Path:
    return (base_dir / relative_path).resolve()


def _load_config_data(source_path: Path) -> dict:
    raw_text = source_path.read_text(encoding="utf-8")
    if source_path.suffix.lower() == ".jsonc":
        return json.loads(_strip_jsonc(raw_text))
    return json.loads(raw_text)


def _strip_jsonc(raw_text: str) -> str:
    cleaned_chars: list[str] = []
    in_string = False
    string_delimiter = ""
    escape_next = False
    i = 0
    length = len(raw_text)

    while i < length:
        char = raw_text[i]
        next_char = raw_text[i + 1] if i + 1 < length else ""

        if in_string:
            cleaned_chars.append(char)
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == string_delimiter:
                in_string = False
            i += 1
            continue

        if char in {'"', "'"}:
            in_string = True
            string_delimiter = char
            cleaned_chars.append(char)
            i += 1
            continue

        if char == "/" and next_char == "/":
            i += 2
            while i < length and raw_text[i] not in "\r\n":
                i += 1
            continue

        if char == "/" and next_char == "*":
            i += 2
            while i + 1 < length and not (raw_text[i] == "*" and raw_text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        cleaned_chars.append(char)
        i += 1

    return _strip_trailing_commas("".join(cleaned_chars))


def _strip_trailing_commas(raw_text: str) -> str:
    cleaned_chars: list[str] = []
    in_string = False
    string_delimiter = ""
    escape_next = False
    i = 0
    length = len(raw_text)

    while i < length:
        char = raw_text[i]

        if in_string:
            cleaned_chars.append(char)
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == string_delimiter:
                in_string = False
            i += 1
            continue

        if char in {'"', "'"}:
            in_string = True
            string_delimiter = char
            cleaned_chars.append(char)
            i += 1
            continue

        if char == ",":
            j = i + 1
            while j < length and raw_text[j] in " \t\r\n":
                j += 1
            if j < length and raw_text[j] in "]}":
                i += 1
                continue

        cleaned_chars.append(char)
        i += 1

    return "".join(cleaned_chars)


def load_config(config_path: str | Path) -> BenchmarkConfig:
    source_path = Path(config_path).resolve()
    data = _load_config_data(source_path)
    base_dir = source_path.parent

    run_data = data["run"]
    user_prompt_path, user_prompt = _resolve_text(base_dir, run_data["user_prompt_file"])

    models = [
        ModelConfig(
            id=item["id"],
            country_of_origin=item.get("country_of_origin", ""),
            weights=item.get("weights", ""),
            artificial_analysis_benchmark_intelligence=item.get(
                "artificial_analysis_benchmark_intelligence", ""
            ),
        )
        for item in data["models"]
    ]
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

    tools = [
        ToolConfig(
            id=item["id"],
            spec=item["spec"],
        )
        for item in data["tools"]
    ]

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
        max_tokens=run_data.get("max_tokens", 32000),
        max_retries=run_data.get("max_retries", 2),
        base_url=run_data.get("base_url", "https://openrouter.ai/api/v1"),
        http_referer=run_data.get("http_referer"),
        x_title=run_data.get("x_title"),
    )

    return BenchmarkConfig(
        source_path=source_path,
        models=models,
        prompts=prompts,
        tools=tools,
        scenarios=scenarios,
        run=run,
    )
