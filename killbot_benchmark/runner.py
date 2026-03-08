from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from killbot_benchmark.config import BenchmarkConfig, ModelConfig, PromptConfig, ScenarioConfig
from killbot_benchmark.openrouter import OpenRouterClient
from killbot_benchmark.reporting import append_jsonl, load_jsonl, write_markdown_report, write_summary_csv


@dataclass(frozen=True)
class BenchmarkCase:
    model: ModelConfig
    prompt: PromptConfig
    scenario: ScenarioConfig


def build_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(model=model, prompt=prompt, scenario=scenario)
        for model in config.models
        for prompt in config.prompts
        for scenario in config.scenarios
    ]


def run_benchmark(config: BenchmarkConfig, client: OpenRouterClient | None = None) -> dict[str, Path]:
    client = client or OpenRouterClient(config.run)

    output_dir = config.run.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "report.md"
    if results_path.exists():
        results_path.unlink()

    for case in build_cases(config):
        record = _run_case(client, config, case)
        append_jsonl(results_path, record)

    records = load_jsonl(results_path)
    write_summary_csv(records, summary_path)
    write_markdown_report(records, report_path)

    return {
        "results": results_path,
        "summary": summary_path,
        "report": report_path,
    }


def regenerate_reports(input_path: Path, output_dir: Path | None = None) -> dict[str, Path]:
    records = load_jsonl(input_path)
    target_dir = output_dir or input_path.parent
    summary_path = target_dir / "summary.csv"
    report_path = target_dir / "report.md"
    write_summary_csv(records, summary_path)
    write_markdown_report(records, report_path)
    return {
        "summary": summary_path,
        "report": report_path,
    }


def _run_case(client: OpenRouterClient, config: BenchmarkConfig, case: BenchmarkCase) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = uuid4().hex
    last_error: Exception | None = None
    for _attempt in range(config.run.max_retries + 1):
        try:
            response = client.run_case(case.model, case.prompt, case.scenario)
            return normalize_result(run_id, timestamp, case, config, response=response)
        except Exception as exc:  # pragma: no cover - covered in integration tests via normalize_result
            last_error = exc
    return normalize_result(run_id, timestamp, case, config, error=str(last_error))


def normalize_result(
    run_id: str,
    timestamp: str,
    case: BenchmarkCase,
    config: BenchmarkConfig,
    response: Any | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    message = _extract_message(response) if response is not None else {}
    tool_call = _extract_tool_call(message)
    arguments = _parse_arguments(tool_call.get("arguments"))

    refusal_text = str(message.get("refusal") or "").strip()
    answer_text = _extract_answer_text(message)

    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_id": case.model.id,
        "prompt_id": case.prompt.id,
        "scenario_id": case.scenario.id,
        "image_path": str(case.scenario.image_path),
        "called_tool": bool(tool_call),
        "tool_name": tool_call.get("name"),
        "x": arguments.get("x"),
        "y": arguments.get("y"),
        "refused": bool(refusal_text),
        "refusal_text": refusal_text or None,
        "answer_text": answer_text or None,
        "error": error,
        "request": {
            "temperature": config.run.temperature,
            "max_tokens": config.run.max_tokens,
            "base_url": config.run.base_url,
        },
        "raw_response": _response_to_dict(response) if response is not None else None,
    }
    return record


def _extract_message(response: Any) -> dict[str, Any]:
    if response is None:
        return {}

    response_dict = _response_to_dict(response)
    choices = response_dict.get("choices") or []
    if not choices:
        return {}
    return choices[0].get("message") or {}


def _extract_tool_call(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return {}
    first_call = tool_calls[0]
    function = first_call.get("function") or {}
    return {
        "name": function.get("name"),
        "arguments": function.get("arguments"),
    }


def _parse_arguments(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"raw_arguments": arguments}
    return {
        "x": parsed.get("x"),
        "y": parsed.get("y"),
        "raw_arguments": parsed,
    }


def _extract_answer_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                texts.append(str(item["text"]).strip())
        return "\n".join(part for part in texts if part).strip()
    return ""


def _response_to_dict(response: Any) -> dict[str, Any]:
    if response is None:
        return {}
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    raise TypeError(f"Unsupported response type: {type(response)!r}")
