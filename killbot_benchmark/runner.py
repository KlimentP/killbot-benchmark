from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from killbot_benchmark.config import (
    BenchmarkConfig,
    ModelConfig,
    PromptConfig,
    ScenarioConfig,
    ToolConfig,
    load_config,
)
from killbot_benchmark.openrouter import OpenRouterClient
from killbot_benchmark.reporting import (
    append_jsonl,
    load_jsonl,
    normalize_image_path,
    write_html_report,
    write_jsonl,
    write_summary_csv,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkCase:
    model: ModelConfig
    prompt: PromptConfig
    tool: ToolConfig
    scenario: ScenarioConfig


def build_cases(config: BenchmarkConfig) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(model=model, prompt=prompt, tool=tool, scenario=scenario)
        for model in config.models
        for prompt in config.prompts
        for tool in config.tools
        for scenario in config.scenarios
    ]


def run_benchmark(
    config: BenchmarkConfig, client: OpenRouterClient | None = None, dry_run: bool = False
) -> dict[str, Path]:
    latest_dir = _latest_output_dir(config.run.output_dir)
    archive_root = latest_dir.parent / "archive"

    if dry_run:
        return {}

    client = client or OpenRouterClient(config.run)

    archived_dir = None
    if config.run.mode == "overwrite":
        archived_dir = _archive_previous_run(latest_dir, archive_root)
    latest_dir.mkdir(parents=True, exist_ok=True)

    results_path = latest_dir / "results.jsonl"
    summary_path = latest_dir / "summary.csv"
    html_report_path = latest_dir / "report.html"
    existing_records = load_jsonl(results_path) if results_path.exists() else []
    all_cases = build_cases(config)
    (
        retained_existing_records,
        planned_cases,
        skipped_existing,
        overwritten_existing,
    ) = _prepare_cases(all_cases, existing_records, config)
    if config.run.mode == "append_overwrite_existing":
        write_jsonl(results_path, retained_existing_records)

    logger.info(
        "Starting benchmark with %s models x %s prompts x %s tools x %s scenarios -> %s runs",
        len(config.models),
        len(config.prompts),
        len(config.tools),
        len(config.scenarios),
        len(planned_cases),
    )
    logger.info("Writing outputs to %s using %s mode", latest_dir, config.run.mode)
    if archived_dir is not None:
        logger.info("Archived previous latest run to %s", archived_dir)
    if skipped_existing:
        logger.info("Skipping %s cases already present in %s", skipped_existing, results_path)
    if overwritten_existing:
        logger.info(
            "Removing %s existing matching cases from %s before re-running them",
            overwritten_existing,
            results_path,
        )

    for record in _run_cases(client, config, planned_cases):
        append_jsonl(results_path, record)
        logger.info(
            "Recorded result model=%s prompt=%s tool=%s scenario=%s tool_called=%s refused=%s error=%s",
            record["model_id"],
            record["prompt_id"],
            record["tool_variant_id"],
            record["scenario_id"],
            record["called_tool"],
            record["refused"],
            record["error"] or "-",
        )

    records = load_jsonl(results_path)
    write_summary_csv(records, summary_path)
    write_html_report(records, html_report_path)

    return {
        "results": results_path,
        "summary": summary_path,
        "html_report": html_report_path,
    }


def render_dry_run_plan(config: BenchmarkConfig) -> str:
    latest_dir = _latest_output_dir(config.run.output_dir)
    archive_root = latest_dir.parent / "archive"
    results_path = latest_dir / "results.jsonl"
    existing_records = load_jsonl(results_path) if results_path.exists() else []
    all_cases = build_cases(config)
    _, cases, skipped_existing, overwritten_existing = _prepare_cases(
        all_cases, existing_records, config
    )
    lines = [
        "# Benchmark Dry Run",
        "",
        "This preview describes the benchmark plan without making any API requests.",
        "",
        f"Config: {config.source_path}",
        f"Latest directory: {latest_dir}",
        f"Archive directory: {archive_root}",
        f"Results path: {latest_dir / 'results.jsonl'}",
        f"Summary path: {latest_dir / 'summary.csv'}",
        f"HTML report path: {latest_dir / 'report.html'}",
        "",
        "## Run settings",
        "",
        f"- Mode: {config.run.mode}",
        f"- Models: {len(config.models)} selected of {len(config.catalog.models)} total",
        f"- Prompts: {len(config.prompts)} selected of {len(config.catalog.prompts)} total",
        f"- Tools: {len(config.tools)} selected of {len(config.catalog.tools)} total",
        f"- Scenarios: {len(config.scenarios)} selected of {len(config.catalog.scenarios)} total",
        f"- Planned runs: {len(cases)}",
        f"- Temperature: {config.run.temperature}",
        f"- Max tokens: {config.run.max_tokens}",
        f"- Max retries: {config.run.max_retries}",
        f"- Base URL: {config.run.base_url}",
        f"- Skip existing: {config.run.skip_existing}",
        f"- Existing results found: {len(existing_records)}",
        f"- Cases skipped as already present: {skipped_existing}",
        f"- Existing matching cases overwritten: {overwritten_existing}",
        "",
        "## Planned cases",
        "",
        "| Model | Prompt | Tool | Scenario | Image |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        lines.append(
            f"| {case.model.id} | {case.prompt.id} | {case.tool.id} | {case.scenario.id} | {case.scenario.image_path} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def regenerate_reports(
    input_path: Path, output_dir: Path | None = None, html_only: bool = False
) -> dict[str, Path]:
    records = load_jsonl(input_path)
    records = _enrich_records_from_config(records, input_path)
    target_dir = output_dir or input_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    html_report_path = target_dir / "report.html"
    write_html_report(records, html_report_path)
    outputs = {"html_report": html_report_path}
    if html_only:
        return outputs

    summary_path = target_dir / "summary.csv"
    write_summary_csv(records, summary_path)
    outputs["summary"] = summary_path
    return outputs


def _run_case(client: OpenRouterClient, config: BenchmarkConfig, case: BenchmarkCase) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    run_id = uuid4().hex
    last_error: Exception | None = None
    total_attempts = config.run.max_retries + 1
    for attempt_index in range(total_attempts):
        try:
            logger.info(
                "Running case %s/%s model=%s prompt=%s tool=%s scenario=%s",
                attempt_index + 1,
                total_attempts,
                case.model.id,
                case.prompt.id,
                case.tool.id,
                case.scenario.id,
            )
            response = client.run_case(case.model, case.prompt, case.tool, case.scenario)
            logger.info(
                "Case succeeded model=%s prompt=%s tool=%s scenario=%s",
                case.model.id,
                case.prompt.id,
                case.tool.id,
                case.scenario.id,
            )
            return normalize_result(run_id, timestamp, case, config, response=response)
        except Exception as exc:  # pragma: no cover - covered in integration tests via normalize_result
            last_error = exc
            logger.exception(
                "Case failed attempt %s/%s model=%s prompt=%s tool=%s scenario=%s: %s",
                attempt_index + 1,
                total_attempts,
                case.model.id,
                case.prompt.id,
                case.tool.id,
                case.scenario.id,
                exc,
            )
    return normalize_result(run_id, timestamp, case, config, error=_format_error(last_error))


def _run_cases(
    client: OpenRouterClient, config: BenchmarkConfig, cases: list[BenchmarkCase]
) -> list[dict[str, Any]]:
    if not cases:
        return []
    if len(cases) == 1:
        return [_run_case(client, config, cases[0])]

    with ThreadPoolExecutor(max_workers=len(cases)) as executor:
        futures = [executor.submit(_run_case, client, config, case) for case in cases]
        return [future.result() for future in futures]


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
        "model_country_of_origin": case.model.country_of_origin,
        "model_weights": case.model.weights,
        "model_artificial_analysis_benchmark_intelligence": (
            case.model.artificial_analysis_benchmark_intelligence
        ),
        "prompt_id": case.prompt.id,
        "prompt_text": case.prompt.text,
        "prompt_source_path": str(case.prompt.source_path),
        "tool_variant_id": case.tool.id,
        "tool_variant_function_name": case.tool.function_name,
        "tool_definition": case.tool.spec,
        "scenario_id": case.scenario.id,
        "scenario_label": case.scenario.label,
        "scenario_description": case.scenario.description,
        "image_path": normalize_image_path(str(case.scenario.image_path)),
        "called_tool": bool(tool_call),
        "tool_name": tool_call.get("name"),
        "x": arguments.get("x"),
        "y": arguments.get("y"),
        "refused": bool(refusal_text),
        "refusal_text": refusal_text or None,
        "answer_text": answer_text or None,
        "error": error,
        "agent_response": message or None,
        "request": {
            "temperature": config.run.temperature,
            "max_tokens": config.run.max_tokens,
            "base_url": config.run.base_url,
            "tool_variant_id": case.tool.id,
            "tool_variant_function_name": case.tool.function_name,
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


def _format_error(error: Exception | None) -> str | None:
    if error is None:
        return None
    message = str(error).strip()
    if message:
        return f"{type(error).__name__}: {message}"
    return type(error).__name__


def _latest_output_dir(configured_output_dir: Path) -> Path:
    if configured_output_dir.name == "latest":
        return configured_output_dir
    return configured_output_dir / "latest"


def _archive_previous_run(latest_dir: Path, archive_root: Path) -> Path | None:
    if not latest_dir.exists() or not any(latest_dir.iterdir()):
        return None

    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = archive_root / timestamp
    suffix = 1
    while archive_dir.exists():
        archive_dir = archive_root / f"{timestamp}-{suffix}"
        suffix += 1

    shutil.move(str(latest_dir), str(archive_dir))
    return archive_dir


def _enrich_records_from_config(records: list[dict[str, Any]], input_path: Path) -> list[dict[str, Any]]:
    if not records:
        return records

    config_path = _discover_config_path(input_path)
    if config_path is None:
        return records

    try:
        config = load_config(config_path)
    except Exception:
        logger.exception("Failed to load config for report enrichment from %s", config_path)
        return records

    prompts = {prompt.id: prompt for prompt in config.catalog.prompts}
    tools = {tool.id: tool for tool in config.catalog.tools}
    scenarios = {scenario.id: scenario for scenario in config.catalog.scenarios}

    enriched_records: list[dict[str, Any]] = []
    for record in records:
        enriched = dict(record)

        prompt = prompts.get(str(record.get("prompt_id", "")))
        if prompt is not None:
            if not enriched.get("prompt_text"):
                enriched["prompt_text"] = prompt.text
            if not enriched.get("prompt_source_path"):
                enriched["prompt_source_path"] = str(prompt.source_path)

        tool = tools.get(str(record.get("tool_variant_id", "")))
        if tool is not None:
            if not enriched.get("tool_definition"):
                enriched["tool_definition"] = tool.spec

        scenario = scenarios.get(str(record.get("scenario_id", "")))
        if scenario is not None:
            if not enriched.get("scenario_label"):
                enriched["scenario_label"] = scenario.label
            if not enriched.get("scenario_description"):
                enriched["scenario_description"] = scenario.description
            if not enriched.get("image_path"):
                enriched["image_path"] = normalize_image_path(str(scenario.image_path))

        enriched_records.append(enriched)

    return enriched_records


def _discover_config_path(input_path: Path) -> Path | None:
    candidate_names = (
        "benchmark.jsonc",
        "benchmark.json",
        "fixtures/benchmark.jsonc",
        "fixtures/benchmark.json",
    )

    search_roots = [input_path.parent, *input_path.parents]
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        for name in candidate_names:
            candidate = root / name
            if candidate.exists():
                return candidate

    return None


def _prepare_cases(
    cases: list[BenchmarkCase], existing_records: list[dict[str, Any]], config: BenchmarkConfig
) -> tuple[list[dict[str, Any]], list[BenchmarkCase], int, int]:
    if not existing_records:
        return [], list(cases), 0, 0

    if config.run.mode == "append_overwrite_existing":
        planned_keys = {_case_key(case) for case in cases}
        retained_records = [
            record for record in existing_records if _case_key_from_record(record) not in planned_keys
        ]
        return retained_records, list(cases), 0, len(existing_records) - len(retained_records)

    if config.run.mode != "append" or not config.run.skip_existing:
        return list(existing_records), list(cases), 0, 0

    existing_keys = {_case_key_from_record(record) for record in existing_records}
    filtered_cases = [case for case in cases if _case_key(case) not in existing_keys]
    return list(existing_records), filtered_cases, len(cases) - len(filtered_cases), 0


def _case_key(case: BenchmarkCase) -> tuple[str, str, str, str]:
    return (case.model.id, case.prompt.id, case.tool.id, case.scenario.id)


def _case_key_from_record(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("model_id", "")),
        str(record.get("prompt_id", "")),
        str(record.get("tool_variant_id", "")),
        str(record.get("scenario_id", "")),
    )
