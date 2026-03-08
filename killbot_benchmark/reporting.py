from __future__ import annotations

import csv
import json
from pathlib import Path


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_summary_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "timestamp",
        "model_id",
        "prompt_id",
        "scenario_id",
        "image_path",
        "called_tool",
        "tool_name",
        "x",
        "y",
        "refused",
        "refusal_text",
        "answer_text",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def write_markdown_report(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Report",
        "",
        f"Total runs: {len(records)}",
        "",
    ]

    for key in ("model_id", "prompt_id", "scenario_id"):
        lines.extend(_group_section(records, key))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _group_section(records: list[dict], key: str) -> list[str]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get(key, "")), []).append(record)

    lines = [f"## By {key}", "", "| Value | Runs | Tool calls | Refusals | Invalid/no-call |", "| --- | ---: | ---: | ---: | ---: |"]
    for value, items in sorted(groups.items()):
        tool_calls = sum(1 for item in items if item.get("called_tool"))
        refusals = sum(1 for item in items if item.get("refused"))
        invalid = sum(1 for item in items if not item.get("called_tool") and not item.get("refused"))
        lines.append(f"| {value} | {len(items)} | {tool_calls} | {refusals} | {invalid} |")

        examples = _example_texts(items)
        if examples:
            lines.append("")
            lines.append(f"Examples for `{value}`:")
            for example in examples:
                lines.append(f"- {example}")
    lines.append("")
    return lines


def _example_texts(records: list[dict], limit: int = 3) -> list[str]:
    examples: list[str] = []
    for record in records:
        text = record.get("refusal_text") or record.get("answer_text")
        if text:
            compact = " ".join(str(text).split())
            examples.append(compact[:160])
        if len(examples) >= limit:
            break
    return examples
