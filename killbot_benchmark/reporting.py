from __future__ import annotations

import csv
import html
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
        "model_country_of_origin",
        "model_weights",
        "model_artificial_analysis_benchmark_intelligence",
        "prompt_id",
        "tool_variant_id",
        "tool_variant_function_name",
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

    for key in ("model_id", "prompt_id", "tool_variant_id", "scenario_id"):
        lines.extend(_group_section(records, key))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_html_report(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    total_runs = len(records)
    tool_calls = sum(1 for record in records if record.get("called_tool"))
    refusals = sum(1 for record in records if record.get("refused"))
    errors = sum(1 for record in records if record.get("error"))
    other = sum(1 for record in records if _outcome_label(record) == "No action")

    country_options = sorted(
        {
            str(record.get("model_country_of_origin", "")).strip()
            for record in records
            if str(record.get("model_country_of_origin", "")).strip()
        }
    )
    weights_options = sorted(
        {
            str(record.get("model_weights", "")).strip()
            for record in records
            if str(record.get("model_weights", "")).strip()
        }
    )

    rows = []
    for record in sorted(
        records,
        key=lambda item: (
            str(item.get("scenario_id", "")),
            str(item.get("model_id", "")),
            str(item.get("prompt_id", "")),
            str(item.get("tool_variant_id", "")),
            str(item.get("timestamp", "")),
        ),
    ):
        outcome = _outcome_label(record)
        detail = _hover_detail(record)
        hover_attr = f' title="{html.escape(detail, quote=True)}"' if detail else ""
        country = str(record.get("model_country_of_origin", "")).strip()
        weights = str(record.get("model_weights", "")).strip()
        intelligence = str(record.get("model_artificial_analysis_benchmark_intelligence", "")).strip()
        rows.append(
            "\n".join(
                [
                    (
                        '      <tr'
                        f' data-country="{html.escape(country, quote=True)}"'
                        f' data-weights="{html.escape(weights, quote=True)}"'
                        f' data-intelligence="{html.escape(intelligence, quote=True)}"'
                        ">"
                    ),
                    f"        <td>{html.escape(str(record.get('scenario_id', '')))}</td>",
                    f"        <td>{html.escape(str(record.get('model_id', '')))}</td>",
                    f"        <td>{html.escape(country or '—')}</td>",
                    f"        <td>{html.escape(weights or '—')}</td>",
                    f"        <td>{html.escape(intelligence or '—')}</td>",
                    f"        <td>{html.escape(str(record.get('prompt_id', '')))}</td>",
                    f"        <td>{html.escape(str(record.get('tool_variant_id', '')))}</td>",
                    (
                        "        <td>"
                        f'<span class="outcome outcome-{_outcome_class(outcome)}"{hover_attr}>'
                        f"{html.escape(outcome)}"
                        "</span>"
                        "</td>"
                    ),
                    "      </tr>",
                ]
            )
        )

    document = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Benchmark Report</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f5f1e8;
        --panel: #fffdf8;
        --ink: #1e1c18;
        --muted: #6b655c;
        --line: #d8d0c3;
        --tool: #2d6a4f;
        --refusal: #9a3412;
        --error: #b91c1c;
        --other: #57534e;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(174, 192, 153, 0.3), transparent 28rem),
          linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
      }}

      main {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 3rem 1.25rem 4rem;
      }}

      h1 {{
        margin: 0 0 0.5rem;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1;
      }}

      p {{
        margin: 0;
        color: var(--muted);
      }}

      .summary {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.9rem;
        margin: 2rem 0;
      }}

      .card, .table-wrap {{
        background: color-mix(in srgb, var(--panel) 90%, white);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 14px 40px rgba(55, 42, 18, 0.08);
      }}

      .card {{
        padding: 1rem 1.1rem;
      }}

      .card strong {{
        display: block;
        font-size: 1.8rem;
        line-height: 1;
        margin-bottom: 0.35rem;
      }}

      .table-wrap {{
        overflow: auto;
      }}

      table {{
        width: 100%;
        border-collapse: collapse;
      }}

      th, td {{
        padding: 0.9rem 1rem;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }}

      th {{
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
      }}

      tr:last-child td {{
        border-bottom: 0;
      }}

      .outcome {{
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.92rem;
        cursor: help;
      }}

      .outcome-tool-call {{
        color: var(--tool);
        background: rgba(45, 106, 79, 0.12);
      }}

      .outcome-refusal {{
        color: var(--refusal);
        background: rgba(154, 52, 18, 0.12);
      }}

      .outcome-error {{
        color: var(--error);
        background: rgba(185, 28, 28, 0.1);
      }}

      .outcome-no-action {{
        color: var(--other);
        background: rgba(87, 83, 78, 0.12);
      }}

      .hint {{
        margin: 0 0 1rem;
      }}

      .controls {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.9rem;
        margin: 0 0 1.25rem;
      }}

      label {{
        display: grid;
        gap: 0.4rem;
        font-size: 0.9rem;
        color: var(--muted);
      }}

      select {{
        width: 100%;
        padding: 0.7rem 0.8rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: var(--panel);
        color: var(--ink);
        font: inherit;
      }}

      .empty-state {{
        text-align: center;
        color: var(--muted);
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Benchmark Report</h1>
      <p>Hover an outcome badge to inspect the model's text, refusal, tool call details, or errors.</p>

      <section class="summary" aria-label="Summary">
        <article class="card"><strong>{total_runs}</strong><span>Total runs</span></article>
        <article class="card"><strong>{tool_calls}</strong><span>Tool calls</span></article>
        <article class="card"><strong>{refusals}</strong><span>Refusals</span></article>
        <article class="card"><strong>{errors}</strong><span>Errors</span></article>
        <article class="card"><strong>{other}</strong><span>No action</span></article>
      </section>

      <section class="controls" aria-label="Filters">
        <label for="country-filter">
          Country of origin
          <select id="country-filter">
            <option value="">All countries</option>
            {"".join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in country_options)}
          </select>
        </label>
        <label for="weights-filter">
          Weights
          <select id="weights-filter">
            <option value="">All weight types</option>
            {"".join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in weights_options)}
          </select>
        </label>
        <label for="sort-select">
          Sort rows
          <select id="sort-select">
            <option value="default">Default</option>
            <option value="intelligence-desc">Intelligence high to low</option>
            <option value="intelligence-asc">Intelligence low to high</option>
          </select>
        </label>
      </section>

      <p class="hint">Filter by model metadata, then sort by intelligence to compare outcomes.</p>

      <section class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Model</th>
              <th>Country</th>
              <th>Weights</th>
              <th>Intelligence</th>
              <th>Prompt</th>
              <th>Tool Variant</th>
              <th>Outcome</th>
            </tr>
          </thead>
          <tbody id="results-body">
{chr(10).join(rows) if rows else '            <tr><td colspan="8" class="empty-state">No records found.</td></tr>'}
          </tbody>
        </table>
      </section>
    </main>
    <script>
      const countryFilter = document.getElementById("country-filter");
      const weightsFilter = document.getElementById("weights-filter");
      const sortSelect = document.getElementById("sort-select");
      const tableBody = document.getElementById("results-body");
      const rows = Array.from(tableBody.querySelectorAll("tr")).filter(
        (row) => !row.classList.contains("empty-state")
      );
      const defaultOrder = rows.slice();

      function parseIntelligence(value) {{
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : null;
      }}

      function rowMatches(row) {{
        const countryMatch = !countryFilter.value || row.dataset.country === countryFilter.value;
        const weightsMatch = !weightsFilter.value || row.dataset.weights === weightsFilter.value;
        return countryMatch && weightsMatch;
      }}

      function compareRows(left, right) {{
        const direction =
          sortSelect.value === "intelligence-asc"
            ? 1
            : sortSelect.value === "intelligence-desc"
              ? -1
              : 0;

        if (!direction) {{
          return defaultOrder.indexOf(left) - defaultOrder.indexOf(right);
        }}

        const leftValue = parseIntelligence(left.dataset.intelligence);
        const rightValue = parseIntelligence(right.dataset.intelligence);

        if (leftValue === null && rightValue === null) {{
          return defaultOrder.indexOf(left) - defaultOrder.indexOf(right);
        }}
        if (leftValue === null) {{
          return 1;
        }}
        if (rightValue === null) {{
          return -1;
        }}
        if (leftValue === rightValue) {{
          return defaultOrder.indexOf(left) - defaultOrder.indexOf(right);
        }}
        return (leftValue - rightValue) * direction;
      }}

      function renderRows() {{
        const visibleRows = rows.filter(rowMatches).sort(compareRows);
        tableBody.innerHTML = "";

        if (!visibleRows.length) {{
          tableBody.innerHTML = '<tr><td colspan="8" class="empty-state">No rows match the selected filters.</td></tr>';
          return;
        }}

        for (const row of visibleRows) {{
          tableBody.appendChild(row);
        }}
      }}

      countryFilter.addEventListener("change", renderRows);
      weightsFilter.addEventListener("change", renderRows);
      sortSelect.addEventListener("change", renderRows);
      renderRows();
    </script>
  </body>
</html>
"""
    path.write_text(document, encoding="utf-8")


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


def _outcome_label(record: dict) -> str:
    if record.get("error"):
        return "Error"
    if record.get("called_tool"):
        return "Tool call"
    if record.get("refused"):
        return "Refusal"
    return "No action"


def _outcome_class(outcome: str) -> str:
    return outcome.lower().replace(" ", "-")


def _hover_detail(record: dict) -> str:
    parts: list[str] = []

    if record.get("tool_name"):
        if record.get("x") is not None and record.get("y") is not None:
            parts.append(f"Tool: {record['tool_name']}({record['x']}, {record['y']})")
        else:
            parts.append(f"Tool: {record['tool_name']}")
    if record.get("tool_variant_id"):
        parts.append(f"Tool variant: {record['tool_variant_id']}")

    for key, label in (("refusal_text", "Refusal"), ("answer_text", "Text"), ("error", "Error")):
        value = record.get(key)
        if value:
            compact = " ".join(str(value).split())
            parts.append(f"{label}: {compact}")

    return " | ".join(parts)
