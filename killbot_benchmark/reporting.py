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
    popovers: dict[str, dict[str, str]] = {}
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
        country = str(record.get("model_country_of_origin", "")).strip()
        weights = str(record.get("model_weights", "")).strip()
        intelligence = str(record.get("model_artificial_analysis_benchmark_intelligence", "")).strip()
        scenario_popover_id = _register_popover(
            popovers,
            title=str(record.get("scenario_id", "")) or "Scenario",
            body_html=_scenario_popover_body(record),
        )
        tool_popover_id = _register_popover(
            popovers,
            title=str(record.get("tool_variant_id", "")) or "Tool",
            body_html=_tool_popover_body(record),
        )
        outcome_popover_id = _register_popover(
            popovers,
            title=f"{outcome}: {record.get('model_id', '') or 'Unknown model'}",
            body_html=_agent_response_popover_body(record),
        )
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
                    (
                        "        <td>"
                        f'<button type="button" class="table-link scenario-link" data-popover-id="{html.escape(scenario_popover_id, quote=True)}">'
                        f"{html.escape(str(record.get('scenario_id', '')))}"
                        "</button>"
                        "</td>"
                    ),
                    f"        <td>{html.escape(str(record.get('model_id', '')))}</td>",
                    f"        <td>{_country_display(country)}</td>",
                    f"        <td>{html.escape(weights or '—')}</td>",
                    f"        <td>{html.escape(intelligence or '—')}</td>",
                    f"        <td>{html.escape(str(record.get('prompt_id', '')))}</td>",
                    (
                        "        <td>"
                        f'<button type="button" class="table-link tool-link" data-popover-id="{html.escape(tool_popover_id, quote=True)}">'
                        f"{html.escape(str(record.get('tool_variant_id', '')))}"
                        "</button>"
                        "</td>"
                    ),
                    (
                        "        <td>"
                        f'<button type="button" class="outcome outcome-{_outcome_class(outcome)}" data-popover-id="{html.escape(outcome_popover_id, quote=True)}">'
                        f"{_outcome_label_html(outcome)}"
                        "</button>"
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
        gap: 0.45rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.92rem;
        border: 0;
        cursor: pointer;
        font: inherit;
      }}

      .outcome-tool-call {{
        color: var(--error);
        background: rgba(185, 28, 28, 0.12);
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

      .table-link {{
        padding: 0;
        border: 0;
        background: transparent;
        color: inherit;
        font: inherit;
        text-align: left;
        cursor: pointer;
        text-decoration: underline;
        text-decoration-color: color-mix(in srgb, var(--muted) 55%, transparent);
        text-underline-offset: 0.18em;
      }}

      .table-link:hover {{
        color: var(--error);
      }}

      .flag {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 1.8rem;
        font-size: 1.2rem;
        line-height: 1;
        cursor: help;
      }}

      .icon-skull {{
        font-size: 0.95rem;
        line-height: 1;
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

      dialog {{
        width: min(840px, calc(100vw - 2rem));
        max-height: calc(100vh - 2rem);
        padding: 0;
        border: 1px solid var(--line);
        border-radius: 24px;
        background: color-mix(in srgb, var(--panel) 94%, white);
        box-shadow: 0 30px 90px rgba(42, 28, 7, 0.24);
      }}

      dialog::backdrop {{
        background: rgba(26, 19, 10, 0.55);
        backdrop-filter: blur(4px);
      }}

      .popover-shell {{
        display: grid;
        gap: 1rem;
        padding: 1.25rem;
      }}

      .popover-header {{
        display: flex;
        align-items: start;
        justify-content: space-between;
        gap: 1rem;
      }}

      .popover-header h2 {{
        margin: 0;
        font-size: 1.45rem;
        line-height: 1.1;
      }}

      .popover-close {{
        padding: 0.55rem 0.85rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: var(--panel);
        color: var(--ink);
        font: inherit;
        cursor: pointer;
      }}

      .popover-content {{
        overflow: auto;
        color: var(--ink);
      }}

      .popover-grid {{
        display: grid;
        grid-template-columns: minmax(220px, 0.9fr) minmax(0, 1.1fr);
        gap: 1rem;
      }}

      .popover-figure {{
        margin: 0;
        display: grid;
        gap: 0.6rem;
      }}

      .popover-figure img {{
        width: 100%;
        max-height: 420px;
        object-fit: contain;
        border-radius: 18px;
        border: 1px solid var(--line);
        background:
          linear-gradient(135deg, rgba(216, 208, 195, 0.22), rgba(255, 255, 255, 0.5)),
          repeating-linear-gradient(
            -45deg,
            rgba(216, 208, 195, 0.25),
            rgba(216, 208, 195, 0.25) 8px,
            transparent 8px,
            transparent 16px
          );
      }}

      .popover-copy {{
        display: grid;
        gap: 0.85rem;
      }}

      .meta {{
        margin: 0;
        font-size: 0.84rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
      }}

      .code-block {{
        margin: 0;
        padding: 0.95rem 1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.72);
        font-family: "SFMono-Regular", "SF Mono", Consolas, "Liberation Mono", monospace;
        font-size: 0.9rem;
        line-height: 1.55;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }}

      .section-stack {{
        display: grid;
        gap: 0.9rem;
      }}

      .section-stack section {{
        display: grid;
        gap: 0.45rem;
      }}

      .section-stack h3 {{
        margin: 0;
        font-size: 0.92rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--muted);
      }}

      @media (max-width: 760px) {{
        .popover-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Benchmark Report</h1>
      <p>Click scenarios for the prompt and image, tools for the definition, and outcomes for the captured agent response.</p>

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
    <dialog id="popover-dialog" aria-labelledby="popover-title">
      <div class="popover-shell">
        <div class="popover-header">
          <h2 id="popover-title">Details</h2>
          <button type="button" class="popover-close" id="popover-close">Close</button>
        </div>
        <div class="popover-content" id="popover-content"></div>
      </div>
    </dialog>
    <script id="popover-data" type="application/json">{html.escape(json.dumps(popovers, ensure_ascii=False), quote=False)}</script>
    <script>
      const countryFilter = document.getElementById("country-filter");
      const weightsFilter = document.getElementById("weights-filter");
      const sortSelect = document.getElementById("sort-select");
      const tableBody = document.getElementById("results-body");
      const popoverData = JSON.parse(document.getElementById("popover-data").textContent);
      const popoverDialog = document.getElementById("popover-dialog");
      const popoverTitle = document.getElementById("popover-title");
      const popoverContent = document.getElementById("popover-content");
      const popoverClose = document.getElementById("popover-close");
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

      function openPopover(id) {{
        const item = popoverData[id];
        if (!item) {{
          return;
        }}
        popoverTitle.textContent = item.title || "Details";
        popoverContent.innerHTML = item.body_html || "<p>No details available.</p>";
        if (popoverDialog.open) {{
          popoverDialog.close();
        }}
        popoverDialog.showModal();
      }}

      countryFilter.addEventListener("change", renderRows);
      weightsFilter.addEventListener("change", renderRows);
      sortSelect.addEventListener("change", renderRows);
      popoverClose.addEventListener("click", () => popoverDialog.close());
      document.addEventListener("click", (event) => {{
        const trigger = event.target.closest("[data-popover-id]");
        if (!trigger) {{
          return;
        }}
        openPopover(trigger.dataset.popoverId);
      }});
      popoverDialog.addEventListener("click", (event) => {{
        const rect = popoverDialog.getBoundingClientRect();
        const clickedOutside =
          event.clientX < rect.left ||
          event.clientX > rect.right ||
          event.clientY < rect.top ||
          event.clientY > rect.bottom;

        if (clickedOutside) {{
          popoverDialog.close();
        }}
      }});
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


def _register_popover(popovers: dict[str, dict[str, str]], title: str, body_html: str) -> str:
    key = f"popover-{len(popovers)}"
    popovers[key] = {"title": title, "body_html": body_html}
    return key


def _country_display(country: str) -> str:
    if not country:
        return "—"

    normalized = country.strip().lower()
    flag = {
        "china": "&#x1F1E8;&#x1F1F3;",
        "us": "&#x1F1FA;&#x1F1F8;",
        "usa": "&#x1F1FA;&#x1F1F8;",
        "united states": "&#x1F1FA;&#x1F1F8;",
        "united states of america": "&#x1F1FA;&#x1F1F8;",
    }.get(normalized)
    if not flag:
        return html.escape(country)
    return (
        f'<span class="flag" title="{html.escape(country, quote=True)}" '
        f'aria-label="{html.escape(country, quote=True)}">{flag}</span>'
    )


def _outcome_label_html(outcome: str) -> str:
    label = html.escape(outcome)
    if outcome == "Tool call":
        return f'<span class="icon-skull" aria-hidden="true">&#9760;</span><span>{label}</span>'
    return f"<span>{label}</span>"


def _scenario_popover_body(record: dict) -> str:
    image_markup = _scenario_image_markup(record)
    prompt_text = str(record.get("prompt_text", "")).strip() or "Prompt text unavailable in this result file."
    prompt_id = str(record.get("prompt_id", "")).strip()
    scenario_description = str(record.get("scenario_description", "")).strip()

    description_markup = (
        f"<p>{html.escape(scenario_description)}</p>" if scenario_description else "<p>No scenario description provided.</p>"
    )

    return (
        '<div class="popover-grid">'
        f"{image_markup}"
        '<div class="popover-copy">'
        f'<p class="meta">Prompt {html.escape(prompt_id or "Unknown")}</p>'
        f"{description_markup}"
        f'<pre class="code-block">{html.escape(prompt_text)}</pre>'
        "</div>"
        "</div>"
    )


def _scenario_image_markup(record: dict) -> str:
    image_path = str(record.get("image_path", "")).strip()
    scenario_label = str(record.get("scenario_label", "")).strip() or str(record.get("scenario_id", "")).strip()
    if image_path:
        try:
            image_uri = Path(image_path).expanduser().resolve().as_uri()
        except ValueError:
            image_uri = ""
        if image_uri:
            return (
                '<figure class="popover-figure">'
                f'<img src="{html.escape(image_uri, quote=True)}" alt="{html.escape(scenario_label or "Scenario image", quote=True)}">'
                f"<figcaption>{html.escape(image_path)}</figcaption>"
                "</figure>"
            )

    return (
        '<figure class="popover-figure">'
        '<div class="code-block">No scenario image path available.</div>'
        "</figure>"
    )


def _tool_popover_body(record: dict) -> str:
    tool_definition = record.get("tool_definition")
    if tool_definition:
        definition_text = json.dumps(tool_definition, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        definition_text = "Tool definition unavailable in this result file."

    return (
        '<div class="section-stack">'
        "<section>"
        "<h3>Function</h3>"
        f'<pre class="code-block">{html.escape(str(record.get("tool_variant_function_name", "")).strip() or "Unknown")}</pre>'
        "</section>"
        "<section>"
        "<h3>Definition</h3>"
        f'<pre class="code-block">{html.escape(definition_text)}</pre>'
        "</section>"
        "</div>"
    )


def _agent_response_popover_body(record: dict) -> str:
    sections = []

    summary = _hover_detail(record)
    if summary:
        sections.append(_popover_section("Summary", summary))

    for heading, key in (("Agent text", "answer_text"), ("Refusal", "refusal_text"), ("Error", "error")):
        raw_value = record.get(key)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value:
            sections.append(_popover_section(heading, value))

    message = _agent_message(record)
    if message:
        sections.append(_popover_section("Agent response", json.dumps(message, ensure_ascii=False, indent=2, sort_keys=True)))

    if not sections:
        sections.append(_popover_section("Details", "No response details were captured for this record."))

    return f'<div class="section-stack">{"".join(sections)}</div>'


def _popover_section(title: str, body: str) -> str:
    return (
        "<section>"
        f"<h3>{html.escape(title)}</h3>"
        f'<pre class="code-block">{html.escape(body)}</pre>'
        "</section>"
    )


def _agent_message(record: dict) -> dict | None:
    agent_response = record.get("agent_response")
    if isinstance(agent_response, dict) and agent_response:
        return agent_response

    raw_response = record.get("raw_response")
    if not isinstance(raw_response, dict):
        return None

    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if isinstance(message, dict) and message:
        return message
    return None
