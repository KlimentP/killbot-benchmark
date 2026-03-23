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


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
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
    actions_pct = f"{round(tool_calls / total_runs * 100)}%" if total_runs > 0 else "0%"
    errors = sum(1 for record in records if record.get("error"))
    other = sum(1 for record in records if _outcome_label(record) == "No action")

    model_options = sorted({str(record.get("model_id", "")).strip() for record in records if str(record.get("model_id", "")).strip()})
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
    scenario_options = sorted({str(record.get("scenario_id", "")).strip() for record in records if str(record.get("scenario_id", "")).strip()})
    prompt_options = sorted({str(record.get("prompt_id", "")).strip() for record in records if str(record.get("prompt_id", "")).strip()})
    tool_variant_options = sorted(
        {str(record.get("tool_variant_id", "")).strip() for record in records if str(record.get("tool_variant_id", "")).strip()}
    )

    rows = []
    popovers: dict[str, dict[str, str]] = {}
    model_popover_ids: dict[str, str] = {}
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
        model_id = str(record.get("model_id", "")).strip()
        model_popover_id = model_popover_ids.get(model_id)
        if model_id and model_popover_id is None:
            model_popover_id = _register_popover(
                popovers,
                title=model_id,
                body_html=_model_info_popover_body(record),
            )
            model_popover_ids[model_id] = model_popover_id
        details_popover_id = _register_popover(
            popovers,
            title=str(record.get("scenario_label", "")).strip() or str(record.get("scenario_id", "")) or "Details",
            body_html=_details_popover_body(record),
        )
        rows.append(
            "\n".join(
                [
                    (
                        '      <tr'
                        f' data-model="{html.escape(str(record.get("model_id", "")).strip(), quote=True)}"'
                        f' data-country="{html.escape(country, quote=True)}"'
                        f' data-weights="{html.escape(weights, quote=True)}"'
                        f' data-scenario="{html.escape(str(record.get("scenario_id", "")).strip(), quote=True)}"'
                        f' data-prompt="{html.escape(str(record.get("prompt_id", "")).strip(), quote=True)}"'
                        f' data-tool-variant="{html.escape(str(record.get("tool_variant_id", "")).strip(), quote=True)}"'
                        f' data-intelligence="{html.escape(intelligence, quote=True)}"'
                        f' data-called-tool="{"1" if record.get("called_tool") else "0"}"'
                        ">"
                    ),
                    f"        <td>{_model_cell_markup(model_id, model_popover_id or '')}</td>",
                    f"        <td>{html.escape(str(record.get('scenario_id', '')))}</td>",
                    f"        <td>{html.escape(str(record.get('prompt_id', '')))}</td>",
                    (
                        "        <td>"
                        f"{html.escape(str(record.get('tool_variant_id', '')))}"
                        "</td>"
                    ),
                    (
                        '        <td class="outcome-cell">'
                        f'<button type="button" class="outcome-trigger" data-popover-id="{html.escape(details_popover_id, quote=True)}" '
                        'title="Show details" '
                        f'aria-label="Show details for {html.escape(str(record.get("scenario_id", "")), quote=True)}">'
                        f'<span class="outcome outcome-{_outcome_class(outcome)}">'
                        f"{_outcome_label_html(outcome, record)}"
                        "</span>"
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
    <title>Killbot Benchmark Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@2.0.1/dist/chartjs-chart-matrix.min.js"></script>
    <style>

      :root {{
        --bg:        #d8d8d4;
        --page:      #ffffff;
        --ink:       #080808;
        --ink-2:     #2a2a2a;
        --ink-3:     #666666;
        --ink-4:     #999999;
        --rule:      #c0c0bc;
        --rule-hvy:  #080808;
        --red:       #b80009;
        --red-dim:   rgba(184,0,9,0.09);
        --amber:     #6b3f00;
        --amber-dim: rgba(107,63,0,0.09);
        --neutral:   #3d3d3d;
        --neutral-dim: rgba(30,30,30,0.07);
        --mono: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
        --sans: 'IBM Plex Sans', system-ui, sans-serif;
      }}

      *, *::before, *::after {{ box-sizing: border-box; margin: 0; }}

      body {{
        font-family: var(--sans);
        font-size: 14px;
        line-height: 1.6;
        color: var(--ink);
        background: var(--bg);
      }}

      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 2rem 4rem;
        background: var(--page);
        min-height: 100vh;
      }}

      /* ── Header ───────────────────────────────── */
      .report-header {{
        border-top: 3px solid var(--ink);
        padding: 1.5rem 0 1.5rem;
        margin-bottom: 0;
      }}

      .report-header h1 {{
        font-family: var(--sans);
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--ink);
        line-height: 1.2;
      }}

      .report-header .subtitle {{
        margin-top: 0.3rem;
        font-size: 0.78rem;
        color: var(--ink-3);
        font-family: var(--mono);
        letter-spacing: 0.04em;
      }}

      /* ── Summary strip ──────────────────────── */
      .summary {{
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        border-top: 1px solid var(--rule);
        border-bottom: 1px solid var(--rule);
        margin: 1.25rem 0;
        background: #f7f7f5;
      }}

      .stat {{
        flex: 1 1 11rem;
        min-width: 0;
        padding: 0.85rem 1.25rem;
        border-right: 1px solid var(--rule);
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
      }}

      .stat:last-child {{ border-right: none; }}

      .stat strong {{
        font-family: var(--mono);
        font-size: 1.6rem;
        font-weight: 600;
        line-height: 1;
        color: var(--ink);
      }}

      .stat span {{
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ink-3);
      }}

      /* ── Heatmap ────────────────────────────── */
      .heatmap-card {{
        margin: 0 0 2rem;
        border: 1px solid var(--rule);
        padding: 1.5rem 1.25rem 1rem;
        background: var(--page);
        overflow-x: auto;
        overflow-y: hidden;
      }}
      .heatmap-card h2 {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-3);
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--rule);
        padding-bottom: 0.5rem;
      }}
      .canvas-wrap {{
        position: relative;
        width: 100%;
        min-width: 0;
        max-width: 100%;
        margin: 0 auto;
        height: 420px;
      }}

      .canvas-wrap canvas {{
        display: block;
        width: 100%;
        height: 100%;
      }}

      /* ── Filters ────────────────────────────── */
      .controls {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 0.6rem;
        margin-bottom: 1rem;
        align-items: end;
      }}

      .filter-group {{
        display: grid;
        gap: 0.28rem;
      }}

      .filter-group label {{
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-3);
      }}

      select {{
        width: 100%;
        padding: 0.42rem 1.6rem 0.42rem 0.55rem;
        border: 1px solid var(--rule);
        border-radius: 0;
        background: var(--page) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23666'/%3E%3C/svg%3E") no-repeat right 0.55rem center;
        color: var(--ink);
        font: 0.8rem/1.4 var(--sans);
        appearance: none;
        cursor: pointer;
      }}

      select:focus {{
        outline: 2px solid var(--ink);
        outline-offset: -1px;
      }}

      button.clear-btn {{
        display: none;
        height: 2.3rem;
        padding: 0 1rem;
        border: 1px solid var(--rule);
        background: #eee;
        color: var(--ink-2);
        font: 600 0.68rem var(--sans);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        cursor: pointer;
        transition: all 0.1s;
      }}
      button.clear-btn:hover {{
        background: #e0e0e0;
        border-color: var(--ink-3);
      }}

      .hint {{
        font-size: 0.76rem;
        color: var(--ink-3);
        margin-bottom: 0.85rem;
      }}

      /* ── Table ──────────────────────────────── */
      .table-wrap {{
        overflow-x: auto;
        border: 1px solid var(--rule-hvy);
      }}

      table {{
        width: max(100%, 64rem);
        border-collapse: collapse;
        font-size: 0.8rem;
      }}

      thead tr {{
        border-bottom: 3px solid var(--ink);
        background: #f0f0ee;
      }}

      th {{
        padding: 0.6rem 0.9rem;
        text-align: left;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-2);
        white-space: nowrap;
        background: #f0f0ee;
      }}

      td {{
        padding: 0.6rem 0.9rem;
        border-bottom: 1px solid var(--rule);
        vertical-align: middle;
        color: var(--ink);
        background: var(--page);
      }}

      tbody tr:last-child td {{ border-bottom: none; }}

      tbody tr:hover td {{ background: #f7f7f5; }}

      .model-col    {{ width: 17rem; }}
      .outcome-col  {{ width: 13rem; }}

      /* ── Outcome badges ─────────────────────── */
      .outcome {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.18rem 0.55rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        white-space: nowrap;
        border-width: 1px;
        border-style: solid;
      }}

      .outcome-tool-call {{
        color: var(--red);
        background: var(--red-dim);
        border-color: var(--red);
      }}

      .outcome-refusal {{
        color: var(--amber);
        background: var(--amber-dim);
        border-color: var(--amber);
      }}

      .outcome-error {{
        color: var(--neutral);
        background: var(--neutral-dim);
        border-color: #aaaaaa;
      }}

      .outcome-no-action {{
        color: var(--neutral);
        background: var(--neutral-dim);
        border-color: #aaaaaa;
      }}

      .outcome-detail {{
        font-family: var(--mono);
        font-weight: 400;
        font-size: 0.68rem;
      }}

      .icon-skull {{ font-size: 0.82rem; }}

      .outcome-label {{ white-space: nowrap; }}

      .outcome-cell {{
        padding: 0.35rem 0.55rem;
      }}

      .outcome-trigger {{
        display: flex;
        align-items: center;
        width: 100%;
        padding: 0.25rem 0.25rem 0.25rem 0.45rem;
        border: none;
        border-left: 2px solid transparent;
        background: transparent;
        cursor: pointer;
        text-align: left;
        transform: translateX(0);
        transition: background 0.08s ease, border-left-color 0.08s ease, transform 0.08s ease;
      }}

      .outcome-trigger:hover,
      .outcome-trigger:focus-visible {{
        background: rgba(8,8,8,0.045);
        border-left-color: var(--ink);
        transform: translateX(2px);
      }}

      .outcome-trigger:focus-visible {{
        outline: 2px solid var(--red);
        outline-offset: 2px;
      }}

      .outcome-trigger:hover .outcome,
      .outcome-trigger:focus-visible .outcome {{
        box-shadow: 0 0 0 1px currentColor;
      }}

      /* ── Model trigger ──────────────────────── */
      .model-trigger {{
        background: none;
        border: none;
        padding: 0;
        font-family: var(--mono);
        font-size: 0.76rem;
        font-weight: 600;
        color: var(--ink);
        cursor: pointer;
        text-align: left;
        text-decoration: underline;
        text-decoration-color: transparent;
        text-underline-offset: 2px;
        transition: color 0.12s, text-decoration-color 0.12s;
      }}

      .model-trigger:hover {{
        color: var(--red);
        text-decoration-color: var(--red);
      }}

      .flag {{
        display: inline-flex;
        align-items: center;
        font-size: 1rem;
        cursor: help;
      }}

      /* ── Dialog ─────────────────────────────── */
      dialog {{
        width: min(860px, calc(100vw - 2rem));
        max-height: calc(100dvh - 3rem);
        padding: 0;
        border: 2px solid var(--ink);
        border-radius: 0;
        background: var(--page);
        box-shadow: 6px 6px 0 rgba(0,0,0,0.15);
      }}

      dialog::backdrop {{
        background: rgba(0,0,0,0.5);
        backdrop-filter: blur(2px);
      }}

      .popover-shell {{
        display: flex;
        flex-direction: column;
        max-height: calc(100dvh - 3rem);
      }}

      .popover-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.9rem 1.25rem;
        border-bottom: 1px solid var(--rule);
        flex-shrink: 0;
      }}

      .popover-header h2 {{
        font-size: 0.9rem;
        font-weight: 600;
        font-family: var(--mono);
        color: var(--ink);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}

      .popover-close {{
        flex-shrink: 0;
        padding: 0.28rem 0.75rem;
        border: 1px solid var(--rule);
        border-radius: 0;
        background: none;
        color: var(--ink-2);
        font: 600 0.72rem/1.4 var(--sans);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        cursor: pointer;
        transition: border-color 0.1s, color 0.1s;
      }}

      .popover-close:hover {{ border-color: var(--ink); color: var(--ink); }}

      .popover-content {{
        overflow: auto;
        padding: 1.25rem;
        flex: 1;
      }}

      /* ── Popover tabs ───────────────────────── */
      .popover-tabs {{
        display: flex;
        margin-bottom: 1.1rem;
        border-bottom: 1px solid var(--rule);
      }}

      .popover-tab {{
        padding: 0.42rem 0.9rem;
        border: none;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        background: none;
        color: var(--ink-3);
        font: 600 0.72rem/1.4 var(--sans);
        letter-spacing: 0.07em;
        text-transform: uppercase;
        cursor: pointer;
        transition: color 0.1s, border-bottom-color 0.1s;
      }}

      .popover-tab:hover {{ color: var(--ink); }}

      .popover-tab.is-active {{
        color: var(--ink);
        border-bottom-color: var(--ink);
      }}

      .popover-panel[hidden] {{ display: none; }}

      /* ── Popover grid ───────────────────────── */
      .popover-grid {{
        display: grid;
        grid-template-columns: minmax(200px, 0.85fr) minmax(0, 1.15fr);
        gap: 1.25rem;
      }}

      @media (max-width: 600px) {{
        .popover-grid {{ grid-template-columns: 1fr; }}
      }}

      .popover-figure {{ margin: 0; }}

      .scenario-image-frame {{ position: relative; }}

      .popover-figure img {{
        display: block;
        width: 100%;
        max-height: 400px;
        object-fit: contain;
        border: 1px solid var(--rule);
        background: repeating-linear-gradient(-45deg, #efefef 0 6px, #f9f9f9 6px 12px);
      }}

      .scenario-marker {{
        position: absolute;
        left: 0; top: 0;
        width: 16px; height: 16px;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        background: var(--red);
        border: 2px solid #fff;
        box-shadow: 0 0 0 1px var(--red);
        pointer-events: none;
      }}

      /* ── Section stack ──────────────────────── */
      .section-stack {{ display: grid; gap: 0.9rem; }}

      .section-stack section {{ display: grid; gap: 0.35rem; }}

      .section-stack h3 {{
        font-size: 0.64rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-3);
      }}

      .code-block {{
        margin: 0;
        padding: 0.7rem 0.85rem;
        border: 1px solid var(--rule);
        background: #f5f5f5;
        font-family: var(--mono);
        font-size: 0.76rem;
        line-height: 1.6;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
      }}

      .detail-chip-row {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
      }}

      .detail-note {{
        font-size: 0.8rem;
        color: var(--ink-3);
      }}

      .meta {{
        font-size: 0.7rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--ink-3);
        font-family: var(--mono);
      }}

      .empty-state {{
        text-align: center;
        padding: 2.5rem;
        color: var(--ink-3);
        font-style: italic;
      }}

      @media (max-width: 700px) {{
        main {{
          padding: 1.25rem 1rem 2.5rem;
        }}

        .report-header {{
          padding-top: 1.1rem;
        }}

        .summary {{
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }}

        .stat {{
          padding: 0.65rem 0.5rem 0.7rem;
          gap: 0.28rem;
          flex-direction: column;
          align-items: flex-start;
          border-right: 1px solid var(--rule);
        }}

        .stat strong {{
          font-size: clamp(1.05rem, 7vw, 1.5rem);
        }}

        .stat span {{
          font-size: 0.58rem;
          letter-spacing: 0.08em;
          line-height: 1.2;
        }}

        .heatmap-card {{
          padding: 1rem 0.85rem 0.85rem;
        }}

        .heatmap-card h2 {{
          margin-bottom: 1rem;
        }}

        .canvas-wrap {{
          height: 320px;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <header class="report-header">
        <h1>Killbot Benchmark — Results Report</h1>
        <p class="subtitle">Autonomous weapons compliance evaluation under different scenarios</p>
      </header>

      <section class="controls" aria-label="Filters">
        <div class="filter-group">
          <label for="model-filter">Model</label>
          <select id="model-filter">
            <option value="">All models</option>
            {"".join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in model_options)}
          </select>
        </div>
        <div class="filter-group">
          <label for="country-filter">Country of origin</label>
          <select id="country-filter">
            <option value="">All countries</option>
            {"".join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in country_options)}
          </select>
        </div>
        <div class="filter-group">
          <label for="weights-filter">Weights</label>
          <select id="weights-filter">
            <option value="">All weight types</option>
            {"".join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in weights_options)}
          </select>
        </div>
        <div class="filter-group">
          <label for="scenario-filter">Scenario</label>
          <select id="scenario-filter">
            <option value="">All scenarios</option>
            {"".join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in scenario_options)}
          </select>
        </div>
        <div class="filter-group">
          <label for="prompt-filter">Prompt</label>
          <select id="prompt-filter">
            <option value="">All prompts</option>
            {"".join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in prompt_options)}
          </select>
        </div>
        <div class="filter-group">
          <label for="tool-variant-filter">Tool variant</label>
          <select id="tool-variant-filter">
            <option value="">All tool variants</option>
            {"".join(f'<option value="{html.escape(option, quote=True)}">{html.escape(option)}</option>' for option in tool_variant_options)}
          </select>
        </div>
        <div class="filter-group">
          <label for="sort-select">Sort</label>
          <select id="sort-select">
            <option value="default">Default order</option>
            <option value="intelligence-desc">Intelligence &#x2193;</option>
            <option value="intelligence-asc">Intelligence &#x2191;</option>
          </select>
        </div>
        <div class="filter-group">
          <button type="button" id="clear-filters" class="clear-btn">Clear Filters</button>
        </div>
      </section>

      <section class="summary" aria-label="Summary statistics">
        <div class="stat"><strong id="stat-actions">{tool_calls}</strong><span>Actions</span></div>
        <div class="stat"><strong id="stat-total">{total_runs}</strong><span>Runs</span></div>
        <div class="stat"><strong id="stat-pct">{actions_pct}</strong><span>Compliance</span></div>
      </section>

      <section class="heatmap-card">
        <h2>Compliance Heatmap — Actions by Prompt × Scenario</h2>
        <div class="canvas-wrap">
          <canvas id="heatmapChart"></canvas>
        </div>
      </section>

      <p class="hint">Click a model name for metadata &middot; click an outcome to inspect response and scenario</p>

      <section class="table-wrap" aria-label="Results table">
        <table>
          <colgroup>
            <col class="model-col">
            <col>
            <col>
            <col class="outcome-col">
          </colgroup>
          <thead>
            <tr>
              <th>Model</th>
              <th>Scenario</th>
              <th>Prompt</th>
              <th>Tool Variant</th>
              <th>Outcome</th>
            </tr>
          </thead>
          <tbody id="results-body">
{chr(10).join(rows) if rows else '            <tr><td colspan="5" class="empty-state">No records found.</td></tr>'}
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
    <script id="popover-data" type="application/json">{_json_for_html_script_tag(popovers)}</script>
    <script>
      const modelFilter = document.getElementById("model-filter");
      const countryFilter = document.getElementById("country-filter");
      const weightsFilter = document.getElementById("weights-filter");
      const scenarioFilter = document.getElementById("scenario-filter");
      const promptFilter = document.getElementById("prompt-filter");
      const toolVariantFilter = document.getElementById("tool-variant-filter");
      const sortSelect = document.getElementById("sort-select");
      const tableBody = document.getElementById("results-body");
      const popoverData = JSON.parse(document.getElementById("popover-data").textContent);
      const popoverDialog = document.getElementById("popover-dialog");
      const popoverTitle = document.getElementById("popover-title");
      const popoverContent = document.getElementById("popover-content");
      const popoverClose = document.getElementById("popover-close");
      const statActions = document.getElementById("stat-actions");
      const statTotal = document.getElementById("stat-total");
      const statPct = document.getElementById("stat-pct");
      const clearFiltersBtn = document.getElementById("clear-filters");
      const heatmapCard = document.querySelector(".heatmap-card");
      const heatmapCanvas = document.getElementById("heatmapChart");
      const heatmapWrap = heatmapCanvas.parentElement;
      const heatmapCtx = heatmapCanvas.getContext("2d");
      let heatmapChart = null;
      const rows = Array.from(tableBody.querySelectorAll("tr")).filter(
        (row) => !row.classList.contains("empty-state")
      );
      const defaultOrder = rows.slice();

      function parseIntelligence(value) {{
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : null;
      }}

      function formatHeatmapAxisLabel(label) {{
        const [scenario = "", toolVariant = ""] = String(label).split(" · ");
        const prettify = (value) => value.replace(/-/g, " ").trim();
        return [prettify(scenario), prettify(toolVariant)];
      }}

      function rowMatches(row) {{
        const modelMatch = !modelFilter.value || row.dataset.model === modelFilter.value;
        const countryMatch = !countryFilter.value || row.dataset.country === countryFilter.value;
        const weightsMatch = !weightsFilter.value || row.dataset.weights === weightsFilter.value;
        const scenarioMatch = !scenarioFilter.value || row.dataset.scenario === scenarioFilter.value;
        const promptMatch = !promptFilter.value || row.dataset.prompt === promptFilter.value;
        const toolVariantMatch = !toolVariantFilter.value || row.dataset.toolVariant === toolVariantFilter.value;
        return modelMatch && countryMatch && weightsMatch && scenarioMatch && promptMatch && toolVariantMatch;
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

        // Update stats
        const totalRuns = visibleRows.length;
        const actions = visibleRows.reduce((a, r) => a + (r.dataset.calledTool === "1" ? 1 : 0), 0);
        const actionsPct = totalRuns > 0 ? Math.round((actions / totalRuns) * 100) : 0;
        statActions.textContent = actions;
        statTotal.textContent = totalRuns;
        statPct.textContent = actionsPct + "%";

        const isFiltered = modelFilter.value || countryFilter.value || weightsFilter.value || 
                          scenarioFilter.value || promptFilter.value || toolVariantFilter.value || 
                          sortSelect.value !== "default";
        clearFiltersBtn.style.display = isFiltered ? "inline-block" : "none";

        updateHeatmap(visibleRows);

        if (!visibleRows.length) {{
          tableBody.innerHTML = '<tr><td colspan="5" class="empty-state">No rows match the selected filters.</td></tr>';
          return;
        }}

        for (const row of visibleRows) {{
          tableBody.appendChild(row);
        }}
      }}

      function updateHeatmap(visibleRows) {{
        if (!visibleRows.length) {{
          if (heatmapChart) heatmapChart.destroy();
          heatmapChart = null;
          heatmapWrap.style.minWidth = "0px";
          heatmapWrap.style.height = window.innerWidth <= 700 ? "320px" : "420px";
          return;
        }}

        const dataGrid = {{}};
        const ySet = new Set();
        const xSet = new Set();
        const modelSet = new Set();

        visibleRows.forEach(row => {{
          const y = row.dataset.prompt;
          const x = `${{row.dataset.scenario}} · ${{row.dataset.toolVariant}}`;
          const key = `${{y}}|${{x}}`;
          if (!dataGrid[key]) dataGrid[key] = {{ v: 0 }};
          if (row.dataset.calledTool === "1") dataGrid[key].v++;
          ySet.add(y);
          xSet.add(x);
          modelSet.add(row.dataset.model);
        }});

        const yLabels = Array.from(ySet).sort();
        const xLabels = Array.from(xSet).sort();
        const totalModels = modelSet.size;
        const mobile = window.innerWidth <= 700;
        const targetColumnWidth = mobile ? 144 : 148;
        const targetRowHeight = mobile ? 82 : 128;
        const chartWidth = Math.max(heatmapCard.clientWidth - (mobile ? 28 : 40), xLabels.length * targetColumnWidth);
        const chartHeight = Math.max(mobile ? 280 : 340, yLabels.length * targetRowHeight + (mobile ? 84 : 72));

        heatmapWrap.style.minWidth = `${{chartWidth}}px`;
        heatmapWrap.style.height = `${{chartHeight}}px`;

        const dataPoints = [];
        yLabels.forEach(y => {{
          xLabels.forEach(x => {{
            const key = `${{y}}|${{x}}`;
            dataPoints.push({{ x, y, v: dataGrid[key] ? dataGrid[key].v : 0 }});
          }});
        }});

        if (heatmapChart) {{
          heatmapChart.destroy();
        }}

        heatmapChart = new Chart(heatmapCtx, {{
            type: 'matrix',
            data: {{
              datasets: [{{
                label: 'Tool compliance',
                data: dataPoints,
                backgroundColor(context) {{
                  const val = context.dataset.data[context.dataIndex];
                  const tModels = context.chart.options.totalModels;
                  if (!val || tModels === 0) return 'rgba(0,0,0,0.05)';
                  const t = val.v / tModels;
                  const stops = [
                    {{ at: 0.00, r: 26,  g: 110, b: 60  }},  
                    {{ at: 0.43, r: 192, g: 120, b: 0   }},  
                    {{ at: 1.00, r: 184, g: 0,   b: 9   }},  
                  ];
                  let lo = stops[0], hi = stops[stops.length-1];
                  for(let i=0; i<stops.length-1; i++) {{
                    if (t >= stops[i].at && t <= stops[i+1].at) {{ lo = stops[i]; hi = stops[i+1]; break; }}
                  }}
                  const localT = (hi.at === lo.at) ? 0 : (t - lo.at) / (hi.at - lo.at);
                  const lerp = (a, b, f) => Math.round(a + (b - a) * f);
                  return `rgb(${{lerp(lo.r, hi.r, localT)}}, ${{lerp(lo.g, hi.g, localT)}}, ${{lerp(lo.b, hi.b, localT)}})`;
                }},
                borderColor: '#ffffff',
                borderWidth: 2,
                width: ({{chart}}) => {{
                  const area = chart.chartArea;
                  return area ? (area.right - area.left) / xLabels.length - 2 : 0;
                }},
                height: ({{chart}}) => {{
                  const area = chart.chartArea;
                  return area ? (area.bottom - area.top) / yLabels.length - 2 : 0;
                }}
              }}]
            }},
            plugins: [{{
              id: 'cellLabels',
              afterDraw(chart) {{
                const {{ctx, data}} = chart;
                const tModels = chart.options.totalModels;
                const meta = chart.getDatasetMeta(0);
                ctx.save();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                meta.data.forEach((el, idx) => {{
                  const {{v}} = data.datasets[0].data[idx];
                  const {{x, y}} = el.getCenterPoint();
                  ctx.fillStyle = '#ffffff';
                  ctx.font = `600 11px var(--mono)`;
                  ctx.fillText(`${{v}}/${{tModels}}`, x, y);
                }});
                ctx.restore();
              }}
            }}],
            options: {{
              responsive: true,
              maintainAspectRatio: false,
              totalModels: totalModels,
              layout: {{
                padding: {{
                  top: mobile ? 10 : 14,
                  right: 8,
                  bottom: 8,
                  left: 8
                }}
              }},
              scales: {{
                x: {{ 
                  type: 'category', 
                  labels: xLabels, 
                  position: 'top', 
                  grid: {{display: false}},
                  offset: true,
                  ticks: {{ 
                    color: '#333333',
                    font: {{family: "'IBM Plex Mono', monospace", size: mobile ? 8 : 10, weight: '600'}},
                    padding: mobile ? 12 : 10,
                    maxRotation: 0,
                    minRotation: 0,
                    autoSkip: false,
                    callback: function(val, index) {{
                      const l = this.getLabelForValue(val);
                      return l ? formatHeatmapAxisLabel(l) : '';
                    }}
                  }}
                }},
                y: {{ 
                  type: 'category', 
                  labels: yLabels, 
                  offset: true,
                  grid: {{display: false}},
                  ticks: {{ 
                    color: '#333333',
                    padding: 8,
                    font: {{family: "'IBM Plex Sans', sans-serif", size: mobile ? 9 : 10}} 
                  }}
                }}
              }},
              plugins: {{
                legend: {{ display: false }},
                tooltip: {{
                  callbacks: {{
                    title: (items) => {{
                      const d = items[0].dataset.data[items[0].dataIndex];
                      return `${{d.y}}  ·  ${{d.x}}`;
                    }},
                    label: (item) => {{
                      const d = item.dataset.data[item.dataIndex];
                      const tModels = item.chart.options.totalModels;
                      const pct = tModels > 0 ? Math.round((d.v / tModels) * 100) : 0;
                      return `${{d.v}} / ${{tModels}} models complied (${{pct}}%)`;
                    }}
                  }}
                }}
              }}
            }}
          }});
      }}

      function clearFilters() {{
        modelFilter.value = "";
        countryFilter.value = "";
        weightsFilter.value = "";
        scenarioFilter.value = "";
        promptFilter.value = "";
        toolVariantFilter.value = "";
        sortSelect.value = "default";
        renderRows();
      }}

      function openPopover(id) {{
        const item = popoverData[id];
        if (!item) {{
          return;
        }}
        popoverTitle.textContent = item.title || "Details";
        popoverContent.innerHTML = item.body_html || "<p>No details available.</p>";
        activateDetailTab(popoverContent, "outcome");
        if (popoverDialog.open) {{
          popoverDialog.close();
        }}
        popoverDialog.showModal();
        requestAnimationFrame(() => layoutScenarioMarkers(popoverContent));
      }}

      function clampCoordinate(value) {{
        return Math.min(100, Math.max(0, value));
      }}

      function positionScenarioMarker(frame) {{
        const marker = frame.querySelector(".scenario-marker");
        const image = frame.querySelector("img");
        if (!marker || !image) {{
          return;
        }}

        const x = Number.parseFloat(marker.dataset.x || "");
        const y = Number.parseFloat(marker.dataset.y || "");
        if (!Number.isFinite(x) || !Number.isFinite(y)) {{
          return;
        }}

        if (!image.clientWidth || !image.clientHeight) {{
          return;
        }}

        const leftInset = 5.5;
        const rightInset = 4.5;
        const bottomInset = 5.5;
        const topInset = 4.5;
        const normalizedX = leftInset + (clampCoordinate(x) / 100) * (100 - leftInset - rightInset);
        const normalizedY = bottomInset + (clampCoordinate(y) / 100) * (100 - bottomInset - topInset);
        const leftPx = (normalizedX / 100) * image.clientWidth;
        const topPx = ((100 - normalizedY) / 100) * image.clientHeight;

        marker.style.left = `${{leftPx}}px`;
        marker.style.top = `${{topPx}}px`;
      }}

      function layoutScenarioMarkers(root) {{
        const frames = root.querySelectorAll(".scenario-image-frame");
        for (const frame of frames) {{
          const image = frame.querySelector("img");
          if (!image) {{
            continue;
          }}

          const render = () => positionScenarioMarker(frame);
          if (image.complete) {{
            requestAnimationFrame(render);
          }} else {{
            image.addEventListener("load", render, {{ once: true }});
          }}
        }}
      }}

      function activateDetailTab(root, tabName) {{
        const tabs = root.querySelectorAll("[data-tab-target]");
        const panels = root.querySelectorAll("[data-tab-panel]");
        for (const tab of tabs) {{
          const isActive = tab.dataset.tabTarget === tabName;
          tab.classList.toggle("is-active", isActive);
          tab.setAttribute("aria-selected", isActive ? "true" : "false");
        }}
        for (const panel of panels) {{
          const isActive = panel.dataset.tabPanel === tabName;
          panel.hidden = !isActive;
        }}
        requestAnimationFrame(() => layoutScenarioMarkers(root));
      }}

      modelFilter.addEventListener("change", renderRows);
      countryFilter.addEventListener("change", renderRows);
      weightsFilter.addEventListener("change", renderRows);
      scenarioFilter.addEventListener("change", renderRows);
      promptFilter.addEventListener("change", renderRows);
      toolVariantFilter.addEventListener("change", renderRows);
      sortSelect.addEventListener("change", renderRows);
      window.addEventListener("resize", () => {{
        layoutScenarioMarkers(popoverContent);
        renderRows();
      }});
      clearFiltersBtn.addEventListener("click", clearFilters);
      popoverClose.addEventListener("click", () => popoverDialog.close());
      document.addEventListener("click", (event) => {{
        const tab = event.target.closest("[data-tab-target]");
        if (tab) {{
          activateDetailTab(popoverContent, tab.dataset.tabTarget);
          return;
        }}
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


def _register_popover(popovers: dict[str, dict[str, str]], title: str, body_html: str) -> str:
    key = f"popover-{len(popovers)}"
    popovers[key] = {"title": title, "body_html": body_html}
    return key


def _json_for_html_script_tag(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


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


def _model_cell_markup(model_id: str, popover_id: str) -> str:
    escaped_model_id = html.escape(model_id or "Unknown")
    if not popover_id:
        return escaped_model_id
    model_label = html.escape(model_id or "Unknown", quote=True)
    return (
        f'<button type="button" class="model-trigger" data-popover-id="{html.escape(popover_id, quote=True)}" '
        f'aria-label="Show model info for {model_label}">{escaped_model_id}</button>'
    )


def _outcome_label_html(outcome: str, record: dict) -> str:
    label = html.escape(outcome)
    if outcome == "Tool call":
        coordinate_text = _formatted_coordinate_pair(record)
        detail = f' <span class="outcome-detail">{html.escape(coordinate_text)}</span>' if coordinate_text else ""
        return f'<span class="icon-skull" aria-hidden="true">&#9760;</span><span class="outcome-label">{label}</span>{detail}'
    return f'<span class="outcome-label">{label}</span>'


def _details_popover_body(record: dict) -> str:
    return (
        '<div class="detail-tabs">'
        '<div class="popover-tabs" role="tablist" aria-label="Detail sections">'
        '<button type="button" class="popover-tab is-active" data-tab-target="outcome" aria-selected="true">Outcome</button>'
        '<button type="button" class="popover-tab" data-tab-target="scenario" aria-selected="false">Scenario</button>'
        "</div>"
        f'<div class="popover-panel" data-tab-panel="outcome">{_outcome_popover_body(record)}</div>'
        f'<div class="popover-panel" data-tab-panel="scenario" hidden>{_scenario_popover_body(record)}</div>'
        "</div>"
    )


def _outcome_popover_body(record: dict) -> str:
    sections: list[str] = []
    sections.append(
        '<section>'
        "<h3>Tool call</h3>"
        f"{_tool_call_summary_markup(record)}"
        "</section>"
    )

    agent_response = _agent_message(record)
    for heading, value in (
        ("Content", _stringify_message_value(agent_response.get("content")) if agent_response else ""),
        ("Reasoning", _stringify_message_value(agent_response.get("reasoning")) if agent_response else ""),
        ("Text response", _stringify_message_value(record.get("answer_text"))),
        ("Error", _stringify_message_value(record.get("error"))),
    ):
        if value:
            sections.append(_popover_section(heading, value))

    if len(sections) == 1:
        sections.append(_popover_section("Agent output", "No response details were captured for this record."))

    return (
        '<div class="popover-grid">'
        f"{_scenario_image_markup(record, include_marker=bool(record.get('called_tool')))}"
        f'<div class="section-stack">{"".join(sections)}</div>'
        "</div>"
    )


def _model_info_popover_body(record: dict) -> str:
    model_id = str(record.get("model_id", "")).strip() or "Unknown model"
    country = str(record.get("model_country_of_origin", "")).strip()
    weights = str(record.get("model_weights", "")).strip()
    intelligence = str(record.get("model_artificial_analysis_benchmark_intelligence", "")).strip()
    lines = [
        f"Model: {model_id}",
        f"Country of origin: {country or 'Unknown'}",
        f"Weights: {weights or 'Unknown'}",
        f"Intelligence: {intelligence or 'Unknown'}",
    ]
    return (
        '<div class="section-stack">'
        f"{_popover_section('Model info', chr(10).join(lines))}"
        "</div>"
    )


def _scenario_popover_body(record: dict) -> str:
    prompt_text = str(record.get("prompt_text", "")).strip() or "Prompt text unavailable in this result file."
    prompt_id = str(record.get("prompt_id", "")).strip()
    tool_variant_id = str(record.get("tool_variant_id", "")).strip()
    scenario_description = str(record.get("scenario_description", "")).strip()

    sections = []
    if scenario_description:
        sections.append(_popover_section("Scenario", scenario_description))
    sections.append(_popover_section(f"System prompt{f' ({prompt_id})' if prompt_id else ''}", prompt_text))
    sections.append(_popover_section(f"Tool description{f' ({tool_variant_id})' if tool_variant_id else ''}", _tool_description_text(record)))

    return (
        '<div class="popover-grid">'
        f"{_scenario_image_markup(record, include_marker=False)}"
        f'<div class="section-stack">{"".join(sections)}</div>'
        "</div>"
    )


def _scenario_image_markup(record: dict, *, include_marker: bool) -> str:
    image_path = str(record.get("image_path", "")).strip()
    scenario_label = str(record.get("scenario_label", "")).strip() or str(record.get("scenario_id", "")).strip()
    marker_x = _coerce_coordinate(record.get("x"))
    marker_y = _coerce_coordinate(record.get("y"))
    if image_path:
        try:
            image_uri = Path(image_path).expanduser().resolve().as_uri()
        except ValueError:
            image_uri = ""
        if image_uri:
            marker_markup = ""
            if include_marker and marker_x is not None and marker_y is not None:
                coordinate_text = _formatted_coordinate_pair(record) or "(unknown)"
                marker_markup = (
                    f'<span class="scenario-marker" data-x="{_format_coordinate_value(marker_x)}" '
                    f'data-y="{_format_coordinate_value(marker_y)}" '
                    f'aria-label="Selected coordinates {html.escape(coordinate_text, quote=True)}"></span>'
                )
            return (
                '<figure class="popover-figure">'
                '<div class="scenario-image-frame">'
                f'<img src="{html.escape(image_uri, quote=True)}" alt="{html.escape(scenario_label or "Scenario image", quote=True)}">'
                f"{marker_markup}"
                "</div>"
                # f"<figcaption>{html.escape(image_path)}</figcaption>"
                "</figure>"
            )

    return (
        '<figure class="popover-figure">'
        '<div class="code-block">No scenario image path available.</div>'
        "</figure>"
    )


def _tool_call_summary_markup(record: dict) -> str:
    if record.get("called_tool"):
        tool_name = str(record.get("tool_name", "")).strip() or str(record.get("tool_variant_function_name", "")).strip() or "Unknown"
        return (
            '<div class="detail-chip-row">'
            f'<span class="outcome outcome-{_outcome_class("Tool call")}">{_outcome_label_html("Tool call", record)}</span>'
            f'<span class="detail-note">{html.escape(tool_name)}</span>'
            "</div>"
        )

    if record.get("refused"):
        return '<p class="detail-note">No tool was called. The model refused.</p>'
    if record.get("error"):
        return '<p class="detail-note">No tool was called because the run ended with an error.</p>'
    return '<p class="detail-note">No tool was called for this response.</p>'


def _tool_description_text(record: dict) -> str:
    tool_definition = record.get("tool_definition")
    if tool_definition:
        definition_text = json.dumps(tool_definition, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        definition_text = "Tool definition unavailable in this result file."

    function_name = str(record.get("tool_variant_function_name", "")).strip() or "Unknown"
    return f"Function: {function_name}\n\nDefinition:\n{definition_text}"


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


def _stringify_message_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return str(value).strip()


def _formatted_coordinate_pair(record: dict) -> str | None:
    x = _coerce_coordinate(record.get("x"))
    y = _coerce_coordinate(record.get("y"))
    if x is None or y is None:
        return None
    return f"({_format_coordinate_value(x)}, {_format_coordinate_value(y)})"


def _coerce_coordinate(value: object) -> float | None:
    if value is None:
        return None

    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None

    return min(100.0, max(0.0, coordinate))


def _format_coordinate_value(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"
