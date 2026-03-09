# killbot-benchmark

A config-driven CLI benchmark for testing whether OpenRouter-hosted multimodal models call a `drone_strike(x, y)` tool or refuse when shown an image grid under different system prompts.

## What it does

- runs the full `model x prompt x scenario image` matrix
- sends a single-turn prompt with one attached image
- exposes one available tool, `drone_strike`
- records tool calls, parsed coordinates, refusal text, answer text, and the raw API payload
- writes JSONL, CSV, Markdown, and HTML summaries

## Quick start

1. Set `OPENROUTER_API_KEY`.
2. Update or replace the sample fixtures under `fixtures/`.
3. Run:

```bash
python3 -m killbot_benchmark run --config fixtures/benchmark.toml
```

Before each new run, any existing contents in `runs/latest/` are moved into `runs/archive/<timestamp>/`.

4. Preview the full matrix without making API requests:

```bash
python3 -m killbot_benchmark run --config fixtures/benchmark.toml --dry-run
```

5. Regenerate reports from a previous run:

```bash
python3 -m killbot_benchmark report --input runs/latest/results.jsonl
```

Generate only the HTML report from existing results:

```bash
python3 -m killbot_benchmark report --input runs/latest/results.jsonl --html-only
```

6. Regenerate grid-labeled fixture images from raw source images:

```bash
python3 scripts/add_image_grids.py
```

## Fixture layout

- `fixtures/benchmark.toml`: benchmark config
- `fixtures/prompts/*.txt`: system prompts and the shared user prompt
- `fixtures/images/*`: scenario images
- `fixtures/images/raw/*`: source images used to regenerate gridded scenario images

## Result files

- `results.jsonl`: source-of-truth, one record per run
- `summary.csv`: flattened summary for analysis
- `report.md`: grouped aggregate report
- `report.html`: interactive table with hoverable outcome details
- `dry_run_plan.md`: preview of the planned benchmark cases and output locations
