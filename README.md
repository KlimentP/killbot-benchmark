# killbot-benchmark

A config-driven CLI benchmark for testing whether OpenRouter-hosted multimodal models call a tool or refuse when shown an image grid under different system prompts and tool definitions.

## What it does

- runs the selected `model x prompt x tool x scenario image` matrix
- keeps a full catalog of models, prompts, tools, and scenarios in config so reports can still enrich older or incremental results
- sends a single-turn prompt with one attached image
- exposes exactly one available tool per run, selected from a configured list of tool definitions
- records tool calls, parsed coordinates, refusal text, answer text, and the raw API payload
- writes JSONL, CSV, and HTML summaries

## Quick start

1. Set `OPENROUTER_API_KEY`.
2. Update or replace the sample fixtures under `fixtures/`.
3. Run:

```bash
python3 -m killbot_benchmark run --config fixtures/benchmark.jsonc
```

By default, runs use append mode and add the selected cases to `runs/latest/results.jsonl`, then regenerate the summaries from the combined file.

Set `"run": { "mode": "overwrite" }` in the config when you want the old behavior: move the current `runs/latest/` into `runs/archive/<timestamp>/` and rebuild `latest` from scratch.

Append mode also defaults to `"skip_existing": true`, so previously recorded `model + prompt + tool + scenario` combinations are skipped instead of duplicated.

4. Preview the full matrix without making API requests. The plan is printed to the console and no files are written:

```bash
python3 -m killbot_benchmark run --config fixtures/benchmark.jsonc --dry-run
```

5. Regenerate summary and HTML outputs from a previous run:

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

- `fixtures/benchmark.jsonc`: benchmark config. `catalog` holds every available model/prompt/tool/scenario, while `selection` holds the ids that will be run right now. JSONC lets you leave comments while editing.
- `fixtures/prompts/*.txt`: system prompts and the shared user prompt
- `fixtures/images/*`: scenario images
- `fixtures/images/raw/*`: source images used to regenerate gridded scenario images

## Result files

- `results.jsonl`: source-of-truth, one record per run
- `summary.csv`: flattened summary for analysis
- `report.html`: interactive table with hoverable outcome details, model metadata filters, and intelligence sorting
