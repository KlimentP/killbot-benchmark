# killbot-benchmark

A config-driven CLI benchmark for testing whether OpenRouter-hosted multimodal models call a `drone_strike(x, y)` tool or refuse when shown an image grid under different system prompts.

## What it does

- runs the full `model x prompt x scenario image` matrix
- sends a single-turn prompt with one attached image
- exposes one available tool, `drone_strike`
- records tool calls, parsed coordinates, refusal text, answer text, and the raw API payload
- writes JSONL, CSV, and Markdown summaries

## Quick start

1. Set `OPENROUTER_API_KEY`.
2. Update or replace the sample fixtures under `fixtures/`.
3. Run:

```bash
python3 -m killbot_benchmark run --config fixtures/benchmark.toml
```

4. Regenerate reports from a previous run:

```bash
python3 -m killbot_benchmark report --input runs/latest/results.jsonl
```

## Fixture layout

- `fixtures/benchmark.toml`: benchmark config
- `fixtures/prompts/*.txt`: system prompts and the shared user prompt
- `fixtures/images/*`: scenario images

## Result files

- `results.jsonl`: source-of-truth, one record per run
- `summary.csv`: flattened summary for analysis
- `report.md`: grouped aggregate report
