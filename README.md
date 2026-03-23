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

Append mode also defaults to `"skip_existing": true`, so previously recorded `model + prompt + tool + scenario` combinations are skipped instead of duplicated.

Set `"run": { "mode": "append_overwrite_existing" }` when you want to keep `runs/latest` but replace any existing records that match the selected `model + prompt + tool + scenario` combinations before appending fresh results.

Set `"run": { "mode": "overwrite" }` when you want to archive the current `runs/latest/` into `runs/archive/<timestamp>/` and rebuild `latest` from scratch.

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

## Adding new fixtures

Add new benchmark items in two places:

1. Add the full definition to the matching `catalog` array in `fixtures/benchmark.jsonc`.
2. Add its `id` to the matching `selection` array if you want it included in the next run.

### Add a new tool

Add a new object under `catalog.tools` in `fixtures/benchmark.jsonc`:

- `id`: stable benchmark id used in reports and selections
- `spec`: the exact OpenRouter tool definition that should be exposed to the model for that run
- `spec.function.name`: the callable function name the model will see
- `spec.function.description` and `spec.function.parameters`: the behavior and schema you want to test

After adding the tool definition, add the tool `id` to `selection.tools` to run it.

### Add a new scenario image

Add the image file under `fixtures/images/` and then add a new object under `catalog.scenarios` in `fixtures/benchmark.jsonc`:

- `id`: stable scenario id
- `image`: relative path such as `images/my-scenario.jpeg`
- `label`: short human-readable label for reports
- `description`: summary shown in outputs and useful for later analysis

Then add the scenario `id` to `selection.scenarios`.

If you are starting from an ungridded source image, place the source file in `fixtures/images/raw/` and regenerate the benchmark images with:

```bash
python3 scripts/add_image_grids.py
```

### Add a new model

Add a new object under `catalog.models` in `fixtures/benchmark.jsonc`:

- `id`: OpenRouter model id, for example `openai/gpt-5.2`
- `country_of_origin`
- `weights`
- `artificial_analysis_benchmark_intelligence`

Then add the model `id` to `selection.models`.

### Add a new system prompt

Create a new prompt file in `fixtures/prompts/`, then add a new object under `catalog.prompts` in `fixtures/benchmark.jsonc`:

- `id`: stable prompt id
- `file`: relative path such as `prompts/my-prompt.txt`

Then add the prompt `id` to `selection.prompts`.

The benchmark sends the selected system prompt file together with the shared user prompt configured by `run.user_prompt_file`.

## Result files

- `results.jsonl`: source-of-truth, one record per run
- `summary.csv`: flattened summary for analysis
- `report.html`: interactive table with hoverable outcome details, model metadata filters, and intelligence sorting
