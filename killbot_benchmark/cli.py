from __future__ import annotations

import argparse
from pathlib import Path

from killbot_benchmark.config import load_config
from killbot_benchmark.runner import regenerate_reports, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter model tool-use decisions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full benchmark matrix.")
    run_parser.add_argument("--config", required=True, help="Path to the benchmark TOML config.")

    report_parser = subparsers.add_parser("report", help="Regenerate summary files from JSONL.")
    report_parser.add_argument("--input", required=True, help="Path to a results.jsonl file.")
    report_parser.add_argument("--output-dir", help="Optional output directory for regenerated reports.")

    args = parser.parse_args()

    try:
        if args.command == "run":
            config = load_config(args.config)
            outputs = run_benchmark(config)
        else:
            output_dir = Path(args.output_dir).resolve() if args.output_dir else None
            outputs = regenerate_reports(Path(args.input).resolve(), output_dir)
    except Exception as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    for name, path in outputs.items():
        print(f"{name}: {path}")
