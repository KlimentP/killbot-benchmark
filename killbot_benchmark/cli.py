from __future__ import annotations

import argparse
import logging
from pathlib import Path

from killbot_benchmark.config import load_config
from killbot_benchmark.env import load_dotenv
from killbot_benchmark.runner import regenerate_reports, render_dry_run_plan, run_benchmark


def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Benchmark OpenRouter model tool-use decisions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full benchmark matrix.")
    run_parser.add_argument(
        "--config", required=True, help="Path to the benchmark JSON or JSONC config."
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the benchmark plan and output paths without making API requests.",
    )

    report_parser = subparsers.add_parser(
        "report", help="Regenerate summary.csv and report.html from JSONL."
    )
    report_parser.add_argument("--input", required=True, help="Path to a results.jsonl file.")
    report_parser.add_argument("--output-dir", help="Optional output directory for regenerated reports.")
    report_parser.add_argument(
        "--html-only",
        action="store_true",
        help="Generate only report.html from an existing results.jsonl file.",
    )

    args = parser.parse_args()

    try:
        if args.command == "run":
            config = load_config(args.config)
            if args.dry_run:
                print(render_dry_run_plan(config), end="")
                return
            outputs = run_benchmark(config)
        else:
            output_dir = Path(args.output_dir).resolve() if args.output_dir else None
            outputs = regenerate_reports(
                Path(args.input).resolve(),
                output_dir,
                html_only=args.html_only,
            )
    except Exception as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    for name, path in outputs.items():
        print(f"{name}: {path}")
