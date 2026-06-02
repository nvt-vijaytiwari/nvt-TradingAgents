"""
Run TradingAgents analysis for every Dhan CSV in the configured data directory.

Defaults to the Codex OAuth provider so the batch run does not need API keys.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from tradingagents.dataflows.dhan_paths import default_dhan_data_dir

from tradingagents.batch_analysis import (
    BatchReportWriter,
    DEFAULT_DEEP_MODEL,
    DEFAULT_QUICK_MODEL,
    discover_dhan_tickers,
    run_batch_analysis,
    save_batch_results,
)


DEFAULT_ANALYSIS_DATE = "2026-05-29"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "batch_reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TradingAgents for all Dhan stocks.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to the Dhan raw CSV directory. Defaults to DHAN_DATA_DIR or the repo default.",
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_ANALYSIS_DATE,
        help="Analysis date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of tickers to analyze.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Accepted for compatibility, but Codex batch runs currently ignore checkpointing.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Directory for the committed batch artifacts. Defaults to ./batch_reports.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    data_dir = args.data_dir or default_dhan_data_dir()

    tickers = discover_dhan_tickers(data_dir)
    if args.limit is not None:
        tickers = tickers[: args.limit]

    print(f"Using Codex OAuth with {DEFAULT_DEEP_MODEL}/{DEFAULT_QUICK_MODEL}")
    print(f"Loading {len(tickers)} ticker(s) from {Path(data_dir)}")
    print(f"Analysis date: {args.date}")
    if args.checkpoint:
        print("Checkpointing is currently disabled for Codex batch runs; proceeding without it.")

    report_dir = Path(args.report_dir) if args.report_dir else DEFAULT_REPORT_DIR
    csv_path = report_dir / f"all_stocks_codex_{args.date}.csv"
    md_path = report_dir / f"all_stocks_codex_{args.date}.md"
    json_path = report_dir / f"all_stocks_codex_{args.date}.json"
    writer = BatchReportWriter(csv_path=csv_path, markdown_path=md_path)

    print(f"Writing progress table to {csv_path}")
    print(f"Writing markdown report to {md_path}")

    def on_result(index, total, result):
        writer.record(index, total, result)
        if result.error:
            print(f"[{index}/{total}] {result.ticker}: ERROR: {result.error}")
        else:
            print(f"[{index}/{total}] {result.ticker}: completed")

    results = run_batch_analysis(
        data_dir=data_dir,
        analysis_date=args.date,
        tickers=tickers,
        checkpoint_enabled=False,
        on_result=on_result,
    )

    save_batch_results(results, json_path)
    print(f"Saved batch summary to {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
