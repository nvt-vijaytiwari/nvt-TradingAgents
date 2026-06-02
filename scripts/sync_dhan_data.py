"""Sync the latest Dhan raw CSV download into the repo-local mirror."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tradingagents.dataflows.dhan_paths import repo_dhan_raw_dir, sync_dhan_raw_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy Dhan raw CSVs into the repo-local mirror.")
    parser.add_argument(
        "--source",
        default=os.getenv("DHAN_DATA_DIR", ""),
        help="Path to the original Dhan raw CSV directory. Defaults to DHAN_DATA_DIR.",
    )
    parser.add_argument(
        "--destination",
        default=str(repo_dhan_raw_dir()),
        help="Destination mirror directory inside the TradingAgents repo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source:
        raise SystemExit("Set --source or DHAN_DATA_DIR to the original Dhan raw CSV directory.")

    copied_files = sync_dhan_raw_data(Path(args.source), Path(args.destination))
    print(f"Copied {len(copied_files)} Dhan CSV file(s) into {Path(args.destination).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
