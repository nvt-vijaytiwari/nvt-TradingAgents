"""Helpers for locating and syncing the repo-local Dhan raw data mirror."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def repository_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def repo_dhan_raw_dir() -> Path:
    """Return the repo-local mirror location for raw Dhan CSV files."""
    return repository_root() / "data" / "dhan" / "raw"


def default_dhan_data_dir() -> str:
    """Return the active Dhan data directory.

    The explicit DHAN_DATA_DIR environment variable wins. Otherwise the
    repo-local mirror under data/dhan/raw is used so cloned checkouts can
    run without needing a separate absolute path.
    """
    return os.getenv("DHAN_DATA_DIR") or str(repo_dhan_raw_dir())


def sync_dhan_raw_data(source_dir: Path | str, destination_dir: Path | str | None = None) -> list[Path]:
    """Copy raw Dhan CSV files from ``source_dir`` into ``destination_dir``.

    The copy is recursive and preserves relative paths. Existing files are
    overwritten so the mirror stays in sync with the latest download.
    """
    source_path = Path(source_dir).expanduser().resolve()
    destination_path = Path(destination_dir).expanduser().resolve() if destination_dir is not None else repo_dhan_raw_dir().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Dhan source directory does not exist: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Dhan source path is not a directory: {source_path}")

    if source_path == destination_path:
        return sorted(source_path.rglob("*.csv"))

    copied_files: list[Path] = []
    for source_file in sorted(source_path.rglob("*.csv")):
        if not source_file.is_file():
            continue
        relative_path = source_file.relative_to(source_path)
        destination_file = destination_path / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        copied_files.append(destination_file)
    return copied_files
