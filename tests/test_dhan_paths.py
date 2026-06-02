from pathlib import Path

from tradingagents.dataflows.dhan_paths import (
    default_dhan_data_dir,
    repo_dhan_raw_dir,
    sync_dhan_raw_data,
)


def test_default_dhan_data_dir_falls_back_to_repo_mirror(monkeypatch):
    monkeypatch.delenv("DHAN_DATA_DIR", raising=False)

    assert default_dhan_data_dir() == str(repo_dhan_raw_dir())


def test_default_dhan_data_dir_uses_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DHAN_DATA_DIR", str(tmp_path / "original" / "dhan"))

    assert default_dhan_data_dir() == str(tmp_path / "original" / "dhan")


def test_sync_dhan_raw_data_copies_csv_files(tmp_path):
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "repo" / "data" / "dhan" / "raw"
    source_dir.mkdir(parents=True)

    source_file = source_dir / "RELIANCE.csv"
    source_file.write_text(
        "date,open,high,low,close,volume\n2026-05-29,1,2,3,4,5\n",
        encoding="utf-8",
    )

    copied_files = sync_dhan_raw_data(source_dir, destination_dir)

    assert copied_files == [destination_dir / "RELIANCE.csv"]
    assert (destination_dir / "RELIANCE.csv").read_text(encoding="utf-8") == source_file.read_text(encoding="utf-8")
