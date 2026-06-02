from tradingagents.batch_analysis import (
    BatchReportWriter,
    discover_dhan_tickers,
    run_batch_analysis,
)


def test_discover_dhan_tickers_returns_sorted_uppercase_csv_stems(tmp_path):
    (tmp_path / "SBIN.csv").write_text(
        "date,open,high,low,close,volume\n2026-05-29,1,1,1,1,1\n",
        encoding="utf-8",
    )
    (tmp_path / "readme.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "reliance.csv").write_text(
        "date,open,high,low,close,volume\n2026-05-29,1,1,1,1,1\n",
        encoding="utf-8",
    )

    assert discover_dhan_tickers(tmp_path) == ["RELIANCE", "SBIN"]


def test_run_batch_analysis_calls_runner_for_each_ticker(tmp_path):
    (tmp_path / "SBIN.csv").write_text(
        "date,open,high,low,close,volume\n2026-05-29,1,1,1,1,1\n",
        encoding="utf-8",
    )
    (tmp_path / "reliance.csv").write_text(
        "date,open,high,low,close,volume\n2026-05-29,1,1,1,1,1\n",
        encoding="utf-8",
    )

    calls = []

    def fake_runner(ticker, analysis_date):
        calls.append((ticker, analysis_date))
        return {"ticker": ticker}, f"decision-{ticker}"

    results = run_batch_analysis(
        data_dir=tmp_path,
        analysis_date="2026-05-29",
        runner=fake_runner,
    )

    assert calls == [
        ("RELIANCE", "2026-05-29"),
        ("SBIN", "2026-05-29"),
    ]
    assert [result.ticker for result in results] == ["RELIANCE", "SBIN"]
    assert [result.decision for result in results] == [
        "decision-RELIANCE",
        "decision-SBIN",
    ]


def test_batch_report_writer_updates_csv_and_markdown(tmp_path):
    csv_path = tmp_path / "batch" / "progress.csv"
    md_path = tmp_path / "batch" / "progress.md"
    writer = BatchReportWriter(csv_path=csv_path, markdown_path=md_path)

    writer.record(
        1,
        2,
        {
            "ticker": "RELIANCE",
            "analysis_date": "2026-05-29",
            "decision": "Underweight",
            "error": None,
        },
    )
    writer.record(
        2,
        2,
        {
            "ticker": "SBIN",
            "analysis_date": "2026-05-29",
            "decision": None,
            "error": "503 Server Error",
        },
    )

    assert csv_path.exists()
    assert md_path.exists()
    assert "RELIANCE" in csv_path.read_text(encoding="utf-8")
    assert "Underweight" in md_path.read_text(encoding="utf-8")
    assert "503 Server Error" in md_path.read_text(encoding="utf-8")


def test_run_batch_analysis_updates_report_writer(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "SBIN.csv").write_text(
        "date,open,high,low,close,volume\n2026-05-29,1,1,1,1,1\n",
        encoding="utf-8",
    )

    report_dir = tmp_path / "reports"
    writer = BatchReportWriter(
        csv_path=report_dir / "progress.csv",
        markdown_path=report_dir / "progress.md",
    )

    def fake_runner(ticker, analysis_date):
        return {"ticker": ticker, "analysis_date": analysis_date}, f"decision-{ticker}"

    results = run_batch_analysis(
        data_dir=data_dir,
        analysis_date="2026-05-29",
        runner=fake_runner,
        on_result=writer.record,
    )

    assert [result.ticker for result in results] == ["SBIN"]
    assert "decision-SBIN" in writer.csv_path.read_text(encoding="utf-8")
    assert "decision-SBIN" in writer.markdown_path.read_text(encoding="utf-8")
