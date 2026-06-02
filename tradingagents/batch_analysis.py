from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from tradingagents.default_config import DEFAULT_CONFIG

DEFAULT_ANALYSTS: tuple[str, ...] = ("market", "news", "fundamentals", "technical")
DEFAULT_DEEP_MODEL = "gpt-5.4"
DEFAULT_QUICK_MODEL = "gpt-5.4-mini"


@dataclass(slots=True)
class BatchResult:
    ticker: str
    analysis_date: str
    decision: str | None
    error: str | None = None

    @property
    def status(self) -> str:
        return "error" if self.error else "completed"


@dataclass(slots=True)
class BatchReportWriter:
    csv_path: Path
    markdown_path: Path
    results: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)
        self.markdown_path = Path(self.markdown_path)

    def record(self, index: int, total: int, result: BatchResult | dict[str, Any]) -> None:
        row = _build_report_row(index, total, result)
        self.results.append(row)
        self.write()

    def write(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(self.csv_path, self.results)
        _write_markdown(self.markdown_path, self.results)


def _coerce_batch_result(result: BatchResult | dict[str, Any]) -> BatchResult:
    if isinstance(result, BatchResult):
        return result

    decision = result.get("decision")
    error = result.get("error")
    return BatchResult(
        ticker=str(result["ticker"]),
        analysis_date=str(result["analysis_date"]),
        decision=None if decision in (None, "") else str(decision),
        error=None if error in (None, "") else str(error),
    )


def _build_report_row(index: int, total: int, result: BatchResult | dict[str, Any]) -> dict[str, str]:
    batch_result = _coerce_batch_result(result)
    return {
        "index": str(index),
        "total": str(total),
        "progress": f"{index}/{total}",
        "ticker": batch_result.ticker,
        "analysis_date": batch_result.analysis_date,
        "status": batch_result.status,
        "verdict": batch_result.decision or "",
        "error": batch_result.error or "",
    }


def _write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    fieldnames = ["index", "total", "progress", "ticker", "analysis_date", "status", "verdict", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _escape_markdown(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _write_markdown(path: Path, rows: Sequence[dict[str, str]]) -> None:
    analysis_date = rows[0]["analysis_date"] if rows else ""
    completed = sum(1 for row in rows if row.get("status") == "completed")
    errored = sum(1 for row in rows if row.get("status") == "error")
    total = int(rows[0]["total"]) if rows else 0

    lines = [
        "# TradingAgents Batch Report",
        "",
        f"- Analysis date: `{analysis_date}`" if analysis_date else "- Analysis date: `unknown`",
        f"- Progress: `{completed}/{total}` completed, `{errored}` errored",
        "",
        "| # | Progress | Ticker | Status | Verdict | Error |",
        "|---|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            "| {index} | {progress} | {ticker} | {status} | {verdict} | {error} |".format(
                index=_escape_markdown(row.get("index", "")),
                progress=_escape_markdown(row.get("progress", "")),
                ticker=_escape_markdown(row.get("ticker", "")),
                status=_escape_markdown(row.get("status", "")),
                verdict=_escape_markdown(row.get("verdict", "")),
                error=_escape_markdown(row.get("error", "")),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover_dhan_tickers(data_dir: str | Path) -> list[str]:
    directory = Path(data_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Dhan data directory does not exist: {directory}")

    tickers = {
        entry.stem.upper()
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() == ".csv"
    }
    return sorted(tickers)


def build_codex_batch_config(
    data_dir: str | Path,
    *,
    checkpoint_enabled: bool = False,
) -> dict:
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "codex"
    config["deep_think_llm"] = DEFAULT_DEEP_MODEL
    config["quick_think_llm"] = DEFAULT_QUICK_MODEL
    config["dhan_data_dir"] = str(Path(data_dir))
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["selected_analysts"] = list(DEFAULT_ANALYSTS)
    config["checkpoint_enabled"] = checkpoint_enabled
    return config


def run_batch_analysis(
    data_dir: str | Path,
    analysis_date: str,
    *,
    tickers: Sequence[str] | None = None,
    runner: Callable[[str, str], tuple[object, object]] | None = None,
    config: dict | None = None,
    checkpoint_enabled: bool = False,
    on_result: Callable[[int, int, BatchResult], None] | None = None,
    limit: int | None = None,
) -> list[BatchResult]:
    stock_list = list(tickers) if tickers is not None else discover_dhan_tickers(data_dir)
    if limit is not None:
        stock_list = stock_list[:limit]

    if runner is None:
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        effective_config = build_codex_batch_config(
            data_dir,
            checkpoint_enabled=checkpoint_enabled,
        )
        if config is not None:
            effective_config.update(config)
            effective_config["dhan_data_dir"] = str(Path(data_dir))
        graph = TradingAgentsGraph(debug=False, config=effective_config)
        runner = graph.propagate

    results: list[BatchResult] = []
    total = len(stock_list)

    for index, ticker in enumerate(stock_list, start=1):
        try:
            _, decision = runner(ticker, analysis_date)
            result = BatchResult(
                ticker=ticker,
                analysis_date=analysis_date,
                decision=str(decision),
            )
        except Exception as exc:  # pragma: no cover - exercised in live runs
            result = BatchResult(
                ticker=ticker,
                analysis_date=analysis_date,
                decision=None,
                error=str(exc),
            )

        results.append(result)
        if on_result is not None:
            on_result(index, total, result)

    return results


def save_batch_results(results: Sequence[BatchResult], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(result) for result in results], handle, ensure_ascii=False, indent=2)
    return path
