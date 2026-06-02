"""
Dhan local data vendor.

Reads pre-downloaded OHLCV CSV files from a local directory instead of
fetching from yfinance or Alpha Vantage.  The CSVs are expected to have
the format produced by the Dhan data pipeline:

    date,open,high,low,close,volume
    2024-01-02,100.5,105.0,99.0,103.2,1234567
    ...

The directory is configured via:
    config["dhan_data_dir"] = "/path/to/raw"
or the environment variable DHAN_DATA_DIR.

When dhan_data_dir is configured it is the SOLE source of truth.
yfinance is never contacted — not as a fallback, not for missing tickers,
not for dates beyond what is available in the files.

Ticker resolution
-----------------
Files are stored as <SYMBOL>.csv  (e.g. RELIANCE.csv).
If the caller passes RELIANCE.NS  (yfinance-style NSE suffix) or
RELIANCE.BO  (BSE suffix) the suffix is stripped automatically so
the right file is found.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated

import pandas as pd

from .config import get_config


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DhanDataError(RuntimeError):
    """Raised when required data is not available in the Dhan raw directory."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dhan_dir() -> str:
    """Return the configured Dhan data directory."""
    config = get_config()
    dhan_dir = config.get("dhan_data_dir") or os.environ.get("DHAN_DATA_DIR", "")
    if not dhan_dir:
        raise FileNotFoundError(
            "Dhan data directory not configured. "
            "Set config['dhan_data_dir'], the DHAN_DATA_DIR environment variable, "
            "or sync the raw CSVs into data/dhan/raw with scripts/sync_dhan_data.py."
        )
    return dhan_dir


def _normalize_symbol(symbol: str) -> str:
    """Strip exchange suffixes (.NS, .BO, etc.) to get the bare ticker."""
    return symbol.split(".")[0].upper()


def resolve_yf_ticker(symbol: str) -> str:
    """Return the yfinance-compatible ticker for a given symbol.

    When dhan_data_dir is configured the tickers are bare NSE symbols
    (e.g. RELIANCE).  yfinance requires the .NS suffix to find Indian
    stocks (e.g. RELIANCE.NS).  This helper appends .NS when:
      - dhan_data_dir is configured (i.e. we are in Indian-market mode), AND
      - the symbol has no exchange suffix yet.
    Symbols that already carry a suffix (RELIANCE.NS, RELIANCE.BO) are
    returned unchanged.
    """
    config = get_config()
    if config.get("dhan_data_dir") and "." not in symbol:
        return symbol.upper() + ".NS"
    return symbol


def _load_csv(symbol: str) -> pd.DataFrame:
    """Load the raw Dhan CSV for *symbol* and normalise column names."""
    bare = _normalize_symbol(symbol)
    path = os.path.join(_dhan_dir(), f"{bare}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No Dhan data file found for '{symbol}' at {path}"
        )

    df = pd.read_csv(path, parse_dates=["date"], on_bad_lines="skip")

    # Normalise column names to Title Case so the rest of the framework
    # (stockstats, _clean_dataframe, …) can work with them unchanged.
    df.rename(
        columns={
            "date":   "Date",
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API — matches the signatures expected by interface.py
# ---------------------------------------------------------------------------

def get_dhan_stock_data(
    symbol: Annotated[str, "Ticker symbol, e.g. RELIANCE or RELIANCE.NS"],
    start_date: Annotated[str, "Start date YYYY-MM-DD"],
    end_date: Annotated[str, "End date YYYY-MM-DD"],
) -> str:
    """
    Return OHLCV data for *symbol* between *start_date* and *end_date* as a
    CSV string, reading from the local Dhan data directory.

    This is the drop-in replacement for get_YFin_data_online used when
    config['data_vendors']['core_stock_apis'] == 'dhan'.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    try:
        df = _load_csv(symbol)
    except FileNotFoundError as exc:
        raise DhanDataError(str(exc)) from exc

    mask = (df["Date"] >= pd.Timestamp(start_date)) & (
        df["Date"] <= pd.Timestamp(end_date)
    )
    df = df.loc[mask]

    if df.empty:
        raise DhanDataError(
            f"No Dhan data available for '{symbol}' "
            f"between {start_date} and {end_date}. "
            f"Data in file ends at {_load_csv(symbol)['Date'].max().date()}."
        )

    # Round prices for cleaner output
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_string = df.to_csv(index=False)
    header = (
        f"# Stock data for {_normalize_symbol(symbol)} "
        f"from {start_date} to {end_date} (Dhan local data)\n"
        f"# Total records: {len(df)}\n\n"
    )
    return header + csv_string


def load_ohlcv_from_dhan(symbol: str, curr_date: str) -> pd.DataFrame | None:
    """
    Try to load OHLCV data from the local Dhan directory.

    Returns a DataFrame filtered to curr_date (no look-ahead), or None if
    the file is not found so the caller can fall back to yfinance.
    """
    try:
        df = _load_csv(symbol)
    except FileNotFoundError:
        return None

    curr_dt = pd.to_datetime(curr_date)
    df = df[df["Date"] <= curr_dt].copy()
    return df if not df.empty else None
