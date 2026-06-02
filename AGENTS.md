# TradingAgents — Agent & Developer Reference

> Everything an AI agent or new developer needs to understand, run, and extend this project.

---

## What This Project Does

TradingAgents is a **multi-agent LLM trading framework** that mimics a real-world trading firm.
You give it a stock ticker and a date; a pipeline of specialised AI agents collaborates to produce
a final **buy / sell / hold recommendation** with full reasoning.

### Agent Pipeline (in order)

```
Market Data (Dhan CSVs + yfinance)
        ↓
Analyst Team  (runs in parallel)
  • Market Analyst      — OHLCV trends, price action
  • News Analyst        — macro news, global events
  • Fundamentals Analyst — P/E, balance sheet, cash flow
  • Technical Analyst   — RSI, MACD, Bollinger, ATR …
        ↓
Researcher Team  (structured debate)
  • Bull Researcher  vs  Bear Researcher
  • Research Manager  →  summarises debate
        ↓
Trader Agent  →  proposes trade (size + direction)
        ↓
Risk Management Team  (3-way debate)
  • Aggressive / Neutral / Conservative Analyst
        ↓
Portfolio Manager  →  approves / rejects  →  FINAL DECISION
```

Results are persisted to `~/.tradingagents/memory/trading_memory.md` and used as
**learning memory** on subsequent runs for the same ticker.

---

## Repository Layout

```
TradingAgents/
├── run_analysis.py                         # ← Main entry point (our custom script)
├── main.py                                 # Original upstream demo script
├── cli/                                    # Interactive CLI (tradingagents run)
├── tradingagents/
│   ├── default_config.py                   # All config keys + defaults
│   ├── graph/
│   │   └── trading_graph.py               # TradingAgentsGraph — top-level orchestrator
│   ├── agents/
│   │   ├── analysts/                      # Market, News, Fundamentals, Technical, Social
│   │   ├── researchers/                   # Bull + Bear researchers
│   │   ├── managers/                      # Research Manager, Portfolio Manager
│   │   ├── risk_mgmt/                     # Aggressive / Neutral / Conservative debators
│   │   └── trader/                        # Trader agent
│   ├── dataflows/
│   │   ├── dhan_data.py                   # ★ OUR ADDITION — local Dhan CSV loader
│   │   ├── interface.py                   # Vendor routing (yfinance / alpha_vantage / dhan)
│   │   ├── stockstats_utils.py            # OHLCV cache + technical indicator engine
│   │   ├── y_finance.py                   # yfinance wrappers (news, fundamentals)
│   │   └── config.py                      # Runtime config accessor
│   └── llm_clients/
│       ├── codex_oauth_client.py          # ★ OUR ADDITION — Codex CLI OAuth provider
│       ├── factory.py                     # Provider factory (openai/groq/openrouter/codex…)
│       ├── openai_client.py               # OpenAI-compatible client + rate-limit retry
│       └── …
├── .env                                    # API keys (never commit)
└── AGENTS.md                               # ← this file
```

---

## Our Custom Changes

All changes we made on top of the upstream repo are marked ★ above and described here.

### 1. Dhan Local Data Integration

**File:** `tradingagents/dataflows/dhan_data.py`

Reads pre-downloaded OHLCV CSV files from a local directory instead of calling yfinance
for price data. The Dhan raw directory is the **sole source of truth** for price and
technical indicator data — yfinance is never contacted for OHLCV when this is configured.

**CSV format expected:**
```
date,open,high,low,close,volume
2024-01-02,100.5,105.0,99.0,103.2,1234567
```

**Ticker resolution:** `RELIANCE.NS` → strips `.NS` → reads `RELIANCE.csv`

**Error behaviour:**
- Ticker file missing → raises `DhanDataError` (no yfinance fallback)
- Date beyond available data → raises `DhanDataError` with last-available-date message

**What still uses yfinance (internet required, no API key):**
- News articles
- Fundamentals (P/E, EPS, margins…)
- Balance sheet / cash flow / income statement
- Insider transactions

**Modified files:**
- `tradingagents/dataflows/stockstats_utils.py` — `load_ohlcv()` checks Dhan first
- `tradingagents/dataflows/interface.py` — registered `"dhan"` as a vendor
- `tradingagents/default_config.py` — added `"dhan_data_dir"` key

---

### 2. New LLM Providers

#### Groq (`llm_provider = "groq"`)
Added to `openai_client.py` provider config and `factory.py`.

```
Base URL : https://api.groq.com/openai/v1
Env var  : GROQ_API_KEY
```

**Note:** Groq free tier has a 6000 TPM limit per model — too low for this app
(single analyst call uses ~15,000–62,000 tokens). Paid tier works fine.

#### Codex CLI OAuth (`llm_provider = "codex"`)

**File:** `tradingagents/llm_clients/codex_oauth_client.py`

Reads the OAuth access token stored by the Codex CLI at `~/.codex/auth.json`.
Auto-refreshes the token 5 minutes before expiry using the stored `refresh_token`.
No separate API key needed — uses ChatGPT subscription authentication.

The client talks to the ChatGPT-backed Codex Responses endpoint
(`https://chatgpt.com/backend-api/codex/responses`) and translates the
streamed events into LangChain `AIMessage` objects, including function tool
calls. Supported models are the GPT-5.4 family available to Codex accounts
(for this repo: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.2`).

#### Rate-limit retry (all OpenAI-compatible providers)

`openai_client.py` `NormalizedChatOpenAI.invoke` now retries on `429 RateLimitError`
up to 8 times, honouring the server's `retry_after_seconds` hint from the response body
(plus a small random jitter). Falls back to exponential backoff when no hint is present.

---

## Setup & Running

### Prerequisites

```bash
# Clone and enter the project
cd /Users/vijaytiwari/work/AI/TradingAgents

# Install dependencies (uv lock file is present)
uv sync
# or: pip install -e .
```

### Environment Variables

Create `.env` in the project root (already present — never commit it):

```bash
# ── LLM Provider (pick one) ──────────────────────────────────────────
OPENROUTER_API_KEY=sk-or-...       # Recommended: cheap + reliable
# OPENAI_API_KEY=sk-...           # Direct OpenAI API
# ANTHROPIC_API_KEY=sk-ant-...    # Direct Anthropic API
# GROQ_API_KEY=gsk_...            # Groq (paid tier only for this app)

# ── Data ─────────────────────────────────────────────────────────────
# Optional override. Defaults to the repo-local mirror at ./data/dhan/raw.
DHAN_DATA_DIR=/Users/vijaytiwari/work/AI/TradingAgents/data/dhan/raw
```

### Run a Single Stock Analysis

Edit `run_analysis.py` to set `TICKER` and `DATE`, then:

```bash
.venv/bin/python3 run_analysis.py
```

### Run All Stocks (Batch)

Run every Dhan CSV through the Codex-backed pipeline with one command:

```bash
.venv/bin/python3 run_all_stocks.py --date 2026-05-29
```

Useful flags:
- `--limit N` to smoke-test a smaller slice first
- `--report-dir PATH` to write the committed batch artifacts somewhere specific
- `--checkpoint` is accepted but ignored for Codex batch runs until checkpoint metadata is sanitized

Defaults:
- Scans every `*.csv` under `DHAN_DATA_DIR`, or `./data/dhan/raw` when the env var is unset
- Uses Codex OAuth with `gpt-5.4` / `gpt-5.4-mini`
- Writes committed artifacts to `./batch_reports/`:
  - `all_stocks_codex_<date>.csv`
  - `all_stocks_codex_<date>.md`
  - `all_stocks_codex_<date>.json`

To refresh the repo-local Dhan mirror after downloading a new raw export:

```bash
.venv/bin/python3 scripts/sync_dhan_data.py --source /path/to/original/dhan/raw
```

### Run via CLI (interactive)

```bash
tradingagents run
# or
.venv/bin/python3 -m cli.main run
```

### Run Tests

```bash
.venv/bin/python3 -m pytest tests/ -q
# Expected: 122 passed
```

---

## Configuration Reference

All keys live in `tradingagents/default_config.py` and can be overridden at runtime:

```python
from tradingagents.default_config import DEFAULT_CONFIG
config = DEFAULT_CONFIG.copy()
config["key"] = "value"
```

| Key | Default | Description |
|---|---|---|
| `llm_provider` | `"openai"` | `openai`, `anthropic`, `google`, `openrouter`, `groq`, `deepseek`, `azure`, `codex` |
| `deep_think_llm` | `"gpt-5.4"` | Model for Research Manager, Trader, Portfolio Manager |
| `quick_think_llm` | `"gpt-5.4-mini"` | Model for the 4 analysts + risk team |
| `dhan_data_dir` | `./data/dhan/raw` | Path to Dhan raw CSV directory. When set or mirrored locally, **sole source of truth** for OHLCV |
| `max_debate_rounds` | `1` | Bull vs Bear debate rounds |
| `max_risk_discuss_rounds` | `1` | Risk team debate rounds |
| `checkpoint_enabled` | `False` | LangGraph checkpoint resume on crash |
| `data_vendors.core_stock_apis` | `"yfinance"` | `yfinance`, `alpha_vantage`, `dhan` |
| `data_vendors.technical_indicators` | `"yfinance"` | Same options |
| `output_language` | `"English"` | Language for final reports |
| `results_dir` | `~/.tradingagents/logs` | Override with `TRADINGAGENTS_RESULTS_DIR` |
| `data_cache_dir` | `~/.tradingagents/cache` | Override with `TRADINGAGENTS_CACHE_DIR` |
| `batch_reports/` | `./batch_reports` | Repo-local batch artifacts updated after each ticker |
| `data/dhan/raw/` | `./data/dhan/raw` | Checked-in Dhan CSV mirror used when `DHAN_DATA_DIR` is unset |

---

## Data Sources

| Data Type | Source | Requires |
|---|---|---|
| OHLCV prices | **Dhan raw CSVs** (local mirror) | `DHAN_DATA_DIR` set or `./data/dhan/raw` populated |
| Technical indicators (RSI, MACD…) | Computed from Dhan CSVs | `DHAN_DATA_DIR` set or `./data/dhan/raw` populated |
| News articles | yfinance (internet) | No key |
| Fundamentals / P/E / EPS | yfinance (internet) | No key |
| Balance sheet / cash flow | yfinance (internet) | No key |
| Insider transactions | yfinance (internet) | No key |

**Dhan data location:** `./data/dhan/raw/`
- 500 NSE stocks as individual CSVs
- Data from ~2002 to **2026-05-29** (latest download)
- Format: `date,open,high,low,close,volume` (lowercase headers)
- Refresh the mirror with `.venv/bin/python3 scripts/sync_dhan_data.py --source /path/to/original/dhan/raw`

---

## Output & Storage

| Location | Contents |
|---|---|
| `~/.tradingagents/memory/trading_memory.md` | Decision log — every run appended here. Used as learning memory on next run for same ticker |
| `~/.tradingagents/logs/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json` | Full raw agent output (all messages, tool calls, intermediate results) |

---

## Recommended LLM Providers (Tested)

| Provider | Model | Cost/stock | Status |
|---|---|---|---|
| **OpenRouter** | `anthropic/claude-sonnet-4.5` | ~$0.28 | ✅ Works, expensive |
| **OpenRouter** | `google/gemini-2.0-flash-lite-001` | ~$0.03–0.05 | ✅ Recommended for OpenRouter |
| Codex OAuth | `gpt-5.4-mini` | ChatGPT sub | ✅ Works |
| Codex OAuth | `gpt-5.4` | ChatGPT sub | ✅ Works |
| OpenRouter | `deepseek/deepseek-v4-flash:free` | $0 | ❌ Rate limited (free tier congested) |
| OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | $0 | ❌ Rate limited (free tier congested) |
| Groq | `qwen/qwen3-32b` | $0 | ❌ 6000 TPM limit too low |

**Current `run_analysis.py` setting:** `codex` provider with `gpt-5.4` / `gpt-5.4-mini`
**Current `run_all_stocks.py` setting:** same Codex OAuth defaults, but for every Dhan CSV

---

## Switching LLM Provider in run_analysis.py

```python
# Codex OAuth — no API key required
config["llm_provider"]   = "codex"
config["deep_think_llm"] = "gpt-5.4"
config["quick_think_llm"] = "gpt-5.4-mini"

# OpenRouter — Recommended (add credits at openrouter.ai/credits)
config["llm_provider"]   = "openrouter"
config["deep_think_llm"] = "google/gemini-2.0-flash-lite-001"
config["quick_think_llm"] = "google/gemini-2.0-flash-lite-001"

# OpenRouter — High quality
config["llm_provider"]   = "openrouter"
config["deep_think_llm"] = "anthropic/claude-sonnet-4.5"
config["quick_think_llm"] = "anthropic/claude-3.5-haiku"
```

---

## Adding a New Stock to Analyse

1. Ensure `<TICKER>.csv` exists in `DHAN_DATA_DIR` or `./data/dhan/raw`
2. Edit `run_analysis.py`:
   ```python
   TICKER = "TCS"
   DATE   = "2026-05-29"   # must be <= last date in the CSV
   ```
3. Run: `.venv/bin/python3 run_analysis.py`

---

## Known Issues / Gotchas

| Issue | Cause | Fix |
|---|---|---|
| `DhanDataError: No Dhan data found for X` | CSV not in raw dir | Add CSV or check ticker spelling |
| `DhanDataError: No data available between …` | Date beyond CSV coverage | Use a date ≤ 2026-05-29 |
| `429 Rate limit` on free OpenRouter models | Free tier congested | Use paid model or wait and retry |
| `413 Request too large` on Groq | 6000 TPM free limit | Use paid Groq tier |
| `400 The '<model>' model is not supported when using Codex with a ChatGPT account` | Selected model is outside the Codex allowlist | Use `gpt-5.4`, `gpt-5.4-mini`, or `gpt-5.2` |
| `Object of type AIMessage is not JSON serializable` during checkpointed batch runs | LangGraph checkpoint metadata still includes raw AIMessage objects | Run `run_all_stocks.py` without `--checkpoint` for Codex batches |
| `tool call validation failed` on Groq llama-3.3-70b | Wrong tool-call format from that model | Use qwen3-32b on Groq |
