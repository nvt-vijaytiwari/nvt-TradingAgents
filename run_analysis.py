"""
Run TradingAgents analysis using:
- Codex OAuth as the LLM provider
- Dhan raw CSV files as the sole price/indicator data source
- yfinance for news & fundamentals (no API key needed)
"""

from dotenv import load_dotenv
load_dotenv()

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.dhan_paths import default_dhan_data_dir

config = DEFAULT_CONFIG.copy()

# LLM — Codex CLI OAuth (uses your ChatGPT subscription, no API key needed)
config["llm_provider"]    = "codex"
config["deep_think_llm"]  = "gpt-5.4"       # strong reasoning
config["quick_think_llm"] = "gpt-5.4-mini"  # fast for analyst agents

# Dhan data as sole price source. Use the explicit env override when set,
# otherwise fall back to the repo-local mirror under data/dhan/raw.
config["dhan_data_dir"] = default_dhan_data_dir()

# Keep debate tight for first run (cheaper + faster)
config["max_debate_rounds"]      = 1
config["max_risk_discuss_rounds"] = 1

# Analysts to run
config["selected_analysts"] = ["market", "news", "fundamentals", "technical"]

ta = TradingAgentsGraph(debug=True, config=config)

TICKER   = "RELIANCE"
DATE     = "2026-05-29"   # latest available date in Dhan data

print(f"\n{'='*60}")
print(f"  Analysing {TICKER} as of {DATE}")
print(f"{'='*60}\n")

_, decision = ta.propagate(TICKER, DATE)

print(f"\n{'='*60}")
print("  FINAL DECISION")
print(f"{'='*60}")
print(decision)
