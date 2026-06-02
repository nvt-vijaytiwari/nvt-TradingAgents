import json
from pathlib import Path

from langchain_core.messages import AIMessage

from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_log_state_serializes_ai_message(tmp_path):
    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.ticker = "RELIANCE"
    graph.config = {"results_dir": str(tmp_path)}
    graph.log_states_dict = {}

    final_state = {
        "company_of_interest": "RELIANCE",
        "trade_date": "2026-05-29",
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_debate_state": {
            "bull_history": "bull",
            "bear_history": "bear",
            "history": [],
            "current_response": AIMessage(content="hello"),
            "judge_decision": "hold",
        },
        "trader_investment_plan": "trade",
        "risk_debate_state": {
            "aggressive_history": "aggressive",
            "conservative_history": "conservative",
            "neutral_history": "neutral",
            "history": [],
            "judge_decision": "underweight",
        },
        "investment_plan": "plan",
        "final_trade_decision": "decision",
    }

    graph._log_state("2026-05-29", final_state)

    log_path = (
        Path(tmp_path)
        / "RELIANCE"
        / "TradingAgentsStrategy_logs"
        / "full_states_log_2026-05-29.json"
    )
    with log_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert data["investment_debate_state"]["current_response"]["content"] == "hello"
