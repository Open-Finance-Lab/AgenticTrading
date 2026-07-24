"""Offline ExecutionBackend for deterministic v2 tests (no Alpaca, no network)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dashboard.backend.api.v2.models import SCHEMA_VERSION, UNIVERSE
from dashboard.backend.execution.base import ExecutionBackend, TERMINAL_STATUSES


class FakeBackend(ExecutionBackend):
    loop = "lockstep"

    def __init__(self, run_id: str = "run_fake", total_steps: int = 2,
                 session_id: str = "sess_fake"):
        self.run_id = run_id
        self.session_id = session_id
        self.total_steps = total_steps
        self.step_index = 0
        self._status = "waiting_decision"
        self._executed_log: List[Dict[str, Any]] = []
        self._decisions: List[Dict[str, Any]] = []

    def current_step_index(self) -> int:
        return self.step_index

    def build_context(self) -> Dict[str, Any]:
        if self._status in ("completed", "closed"):
            return {
                "schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                "mode": "backtest", "loop": self.loop, "status": self._status,
                "step_index": self.step_index, "total_steps": self.total_steps,
                "universe": list(UNIVERSE), "news_sentiment": {}, "news_overview": None,
            }
        return {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id, "mode": "backtest",
            "loop": self.loop, "status": "waiting_decision",
            "step_index": self.step_index, "total_steps": self.total_steps,
            "timestamp": "2026-04-15T13:30:00+00:00",
            "decision_deadline_at": "2026-04-15T13:31:00+00:00",
            "decision_timeout_seconds": 60, "universe": list(UNIVERSE),
            "portfolio": {"cash": 100000.0, "positions_value": 0.0,
                          "total_equity": 100000.0, "num_positions": 0},
            "current_holdings": {}, "recent_trades": [], "top_signals": {},
            "news_sentiment": {}, "news_overview": None,
            "decision_format": {"actions": []},
        }

    def apply_decisions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        executed = [{"action": a.get("action"), "symbol": a.get("symbol"),
                     "shares": a.get("position_size", 0), "price": 100.0}
                    for a in actions if a.get("action") in ("buy", "sell")]
        self._executed_log.extend(executed)
        self._decisions.append({
            "step_index": self.step_index,
            "decision_source": "external_agent",
            "actions_executed": len(executed),
        })
        self.step_index += 1
        if self.step_index >= self.total_steps:
            self._status = "completed"
        return {
            "accepted": True, "executed": executed, "rejected": [],
            "decision_source": "external_agent", "next_step": self.step_index,
            "status": self._status, "run_id": self.run_id,
            "metrics": self._metrics() if self._status == "completed" else None,
        }

    def status(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "status": self._status,
                "step_index": self.step_index, "total_steps": self.total_steps,
                "mode": "backtest", "loop": self.loop}

    def _metrics(self) -> Dict[str, Any]:
        return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
                "num_trades": len(self._executed_log), "final_equity": 100000.0,
                "llm_calls": self.total_steps, "input_tokens": 0,
                "output_tokens": 0, "est_cost_usd": 0.0}

    def result(self) -> Optional[Dict[str, Any]]:
        if self._status != "completed":
            return None
        return {
            "run": {"run_id": self.run_id, "agent_name": "fake", "mode": "backtest"},
            "equity_curve": [{"timestamp": "2026-04-15T13:30:00+00:00", "equity": 100000.0,
                              "cash": 100000.0, "positions_value": 0.0}],
            "trades": [], "decisions": [], "metrics": self._metrics(),
            "manifest": {"agent_name": "fake", "model_name": "m", "mode": "backtest",
                         "universe": "djia_30", "start_date": "2026-04-15",
                         "end_date": "2026-04-16", "decision_timeout_seconds": 60,
                         "schema_version": SCHEMA_VERSION, "news_sentiment_source": None},
        }

    def decisions(self) -> List[Dict[str, Any]]:
        return self._decisions

    def cancel(self) -> None:
        self._status = "closed"

    def is_active(self) -> bool:
        # Honest liveness (the ExecutionBackend default is always-True): the
        # reaper sweep and cap reconcile rely on terminal fakes reporting done.
        return self._status not in TERMINAL_STATUSES
