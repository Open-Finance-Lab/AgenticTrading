"""Read-side presentation of backtest_attempts journal rows.

Interrupted classification lives here and nowhere else (spec D3): a 'running'
row older than its own subprocess budget plus a grace margin is *presented*
as interrupted — no writer ever stamps that state, so a dev process pointed
at the shared AGENT_RUNS_DATABASE_URL cannot clobber a genuine prod run.
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Margin past the run's own subprocess timeout before a still-'running' row is
# presented as interrupted (covers finalize lag between subprocess exit and
# the journal write, plus a restart landing mid-write).
INTERRUPTED_GRACE_SECONDS = 600
# Rows that predate timeout_seconds, or whose insert dropped it.
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 1800

INTERRUPTED_MESSAGE = "Backtest interrupted (server restarted)."

_JOURNAL_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_journal_timestamp(value: Any) -> Optional[datetime]:
    """Journal timestamps are UTC CURRENT_TIMESTAMP strings on both twins."""
    if not value:
        return None
    text = str(value).replace("T", " ")[:19]
    try:
        return datetime.strptime(text, _JOURNAL_TS_FORMAT).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def present_attempt(row: Dict[str, Any], *,
                     now: Optional[datetime] = None) -> Dict[str, Any]:
    """Return a copy of ``row`` with stale 'running' presented as interrupted."""
    presented = dict(row)
    if presented.get("status") != "running":
        return presented
    created = _parse_journal_timestamp(presented.get("created_at"))
    if created is None:
        return presented
    budget = presented.get("timeout_seconds") or DEFAULT_ATTEMPT_TIMEOUT_SECONDS
    current = now or datetime.now(timezone.utc)
    if (current - created).total_seconds() > budget + INTERRUPTED_GRACE_SECONDS:
        presented["status"] = "interrupted"
        presented["error"] = INTERRUPTED_MESSAGE
    return presented


def summarize_attempt(row: Dict[str, Any], *,
                       now: Optional[datetime] = None) -> Dict[str, Any]:
    """The agent-card payload: latest_backtest_attempt's five keys."""
    presented = present_attempt(row, now=now)
    return {
        key: presented.get(key)
        for key in ("run_id", "status", "error", "created_at", "finished_at")
    }


def attempt_as_run_entry(presented: Dict[str, Any]) -> Dict[str, Any]:
    """A journal row shaped like an agent_runs entry for history-list merges.

    ``presented`` must already be the output of ``present_attempt`` — this
    function does no classification of its own.

    initial_equity is 0.0 on purpose: the frontend's metrics guard
    (``!metrics.initial_equity``) then routes failed entries to the no-metrics
    path without a second status check.
    """
    return {
        "run_id": presented.get("run_id"),
        "agent_name": presented.get("agent_name") or "Agent",
        "mode": "backtest",
        "start_date": presented.get("start_date") or "",
        "end_date": presented.get("end_date") or "",
        "initial_equity": 0.0,
        "num_trades": 0,
        "created_at": presented.get("created_at") or "",
        "status": presented.get("status"),
        "error": presented.get("error"),
    }
