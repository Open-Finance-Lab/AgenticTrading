#!/usr/bin/env python3
"""Where do two backtests of one configuration first disagree?

Prints, as JSON, the first divergent bar, how many bars diverged, and the
final-equity gap between two saved runs, plus the sampling each run recorded.
Reads ``agent_runs``, ``equity_timeseries`` and ``backtest_decisions``
through the same ``db`` the dashboard uses, so it works against local SQLite
or, with ``AGENT_RUNS_DATABASE_URL`` set, against Postgres.

    python dashboard/scripts/diff_backtest_runs.py <run_id_a> <run_id_b>

**Two axes, and ``basis`` says which one answered.** The decision log is the
sharper of the two, but it is not always written: ``run_agent_backtest``
calls ``db.insert_decisions`` only on the AI Hedge Fund runtime (the guard in
``HourlyBacktester`` keyed on ``AI_HEDGE_FUND_RUNTIME_TYPE``), and the
external-agent surface (``external_run_service``) is the only other writer. A
pipeline-runtime backtest -- the ordinary dashboard run -- has no rows there,
so the decision fields come back ``None`` rather than ``0`` and the equity
curve, which every run writes, carries the number. A zero out of an empty
table is the strongest claim this script can make, and it is the one claim it
must never make by accident.

The number this exists for is *later and smaller*, not zero: providers are
not deterministic at temperature 0, and each bar's prompt embeds the previous
bar's answer, so one different draw is carried to the end of the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

from dashboard.backend.database import db  # noqa: E402


def _normalised_actions(entry: Dict[str, Any]) -> str:
    return json.dumps(entry.get("actions_submitted") or [], sort_keys=True)


def compare_decisions(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bar-by-bar comparison of two decision logs, in step order."""
    divergent = 0
    first: Optional[Dict[str, Any]] = None
    for x, y in zip(a, b):
        if _normalised_actions(x) != _normalised_actions(y):
            divergent += 1
            if first is None:
                first = {
                    "step_index": int(x.get("step_index", 0)),
                    "timestamp": x.get("timestamp"),
                }
    return {
        "steps_compared": min(len(a), len(b)),
        "steps_a": len(a),
        "steps_b": len(b),
        "divergent_steps": divergent,
        "first_divergence": first,
    }


def _normalised_point(point: Dict[str, Any]) -> str:
    return json.dumps([str(point.get("timestamp")), point.get("equity")])


def compare_equity(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bar-by-bar comparison of two equity curves, in stored order.

    The axis that always exists: ``run_agent_backtest`` writes
    ``equity_timeseries`` for every run, unconditionally.

    Exact comparison, no tolerance. Two runs whose decisions agreed ran the
    same arithmetic over the same bars, so any difference at all is real, and
    a tolerance would swallow exactly the smallest and earliest divergence
    this script exists to find.
    """
    divergent = 0
    first: Optional[Dict[str, Any]] = None
    for index, (x, y) in enumerate(zip(a, b)):
        if _normalised_point(x) != _normalised_point(y):
            divergent += 1
            if first is None:
                first = {"index": index, "timestamp": x.get("timestamp")}
    return {
        "equity_points_compared": min(len(a), len(b)),
        "equity_points_a": len(a),
        "equity_points_b": len(b),
        "divergent_equity_points": divergent,
        "first_equity_divergence": first,
    }


#: What the decision axis reports when there is no decision log to read.
#: ``None``, never ``0``: ``backtest_decisions`` is written only by the AI
#: Hedge Fund runtime and the external-agent surface, so an empty log on a
#: pipeline run means *unmeasured*. A ``0`` there is this script asserting
#: that two runs agreed on every bar, out of a table nobody wrote to -- and
#: that number is the headline of the table it gets pasted into.
_NO_DECISION_LOG = {
    "steps_compared": None,
    "divergent_steps": None,
    "first_divergence": None,
}


def compare_runs(run_a: str, run_b: str) -> Dict[str, Any]:
    row_a = db.get_run(run_a)
    row_b = db.get_run(run_b)
    missing = [rid for rid, row in ((run_a, row_a), (run_b, row_b)) if row is None]
    if missing:
        raise SystemExit(f"unknown run id(s): {', '.join(missing)}")
    decisions_a = db.get_decisions(run_a)
    decisions_b = db.get_decisions(run_b)
    decisions_recorded = bool(decisions_a) and bool(decisions_b)
    if decisions_recorded:
        report = compare_decisions(decisions_a, decisions_b)
    else:
        report = dict(_NO_DECISION_LOG)
        report["steps_a"] = len(decisions_a)
        report["steps_b"] = len(decisions_b)
    report.update(
        compare_equity(db.get_equity_curve(run_a), db.get_equity_curve(run_b))
    )
    final_a = row_a.get("final_equity")
    final_b = row_b.get("final_equity")
    gap_pct = None
    if final_a and final_b is not None:
        gap_pct = 100.0 * (float(final_b) - float(final_a)) / float(final_a)
    report.update(
        {
            "run_a": run_a,
            "run_b": run_b,
            "decisions_recorded": decisions_recorded,
            "basis": "decisions" if decisions_recorded else "equity",
            "final_equity_a": final_a,
            "final_equity_b": final_b,
            "final_equity_gap_pct": gap_pct,
            "sampling_a": (row_a.get("metadata") or {}).get("llm_sampling"),
            "sampling_b": (row_b.get("metadata") or {}).get("llm_sampling"),
        }
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Where do two backtests of one configuration first disagree?"
    )
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args(argv)
    print(json.dumps(compare_runs(args.run_a, args.run_b), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
