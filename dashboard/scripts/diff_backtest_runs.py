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
from typing import Any, Dict, List, Optional, Tuple

if not __package__:
    from _bootstrap import ensure_repo_root

    ensure_repo_root()

from dashboard.backend.database import db  # noqa: E402
from dashboard.backend.db_url import describe_database_url  # noqa: E402


#: Fields that must agree before two runs are one configuration measured
#: twice. A mismatch does not stop the report, it names itself in
#: ``mismatches`` and fails the CLI: a before/after pair that differs in
#: window, model or universe diverges for reasons that have nothing to do
#: with sampling. ``initial_pipeline`` (the strategy prompt and step count)
#: and ``data_source`` live in the row's metadata, and are as much a part of
#: "one configuration" as the window: a pair that differs in either is not a
#: sampling measurement.
#:
#: The rest are the inputs that move the curve without any model call
#: changing. The tape (``market_data_feed``, ``sip_fallback_to_iex``,
#: ``end_clamped``): curves priced off different feeds diverge from bar 1, so
#: a before run that fell back to IEX is not a sampling measurement against an
#: after run on SIP. ``initial_equity`` scales every order. ``frequency_contract``
#: is the bar cadence the decisions ran on. ``llm_max_output_tokens`` is the
#: ceiling truncation retries hinge on, which is how a ceiling change shows up
#: as divergence.
_COMPARABILITY_FIELDS = (
    "start_date",
    "end_date",
    "llm_model",
    "symbols",
    "initial_pipeline",
    "data_source",
    "market_data_feed",
    "sip_fallback_to_iex",
    "end_clamped",
    "initial_equity",
    "frequency_contract",
    "llm_max_output_tokens",
)

#: Read off the row's metadata rather than its columns.
_METADATA_FIELDS = frozenset(
    {
        "data_source",
        "market_data_feed",
        "sip_fallback_to_iex",
        "end_clamped",
        "llm_max_output_tokens",
    }
)


def _normalised_actions(entry: Dict[str, Any]) -> str:
    return json.dumps(entry.get("actions_submitted") or [], sort_keys=True)


def _compare_keyed(
    a: Dict[Any, Any], b: Dict[Any, Any], same
) -> Tuple[int, int, List[Any], int, int]:
    """Join two keyed series; return (shared, divergent, divergent_keys, only_a, only_b).

    ``divergent_keys`` is in key order, so its first element is the earliest
    shared key that differs.
    """
    shared = sorted(a.keys() & b.keys())
    divergent_keys = [k for k in shared if not same(a[k], b[k])]
    return (
        len(shared),
        len(divergent_keys),
        divergent_keys,
        len(a.keys() - b.keys()),
        len(b.keys() - a.keys()),
    )


def compare_decisions(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare two decision logs, joined on ``step_index``.

    ``step_index`` is the key ``backtest_decisions`` is ordered and indexed
    on. Joining on it rather than zipping by position means a log with a
    missing middle step, or one cut short by a timeout, is compared only
    where both runs have an answer, and the unmatched steps are counted in
    ``decisions_only_in_a`` / ``decisions_only_in_b`` instead of shifting every
    later step against the wrong partner.
    """
    by_a = {int(x.get("step_index", 0)): x for x in a}
    by_b = {int(y.get("step_index", 0)): y for y in b}
    shared, divergent, keys, only_a, only_b = _compare_keyed(
        by_a, by_b, lambda x, y: _normalised_actions(x) == _normalised_actions(y)
    )
    first: Optional[Dict[str, Any]] = None
    if keys:
        first = {"step_index": keys[0], "timestamp": by_a[keys[0]].get("timestamp")}
    return {
        "steps_compared": shared,
        "steps_a": len(by_a),
        "steps_b": len(by_b),
        "decisions_only_in_a": only_a,
        "decisions_only_in_b": only_b,
        # Nothing shared is nothing measured, not agreement.
        "divergent_steps": divergent if shared else None,
        "first_divergence": first,
    }


def compare_equity(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare two equity curves, joined on timestamp.

    The axis that always exists: ``run_agent_backtest`` writes
    ``equity_timeseries`` for every run, unconditionally.

    Joined on timestamp, not position, so a partial or timed-out run is
    compared only over the bars both runs have; the bars only one has are
    counted in ``equity_only_in_a`` / ``equity_only_in_b`` and are not
    divergence.

    Exact comparison, no tolerance. Two runs whose decisions agreed ran the
    same arithmetic over the same bars, so any difference at all is real, and
    a tolerance would swallow exactly the smallest and earliest divergence
    this script exists to find.
    """
    by_a = {str(p.get("timestamp")): p for p in a}
    by_b = {str(p.get("timestamp")): p for p in b}
    shared, divergent, keys, only_a, only_b = _compare_keyed(
        by_a, by_b, lambda x, y: x.get("equity") == y.get("equity")
    )
    first: Optional[Dict[str, Any]] = None
    if keys:
        first = {"index": sorted(by_a).index(keys[0]), "timestamp": keys[0]}
    return {
        "equity_points_compared": shared,
        "equity_points_a": len(by_a),
        "equity_points_b": len(by_b),
        "equity_only_in_a": only_a,
        "equity_only_in_b": only_b,
        "divergent_equity_points": divergent if shared else None,
        "first_equity_divergence": first,
    }


#: What the decision axis reports when there is no decision log to read.
#: ``None``, never ``0``: ``backtest_decisions`` is written only by the AI
#: Hedge Fund runtime and the external-agent surface, so an empty log on a
#: pipeline run means *unmeasured*. A ``0`` there is this script asserting
#: that two runs agreed on every bar, out of a table nobody wrote to.
_NO_DECISION_LOG = {
    "steps_compared": None,
    "divergent_steps": None,
    "first_divergence": None,
}


def _run_field(row: Dict[str, Any], name: str) -> Any:
    """A comparability field off an ``agent_runs`` row, or its metadata."""
    metadata = row.get("metadata") or {}
    if name == "symbols":
        symbols = metadata.get("symbols")
        return sorted(symbols) if symbols else None
    if name == "initial_pipeline":
        # Normalised, so key order inside a step is not a difference, and
        # without step ids: seeding mints a random id per agent, so two agents
        # running the same instruction would otherwise never compare. An id is
        # identity, not configuration -- nothing the model sees depends on it.
        pipeline = metadata.get("initial_pipeline")
        if pipeline is None:
            return None
        if isinstance(pipeline, list):
            pipeline = [
                {key: value for key, value in step.items() if key != "id"}
                if isinstance(step, dict)
                else step
                for step in pipeline
            ]
        return json.dumps(pipeline, sort_keys=True)
    if name == "frequency_contract":
        contract = metadata.get("frequency_contract")
        return json.dumps(contract, sort_keys=True) if contract is not None else None
    if name in _METADATA_FIELDS:
        return metadata.get(name)
    return row.get(name)


def _final_equity_gap(final_a: Any, final_b: Any) -> Tuple[Optional[float], Optional[str]]:
    if final_a is None or final_b is None:
        return None, "final equity not recorded for " + " and ".join(
            name
            for name, value in (("run_a", final_a), ("run_b", final_b))
            if value is None
        )
    if not float(final_a):
        return None, "run_a final equity is zero"
    return 100.0 * (float(final_b) - float(final_a)) / float(final_a), None


def _lane_warnings(label: str, row: Dict[str, Any]) -> List[str]:
    """Say when a run's calls were answered on more than one lane.

    One sampling policy takes a different shape per lane (``llm_sampling.wire``),
    so a run that failed over part-way did not send every bar the same request,
    and divergence after the switch is not sampling noise alone. A warning, not
    a mismatch: the configuration was the same, the route was not.
    """
    sampling = (row.get("metadata") or {}).get("llm_sampling") or {}
    wire = sampling.get("wire") if isinstance(sampling, dict) else None
    if isinstance(wire, dict) and len(wire) > 1:
        shapes = ", ".join(f"{lane}: {controls}" for lane, controls in wire.items())
        return [f"{label} answered on more than one lane ({shapes})"]
    return []


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
    gap_pct, gap_reason = _final_equity_gap(final_a, final_b)

    mismatches: List[str] = []
    fields: Dict[str, Any] = {}
    for name in _COMPARABILITY_FIELDS:
        value_a, value_b = _run_field(row_a, name), _run_field(row_b, name)
        fields[f"{name}_a"], fields[f"{name}_b"] = value_a, value_b
        # A run that never recorded a field (a legacy row has no symbols)
        # cannot be shown to differ on it; only two recorded values can.
        if value_a is not None and value_b is not None and value_a != value_b:
            mismatches.append(name)

    report.update(
        {
            "run_a": run_a,
            "run_b": run_b,
            **fields,
            "comparable": not mismatches,
            "mismatches": mismatches,
            "decisions_recorded": decisions_recorded,
            "basis": "decisions" if decisions_recorded else "equity",
            "final_equity_a": final_a,
            "final_equity_b": final_b,
            "final_equity_gap_pct": gap_pct,
            "sampling_a": (row_a.get("metadata") or {}).get("llm_sampling"),
            "sampling_b": (row_b.get("metadata") or {}).get("llm_sampling"),
            "warnings": _lane_warnings("run_a", row_a) + _lane_warnings("run_b", row_b),
        }
    )
    if gap_reason:
        report["final_equity_gap_reason"] = gap_reason
    return report


def describe_run_history_backend() -> str:
    """Name the store ``db`` is bound to, credentials never included.

    A wrong ``DATABASE_PATH`` (or a missing ``AGENT_RUNS_DATABASE_URL``)
    answers "unknown run id" or, worse, reads a different file's runs, so the
    source is printed on every invocation.
    """
    url = getattr(db, "database_url", None)
    if url:
        return f"postgres ({describe_database_url(url)})"
    return f"sqlite ({getattr(db, 'db_path', '?')})"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Where do two backtests of one configuration first disagree?"
    )
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help=(
            "exit 0 even when the runs differ in window, model, universe, "
            "pipeline, data source, tape, capital, cadence or output ceiling"
        ),
    )
    args = parser.parse_args(argv)
    print(f"run history: {describe_run_history_backend()}", file=sys.stderr)
    report = compare_runs(args.run_a, args.run_b)
    print(json.dumps(report, indent=2, default=str))
    for warning in report["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    if report["mismatches"] and not args.allow_mismatch:
        print(
            "runs are not comparable, they differ in: "
            + ", ".join(report["mismatches"])
            + " (pass --allow-mismatch to accept)",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
