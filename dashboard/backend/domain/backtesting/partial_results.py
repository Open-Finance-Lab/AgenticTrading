"""Startup reclaimer for interrupted backtests (design: partial results).

A dashboard backtest is a subprocess whose results only reach the database at
completion. If the server restarts mid-run (a deploy, a crash), the run leaves
no row, no curve — the user's wait simply vanishes. Since the engine now
mirrors its progress payload into ``run_live_progress`` while running, this
module turns whatever snapshots survived a restart back into honest
*interrupted* runs: an ``agent_runs`` row carrying ``interrupted`` metadata
and the step it reached, plus the partial equity curve and trades in the
regular tables — so every existing reader (run lists, chart endpoints,
comparisons) serves the partial result with zero changes.

Snapshots whose run has a terminal ``agent_runs`` row are stale leftovers
from a completed run (the engine does not delete on success); the reclaimer
drops them. Idempotent: reclaiming twice writes the same interrupted row
(insert_run upserts) and re-inserting the same equity points is a no-op.
"""

from __future__ import annotations

from typing import Any, Dict


def reclaim_interrupted_backtests() -> Dict[str, int]:
    """Run once at startup. Returns counts for the boot log."""
    from dashboard.backend.database import db

    reclaimed = 0
    dropped_stale = 0
    failed = 0
    try:
        snapshots = db.list_live_progress()
    except Exception as exc:  # noqa: BLE001 - reclaim must never break boot
        print(f"⚠️  interrupted-run reclaim: snapshot listing failed: {exc}")
        return {"reclaimed": 0, "dropped_stale": 0, "failed": 0}

    for snapshot in snapshots:
        run_id = str(snapshot.get("run_id") or "").strip()
        if not run_id:
            continue
        try:
            existing = db.get_run(run_id)
            if existing is not None:
                # Terminal row present: the run finished after this snapshot
                # was written (or was reclaimed already). Drop the stale copy.
                db.delete_live_progress(run_id)
                dropped_stale += 1
                continue

            payload = snapshot.get("payload") or {}
            meta = payload.get("run_metadata") or {}
            curve = payload.get("equity_curve") or []
            trades = payload.get("trades") or []
            step = int(payload.get("step") or 0)
            total_steps = int(payload.get("total_steps") or 0)

            last_equity = float(curve[-1]["equity"]) if curve else None
            initial_equity = float(curve[0]["equity"]) if curve else 0.0

            db.insert_run(
                run_id,
                str(snapshot.get("session_id") or ""),
                str(snapshot.get("agent_name") or "agent"),
                str(meta.get("mode") or "llm"),
                str(meta.get("start_date") or ""),
                str(meta.get("end_date") or ""),
                initial_equity,
                final_equity=last_equity,
                num_trades=len(trades),
                llm_model=str(meta.get("llm_model") or "rule-based"),
                metadata={
                    "interrupted": True,
                    "interrupted_reason": "server restart",
                    "interrupted_step": step,
                    "interrupted_total_steps": total_steps,
                    "partial": True,
                },
            )
            # Curve points already carry timestamp/equity/cash/positions_value
            # in the shape insert_equity_points expects.
            db.insert_equity_points(run_id, curve, replace=True)
            if trades:
                db.insert_trades(run_id, trades)
            db.delete_live_progress(run_id)
            reclaimed += 1
        except Exception as exc:  # noqa: BLE001 - one bad row starves the rest
            failed += 1
            print(f"⚠️  interrupted-run reclaim: {run_id} failed: {exc}")

    return {"reclaimed": reclaimed, "dropped_stale": dropped_stale, "failed": failed}
