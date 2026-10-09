#!/usr/bin/env python3
"""Recompute leaderboard baseline runs for the configured contest window.

Deletes cached leaderboard runs outside the configured contest window, then
recomputes baselines. Mean-Variance uses the prior month (reference_start_date)
for weight estimation only; equity curves cover the contest window.

Usage (from repo root):

    python dashboard/scripts/refresh_leaderboard_baselines.py

Requires Alpaca credentials for Buy & Hold / Mean-Variance / Equal-Weight
(DJIA and SPY index lines use Yahoo Finance and do not need Alpaca).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

DASHBOARD_DIR = Path(__file__).resolve().parent.parent

from _bootstrap import ensure_repo_root

ensure_repo_root()

load_dotenv(DASHBOARD_DIR / ".env")
load_dotenv(DASHBOARD_DIR.parent / ".env")

from dashboard.backend.database import db  # noqa: E402
from dashboard.backend.domain.leaderboard.service import (  # noqa: E402
    ensure_leaderboard_runs,
    load_leaderboard_config,
)

def _delete_stale_runs(session_id: str, keep_start: str, keep_end: str) -> int:
    """Remove leaderboard runs for any window other than the configured one."""
    removed = 0
    for run in db.get_runs_by_session(session_id) or []:
        if run.get("mode") != "leaderboard":
            continue
        start = run.get("start_date")
        end = run.get("end_date")
        if start == keep_start and end == keep_end:
            continue
        db.delete_run(run["run_id"])
        print(f"  removed {run['run_id']} ({start} → {end})")
        removed += 1
    return removed


def main() -> int:
    config = load_leaderboard_config()
    session_id = config["session_id"]
    start = config["start_date"]
    end = config["end_date"]

    print(f"Leaderboard window: {start} → {end}")
    print("Removing stale cached leaderboard runs…")
    removed = _delete_stale_runs(session_id, start, end)
    print(f"Removed {removed} run(s).")

    print("Recomputing baselines (Alpaca + Yahoo)…")
    try:
        meta = ensure_leaderboard_runs(force_refresh=True)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Tip: set ALPACA_API_KEY / ALPACA_SECRET_KEY or credentials/alpaca.json "
            "for stock-based baselines.",
            file=sys.stderr,
        )
        return 1

    print(f"Done — created {meta.get('created', 0)} run(s).")
    print("Refresh the Competition tab, or force a recompute: GET /api/v1/leaderboard?refresh=true with X-Leaderboard-Refresh-Secret or an admin session")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
