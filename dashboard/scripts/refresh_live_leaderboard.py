#!/usr/bin/env python3
"""Persist the Live Trading Leaderboard freeze snapshot for the current month.

1. Recomputes cheap baselines (indices + rule-based strategies) for
   month-open → last completed US cash session under ``leaderboard-live``.
2. Optionally deploys every competition ``llm_agent`` over that same freeze
   (real API calls — this is how model curves appear on GET ?period=live).

Public GET never runs this. After each close, re-run with --models to append
a new snapshot row (start stays month-open; end_date is that freeze). The
board keeps serving the latest snapshot that does not extend past freeze.

    python dashboard/scripts/refresh_live_leaderboard.py --models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

DASHBOARD_DIR = Path(__file__).resolve().parent.parent

from _bootstrap import ensure_repo_root

ensure_repo_root()

load_dotenv(DASHBOARD_DIR / ".env")
load_dotenv(DASHBOARD_DIR.parent / ".env")

from dashboard.backend.domain.leaderboard.live import (  # noqa: E402
    clear_live_session_runs,
    live_freeze_config,
    live_llm_entries,
    refresh_live_leaderboard,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh the Live Trading Leaderboard freeze snapshot"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Also deploy the Live LLM roster (GPT / DeepSeek / Nemotron) for the freeze window",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow publishing LLM entries that fell back to rule-based trading",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all leaderboard-live runs before refreshing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the per-freeze refresh cache and recompute even if cached",
    )
    args = parser.parse_args()

    freeze = live_freeze_config()
    if freeze is None:
        print(
            "ERROR: no completed cash session this month yet — nothing to freeze.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Live freeze window: {freeze['start_date']} → {freeze['end_date']}"
    )
    print(f"Session: {freeze['session_id']}")
    roster = [e["id"] for e in live_llm_entries(freeze)]
    print(f"Live LLM roster: {', '.join(roster)}")

    if args.clear:
        n = clear_live_session_runs()
        print(f"Cleared {n} leaderboard-live run(s).")

    try:
        result = refresh_live_leaderboard(
            deploy_models=args.models,
            force_refresh=args.force,
            allow_fallback=args.allow_fallback,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if result.get("skipped"):
        print("Already refreshed for this freeze — skipped.")
        return 0

    baselines = result.get("baselines") or {}
    print(
        f"Baselines created: {baselines.get('created', 0)} "
        f"(skipped {baselines.get('skipped', 0)})"
    )

    if not args.models:
        print("Done (baselines only). Pass --models to deploy LLM freeze snapshots.")
        print("View: GET /api/v1/leaderboard?period=live")
        return 0

    failures = result.get("model_failures") or []
    for row in result.get("model_results") or []:
        ret = row.get("total_return")
        ret_s = f"{ret * 100:+.2f}%" if ret is not None else "—"
        print(f"  ok  {row.get('entry_id')}  run={row.get('run_id')}  return={ret_s}")
    for fail in failures:
        print(f"  FAIL {fail.get('entry_id')}: {fail.get('error')}", file=sys.stderr)

    print(f"\nDone. Failures: {len(failures)}")
    print("View: GET /api/v1/leaderboard?period=live")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
