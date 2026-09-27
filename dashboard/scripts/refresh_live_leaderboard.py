#!/usr/bin/env python3
"""Persist the Live Trading Leaderboard freeze snapshot for the current month.

1. Recomputes cheap baselines (indices + rule-based strategies) for
   month-open → last settled US cash session under ``leaderboard-live``.
2. With ``--models`` (billable), appends each Live LLM roster entry for
   sessions not yet stored (usually one cash day), restoring cash/positions
   from yesterday's snapshot.
3. Deletes this month's freeze rows the new ones supersede.

Public GET never runs this; it only reads the rows written here.

    python dashboard/scripts/refresh_live_leaderboard.py --models

``--clear`` also forgets the last refresh's window, so the refresh after it
always runs instead of reporting "already refreshed" over an empty board.

Remote prod (Render) without shell access — enqueues a background refresh
(HTTP 202); poll ``GET /api/v1/leaderboard?period=live``:

    curl -X POST "$ATL_API/api/v1/leaderboard/live/refresh?deploy_models=true" \\
      -H "X-Leaderboard-Refresh-Secret: $LEADERBOARD_DAILY_REFRESH_SECRET"
"""

from __future__ import annotations

import argparse
import os
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
        help="Append the Live LLM roster (GPT / DeepSeek / Nemotron) for new cash sessions",
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
        help="Replay the whole month from the 1st (ignores the daily snapshot)",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="POST to ATL_API instead of running locally (uses LEADERBOARD_DAILY_REFRESH_SECRET)",
    )
    args = parser.parse_args()

    if args.remote:
        return _refresh_remote(
            deploy_models=args.models,
            force=args.force,
            allow_fallback=args.allow_fallback,
        )

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
        print("Done (baselines only). Pass --models to append LLM freeze snapshots.")
        print("View: GET /api/v1/leaderboard?period=live")
        return 0

    failures = result.get("model_failures") or []
    for row in result.get("model_results") or []:
        ret = row.get("total_return")
        ret_s = f"{ret * 100:+.2f}%" if ret is not None else "—"
        seg = row.get("segment") or {}
        extra = ""
        if seg:
            extra = f"  segment={seg.get('start_date')}→{seg.get('end_date')}"
        print(
            f"  ok  {row.get('entry_id')}  run={row.get('run_id')}  "
            f"return={ret_s}{extra}"
        )
    for fail in failures:
        print(f"  FAIL {fail.get('entry_id')}: {fail.get('error')}", file=sys.stderr)

    print(f"\nDone. Failures: {len(failures)}")
    print("View: GET /api/v1/leaderboard?period=live")
    return 1 if failures else 0


def _refresh_remote(*, deploy_models: bool, force: bool, allow_fallback: bool) -> int:
    import httpx

    secret = (os.getenv("LEADERBOARD_DAILY_REFRESH_SECRET") or "").strip()
    if not secret:
        print("ERROR: LEADERBOARD_DAILY_REFRESH_SECRET is not set", file=sys.stderr)
        return 1
    if allow_fallback:
        print(
            "ERROR: --allow-fallback cannot be combined with --remote; run the "
            "refresh locally against the target database instead.",
            file=sys.stderr,
        )
        return 1
    base = (os.getenv("ATL_API") or os.getenv("ATL_API_BASE") or "http://localhost:8000").rstrip("/")
    params = {
        "deploy_models": str(deploy_models).lower(),
        "force": str(force).lower(),
    }
    url = f"{base}/api/v1/leaderboard/live/refresh"
    print(f"POST {url}")
    try:
        resp = httpx.post(
            url,
            params=params,
            headers={"X-Leaderboard-Refresh-Secret": secret},
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if resp.status_code >= 400:
        print(f"ERROR {resp.status_code}: {resp.text}", file=sys.stderr)
        return 1
    print(resp.text)
    if resp.status_code == 202:
        print("Accepted (background). Poll GET /api/v1/leaderboard?period=live")
    return 0


if __name__ == "__main__":
    sys.exit(main())
