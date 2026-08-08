#!/usr/bin/env python3
"""Refresh the Daily Leaderboard for the last completed US weekday.

1. Recomputes cheap baselines (indices + rule-based strategies) for the daily window.
2. Optionally redeploys every competition ``llm_agent`` entry (real API calls).

Schedule nightly after US market close, e.g. via GitHub Actions or:

    python dashboard/scripts/refresh_daily_leaderboard.py --models

Remote prod (Render) without shell access — enqueues a background refresh
(HTTP 202); poll ``GET /api/v1/leaderboard?period=daily`` for progress:

    curl -X POST "$ATL_API/api/v1/leaderboard/daily/refresh?deploy_models=true" \\
      -H "X-Leaderboard-Refresh-Secret: $LEADERBOARD_DAILY_REFRESH_SECRET"

Window math is America/New_York + 16:00 cash close: after the close the board
is *that* weekday; before the close (and on weekends) it rolls to the prior
session. There is no Saturday/Sunday session.
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

from dashboard.backend.domain.leaderboard.service import (  # noqa: E402
    refresh_daily_leaderboard,
    resolve_leaderboard_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh the Daily Leaderboard window")
    parser.add_argument(
        "--models",
        action="store_true",
        help="Also redeploy every llm_agent entry for the daily window (expensive)",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow publishing LLM entries that fell back to rule-based trading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the per-window refresh cache",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="POST to ATL_API instead of running locally (uses LEADERBOARD_DAILY_REFRESH_SECRET)",
    )
    args = parser.parse_args()

    if args.remote:
        return _refresh_remote(deploy_models=args.models, force=args.force, allow_fallback=args.allow_fallback)

    config = resolve_leaderboard_config("daily")
    print(f"Daily leaderboard window: {config['start_date']} → {config['end_date']}")
    print(f"Session: {config['session_id']}")

    try:
        result = refresh_daily_leaderboard(
            deploy_models=args.models,
            force_refresh=args.force,
            allow_fallback=args.allow_fallback,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if result.get("skipped"):
        print("Already refreshed for this window — skipped.")
        return 0

    baselines = result.get("baselines") or {}
    print(
        f"Baselines created: {baselines.get('created', 0)} "
        f"(skipped {baselines.get('skipped', 0)})"
    )

    if not args.models:
        print("Done (baselines only). Pass --models to redeploy LLM entries.")
        print("View: GET /api/v1/leaderboard?period=daily")
        return 0

    failures = result.get("model_failures") or []
    for row in result.get("model_results") or []:
        ret = row.get("total_return")
        ret_s = f"{ret * 100:+.2f}%" if ret is not None else "—"
        print(f"  ok  {row.get('entry_id')}  run={row.get('run_id')}  return={ret_s}")
    for fail in failures:
        print(f"  FAIL {fail.get('entry_id')}: {fail.get('error')}", file=sys.stderr)

    print(f"\nDone. Failures: {len(failures)}")
    print("View: GET /api/v1/leaderboard?period=daily")
    return 1 if failures else 0


def _refresh_remote(*, deploy_models: bool, force: bool, allow_fallback: bool) -> int:
    import httpx

    secret = (os.getenv("LEADERBOARD_DAILY_REFRESH_SECRET") or "").strip()
    if not secret:
        print("ERROR: LEADERBOARD_DAILY_REFRESH_SECRET is not set", file=sys.stderr)
        return 1
    if allow_fallback:
        # The H6 integrity guard is deliberately not waivable over HTTP, so the
        # endpoint has no such parameter. Fail loudly rather than silently
        # publishing under the guard the operator thought they had lifted.
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
    url = f"{base}/api/v1/leaderboard/daily/refresh"
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
        print("Accepted (background). Poll GET /api/v1/leaderboard?period=daily")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
