#!/usr/bin/env python3
"""Deploy an LLM model onto the leaderboard.

Runs a configured model's hourly backtest over the contest window and caches the
result (equity curve + metrics + token cost) so the web leaderboard shows it
without recomputing. This is how you "permanently deploy" a model:

  1. Add an entry to dashboard/config/leaderboard.json with "strategy": "llm_agent",
     "integration": "commonstack" | "openrouter" | "anthropic", and
     "auto_compute": false (see claude_haiku_4_5 / nemotron_3_nano_30b).
  2. Run this script once (it makes real LLM API calls):

       # Quick smoke test on a short window (cheap):
       python3 dashboard/scripts/deploy_leaderboard_model.py \
         --entry claude_haiku_4_5 --start 2026-04-15 --end 2026-04-16

       # Full contest window (the one the leaderboard displays):
       python3 dashboard/scripts/deploy_leaderboard_model.py --entry claude_haiku_4_5

  3. Refresh the leaderboard — the model appears as a provided baseline.

Requires the API key for the entry's integration in dashboard/.env
(COMMONSTACK_API_KEY, OPENROUTER_API_KEY, and/or ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

DASHBOARD_DIR = Path(__file__).resolve().parent.parent

# Direct-execution bootstrap: make the repo root importable so canonical
# `dashboard.backend.*` imports resolve (no-op when run as part of the package).
from _bootstrap import ensure_repo_root

ensure_repo_root()

# Load secrets (ANTHROPIC_API_KEY, etc.) from dashboard/.env then repo root .env.
load_dotenv(DASHBOARD_DIR / ".env")
load_dotenv(DASHBOARD_DIR.parent / ".env")

from dashboard.backend.domain.leaderboard.service import (  # noqa: E402
    LeaderboardFallbackError,
    deploy_model_run,
    load_leaderboard_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy an LLM model onto the leaderboard")
    parser.add_argument("--entry", required=True, help="Leaderboard entry id (e.g. claude_haiku_4_5)")
    parser.add_argument("--start", default=None, help="Override window start (YYYY-MM-DD) for testing")
    parser.add_argument("--end", default=None, help="Override window end (YYYY-MM-DD) for testing")
    parser.add_argument("--force", action="store_true", help="Recompute even if a cached run exists")
    parser.add_argument(
        "--period",
        choices=("contest", "daily", "live"),
        default="contest",
        help="Target board: contest, daily, or live (current-month freeze snapshot)",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Publish even if the LLM entry fell back to rule-based trading "
        "(by default that is refused so a rule-based curve is not shown as an LLM result)",
    )
    parser.add_argument("--list", action="store_true", help="List configured entries and exit")
    args = parser.parse_args()

    if args.list:
        config = load_leaderboard_config()
        print("Configured leaderboard entries:")
        for s in config.get("strategies", []):
            auto = s.get("auto_compute", True)
            integ = s.get("integration") or "-"
            print(
                f"  - {s['id']:<22} model={s.get('model'):<22} "
                f"strategy={s.get('strategy'):<18} integration={integ:<12} "
                f"auto_compute={auto}"
            )
        return 0

    print(f"Deploying '{args.entry}' to the {args.period} leaderboard...")
    if args.start or args.end:
        print(f"  (test window override: {args.start or 'config'} → {args.end or 'config'})")

    try:
        result = deploy_model_run(
            args.entry,
            force_refresh=args.force,
            start_date=args.start,
            end_date=args.end,
            allow_fallback=args.allow_fallback,
            period=args.period,
        )
    except LeaderboardFallbackError as exc:
        print("\n❌ Refused to publish a rule-based fallback under an LLM name:")
        print(f"   {exc}")
        return 2

    print("\n" + "=" * 60)
    if result.get("cached"):
        print(f"✅ Already deployed (cached). Use --force to recompute.")
    else:
        print(f"✅ Deployed.")
    print("=" * 60)
    print(f"  Entry        : {result['entry_id']}")
    print(f"  Model        : {result.get('model')}")
    print(f"  Run ID       : {result['run_id']}")
    ret = result.get("total_return")
    if ret is not None:
        print(f"  Return       : {ret * 100:+.2f}%")
    if result.get("sharpe_ratio") is not None:
        print(f"  Sharpe       : {float(result['sharpe_ratio']):.2f}")
    if result.get("max_drawdown") is not None:
        print(f"  Max Drawdown : {abs(float(result['max_drawdown'])) * 100:.2f}%")
    if result.get("final_equity") is not None:
        print(f"  Final Equity : ${float(result['final_equity']):,.0f}")
    print(f"  Trades       : {result.get('num_trades')}")
    print(f"  LLM Calls    : {result.get('llm_calls')}")
    print(
        f"  Tokens       : {int(result.get('input_tokens') or 0):,} in / "
        f"{int(result.get('output_tokens') or 0):,} out"
    )
    print(f"  Est. Cost    : ${float(result.get('est_cost_usd') or 0):.4f}")
    print("\nRefresh the leaderboard (or GET /api/v1/leaderboard?refresh=true) to see it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
