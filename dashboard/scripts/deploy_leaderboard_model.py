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

One model run is one draw (#602). To publish a median and a range instead:

       python3 dashboard/scripts/deploy_leaderboard_model.py \
         --entry deepseek_v4_pro --samples 3

writes repeat runs 1..N beside the primary row (existing ones are skipped
unless --force). The board publishes the median once two comparable repeats
exist; the primary row is left untouched. Each repeat is billed in full.

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
    describe_entry_publication,
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
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Write repeat runs 1..N (#602); the board publishes their median. "
        "Contest/daily only. Each repeat is a full billable run.",
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

    if args.samples is not None:
        if args.period == "live":
            print("--samples is not supported for the live board")
            return 1
        if args.samples < 1:
            print("--samples must be at least 1")
            return 1
        return _deploy_samples(args)

    try:
        if args.period == "live":
            result = _deploy_live(args)
            if result is None:
                return 1
        else:
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
    print("\nRefresh the leaderboard to see it (forcing a recompute: GET /api/v1/leaderboard?refresh=true with X-Leaderboard-Refresh-Secret or an admin session).")
    return 0


def _deploy_samples(args) -> int:
    """Repeat runs 1..N, one line each, then what the board will publish.

    Stops at the first failure: a fallback or a broken run is likelier to
    repeat than to clear on the next billable attempt. Whatever was written
    still counts, so the closing line is printed either way.
    """
    status = 0
    for n in range(1, args.samples + 1):
        try:
            result = deploy_model_run(
                args.entry,
                force_refresh=args.force,
                start_date=args.start,
                end_date=args.end,
                allow_fallback=args.allow_fallback,
                period=args.period,
                sample=n,
            )
        except LeaderboardFallbackError as exc:
            print(f"\n❌ Sample {n} refused (rule-based fallback): {exc}")
            status = 2
            break
        except (RuntimeError, ValueError) as exc:
            print(f"\n❌ Sample {n} failed: {exc}")
            status = 1
            break
        ret = result.get("total_return")
        drift = result.get("config_drift") or []
        if not result.get("cached"):
            state = "new"
        elif drift:
            state = f"cached, recorded under a different {', '.join(drift)}; --force re-runs it"
        else:
            state = "cached"
        ret_text = f"{ret * 100:+.2f}%" if ret is not None else "—"
        print(
            f"  sample {n}: {result['run_id']} ({state}) return {ret_text} "
            f"cost ${float(result.get('est_cost_usd') or 0):.4f}"
        )
    print(_publication_line(args))
    return status


def _publication_line(args) -> str:
    """The board's own answer for this entry -- see ``describe_entry_publication``.

    Not recomputed here from the runs just written: the board pools by recorded
    config and counts repeats this invocation did not write, so a local median
    can name a number the page never shows.
    """
    published = describe_entry_publication(
        args.entry, period=args.period, start_date=args.start, end_date=args.end
    )
    samples = published["samples"]
    if published["run_id"] is None:
        return "\nBoard publishes nothing for this entry yet."
    if not samples or samples["count"] < 2:
        return f"\nBoard publishes a single run: {published['run_id']}."
    kind = "median" if samples["count"] % 2 else "a middle run"
    return (
        f"\nBoard publishes {kind} of {samples['count']} runs "
        f"({published['run_id']}, {float(published['total_return']) * 100:+.2f}%), "
        f"range {samples['min_return'] * 100:+.2f}% to {samples['max_return'] * 100:+.2f}%."
    )


def _deploy_live(args):
    """One Live-board entry through the snapshot-carrying increment.

    ``deploy_model_run`` writes no portfolio snapshot, so a live row made there
    is a month replay the next nightly increment has to pay for again.
    """
    from dashboard.backend.domain.leaderboard.live import (
        deploy_live_model_increment,
        live_freeze_config,
        live_llm_entries,
    )

    if args.start or args.end:
        print("ERROR: --start/--end do not apply to --period live (the window is the month freeze).")
        return None
    freeze = live_freeze_config()
    if freeze is None:
        print("ERROR: no settled cash session this month yet — nothing to freeze.")
        return None
    entry = next((e for e in live_llm_entries(freeze) if e["id"] == args.entry), None)
    if entry is None:
        print(f"ERROR: '{args.entry}' is not on the Live roster.")
        return None
    return deploy_live_model_increment(
        entry,
        freeze,
        force_refresh=args.force,
        allow_fallback=args.allow_fallback,
    )


if __name__ == "__main__":
    sys.exit(main())
