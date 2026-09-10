#!/usr/bin/env python3
"""Run one Alpaca live-trading decision cycle from the command line.

This is the manual on-ramp to ``dashboard.backend.execution.alpaca_live_service``
for a single local account — no dashboard UI required. It is **review-only by
default**: pass ``--execute`` AND set ``ALPACA_LIVE_EXECUTE=true`` in ``.env``
to let orders actually reach the market (the same two-gate pattern the app
already uses for Robinhood live trading).

Examples
--------
Review what the agent would do, with no risk at all:
    python dashboard/scripts/run_alpaca_live_agent.py --instruction "Trade conservatively across DJIA names"

Actually place orders (after you've watched several dry runs and set
ALPACA_LIVE_EXECUTE=true in .env):
    python dashboard/scripts/run_alpaca_live_agent.py --execute --instruction "..."
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

DASHBOARD_DIR = Path(__file__).resolve().parent.parent

from _bootstrap import ensure_repo_root

ensure_repo_root()

# Load secrets (ALPACA_LIVE_API_KEY, etc.) from dashboard/.env then repo root .env.
load_dotenv(DASHBOARD_DIR / ".env")
load_dotenv(DASHBOARD_DIR.parent / ".env")

from dashboard.backend.execution import alpaca_live_service as svc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--instruction",
        default="Trade conservatively across DJIA names. Prefer holding cash when uncertain.",
        help="Trading prompt/strategy instruction given to the model.",
    )
    parser.add_argument("--model", default=None, help="Override the model name (defaults to the app's default).")
    parser.add_argument(
        "--symbols", nargs="*", default=None, help="Symbol universe to consider (defaults to the DJIA 30)."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Allow real order placement for this run. Still requires "
            "ALPACA_LIVE_EXECUTE=true in .env — without it, this run stays "
            "review-only regardless of this flag."
        ),
    )
    args = parser.parse_args()

    if args.execute and not svc.execute_enabled():
        print(
            "--execute was passed, but ALPACA_LIVE_EXECUTE is not 'true' in the environment.\n"
            "This run will stay in review-only mode. Set ALPACA_LIVE_EXECUTE=true in .env "
            "once you're ready for real orders.\n",
            file=sys.stderr,
        )

    result = asyncio.run(
        svc.run_live_for_agent(
            instruction=args.instruction,
            model_name=args.model,
            symbols=args.symbols,
            dry_run=not args.execute,
        )
    )

    print(json.dumps(result, indent=2, default=str))

    if result.get("dry_run"):
        print("\n[dry run] No orders were sent to the broker.", file=sys.stderr)
    else:
        print("\n[LIVE] Orders were sent to your real Alpaca account.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
