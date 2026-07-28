#!/usr/bin/env python3
"""Bulk-create agents via POST /api/v1/agents for load / UI testing.

Each agent gets a small cash_allocation (default $10) so many can fit in the
$10k account portfolio. Requires a logged-in session (Bearer token).

Quick start (local web UI already open and signed in):

  1. In the browser console on /app:
       localStorage.getItem('auth-token')
       localStorage.getItem('browser-owner-id')   # optional but recommended

  2. From repo root:
       python dashboard/scripts/bulk_create_agents.py \\
         --token "<paste auth-token>" \\
         --browser-id "<paste browser-owner-id>" \\
         --count 100 \\
         --cash 10

  Prod example:
       python dashboard/scripts/bulk_create_agents.py \\
         --base-url https://your-atl-host \\
         --token "..." --browser-id "..." --count 100 --cash 10

  Login instead of --token:
       python dashboard/scripts/bulk_create_agents.py \\
         --email you@example.com --password '...' --count 10 --cash 10

Use --dry-run to print the plan without creating anything.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any, Optional

import requests

from _bootstrap import ensure_repo_root

ensure_repo_root()


def _headers(token: str, browser_id: Optional[str], session_id: Optional[str]) -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if browser_id:
        h["x-browser-id"] = browser_id
    if session_id:
        h["x-session-id"] = session_id
    return h


def login(base_url: str, email: str, password: str) -> str:
    url = f"{base_url.rstrip('/')}/auth/login"
    resp = requests.post(url, json={"email": email, "password": password}, timeout=30)
    if resp.status_code != 200:
        raise SystemExit(f"Login failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    token = data.get("token")
    if not token:
        raise SystemExit("Login response missing token")
    user = data.get("user") or {}
    print(f"Logged in as {user.get('display_name') or user.get('email') or '?'}")
    return token


def get_portfolio(base_url: str, token: str, browser_id: Optional[str]) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/portfolio"
    resp = requests.get(url, headers=_headers(token, browser_id, None), timeout=30)
    if resp.status_code != 200:
        return {}
    return resp.json().get("portfolio") or {}


def list_agents(base_url: str, token: str, browser_id: Optional[str]) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/v1/agents"
    resp = requests.get(url, headers=_headers(token, browser_id, None), timeout=120)
    if resp.status_code != 200:
        raise SystemExit(f"List agents failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json().get("agents") or []


def create_agent(
    base_url: str,
    token: str,
    browser_id: Optional[str],
    *,
    name: str,
    cash: float,
    model_name: str,
    agent_type: str,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/agents"
    body = {
        "name": name,
        "model_name": model_name,
        "agent_type": agent_type,
        "cash_allocation": cash,
    }
    resp = requests.post(
        url,
        headers=_headers(token, browser_id, None),
        json=body,
        timeout=60,
    )
    if resp.status_code != 200:
        detail = resp.text[:400]
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass  # non-JSON body -- keep the raw-text fallback set above
        raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk-create agents for web load testing")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="ATL backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument("--token", default=None, help="Bearer auth token (from localStorage auth-token)")
    parser.add_argument("--email", default=None, help="Login email (alternative to --token)")
    parser.add_argument("--password", default=None, help="Login password")
    parser.add_argument(
        "--browser-id",
        default=None,
        help="x-browser-id header (localStorage browser-owner-id); auto-generated if omitted",
    )
    parser.add_argument("--count", type=int, default=100, help="Number of agents to create (default: 100)")
    parser.add_argument("--cash", type=float, default=10.0, help="cash_allocation per agent (default: 10)")
    parser.add_argument("--prefix", default="load-test", help="Agent name prefix (default: load-test)")
    parser.add_argument(
        "--model",
        default="local-model",
        help="model_name for each agent (default: local-model, no LLM cost)",
    )
    parser.add_argument(
        "--type",
        dest="agent_type",
        choices=("builtin", "external"),
        default="builtin",
        help="agent_type (default: builtin, matches Foundation agents in UI)",
    )
    parser.add_argument("--start-index", type=int, default=1, help="First suffix number (default: 1)")
    parser.add_argument("--delay-ms", type=int, default=0, help="Pause between creates (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, do not POST")
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.cash < 0:
        raise SystemExit("--cash must be >= 0")

    token = args.token
    if not token:
        if not args.email or not args.password:
            raise SystemExit("Provide --token or both --email and --password")
        token = login(args.base_url, args.email, args.password)

    browser_id = args.browser_id or f"bulk-script-{uuid.uuid4().hex[:12]}"

    total_needed = args.count * args.cash
    print(f"Target : {args.base_url}")
    print(f"Plan   : create {args.count} agents × ${args.cash:.2f} = ${total_needed:.2f} allocated")
    print(f"Names  : {args.prefix}-{args.start_index:03d} … {args.prefix}-{args.start_index + args.count - 1:03d}")
    print(f"Model  : {args.model}  type={args.agent_type}")

    portfolio = get_portfolio(args.base_url, token, browser_id)
    if portfolio:
        avail = float(portfolio.get("cash_available") or 0)
        allocated = float(portfolio.get("allocated") or 0)
        equity = float(portfolio.get("equity") or portfolio.get("total_value") or 0)
        print(
            f"Portfolio: equity=${equity:,.2f}  allocated=${allocated:,.2f}  "
            f"unallocated=${avail:,.2f}"
        )
        if total_needed > avail + 0.01:
            max_fit = int(avail // args.cash) if args.cash > 0 else args.count
            print(
                f"\nWARN: Need ${total_needed:,.2f} unallocated cash but only ${avail:,.2f} available."
            )
            print(f"   At ${args.cash:.2f}/agent you can create at most ~{max_fit} more.")
            if not args.dry_run:
                ans = input("Continue anyway? [y/N] ").strip().lower()
                if ans != "y":
                    return 1

    try:
        existing = list_agents(args.base_url, token, browser_id)
        print(f"Existing agents: {len(existing)}")
    except SystemExit:
        pass  # informational only -- listing failure shouldn't block creation

    if args.dry_run:
        print("\n[dry-run] No agents created.")
        return 0

    ok = 0
    failed: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    for i in range(args.start_index, args.start_index + args.count):
        name = f"{args.prefix}-{i:03d}"
        try:
            result = create_agent(
                args.base_url,
                token,
                browser_id,
                name=name,
                cash=args.cash,
                model_name=args.model,
                agent_type=args.agent_type,
            )
            agent = result.get("agent") or {}
            ok += 1
            print(
                f"  OK [{ok}/{args.count}] {name} -> {agent.get('agent_id', '?')} "
                f"(${agent.get('cash_allocation', args.cash):.2f})"
            )
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"  FAIL {name}: {exc}")
            if "Insufficient unallocated cash" in str(exc):
                print("     Stopping: portfolio out of unallocated cash.")
                break

        if args.delay_ms > 0 and i < args.start_index + args.count - 1:
            time.sleep(args.delay_ms / 1000.0)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 60)
    print(f"Created: {ok}/{args.count} in {elapsed:.1f}s")
    if failed:
        print(f"Failed : {len(failed)}")
        for name, err in failed[:5]:
            print(f"  - {name}: {err}")
        if len(failed) > 5:
            print(f"  … and {len(failed) - 5} more")

    portfolio = get_portfolio(args.base_url, token, browser_id)
    if portfolio:
        print(
            f"Portfolio now: allocated=${float(portfolio.get('allocated') or 0):,.2f}  "
            f"unallocated=${float(portfolio.get('cash_available') or 0):,.2f}"
        )

    print("\nRefresh My Agents in the browser to see the new cards.")
    return 0 if ok == args.count else 1


if __name__ == "__main__":
    sys.exit(main())
