#!/usr/bin/env python3
"""
Run a simple trading policy against the Agent-Environment Protocol (v1).

This is a readable, hackable reference: it authenticates with an Agent API key,
creates an immutable AgentVersion, opens a Run, then steps through the protocol
(Observation -> Decision -> ExecutionResult) printing what happens each step.

Policy = "equal-weight buy & hold": on the first step it buys a fixed dollar
amount (notional order) of a few symbols, then holds for the rest of the run.
Swap out `decide()` to plug in your own logic / LLM.

Usage:
  python3 dashboard/examples/protocol_policy_demo.py \\
    --api http://127.0.0.1:8000 \\
    --api-key ag_xxxxxxxx \\
    --start 2026-04-15 --end 2026-04-30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(method: str, url: str, api_key: str, body: dict | None = None, timeout: int = 90) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc


# How many names to buy on the first step, and dollars per name.
# The us-equity-hourly-v1 environment starts with $1,000 and caps one position at 25% of
# equity ($250), so five $200 orders fit. A name priced above $200 would round to zero
# shares and be rejected, so the examples skip those.
NUM_NAMES = 5
DOLLARS_PER_NAME = 200


def decide(observation: dict, sequence: int) -> list[dict]:
    """Return protocol orders for this step. First step buys, then holds.

    Buys the symbols actually present in the observation (the environment only
    surfaces its top signals each step), so the demo reliably produces fills.
    """
    if sequence != 0:
        return []  # HOLD
    features = (observation.get("market") or {}).get("features") or {}
    tradable = [s for s, f in features.items() if 0 < float(f.get("price") or 0) <= DOLLARS_PER_NAME]
    orders = []
    for symbol in tradable[:NUM_NAMES]:
        orders.append({
            "symbol": symbol,
            "side": "buy",
            "quantity_type": "notional",   # spend a dollar amount; server -> shares
            "quantity": DOLLARS_PER_NAME,
            "order_type": "market",
        })
    return orders


def main() -> int:
    p = argparse.ArgumentParser(description="Protocol v1 policy demo")
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", required=True)
    p.add_argument("--start", default="2026-04-15")
    p.add_argument("--end", default="2026-04-30")
    args = p.parse_args()
    base = args.api.rstrip("/")
    key = args.api_key

    agent = call("GET", f"{base}/api/v1/agents/resolve", key)
    print(f"Agent: {agent['name']} ({agent['agent_id']})")

    version = call("POST", f"{base}/api/v1/agents/{agent['agent_id']}/versions", key, {
        "version": "0.1.0",
        "architecture": "equal_weight_buy_hold",
        "model_backbones": ["rule-based"],
    })["agent_version"]
    print(f"AgentVersion: {version['agent_version_id']}")

    run = call("POST", f"{base}/api/v1/runs", key, {
        "agent_version_id": version["agent_version_id"],
        "environment": {"type": "backtest", "environment_id": "us-equity-hourly-v1"},
        "config": {"start_date": args.start, "end_date": args.end},
    })
    run_id = run["run_id"]
    print(f"Run: {run_id}\n")

    n = 0
    while True:
        step = call("GET", f"{base}/api/v1/runs/{run_id}/steps/next", key)
        status = step.get("status")
        if status == "loading":
            print("  loading market data...")
            time.sleep(2)
            continue
        if status == "completed":
            break
        if status != "awaiting_decision":
            time.sleep(0.5)
            continue

        seq = step["sequence"]
        obs = step["observation"]
        pf = obs["portfolio"]
        orders = decide(obs, seq)
        res = call(
            "POST",
            f"{base}/api/v1/runs/{run_id}/steps/{step['step_id']}/decision",
            key,
            {
                "idempotency_key": f"{run_id}-{seq}",
                "orders": orders,
                "confidence": 0.9,
                "rationale": "equal-weight buy & hold demo",
            },
        )
        fills = res.get("fills") or []
        rej = res.get("validation", {}).get("rejections") or []
        after = res.get("portfolio_after", {})
        n += 1
        line = (
            f"step {seq:>2} | equity_before=${pf['equity']:>10,.0f} "
            f"| orders={len(orders)} fills={len(fills)} rejections={len(rej)} "
            f"| cash_after=${after.get('cash', 0):>10,.0f} equity_after=${after.get('equity', 0):>10,.0f}"
        )
        print(line)
        for f in fills:
            print(f"        FILL {f['side'].upper():4} {f['symbol']:4} x{f['filled_quantity']} @ ${f['fill_price']:.2f}")
        for r in rej:
            print(f"        REJECT {r['order'].get('symbol')}: {r['reason']}")

    print(f"\nStepped through {n} decisions.")
    result = call("GET", f"{base}/api/v1/runs/{run_id}/result", key)
    m = result.get("metrics") or {}
    print("\n=== Result ===")
    print(f"  total_return : {m.get('total_return')}")
    print(f"  sharpe_ratio : {m.get('sharpe_ratio')}")
    print(f"  max_drawdown : {m.get('max_drawdown')}")
    print(f"  num_trades   : {m.get('num_trades')}")
    print(f"  final_equity : {m.get('final_equity')}")
    trades = call("GET", f"{base}/api/v1/runs/{run_id}/trades", key)
    print(f"  trades logged: {trades.get('count')}")
    print(f"\nInspect: {base}/api/v1/runs/{run_id}/result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
