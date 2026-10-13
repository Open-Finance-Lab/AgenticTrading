"""Reference client for the /api/v2 agent contract (Plan 2).

Demonstrates the four canonical verbs end-to-end. The agent's LLM runs CLIENT-SIDE:
this script fetches context, decides locally (here: a trivial rule), and submits.
The backend only serves context and validates — it never calls your model.

Usage:
    python external_agent_client_v2.py --api http://localhost:8000 \
        --start 2026-04-15 --end 2026-04-16
"""

from __future__ import annotations

import argparse
import time
import uuid

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--name", default="v2-reference-agent")
    ap.add_argument("--model", default="local-model")
    args = ap.parse_args()

    base = args.api.rstrip("/")

    # 1) register → api_key (shown once)
    reg = requests.post(f"{base}/api/v2/agents",
                        json={"name": args.name, "model_name": args.model}).json()
    key = reg["api_key"]
    headers = {"X-API-Key": key}
    print(f"registered agent {reg['agent_id']} (scopes: {reg['scopes']})")

    # 2) create a run
    run = requests.post(f"{base}/api/v2/runs", headers=headers, json={
        "mode": "backtest", "universe": "djia_30",
        "start_date": args.start, "end_date": args.end,
        "agent_name": args.name, "model_name": args.model,
    }).json()
    run_id = run["run_id"]
    print(f"run {run_id} → {run['status']}")

    # 3) loop: get_context → decide locally → submit_decision
    while True:
        ctx = requests.get(f"{base}/api/v2/runs/{run_id}/context", headers=headers).json()
        status = ctx.get("status")
        if status == "loading":
            time.sleep(1.0)
            continue
        if status in ("completed", "closed", "failed"):
            break

        # ---- CLIENT-SIDE decision (replace with your LLM call) ----
        # Size each buy as a third of the cash on hand, 5% kept back for the next
        # bar's fill price: a fixed share count overshoots cash on pricier names.
        cash = float((ctx.get("portfolio") or {}).get("cash") or 0)
        budget = 0.95 * cash / 3
        actions = []
        for sym, sig in list(ctx.get("top_signals", {}).items())[:3]:
            news = ctx.get("news_sentiment", {}).get(sym, {})
            shares = int(budget // sig["price"]) if sig["price"] > 0 else 0
            if shares > 0 and (sig["rsi"] < 35 or news.get("sentiment") == "bullish"):
                actions.append({
                    "action": "buy", "symbol": sym, "confidence": 0.7,
                    "reasoning": f"rsi={sig['rsi']:.0f}, news={news.get('sentiment','n/a')}",
                    "position_size": shares,
                })

        ack = requests.post(f"{base}/api/v2/runs/{run_id}/decisions", headers=headers, json={
            "idempotency_key": str(uuid.uuid4()), "actions": actions,
        }).json()
        print(f"step {ctx['step_index']}/{ctx['total_steps']} → "
              f"executed {len(ack.get('executed', []))}, status {ack.get('status')}")
        if ack.get("status") == "completed":
            break

    # 4) get_result
    result = requests.get(f"{base}/api/v2/runs/{run_id}/result", headers=headers).json()
    print("metrics:", result.get("metrics"))


if __name__ == "__main__":
    main()
