#!/usr/bin/env python3
"""
Minimal external-agent backtest client.

Uses simple RSI rules (no LLM required). Point at your API server and reuse
the dashboard session id so results appear on the website.

Two modes:

  * --protocol v1  (recommended) uses the versioned Agent-Environment Protocol:
      AgentVersion -> Run -> Step -> Decision -> ExecutionResult.
      Authenticates directly with the Agent API key (X-API-Key); no session id.

  * --protocol legacy (default, kept for backward compatibility) uses the
      original /api/v1/backtest/* endpoints with X-Session-Id.

Usage:
  # New Run API (protocol v1) — only needs the agent API key:
  python3 examples/external_agent_client.py \\
    --protocol v1 \\
    --api https://agentictrading.onrender.com \\
    --api-key ag_xxxxxxxx \\
    --start 2026-04-15 --end 2026-04-16

  # Legacy backtest API with an API key (auto-resolves session):
  python3 examples/external_agent_client.py \\
    --api http://localhost:8000 \\
    --api-key ag_xxxxxxxx \\
    --start 2026-04-15 --end 2026-04-16

  # Legacy with a manual session id:
  python3 examples/external_agent_client.py \\
    --api http://localhost:8000 \\
    --session-id "$SESSION_ID" \\
    --start 2026-04-15 --end 2026-04-16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def api_request(
    method: str,
    url: str,
    session_id: str,
    body: dict | None = None,
    timeout: int = 120,
) -> dict:
    data = None
    headers = {
        "X-Session-Id": session_id,
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 409:
            try:
                return json.loads(detail).get("detail", json.loads(detail))
            except json.JSONDecodeError:
                pass
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and "detail" in parsed:
                detail = parsed["detail"]
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Request timed out after {timeout}s: {url}. "
            "For long date ranges on cloud, redeploy the API or retry — /start now loads data in the background."
        ) from exc


def rule_based_decision(snapshot: dict) -> dict:
    """Demo agent: buy oversold, sell overbought, else hold."""
    actions = []
    holdings = snapshot.get("current_holdings") or {}
    signals = snapshot.get("top_signals") or {}

    for symbol, sig in signals.items():
        rsi = float(sig.get("rsi") or 50)
        price = float(sig.get("price") or 0)
        if price <= 0:
            continue

        owned = symbol in holdings and holdings[symbol].get("shares", 0) > 0

        if not owned and rsi < 35:
            shares = max(1, int(2000 / price))
            actions.append({
                "action": "buy",
                "symbol": symbol,
                "confidence": 0.75,
                "reasoning": "RSI oversold entry",
                "position_size": shares,
                "stop_loss_price": None,
                "take_profit_price": None,
            })
        elif owned and rsi > 65:
            actions.append({
                "action": "sell",
                "symbol": symbol,
                "confidence": 0.8,
                "reasoning": "RSI overbought exit",
                "position_size": holdings[symbol]["shares"],
                "stop_loss_price": None,
                "take_profit_price": None,
            })

    if not actions:
        actions.append({
            "action": "hold",
            "symbol": next(iter(signals), "AAPL"),
            "confidence": 0.5,
            "reasoning": "No signal this hour",
            "position_size": 0,
            "stop_loss_price": None,
            "take_profit_price": None,
        })

    return {"actions": actions}


def resolve_api_key(base: str, api_key: str) -> dict:
    req = urllib.request.Request(
        f"{base.rstrip('/')}/api/v1/agents/resolve",
        headers={"X-API-Key": api_key, "Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Protocol v1 (Run API) helpers
# ---------------------------------------------------------------------------


def api_request_key(
    method: str,
    url: str,
    api_key: str,
    body: dict | None = None,
    timeout: int = 120,
) -> dict:
    """Authenticated request for the Run API using the Agent API key."""
    data = None
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(detail)
            detail = parsed.get("detail", parsed)
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc


def rule_based_orders(observation: dict) -> list[dict]:
    """Demo agent: translate features into protocol orders (orders-v1)."""
    orders = []
    features = (observation.get("market") or {}).get("features") or {}
    holdings = {
        p["symbol"]: p
        for p in (observation.get("portfolio", {}).get("positions") or [])
    }
    for symbol, sig in features.items():
        rsi = float(sig.get("rsi") or 50)
        price = float(sig.get("price") or 0)
        if price <= 0:
            continue
        owned = symbol in holdings and holdings[symbol].get("quantity", 0) > 0
        if not owned and rsi < 35:
            orders.append({
                "symbol": symbol,
                "side": "buy",
                "quantity_type": "shares",
                "quantity": max(1, int(2000 / price)),
                "order_type": "market",
            })
        elif owned and rsi > 65:
            orders.append({
                "symbol": symbol,
                "side": "sell",
                "quantity_type": "shares",
                "quantity": holdings[symbol]["quantity"],
                "order_type": "market",
            })
    return orders


def run_protocol_v1(base: str, api_key: str, args) -> int:
    resolved = resolve_api_key(base, api_key)
    agent_id = resolved["agent_id"]
    print(f"Agent: {resolved.get('name')} ({agent_id})")

    version = api_request_key(
        "POST",
        f"{base}/api/v1/agents/{agent_id}/versions",
        api_key,
        {
            "version": "0.1.0",
            "execution_mode": "external",
            "architecture": "single_agent_rule_based",
            "model_backbones": [resolved.get("model_name") or "rule-based"],
            "decision_frequency": "1h",
        },
    )["agent_version"]
    agent_version_id = version["agent_version_id"]
    print(f"AgentVersion: {agent_version_id}")

    run = api_request_key(
        "POST",
        f"{base}/api/v1/runs",
        api_key,
        {
            "agent_version_id": agent_version_id,
            "environment": {"type": "backtest", "environment_id": "us-equity-hourly-v1"},
            "config": {"start_date": args.start, "end_date": args.end},
        },
        timeout=60,
    )
    run_id = run["run_id"]
    print(f"Run: {run_id} (status={run.get('status')})")

    idem = 0
    while True:
        step = api_request_key("GET", f"{base}/api/v1/runs/{run_id}/steps/next", api_key)
        status = step.get("status")
        if status == "loading":
            print("  Loading market data…")
            time.sleep(2)
            continue
        if status == "completed":
            break
        if status != "awaiting_decision":
            time.sleep(0.5)
            continue

        step_id = step["step_id"]
        seq = step.get("sequence")
        print(f"Step seq={seq} step_id={step_id} deadline={step.get('deadline_at')}")
        orders = rule_based_orders(step.get("observation") or {})
        idem += 1
        result = api_request_key(
            "POST",
            f"{base}/api/v1/runs/{run_id}/steps/{step_id}/decision",
            api_key,
            {
                "idempotency_key": f"{run_id}-{seq}-{idem}",
                "run_id": run_id,
                "step_id": step_id,
                "orders": orders,
                "confidence": 0.7,
                "rationale": "rule-based demo decision",
            },
        )
        fills = result.get("fills") or []
        rejections = result.get("validation", {}).get("rejections") or []
        print(f"  fills={len(fills)} rejections={len(rejections)}")
        for fill in fills:
            print(f"    {fill['side'].upper()} {fill['symbol']} x{fill['filled_quantity']} @ {fill['fill_price']}")

    print("\nDone!")
    result = api_request_key("GET", f"{base}/api/v1/runs/{run_id}/result", api_key)
    metrics = result.get("metrics") or {}
    print(f"Total return: {metrics.get('total_return')}")
    print(f"Sharpe: {metrics.get('sharpe_ratio')}  Trades: {metrics.get('num_trades')}")
    print(f"\nView on website: My Agents → View in Playground")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="External agent backtest client (demo)")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", help="Registered agent API key (auto-resolves session)")
    parser.add_argument("--session-id", help="X-Session-Id (optional if --api-key is set)")
    parser.add_argument("--agent-name", default="my-local-agent")
    parser.add_argument("--model-name", default="rule-based-demo")
    parser.add_argument("--start", default="2026-04-15")
    parser.add_argument("--end", default="2026-04-16")
    parser.add_argument(
        "--protocol",
        choices=["legacy", "v1"],
        default="legacy",
        help="legacy=/api/v1/backtest/* (session), v1=/api/v1/runs/* (Agent-Environment Protocol)",
    )
    args = parser.parse_args()

    base = args.api.rstrip("/")

    if args.protocol == "v1":
        if not args.api_key:
            parser.error("--protocol v1 requires --api-key")
        return run_protocol_v1(base, args.api_key, args)

    session_id = args.session_id
    agent_name = args.agent_name
    model_name = args.model_name

    if args.api_key:
        resolved = resolve_api_key(base, args.api_key)
        session_id = resolved["session_id"]
        if args.agent_name == "my-local-agent" and resolved.get("name"):
            agent_name = resolved["name"]
        if args.model_name == "rule-based-demo" and resolved.get("model_name"):
            model_name = resolved["model_name"]
        print(f"Agent: {resolved.get('name')} ({resolved.get('agent_id')})")
    elif not session_id:
        parser.error("Provide --api-key or --session-id")

    print(f"Session: {session_id}")
    print("Dashboard: open My Agents and click View in Playground (no console needed).")
    print()

    try:
        schema = api_request("GET", f"{base}/api/v1/backtest/schema", session_id)
        print(f"Decision timeout: {schema.get('decision_timeout_seconds')}s")
    except RuntimeError as exc:
        print(f"Warning: could not fetch schema ({exc})")

    started = api_request(
        "POST",
        f"{base}/api/v1/backtest/start",
        session_id,
        {
            "start_date": args.start,
            "end_date": args.end,
            "agent_name": agent_name,
            "model_name": model_name,
            "mode": "safe_trading",
        },
        timeout=60,
    )
    backtest_id = started["backtest_id"]
    total = started.get("total_steps") or 0
    if started.get("status") == "loading":
        print(f"Backtest {backtest_id} started — loading market data…")
    else:
        print(f"Started backtest {backtest_id} ({total} steps)")

    step_num = 0
    loading_announced = False
    while True:
        ctx = api_request(
            "GET",
            f"{base}/api/v1/backtest/{backtest_id}/steps/current",
            session_id,
        )
        status = ctx.get("status")

        if status == "loading":
            if not loading_announced:
                print("  Fetching Alpaca bars (long ranges may take 1–3 min on cloud)…")
                loading_announced = True
            time.sleep(2)
            continue

        if status == "completed":
            print("\nDone!")
            run_id = ctx.get("run_id")
            compare_path = ctx.get("compare_url")
            print(json.dumps(ctx, indent=2))
            if compare_path:
                print(f"\nEquity chart: {base}{compare_path}")
            if run_id:
                try:
                    summary = api_request(
                        "GET",
                        f"{base}/api/v1/backtest/runs/{run_id}/result",
                        session_id,
                    )
                    print(f"\nTrades: {len(summary.get('trades', []))}")
                    print(f"Decisions: {len(summary.get('decisions', []))}")
                except RuntimeError as exc:
                    print(f"Result fetch: {exc}")
            print(f"\nView on website: My Agents → View in Playground")
            return 0

        if status == "failed":
            print("Backtest failed:", ctx.get("error"))
            return 1

        if status != "waiting_decision":
            time.sleep(0.5)
            continue

        if not total:
            total = ctx.get("total_steps") or total
            print(f"Loaded {total} trading hours")

        step_num = ctx.get("step_index", 0) + 1
        deadline = ctx.get("decision_deadline_at", "")
        print(f"Step {step_num}/{total} deadline={deadline}")

        payload = rule_based_decision(ctx.get("market_snapshot") or {})
        result = api_request(
            "POST",
            f"{base}/api/v1/backtest/{backtest_id}/steps/current/decisions",
            session_id,
            payload,
        )
        if isinstance(result, dict) and result.get("error") == "step_already_closed":
            print("  Step closed (timeout), continuing...")
            continue
        if not result.get("accepted"):
            print("  Decision rejected:", result)
        else:
            executed = result.get("executed") or []
            print(f"  Executed {result.get('executed_count', 0)} action(s)")
            for item in executed:
                print(f"    {item.get('action', '').upper()} {item.get('symbol')} x{item.get('shares')}")
            if result.get("status") == "completed" and result.get("compare_url"):
                print(f"\nCompleted early. Chart: {base}{result['compare_url']}")
                return 0


if __name__ == "__main__":
    sys.exit(main())
