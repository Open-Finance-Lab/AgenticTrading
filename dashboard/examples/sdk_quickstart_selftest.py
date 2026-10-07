#!/usr/bin/env python3
"""One-command SDK self-test against a running Agentic Trading Lab backend.

What it does:
  1. Logs in with your account (email + password from env) to get an auth token.
  2. Creates an agent owned by that account (so the run shows up under
     "My Agents" on the website) and obtains its API key.
  3. Creates an AgentVersion via the SDK (ATLClient).
  4. Runs a short backtest with AgentRunner using a tiny equal-dollar rule.
  5. Prints the final metrics.

Neither the password nor the API key is ever printed.

Prerequisites:
  - Backend running, e.g.:
      uvicorn dashboard.backend.app:app --host 127.0.0.1 --port 8000   # from the repo root
  - The agentictrading package importable (run via the command below).
  - Your account credentials in the environment:
      export ATL_EMAIL="you@example.com"
      export ATL_PASSWORD="your-password"

Run it:
  PYTHONPATH=packaging/agentictrading/src \
    python3 dashboard/examples/sdk_quickstart_selftest.py

Optional env overrides:
  ATL_BASE_URL   (default http://127.0.0.1:8000)
  ATL_START_DATE (default 2026-04-15)
  ATL_END_DATE   (default 2026-04-30)
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import uuid

from agentictrading import AgentRunner, ATLClient

# The us-equity-hourly-v1 environment starts with $1,000 and caps one position at 25% of
# equity ($250), so five $200 orders fit. A name priced above $200 would round to zero
# shares and be rejected, so the examples skip those.
NUM_NAMES = 5
DOLLARS_PER_NAME = 200


def login(base_url: str, email: str, password: str) -> tuple[str, dict]:
    """Authenticate and return (token, public_user)."""
    body = json.dumps({"email": email, "password": password}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/auth/login",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["token"], data.get("user", {})


def create_account_agent(base_url: str, token: str) -> tuple[str, str]:
    """Create an agent owned by the logged-in account; return (agent_id, api_key).

    Passing the bearer token sets the agent's owner_user_id, so the agent and its
    runs appear under "My Agents" for that account on the website.
    """
    session = f"sdk-selftest-{uuid.uuid4().hex[:8]}"
    body = json.dumps({"name": "sdk-selftest-agent", "model_name": "rule-based"}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/v1/agents",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Session-Id": session,
            "X-Browser-Id": session,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["agent"]["agent_id"], data["api_key"]


class EqualDollarAgent:
    """Buys a few names on the first step, then holds."""

    def __init__(self):
        self._invested = False

    def decide(self, observation):
        if self._invested:
            return {"orders": [], "rationale": "Holding."}
        tradable = [
            sym for sym, feat in observation.features.items()
            if 0 < float(feat.get("price") or 0) <= DOLLARS_PER_NAME
        ]
        orders = [
            {
                "symbol": sym,
                "side": "buy",
                "quantity_type": "notional",
                "quantity": DOLLARS_PER_NAME,
                "order_type": "market",
            }
            for sym in tradable[:NUM_NAMES]
        ]
        if orders:
            self._invested = True
        return {"orders": orders, "confidence": 0.7, "rationale": "Equal-dollar entry."}

    def on_execution_result(self, result):
        for fill in result.fills:
            print(f"  FILL {fill['side'].upper():4} {fill['symbol']:5} "
                  f"x{fill['filled_quantity']} @ ${fill['fill_price']:.2f}")


def main() -> int:
    base_url = os.environ.get("ATL_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    start = os.environ.get("ATL_START_DATE", "2026-04-15")
    end = os.environ.get("ATL_END_DATE", "2026-04-30")

    email = os.environ.get("ATL_EMAIL")
    password = os.environ.get("ATL_PASSWORD")
    if not email or not password:
        print("Set ATL_EMAIL and ATL_PASSWORD to log in with your account, e.g.:")
        print('  export ATL_EMAIL="you@example.com"')
        print('  export ATL_PASSWORD="your-password"')
        return 1

    print(f"Backend: {base_url}")
    try:
        token, user = login(base_url, email, password)
    except Exception as exc:  # noqa: BLE001 - surface a friendly hint
        print(f"Login failed for {email} ({exc}).")
        print("Check ATL_EMAIL / ATL_PASSWORD and that the backend is running on", base_url)
        return 1
    print(f"Logged in as {user.get('email', email)} (user id {user.get('id')})")

    try:
        agent_id, api_key = create_account_agent(base_url, token)
    except Exception as exc:  # noqa: BLE001 - surface a friendly hint
        print(f"Could not create an agent ({exc}).")
        return 1
    print(f"Created agent owned by your account: {agent_id} (api key length {len(api_key)})\n")

    client = ATLClient(base_url=base_url, api_key=api_key)

    version = client.create_agent_version(
        agent_id,
        version="0.1.0",
        architecture="equal_dollar_buy_hold",
        model_backbones=["rule-based"],
    )
    print(f"AgentVersion: {version.id}")

    runner = AgentRunner(client=client, agent=EqualDollarAgent())
    print(f"Running backtest {start} -> {end} ...\n")
    result = runner.run_backtest(
        version.id,
        environment_id="us-equity-hourly-v1",
        start_date=start,
        end_date=end,
    )

    print("\n=== Metrics ===")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    print(f"\nInspect raw result: {base_url}/api/v1/runs/{result.run_id}/result")
    return 0


if __name__ == "__main__":
    sys.exit(main())
