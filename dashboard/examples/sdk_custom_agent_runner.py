#!/usr/bin/env python3
"""AgentRunner example: implement only decide(), let the SDK run the loop.

A contributor writes a class with a single required method, ``decide``. The
optional hooks ``on_execution_result`` and ``on_run_completed`` are detected
automatically if present. ``AgentRunner`` handles create-run, polling,
loading waits, decision submission, and final result retrieval.

Credentials come from the environment (never hard-code keys)::

    export ATL_API_KEY="ag_xxx"
    export ATL_BASE_URL="http://127.0.0.1:8000"
    export ATL_AGENT_VERSION_ID="agv_xxx"   # or ATL_AGENT_ID to create one

Run:
    python3 dashboard/examples/sdk_custom_agent_runner.py
"""

from __future__ import annotations

import os
import sys

from agentictrading import AgentRunner, ATLClient


class MomentumAgent:
    """Buys the strongest positive-momentum names once, then holds."""

    # AgentRunner hands decide() the observation only, not step.constraints, so the
    # position cap is a parameter. 0.25 is us-equity-hourly-v1's max_position_weight.
    def __init__(self, num_names: int = 5, max_position_weight: float = 0.25) -> None:
        self.num_names = num_names
        self.max_position_weight = max_position_weight
        self._invested = False

    def dollars_per_name(self, portfolio: dict) -> float:
        """Size each buy from the current cash and equity, under the position cap.

        Orders are sized at this bar's price but fill at the next bar's, plus
        costs, so 5% is kept back or the last buy can come up short of cash.
        """
        cash = float(portfolio.get("cash") or 0)
        equity = float(portfolio.get("equity") or cash)
        return min(self.max_position_weight * equity, cash / self.num_names) * 0.95

    def decide(self, observation):
        if self._invested:
            return {"orders": [], "rationale": "Holding."}

        budget = self.dollars_per_name(observation.portfolio)
        features = observation.features
        ranked = sorted(
            (
                (sym, float(feat.get("price") or 0), float(feat.get("momentum") or feat.get("return_1d") or 0))
                for sym, feat in features.items()
            ),
            key=lambda row: row[2],
            reverse=True,
        )
        orders = [
            {
                "symbol": sym,
                "side": "buy",
                "quantity_type": "notional",
                "quantity": round(budget, 2),
                "order_type": "market",
            }
            for sym, price, _ in ranked
            if 0 < price <= budget  # a pricier name would round to zero shares
        ][: self.num_names]

        if orders:
            self._invested = True
        return {"orders": orders, "confidence": 0.7, "rationale": "Top momentum entry."}

    # Optional hooks - detected automatically by AgentRunner.
    def on_execution_result(self, result):
        if result.fills:
            print(f"  filled {len(result.fills)} order(s); equity={result.portfolio_after.get('equity')}")

    def on_run_completed(self, result):
        print(f"  run completed: {result.metrics.get('total_return')} total return")


def resolve_agent_version_id(client: ATLClient) -> str:
    version_id = os.environ.get("ATL_AGENT_VERSION_ID")
    if version_id:
        return version_id
    agent_id = os.environ.get("ATL_AGENT_ID")
    if not agent_id:
        raise SystemExit(
            "Set ATL_AGENT_VERSION_ID (preferred) or ATL_AGENT_ID to create one."
        )
    version = client.create_agent_version(
        agent_id,
        version="0.1.0",
        architecture="momentum",
        model_backbones=["rule-based"],
    )
    print(f"Created AgentVersion: {version.id}")
    return version.id


def main() -> int:
    api_key = os.environ.get("ATL_API_KEY")
    if not api_key:
        raise SystemExit("ATL_API_KEY is required")
    base_url = os.environ.get("ATL_BASE_URL", "http://127.0.0.1:8000")
    start = os.environ.get("ATL_START_DATE", "2026-04-15")
    end = os.environ.get("ATL_END_DATE", "2026-04-30")

    client = ATLClient(base_url=base_url, api_key=api_key)
    agent_version_id = resolve_agent_version_id(client)

    runner = AgentRunner(client=client, agent=MomentumAgent())
    result = runner.run_backtest(
        agent_version_id,
        environment_id="us-equity-hourly-v1",
        start_date=start,
        end_date=end,
    )

    print("\n=== Metrics ===")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
