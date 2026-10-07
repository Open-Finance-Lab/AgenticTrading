#!/usr/bin/env python3
"""Rule-based SDK example: read the observation, submit at least one real order.

The rule is intentionally simple and deterministic: on the first step, spend an
equal dollar amount on the first few tradable symbols surfaced in the
observation, then hold. It demonstrates reading from:

    step.observation.market       (features / prices)
    step.observation.portfolio    (cash / equity / positions)
    step.constraints              (allowed symbols, max orders, ...)

Credentials come from the environment (never hard-code keys)::

    export ATL_API_KEY="ag_xxx"
    export ATL_BASE_URL="http://127.0.0.1:8000"
    export ATL_AGENT_VERSION_ID="agv_xxx"   # or ATL_AGENT_ID to create one

Run:
    python3 dashboard/examples/sdk_rule_based_agent.py
"""

from __future__ import annotations

import os
import sys

from agentictrading import ATLClient, Decision, Order

# The us-equity-hourly-v1 environment starts with $1,000 and caps one position at 25% of
# equity ($250), so five $200 orders fit. A name priced above $200 would round to zero
# shares and be rejected, so the examples skip those.
NUM_NAMES = 5
DOLLARS_PER_NAME = 200


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
        architecture="equal_weight_buy_hold",
        model_backbones=["rule-based"],
    )
    print(f"Created AgentVersion: {version.id}")
    return version.id


def decide(step) -> Decision:
    """Equal-dollar buy on the first step; HOLD afterwards."""
    if step.sequence:  # only act on the first step (sequence 0)
        return Decision(orders=[], rationale="Holding existing positions.")

    features = step.observation.features
    constraints = step.constraints or {}
    allowed = set(constraints.get("allowed_symbols") or [])
    max_orders = int(constraints.get("max_orders") or NUM_NAMES)

    tradable = [
        sym
        for sym, feat in features.items()
        if 0 < float(feat.get("price") or 0) <= DOLLARS_PER_NAME
        and (not allowed or sym in allowed)
    ]

    orders = [
        Order(
            symbol=sym,
            side="buy",
            quantity_type="notional",
            quantity=DOLLARS_PER_NAME,
            order_type="market",
        )
        for sym in tradable[: min(NUM_NAMES, max_orders)]
    ]
    rationale = (
        f"Equal-dollar entry into {len(orders)} names." if orders else "No tradable names."
    )
    return Decision(orders=orders, confidence=0.8, rationale=rationale)


def main() -> int:
    api_key = os.environ.get("ATL_API_KEY")
    if not api_key:
        raise SystemExit("ATL_API_KEY is required")
    base_url = os.environ.get("ATL_BASE_URL", "http://127.0.0.1:8000")
    start = os.environ.get("ATL_START_DATE", "2026-04-15")
    end = os.environ.get("ATL_END_DATE", "2026-04-30")

    client = ATLClient(base_url=base_url, api_key=api_key)
    agent_version_id = resolve_agent_version_id(client)

    run = client.create_run(
        agent_version_id,
        environment_id="us-equity-hourly-v1",
        start_date=start,
        end_date=end,
    )
    print(f"Run: {run.id}\n")

    while True:
        step = client.get_next_step(run.id)
        if step.status == "completed":
            break
        if step.status == "loading":
            client.wait(2)
            continue
        if step.status != "awaiting_decision":
            client.wait(0.5)
            continue

        decision = decide(step)
        result = client.submit_decision(run.id, step.id, decision)
        if result.fills:
            for fill in result.fills:
                print(
                    f"  step {step.sequence:>2} FILL {fill['side'].upper():4} "
                    f"{fill['symbol']:5} x{fill['filled_quantity']} @ ${fill['fill_price']:.2f}"
                )
        for rej in result.rejections:
            print(f"  step {step.sequence:>2} REJECT {rej.get('order', {}).get('symbol')}: {rej.get('reason')}")

    result = client.get_run_result(run.id)
    print("\n=== Metrics ===")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
