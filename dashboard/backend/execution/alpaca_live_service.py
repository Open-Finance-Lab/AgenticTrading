"""Alpaca live trading orchestration + audit logging.

This module is the **live-money** path for Alpaca: every order it emits can
reach a real brokerage account. It deliberately mirrors the risk-gate /
two-gate design already shipped for Robinhood in
``dashboard.backend.execution.robinhood_live_service`` (off by default, a
dry-run review mode, a per-order USD cap, a share cap, no shorting) rather
than inventing a looser one. The risk-gate and status-mapping helpers here
are pure functions so they can be unit tested without network or broker
access, same as the Robinhood ones.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dashboard.backend.infrastructure.brokers.alpaca_live import (
    AlpacaLiveCredentialsError,
    AlpacaLiveTradingClient,
)
from dashboard.backend.infrastructure.llm.backtest_harness import (
    default_model_name,
    extract_response_text,
    make_llm_client,
    parse_llm_response,
    request_trading_decision,
)
from dashboard.backend.infrastructure.llm.validator import DJIA_30, MAX_ORDER_SHARES
from dashboard.backend.paths import REPO_ROOT

logger = logging.getLogger(__name__)

AUDIT_DIR = REPO_ROOT / "dashboard" / "storage" / "audit" / "alpaca_live"

#: Fallback notional cap, in USD, per order. Used whenever
#: ``ALPACA_MAX_ORDER_USD`` is missing, unparseable or non-positive.
DEFAULT_MAX_ORDER_USD = 25.0

#: Orders below this share count are dropped rather than sent to the broker.
MIN_ORDER_QUANTITY = 0.0001

#: A live run is single-flight for the whole process — this is a local,
#: single-account setup, not the multi-tenant Robinhood path, so one lock is
#: enough (no per-user keying).
_run_lock = asyncio.Lock()


def execute_enabled() -> bool:
    """True when live order placement is armed (read fresh on every call, like
    the kill switch it doubles as)."""
    return os.getenv("ALPACA_LIVE_EXECUTE", "false").strip().lower() in {"1", "true", "yes", "on"}


def max_order_usd() -> float:
    """Per-order notional cap in USD, read fresh from the environment. A missing,
    unparseable, non-finite or non-positive value clamps to
    :data:`DEFAULT_MAX_ORDER_USD` rather than disabling the cap."""
    raw = os.getenv("ALPACA_MAX_ORDER_USD")
    if raw is None or not str(raw).strip():
        return DEFAULT_MAX_ORDER_USD
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_ORDER_USD
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_MAX_ORDER_USD
    return value


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _audit_sync(event: str, payload: Dict[str, Any]) -> None:
    """Append one audit record. Never raises — an unwritable audit dir must not
    abort a live run mid-flight."""
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        record = {"ts": _utcnow_iso(), "event": event, **payload}
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = AUDIT_DIR / f"audit_{day}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.exception("Alpaca live audit write failed for event %s", event)


async def _audit(event: str, payload: Dict[str, Any]) -> None:
    await asyncio.to_thread(_audit_sync, event, payload)


def _floor_quantity(qty: float) -> float:
    """Floor to 4 decimal places. Rounding *down* matters: round-half-up could
    push the resulting notional back above the USD cap that produced it."""
    try:
        value = float(qty)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return math.floor(value * 10000.0) / 10000.0


def _rejection(order: Dict[str, Any], reason: str, detail: str) -> Dict[str, Any]:
    return {"order": order, "reason": reason, "detail": detail}


def risk_gate_orders(
    orders: List[Dict[str, Any]],
    prices: Dict[str, float],
    holdings: Dict[str, float],
    max_usd: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Clamp or reject every order, on **both** sides. Pure function.

    Rules (same as the Robinhood risk gate):
      * no usable price -> reject (``no_quote``); a missing quote must never
        become a pass-through.
      * buys and sells are both clamped to ``max_usd / price`` and to
        ``MAX_ORDER_SHARES``.
      * sells additionally clamp to the held quantity and are rejected outright
        (``no_position``) when nothing is held — never open a short.
      * a residual quantity below ``MIN_ORDER_QUANTITY`` -> ``below_min_quantity``.

    Returns ``(accepted_orders, rejections)``.
    """
    accepted: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []

    try:
        cap_usd = float(max_usd)
    except (TypeError, ValueError):
        cap_usd = DEFAULT_MAX_ORDER_USD
    if not math.isfinite(cap_usd) or cap_usd <= 0:
        cap_usd = DEFAULT_MAX_ORDER_USD

    for order in orders or []:
        symbol = str(order.get("symbol") or "").upper()
        side = str(order.get("side") or "").lower()
        try:
            requested = float(order.get("quantity"))
        except (TypeError, ValueError):
            requested = 0.0

        price = prices.get(symbol)
        if not price or not math.isfinite(price) or price <= 0:
            rejections.append(
                _rejection(
                    order,
                    "no_quote",
                    f"No usable quote for {symbol or '?'}; cannot size the order against the USD cap.",
                )
            )
            continue

        limits = [requested, cap_usd / price, float(MAX_ORDER_SHARES)]
        if side == "sell":
            available = float(holdings.get(symbol, 0) or 0)
            if available <= 0:
                rejections.append(
                    _rejection(
                        order,
                        "no_position",
                        f"No {symbol} position held; refusing to open a short.",
                    )
                )
                continue
            limits.append(available)
        elif side != "buy":
            rejections.append(_rejection(order, "bad_side", f"Unknown side {side!r}."))
            continue

        qty = _floor_quantity(min(limits))
        if qty < MIN_ORDER_QUANTITY:
            rejections.append(
                _rejection(
                    order,
                    "below_min_quantity",
                    (
                        f"{symbol} {side} sized down to {qty} shares at ${price} "
                        f"(cap ${cap_usd}); below the {MIN_ORDER_QUANTITY} minimum."
                    ),
                )
            )
            continue

        gated = dict(order)
        gated["symbol"] = symbol
        gated["side"] = side
        gated["quantity"] = qty
        gated["notional_usd"] = round(qty * price, 2)
        accepted.append(gated)

    return accepted, rejections


def _actions_to_orders(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    orders: List[Dict[str, Any]] = []
    for action in actions:
        act = str(action.get("action") or "").lower()
        symbol = str(action.get("symbol") or "").upper()
        shares = action.get("shares")
        if shares is None:
            shares = action.get("position_size")
        if act == "hold" or not symbol or act not in {"buy", "sell"}:
            continue
        try:
            qty = float(shares)
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        orders.append({"symbol": symbol, "side": act, "quantity": qty})
    return orders


async def _llm_decision(
    *, model_name: Optional[str], instruction: str, portfolio: Dict[str, Any], prices: Dict[str, float]
) -> Dict[str, Any]:
    """Ask the configured model for actions. Raises ``ValueError("llm_unavailable")``
    when no LLM client is configured or the response has no usable structure —
    a keyless deployment must never fall back to silent holds, which would be
    indistinguishable from a deliberate all-hold decision."""
    client = make_llm_client()
    if client is None:
        raise ValueError("llm_unavailable")
    model = model_name or default_model_name()

    user_payload = {
        "instruction": instruction or "Trade conservatively across DJIA names.",
        "portfolio": portfolio,
        "prices": prices,
        "allowed_symbols": list(DJIA_30),
    }

    def _call() -> Dict[str, Any]:
        response = request_trading_decision(client, prompt=json.dumps(user_payload, default=str), model=model)
        text = extract_response_text(response)
        parsed = parse_llm_response(text)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("actions"), list):
            raise ValueError("llm_unavailable")
        return parsed

    return await asyncio.to_thread(_call)


async def run_live_for_agent(
    *,
    instruction: str,
    model_name: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Run one live decision cycle against the real Alpaca account.

    ``dry_run=True`` (the default) always forces a review-only cycle, even if
    ``ALPACA_LIVE_EXECUTE`` is set. Real orders require **both**
    ``dry_run=False`` on the call **and** ``ALPACA_LIVE_EXECUTE=true`` in the
    environment — same two-gate pattern as the Robinhood path.
    """
    if _run_lock.locked():
        raise ValueError("live_run_in_progress")

    async with _run_lock:
        return await _execute_live_run(
            instruction=instruction, model_name=model_name, symbols=symbols, dry_run=dry_run
        )


async def _execute_live_run(
    *, instruction: str, model_name: Optional[str], symbols: Optional[List[str]], dry_run: bool
) -> Dict[str, Any]:
    run_id = f"alpaca_live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    try:
        client = AlpacaLiveTradingClient()
    except AlpacaLiveCredentialsError as exc:
        await _audit("credentials_missing", {"run_id": run_id, "error": str(exc)})
        raise

    account = await asyncio.to_thread(client.get_account)
    if account is None:
        raise ValueError("alpaca_account_unavailable")
    holdings = await asyncio.to_thread(client.get_positions)

    universe = sorted(set(symbols or DJIA_30) | set(holdings.keys()))
    prices = await asyncio.to_thread(client.get_quotes, universe)

    portfolio = {"cash": account["cash"], "buying_power": account["buying_power"], "holdings": holdings}

    await _audit(
        "context_snapshot",
        {"run_id": run_id, "portfolio": portfolio, "prices": prices, "universe": universe},
    )

    try:
        llm_result = await _llm_decision(
            model_name=model_name, instruction=instruction, portfolio=portfolio, prices=prices
        )
    except Exception as exc:
        await _audit("decision_failed", {"run_id": run_id, "error": str(exc)[:200]})
        raise

    actions = llm_result.get("actions") or []
    await _audit("decision", {"run_id": run_id, "actions": actions})

    cap_usd = max_order_usd()
    orders, rejections = risk_gate_orders(_actions_to_orders(actions), prices, holdings, cap_usd)
    if rejections:
        await _audit("orders_rejected", {"run_id": run_id, "max_order_usd": cap_usd, "rejections": rejections})

    executions: List[Dict[str, Any]] = []
    should_execute = execute_enabled() and not dry_run

    for order in orders:
        if not should_execute:
            executions.append({"order": order, "status": "skipped", "reason": "dry_run_or_execute_disabled"})
            continue
        try:
            result = await asyncio.to_thread(
                client.submit_market_order, order["symbol"], order["quantity"], order["side"]
            )
            executions.append({"order": order, "status": result.status or "submitted", "result": result.raw})
            await _audit(
                "order_placed",
                {"run_id": run_id, "order": order, "status": result.status, "result": result.raw},
            )
        except Exception as exc:
            detail = str(exc)[:300]
            logger.exception("Alpaca live order placement failed for %s", order.get("symbol"))
            executions.append({"order": order, "status": "failed", "reason": "place_order_failed", "error": detail})
            await _audit("order_failed", {"run_id": run_id, "order": order, "error": detail})

    return {
        "run_id": run_id,
        "status": "completed",
        "dry_run": dry_run or not execute_enabled(),
        "execute_enabled": execute_enabled(),
        "max_order_usd": cap_usd,
        "account": account,
        "decision": {"actions": actions},
        "orders_reviewed": orders,
        "rejected_orders": rejections,
        "executions": executions,
    }
