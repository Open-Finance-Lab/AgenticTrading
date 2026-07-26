"""Robinhood live trading orchestration + audit logging."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dashboard.backend.api import robinhood_oauth
from dashboard.backend.domain.brokers.repository import broker_store
from dashboard.backend.infrastructure.brokers.robinhood_mcp import (
    RobinhoodMCPClient,
    extract_buying_power,
    extract_portfolio_value,
)
from dashboard.backend.api.v2.models import validate_actions
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

AUDIT_DIR = REPO_ROOT / "dashboard" / "storage" / "audit" / "robinhood"
MAX_ORDER_USD = float(os.getenv("ROBINHOOD_MAX_ORDER_USD", "25"))


def execute_enabled() -> bool:
    return os.getenv("ROBINHOOD_EXECUTE", "false").strip().lower() in {"1", "true", "yes"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _audit(event: str, payload: Dict[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"ts": _utcnow_iso(), "event": event, **payload}
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = AUDIT_DIR / f"audit_{day}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def _ensure_access_token(user_id: int) -> Dict[str, Any]:
    tokens = broker_store.get_tokens(user_id)
    if not tokens or not tokens.get("access_token"):
        raise ValueError("robinhood_not_connected")
    return tokens


async def _refresh_if_needed(user_id: int, tokens: Dict[str, Any]) -> str:
    access = tokens["access_token"]
    refresh = tokens.get("refresh_token")
    client_id = tokens.get("client_id")
    if not refresh or not client_id:
        return access
    try:
        refreshed = await asyncio.to_thread(
            robinhood_oauth.refresh_access_token,
            refresh_token=refresh,
            client_id=client_id,
        )
        new_access = refreshed.get("access_token")
        if new_access:
            broker_store.update_tokens(
                user_id,
                access_token=new_access,
                refresh_token=refreshed.get("refresh_token") or refresh,
                token_expires_at=robinhood_oauth.token_expires_at_iso(refreshed.get("expires_in")),
            )
            return new_access
    except Exception:
        logger.exception("Robinhood token refresh failed for user %s", user_id)
    return access


async def get_connection_status(user_id: int, *, include_portfolio: bool = False) -> Dict[str, Any]:
    """Return Robinhood link state. Portfolio/MCP calls are optional and time-bounded."""
    base = {
        "broker": "robinhood",
        "execute_enabled": execute_enabled(),
        "max_order_usd": MAX_ORDER_USD,
    }
    public = broker_store.get_public(user_id)
    if not public:
        return {**base, "connected": False}

    result = {
        **base,
        **public,
        "connected": True,
    }
    if not include_portfolio:
        return result

    try:
        tokens = _ensure_access_token(user_id)
        access = await _refresh_if_needed(user_id, tokens)
        client = RobinhoodMCPClient(access)
        portfolio = await client.get_portfolio()
        result["buying_power"] = extract_buying_power(portfolio)
        result["portfolio_value"] = extract_portfolio_value(portfolio)
    except Exception as exc:
        logger.warning("Robinhood portfolio snapshot failed for user %s: %s", user_id, exc)
        result["portfolio_error"] = str(exc)[:200]
    return result


def _portfolio_state(positions: Any, buying_power: Optional[float]) -> Dict[str, Any]:
    holdings: Dict[str, float] = {}
    if isinstance(positions, list):
        for row in positions:
            if not isinstance(row, dict):
                continue
            symbol = (row.get("symbol") or row.get("instrument_symbol") or "").upper()
            qty = row.get("quantity") or row.get("qty") or row.get("shares")
            if symbol and qty is not None:
                try:
                    holdings[symbol] = float(qty)
                except (TypeError, ValueError):
                    continue
    elif isinstance(positions, dict):
        rows = positions.get("positions") or positions.get("results") or []
        return _portfolio_state(rows, buying_power)
    cash = float(buying_power or 0.0)
    return {"cash": cash, "holdings": holdings, "positions": holdings}


def _build_market_snapshot(quotes: Any, symbols: List[str]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {"symbols": {}}
    rows: List[Any]
    if isinstance(quotes, list):
        rows = quotes
    elif isinstance(quotes, dict):
        rows = quotes.get("quotes") or quotes.get("results") or []
    else:
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = (row.get("symbol") or "").upper()
        if not symbol:
            continue
        price = row.get("last_trade_price") or row.get("price") or row.get("mark_price")
        snapshot["symbols"][symbol] = {"price": price, "raw": row}
    for symbol in symbols:
        snapshot["symbols"].setdefault(symbol, {"price": None})
    return snapshot


def _actions_to_robinhood_orders(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    orders: List[Dict[str, Any]] = []
    for action in actions:
        act = str(action.get("action") or "").lower()
        symbol = str(action.get("symbol") or "").upper()
        shares = action.get("shares")
        if shares is None:
            shares = action.get("position_size")
        if act == "hold" or not symbol:
            continue
        if act not in {"buy", "sell"}:
            continue
        try:
            qty = float(shares)
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "side": act,
                "quantity": qty,
                "order_type": "market",
                "time_in_force": "gfd",
            }
        )
    return orders


def _cap_orders_by_usd(
    orders: List[Dict[str, Any]],
    quotes: Dict[str, Any],
    max_usd: float,
) -> List[Dict[str, Any]]:
    capped: List[Dict[str, Any]] = []
    symbol_prices = (quotes or {}).get("symbols") or {}
    for order in orders:
        if order.get("side") != "buy":
            capped.append(order)
            continue
        symbol = order["symbol"]
        price_raw = (symbol_prices.get(symbol) or {}).get("price")
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            capped.append(order)
            continue
        if price <= 0:
            continue
        max_shares = max_usd / price
        qty = min(float(order["quantity"]), max_shares, float(MAX_ORDER_SHARES))
        if qty >= 0.0001:
            patched = dict(order)
            patched["quantity"] = round(qty, 4)
            capped.append(patched)
    return capped


async def _llm_decision(
    *,
    agent: Dict[str, Any],
    market_snapshot: Dict[str, Any],
    portfolio_state: Dict[str, Any],
) -> Dict[str, Any]:
    client = make_llm_client()
    model = agent.get("model_name") or default_model_name()
    if client is None:
        return {"actions": [{"action": "hold", "symbol": sym, "shares": 0} for sym in DJIA_30[:5]]}

    pipeline = agent.get("pipeline") or []
    instruction = ""
    if pipeline:
        instruction = "\n\n".join(
            f"[{step.get('label', 'step')}]\n{step.get('prompt', '')}" for step in pipeline if isinstance(step, dict)
        )
    user_payload = {
        "instruction": instruction or "Trade conservatively across DJIA names.",
        "portfolio": portfolio_state,
        "market": market_snapshot,
        "allowed_symbols": DJIA_30,
    }
    def _call() -> Dict[str, Any]:
        response = request_trading_decision(
            client,
            prompt=json.dumps(user_payload, default=str),
            model=model,
        )
        text = extract_response_text(response)
        parsed = parse_llm_response(text)
        return parsed or {"actions": []}

    return await asyncio.to_thread(_call)


async def run_live_for_agent(
    *,
    user_id: int,
    agent: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    if not agent.get("live_trading_enabled"):
        raise ValueError("live_trading_not_enabled")

    run_id = f"rh_live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    tokens = _ensure_access_token(user_id)
    access = await _refresh_if_needed(user_id, tokens)
    client = RobinhoodMCPClient(access)

    portfolio = await client.get_portfolio()
    positions = await client.get_equity_positions()
    buying_power = extract_buying_power(portfolio)
    portfolio_state = _portfolio_state(positions, buying_power)

    symbols = sorted(set(DJIA_30[:10]) | set(portfolio_state.get("holdings", {}).keys()))
    quotes = await client.get_equity_quotes(symbols)
    market_snapshot = _build_market_snapshot(quotes, symbols)

    _audit(
        "context_snapshot",
        {
            "run_id": run_id,
            "user_id": user_id,
            "agent_id": agent.get("agent_id"),
            "portfolio": portfolio_state,
            "market": market_snapshot,
        },
    )

    llm_result = await _llm_decision(
        agent=agent,
        market_snapshot=market_snapshot,
        portfolio_state=portfolio_state,
    )
    raw_actions = llm_result.get("actions") or []
    actions, rejected = validate_actions(raw_actions)

    _audit(
        "decision_validated",
        {
            "run_id": run_id,
            "agent_id": agent.get("agent_id"),
            "actions": actions,
            "rejected": rejected,
        },
    )

    orders = _cap_orders_by_usd(
        _actions_to_robinhood_orders(actions),
        market_snapshot,
        MAX_ORDER_USD,
    )

    reviews: List[Dict[str, Any]] = []
    executions: List[Dict[str, Any]] = []

    for order in orders:
        review = await client.review_equity_order(order)
        reviews.append({"order": order, "review": review})
        _audit("pre_trade_review", {"run_id": run_id, "order": order, "review": review})

        should_execute = execute_enabled() and not dry_run
        if not should_execute:
            executions.append({"order": order, "status": "skipped", "reason": "dry_run_or_execute_disabled"})
            continue

        result = await client.place_equity_order(order)
        executions.append({"order": order, "status": "submitted", "result": result})
        _audit("order_placed", {"run_id": run_id, "order": order, "result": result})

    return {
        "run_id": run_id,
        "status": "completed",
        "dry_run": dry_run or not execute_enabled(),
        "execute_enabled": execute_enabled(),
        "portfolio_value": extract_portfolio_value(portfolio),
        "buying_power": buying_power,
        "decision": {"actions": actions},
        "orders_reviewed": reviews,
        "executions": executions,
    }
