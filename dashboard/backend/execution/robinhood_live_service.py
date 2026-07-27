"""Robinhood live trading orchestration + audit logging.

This module is the **live-money** path: every order it emits can reach a real
brokerage account. The risk gate (:func:`_risk_gate_orders`), the pre-trade
review gate (:func:`_review_blocks_order`) and the execution-status reader
(:func:`_execution_status`) are deliberately pure functions so they can be unit
tested without network, broker or database access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from dashboard.backend.api.v2.models import validate_actions
from dashboard.backend.domain.brokers.repository import broker_store
from dashboard.backend.infrastructure.brokers import robinhood_oauth
from dashboard.backend.infrastructure.brokers.robinhood_mcp import (
    RobinhoodMCPClient,
    extract_buying_power,
    extract_portfolio_value,
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

AUDIT_DIR = REPO_ROOT / "dashboard" / "storage" / "audit" / "robinhood"

#: Fallback notional cap, in USD, per order. Used whenever
#: ``ROBINHOOD_MAX_ORDER_USD`` is missing, unparseable or non-positive.
DEFAULT_MAX_ORDER_USD = 25.0

#: ``RobinhoodMCPClient.get_equity_quotes`` truncates to ``symbols[:20]``, so the
#: universe has to be requested in batches or symbols silently lose their quote.
QUOTE_BATCH_SIZE = 20

#: Orders below this share count are dropped rather than sent to the broker.
MIN_ORDER_QUANTITY = 0.0001

#: Skip an OAuth refresh while the stored token still has this much headroom.
TOKEN_REFRESH_HEADROOM_SECONDS = 120

#: Idempotency replay window and cache size for ``run_live_for_agent``.
IDEMPOTENCY_TTL_SECONDS = 900.0
IDEMPOTENCY_MAX_ENTRIES = 256

_REVOKE_TIMEOUT_SECONDS = 20.0

_NEGATIVE_STATUS_VALUES = {"rejected", "failed", "denied", "error", "blocked"}
_ERROR_KEYS = ("error", "errors", "rejection_reason")
_APPROVAL_KEYS = ("approved", "allowed", "can_place", "is_valid")
_ORDER_ID_KEYS = ("id", "order_id", "orderId", "order_number", "client_order_id")

#: One lock per user id: a user may only have one live run in flight at a time.
_user_locks: Dict[int, asyncio.Lock] = {}

#: (user_id, agent_id, idempotency_key) -> (monotonic_ts, result dict)
_idempotency_cache: "OrderedDict[Tuple[int, str, str], Tuple[float, Dict[str, Any]]]" = OrderedDict()


def execute_enabled() -> bool:
    """True when live order placement is armed (read fresh on every call)."""
    return os.getenv("ROBINHOOD_EXECUTE", "false").strip().lower() in {"1", "true", "yes"}


def max_order_usd() -> float:
    """Per-order notional cap in USD, read fresh from the environment.

    Read per call (not frozen at import) so tightening the cap takes effect
    without a restart — the kill switch already behaves that way. A missing,
    unparseable, non-finite or non-positive value clamps to
    :data:`DEFAULT_MAX_ORDER_USD` rather than disabling the cap.
    """
    raw = os.getenv("ROBINHOOD_MAX_ORDER_USD")
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
        logger.exception("Robinhood audit write failed for event %s", event)


async def _audit(event: str, payload: Dict[str, Any]) -> None:
    """Async wrapper: the audit write is blocking file I/O, so keep it off the loop."""
    await asyncio.to_thread(_audit_sync, event, payload)


def _ensure_access_token(user_id: int) -> Dict[str, Any]:
    tokens = broker_store.get_tokens(user_id)
    if not tokens or not tokens.get("access_token"):
        raise ValueError("robinhood_not_connected")
    return tokens


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp; naive values are treated as UTC."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _token_still_valid(
    tokens: Dict[str, Any],
    *,
    headroom_seconds: int = TOKEN_REFRESH_HEADROOM_SECONDS,
) -> bool:
    """True when ``token_expires_at`` is parseable and still has headroom left."""
    expires_at = _parse_iso_timestamp((tokens or {}).get("token_expires_at"))
    if expires_at is None:
        return False
    return expires_at - datetime.now(timezone.utc) > timedelta(seconds=headroom_seconds)


async def _refresh_if_needed(user_id: int, tokens: Dict[str, Any]) -> str:
    access = tokens["access_token"]
    refresh = tokens.get("refresh_token")
    client_id = tokens.get("client_id")
    if not refresh or not client_id:
        return access
    if _token_still_valid(tokens):
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
        "max_order_usd": max_order_usd(),
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


async def revoke_connection(user_id: int) -> bool:
    """Best-effort upstream revocation of a user's Robinhood tokens.

    Dropping the local row leaves the broker-side grant alive, so the disconnect
    route calls this first. Returns True only when the provider accepted a
    revocation; never raises.
    """
    try:
        tokens = broker_store.get_tokens(user_id)
    except Exception:
        logger.exception("Robinhood token lookup failed during revoke for user %s", user_id)
        return False
    if not tokens or not tokens.get("access_token"):
        return False
    try:
        return await asyncio.to_thread(_revoke_tokens_sync, tokens)
    except Exception:
        logger.exception("Robinhood token revocation failed for user %s", user_id)
        return False


def _revoke_tokens_sync(tokens: Dict[str, Any]) -> bool:
    """POST the stored tokens to the provider's RFC-7009 revocation endpoint."""
    try:
        metadata = robinhood_oauth.fetch_metadata()
    except Exception:
        logger.warning("Robinhood OAuth metadata unavailable; cannot revoke", exc_info=True)
        return False
    endpoint = (metadata or {}).get("revocation_endpoint")
    if not endpoint:
        logger.warning("Robinhood OAuth metadata advertises no revocation_endpoint")
        return False

    client_id = tokens.get("client_id")
    revoked = False
    candidates = (
        (tokens.get("refresh_token"), "refresh_token"),
        (tokens.get("access_token"), "access_token"),
    )
    for token, hint in candidates:
        if not token:
            continue
        payload = {"token": token, "token_type_hint": hint}
        if client_id:
            payload["client_id"] = client_id
        try:
            with httpx.Client(timeout=_REVOKE_TIMEOUT_SECONDS) as http_client:
                response = http_client.post(str(endpoint), data=payload)
        except Exception:
            logger.warning("Robinhood %s revocation request failed", hint, exc_info=True)
            continue
        if response.status_code < 400:
            revoked = True
        else:
            logger.warning(
                "Robinhood %s revocation returned HTTP %s", hint, response.status_code
            )
    return revoked


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


def _chunk_symbols(symbols: List[str], size: int = QUOTE_BATCH_SIZE) -> List[List[str]]:
    """Split a symbol list into batches the MCP quote tool will not truncate."""
    if size <= 0:
        size = QUOTE_BATCH_SIZE
    ordered = list(symbols or [])
    return [ordered[i : i + size] for i in range(0, len(ordered), size)]


async def _fetch_market_snapshot(client: Any, symbols: List[str]) -> Dict[str, Any]:
    """Quote the *whole* tradable universe, in ``QUOTE_BATCH_SIZE`` chunks.

    ``get_equity_quotes`` silently drops everything past the 20th symbol, so a
    single call for the 30-name DJIA universe would leave 10 symbols priceless —
    and a priceless symbol used to bypass the notional cap entirely.
    """
    merged: Dict[str, Any] = {"symbols": {}}
    errors: List[Dict[str, str]] = []
    for batch in _chunk_symbols(symbols):
        try:
            quotes = await client.get_equity_quotes(batch)
        except Exception as exc:
            logger.exception("Robinhood quote batch failed for %s", ",".join(batch))
            errors.append({"symbols": ",".join(batch), "error": str(exc)[:200]})
            quotes = None
        merged["symbols"].update(_build_market_snapshot(quotes, batch).get("symbols") or {})
    if errors:
        merged["quote_errors"] = errors
    return merged


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


def _usable_price(entry: Any) -> Optional[float]:
    """Return a strictly positive float price, or None when unusable."""
    if not isinstance(entry, dict):
        return None
    try:
        price = float(entry.get("price"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return price


def _floor_quantity(qty: float) -> float:
    """Floor to 4 decimal places.

    Rounding *down* matters: round-half-up could push the resulting notional back
    above the USD cap that produced the quantity in the first place.
    """
    try:
        value = float(qty)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return math.floor(value * 10000.0) / 10000.0


def _rejection(order: Dict[str, Any], reason: str, detail: str) -> Dict[str, Any]:
    return {"order": order, "reason": reason, "detail": detail}


def _risk_gate_orders(
    orders: List[Dict[str, Any]],
    market_snapshot: Dict[str, Any],
    portfolio_state: Dict[str, Any],
    max_usd: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Clamp or reject every order, on **both** sides. Pure function.

    Rules:
      * no usable price -> reject (``no_quote``); a missing quote must never
        become a pass-through, which is how a 10,000-share order once shipped
        under a nominal 25 USD cap.
      * buys and sells are both clamped to ``max_usd / price`` and to
        ``MAX_ORDER_SHARES``.
      * sells additionally clamp to the held quantity and are rejected outright
        (``no_position``) when nothing is held — never open a short.
      * a residual quantity below ``MIN_ORDER_QUANTITY`` -> ``below_min_quantity``.

    Returns ``(accepted_orders, rejections)`` where each rejection is
    ``{"order": ..., "reason": ..., "detail": ...}``.
    """
    accepted: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    symbol_prices = (market_snapshot or {}).get("symbols") or {}
    holdings = (portfolio_state or {}).get("holdings") or {}

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

        price = _usable_price(symbol_prices.get(symbol))
        if price is None:
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
            try:
                available = float(holdings.get(symbol, 0) or 0)
            except (TypeError, ValueError):
                available = 0.0
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


def _has_negative_signal(payload: Dict[str, Any]) -> bool:
    """True when a broker payload carries an explicit error/refusal signal."""
    for key in _ERROR_KEYS:
        if payload.get(key):
            return True
    for key in _APPROVAL_KEYS:
        if payload.get(key) is False:
            return True
    for key in ("status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip().lower() in _NEGATIVE_STATUS_VALUES:
            return True
    return False


def _review_blocks_order(review: Any) -> Tuple[bool, str]:
    """Decide whether a pre-trade review forbids placement.

    Returns ``(blocked, reason)``. The review used to be logged and ignored, so
    an explicit broker refusal did nothing to stop the order that followed.
    """
    if review is None:
        return True, "review_empty"
    if not isinstance(review, dict):
        return False, ""
    if _has_negative_signal(review):
        return True, "review_rejected"
    return False, ""


def _looks_like_placed_order(payload: Dict[str, Any]) -> bool:
    for key in _ORDER_ID_KEYS:
        if payload.get(key):
            return True
    for key in ("status", "state"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return True
    for key in _APPROVAL_KEYS:
        if payload.get(key) is True:
            return True
    return False


def _execution_status(result: Any) -> str:
    """Map a broker placement response to ``rejected`` / ``submitted`` / ``unknown``.

    Recording every placement as ``submitted`` regardless of the response made
    the audit trail unusable — a refusal looked identical to a fill.
    """
    if not isinstance(result, dict):
        return "unknown"
    if _has_negative_signal(result):
        return "rejected"
    if _looks_like_placed_order(result):
        return "submitted"
    for key in ("order", "result", "data"):
        nested = result.get(key)
        if isinstance(nested, dict):
            nested_status = _execution_status(nested)
            if nested_status != "unknown":
                return nested_status
    return "unknown"


async def _llm_decision(
    *,
    agent: Dict[str, Any],
    market_snapshot: Dict[str, Any],
    portfolio_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Ask the agent's model for actions.

    Raises ``ValueError("llm_unavailable")`` when no LLM client is configured or
    the response has no usable structure. Fabricating hold actions here made a
    keyless deployment indistinguishable from a deliberate all-hold run.
    """
    client = make_llm_client()
    if client is None:
        raise ValueError("llm_unavailable")
    model = agent.get("model_name") or default_model_name()

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
        "allowed_symbols": list(DJIA_30),
    }

    def _call() -> Dict[str, Any]:
        response = request_trading_decision(
            client,
            prompt=json.dumps(user_payload, default=str),
            model=model,
        )
        text = extract_response_text(response)
        parsed = parse_llm_response(text)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("actions"), list):
            raise ValueError("llm_unavailable")
        return parsed

    return await asyncio.to_thread(_call)


def _user_lock(user_id: int) -> asyncio.Lock:
    lock = _user_locks.get(int(user_id))
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[int(user_id)] = lock
    return lock


def _idempotency_cache_key(
    user_id: int, agent_id: Any, idempotency_key: str
) -> Tuple[int, str, str]:
    return (int(user_id), str(agent_id or ""), str(idempotency_key))


def _prune_idempotency_cache(now: Optional[float] = None) -> None:
    current = time.monotonic() if now is None else now
    stale = [
        key
        for key, (stamped_at, _) in _idempotency_cache.items()
        if current - stamped_at > IDEMPOTENCY_TTL_SECONDS
    ]
    for key in stale:
        _idempotency_cache.pop(key, None)
    while len(_idempotency_cache) > IDEMPOTENCY_MAX_ENTRIES:
        _idempotency_cache.popitem(last=False)


def _idempotent_replay(key: Tuple[int, str, str]) -> Optional[Dict[str, Any]]:
    _prune_idempotency_cache()
    entry = _idempotency_cache.get(key)
    if entry is None:
        return None
    return {**entry[1], "idempotent_replay": True}


def _remember_idempotent_result(key: Tuple[int, str, str], result: Dict[str, Any]) -> None:
    _idempotency_cache[key] = (time.monotonic(), dict(result))
    _idempotency_cache.move_to_end(key)
    _prune_idempotency_cache()


async def run_live_for_agent(
    *,
    user_id: int,
    agent: Dict[str, Any],
    dry_run: bool = False,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one live decision cycle for ``agent`` on behalf of ``user_id``.

    Concurrent runs for the same user raise ``ValueError("live_run_in_progress")``,
    and a repeated ``idempotency_key`` replays the previous result instead of
    placing the orders a second time.
    """
    if not agent.get("live_trading_enabled"):
        raise ValueError("live_trading_not_enabled")

    cache_key: Optional[Tuple[int, str, str]] = None
    if idempotency_key:
        cache_key = _idempotency_cache_key(user_id, agent.get("agent_id"), idempotency_key)
        replay = _idempotent_replay(cache_key)
        if replay is not None:
            return replay

    lock = _user_lock(user_id)
    if lock.locked():
        raise ValueError("live_run_in_progress")

    async with lock:
        if cache_key is not None:
            replay = _idempotent_replay(cache_key)
            if replay is not None:
                return replay
        result = await _execute_live_run(user_id=user_id, agent=agent, dry_run=dry_run)
        if cache_key is not None:
            _remember_idempotent_result(cache_key, result)
        return result


async def _execute_live_run(
    *,
    user_id: int,
    agent: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"rh_live_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    agent_id = agent.get("agent_id")
    tokens = _ensure_access_token(user_id)
    access = await _refresh_if_needed(user_id, tokens)
    client = RobinhoodMCPClient(access)

    portfolio = await client.get_portfolio()
    positions = await client.get_equity_positions()
    buying_power = extract_buying_power(portfolio)
    portfolio_state = _portfolio_state(positions, buying_power)

    # Quote exactly what the model is allowed to trade (plus anything held), so a
    # symbol can never reach the risk gate without a price.
    universe = sorted(set(DJIA_30) | set(portfolio_state.get("holdings", {}).keys()))
    market_snapshot = await _fetch_market_snapshot(client, universe)

    await _audit(
        "context_snapshot",
        {
            "run_id": run_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "portfolio": portfolio_state,
            "market": market_snapshot,
        },
    )

    try:
        llm_result = await _llm_decision(
            agent=agent,
            market_snapshot=market_snapshot,
            portfolio_state=portfolio_state,
        )
    except Exception as exc:
        await _audit(
            "decision_failed",
            {"run_id": run_id, "agent_id": agent_id, "error": str(exc)[:200]},
        )
        raise

    raw_actions = llm_result.get("actions") or []
    actions, rejected = validate_actions(raw_actions)

    await _audit(
        "decision_validated",
        {
            "run_id": run_id,
            "agent_id": agent_id,
            "actions": actions,
            "rejected": rejected,
        },
    )

    cap_usd = max_order_usd()
    orders, order_rejections = _risk_gate_orders(
        _actions_to_robinhood_orders(actions),
        market_snapshot,
        portfolio_state,
        cap_usd,
    )
    if order_rejections:
        await _audit(
            "orders_rejected",
            {
                "run_id": run_id,
                "agent_id": agent_id,
                "max_order_usd": cap_usd,
                "rejections": order_rejections,
            },
        )

    reviews: List[Dict[str, Any]] = []
    executions: List[Dict[str, Any]] = []

    for order in orders:
        try:
            review = await client.review_equity_order(order)
        except Exception as exc:
            detail = str(exc)[:200]
            logger.warning("Robinhood pre-trade review failed for %s: %s", order.get("symbol"), detail)
            reviews.append({"order": order, "review": None, "error": detail})
            executions.append(
                {"order": order, "status": "skipped", "reason": "review_failed", "error": detail}
            )
            await _audit(
                "pre_trade_review_failed",
                {"run_id": run_id, "order": order, "error": detail},
            )
            continue

        reviews.append({"order": order, "review": review})
        await _audit("pre_trade_review", {"run_id": run_id, "order": order, "review": review})

        blocked, block_reason = _review_blocks_order(review)
        if blocked:
            executions.append(
                {"order": order, "status": "skipped", "reason": block_reason, "review": review}
            )
            await _audit(
                "order_blocked",
                {"run_id": run_id, "order": order, "reason": block_reason, "review": review},
            )
            continue

        should_execute = execute_enabled() and not dry_run
        if not should_execute:
            executions.append({"order": order, "status": "skipped", "reason": "dry_run_or_execute_disabled"})
            continue

        try:
            result = await client.place_equity_order(order)
        except Exception as exc:
            detail = str(exc)[:200]
            logger.exception("Robinhood order placement failed for %s", order.get("symbol"))
            executions.append(
                {"order": order, "status": "failed", "reason": "place_order_failed", "error": detail}
            )
            await _audit("order_failed", {"run_id": run_id, "order": order, "error": detail})
            continue

        status = _execution_status(result)
        executions.append({"order": order, "status": status, "result": result})
        await _audit(
            "order_placed",
            {"run_id": run_id, "order": order, "status": status, "result": result},
        )

    return {
        "run_id": run_id,
        "status": "completed",
        "dry_run": dry_run or not execute_enabled(),
        "execute_enabled": execute_enabled(),
        "max_order_usd": cap_usd,
        "portfolio_value": extract_portfolio_value(portfolio),
        "buying_power": buying_power,
        "decision": {"actions": actions},
        "rejected_actions": rejected,
        "orders_reviewed": reviews,
        "executions": executions,
        "rejected_orders": order_rejections,
    }
