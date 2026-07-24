"""Agent-Environment Protocol primitives (v1).

Pure helpers shared by the Run API and the orchestration service: protocol
version, the standard error envelope, request models for the Decision schema,
and translation between protocol ``orders`` and the engine's legacy ``actions``
payload. Execution/validation itself is reused from ``llm_validator`` and the
backtest engine — nothing here re-implements trading logic.

Moved verbatim (Phase 3B2) from ``dashboard/backend/protocol.py``; the original
module was removed in Phase 4A. Public classes, constants, schemas, signatures,
and behavior are unchanged; only the module location moved.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "1.0"

VALID_SIDES = {"buy", "sell"}
VALID_QUANTITY_TYPES = {"shares", "notional", "weight"}
VALID_ORDER_TYPES = {"market"}


class ProtocolError(Exception):
    """Raised for protocol-level failures; mapped to an HTTP error envelope."""

    def __init__(self, code: str, message: str, status_code: int = 400,
                 details: Any = None, headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details
        # Response headers (e.g. Retry-After on 429) — forwarded to the
        # HTTPException by the router's _handle_protocol_error.
        self.headers = headers or {}

    def to_body(self) -> Dict[str, Any]:
        return error_body(self.code, self.message, self.details)


def error_body(code: str, message: str, details: Any = None) -> Dict[str, Any]:
    """Standard protocol error envelope."""
    body: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "error": {"code": code, "message": message},
    }
    if details is not None:
        body["error"]["details"] = details
    return body


class OrderIn(BaseModel):
    symbol: str
    side: str = Field(description="buy or sell")
    quantity_type: str = Field(default="shares")
    # Bounded + finite: reject NaN/Infinity (which otherwise reach
    # ``resolve_order_quantity`` and blow up ``int(inf)`` with a 500) and cap
    # absurd magnitudes at the schema boundary so a single order can never
    # overflow the engine's integer share math.
    quantity: float = Field(ge=0, lt=1e12, allow_inf_nan=False)
    order_type: str = Field(default="market")


class DecisionIn(BaseModel):
    protocol_version: Optional[str] = None
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    idempotency_key: str = Field(min_length=1, max_length=200)
    orders: List[OrderIn] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rationale: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None


def _clean_rationale(rationale: Optional[str]) -> str:
    text = (rationale or "").strip()
    if len(text) < 5:
        text = "external agent decision"
    return text[:500]


def resolve_order_quantity(
    order: OrderIn,
    *,
    price: float,
    equity: float,
) -> Tuple[int, Optional[str]]:
    """Resolve an order's share count from its quantity_type.

    Returns (shares, rejection_reason). ``shares`` is 0 when the order cannot
    be sized (rejection_reason explains why).
    """
    if order.quantity_type not in VALID_QUANTITY_TYPES:
        return 0, f"unsupported_quantity_type:{order.quantity_type}"
    if order.order_type not in VALID_ORDER_TYPES:
        return 0, f"unsupported_order_type:{order.order_type}"

    if order.quantity_type == "shares":
        return int(order.quantity), None

    if price <= 0:
        return 0, "missing_price"

    if order.quantity_type == "notional":
        return int(order.quantity // price), None

    # weight: fraction of current equity
    if order.quantity_type == "weight":
        notional = max(0.0, equity) * float(order.quantity)
        return int(notional // price), None

    return 0, "unsupported_quantity_type"


def order_to_action(
    order: OrderIn,
    *,
    shares: int,
    confidence: float,
    rationale: Optional[str],
) -> Dict[str, Any]:
    """Translate a protocol order into a legacy engine action dict."""
    return {
        "action": order.side.lower(),
        "symbol": order.symbol,
        "confidence": confidence,
        "reasoning": _clean_rationale(rationale),
        "position_size": int(shares),
        "stop_loss_price": None,
        "take_profit_price": None,
    }
