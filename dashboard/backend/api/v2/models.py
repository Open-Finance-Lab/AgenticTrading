"""Typed v2 wire contract (spec §5). Source of the auto-generated OpenAPI doc."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from dashboard.backend.infrastructure.llm.validator import DJIA_30, normalize_symbol

SCHEMA_VERSION = "2.0"
UNIVERSE_KEY = "djia_30"
UNIVERSE: List[str] = list(DJIA_30)
_UNIVERSE_SET = set(UNIVERSE)
# Every DJIA-30 ticker is at most 5 characters; this leaves room for padding.
# It bounds what a rejection echoes into the ack, its idempotency row and the
# trace, which an agent could otherwise fill with an arbitrarily long string
# on every submission.
SYMBOL_MAX_LENGTH = 32
_SYMBOL_RULE = (
    "Surrounding ASCII whitespace is trimmed and ASCII letters are uppercased "
    "(normalize_symbol), so ' aapl ' is AAPL. A non-ASCII value is never "
    "case-folded and so never matches a ticker.")


# --- Context envelope ------------------------------------------------------

class NewsSentimentEntry(BaseModel):
    sentiment: str = Field(pattern="^(bullish|bearish|neutral)$")
    score: float = Field(ge=-1.0, le=1.0)
    headline: str
    source: str
    url: str
    age_hours: float = Field(ge=0.0)
    n_articles: int = Field(ge=0)
    rationale: Optional[str] = None  # producer's one-line directional reasoning (additive, 2026-07-13)


class PortfolioState(BaseModel):
    cash: float
    positions_value: float
    total_equity: float
    num_positions: int


class HoldingItem(BaseModel):
    shares: float
    entry_price: float
    current_price: float
    position_value: float
    pnl_pct: float


class SignalItem(BaseModel):
    price: float
    rsi: float
    macd: float
    macd_signal: float
    sma20: float
    sma50: float
    bb_upper: float
    bb_lower: float


class ContextEnvelope(BaseModel):
    schema_version: str
    run_id: str
    mode: str
    step_index: int
    total_steps: int
    timestamp: Optional[str] = None
    loop: str  # "lockstep" | "realtime"
    decision_deadline_at: Optional[str] = None
    decision_timeout_seconds: Optional[int] = None
    status: str  # loading | waiting_decision | completed | closed | failed
    universe: List[str]
    portfolio: Optional[PortfolioState] = None
    current_holdings: Dict[str, HoldingItem] = Field(default_factory=dict)
    recent_trades: List[Dict[str, Any]] = Field(default_factory=list)
    top_signals: Dict[str, SignalItem] = Field(default_factory=dict)
    news_sentiment: Dict[str, NewsSentimentEntry] = Field(default_factory=dict)
    news_overview: Optional[str] = None
    decision_format: Optional[Dict[str, Any]] = None


# --- Decision request ------------------------------------------------------

class ActionItem(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    action: str = Field(pattern="^(buy|sell|hold)$")
    symbol: str = Field(
        max_length=SYMBOL_MAX_LENGTH,
        description=f"DJIA-30 ticker, checked after normalizing. {_SYMBOL_RULE}")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=5, max_length=500)
    position_size: int = Field(ge=0, le=10000)
    stop_loss_price: Optional[float] = Field(default=None, gt=0)
    take_profit_price: Optional[float] = Field(default=None, gt=0)

    @field_validator("symbol")
    @classmethod
    def _symbol_in_universe(cls, v: str) -> str:
        # Normalize before the membership check, with the helper every
        # decision path shares: "aapl" or "AAPL " is AAPL, not a universe
        # violation.
        symbol = normalize_symbol(v)
        if symbol not in _UNIVERSE_SET:
            raise ValueError(f"universe_violation: {v} not in DJIA-30")
        return symbol


class DecisionRequest(BaseModel):
    idempotency_key: str = Field(min_length=1)
    # Raw action dicts (each should match ActionItem). They are validated per-item
    # at the v2 boundary via validate_actions() so that a single malformed action is
    # dropped *with a reason* (spec §5.3 partial execution) rather than 422-ing the
    # whole submission. The action shape stays discoverable via GET /api/v2/schema.
    actions: List[Dict[str, Any]]


def validate_actions(
    raw_actions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Split raw actions into (valid ActionItem dicts, rejected {symbol, reason}).

    Realizes spec §5.3: schema-invalid actions are dropped with a reason instead of
    failing the whole request. Keeping this at the boundary means every backend
    receives only valid actions and the rejected list is uniform across backends.
    """
    valid: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    for raw in raw_actions:
        try:
            valid.append(ActionItem(**raw).model_dump())
        except ValidationError as exc:
            msg = exc.errors()[0].get("msg", "validation_failed")
            reason = "universe_violation" if "universe_violation" in msg else "validation_failed"
            rejected.append({"symbol": _rejected_symbol(raw.get("symbol", "?")),
                             "reason": reason})
    return valid, rejected


def _rejected_symbol(raw_symbol: Any) -> str:
    """The symbol a rejection echoes: normalized exactly like an executed one,
    so an agent matches every ack entry to what it sent with one rule, and
    bounded to SYMBOL_MAX_LENGTH."""
    text = raw_symbol if isinstance(raw_symbol, str) else str(raw_symbol)
    return normalize_symbol(text)[:SYMBOL_MAX_LENGTH]


# --- Submit ack ------------------------------------------------------------

class ExecutedItem(BaseModel):
    action: str
    symbol: str = Field(description=f"The submitted symbol, normalized. {_SYMBOL_RULE}")
    shares: float
    price: Optional[float] = None


class RejectedItem(BaseModel):
    symbol: str = Field(description=(
        f"The submitted symbol, normalized like an executed one and cut to "
        f"{SYMBOL_MAX_LENGTH} characters. {_SYMBOL_RULE}"))
    reason: str


class SubmitAck(BaseModel):
    accepted: bool
    executed: List[ExecutedItem] = Field(default_factory=list)
    rejected: List[RejectedItem] = Field(default_factory=list)
    decision_source: str  # external_agent | timeout_hold | validation_hold
    next_step: Optional[int] = None
    status: str
    run_id: str
    metrics: Optional[Dict[str, Any]] = None


# --- Result ----------------------------------------------------------------

class RunManifest(BaseModel):
    agent_name: str
    model_name: str
    mode: str
    universe: str = UNIVERSE_KEY
    start_date: str
    end_date: str
    decision_timeout_seconds: int
    schema_version: str = SCHEMA_VERSION
    news_sentiment_source: Optional[str] = None


class ResultEnvelope(BaseModel):
    run: Dict[str, Any]
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    manifest: Optional[RunManifest] = None


# --- Error -----------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    retryable: bool = False


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
