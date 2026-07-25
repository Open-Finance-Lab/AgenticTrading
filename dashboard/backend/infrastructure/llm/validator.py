"""
LLM Trading Decision Validator

Provides strict validation for LLM trading responses:
- Enforces JSON-only responses (no tool_calls)
- Validates against trading schema
- Checks portfolio constraints
- Logs all decisions for audit trail

Usage:
    decision = validate_llm_response(raw_response, portfolio_state)
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, field_validator, ValidationError, ConfigDict

logger = logging.getLogger(__name__)

# Any C0 control char (incl. CR/LF) or DEL — the log-injection surface.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _safe_log_value(value: Any, limit: int = 500) -> str:
    """Neutralize untrusted (LLM/attacker-controlled) text before it reaches a log
    record. Strips CR/LF — the sanitizer CodeQL py/log-injection recognizes — plus
    other control chars, so a crafted response cannot forge or split log lines, and
    bounds length. Sanitizes only what is *logged*; never the value trading
    validation runs against, so the LLM safety boundary is unchanged.
    """
    text = str(value)[:limit].replace("\r", " ").replace("\n", " ")
    return _CONTROL_CHARS.sub(" ", text)


# DJIA 30 constituents. Source: S&P Dow Jones Indices, effective 2026-06-29
# (GOOGL replaced VZ). Canonical for ATL — the backtest script and the v2 API
# contract import this (guarded by tests/test_djia30_universe.py). Reconcile
# against the official index on change.
DJIA_30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GOOGL", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "WMT",
]

# Top 10 DJIA stocks (for 10-stock buy-and-hold mode)
TOP_10_STOCKS = ["AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "AXP", "HD", "DIS"]

# Ticker shape only — universe membership is enforced by callers that know the
# selected asset universe (default: DJIA_30). Mag7 / custom lists include names
# outside the Dow (e.g. TSLA, META).
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


class TradingAction(str, Enum):
    """Allowed trading actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


# Hard per-order share ceiling enforced by ``LLMTradingDecision``. Exposed as a
# module constant so upstream layers (e.g. the protocol Run service) can reject
# an over-size order per-order *before* it reaches this all-or-nothing batch
# validator, instead of letting it void an entire multi-order decision.
MAX_ORDER_SHARES = 10000


class LLMTradingDecision(BaseModel):
    """
    Strict schema for LLM trading responses.
    
    The LLM MUST respond with JSON matching this schema.
    Any attempt to call tools or functions will be rejected.
    """
    model_config = ConfigDict(use_enum_values=True)
    
    action: TradingAction
    symbol: str
    confidence: float
    reasoning: str
    position_size: int
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Normalize ticker and enforce shape (universe checked later)."""
        if not isinstance(v, str):
            raise ValueError(f"Symbol must be a string, got {type(v)}")
        symbol = v.strip().upper()
        if not _TICKER_RE.fullmatch(symbol):
            raise ValueError(f"Invalid symbol format: {v}")
        return symbol
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Confidence must be 0.0 to 1.0"""
        if not isinstance(v, (int, float)):
            raise ValueError(f"Confidence must be numeric, got {type(v)}")
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence {v} out of range [0.0, 1.0]")
        return float(v)
    
    @field_validator('reasoning')
    @classmethod
    def validate_reasoning(cls, v):
        """Reasoning must be string, max 500 chars"""
        if not isinstance(v, str):
            raise ValueError(f"Reasoning must be string, got {type(v)}")
        if len(v) > 500:
            raise ValueError(f"Reasoning too long: {len(v)} > 500 chars")
        if len(v) < 5:
            raise ValueError(f"Reasoning too short: {len(v)} < 5 chars")
        return v
    
    @field_validator('position_size')
    @classmethod
    def validate_position_size(cls, v):
        """Position size must be non-negative integer"""
        if not isinstance(v, int):
            raise ValueError(f"Position size must be integer, got {type(v)}")
        if v < 0:
            raise ValueError(f"Position size cannot be negative: {v}")
        if v > MAX_ORDER_SHARES:  # Reasonable max: no single position > 10k shares
            raise ValueError(f"Position size too large: {v} > {MAX_ORDER_SHARES}")
        return v
    
    @field_validator('stop_loss_price', mode='before')
    @classmethod
    def validate_stop_loss(cls, v):
        """Stop loss must be valid and positive"""
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise ValueError(f"Stop loss must be numeric, got {type(v)}")
        if v <= 0:
            raise ValueError(f"Stop loss must be positive: {v}")
        return float(v)
    
    @field_validator('take_profit_price', mode='before')
    @classmethod
    def validate_take_profit(cls, v):
        """Take profit must be valid and positive"""
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise ValueError(f"Take profit must be numeric, got {type(v)}")
        if v <= 0:
            raise ValueError(f"Take profit must be positive: {v}")
        return float(v)


class PortfolioConstraints(BaseModel):
    """Portfolio constraints to validate against"""
    cash_available: float
    max_position_size: int = 5000
    max_daily_trades: int = 20
    max_position_value: float = 50000
    min_confidence: float = 0.3  # Don't execute low-confidence decisions


def validate_llm_response(
    raw_response: str,
    portfolio_state: Dict[str, Any],
    current_prices: Dict[str, float],
    allowed_symbols: Optional[List[str]] = None,
) -> Optional[LLMTradingDecision]:
    """
    Validate LLM response and check portfolio constraints.
    
    Args:
        raw_response: Raw string from LLM
        portfolio_state: Current portfolio state
        current_prices: Current market prices {symbol: price}
        allowed_symbols: Tradeable universe (defaults to DJIA_30)
    
    Returns:
        LLMTradingDecision if valid, None if invalid (reason logged)
    
    Raises:
        ValidationError: If response is malformed
    """
    
    # =========================================================================
    # CRITICAL CHECK 1: Reject tool_calls or function_calls
    # =========================================================================
    if "tool_use" in raw_response or "tool_calls" in raw_response or \
       "function_calls" in raw_response or "invoke" in raw_response:
        logger.error(
            "🚨 SECURITY: LLM attempted tool calling! Rejecting response entirely.",
            extra={"raw_response_preview": _safe_log_value(raw_response, 200)}
        )
        return None
    
    # =========================================================================
    # CRITICAL CHECK 2: Parse JSON
    # =========================================================================
    try:
        json_data = json.loads(raw_response)
    except json.JSONDecodeError as e:
        logger.warning(
            "❌ Invalid JSON from LLM: %s",
            _safe_log_value(e, 200),
            extra={"raw_response_preview": _safe_log_value(raw_response, 200)}
        )
        return None
    
    # =========================================================================
    # CRITICAL CHECK 3: Schema validation
    # =========================================================================
    try:
        decision = LLMTradingDecision(**json_data)
    except ValidationError as e:
        logger.warning(
            "❌ Schema validation failed: %s",
            _safe_log_value(e, 500),
            extra={
                "errors": _safe_log_value(e.errors(), 500),
                "json_data": _safe_log_value(json_data, 500),
            }
        )
        return None

    allow = {
        str(s).strip().upper()
        for s in (allowed_symbols if allowed_symbols is not None else DJIA_30)
        if s
    }
    if decision.symbol not in allow:
        logger.warning(
            f"❌ Symbol {decision.symbol} not in allowed universe",
            extra={"symbol": decision.symbol, "allowed": sorted(allow)},
        )
        return None
    
    # =========================================================================
    # CHECK 4: Portfolio constraints
    # =========================================================================
    constraints = PortfolioConstraints(**portfolio_state.get("constraints", {}))
    
    # Check confidence threshold
    if decision.confidence < constraints.min_confidence:
        logger.info(
            f"⏭️ Decision confidence {decision.confidence} below minimum {constraints.min_confidence}. "
            f"Treating as HOLD.",
            extra={"decision": decision.model_dump()}
        )
        decision.action = TradingAction.HOLD
        decision.position_size = 0
        return decision
    
    # Check cash availability for BUY
    if decision.action == TradingAction.BUY:
        current_price = current_prices.get(decision.symbol, 0)
        total_cost = decision.position_size * current_price
        
        if total_cost > constraints.cash_available:
            logger.warning(
                f"❌ BUY {decision.symbol}: Insufficient cash. "
                f"Need ${total_cost:,.2f}, have ${constraints.cash_available:,.2f}",
                extra={"decision": decision.model_dump(), "cost": total_cost}
            )
            return None
        
        if decision.position_size > constraints.max_position_size:
            logger.warning(
                f"❌ BUY {decision.symbol}: Position size {decision.position_size} "
                f"exceeds max {constraints.max_position_size}",
                extra={"decision": decision.model_dump()}
            )
            return None
        
        if total_cost > constraints.max_position_value:
            logger.warning(
                f"❌ BUY {decision.symbol}: Position value ${total_cost:,.2f} "
                f"exceeds max ${constraints.max_position_value:,.2f}",
                extra={"decision": decision.model_dump()}
            )
            return None
    
    # Check that SELL has valid position
    if decision.action == TradingAction.SELL:
        positions = {p["symbol"]: p["shares"] for p in portfolio_state.get("positions", [])}
        if decision.symbol not in positions:
            logger.warning(
                f"❌ SELL {decision.symbol}: No position to sell",
                extra={"decision": decision.model_dump(), "positions": positions}
            )
            return None
        
        if decision.position_size > positions[decision.symbol]:
            logger.warning(
                f"❌ SELL {decision.symbol}: Trying to sell {decision.position_size} "
                f"shares, only have {positions[decision.symbol]}",
                extra={"decision": decision.model_dump()}
            )
            return None
    
    logger.info(
        f"✅ LLM decision validated: {decision.action.upper()} {decision.symbol} "
        f"{decision.position_size} @ confidence {decision.confidence:.0%}",
        extra={"decision": decision.model_dump()}
    )
    
    return decision


class LLMTradingDecisionBatch(BaseModel):
    """Batch trading decisions from an external agent."""

    actions: List[LLMTradingDecision]


def parse_actions_payload(payload: Dict[str, Any]) -> Tuple[Optional[List[LLMTradingDecision]], Optional[str]]:
    """
    Parse and validate an external agent decision payload.

    Returns:
        (list of decisions, None) on success, or (None, error_message) on failure.
    """
    if not isinstance(payload, dict):
        return None, "Payload must be a JSON object"

    actions_raw = payload.get("actions")
    if not isinstance(actions_raw, list):
        return None, 'Payload must include an "actions" array'

    decisions: List[LLMTradingDecision] = []
    for idx, item in enumerate(actions_raw):
        try:
            decisions.append(LLMTradingDecision(**item))
        except ValidationError as exc:
            return None, f"actions[{idx}]: {exc.errors()[0]['msg']}"

    return decisions, None


def actions_to_executable(
    decisions: List[LLMTradingDecision],
    *,
    cash: float,
    positions: Dict[str, int],
    current_prices: Dict[str, float],
    min_confidence: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Convert validated external decisions into backtest executable actions.

    Matches logic in backtest_hourly_agent.make_trading_decision_with_llm().
    """
    executable: List[Dict[str, Any]] = []

    for decision in decisions:
        symbol = decision.symbol
        action_type = decision.action
        confidence = decision.confidence
        reasoning = decision.reasoning

        if confidence < min_confidence:
            continue

        if action_type == TradingAction.HOLD:
            continue

        price = current_prices.get(symbol, 0)
        if price <= 0:
            continue

        if action_type == TradingAction.BUY:
            shares = decision.position_size
            if shares <= 0:
                continue
            cost = shares * price
            if cost <= cash:
                executable.append({
                    "symbol": symbol,
                    "action": "buy",
                    "shares": shares,
                    "reason": f"[External] {reasoning} (confidence: {confidence:.0%})",
                    "confidence": confidence,
                })

        elif action_type == TradingAction.SELL:
            if symbol in positions and positions[symbol] > 0:
                sell_shares = min(decision.position_size or positions[symbol], positions[symbol])
                if sell_shares > 0:
                    executable.append({
                        "symbol": symbol,
                        "action": "sell",
                        "shares": sell_shares,
                        "reason": f"[External] {reasoning} (confidence: {confidence:.0%})",
                        "confidence": confidence,
                    })

    return executable


def log_audit_trail(
    session_id: str,
    decision: LLMTradingDecision,
    llm_raw_response: str,
    execution_status: str,
    error_msg: Optional[str] = None
):
    """
    Log LLM decision to audit trail for compliance and debugging.
    
    Args:
        session_id: User session ID
        decision: Validated trading decision
        llm_raw_response: Raw response from LLM
        execution_status: "success", "rejected", "failed"
        error_msg: Optional error message
    """
    audit_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "decision": decision.model_dump() if decision else None,
        "execution_status": execution_status,
        "error": error_msg,
        "llm_response_preview": llm_raw_response[:300]
    }
    
    logger.info(
        f"📋 AUDIT: {execution_status.upper()} decision for session {session_id[:8]}",
        extra=audit_record
    )


# ============================================================================
# Safe Prompt Template (for integration)
# ============================================================================
# ============================================================================
# AGENT MODE PROMPTS
# ============================================================================
BUY_AND_HOLD_PROMPT = """You are a buy-and-hold backtest agent.

Return ONLY valid JSON. No markdown. No code fences. No explanations.
Goal:
- First hour: buy 10 stocks, about $10,000 per stock.
- Later hours: hold existing positions.
- Never sell.

Rules:
1. Use only market_snapshot and VALID SYMBOLS.
2. If current_holdings is empty, generate BUY actions.
3. If current_holdings is not empty, generate HOLD actions for existing positions.
4. Do not sell.
5. Do not return all HOLD when current_holdings is empty.
6. Do not change the JSON format.

Stocks to buy on first hour:
Top 10 DJIA stocks: AAPL, MSFT, JPM, V, JNJ, WMT, PG, AXP, HD, DIS

First-hour buy logic:
- allocation_per_stock = 100 (which is 1000 / 10)
- For each stock, find its current price in market_snapshot.
- position_size = int(10000 / current_price)
- If the stock is not in VALID SYMBOLS or price is missing, skip it.
- Confidence should be 0.95.
- Reasoning should be: "Buy-and-hold initial purchase - equal 10-stock allocation."

Later-hour hold logic:
- For each of the 10 holdings, return action="hold".
- position_size must be 0.
- Confidence should be 0.90.
- Reasoning should be: "Buy-and-hold: maintain position - 10-stock portfolio."
- Generate exactly 10 HOLD actions (one per stock).

MARKET SNAPSHOT:
{market_snapshot}

VALID SYMBOLS:
{valid_symbols}

Return ONLY this JSON format:

{{
  "actions": [
    {{
      "action": "buy|sell|hold",
      "symbol": "<DJIA symbol from VALID SYMBOLS>",
      "confidence": <float 0.0-1.0>,
      "reasoning": "<short reason, max 30 chars>",
      "position_size": <integer shares, 0 for hold>,
      "stop_loss_price": <float or null>,
      "take_profit_price": <float or null>
    }}
  ]
}}

Output ONLY valid JSON.
"""

# SAFE_TRADING_PROMPT = """You are testing a BUY-AND-HOLD strategy for multiple DJIA stocks.

# === STRATEGY ===
# 1. On FIRST hour: Equally buy the top 10 DJIA stocks (by signal strength)
#    - Divide $100,000 by 10 = $10,000 per stock
#    - Buy as many shares as possible for each
# 2. On ALL other hours: HOLD (do nothing)
# 3. Never sell during the period

# === CRITICAL CONSTRAINTS ===
# 1. You CANNOT access the internet, call APIs, or use web searches
# 2. You CANNOT use tools, functions, or execute code
# 3. Your response MUST be ONLY valid JSON (no other text, no explanations)

# === RESPONSE FORMAT (JSON ONLY) ===
# {{
#   "actions": [
#     {{
#       "action": "buy|hold",
#       "symbol": "<DJIA stock>",
#       "confidence": 1.0,
#       "reasoning": "Buy and hold DJIA stocks",
#       "position_size": <integer shares, or 0 for hold>,
#       "stop_loss_price": null,
#       "take_profit_price": null
#     }},
#     ...
#   ]
# }}

# === MARKET DATA ===
# {market_snapshot}

# === INSTRUCTIONS ===
# FIRST TRADE ONLY:
# - For each of the 10 stocks in top_signals:
#   - Allocate: $100,000 / 10 = $10,000 per stock
#   - Calculate shares: floor($10,000 / stock_price)
#   - Include in actions array as BUY

# ALL OTHER TRADES:
# - For all 10 stocks: return HOLD action with position_size=0

# Respond with ONLY valid JSON."""
SAFE_TRADING_PROMPT = """You are an active DJIA portfolio trading agent.

Goal:
Trade actively enough to beat passive baselines, but avoid random trades.
Use only the provided market_snapshot. Do not use internet, tools, APIs, or code.

Output rules:
- Return ONLY valid JSON.
- No markdown.
- No code fences.
- No explanations outside JSON.
- Use only symbols from VALID SYMBOLS.
- Only sell symbols that are currently owned.
- Keep reasoning short.

Trading style:
- Prefer BUY when a stock has bullish trend or momentum.
- Prefer HOLD when signals are mixed.
- Prefer SELL only when an owned stock clearly weakens.
- Do not return all HOLD if there is cash available and at least one reasonable bullish setup.
- Do not over-focus on RSI alone.

BUY logic:
Buy when at least 2 of these are true:
- price is above SMA20
- price is above SMA50
- SMA20 is above SMA50
- MACD is above signal
- RSI is between 35 and 70
- recent return or relative strength is positive
- price is recovering from oversold conditions

Strong BUY when at least 4 of the above are true.

Avoid BUY when:
- price is below SMA50 and MACD is bearish
- RSI is above 80
- the same symbol was bought very recently
- cash is too low

SELL logic:
Only sell owned stocks.

Sell when at least 2 of these are true:
- price is below SMA20
- price is below SMA50
- MACD is below signal
- RSI is above 75 and momentum is weakening
- recent return or relative strength is poor
- the position has a meaningful unrealized loss and trend is weak

Avoid SELL when:
- the stock is still above SMA20 and SMA50
- MACD is bullish
- the only issue is high RSI in a strong uptrend

HOLD logic:
Hold when:
- signals are mixed
- the stock is already owned and trend is still acceptable
- cash is limited
- recent_trades show the symbol was traded too recently

Capital deployment (IMPORTANT - avoid cash drag):
- Idle cash earns nothing. In a flat or rising market, sitting on a large cash pile guarantees you underperform fully-invested baselines.
- Target staying at least 80% invested whenever there are reasonable bullish or neutral setups available.
- Only hold a large cash balance (more than ~20%) when most stocks show clearly bearish trends (price below SMA50 and bearish MACD).
- If cash is above ~20% of total equity and any qualifying BUY setups exist, you MUST deploy cash into them this step.
- Do not hoard cash waiting for a perfect entry.

Activity rule:
- Prioritize the best opportunities from top_signals and current_holdings.
- It is fine to HOLD existing winners instead of churning - do not sell a strong uptrend just to trade.
- But do not leave qualifying capital idle: if you are under-invested, add or size up the strongest setups.
- Do not output more than 10 actions total.

Position sizing:
- Medium BUY: use about 10% of portfolio value.
- Strong BUY: use about 15% to 20% of portfolio value.
- Never use more than 25% of portfolio value on one new stock.
- Size positions so that, across the portfolio, you stay close to fully invested when setups justify it.
- position_size must be an integer share count.
- If price or cash is missing, use position_size 1 for BUY.
- For HOLD, position_size must be 0.
- For SELL, position_size should be the shares to sell if available; otherwise use 1.

Confidence:
- BUY confidence should usually be 0.65 to 0.90.
- SELL confidence should usually be 0.65 to 0.90.
- HOLD confidence should usually be 0.30 to 0.60.
- Use confidence above 0.85 only for very clear setups.

Constraints (fixed):
- Use ONLY the provided market_snapshot. No internet, tools, APIs, or code.
- Trade ONLY symbols listed in VALID SYMBOLS.

MARKET SNAPSHOT:
{market_snapshot}

VALID SYMBOLS:
{valid_symbols}

Return exactly this JSON shape:

{{
  "actions": [
    {{
      "action": "buy|sell|hold",
      "symbol": "<DJIA symbol>",
      "confidence": 0.75,
      "reasoning": "<short reason using trend, momentum, RSI, or risk>",
      "position_size": 1,
      "stop_loss_price": null,
      "take_profit_price": null
    }}
  ]
}}

Return ONLY valid JSON.
"""


# ============================================================================
# Custom (free-form) strategy prompt
# ============================================================================
# A free-form strategy from a caller REPLACES the baked-in SAFE_TRADING_PROMPT
# strategy body, but every run still needs a fixed scaffold so it keeps working:
# the execution context (market snapshot + valid-symbol allowlist) and the strict
# JSON ``actions`` output contract that the backtest parser/validator/executor
# depend on. We keep that fixed part as its own constant and CONCATENATE it after
# the user's prompt. Only this fixed scaffold goes through str.format, so any
# braces in the user's free-form prompt can never break formatting.
CUSTOM_STRATEGY_OUTPUT_CONTRACT = """

=== EXECUTION CONTRACT (fixed — always obey, even if it conflicts with the strategy above) ===
Run the strategy above as a DJIA portfolio trading agent on historical hourly data.
- Use ONLY the provided market_snapshot. No internet, tools, APIs, or code.
- Trade ONLY symbols listed in VALID SYMBOLS.
- Only SELL symbols that are currently owned (see current_holdings).
- position_size must be an integer share count; use 0 for hold.
- Output at most 10 actions.
- Return ONLY valid JSON. No markdown, no code fences, no text outside the JSON.

MARKET SNAPSHOT:
{market_snapshot}

VALID SYMBOLS:
{valid_symbols}

Return exactly this JSON shape:

{{
  "actions": [
    {{
      "action": "buy|sell|hold",
      "symbol": "<DJIA symbol>",
      "confidence": 0.75,
      "reasoning": "<short reason grounded in the strategy and the snapshot>",
      "position_size": 1,
      "stop_loss_price": null,
      "take_profit_price": null
    }}
  ]
}}

Return ONLY valid JSON.
"""


def _format_valid_symbols(allowed_symbols: Optional[List[str]] = None) -> str:
    symbols = allowed_symbols if allowed_symbols is not None else DJIA_30
    return ", ".join(str(s).strip().upper() for s in symbols if s)


def create_safe_prompt(
    market_snapshot: Dict[str, Any],
    allowed_symbols: Optional[List[str]] = None,
) -> str:
    """Create a safe prompt for LLM trading decision."""
    return SAFE_TRADING_PROMPT.format(
        market_snapshot=json.dumps(market_snapshot, indent=2),
        valid_symbols=_format_valid_symbols(allowed_symbols),
    )


def create_custom_prompt(
    market_snapshot: Dict[str, Any],
    strategy_prompt: str,
    allowed_symbols: Optional[List[str]] = None,
) -> str:
    """Concatenate a user's free-form strategy with the fixed execution contract.

    The free-form ``strategy_prompt`` is included verbatim and REPLACES the
    hardcoded SAFE_TRADING_PROMPT strategy body. A fixed scaffold
    (``CUSTOM_STRATEGY_OUTPUT_CONTRACT``) is appended to pin the market context,
    the valid-symbol allowlist, and the strict JSON ``actions`` output shape, so
    downstream parsing/validation/execution are unchanged regardless of what the
    user wrote. Only the fixed scaffold is formatted, so user braces are safe.
    """
    strategy = (strategy_prompt or "").strip()
    contract = CUSTOM_STRATEGY_OUTPUT_CONTRACT.format(
        market_snapshot=json.dumps(market_snapshot, indent=2),
        valid_symbols=_format_valid_symbols(allowed_symbols),
    )
    return f"=== USER STRATEGY ===\n{strategy}\n{contract}"


def create_prompt(
    market_snapshot: Dict[str, Any],
    mode: str = "safe_trading",
    custom_prompt: str | None = None,
    allowed_symbols: Optional[List[str]] = None,
) -> str:
    """
    Create a prompt for LLM trading decision based on mode.
    
    Args:
        market_snapshot: Market data snapshot
        mode: "safe_trading" or "buy_and_hold"
        custom_prompt: optional free-form strategy that REPLACES the built-in
            strategy body for this run (the market snapshot + JSON output
            contract are still enforced). When set, ``mode`` is ignored.
        allowed_symbols: Tradeable universe listed in VALID SYMBOLS (default DJIA_30).
    
    Returns:
        Formatted prompt string
    """
    if custom_prompt and custom_prompt.strip():
        return create_custom_prompt(
            market_snapshot, custom_prompt, allowed_symbols=allowed_symbols
        )
    if mode == "buy_and_hold":
        return BUY_AND_HOLD_PROMPT.format(
            market_snapshot=json.dumps(market_snapshot, indent=2),
            valid_symbols=_format_valid_symbols(allowed_symbols),
        )
    return create_safe_prompt(market_snapshot, allowed_symbols=allowed_symbols)
