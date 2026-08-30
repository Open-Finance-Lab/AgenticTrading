"""Backtesting portfolio-manager orchestration facade.

Moved (Phase 2C3) from ``dashboard/scripts/backtest_hourly_agent.py`` during the
domain-layer extraction. ``PortfolioManager`` coordinates the backtest agent:
portfolio state/valuation, deterministic reference-agent decisions, the LLM
decision workflow, in-memory execution, equity history, and token counters. The
lower-level logic already lives under ``dashboard.backend.domain.trading`` and
``dashboard.backend.infrastructure.llm``; this class delegates to it.

Most of the class is a mechanical move (only the imports became canonical), and
the legacy script re-exports this exact class object so ``bha.PortfolioManager``
and existing subclasses keep working unchanged. **The trading logic is not fully
unchanged, though:** the ``safe_trading`` candidate selection in
``make_trading_decision_with_llm`` no longer ranks the top-10 candidates by RSI
extremity (``|RSI - 50|``, a mean-reversion heuristic) — it ranks the top-12 by a
multi-factor trend/momentum score *and always appends current holdings*. That
ranking change was made separately from (and independently of) this file's
domain-layer move, so backtests from before it are **not** directly comparable
with current ones (see the inline comment on that branch for the rationale).

This module is domain-level orchestration: it must NOT import dashboard scripts,
``HourlyBacktester``, FastAPI routers, the database singleton, or Alpaca clients.
"""

import json
import math
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional

import pandas as pd

from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
from dashboard.backend.infrastructure.llm.validator import (
    DJIA_30,
    MAX_ORDER_SHARES,
    create_prompt,
)
from dashboard.backend.domain.trading.portfolio import (
    append_equity_record as _append_equity_record,
    build_portfolio_state as _build_portfolio_state,
    get_equity_curve as _get_equity_curve,
)
from dashboard.backend.domain.trading.execution import (
    execute_actions as _execute_actions,
)
from dashboard.backend.domain.backtesting.reference_agent import (
    make_rule_based_decision as _make_rule_based_decision,
)
from dashboard.backend.infrastructure.llm.backtest_harness import (
    HAS_ANTHROPIC,
    extract_response_text as _extract_response_text,
    extract_token_usage as _extract_token_usage,
    parse_llm_response as _parse_llm_response,
    request_trading_decision as _request_trading_decision,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import (
    RECOVERY_MAX_OUTPUT_TOKENS,
    response_text_or_none,
    run_pipeline_decision,
    truncation_reason,
)


class LLMDecisionError(RuntimeError):
    """Raised when an explicitly required LLM cannot drive a decision step."""


# A strict-LLM run may absorb this fraction of its steps as unusable model
# responses before it aborts. One truncated response should not discard hours
# of a multi-hour backtest, but the surviving curve still has to be honestly
# publishable, so this stays well inside the leaderboard's H6 coverage floor
# (``domain/leaderboard/service.py::MIN_LLM_DECISION_COVERAGE``, 0.95 — it
# cannot be imported here: that module pulls in the database singleton, which
# this module's docstring forbids). ``test_strict_llm_budget_clears_h6``
# asserts the two stay consistent.
STRICT_LLM_MAX_FALLBACK_RATIO = 0.02


class PortfolioManager:
    """Manages portfolio with hourly trading decisions based on indicators."""
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        allowed_symbols: Optional[List[str]] = None,
        t_plus_one_enabled: bool = False,
        lot_size: int = 1,
        transaction_cost_profile=None,
        market_rule_calendar=None,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: num_shares}
        self.entry_prices = {}  # {symbol: entry_price}
        self.trades = []
        self.order_events = []
        # {(symbol, side, reason, trading_date): record} — the same
        # bounded-by-construction shape as t1_deferrals below, pointing at the
        # order_events entry to bump instead of re-appending. Owned here, not
        # by the executor, because it must survive across per-step calls.
        self._order_event_repeats = {}
        self.t_plus_one_enabled = t_plus_one_enabled
        self.lot_size = lot_size
        self.transaction_cost_profile = transaction_cost_profile
        self.market_rule_calendar = market_rule_calendar
        self.transaction_cost_totals = {
            "gross_value": 0.0,
            "slippage_amount": 0.0,
            "commission": 0.0,
            "stamp_duty": 0.0,
            "transfer_fee": 0.0,
            "total_fees": 0.0,
        }
        self.available_positions = {}
        self.frozen_lots = {}
        self.rejected_orders = []
        # {(symbol, trading_date): record} — one entry per symbol-day on which
        # T+1 shrank an intended exit. Deduped by key rather than appended per
        # bar, so it is bounded by symbols x trading days by construction.
        self.t1_deferrals = {}
        self.equity_history = []
        # Real LLM token usage (server-side calls report actual counts)
        self.llm_calls = 0        # billed API calls (any response with usage)
        self.llm_decisions = 0    # steps the model actually drove (H6 coverage)
        self.input_tokens = 0
        self.output_tokens = 0
        # Strict-LLM tolerance. The runner sets ``strict_llm_total_steps`` once
        # it knows the run length; left None the budget is 0, so an unusable
        # response is fatal on the first strike (the conservative default for
        # callers that never declare a length).
        self.strict_llm_total_steps: Optional[int] = None
        self.strict_llm_fallbacks = 0
        # Latest decision-pipeline step outputs (for daily post-trade analysis).
        self.last_pipeline_step_outputs = []
        # Tradeable universe for this run (defaults to DJIA_30).
        symbols = allowed_symbols if allowed_symbols is not None else DJIA_30
        self.allowed_symbols = [str(s).strip().upper() for s in symbols if s]
        self._allowed_set = set(self.allowed_symbols)
    @property
    def sellable_positions(self) -> Optional[Mapping]:
        """The T+1 sellable balance, or ``None`` when settlement is immediate.

        ``None`` rather than ``self.positions`` on purpose: it is the sentinel
        every downstream helper uses to keep the pre-T+1 schema and behavior
        byte-identical for non-A-share runs.

        Returned as a read-only view. It reads like a query but would otherwise
        hand out the live ledger, so a consumer could rebalance the portfolio
        through it by accident. The view still tracks later mutations; execution
        writes through ``self.available_positions`` directly.
        """
        if not self.t_plus_one_enabled:
            return None
        return MappingProxyType(self.available_positions)

    def sellable_shares(self, symbol: str) -> float:
        """Shares of ``symbol`` an order submitted right now could actually fill."""
        held = self.positions.get(symbol, 0)
        if not self.t_plus_one_enabled:
            return held
        return min(self.available_positions.get(symbol, 0), held)

    def record_t1_deferral(
        self,
        symbol: str,
        requested_shares: float,
        sellable_shares: float,
    ) -> None:
        """Note that T+1 shrank an intended exit, once per symbol-trading-day.

        The trading date is read off the symbol's own frozen lots rather than a
        clock: a lot is only still frozen on the date it was bought (the release
        sweep moves every earlier lot out), so ``max(buy_date)`` *is* today
        whenever anything is blocking. That keeps this callable from the
        rule-based path, which is handed no timestamp.

        Keeping the largest deferral per symbol-day rather than the first means
        the record reflects the worst the constraint bound that day.
        """
        deferred = requested_shares - sellable_shares
        if deferred <= 0:
            return
        lots = self.frozen_lots.get(symbol) or []
        trading_date = max((lot["buy_date"] for lot in lots), default=None)
        if trading_date is None:
            return
        key = (symbol, trading_date)
        prior = self.t1_deferrals.get(key)
        if prior is None or deferred > prior["deferred_shares"]:
            self.t1_deferrals[key] = {
                "date": trading_date,
                "symbol": symbol,
                "requested_shares": requested_shares,
                "sellable_shares": sellable_shares,
                "deferred_shares": deferred,
            }

    def get_portfolio_state(self, market_data: Dict[str, pd.Series], price_cache: Dict = None, timestamp = None) -> Dict:
        """Get current portfolio state with market indicators.

        Uses real data for signals, forward-filled prices for valuation.
        """
        return _build_portfolio_state(
            self.cash,
            self.positions,
            self.entry_prices,
            market_data,
            price_cache,
            timestamp,
            self.sellable_positions,
        )

    def make_trading_decision(self, portfolio_state: Dict) -> Dict:
        """
        Agent makes trading decisions based on technical indicators.
        
        Rules:
        - BUY: RSI < 30 AND Price < SMA20 (oversold + downtrend)
        - SELL: RSI > 70 OR Price > SMA50 + 2% (overbought)
        - HOLD: Otherwise
        
        Returns:
            {"actions": [{"symbol": "AAPL", "action": "buy", "shares": 10}, ...]}
        """
        decision = _make_rule_based_decision(
            portfolio_state=portfolio_state,
            positions=self.positions,
            cash=self.cash,
            available_positions=self.sellable_positions,
            lot_size=self.lot_size,
        )
        # Drain into the run-level ledger and pop, so the returned shape stays
        # exactly {"actions": [...]} for every existing caller.
        for deferral in decision.pop("t1_deferrals", ()):
            self.record_t1_deferral(
                deferral["symbol"],
                deferral["requested_shares"],
                deferral["sellable_shares"],
            )
        return decision
    
    def make_trading_decision_with_llm(
        self,
        portfolio_state: Dict,
        llm_client,
        mode: str = "safe_trading",
        model: str = None,
        strategy_prompt: str = None,
        pipeline: List[Dict] = None,
        temperature: Optional[float] = None,
        market_context: Optional[Dict] = None,
        strict_llm: bool = False,
    ) -> Dict:
        """
        Make trading decisions using Claude LLM with technical indicators.
        
        The LLM receives:
        - All technical indicators (RSI, MACD, Bollinger Bands, SMAs)
        - Current portfolio state
        - Recent trade history (last 24 hours) for context and memory
        - Clear instructions on how to interpret signals
        
        Args:
            portfolio_state: Current portfolio state with market signals
            llm_client: Anthropic client instance
            mode: "safe_trading" (risk management) or "buy_and_hold" (debug mode)
            temperature: Optional model sampling temperature
        
        Returns:
            {"actions": [list of trading actions]}
        """
        unified_client = bool(getattr(llm_client, "execution_service", None))
        if (not llm_client) or (not HAS_ANTHROPIC and not unified_client):
            if strict_llm:
                raise LLMDecisionError("Required LLM client is not available")
            print("\u26a0️  LLM client not available, using rule-based fallback")
            return self.make_trading_decision(portfolio_state)
        
        try:
            # ================================================================
            # STEP 1: Create prompt with all technical indicators
            # ================================================================
            # Convert timestamp to ISO format string (handle pandas Timestamp)
            timestamp = portfolio_state.get("timestamp", datetime.now())
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
            
            # Extract current holdings for LLM decision-making
            holdings = {}
            for position in portfolio_state["positions"]:
                holding = {
                    "shares": position["shares"],
                    "entry_price": round(position["entry_price"], 2),
                    "current_price": round(position["current_price"], 2),
                    "position_value": round(position["position_value"], 2),
                    "pnl_pct": round(position["pnl_pct"], 2)
                }
                # Present only on T+1 markets, so non-A-share prompts are
                # byte-identical. Without forwarding it the model cannot see the
                # settlement rule at all -- its order would still be capped
                # downstream, but it would reason as though it could exit in
                # full and never learn otherwise.
                if "sellable_shares" in position:
                    holding["sellable_shares"] = position["sellable_shares"]
                holdings[position["symbol"]] = holding
            
            # Extract recent trade history (last 24 hours) for LLM memory
            # This prevents LLM from re-entering stocks too soon
            recent_trades = []
            cutoff_time = timestamp - timedelta(hours=24)
            for trade in self.trades:
                if trade["timestamp"] > cutoff_time:
                    recent_trades.append({
                        "symbol": trade["symbol"],
                        "side": trade["side"],
                        "shares": trade["shares"],
                        "price": round(float(trade["price"]), 2),
                        "timestamp": trade["timestamp"].isoformat() if hasattr(trade["timestamp"], 'isoformat') else str(trade["timestamp"])
                    })
            
            market_snapshot = {
                "timestamp": timestamp_str,
                "portfolio": {
                    "cash": round(portfolio_state["cash"], 2),
                    "positions_value": round(portfolio_state["positions_value"], 2),
                    "total_equity": round(portfolio_state["total_equity"], 2),
                    "num_positions": len(portfolio_state["positions"])
                },
                "current_holdings": holdings,  # What we currently own
                "recent_trades": recent_trades,  # Last 24h of trades (memory)
                "top_signals": {}
            }
            if market_context:
                market_snapshot["market"] = dict(market_context)
            
            # Add market signals to snapshot
            signals = portfolio_state["market_signals"]
            
            # For buy-and-hold mode, offer the full selected universe
            if mode == "buy_and_hold":
                symbols_to_include = [s for s in self.allowed_symbols if s in signals]
            else:
                # For safe_trading, rank by trend/momentum (NOT RSI extremity).
                # Ranking by |RSI-50| seeds a mean-reversion bias (fade winners,
                # buy losers) which loses in trending markets. Instead score each
                # name by trend confluence so the model is offered genuine momentum
                # opportunities, and ALWAYS include current holdings so it can
                # actively manage / exit weak positions.
                def _trend_score(sig: Dict) -> float:
                    price = float(sig.get("price", 0) or 0)
                    sma20 = float(sig.get("sma20", 0) or 0)
                    sma50 = float(sig.get("sma50", 0) or 0)
                    macd = float(sig.get("macd", 0) or 0)
                    macd_sig = float(sig.get("macd_signal", 0) or 0)
                    rsi = float(sig.get("rsi", 50) or 50)
                    score = 0.0
                    if sma20 and price > sma20:
                        score += 1.0
                    if sma50 and price > sma50:
                        score += 1.0
                    if sma20 and sma50 and sma20 > sma50:
                        score += 1.0
                    if macd > macd_sig:
                        score += 1.0
                    # Reward healthy (not overheated) momentum; penalize extreme RSI
                    if 45 <= rsi <= 70:
                        score += 0.5
                    elif rsi > 80:
                        score -= 0.5
                    # Continuous tiebreak: distance above the 50-day trend line
                    if sma50:
                        score += max(min((price / sma50) - 1.0, 0.25), -0.25)
                    return score

                trend_sorted = sorted(
                    signals.items(),
                    key=lambda kv: _trend_score(kv[1]),
                    reverse=True,
                )
                symbols_to_include = [sym for sym, _ in trend_sorted[:12]]
                # Guarantee every currently-held symbol is visible to the model
                for sym in holdings:
                    if sym in signals and sym not in symbols_to_include:
                        symbols_to_include.append(sym)
            
            for symbol in symbols_to_include:
                signal = signals[symbol]
                
                # Extract values, allowing zero values (they're still valid prices)
                rsi = float(signal.get("rsi", 50)) if pd.notna(signal.get("rsi")) else 50.0
                macd = float(signal.get("macd", 0)) if pd.notna(signal.get("macd")) else 0.0
                macd_sig = float(signal.get("macd_signal", 0)) if pd.notna(signal.get("macd_signal")) else 0.0
                sma20 = float(signal.get("sma20", 0)) if pd.notna(signal.get("sma20")) else 0.0
                sma50 = float(signal.get("sma50", 0)) if pd.notna(signal.get("sma50")) else 0.0
                bb_upper = float(signal.get("bb_upper", 0)) if pd.notna(signal.get("bb_upper")) else 0.0
                bb_lower = float(signal.get("bb_lower", 0)) if pd.notna(signal.get("bb_lower")) else 0.0
                price = float(signal.get("price", 0)) if pd.notna(signal.get("price")) else 0.0
                
                # Always include these stocks with their price (critical for LLM calculation)
                market_snapshot["top_signals"][symbol] = {
                    "price": price,
                    "rsi": rsi,
                    "macd": macd,
                    "macd_signal": macd_sig,
                    "sma20": sma20,
                    "sma50": sma50,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                }
            
            # Ensure market_snapshot is fully JSON-serializable before sending
            try:
                json.dumps(market_snapshot)  # Verify it's serializable
            except TypeError as e:
                if strict_llm:
                    raise LLMDecisionError(
                        "LLM market snapshot could not be serialized"
                    ) from e
                print(f"   ⚠️  Market snapshot serialization error: {e}")
                print(f"   Falling back to rule-based logic")
                return self.make_trading_decision(portfolio_state)
            
            # DEBUG: Show what's in market_snapshot for buy-and-hold mode
            if mode == "buy_and_hold" and not self.positions:
                print(f"\n   DEBUG market_snapshot:")
                print(f"     Cash: ${market_snapshot['portfolio']['cash']}")
                print(f"     Top signals count: {len(market_snapshot['top_signals'])}")
                if market_snapshot['top_signals']:
                    for symbol, signal in list(market_snapshot['top_signals'].items())[:3]:
                        print(f"       {symbol}: price=${signal.get('price', 'MISSING')}")
                print()
            
            print(f"\n🤖 Calling LLM for trading decision...")
            print(f"   Signals analyzed: {len(market_snapshot['top_signals'])} stocks")
            print(f"   Portfolio: Cash=${market_snapshot['portfolio']['cash']:.0f}, Equity=${market_snapshot['portfolio']['total_equity']:.0f}")
            print(f"   Top signals:")
            for symbol, signal in list(market_snapshot['top_signals'].items())[:3]:
                rsi = signal.get('rsi', 50)
                price = signal.get('price', 0)
                print(f"      {symbol}: ${price:.2f} (RSI={rsi:.1f})")

            if pipeline:
                print(f"   Sub-agent pipeline: {len(pipeline)} step(s)")
                decision, (input_delta, output_delta), pipeline_calls, step_outputs = (
                    run_pipeline_decision(
                        llm_client,
                        pipeline=pipeline,
                        market_snapshot=market_snapshot,
                        model=model,
                    )
                )
                self.input_tokens += input_delta
                self.output_tokens += output_delta
                self.llm_calls += pipeline_calls
                self.last_pipeline_step_outputs = step_outputs or []
                if decision is None:
                    if strict_llm:
                        raise LLMDecisionError(
                            "LLM pipeline returned no parseable decision"
                        )
                    print("   Falling back to rule-based logic")
                    return self.make_trading_decision(portfolio_state)
            else:
                prompt = create_prompt(
                    market_snapshot,
                    mode=mode,
                    custom_prompt=strategy_prompt,
                    allowed_symbols=self.allowed_symbols,
                )

                # ================================================================
                # STEP 2: Call Claude with technical indicator analysis
                # ================================================================
                # Reasoning models (Nemotron) sometimes return only thinking
                # blocks. Retry, then one last JSON-only rescue call so we do
                # not tank H6 coverage on intermittent empty responses.
                llm_response = None
                no_text_retries = 4
                # One recovery budget (``RECOVERY_MAX_OUTPUT_TOKENS``) per
                # step, spent by whichever unusable reply claims it first: the
                # final rescue call below, or the truncation retry after the
                # parse. A reply still unusable at that budget has nothing
                # different left to ask for.
                recovery_spent = False
                output_delta = 0
                for attempt in range(no_text_retries + 1):
                    if attempt == no_text_retries:
                        # Final rescue: preserve the provider's reasoning mode,
                        # but give the model more room for reasoning plus JSON.
                        print(
                            "   ⚠️  Final rescue call with reasoning preserved "
                            f"and max_tokens={RECOVERY_MAX_OUTPUT_TOKENS}"
                        )
                        recovery_spent = True
                        response = _request_trading_decision(
                            llm_client,
                            prompt=prompt,
                            model=model,
                            max_tokens=RECOVERY_MAX_OUTPUT_TOKENS,
                            temperature=temperature,
                            market_context=market_context,
                        )
                    else:
                        response = _request_trading_decision(
                            llm_client,
                            prompt=prompt,
                            model=model,
                            temperature=temperature,
                            market_context=market_context,
                        )
                    output_delta = self._record_llm_usage(response)
                    try:
                        llm_response = _extract_response_text(response)
                        break
                    except AttributeError as extract_err:
                        if "No text content" not in str(extract_err):
                            raise
                        if attempt < no_text_retries:
                            print(
                                f"   ⚠️  {extract_err}; "
                                f"retry {attempt + 1}/{no_text_retries}"
                            )
                            continue
                        raise

                # ================================================================
                # STEP 3: Parse and validate LLM response
                # ================================================================
                decision = _parse_llm_response(llm_response)
                if decision is None and not recovery_spent:
                    # The same single recovery attempt the pipeline path makes:
                    # a reply the provider cut at the output ceiling, or one
                    # that opened the decision envelope and never closed it,
                    # gets the same request once more with room for reasoning
                    # plus the JSON. Without it the leaderboard ``llm_agent`` —
                    # which never runs a pipeline — turned every over-long
                    # reasoning burst into a billed step with no H6 decision.
                    reason = truncation_reason(response, output_delta, llm_response)
                    if reason is not None:
                        print(
                            f"   ⚠️  LLM response {reason}; retrying once with "
                            "reasoning preserved and "
                            f"max_tokens={RECOVERY_MAX_OUTPUT_TOKENS}"
                        )
                        retry_response = _request_trading_decision(
                            llm_client,
                            prompt=prompt,
                            model=model,
                            max_tokens=RECOVERY_MAX_OUTPUT_TOKENS,
                            temperature=temperature,
                            market_context=market_context,
                        )
                        self._record_llm_usage(retry_response)
                        retry_text = response_text_or_none(retry_response)
                        if retry_text is None:
                            print(
                                "   ⚠️  Retry returned no text block; "
                                "keeping the first attempt"
                            )
                        else:
                            decision = _parse_llm_response(retry_text)
                if decision is None:
                    if strict_llm:
                        raise LLMDecisionError(
                            "LLM response could not be parsed as a decision"
                        )
                    return {"actions": []}

            # ================================================================
            # STEP 4: Convert LLM decisions to actions
            # ================================================================
            actions = []
            llm_actions = decision.get("actions", [])

            if not llm_actions:
                if strict_llm:
                    # H6 coverage increment #1 of 2 (see the success exit
                    # below). An empty action list is a *valid* model decision
                    # — "do nothing this step" — and under strict_llm nothing
                    # rule-based ran, so the model genuinely drove the step and
                    # it must count. The non-strict branch below is the
                    # opposite case: it silently swaps in a rule-based decision,
                    # which is exactly what must NOT count.
                    self.llm_decisions += 1
                    return {"actions": []}
                print(f"   ⚠️  LLM returned no actions. Decision object: {decision}")
                print(f"   Falling back to rule-based logic")
                return self.make_trading_decision(portfolio_state)

            # The prompt contract asks for at most one action per universe symbol.
            # A response with more than that is degenerate or hostile (e.g. a
            # free-form strategy_prompt goading the model into spamming
            # actions) — bound the work instead of iterating it all.
            max_actions = max(len(self.allowed_symbols), 1)
            if len(llm_actions) > max_actions:
                if strict_llm:
                    raise LLMDecisionError(
                        "LLM returned an invalid action batch"
                    )
                print(
                    f"   ⚠️  LLM returned {len(llm_actions)} actions; "
                    f"processing only the first {max_actions}"
                )
                llm_actions = llm_actions[:max_actions]

            if strict_llm:
                valid_action_types = {"buy", "sell", "hold"}
                for llm_action in llm_actions:
                    if not isinstance(llm_action, dict):
                        raise LLMDecisionError(
                            "LLM returned an invalid action batch"
                        )
                    symbol = str(llm_action.get("symbol") or "").strip().upper()
                    action_type = str(
                        llm_action.get("action") or ""
                    ).strip().lower()
                    if (
                        symbol not in self._allowed_set
                        or action_type not in valid_action_types
                    ):
                        raise LLMDecisionError(
                            "LLM returned an invalid action batch"
                        )

            for llm_action in llm_actions:
                symbol = str(llm_action.get("symbol") or "").strip().upper()
                action_type = llm_action.get("action", "hold").lower()
                confidence = llm_action.get("confidence", 0.5)
                reasoning = llm_action.get("reasoning", "")
                
                print(f"\n   Processing: {symbol} ({action_type.upper()}, conf={confidence:.0%})")
                print(f"      Reasoning: {reasoning[:60]}...")
                
                # Skip low-confidence decisions
                if confidence < 0.3:
                    print(f"      ⏸️  Skipping (confidence {confidence:.0%} too low)")
                    continue
                
                if symbol not in self._allowed_set:
                    print(f"   ❌ {symbol}: Outside selected universe, skipping")
                    continue
                
                signal = signals.get(symbol, {})
                price = float(signal.get("price", 0)) if signal.get("price") else 0.0
                
                if action_type == "buy":
                    # Use position_size from LLM directly. Coerce defensively:
                    # json.loads happily yields strings, floats, Infinity and
                    # NaN here, and an unparseable value must skip THIS action,
                    # not throw the whole decision into the rule-based fallback.
                    raw_size = llm_action.get("position_size", 0)
                    try:
                        shares = (
                            float(raw_size or 0)
                            if self.lot_size > 1
                            else int(raw_size or 0)
                        )
                    except (TypeError, ValueError, OverflowError):
                        print(f"      ⚠️  BUY {symbol}: Skip (unparseable position_size {raw_size!r})")
                        continue
                    if isinstance(shares, float) and not math.isfinite(shares):
                        print(f"      ⚠️  BUY {symbol}: Skip (unparseable position_size {raw_size!r})")
                        continue

                    # If LLM didn't provide position_size, calculate from confidence
                    if shares == 0:
                        base_risk = portfolio_state["total_equity"] * 0.02
                        risk_amount = base_risk * confidence
                        if price > 0:
                            calculated = risk_amount / price
                            if self.lot_size > 1:
                                # This size is ours, not the model's: rounding
                                # it down to a whole lot is not the silent
                                # correction the executor refuses to make. A
                                # raw risk budget is essentially never a lot
                                # multiple, so submitting it unrounded would
                                # mint a guaranteed `invalid_lot_size`
                                # rejection on every fallback -- punishing the
                                # agent for the harness's own arithmetic. A
                                # quantity the LLM *did* request still passes
                                # through untouched above, so a genuinely
                                # invalid request stays visible as a rejection.
                                shares = (
                                    int(calculated) // self.lot_size
                                ) * self.lot_size
                            else:
                                shares = int(calculated)
                        else:
                            shares = 0

                    if shares > MAX_ORDER_SHARES:
                        # Same per-order ceiling validate_llm_response enforces
                        # on the safe path; a free-form strategy_prompt must
                        # not push an unbounded order through this loop.
                        print(f"      ⚠️  BUY {symbol}: Skip (position_size {shares} > per-order cap {MAX_ORDER_SHARES})")
                    elif shares > 0 and (
                        self.lot_size > 1 or shares * price <= self.cash
                    ):
                        actions.append({
                            "symbol": symbol,
                            "action": "buy",
                            "shares": shares,
                            "reason": f"[LLM] {reasoning} (confidence: {confidence:.0%})",
                            "confidence": confidence
                        })
                        print(f"      ✅ BUY {symbol}: {shares} shares @ ${price:.2f} (conf: {confidence:.0%})")
                    else:
                        print(f"      ⚠️  BUY {symbol}: Skip (insufficient cash: need ${shares*price:,.0f}, have ${self.cash:,.0f})")
                
                elif action_type == "sell":
                    # Size at the sellable quantity, not the held one: under T+1
                    # they differ, and the difference is unfillable.
                    held = self.positions.get(symbol, 0)
                    sellable = self.sellable_shares(symbol)
                    if sellable < held:
                        self.record_t1_deferral(symbol, held, sellable)
                    if sellable > 0:
                        action = {
                            "symbol": symbol,
                            "action": "sell",
                            "shares": sellable,
                            "reason": f"[LLM] {reasoning} (confidence: {confidence:.0%})",
                            "confidence": confidence
                        }
                        if sellable < held:
                            # The reason string still carries the model's own
                            # words ("exit the full position"); without this the
                            # log would read as though it asked for `sellable`.
                            action["requested_shares"] = held
                        actions.append(action)
                        print(f"      ✅ SELL {symbol}: {sellable} shares @ ${price:.2f} (conf: {confidence:.0%})")
                    elif symbol in self.positions and self.positions[symbol] > 0:
                        print(f"      ⚠️  SELL {symbol}: Skip (T+1 frozen, {self.positions[symbol]} shares settle next trading day)")
                    else:
                        print(f"      ⚠️  SELL {symbol}: Skip (not in portfolio, only owns: {list(self.positions.keys())})")
                
                # else: HOLD is implicit (don't add to actions)
            
            # H6 coverage increment #2 of 2 — the main success exit: the model's
            # actions were parsed AND processed without error, so the returned
            # decision is genuinely the model's. This is the H6 coverage
            # numerator, distinct from llm_calls (billed calls): any exception in
            # the processing loop above is caught below and swapped for a
            # rule-based decision, which must NOT count. (Actions filtered for
            # low confidence / invalid symbol still count: the model drove the
            # step, our policy merely declined to act on it.)
            #
            # These two sites are the ONLY writes to llm_decisions. Adding a
            # third means re-checking that the step really was model-driven with
            # no rule-based fallback — that invariant is what the leaderboard's
            # H6 guard rests on.
            self.llm_decisions += 1
            print(f"   ✅ Total actions: {len(actions)}\n")
            return {"actions": actions}

        except LLMDecisionError as strict_error:
            if getattr(llm_client, "fail_closed", False):
                raise
            return self._absorb_strict_llm_failure(portfolio_state, strict_error)
        except Exception as e:
            if strict_llm:
                if getattr(llm_client, "fail_closed", False):
                    raise
                # Same treatment as an explicit strict violation. Built rather
                # than raised because the budget may still absorb it; the cause
                # is attached by hand so the traceback survives either way.
                strict_error = LLMDecisionError("LLM decision processing failed")
                strict_error.__cause__ = e
                return self._absorb_strict_llm_failure(portfolio_state, strict_error)
            print(f"\n❌ LLM decision error: {e}")
            print(f"   Falling back to rule-based logic\n")
            return self.make_trading_decision(portfolio_state)

    def _record_llm_usage(self, response) -> int:
        """Bill one received response; returns its output-token delta.

        A response whose usage cannot be read is logged and not counted as a
        call — ``llm_calls`` is "billed API calls (any response with usage)".
        """
        try:
            input_delta, output_delta = _extract_token_usage(response)
        except Exception as usage_err:
            print(f"   ⚠️  Could not read token usage: {usage_err}")
            return 0
        self.input_tokens += input_delta
        self.output_tokens += output_delta
        self.llm_calls += 1
        return output_delta

    def strict_llm_fallback_budget(self) -> int:
        """How many unusable model responses this strict run may absorb."""
        total_steps = self.strict_llm_total_steps
        if not total_steps or total_steps <= 0:
            return 0
        return int(total_steps * STRICT_LLM_MAX_FALLBACK_RATIO)

    def _absorb_strict_llm_failure(
        self,
        portfolio_state: Dict,
        error: LLMDecisionError,
    ) -> Dict:
        """Spend one strike on an unusable response, or abort the run.

        The step falls back to rule-based logic and deliberately does NOT
        increment ``llm_decisions``: the model did not drive it, so it counts
        against H6 coverage exactly as an ordinary fallback would.
        """
        self.strict_llm_fallbacks += 1
        budget = self.strict_llm_fallback_budget()
        if self.strict_llm_fallbacks > budget:
            raise error
        print(
            f"\n⚠️  Strict LLM step failed ({error}); using rule-based logic for "
            f"this step. Strike {self.strict_llm_fallbacks}/{budget}.\n"
        )
        return self.make_trading_decision(portfolio_state)
    
    def execute_actions(
        self,
        actions: List[Dict],
        market_data: Dict,
        timestamp: datetime,
        fallback_prices: Optional[Dict] = None,
    ):
        """Execute trading decisions."""
        trades_before = len(self.trades)
        market_rules = None
        if self.market_rule_calendar is not None:
            market_rules = {
                action.get("symbol"): self.market_rule_calendar.rule_for_timestamp(
                    action.get("symbol"), timestamp
                )
                for action in actions
                if action.get("action") in {"buy", "sell"}
                and action.get("symbol") in self._allowed_set
            }
        self.cash = _execute_actions(
            actions=actions,
            market_data=market_data,
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions,
            entry_prices=self.entry_prices,
            trades=self.trades,
            t_plus_one_enabled=self.t_plus_one_enabled,
            available_positions=self.available_positions,
            frozen_lots=self.frozen_lots,
            rejected_orders=self.rejected_orders,
            lot_size=self.lot_size,
            order_events=self.order_events,
            order_event_repeats=self._order_event_repeats,
            transaction_cost_profile=self.transaction_cost_profile,
            market_rules=market_rules,
            fallback_prices=fallback_prices,
        )
        for trade in self.trades[trades_before:]:
            for field in self.transaction_cost_totals:
                self.transaction_cost_totals[field] += float(trade.get(field, 0.0) or 0.0)
    
    def update_equity(self, market_data: Dict, price_cache: Dict = None, timestamp = None):
        """Update equity snapshot using real data or forward-filled prices.

        Prefers real data, falls back to last-known price for smooth valuation.
        """
        _append_equity_record(
            self.equity_history,
            self.cash,
            self.positions,
            market_data,
            price_cache,
            timestamp,
        )

    def get_equity_curve(self) -> List[Dict]:
        """Return equity history."""
        return _get_equity_curve(self.equity_history)
