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
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
from dashboard.backend.infrastructure.llm.pipeline_runner import run_pipeline_decision


class PortfolioManager:
    """Manages portfolio with hourly trading decisions based on indicators."""
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        allowed_symbols: Optional[List[str]] = None,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: num_shares}
        self.entry_prices = {}  # {symbol: entry_price}
        self.trades = []
        self.equity_history = []
        # Real LLM token usage (server-side calls report actual counts)
        self.llm_calls = 0        # billed API calls (any response with usage)
        self.llm_decisions = 0    # steps the model actually drove (H6 coverage)
        self.input_tokens = 0
        self.output_tokens = 0
        # Latest decision-pipeline step outputs (for daily post-trade analysis).
        self.last_pipeline_step_outputs = []
        # Tradeable universe for this run (defaults to DJIA_30).
        symbols = allowed_symbols if allowed_symbols is not None else DJIA_30
        self.allowed_symbols = [str(s).strip().upper() for s in symbols if s]
        self._allowed_set = set(self.allowed_symbols)    
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
        return _make_rule_based_decision(
            portfolio_state=portfolio_state,
            positions=self.positions,
            cash=self.cash,
        )
    
    def make_trading_decision_with_llm(
        self,
        portfolio_state: Dict,
        llm_client,
        mode: str = "safe_trading",
        model: str = None,
        strategy_prompt: str = None,
        pipeline: List[Dict] = None,
        temperature: Optional[float] = None,
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
        if not HAS_ANTHROPIC or not llm_client:
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
                holdings[position["symbol"]] = {
                    "shares": position["shares"],
                    "entry_price": round(position["entry_price"], 2),
                    "current_price": round(position["current_price"], 2),
                    "position_value": round(position["position_value"], 2),
                    "pnl_pct": round(position["pnl_pct"], 2)
                }
            
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
                for attempt in range(no_text_retries + 1):
                    if attempt == no_text_retries:
                        # Final rescue: force reasoning off for this one call.
                        print(
                            "   ⚠️  Final rescue call with reasoning disabled "
                            "to obtain JSON text"
                        )
                        prev_effort = os.environ.get("OPENROUTER_REASONING_EFFORT")
                        os.environ["OPENROUTER_REASONING_EFFORT"] = "none"
                        try:
                            response = _request_trading_decision(
                                llm_client,
                                prompt=prompt,
                                model=model,
                                temperature=temperature,
                            )
                        finally:
                            if prev_effort is None:
                                os.environ.pop("OPENROUTER_REASONING_EFFORT", None)
                            else:
                                os.environ["OPENROUTER_REASONING_EFFORT"] = prev_effort
                    else:
                        response = _request_trading_decision(
                            llm_client,
                            prompt=prompt,
                            model=model,
                            temperature=temperature,
                        )
                    try:
                        input_delta, output_delta = _extract_token_usage(response)
                        self.input_tokens += input_delta
                        self.output_tokens += output_delta
                        self.llm_calls += 1
                    except Exception as usage_err:
                        print(f"   ⚠️  Could not read token usage: {usage_err}")
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
                if decision is None:
                    return {"actions": []}

            # ================================================================
            # STEP 4: Convert LLM decisions to actions
            # ================================================================
            actions = []
            llm_actions = decision.get("actions", [])

            if not llm_actions:
                print(f"   ⚠️  LLM returned no actions. Decision object: {decision}")
                print(f"   Falling back to rule-based logic")
                return self.make_trading_decision(portfolio_state)

            # The prompt contract asks for at most one action per universe symbol.
            # A response with more than that is degenerate or hostile (e.g. a
            # free-form strategy_prompt goading the model into spamming
            # actions) — bound the work instead of iterating it all.
            max_actions = max(len(self.allowed_symbols), 1)
            if len(llm_actions) > max_actions:
                print(
                    f"   ⚠️  LLM returned {len(llm_actions)} actions; "
                    f"processing only the first {max_actions}"
                )
                llm_actions = llm_actions[:max_actions]

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
                        shares = int(raw_size or 0)
                    except (TypeError, ValueError, OverflowError):
                        print(f"      ⚠️  BUY {symbol}: Skip (unparseable position_size {raw_size!r})")
                        continue

                    # If LLM didn't provide position_size, calculate from confidence
                    if shares == 0:
                        base_risk = portfolio_state["total_equity"] * 0.02
                        risk_amount = base_risk * confidence
                        shares = int(risk_amount / price) if price > 0 else 0

                    if shares > MAX_ORDER_SHARES:
                        # Same per-order ceiling validate_llm_response enforces
                        # on the safe path; a free-form strategy_prompt must
                        # not push an unbounded order through this loop.
                        print(f"      ⚠️  BUY {symbol}: Skip (position_size {shares} > per-order cap {MAX_ORDER_SHARES})")
                    elif shares > 0 and shares * price <= self.cash:
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
                    if symbol in self.positions and self.positions[symbol] > 0:
                        actions.append({
                            "symbol": symbol,
                            "action": "sell",
                            "shares": self.positions[symbol],
                            "reason": f"[LLM] {reasoning} (confidence: {confidence:.0%})",
                            "confidence": confidence
                        })
                        print(f"      ✅ SELL {symbol}: {self.positions[symbol]} shares @ ${price:.2f} (conf: {confidence:.0%})")
                    else:
                        print(f"      ⚠️  SELL {symbol}: Skip (not in portfolio, only owns: {list(self.positions.keys())})")
                
                # else: HOLD is implicit (don't add to actions)
            
            # Count this step as model-driven only here, at the single success
            # exit — the model's actions were parsed AND processed without error,
            # so the returned decision is genuinely the model's. This is the H6
            # coverage numerator, distinct from llm_calls (billed calls): any
            # exception in the processing loop above is caught below and swapped
            # for a rule-based decision, which must NOT count. (Actions filtered
            # for low confidence / invalid symbol still count: the model drove
            # the step, our policy merely declined to act on it.)
            self.llm_decisions += 1
            print(f"   ✅ Total actions: {len(actions)}\n")
            return {"actions": actions}

        except Exception as e:
            print(f"\n❌ LLM decision error: {e}")
            print(f"   Falling back to rule-based logic\n")
            return self.make_trading_decision(portfolio_state)
    
    def execute_actions(self, actions: List[Dict], market_data: Dict, timestamp: datetime):
        """Execute trading decisions."""
        self.cash = _execute_actions(
            actions=actions,
            market_data=market_data,
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions,
            entry_prices=self.entry_prices,
            trades=self.trades,
        )
    
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
