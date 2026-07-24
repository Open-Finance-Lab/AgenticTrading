"""Hourly DJIA backtest engine.

Moved verbatim (Phase 2C5) from ``dashboard/scripts/backtest_hourly_agent.py``.
``HourlyBacktester`` runs the hourly agent backtest plus the buy-and-hold and DJIA
baselines, persisting results to the database. The class body is functionally
identical to the legacy implementation; only the imports are canonical. The
legacy script re-exports this exact class so ``bha.HourlyBacktester`` and existing
subclasses (e.g. ``backtest_custom_algo``) keep working unchanged.

This module is backend domain code: it must NOT import dashboard scripts,
``backtest_hourly_agent``, FastAPI routers, or the CLI bootstrap helpers.

Baseline methods and result assembly intentionally remain here for now; they can
be extracted in a later phase.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dashboard.backend.database import db
import dashboard.backend.infrastructure.llm.token_cost as token_cost
from dashboard.backend.baseline_generator import generate_baselines
from dashboard.backend.infrastructure.llm.validator import DJIA_30, TOP_10_STOCKS as TOP_10
from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.metrics import (
    calculate_sharpe,
    calculate_max_drawdown,
)
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.market_data.alpaca_bars import MarketDataUnavailableError
from dashboard.backend.infrastructure.market_data.provider import (
    ALPACA,
    VNPY_SIMULATION,
    create_market_data_provider,
)
import dashboard.backend.infrastructure.llm.backtest_harness as llm_harness
from dashboard.backend.infrastructure.llm.backtest_harness import (
    HAS_ANTHROPIC,
    default_model_name,
    make_llm_client,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import (
    is_last_bar_of_trading_day,
    recombine_pipeline,
    run_post_trade_analysis,
    split_pipeline,
    trading_day_key,
)


class HourlyBacktester:
    """Runs hourly backtest with agent and baselines."""
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        session_id: str = "legacy-demo-session",
        use_llm: bool = True,
        mode: str = "safe_trading",
        strategy_prompt: str = None,
        model: str = None,
        pipeline: list = None,
        live_run_id: str = None,
        progress_file: str = None,
        data_source: str = ALPACA,
        initial_capital: float = None,
        symbols: Optional[List[str]] = None,
    ):
        # Validate and swap dates if they're in the wrong order
        from datetime import datetime as dt_parser
        try:
            start = dt_parser.strptime(start_date, "%Y-%m-%d")
            end = dt_parser.strptime(end_date, "%Y-%m-%d")
            
            if start > end:
                print(f"⚠️  Dates were backwards ({start_date} > {end_date}). Swapping...")
                start_date, end_date = end_date, start_date
        except ValueError:
            pass  # Invalid date format, let Alpaca handle the error
        
        self.start_date = start_date
        self.end_date = end_date
        self.session_id = session_id
        self.initial_capital = float(INITIAL_CAPITAL if initial_capital is None else initial_capital)
        self.mode = mode  # "safe_trading" or "buy_and_hold"
        # Optional free-form strategy that REPLACES the built-in prompt for this run.
        self.strategy_prompt = (strategy_prompt or "").strip() or None
        # Optional sub-agent pipeline (when set, overrides strategy_prompt).
        self.pipeline = pipeline if pipeline else None
        self.initial_pipeline = (
            json.loads(json.dumps(self.pipeline)) if self.pipeline else None
        )
        self.prompt_adaptations: List[Dict] = []
        # Model id; defaults to the gateway-appropriate slug (CommonStack vs native).
        self.model = model or default_model_name()
        self.live_run_id = (live_run_id or "").strip() or None
        self.progress_file = (progress_file or "").strip() or None
        self.data_source = data_source
        self.data_loader = create_market_data_provider(data_source)
        # Selected asset universe for this run (defaults to full DJIA_30).
        if symbols:
            cleaned = []
            seen = set()
            for raw in symbols:
                sym = str(raw or "").strip().upper()
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                cleaned.append(sym)
            self.symbols = cleaned or list(DJIA_30)
        else:
            self.symbols = list(DJIA_30)
        self.all_data = {}
        self.use_llm = use_llm and HAS_ANTHROPIC and data_source != VNPY_SIMULATION
        self.llm_client = None
        
        # Initialize LLM client if enabled. Prefer CommonStack (the model we host)
        # via make_llm_client(); it falls back to native Anthropic when only
        # ANTHROPIC_API_KEY is set, and returns None when no key/SDK is available.
        if self.use_llm:
            self.llm_client = make_llm_client()
            if self.llm_client is None:
                print(
                    "⚠️  No LLM key (COMMONSTACK_API_KEY / OPENROUTER_API_KEY / "
                    "ANTHROPIC_API_KEY) set. Running without LLM."
                )
                self.use_llm = False
            else:
                print(f"✅ LLM initialized (model={self.model})")
    
    def _serialize_trades(self, trades: List[Dict]) -> List[Dict]:
        serialized = []
        for trade in trades:
            ts = trade.get("timestamp")
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            side = str(trade.get("side", "")).upper()
            quantity = int(trade.get("shares") or trade.get("quantity") or 0)
            price = float(trade.get("price") or 0)
            value = float(
                trade.get("cost")
                or trade.get("proceeds")
                or trade.get("value")
                or quantity * price
            )
            serialized.append(
                {
                    "timestamp": ts,
                    "symbol": trade.get("symbol"),
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "value": value,
                    "reason": trade.get("reason", ""),
                }
            )
        return serialized

    def _publish_live_progress(self, step: int, total_steps: int, manager) -> None:
        """Write incremental equity curve snapshots for live dashboard charting."""
        if not self.progress_file:
            return
        from pathlib import Path

        curve = manager.get_equity_curve()
        serialized = []
        for entry in curve:
            ts = entry.get("timestamp")
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            serialized.append(
                {
                    "timestamp": ts,
                    "equity": float(entry.get("equity", 0) or 0),
                    "cash": float(entry.get("cash", 0) or 0),
                    "positions_value": float(entry.get("positions_value", 0) or 0),
                }
            )
        payload = {
            "run_id": self.live_run_id,
            "step": step,
            "total_steps": total_steps,
            "equity_curve": serialized,
            "trades": self._serialize_trades(manager.trades),
        }
        try:
            Path(self.progress_file).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            print(f"   ⚠️  Could not write live progress: {exc}")
    
    def load_data(self):
        """Fetch hourly data from the selected normalized provider."""
        print(f"   Universe: {len(self.symbols)} symbols ({', '.join(self.symbols[:8])}"
              f"{'…' if len(self.symbols) > 8 else ''})")
        self.all_data = self.data_loader.fetch_bars(
            self.symbols, self.start_date, self.end_date
        )
        if not self.all_data:
            # Raise, don't sys.exit(1): this runs inside server threads
            # (external runs, algo service) where SystemExit evades
            # `except Exception` and strands the run (the B0 class).
            print("❌ No data fetched.")
            raise MarketDataUnavailableError(
                f"No {self.data_source} market data available for "
                f"{self.start_date}..{self.end_date}"
            )
    
    def calculate_indicators(self):
        """Calculate technical indicators for all symbols."""
        print("\n📈 Calculating technical indicators...")
        count = 0
        for symbol, df in self.all_data.items():
            self.all_data[symbol] = TechnicalIndicators.calculate_indicators(df)
            count += 1
            if count % 5 == 0:
                print(f"  ✅ {count}/{len(self.all_data)} symbols...")
        print(f"  ✅ All indicators calculated\n")
    
    def _run_metadata(self) -> Dict:
        """Data provenance recorded on EVERY run row, agent and baseline alike.

        Provenance is the only thing the baselines share with the agent: they
        make no model calls and run no pipeline, so anything LLM-shaped belongs
        in ``_agent_run_metadata`` instead."""
        return {
            "data_source": self.data_source,
            "symbols": list(self.symbols),
        }

    def _agent_run_metadata(self) -> Dict:
        """Provenance plus the effective config the agent run actually used.

        LLM_MAX_OUTPUT_TOKENS is an env knob that changes a run's spend and
        response truncation; recording the EFFECTIVE value (post defensive
        parse) makes runs auditable after the env changes."""
        meta: Dict = self._run_metadata()
        if self.use_llm:
            meta["llm_max_output_tokens"] = llm_harness.DEFAULT_MAX_OUTPUT_TOKENS
        if self.prompt_adaptations:
            meta["prompt_adaptations"] = self.prompt_adaptations
        if self.initial_pipeline is not None:
            meta["initial_pipeline"] = self.initial_pipeline
        if self.pipeline is not None:
            meta["final_pipeline"] = self.pipeline
        return meta

    def _current_equity(self, manager: PortfolioManager) -> float:
        if manager.equity_history:
            return float(manager.equity_history[-1].get("equity") or manager.cash)
        return float(manager.cash)

    def _run_daily_post_trade(
        self,
        *,
        manager: PortfolioManager,
        day_episode: Dict,
        post_trade_steps: List[Dict],
    ) -> None:
        """Run once-per-day post-trade analysis and mutate ``self.pipeline``."""
        if not post_trade_steps or not self.use_llm or not self.llm_client:
            return
        if not day_episode.get("trading_day"):
            return

        decision_steps, _ = split_pipeline(self.pipeline)
        start_eq = float(day_episode.get("day_start_equity") or 0)
        end_eq = self._current_equity(manager)
        day_return = ((end_eq - start_eq) / start_eq) if start_eq else 0.0
        trade_start = int(day_episode.get("trade_start_index") or 0)
        day_trades = manager.trades[trade_start:]

        episode_context = {
            "trading_day": day_episode.get("trading_day"),
            "day_start_equity": start_eq,
            "day_end_equity": end_eq,
            "day_return": day_return,
            "trade_count": len(day_trades),
            "trades": day_trades,
            "latest_step_outputs": day_episode.get("latest_step_outputs") or [],
        }

        patched, record, (in_tok, out_tok), calls = run_post_trade_analysis(
            self.llm_client,
            post_trade_steps=post_trade_steps,
            episode_context=episode_context,
            decision_pipeline=decision_steps,
            model=self.model,
        )
        manager.input_tokens += in_tok
        manager.output_tokens += out_tok
        manager.llm_calls += calls
        if record:
            self.prompt_adaptations.append(record)
        self.pipeline = recombine_pipeline(patched, post_trade_steps)

    def run_agent_backtest(self) -> Tuple[str, List[Dict]]:
        """Run backtest with agent making hourly decisions."""
        print("🤖 Running Agent backtest (hourly decisions)...\n")
        
        # Track LLM usage for results metadata
        llm_calls_count = 0
        llm_model = "rule-based"  # Default
        
        manager = PortfolioManager(
            initial_capital=self.initial_capital,
            allowed_symbols=self.symbols,
        )
        _decision_steps, post_trade_steps = split_pipeline(self.pipeline)
        if post_trade_steps:
            print(
                f"   Post-trade analysis: {len(post_trade_steps)} step(s), "
                "once per trading day\n"
            )
        
        # Get all timestamps
        all_timestamps = set()
        for df in self.all_data.values():
            all_timestamps.update(df.index)
        all_timestamps = sorted(all_timestamps)
        
        # Filter: only keep hours with real data for 80%+ of symbols
        min_required = int(len(self.all_data) * 0.8)
        filtered = []
        for ts in all_timestamps:
            real_data_count = sum(1 for df in self.all_data.values() if ts in df.index)
            if real_data_count >= min_required:
                filtered.append(ts)
        
        all_timestamps = filtered if filtered else all_timestamps
        
        # Filter: only keep regular market hours (9:30 AM - 4:00 PM ET)
        # Exclude pre-market (before 9:30 AM) and after-hours (after 4:00 PM)
        import pytz
        et_tz = pytz.timezone('US/Eastern')
        market_hours_only = []
        
        for ts in all_timestamps:
            # Convert to ET
            ts_et = ts.astimezone(et_tz)
            hour = ts_et.hour
            minute = ts_et.minute
            
            # Market hours: 9:30 AM (hour 9, min 30+) through 4:00 PM (hour 16, min 0)
            is_market_hours = (hour > 9 and hour < 16) or \
                             (hour == 9 and minute >= 30) or \
                             (hour == 16 and minute == 0)
            
            if is_market_hours:
                market_hours_only.append(ts)
        
        all_timestamps = market_hours_only
        
        print(f"   Trading {len(all_timestamps)} hours during regular market hours (9:30 AM - 4:00 PM ET)...\n")
        total_steps = len(all_timestamps)
        
        # Build forward-filled price cache to handle missing hourly data
        print("   Pre-computing forward-filled price cache...")
        price_cache = {}
        for symbol, df in self.all_data.items():
            price_cache[symbol] = {}
            last_price = None
            
            for timestamp in all_timestamps:
                if timestamp in df.index:
                    last_price = df.loc[timestamp, "close"]
                    price_cache[symbol][timestamp] = last_price
                else:
                    # Fallback (shouldn't happen with daily data)
                    if last_price is not None:
                        price_cache[symbol][timestamp] = last_price
        
        print("   ✅ Cache ready\n")

        day_episode: Dict = {
            "trading_day": None,
            "day_start_equity": None,
            "trade_start_index": 0,
            "latest_step_outputs": [],
        }
        
        # Hourly loop
        for i, timestamp in enumerate(all_timestamps):
            day_key = trading_day_key(timestamp)
            if day_episode["trading_day"] != day_key:
                day_episode = {
                    "trading_day": day_key,
                    "day_start_equity": self._current_equity(manager),
                    "trade_start_index": len(manager.trades),
                    "latest_step_outputs": [],
                }

            # Get market data for this hour (real data when available)
            market_data = {}
            for symbol in self.symbols:
                if symbol not in self.all_data:
                    continue
                df = self.all_data[symbol]
                if timestamp not in df.index:
                    continue
                market_data[symbol] = df.loc[timestamp]
            
            # Get portfolio state (uses real data for signals, forward-fill for valuation)
            state = manager.get_portfolio_state(market_data, price_cache, timestamp)
            state["timestamp"] = timestamp  # Add timestamp for LLM context
            
            # Make decision (LLM if available, else rule-based)
            if self.use_llm and self.llm_client:
                decision = manager.make_trading_decision_with_llm(
                    state,
                    self.llm_client,
                    mode=self.mode,
                    model=self.model,
                    strategy_prompt=self.strategy_prompt,
                    pipeline=self.pipeline,
                )
                llm_calls_count += 1  # Track that LLM was used
                if llm_calls_count == 1:  # Set on first call
                    llm_model = self.model
                if manager.last_pipeline_step_outputs:
                    day_episode["latest_step_outputs"] = manager.last_pipeline_step_outputs
            else:
                decision = manager.make_trading_decision(state)
            
            # Execute trades (only if real data available)
            manager.execute_actions(decision["actions"], market_data, timestamp)
            
            # Update equity (uses forward-filled prices for smooth valuation)
            manager.update_equity(market_data, price_cache, timestamp)
            self._publish_live_progress(i + 1, total_steps, manager)

            if post_trade_steps and is_last_bar_of_trading_day(all_timestamps, i):
                self._run_daily_post_trade(
                    manager=manager,
                    day_episode=day_episode,
                    post_trade_steps=post_trade_steps,
                )
            
            # Progress
            if (i + 1) % 100 == 0:
                equity = manager.equity_history[-1]["equity"]
                pct_return = ((equity - self.initial_capital) / self.initial_capital) * 100
                print(f"   Hour {i+1}/{len(all_timestamps)}: Equity ${equity:,.0f} ({pct_return:+.1f}%)")
        
        equity_curve = manager.get_equity_curve()
        
        # Convert timestamps to strings
        for entry in equity_curve:
            if hasattr(entry["timestamp"], "isoformat"):
                entry["timestamp"] = entry["timestamp"].isoformat()
        
        # Store in database
        run_id = self.live_run_id or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        initial_eq = equity_curve[0]["equity"] if equity_curve else self.initial_capital
        final_eq = equity_curve[-1]["equity"] if equity_curve else self.initial_capital
        total_return = (final_eq - self.initial_capital) / self.initial_capital

        est_cost = token_cost.estimate_cost_usd(
            llm_model, manager.input_tokens, manager.output_tokens
        )

        db.insert_run(
            run_id=run_id,
            session_id=self.session_id,
            agent_name="Agent",
            mode="backtest",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=initial_eq,
            final_equity=final_eq,
            total_return=total_return,
            sharpe_ratio=self._calc_sharpe(equity_curve),
            max_drawdown=self._calc_max_dd(equity_curve),
            num_trades=len(manager.trades),
            llm_model=llm_model,  # Track which model was used
            llm_calls=manager.llm_calls,
            input_tokens=manager.input_tokens,
            output_tokens=manager.output_tokens,
            est_cost_usd=est_cost,
            metadata=self._agent_run_metadata(),
        )

        db.insert_equity_points(run_id, equity_curve)
        db.insert_trades(run_id, manager.trades)
        
        print(f"\n  ✅ Agent backtest complete")
        print(f"     • Run ID: {run_id}")
        model_display = self.model if llm_calls_count > 0 else "rule-based"
        print(f"     • Model: {model_display} (✅ LLM enabled)" if llm_calls_count > 0 else f"     • Model: {model_display} (❌ fallback)")
        print(f"     • LLM Calls: {llm_calls_count}")
        print(f"     • Tokens: {manager.input_tokens:,} in / {manager.output_tokens:,} out (est. cost ${est_cost:.4f})")
        print(f"     • Trades: {len(manager.trades)}")
        if self.prompt_adaptations:
            print(f"     • Post-trade adaptations: {len(self.prompt_adaptations)} day(s)")
        print(f"     • Final: ${final_eq:,.0f}")
        print(f"     • Return: {total_return*100:+.2f}%\n")
        
        return run_id, equity_curve
    
    def run_buyhold_baseline(self) -> Tuple[str, List[Dict]]:
        """Buy and hold baseline using shared baseline generator."""
        print("📊 Running Buy & Hold baseline...\n")

        # Full DJIA runs keep the historical 10-stock B&H sleeve; other universes
        # buy-and-hold the selected assets so the baseline matches the agent book.
        if set(self.symbols) == set(DJIA_30):
            bh_symbols = [s for s in TOP_10 if s in self.all_data]
        else:
            bh_symbols = [s for s in self.symbols if s in self.all_data]
        
        equity_history, _ = generate_baselines(
            bars_by_symbol=self.all_data,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            symbols_list=bh_symbols,
        )
        
        if not equity_history:
            return None, []
        
        # Store in database
        run_id = f"buyhold_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        initial_eq = equity_history[0]["equity"]
        final_eq = equity_history[-1]["equity"]
        total_return = (final_eq - self.initial_capital) / self.initial_capital
        
        db.insert_run(
            run_id=run_id,
            session_id=self.session_id,
            agent_name="buy-and-hold",
            mode="backtest",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=initial_eq,
            final_equity=final_eq,
            total_return=total_return,
            sharpe_ratio=self._calc_sharpe(equity_history),
            max_drawdown=self._calc_max_dd(equity_history),
            num_trades=1,
            metadata=self._run_metadata(),
        )
        
        db.insert_equity_points(run_id, equity_history)
        
        print(f"  ✅ Buy & Hold baseline complete")
        print(f"     • Run ID: {run_id}")
        print(f"     • Final: ${final_eq:,.0f}")
        print(f"     • Return: {total_return*100:+.2f}%\n")
        
        return run_id, equity_history
    
    def run_djia_baseline(self) -> Tuple[str, List[Dict]]:
        """DJIA index baseline using shared baseline generator."""
        print("📊 Running DJIA Index baseline...\n")
        
        # True DJIA equal-weight needs all 30 names. When the agent universe is a
        # subset (Mag7 / custom), fetch DJIA bars separately so the chart baseline
        # stays a real Dow proxy rather than the selected sleeve.
        if set(self.symbols) == set(DJIA_30) and set(self.all_data.keys()) >= set(DJIA_30):
            bars = self.all_data
        else:
            print("   Fetching full DJIA bars for index baseline…")
            bars = self.data_loader.fetch_bars(DJIA_30, self.start_date, self.end_date)
            if not bars:
                print("   ⚠️  No DJIA bars available; skipping index baseline")
                return None, []

        _, equity_history = generate_baselines(
            bars_by_symbol=bars,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            symbols_list=list(DJIA_30),
        )
        
        if not equity_history:
            return None, []
        
        # Store in database
        run_id = f"djia_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        initial_eq = equity_history[0]["equity"]
        final_eq = equity_history[-1]["equity"]
        total_return = (final_eq - self.initial_capital) / self.initial_capital
        
        db.insert_run(
            run_id=run_id,
            session_id=self.session_id,
            agent_name="DJIA",
            mode="backtest",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=initial_eq,
            final_equity=final_eq,
            total_return=total_return,
            sharpe_ratio=self._calc_sharpe(equity_history),
            max_drawdown=self._calc_max_dd(equity_history),
            num_trades=0,
            metadata=self._run_metadata(),
        )
        
        db.insert_equity_points(run_id, equity_history)
        
        print(f"  ✅ DJIA Index baseline complete")
        print(f"     • Run ID: {run_id}")
        print(f"     • Final: ${final_eq:,.0f}")
        print(f"     • Return: {total_return*100:+.2f}%\n")
        
        return run_id, equity_history
    
    @staticmethod
    def _calc_sharpe(equity_curve: List[Dict]) -> float:
        """Annualized hourly Sharpe ratio.

        Delegates to dashboard.backend.domain.backtesting.metrics.calculate_sharpe;
        inputs, outputs, edge cases, and the hourly annualization factor are
        unchanged.
        """
        return calculate_sharpe(equity_curve)

    @staticmethod
    def _calc_max_dd(equity_curve: List[Dict]) -> float:
        """Maximum drawdown of the equity curve.

        Delegates to
        dashboard.backend.domain.backtesting.metrics.calculate_max_drawdown.
        """
        return calculate_max_drawdown(equity_curve)
