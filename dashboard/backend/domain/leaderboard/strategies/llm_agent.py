"""LLM trading agent as a leaderboard baseline.

Replays the hourly DJIA backtest over the contest window, asking an LLM
(e.g. Claude Haiku 4.5) for buy/sell/hold decisions each market hour. Every
model on the leaderboard is just a config entry with a different ``model_id`` —
the trading logic is shared and lives in ``backtest_hourly_agent``.

Unlike the deterministic baselines, this strategy makes real LLM API calls, so
it is expensive and slow. It is intended to be precomputed by the deploy script
(``scripts/deploy_leaderboard_model.py``) and cached, not recomputed on every
web request. It records token usage / cost so cost can be shown per run.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd

from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.llm.backtest_harness import (
    HAS_ANTHROPIC,
    default_model_name,
    make_llm_client,
)
from dashboard.backend.infrastructure.llm.providers import KNOWN_INTEGRATIONS

from .base import BaselineStrategy
from ._common import build_price_cache, market_timestamps, subset_bars, timestamps_in_contest


class LLMAgentStrategy(BaselineStrategy):
    key = "llm_agent"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mode = self.config.get("mode", "safe_trading")
        self.model_id = self.config.get("model_id")
        # Gateway selection: "commonstack" | "openrouter" | "anthropic".
        # Omitted → env auto-detect (CommonStack key prefers CommonStack).
        self.integration = self.config.get("integration")
        self.reasoning_effort = self.config.get("reasoning_effort")
        temperature = self.config.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError(
                "temperature must be a finite number between 0 and 2; "
                f"got {temperature!r}"
            )
        self.temperature = temperature
        # Populated during run() for reporting / cost tracking.
        self.llm_calls = 0
        self.llm_decisions = 0  # steps the model actually drove (H6 guard numerator)
        self.decision_steps = 0  # total steps offered to the model (guard denominator)
        self.input_tokens = 0
        self.output_tokens = 0
        self._num_trades = 0
        self.used_llm = False

    def required_symbols(self) -> List[str]:
        symbols = self.config.get("symbols")
        return list(symbols) if symbols else list(DJIA_30)

    def _make_client(self):
        if not HAS_ANTHROPIC:
            print("⚠️  Anthropic SDK unavailable — llm_agent falls back to rule-based.")
            return None
        client = make_llm_client(
            self.integration,
            reasoning_effort=self.reasoning_effort,
        )
        if client is None:
            chosen = self.integration or "auto"
            print(
                "⚠️  No LLM key for integration "
                f"{chosen!r} (need OPENROUTER_API_KEY / COMMONSTACK_API_KEY / "
                "ANTHROPIC_API_KEY as appropriate) — llm_agent falls back to rule-based. "
                f"Known integrations: {', '.join(KNOWN_INTEGRATIONS)}"
            )
        return client

    def run(
        self,
        bars_by_symbol: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> List[Dict[str, Any]]:
        symbols = self.required_symbols()
        bars_subset = subset_bars(bars_by_symbol, symbols)
        if not bars_subset:
            return []

        # Technical indicators are required for the LLM prompt context.
        # Bars may include a lookback period before start_date (needed for a
        # 1-day daily board); decision steps are clipped to the contest window.
        data = {
            sym: TechnicalIndicators.calculate_indicators(df)
            for sym, df in bars_subset.items()
        }

        timestamps = timestamps_in_contest(
            market_timestamps(data), start_date, end_date
        )
        if not timestamps:
            return []
        price_cache = build_price_cache(data, timestamps)

        client = self._make_client()
        self.used_llm = client is not None
        # When no explicit model id is configured, use the default that matches
        # the gateway make_llm_client actually built (CommonStack / OpenRouter
        # slug vs native Anthropic id) rather than a hardcoded id the gateway
        # rejects.
        model_id = self.model_id or default_model_name(self.integration)

        manager = PortfolioManager(initial_capital=initial_capital)
        total = len(timestamps)
        integration_label = self.integration or "auto"
        print(
            f"\n   🤖 LLM agent baseline: model={model_id} "
            f"integration={integration_label} mode={self.mode} "
            f"steps={total} llm={'on' if self.used_llm else 'off (rule-based)'}"
        )

        for i, ts in enumerate(timestamps):
            market_data = {}
            for sym in symbols:
                df = data.get(sym)
                if df is not None and ts in df.index:
                    market_data[sym] = df.loc[ts]

            state = manager.get_portfolio_state(market_data, price_cache, ts)
            state["timestamp"] = ts

            if client is not None:
                decision = manager.make_trading_decision_with_llm(
                    state,
                    client,
                    mode=self.mode,
                    model=model_id,
                    temperature=self.temperature,
                )
            else:
                decision = manager.make_trading_decision(state)

            manager.execute_actions(decision.get("actions", []), market_data, ts)
            manager.update_equity(market_data, price_cache, ts)

            if (i + 1) % 25 == 0 or (i + 1) == total:
                equity = manager.equity_history[-1]["equity"] if manager.equity_history else initial_capital
                print(f"      step {i + 1}/{total} · equity ${equity:,.0f} · calls {manager.llm_calls}")

        curve = manager.get_equity_curve()
        for entry in curve:
            if hasattr(entry["timestamp"], "isoformat"):
                entry["timestamp"] = entry["timestamp"].isoformat()

        self._num_trades = len(manager.trades)
        self.llm_calls = manager.llm_calls
        self.llm_decisions = manager.llm_decisions  # steps the model actually drove
        self.decision_steps = total  # how many steps the model was asked to decide
        self.input_tokens = manager.input_tokens
        self.output_tokens = manager.output_tokens
        self.model_id = model_id
        return curve

    def num_trades(self) -> int:
        return self._num_trades
