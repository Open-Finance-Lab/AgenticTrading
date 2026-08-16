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
from dashboard.backend.infrastructure.market_data.profiles import (
    ALPACA,
    get_market_profile,
)
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.llm.backtest_harness import (
    HAS_ANTHROPIC,
    default_model_name,
    make_llm_client,
)
from dashboard.backend.infrastructure.llm.providers import KNOWN_INTEGRATIONS
from dashboard.backend.infrastructure.llm.providers.openrouter import (
    reasoning_is_explicitly_enabled,
)

from .base import BaselineStrategy
from ._common import build_price_cache, market_timestamps, subset_bars, timestamps_in_contest


# Parity with api/routers/backtests.py::MAX_STRATEGY_PROMPT_CHARS, which caps the
# same field on /backtest/run for the same reason (the text rides every LLM call).
# Duplicated rather than imported because domain/ must not import api/
# (test_architecture_boundaries); test_llm_agent_instruction.py asserts the two
# constants stay equal so the copies cannot drift apart silently.
MAX_STRATEGY_PROMPT_CHARS = 4000


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
        # 0–1, not the OpenAI 0–2 range: every gateway here is reached through
        # an Anthropic Messages client, which caps temperature at 1.0. A value
        # above it would only surface as a per-request 400 — i.e. a silent
        # rule-based fallback that the H6 guard then blocks from publishing.
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 1
        ):
            raise ValueError(
                "temperature must be a finite number between 0 and 1; "
                f"got {temperature!r}"
            )
        # Anthropic rejects temperature alongside extended thinking, and the
        # OpenRouter wrapper injects a `thinking` block whenever the configured
        # effort enables it. Catch the conflict here rather than one 400 per
        # decision step.
        if temperature is not None and reasoning_is_explicitly_enabled(
            self.reasoning_effort
        ):
            raise ValueError(
                f"temperature ({temperature!r}) cannot be combined with "
                f"reasoning_effort={self.reasoning_effort!r}: extended thinking "
                "and temperature are mutually exclusive on the Anthropic "
                "Messages API. Disable reasoning or drop temperature."
            )
        self.temperature = temperature
        # C1: the Open Track's competing variable. Absent on the seven published
        # Model Track entries, which must keep producing their existing curves —
        # so blank collapses to None rather than "".
        strategy_prompt = self.config.get("strategy_prompt")
        if strategy_prompt is not None and not isinstance(strategy_prompt, str):
            # Typed error, not the AttributeError `.strip()` would raise: this
            # constructor runs on every public, unauthenticated
            # GET /api/v1/leaderboard via _symbols_for_config/_config_needs_alpaca,
            # so an uncaught one 500s the whole board — which surfaces to users as
            # a CORS failure rather than an error anyone can read.
            raise ValueError(
                "strategy_prompt must be a string; got "
                f"{type(strategy_prompt).__name__}"
            )
        self.strategy_prompt = (strategy_prompt or "").strip() or None
        if (
            self.strategy_prompt is not None
            and len(self.strategy_prompt) > MAX_STRATEGY_PROMPT_CHARS
        ):
            raise ValueError(
                f"strategy_prompt is {len(self.strategy_prompt)} characters, over the "
                f"{MAX_STRATEGY_PROMPT_CHARS} limit. It is injected into every LLM call "
                "of the run, and an over-long body pushes responses past "
                "LLM_MAX_OUTPUT_TOKENS — which truncates decisions and fails H6 while "
                "billing in full."
            )
        # create_prompt ignores `mode` entirely once a custom prompt is set
        # (validator.py:772), so an entry configured buy_and_hold + instruction
        # would be published under a mode it never actually ran. safe_trading is
        # the default and replacing *its* body is the whole point of the feature,
        # so only an explicit, different mode is a conflict. Same treatment as the
        # temperature/reasoning_effort pair above: fail here, not once per step.
        configured_mode = self.config.get("mode")
        if (
            self.strategy_prompt is not None
            and configured_mode is not None
            and configured_mode != "safe_trading"
        ):
            raise ValueError(
                f"mode={configured_mode!r} cannot be combined with strategy_prompt: a "
                "custom instruction REPLACES the mode's prompt body, so the entry would "
                "publish a curve labelled with a mode that never ran."
            )
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

    def _run_decision_loop(
        self,
        client,
        timestamps,
        symbols,
        data,
        price_cache,
        initial_capital,
        model_id,
    ):
        """One decision per timestamp, executed against a fresh PortfolioManager.

        Extracted from run() so the strategy_prompt hand-off is reachable in a
        test without live bars or an LLM client — an untestable call site is
        exactly where a silently-dropped keyword hides.
        """
        # Contest entries replay the hourly DJIA window over Alpaca bars, so the
        # execution rules come from that profile rather than a constructor
        # default — a leaderboard curve must never be produced under settlement
        # semantics its market does not have.
        profile = get_market_profile(ALPACA)
        manager = PortfolioManager(
            initial_capital=initial_capital,
            t_plus_one_enabled=profile.t_plus_one_enabled,
        )
        # Resolution lives in run() alone. Re-applying the
        # `or self.model_id or default_model_name(...)` chain here meant two
        # copies of it: change one and the header can advertise one model while
        # the calls use another, with the stored run metadata agreeing with the
        # header rather than with what was billed.
        total = len(timestamps)

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
                    strategy_prompt=self.strategy_prompt,
                    temperature=self.temperature,
                )
            else:
                decision = manager.make_trading_decision(state)

            manager.execute_actions(decision.get("actions", []), market_data, ts)
            manager.update_equity(market_data, price_cache, ts)

            if (i + 1) % 25 == 0 or (i + 1) == total:
                equity = manager.equity_history[-1]["equity"] if manager.equity_history else initial_capital
                print(f"      step {i + 1}/{total} · equity ${equity:,.0f} · calls {manager.llm_calls}")

        return manager

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

        total = len(timestamps)
        integration_label = self.integration or "auto"
        print(
            f"\n   🤖 LLM agent baseline: model={model_id} "
            f"integration={integration_label} mode={self.mode} "
            f"steps={total} llm={'on' if self.used_llm else 'off (rule-based)'}"
        )

        manager = self._run_decision_loop(
            client=client,
            timestamps=timestamps,
            symbols=symbols,
            data=data,
            price_cache=price_cache,
            initial_capital=initial_capital,
            model_id=model_id,
        )

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
