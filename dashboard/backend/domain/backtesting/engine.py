"""Profile-driven hourly backtest engine.

Moved verbatim (Phase 2C5) from ``dashboard/scripts/backtest_hourly_agent.py``.
``HourlyBacktester`` runs the hourly agent backtest plus the configured buy-and-hold
and optional index baselines, persisting results to the database. The class body is functionally
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
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Tuple

from dashboard.backend.database import db
import dashboard.backend.infrastructure.llm.token_cost as token_cost
from dashboard.backend.baseline_generator import generate_baselines
from dashboard.backend.infrastructure.llm.validator import DJIA_30, TOP_10_STOCKS as TOP_10
from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
from dashboard.backend.domain.backtesting.currency import (
    CurrencyContext,
    CurrencyContextError,
)
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.metrics import (
    calculate_sharpe,
    calculate_max_drawdown,
)
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_RUNTIME_TYPE,
    DEFAULT_RUNTIME_TYPE,
    PIPELINE_RUNTIME_TYPE,
    AgentRuntimeConfigurationError,
    AgentRuntimeContext,
    AgentRuntimeError,
    RuntimeDispatcher,
    normalize_runtime_config,
    normalize_runtime_type,
)
from dashboard.backend.infrastructure.ai_hedge_fund.adapter import AiHedgeFundRuntime
from dashboard.backend.infrastructure.market_data.alpaca_bars import MarketDataUnavailableError
from dashboard.backend.infrastructure.market_data.ifind_client import IFindClientError
from dashboard.backend.infrastructure.market_data.ifind_fx import (
    IFindFxError,
    MAX_RELATIVE_DEVIATION,
)
from dashboard.backend.domain.backtesting.market_rules import MarketRuleDataError
from dashboard.backend.infrastructure.market_data.provider import (
    ALPACA,
    create_market_data_provider,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    IFIND_ASHARE,
    LLM_DECISION_SOURCE,
    RULE_BASED_DECISION_SOURCE,
    MarketProfile,
    get_market_profile,
    resolve_decision_source,
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

# Hosted-runtime failure tolerance. A hosted decision crosses a subprocess
# boundary and a third-party API, so one timeout must not discard a run's worth
# of completed steps -- but a run where most steps failed is not the run the
# agent describes, and persisting it as one is the "fail-closed is not
# fail-visible" trap. Absorb a minority as holds, abort past that.
HOSTED_RUNTIME_MAX_FAILURE_RATIO = 0.2
HOSTED_RUNTIME_MIN_FAILURE_BUDGET = 2

# Rejected-order audit records are per-step, unbounded, and land in the
# agent_runs.metadata JSON cell -- which for a run-history Postgres deployment
# is a row in the free-tier ATL-runs-main project. A 20-symbol A-share run can
# emit thousands. Persist a bounded head sample plus the true total, the same
# shape runtime_step_failures already uses below, so the cap is never silent.
# Head rather than tail, matching runtime_step_failure_samples: the T+1 pattern
# repeats, so the earliest records characterise it and the count carries scale.
# Also bounds the (already much smaller) t1_deferrals sample.
REJECTED_ORDER_SAMPLE_LIMIT = 200
# The live-progress file is rewritten in full on every step, so embedding the
# whole growing list makes write volume quadratic in the run length. The live
# view only ever shows the latest activity, so carry a tail window.
LIVE_PROGRESS_REJECTED_ORDER_LIMIT = 50
LIVE_PROGRESS_ORDER_EVENT_LIMIT = 50


def _unfilled_order_events(order_events) -> List[Dict]:
    """Return only the order outcomes ``trades`` cannot already reconstruct.

    The executor keeps a complete order ledger, but a *fully filled* order is
    byte-for-byte recoverable from the trade it produced -- which is already
    persisted, relationally and without a cap, in the ``trades`` table. Copying
    fills into ``agent_runs.metadata`` as well would duplicate the run's single
    largest table into one JSON cell on free-tier Postgres, carrying the whole
    ``[LLM] <reasoning>`` prose a second time.

    Worse, it would make the cap *lossy*: the sample is bounded at
    ``REJECTED_ORDER_SAMPLE_LIMIT``, so a busy run's fills would push its
    rejections out of the persisted window entirely, and the Trading Log would
    silently show the oldest 200 orders of a run that placed thousands.

    Keeping only the non-filled outcomes makes the stored set small, bounded in
    practice, and strictly additive to ``trades``: the UI reassembles the full
    order history by merging the two, so nothing is hidden. Partial fills stay
    here because their shortfall (``unfilled_shares`` + ``reason``) is exactly
    the part no trade row records.
    """
    return [
        event for event in (order_events or [])
        if event.get("status") != "filled"
    ]


def _prior_market_date_by_decision_date(
    timestamps,
) -> Dict[date, Optional[date]]:
    """Map each ATL trading date to its latest strictly earlier trading date."""
    market_dates = sorted({timestamp.date() for timestamp in timestamps})
    return {
        decision_date: market_dates[index - 1] if index else None
        for index, decision_date in enumerate(market_dates)
    }


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
        universe: Optional[str] = None,
        decision_source: Optional[str] = None,
        runtime_type: str = DEFAULT_RUNTIME_TYPE,
        runtime_config: Optional[Dict] = None,
        execution_client: Any = None,
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
        self.rejected_orders: List[Dict] = []
        self.order_events: List[Dict] = []
        self.market_rule_calendar = None
        self.t1_deferrals: List[Dict] = []
        # Model id; defaults to the gateway-appropriate slug (CommonStack vs native).
        self.model = model or default_model_name()
        self.execution_client = execution_client
        self.live_run_id = (live_run_id or "").strip() or None
        self.progress_file = (progress_file or "").strip() or None
        self.data_source = data_source
        self.runtime_type = normalize_runtime_type(runtime_type)
        self.runtime_config = normalize_runtime_config(
            self.runtime_type, runtime_config or {}
        )
        # The concrete runtime is constructed here, the single production caller,
        # rather than inside RuntimeDispatcher: domain/agents/runtime.py must not
        # import an infrastructure adapter, and this file already depends on
        # infrastructure for market data and LLM access.
        self.runtime_dispatcher = (
            RuntimeDispatcher(
                self.runtime_type,
                self.runtime_config,
                runtime=AiHedgeFundRuntime(self.runtime_config),
            )
            if self.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE
            else None
        )
        # Transient hosted-runtime failures absorbed by this run, kept for the
        # run metadata so a partially-degraded run is legible after the fact.
        self.runtime_step_failures: List[str] = []
        self.profile = get_market_profile(data_source, universe)
        self.currency_context: CurrencyContext | None = None
        self.native_initial_capital = self.initial_capital
        if self.profile.native_currency == self.profile.reporting_currency:
            self.currency_context = CurrencyContext.identity(
                self.profile.native_currency,
                self.profile.timezone,
            )
        requested_decision_source = (
            decision_source
            if decision_source is not None
            else (None if use_llm else RULE_BASED_DECISION_SOURCE)
        )
        self.decision_source = resolve_decision_source(
            self.profile,
            requested_decision_source,
        )
        self.strict_llm = bool(execution_client) or (
            decision_source is not None
            and data_source == IFIND_ASHARE
            and self.decision_source == LLM_DECISION_SOURCE
        )
        if self.runtime_type != PIPELINE_RUNTIME_TYPE:
            self.strict_llm = False
        # iFinD is backend-owned; US providers can use the selected run assets.
        if data_source == IFIND_ASHARE:
            self.symbols = self.profile.symbols
        elif symbols:
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
        wants_llm = self.decision_source == LLM_DECISION_SOURCE
        self.use_llm = wants_llm and (execution_client is not None or HAS_ANTHROPIC)
        if self.runtime_type != PIPELINE_RUNTIME_TYPE:
            self.use_llm = False
        self.llm_client = None

        if (
            self.runtime_type == PIPELINE_RUNTIME_TYPE
            and wants_llm
            and not HAS_ANTHROPIC
            and execution_client is None
        ):
            if self.strict_llm:
                raise llm_harness.LLMConfigurationError(
                    "LLM client is unavailable because the required SDK is not installed"
                )
            self.decision_source = RULE_BASED_DECISION_SOURCE

        # A worker handoff supplies the provider-neutral compatibility client.
        # Legacy direct LLM setup remains available only for non-worker callers.
        if self.use_llm and execution_client is not None:
            self.llm_client = execution_client
            print(f"✅ Unified LLM execution initialized (model={self.model})")

        # Initialize legacy LLM client if enabled. Prefer CommonStack (the model we host)
        # via make_llm_client(); it falls back to native Anthropic when only
        # ANTHROPIC_API_KEY is set, and returns None when no key/SDK is available.
        if self.use_llm and execution_client is None:
            self.llm_client = make_llm_client()
            if self.llm_client is None:
                if self.strict_llm:
                    raise llm_harness.LLMConfigurationError(
                        "LLM client is unavailable for the requested decision source"
                    )
                print(
                    "⚠️  No LLM key (COMMONSTACK_API_KEY / OPENROUTER_API_KEY / "
                    "ANTHROPIC_API_KEY) set. Running without LLM."
                )
                self.use_llm = False
                self.decision_source = RULE_BASED_DECISION_SOURCE
            else:
                print(f"✅ LLM initialized (model={self.model})")

        self.data_loader = create_market_data_provider(
            data_source,
            self.profile.universe,
        )
    
    def _serialize_trades(self, trades: List[Dict]) -> List[Dict]:
        serialized = []
        currency_context = self._require_currency_context()
        for native_trade in trades:
            trade = currency_context.reporting_trade(native_trade)
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
            record = {
                "timestamp": ts,
                "symbol": trade.get("symbol"),
                "side": side,
                "quantity": quantity,
                "price": price,
                "value": value,
                "reason": trade.get("reason", ""),
            }
            for field in (
                "reference_price",
                "gross_value",
                "slippage_amount",
                "commission",
                "stamp_duty",
                "transfer_fee",
                "total_fees",
                "net_cash_impact",
            ):
                if field in trade:
                    record[field] = float(trade[field])
            for field in (
                "market_rule_date",
                "market_rule_suspended",
                "market_rule_closing_limit_state",
                "market_rule_closing_gate_effective",
            ):
                if field in trade:
                    record[field] = trade[field]
            if trade.get("market_rule_official_close") is not None:
                record["market_rule_official_close"] = float(
                    trade["market_rule_official_close"]
                )
            for field in (
                # No native_executed_value here: executed_value belongs to an
                # order event, never to a trade row, and the column list the
                # trades table persists does not carry it either.
                "native_price",
                "native_reference_price",
                "native_value",
                "native_gross_value",
                "native_slippage_amount",
                "native_commission",
                "native_stamp_duty",
                "native_transfer_fee",
                "native_total_fees",
                "native_net_cash_impact",
                "fx_rate",
            ):
                if field in trade:
                    record[field] = float(trade[field])
            serialized.append(record)
        return serialized

    @staticmethod
    def _serialize_rejected_orders(rejected_orders: List[Dict]) -> List[Dict]:
        """JSON-safe rejected-order records.

        Static where the sibling ``_serialize_trades`` is an instance method,
        and deliberately so: trades carry prices and values that must be
        converted through ``self._require_currency_context()``, while a rejected
        order records only share counts, which are currency-free. Taking ``self``
        here just to look symmetric would imply a conversion that does not exist.
        """
        serialized = []
        for item in rejected_orders:
            record = dict(item)
            timestamp = record.get("timestamp")
            if hasattr(timestamp, "isoformat"):
                record["timestamp"] = timestamp.isoformat()
            for field in (
                "requested_shares",
                "executed_shares",
                "unfilled_shares",
            ):
                value = record.get(field)
                if hasattr(value, "item"):
                    record[field] = value.item()
            serialized.append(record)
        return serialized

    def _serialize_order_events(self, order_events: List[Dict]) -> List[Dict]:
        """Convert order outcomes to reporting currency and JSON-safe values."""
        serialized = []
        currency_context = self._require_currency_context()
        for native_event in order_events:
            record = currency_context.reporting_order_event(native_event)
            timestamp = record.get("timestamp")
            if hasattr(timestamp, "isoformat"):
                record["timestamp"] = timestamp.isoformat()
            for field in (
                "requested_shares",
                "executed_shares",
                "unfilled_shares",
            ):
                value = record.get(field)
                if hasattr(value, "item"):
                    record[field] = value.item()
            for field in (
                "price",
                "reference_price",
                "executed_value",
                "gross_value",
                "slippage_amount",
                "commission",
                "stamp_duty",
                "transfer_fee",
                "total_fees",
                "net_cash_impact",
                "native_price",
                "native_reference_price",
                "native_value",
                "native_executed_value",
                "native_gross_value",
                "native_slippage_amount",
                "native_commission",
                "native_stamp_duty",
                "native_transfer_fee",
                "native_total_fees",
                "native_net_cash_impact",
                "fx_rate",
            ):
                if field in record:
                    record[field] = float(record[field])
            if record.get("market_rule_official_close") is not None:
                record["market_rule_official_close"] = float(
                    record["market_rule_official_close"]
                )
            serialized.append(record)
        return serialized

    @staticmethod
    def _serialize_t1_deferrals(t1_deferrals: Dict) -> List[Dict]:
        """JSON-safe T+1 deferral records, oldest symbol-day first.

        Static for the same reason as ``_serialize_rejected_orders``: share
        counts carry no currency. Sorted so the persisted sample is a stable
        prefix of the run rather than dict-insertion order.
        """
        serialized = []
        for record in sorted(
            t1_deferrals.values(),
            key=lambda item: (item["date"], item["symbol"]),
        ):
            item = dict(record)
            date_value = item.get("date")
            if hasattr(date_value, "isoformat"):
                item["date"] = date_value.isoformat()
            for field in (
                "requested_shares",
                "sellable_shares",
                "deferred_shares",
            ):
                value = item.get(field)
                if hasattr(value, "item"):
                    item[field] = value.item()
            serialized.append(item)
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
                    **{
                        field: float(entry[field])
                        for field in (
                            "native_equity",
                            "native_cash",
                            "native_positions_value",
                            "fx_rate",
                        )
                        if field in entry
                    },
                }
            )
        unfilled_order_events = _unfilled_order_events(
            getattr(manager, "order_events", [])
        )
        payload = {
            "run_id": self.live_run_id,
            "step": step,
            "total_steps": total_steps,
            "equity_curve": serialized,
            "trades": self._serialize_trades(manager.trades),
            "rejected_orders": self._serialize_rejected_orders(
                manager.rejected_orders[-LIVE_PROGRESS_REJECTED_ORDER_LIMIT:]
            ),
            "rejected_orders_count": len(manager.rejected_orders),
            # Non-filled outcomes only; the complete `trades` list above already
            # carries every fill, and the frontend merges the two. Tail window
            # for the same reason rejected_orders uses one -- this file is
            # rewritten whole on every step.
            "order_events": self._serialize_order_events(
                unfilled_order_events[-LIVE_PROGRESS_ORDER_EVENT_LIMIT:]
            ),
            "order_events_count": len(unfilled_order_events),
        }
        try:
            Path(self.progress_file).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            print(f"   ⚠️  Could not write live progress: {exc}")
    
    def load_data(self):
        """Fetch hourly data from the selected normalized provider."""
        # Keep the error path usable for legacy callers that construct an
        # instance with ``__new__`` (or inject a loader) before initialization.
        symbols = getattr(self, "symbols", ())
        print(
            f"   Universe: {len(symbols)} symbols ({', '.join(symbols[:8])}"
            f"{'…' if len(symbols) > 8 else ''})"
        )
        self.all_data = self.data_loader.fetch_bars(
            symbols, self.start_date, self.end_date
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
        if self.data_source == IFIND_ASHARE:
            self._ifind_common_start = self._validate_ifind_loaded_data()
            self._initialize_ifind_market_rules()
            self._initialize_ifind_currency_context()

    def _initialize_ifind_market_rules(self) -> None:
        """Load and validate official rules before any order can execute."""
        if not hasattr(self.data_loader, "fetch_market_rules"):
            raise MarketDataUnavailableError(
                "Market rule data unavailable: iFinD provider has no rule capability"
            )
        try:
            self.market_rule_calendar = self.data_loader.fetch_market_rules(
                self.symbols,
                self.start_date,
                self.end_date,
                bars_by_symbol=self.all_data,
            )
        except (IFindClientError, MarketRuleDataError, ValueError) as exc:
            raise MarketDataUnavailableError(
                f"Market rule data unavailable: {exc}"
            ) from exc

    def _initialize_ifind_currency_context(self) -> None:
        """Load the iFinD historical conversion series for this A-share run."""
        # Assert the capability instead of catching AttributeError around the
        # whole block: a bare AttributeError catch also swallows typos inside
        # it and reports them as a credentials problem.
        if not hasattr(self.data_loader, "fetch_usd_cny"):
            raise MarketDataUnavailableError(
                f"{self.data_source} market data provider cannot supply the "
                "historical conversion rate an A-share run requires"
            )
        try:
            rates = self.data_loader.fetch_usd_cny(
                self.symbols,
                self.start_date,
                self.end_date,
            )
            context = CurrencyContext(
                native_currency=self.profile.native_currency,
                reporting_currency=self.profile.reporting_currency,
                timezone=self.profile.timezone,
                rates=rates,
                fx_source="ifind_history_currency_conversion",
                fx_policy="daily_implied_median_forward_fill",
            )
            context.rate_at(self._ifind_common_start)
        except IFindClientError as exc:
            # Transport/auth/business failures really are a credentials or
            # permission story; the client's message is already sanitized.
            raise MarketDataUnavailableError(
                "iFinD historical conversion rate is unavailable; check the "
                "historical quotation currency permission, token, and date range"
            ) from exc
        except (IFindFxError, CurrencyContextError) as exc:
            # These say *why* the series is unusable (gaps, disagreement, no
            # rate before the first bar). Both message families are built from
            # counts only, so they are safe to surface verbatim — and pointing
            # the operator at credentials here would send them to the wrong place.
            raise MarketDataUnavailableError(
                f"iFinD historical conversion rate is unusable: {exc}"
            ) from exc

        self.currency_context = context
        self.native_initial_capital = context.to_native(
            self.initial_capital,
            self._ifind_common_start,
        )

    def _require_currency_context(self) -> CurrencyContext:
        context = getattr(self, "currency_context", None)
        if context is None:
            raise MarketDataUnavailableError(
                "Currency context is not initialized; load market data before running"
            )
        return context

    def _validate_ifind_loaded_data(self):
        """Require an exact, sufficiently deep A-share batch with a common bar."""
        expected = tuple(self.symbols)
        actual = tuple(self.all_data)
        expected_set = set(expected)
        actual_set = set(actual)
        if len(actual) != len(expected) or actual_set != expected_set:
            missing = [symbol for symbol in expected if symbol not in actual_set]
            unexpected = [symbol for symbol in actual if symbol not in expected_set]
            raise MarketDataUnavailableError(
                "iFinD provider result is incomplete: "
                f"missing={missing!r} unexpected={unexpected!r}"
            )

        short = {
            symbol: len(self.all_data[symbol])
            for symbol in expected
            if len(self.all_data[symbol]) < 50
        }
        if short:
            raise MarketDataUnavailableError(
                f"iFinD symbols have fewer than 50 bars: {short!r}"
            )

        common_index = self.all_data[expected[0]].index
        for symbol in expected[1:]:
            common_index = common_index.intersection(self.all_data[symbol].index)
            if common_index.empty:
                break
        if common_index.empty:
            raise MarketDataUnavailableError(
                "iFinD symbols do not share a common timestamp for baseline start"
            )
        return common_index.min()
    
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
    
    def _effective_profile(self) -> MarketProfile:
        profile = getattr(self, "profile", None)
        if profile is not None:
            return profile
        return get_market_profile(self.data_source)

    def _run_metadata(
        self,
        transaction_cost_totals: Optional[Dict] = None,
        costs_applied: bool = False,
        baseline_allocation: Optional[Dict] = None,
    ) -> Dict:
        """Data provenance recorded on EVERY run row, agent and baseline alike.

        Provenance is the only thing the baselines share with the agent: they
        make no model calls and run no pipeline, so anything LLM-shaped belongs
        in ``_agent_run_metadata`` instead.

        ``costs_applied`` separates the market's rule from this run's ledger.
        The cost profile is provenance — it describes the market and belongs on
        every row of it, exactly like ``lot_size``. Whether the run actually
        paid those fees is a different fact: the index reference curve is a
        price series that never places an order, so it must not read as a
        costed book.
        """
        profile = self._effective_profile()
        metadata = {
            "data_source": self.data_source,
            "symbols": list(self.symbols),
            "native_currency": profile.native_currency,
            "reporting_currency": profile.reporting_currency,
            "lot_size": profile.lot_size,
        }
        if profile.transaction_cost_profile is not None:
            metadata["transaction_cost_profile"] = (
                profile.transaction_cost_profile.to_metadata()
            )
            metadata["transaction_costs_applied"] = bool(costs_applied)
            if costs_applied and transaction_cost_totals and any(
                float(value or 0) != 0 for value in transaction_cost_totals.values()
            ):
                metadata["transaction_cost_totals"] = dict(transaction_cost_totals)
        if baseline_allocation:
            metadata["baseline_allocation"] = dict(baseline_allocation)
        if self.data_source == IFIND_ASHARE:
            metadata.update(
                {
                    "market": profile.market,
                    "universe": profile.universe,
                    "timeframe": profile.timeframe,
                    "timezone": profile.timezone,
                    # Baselines make no model calls, so rule_based is the honest
                    # value here. Agent runs overwrite it in _agent_run_metadata
                    # with the decision source they actually resolved.
                    "decision_source": RULE_BASED_DECISION_SOURCE,
                    "benchmark": profile.benchmark,
                    # Market provenance, not execution provenance: this records
                    # that the run's market settles T+1, which is true of the
                    # buy-and-hold baseline rows too even though they never
                    # build a T+1 PortfolioManager (they never sell, so the
                    # rule cannot bind). Read it as "which market", not "which
                    # executor ran".
                    "t_plus_one_enabled": profile.t_plus_one_enabled,
                }
            )
            calendar = getattr(self, "market_rule_calendar", None)
            if calendar is not None:
                metadata["market_rule_profile"] = calendar.to_metadata()
            context = self._require_currency_context()
            # Frames arrive sorted (the adapter rejects non-increasing
            # timestamps), so read the ends instead of materializing every
            # timestamp across every symbol just to take min/max.
            first_timestamp = min(frame.index[0] for frame in self.all_data.values())
            last_timestamp = max(frame.index[-1] for frame in self.all_data.values())
            metadata.update(
                {
                    "fx_pair": context.fx_pair,
                    "fx_source": context.fx_source,
                    "fx_policy": context.fx_policy,
                    "fx_symbols": list(self.symbols),
                    "fx_max_relative_deviation": MAX_RELATIVE_DEVIATION,
                    "fx_start_rate": context.rate_at(first_timestamp),
                    "fx_end_rate": context.rate_at(last_timestamp),
                    "fx_market_start_date": context.market_date(first_timestamp).isoformat(),
                    "fx_market_end_date": context.market_date(last_timestamp).isoformat(),
                    "fx_observation_start_date": min(context.rates).isoformat(),
                    "fx_observation_end_date": max(context.rates).isoformat(),
                    "native_initial_capital": self.native_initial_capital,
                }
            )
        return metadata

    def _agent_run_metadata(self) -> Dict:
        """Provenance plus the effective config the agent run actually used.

        LLM_MAX_OUTPUT_TOKENS is an env knob that changes a run's spend and
        response truncation; recording the EFFECTIVE value (post defensive
        parse) makes runs auditable after the env changes."""
        # An agent run always executes through the cost path when the market
        # defines one, so the profile is the honest gate here. Keying this on
        # the data source instead would silently drop the ledger the day a
        # second costed market appears.
        profile = self._effective_profile()
        meta: Dict = self._run_metadata(
            transaction_cost_totals=getattr(self, "transaction_cost_totals", None),
            costs_applied=profile.transaction_cost_profile is not None,
        )
        decision_source = getattr(self, "decision_source", None)
        if decision_source is not None:
            meta["decision_source"] = decision_source
        if self.use_llm:
            meta["llm_max_output_tokens"] = llm_harness.DEFAULT_MAX_OUTPUT_TOKENS
        llm_execution = getattr(self, "_llm_execution_evidence", None)
        execution_client = getattr(self, "execution_client", None)
        if llm_execution is None and execution_client is not None:
            summary = getattr(execution_client, "execution_summary", None)
            if callable(summary):
                llm_execution = summary()
        if llm_execution is not None:
            meta["llm_execution"] = llm_execution.model_dump(mode="json")
        if self.prompt_adaptations:
            meta["prompt_adaptations"] = self.prompt_adaptations
        if self.initial_pipeline is not None:
            meta["initial_pipeline"] = self.initial_pipeline
        if self.pipeline is not None:
            meta["final_pipeline"] = self.pipeline
        if self.data_source == IFIND_ASHARE:
            rejected = list(getattr(self, "rejected_orders", []) or [])
            if rejected:
                # Count first, sample second: a consumer must be able to tell
                # "3 rejections" from "the first 200 of 7,000".
                meta["rejected_orders_count"] = len(rejected)
                meta["rejected_orders"] = rejected[:REJECTED_ORDER_SAMPLE_LIMIT]
                truncated = len(rejected) - REJECTED_ORDER_SAMPLE_LIMIT
                if truncated > 0:
                    meta["rejected_orders_truncated"] = truncated
                market_rule_rejections = {
                    reason: sum(1 for item in rejected if item.get("reason") == reason)
                    for reason in (
                        "suspended",
                        "limit_up_buy_blocked",
                        "limit_down_sell_blocked",
                        "market_rule_unavailable",
                    )
                }
                market_rule_rejections = {
                    reason: count
                    for reason, count in market_rule_rejections.items()
                    if count
                }
                if market_rule_rejections:
                    meta["market_rule_rejections"] = market_rule_rejections
            deferrals = list(getattr(self, "t1_deferrals", []) or [])
            if deferrals:
                # "How often did T+1 stop this agent exiting?" — the question a
                # capped order can no longer answer, because it fills exactly
                # and leaves the executor nothing to audit.
                meta["t1_deferred_events"] = len(deferrals)
                meta["t1_deferred_shares"] = sum(
                    item["deferred_shares"] for item in deferrals
                )
                meta["t1_deferrals"] = deferrals[:REJECTED_ORDER_SAMPLE_LIMIT]
                deferrals_truncated = len(deferrals) - REJECTED_ORDER_SAMPLE_LIMIT
                if deferrals_truncated > 0:
                    meta["t1_deferrals_truncated"] = deferrals_truncated
        # Already filtered to non-filled outcomes by run_agent_backtest, so this
        # list is small: fills live in `trades`, and repeated rejections were
        # collapsed per symbol-trading-day. Head sample, matching
        # rejected_orders -- the earliest outcomes characterise the run, and the
        # count carries scale. That is deliberately the opposite end from the
        # live tail window: a live viewer wants the latest activity, a reader
        # of a finished run wants the start of the story. The Trading Log
        # labels whichever end it is showing, so neither reads as complete.
        order_events = list(getattr(self, "order_events", []) or [])
        if order_events:
            meta["order_events_count"] = len(order_events)
            meta["order_events"] = order_events[:REJECTED_ORDER_SAMPLE_LIMIT]
            order_events_truncated = len(order_events) - REJECTED_ORDER_SAMPLE_LIMIT
            if order_events_truncated > 0:
                meta["order_events_truncated"] = order_events_truncated
        runtime_type = getattr(self, "runtime_type", PIPELINE_RUNTIME_TYPE)
        if runtime_type != PIPELINE_RUNTIME_TYPE:
            meta["runtime_type"] = runtime_type
            meta["runtime_config"] = dict(getattr(self, "runtime_config", {}) or {})
            meta["runtime_calls"] = self.runtime_dispatcher.calls
            # Absorbing failures as holds keeps a run alive, but a run that
            # held its way through half the period must not read as a clean
            # one. Record the count (and a bounded sample) on the run itself.
            if self.runtime_step_failures:
                meta["runtime_step_failures"] = len(self.runtime_step_failures)
                meta["runtime_step_failure_samples"] = [
                    text[:300] for text in self.runtime_step_failures[:5]
                ]
        return meta

    def _llm_market_context(self) -> Dict:
        """Return the fixed market facts supplied to every LLM decision."""
        context = self._require_currency_context()
        result = {
            "market": self.profile.market,
            "timezone": self.profile.timezone,
            "timeframe": self.profile.timeframe,
            "symbols": list(self.symbols),
            "paper_backtest": True,
            "native_currency": self.profile.native_currency,
            "reporting_currency": self.profile.reporting_currency,
        }
        if self.profile.lot_size > 1:
            # Conditional for the same reason `settlement` below is: this dict
            # is serialized straight into the LLM prompt, so an unconditional
            # key changes every DJIA prompt and makes new runs non-comparable
            # with the historical ones on the leaderboard. A market that trades
            # in single shares has no lot rule to state.
            result["lot_size"] = self.profile.lot_size
            result["lot_size_note"] = (
                "Order quantities must be positive whole multiples of "
                f"{self.profile.lot_size} shares; any other size is rejected "
                "in full rather than rounded."
            )
        if self.profile.t_plus_one_enabled:
            # A bare `sellable_shares` number in each holding is not
            # self-explanatory. Name the rule that produces it, or the model has
            # to infer a settlement regime from an unlabelled integer.
            result["settlement"] = "T+1"
            result["settlement_note"] = (
                "Shares bought today cannot be sold until the next trading day. "
                "Each holding reports sellable_shares; a sell above that amount "
                "is truncated to it."
            )
        if context.requires_conversion:
            result["fx_pair"] = context.fx_pair
            result["fx_source"] = "iFinD Historical Conversion Rate"
        return result

    def _current_equity(self, manager: PortfolioManager, timestamp=None) -> float:
        if manager.equity_history:
            equity = manager.equity_history[-1].get("equity")
            if equity is not None:
                return float(equity)
        if timestamp is not None:
            return self._require_currency_context().to_reporting(manager.cash, timestamp)
        return float(manager.cash)

    def _market_hours_only(self, timestamps):
        """Filter timestamps using the selected market's local sessions."""
        import pytz

        profile = self._effective_profile()
        market_tz = pytz.timezone(profile.timezone)
        kept = []
        for timestamp in timestamps:
            local = timestamp.astimezone(market_tz)
            local_time = local.time()
            if profile.market == "CN":
                is_market_hours = (
                    time(9, 30) <= local_time <= time(11, 30)
                    or time(13, 0) <= local_time <= time(15, 0)
                )
            else:
                is_market_hours = (
                    (local.hour > 9 and local.hour < 16)
                    or (local.hour == 9 and local.minute >= 30)
                    or (local.hour == 16 and local.minute == 0)
                )
            if is_market_hours:
                kept.append(timestamp)
        return kept

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
        llm_model = "rule-based"  # Default; hosted runs are attributed below,
        # after the loop, from the calls that actually succeeded.

        manager = PortfolioManager(
            initial_capital=self.native_initial_capital,
            allowed_symbols=self.symbols,
            t_plus_one_enabled=self.profile.t_plus_one_enabled,
            lot_size=self.profile.lot_size,
            transaction_cost_profile=self.profile.transaction_cost_profile,
            market_rule_calendar=self.market_rule_calendar,
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
        
        all_timestamps = self._market_hours_only(all_timestamps)
        prior_market_dates = (
            _prior_market_date_by_decision_date(all_timestamps)
            if self.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE
            else {}
        )
        # Hosted runtimes decide once per trading day, so the budget is a share
        # of decision days rather than of hourly bars.
        runtime_failure_budget = max(
            HOSTED_RUNTIME_MIN_FAILURE_BUDGET,
            int(len(prior_market_dates) * HOSTED_RUNTIME_MAX_FAILURE_RATIO),
        )
        if self.runtime_type != PIPELINE_RUNTIME_TYPE:
            print(
                f"   Hosted runtime: holding on up to {runtime_failure_budget} "
                f"failed step(s) before aborting\n"
            )

        print(
            f"   Trading {len(all_timestamps)} bars during "
            f"{self.profile.market} {self.profile.timeframe} sessions...\n"
        )
        total_steps = len(all_timestamps)
        # Declare the run length so a strict-LLM run can absorb a small number
        # of unusable responses instead of discarding hours of work on the
        # first one. Left unset the budget is 0 (fatal on the first strike).
        manager.strict_llm_total_steps = total_steps
        if self.strict_llm:
            print(
                "   Strict LLM: aborting after more than "
                f"{manager.strict_llm_fallback_budget()} unusable response(s)\n"
            )

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
                    "day_start_equity": self._current_equity(manager, timestamp),
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
            runtime_invoked = False
            
            # Keep the established pipeline execution path unchanged. Hosted
            # runtimes alone cross the runtime-dispatch boundary, then return
            # the same ATL action envelope for PortfolioManager to execute.
            if self.runtime_type == PIPELINE_RUNTIME_TYPE:
                if self.use_llm and self.llm_client:
                    decision = manager.make_trading_decision_with_llm(
                        state,
                        self.llm_client,
                        mode=self.mode,
                        model=self.model,
                        strategy_prompt=self.strategy_prompt,
                        pipeline=self.pipeline,
                        market_context=self._llm_market_context(),
                        strict_llm=self.strict_llm,
                    )
                    llm_calls_count += 1  # Track that LLM was used
                    if llm_calls_count == 1:  # Set on first call
                        llm_model = self.model
                    if manager.last_pipeline_step_outputs:
                        day_episode["latest_step_outputs"] = manager.last_pipeline_step_outputs
                else:
                    decision = manager.make_trading_decision(state)
            else:
                runtime_calls_before = self.runtime_dispatcher.calls
                runtime_context = AgentRuntimeContext(
                    timestamp=timestamp,
                    backtest_start_date=self.start_date,
                    symbols=list(self.symbols),
                    cash=float(manager.cash),
                    total_equity=float(state["total_equity"]),
                    positions=dict(manager.positions),
                    entry_prices=dict(manager.entry_prices),
                    current_prices={
                        symbol: float(row["close"])
                        for symbol, row in market_data.items()
                    },
                    latest_market_date_before_decision=prior_market_dates.get(
                        timestamp.date()
                    ),
                    market=self._llm_market_context(),
                )
                try:
                    decision = self.runtime_dispatcher.dispatch(
                        runtime_context,
                        pipeline_handler=lambda: manager.make_trading_decision(state),
                    )
                except AgentRuntimeConfigurationError:
                    # Deployment-level and identical on every step. Surface it
                    # on the first one instead of holding through the run.
                    raise
                except AgentRuntimeError as exc:
                    self.runtime_step_failures.append(
                        f"{timestamp.isoformat()}: {exc}"
                    )
                    failures = len(self.runtime_step_failures)
                    print(
                        f"   ⚠️  Runtime step failed ({failures}/"
                        f"{runtime_failure_budget} tolerated): {exc}",
                        flush=True,
                    )
                    if failures > runtime_failure_budget:
                        raise AgentRuntimeError(
                            f"{self.runtime_type} runtime failed {failures} step(s), "
                            f"over the {runtime_failure_budget} tolerated for this "
                            f"run; last error: {exc}"
                        ) from exc
                    # Hold this step. Trading on a stale view would be worse
                    # than not trading, and the run keeps its completed steps.
                    decision = {"actions": []}
                runtime_invoked = self.runtime_dispatcher.calls > runtime_calls_before
            
            # Execute trades (only if real data available)
            trades_before_execution = len(manager.trades)
            fallback_prices = {
                symbol: values[timestamp]
                for symbol, values in price_cache.items()
                if timestamp in values
            }
            manager.execute_actions(
                decision["actions"],
                market_data,
                timestamp,
                fallback_prices=fallback_prices,
            )
            if runtime_invoked:
                self.runtime_dispatcher.record_latest_execution(
                    len(manager.trades) - trades_before_execution
                )
            
            # Update equity (uses forward-filled prices for smooth valuation)
            manager.update_equity(market_data, price_cache, timestamp)
            manager.equity_history[-1] = (
                self._require_currency_context().reporting_equity_record(
                    manager.equity_history[-1]
                )
            )
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

        # Attribute a hosted run to its model only if the model actually drove
        # steps. A row naming a model beside ``llm_calls=0`` is precisely the
        # shape the H6 integrity guard reads as a rule-based curve wearing an
        # LLM's name, so persist the honest label when every step held.
        runtime_calls = (
            self.runtime_dispatcher.calls
            if self.runtime_type != PIPELINE_RUNTIME_TYPE
            else 0
        )
        if self.runtime_type != PIPELINE_RUNTIME_TYPE:
            llm_model = (
                str(self.runtime_dispatcher.model_name or "unknown")
                if runtime_calls
                else "rule-based"
            )
        # Upstream does not report token usage back across the bridge, so a
        # hosted run's tokens and cost stay 0. The call count is what makes the
        # row self-consistent rather than silently understating the run.
        llm_calls_total = manager.llm_calls + runtime_calls

        llm_execution = None
        execution_client = getattr(self, "execution_client", None)
        if execution_client is not None:
            summary = getattr(execution_client, "execution_summary", None)
            if callable(summary):
                llm_execution = summary()
        if llm_execution is not None:
            est_cost = (
                llm_execution.provider_cost_usd
                if llm_execution.provider_cost_usd is not None
                else llm_execution.estimated_cost_usd
            )
            est_cost = float(est_cost or 0)
        else:
            est_cost = token_cost.estimate_cost_usd(
                llm_model, manager.input_tokens, manager.output_tokens
            )
        self._llm_execution_evidence = llm_execution

        self.t1_deferrals = self._serialize_t1_deferrals(manager.t1_deferrals)
        self.rejected_orders = self._serialize_rejected_orders(
            manager.rejected_orders
        )
        self.order_events = self._serialize_order_events(
            _unfilled_order_events(manager.order_events)
        )
        self.transaction_cost_totals = dict(manager.transaction_cost_totals)
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
            llm_calls=llm_calls_total,
            input_tokens=manager.input_tokens,
            output_tokens=manager.output_tokens,
            est_cost_usd=est_cost,
            metadata=self._agent_run_metadata(),
        )

        db.insert_equity_points(run_id, equity_curve)
        db.insert_trades(run_id, self._serialize_trades(manager.trades))
        if self.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE:
            db.insert_decisions(run_id, self.runtime_dispatcher.decision_audit_rows)
        
        print(f"\n  ✅ Agent backtest complete")
        print(f"     • Run ID: {run_id}")
        if self.runtime_type == PIPELINE_RUNTIME_TYPE:
            model_display = self.model if llm_calls_count > 0 else "rule-based"
            print(f"     • Model: {model_display} (✅ LLM enabled)" if llm_calls_count > 0 else f"     • Model: {model_display} (❌ fallback)")
            print(f"     • LLM Calls: {llm_calls_count}")
        else:
            print(f"     • Model: {llm_model} (✅ agent runtime)" if runtime_calls else f"     • Model: {llm_model} (❌ fallback)")
            print(f"     • Runtime calls: {runtime_calls}")
            if self.runtime_step_failures:
                print(
                    f"     • Runtime steps held after failure: "
                    f"{len(self.runtime_step_failures)}"
                )
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
        
        bars = self.all_data
        if self.data_source == IFIND_ASHARE:
            common_start = getattr(self, "_ifind_common_start", None)
            if common_start is None:
                common_start = self._validate_ifind_loaded_data()
            bars = {
                symbol: self.all_data[symbol].loc[common_start:]
                for symbol in self.symbols
            }

        profile = self._effective_profile()
        baseline_cost_totals: Dict[str, float] = {}
        baseline_allocation: Dict[str, Any] = {}
        equity_history, _ = generate_baselines(
            bars_by_symbol=bars,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            symbols_list=bh_symbols,
            market_timezone=profile.timezone,
            currency_context=self._require_currency_context(),
            transaction_cost_profile=profile.transaction_cost_profile,
            transaction_cost_totals=baseline_cost_totals,
            # The board lot is its own market rule. Deriving it from the cost
            # profile would floor buys to 100 in any future market that charges
            # fees but trades in single shares.
            lot_size=profile.lot_size,
            allocation_summary=baseline_allocation,
            market_rule_calendar=self.market_rule_calendar,
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
            metadata=self._run_metadata(
                baseline_cost_totals,
                costs_applied=profile.transaction_cost_profile is not None,
                baseline_allocation=baseline_allocation,
            ),
        )
        
        db.insert_equity_points(run_id, equity_history)
        
        print(f"  ✅ Buy & Hold baseline complete")
        print(f"     • Run ID: {run_id}")
        print(f"     • Final: ${final_eq:,.0f}")
        print(f"     • Return: {total_return*100:+.2f}%\n")
        
        return run_id, equity_history
    
    def run_djia_baseline(self) -> Tuple[str, List[Dict]]:
        """DJIA index baseline using shared baseline generator."""
        if not self._effective_profile().index_baseline_enabled:
            return None, []

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
            market_timezone=self._effective_profile().timezone,
            symbols_list=list(DJIA_30),
            currency_context=self._require_currency_context(),
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
