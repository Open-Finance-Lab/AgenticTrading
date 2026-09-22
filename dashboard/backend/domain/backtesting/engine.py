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

import inspect
import json
import uuid
from bisect import bisect_left
from datetime import date, datetime, time
from math import ceil
# `from time import ...`, not `import time`: line 21's
# `from datetime import date, datetime, time` binds the bare name `time` to
# `datetime.time`, which this module *calls as a constructor* for the A-share
# session bounds (`time(9, 30) <= local_time <= time(11, 30)`, :1209-1210). A
# plain `import time` above it is rebound by that later line, so `time.time()`
# raises AttributeError; below it, those four session bounds silently become
# calls on the time module instead. Green on every US run either way, red only
# on an A-share one.
#
# Two clocks, deliberately. `wall_clock` stamps INSTANTS an operator or another
# process reads -- `started_at`/`ended_at` in `phases[]`, which are compared
# against `--launched-at` and the two stamps the launch script takes. `steady`
# measures DURATIONS, because `time.time()` is not monotonic: an NTP step
# between two reads distorts the interval and a backward step makes it
# negative, and these numbers are published on the card and are the measurement
# the next latency decision is made on.
#
# `starting` is the one phase whose TOTAL cannot move to `steady`, and not for
# a reason worth working around: `monotonic()` has a per-process epoch, so the
# parent's reading and the child's are not comparable at all. That phase opens
# at the parent's `--launched-at` and closes in the child, so the wall clock is
# the only shared reference. It is also the interval least exposed to the
# hazard: a single span bounded by a process spawn, not a loop.
#
# Its SPLITS are not all like that, and reading "the phase is cross-process" as
# "every number under it is" is how the fix for #509 first shipped with two
# wall-clock differences still in it. Only `spawn+interpreter` reaches back
# into the parent; `imports+stores` and `preflight` both begin and end inside
# the child, so both take the steady clock -- bounded by the two `monotonic()`
# marks the launch script hands over in `startup_clock`. See
# `_set_progress_phase`.
from time import monotonic as steady_clock, time as wall_clock
from typing import Any, Dict, List, Optional, Tuple

from dashboard.backend.database import db
import dashboard.backend.infrastructure.llm.token_cost as token_cost
from dashboard.backend.baseline_generator import generate_baselines
from dashboard.backend.infrastructure.llm.validator import DJIA_30, TOP_10_STOCKS as TOP_10
from dashboard.backend.infrastructure.market_data.strategy_universe import (
    resolve_strategy_universe, validate_selection,
)
from dashboard.backend.domain.backtesting.constants import (
    INITIAL_CAPITAL,
    fractional_return,
)
from dashboard.backend.domain.backtesting.currency import (
    CurrencyContext,
    CurrencyContextError,
)
from dashboard.backend.domain.backtesting.features import TechnicalIndicators
from dashboard.backend.domain.backtesting.bar_aggregation import (
    aggregate_bars_by_symbol,
    summarize_aggregation_quality,
)
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
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
    feed_provenance,
)
from dashboard.backend.infrastructure.market_data.equity_metadata import (
    EquityMetadataUnavailableError,
    load_and_enrich_us_equity_bars,
)
from dashboard.backend.infrastructure.market_data.ifind_ashare import (
    minimum_bars_for_window,
)
from dashboard.backend.infrastructure.market_data.ifind_client import IFindClientError
from dashboard.backend.infrastructure.market_data.ifind_fx import (
    IFindFxError,
    MAX_RELATIVE_DEVIATION,
)
from dashboard.backend.domain.backtesting.market_rules import (
    CorporateActionGapError,
    MarketRuleDataError,
)
from dashboard.backend.infrastructure.market_data.provider import (
    ALPACA,
    create_market_data_provider,
)
from dashboard.backend.infrastructure.market_data.frequency import (
    FrequencyConfigError,
    build_verified_intraday_contract,
    normalize_bar_timeframe,
    timeframe_minutes,
    verify_source_timeframe,
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

#: Every phase the child publishes to the progress file, in the order a run
#: passes through them. `running` is set by `_publish_live_progress`; the rest
#: by `publish_phase`. The status route and the card key on these names.
PROGRESS_PHASES = (
    "starting",
    "loading_bars",
    "indicators",
    "first_decision",
    "running",
    "saving",
)


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
        stock_pool: Optional[str] = None,
        pool_mode: Optional[str] = None,
        universe_selection: Optional[Dict] = None,
        source_timeframe: Optional[str] = None,
        launched_at: Optional[float] = None,
        startup_clock: Optional[Dict[str, float]] = None,
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
        self._init_progress_phases(launched_at, startup_clock)
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
        self.requested_source_timeframe = normalize_bar_timeframe(
            source_timeframe
            if source_timeframe is not None
            else getattr(self.profile, "source_timeframe", self.profile.timeframe)
        )
        self.source_timeframe = self.requested_source_timeframe
        self.decision_timeframe = getattr(
            self.profile, "decision_timeframe", self.profile.timeframe
        )
        self.execution_timeframe = getattr(
            self.profile, "execution_timeframe", self.profile.timeframe
        )
        self.valuation_frequency = getattr(
            self.profile, "valuation_frequency", self.profile.timeframe
        )
        self.intraday_mode = False
        self.source_data = {}
        self.data_quality = {}
        self.frequency_contract = None
        self.market_data_provenance = {}
        self.equity_metadata = {}
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
        # The resolved *request*, frozen here because the two availability
        # downgrades below rewrite `self.decision_source` in place and
        # `_agent_run_metadata` persists whatever it holds at the end of the
        # run. Without this the headline case of #169 -- "I asked for a model
        # and every step traded rule-based" -- persisted identically to a run
        # the caller deliberately ordered rule-based, and the surfaces reading
        # the row reported the silent fallback as exactly what was ordered.
        #
        # Resolved rather than raw: `resolve_decision_source` is what validates
        # the request against the market profile and fills in that profile's
        # default, so the raw argument is None on the common path and says
        # nothing about what the run was going to do.
        self.requested_decision_source = self.decision_source
        self.strict_llm = bool(execution_client) or (
            decision_source is not None
            and data_source == IFIND_ASHARE
            and self.decision_source == LLM_DECISION_SOURCE
        )
        if self.runtime_type != PIPELINE_RUNTIME_TYPE:
            self.strict_llm = False
        self.universe_selection = None
        if stock_pool is not None or pool_mode is not None or universe_selection is not None:
            if data_source == IFIND_ASHARE or symbols is not None:
                raise ValueError("stock_pool requires a US source and cannot be combined with symbols")
            if universe_selection is not None:
                if stock_pool is not None or pool_mode is not None:
                    raise ValueError("Use a frozen universe selection or stock_pool, not both")
                self.universe_selection = validate_selection(universe_selection)
            else:
                if stock_pool is None:
                    raise ValueError("pool_mode requires stock_pool")
                self.universe_selection = resolve_strategy_universe(stock_pool, pool_mode or "top30")
            self.symbols = list(self.universe_selection["symbols"])
        # iFinD is backend-owned; US providers can use the selected run assets.
        elif data_source == IFIND_ASHARE:
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

        self.data_loader = self._create_market_data_provider()

    def _create_market_data_provider(self):
        """Create the selected provider without breaking legacy test doubles."""
        factory = create_market_data_provider
        try:
            parameters = inspect.signature(factory).parameters.values()
            accepts_source_timeframe = any(
                parameter.name == "source_timeframe"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_source_timeframe = False
        if accepts_source_timeframe:
            return factory(
                self.data_source,
                self.profile.universe,
                source_timeframe=self.requested_source_timeframe,
            )
        return factory(self.data_source, self.profile.universe)
    
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

    # -- Progress phases -----------------------------------------------------
    #
    # `_publish_live_progress` is the only writer once the bar loop starts;
    # before it, nothing wrote anything, and the card sat on its launch
    # sentence for the whole start (imports, six stores' DDL, the bar fetch,
    # aggregation, indicators, then bar 1's full pipeline). Each phase write
    # carries the phase in progress plus the finished ones with timestamps, so
    # the same payload that drives the card is the measurement of the start.
    # Every accessor tolerates an instance built with __new__ (tests, legacy
    # tools) that never ran _init_progress_phases -- and `publish_phase`
    # additionally tolerates `progress_file` being *absent* rather than None,
    # because Step 6 makes it the first statement of `load_data`, which is
    # documented as usable on exactly such an instance.

    def _init_progress_phases(
        self,
        launched_at: Optional[float] = None,
        startup_clock: Optional[Dict[str, float]] = None,
    ) -> None:
        """Open the implicit `starting` phase at the parent's launch time.

        The parent alone can see the gap between spawning the child and the
        child's first write (imports, store startup); passing its clock in is
        how that gap becomes a measured phase instead of a missing one. A
        caller that passes nothing (the CLI without --launched-at, tests,
        `__new__` instances) opens no phase at all, so the first
        `publish_phase` records no `starting` entry rather than inventing one
        from the child's own clock -- which would measure zero and read as a
        start that cost nothing.

        `starting` is deliberately ONE phase name -- the card has nothing
        useful to say about spawn vs imports vs stores, and a name the status
        route must translate is a name the frontend must learn. But one
        undifferentiated number cannot justify an optimisation either: it lumps
        process spawn, pandas and three SDK imports, seven store constructions
        (six of them Postgres twins, and only those six run DDL) and that DDL
        into a single figure that moves for reasons nobody can attribute. So
        the *record* is split even though the *phase* is not.
        With the script's two stamps and its reading of db_url's accumulator,
        the `starting` entry yields four numbers from one run:

            spawn + interpreter    = child_entered_at - started_at
            imports incl. stores   = imports_seconds       (steady clock)
            of which schema DDL    = schema_init_seconds   (0.0 in a worker)
            preflight remainder    = preflight_seconds     (steady clock)

        The middle two are measured, not re-derived from the stamps beside
        them: both intervals begin and end inside the child, so both belong on
        the steady clock for the reason the module header gives. The stamps
        stay because `spawn + interpreter` genuinely needs them and because
        they are what a log line is correlated against.

        Undo the split and the Final verification table can say that `starting`
        got shorter but not which of those four moved -- which is the same as
        not knowing whether removing the DDL did anything.

        A mark nobody passed is *absent*, not zero: an in-process engine gets a
        plain three-key record. `schema_init_seconds` is the one exception --
        present and 0.0 in a worker, because that zero is the evidence the flag
        fired, not a gap.
        """
        # `is not None`, not truthiness: 0.0 is a launch time, and argparse's
        # `type=float` hands it over intact. A falsy gate dropped the whole
        # `starting` phase for `--launched-at 0` -- deleting the one number
        # this plan exists to produce, with nothing in the file to say so.
        # Deliberately NOT range-checked: `starting` is never a live phase, so
        # an absurd clock lands only in `phases[]`, where it reads as an
        # absurdly long row -- visible, and therefore fixable. A bounds check
        # would turn that back into a silent absence, which is the one outcome
        # this repo never accepts (see IFIND_ALLOW_CORPORATE_ACTION_GAPS in
        # CLAUDE.md: a labelled wrong number, never a silent one). The other
        # malformed forms cannot reach here at all -- argparse's `type=float`
        # rejects an empty string and a BSD `date +%s.%N`'s trailing "N" at the
        # CLI boundary, loudly.
        self._progress_phase: Optional[str] = (
            "starting" if launched_at is not None else None
        )
        self._progress_phase_started_at: Optional[float] = (
            float(launched_at) if launched_at is not None else None
        )
        # Deliberately None even when `starting` opens: that phase began in the
        # PARENT, and a monotonic reading is meaningless across processes.
        # `_set_progress_phase` reads None as "fall back to the wall-clock
        # difference", which is correct for exactly and only this phase.
        self._progress_phase_started_steady: Optional[float] = None
        extra: Dict = {}
        # Seed the startup clock only when `starting` is actually open.
        # backtest_hourly_agent.py always passes `startup_clock` but
        # `launched_at` only from --launched-at, which a bare CLI run omits;
        # today the orphaned keys are dropped because `starting` never closes,
        # and once extras merge into whichever phase closes they would
        # otherwise land on `loading_bars` -- a spawn time and a DDL cost on
        # the phase that did neither.
        # The steady marks ride the same gate but land in their own attribute
        # rather than in `extra`, because `extra` is persisted into `phases[]`
        # and a raw `monotonic()` reading has a per-process epoch: in the file
        # it is a large meaningless float, and an operator differencing it
        # against anything else in the payload gets nonsense. What gets
        # published is the two durations `_set_progress_phase` derives from
        # them.
        steady: Dict[str, float] = {}
        if launched_at is not None:
            for key in ("child_entered_at", "imports_done_at", "schema_init_seconds"):
                value = (startup_clock or {}).get(key)
                if value is not None:
                    extra[key] = float(value)
            for key in ("child_entered_steady", "imports_done_steady"):
                value = (startup_clock or {}).get(key)
                if value is not None:
                    steady[key] = float(value)
        self._progress_startup_steady: Dict[str, float] = steady
        self._progress_phase_extra: Dict = extra
        self._progress_phases: List[Dict] = []
        self._progress_total_steps: int = 0

    def _set_progress_phase(
        self, name: str, *, total_steps: Optional[int] = None
    ) -> bool:
        """Close the current phase and open ``name``. True if the phase moved."""
        if name not in PROGRESS_PHASES:
            raise ValueError(f"unknown progress phase: {name!r}")
        if not hasattr(self, "_progress_phases"):
            self._init_progress_phases()
        if total_steps is not None:
            self._progress_total_steps = int(total_steps)
        if self._progress_phase == name:
            return False
        now = wall_clock()
        now_steady = steady_clock()
        if self._progress_phase is not None:
            finished = {
                "name": self._progress_phase,
                "started_at": self._progress_phase_started_at,
                "ended_at": now,
            }
            # Extras belong to whichever phase was open, not to `starting`
            # alone. `_init_progress_phases` seeds them for `starting` -- and
            # ONLY when it actually opens that phase, see below;
            # `record_phase_metric` adds them for any later phase, and only
            # while one is open. Cleared on every transition so a number can
            # never be reported against the wrong phase.
            finished.update(getattr(self, "_progress_phase_extra", {}))
            self._progress_phase_extra = {}
            self._progress_phases.append(finished)
            # stdout as well as the file, and from here rather than from
            # main(). The parent unlinks the progress file the moment the run
            # ends (`backtests.py:1993`, inside run_backtest_background's
            # `finally` at `:1954`), so `phases[]` can only be read by racing a
            # live poll; the child's stdout is captured head+tail
            # (SUBPROCESS_LOG_HEAD_CHARS / _TAIL_CHARS, 32k each, `:2482-2483`)
            # and dumped into the parent's log under
            # `=== BACKTEST SCRIPT OUTPUT ===` (`:1773`), where it keeps. It is
            # also the ONLY phase record a CLI run, the external-run session or
            # the algo service has -- none of them writes a progress file.
            # main() could not do this job: it cannot see `first_decision` open
            # and close inside run_agent_backtest, and a second elapsed clock in
            # the script is the two-owners pattern this repo already documents.
            # `is None`, not `or`: `started_at` is 0.0 for the very launch
            # clock this plan exists to measure, and `0.0 or now` would print
            # that phase as having cost nothing -- the same falsy-zero trap
            # _init_progress_phases's gate is about, one line further on.
            opened = finished["started_at"]
            opened_steady = getattr(self, "_progress_phase_started_steady", None)
            if opened_steady is None:
                # `starting` (opened in the parent), or a legacy `__new__`
                # instance predating the steady mark. The wall clock is the
                # only shared reference across that process boundary.
                elapsed = finished["ended_at"] - (now if opened is None else opened)
            else:
                elapsed = now_steady - opened_steady
            # Recorded, not just printed: `phases[]` is what an operator reads
            # off the progress file, and leaving only the two wall-clock
            # instants there would make them re-derive the distorted number the
            # steady clock exists to avoid.
            finished["duration_seconds"] = elapsed
            print(f"⏱  phase {name} (after {finished['name']} {elapsed:.2f}s)", flush=True)
            if finished["name"] == "starting" and {
                "child_entered_at",
                "imports_done_at",
            } <= finished.keys():
                # Printed on this transition, not saved for a summary at the
                # end: `saving` fires before the agent's insert_run with two
                # baseline runs still to print, so a closing table can land in
                # the truncated middle of the parent's bounded capture. These
                # lines are at the head.
                #
                # Two of the four numbers are MEASURED, not differenced.
                # `imports+stores` and `preflight` both begin and end inside
                # this process, so an NTP step between the stamps that bound
                # them is the #509 hazard exactly -- a negative
                # `imports+stores`, or one smaller than the monotonic `schema
                # DDL` printed beside it on the same line, i.e. an internally
                # contradictory measurement. They come off the steady marks
                # the launch script hands over. `spawn+interpreter` cannot:
                # its left edge is the PARENT's clock, and that is the same
                # irreducible cross-process interval the module header
                # describes.
                #
                # Derived and stored HERE rather than pre-differenced in the
                # script, for one owner each: `preflight` closes on this very
                # transition, so only the engine can compute it, and splitting
                # the pair across two files is how they end up on two clocks.
                # Stored as well as printed for the same reason
                # `duration_seconds` is -- `phases[]` is what an operator
                # reads off the progress file, and leaving only the stamps
                # there makes them re-derive the distorted number.
                steady_marks = getattr(self, "_progress_startup_steady", None) or {}
                entered_steady = steady_marks.get("child_entered_steady")
                imports_done_steady = steady_marks.get("imports_done_steady")
                # A child predating the steady marks (or a test handing over
                # the three stamps alone) still gets a number: the wall-clock
                # difference this replaced. Absent beats silently zero.
                if entered_steady is None or imports_done_steady is None:
                    imports_seconds = (
                        finished["imports_done_at"] - finished["child_entered_at"]
                    )
                else:
                    imports_seconds = imports_done_steady - entered_steady
                if imports_done_steady is None:
                    preflight_seconds = (
                        finished["ended_at"] - finished["imports_done_at"]
                    )
                else:
                    preflight_seconds = now_steady - imports_done_steady
                finished["imports_seconds"] = imports_seconds
                finished["preflight_seconds"] = preflight_seconds
                print(
                    f"     spawn+interpreter "
                    f"{finished['child_entered_at'] - finished['started_at']:.2f}s"
                    f" | imports+stores {imports_seconds:.2f}s"
                    f" (schema DDL {finished.get('schema_init_seconds', 0.0):.2f}s)"
                    f" | preflight {preflight_seconds:.2f}s",
                    flush=True,
                )
            elif finished["name"] == "loading_bars" and "fetch_seconds" in finished:
                # The design's measurement gate: with a warm cache the residual
                # here IS the aggregation cost, directly -- no synthetic
                # benchmark, no subtraction. It decides whether caching the
                # AGGREGATED output is worth a second change or whether the
                # fetch was the whole story. `elapsed` is the phase total
                # computed above, and for THIS phase it always comes off the
                # steady clock -- only `starting` can reach the wall-clock
                # branch, because it is the only phase `_init_progress_phases`
                # opens and so the only one whose steady mark can be None.
                # That matters here: `fetch_seconds` is a steady-clock delta
                # (`load_data`), so the subtraction below is same-clock and
                # cannot go negative from a step. It is also why the test
                # bounding `fetch_seconds` has to bound it by
                # `duration_seconds` and not by `ended_at - started_at`.
                fetch_seconds = float(finished["fetch_seconds"])
                print(
                    f"     fetch {fetch_seconds:.2f}s"
                    f" | aggregate+verify {elapsed - fetch_seconds:.2f}s",
                    flush=True,
                )
        else:
            print(f"⏱  phase {name}", flush=True)
        self._progress_phase = name
        self._progress_phase_started_at = now
        self._progress_phase_started_steady = now_steady
        return True

    def _progress_phase_fields(self) -> Dict:
        if not hasattr(self, "_progress_phases"):
            self._init_progress_phases()
        return {
            "phase": self._progress_phase,
            "phase_started_at": self._progress_phase_started_at,
            "phases": list(self._progress_phases),
        }

    def record_phase_metric(self, key: str, value: float) -> None:
        """Attach a number to the phase that is currently open.

        The phase *name* stays one word -- the card has nothing useful to say
        about fetch versus aggregate, and a name the status route must
        translate is a name the frontend must learn. But one undifferentiated
        number cannot justify an optimisation either, which is the same
        argument `_init_progress_phases` makes for splitting `starting` into
        four. So the record is split even though the phase is not.

        Tolerates an instance built with `__new__` that never ran
        `_init_progress_phases`, like every other accessor here.

        With no phase open the number has no owner and is dropped: the
        alternative is handing it to whichever phase closes first, which is
        the same misattribution `_init_progress_phases` guards against for
        the startup clock.
        """
        if not hasattr(self, "_progress_phase_extra"):
            self._init_progress_phases()
        if self._progress_phase is None:
            return
        self._progress_phase_extra[str(key)] = float(value)

    def publish_phase(self, name: str, *, total_steps: Optional[int] = None) -> None:
        """Record a phase transition and, with a progress file, publish it.

        **Before** the loop there is nothing to carry, so the payload is the
        skeleton `step: 0` / `equity_curve: []`, and every existing reader of
        the file (chart, trading log, ETA anchor) keeps its early return while
        only the message and the bar count change.

        **After** the loop there is, and writing that skeleton over it was a
        real regression rather than a cosmetic one. `saving` fires at the end
        of a 49-bar run: a payload of `step: 0, equity_curve: []` snaps the
        Backtest panel's bar from 99% to 0 -- `stepPct` is computed straight
        off this file's `step`/`total_steps` (`app.js:8879`, the 1s poller;
        `attachToLiveBacktest` at `:8700` holds a byte-identical second copy,
        which Task 4 replaces with one shared helper) and never passes through
        the fold, so no frontend guard can reach it -- and the My
        Agents fold *replaces* its stored entry (`app.js:8886`), blanking the
        sparkline, the equity label and `49/49` for the whole
        baseline/persistence tail, with nothing red anywhere. So a phase write
        carries the last published payload forward and changes only the phase
        fields and the bar count: the file never says less than it last said.

        The carry is what makes the fix hold at the source. Task 4 Step 6 adds
        a second, independent guard in the browser for any writer that still
        publishes a bare phase tick after real progress; neither is a substitute
        for the other, because the panel's bar bypasses the fold and the fold
        outlives this engine's payload shape.
        """
        self._set_progress_phase(name, total_steps=total_steps)
        # `getattr`, not `self.progress_file`. Step 6 makes this method the
        # first statement of `load_data`, and `load_data` is documented as
        # usable on an instance built with `__new__` (its own comment,
        # engine.py:913-914) -- so this read is now the first attribute such a
        # caller touches. `progress_file` is assigned in `__init__` and is not
        # a class attribute, so on that instance it is *absent*, and a bare
        # read raises AttributeError rather than returning None. Verified by
        # running it. The one such caller in the suite is
        # test_market_data_errors.py::test_engine_load_data_empty_raises_not_exits,
        # which sets five attributes and not this one; it would report an
        # AttributeError naming neither this task nor its own subject, in
        # place of the MarketDataUnavailableError it asserts.
        if not getattr(self, "progress_file", None):
            return
        last = getattr(self, "_progress_last_payload", None) or {}
        payload = dict(last)
        payload.update(
            {
                "run_id": self.live_run_id,
                # `last.get(...)`, not `self._progress_...`: the step and the
                # curve belong to the loop, and re-deriving them here would be
                # a second owner for numbers _publish_live_progress already
                # published. Absent (pre-loop) they default to the skeleton.
                "step": int(last.get("step") or 0),
                "total_steps": self._progress_total_steps,
                "equity_curve": last.get("equity_curve") or [],
                **self._progress_phase_fields(),
            }
        )
        self._write_progress_payload(payload)

    def _write_progress_payload(self, payload: Dict) -> None:
        from pathlib import Path

        # Remembered before the write, not after it: this is the payload this
        # process intends the file to hold, and a phase published after a failed
        # write must still carry the loop's numbers rather than silently fall
        # back to the skeleton. One reference to data the manager already holds.
        self._progress_last_payload: Dict = payload
        try:
            Path(self.progress_file).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            print(f"   ⚠️  Could not write live progress: {exc}")

    def _publish_live_progress(self, step: int, total_steps: int, manager) -> None:
        """Write incremental equity curve snapshots for live dashboard charting."""
        # Above the early return, not below it. The phase clock is state, not a
        # side effect of writing, and the progress file is only one of its
        # readers. Below the return an engine with no progress file -- every
        # CLI run, the external-run session, the algo service -- never leaves
        # `first_decision`, so `publish_phase("saving")` closes a
        # `first_decision` spanning the entire bar loop: the one number this
        # phase exists to produce, published as hours instead of seconds, in
        # the row of the table that is supposed to justify the whole track.
        # `publish_phase` is already the right way round; this is the only
        # asymmetry. `_set_progress_phase`'s hasattr guard self-initialises, so
        # a `__new__`-built instance is unaffected.
        self._set_progress_phase("running", total_steps=total_steps)
        if not self.progress_file:
            return

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
            **self._progress_phase_fields(),
        }
        self._write_progress_payload(payload)

    def load_data(self):
        """Fetch source bars and build the strategy's decision-bar dataset."""
        self.publish_phase("loading_bars")
        # Keep the error path usable for legacy callers that construct an
        # instance with ``__new__`` (or inject a loader) before initialization.
        symbols = getattr(self, "symbols", ())
        print(
            f"   Universe: {len(symbols)} symbols ({', '.join(symbols[:8])}"
            f"{'…' if len(symbols) > 8 else ''})"
        )
        fetch_started_at = steady_clock()
        self.source_data = self.data_loader.fetch_bars(
            symbols, self.start_date, self.end_date
        )
        # Everything after this point in `loading_bars` -- the frequency
        # verification and, in intraday mode, `aggregate_bars_by_symbol` -- is
        # the half the phase name hides. Recorded here so the number survives
        # in `phases[]` as well as on stdout.
        self.record_phase_metric("fetch_seconds", steady_clock() - fetch_started_at)
        if not self.source_data:
            # Raise, don't sys.exit(1): this runs inside server threads
            # (external runs, algo service) where SystemExit evades
            # `except Exception` and strands the run (the B0 class).
            print("❌ No data fetched.")
            raise MarketDataUnavailableError(
                f"No {self.data_source} market data available for "
                f"{self.start_date}..{self.end_date}"
            )

        configured_source_timeframe = getattr(
            self.data_loader, "source_timeframe", None
        )
        if configured_source_timeframe is None:
            # A provider replacement that predates the frequency contract is
            # assumed to return its historical profile resolution. This keeps
            # injected hourly loaders from being accidentally aggregated as if
            # they had honoured the new optional factory argument.
            actual_source_timeframe = normalize_bar_timeframe(
                self.profile.timeframe
            )
        else:
            try:
                actual_source_timeframe = verify_source_timeframe(
                    self.requested_source_timeframe,
                    configured_source_timeframe,
                    evidence="configured",
                )
            except FrequencyConfigError as exc:
                raise MarketDataUnavailableError(
                    f"Market data frequency contract failed: {exc}"
                ) from exc
        fetch_evidence = getattr(self.data_loader, "last_fetch", None)
        if isinstance(fetch_evidence, dict) and fetch_evidence.get(
            "source_timeframe"
        ):
            try:
                actual_source_timeframe = verify_source_timeframe(
                    self.requested_source_timeframe,
                    fetch_evidence["source_timeframe"],
                    evidence="fetch",
                )
            except FrequencyConfigError as exc:
                raise MarketDataUnavailableError(
                    f"Market data frequency contract failed: {exc}"
                ) from exc
        self.source_timeframe = actual_source_timeframe
        self.market_data_provenance = feed_provenance(self.source_data) or {}
        self.intraday_mode = timeframe_minutes(actual_source_timeframe) < timeframe_minutes(
            self.decision_timeframe
        )
        if self.intraday_mode:
            # This Phase 2 engine uses the fetched source bar for both fill and
            # valuation. If a caller explicitly selects 1m instead of the
            # profile's 5m target, metadata must describe the effective clocks.
            self.execution_timeframe = actual_source_timeframe
            self.valuation_frequency = actual_source_timeframe
            self.frequency_contract = build_verified_intraday_contract(
                source_timeframe=actual_source_timeframe,
                decision_timeframe=self.decision_timeframe,
                decision_frequency=self.profile.decision_frequency,
            )
            print(
                f"   Aggregating {actual_source_timeframe} source bars into "
                f"{self.decision_timeframe} decision bars..."
            )
            aggregated_data = aggregate_bars_by_symbol(
                self.source_data,
                source_timeframe=actual_source_timeframe,
                decision_timeframe=self.decision_timeframe,
                market=self.profile.market,
                timezone=self.profile.timezone,
            )
            self.data_quality = summarize_aggregation_quality(aggregated_data)
            self.all_data = self._completed_decision_bars(aggregated_data)
            if not self.all_data:
                raise MarketDataUnavailableError(
                    "Source bars were fetched, but no completed decision bars "
                    "could be built"
                )
        else:
            self.all_data = self.source_data
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
        except (CorporateActionGapError, IFindClientError, MarketRuleDataError, ValueError) as exc:
            if isinstance(exc, CorporateActionGapError):
                # Not prefixed like the branch below: the rule data is
                # available and valid, and telling the user it is missing
                # sends them to look for a credentials problem that does not
                # exist. The message already names the dates and what to do
                # about them. A single ``except`` tuple plus this branch,
                # rather than two ``except`` clauses ordered specific-first:
                # ``CorporateActionGapError`` no longer subclasses
                # ``MarketRuleDataError`` (it is a bad-input condition, not a
                # data-unavailable one), so two clauses would put the choice
                # of message back in the hands of whichever one a future edit
                # happened to list first.
                raise MarketDataUnavailableError(str(exc)) from exc
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

        # Scaled to the requested window, not a flat count. A fixed 50 lived
        # here and in ifind_ashare.py, and both were quietly a ~13-trading-day
        # minimum window: once MAX_BACKTEST_DAYS fell to 14, no legal window
        # could reach it (at most 10 weekdays x 4 A-share 60m sessions = 40
        # bars), so every A-share run raised here instead of returning data.
        # Parsing is bare because the provider has already normalized these
        # same two strings by the time any data exists to validate.
        floor = minimum_bars_for_window(
            date.fromisoformat(self.start_date),
            date.fromisoformat(self.end_date),
        )
        short = {
            symbol: len(self.all_data[symbol])
            for symbol in expected
            if len(self.all_data[symbol]) < floor
        }
        if short:
            raise MarketDataUnavailableError(
                f"iFinD symbols have fewer than {floor} bars: {short!r}"
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
        self.publish_phase("indicators")
        print("\n📈 Calculating technical indicators...")
        count = 0
        for symbol, df in self.all_data.items():
            self.all_data[symbol] = TechnicalIndicators.calculate_indicators(df)
            count += 1
            if count % 5 == 0:
                print(f"  ✅ {count}/{len(self.all_data)} symbols...")
        if getattr(self, "data_source", None) == ALPACA:
            try:
                self.all_data, self.equity_metadata = load_and_enrich_us_equity_bars(
                    self.all_data,
                    timezone=self.profile.timezone,
                )
            except EquityMetadataUnavailableError as exc:
                raise MarketDataUnavailableError(
                    "Configured US equity metadata is unavailable"
                ) from exc
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
            **dict(getattr(self, "market_data_provenance", {}) or {}),
        }
        if getattr(self, "intraday_mode", False):
            frequency_contract = getattr(self, "frequency_contract", None)
            metadata["frequency_contract"] = dict(
                frequency_contract
                or build_verified_intraday_contract(
                    source_timeframe=self.source_timeframe,
                    decision_timeframe=self.decision_timeframe,
                    decision_frequency=profile.decision_frequency,
                )
            )
            metadata["market_data_quality"] = dict(
                getattr(self, "data_quality", {})
            )
        if getattr(self, "universe_selection", None) is not None:
            metadata["universe_selection"] = dict(self.universe_selection)
        equity_metadata = dict(getattr(self, "equity_metadata", {}) or {})
        if equity_metadata.get("status") == "available":
            metadata["equity_metadata"] = equity_metadata
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
        # What the run asked for, beside what it ended up doing. These differ
        # only when the model was unavailable, which is precisely the run that
        # has to be reported as a fallback rather than as a choice -- and
        # `decision_source` above has already been overwritten with the
        # outcome by then. Written unconditionally (not only when they differ)
        # so its *presence* separates a row this code wrote from one that
        # predates the key, the same way `decision_steps` does for the
        # counters: a reader cannot otherwise tell "asked for rule-based" from
        # "nobody recorded the question".
        requested_decision_source = getattr(self, "requested_decision_source", None)
        if requested_decision_source is not None:
            meta["requested_decision_source"] = requested_decision_source
        decision_steps = getattr(self, "llm_decision_steps", None)
        if decision_steps is not None:
            # Denominator for "did the model actually drive this run?" --
            # llm_decisions / decision_steps, the same ratio the leaderboard's
            # H6 guard applies, under the name it already uses for it.
            #
            # Its *presence* is also the witness that this row records
            # llm_decisions at all: that column was added with DEFAULT 0, so
            # every row written before it existed reads back as 0, and a bare 0
            # cannot tell "the model drove nothing" from "nobody was counting".
            # Reading provenance off such a row without this key would accuse
            # every historical run of a fallback it never had.
            meta["decision_steps"] = int(decision_steps)
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
        if getattr(self, "universe_selection", None) is not None:
            result["stock_pool"] = self.universe_selection["stock_pool"]
            result["pool_mode"] = self.universe_selection["pool_mode"]
        equity_metadata = dict(getattr(self, "equity_metadata", {}) or {})
        if equity_metadata.get("status") == "available":
            result["equity_metadata"] = {
                "classification": equity_metadata["classification"],
                "point_in_time": equity_metadata["point_in_time"],
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
            local = (
                market_tz.localize(timestamp)
                if timestamp.tzinfo is None
                else timestamp.astimezone(market_tz)
            )
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

    def _market_day_key(self, timestamp) -> str:
        """Return a trading-day key in the market's local timezone."""
        import pytz

        market_tz = pytz.timezone(self._effective_profile().timezone)
        local = (
            market_tz.localize(timestamp)
            if timestamp.tzinfo is None
            else timestamp.astimezone(market_tz)
        )
        return local.date().isoformat()

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

    @staticmethod
    def _timestamps_for_data(data: Dict[str, Any]) -> List[datetime]:
        timestamps = set()
        for frame in data.values():
            timestamps.update(frame.index)
        return sorted(timestamps)

    @staticmethod
    def _market_data_at(
        data: Dict[str, Any], symbols: List[str], timestamp: datetime
    ) -> Dict[str, Any]:
        return {
            symbol: data[symbol].loc[timestamp]
            for symbol in symbols
            if symbol in data and timestamp in data[symbol].index
        }

    @staticmethod
    def _forward_filled_price_cache(
        data: Dict[str, Any], timestamps: List[datetime]
    ) -> Dict[str, Dict[datetime, Any]]:
        cache: Dict[str, Dict[datetime, Any]] = {}
        for symbol, frame in data.items():
            prices: Dict[datetime, Any] = {}
            last_price = None
            for timestamp in timestamps:
                if timestamp in frame.index:
                    last_price = frame.loc[timestamp, "close"]
                if last_price is not None:
                    prices[timestamp] = last_price
            cache[symbol] = prices
        return cache

    def _annualization_periods(self) -> float | None:
        """Return the Sharpe sampling factor for the curve this run emits."""
        if not getattr(self, "intraday_mode", False):
            return None
        minutes = timeframe_minutes(self.source_timeframe)
        return 252 * 6.5 * (60 / minutes)

    def run_agent_backtest(self) -> Tuple[str, List[Dict]]:
        """Run a backtest with hourly strategy decisions.

        When a finer source dataset is configured, the strategy still receives
        one completed hourly bar per step while fills and mark-to-market use
        the finer source timeline.
        """
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
        
        # Decision timestamps come from the completed decision-bar dataset.
        all_timestamps = self._timestamps_for_data(self.all_data)
        
        # Filter: only keep hours with real data for 80%+ of symbols
        min_required = max(1, ceil(len(self.all_data) * 0.8))
        filtered = []
        for ts in all_timestamps:
            real_data_count = sum(1 for df in self.all_data.values() if ts in df.index)
            if real_data_count >= min_required:
                filtered.append(ts)
        
        all_timestamps = filtered
        
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

        raw_timestamps = all_timestamps
        execution_plan = {timestamp: timestamp for timestamp in all_timestamps}
        if self.intraday_mode:
            raw_timestamps = self._market_hours_only(
                self._timestamps_for_data(self.source_data)
            )
            raw_by_market_day: Dict[str, List[Any]] = {}
            for source_timestamp in raw_timestamps:
                raw_by_market_day.setdefault(
                    self._market_day_key(source_timestamp), []
                ).append(source_timestamp)
            execution_plan = {}
            for timestamp in all_timestamps:
                same_day_sources = raw_by_market_day.get(
                    self._market_day_key(timestamp), []
                )
                source_index = bisect_left(same_day_sources, timestamp)
                next_source_timestamp = (
                    same_day_sources[source_index]
                    if (
                        source_index < len(same_day_sources)
                        and same_day_sources[source_index] == timestamp
                    )
                    else None
                )
                # The final partial session bucket (e.g. 15:30–16:00 ET) has
                # no following source bar at 16:00 and cannot be executed.
                if next_source_timestamp is not None:
                    execution_plan[timestamp] = next_source_timestamp
            all_timestamps = [
                timestamp
                for timestamp in all_timestamps
                if timestamp in execution_plan
            ]

        print(
            f"   Trading {len(all_timestamps)} hourly decision bars during "
            f"{self.profile.market} {self.profile.timeframe} sessions"
            + (
                f"; executing/valuing on {self.source_timeframe} bars"
                if self.intraday_mode
                else ""
            )
            + "...\n"
        )
        total_steps = len(all_timestamps)
        self.publish_phase("first_decision", total_steps=total_steps)
        # Declare the run length so a strict-LLM run can absorb a small number
        # of unusable responses instead of discarding hours of work on the
        # first one. Left unset the budget is 0 (fatal on the first strike).
        manager.strict_llm_total_steps = total_steps
        if self.strict_llm:
            print(
                "   Strict LLM: aborting after more than "
                f"{manager.strict_llm_fallback_budget()} unusable response(s)\n"
            )

        # Build separate decision and valuation caches. In minute mode the
        # strategy sees hourly prices while portfolio valuation sees every
        # source bar.
        print("   Pre-computing forward-filled price cache...")
        price_cache = self._forward_filled_price_cache(
            self.all_data, all_timestamps
        )
        valuation_price_cache = (
            self._forward_filled_price_cache(self.source_data, raw_timestamps)
            if self.intraday_mode
            else price_cache
        )
        valuation_cursor = 0

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

            # Decision signals always use the completed hourly bar.
            market_data = self._market_data_at(
                self.all_data, self.symbols, timestamp
            )
            
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
            
            # In minute mode, execute at the next source bar's open. The
            # decision bar closes at ``timestamp``; the source bar opening at
            # that same instant is the first non-look-ahead fill opportunity.
            execution_timestamp = execution_plan[timestamp]
            execution_market_data = market_data
            execution_fallback_prices = {
                symbol: values[execution_timestamp]
                for symbol, values in valuation_price_cache.items()
                if execution_timestamp in values
            }
            execution_prices = None
            if self.intraday_mode:
                execution_market_data = self._market_data_at(
                    self.source_data,
                    self.symbols,
                    execution_timestamp,
                )
                execution_prices = {
                    symbol: row["open"]
                    for symbol, row in execution_market_data.items()
                    if "open" in row
                }

            # Execute trades (only if real data available)
            trades_before_execution = len(manager.trades)
            manager.execute_actions(
                decision["actions"],
                execution_market_data,
                execution_timestamp,
                fallback_prices=execution_fallback_prices,
                execution_prices=execution_prices,
            )
            if runtime_invoked:
                self.runtime_dispatcher.record_latest_execution(
                    len(manager.trades) - trades_before_execution
                )
            
            # Update equity. The minute path emits one mark for every source
            # bar through the fill, while the legacy path emits one hourly mark.
            if self.intraday_mode:
                while (
                    valuation_cursor < len(raw_timestamps)
                    and raw_timestamps[valuation_cursor] <= execution_timestamp
                ):
                    valuation_timestamp = raw_timestamps[valuation_cursor]
                    valuation_market_data = self._market_data_at(
                        self.source_data,
                        self.symbols,
                        valuation_timestamp,
                    )
                    manager.update_equity(
                        valuation_market_data,
                        valuation_price_cache,
                        valuation_timestamp,
                    )
                    manager.equity_history[-1] = (
                        self._require_currency_context().reporting_equity_record(
                            manager.equity_history[-1]
                        )
                    )
                    valuation_cursor += 1
            else:
                manager.update_equity(
                    execution_market_data,
                    price_cache,
                    execution_timestamp,
                )
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
                pct_return = fractional_return(equity, self.initial_capital) * 100
                print(f"   Decision {i+1}/{len(all_timestamps)}: Equity ${equity:,.0f} ({pct_return:+.1f}%)")

        if self.intraday_mode:
            # Mark any remaining source bars after the final decision/fill so
            # the curve closes at the end of the requested market window.
            while valuation_cursor < len(raw_timestamps):
                valuation_timestamp = raw_timestamps[valuation_cursor]
                valuation_market_data = self._market_data_at(
                    self.source_data,
                    self.symbols,
                    valuation_timestamp,
                )
                manager.update_equity(
                    valuation_market_data,
                    valuation_price_cache,
                    valuation_timestamp,
                )
                manager.equity_history[-1] = (
                    self._require_currency_context().reporting_equity_record(
                        manager.equity_history[-1]
                    )
                )
                valuation_cursor += 1
        
        equity_curve = manager.get_equity_curve()
        
        # Convert timestamps to strings
        for entry in equity_curve:
            if hasattr(entry["timestamp"], "isoformat"):
                entry["timestamp"] = entry["timestamp"].isoformat()
        
        # Store in database
        run_id = self.live_run_id or f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        initial_eq = equity_curve[0]["equity"] if equity_curve else self.initial_capital
        final_eq = equity_curve[-1]["equity"] if equity_curve else self.initial_capital
        total_return = fractional_return(final_eq, self.initial_capital)

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

        # Steps the model actually drove, and the steps it was asked to drive.
        # Persisted beside llm_calls because they answer different questions:
        # llm_calls is the *billing* counter and ticks on a truncated or
        # unparseable response that then traded rule-based, so a run with
        # llm_calls == decision_steps and llm_decisions == 0 is a total
        # fallback wearing the model's name. Nothing downstream could see that
        # before -- dashboard backtests are subprocesses, so PortfolioManager's
        # in-process counter (and the H6 guard that reads it) never crossed
        # back to the parent. Issue #169.
        if self.runtime_type == PIPELINE_RUNTIME_TYPE:
            llm_decisions_total = manager.llm_decisions
            decision_steps_total = total_steps
        else:
            # A hosted runtime decides once per trading day and holds on every
            # other bar, so the hourly bar count is not its denominator: it
            # would report a genuine run as ~1/7 covered. `runtime_calls` is
            # every step it drove, and each entry in `runtime_step_failures` is
            # a step it was asked for and could not answer -- their sum is the
            # steps the model was actually asked to decide. There is no
            # billed-but-unusable case on this path (the bridge reports no
            # usage at all), which is why numerator and billing counter
            # coincide here and must not be collapsed anywhere else.
            llm_decisions_total = runtime_calls
            decision_steps_total = runtime_calls + len(self.runtime_step_failures)
        # Read back out of `self` by _agent_run_metadata below, the same way it
        # already reads decision_source and the execution evidence. Not named
        # `decision_steps`: that spelling is already taken in this module for
        # the *pipeline's* decision stages, which is a different thing entirely.
        self.llm_decision_steps = decision_steps_total

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
        self.publish_phase("saving")
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
            sharpe_ratio=self._calc_sharpe(
                equity_curve, periods_per_year=self._annualization_periods()
            ),
            max_drawdown=self._calc_max_dd(equity_curve),
            num_trades=len(manager.trades),
            llm_model=llm_model,  # Track which model was used
            llm_calls=llm_calls_total,
            llm_decisions=llm_decisions_total,
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
        # ``generate_baselines`` at $0 returns a flat zero history, which is
        # then written to ``agent_runs`` as a real row -- so a $0 run recorded
        # buy-and-hold as having returned 0.00% over its window. The chart is
        # rebuilt per request; these rows outlive it.
        if not self.initial_capital > 0:
            print("📊 Buy & Hold baseline skipped: this run has no capital\n")
            return None, []

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
        total_return = fractional_return(final_eq, self.initial_capital)
        
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
        # Same reasoning as run_buyhold_baseline: a benchmark scaled to $0 is a
        # row claiming the Dow returned 0.00%, not a benchmark.
        if not self.initial_capital > 0:
            print("📊 DJIA Index baseline skipped: this run has no capital\n")
            return None, []

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
            if self.intraday_mode:
                bars = aggregate_bars_by_symbol(
                    bars,
                    source_timeframe=self.source_timeframe,
                    decision_timeframe=self.decision_timeframe,
                    market=self.profile.market,
                    timezone=self.profile.timezone,
                )
                bars = self._completed_decision_bars(bars)

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
        total_return = fractional_return(final_eq, self.initial_capital)
        
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
    def _completed_decision_bars(
        bars_by_symbol: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep only decision bars whose source-bar coverage is complete."""
        return {
            symbol: (
                frame.loc[
                    frame["is_complete"].fillna(False).astype(bool)
                ].copy()
                if "is_complete" in frame.columns
                else frame
            )
            for symbol, frame in bars_by_symbol.items()
            if not frame.empty
        }
    
    @staticmethod
    def _calc_sharpe(
        equity_curve: List[Dict], periods_per_year: Optional[float] = None
    ) -> float:
        """Annualized Sharpe ratio for the curve's sampling frequency.

        Delegates to dashboard.backend.domain.backtesting.metrics.calculate_sharpe;
        Omitting ``periods_per_year`` preserves the historical hourly factor;
        minute-valued runs pass the finer sampling factor explicitly.
        """
        return calculate_sharpe(equity_curve, periods_per_year=periods_per_year)

    @staticmethod
    def _calc_max_dd(equity_curve: List[Dict]) -> float:
        """Maximum drawdown of the equity curve.

        Delegates to
        dashboard.backend.domain.backtesting.metrics.calculate_max_drawdown.
        """
        return calculate_max_drawdown(equity_curve)
