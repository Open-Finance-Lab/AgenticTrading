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
extremity (``|RSI - 50|``, a mean-reversion heuristic) — it ranks candidates by a
multi-factor trend/momentum score and shows the top
``SNAPSHOT_SHORTLIST_SIZE`` -- the whole universe instead, for a pipeline agent
on up to ``SNAPSHOT_FULL_UNIVERSE_MAX`` names -- *always appending current
holdings*. That
ranking change was made separately from (and independently of) this file's
domain-layer move, so backtests from before it are **not** directly comparable
with current ones (see the inline comment on that branch for the rationale).

This module is domain-level orchestration: it must NOT import dashboard scripts,
``HourlyBacktester``, FastAPI routers, the database singleton, or Alpaca clients.
"""

import json
import math
from datetime import date, datetime, timedelta
from itertools import chain
from numbers import Real
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
from dashboard.backend.infrastructure.llm.validator import (
    DJIA_30,
    LLM_ACTION_DEFAULT_CONFIDENCE,
    MAX_ORDER_SHARES,
    TRADING_ACTIONS,
    create_prompt,
    model_number,
    normalize_action,
    normalize_symbol,
    read_confidence,
    requested_sell_shares,
)
from dashboard.backend.domain.trading.portfolio import (
    append_equity_record as _append_equity_record,
    build_portfolio_state as _build_portfolio_state,
    get_equity_curve as _get_equity_curve,
)
from dashboard.backend.domain.trading.execution import (
    execute_actions as _execute_actions,
    record_refused_order as _record_refused_order,
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
from dashboard.backend.infrastructure.llm.execution.errors import (
    LLMExecutionError,
    is_empty_reply,
    is_transient_provider_failure,
)
from dashboard.backend.infrastructure.llm.pipeline_runner import (
    RECOVERY_MAX_OUTPUT_TOKENS,
    escalate_ceiling_on_retry,
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

# The ratio rounds to zero below 50 steps, so the onboarding modal's 49-bar run
# had no tolerance at all. From this length on, one strike still leaves the run
# above H6 (1/25 = 4% < 5%); ``test_strict_llm_budget_clears_h6`` pins that.
STRICT_LLM_MIN_STEPS_FOR_ONE_STRIKE = 25

# The version of the pre-trade gate in ``make_trading_decision_with_llm``,
# recorded on every LLM run (``agent_runs.metadata.llm_gate_version``). Bump it
# whenever the gate's rules change what the same reply trades: a cached
# leaderboard row and a recorded decision tape are only comparable with output
# from the gate that produced them. Absent on rows written before it existed.
# 2: a confidence that is not a number from 0 to 1 is not read at all -- it
#    neither sizes a buy nor decides the 0.3 floor (#674). The first version
#    recorded, so it also covers what landed just before it: actions read by
#    ``normalize_action`` (#658) and sells sized as asked (#661, recorded on
#    its own as ``llm_sell_sizing``).
LLM_GATE_VERSION = 2


# The order-event reason of a SELL the gate refused because it cannot read
# its size. The same word the conformance reference uses for it.
SELL_SIZE_REFUSED_REASON = "invalid_quantity"


# Up to this many symbols a pipeline agent's market snapshot carries the whole
# universe, trend-ordered. A cut inside a universe this small (DJIA_30, the
# 30-name pools) made "spread across the universe" mean "across today's top 12
# by trend", and the shortlist changed bar to bar (#541).
SNAPSHOT_FULL_UNIVERSE_MAX = 30

# Above that -- and always on the single-prompt path -- the snapshot keeps this
# many top-trend names plus current holdings. A pipeline prompt says so
# (``universe_note``); the single-prompt one lists every VALID SYMBOL instead.
SNAPSHOT_SHORTLIST_SIZE = 12

# How far back a decision's ``recent_trades`` block looks. The Live board
# carries this memory across its nightly increments through ``snapshot_state``,
# so it is also how much trade history a snapshot keeps.
TRADE_MEMORY_WINDOW = timedelta(hours=24)

# Runaway bound on the trades a snapshot carries (and a restore accepts). Six
# decisions a session over a 30-name universe is at most 180 trades inside the
# window, so the window, not this cap, normally decides what is kept.
SNAPSHOT_TRADE_MEMORY_LIMIT = 200

# Marks a snapshot that carries ``recent_trades``; ``trade_memory_since`` beside
# it says from which bar the chain of snapshots has carried it (see
# ``PortfolioManager._trade_memory_since``). The Live month the memory shipped
# in joins nights that opened with no memory of the night before to nights
# that opened with it, and these two keys are the record of where.
TRADE_MEMORY_VERSION = 1


def _memory_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """``value`` as a tz-aware Timestamp, or None when it is not a readable one.

    A naive stamp is read as UTC, the zone the board's bars are stamped in, so
    a restored trade can always be compared with a tz-aware decision clock
    (comparing naive with aware raises, and the decision path would swallow
    that into a rule-based fallback). Text must open with an ISO date:
    pandas also reads "now" and "today", which would stamp a corrupt row at
    the wall clock, later than every bar.
    """
    if not isinstance(value, (str, datetime)):
        return None
    try:
        if isinstance(value, str):
            date.fromisoformat(value[:10])
        ts = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _memory_trade(trade: Any) -> Optional[Dict[str, Any]]:
    """The prompt's five fields of one trade, or None for a row it cannot use.

    Snapshotting and restoring both pass through here, so a corrupt row is
    dropped on either side instead of failing the session it was read in.
    """
    if not isinstance(trade, Mapping):
        return None
    symbol, side = trade.get("symbol"), trade.get("side")
    if not (isinstance(symbol, str) and symbol and isinstance(side, str) and side):
        return None
    ts = _memory_timestamp(trade.get("timestamp"))
    if ts is None:
        return None
    # Numbers only: the writer stores floats, so text or a bool (which
    # float() reads as 1.0) is a corrupt row, not a quantity.
    numbers = (trade.get("shares"), trade.get("price"))
    if not all(isinstance(n, Real) and not isinstance(n, bool) for n in numbers):
        return None
    shares, price = (float(n) for n in numbers)
    if not (math.isfinite(shares) and math.isfinite(price)):
        return None
    return {
        "symbol": symbol,
        "side": side,
        "shares": shares,
        "price": price,
        "timestamp": ts,
    }


def _universe_note(*, ranked: int, held: int, universe_size: int) -> str:
    """Describe a partial snapshot by what it actually holds this bar.

    ``ranked`` can fall short of the shortlist size when few names have a bar,
    and ``held`` counts only holdings the ranking did not already include.
    """
    shown = ranked + held
    note = (
        f"Snapshot shows {shown} of {universe_size} symbols: "
        f"the top {ranked} by trend score"
    )
    if held:
        noun = "holding" if held == 1 else "holdings"
        note += f", plus {held} current {noun} outside them"
    return note + "."


def _insert_before(
    mapping: Dict[str, Any], anchor: str, key: str, value: Any
) -> Dict[str, Any]:
    """A copy of ``mapping`` with ``key`` placed just ahead of ``anchor``."""
    out: Dict[str, Any] = {}
    for existing, existing_value in mapping.items():
        if existing == anchor:
            out[key] = value
        out[existing] = existing_value
    return out


def _is_legacy_empty_reply(error: BaseException) -> bool:
    """The legacy client's empty reply: a response with no text block.

    Unlike the execution layer's (``is_empty_reply``), it arrives as a response
    whose usage was already recorded, so its retries are billed calls.
    """
    return isinstance(error, AttributeError) and "No text content" in str(error)


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
        # Trades from earlier sessions, restored from a snapshot only so the
        # prompt's 24h ``recent_trades`` block can see them. Never merged into
        # ``trades``: LLMAgentStrategy copies ``trades`` into its trade count and
        # the Live board folds that copy into the month's trade stats on every
        # checkpoint (live.py ``_write_live_checkpoint``), so a restored trade
        # there would be counted again on every later night.
        self.restored_trades: List[Dict[str, Any]] = []
        # The bar this chain of snapshots began carrying trade memory from,
        # restored from a snapshot that records it. None means the chain
        # begins with this manager (``_trade_memory_since``).
        self.trade_memory_since: Optional[pd.Timestamp] = None
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
        # Actions whose confidence was not a number from 0 to 1; of those, the
        # buys skipped because they had no size of their own to trade at; and
        # the steps that lost at least one such buy. A model answering on a
        # 0-100 scale on every bar shows up here, not as a quiet flat curve:
        # the leaderboard's H6 guard subtracts the steps from its numerator.
        self.unusable_confidence_actions = 0
        self.unusable_confidence_skipped = 0
        self.confidence_skipped_steps = 0
        self._unusable_confidence_reported = False
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
        self.allowed_symbols = [normalize_symbol(str(s)) for s in symbols if s]
        self._allowed_set = set(self.allowed_symbols)

    def snapshot_state(self) -> Dict[str, Any]:
        """JSON-safe book for resuming a later session (Live daily increment).

        Carries cash, positions, T+1 lots and ``recent_trades``: the trade
        memory the next session's prompts read (see ``_trade_memory_tail``),
        marked by ``trade_memory`` and ``trade_memory_since``.
        """

        def _lot(lot: Dict) -> Dict[str, Any]:
            buy = lot.get("buy_date")
            if hasattr(buy, "isoformat"):
                buy = buy.isoformat()
            return {
                "quantity": float(lot.get("quantity") or 0),
                "buy_date": str(buy) if buy is not None else None,
            }

        return {
            "cash": float(self.cash),
            "positions": {
                str(symbol): float(shares)
                for symbol, shares in (self.positions or {}).items()
                if shares
            },
            "entry_prices": {
                str(symbol): float(price)
                for symbol, price in (self.entry_prices or {}).items()
            },
            "available_positions": {
                str(symbol): float(shares)
                for symbol, shares in (self.available_positions or {}).items()
            },
            "frozen_lots": {
                str(symbol): [_lot(lot) for lot in (lots or [])]
                for symbol, lots in (self.frozen_lots or {}).items()
            },
            "t_plus_one_enabled": bool(self.t_plus_one_enabled),
            "recent_trades": self._trade_memory_tail(),
            "trade_memory": TRADE_MEMORY_VERSION,
            "trade_memory_since": self._trade_memory_since(),
        }

    def _trade_memory_tail(self) -> List[Dict[str, Any]]:
        """Restored and own trades a later prompt can still be shown, JSON-safe.

        Every later decision is at or after the newest trade and after this
        manager's last bar, so a trade a full ``TRADE_MEMORY_WINDOW`` older than
        the later of the two can never pass a later decision's cutoff: the tail
        anchored there is a superset of what any later ``recent_trades`` block
        reads, and a model that stops trading lets its memory expire instead of
        carrying its last trades in every later snapshot. Oldest first, at most
        ``SNAPSHOT_TRADE_MEMORY_LIMIT`` (the newest), timestamps as ISO text.
        """
        rows = [
            row
            for row in map(_memory_trade, chain(self.restored_trades, self.trades))
            if row is not None
        ]
        if not rows:
            return []
        rows.sort(key=lambda row: row["timestamp"])
        anchor = rows[-1]["timestamp"]
        last_bar = self._bar_stamp(-1)
        if last_bar is not None and last_bar > anchor:
            anchor = last_bar
        cutoff = anchor - TRADE_MEMORY_WINDOW
        tail = [row for row in rows if row["timestamp"] > cutoff]
        return [
            {**row, "timestamp": row["timestamp"].isoformat()}
            for row in tail[-SNAPSHOT_TRADE_MEMORY_LIMIT:]
        ]

    def _trade_memory_since(self) -> Optional[str]:
        """The bar from which this chain of snapshots has carried trade memory,
        as ISO text, or None before this manager has decided anything.

        Restored from a snapshot that records it; otherwise this manager's
        first bar, because a manager that restored no memory is where the
        memory starts. A session dated after the stamp opened with the previous
        session's trades in its prompt; the session it falls in did not, unless
        it was the month's first and had nothing to remember. The Live board
        deletes every superseded checkpoint, so its newest row is the only one
        left to say whether its month joined the two, and a bare version flag
        could not say where.
        """
        since = self.trade_memory_since
        if since is None:
            since = self._bar_stamp(0)
        return since.isoformat() if since is not None else None

    def _bar_stamp(self, index: int) -> Optional[pd.Timestamp]:
        """The stamp of equity point ``index`` (0 is the first bar this manager
        decided, -1 the last), or None before it has decided anything."""
        if not self.equity_history:
            return None
        return _memory_timestamp(self.equity_history[index].get("timestamp"))

    def restore_state(self, snapshot: Optional[Dict[str, Any]]) -> None:
        """Load cash/positions/T+1 lots and the 24h trade memory from
        ``snapshot_state`` (no-op if empty).

        The memory goes to ``restored_trades``, never ``trades``. A snapshot
        written before it existed restores none, quietly; a marked one that
        does not read back whole logs an ERROR (``_restore_trade_memory``).
        """
        if not snapshot:
            return
        self.cash = float(snapshot.get("cash", self.cash))
        self.positions = {
            str(symbol): float(shares)
            for symbol, shares in (snapshot.get("positions") or {}).items()
        }
        self.entry_prices = {
            str(symbol): float(price)
            for symbol, price in (snapshot.get("entry_prices") or {}).items()
        }
        available = snapshot.get("available_positions")
        if available is None:
            self.available_positions = dict(self.positions)
        else:
            self.available_positions = {
                str(symbol): float(shares) for symbol, shares in available.items()
            }
        lots_out: Dict[str, List[Dict[str, Any]]] = {}
        for symbol, lots in (snapshot.get("frozen_lots") or {}).items():
            restored: List[Dict[str, Any]] = []
            for lot in lots or []:
                quantity = float(lot.get("quantity") or 0)
                buy = lot.get("buy_date")
                if isinstance(buy, str):
                    try:
                        buy = date.fromisoformat(buy[:10])
                    except ValueError:
                        buy = None
                if not isinstance(buy, date):
                    # A snapshot is always restored in a later session, so a
                    # lot with no readable buy date was bought before today:
                    # release it rather than keep a lot the T+1 sweep would
                    # crash comparing against a date.
                    self.available_positions[str(symbol)] = (
                        self.available_positions.get(str(symbol), 0.0) + quantity
                    )
                    continue
                restored.append({"quantity": quantity, "buy_date": buy})
            if restored:
                lots_out[str(symbol)] = restored
        self.frozen_lots = lots_out
        self._restore_trade_memory(snapshot)

    def _restore_trade_memory(self, snapshot: Dict[str, Any]) -> None:
        """Load ``recent_trades`` into ``restored_trades`` and, from a snapshot
        that carried it whole, ``trade_memory_since``.

        Rows are re-read through ``_memory_trade``, put in time order and capped
        to the newest ``SNAPSHOT_TRADE_MEMORY_LIMIT``; an unusable one is
        skipped rather than allowed to fail the session.
        """
        memory = snapshot.get("recent_trades")
        parsed = sorted(
            (
                row
                for row in map(_memory_trade, memory if isinstance(memory, list) else [])
                if row is not None
            ),
            key=lambda row: row["timestamp"],
        )
        self.restored_trades = parsed[-SNAPSHOT_TRADE_MEMORY_LIMIT:]
        version = snapshot.get("trade_memory")
        if version is None:
            # Written before the memory existed: nothing was carried, nothing
            # is wrong, and the mark restarts at this manager's first bar.
            self.trade_memory_since = None
            return
        since = _memory_timestamp(snapshot.get("trade_memory_since"))
        # A marked snapshot holds only rows ``_memory_trade`` accepted and, being
        # written after a bar, a readable stamp. Less than that is a broken
        # contract (a renamed key, a changed row shape, a version this code does
        # not know), not a quiet night: nothing else would tell the two apart,
        # since the night just opens with less memory than it was given.
        intact = (
            version == TRADE_MEMORY_VERSION
            and isinstance(memory, list)
            and len(parsed) == len(memory)
            and since is not None
        )
        if not intact:
            if isinstance(memory, list):
                found = len(memory)
            else:
                found = "missing" if memory is None else type(memory).__name__
            print(
                f"ERROR: portfolio.trade_memory_unreadable version={version} "
                f"recent_trades={found} restored={len(parsed)} "
                f"since={'ok' if since is not None else 'unreadable'}",
                flush=True,
            )
        # Lost or unreadable: the mark restarts at this manager's first bar,
        # which can only record the switch later than it happened, never
        # earlier.
        self.trade_memory_since = since if intact else None

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

    def _coerce_position_size(self, raw: Any) -> Optional[float]:
        """A model's ``position_size`` as a number, or None when unparseable.

        json.loads happily yields strings, floats, Infinity and NaN here; a
        missing or null size reads as 0. Whole shares unless the market trades
        in lots, where the executor validates the quantity itself.
        """
        try:
            size = float(raw or 0) if self.lot_size > 1 else int(raw or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        if isinstance(size, float) and not math.isfinite(size):
            return None
        return size

    def _refuse_sell(
        self,
        *,
        symbol: str,
        raw_size: Any,
        timestamp: Any,
        price: float,
        strategy_reason: str,
    ) -> None:
        """Record a SELL the gate skipped because it cannot read its size.

        On the order ledger, like an executor rejection: a skip that is only a
        ``print`` is invisible from a dashboard backtest, whose stdout nobody
        reads, and a run whose model wrote ``"all"`` would publish a curve
        that never sold with nothing to say why.
        """
        asked = model_number(raw_size)
        _record_refused_order(
            rejected_orders=self.rejected_orders,
            order_events=self.order_events,
            order_event_repeats=self._order_event_repeats,
            timestamp=timestamp,
            symbol=symbol,
            action="sell",
            # A finite number, as every ledger field must be: "all" or NaN
            # asked for no count, and the reason says the size was refused.
            requested_shares=asked if asked is not None and math.isfinite(asked) else 0,
            price=price,
            reason=SELL_SIZE_REFUSED_REASON,
            strategy_reason=strategy_reason,
            costs_enabled=self.transaction_cost_profile is not None,
        )

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
            cutoff_time = timestamp - TRADE_MEMORY_WINDOW
            # Restored memory first: it is older than anything this manager
            # has traded itself, so the block stays in time order.
            for trade in chain(self.restored_trades, self.trades):
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
            
            # Named pools can contain thousands of securities: rank the whole
            # pool before creating the model's signal shortlist in either mode.
            # Legacy buy-and-hold retains its complete signal snapshot.
            if mode == "buy_and_hold" and not (market_context or {}).get("stock_pool"):
                symbols_to_include = [s for s in self.allowed_symbols if s in signals]
            else:
                # For safe_trading, rank by trend/momentum (NOT RSI extremity).
                # Ranking by |RSI-50| seeds a mean-reversion bias (fade winners,
                # buy losers) which loses in trending markets. Instead score each
                # name by trend confluence so the model is offered genuine momentum
                # opportunities, and ALWAYS include current holdings so it can
                # actively manage / exit weak positions.
                def _trend_score(sig: Dict) -> float:
                    score = _raw_trend_score(sig)
                    # A non-finite price or SMA makes the score NaN, and
                    # ``sorted`` has no defined order with NaN keys: one such
                    # name could scramble the ranking of every other name.
                    return score if math.isfinite(score) else float("-inf")

                def _raw_trend_score(sig: Dict) -> float:
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
                # Decide on the configured universe, not on how many names
                # happen to have a bar this timestamp: a 31-name universe with
                # one gap would otherwise flip between "everything" and "top 12
                # plus a note" from bar to bar -- the same path dependence this
                # cut was fixed for. ``max`` covers signals outside the
                # configured list (and an empty one); the set, because the
                # configured list is upper-cased but not de-duplicated.
                universe_size = max(len(self._allowed_set), len(trend_sorted))
                # Pipeline only. The single-prompt path keeps its shortlist:
                # it already lists every VALID SYMBOL, its contract caps a
                # reply at 10 actions, and the leaderboard's LLM entries run
                # it -- their cached curves (matched on strategy, window,
                # capital and prompt, nothing recording the snapshot) must
                # stay comparable with any one re-run beside them.
                show_whole_universe = (
                    bool(pipeline) and universe_size <= SNAPSHOT_FULL_UNIVERSE_MAX
                )
                if show_whole_universe:
                    symbols_to_include = [sym for sym, _ in trend_sorted]
                else:
                    symbols_to_include = [
                        sym for sym, _ in trend_sorted[:SNAPSHOT_SHORTLIST_SIZE]
                    ]
                ranked_count = len(symbols_to_include)
                # Guarantee every currently-held symbol is visible to the model
                for sym in holdings:
                    if sym in signals and sym not in symbols_to_include:
                        symbols_to_include.append(sym)
                if pipeline and not show_whole_universe:
                    # The model must be told the snapshot is partial: the
                    # pipeline prompt lists no VALID SYMBOLS and says only
                    # "trade ONLY symbols listed in the market snapshot".
                    # Placed just ahead of ``top_signals`` so the model reads
                    # it first, without moving any other key.
                    market_snapshot = _insert_before(
                        market_snapshot,
                        "top_signals",
                        "universe_note",
                        _universe_note(
                            ranked=ranked_count,
                            held=len(symbols_to_include) - ranked_count,
                            universe_size=universe_size,
                        ),
                    )
            
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
                for field in ("turnover", "market_cap_usd"):
                    value = signal.get(field)
                    if value is not None and pd.notna(value):
                        market_snapshot["top_signals"][symbol][field] = float(value)
                sic_code = signal.get("sic_code")
                if sic_code is not None and pd.notna(sic_code):
                    market_snapshot["top_signals"][symbol]["sic_code"] = int(sic_code)
                for field in ("market_cap_status", "industry", "sector"):
                    value = signal.get(field)
                    if value is not None and pd.notna(value):
                        market_snapshot["top_signals"][symbol][field] = str(value)
            
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
                # How many times the failing ceiling is re-asked before the
                # rescue call raises it. ``pipeline_runner`` escalates on its
                # FIRST retry and then stops: a reply already obtained at the
                # recovery ceiling has nothing different left to ask for, so a
                # second escalated request is the same request again.
                # ``LLM_ESCALATE_CEILING_ON_RETRY=1`` makes this loop agree --
                # one request at the configured ceiling, then the rescue call
                # below -- instead of paying for four identical failures first.
                # An empty reply is what a reply looks like when reasoning ran
                # out the output ceiling before any text was emitted, which is
                # not the kind of failure a same-ceiling retry fixes. The first
                # request and the rescue call are byte-identical either way;
                # only the count of same-ceiling retries between them moves.
                # The count governs the legacy client's empty reply only; the
                # execution layer's skips it (see the retry branch below).
                no_text_retries = 1 if escalate_ceiling_on_retry() else 4
                # One recovery budget (``RECOVERY_MAX_OUTPUT_TOKENS``) per
                # step, spent by whichever unusable reply claims it first: the
                # final rescue call below, or the truncation retry after the
                # parse. A reply still unusable at that budget has nothing
                # different left to ask for.
                recovery_spent = False
                output_delta = 0

                def request_decision(*, recovery: bool):
                    # Only a recovery attempt passes a ceiling, so every other
                    # request keeps the call shape it had before retries.
                    ceiling = (
                        {"max_tokens": RECOVERY_MAX_OUTPUT_TOKENS} if recovery else {}
                    )
                    return _request_trading_decision(
                        llm_client,
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        market_context=market_context,
                        **ceiling,
                    )

                attempt = 0
                while True:
                    rescue = attempt >= no_text_retries
                    if rescue:
                        # Final rescue: preserve the provider's reasoning mode,
                        # but give the model more room for reasoning plus JSON.
                        print(
                            "   ⚠️  Final rescue call with reasoning preserved "
                            f"and max_tokens={RECOVERY_MAX_OUTPUT_TOKENS}"
                        )
                        recovery_spent = True
                    try:
                        response = request_decision(recovery=rescue)
                        output_delta = self._record_llm_usage(response)
                        llm_response = _extract_response_text(response)
                        break
                    except (LLMExecutionError, AttributeError) as unusable:
                        legacy_empty = _is_legacy_empty_reply(unusable)
                        if rescue or not (legacy_empty or is_empty_reply(unusable)):
                            raise
                        if legacy_empty:
                            print(
                                f"   ⚠️  {unusable}; "
                                f"retry {attempt + 1}/{no_text_retries}"
                            )
                            attempt += 1
                        else:
                            # The execution layer rejects an empty reply itself
                            # (#600) and releases its reservation at zero
                            # (#556), so a same-ceiling retry is a generation
                            # the platform pays for and never bills. It goes
                            # straight to the rescue, as pipeline_runner's
                            # steps do. Any other execution error -- a request
                            # or result contract break, billing, credentials,
                            # quota -- was raised above, unretried.
                            print(
                                "   ⚠️  Empty model response; going straight "
                                "to the rescue call"
                            )
                            attempt = no_text_retries

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
                        try:
                            retry_response = request_decision(recovery=True)
                        except LLMExecutionError as retry_err:
                            # An empty second reply is the same outcome as the
                            # truncated first one: keep the first attempt.
                            if not is_empty_reply(retry_err):
                                raise
                            retry_response = None
                        if retry_response is not None:
                            self._record_llm_usage(retry_response)
                        retry_text = (
                            None
                            if retry_response is None
                            else response_text_or_none(retry_response)
                        )
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
                for llm_action in llm_actions:
                    if not isinstance(llm_action, dict):
                        raise LLMDecisionError(
                            "LLM returned an invalid action batch"
                        )
                    symbol = normalize_symbol(str(llm_action.get("symbol") or ""))
                    # A missing action is refused here, unlike the loop below.
                    action_type = normalize_action(llm_action.get("action"))
                    if (
                        symbol not in self._allowed_set
                        or action_type not in TRADING_ACTIONS
                    ):
                        raise LLMDecisionError(
                            "LLM returned an invalid action batch"
                        )
                    # The one order the gate cannot carry out as written: a
                    # buy sized from a confidence that is not a 0-1 number.
                    # Rejected like a bad symbol rather than skipped, so a
                    # billed run whose model answers 0-100 stops on the first
                    # bar instead of charging for a run that never buys.
                    if (
                        action_type == "buy"
                        and read_confidence(
                            llm_action.get("confidence"),
                            LLM_ACTION_DEFAULT_CONFIDENCE,
                        )
                        is None
                        and self._coerce_position_size(
                            llm_action.get("position_size", 0)
                        )
                        == 0
                    ):
                        raise LLMDecisionError(
                            "LLM returned a buy with no size and a confidence "
                            "that is not a number from 0 to 1"
                        )

            skipped_for_confidence = False
            # Whether any action named buy, sell or hold. A batch where none
            # did carried no decision the model can be credited with (#658).
            recognised_any = False
            for llm_action in llm_actions:
                symbol = normalize_symbol(str(llm_action.get("symbol") or ""))
                # Read by the same rule as the strict check above: two readings
                # let "buy " pass that check and then match no branch (#658).
                # A missing or null action holds, as an absent one always has.
                raw_action = llm_action.get("action")
                action_type = normalize_action(raw_action) or "hold"
                raw_confidence = llm_action.get("confidence")
                confidence = read_confidence(
                    raw_confidence, LLM_ACTION_DEFAULT_CONFIDENCE
                )
                raw_reasoning = llm_action.get("reasoning")
                reasoning = "" if raw_reasoning is None else str(raw_reasoning)

                if action_type not in TRADING_ACTIONS:
                    # Only a non-strict run gets here; strict_llm refused the
                    # batch above. Placing nothing silently read as a hold.
                    print(f"\n   ⚠️  {symbol}: Skip (unrecognised action {raw_action!r})")
                    continue
                recognised_any = True

                # A confidence that is not a number from 0 to 1 (25, NaN,
                # "high", a bool) is not read at all: no scale is guessed and
                # no default is put over it. It therefore cannot decide the
                # 0.3 floor either way, so an order that does not need it is
                # carried out as the model wrote it -- a sell (sized by
                # requested_sell_shares) or a buy with its own position_size. Only
                # a buy that would be sized from it is skipped, below: that is
                # where a 0-100 answer used to multiply the order by 100.
                if confidence is None:
                    conf_label = "unread"
                    self._note_unusable_confidence()
                    print(f"\n   Processing: {symbol} ({action_type.upper()}, conf={raw_confidence!r:.40})")
                    print(f"      ⚠️  {symbol}: confidence {raw_confidence!r:.40} is not a number from 0 to 1; not read")
                else:
                    conf_label = f"{confidence:.0%}"
                    print(f"\n   Processing: {symbol} ({action_type.upper()}, conf={conf_label})")
                print(f"      Reasoning: {reasoning[:60]}...")

                # Skip low-confidence decisions
                if confidence is not None and confidence < 0.3:
                    print(f"      ⏸️  Skipping (confidence {conf_label} too low)")
                    continue
                
                if symbol not in self._allowed_set:
                    print(f"   ❌ {symbol}: Outside selected universe, skipping")
                    continue
                
                signal = signals.get(symbol, {})
                price = float(signal.get("price", 0)) if signal.get("price") else 0.0
                
                if action_type == "buy":
                    # Use position_size from LLM directly, coerced defensively
                    # (``_coerce_position_size``).
                    raw_size = llm_action.get("position_size", 0)
                    shares = self._coerce_position_size(raw_size)
                    if shares is None:
                        print(f"      ⚠️  BUY {symbol}: Skip (unparseable position_size {raw_size!r})")
                        continue

                    if shares == 0 and confidence is None:
                        # Nothing to size it from (see above). A strict run
                        # rejected this order before the loop.
                        self.unusable_confidence_skipped += 1
                        skipped_for_confidence = True
                        print(f"      ⚠️  BUY {symbol}: Skip (no position_size, and no 0-1 confidence to size it from)")
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
                            "reason": f"[LLM] {reasoning} (confidence: {conf_label})",
                            "confidence": confidence
                        })
                        print(f"      ✅ BUY {symbol}: {shares} shares @ ${price:.2f} (conf: {conf_label})")
                    else:
                        print(f"      ⚠️  BUY {symbol}: Skip (insufficient cash: need ${shares*price:,.0f}, have ${self.cash:,.0f})")
                
                elif action_type == "sell":
                    held = self.positions.get(symbol, 0)
                    if held <= 0:
                        print(f"      ⚠️  SELL {symbol}: Skip (not in portfolio, only owns: {list(self.positions.keys())})")
                        continue
                    # Sell what was asked for (#661): position_size shares,
                    # else a size_fraction of the position (what a pipeline
                    # risk action sends), else the whole position. A size that
                    # cannot be read skips THIS action and is recorded; it is
                    # never read as "all of it" (requested_sell_shares).
                    reason = f"[LLM] {reasoning} (confidence: {conf_label})"
                    raw_size = llm_action.get("position_size")
                    requested, refusal = requested_sell_shares(
                        raw_size,
                        llm_action.get("size_fraction"),
                        held,
                        lot_size=self.lot_size,
                    )
                    if refusal is not None:
                        print(f"      ⚠️  SELL {symbol}: Skip ({refusal})")
                        self._refuse_sell(
                            symbol=symbol,
                            raw_size=raw_size,
                            timestamp=timestamp,
                            price=price,
                            strategy_reason=reason,
                        )
                        continue
                    # Size at the sellable quantity, not the held one: under T+1
                    # they differ, and the difference is unfillable.
                    sellable = self.sellable_shares(symbol)
                    wanted = min(requested, held)
                    if sellable < wanted:
                        self.record_t1_deferral(symbol, wanted, sellable)
                    shares = min(wanted, sellable)
                    if shares > 0:
                        action = {
                            "symbol": symbol,
                            "action": "sell",
                            "shares": shares,
                            "reason": reason,
                            "confidence": confidence
                        }
                        if shares < requested:
                            # The reason string still carries the model's own
                            # words ("exit the full position"); without this the
                            # log would read as though it asked for `shares`.
                            action["requested_shares"] = requested
                        actions.append(action)
                        print(f"      ✅ SELL {symbol}: {shares} shares @ ${price:.2f} (conf: {conf_label})")
                    else:
                        print(f"      ⚠️  SELL {symbol}: Skip (T+1 frozen, {held} shares settle next trading day)")
                
                # else: HOLD is implicit (don't add to actions)

            if not recognised_any:
                # Like an unparseable reply on this non-strict path: no actions
                # and no H6 credit, since nothing the model sent named a trade
                # or a hold. Counting it published a curve the model never
                # drove under its name (#658).
                print("   ⚠️  No action named buy, sell or hold; holding this step\n")
                return {"actions": []}

            # H6 coverage increment #2 of 2 — the main success exit: the model's
            # actions were parsed AND processed without error, so the returned
            # decision is genuinely the model's. This is the H6 coverage
            # numerator, distinct from llm_calls (billed calls): any exception in
            # the processing loop above is caught below and swapped for a
            # rule-based decision, which must NOT count. (Actions filtered for
            # low confidence / invalid symbol still count: the model drove the
            # step, our policy merely declined to act on it. A batch in which
            # no action was recognised returned above without counting.)
            #
            # An unsized buy skipped for a confidence that is not a 0-1 number
            # still counts here too -- no rule-based trade replaced it, and the
            # decision tape reads an unchanged llm_decisions as a fallback bar
            # (``decision_tape.driver_for``). It is the model's broken contract,
            # not our policy, so the step is counted a second time in
            # confidence_skipped_steps, which the leaderboard's H6 guard
            # subtracts: a model that answers 0-100 on every bar must not
            # publish a flat curve at 100% coverage.
            #
            # These two sites are the ONLY writes to llm_decisions. Adding a
            # third means re-checking that the step really was model-driven with
            # no rule-based fallback — that invariant is what the leaderboard's
            # H6 guard rests on.
            if skipped_for_confidence:
                self.confidence_skipped_steps += 1
            self.llm_decisions += 1
            print(f"   ✅ Total actions: {len(actions)}\n")
            return {"actions": actions}

        except LLMDecisionError as strict_error:
            if getattr(llm_client, "fail_closed", False):
                raise
            return self._absorb_strict_llm_failure(portfolio_state, strict_error)
        except Exception as e:
            if strict_llm:
                if getattr(llm_client, "fail_closed", False) and not (
                    is_transient_provider_failure(e)
                ):
                    raise
                # Same treatment as an explicit strict violation. Built rather
                # than raised because the budget may still absorb it; the cause
                # is attached by hand so the traceback survives either way.
                strict_error = LLMDecisionError("LLM decision processing failed")
                strict_error.__cause__ = e
                return self._absorb_strict_llm_failure(
                    portfolio_state,
                    strict_error,
                    hold=is_transient_provider_failure(e),
                )
            print(f"\n❌ LLM decision error: {e}")
            print(f"   Falling back to rule-based logic\n")
            return self.make_trading_decision(portfolio_state)

    def _note_unusable_confidence(self) -> None:
        """Count one action whose confidence is not a 0-1 number.

        The first one of a run also prints a flushed ``ERROR: llm.`` line,
        which a dashboard backtest's parent relays to the service log as it
        arrives: the per-action lines go to a subprocess's stdout, and a model
        that answers 0-100 on every bar is a whole-run contract break that a
        per-item line cannot report. Fixed tokens only -- the raw value is the
        model's text, and the parent's redaction expects none in this line.
        """
        self.unusable_confidence_actions += 1
        if self._unusable_confidence_reported:
            return
        self._unusable_confidence_reported = True
        print(
            "ERROR: llm.unusable_confidence "
            "detail=not_a_number_from_0_to_1 policy=unsized_buys_skipped",
            flush=True,
        )

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
        budget = int(total_steps * STRICT_LLM_MAX_FALLBACK_RATIO)
        if budget == 0 and total_steps >= STRICT_LLM_MIN_STEPS_FOR_ONE_STRIKE:
            return 1
        return budget

    def _absorb_strict_llm_failure(
        self,
        portfolio_state: Dict,
        error: LLMDecisionError,
        *,
        hold: bool = False,
    ) -> Dict:
        """Spend one strike on an unusable response, or abort the run.

        The step falls back to rule-based logic, or holds when ``hold`` is set,
        and deliberately does NOT increment ``llm_decisions``: the model did
        not drive it, so it counts against H6 coverage exactly as an ordinary
        fallback would.

        A provider outage holds because it says nothing about the market, and
        rule-based trades would put orders into a user's LLM run that neither
        the model nor the user chose. Other strikes keep the rule-based
        fallback that existing strict runs, leaderboard curves included, were
        built with.
        """
        self.strict_llm_fallbacks += 1
        budget = self.strict_llm_fallback_budget()
        if self.strict_llm_fallbacks > budget:
            raise error
        if hold:
            print(
                f"\n⚠️  Strict LLM step failed ({error}); holding this step. "
                f"Strike {self.strict_llm_fallbacks}/{budget}.\n"
            )
            return {"actions": []}
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
        execution_prices: Optional[Dict] = None,
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
            execution_prices=execution_prices,
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
