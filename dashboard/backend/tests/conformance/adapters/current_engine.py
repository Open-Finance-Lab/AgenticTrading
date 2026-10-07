"""Adapter D: the My Agents dashboard engine (``HourlyBacktester``), §8.1-8.2.

The scored run drives the real engine end to end -- ``load_data``,
``calculate_indicators``, ``run_agent_backtest`` -- with three seams replaced:

* market data: each case bar becomes open-stamped 5m source bars that
  aggregate back to it exactly (§8.1), served by an in-memory loader, so no
  Alpaca or network call is made;
* the run database: ``engine.db`` records ``insert_*`` calls in memory;
* the decision: ``PortfolioManager.make_trading_decision`` (the rule-based
  seam, ``engine.py:2189``) returns the case's intents for that decision bar,
  bypassing decision-time sizing and pre-checks. Everything from there on --
  fill planning, execution, marking, metrics -- is the engine's own code.

The DL probe (``llm=True``) instead routes every step through
``PortfolioManager.make_trading_decision_with_llm`` (``engine.py:2173``) with a
stub client that answers the case's intents as LLM JSON, so the LLM
translator's sizing and pre-checks (D18) are what get measured. No network.
It runs non-strict, the path the design doc's C04 prediction describes; a
My Agents worker run is strict (``engine.py:388``) and would abort on C04's
oversized batch instead.

Mapping the engine onto the ``OrderFinal`` vocabulary. Orders come from the
in-memory ledger ``pm.order_events``, never from run metadata, which keeps
only unfilled events, caps them at 200 and carries no ids (``engine.py:193-217``).
A pure rejection that repeats on one trading day is collapsed into one event
with a ``repeat_count`` (``execution.py:197-240``); it is expanded back, the
first copy to the intent named in its ``strategy_reason`` and each further copy
to the next submitted intent with the same symbol, side and trading date.

=========================  ==========================================
engine ``status/reason``   ``OrderFinal``
=========================  ==========================================
``filled``                 ``filled``
``partial`` + ``insufficient_position``  ``partially_filled/clipped_to_position``
``rejected`` + ``insufficient_position`` ``rejected/no_position``
``rejected`` + ``insufficient_cash``     ``rejected/insufficient_cash``
anything else              passed through verbatim (and so fails)
=========================  ==========================================

An intent with no ledger entry -- a decision that is not a step, or a fill
bar with no source bar (``execution.py:526-527``) -- yields no ``OrderFinal``.
Fills come from the persisted trade rows (``engine._serialize_trades``), whose
``quantity`` is an int; every quantity the D runs submit is whole. Fields the
engine has no output for (realized P&L, regulatory fees) are ``None``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd

from dashboard.backend.domain.backtesting import engine as engine_mod
from dashboard.backend.domain.backtesting.engine import HourlyBacktester
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    FRAME_ATTR_END_CLAMPED,
    FRAME_ATTR_FEED,
    FRAME_ATTR_SIP_FALLBACK,
)
from dashboard.backend.infrastructure.market_data.sessions import (
    FRAME_ATTR_OPEN_STAMPED_MINUTES,
)

from ..model import (
    NEW_YORK,
    Case,
    Expected,
    Fill,
    NotExpressible,
    OrderFinal,
    OrderIntent,
    to_decimal,
)
from ..reference import grid

SOURCE_MINUTES = 5
_FIVE = timedelta(minutes=SOURCE_MINUTES)
_LLM_REASON = re.compile(r"^\[LLM\] (\S+) \(confidence")


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"naive timestamp {value!r}")
    return stamp.tz_convert("UTC")


def _number(value: Decimal):
    """The JSON-ish number an agent would send: int when whole."""
    return int(value) if value == value.to_integral_value() else float(value)


# --------------------------------------------------------------------------
# §8.1 bar synthesis
# --------------------------------------------------------------------------


def synthesize_source_bars(case: Case) -> Dict[str, pd.DataFrame]:
    """Open-stamped 5m bars per symbol that aggregate back to the case bars.

    A bar lasting ``n`` x 5m becomes ``sub0 = (O, H, L, C, V/n)`` followed by
    ``n - 1`` bars flat at ``C`` (``V/n`` each): the 5m bar opening at a
    decision stamp is ``sub0`` of the next case bar, so its open is ``O(t+1)``,
    and every intrabar mark equals the bar-close mark. A missing case bar
    contributes no rows. The only padding is before a symbol's first bar
    within the case's first session, flat at that bar's open; nothing is ever
    padded after the last bar (T03 depends on that).
    """
    first_day = min(bar.ts for bar in case.bars).date()
    rows: Dict[str, List[tuple]] = {}
    for symbol in dict.fromkeys(bar.symbol for bar in case.bars):
        bars = sorted((b for b in case.bars if b.symbol == symbol), key=lambda b: b.ts)
        out = rows.setdefault(symbol, [])
        first = bars[0]
        if first.ts.date() == first_day:
            stamp = first.ts.replace(hour=9, minute=30)
            pad = float(first.open)
            while stamp < first.ts:
                out.append((stamp, pad, pad, pad, pad, float(first.volume) / 12))
                stamp += _FIVE
        for bar in bars:
            n = bar.minutes // SOURCE_MINUTES
            volume = float(bar.volume) / n
            o, h, l, c = (float(x) for x in (bar.open, bar.high, bar.low, bar.close))
            out.append((bar.ts, o, h, l, c, volume))
            for k in range(1, n):
                out.append((bar.ts + k * _FIVE, c, c, c, c, volume))
    frames = {}
    for symbol, data in rows.items():
        index = pd.DatetimeIndex([row[0] for row in data]).tz_convert("America/New_York")
        frame = pd.DataFrame(
            [row[1:] for row in data],
            columns=["open", "high", "low", "close", "volume"],
            index=index,
        )
        frame.attrs[FRAME_ATTR_FEED] = "sip"
        frame.attrs[FRAME_ATTR_OPEN_STAMPED_MINUTES] = SOURCE_MINUTES
        frame.attrs[FRAME_ATTR_SIP_FALLBACK] = False
        frame.attrs[FRAME_ATTR_END_CLAMPED] = False
        frames[symbol] = frame
    return frames


class _CaseLoader:
    """The provider double: the case's 5m bars, whatever window is asked."""

    def __init__(self, frames):
        self.frames = frames
        self.source_timeframe = None

    def configure_source_timeframe(self, value):
        self.source_timeframe = value

    def fetch_bars(self, symbols, start_date, end_date):
        return {s: self.frames[s].copy() for s in symbols if s in self.frames}


class _RecordingDB:
    def __init__(self):
        self.runs, self.equity_points, self.trades = [], [], []

    def insert_run(self, **kwargs):
        self.runs.append(kwargs)

    def insert_equity_points(self, run_id, points):
        self.equity_points.append((run_id, list(points)))

    def insert_trades(self, run_id, trades):
        self.trades.extend(trades)

    def insert_decisions(self, run_id, decisions):
        pass


# --------------------------------------------------------------------------
# The DL probe's stub LLM
# --------------------------------------------------------------------------


class _StubLLMClient:
    """Answers each decision with the case's intents as LLM decision JSON."""

    #: ``make_trading_decision_with_llm`` takes the LLM path only when the
    #: Anthropic SDK is importable or the client looks like the unified
    #: execution client (``portfolio_manager.py:406-407``); this keeps the
    #: probe off the rule-based fallback in an environment without the SDK.
    execution_service = "conformance-stub"

    def __init__(self, intents_at, symbols):
        self.intents_at = intents_at
        self.symbols = symbols
        self.state = None  # the portfolio state of the step being decided
        self.pm = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_request):
        actions = [self._action(intent) for intent in self.intents_at.get(_utc(self.state["timestamp"]), ())]
        if not actions:
            # An explicit hold: an empty list would send a non-strict run to
            # the rule-based fallback (portfolio_manager.py:783-796).
            actions = [{"symbol": self.symbols[0], "action": "hold", "confidence": 1.0, "reasoning": "hold"}]
        text = json.dumps({"actions": actions})
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
            stop_reason="end_turn",
        )

    def _action(self, intent: OrderIntent) -> dict:
        held = Decimal(str(self.pm.positions.get(intent.symbol, 0)))
        if intent.target_weight is not None:
            # What a model that wants weight w writes: the share delta at the
            # decision close, in the only unit the schema has (shares).
            price = Decimal(str(self.state["market_signals"][intent.symbol]["price"]))
            equity = Decimal(str(self.state["total_equity"]))
            delta = intent.target_weight * equity / price - held
            side, size = ("buy" if delta > 0 else "sell"), abs(delta)
        else:
            side, size = intent.side, held if intent.qty == "all" else intent.qty
        return {
            "symbol": intent.symbol,
            "action": side,
            "position_size": _number(size),
            "confidence": 1.0,
            "reasoning": intent.id,
        }


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class CurrentEngineAdapter:
    name = "atl-dashboard"
    capabilities = frozenset({"market"})

    def __init__(self, monkeypatch, *, llm: bool = False):
        self.monkeypatch = monkeypatch
        self.llm = llm
        if llm:
            self.name = "atl-dashboard-llm"

    def _check_expressible(self, case: Case) -> None:
        for intent in case.intents:
            if intent.type != "market":
                raise NotExpressible(f"D12: {intent.id} is a {intent.type} order; actions are market-only")
            if intent.tif != "day":
                raise NotExpressible(f"D12: {intent.id} is {intent.tif.upper()}; actions carry no time in force")
            if intent.oco_group is not None:
                raise NotExpressible(f"D12: {intent.id} is an OCO leg; actions carry no order links")
            if intent.target_weight is not None and not self.llm:
                raise NotExpressible(f"D9: {intent.id} is a target weight; actions carry shares only")

    def run(self, case: Case) -> Expected:
        self._check_expressible(case)
        stamps = grid(case)
        minutes_at = dict(stamps)
        symbols = list(dict.fromkeys(bar.symbol for bar in case.bars))
        intents_at: Dict[pd.Timestamp, List[OrderIntent]] = {}
        for intent in case.intents:
            close = intent.decision_bar + timedelta(minutes=minutes_at[intent.decision_bar])
            intents_at.setdefault(_utc(close), []).append(intent)

        loader = _CaseLoader(synthesize_source_bars(case))

        def factory(data_source="alpaca", universe=None, *, source_timeframe=None):
            loader.configure_source_timeframe(source_timeframe)
            return loader

        fake_db = _RecordingDB()
        mp = self.monkeypatch
        mp.setattr(engine_mod, "create_market_data_provider", factory)
        mp.setattr(engine_mod, "db", fake_db)

        holder = SimpleNamespace(pm=None, submitted=[], avg_path=[], fallbacks=0)
        original_init = PortfolioManager.__init__
        original_execute = PortfolioManager.execute_actions
        original_rule_based = PortfolioManager.make_trading_decision
        original_llm = PortfolioManager.make_trading_decision_with_llm

        def init(pm, *args, **kwargs):
            original_init(pm, *args, **kwargs)
            holder.pm = pm

        def execute(pm, actions, market_data, timestamp, *args, **kwargs):
            before = len(pm.trades)
            original_execute(pm, actions, market_data, timestamp, *args, **kwargs)
            for trade in pm.trades[before:]:
                avg = pm.entry_prices.get(trade["symbol"])
                holder.avg_path.append((trade.get("reason", ""), to_decimal(avg)))

        def scripted(pm, state):
            ts = _utc(state["timestamp"])
            actions = []
            for intent in intents_at.get(ts, ()):
                holder.submitted.append((ts, intent))
                shares = pm.positions.get(intent.symbol, 0) if intent.qty == "all" else _number(intent.qty)
                actions.append(
                    {"symbol": intent.symbol, "action": intent.side, "shares": shares, "reason": intent.id}
                )
            return {"actions": actions}

        mp.setattr(PortfolioManager, "__init__", init)
        mp.setattr(PortfolioManager, "execute_actions", execute)

        stub = None
        if self.llm:
            stub = _StubLLMClient(intents_at, symbols)

            def via_llm(pm, state, *args, **kwargs):
                stub.state, stub.pm = state, pm
                for intent in intents_at.get(_utc(state["timestamp"]), ()):
                    holder.submitted.append((_utc(state["timestamp"]), intent))
                return original_llm(pm, state, *args, **kwargs)

            def counted_fallback(pm, state):
                holder.fallbacks += 1
                return original_rule_based(pm, state)

            mp.setattr(PortfolioManager, "make_trading_decision_with_llm", via_llm)
            mp.setattr(PortfolioManager, "make_trading_decision", counted_fallback)
        else:
            mp.setattr(PortfolioManager, "make_trading_decision", scripted)

        first_day, last_day = stamps[0][0].date(), stamps[-1][0].date()
        backtester = HourlyBacktester(
            first_day.isoformat(),
            last_day.isoformat(),
            use_llm=False,
            symbols=symbols,
            initial_capital=float(case.initial_cash),
        )
        # Pin the engine's own exclusive bound (end + 1 day). Unpinned, a
        # window dated after today (T05's 2026-11-27) is clamped to today and
        # refused before any bar loads; the in-memory loader ignores it anyway.
        backtester.provider_end_date = (last_day + timedelta(days=1)).isoformat()
        if stub is not None:
            backtester.use_llm = True
            backtester.llm_client = stub
        backtester.load_data()
        backtester.calculate_indicators()
        _run_id, curve = backtester.run_agent_backtest()
        self.fallbacks = holder.fallbacks
        return self._result(case, stamps, holder, fake_db, curve)

    # ---------------------------------------------------------------- outputs

    def _intent_id(self, reason: str, ids) -> Optional[str]:
        if reason in ids:
            return reason
        match = _LLM_REASON.match(reason or "")
        return match.group(1) if match and match.group(1) in ids else None

    def _result(self, case, stamps, holder, fake_db, curve) -> Expected:
        ids = {intent.id for intent in case.intents}
        pm = holder.pm

        opens = {_utc(ts): ts for ts, _ in stamps}
        closes = {_utc(ts + timedelta(minutes=m)): ts for ts, m in stamps}

        def case_bar(stamp) -> datetime:
            stamp = _utc(stamp)
            if stamp in opens:
                return opens[stamp]
            if stamp in closes:  # a close fill (the 16:00 bucket) belongs to the bar it closes
                return closes[stamp]
            earlier = [ts for key, ts in opens.items() if key < stamp]
            return max(earlier) if earlier else stamp.tz_convert(NEW_YORK).to_pydatetime()

        fills = []
        for row in fake_db.trades:
            fills.append(
                Fill(
                    order_id=self._intent_id(row.get("reason", ""), ids) or row.get("reason", ""),
                    bar=case_bar(row["timestamp"]),
                    price=to_decimal(row["price"]),
                    qty=to_decimal(row["quantity"]),
                    commission=to_decimal(row.get("commission")),
                    sec_fee=None,
                    taf=None,
                    slippage_cost=to_decimal(row.get("slippage_amount")),
                    side=str(row.get("side", "")).lower() or None,
                )
            )

        orders = self._orders(pm, holder.submitted, ids)

        by_stamp = {_utc(point["timestamp"]): to_decimal(point["equity"]) for point in curve}
        equity = []
        for ts, minutes in stamps:
            close = ts + timedelta(minutes=minutes)
            value = by_stamp.get(_utc(close - _FIVE))  # the bar's last 5m mark
            if value is not None:
                equity.append((close, value))

        run = fake_db.runs[0]
        avg_path = [
            (self._intent_id(reason, ids) or reason, avg) for reason, avg in holder.avg_path
        ]
        return Expected(
            fills=fills,
            orders=orders,
            cash=to_decimal(pm.cash),
            positions={s: to_decimal(q) for s, q in pm.positions.items()},
            avg_cost={s: to_decimal(p) for s, p in pm.entry_prices.items()},
            realized_pnl=None,  # D14: no realized P&L field
            initial_equity=to_decimal(run["initial_equity"]),
            equity=equity,
            final_equity=to_decimal(run["final_equity"]),
            total_return=to_decimal(run["total_return"]),
            max_drawdown=to_decimal(run["max_drawdown"]),
            sharpe=to_decimal(run["sharpe_ratio"]),
            avg_cost_path=avg_path,
        )

    def _orders(self, pm, submitted: List[Tuple[pd.Timestamp, OrderIntent]], ids) -> List[OrderFinal]:
        named = {self._intent_id(e.get("strategy_reason", ""), ids) for e in pm.order_events}
        claimed = set()
        out: List[OrderFinal] = []

        def emit(order_id: str, event: dict) -> None:
            status, reason = _map_status(event)
            claimed.add(order_id)
            out.append(OrderFinal(order_id, status, reason, to_decimal(event["executed_shares"])))

        for event in pm.order_events:
            order_id = self._intent_id(event.get("strategy_reason", ""), ids)
            emit(order_id or event.get("strategy_reason", ""), event)
            day = _utc(event["timestamp"]).tz_convert(NEW_YORK).date()
            side = str(event["side"]).lower()
            for _ in range(int(event.get("repeat_count", 1)) - 1):
                # A collapsed repeat: the next submitted intent with the same
                # symbol, side and trading day that has no event of its own.
                for ts, intent in submitted:
                    if (
                        intent.symbol == event["symbol"]
                        and intent.side == side
                        and ts.tz_convert(NEW_YORK).date() == day
                        and intent.id not in claimed
                        and intent.id not in named
                    ):
                        emit(intent.id, event)
                        break
        return out


def _map_status(event: dict) -> Tuple[str, str]:
    status, reason = event.get("status"), event.get("reason") or ""
    if status == "filled":
        return "filled", ""
    if status == "partial":
        return "partially_filled", {"insufficient_position": "clipped_to_position"}.get(reason, reason)
    if status == "rejected":
        return "rejected", {"insufficient_position": "no_position"}.get(reason, reason)
    return str(status), reason
