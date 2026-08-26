"""
External agent backtest service — step-by-step hourly loop via HTTP API.

Each trading hour waits up to DECISION_TIMEOUT_SECONDS for POST /decisions;
otherwise the step auto-holds (no trades).

Canonical location (Phase 3C1). Moved verbatim from
``dashboard/backend/external_backtest_service.py``; the original module was
removed in Phase 4A. Public classes, functions, constants, singletons,
signatures, return schemas, exceptions, logging, persistence, result
serialization, and backtest orchestration are unchanged; only the module
location moved.
"""

from __future__ import annotations

import os
import uuid
import threading
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any, Dict, List, Optional

import pandas as pd

import dashboard.backend.infrastructure.llm.token_cost as token_cost
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation
from dashboard.backend.domain.agents.repository import agent_store
from dashboard.backend.database import db
from dashboard.backend.execution.base import TERMINAL_STATUSES
from dashboard.backend.infrastructure.llm.validator import (
    DJIA_30,
    actions_to_executable,
    parse_actions_payload,
)
from dashboard.backend.domain.backtesting.constants import resolve_initial_capital
from dashboard.backend.domain.backtesting.metrics import (
    calculate_max_drawdown,
    calculate_sharpe,
)
from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader
from dashboard.backend.infrastructure.market_data.profiles import (
    ALPACA,
    get_market_profile,
)

from dashboard.backend.domain.backtesting import (
    baseline_worker,
    constants as _constants,
    features as _features,
    market_data_store,
)

# Canonical re-exports. external_run_service is the single wiring point that names
# these concrete symbols; guard tests (test_canonical_consumers,
# test_external_run_service_move) assert `svc.TechnicalIndicators` is that class and
# `svc.INITIAL_CAPITAL` is the canonical constant. Bound by assignment rather than a
# bare `from ... import X` so each re-export is explicit and static analysis sees it
# as used — no py/unused-import false positive. Do not remove (deleting either
# reddens those tests). resolve_initial_capital stays a direct import: it's called
# in the body, so it's a genuine use, not a re-export.
TechnicalIndicators = _features.TechnicalIndicators
INITIAL_CAPITAL = _constants.INITIAL_CAPITAL

DECISION_TIMEOUT_SECONDS = int(os.getenv("EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS", "60"))

_sessions: Dict[str, "ExternalBacktestSession"] = {}
_lock = threading.Lock()

# Budgets for the legacy ``/api/v1/backtest/*`` surface.
#
# That surface authenticates nothing: ``_require_session`` accepts any non-empty
# ``X-Session-Id``, so a bare POST /api/v1/backtest/start — no account, no agent,
# no API key — used to spawn a thread, load a market-data window into memory and
# add a session to the dict below, without any limit whatsoever. It writes no
# ``protocol_runs`` row either, so none of the three protocol caps
# (per-agent / per-account / global) can see one: the entitlement plane could
# hold every authenticated agent to its quota while this route ran unbounded
# beside it.
#
# Note this is the shipping SDK's path too (packaging/agentictrading calls
# POST /api/v1/backtest/start), not only an abuse surface — hence a per-session
# budget of 5, matching MAX_ACTIVE_RUNS_PER_AGENT, rather than something
# punitive. It bounds one naive or looping client; keyed on a header the caller
# chooses, it is not a bound against someone who rotates it.
#
# MAX_LEGACY_ACTIVE_GLOBAL is the one that holds regardless, and it counts
# **every** resident session in the dict below — including the ones the v1/v2
# protocol surfaces open through run_service.create_run, which share this
# registry. That is deliberate on both halves: the quantity worth bounding is
# resident bar windows on a free-tier box, whoever opened them; and when that
# is scarce, the surface that gets refused first should be the one with no
# account, agent or key behind it. So this ceiling can refuse a legacy start
# while protocol runs (bounded separately by MAX_ACTIVE_RUNS_GLOBAL, 100) keep
# creating — never the reverse.
#
# Both are env-overridable, and 0 disables (an operator who wants the old
# unbounded behaviour back can have it explicitly).
MAX_LEGACY_ACTIVE_PER_SESSION = int(os.getenv("MAX_LEGACY_ACTIVE_PER_SESSION", "5"))
MAX_LEGACY_ACTIVE_GLOBAL = int(os.getenv("MAX_LEGACY_ACTIVE_GLOBAL", "50"))

# Neither of the caps above bounds a session once it goes terminal --
# _count_active_locked skips anything in TERMINAL_STATUSES, and reap_runs
# only walks _runs (the protocol registry), which this legacy surface never
# populates. A terminal session left in _sessions is otherwise permanent,
# each one still pinning a loaded bar window. sweep_terminal_sessions()
# below rides the existing reaper pass to drop them after this many seconds.
#
# Parsed defensively, same shape as ``_max_active_dashboard_backtests`` in
# ``api/routers/backtests.py``: a typo'd operator value must not raise at
# import, so a bad value falls back to the default with a log line. No range
# check -- unlike a concurrency cap, there is no "0 is a legitimate setting,
# negative is a typo" distinction to enforce here.
_DEFAULT_LEGACY_SESSION_RETENTION_SECONDS = 300


def _legacy_session_retention_seconds() -> int:
    raw = os.getenv("LEGACY_SESSION_RETENTION_SECONDS")
    if raw is None or not str(raw).strip():
        return _DEFAULT_LEGACY_SESSION_RETENTION_SECONDS
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        print(
            "LEGACY_SESSION_RETENTION_SECONDS is not an integer "
            f"({raw!r}); using {_DEFAULT_LEGACY_SESSION_RETENTION_SECONDS}",
            flush=True,
        )
        return _DEFAULT_LEGACY_SESSION_RETENTION_SECONDS


LEGACY_SESSION_RETENTION_SECONDS = _legacy_session_retention_seconds()


class BacktestCapacityError(Exception):
    """Legacy start refused: a session or the server is at capacity.

    Carries the numbers so the router can build a 429 body without re-deriving
    them (and without a second, racy count).
    """

    def __init__(self, scope: str, active: int, limit: int) -> None:
        self.scope = scope
        self.active = active
        self.limit = limit
        super().__init__(f"{scope} at capacity ({active} of {limit} active)")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _new_ext_run_id() -> str:
    """Unique id for a finalized external run.

    Second-resolution timestamp for human-readable ordering PLUS a uuid suffix.
    The bare ``ext_<YYYYMMDD_HHMMSS>`` scheme used before collided when two runs
    finalized in the same second: the id is a PRIMARY KEY written with
    ``INSERT OR REPLACE``, so a collision silently overwrote the earlier run's
    rows — and a cached ``/runs/{id}/plot.png`` would then serve the wrong run's
    chart forever. Only ``str.startswith("ext_")`` and a ``created_at`` sort ever
    depend on this value, so the suffix is safe.
    """
    return f"ext_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def build_compare_url(
    run_id: Optional[str], baseline_run_ids: Dict[str, str],
) -> Optional[str]:
    """Compare-view link: the run against its baselines (run, djia, buy&hold).

    Module-level so the live session and the archived-run tombstone build the
    exact same URL — the ordering is part of the contract (finding: the
    tombstone used to drop compare_url entirely)."""
    if not run_id:
        return None
    ids = [run_id]
    if baseline_run_ids.get("djia"):
        ids.append(baseline_run_ids["djia"])
    if baseline_run_ids.get("buy_and_hold"):
        ids.append(baseline_run_ids["buy_and_hold"])
    return f"/compare?run_ids={','.join(ids)}"


def build_final_metrics(run: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The completed-run metrics block, from an agent_runs row.

    Module-level (like build_compare_url) so the live session's get_status and
    the archived-run tombstone hand out the exact same shape — a new metric
    can't be added to one surface and silently dropped from the other. Returns
    {} for a missing row."""
    if not run:
        return {}
    return {
        "total_return": run.get("total_return"),
        "sharpe_ratio": run.get("sharpe_ratio"),
        "max_drawdown": run.get("max_drawdown"),
        "num_trades": run.get("num_trades"),
        "final_equity": run.get("final_equity"),
        "llm_calls": run.get("llm_calls"),
        "input_tokens": run.get("input_tokens"),
        "output_tokens": run.get("output_tokens"),
        "est_cost_usd": run.get("est_cost_usd"),
        "timeout_holds": (run.get("metadata") or {}).get("timeout_holds")
        if isinstance(run.get("metadata"), dict) else None,
    }


class ExternalBacktestSession:
    """One external-agent backtest driven hour-by-hour through the API."""

    def __init__(
        self,
        *,
        backtest_id: str,
        session_id: str,
        agent_name: str,
        model_name: str,
        start_date: str,
        end_date: str,
        mode: str = "safe_trading",
        symbols: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        initial_capital: Optional[float] = None,
    ):
        self.backtest_id = backtest_id
        self.session_id = session_id
        self.agent_name = agent_name
        self.model_name = model_name
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        # Declared tradeable allow-list (protocol runs' config.symbols).
        # When set, market snapshots carry features for every one of these
        # symbols; when None (legacy external flow), the historical top-10
        # RSI sampling is preserved.
        self.symbols = list(symbols) if symbols else None

        self.status = "loading"
        self.error: Optional[str] = None
        self.step_index = 0
        self.total_steps = 0
        self.run_id: Optional[str] = run_id
        self.baseline_run_ids: Dict[str, str] = {}
        # v2 backends record the hash of the exact context served per step here;
        # threaded into the decision log so each decision traces to its context.
        self.context_ref_by_step: Dict[int, str] = {}

        self.initial_capital = resolve_initial_capital(initial_capital)
        # This service is Alpaca-only by construction (it loads bars through
        # AlpacaDataLoader and neither /api/v1 nor /api/v2 accepts a
        # data_source). Resolve execution rules from the profile registry
        # anyway, so wiring a second data source here carries its settlement
        # semantics with it. Defaulting the flag instead would make a future
        # A-share run silently execute with US same-day settlement — no error,
        # no log, just wrong fills.
        self.profile = get_market_profile(ALPACA)
        self.manager = PortfolioManager(
            initial_capital=self.initial_capital,
            t_plus_one_enabled=self.profile.t_plus_one_enabled,
        )
        self.all_data: Dict[str, pd.DataFrame] = {}
        self.timestamps: List[Any] = []
        self.price_cache: Dict[str, Dict[Any, float]] = {}

        self.step_opened_at: Optional[datetime] = None
        # Stamped by sweep_terminal_sessions() the first time it sees this
        # session in a terminal status; None until then. Explicit default so
        # the attribute is discoverable rather than a bare getattr() away.
        self.terminal_seen_at: Optional[datetime] = None
        self.last_decision_source: Optional[str] = None
        self.decision_log: List[Dict[str, Any]] = []
        self.last_executed: List[Dict[str, Any]] = []

        # Estimated token usage. The agent's LLM runs client-side, so we
        # approximate input tokens from the context we serve each hour and
        # output tokens from the decisions the agent submits.
        self.llm_calls = 0
        self.est_input_tokens = 0
        self.est_output_tokens = 0
        # Integrity counter: steps auto-held because their decision deadline
        # passed (see _advance_step). Surfaced at every result surface.
        self.timeout_holds = 0

        self._step_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_market_data(self) -> None:
        # loader_factory resolves the module-global name at call time, so every
        # existing monkeypatch of ebs.AlpacaDataLoader still controls the fetch.
        dataset = market_data_store.get_dataset(
            DJIA_30, self.start_date, self.end_date,
            loader_factory=AlpacaDataLoader,
        )
        self.adopt_dataset(dataset)

    def adopt_dataset(self, dataset: "market_data_store.MarketDataset") -> None:
        """Attach a built shared dataset and open step 0.

        SHARED + READ-ONLY: all_data/timestamps/price_cache belong to the
        store and other sessions — never mutate them (see the store docstring).

        Publish the loaded state under the step lock, and only if a concurrent
        cancel() hasn't already moved the run to a terminal state (cancel()
        writes "closed" under this same lock; an unlocked write here used to
        resurrect a cancelled run back to waiting_decision).
        """
        self.all_data = dataset.all_data
        self.timestamps = dataset.timestamps
        self.price_cache = dataset.price_cache
        self.total_steps = dataset.total_steps

        with self._step_lock:
            if self.status in TERMINAL_STATUSES:
                return
            self.status = "waiting_decision"
            self._open_current_step()

    def _market_data_at(self, timestamp) -> Dict[str, pd.Series]:
        market_data = {}
        for symbol in DJIA_30:
            if symbol not in self.all_data:
                continue
            df = self.all_data[symbol]
            if timestamp in df.index:
                market_data[symbol] = df.loc[timestamp]
        return market_data

    def _open_current_step(self) -> None:
        self.step_opened_at = _utcnow()
        self.last_decision_source = None

    def _deadline_at(self) -> datetime:
        opened = self.step_opened_at or _utcnow()
        return opened + timedelta(seconds=DECISION_TIMEOUT_SECONDS)

    # ------------------------------------------------------------------
    # Snapshot for external agents
    # ------------------------------------------------------------------

    def build_market_snapshot(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = portfolio_state.get("timestamp", datetime.now())
        if hasattr(timestamp, "isoformat"):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp)

        holdings = {}
        for position in portfolio_state["positions"]:
            holdings[position["symbol"]] = {
                "shares": position["shares"],
                "entry_price": round(position["entry_price"], 2),
                "current_price": round(position["current_price"], 2),
                "position_value": round(position["position_value"], 2),
                "pnl_pct": round(position["pnl_pct"], 2),
            }

        recent_trades = []
        cutoff = timestamp - timedelta(hours=24)
        for trade in self.manager.trades:
            if trade["timestamp"] > cutoff:
                recent_trades.append({
                    "symbol": trade["symbol"],
                    "side": trade["side"],
                    "shares": trade["shares"],
                    "price": round(float(trade["price"]), 2),
                    "timestamp": trade["timestamp"].isoformat()
                    if hasattr(trade["timestamp"], "isoformat")
                    else str(trade["timestamp"]),
                })

        snapshot: Dict[str, Any] = {
            "timestamp": timestamp_str,
            "portfolio": {
                "cash": round(portfolio_state["cash"], 2),
                "positions_value": round(portfolio_state["positions_value"], 2),
                "total_equity": round(portfolio_state["total_equity"], 2),
                "num_positions": len(portfolio_state["positions"]),
            },
            "current_holdings": holdings,
            "recent_trades": recent_trades,
            "top_signals": {},
        }

        signals = portfolio_state["market_signals"]
        if self.symbols:
            # Protocol runs declare a tradeable allow-list: the agent must see
            # features for every symbol it may trade (it can't trade what it
            # can't see). No LLM prompt is built from this snapshot on the
            # external path, so there is no prompt-size reason to sample.
            symbols = [s for s in self.symbols if s in signals]
        elif self.mode == "buy_and_hold":
            symbols = [s for s in DJIA_30 if s in signals]
        else:
            rsi_sorted = sorted(
                [(sym, sig.get("rsi", 50)) for sym, sig in signals.items()],
                key=lambda x: abs(x[1] - 50),
                reverse=True,
            )
            symbols = [sym for sym, _ in rsi_sorted[:10]]

        for symbol in symbols:
            signal = signals[symbol]
            snapshot["top_signals"][symbol] = {
                "price": float(signal.get("price") if pd.notna(signal.get("price")) else 0),
                "rsi": float(signal.get("rsi") if pd.notna(signal.get("rsi")) else 50),
                "macd": float(signal.get("macd") if pd.notna(signal.get("macd")) else 0),
                "macd_signal": float(
                    signal.get("macd_signal") if pd.notna(signal.get("macd_signal")) else 0
                ),
                "sma20": float(signal.get("sma20") if pd.notna(signal.get("sma20")) else 0),
                "sma50": float(signal.get("sma50") if pd.notna(signal.get("sma50")) else 0),
                "bb_upper": float(
                    signal.get("bb_upper") if pd.notna(signal.get("bb_upper")) else 0
                ),
                "bb_lower": float(
                    signal.get("bb_lower") if pd.notna(signal.get("bb_lower")) else 0
                ),
            }

        return snapshot

    # ------------------------------------------------------------------
    # Step API
    # ------------------------------------------------------------------

    def _build_step_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """The payload an agent consumes to decide — used as the input estimate."""
        return {
            "market_snapshot": self.build_market_snapshot(state),
            "valid_symbols": DJIA_30,
            "decision_format": get_decision_format(),
        }

    def _account_step_tokens(
        self,
        state: Dict[str, Any],
        decision_payload: Dict[str, Any],
    ) -> None:
        """Accumulate estimated input/output tokens for one agent decision."""
        try:
            self.est_input_tokens += token_cost.estimate_tokens(
                self._build_step_context(state)
            )
            self.est_output_tokens += token_cost.estimate_tokens(decision_payload)
            self.llm_calls += 1
        except Exception as exc:  # never let estimation break a backtest
            print(f"⚠️ Token estimate skipped for step {self.step_index}: {exc}")

    def _maybe_apply_timeout(self) -> bool:
        """Auto-hold if the decision window expired. Returns True if advanced."""
        if self.status != "waiting_decision":
            return False
        if _utcnow() <= self._deadline_at():
            return False
        self._advance_step(executable=[], decision_source="timeout_hold")
        return True

    def _emit_deadline_hold_notice(self, first_index: int, count: int) -> None:
        """Print once per poll when this call just auto-held one or more steps.

        Every caller drives ``_maybe_apply_timeout`` (looped or single-call)
        under ``self._step_lock`` and must capture ``first_index``
        (``self.step_index``) *before* doing so — ``_advance_step`` increments
        ``step_index`` inside the loop, so a value read afterward would name a
        step that was never held. Never take ``self._step_lock`` here: every
        caller already holds it, and ``threading.Lock`` is non-reentrant, so
        re-acquiring it deadlocks. A published equity curve containing these
        steps is not the agent's curve; this is the one place that says so.
        """
        last_index = first_index + count - 1
        # agent_name is caller-supplied on the legacy surface (StartBacktestRequest,
        # no character restriction beyond length) and that surface authenticates
        # nothing -- an embedded \n/\r would forge additional log lines. Strip them
        # at the interpolation site rather than trusting the value.
        safe_agent_name = self.agent_name.replace("\r", " ").replace("\n", " ")
        print(
            f"⚠️ decision deadline: auto-held {count} step(s) for {self.backtest_id}\n"
            f"   (agent={safe_agent_name}, steps={first_index}..{last_index}, "
            f"total_holds={self.timeout_holds})\n"
            f"   — these steps are NOT the agent's decisions"
        )

    def get_current_step(self) -> Dict[str, Any]:
        with self._step_lock:
            start_index = self.step_index
            holds_before = self.timeout_holds
            while self._maybe_apply_timeout():
                if self.status == "completed":
                    break
            if self.timeout_holds > holds_before:
                self._emit_deadline_hold_notice(
                    start_index, self.timeout_holds - holds_before
                )

            if self.status == "loading":
                return {"status": "loading", "message": "Loading market data..."}

            if self.status == "failed":
                return {"status": "failed", "error": self.error}

            if self.status == "completed":
                baseline_ids = self.baseline_run_ids  # one snapshot for both fields
                return {
                    "status": "completed",
                    "backtest_id": self.backtest_id,
                    "run_id": self.run_id,
                    "baseline_run_ids": baseline_ids,
                    "total_steps": self.total_steps,
                    "timeout_holds": self.timeout_holds,
                    "metrics": self._final_metrics(),
                    "compare_url": self._compare_url(baseline_ids),
                    "session_id": self.session_id,
                }

            if self.status == "closed":
                return {
                    "status": "closed",
                    "backtest_id": self.backtest_id,
                    "run_id": self.run_id,
                    "step_index": self.step_index,
                    "total_steps": self.total_steps,
                    "session_id": self.session_id,
                }

            timestamp = self.timestamps[self.step_index]
            market_data = self._market_data_at(timestamp)
            state = self.manager.get_portfolio_state(
                market_data, self.price_cache, timestamp
            )
            state["timestamp"] = timestamp

            return {
                "status": "waiting_decision",
                "backtest_id": self.backtest_id,
                "step_index": self.step_index,
                "total_steps": self.total_steps,
                "timestamp": timestamp.isoformat()
                if hasattr(timestamp, "isoformat")
                else str(timestamp),
                "decision_timeout_seconds": DECISION_TIMEOUT_SECONDS,
                "decision_deadline_at": _iso(self._deadline_at()),
                "market_snapshot": self.build_market_snapshot(state),
                "valid_symbols": DJIA_30,
                "decision_format": get_decision_format(),
            }

    def submit_decisions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._step_lock:
            if self.status == "completed":
                return {
                    "accepted": False,
                    "error": "backtest_already_completed",
                    "run_id": self.run_id,
                }

            if self.status != "waiting_decision":
                return {
                    "accepted": False,
                    "error": f"invalid_status:{self.status}",
                }

            if _utcnow() > self._deadline_at():
                self._advance_step(executable=[], decision_source="timeout_hold")
                return {
                    "accepted": False,
                    "error": "step_already_closed",
                    "outcome": "timeout_hold",
                    "next_step": self.step_index,
                    "status": self.status,
                }

            decisions, err = parse_actions_payload(payload)
            if err:
                self._advance_step(executable=[], decision_source="validation_hold")
                return {
                    "accepted": False,
                    "error": err,
                    "outcome": "validation_hold",
                    "next_step": self.step_index,
                    "status": self.status,
                }

            timestamp = self.timestamps[self.step_index]
            market_data = self._market_data_at(timestamp)
            state = self.manager.get_portfolio_state(
                market_data, self.price_cache, timestamp
            )
            state["timestamp"] = timestamp
            self._account_step_tokens(state, payload)
            current_prices = {
                sym: float(row.get("price", 0))
                for sym, row in state["market_signals"].items()
            }

            executable = actions_to_executable(
                decisions,
                cash=self.manager.cash,
                positions=self.manager.positions,
                current_prices=current_prices,
            )

            self._advance_step(
                executable=executable,
                decision_source="external_agent",
                raw_actions=[d.model_dump() for d in decisions],
            )

            return {
                "accepted": True,
                "executed_count": len(self.last_executed),
                "executed": self.last_executed,
                "decision_source": self.last_decision_source,
                "next_step": self.step_index,
                "status": self.status,
                "run_id": self.run_id if self.status == "completed" else None,
                "compare_url": self._compare_url() if self.status == "completed" else None,
                "metrics": self._final_metrics() if self.status == "completed" else None,
            }

    def _advance_step(
        self,
        *,
        executable: List[Dict[str, Any]],
        decision_source: str,
        raw_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if decision_source == "timeout_hold":
            # Integrity counter: steps the server auto-held past the deadline.
            # A latency-corrupted run stays green but is now VISIBLE at every
            # surface where its results appear (status, result metrics,
            # agent_runs.metadata).
            self.timeout_holds += 1
        timestamp = self.timestamps[self.step_index]
        market_data = self._market_data_at(timestamp)

        self.last_executed = []
        for action in executable:
            self.last_executed.append({
                "symbol": action.get("symbol"),
                "action": action.get("action"),
                "shares": action.get("shares"),
                "reason": action.get("reason"),
            })

        self.manager.execute_actions(executable, market_data, timestamp)
        self.manager.update_equity(market_data, self.price_cache, timestamp)

        self.decision_log.append({
            "step_index": self.step_index,
            "timestamp": timestamp.isoformat()
            if hasattr(timestamp, "isoformat")
            else str(timestamp),
            "decision_source": decision_source,
            "actions_submitted": raw_actions or [],
            "actions_executed": len(executable),
            "context_ref": self.context_ref_by_step.get(self.step_index),
        })
        self.last_decision_source = decision_source

        self.step_index += 1
        if self.step_index >= self.total_steps:
            self._finalize()
        else:
            self.status = "waiting_decision"
            self._open_current_step()

    def _finalize(self) -> None:
        equity_curve = self.manager.get_equity_curve()
        for entry in equity_curve:
            if hasattr(entry["timestamp"], "isoformat"):
                entry["timestamp"] = entry["timestamp"].isoformat()

        # v2 backends pre-assign a canonical run_id at creation; only the
        # legacy flow (no pre-set id) mints the collision-safe ext_ id here.
        if not self.run_id:
            self.run_id = _new_ext_run_id()
        initial_eq = equity_curve[0]["equity"] if equity_curve else self.initial_capital
        final_eq = equity_curve[-1]["equity"] if equity_curve else self.initial_capital
        total_return = (final_eq - self.initial_capital) / self.initial_capital

        est_cost = token_cost.estimate_cost_usd(
            self.model_name, self.est_input_tokens, self.est_output_tokens
        )

        db.insert_run(
            run_id=self.run_id,
            session_id=self.session_id,
            agent_name=self.agent_name,
            mode="backtest",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_equity=initial_eq,
            final_equity=final_eq,
            total_return=total_return,
            sharpe_ratio=calculate_sharpe(equity_curve),
            max_drawdown=calculate_max_drawdown(equity_curve),
            num_trades=len(self.manager.trades),
            llm_model=self.model_name,
            llm_calls=self.llm_calls,
            input_tokens=self.est_input_tokens,
            output_tokens=self.est_output_tokens,
            est_cost_usd=est_cost,
            metadata={
                "decision_timeout_seconds": DECISION_TIMEOUT_SECONDS,
                "timeout_holds": self.timeout_holds,
            },
        )
        db.insert_equity_points(self.run_id, equity_curve)
        db.insert_trades(self.run_id, self.manager.trades)
        db.insert_decisions(self.run_id, self.decision_log)

        analytics_user_id = getattr(self, "analytics_user_id", None)
        if analytics_user_id is not None:
            analytics_instrumentation.emit_run_event(
                event_name="backtest_completed",
                user_id=int(analytics_user_id),
                run_id=self.run_id,
                correlation_id=self.backtest_id,
            )

        # The run is fully persisted now — publish "completed" here, in-request.
        # is_active()/cancel() read self.status without _step_lock, so the flip
        # must land before this method returns (T1 already moved the slow work —
        # the market-data build — out of finalize; T2 below moves baselines out
        # too). baseline_run_ids stays {} in the envelope this finalize returns;
        # the worker fills it in asynchronously and later polls self-heal (see
        # _publish_baselines).
        self.status = "completed"

        # Background half (T2): baselines depend only on the run config, so
        # they move to the deduping worker. The job pins all_data (the shared
        # T1 bundle) so cache eviction can't force a rebuild, and publishes
        # back via _publish_baselines. Queue-full/failure degrade exactly like
        # the old best-effort inline path: run saved, baselines absent.
        baseline_worker.submit(baseline_worker.BaselineJob(
            run_id=self.run_id,
            session_id=self.session_id,
            start_date=self.start_date,
            end_date=self.end_date,
            mode=self.mode,
            all_data=self.all_data,
            publish=self._publish_baselines,
        ))

        try:
            agent_store.register_or_get_agent(
                session_id=self.session_id,
                name=self.agent_name,
                model_name=self.model_name,
            )
        except Exception as exc:
            print(f"⚠️ Agent auto-register failed (run saved): {exc}")
        # status is already "completed" (set before the baseline block above).

    def _publish_baselines(self, ids: Dict[str, str]) -> None:
        """Worker-thread callback: swap in a NEW dict in one statement.

        get_status()/get_current_step() alias baseline_run_ids out from under
        _step_lock and serialize it after releasing the lock — an in-place
        write from the worker could tear a poll response mid-iteration. A
        single reference assignment is GIL-atomic: readers see the old (empty)
        dict or the new (complete) one, never a hybrid.
        """
        self.baseline_run_ids = dict(ids)

    def _compare_url(self, baseline_run_ids: Optional[Dict[str, str]] = None) -> Optional[str]:
        # Callers that ALSO surface baseline_run_ids in the same response pass a
        # snapshot so the two fields can't straddle the worker's one-time async
        # publish (T2: _publish_baselines rebinds the dict). Default reads live.
        ids = self.baseline_run_ids if baseline_run_ids is None else baseline_run_ids
        return build_compare_url(self.run_id, ids)

    def _final_metrics(self) -> Dict[str, Any]:
        if not self.run_id:
            return {}
        return build_final_metrics(db.get_run(self.run_id))

    def get_status(self) -> Dict[str, Any]:
        with self._step_lock:
            start_index = self.step_index
            holds_before = self.timeout_holds
            self._maybe_apply_timeout()
            if self.timeout_holds > holds_before:
                self._emit_deadline_hold_notice(
                    start_index, self.timeout_holds - holds_before
                )
            base = {
                "backtest_id": self.backtest_id,
                "status": self.status,
                "step_index": self.step_index,
                "total_steps": self.total_steps,
                "agent_name": self.agent_name,
                "model_name": self.model_name,
                "run_id": self.run_id,
                "timeout_holds": self.timeout_holds,
            }
            if self.status == "waiting_decision":
                base["decision_deadline_at"] = _iso(self._deadline_at())
            if self.status == "completed":
                baseline_ids = self.baseline_run_ids  # one snapshot for both fields
                base["metrics"] = self._final_metrics()
                base["baseline_run_ids"] = baseline_ids
                base["compare_url"] = self._compare_url(baseline_ids)
            if self.error:
                base["error"] = self.error
            return base

    def drain_expired(self) -> str:
        """Apply any elapsed decision deadlines (auto-hold) without an agent,
        driving an abandoned run forward. No-op unless the current step is past
        its deadline. Returns the resulting status."""
        with self._step_lock:
            start_index = self.step_index
            holds_before = self.timeout_holds
            while self._maybe_apply_timeout():
                if self.status == "completed":
                    break
            if self.timeout_holds > holds_before:
                self._emit_deadline_hold_notice(
                    start_index, self.timeout_holds - holds_before
                )
            return self.status

    def get_decisions(self) -> List[Dict[str, Any]]:
        with self._step_lock:
            return list(self.decision_log)

    # ------------------------------------------------------------------
    # Protocol adapters (read-only; used by the Agent-Environment Protocol)
    # ------------------------------------------------------------------

    def _portfolio_state_at(
        self, timestamp, market_data: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Any]:
        if market_data is None:
            market_data = self._market_data_at(timestamp)
        state = self.manager.get_portfolio_state(market_data, self.price_cache, timestamp)
        state["timestamp"] = timestamp
        return state

    def protocol_market_data(self, timestamp) -> Dict[str, pd.Series]:
        """Raw per-symbol OHLCV rows at ``timestamp``, for callers building more than
        one protocol view (e.g. bars + portfolio) from a single snapshot without
        repeating the lookup."""
        return self._market_data_at(timestamp)

    def protocol_portfolio(
        self, timestamp=None, market_data: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Any]:
        """Normalized {cash, equity, positions[]} snapshot for the protocol."""
        if timestamp is None and self.timestamps:
            idx = self.step_index if self.step_index < self.total_steps else self.total_steps - 1
            idx = max(0, idx)
            timestamp = self.timestamps[idx]
        if timestamp is None:
            return {
                "cash": round(self.manager.cash, 2),
                "equity": round(self.manager.cash, 2),
                "positions": [],
            }
        state = self._portfolio_state_at(timestamp, market_data)
        positions = [
            {
                "symbol": p["symbol"],
                "quantity": p["shares"],
                "entry_price": round(p["entry_price"], 4),
                "current_price": round(p["current_price"], 4),
                "market_value": round(p["position_value"], 2),
                "unrealized_pnl_pct": round(p["pnl_pct"], 4),
            }
            for p in state["positions"]
        ]
        return {
            "cash": round(state["cash"], 2),
            "equity": round(state["total_equity"], 2),
            "positions": positions,
        }

    def protocol_current_prices(self, timestamp=None) -> Dict[str, float]:
        if timestamp is None and self.timestamps:
            idx = max(0, min(self.step_index, self.total_steps - 1))
            timestamp = self.timestamps[idx]
        if timestamp is None:
            return {}
        state = self._portfolio_state_at(timestamp)
        return {
            sym: float(sig.get("price") or 0)
            for sym, sig in state["market_signals"].items()
        }

    def protocol_bars(
        self, timestamp=None, market_data: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Return normalized current OHLCV bars for protocol observations.

        Called unconditionally on every ``GET /steps/next`` poll for every running
        agent, so a malformed bar must never raise out of here: that would
        permanently wedge the polling run with an unstructured 500 (the step
        timestamp doesn't advance until a decision is submitted, so the same bad
        row would be hit again on every retry). Symbols that fail the sanity
        check are dropped instead, mirroring protocol_current_prices's leniency.
        """
        if timestamp is None and self.timestamps:
            idx = max(0, min(self.step_index, self.total_steps - 1))
            timestamp = self.timestamps[idx]
        if timestamp is None:
            return {}

        allowed = set(self.symbols) if self.symbols else None
        if market_data is None:
            market_data = self._market_data_at(timestamp)
        bars: Dict[str, Dict[str, Any]] = {}
        for symbol, row in market_data.items():
            if allowed is not None and symbol not in allowed:
                continue

            values = {
                field: float(row[field])
                for field in ("open", "high", "low", "close", "volume")
            }
            prices = [values[field] for field in ("open", "high", "low", "close")]
            if (
                not all(isfinite(price) and price > 0 for price in prices)
                or not isfinite(values["volume"])
                or values["volume"] < 0
                or values["high"] < max(values["open"], values["close"])
                or values["low"] > min(values["open"], values["close"])
            ):
                print(
                    f"⚠️ Dropping invalid protocol OHLCV bar for {symbol} at "
                    f"{timestamp!r}: {values!r}"
                )
                continue

            bars[symbol] = {
                "timestamp": timestamp.isoformat()
                if hasattr(timestamp, "isoformat")
                else str(timestamp),
                **values,
            }
        return bars

    def executed_step_timestamp(self):
        """Timestamp of the most recently advanced step (post-submit)."""
        if self.step_index > 0 and self.timestamps:
            return self.timestamps[self.step_index - 1]
        return None

    def trade_count(self) -> int:
        return len(self.manager.trades)

    def fills_since(self, baseline_count: int) -> List[Dict[str, Any]]:
        new_trades = self.manager.trades[baseline_count:]
        fills: List[Dict[str, Any]] = []
        for trade in new_trades:
            qty = int(trade.get("shares") or 0)
            fills.append({
                "symbol": trade.get("symbol"),
                "side": str(trade.get("side", "")).lower(),
                "requested_quantity": qty,
                "filled_quantity": qty,
                "fill_price": round(float(trade.get("price") or 0), 4),
            })
        return fills


def get_decision_format() -> Dict[str, Any]:
    """Document the expected external agent decision payload."""
    return {
        "actions": [
            {
                "action": "buy|sell|hold",
                "symbol": "<DJIA symbol>",
                "confidence": 0.0,
                "reasoning": "<short reason>",
                "position_size": 0,
                "stop_loss_price": None,
                "take_profit_price": None,
            }
        ]
    }


def verify_session(session: ExternalBacktestSession, session_id: str) -> bool:
    return session.session_id == session_id


def count_active_sessions(session_id: Optional[str] = None) -> int:
    """Non-terminal in-memory sessions, all of them or just one caller's.

    Counts what is actually resident, which is the quantity worth bounding:
    a live session pins its loaded bar window whether it is still loading, or
    parked mid-run waiting on a decision that will never come.
    """
    with _lock:
        return _count_active_locked(session_id)


def _count_active_locked(session_id: Optional[str]) -> int:
    return sum(
        1
        for s in _sessions.values()
        if s.status not in TERMINAL_STATUSES
        and (session_id is None or s.session_id == session_id)
    )


def _raise_if_at_capacity(session_id: str) -> None:
    """Caller must hold ``_lock``."""
    if MAX_LEGACY_ACTIVE_GLOBAL > 0:
        total = _count_active_locked(None)
        if total >= MAX_LEGACY_ACTIVE_GLOBAL:
            raise BacktestCapacityError("server", total, MAX_LEGACY_ACTIVE_GLOBAL)
    if MAX_LEGACY_ACTIVE_PER_SESSION > 0:
        mine = _count_active_locked(session_id)
        if mine >= MAX_LEGACY_ACTIVE_PER_SESSION:
            raise BacktestCapacityError(
                "session", mine, MAX_LEGACY_ACTIVE_PER_SESSION
            )


# ------------------------------------------------------------------
# Public registry
# ------------------------------------------------------------------


def start_backtest(
    *,
    session_id: str,
    agent_name: str,
    model_name: str,
    start_date: str,
    end_date: str,
    mode: str = "safe_trading",
    symbols: Optional[List[str]] = None,
    initial_capital: Optional[float] = None,
    enforce_session_cap: bool = False,
    emit_analytics: bool = False,
) -> Dict[str, Any]:
    """Open an external-agent backtest session.

    ``enforce_session_cap`` is opt-in and set by the legacy
    ``/api/v1/backtest/start`` route only. The protocol surfaces (v1 ``/runs``
    and v2) reach this through ``run_service.create_run``, which has already
    applied the per-agent, per-owner and global caps under ``_create_lock``;
    charging them a second, differently-keyed budget here would refuse creates
    the surface above just authorised. Raises ``BacktestCapacityError`` when the
    caller is over budget.
    """
    backtest_id = f"bt_{uuid.uuid4().hex[:12]}"
    session = ExternalBacktestSession(
        backtest_id=backtest_id,
        session_id=session_id,
        agent_name=agent_name,
        model_name=model_name,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        symbols=symbols,
        initial_capital=initial_capital,
    )
    session.status = "loading"
    session.analytics_user_id = None
    if emit_analytics:
        try:
            agent = agent_store.get_agent_by_session(session_id)
            owner_user_id = agent.get("owner_user_id") if agent else None
            if owner_user_id is not None:
                session.analytics_user_id = int(owner_user_id)
        except Exception:
            pass

    with _lock:
        # Counted and inserted under one acquisition: as a check in the router
        # followed by an insert here, N concurrent starts would all read the
        # same under-limit count and all proceed (the check-then-act race the
        # protocol path closes with _create_lock).
        if enforce_session_cap:
            _raise_if_at_capacity(session_id)
        _sessions[backtest_id] = session

    if session.analytics_user_id is not None:
        analytics_instrumentation.emit_run_event(
            event_name="backtest_requested",
            user_id=session.analytics_user_id,
            run_id=backtest_id,
        )
        analytics_instrumentation.emit_run_event(
            event_name="backtest_started",
            user_id=session.analytics_user_id,
            run_id=backtest_id,
        )

    # Fast path: creation runs under the caller's _create_lock, where blocking
    # is forbidden — peek() is non-blocking. A resident dataset means no loader
    # thread at all: attach it and open step 0 synchronously. Miss or
    # build-in-flight falls through to the loader thread exactly as before
    # (get_dataset inside the thread blocks/dedupes there).
    dataset = market_data_store.peek(DJIA_30, start_date, end_date)
    if dataset is not None:
        session.adopt_dataset(dataset)
    else:
        def _load_in_background() -> None:
            # load_market_data() constructs AlpacaDataLoader; missing credentials
            # now raise MarketDataUnavailableError (a plain Exception, B0 deep
            # fix). The SystemExit catch stays as defense-in-depth: on this daemon
            # thread the default threading.excepthook silently swallows
            # SystemExit, so a regression back to sys.exit() would strand the run
            # in "loading" forever. (Mirrors the _finalize() catch above.)
            try:
                session.load_market_data()
            except (Exception, SystemExit) as exc:
                session.status = "failed"
                session.error = str(exc)
                if session.analytics_user_id is not None:
                    analytics_instrumentation.emit_run_event(
                        event_name="backtest_failed",
                        user_id=session.analytics_user_id,
                        run_id=backtest_id,
                        error_category="provider_unavailable",
                    )

        threading.Thread(target=_load_in_background, daemon=True).start()

    return {
        "backtest_id": backtest_id,
        "status": "loading",
        "total_steps": 0,
        "current_step": 0,
        "agent_name": agent_name,
        "model_name": model_name,
        "session_id": session_id,
        "decision_timeout_seconds": DECISION_TIMEOUT_SECONDS,
        "decision_format": get_decision_format(),
        "message": "Loading market data from Alpaca (poll /steps/current)",
        "next": {
            "get_context": f"/api/v1/backtest/{backtest_id}/steps/current",
            "submit_decisions": f"/api/v1/backtest/{backtest_id}/steps/current/decisions",
            "status": f"/api/v1/backtest/{backtest_id}/status",
        },
    }


def get_session(backtest_id: str) -> Optional[ExternalBacktestSession]:
    with _lock:
        return _sessions.get(backtest_id)


def evict_session(backtest_id: str) -> bool:
    """Drop a finished session from memory (frees its market-data buffers).

    Protocol-surface reads (``/api/v1/runs/{run_id}/...``) are DB-backed via
    ``run_store``, so eviction there is transparent. The legacy
    ``/api/v1/backtest/*`` surface is not: its ``{backtest_id}``-keyed reads
    (``status``, ``decisions``, ``steps/current``) go through
    ``_require_backtest`` -> ``get_session``, which has no DB fallback and
    404s once the session is gone. Only that surface's ``runs/{run_id}/*``
    result/trades/decisions endpoints (DB-backed via
    ``db.get_run_with_session``) stay readable after eviction. Returns True
    if one was removed.
    """
    with _lock:
        return _sessions.pop(backtest_id, None) is not None


def sweep_terminal_sessions() -> int:
    """TTL eviction for terminal sessions left in ``_sessions``.

    ``_sessions`` is written by ``start_backtest`` for both the legacy
    ``/api/v1/backtest/*`` surface and the protocol surface (via
    ``run_service.create_run``). The protocol side is separately reaped by
    ``reap_runs`` walking ``_runs`` and calling ``evict_session`` once a run
    goes terminal, but the legacy surface has no such registry, and neither
    of the ``MAX_LEGACY_ACTIVE_*`` caps sees a terminal session either (they
    filter on ``TERMINAL_STATUSES``). Left alone, a terminal legacy session
    sits in memory forever, still pinning a loaded bar window. This sweep is
    the backstop for both: it walks the whole dict rather than filtering to
    legacy-only, which is correct because reap_runs already evicts terminal
    protocol sessions before invoking registered sweeps like this one -- it
    only ever sees a protocol session here if that eviction failed, where the
    TTL is a second line of defense, not the primary path.

    Stamps ``terminal_seen_at`` the first time a session is seen terminal
    (never evicting on that same pass) rather than expecting a terminal
    transition to stamp it -- a finalize/cancel path that forgets to stamp
    would otherwise leak silently instead of just running one sweep late.
    Once a legacy session is dropped here, its ``{backtest_id}``-keyed reads
    (status/decisions/steps-current) start 404ing -- the run's result stays
    readable via the DB-backed ``GET /api/v1/backtest/runs/{run_id}/result``
    (and ``/trades``, ``/decisions``); see ``evict_session``'s docstring.

    Called by the registered reaper sweep in app.py; not a new thread.
    Pops directly rather than calling ``evict_session`` because that also
    acquires ``_lock``, which is not reentrant -- calling it from in here
    would deadlock the reaper thread. Returns the number of sessions
    evicted this pass; idempotent.
    """
    now = _utcnow()
    dropped = 0
    with _lock:
        for backtest_id, s in list(_sessions.items()):
            # Defensive: _sessions is shared with the protocol surface, and a
            # foreign/fake object injected there (e.g. in tests) may not
            # carry .status at all. Treat that as "not terminal" and skip --
            # never let a missing attribute here raise through the reaper.
            status = getattr(s, "status", None)
            if status not in TERMINAL_STATUSES:
                continue
            if s.terminal_seen_at is None:
                s.terminal_seen_at = now
                continue
            age = (now - s.terminal_seen_at).total_seconds()
            if age >= LEGACY_SESSION_RETENTION_SECONDS:
                _sessions.pop(backtest_id, None)
                dropped += 1
    return dropped


def get_current_step(backtest_id: str) -> Optional[Dict[str, Any]]:
    session = get_session(backtest_id)
    if not session:
        return None
    return session.get_current_step()


def submit_decisions(backtest_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    session = get_session(backtest_id)
    if not session:
        return None
    return session.submit_decisions(payload)


def get_status(backtest_id: str) -> Optional[Dict[str, Any]]:
    session = get_session(backtest_id)
    if not session:
        return None
    return session.get_status()


def get_backtest_decisions(backtest_id: str) -> Optional[List[Dict[str, Any]]]:
    session = get_session(backtest_id)
    if not session:
        return None
    return session.get_decisions()


def get_run_trades(run_id: str, session_id: str) -> Optional[List[Dict]]:
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        return None
    return db.get_trades(run_id)


def get_run_decisions(run_id: str, session_id: str) -> Optional[List[Dict]]:
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        return None
    return db.get_decisions(run_id)


def get_run_result(run_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        return None
    equity = db.get_equity_curve(run_id)
    trades = db.get_trades(run_id)
    decisions = db.get_decisions(run_id)
    return {
        "run": run,
        "equity_curve": equity,
        "trades": trades,
        "decisions": decisions,
        "metrics": {
            "total_return": run.get("total_return"),
            "sharpe_ratio": run.get("sharpe_ratio"),
            "max_drawdown": run.get("max_drawdown"),
            "num_trades": run.get("num_trades"),
            "final_equity": run.get("final_equity"),
            "llm_calls": run.get("llm_calls"),
            "input_tokens": run.get("input_tokens"),
            "output_tokens": run.get("output_tokens"),
            "est_cost_usd": run.get("est_cost_usd"),
            "timeout_holds": (run.get("metadata") or {}).get("timeout_holds")
            if isinstance(run.get("metadata"), dict) else None,
        },
    }
