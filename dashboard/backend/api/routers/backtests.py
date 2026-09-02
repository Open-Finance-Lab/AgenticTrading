"""Backtest, run, and comparison routes (Phase 3D4A).

Moved verbatim from ``dashboard/backend/app.py``. All external paths
(``/backtest/*``, ``/api/backtest/*``, ``/runs*``, ``/compare``), methods,
endpoint names, response models, market-hours filtering, and the background
backtest workflow are unchanged. This router is registered directly on the app
(routes carry their full absolute paths; no extra prefix is applied), so the
``/api/backtest/...`` paths remain exactly as before.

The decorator order is preserved so that ``/api/backtest/compare/latest`` is
registered before ``/api/backtest/{run_id}`` and ``/runs/latest/metrics`` before
``/runs/{run_id}``.
"""

import json
import os
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
# Module-local alias, not `import threading`, so a test can monkeypatch the
# thread factory HERE. Patching `backtests_router.threading.Thread` reaches
# through to the shared stdlib module object and swaps Thread process-wide,
# which leaks into every later test in the session.
from threading import Lock as _PlotCacheLock
from threading import Lock as _BacktestSlotsLock
from threading import Thread as _BackgroundThread
from typing import Any, Dict, List, Literal, Optional, Tuple

import pytz
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

# matplotlib is imported and configured (headless Agg backend) once at module
# import, not per request: the plot endpoint previously re-imported it and
# re-called matplotlib.use("Agg") on every call. Agg must be selected before any
# pyplot import elsewhere in the process, so it belongs at module scope.
import matplotlib
matplotlib.use("Agg")

from dashboard.backend.database import db, DB_PATH
from dashboard.backend.paths import DASHBOARD_DIR, REPO_ROOT, SCRIPTS_DIR
from dashboard.backend.middleware import get_session_id_from_request
from dashboard.backend.infrastructure.market_data.provider import (
    ALPACA,
    IFIND_ASHARE,
    MarketDataCredentialsError,
    MarketDataDependencyError,
    MarketDataSourceDisabled,
    UnsupportedMarketDataSource,
    ensure_market_data_source_available,
    validate_market_data_source,
)
from dashboard.backend.infrastructure.market_data.profiles import (
    LLM_DECISION_SOURCE,
    MarketProfile,
    get_market_profile,
    resolve_decision_source,
)
from dashboard.backend.infrastructure.llm.execution.errors import LLMExecutionError
from dashboard.backend.infrastructure.llm.execution.service import LLMExecutionService
from dashboard.backend.infrastructure.llm.execution.handoff import (
    create_execution_handoff,
)
from dashboard.backend.infrastructure.llm.execution.models import (
    BillingMode,
    LLMRunEvidence,
)
from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
)
from dashboard.backend.domain.model_providers.service import (
    CredentialResolutionError,
    get_model_provider_service,
)
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation
from dashboard.backend.domain.credits.service import credits_service
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter, client_key
from dashboard.backend.domain.agents.service import agent_service
from dashboard.backend.domain.agents.credential_store import (
    FINANCIAL_DATASETS_CREDENTIAL,
    agent_credential_store,
)
from dashboard.backend.api.dependencies import _owner_context, _require_agent_access
from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_RUNTIME_TYPE,
    DEFAULT_RUNTIME_TYPE,
    PIPELINE_RUNTIME_TYPE,
    normalize_runtime_config,
    normalize_runtime_type,
)
from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
    AiHedgeFundConfigurationError,
    resolve_step_timeout_seconds,
    runtime_unavailable_reason,
)
from dashboard.backend.domain.backtesting.constants import (
    MAX_BACKTEST_INITIAL_CAPITAL,
    resolve_initial_capital,
)
from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.equity_plot import (
    align_equity,
    build_backtest_chart_data,
    curve_timestamps_and_values,
    equity_lookup,
    market_index_baselines_with_status,
    render_backtest_equity_png,
    resolve_agent_chart_label,
)

router = APIRouter()


# ============================================================================
# Helper: Filter to Market Hours Only
# ============================================================================

def filter_market_hours(
    equity_points: List[dict],
    *,
    market: str = "US",
    market_timezone: str = "US/Eastern",
) -> List[dict]:
    """
    Filter equity data to only include market hours.
    Requirements:
    - Weekday (Monday-Friday): 0=Mon, 6=Sun
    - US: 9:30 AM - 4:00 PM local time
    - CN: 9:30 AM - 11:30 AM and 1:00 PM - 3:00 PM local time
    - Removes weekends, pre-market, after-hours, and overnight data
    """
    if not equity_points:
        return []
    
    local_tz = pytz.timezone(market_timezone)
    filtered = []
    removed_count = 0
    
    for point in equity_points:
        try:
            # Parse timestamp
            ts = datetime.fromisoformat(point['timestamp'].replace('Z', '+00:00'))
            ts_local = ts.astimezone(local_tz)
            
            # Check weekday (0=Mon, 4=Fri, 5=Sat, 6=Sun)
            weekday = ts_local.weekday()
            is_weekday = weekday < 5  # Monday-Friday only
            
            # Check the configured market's local trading sessions.
            hour = ts_local.hour
            minute = ts_local.minute
            minutes = hour * 60 + minute
            if market == "CN":
                is_market_hours = (
                    9 * 60 + 30 <= minutes <= 11 * 60 + 30
                    or 13 * 60 <= minutes <= 15 * 60
                )
            else:
                is_market_hours = 9 * 60 + 30 <= minutes <= 16 * 60
            
            if is_weekday and is_market_hours:
                filtered.append(point)
            else:
                removed_count += 1
        except Exception as e:
            print(f"Warning: Could not parse timestamp {point.get('timestamp')}: {e}")
            removed_count += 1
            continue
    
    if removed_count > 0:
        print(f"✅ filter_market_hours: {len(equity_points)} → {len(filtered)} points (removed {removed_count} non-market-hours)")
    
    if len(filtered) == 0 and len(equity_points) > 0:
        print(f"⚠️ WARNING: filter_market_hours removed ALL {len(equity_points)} points! Check timezone or data format.")
    
    return filtered


def _market_profile_for_run(run: Dict[str, Any]) -> MarketProfile:
    metadata = run.get("metadata")
    data_source = (
        metadata.get("data_source") if isinstance(metadata, dict) else ALPACA
    ) or ALPACA
    universe = metadata.get("universe") if isinstance(metadata, dict) else None
    try:
        return get_market_profile(data_source, universe)
    except ValueError:
        return get_market_profile(ALPACA)


def _filter_equity_for_run(
    run: Dict[str, Any], equity_points: List[dict]
) -> List[dict]:
    profile = _market_profile_for_run(run)
    if profile.market == "US" and profile.timezone == "US/Eastern":
        return filter_market_hours(equity_points)
    return filter_market_hours(
        equity_points,
        market=profile.market,
        market_timezone=profile.timezone,
    )


def _stored_buyhold_baseline(
    run: Dict[str, Any],
) -> List[tuple[str, str, List[dict]]]:
    run_id = run.get("baseline_buyhold_run_id")
    if not run_id:
        return []
    baseline_run = db.get_run(run_id)
    baseline_curve = db.get_equity_curve(run_id)
    if not baseline_curve:
        return []
    label = (baseline_run or {}).get("agent_name") or "buy-and-hold"
    return [(label, run_id, baseline_curve)]


# ============================================================================
# Pydantic Models (Response structures)
# ============================================================================

class EquityPoint(BaseModel):
    timestamp: str
    equity: float
    cash: float
    positions_value: float
    daily_return: Optional[float] = None
    native_equity: Optional[float] = None
    native_cash: Optional[float] = None
    native_positions_value: Optional[float] = None
    fx_rate: Optional[float] = None


class RunMetadata(BaseModel):
    run_id: str
    agent_name: str
    mode: str
    start_date: str
    end_date: str
    initial_equity: float
    final_equity: Optional[float] = None
    total_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    num_trades: int = 0
    created_at: str
    baseline_djia_run_id: Optional[str] = None
    baseline_buyhold_run_id: Optional[str] = None
    llm_model: Optional[str] = None
    llm_execution: Optional[Dict[str, Any]] = None
    data_source: str = ALPACA
    market: Optional[str] = None
    universe: Optional[str] = None
    timeframe: Optional[str] = None
    timezone: Optional[str] = None
    decision_source: Optional[str] = None
    benchmark: Optional[str] = None
    symbols: Optional[List[str]] = None
    native_currency: Optional[str] = None
    reporting_currency: Optional[str] = None
    native_initial_capital: Optional[float] = None
    fx_pair: Optional[str] = None
    fx_source: Optional[str] = None
    fx_policy: Optional[str] = None
    fx_start_rate: Optional[float] = None
    fx_end_rate: Optional[float] = None
    t_plus_one_enabled: Optional[bool] = None
    lot_size: Optional[int] = None
    transaction_cost_profile: Optional[Dict[str, Any]] = None
    # The profile above is market provenance and rides every row of that
    # market; this says whether THIS run actually paid it. The index reference
    # curve places no orders, so it carries the profile with the flag false.
    transaction_costs_applied: Optional[bool] = None
    transaction_cost_totals: Optional[Dict[str, Any]] = None
    market_rule_profile: Optional[Dict[str, Any]] = None
    market_rule_rejections: Optional[Dict[str, int]] = None
    # How much of the buy & hold sleeve filled. A lot-constrained benchmark on
    # a small account can place fewer symbols than requested, and a partly
    # placed benchmark must be legible rather than read as a real flat curve.
    baseline_allocation: Optional[Dict[str, Any]] = None
    # Scalars only. The rejected-order records themselves are unbounded per-step
    # audit data and this model is the response_model for two *list* routes
    # (/api/backtest/runs and the public, unpaginated /runs), so shipping them
    # inline would multiply a multi-megabyte array by the run count on a payload
    # the dashboard fetches on every load. The records are served per run by
    # GET /runs/{run_id}/rejected-orders instead, mirroring /runs/{run_id}/trades.
    # None (not 0) for runs that predate the feature, matching t_plus_one_enabled.
    rejected_orders_count: Optional[int] = None
    rejected_orders_truncated: Optional[int] = None
    order_events_count: Optional[int] = None
    order_events_truncated: Optional[int] = None
    # How hard T+1 actually bound: symbol-days on which the agent wanted to exit
    # more than it could, and the total shares that deferred. Scalars for the
    # same reason as above — the records live on the detail endpoint.
    t1_deferred_events: Optional[int] = None
    t1_deferred_shares: Optional[float] = None


class EquityCurve(BaseModel):
    run_id: str
    agent_name: str
    data: List[EquityPoint]
    metrics: dict


class ComparisonResponse(BaseModel):
    runs: List[EquityCurve]
    summary: dict


class ChartSeries(BaseModel):
    run_id: str
    label: str
    values: List[float]
    color: str
    dashed: bool = False


class BacktestChartData(BaseModel):
    agent_run_id: str
    timestamps: List[str]
    x_labels: List[str]
    series: List[ChartSeries]
    # False = the index benchmarks are absent because Yahoo was unreachable, not
    # because this run has none. Defaults True so an older cached client that
    # never reads it behaves exactly as before.
    index_baselines_ok: bool = True


def _run_metadata_response(run: Dict[str, Any]) -> RunMetadata:
    """Expose data provenance while keeping historical runs backward compatible."""
    metadata = run.get("metadata")
    data_source = metadata.get("data_source") if isinstance(metadata, dict) else None
    payload = dict(run)
    payload["data_source"] = data_source or ALPACA
    if isinstance(metadata, dict):
        for field in (
            "market",
            "universe",
            "timeframe",
            "timezone",
            "decision_source",
            "benchmark",
            "symbols",
            "native_currency",
            "reporting_currency",
            "native_initial_capital",
            "fx_pair",
            "fx_source",
            "fx_policy",
            "fx_start_rate",
            "fx_end_rate",
            "t_plus_one_enabled",
            "lot_size",
            "transaction_cost_profile",
            "transaction_costs_applied",
            "transaction_cost_totals",
            "market_rule_profile",
            "market_rule_rejections",
            "baseline_allocation",
            "rejected_orders_count",
            "rejected_orders_truncated",
            "order_events_count",
            "order_events_truncated",
            "t1_deferred_events",
            "t1_deferred_shares",
            "llm_execution",
        ):
            if field in metadata:
                if field == "llm_execution" and isinstance(metadata[field], dict):
                    safe_evidence = {
                        name: metadata[field][name]
                        for name in LLMRunEvidence.model_fields
                        if name in metadata[field]
                    }
                    try:
                        payload[field] = LLMRunEvidence.model_validate(
                            safe_evidence
                        ).model_dump(mode="json")
                    except Exception:  # noqa: BLE001 - legacy/malformed metadata
                        continue
                else:
                    payload[field] = metadata[field]
    return RunMetadata(**payload)


# ============================================================================
# Background backtest state + worker
# ============================================================================

# Global state for background backtests.
#
# Historically a single process-wide ``backtest_status`` dict enforced
# single-flight: one running dashboard backtest at a time. Entitlements now
# allow N concurrent runs per signed-in user (anonymous / no entitlement stays
# at 1). ``_active_slots`` is the concurrency ledger; ``backtest_status`` remains
# as a compatibility mirror of the most recently started/updated slot so
# existing tests and callers that poke the dict directly keep working.

backtest_status = {
    "running": False,
    "error": None,
    "runs_count": 0,
    "started_at": None,
    "progress_file": None,
    "live_run_id": None,
}
backtest_session_id = None  # Track which session owns the mirrored status

_backtest_slots_lock = _BacktestSlotsLock()
# live_run_id -> slot dict (running or just-finished, briefly retained)
_active_slots: Dict[str, Dict[str, Any]] = {}
_recent_slots: Dict[str, Dict[str, Any]] = {}

# Server-wide default. Deliberately equal to ``DEFAULT_MAX_CONCURRENT_BACKTESTS``
# so one default-entitlement account can actually reach its own quota, and no
# higher: a dashboard backtest is a *subprocess* (unlike the protocol surfaces'
# in-process step sessions, whose global caps are 50/100), and each one pins a
# loaded bar window inside a 512MB free-tier instance. It is also the LLM spend
# bound -- before this module grew slots the runner was single-flight, so the
# ceiling was exactly 1. Operators on a larger plan raise it explicitly.
_DEFAULT_MAX_ACTIVE_DASHBOARD_BACKTESTS = 5


def _max_active_dashboard_backtests() -> int:
    """Server-wide ceiling on concurrent dashboard backtests.

    Parsed defensively for the same reason every other operator-set integer in
    this repo is: a typo in the Render field must not take the whole app down
    at import, and a negative value must not silently refuse every backtest
    (0 is a legitimate "drain the runner" setting, ``-1`` is a typo). The
    default is deliberately conservative -- each in-flight run pins a loaded
    bar window in a 512MB free-tier dyno, and an LLM run spends operator money
    per trading hour, so this is a spend bound as much as a memory one.
    """
    raw = os.getenv("MAX_ACTIVE_DASHBOARD_BACKTESTS")
    if raw is None or not str(raw).strip():
        return _DEFAULT_MAX_ACTIVE_DASHBOARD_BACKTESTS
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        print(
            "MAX_ACTIVE_DASHBOARD_BACKTESTS is not an integer "
            f"({raw!r}); using {_DEFAULT_MAX_ACTIVE_DASHBOARD_BACKTESTS}",
            flush=True,
        )
        return _DEFAULT_MAX_ACTIVE_DASHBOARD_BACKTESTS
    if value < 0:
        print(
            f"MAX_ACTIVE_DASHBOARD_BACKTESTS is negative ({value}); "
            f"using {_DEFAULT_MAX_ACTIVE_DASHBOARD_BACKTESTS}",
            flush=True,
        )
        return _DEFAULT_MAX_ACTIVE_DASHBOARD_BACKTESTS
    return value


MAX_ACTIVE_DASHBOARD_BACKTESTS = _max_active_dashboard_backtests()


def _reset_slots_for_tests() -> None:
    """Drop in-flight/recent slots and the legacy status mirror.

    Tests mock ``run_backtest_background``, so ``_finalize_slot`` never runs.
    The suite must call this between cases or the process-wide cap refuses
    later ``POST /backtest/run`` with ``success: false``.
    """
    global backtest_session_id
    with _backtest_slots_lock:
        _active_slots.clear()
        _recent_slots.clear()
    backtest_status.update(
        {
            "running": False,
            "error": None,
            "runs_count": 0,
            "started_at": None,
            "progress_file": None,
            "live_run_id": None,
        }
    )
    backtest_session_id = None


def _backtest_owner_key(user_id: Optional[int], owner_session: str) -> str:
    """Who this run's concurrency is billed to.

    ``owner_session`` is the *caller's* browser session, never the session the
    run's results are filed under. For a built-in agent those differ: every
    anonymous visitor who runs the same built-in agent inherits that agent's
    session id, so keying the cap on it put the whole internet into one bucket
    -- one visitor's backtest would refuse everyone else's.

    A browser session is a caller-chosen header, so this is an incentive fix
    rather than a bound, exactly as ``resolve_owner_cap_context`` documents for
    the protocol surface: it stops signing out from being the cheaper option.
    Rotating the header still buys concurrency; what actually bounds the server
    is ``MAX_ACTIVE_DASHBOARD_BACKTESTS``.
    """
    if user_id is not None:
        return f"user:{int(user_id)}"
    return f"session:{owner_session}"


def _max_concurrent_for_user(user_id: Optional[int]) -> int:
    """Per-owner concurrent dashboard backtests.

    Anonymous callers get 1 -- the pre-PR behaviour -- because there is no
    account to hold an entitlement and no store lookup worth a round-trip.

    Signed-in callers get their ``max_concurrent_backtests`` entitlement. Note
    this is the SAME number the protocol surface applies to its own runs, and
    the two are counted separately, so an account's true ceiling is that
    entitlement on each surface rather than across both. Deliberate: making it
    one shared budget would silently cut every existing protocol user's
    capacity the moment this shipped, and the server-wide cap is the bound that
    actually protects the instance.

    Fails open on a store error for the same reason ``resolve_owner_cap_context``
    does: concurrency must not gain a hard dependency on the users database
    being reachable, and the server-wide cap still applies. The print marks the
    boundary -- an outage here otherwise looks exactly like "no cap configured".
    """
    if user_id is None:
        return 1
    from dashboard.backend.users import user_store

    try:
        return int(user_store.get_entitlements(user_id)["max_concurrent_backtests"])
    except Exception as exc:  # noqa: BLE001 - see docstring
        print(
            f"⚠️ entitlement lookup failed for user {user_id}; "
            f"falling back to server-wide cap only: {exc}",
            flush=True,
        )
        return MAX_ACTIVE_DASHBOARD_BACKTESTS


def _slot_snapshot(slot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "running": bool(slot.get("running")),
        "error": slot.get("error"),
        "runs_count": int(slot.get("runs_count") or 0),
        "started_at": slot.get("started_at"),
        "progress_file": slot.get("progress_file"),
        "live_run_id": slot.get("live_run_id"),
        "session_id": slot.get("session_id"),
        "owner_session": slot.get("owner_session"),
        "user_id": slot.get("user_id"),
        "owner_key": slot.get("owner_key"),
    }


def _mirror_slot_to_legacy(slot: Dict[str, Any]) -> None:
    """Keep ``backtest_status`` / ``backtest_session_id`` in sync with one slot."""
    global backtest_session_id
    backtest_status["running"] = bool(slot.get("running"))
    backtest_status["error"] = slot.get("error")
    backtest_status["runs_count"] = int(slot.get("runs_count") or 0)
    backtest_status["started_at"] = slot.get("started_at")
    backtest_status["progress_file"] = slot.get("progress_file")
    backtest_status["live_run_id"] = slot.get("live_run_id")
    backtest_session_id = slot.get("session_id")


def _count_active_for_owner(owner_key: str) -> int:
    return sum(
        1
        for slot in _active_slots.values()
        if slot.get("running") and slot.get("owner_key") == owner_key
    )


def _try_acquire_backtest_slot(
    *,
    live_run_id: str,
    session_id: str,
    owner_session: Optional[str] = None,
    user_id: Optional[int],
) -> Optional[str]:
    """Register a running slot or return a human-readable refusal reason.

    ``session_id`` files the *results* (for a built-in agent that is the
    agent's own session, so its runs land on its public card).
    ``owner_session`` is the caller's browser session and is what the cap is
    billed to; it defaults to ``session_id`` for callers that are the same
    thing. Keeping them apart is what stops every anonymous visitor to one
    built-in agent from sharing a single slot.
    """
    owner_session = owner_session or session_id
    owner_key = _backtest_owner_key(user_id, owner_session)
    max_for_owner = _max_concurrent_for_user(user_id)
    with _backtest_slots_lock:
        active_global = sum(1 for s in _active_slots.values() if s.get("running"))
        if active_global >= MAX_ACTIVE_DASHBOARD_BACKTESTS:
            return (
                "Server is at capacity for concurrent backtests. "
                "Please wait for one to finish."
            )
        if _count_active_for_owner(owner_key) >= max_for_owner:
            if max_for_owner <= 0:
                # An admin set this account's quota to 0, which the entitlement
                # range documents as "suspended". Folded into the <= 1 branch it
                # told a suspended user to "wait for it to complete" — waiting
                # for a run they do not have, forever.
                return (
                    "Backtests are disabled for this account. "
                    "Contact an administrator."
                )
            if max_for_owner == 1:
                return "Backtest already running. Please wait for it to complete."
            return (
                f"You already have {max_for_owner} backtests running. "
                "Please wait for one to finish."
            )
        slot = {
            "live_run_id": live_run_id,
            "session_id": session_id,
            "owner_session": owner_session,
            "user_id": user_id,
            "owner_key": owner_key,
            "running": True,
            "error": None,
            "runs_count": 0,
            "started_at": time.time(),
            "progress_file": None,
        }
        _active_slots[live_run_id] = slot
        _mirror_slot_to_legacy(slot)
    return None


def _update_slot(live_run_id: str, **fields: Any) -> None:
    with _backtest_slots_lock:
        slot = _active_slots.get(live_run_id) or _recent_slots.get(live_run_id)
        if not slot:
            # Legacy / test path: no slot was ever registered, so the global
            # mirror is the only place this run exists.
            mirrored = backtest_status.get("live_run_id")
            if mirrored and mirrored != live_run_id and mirrored in _active_slots:
                # The mirror describes a registered run. Writing this
                # un-slotted run's progress_file/started_at over it would
                # publish a pair that never coexisted -- one run's id beside
                # another's progress -- so leave it alone.
                return
            for key, value in fields.items():
                if key in backtest_status:
                    backtest_status[key] = value
            # Stamp the id too. Without it the mirror advertised whichever run
            # last owned it while carrying this run's fields, and both the
            # status route and the concurrency count read that pair.
            backtest_status["live_run_id"] = live_run_id
            return
        slot.update(fields)
        if backtest_status.get("live_run_id") == live_run_id:
            _mirror_slot_to_legacy(slot)


def _slot_analytics_user_id(live_run_id: str) -> Optional[int]:
    with _backtest_slots_lock:
        slot = _active_slots.get(live_run_id) or _recent_slots.get(live_run_id)
        user_id = slot.get("user_id") if slot else None
    return int(user_id) if user_id is not None else None


def _finalize_slot(live_run_id: str, *, error: Optional[str], runs_count: int) -> None:
    with _backtest_slots_lock:
        slot = _active_slots.pop(live_run_id, None)
        if not slot:
            backtest_status["running"] = False
            backtest_status["started_at"] = None
            backtest_status["live_run_id"] = None
            backtest_status["progress_file"] = None
            if error is not None:
                backtest_status["error"] = error
            backtest_status["runs_count"] = runs_count
            return
        slot["running"] = False
        slot["error"] = error
        slot["runs_count"] = runs_count
        slot["started_at"] = None
        slot["progress_file"] = None
        _recent_slots[live_run_id] = slot
        # Bound retention so a long-lived process does not grow forever.
        if len(_recent_slots) > 50:
            oldest = next(iter(_recent_slots))
            _recent_slots.pop(oldest, None)
        if backtest_status.get("live_run_id") == live_run_id:
            _mirror_slot_to_legacy(slot)
            backtest_status["progress_file"] = None
            backtest_status["started_at"] = None
    user_id = slot.get("user_id")
    if user_id is not None:
        succeeded = error is None and runs_count > 0
        analytics_instrumentation.emit_run_event(
            event_name=("backtest_completed" if succeeded else "backtest_failed"),
            user_id=int(user_id),
            run_id=live_run_id,
            error_category=(None if succeeded else "internal_error"),
        )


def _release_slot(live_run_id: str) -> None:
    """Drop a slot acquired for a run that never started.

    Distinct from ``_finalize_slot``: nothing ran, so there is no outcome to
    retain and nothing the poller should be able to find afterwards. Finalising
    instead would park a ``runs_count: 0`` entry in ``_recent_slots``, and the
    status route reads that as "completed, but no runs found for this session"
    -- a failure message for a request that was simply refused.
    """
    with _backtest_slots_lock:
        _active_slots.pop(live_run_id, None)
        _recent_slots.pop(live_run_id, None)
        if backtest_status.get("live_run_id") == live_run_id:
            backtest_status["running"] = False
            backtest_status["started_at"] = None
            backtest_status["live_run_id"] = None
            backtest_status["progress_file"] = None


def _slot_visible_to(
    slot: Dict[str, Any], *, session_id: str, user_id: Optional[int]
) -> bool:
    """Is this slot a run the caller is entitled to see?

    Ownership is the same pair the cap ledger keys on. A signed-in caller sees
    their own runs from any browser session; a run started anonymously stays
    bound to the browser session that started it, including after that same
    session signs in -- its slot predates the account and would otherwise
    vanish from the poller mid-run.
    """
    if user_id is not None and slot.get("user_id") is not None:
        return int(slot["user_id"]) == int(user_id)
    if not session_id:
        return False
    # Either identity on the slot counts. For a built-in agent the two differ:
    # ``session_id`` is the agent's (where results file) and ``owner_session``
    # is the visitor's (who started it), and the visitor polls with their own.
    return session_id in (slot.get("session_id"), slot.get("owner_session"))


def _resolve_status_slot(
    *,
    session_id: str,
    user_id: Optional[int],
    live_run_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Resolve the slot a status poll is asking about.

    An explicit ``live_run_id`` is an *exact* lookup: it answers 404 unless it
    resolves to a run this caller owns. Two properties matter and neither is
    optional.

    First, the ownership check. Without it the route hands any caller who
    knows (or guesses) a run id that run's ``session_id`` -- and in this
    codebase a session id is an access grant, not just a label (see
    ``_owner_context``), so leaking one is an authorization break, not an
    information leak. Unknown id and someone else's id return the same 404 on
    purpose, so the route cannot be used to test whether a run id exists.

    Second, no fallback. The unknown-id case must NOT drop through to the
    session scan below: the caller supplied an id precisely to disambiguate
    between their own concurrent runs, so answering with a sibling run turns
    "how is run B doing?" into run A's progress -- silently, with HTTP 200.
    """
    with _backtest_slots_lock:
        if live_run_id:
            slot = _active_slots.get(live_run_id) or _recent_slots.get(live_run_id)
            if slot is None and backtest_status.get("live_run_id") == live_run_id:
                # Legacy mirror: a run that never registered a slot (tests, and
                # the pre-slot code path) still sets the global status dict.
                slot = {
                    **backtest_status,
                    "session_id": backtest_session_id,
                    "user_id": None,
                }
            if slot is None or not _slot_visible_to(
                slot, session_id=session_id, user_id=user_id
            ):
                raise HTTPException(status_code=404, detail="Backtest run not found")
            return _slot_snapshot(slot)
        # Prefer an active run owned by this backtest session. Matched on
        # session identity only, not on user_id: this branch answers "what is
        # THIS browser doing?", and widening it to the account would surface a
        # run started in another tab or on another machine as though it were
        # this page's.
        for slot in reversed(list(_active_slots.values())):
            if slot.get("running") and session_id in (
                slot.get("session_id"),
                slot.get("owner_session"),
            ):
                return _slot_snapshot(slot)
        for slot in reversed(list(_recent_slots.values())):
            if session_id in (slot.get("session_id"), slot.get("owner_session")):
                return _slot_snapshot(slot)
    return None


def count_active_dashboard_backtests() -> int:
    """How many dashboard-UI backtests are in flight on this process.

    Counts the slot ledger, not the legacy ``backtest_status`` mirror. The
    mirror tracks whichever slot changed most recently, so under the
    multi-slot runner it reports 1 while five runs are live -- which is
    exactly the "a future multi-slot runner changes this function, not its
    callers" case its previous docstring anticipated.
    """
    with _backtest_slots_lock:
        active = sum(1 for slot in _active_slots.values() if slot.get("running"))
    if active:
        return active
    # Legacy/test path: a run that never registered a slot still sets the mirror.
    return 1 if backtest_status.get("running") else 0


def _read_progress_file(progress_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load incremental equity snapshots written by the backtest subprocess.

    ``progress_updated_at`` (the file's mtime) and ``progress_age_seconds`` (how
    old that is) are not fields the writer emits. Together they answer "are these
    numbers current?", which the payload alone cannot.

    The *age* is what the UI reads, and it is computed here rather than in the
    browser deliberately: differencing a server mtime against the client clock
    makes any machine more than the staleness threshold out of step
    indistinguishable from a wedged run -- a fast clock pins a permanent "No
    progress for 47m" onto a healthy backtest, a slow one suppresses the warning
    forever, and suspended laptops drift by minutes routinely. Both ends of this
    subtraction are read in this process, so it carries no skew.

    stat() and read_text() are separate syscalls, so a file rewritten between
    them yields an mtime marginally older than the payload -- immaterial against
    a 120s staleness threshold, and not worth a lock to avoid.
    """
    if not progress_file:
        return None
    path = Path(progress_file)
    if not path.is_file():
        return None
    try:
        updated_at = path.stat().st_mtime
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        **payload,
        "progress_updated_at": updated_at,
        # Clamped at zero: a clock stepping backwards between the write and this
        # read would otherwise report a negative age, and "-3s" reads as a bug.
        "progress_age_seconds": max(0.0, time.time() - updated_at),
    }


def _read_backtest_progress(progress_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Legacy reader: fall back to the global mirror when given no file.

    Kept separate from ``_read_progress_file`` because that fallback is only
    ever right for a *single-flight* caller. Folding it into the reader made a
    slot whose ``progress_file`` is still None -- a run accepted a moment ago,
    before the subprocess has written anything -- report whichever sibling run
    last touched the mirror, so a freshly started backtest opened at 60%.
    The status route reads ``_read_progress_file`` for that reason.
    """
    return _read_progress_file(progress_file or backtest_status.get("progress_file"))


def run_backtest_background(
    start_date: str,
    end_date: str,
    session_id: str,
    strategy_prompt: Optional[str] = None,
    model: Optional[str] = None,
    pipeline: Optional[List[Dict[str, Any]]] = None,
    agent_id: Optional[str] = None,
    data_source: str = ALPACA,
    live_run_id: Optional[str] = None,
    universe: Optional[str] = None,
    timeframe: Optional[str] = None,
    initial_capital: Optional[float] = None,
    assets: Optional[List[str]] = None,
    decision_source: Optional[str] = None,
    runtime_type: str = DEFAULT_RUNTIME_TYPE,
    runtime_config: Optional[Dict[str, Any]] = None,
    financial_datasets_api_key: Optional[str] = None,
    execution_handoff_payload: Optional[str] = None,
):
    """Run backtest in background thread.

    The execution handoff is passed through stdin so no credential or signed
    worker payload appears in subprocess arguments or environment variables.
    """
    global backtest_status, backtest_session_id

    strategy_prompt_path = None
    pipeline_path = None
    runtime_config_path = None
    progress_file = None
    # Bound so finally can always finalize even if minting the id fails early.
    resolved_live_run_id = live_run_id
    execution_run_id = None
    # Snapshot the agent's pipeline as this run sees it, so the adapted-pipeline
    # write-back at the end can tell "nobody touched it" from "the user edited
    # it mid-run" (Configure stays open, and sibling runs adapt too).
    baseline_pipeline = _normalized_pipeline(pipeline) or _agent_pipeline_snapshot(agent_id)
    try:
        import subprocess
        import sys
        import tempfile

        profile = get_market_profile(data_source, universe)
        decision_source = resolve_decision_source(profile, decision_source)
        uses_llm = decision_source == LLM_DECISION_SOURCE
        universe = profile.universe
        timeframe = timeframe or profile.timeframe
        if timeframe != profile.timeframe:
            raise ValueError("Backtest market profile does not match the data source")

        if not resolved_live_run_id:
            resolved_live_run_id = (
                f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )
        if execution_handoff_payload:
            execution_run_id = resolved_live_run_id
        progress_file = str(
            Path(tempfile.gettempdir()) / f"backtest_progress_{resolved_live_run_id}.json"
        )
        _update_slot(
            resolved_live_run_id,
            running=True,
            error=None,
            started_at=time.time(),
            progress_file=progress_file,
            session_id=session_id,
        )
        analytics_user_id = _slot_analytics_user_id(resolved_live_run_id)
        if analytics_user_id is not None:
            analytics_instrumentation.emit_run_event(
                event_name="backtest_started",
                user_id=analytics_user_id,
                run_id=resolved_live_run_id,
            )
        # Legacy mirror for code paths that still read the globals directly.
        backtest_session_id = session_id

        print(f"🚀 Background: Running backtest: {start_date} to {end_date}", flush=True)
        print(f"   Session: {session_id[:8]}...", flush=True)
        
        script_path = SCRIPTS_DIR / "backtest_hourly_agent.py"
        db_path = DB_PATH
        venv_dir = REPO_ROOT / ".venv"
        
        # Determine the Python executable to use (from venv if available)
        if venv_dir.exists():
            python_exe = str(venv_dir / "bin" / "python3")
            print(f"🐍 Using venv Python: {python_exe}", flush=True)
        else:
            python_exe = sys.executable
            print(f"🐍 Using system Python: {python_exe}", flush=True)
        
        # Check database directory
        print(f"📁 Database path: {db_path}", flush=True)
        print(f"📁 Database dir exists: {db_path.parent.exists()}", flush=True)
        print(f"📁 Can write to {db_path.parent}: {os.access(db_path.parent, os.W_OK)}", flush=True)
        
        env = os.environ.copy()
        if runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE:
            # A Financial Datasets key is agent-owner material, never a platform
            # fallback. Isolate it only for the hosted runtime; pipeline
            # subprocesses retain their established environment unchanged.
            env.pop("FINANCIAL_DATASETS_API_KEY", None)
            if financial_datasets_api_key:
                env["FINANCIAL_DATASETS_API_KEY"] = financial_datasets_api_key
        if uses_llm:
            print(f"{data_source} selected; LLM decision source enabled", flush=True)
        else:
            print(f"{data_source} selected; rule-based decision source", flush=True)
        
        cmd = [
            python_exe, str(script_path),
            "--start", start_date, "--end", end_date,
            "--session-id", session_id,
            "--data-source", data_source,
            "--universe", universe,
            "--timeframe", timeframe,
            "--decision-source", decision_source,
        ]

        if runtime_type != PIPELINE_RUNTIME_TYPE:
            cmd += ["--runtime-type", runtime_type]

        if runtime_config:
            fd, runtime_config_path = tempfile.mkstemp(
                prefix="agent_runtime_", suffix=".json"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(runtime_config, f)
            cmd += ["--runtime-config-file", runtime_config_path]

        # Optional free-form strategy prompt: written to a temp file (avoids
        # shell-escaping a long prompt) and passed via --strategy-prompt-file.
        if uses_llm and strategy_prompt and strategy_prompt.strip() and not pipeline:
            fd, strategy_prompt_path = tempfile.mkstemp(prefix="strategy_prompt_", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(strategy_prompt.strip())
            cmd += ["--strategy-prompt-file", strategy_prompt_path]

        if uses_llm and pipeline:
            fd, pipeline_path = tempfile.mkstemp(prefix="agent_pipeline_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(pipeline, f)
            cmd += ["--pipeline-file", pipeline_path]

        if uses_llm and model and model.strip():
            cmd += ["--model", model.strip()]

        if execution_handoff_payload:
            cmd += ["--execution-handoff-stdin"]

        cmd += ["--run-id", resolved_live_run_id, "--progress-file", progress_file]

        # Simulation capital is independent of the agent's portfolio sleeve.
        cmd += ["--initial-capital", str(resolve_initial_capital(initial_capital))]

        if assets:
            cmd += ["--assets", ",".join(assets)]
            print(f"   Assets: {', '.join(assets)}", flush=True)

        print(f"📋 Running: {' '.join(cmd)}", flush=True)

        subprocess_timeout = _backtest_subprocess_timeout(
            runtime_type, start_date, end_date
        )
        print(f"⏱️  Subprocess timeout: {subprocess_timeout}s", flush=True)

        result = subprocess.run(
            cmd,
            cwd=str(DASHBOARD_DIR),
            capture_output=True,
            text=True,
            timeout=subprocess_timeout,
            env=env,
            input=execution_handoff_payload or "",
        )
        
        # Print script output for debugging
        print(f"\n📋 === BACKTEST SCRIPT OUTPUT ===", flush=True)
        # Redact, but do NOT truncate: print() is the only log channel that
        # survives in the deployed config, so trimming the dump would drop the
        # head of every run (universe, decision source, FX bootstrap).
        if result.stdout:
            print(
                f"STDOUT:\n{_redact_credentials(result.stdout, financial_datasets_api_key)}",
                flush=True,
            )
        if result.stderr:
            print(
                f"STDERR:\n{_redact_credentials(result.stderr, financial_datasets_api_key)}",
                flush=True,
            )
        print(f"Return code: {result.returncode}", flush=True)
        print(f"=== END BACKTEST OUTPUT ===", flush=True)

        slot_error = None
        slot_runs_count = 0
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            summary = _sanitize_backtest_error(
                error_msg,
                500,
                extra_secret=financial_datasets_api_key,
            )
            slot_error = (
                f"Backtest failed with return code {result.returncode}. {summary}"
            )
            print(f"❌ Backtest failed (returncode={result.returncode})", flush=True)
        else:
            runs = db.get_runs_by_mode("backtest")
            slot_runs_count = len(runs)
            print(f"✅ Backtest completed. Found {len(runs)} runs in database.", flush=True)
            if len(runs) > 0:
                print(f"   Latest run IDs: {[r['run_id'] for r in runs[:3]]}", flush=True)
            _maybe_writeback_adapted_pipeline(
                agent_id, resolved_live_run_id, baseline_pipeline
            )
        if resolved_live_run_id:
            _finalize_slot(
                resolved_live_run_id, error=slot_error, runs_count=slot_runs_count
            )
            resolved_live_run_id = None  # finally must not double-finalize
    except Exception as e:
        summary = _sanitize_backtest_error(
            e,
            500,
            extra_secret=financial_datasets_api_key,
        )
        print(f"❌ Backtest exception: {summary}", flush=True)
        if resolved_live_run_id:
            _finalize_slot(resolved_live_run_id, error=summary, runs_count=0)
            resolved_live_run_id = None
    finally:
        if execution_handoff_payload and execution_run_id:
            try:
                # The child normally finalizes itself. Repeating this from the
                # parent also clears reservations when the subprocess is killed
                # by timeout or exits before its own finally block runs.
                LLMExecutionService(
                    providers=get_model_provider_service(),
                    credits=credits_service,
                ).finalize_run(execution_run_id)
            except LLMExecutionError as exc:
                print(
                    f"❌ LLM execution cleanup failed: {exc.safe_message}",
                    flush=True,
                )
        if resolved_live_run_id:
            _finalize_slot(resolved_live_run_id, error=None, runs_count=0)
        elif not live_run_id:
            # No slot was ever registered (a caller that minted no run id), so
            # _finalize_slot never ran and the legacy mirror is the only record
            # of this run. Clear it, or the single-flight fallback stays wedged
            # at running=True for the life of the process.
            backtest_status["running"] = False
            backtest_status["started_at"] = None
            backtest_status["live_run_id"] = None
            backtest_status["progress_file"] = None
        if progress_file:
            try:
                Path(progress_file).unlink(missing_ok=True)
            except OSError:
                pass
        if strategy_prompt_path:
            try:
                os.remove(strategy_prompt_path)
            except OSError:
                pass
        if pipeline_path:
            try:
                os.remove(pipeline_path)
            except OSError:
                pass
        if runtime_config_path:
            try:
                os.remove(runtime_config_path)
            except OSError:
                # Best-effort cleanup of a temp file the run no longer needs;
                # the OS reclaims it regardless, and failing here would mask
                # the backtest's own outcome.
                pass
        print("✋ Backtest background thread finished", flush=True)


# The dashboard pipeline parent has a bounded 60-minute wall-clock budget. A
# hosted run instead spends one *upstream subprocess* per trading day, each
# allowed AI_HEDGE_FUND_TIMEOUT_SECONDS, and is sized dynamically below.
# Hosted runtimes below retain their own per-decision sizing; this fixed value
# only applies to the normal pipeline subprocess.
PIPELINE_SUBPROCESS_TIMEOUT_SECONDS = 3600
# Data load, baseline generation and persistence sit outside the decision loop.
SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS = 600
# Ceiling, so a long date range cannot pin a worker thread indefinitely.
MAX_SUBPROCESS_TIMEOUT_SECONDS = 14400


def _estimated_decision_days(start_date: str, end_date: str) -> int:
    """Upper-bound the trading days in an inclusive date range.

    Weekday count, not a market calendar: holidays only make the real number
    smaller, and over-provisioning the parent timeout is the safe direction.
    """
    try:
        start = datetime.strptime(str(start_date)[:10], "%Y-%m-%d").date()
        end = datetime.strptime(str(end_date)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return 0
    if end < start:
        return 0
    total_days = (end - start).days + 1
    whole_weeks, remainder = divmod(total_days, 7)
    weekdays = whole_weeks * 5
    start_weekday = start.weekday()
    for offset in range(remainder):
        if (start_weekday + offset) % 7 < 5:
            weekdays += 1
    return weekdays


def _backtest_subprocess_timeout(
    runtime_type: str, start_date: str, end_date: str
) -> int:
    """Return the parent subprocess timeout for this run's runtime."""
    if runtime_type == PIPELINE_RUNTIME_TYPE:
        return PIPELINE_SUBPROCESS_TIMEOUT_SECONDS
    step_seconds = resolve_step_timeout_seconds()
    required = (
        step_seconds * _estimated_decision_days(start_date, end_date)
        + SUBPROCESS_TIMEOUT_OVERHEAD_SECONDS
    )
    budget = max(PIPELINE_SUBPROCESS_TIMEOUT_SECONDS, required)
    if budget > MAX_SUBPROCESS_TIMEOUT_SECONDS:
        # Say so rather than truncating quietly: past this point the parent is
        # the binding constraint again, and the run can be killed mid-flight.
        print(
            f"⚠️  Hosted backtest needs ~{budget}s but is capped at "
            f"{MAX_SUBPROCESS_TIMEOUT_SECONDS}s; shorten the date range or "
            f"lower AI_HEDGE_FUND_TIMEOUT_SECONDS (currently {step_seconds}s)",
            flush=True,
        )
        return MAX_SUBPROCESS_TIMEOUT_SECONDS
    return budget


def _redact_credentials(text: object, extra_secret: Optional[str] = None) -> str:
    """Strip credentials from text without dropping any of it.

    Kept separate from truncation on purpose: the subprocess log dump needs
    redaction over its FULL length, while only the operator-facing error
    summary needs a length bound.
    """
    message = str(text)
    for environment_variable in ("IFIND_REFRESH_TOKEN", "IFIND_ACCESS_TOKEN"):
        token = os.getenv(environment_variable, "").strip()
        if token:
            message = message.replace(token, "[REDACTED]")
    if extra_secret:
        message = message.replace(extra_secret, "[REDACTED]")
    message = re.sub(
        r"(?i)(access[_-]?token\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)(refresh[_-]?token\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)(authorization\s*[=:]\s*)(?:bearer\s+)?[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    return message


def _sanitize_backtest_error(
    error: object,
    max_chars: int = 500,
    *,
    extra_secret: Optional[str] = None,
) -> str:
    """Return a bounded background error summary without credentials."""
    return _redact_credentials(error, extra_secret)[-max_chars:]


def _normalized_pipeline(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Coerce a stored-or-passed pipeline to a comparable list, else None."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, list) else None


def _agent_pipeline_snapshot(agent_id: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """The agent's pipeline as it stands right now, for a later staleness check."""
    if not agent_id:
        return None
    try:
        agent = agent_service.get_agent(agent_id) or {}
    except Exception as exc:  # noqa: BLE001 - snapshot is best-effort
        print(f"⚠️  Could not snapshot pipeline for agent {agent_id}: {exc}", flush=True)
        return None
    return _normalized_pipeline(agent.get("pipeline"))


def _maybe_writeback_adapted_pipeline(
    agent_id: Optional[str],
    run_id: Optional[str],
    started_from_pipeline: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Persist post-trade adapted pipeline back onto the agent row.

    Declines to write when the agent's stored pipeline no longer matches the
    one this run started from. That was always possible and is now routine:
    Configure stays editable while a backtest runs, and several runs can be in
    flight at once, so an unconditional write silently discards whatever the
    user saved -- or whatever a sibling run adapted -- in the minutes since
    this run began. Losing the user's own edit is the worst outcome available
    here and the only unrecoverable one; skipping the write costs nothing,
    because ``final_pipeline`` stays in the run's metadata either way.

    ``started_from_pipeline`` of None means the caller could not establish a
    baseline, which is treated as "cannot prove it is safe" -- the write is
    skipped rather than forced.
    """
    if not agent_id or not run_id:
        return
    run = db.get_run(run_id)
    if not run:
        return
    metadata = run.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = None
    if not isinstance(metadata, dict):
        return
    adaptations = metadata.get("prompt_adaptations")
    final_pipeline = metadata.get("final_pipeline")
    if not adaptations or not isinstance(final_pipeline, list) or not final_pipeline:
        return
    current_pipeline = _agent_pipeline_snapshot(agent_id)
    if started_from_pipeline is None or current_pipeline != started_from_pipeline:
        print(
            f"↩️  Skipping adapted-pipeline write-back for agent {agent_id}: "
            "its pipeline changed while this run was in flight",
            flush=True,
        )
        return
    try:
        agent_service.update_agent(agent_id, pipeline=final_pipeline)
        print(
            f"✅ Wrote adapted pipeline back to agent {agent_id} "
            f"({len(adaptations)} adaptation day(s))",
            flush=True,
        )
    except Exception as exc:
        print(f"⚠️  Could not write adapted pipeline to agent {agent_id}: {exc}", flush=True)

class BacktestRunRequest(BaseModel):
    """Optional JSON body for POST /backtest/run.

    All fields are optional; when present they override the query-param
    defaults. ``strategy_prompt`` is a free-form strategy that REPLACES the
    built-in agent prompt for this run, and ``model`` overrides the LLM model id.
    ``agent_id`` targets a built-in agent's trading session (Discord / website).
    ``pipeline`` is the sub-agent step chain from the agent editor; when set it
    overrides ``strategy_prompt``. Long prompts belong in the body (not the query string).
    """
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    strategy_prompt: Optional[str] = None
    model: Optional[str] = None
    agent_id: Optional[str] = None
    pipeline: Optional[List[Dict[str, Any]]] = None
    data_source: Optional[Literal["alpaca", "vnpy_simulation", "ifind_ashare"]] = None
    universe: Optional[str] = None
    timeframe: Optional[str] = None
    decision_source: Optional[Literal["rule_based", "llm"]] = None
    billing_mode: Optional[BillingMode] = None
    provider_id: Optional[str] = None
    # Simulation starting cash for this run only — independent of portfolio sleeves.
    initial_capital: Optional[float] = None
    # Tradeable universe for this run. Accepts a list or a comma-separated string.
    assets: Optional[Any] = None


# /backtest/run spends real operator LLM credits per trading hour of the run, on
# an anonymous (session-id-only) surface. The params arrive as EITHER query
# params or a JSON body, so validation runs on the merged effective values in the
# handler rather than only on the Pydantic body.
MAX_STRATEGY_PROMPT_CHARS = 4000
MAX_BACKTEST_DAYS = 31
MAX_PIPELINE_STEPS = 20
MAX_PIPELINE_JSON_CHARS = 32000
MAX_BACKTEST_ASSETS = 30
_ASSET_TICKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.]{0,9}$")

# A model id is a provider/model slug: letters, digits, and . _ / - only, bounded
# length. This rejects a garbage/injection string reaching the backtest subprocess
# — it deliberately does NOT gate model *tier*: the dashboard UI intentionally
# offers expensive models (e.g. claude-opus), so tiering is a product/auth decision,
# not enforced here, and gating by the pricing table would 422 the UI's own options.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,63}$")

# Per-client run budget: a best-effort throttle only. The global
# ``backtest_status["running"]`` flag blocks *concurrent* runs; this throttles
# *serial* abuse from a well-behaved client. A client rotating its self-minted
# session id can evade it (see api/rate_limit) — the per-request caps above
# (model shape, prompt length, date range) are the hard limits.
_backtest_rate_limiter = FixedWindowRateLimiter(max_events=10, window_seconds=3600)


def _parse_ymd(value: str, field: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail=f"{field} must be a date in YYYY-MM-DD format.")


def _normalize_backtest_assets(raw: Any) -> Optional[List[str]]:
    """Parse / validate a caller-supplied asset universe.

    Returns ``None`` when the caller omitted assets (engine defaults to DJIA_30).
    Rejects empty lists, oversized universes, and malformed tickers.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw]
    else:
        raise HTTPException(
            status_code=422,
            detail="assets must be a list of tickers or a comma-separated string.",
        )
    cleaned: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        ticker = item.upper()
        if not _ASSET_TICKER_RE.fullmatch(ticker):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid asset ticker '{item}'.",
            )
        if ticker in seen:
            continue
        seen.add(ticker)
        cleaned.append(ticker)
    if not cleaned:
        raise HTTPException(status_code=422, detail="assets must include at least one ticker.")
    if len(cleaned) > MAX_BACKTEST_ASSETS:
        raise HTTPException(
            status_code=422,
            detail=f"assets too large (max {MAX_BACKTEST_ASSETS} tickers).",
        )
    return cleaned


def _validate_backtest_params(start_date, end_date, strategy_prompt, model, pipeline=None) -> None:
    """Reject malformed / cost-abuse inputs before scheduling the background run.

    - ``model`` must look like a model id (charset + length), which rejects an
      arbitrary/garbage string reaching the backtest subprocess. It does NOT cap
      model tier (the UI intentionally offers expensive models).
    - ``strategy_prompt`` is length-capped (it is injected into every LLM call).
    - the date range must be well-formed and bounded (each extra day is more
      hourly LLM calls).
    """
    if model and not _MODEL_ID_RE.match(model.strip()):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid model id '{model}'.",
        )
    if strategy_prompt and len(strategy_prompt) > MAX_STRATEGY_PROMPT_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"strategy_prompt too long (max {MAX_STRATEGY_PROMPT_CHARS} characters).",
        )
    if pipeline is not None:
        if not isinstance(pipeline, list) or not pipeline:
            raise HTTPException(status_code=422, detail="pipeline must be a non-empty array.")
        if len(pipeline) > MAX_PIPELINE_STEPS:
            raise HTTPException(
                status_code=422,
                detail=f"pipeline too long (max {MAX_PIPELINE_STEPS} steps).",
            )
        try:
            encoded = json.dumps(pipeline)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="pipeline must be JSON-serializable.")
        if len(encoded) > MAX_PIPELINE_JSON_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"pipeline too large (max {MAX_PIPELINE_JSON_CHARS} characters).",
            )
    start = _parse_ymd(start_date, "start_date")
    end = _parse_ymd(end_date, "end_date")
    if end < start:
        raise HTTPException(status_code=422, detail="end_date must not be before start_date.")
    if (end - start).days > MAX_BACKTEST_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"Date range too large (max {MAX_BACKTEST_DAYS} days).",
        )


def _resolve_market_profile_request(
    data_source: str,
    universe: Optional[str],
    timeframe: Optional[str],
    decision_source: Optional[str],
) -> tuple[MarketProfile, str]:
    """Validate source, profile, decision capability, then credentials."""
    try:
        validate_market_data_source(data_source)
    except UnsupportedMarketDataSource as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MarketDataSourceDisabled as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    try:
        profile = get_market_profile(data_source, universe)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    if timeframe is not None and timeframe != profile.timeframe:
        raise HTTPException(
            status_code=422,
            detail=(
                f"data_source={data_source!r} requires "
                f"timeframe={profile.timeframe!r}."
            ),
        )

    try:
        resolved_decision_source = resolve_decision_source(
            profile,
            decision_source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        ensure_market_data_source_available(data_source)
    except MarketDataSourceDisabled as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (MarketDataDependencyError, MarketDataCredentialsError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return profile, resolved_decision_source


def _resolve_backtest_pipeline(
    agent_id: Optional[str],
    body_pipeline: Any,
) -> Optional[List[Dict[str, Any]]]:
    """Resolve the sub-agent pipeline for a backtest run."""
    if body_pipeline is not None:
        return body_pipeline
    if not agent_id:
        return None
    agent = agent_service.get_agent(agent_id)
    if not agent:
        return None
    pipeline = agent.get("pipeline")
    if isinstance(pipeline, list) and pipeline:
        return pipeline
    return None


def _resolve_backtest_runtime(
    agent_id: Optional[str],
) -> tuple[str, Dict[str, Any]]:
    """Return the persisted hosted runtime for an agent-backed run."""
    if not agent_id:
        return DEFAULT_RUNTIME_TYPE, {}
    agent = agent_service.get_agent(agent_id)
    if not agent:
        # The session resolver owns the established 404 response.
        return DEFAULT_RUNTIME_TYPE, {}
    runtime_type = normalize_runtime_type(agent.get("runtime_type"))
    runtime_config = normalize_runtime_config(
        runtime_type, agent.get("runtime_config") or {}
    )
    return runtime_type, runtime_config


def _resolve_ai_hedge_fund_credential(request: Request, agent_id: Optional[str]) -> str:
    """Authorize and decrypt the per-agent market-data credential for one run."""
    if not agent_id:
        raise HTTPException(
            status_code=422,
            detail="AI Hedge Fund backtests must reference an owned agent",
        )
    ctx = _owner_context(request, request.headers.get("authorization"))
    agent = _require_agent_access(agent_id, ctx)
    if (agent.get("runtime_type") or DEFAULT_RUNTIME_TYPE) != AI_HEDGE_FUND_RUNTIME_TYPE:
        raise HTTPException(status_code=422, detail="Agent runtime is not AI Hedge Fund")
    if not (os.getenv("OPENROUTER_API_KEY") or "").strip():
        raise HTTPException(
            status_code=503,
            detail="AI Hedge Fund's platform-managed OpenRouter provider is not configured",
        )
    # The isolated venv is created by the deploy build, and render.yaml is
    # documentation rather than the deploy mechanism for this service -- so a
    # deployment without it is a live possibility. Reject the run here instead
    # of accepting it and failing inside a background subprocess minutes later.
    try:
        unavailable = runtime_unavailable_reason()
    except AiHedgeFundConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if unavailable:
        raise HTTPException(status_code=503, detail=unavailable)
    try:
        resolve_step_timeout_seconds()
    except AiHedgeFundConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        credential = agent_credential_store.get_secret(
            agent_id, FINANCIAL_DATASETS_CREDENTIAL
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not credential:
        raise HTTPException(
            status_code=422,
            detail=(
                "Configure a Financial Datasets API key on this AI Hedge Fund "
                "agent before running a backtest"
            ),
        )
    return credential


def _resolve_backtest_session(request: Request, agent_id: Optional[str]) -> str:
    """Return the session that should own this backtest run.

    When ``agent_id`` references a built-in agent, use that agent's session so
    results appear on its website card (without exposing ``session_id`` in public
    listings). Otherwise fall back to the caller's ``X-Session-Id``.
    """
    if not agent_id:
        return request.state.session_id
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if (agent.get("agent_type") or "external") != "builtin":
        raise HTTPException(
            status_code=422,
            detail="agent_id must reference a built-in agent",
        )
    return agent["session_id"]


@router.post("/backtest/run")
def run_backtest_endpoint(
    request: Request,
    start_date: str = "2026-05-01",
    end_date: str = "2026-05-07",
    strategy_prompt: Optional[str] = None,
    model: Optional[str] = None,
    data_source: str = ALPACA,
    universe: Optional[str] = None,
    timeframe: Optional[str] = None,
    decision_source: Optional[Literal["rule_based", "llm"]] = None,
    assets: Optional[str] = None,
    body: Optional[BacktestRunRequest] = None,
):
    """
    Trigger backtest in background (non-blocking).
    
    Returns immediately with status. Check /backtest/status to monitor progress.

    Accepts an optional JSON body (preferred for a long ``strategy_prompt``);
    body fields override the equivalent query params. Backward compatible with
    callers that pass only ``start_date``/``end_date`` as query params.
    """
    # Body (when provided) overrides query params.
    agent_id: Optional[str] = None
    pipeline: Optional[List[Dict[str, Any]]] = None
    initial_capital: Optional[float] = None
    billing_mode: Optional[BillingMode] = None
    provider_id: Optional[str] = None
    raw_assets: Any = assets
    if body is not None:
        start_date = body.start_date or start_date
        end_date = body.end_date or end_date
        strategy_prompt = body.strategy_prompt or strategy_prompt
        model = body.model or model
        data_source = body.data_source or data_source
        universe = body.universe or universe
        timeframe = body.timeframe or timeframe
        if body.decision_source is not None:
            decision_source = body.decision_source
        agent_id = body.agent_id
        if body.pipeline is not None:
            pipeline = body.pipeline
        if body.initial_capital is not None:
            initial_capital = body.initial_capital
        if body.billing_mode is not None:
            billing_mode = body.billing_mode
        if body.provider_id is not None:
            provider_id = body.provider_id
        if body.assets is not None:
            raw_assets = body.assets

    try:
        runtime_type, runtime_config = _resolve_backtest_runtime(agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    decision_source_was_explicit = decision_source is not None
    profile, resolved_decision_source = _resolve_market_profile_request(
        data_source,
        universe,
        timeframe,
        decision_source,
    )
    if runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE:
        if data_source != ALPACA:
            raise HTTPException(
                status_code=422,
                detail=(
                    "AI Hedge Fund currently supports the Alpaca US-equity "
                    "profile only."
                ),
            )
        if resolved_decision_source != LLM_DECISION_SOURCE:
            raise HTTPException(
                status_code=422,
                detail="AI Hedge Fund requires decision_source='llm'.",
            )
        financial_datasets_api_key = _resolve_ai_hedge_fund_credential(
            request, agent_id
        )
    else:
        financial_datasets_api_key = None
    selected_assets = (
        list(profile.symbols)
        if data_source == IFIND_ASHARE
        else _normalize_backtest_assets(raw_assets)
    )

    if initial_capital is not None:
        try:
            initial_capital = float(initial_capital)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="initial_capital must be a number.")
        if initial_capital <= 0:
            raise HTTPException(status_code=422, detail="initial_capital must be greater than 0.")
        if initial_capital > float(MAX_BACKTEST_INITIAL_CAPITAL):
            raise HTTPException(
                status_code=422,
                detail=f"initial_capital cannot exceed {MAX_BACKTEST_INITIAL_CAPITAL:g}.",
            )

    ignored_llm_fields: List[str] = []
    if resolved_decision_source == LLM_DECISION_SOURCE:
        if runtime_type == PIPELINE_RUNTIME_TYPE:
            pipeline = _resolve_backtest_pipeline(agent_id, pipeline)
            if agent_id and not model:
                agent = agent_service.get_agent(agent_id)
                if agent and agent.get("model_name"):
                    model = agent["model_name"]
        else:
            ignored_llm_fields = [
                name
                for name, value in (
                    ("strategy_prompt", strategy_prompt),
                    ("model", model),
                    ("pipeline", pipeline),
                )
                if value
            ]
            strategy_prompt = None
            model = None
            pipeline = None
    else:
        # A rule-based run drops the LLM-only fields — but validate them FIRST.
        # Dropping them before _validate_backtest_params meant a malformed model
        # was answered 200 instead of 422, so the caller never learned their
        # input was garbage. Rejecting the *combination* outright is not an
        # option: a body-level decision_source deliberately overrides a query
        # one, and leftover query params are exactly what that override exists
        # to neutralize. Validate, drop, then say what was dropped.
        _validate_backtest_params(start_date, end_date, strategy_prompt, model, pipeline)
        ignored_llm_fields = [
            name
            for name, value in (
                ("strategy_prompt", strategy_prompt),
                ("model", model),
                ("pipeline", pipeline),
            )
            if value
        ]
        strategy_prompt = None
        model = None
        pipeline = None

    if (
        decision_source_was_explicit
        and resolved_decision_source == LLM_DECISION_SOURCE
        and runtime_type == PIPELINE_RUNTIME_TYPE
        and not (model or "").strip()
    ):
        raise HTTPException(
            status_code=422,
            detail="model is required when decision_source='llm'.",
        )

    # Validate before taking rate-limit capacity or scheduling the worker.
    _validate_backtest_params(start_date, end_date, strategy_prompt, model, pipeline)

    if not _backtest_rate_limiter.allow(client_key(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many backtests started recently; please try again later.",
        )

    session_id = _resolve_backtest_session(request, agent_id)
    print(f"📌 /backtest/run endpoint called: start_date={start_date}, end_date={end_date}", flush=True)
    print(f"   Session: {session_id[:8]}...", flush=True)
    print(f"   Market data: {data_source}", flush=True)
    print(f"   Decision source: {resolved_decision_source}", flush=True)
    print(f"   Agent runtime: {runtime_type}", flush=True)
    if strategy_prompt and not pipeline:
        print(f"   Custom strategy prompt: {len(strategy_prompt)} chars", flush=True)
    if pipeline:
        print(f"   Sub-agent pipeline: {len(pipeline)} step(s)", flush=True)
    if model:
        print(f"   Model override: {model}", flush=True)
    if selected_assets:
        print(f"   Assets ({len(selected_assets)}): {', '.join(selected_assets)}", flush=True)
    else:
        print(f"   Assets: default DJIA ({len(DJIA_30)})", flush=True)

    from dashboard.backend.api.dependencies import _optional_user

    optional_user = _optional_user(
        request,
        request.headers.get("authorization") or request.headers.get("Authorization"),
    )
    user_id = optional_user["id"] if optional_user else None

    # Mint the id before constructing the signed worker handoff. It is both the
    # client-visible run identity and part of the handoff's tamper boundary.
    live_run_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    execution_handoff_payload: Optional[str] = None
    if (
        resolved_decision_source == LLM_DECISION_SOURCE
        and runtime_type == PIPELINE_RUNTIME_TYPE
    ):
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Sign in before running an LLM backtest.",
            )
        if billing_mode is None:
            raise HTTPException(
                status_code=422,
                detail="billing_mode is required for LLM backtests.",
            )
        if provider_id and not re.fullmatch(
            r"^[a-z0-9_]{2,64}$", provider_id.strip()
        ):
            raise HTTPException(status_code=422, detail="Invalid provider id.")
        if not model or not model.strip():
            raise HTTPException(
                status_code=422,
                detail="model is required for LLM backtests.",
            )
        provider_service = get_model_provider_service()
        provider_ids: tuple[str, ...]
        try:
            if billing_mode is BillingMode.BYOK:
                if not provider_id or not provider_id.strip():
                    raise HTTPException(
                        status_code=422,
                        detail="provider_id is required for BYOK backtests.",
                    )
                provider_id = provider_id.strip()
                route = provider_service.preflight_execution_model(
                    provider_id,
                    model.strip(),
                )
                provider_service.preflight_user_default_credential(
                    int(user_id), provider_id
                )
                provider_ids = (provider_id,)
            else:
                provider_ids = provider_service.resolve_platform_execution_candidates(
                    model.strip(),
                    preferred_provider_id=provider_id,
                )
                if not provider_ids:
                    raise HTTPException(
                        status_code=422,
                        detail="ATL Credits model execution is unavailable right now.",
                    )
                provider_id = provider_ids[0]
                route = provider_service.preflight_execution_model(
                    provider_id,
                    model.strip(),
                )
        except UnsupportedExecutionModel as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The selected model is not available "
                    "from this provider."
                ),
            ) from exc
        except CredentialResolutionError as exc:
            if billing_mode is BillingMode.PLATFORM_CREDITS:
                raise HTTPException(
                    status_code=422,
                    detail="ATL Credits model execution is unavailable right now.",
                ) from exc
            raise HTTPException(status_code=422, detail=exc.safe_message) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - never expose provider internals
            raise HTTPException(
                status_code=503,
                detail=LLMExecutionError.safe("provider_unavailable").safe_message,
            ) from exc

        execution_handoff_payload = create_execution_handoff(
            user_id=int(user_id),
            run_id=live_run_id,
            billing_mode=billing_mode,
            provider_id=provider_id,
            provider_ids=provider_ids,
            model_id=route.catalog_id,
            prompt_metadata={
                "start_date": start_date,
                "end_date": end_date,
                "strategy_prompt": strategy_prompt,
                "pipeline": pipeline,
                "data_source": data_source,
                "universe": profile.universe,
                "assets": selected_assets,
            },
        )

    refusal = _try_acquire_backtest_slot(
        live_run_id=live_run_id,
        session_id=session_id,
        # The caller's OWN session pays for the slot, even when the results
        # file under a built-in agent's session — see _backtest_owner_key.
        owner_session=request.state.session_id,
        user_id=user_id,
    )
    if refusal:
        print(f"⚠️ Backtest refused: {refusal}", flush=True)
        return {
            "success": False,
            "error": refusal,
        }

    if user_id is not None:
        analytics_instrumentation.emit_run_event(
            event_name="backtest_requested",
            user_id=int(user_id),
            run_id=live_run_id,
        )
        analytics_instrumentation.emit_run_event(
            event_name="backtest_queued",
            user_id=int(user_id),
            run_id=live_run_id,
        )

    # Start backtest in background thread
    print(f"🧵 Starting background thread for backtest", flush=True)
    # Keyword args, not positional: this call passes 14 of them and universe /
    # timeframe were inserted mid-signature. By name, a future insertion in the
    # wrong slot is a TypeError instead of a silently shifted argument.
    thread = _BackgroundThread(
        target=run_backtest_background,
        kwargs={
            "start_date": start_date,
            "end_date": end_date,
            "session_id": session_id,
            "strategy_prompt": strategy_prompt,
            "model": model,
            "pipeline": pipeline,
            "runtime_type": runtime_type,
            "runtime_config": runtime_config,
            "financial_datasets_api_key": financial_datasets_api_key,
            "agent_id": agent_id,
            "data_source": data_source,
            "live_run_id": live_run_id,
            "universe": profile.universe,
            "timeframe": profile.timeframe,
            "initial_capital": initial_capital,
            "assets": selected_assets,
            "decision_source": resolved_decision_source,
            "execution_handoff_payload": execution_handoff_payload,
        },
        daemon=True
    )
    try:
        thread.start()
    except Exception:
        # Releasing the slot is part of the same
        # unwind: leaving it held would burn one of the owner's concurrent
        # slots, and one of the server's, for the life of the process.
        _release_slot(live_run_id)
        if user_id is not None:
            analytics_instrumentation.emit_run_event(
                event_name="backtest_failed",
                user_id=int(user_id),
                run_id=live_run_id,
                error_category="internal_error",
            )
        raise

    response = {
        "success": True,
        "message": "Backtest started in background. Check /backtest/status for progress.",
        "status_url": "/backtest/status",
        "session_id": session_id,
        "data_source": data_source,
        "live_run_id": live_run_id,
        "run_id": live_run_id,
        "market": profile.market,
        "universe": profile.universe,
        "timeframe": profile.timeframe,
        "timezone": profile.timezone,
        "decision_source": resolved_decision_source,
        "benchmark": profile.benchmark,
        "assets": selected_assets or list(DJIA_30),
    }
    if runtime_type != PIPELINE_RUNTIME_TYPE:
        response["runtime_type"] = runtime_type
    if ignored_llm_fields:
        # Say what a rule-based run threw away. Dropping LLM-only fields is
        # correct, doing it invisibly is not: the caller otherwise cannot tell
        # a honoured model from an ignored one.
        response["ignored_fields"] = ignored_llm_fields
    if execution_handoff_payload:
        response["billing_mode"] = billing_mode.value
        response["provider_id"] = provider_id
    return response

@router.get("/backtest/status")
def get_backtest_status(
    request: Request,
    live_run_id: Optional[str] = Query(default=None),
):
    """Get backtest status (running, error, or completed).

    Pass ``live_run_id`` when following a specific concurrent job; otherwise the
    newest active (or recently finished) run for this session is returned.
    """
    session_id = request.state.session_id
    from dashboard.backend.api.dependencies import _optional_user

    viewer = _optional_user(
        request,
        request.headers.get("authorization") or request.headers.get("Authorization"),
    )
    slot = _resolve_status_slot(
        session_id=session_id,
        user_id=viewer["id"] if viewer else None,
        live_run_id=live_run_id,
    )

    # Tests and legacy callers still mutate ``backtest_status`` directly without
    # registering a slot — honour that mirror when no slot resolves.
    if slot is None:
        slot = {
            "running": bool(backtest_status.get("running")),
            "error": backtest_status.get("error"),
            "runs_count": int(backtest_status.get("runs_count") or 0),
            "started_at": backtest_status.get("started_at"),
            "progress_file": backtest_status.get("progress_file"),
            "live_run_id": backtest_status.get("live_run_id"),
            "session_id": backtest_session_id,
        }

    if slot.get("running"):
        elapsed = 0
        started_at = slot.get("started_at")
        if started_at:
            elapsed = max(0, int(time.time() - started_at))
        # _read_progress_file, not _read_backtest_progress: a slot whose
        # progress_file is still None has simply not written one yet, and the
        # legacy reader's fall-back to the global mirror would answer with
        # whichever sibling run touched it last.
        progress = _read_progress_file(slot.get("progress_file"))
        message = "Backtest is running… (multi-step agent pipeline; may take several minutes)"
        if progress:
            step = int(progress.get("step") or 0)
            total = int(progress.get("total_steps") or 0)
            if total > 0:
                pct = min(99, round(100 * step / total))
                message = f"Backtest running… step {step}/{total} ({pct}%)"
        payload = {
            "running": True,
            "message": message,
            "elapsed_seconds": elapsed,
            "live_run_id": slot.get("live_run_id"),
            "session_id": slot.get("session_id") or backtest_session_id,
        }
        if progress:
            payload["progress"] = progress
        return payload
    elif slot.get("error"):
        return {
            "running": False,
            "error": slot.get("error"),
            "live_run_id": slot.get("live_run_id"),
            "message": "Backtest failed",
        }
    elif int(slot.get("runs_count") or 0) > 0:
        # Verify the completed backtest belongs to this session
        runs = db.get_runs_by_session(session_id)
        if not runs:
            return {
                "running": False,
                "error": "Backtest completed but no runs found for this session",
                "live_run_id": slot.get("live_run_id"),
                "message": "Session mismatch",
            }

        return {
            "running": False,
            "success": True,
            "runs_count": int(slot.get("runs_count") or 0),
            "session_id": session_id,
            "live_run_id": slot.get("live_run_id"),
            "message": "Backtest completed successfully",
        }
    else:
        return {
            "running": False,
            "message": "No backtest has been run yet",
        }


# ============================================================================
# Backtest Routes
# ============================================================================

@router.get("/api/backtest/runs", response_model=List[RunMetadata])
def get_backtest_runs(request: Request):
    """Get all backtest runs for this session."""
    session_id = get_session_id_from_request(request)
    runs = db.get_runs_by_session(session_id)
    runs = [r for r in runs if r['mode'] == 'backtest']
    return [_run_metadata_response(run) for run in runs]


# IMPORTANT: Register /compare/latest BEFORE /{run_id} to prevent {run_id} from matching "compare/latest"

@router.get("/api/backtest/compare/latest", response_model=ComparisonResponse)
def compare_latest_backtests(request: Request):
    """Compare the latest backtest runs + baselines for this session."""
    session_id = get_session_id_from_request(request)
    
    # Get this session's runs
    all_runs = db.get_runs_by_session(session_id) or []
    backtest_runs = [r for r in all_runs if r['mode'] == 'backtest']
    baseline_runs = [r for r in all_runs if r['mode'] == 'baseline']
    runs = backtest_runs + baseline_runs
    
    if not runs:
        raise HTTPException(status_code=404, detail="No backtest or baseline runs found for this session")
    
    # Group by agent and get latest for each
    latest_by_agent = {}
    for run in runs:
        agent = run['agent_name']
        if agent not in latest_by_agent or run['created_at'] > latest_by_agent[agent]['created_at']:
            latest_by_agent[agent] = run
    
    # Build comparison response
    comparison_runs = []
    for agent, run in latest_by_agent.items():
        equity_data = db.get_equity_curve(run['run_id'])
        equity_data = _filter_equity_for_run(run, equity_data)
        
        if equity_data:
            comparison_runs.append(EquityCurve(
                run_id=run['run_id'],
                agent_name=agent,
                data=[EquityPoint(**point) for point in equity_data],
                metrics={
                    'total_return': run['total_return'],
                    'sharpe_ratio': run['sharpe_ratio'],
                    'max_drawdown': run['max_drawdown'],
                    'num_trades': run['num_trades']
                }
            ))
    
    if not comparison_runs:
        raise HTTPException(status_code=404, detail="No equity data found for session")
    
    best_run = max(comparison_runs, key=lambda r: r.metrics['total_return'] or 0)
    
    return ComparisonResponse(
        runs=comparison_runs,
        summary={
            'num_runs': len(comparison_runs),
            'best_performer': best_run.agent_name,
            'best_return': best_run.metrics['total_return']
        }
    )


@router.get("/api/backtest/{run_id}/chart-data", response_model=BacktestChartData)
def get_backtest_chart_data(run_id: str, request: Request):
    """Chart-ready equity series for the Playground backtest page.

    Uses the same DJIA index + Nasdaq-100 baselines and gapless market-hour
    x-axis as ``/runs/{run_id}/plot.png`` (Discord chart), plus the paired
    stored Buy & Hold curve when one exists.
    """
    session_id = get_session_id_from_request(request)
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found or not yours")

    profile = _market_profile_for_run(run)
    agent_curve = _filter_equity_for_run(run, db.get_equity_curve(run_id))
    if not agent_curve:
        raise HTTPException(status_code=404, detail="No equity data to plot for this run")

    initial_capital = float(
        run.get("initial_equity") or agent_curve[0].get("equity") or 1_000
    )
    agent_card = agent_service.agents.get_agent_by_session(session_id)
    card_name = (agent_card or {}).get("name")

    try:
        payload = build_backtest_chart_data(
            run_id=run_id,
            agent_name=run.get("agent_name") or "Agent",
            llm_model=run.get("llm_model"),
            start_date=run.get("start_date") or "",
            end_date=run.get("end_date") or "",
            initial_capital=initial_capital,
            agent_curve=agent_curve,
            card_name=card_name,
            stored_baselines=_stored_buyhold_baseline(run),
            include_market_indexes=profile.index_baseline_enabled,
            market_timezone=profile.timezone,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return BacktestChartData(**payload)


@router.get("/api/backtest/{run_id}", response_model=EquityCurve)
def get_backtest_run(run_id: str, request: Request):
    """Get specific backtest run with equity curve."""
    session_id = get_session_id_from_request(request)
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found or not yours")
    
    equity_data = db.get_equity_curve(run_id)
    
    return EquityCurve(
        run_id=run_id,
        agent_name=run['agent_name'],
        data=[EquityPoint(**point) for point in equity_data],
        metrics={
            'total_return': run['total_return'],
            'sharpe_ratio': run['sharpe_ratio'],
            'max_drawdown': run['max_drawdown'],
            'num_trades': run['num_trades']
        }
    )


@router.get("/runs/latest/metrics", response_model=RunMetadata)
def get_latest_metrics(request: Request):
    """Get metrics for the latest Agent backtest run in this session (excludes baselines)."""
    session_id = request.state.session_id
    runs = [r for r in db.get_runs_by_session(session_id) or [] 
            if r['mode'] == 'backtest' and r['agent_name'] == 'Agent']
    if not runs:
        raise HTTPException(status_code=404, detail="No Agent backtest runs found for this session")
    
    latest_run = max(runs, key=lambda r: r['created_at'])
    return _run_metadata_response(latest_run)


@router.get("/runs", response_model=List[RunMetadata])
def get_runs(request: Request, mode: Optional[str] = None):
    """
    Get all backtest runs (public, not filtered by session).
    Backtest results are meant to be shared/viewed, not isolated per user.
    
    Query params:
    - mode: 'backtest' or 'paper' (optional)
    """
    # Get ALL runs - backtest results are public
    all_runs = db.get_all_runs()
    
    if mode:
        runs = [r for r in all_runs if r['mode'] == mode]
    else:
        # Default: backtest runs only
        runs = [r for r in all_runs if r['mode'] == 'backtest']
    
    print(f"\n📍 /runs: returning {len(runs)} backtest runs")
    
    return [_run_metadata_response(run) for run in runs]


@router.get("/runs/{run_id}", response_model=RunMetadata)
def get_run(run_id: str, request: Request):
    """Get metadata for a specific run."""
    session_id = request.state.session_id
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or not yours")
    return _run_metadata_response(run)


@router.get("/runs/{run_id}/equity", response_model=EquityCurve)
def get_equity_curve(run_id: str, request: Request):
    """
    Get equity curve for a specific run.
    
    Returns time-series data with equity, cash, positions_value, daily_return.
    Filtered to market hours only (9:30 AM - 4:00 PM ET).
    """
    session_id = request.state.session_id
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or not yours")
    
    equity_data = db.get_equity_curve(run_id)
    equity_data = _filter_equity_for_run(run, equity_data)
    
    return EquityCurve(
        run_id=run_id,
        agent_name=run['agent_name'],
        data=[EquityPoint(**point) for point in equity_data],
        metrics={
            'total_return': run['total_return'],
            'sharpe_ratio': run['sharpe_ratio'],
            'max_drawdown': run['max_drawdown'],
            'num_trades': run['num_trades']
        }
    )


@router.get("/runs/{run_id}/trades")
def get_run_trades(run_id: str, request: Request):
    """Trades plus the orders that did *not* fill, for an owned run.

    The two lists are complementary, not overlapping: ``trades`` is the
    complete, uncapped fill history straight out of the trades table, and
    ``order_events`` carries only the rejected and partially-filled orders,
    which are the outcomes a trade row cannot express. Clients reassemble the
    full order history by merging them. That split is what keeps the metadata
    sample from having to hold a copy of every fill -- see
    ``engine._unfilled_order_events``.
    """
    session_id = request.state.session_id
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or not yours")
    trades = db.get_trades(run_id)
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    order_events = metadata.get("order_events")
    if not isinstance(order_events, list):
        order_events = []
    order_event_count = metadata.get("order_events_count")
    if not isinstance(order_event_count, int) or order_event_count < len(order_events):
        order_event_count = len(order_events)
    order_events_truncated = metadata.get("order_events_truncated")
    if not isinstance(order_events_truncated, int) or order_events_truncated < 0:
        order_events_truncated = max(order_event_count - len(order_events), 0)
    return {
        "run_id": run_id,
        "trades": trades,
        "count": len(trades),
        "order_events": order_events,
        "order_event_count": order_event_count,
        "order_events_returned": len(order_events),
        "order_events_truncated": order_events_truncated,
    }


@router.get("/runs/{run_id}/rejected-orders")
def get_run_rejected_orders(run_id: str, request: Request):
    """Rejected / partially-filled order records for a run owned by this session.

    Served here rather than on RunMetadata because these are per-step audit
    records — a T+1 A-share run can emit thousands — and RunMetadata is the
    response_model for two list routes the dashboard fetches on every load.

    ``count`` is the run's true total; ``returned`` is how many this response
    carries. They differ when the engine capped the persisted sample, in which
    case ``truncated`` says by how much.

    ``t1_deferrals`` answers the complementary question. A rejected order means
    the agent *submitted* something unfillable; a deferral means it wanted to
    exit and sized down because it could not. The built-in agents now do the
    latter, so for them this list — not ``rejected_orders`` — is where T+1's
    effect on strategy shows up.
    """
    session_id = request.state.session_id
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or not yours")
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    records = metadata.get("rejected_orders") or []
    deferrals = metadata.get("t1_deferrals") or []
    return {
        "run_id": run_id,
        "rejected_orders": records,
        "count": metadata.get("rejected_orders_count", len(records)),
        "returned": len(records),
        "truncated": metadata.get("rejected_orders_truncated", 0),
        "t1_deferrals": deferrals,
        "t1_deferred_events": metadata.get("t1_deferred_events", len(deferrals)),
        "t1_deferred_shares": metadata.get("t1_deferred_shares", 0),
        "t1_deferrals_truncated": metadata.get("t1_deferrals_truncated", 0),
    }


@router.get("/runs/{run_id}/plot.png", include_in_schema=False)
def get_run_plot(run_id: str):
    """Render an equity-curve comparison PNG (agent vs baselines) for a run.

    Public endpoint: the path ends in ``.png`` so it is exempt from the session
    middleware. Used by the Discord bot to post a chart after a backtest, and
    usable directly as an <img> src. Uses the gapless market-hour axis from
    ``docs/examples/simple_trading_agent_backtest.py`` with Playground colors.

    Sync ``def`` so FastAPI runs the CPU-bound matplotlib render in its
    threadpool rather than blocking the event loop; the PNG is cached per run_id.
    """
    return Response(content=_run_plot_png(run_id), media_type="image/png")


class _UncachedPlotPng(Exception):
    """Carries a rendered PNG that must *not* be memoized.

    Raised when Yahoo was unreachable, so the chart is missing its index
    baselines. ``lru_cache`` never stores a call that raised, which is what
    keeps a degraded render out of the cache — otherwise one Yahoo 429 would
    pin a baseline-free chart to that run for the life of the process.
    """

    def __init__(self, png: bytes) -> None:
        super().__init__("index baselines unavailable; render not cached")
        self.png = png


_DEGRADED_PLOT_NOTE = (
    "⚠ Index benchmarks unavailable — market-data provider unreachable"
)

# A short negative cache for degraded renders. Keeping them out of the lru_cache
# entirely (see _UncachedPlotPng) is right for a blip, but a *persistent* Yahoo
# block is a steady state on a free-tier host with shared egress IPs — and this
# route is public, unauthenticated and exempt from the session middleware. With
# no bound at all, that state re-runs the full matplotlib render on every hit,
# forever, which is precisely the cost the lru_cache exists to avoid. One retry
# per run per minute keeps the recovery behaviour without the amplification.
_DEGRADED_PLOT_TTL_SECONDS = 60.0
_DEGRADED_PLOT_MAX_ENTRIES = 128
_degraded_plot_lock = _PlotCacheLock()
_degraded_plot_cache: Dict[str, Tuple[float, bytes]] = {}


def _degraded_plot_cached(run_id: str) -> Optional[bytes]:
    with _degraded_plot_lock:
        entry = _degraded_plot_cache.get(run_id)
        if not entry:
            return None
        stored_at, png = entry
        if (time.monotonic() - stored_at) >= _DEGRADED_PLOT_TTL_SECONDS:
            _degraded_plot_cache.pop(run_id, None)
            return None
        return png


def _degraded_plot_store(run_id: str, png: bytes) -> None:
    now = time.monotonic()
    with _degraded_plot_lock:
        expired = [
            key
            for key, (stored_at, _png) in _degraded_plot_cache.items()
            if (now - stored_at) >= _DEGRADED_PLOT_TTL_SECONDS
        ]
        for key in expired:
            _degraded_plot_cache.pop(key, None)
        # Bound the dict even if every entry is still live (many distinct runs
        # requested inside one outage window): evict oldest-inserted first.
        while len(_degraded_plot_cache) >= _DEGRADED_PLOT_MAX_ENTRIES:
            _degraded_plot_cache.pop(next(iter(_degraded_plot_cache)), None)
        _degraded_plot_cache[run_id] = (now, png)


def _clear_degraded_plot_cache() -> None:
    """Test hook: the TTL is wall-clock, so tests must reset it explicitly."""
    with _degraded_plot_lock:
        _degraded_plot_cache.clear()


def _run_plot_png(run_id: str) -> bytes:
    """``_render_run_plot_png`` with the uncached-degraded-render escape unwrapped.

    A degraded render is served from the short negative cache for
    ``_DEGRADED_PLOT_TTL_SECONDS`` before Yahoo is tried again, so a sustained
    outage costs one re-render per run per minute instead of one per request.
    """
    cached = _degraded_plot_cached(run_id)
    if cached is not None:
        return cached
    try:
        png = _render_run_plot_png(run_id)
    except _UncachedPlotPng as exc:
        _degraded_plot_store(run_id, exc.png)
        return exc.png
    with _degraded_plot_lock:
        _degraded_plot_cache.pop(run_id, None)
    return png


@lru_cache(maxsize=128)
def _render_run_plot_png(run_id: str) -> bytes:
    """Render (and memoize) the equity-curve comparison PNG for ``run_id``.

    A run's equity data is immutable once written and run_ids are unique per
    run, so the rendered bytes are reused without re-querying the DB or
    re-rendering. HTTPExceptions (missing run / no equity data) are raised, not
    cached — so data that appears later is still picked up on a retry. A render
    whose index baselines were lost to a Yahoo outage leaves the same way, via
    ``_UncachedPlotPng``; call through ``_run_plot_png`` to get its bytes.
    """
    run = db.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    agent_card = agent_service.agents.get_agent_by_session(run.get("session_id") or "")
    agent_label = resolve_agent_chart_label(
        run.get("agent_name"),
        run.get("llm_model"),
        (agent_card or {}).get("name"),
    )
    profile = _market_profile_for_run(run)
    agent_curve = _filter_equity_for_run(run, db.get_equity_curve(run_id))
    timestamps, agent_values = curve_timestamps_and_values(agent_curve)
    if not timestamps:
        raise HTTPException(status_code=404, detail="No equity data to plot for this run")

    initial_capital = float(run.get("initial_equity") or agent_values[0] or 1_000)
    index_baselines_ok = True
    if profile.index_baseline_enabled:
        baselines, index_baselines_ok = market_index_baselines_with_status(
            timestamps,
            run.get("start_date") or "",
            run.get("end_date") or "",
            initial_capital,
            context=run_id,
        )
    else:
        baselines = [
            (label, baseline_run_id, align_equity(
                timestamps, equity_lookup(curve)
            ))
            for label, baseline_run_id, curve in _stored_buyhold_baseline(run)
        ]

    try:
        png = render_backtest_equity_png(
            agent_label=agent_label,
            agent_run_id=run_id,
            timestamps=timestamps,
            agent_values=agent_values,
            baselines=baselines,
            market_timezone=profile.timezone,
            note=None if index_baselines_ok else _DEGRADED_PLOT_NOTE,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not index_baselines_ok:
        raise _UncachedPlotPng(png)
    return png


@router.get("/compare", response_model=ComparisonResponse)
def compare_runs(run_ids: str, request: Request):
    """
    Compare multiple runs (public, not filtered by session).
    
    Query params:
    - run_ids: comma-separated list of run IDs (e.g., "run1,run2,run3")
    
    Returns equity curves for all specified runs, ready for multi-line chart.
    """
    ids = [rid.strip() for rid in run_ids.split(',') if rid.strip()]
    
    if not ids:
        raise HTTPException(status_code=400, detail="At least one run_id required")
    
    runs = []
    final_equities = []
    
    for run_id in ids:
        # Get run without session filter - backtest results are public
        run = db.get_run(run_id)
        if not run:
            continue
        
        equity_data = db.get_equity_curve(run_id)
        equity_data = _filter_equity_for_run(run, equity_data)
        if equity_data:
            final_equities.append(run['final_equity'] or 0)
            
            runs.append(EquityCurve(
                run_id=run_id,
                agent_name=run['agent_name'],
                data=[EquityPoint(**point) for point in equity_data],
                metrics={
                    'total_return': run['total_return'],
                    'sharpe_ratio': run['sharpe_ratio'],
                    'max_drawdown': run['max_drawdown'],
                    'num_trades': run['num_trades']
                }
            ))
    
    if not runs:
        raise HTTPException(status_code=404, detail="No data found for specified runs")
    
    # Build summary: identify winner (highest final equity)
    best_run = max(runs, key=lambda r: r.metrics['total_return'] or 0) if runs else None
    
    return ComparisonResponse(
        runs=runs,
        summary={
            'num_runs': len(runs),
            'best_performer': best_run.agent_name if best_run else None,
            'best_return': best_run.metrics['total_return'] if best_run else None
        }
    )
