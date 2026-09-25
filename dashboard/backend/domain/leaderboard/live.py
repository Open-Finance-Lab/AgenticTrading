"""Calendar-month Live Trading Leaderboard.

This board is not Daily stretched to 30 days and not the fixed contest window.
One continuous paper-trading month per entry: the chart axis is the current
America/New_York calendar month, hourly nodes are US cash-session hours
(10:00–16:00 ET, weekdays), and points after the freeze close stay empty so
the line cannot interpolate into the live tail or the future.

Phase 1 caches auto-compute baselines/indices for month-open → last completed
session under ``leaderboard-live``. LLM models are written by
``refresh_live_leaderboard(deploy_models=True)`` (or the deploy script), never
on public GET. Each freeze close appends one cash session onto the prior
snapshot (cash + positions), so a month costs about one full backtest, not a
replay from the 1st every night.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from calendar import monthrange
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

from dashboard.backend.database import db
from dashboard.backend.domain.leaderboard.baselines import INITIAL_CAPITAL, calc_metrics
from dashboard.backend.domain.leaderboard.service import (
    _rank_entries,
    load_leaderboard_config,
)
from dashboard.backend.domain.leaderboard.strategies._common import reference_start_date
from dashboard.backend.paths import DATA_DIR
import dashboard.backend.domain.leaderboard.service as lb_service

_US_EASTERN = ZoneInfo("America/New_York")
_US_CASH_OPEN = time(9, 30)
_US_CASH_CLOSE = time(16, 0)
# Alpaca 1h bars for the cash session typically print on the hour from 10:00
# through the 16:00 close (7 nodes). 9:30 is the open, not a separate hourly
# close. Phase 1 will replace this synthetic grid with the actual tape's
# timestamps and only *pad* the tail out to month-end.
_RTH_HOURS = (10, 11, 12, 13, 14, 15, 16)

LIVE_SESSION_ID = "leaderboard-live"
LIVE_PHASE = 1
# Season-0 local roster: only these LLM curves are deployed and shown on Live.
# Contest/daily still use the full leaderboard.json list.
LIVE_MODEL_IDS = (
    "gpt_5_5",
    "deepseek_v4_pro",
    "nemotron_3_nano_30b",
)
LIVE_SNAPSHOT_KEY = "live_portfolio_snapshot"
_LIVE_REFRESH_STATE_PATH = DATA_DIR / "leaderboard_live_refresh.json"
_live_refresh_lock = threading.Lock()
_live_refresh_running = False


def _coerce_as_of_eastern(as_of: Optional[Union[date, datetime]] = None) -> datetime:
    if as_of is None:
        return datetime.now(_US_EASTERN)
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=_US_EASTERN)
        return as_of.astimezone(_US_EASTERN)
    return datetime(
        as_of.year, as_of.month, as_of.day,
        _US_CASH_CLOSE.hour, _US_CASH_CLOSE.minute,
        tzinfo=_US_EASTERN,
    )


def live_month_dates(as_of: Optional[Union[date, datetime]] = None) -> Tuple[str, str]:
    """First and last calendar day of the current Eastern month."""
    now = _coerce_as_of_eastern(as_of)
    start = date(now.year, now.month, 1)
    end = date(now.year, now.month, monthrange(now.year, now.month)[1])
    return start.isoformat(), end.isoformat()


def _weekdays_inclusive(start: date, end: date) -> List[date]:
    if end < start:
        return []
    days: List[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def live_month_hourly_axis(start_date: str, end_date: str) -> List[str]:
    """RTH hourly ISO timestamps covering the calendar month (weekends skipped)."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    out: List[str] = []
    for day in _weekdays_inclusive(start, end):
        for hour in _RTH_HOURS:
            ts = datetime(day.year, day.month, day.day, hour, 0, tzinfo=_US_EASTERN)
            out.append(ts.isoformat())
    return out


def _previous_weekday(day: date) -> date:
    cursor = day - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def next_session_day(day: date) -> date:
    """Next US weekday after ``day`` (Sat/Sun → Monday)."""
    cursor = day + timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor += timedelta(days=1)
    return cursor


def live_increment_bounds(
    prior_end: Optional[str],
    *,
    month_start: str,
    freeze_end: str,
) -> Optional[Tuple[str, str]]:
    """Inclusive window of sessions not yet on the stored snapshot.

    ``None`` means the snapshot already covers ``freeze_end``. With no prior
    row the window is month-open → freeze (first backtest of the month).
    """
    freeze = date.fromisoformat(freeze_end)
    start = date.fromisoformat(month_start)
    if not prior_end:
        if freeze < start:
            return None
        return month_start, freeze_end
    prior = date.fromisoformat(prior_end)
    if prior >= freeze:
        return None
    nxt = next_session_day(prior)
    if nxt < start:
        nxt = start
    if nxt > freeze:
        return None
    return nxt.isoformat(), freeze.isoformat()


def live_clock(as_of: Optional[Union[date, datetime]] = None) -> Dict[str, Any]:
    """Session state for the live month at ``as_of`` (America/New_York).

    - ``weekend`` / ``preopen``: yesterday (or Friday) is frozen; no live tail.
    - ``rth``: yesterday is frozen; today is the live tail through the last
      hourly node ``<= as_of``.
    - ``closed``: today's cash session is complete and joins the freeze.
    """
    now = _coerce_as_of_eastern(as_of)
    today = now.date()
    weekday = now.weekday()

    if weekday >= 5:
        session_state = "weekend"
        frozen_through = today
        while frozen_through.weekday() >= 5:
            frozen_through -= timedelta(days=1)
        live_day = None
    elif now.time() < _US_CASH_OPEN:
        session_state = "preopen"
        frozen_through = _previous_weekday(today)
        live_day = today
    elif now.time() >= _US_CASH_CLOSE:
        session_state = "closed"
        frozen_through = today
        live_day = None
    else:
        session_state = "rth"
        frozen_through = _previous_weekday(today)
        live_day = today

    return {
        "as_of": now.isoformat(),
        "as_of_date": today.isoformat(),
        "session_state": session_state,
        "frozen_through": frozen_through.isoformat(),
        "live_day": live_day.isoformat() if live_day else None,
    }


def _parse_axis_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def printed_through_timestamp(axis: List[str], as_of: datetime) -> Optional[str]:
    """Last axis node that is allowed to hold a printed value (not the future)."""
    last: Optional[str] = None
    for ts in axis:
        if _parse_axis_ts(ts) <= as_of:
            last = ts
        else:
            break
    return last


def _next_tick(axis: List[str], printed_through: Optional[str]) -> Optional[str]:
    if not axis:
        return None
    if printed_through is None:
        return axis[0]
    try:
        idx = axis.index(printed_through)
    except ValueError:
        return None
    if idx + 1 < len(axis):
        return axis[idx + 1]
    return None


def live_freeze_config(
    as_of: Optional[Union[date, datetime]] = None,
) -> Optional[Dict[str, Any]]:
    """Contest roster, cached under ``leaderboard-live`` for month-open → freeze.

    ``end_date`` is the last *completed* cash session, not month-end — we must
    not backtest into days that have not closed. Returns ``None`` when this
    month has no completed session yet (e.g. Aug 1 weekend).
    """
    now = _coerce_as_of_eastern(as_of)
    month_start, _month_end = live_month_dates(now)
    clock = live_clock(now)
    freeze_end = clock["frozen_through"]
    if date.fromisoformat(freeze_end) < date.fromisoformat(month_start):
        return None
    base = load_leaderboard_config()
    live_base = {k: v for k, v in base.items() if k != "reference_start_date"}
    return {
        **live_base,
        "session_id": LIVE_SESSION_ID,
        "start_date": month_start,
        "end_date": freeze_end,
        "reference_start_date": reference_start_date(month_start, None),
        "period": "live",
        "board_title": "Live Trading Leaderboard",
        "phase_label": "Season 0",
        "standings_label": "Ranking",
    }


def live_board_strategies(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Baselines plus the Season-0 Live LLM roster (not the full contest list)."""
    cfg = config or load_leaderboard_config()
    out: List[Dict[str, Any]] = []
    for strategy in cfg.get("strategies", []):
        if strategy.get("strategy") == "llm_agent" and strategy.get("id") not in LIVE_MODEL_IDS:
            continue
        out.append(strategy)
    return out


def live_llm_entries(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return [
        s for s in live_board_strategies(config)
        if s.get("strategy") == "llm_agent"
    ]


def clear_live_session_runs() -> int:
    """Delete every ``leaderboard-live`` run (curves, trades, decisions)."""
    runs = db.get_runs_by_session(LIVE_SESSION_ID) or []
    for run in runs:
        run_id = run.get("run_id")
        if run_id:
            db.delete_run(run_id)
    return len(runs)


def _live_window_key(config: Dict[str, Any]) -> str:
    return f"{config['session_id']}|{config['start_date']}|{config['end_date']}"


def _live_refresh_state() -> Dict[str, Any]:
    if not _LIVE_REFRESH_STATE_PATH.exists():
        return {}
    try:
        with open(_LIVE_REFRESH_STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_live_refresh_state(state: Dict[str, Any]) -> None:
    dest_dir = _LIVE_REFRESH_STATE_PATH.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest_dir), prefix=".live_refresh_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, _LIVE_REFRESH_STATE_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _clip_end(left: str, right: str) -> str:
    return left if left <= right else right


def _run_metadata_dict(run: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not run:
        return {}
    meta = run.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return meta if isinstance(meta, dict) else {}


def _snapshot_from_run(run: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    snap = _run_metadata_dict(run).get(LIVE_SNAPSHOT_KEY)
    if isinstance(snap, dict) and snap.get("cash") is not None:
        return snap
    return None


def _stitch_equity_curves(
    prior: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not prior:
        return list(new)
    seen = {str(point.get("timestamp")) for point in prior}
    extra = [point for point in new if str(point.get("timestamp")) not in seen]
    return list(prior) + extra


def _set_live_refresh_running(value: bool) -> None:
    global _live_refresh_running
    _live_refresh_running = value


def deploy_live_model_increment(
    entry: Dict[str, Any],
    freeze_cfg: Dict[str, Any],
    *,
    force_refresh: bool = False,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Append unseen cash sessions onto a live-month LLM snapshot.

    Trades only ``live_increment_bounds`` (usually one day) and restores cash
    plus positions from the previous freeze row. A missing snapshot falls back
    to a full month-open → freeze replay once, then stores the book for later
    nights. Does not change contest/daily ``deploy_model_run``.
    """
    entry_id = entry["id"]
    month_start = freeze_cfg["start_date"]
    freeze_end = freeze_cfg["end_date"]
    session_id = freeze_cfg["session_id"]
    initial_capital = float(freeze_cfg.get("initial_capital", INITIAL_CAPITAL))

    prior = None if force_refresh else lb_service._find_live_month_run(
        entry_id, month_start, freeze_end, session_id
    )
    snapshot = _snapshot_from_run(prior)
    if prior and str(prior.get("end_date") or "") == freeze_end and not force_refresh:
        return {
            "entry_id": entry_id,
            "run_id": prior.get("run_id"),
            "cached": True,
            "increment": False,
            "model": entry.get("model"),
            "window": {"start_date": month_start, "end_date": freeze_end},
            "segment": None,
            "total_return": prior.get("total_return"),
            "final_equity": prior.get("final_equity"),
            "llm_calls": prior.get("llm_calls"),
        }

    if force_refresh or prior is None or snapshot is None:
        segment_start, segment_end = month_start, freeze_end
        snapshot = None
        prior_curve: List[Dict[str, Any]] = []
        resumed = False
        if prior is not None and snapshot is None and not force_refresh:
            print(
                f"⚠️ Live {entry_id}: no portfolio snapshot on "
                f"{prior.get('run_id')}; replaying {month_start} → {freeze_end} once"
            )
    else:
        bounds = live_increment_bounds(
            str(prior.get("end_date") or ""),
            month_start=month_start,
            freeze_end=freeze_end,
        )
        if bounds is None:
            return {
                "entry_id": entry_id,
                "run_id": prior.get("run_id"),
                "cached": True,
                "increment": False,
                "model": entry.get("model"),
                "window": {"start_date": month_start, "end_date": freeze_end},
                "segment": None,
            }
        segment_start, segment_end = bounds
        prior_curve = db.get_equity_curve(prior["run_id"]) or []
        resumed = True

    strategy_impl = lb_service.get_strategy(entry)
    bars_start = lb_service.reference_start_date(segment_start, freeze_cfg)
    if bars_start > segment_start:
        bars_start = segment_start
    bars = lb_service.fetch_hourly_bars(
        strategy_impl.required_symbols(), bars_start, segment_end
    )
    if not bars:
        raise RuntimeError(
            f"No market data returned for live increment {bars_start} → {segment_end}"
        )
    print(
        f"  live increment {entry_id}: trade {segment_start} → {segment_end} "
        f"(month {month_start} → {freeze_end}, resume={resumed})"
    )

    curve = strategy_impl.run(
        bars,
        segment_start,
        segment_end,
        initial_capital,
        starting_snapshot=snapshot,
    )
    if not curve:
        raise RuntimeError(
            f"No equity curve produced for live increment '{entry_id}' "
            f"{segment_start} → {segment_end}"
        )

    stitched = _stitch_equity_curves(prior_curve, curve)
    metrics = calc_metrics(stitched, initial_capital)
    run_id = lb_service._run_id(entry_id, month_start, freeze_end)

    input_tokens = int(getattr(strategy_impl, "input_tokens", 0) or 0)
    output_tokens = int(getattr(strategy_impl, "output_tokens", 0) or 0)
    llm_calls = int(getattr(strategy_impl, "llm_calls", 0) or 0)
    llm_decisions = lb_service._reported_int(strategy_impl, "llm_decisions")
    decision_steps = int(getattr(strategy_impl, "decision_steps", 0) or 0)
    model_id = getattr(strategy_impl, "model_id", None) or entry.get("model_id")
    if resumed and prior:
        input_tokens += int(prior.get("input_tokens") or 0)
        output_tokens += int(prior.get("output_tokens") or 0)
        llm_calls += int(prior.get("llm_calls") or 0)
    est_cost = lb_service.token_cost.estimate_cost_usd(
        model_id, input_tokens, output_tokens
    )

    lb_service._reject_if_llm_fallback(
        entry_id,
        strategy_impl,
        int(getattr(strategy_impl, "llm_calls", 0) or 0),
        llm_decisions=llm_decisions,
        decision_steps=decision_steps,
        model=entry.get("model"),
        model_id=model_id,
        allow_fallback=allow_fallback,
    )

    new_snapshot = getattr(strategy_impl, "last_portfolio_snapshot", None)
    meta = lb_service._llm_run_metadata(
        entry_id,
        entry,
        strategy_impl,
        model_id=model_id,
        initial_capital=initial_capital,
        start_date=month_start,
        end_date=freeze_end,
    ) or {}
    meta[LIVE_SNAPSHOT_KEY] = new_snapshot
    meta["live_increment"] = {
        "segment_start": segment_start,
        "segment_end": segment_end,
        "resumed_from_run_id": prior.get("run_id") if resumed else None,
        "full_replay": not resumed,
    }

    trades = int(strategy_impl.num_trades() or 0)
    if resumed and prior:
        trades += int(prior.get("num_trades") or 0)

    stored_decisions = llm_calls if llm_decisions is None else llm_decisions
    if resumed and prior and llm_decisions is not None:
        stored_decisions = int(prior.get("llm_decisions") or 0) + int(llm_decisions)

    db.insert_run(
        run_id=run_id,
        session_id=session_id,
        agent_name=entry["name"],
        mode=lb_service.LEADERBOARD_MODE,
        start_date=month_start,
        end_date=freeze_end,
        initial_equity=metrics["initial_equity"],
        final_equity=metrics["final_equity"],
        total_return=metrics["total_return"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        num_trades=trades,
        llm_model=entry_id,
        llm_calls=llm_calls,
        llm_decisions=stored_decisions,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        est_cost_usd=est_cost,
        metadata=lb_service._with_market_data_provenance(
            meta,
            lb_service.feed_provenance(bars),
        ),
    )
    db.insert_equity_points(run_id, stitched)

    return {
        "entry_id": entry_id,
        "run_id": run_id,
        "cached": False,
        "increment": resumed,
        "model": entry.get("model"),
        "model_id": model_id,
        "window": {"start_date": month_start, "end_date": freeze_end},
        "segment": {"start_date": segment_start, "end_date": segment_end},
        "total_return": metrics["total_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "final_equity": metrics["final_equity"],
        "num_trades": trades,
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "est_cost_usd": est_cost,
    }


def refresh_live_leaderboard(
    *,
    deploy_models: bool = False,
    force_refresh: bool = False,
    allow_fallback: bool = False,
    as_of: Optional[Union[date, datetime]] = None,
) -> Dict[str, Any]:
    """Persist this month's freeze-window runs into ``agent_runs``.

    Always refreshes cheap baselines/indices for month-open → last completed
    session. When ``deploy_models`` is True, each LLM entry continues from the
    latest snapshot and only trades sessions not yet stored (typically one
    cash day). Public GET never calls this.
    """
    freeze_cfg = live_freeze_config(as_of)
    if freeze_cfg is None:
        raise RuntimeError(
            "Live leaderboard has no completed cash session this month yet"
        )
    window_key = _live_window_key(freeze_cfg)
    prior = _live_refresh_state()
    if (
        not force_refresh
        and prior.get("window_key") == window_key
        and prior.get("baselines_refreshed")
        and (not deploy_models or prior.get("models_deployed"))
    ):
        return {
            **prior,
            "skipped": True,
            "window": {
                "start_date": freeze_cfg["start_date"],
                "end_date": freeze_cfg["end_date"],
                "label": f"{freeze_cfg['start_date']} → {freeze_cfg['end_date']}",
            },
        }

    baseline_meta = lb_service.ensure_leaderboard_runs(
        force_refresh=force_refresh, config=freeze_cfg
    )
    result: Dict[str, Any] = {
        "window_key": window_key,
        "window": {
            "start_date": freeze_cfg["start_date"],
            "end_date": freeze_cfg["end_date"],
            "label": f"{freeze_cfg['start_date']} → {freeze_cfg['end_date']}",
        },
        "period": "live",
        "session_id": LIVE_SESSION_ID,
        "baselines_refreshed": True,
        "baselines": baseline_meta,
        "models_deployed": False,
        "model_results": [],
        "model_failures": [],
        "refreshed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "skipped": False,
    }

    if deploy_models:
        failures: List[Dict[str, str]] = []
        successes: List[Dict[str, Any]] = []
        for entry in live_llm_entries(freeze_cfg):
            entry_id = entry["id"]
            try:
                row = deploy_live_model_increment(
                    entry,
                    freeze_cfg,
                    force_refresh=force_refresh,
                    allow_fallback=allow_fallback,
                )
                successes.append(row)
            except (lb_service.LeaderboardFallbackError, ValueError, RuntimeError) as exc:
                failures.append({"entry_id": entry_id, "error": str(exc)})
        result["models_deployed"] = not failures
        result["model_results"] = successes
        result["model_failures"] = failures

    _save_live_refresh_state(result)
    return result


def _run_live_refresh_background(
    *,
    deploy_models: bool,
    force_refresh: bool,
) -> None:
    try:
        refresh_live_leaderboard(
            deploy_models=deploy_models,
            force_refresh=force_refresh,
        )
    except Exception as exc:
        print(f"⚠️ Live leaderboard background refresh failed: {exc}")
    finally:
        with _live_refresh_lock:
            _set_live_refresh_running(False)


def maybe_schedule_live_leaderboard_refresh(
    *,
    deploy_models: bool = True,
    force_refresh: bool = False,
) -> bool:
    """Start a background live refresh if one is not already running."""
    freeze_cfg = live_freeze_config()
    if freeze_cfg is None:
        raise RuntimeError(
            "Live leaderboard has no completed cash session this month yet"
        )
    if not force_refresh and not deploy_models:
        prior = _live_refresh_state()
        if prior.get("window_key") == _live_window_key(freeze_cfg) and prior.get(
            "baselines_refreshed"
        ):
            return False

    with _live_refresh_lock:
        global _live_refresh_running
        if _live_refresh_running:
            return False
        _set_live_refresh_running(True)
        thread = threading.Thread(
            target=_run_live_refresh_background,
            kwargs={
                "deploy_models": deploy_models,
                "force_refresh": force_refresh,
            },
            name="live-leaderboard-refresh",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            _set_live_refresh_running(False)
            raise
        return True


def enqueue_live_leaderboard_refresh(
    *,
    deploy_models: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Cron/API entrypoint: accept a live refresh and run it in a background thread.

    Never blocks on model deploys. GET never calls this. No ``allow_fallback``.
    """
    freeze_cfg = live_freeze_config()
    if freeze_cfg is None:
        raise RuntimeError(
            "Live leaderboard has no completed cash session this month yet"
        )
    started = maybe_schedule_live_leaderboard_refresh(
        deploy_models=deploy_models,
        force_refresh=force_refresh,
    )
    in_progress = started or _live_refresh_running
    return {
        "accepted": True,
        "started": started,
        "refresh_in_progress": in_progress,
        "period": "live",
        "window": {
            "start_date": freeze_cfg["start_date"],
            "end_date": freeze_cfg["end_date"],
            "label": f"{freeze_cfg['start_date']} → {freeze_cfg['end_date']}",
        },
        "message": (
            "Live leaderboard refresh started in the background."
            if started
            else (
                "Live leaderboard refresh already in progress."
                if in_progress
                else "No new live refresh scheduled (window already satisfied)."
            )
        ),
    }


def _parse_to_et(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
    else:
        raw = str(ts).replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_US_EASTERN)


def _hour_key(dt: datetime) -> str:
    hour = dt.hour
    if hour == 9 and dt.minute >= 30:
        hour = 10
    return f"{dt.date().isoformat()}T{hour:02d}:00"


def reindex_frozen_curve(
    hourly_points: List[Dict[str, Any]],
    axis: List[str],
    freeze_end: str,
    initial_capital: float,
) -> List[Dict[str, Any]]:
    """Map a freeze-window hourly curve onto the calendar-month axis.

    Points after the freeze close are omitted (frontend leaves those axis
    nodes null). Missing hours inside the freeze as-of fill from the last
    print so the line is continuous across sparse bars, not into the future.
    """
    by_hour: Dict[str, float] = {}
    for pt in hourly_points:
        dt = _parse_to_et(pt.get("timestamp"))
        if dt is None:
            continue
        by_hour[_hour_key(dt)] = float(pt.get("equity") or 0)

    freeze_dt = datetime.combine(
        date.fromisoformat(freeze_end), _US_CASH_CLOSE, tzinfo=_US_EASTERN
    )
    last: Optional[float] = None
    out: List[Dict[str, Any]] = []
    for ts in axis:
        ts_dt = _parse_axis_ts(ts)
        if ts_dt > freeze_dt:
            break
        key = f"{ts_dt.date().isoformat()}T{ts_dt.hour:02d}:00"
        if key in by_hour:
            last = by_hour[key]
        elif last is None:
            last = float(initial_capital)
        out.append({"timestamp": ts, "equity": last})
    return out


def _entry_from_strategy(
    strategy: Dict[str, Any],
    *,
    display_capital: float,
    curve: List[Dict[str, Any]],
    run: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    is_model = strategy.get("strategy") == "llm_agent" or strategy.get("label") == "Model"
    printed = [p for p in curve if p.get("equity") is not None]
    if printed:
        last_eq = float(printed[-1]["equity"])
        metrics = calc_metrics(printed, display_capital)
        portfolio_value = last_eq
        total_return = metrics["total_return"]
        sharpe = metrics["sharpe_ratio"]
        max_dd = metrics["max_drawdown"]
    elif run:
        scale = 1.0
        stored = float(run.get("initial_equity") or display_capital)
        if stored:
            scale = display_capital / stored
        final = run.get("final_equity")
        portfolio_value = float(final) * scale if final is not None else display_capital
        total_return = run.get("total_return") or 0.0
        sharpe = run.get("sharpe_ratio") or 0.0
        max_dd = run.get("max_drawdown") or 0.0
    else:
        portfolio_value = display_capital
        total_return = 0.0
        sharpe = 0.0
        max_dd = 0.0
    return {
        "entry_id": strategy["id"],
        "team_name": strategy.get("name") or "Agentic Trading Lab",
        "team_badge": strategy.get("label", "Baseline Strategy"),
        "model": strategy.get("model", "Baseline"),
        "entry_type": "baseline",
        "is_model": is_model,
        "initial_equity": display_capital,
        "portfolio_value": portfolio_value,
        "cumulative_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "status": "frozen" if printed else "pending",
        "run_id": run.get("run_id") if run else None,
        "llm_calls": (run or {}).get("llm_calls") or 0,
        "input_tokens": (run or {}).get("input_tokens") or 0,
        "output_tokens": (run or {}).get("output_tokens") or 0,
        "est_cost_usd": (run or {}).get("est_cost_usd") or 0,
        "equity_curve": curve,
    }


def get_live_leaderboard(
    *,
    as_of: Optional[Union[date, datetime]] = None,
) -> Dict[str, Any]:
    """Calendar-month board: freeze-window curves on a full-month axis.

    Auto-compute baselines/indices run for month-open → last completed session
    and cache under ``leaderboard-live``. LLM entries stay pending until
    ``refresh_live_leaderboard(deploy_models=True)`` (this path never calls the
    models). Points after the stored snapshot — and after the freeze close —
    stay off the series so a stale cache cannot paint the next session.
    """
    now = _coerce_as_of_eastern(as_of)
    start_date, end_date = live_month_dates(now)
    clock = live_clock(now)
    axis = live_month_hourly_axis(start_date, end_date)
    printed_through = printed_through_timestamp(axis, now)
    month_start = date.fromisoformat(start_date)
    month_end = date.fromisoformat(end_date)
    frozen_day = date.fromisoformat(clock["frozen_through"])
    trading_days = _weekdays_inclusive(month_start, month_end)
    elapsed_end = min(frozen_day, month_end)
    elapsed = _weekdays_inclusive(month_start, elapsed_end) if elapsed_end >= month_start else []

    config = load_leaderboard_config()
    strategies = live_board_strategies(config)
    display_capital = float(config.get("initial_capital", INITIAL_CAPITAL))
    freeze_cfg = live_freeze_config(now)
    freeze_error: Optional[str] = None
    if freeze_cfg is not None:
        try:
            lb_service.ensure_leaderboard_runs(config=freeze_cfg)
        except Exception:
            print("⚠️ Live freeze refresh failed; serving any cached live-month runs")
            freeze_error = "freeze_unavailable"

    entries: List[Dict[str, Any]] = []
    freeze_start = freeze_cfg["start_date"] if freeze_cfg else start_date
    freeze_end = freeze_cfg["end_date"] if freeze_cfg else clock["frozen_through"]
    session_id = LIVE_SESSION_ID
    snapshot_ends: List[str] = []

    for strategy in strategies:
        run = None
        if freeze_cfg is not None:
            run = lb_service._find_live_month_run(
                strategy["id"], freeze_start, freeze_end, session_id
            )
        hourly: List[Dict[str, Any]] = []
        curve_end = freeze_end
        if run:
            snapshot_ends.append(str(run.get("end_date") or freeze_end))
            curve_end = _clip_end(str(run.get("end_date") or freeze_end), freeze_end)
            hourly = db.get_equity_curve(run["run_id"]) or []
            stored = float(run.get("initial_equity") or display_capital)
            scale = (display_capital / stored) if stored else 1.0
            if scale != 1.0:
                hourly = [
                    {**pt, "equity": float(pt.get("equity") or 0) * scale}
                    for pt in hourly
                ]
        curve = reindex_frozen_curve(
            hourly, axis, curve_end, display_capital
        ) if hourly else []
        entries.append(
            _entry_from_strategy(
                strategy,
                display_capital=display_capital,
                curve=curve,
                run=run,
            )
        )

    entries = _rank_entries(entries)
    printed_entries = [e for e in entries if e.get("equity_curve")]
    models_with_prints = [e for e in printed_entries if e.get("is_model")]
    if models_with_prints:
        leader = models_with_prints[0].get("model") or models_with_prints[0].get("team_name") or "—"
    elif printed_entries:
        top = max(printed_entries, key=lambda e: e.get("portfolio_value") or 0)
        leader = top.get("model") or top.get("team_name") or "—"
    else:
        leader = "—"

    printed_count = max((len(e.get("equity_curve") or []) for e in entries), default=0)
    models_cached = sum(1 for e in entries if e.get("is_model") and e.get("equity_curve"))
    models_total = sum(1 for e in entries if e.get("is_model"))
    month_label = now.strftime("%B %Y")
    live_status = {
        "phase": LIVE_PHASE,
        "month": f"{now.year:04d}-{now.month:02d}",
        "session_id": LIVE_SESSION_ID,
        "as_of": clock["as_of"],
        "session_state": clock["session_state"],
        "frozen_through": clock["frozen_through"],
        "live_day": clock["live_day"],
        "printed_through": printed_through,
        "now_index": axis.index(printed_through) if printed_through in axis else None,
        "next_tick": _next_tick(axis, printed_through),
        "trading_days_elapsed": len(elapsed),
        "trading_days_total": len(trading_days),
        "axis_count": len(axis),
        "printed_count": printed_count,
        "models_cached": models_cached,
        "models_pending": max(models_total - models_cached, 0),
        "roster": list(LIVE_MODEL_IDS),
        "snapshot_end": max(snapshot_ends) if snapshot_ends else None,
        "has_prints": bool(printed_entries),
        "freeze_start": freeze_start,
        "freeze_end": freeze_end,
        "freeze_error": freeze_error,
    }

    return {
        "period": "live",
        "board_title": "Live Trading Leaderboard",
        "phase_label": "Season 0",
        "standings_label": "Ranking",
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "label": f"{start_date} — {end_date}",
            "description": (
                f"Live month {month_label}. Axis is the calendar month; hourly "
                "nodes are US cash-session hours (10:00–16:00 America/New_York). "
                f"Frozen history is the hourly backtest through {clock['frozen_through']}. "
                "Each completed freeze is stored as a monthly snapshot in agent_runs "
                "(session leaderboard-live); public GET never calls the models."
            ),
        },
        "chart_axis": axis,
        "updated_at": now.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
        "total_entries": len(entries),
        "display_capital": display_capital,
        "leader": leader,
        "entries": entries,
        "live_status": live_status,
    }
