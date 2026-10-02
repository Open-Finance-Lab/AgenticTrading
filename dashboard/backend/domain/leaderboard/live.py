"""Calendar-month Live Trading Leaderboard.

This board is not Daily stretched to 30 days and not the fixed contest window.
One continuous paper-trading month per entry: the chart axis is the current
America/New_York calendar month, hourly nodes are the closes of the US cash
session's hourly bars (10:00–16:00 ET, NYSE trading days), and points after
the freeze close stay empty so the line cannot interpolate into the future.

Every curve is written by ``refresh_live_leaderboard`` (the cron hook or the
refresh script), never on a public GET: GET only reads the latest stored
freeze row per entry. Baselines/indices are recomputed for month-open → last
settled session on every refresh; LLM models only when ``deploy_models=True``,
and each freeze appends one cash session onto the prior snapshot (cash +
positions), so a month costs about one full backtest, not a replay from the
1st every night.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

from dashboard.backend.database import db
from dashboard.backend.domain.leaderboard.baselines import INITIAL_CAPITAL, calc_metrics
from dashboard.backend.domain.leaderboard.strategies._common import reference_start_date
from dashboard.backend.domain.leaderboard.us_market_calendar import is_trading_day
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    allow_recent_sip,
    sip_delay_minutes,
)
from dashboard.backend.infrastructure.market_data.sessions import session_windows
from dashboard.backend.paths import DATA_DIR
import dashboard.backend.domain.leaderboard.service as lb_service

_US_EASTERN = ZoneInfo("America/New_York")
# The US cash session has one owner (market_data/sessions.py); restating it here
# is how the board's copies drifted apart before #529.
_US_CASH_OPEN, _US_CASH_CLOSE = session_windows("US")[0]
# Axis nodes are hourly bar *closes*. Since #529 the board's Alpaca bars are
# open-stamped (09:00 … 15:00, each closing an hour later) and Yahoo's index
# bars are open-stamped on the half hour (09:30 … 15:30), so a stored point is
# placed at the hour its bar closed in (see ``_axis_node_key``): seven nodes a
# day, the last one holding the 16:00 close.
_RTH_HOURS = (10, 11, 12, 13, 14, 15, 16)
_BAR_MINUTES = 60
# A session joins the freeze only once its closing bar is final on the tape the
# board is priced off. A Basic plan clamps a SIP request to now − delay, so a
# refresh at 16:02 would fetch a truncated 15:00 bar and cache that curve as
# the day's close. The margin covers late prints and the cron's own jitter.
_FREEZE_SETTLE_MARGIN_MINUTES = 15

LIVE_SESSION_ID = "leaderboard-live"
LIVE_PHASE = 1
# Season-0 local roster: only these LLM curves are deployed and shown on Live.
# Contest/daily still use the full leaderboard.json list. GPT-5.5 is left out
# because it was most of the nightly token spend.
#
# A dropped model's rows are deliberately left in place, not pruned: GET only
# renders roster entries, so they are inert, and if the model comes back
# mid-month it resumes from its last snapshot and trades only the sessions it
# missed. Deleting them would make a re-add replay the month from the 1st.
LIVE_MODEL_IDS = (
    "deepseek_v4_pro",
    "nemotron_3_nano_30b",
)
LIVE_SNAPSHOT_KEY = "live_portfolio_snapshot"
# Present (True) on a checkpoint row only while it is being written. The row
# goes in first, then its curve, then the row again with the snapshot and
# without this flag; a writer killed in between leaves a flagged row that no
# reader treats as a freeze, instead of a row that has a snapshot and no curve.
LIVE_CHECKPOINT_PENDING_KEY = "live_checkpoint_pending"
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


def _trading_days_inclusive(start: date, end: date) -> List[date]:
    if end < start:
        return []
    days: List[date] = []
    cursor = start
    while cursor <= end:
        if is_trading_day(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def live_month_hourly_axis(start_date: str, end_date: str) -> List[str]:
    """Hourly bar-close ISO timestamps covering the month's NYSE trading days."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    out: List[str] = []
    for day in _trading_days_inclusive(start, end):
        for hour in _RTH_HOURS:
            ts = datetime(day.year, day.month, day.day, hour, 0, tzinfo=_US_EASTERN)
            out.append(ts.isoformat())
    return out


def _previous_trading_day(day: date) -> date:
    cursor = day - timedelta(days=1)
    while not is_trading_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def next_session_day(day: date) -> date:
    """Next NYSE trading day after ``day`` (skips weekends and holidays)."""
    cursor = day + timedelta(days=1)
    while not is_trading_day(cursor):
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


def freeze_settle_time(day: date) -> datetime:
    """When ``day``'s closing bar is final on the board's tape."""
    delay = 0 if allow_recent_sip() else sip_delay_minutes()
    close = datetime.combine(day, _US_CASH_CLOSE, tzinfo=_US_EASTERN)
    return close + timedelta(minutes=delay + _FREEZE_SETTLE_MARGIN_MINUTES)


def live_clock(as_of: Optional[Union[date, datetime]] = None) -> Dict[str, Any]:
    """Session state for the live month at ``as_of`` (America/New_York).

    - ``weekend`` / ``holiday``: the last trading day is frozen; no session.
    - ``preopen``: the previous trading day is frozen; today has not opened.
    - ``rth``: the previous trading day is frozen; today's session is in
      progress. Nothing prints intraday — today appends once it settles.
    - ``settling``: the cash session has closed but its closing bar is not
      yet final on the tape (``freeze_settle_time``); still not frozen.
    - ``closed``: today's cash session is settled and joins the freeze.
    """
    now = _coerce_as_of_eastern(as_of)
    today = now.date()

    if not is_trading_day(today):
        session_state = "weekend" if today.weekday() >= 5 else "holiday"
        frozen_through = _previous_trading_day(today)
        live_day = None
    elif now.time() < _US_CASH_OPEN:
        session_state = "preopen"
        frozen_through = _previous_trading_day(today)
        live_day = None
    elif now.time() < _US_CASH_CLOSE:
        session_state = "rth"
        frozen_through = _previous_trading_day(today)
        live_day = today
    elif now < freeze_settle_time(today):
        session_state = "settling"
        frozen_through = _previous_trading_day(today)
        live_day = today
    else:
        session_state = "closed"
        frozen_through = today
        live_day = None

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
    base = lb_service.load_leaderboard_config()
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
    cfg = config or lb_service.load_leaderboard_config()
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
    """Delete every ``leaderboard-live`` run (curves, trades, decisions).

    Also forgets which window the last refresh satisfied: that record describes
    rows that no longer exist, and left in place it makes the very next
    refresh report "already refreshed" and skip, leaving the board empty.
    """
    runs = db.get_runs_by_session(LIVE_SESSION_ID) or []
    for run in runs:
        run_id = run.get("run_id")
        if run_id:
            db.delete_run(run_id)
    _LIVE_REFRESH_STATE_PATH.unlink(missing_ok=True)
    return len(runs)


def _is_month_row(run: Dict[str, Any], month_start: str, freeze_end: str) -> bool:
    end = str(run.get("end_date") or "")
    return bool(
        run.get("llm_model")
        and run.get("mode") == lb_service.LEADERBOARD_MODE
        and run.get("start_date") == month_start
        and end
        and end <= freeze_end
    )


def _is_pending_checkpoint(run: Dict[str, Any]) -> bool:
    return bool(_run_metadata_dict(run).get(LIVE_CHECKPOINT_PENDING_KEY))


def latest_live_month_runs(
    month_start: str,
    freeze_end: str,
    *,
    runs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Latest freeze row per entry for this month that ends by ``freeze_end``.

    Each freeze close writes its own ``agent_runs`` row (``start_date`` is
    month-open, ``end_date`` is that freeze), so GET keeps serving yesterday's
    snapshot after the clock rolls, until the next refresh lands. One session
    scan serves the whole board; pass ``runs`` to reuse a scan already made.
    A checkpoint still being written is never the latest freeze.
    """
    if runs is None:
        runs = db.get_runs_by_session(LIVE_SESSION_ID) or []
    best: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        if not _is_month_row(run, month_start, freeze_end) or _is_pending_checkpoint(run):
            continue
        entry_id = run["llm_model"]
        end = str(run["end_date"])
        current = best.get(entry_id)
        if current is None or end > str(current.get("end_date") or ""):
            best[entry_id] = run
    return best


def _entry_month_runs(
    entry_id: str,
    month_start: str,
    freeze_end: str,
) -> List[Dict[str, Any]]:
    """Every row this entry has for the month, pending ones included, newest first."""
    runs = [
        run for run in db.get_runs_by_session(LIVE_SESSION_ID) or []
        if run.get("llm_model") == entry_id and _is_month_row(run, month_start, freeze_end)
    ]
    return sorted(runs, key=lambda run: str(run["end_date"]), reverse=True)


def prune_superseded_live_runs(month_start: str, freeze_end: str) -> int:
    """Delete this month's freeze rows that a later freeze row has replaced.

    Every row stores the whole month-to-date curve, so keeping one per day
    grows storage with the square of the day of the month while nothing but
    the latest row per entry is ever read again. The row kept is the latest
    *complete* one: a checkpoint still flagged pending at the end of a refresh
    is a write that was cut short, and goes too.
    """
    runs = db.get_runs_by_session(LIVE_SESSION_ID) or []
    keep = {
        run.get("run_id")
        for run in latest_live_month_runs(month_start, freeze_end, runs=runs).values()
    }
    deleted = 0
    for run in runs:
        run_id = run.get("run_id")
        if (
            run_id
            and run_id not in keep
            and run.get("start_date") == month_start
            and str(run.get("end_date") or "") <= freeze_end
        ):
            db.delete_run(run_id)
            deleted += 1
    return deleted


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
        # Best-effort cleanup of our own temp file; the original error is the
        # one worth raising.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
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


class LiveSessionGapError(RuntimeError):
    """A trading day inside the segment came back from the tape with no bars."""


def _session_gap(
    entry_id: str,
    missing: List[date],
    segment_start: str,
    segment_end: str,
) -> LiveSessionGapError:
    """Report a trading day with no bars, loudly, and build the error to raise.

    Days are drawn from the NYSE calendar, so a session with no bars is broken
    data, never an absent one: the feed failed, or the calendar is missing an
    unscheduled closure. Skipping it would carry the book across a session it
    never traded and mark the window deployed. Instead nothing from the gap on
    is stored (a gap is only seen at the next session's close, so that session
    is dropped too), the refresh reports the entry failed, and the next one
    resumes from the gap.
    """
    days = ", ".join(day.isoformat() for day in missing)
    print(
        f"ERROR: live.session_gap entry={entry_id} missing={days} "
        f"segment={segment_start}..{segment_end}",
        flush=True,
    )
    return LiveSessionGapError(
        f"Live '{entry_id}': no bars for trading day(s) {days} in "
        f"{segment_start} → {segment_end}; stored nothing from there on rather "
        f"than carry the book across a session it never traded"
    )


def _resume_point(
    rows: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Newest complete checkpoint among ``rows`` (newest first), with its curve.

    A row counts only if it carries a snapshot *and* a stored curve. One with a
    snapshot but no curve was cut short between its two writes (rows from
    before the pending flag existed have no other tell); resuming from it
    would stitch the rest of the month onto an empty curve, so it is dropped
    and the next row down is tried. Returns the dropped run ids as well.
    """
    dropped: List[str] = []
    for run in rows:
        if _is_pending_checkpoint(run) or _snapshot_from_run(run) is None:
            continue
        curve = db.get_equity_curve(run["run_id"]) or []
        if curve:
            return run, curve, dropped
        print(
            f"⚠️ Live {run.get('llm_model')}: {run['run_id']} has a snapshot but "
            f"no curve (its write was cut short); dropping it"
        )
        db.delete_run(run["run_id"])
        dropped.append(run["run_id"])
    return None, [], dropped


# The entry's own config keys, as ``_llm_run_metadata`` records them: what the
# model is asked, and how it is sampled. Every checkpoint re-records them, so
# resuming a month written under other values would stitch two experiments into
# one curve and label the whole month with the newer one.
_RESUME_CONFIG_KEYS = (
    "model_id",
    "integration",
    "temperature",
    "reasoning_effort",
    "strategy_prompt",
)


def _config_changes(
    resume_from: Dict[str, Any],
    entry: Dict[str, Any],
    strategy_impl: Any,
    *,
    initial_capital: float,
    month_start: str,
    freeze_end: str,
) -> List[str]:
    """Config keys whose value differs from the one ``resume_from`` was run under.

    The current side comes from ``_llm_run_metadata``, the function that wrote
    the recorded side, so the two cannot drift apart by construction. A key
    the checkpoint never recorded is unknown, not a change. So is a
    ``model_id`` the entry leaves to the gateway default, which only ``run()``
    resolves.
    """
    recorded = _run_metadata_dict(resume_from)
    current = lb_service._llm_run_metadata(
        entry["id"],
        entry,
        strategy_impl,
        model_id=getattr(strategy_impl, "model_id", None) or entry.get("model_id"),
        initial_capital=initial_capital,
        start_date=month_start,
        end_date=freeze_end,
    ) or {}
    return [
        key
        for key in _RESUME_CONFIG_KEYS
        if key in recorded
        and not (key == "model_id" and current.get(key) is None)
        and recorded[key] != current.get(key)
    ]


def _checkpoint_passes_h6(
    entry_id: str,
    entry: Dict[str, Any],
    strategy_impl: Any,
    *,
    allow_fallback: bool,
) -> bool:
    """Whether the segment traded so far clears H6; raises once it never can.

    H6 applies to the segment this refresh trades, as it did when a segment
    was one write. Checking each session on its own would put a catch-up's
    every day to the test over seven steps, where a single unusable reply is
    already below 95%. So a session whose running coverage falls short is just
    not checkpointed, and the run carries on. It stops only when even a perfect
    remainder could not bring the segment back over the threshold, so a dead
    model is not billed for the rest of a multi-day replay.
    """
    llm_calls = int(getattr(strategy_impl, "llm_calls", 0) or 0)
    decisions = lb_service._reported_int(strategy_impl, "llm_decisions")
    steps = int(getattr(strategy_impl, "decision_steps", 0) or 0)
    try:
        lb_service._reject_if_llm_fallback(
            entry_id,
            strategy_impl,
            llm_calls,
            llm_decisions=decisions,
            decision_steps=steps,
            model=entry.get("model"),
            model_id=getattr(strategy_impl, "model_id", None) or entry.get("model_id"),
            allow_fallback=allow_fallback,
        )
        return True
    except lb_service.LeaderboardFallbackError:
        if not getattr(strategy_impl, "used_llm", False) or llm_calls == 0:
            raise
        planned = int(getattr(strategy_impl, "planned_decision_steps", 0) or 0)
        misses = max(steps - (llm_calls if decisions is None else decisions), 0)
        if planned > steps and planned - misses >= (
            lb_service.MIN_LLM_DECISION_COVERAGE * planned
        ):
            return False
        raise


def _drop_superseded_live_rows(
    entry_id: str,
    month_start: str,
    freeze_end: str,
    *,
    keep: str,
    day_iso: str,
    supersede_later: bool,
) -> None:
    """Leave ``keep`` as the entry's checkpoint once it is complete.

    Every earlier row is a prefix of its curve, and a pending row is a write
    that never finished. Later rows go too when the segment replaces them (a
    forced replay): deleting them as the first checkpoint lands is what makes
    the replay durable, since a killed replay would otherwise leave the
    pre-force row as the newest and the next refresh would call it done.
    Pruning here rather than at the end of the refresh also keeps an aborted
    catch-up from leaving a row per day behind.
    """
    for run in _entry_month_runs(entry_id, month_start, freeze_end):
        run_id = run.get("run_id")
        if not run_id or run_id == keep:
            continue
        if (
            str(run["end_date"]) < day_iso
            or supersede_later
            or _is_pending_checkpoint(run)
        ):
            db.delete_run(run_id)


def _write_live_checkpoint(
    entry: Dict[str, Any],
    freeze_cfg: Dict[str, Any],
    strategy_impl: Any,
    *,
    day_iso: str,
    curve: List[Dict[str, Any]],
    base: Optional[Dict[str, Any]],
    lineage: Dict[str, Any],
    provenance: Optional[Dict[str, Any]],
    supersede_later: bool,
) -> Dict[str, Any]:
    """Store the month curve through ``day_iso`` as the entry's newest freeze.

    Three writes, ordered so that no reader ever sees a snapshot without its
    curve: the row flagged pending, the curve, then the row again carrying the
    snapshot. ``equity_timeseries`` has a live foreign key on Postgres, so the
    curve cannot go in before its row; the flag is what makes the gap safe.
    Counters are the resumed row's plus the run's so far (the run is one
    continuous segment, so its own counters are already cumulative).
    """
    entry_id = entry["id"]
    month_start = freeze_cfg["start_date"]
    initial_capital = float(freeze_cfg.get("initial_capital", INITIAL_CAPITAL))

    def carried(key: str) -> int:
        return int((base or {}).get(key) or 0)

    llm_calls = int(getattr(strategy_impl, "llm_calls", 0) or 0)
    decisions = lb_service._reported_int(strategy_impl, "llm_decisions")
    totals = {
        "input_tokens": carried("input_tokens")
        + int(getattr(strategy_impl, "input_tokens", 0) or 0),
        "output_tokens": carried("output_tokens")
        + int(getattr(strategy_impl, "output_tokens", 0) or 0),
        "llm_calls": carried("llm_calls") + llm_calls,
        "llm_decisions": carried("llm_decisions")
        + (llm_calls if decisions is None else decisions),
        "num_trades": carried("num_trades") + int(strategy_impl.num_trades() or 0),
    }
    model_id = getattr(strategy_impl, "model_id", None) or entry.get("model_id")
    est_cost = lb_service.token_cost.estimate_cost_usd(
        model_id, totals["input_tokens"], totals["output_tokens"]
    )
    metrics = calc_metrics(curve, initial_capital)
    run_id = lb_service._run_id(entry_id, month_start, day_iso)

    meta = lb_service._llm_run_metadata(
        entry_id,
        entry,
        strategy_impl,
        model_id=model_id,
        initial_capital=initial_capital,
        start_date=month_start,
        end_date=day_iso,
    ) or {}
    meta["live_increment"] = {**lineage, "segment_end": day_iso}
    meta = lb_service._with_market_data_provenance(meta, provenance) or {}

    row = {
        "run_id": run_id,
        "session_id": freeze_cfg["session_id"],
        "agent_name": entry["name"],
        "mode": lb_service.LEADERBOARD_MODE,
        "start_date": month_start,
        "end_date": day_iso,
        "initial_equity": metrics["initial_equity"],
        "final_equity": metrics["final_equity"],
        "total_return": metrics["total_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "num_trades": totals["num_trades"],
        "llm_model": entry_id,
        "llm_calls": totals["llm_calls"],
        "llm_decisions": totals["llm_decisions"],
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "est_cost_usd": est_cost,
    }
    db.insert_run(**row, metadata={**meta, LIVE_CHECKPOINT_PENDING_KEY: True})
    db.insert_equity_points(run_id, curve)
    db.insert_run(
        **row,
        metadata={
            **meta,
            LIVE_SNAPSHOT_KEY: getattr(strategy_impl, "last_portfolio_snapshot", None),
        },
    )
    _drop_superseded_live_rows(
        entry_id,
        month_start,
        freeze_cfg["end_date"],
        keep=run_id,
        day_iso=day_iso,
        supersede_later=supersede_later,
    )
    return {
        "run_id": run_id,
        "model_id": model_id,
        "end_date": day_iso,
        "metrics": metrics,
        "totals": totals,
        "est_cost_usd": est_cost,
    }


def deploy_live_model_increment(
    entry: Dict[str, Any],
    freeze_cfg: Dict[str, Any],
    *,
    force_refresh: bool = False,
    allow_fallback: bool = False,
    bars_memo: Optional[Dict[Tuple[Any, ...], Any]] = None,
) -> Dict[str, Any]:
    """Append unseen cash sessions onto a live-month LLM snapshot.

    Trades only ``live_increment_bounds`` (usually one day) and restores cash
    plus positions from the newest complete checkpoint. With none, or on
    ``force_refresh``, it replays from month-open. Either way the segment is
    one continuous run that checkpoints each session as it closes, so a
    catch-up killed part-way resumes from the last session it stored rather
    than from month-open, and nothing but the checkpoint writes depends on the
    day boundary (the book, prices and trade memory run straight through).
    Does not change contest/daily ``deploy_model_run``.

    ``bars_memo`` lets one refresh share a bar fetch across the roster: the
    on-disk bar cache refuses windows under 24 hours old, so without it every
    model re-downloads the same increment window.
    """
    entry_id = entry["id"]
    month_start = freeze_cfg["start_date"]
    freeze_end = freeze_cfg["end_date"]
    initial_capital = float(freeze_cfg.get("initial_capital", INITIAL_CAPITAL))

    rows = _entry_month_runs(entry_id, month_start, freeze_end)
    resume_from, prior_curve, dropped = _resume_point(rows)
    latest = next(
        (
            run for run in rows
            if not _is_pending_checkpoint(run) and run["run_id"] not in dropped
        ),
        None,
    )
    if latest and str(latest["end_date"]) == freeze_end and not force_refresh:
        return {
            "entry_id": entry_id,
            "run_id": latest.get("run_id"),
            "cached": True,
            "increment": False,
            "model": entry.get("model"),
            "window": {"start_date": month_start, "end_date": freeze_end},
            "segment": None,
            "total_return": latest.get("total_return"),
            "final_equity": latest.get("final_equity"),
            "llm_calls": latest.get("llm_calls"),
        }

    strategy_impl = lb_service.get_strategy(entry)
    config_changed: List[str] = []
    if resume_from is not None and not force_refresh:
        config_changed = _config_changes(
            resume_from,
            entry,
            strategy_impl,
            initial_capital=initial_capital,
            month_start=month_start,
            freeze_end=freeze_end,
        )
    if config_changed:
        # Replayed like a forced refresh, rows past the first checkpoint
        # included: a replay killed part-way must not leave an old-config row
        # as the newest one. If it does anyway, the next refresh finds the
        # same change and replays again.
        print(
            f"WARNING: live.config_changed entry={entry_id} "
            f"keys={','.join(config_changed)} checkpoint={resume_from.get('run_id')} "
            f"replaying={month_start}..{freeze_end}",
            flush=True,
        )
    replace_later = force_refresh or bool(config_changed)
    if replace_later:
        resume_from, prior_curve = None, []
    if resume_from is None:
        segment_start, segment_end = month_start, freeze_end
        snapshot = None
        if latest is not None and not replace_later:
            print(
                f"⚠️ Live {entry_id}: no resumable snapshot on this month's rows; "
                f"replaying {month_start} → {freeze_end}"
            )
    else:
        bounds = live_increment_bounds(
            str(resume_from.get("end_date") or ""),
            month_start=month_start,
            freeze_end=freeze_end,
        )
        if bounds is None:
            return {
                "entry_id": entry_id,
                "run_id": resume_from.get("run_id"),
                "cached": True,
                "increment": False,
                "model": entry.get("model"),
                "window": {"start_date": month_start, "end_date": freeze_end},
                "segment": None,
            }
        segment_start, segment_end = bounds
        snapshot = _snapshot_from_run(resume_from)
    resumed = resume_from is not None

    # The indicator lookback is relative to the segment being traded, not to
    # month-open. ``freeze_cfg`` pins ``reference_start_date`` to a month before
    # the 1st, so passing it here made a one-day increment on the 28th fetch
    # about two months of bars.
    bars_start = reference_start_date(segment_start, None)
    symbols = strategy_impl.required_symbols()
    memo_key = (tuple(sorted(symbols)), bars_start, segment_end)
    if bars_memo is not None and memo_key in bars_memo:
        bars = bars_memo[memo_key]
    else:
        bars = lb_service.fetch_hourly_bars(symbols, bars_start, segment_end)
        if bars_memo is not None and bars:
            bars_memo[memo_key] = bars
    if not bars:
        raise RuntimeError(
            f"No market data returned for live increment {bars_start} → {segment_end}"
        )
    print(
        f"  live increment {entry_id}: trade {segment_start} → {segment_end} "
        f"(month {month_start} → {freeze_end}, resume={resumed})"
    )

    expected = _trading_days_inclusive(
        date.fromisoformat(segment_start), date.fromisoformat(segment_end)
    )
    provenance = lb_service.feed_provenance(bars)
    # ``resumed_from_run_id`` names the row this segment started from, which
    # the first checkpoint supersedes and deletes; the end date is the part of
    # the lineage that stays readable after that.
    lineage = {
        "segment_start": segment_start,
        "full_replay": not resumed,
        "forced": bool(force_refresh),
        "config_changed": config_changed or None,
        "resumed_from_run_id": resume_from.get("run_id") if resume_from else None,
        "resumed_from_end_date": resume_from.get("end_date") if resume_from else None,
    }
    cursor = {"next": 0}
    written: List[Dict[str, Any]] = []

    def on_session_close(day: date, segment_curve: List[Dict[str, Any]]) -> None:
        missing: List[date] = []
        while cursor["next"] < len(expected) and expected[cursor["next"]] < day:
            missing.append(expected[cursor["next"]])
            cursor["next"] += 1
        if missing:
            raise _session_gap(entry_id, missing, segment_start, segment_end)
        if cursor["next"] < len(expected) and expected[cursor["next"]] == day:
            cursor["next"] += 1
        if not _checkpoint_passes_h6(
            entry_id, entry, strategy_impl, allow_fallback=allow_fallback
        ):
            print(
                f"  live {entry_id}: {day.isoformat()} not checkpointed; the "
                f"segment's model coverage is below the H6 threshold so far"
            )
            return
        written.append(
            _write_live_checkpoint(
                entry,
                freeze_cfg,
                strategy_impl,
                day_iso=day.isoformat(),
                curve=_stitch_equity_curves(prior_curve, segment_curve),
                base=resume_from,
                lineage=lineage,
                provenance=provenance,
                supersede_later=replace_later,
            )
        )

    strategy_impl.run(
        bars,
        segment_start,
        segment_end,
        initial_capital,
        starting_snapshot=snapshot,
        on_session_close=on_session_close,
    )

    # Sessions still expected after the run returned had no bars at all; this
    # also covers a segment with none (a run that returns before its first step
    # reports no model use, which H6 below would misname as a fallback).
    if cursor["next"] < len(expected):
        raise _session_gap(entry_id, expected[cursor["next"]:], segment_start, segment_end)
    # The segment as a whole must clear H6, as it did before it was split into
    # checkpoints. The last session close has already checked this; repeat it
    # here so a run that never reached that close cannot slip through.
    lb_service._reject_if_llm_fallback(
        entry_id,
        strategy_impl,
        int(getattr(strategy_impl, "llm_calls", 0) or 0),
        llm_decisions=lb_service._reported_int(strategy_impl, "llm_decisions"),
        decision_steps=int(getattr(strategy_impl, "decision_steps", 0) or 0),
        model=entry.get("model"),
        model_id=getattr(strategy_impl, "model_id", None) or entry.get("model_id"),
        allow_fallback=allow_fallback,
    )
    if not written or written[-1]["end_date"] != segment_end:
        raise RuntimeError(
            f"Live '{entry_id}': no checkpoint stored through {segment_end} for "
            f"{segment_start} → {segment_end}"
        )

    last = written[-1]
    metrics = last["metrics"]
    totals = last["totals"]
    return {
        "entry_id": entry_id,
        "run_id": last["run_id"],
        "cached": False,
        "increment": resumed,
        "model": entry.get("model"),
        "model_id": last["model_id"],
        "window": {"start_date": month_start, "end_date": last["end_date"]},
        "segment": {"start_date": segment_start, "end_date": segment_end},
        "total_return": metrics["total_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "final_equity": metrics["final_equity"],
        "num_trades": totals["num_trades"],
        "llm_calls": totals["llm_calls"],
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "est_cost_usd": last["est_cost_usd"],
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
        bars_memo: Dict[Tuple[Any, ...], Any] = {}
        for entry in live_llm_entries(freeze_cfg):
            entry_id = entry["id"]
            try:
                row = deploy_live_model_increment(
                    entry,
                    freeze_cfg,
                    force_refresh=force_refresh,
                    allow_fallback=allow_fallback,
                    bars_memo=bars_memo,
                )
                successes.append(row)
            except (lb_service.LeaderboardFallbackError, ValueError, RuntimeError) as exc:
                # The background refresh keeps this list only in a state file on
                # an ephemeral disk; the log line is the part an operator sees.
                print(
                    f"ERROR: live.model_deploy_failed entry={entry_id} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                failures.append({"entry_id": entry_id, "error": str(exc)})
        result["models_deployed"] = not failures
        result["model_results"] = successes
        result["model_failures"] = failures

    result["pruned_runs"] = prune_superseded_live_runs(
        freeze_cfg["start_date"], freeze_cfg["end_date"]
    )
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
    deploy_models: bool = False,
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
    deploy_models: bool = False,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Cron/API entrypoint: accept a live refresh and run it in a background thread.

    Never blocks on model deploys. GET never calls this. No ``allow_fallback``.
    ``deploy_models`` defaults off: it is the billable half (every Live LLM
    trades a session at the operator's API cost), so each caller opts in.
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


def _axis_node_key(bar_open: datetime) -> Optional[str]:
    """The axis node an open-stamped hourly point belongs to: its bar's close.

    Alpaca's 09:00 bar (09:30 open → 10:00) lands on 10:00 and its 15:00 bar on
    the 16:00 close; Yahoo's 09:30 bar closes 10:30 and lands on 10:00, and its
    15:30 half-bar is capped at the 16:00 close. A bar opening at or after the
    close is after-hours and has no node.
    """
    if bar_open.time() >= _US_CASH_CLOSE:
        return None
    close = bar_open + timedelta(minutes=_BAR_MINUTES)
    session_close = datetime.combine(close.date(), _US_CASH_CLOSE, tzinfo=_US_EASTERN)
    if close > session_close:
        close = session_close
    return f"{close.date().isoformat()}T{close.hour:02d}:00"


def reindex_frozen_curve(
    hourly_points: List[Dict[str, Any]],
    axis: List[str],
    freeze_end: str,
    initial_capital: float,
    scale: float = 1.0,
) -> List[Dict[str, Any]]:
    """Map a freeze-window hourly curve onto the calendar-month axis.

    Points after the freeze close are omitted (frontend leaves those axis
    nodes null). Missing hours inside the freeze as-of fill from the last
    print so the line is continuous across sparse bars, not into the future.
    A stored NULL equity is "no observation", never $0 (issue #390).
    """
    by_hour: Dict[str, float] = {}
    for pt in hourly_points:
        dt = _parse_to_et(pt.get("timestamp"))
        equity = pt.get("equity")
        if dt is None or equity is None:
            continue
        key = _axis_node_key(dt)
        if key is not None:
            by_hour[key] = float(equity) * scale

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


def _recorded_seed(run: Dict[str, Any]) -> Optional[float]:
    """The capital a live row was run at, when the row recorded it.

    ``agent_runs.initial_equity`` is not that number: ``calc_metrics`` stores
    the curve's *first mark*, which already carries the first hour's P&L, so
    rescaling by it erased that hour from every published figure.
    """
    value = _run_metadata_dict(run).get("initial_capital")
    try:
        seed = float(value)
    except (TypeError, ValueError):
        return None
    return seed if seed > 0 else None


def _entry_from_strategy(
    strategy: Dict[str, Any],
    *,
    display_capital: float,
    curve: List[Dict[str, Any]],
    run: Optional[Dict[str, Any]],
    scale: float = 1.0,
    catching_up: bool = False,
) -> Dict[str, Any]:
    """One board row. Returns and risk come off the stored run untouched.

    The stored run's metrics were computed against the capital it was seeded
    with, over exactly the curve this row plots, so only the dollar axis is
    scaled (the contest board's rule). A row with no run is *pending*: it has
    no value, return or rank — publishing the seed and 0% would rank an entry
    that never traded against ones that did. A row *catching up* is several
    sessions behind the rest of the board (a replay checkpointing its way
    through the month): it keeps its curve and figures, but is not ranked,
    because its value is from a different day than the rows it would sit by.
    """
    is_model = strategy.get("strategy") == "llm_agent" or strategy.get("label") == "Model"
    printed = bool(run) and any(p.get("equity") is not None for p in curve)
    if not printed:
        status = "pending"
    elif catching_up:
        status = "catching_up"
    else:
        status = "frozen"
    if printed:
        final = run.get("final_equity")
        portfolio_value = float(final) * scale if final is not None else None
        total_return = run.get("total_return")
        sharpe = run.get("sharpe_ratio")
        max_dd = run.get("max_drawdown")
    else:
        portfolio_value = None
        total_return = None
        sharpe = None
        max_dd = None
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
        "status": status,
        "rank": None,
        "run_id": run.get("run_id") if run else None,
        "snapshot_end": run.get("end_date") if run else None,
        "llm_calls": (run or {}).get("llm_calls") or 0,
        "input_tokens": (run or {}).get("input_tokens") or 0,
        "output_tokens": (run or {}).get("output_tokens") or 0,
        "est_cost_usd": (run or {}).get("est_cost_usd") or 0,
        "equity_curve": curve if printed else [],
    }


def get_live_leaderboard(
    *,
    as_of: Optional[Union[date, datetime]] = None,
) -> Dict[str, Any]:
    """Calendar-month board: freeze-window curves on a full-month axis.

    Read-only. This is a public, unauthenticated GET, and the freeze window
    moves every trading day, so computing here would miss the run cache on
    the first request of every day and fetch 30 symbols of bars plus the index
    series inside a request thread — once per concurrent request. Every row is
    written by ``refresh_live_leaderboard`` instead; this serves the latest
    stored freeze row per entry and leaves the rest pending. Points after the
    stored snapshot — and after the freeze close — stay off the series so a
    stale row cannot paint the next session.
    """
    now = _coerce_as_of_eastern(as_of)
    start_date, end_date = live_month_dates(now)
    clock = live_clock(now)
    axis = live_month_hourly_axis(start_date, end_date)
    printed_through = printed_through_timestamp(axis, now)
    month_start = date.fromisoformat(start_date)
    month_end = date.fromisoformat(end_date)
    frozen_day = date.fromisoformat(clock["frozen_through"])
    trading_days = _trading_days_inclusive(month_start, month_end)
    elapsed_end = min(frozen_day, month_end)
    elapsed = (
        _trading_days_inclusive(month_start, elapsed_end)
        if elapsed_end >= month_start
        else []
    )

    config = lb_service.load_leaderboard_config()
    strategies = live_board_strategies(config)
    display_capital = float(config.get("initial_capital", INITIAL_CAPITAL))
    freeze_cfg = live_freeze_config(now)
    freeze_start = freeze_cfg["start_date"] if freeze_cfg else start_date
    freeze_end = freeze_cfg["end_date"] if freeze_cfg else clock["frozen_through"]
    runs_by_entry = (
        latest_live_month_runs(freeze_start, freeze_end) if freeze_cfg else {}
    )

    ranked: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    snapshot_ends: List[str] = []
    catching_up: List[str] = []
    # One session behind the newest row is the nightly model append still in
    # flight (baselines land first) and stays ranked, as it always has. More
    # than that is a replay part-way through the month. Measured against the
    # newest row rather than the clock, so a board the cron missed for days
    # is stale as a whole (``snapshot_stale``) but still ranks like with like.
    board_ends = [
        str(runs_by_entry[s["id"]].get("end_date") or "")
        for s in strategies
        if s["id"] in runs_by_entry
    ]
    newest_end = max((end for end in board_ends if end), default=None)
    catch_up_floor = (
        _previous_trading_day(date.fromisoformat(newest_end)).isoformat()
        if newest_end
        else None
    )

    for strategy in strategies:
        run = runs_by_entry.get(strategy["id"])
        curve: List[Dict[str, Any]] = []
        scale = 1.0
        behind = False
        if run:
            behind = bool(
                catch_up_floor and str(run.get("end_date") or "") < catch_up_floor
            )
            snapshot_ends.append(str(run.get("end_date") or freeze_end))
            curve_end = _clip_end(str(run.get("end_date") or freeze_end), freeze_end)
            seed = _recorded_seed(run)
            if seed is not None:
                scale = display_capital / seed
            hourly = db.get_equity_curve(run["run_id"]) or []
            if hourly:
                curve = reindex_frozen_curve(
                    hourly, axis, curve_end, display_capital, scale=scale
                )
        entry = _entry_from_strategy(
            strategy,
            display_capital=display_capital,
            curve=curve,
            run=run,
            scale=scale,
            catching_up=behind,
        )
        if entry["status"] == "catching_up":
            catching_up.append(entry["entry_id"])
        (ranked if entry["status"] == "frozen" else pending).append(entry)

    entries = lb_service._rank_entries(ranked) + pending
    models_with_prints = [e for e in ranked if e.get("is_model")]
    if models_with_prints:
        leader = models_with_prints[0].get("model") or models_with_prints[0].get("team_name") or "—"
    elif ranked:
        leader = ranked[0].get("model") or ranked[0].get("team_name") or "—"
    else:
        leader = "—"

    printed_count = max((len(e.get("equity_curve") or []) for e in entries), default=0)
    models_cached = sum(1 for e in ranked if e.get("is_model"))
    models_total = sum(1 for e in entries if e.get("is_model"))
    snapshot_end = max(snapshot_ends) if snapshot_ends else None
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
        "snapshot_end": snapshot_end,
        # Entries shown but not ranked: their newest row is more than one
        # session behind the board's, so their value is from another day.
        "catching_up": catching_up,
        # The newest stored freeze is behind the clock's: tonight's refresh
        # has not landed (or failed). GET never fills that gap itself.
        "snapshot_stale": bool(
            freeze_cfg is not None and (snapshot_end is None or snapshot_end < freeze_end)
        ),
        "has_prints": bool(ranked),
        "freeze_start": freeze_start,
        "freeze_end": freeze_end,
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
                f"Live month {month_label}. Axis is the calendar month's NYSE "
                "trading days; each node is the close of an hourly US cash-session "
                "bar (10:00–16:00 America/New_York). "
                f"Frozen history is the hourly backtest through {clock['frozen_through']}; "
                "a session is appended once, after it closes and settles. "
                "Each freeze is stored as a monthly snapshot in agent_runs "
                "(session leaderboard-live); public GET only reads it."
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


lb_service.register_period_board("live", get_live_leaderboard)
