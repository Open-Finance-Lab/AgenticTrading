"""Leaderboard contest: baseline strategies on a fixed backtest window.

Canonical location (Phase 3C3). Moved from
``dashboard/backend/services/leaderboard_service.py``; the original module was
removed in Phase 4A. Public functions, ranking behavior, filtering,
ordering, metrics, result schemas, constants, and database behavior are
unchanged; only the module location and the leaderboard-domain import paths
moved.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import tempfile
import threading
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import dashboard.backend.infrastructure.llm.token_cost as token_cost
from dashboard.backend.database import db
import dashboard.backend.domain.leaderboard.baselines as _baselines
from dashboard.backend.domain.leaderboard.strategies._common import reference_start_date
from dashboard.backend.domain.leaderboard.strategies import get_strategy
from dashboard.backend.infrastructure.llm import backtest_harness as llm_harness
from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    MarketDataUnavailableError,
    configured_feed_name,
    feed_provenance,
)
from dashboard.backend.paths import CONFIG_DIR, DATA_DIR

# Bound by assignment from the module alias rather than a bare
# `from ... import X`. Five of these are used below, but `downsample_daily` is a
# pure re-export: test_service_move asserts `service.downsample_daily is
# baselines.downsample_daily`, a cross-module contract `py/unused-import`
# (intra-file only) cannot see. One import form for the module, so this does not
# trade that alert for `py/import-and-import-from`. Kept below the import block
# rather than inside it so the imports stay one contiguous section.
INITIAL_CAPITAL = _baselines.INITIAL_CAPITAL
align_equity_curves = _baselines.align_equity_curves
calc_metrics = _baselines.calc_metrics
chart_equity_curve = _baselines.chart_equity_curve
downsample_daily = _baselines.downsample_daily
fetch_hourly_bars = _baselines.fetch_hourly_bars

LEADERBOARD_MODE = "leaderboard"
VALID_PERIODS = ("contest", "daily", "live")
_SKIP_CACHE_PATH = DATA_DIR / "leaderboard_skip_cache.json"
_DAILY_REFRESH_STATE_PATH = DATA_DIR / "leaderboard_daily_refresh.json"
_daily_refresh_lock = threading.Lock()
_daily_refresh_running = False
# (configured_feed, stale_feeds) pairs already reported — see _warn_on_feed_drift.
_warned_feed_drift: set[Tuple[str, Tuple[str, ...]]] = set()
# Two seeds are "the same seed" within this many dollars. A stored
# ``initial_equity`` has been through a backtest and a JSON round-trip — the
# committed board carries 100000.00000000003 for one entry — so `==` here would
# report a mismatch nobody made.
_SEED_MATCH_TOLERANCE = 0.01
# (entry_id, stored_seed, display_capital) triples already reported — see
# _warn_on_seed_mismatch.
_warned_seed_mismatch: set[Tuple[str, float, float]] = set()
# (configured_capital, stale_capitals) pairs already reported — see
# _warn_on_capital_drift.
_warned_capital_drift: set[Tuple[float, Tuple[float, ...]]] = set()
# (entry_id, run_id) pairs already reported — see _warn_on_prompt_drift.
_warned_prompt_drift: set[Tuple[str, str]] = set()
# (entry_id, run_id, rendered_condition) triples already reported — see
# _report_curve_integrity.
_warned_curve_integrity: set[Tuple[str, str, str]] = set()
# How a cached row compares to what the board is asking for. STALE is the only
# value that refuses a row, and it is deliberately the only one a *recorded*
# disagreement can produce — see _cache_match_rank.
_CACHE_MATCH = 0
_CACHE_UNRECORDED = 1
_CACHE_STALE = 2

# Daily board window is the last *completed* US cash session, not UTC-yesterday.
_US_EASTERN = ZoneInfo("America/New_York")
_US_CASH_CLOSE = time(16, 0)  # 16:00 America/New_York regular-session close

# H6 integrity threshold: an LLM entry must have decided at least this fraction
# of its steps with the model itself. Below it, the curve is mostly a rule-based
# fallback and publishing it would misrepresent that model's result. 0.95 leaves
# a small margin for transient API blips on a genuine run (e.g. 159/161) while
# still rejecting partial-fallback curves (e.g. the 1/161 run that topped the
# board). Override per-deploy with allow_fallback=True / --allow-fallback.
MIN_LLM_DECISION_COVERAGE = 0.95


def _auto_compute(strategy: Dict[str, Any]) -> bool:
    """Whether a strategy is cheap enough to compute on-demand during a web request.

    LLM-backed entries make real API calls, so they default to manual deploy
    (precomputed by scripts/deploy_leaderboard_model.py) and are flagged with
    ``"auto_compute": false`` in the config.
    """
    return bool(strategy.get("auto_compute", True))


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_leaderboard_config() -> Dict[str, Any]:
    path = CONFIG_DIR / "leaderboard.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_period(period: Optional[str]) -> str:
    value = (period or "contest").strip().lower()
    return value if value in VALID_PERIODS else "contest"


def _coerce_as_of_eastern(as_of: Optional[Union[date, datetime]]) -> datetime:
    """Normalize ``as_of`` to a timezone-aware America/New_York datetime.

    - ``None`` → now in Eastern.
    - aware ``datetime`` → converted to Eastern.
    - naive ``datetime`` → interpreted as already Eastern.
    - bare ``date`` → that Eastern calendar day at the cash close (16:00), so a
      weekday date means "that session is eligible" without callers inventing a
      clock time.
    """
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


def daily_window_dates(as_of: Optional[Union[date, datetime]] = None) -> Tuple[str, str]:
    """Most recently completed US cash equity session (Mon–Fri).

    Anchored on America/New_York so the board flips to **today** once the
    regular session has closed (16:00 ET). Before the close on a weekday — and
    all weekend — the window rolls back to the previous weekday. That is what
    makes a post-close cron (e.g. 22:30 UTC ≈ 18:30 EDT) refresh the session
    that just finished, instead of UTC-yesterday / last Friday.

    Market holidays are not special-cased: an exchange holiday still selects
    that weekday after 16:00 ET (bars may be empty/sparse).
    """
    now_et = _coerce_as_of_eastern(as_of)
    d = now_et.date()
    # Wall-clock compare in Eastern (DST already applied by ZoneInfo).
    session_complete = now_et.weekday() < 5 and now_et.time() >= _US_CASH_CLOSE
    if not session_complete:
        d -= timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun → Friday
        d -= timedelta(days=1)
    iso = d.isoformat()
    return iso, iso


def daily_window_label(start_date: str, end_date: str) -> str:
    """Human-readable label for the daily board header."""
    if start_date == end_date:
        return start_date
    return f"{start_date} → {end_date}"


def llm_leaderboard_entries(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Configured competition LLM models (same roster as the contest board)."""
    cfg = config or load_leaderboard_config()
    return [
        s for s in cfg.get("strategies", [])
        if s.get("strategy") == "llm_agent"
    ]


def _daily_refresh_state() -> Dict[str, Any]:
    if not _DAILY_REFRESH_STATE_PATH.exists():
        return {}
    try:
        with open(_DAILY_REFRESH_STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_daily_refresh_state(state: Dict[str, Any]) -> None:
    dest_dir = _DAILY_REFRESH_STATE_PATH.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest_dir), prefix=".daily_refresh_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp, _DAILY_REFRESH_STATE_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            # Best-effort cleanup of the temp file; the original error is re-raised.
            pass
        raise


def _daily_window_key(config: Dict[str, Any]) -> str:
    return f"{config['session_id']}|{config['start_date']}|{config['end_date']}"


def _set_daily_refresh_running(value: bool) -> None:
    """Single writer for the in-progress flag; callers hold ``_daily_refresh_lock``.

    Deliberately does *not* take the lock itself:
    ``maybe_schedule_daily_leaderboard_refresh`` calls this from inside its own
    ``with`` block, and ``_daily_refresh_lock`` is a plain Lock, not an RLock,
    so acquiring here would deadlock.
    """
    global _daily_refresh_running
    _daily_refresh_running = value


def _cached_run_index(
    start_date: str,
    end_date: str,
    session_id: str,
    initial_capital: Optional[float] = None,
    prompt_by_entry: Optional[Dict[str, Optional[str]]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], set]:
    """``(cached run by llm_model, entry ids whose row drifted from the config)``.

    ``_find_cached_run`` rescans the whole session per lookup, so calling it in
    a loop is O(entries × runs) DB work on a request path. Same predicate and
    the same ranking, one query — see ``_cache_match_rank``.

    A drifted entry appears in **both** returns: it is still a cached row (so
    ``models_pending`` cannot count it and trigger a billable redeploy — see
    ``_resolve_cached_run``), and it is named so the caller can report it as
    something other than healthy.
    """
    wanted = _finite_positive(initial_capital)
    prompts = prompt_by_entry or {}
    index: Dict[str, Dict[str, Any]] = {}
    ranks: Dict[str, int] = {}
    for run in db.get_runs_by_session(session_id) or []:
        if not (
            run.get("mode") == LEADERBOARD_MODE
            and run.get("start_date") == start_date
            and run.get("end_date") == end_date
        ):
            continue
        key = run.get("llm_model")
        rank = _cache_match_rank(run, wanted, _wanted_prompt(prompts.get(key)))
        if key not in index or rank < ranks[key]:
            index[key] = run
            ranks[key] = rank
    drifted = {key for key, rank in ranks.items() if rank == _CACHE_STALE}
    return index, drifted


def _daily_models_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """How many competition LLM curves exist for the current daily window."""
    entries = llm_leaderboard_entries(config)
    # Single scan: this runs on every public GET of the daily board.
    cached_runs, drifted_ids = _cached_run_index(
        config["start_date"],
        config["end_date"],
        config["session_id"],
        config.get("initial_capital", INITIAL_CAPITAL),
        {e["id"]: e.get("strategy_prompt") for e in entries},
    )
    cached = 0
    drifted = 0
    pending_ids: List[str] = []
    for entry in entries:
        entry_id = entry["id"]
        if cached_runs.get(entry_id):
            # Drifted rows count as CACHED, never pending: pending is what
            # `maybe_schedule_daily_leaderboard_refresh` spends money on. See
            # `_resolve_cached_run`.
            cached += 1
            # ⚠ Counted HERE, not as `len(drifted_ids)`. `_cached_run_index`
            # ranks every run in the session and window — the five baselines
            # included — while every other number in this dict is computed over
            # `llm_leaderboard_entries` alone. Taking the raw set size let a
            # drifted baseline report `models_config_drift` above `models_total`,
            # i.e. more of a population drifted than the population has members.
            # Intersecting with the entries actually counted keeps the invariant
            # drift ≤ cached ≤ total, which is what makes the dict readable.
            if entry_id in drifted_ids:
                drifted += 1
        else:
            pending_ids.append(entry_id)
    total = len(entries)
    return {
        "trading_date": config["start_date"],
        "models_total": total,
        "models_cached": cached,
        "models_config_drift": drifted,
        "models_pending": len(pending_ids),
        "pending_entry_ids": pending_ids,
        "refresh_in_progress": _daily_refresh_running,
    }


def _auto_deploy_daily_models_enabled() -> bool:
    """Whether missing daily LLM curves should backfill in a background thread.

    **Strict opt-in.** ``GET /api/v1/leaderboard?period=daily`` is public and
    unauthenticated, and this flag decides whether serving it may kick off
    ``deploy_model_run`` for every competition entry — i.e. real, billable LLM
    API calls. Defaulting on for "not Render" would have armed that path on the
    Docker image, every self-host and fork, and the test suite (``conftest``
    deliberately strips ``RENDER``). An operator must ask for it by name.
    """
    raw = (os.getenv("LEADERBOARD_DAILY_AUTO_DEPLOY") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def refresh_daily_leaderboard(
    *,
    deploy_models: bool = False,
    force_refresh: bool = False,
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Recompute the rolling daily board for the last completed weekday.

    Always refreshes cheap baselines/index lines. When ``deploy_models`` is
    True, also runs every configured ``llm_agent`` entry over the daily window
    (real LLM API calls — use the cron script or POST refresh endpoint).
    """
    config = resolve_leaderboard_config("daily")
    window_key = _daily_window_key(config)
    prior = _daily_refresh_state()
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
                "start_date": config["start_date"],
                "end_date": config["end_date"],
                "label": daily_window_label(config["start_date"], config["end_date"]),
            },
        }

    baseline_meta = ensure_leaderboard_runs(
        force_refresh=force_refresh, period="daily", config=config
    )
    result: Dict[str, Any] = {
        "window_key": window_key,
        "window": {
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "label": daily_window_label(config["start_date"], config["end_date"]),
        },
        "period": "daily",
        "baselines_refreshed": True,
        "baselines": baseline_meta,
        "models_deployed": False,
        "model_results": [],
        "model_failures": [],
        "refreshed_at": _utcnow_iso(),
        "skipped": False,
    }

    if deploy_models:
        failures: List[Dict[str, str]] = []
        successes: List[Dict[str, Any]] = []
        for entry in llm_leaderboard_entries(config):
            entry_id = entry["id"]
            try:
                row = deploy_model_run(
                    entry_id,
                    force_refresh=force_refresh,
                    period="daily",
                    allow_fallback=allow_fallback,
                )
                successes.append(row)
            except (LeaderboardFallbackError, ValueError, RuntimeError) as exc:
                failures.append({"entry_id": entry_id, "error": str(exc)})
        # Only mark complete when every model succeeded; a partial run must not
        # skip the remaining entries on the next cron (force=false).
        result["models_deployed"] = not failures
        result["model_results"] = successes
        result["model_failures"] = failures

    _save_daily_refresh_state(result)
    return result


def _run_daily_refresh_background(
    *,
    deploy_models: bool,
    force_refresh: bool,
) -> None:
    """Worker body. Deliberately no ``allow_fallback``: the H6 integrity guard
    is never waived on a background/HTTP-triggered run, only by an operator
    running ``scripts/refresh_daily_leaderboard.py --allow-fallback`` locally."""
    try:
        refresh_daily_leaderboard(
            deploy_models=deploy_models,
            force_refresh=force_refresh,
        )
    except Exception as exc:
        print(f"⚠️ Daily leaderboard background refresh failed: {exc}")
    finally:
        with _daily_refresh_lock:
            _set_daily_refresh_running(False)


def maybe_schedule_daily_leaderboard_refresh(
    *,
    deploy_models: Optional[bool] = None,
    force_refresh: bool = False,
) -> bool:
    """Start a background daily refresh if one is not already running.

    Returns True when a new worker thread was started.

    The in-progress guard (``_daily_refresh_running``) and the state file are
    **per-process**: two workers or two instances can each schedule the same
    window. Adequate for the current single-instance Render deploy; a
    multi-replica deploy would need a shared lock, or duplicate model deploys.
    """
    config = resolve_leaderboard_config("daily")
    status = _daily_models_status(config)
    should_deploy = (
        deploy_models
        if deploy_models is not None
        else (_auto_deploy_daily_models_enabled() and status["models_pending"] > 0)
    )
    if not force_refresh and not should_deploy:
        prior = _daily_refresh_state()
        if prior.get("window_key") == _daily_window_key(config) and prior.get(
            "baselines_refreshed"
        ):
            return False

    with _daily_refresh_lock:
        if _daily_refresh_running:
            return False
        # Claim before starting, never after: a fast worker can reach its
        # finally and clear the flag before start() even returns, and a
        # set-after-start would then leave it stuck True forever.
        _set_daily_refresh_running(True)
        thread = threading.Thread(
            target=_run_daily_refresh_background,
            kwargs={
                "deploy_models": should_deploy,
                "force_refresh": force_refresh,
            },
            daemon=True,
            name="daily-leaderboard-refresh",
        )
        try:
            thread.start()
        except Exception:
            # A failed start would otherwise strand the flag at True forever:
            # every later refresh returns False and the UI polls a worker that
            # does not exist. Release it and let the caller see "not started".
            _set_daily_refresh_running(False)
            raise
        return True


def enqueue_daily_leaderboard_refresh(
    *,
    deploy_models: bool = True,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Cron/API entrypoint: accept a refresh and run it in a background thread.

    Never blocks on model deploys — callers (GitHub Actions curl, ``--remote``)
    get an immediate acknowledgement. Progress is visible via
    ``GET /api/v1/leaderboard?period=daily`` (``daily_status``).

    No ``allow_fallback``: the H6 integrity guard is not waivable over HTTP.
    """
    config = resolve_leaderboard_config("daily")
    started = maybe_schedule_daily_leaderboard_refresh(
        deploy_models=deploy_models,
        force_refresh=force_refresh,
    )
    status = _daily_models_status(config)
    # The worker flips the flag under the lock; treat a just-started thread as
    # in-progress even if the status snapshot raced ahead of the assignment.
    in_progress = bool(status.get("refresh_in_progress") or started)
    return {
        "accepted": True,
        "started": started,
        "refresh_in_progress": in_progress,
        "window": {
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "label": daily_window_label(config["start_date"], config["end_date"]),
        },
        "daily_status": status,
        "message": (
            "Daily leaderboard refresh started in the background."
            if started
            else (
                "Daily leaderboard refresh already in progress."
                if in_progress
                else "No new daily refresh scheduled (window already satisfied)."
            )
        ),
    }


def verify_daily_refresh_secret(provided: Optional[str]) -> None:
    """Raise when the cron secret is unconfigured (ValueError) or wrong (PermissionError)."""
    expected = (os.getenv("LEADERBOARD_DAILY_REFRESH_SECRET") or "").strip()
    if not expected:
        raise ValueError("LEADERBOARD_DAILY_REFRESH_SECRET is not configured")
    token = (provided or "").strip()
    # Compare as bytes: Starlette latin-1-decodes headers, and compare_digest
    # raises TypeError on a str containing non-ASCII — which would surface as an
    # unhandled 500 instead of a clean auth failure.
    # surrogateescape, not strict: os.getenv round-trips undecodable env bytes as
    # surrogates on Linux, and strict encoding would raise on those too.
    if not token or not secrets.compare_digest(
        token.encode("utf-8", "surrogateescape"),
        expected.encode("utf-8", "surrogateescape"),
    ):
        raise PermissionError("Invalid daily leaderboard refresh secret")


# Season 0 is the shakedown season by convention: numbered, so the board has a
# real identity to show and so Season 1 means "the first one that counted", but
# explicitly the one whose results nobody should read as a standing. Mirrors
# PREVIEW_SEASON_NUMBER in dashboard/frontend/js/leaderboard.js.
PREVIEW_SEASON_NUMBER = 0
DEFAULT_SEASON_TRADING_DAYS = 10
# One calendar year of weekday sessions. An upper bound rather than a sanity
# check: `season_window` walks the calendar one day at a time and runs inside a
# public, unauthenticated GET, so the config number is the loop bound. See
# `_season_trading_days`.
MAX_SEASON_TRADING_DAYS = 260

# Preview-season windows already reported as elapsed — see `_preview_season_dates`.
# Warned once per (start, end) rather than per request, because the request is a
# public GET and the condition, once true, is true forever.
_warned_elapsed_seasons: set[Tuple[str, str]] = set()


def _season_trading_days(season_cfg: Dict[str, Any]) -> int:
    """The configured season length, clamped to a length a season can have.

    Parsed and clamped **once**, here, so the ``trading_days_total`` the payload
    reports is the number the window was actually built from. Reporting the raw
    config value instead was not cosmetic: the client computes
    ``elapsed / total``, so a negative length rendered a **100%-full** progress
    bar directly underneath the banner whose entire job is denying that anything
    advanced.

    The parse is guarded for the same reason the clamp is. A bare ``int()`` on a
    config string raises ``ValueError`` out of a public, unauthenticated GET, and
    a large value spins the ``season_window`` calendar walk on every request.
    """
    raw = season_cfg.get("length_trading_days")
    if raw is None or raw == "":
        return DEFAULT_SEASON_TRADING_DAYS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        print(
            "[leaderboard] season.length_trading_days is not an integer "
            f"({raw!r}); using {DEFAULT_SEASON_TRADING_DAYS}"
        )
        return DEFAULT_SEASON_TRADING_DAYS
    clamped = max(1, min(value, MAX_SEASON_TRADING_DAYS))
    if clamped != value:
        print(
            f"[leaderboard] season.length_trading_days={value} is outside "
            f"1..{MAX_SEASON_TRADING_DAYS}; using {clamped}"
        )
    return clamped


def season_window(start_date: str, trading_days: int) -> Tuple[str, str]:
    """The (start, end) dates of a season ``trading_days`` sessions long.

    Ten sessions is two calendar weeks of US cash trading, Monday through
    Friday. Not a new number: ``js/leaderboard.js`` already declares
    ``const SEASON_TRADING_DAYS = 10;`` with exactly that comment.

    Weekdays only -- market holidays are NOT modelled. That is correct for
    Season 0, whose window (2026-08-12 → 2026-08-25) contains none, and it is
    knowingly insufficient for the advance engine, which will need a real
    calendar. It is stated here rather than left for that engine to discover:
    the failure would be a season that ends one session short with nothing
    reporting it.

    A **weekend ``start_date`` rolls forward** to the following Monday instead of
    being echoed back. The returned start is the season's first *session*, and
    the client prints it as the window's first day; a Saturday there is a date on
    which, by this function's own weekday rule, nothing traded.

    ``trading_days`` is clamped to ``1..MAX_SEASON_TRADING_DAYS`` **in here**, not
    only in the caller: this is a module-level function reached from a public GET
    and the argument is the bound on a day-at-a-time calendar walk.
    """
    cursor = date.fromisoformat(start_date)
    while cursor.weekday() >= 5:
        cursor += timedelta(days=1)
    start = cursor
    target = max(1, min(int(trading_days), MAX_SEASON_TRADING_DAYS))
    counted = 0
    last = start
    while counted < target:
        if cursor.weekday() < 5:
            counted += 1
            last = cursor
        cursor += timedelta(days=1)
    return start.isoformat(), last.isoformat()


def _preview_season_dates(
    season_cfg: Dict[str, Any],
    trading_days: int,
    as_of: Optional[Union[date, datetime]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """The preview season's window, or ``(None, None)`` when it cannot be stated.

    Two ways it cannot be, and both used to answer with a confident wrong window:

    - **No ``season_zero_start``.** This fell back to ``config["start_date"]`` --
      the *contest* window -- and published 2026-04-15 → 2026-04-28 as a season.
      "Nobody configured a season" and "the season is the April contest window"
      came back byte-identical, both ``status: preview``, HTTP 200. That is the
      exact shape of CLAUDE.md's fail-closed-is-not-fail-visible section.
    - **The configured window has already elapsed.** *Nothing advances Season 0* --
      there is no engine -- so a fixed fortnight becomes a fixed **past** fortnight
      the day after it ends, and the strip renders "Aug 12 – Aug 25 · Day 0 of 10"
      for as long as the preview ships. No operator error is required, only time,
      which is why it is handled here rather than left to a config bump.

    ``(None, None)`` lands on copy the client already carries for this exact
    state: "Dates set when the first season opens", and a subtitle with the date
    clause dropped. A season that has not opened has no window -- that is the
    true thing to say, and the only one that does not rot.

    The elapsed case prints once per window. Silently swapping a stale window for
    no window is still a silent swap; the operator signal is a log line because
    the caller is an anonymous GET (same convention as `_warn_on_feed_drift`).
    """
    raw = season_cfg.get("season_zero_start")
    raw_start = raw.strip() if isinstance(raw, str) else ""
    if not raw_start:
        return None, None
    try:
        start, end = season_window(raw_start, trading_days)
    except ValueError:
        print(
            "[leaderboard] season.season_zero_start is not an ISO date "
            f"({raw!r}); the preview season reports no window"
        )
        return None, None
    if date.fromisoformat(end) < _coerce_as_of_eastern(as_of).date():
        key = (start, end)
        if key not in _warned_elapsed_seasons:
            _warned_elapsed_seasons.add(key)
            print(
                f"[leaderboard] preview season window {start} → {end} has fully "
                "elapsed and no advance engine exists; the season block now "
                "reports no dates. Move season.season_zero_start forward in "
                "dashboard/config/leaderboard.json, or ship the advance engine."
            )
        return None, None
    return start, end


def build_season_payload(
    config: Dict[str, Any],
    as_of: Optional[Union[date, datetime]] = None,
) -> Dict[str, Any]:
    """The season block the Live Trading tab renders.

    NOTHING HERE MAY CLAIM AN ADVANCE. ``last_advanced_date`` stays None and
    ``trading_days_elapsed`` stays 0 because no season has advanced -- there is
    no advance engine. ``seasonHasAdvanced()`` on the client tests exactly those
    two fields, deliberately rather than the period string, precisely so that
    teaching the server the word "live" cannot clear the preview banner. A date
    here flips the badge to "Running" and prints "Next advance: nightly after
    the 16:00 ET close" under a board nothing updates.

    Every key the client reads is present. A missing one is not a crash there --
    the render path uses optional chaining throughout -- it is a silently blank
    season strip, which is worse.

    ``start_date``/``end_date`` are the two that may legitimately be ``None``;
    see `_preview_season_dates` for the two states that produce it. Present-but-
    null, never absent.
    """
    season_cfg = config.get("season") or {}
    trading_days = _season_trading_days(season_cfg)
    start, end = _preview_season_dates(season_cfg, trading_days, as_of)
    return {
        "number": PREVIEW_SEASON_NUMBER,
        "status": "preview",
        "start_date": start,
        "end_date": end,
        "last_advanced_date": None,
        "trading_days_elapsed": 0,
        "trading_days_total": trading_days,
        "entries_open": False,
        "entry_closes_at": None,
        "entry_count": 0,
        "next_advance_at": None,
        "gaps": [],
    }


def resolve_leaderboard_config(period: Optional[str] = "contest") -> Dict[str, Any]:
    """Return the effective leaderboard config for ``contest``, ``daily``, or ``live``.

    Daily reuses the same strategy roster as the contest board, but caches under
    a separate session and a rolling 1-day (last completed weekday) window.
    Live reuses the contest session and window verbatim — it is a Season 0
    preview of the Competition board under season chrome, not a distinct
    computed window — so every entry still hits the same cache.
    """
    base = load_leaderboard_config()
    period_key = _normalize_period(period)
    if period_key == "daily":
        start_date, end_date = daily_window_dates()
        # Drop fixed contest reference so mean-variance gets a fresh prior month.
        daily_base = {k: v for k, v in base.items() if k != "reference_start_date"}
        return {
            **daily_base,
            "session_id": base.get("daily_session_id", "leaderboard-daily"),
            "start_date": start_date,
            "end_date": end_date,
            "reference_start_date": reference_start_date(start_date, None),
            "description": (
                f"Daily leaderboard for {start_date} (last completed US cash session). "
                "After 16:00 America/New_York the board advances to that weekday; "
                "before the close (and on weekends) it shows the prior session. "
                "Baselines refresh automatically; competition models deploy via the "
                "nightly refresh job or LEADERBOARD_DAILY_AUTO_DEPLOY locally."
            ),
            "period": "daily",
            "board_title": "Daily Leaderboard",
            "phase_label": "Daily",
            "standings_label": "Ranking",
        }
    if period_key == "live":
        # The contest session and the contest window, deliberately. This board
        # is a Season 0 PREVIEW: real Competition curves under season chrome,
        # with a banner saying nothing here has advanced. Inventing a window
        # would make `_find_cached_run` miss on all twelve entries and start
        # recomputing baselines -- and with LEADERBOARD_DAILY_AUTO_DEPLOY armed,
        # LLM deploys -- from a public, unauthenticated GET.
        return {
            **base,
            # Overridden, not inherited. The contest description ("One-month
            # contest window; stock baselines use prior-month data only as
            # reference…") describes the *rules of the Competition board*, and
            # inheriting it shipped those rules to `window.description` on a tab
            # that does not run under them -- to every non-browser consumer of
            # this payload, since /app never renders the field and so never
            # showed the mismatch.
            "description": (
                f"Season {PREVIEW_SEASON_NUMBER} preview. The curves are the "
                "Competition board's fixed window, shown under season chrome so "
                "the layout can be reviewed: no season has advanced, and no "
                "ranking on this board counts. There is no advance engine yet."
            ),
            "period": "live",
            "board_title": "Live Trading Leaderboard",
            # Derived, never spelled out: a literal "Season 0" here is a second
            # owner of the season number, and the first thing the advance engine
            # does is bump PREVIEW_SEASON_NUMBER.
            "phase_label": f"Season {PREVIEW_SEASON_NUMBER}",
            "standings_label": "Ranking",
        }
    return {
        **base,
        "period": "contest",
        "board_title": "Competition Leaderboard",
        "phase_label": "Preseason",
        "standings_label": "Ranking",
    }


def _finite_positive(value: Any) -> Optional[float]:
    """``value`` as a positive finite float, or None when it is not one.

    ``None``, ``NaN``, ``inf``, a non-numeric string and ``0`` all answer None:
    every one of them is a number this module must neither divide by nor
    publish as a seed. ``bool`` is excluded explicitly because ``float(True)``
    is ``1.0``, which would sail through every other check.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num) or num <= 0:
        return None
    return num


def _finite(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None when it is not a number at all.

    The twin of ``_finite_positive`` for the columns where ``0`` and a negative
    are *real observations* rather than a missing one: a final equity, a return,
    a drawdown. ``_finite_positive`` must reject zero because its callers divide
    by the result; these callers rank on it, and refusing an account that truly
    went to zero would be the same absent-as-a-value conflation one column over.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _run_seed(run: Dict[str, Any]) -> Optional[float]:
    """The seed a stored run was actually executed at, or None if unrecorded."""
    return _finite_positive(run.get("initial_equity"))


def _run_id(strategy_id: str, start_date: str, end_date: str) -> str:
    """Deterministic id for one leaderboard run — deliberately seed-free.

    ⚠ **Do not put the seed capital in here.** It was tried: the reasoning was
    that ``insert_run`` is INSERT OR REPLACE on this id, so a re-run at a new
    seed overwrites the old row rather than sitting beside it, and the
    seed-aware lookup below then has nothing to choose between. True, and still
    the wrong trade — because the twelve rows in the committed seed database
    carry these *seed-free* ids. A suffixed id does not replace them, it
    **inserts alongside** them, so the first force-refresh after such a change
    doubles every entry on the board and orphans twelve equity curves that
    nothing ever prunes. Duplicate history is worse than the ambiguity it buys.

    One id per (entry, window) is the invariant: a refresh replaces the row and
    its curve, so a window can hold exactly one seed and ``_cache_match_rank``'s
    job is to notice when that seed is not the one the config publishes — not to
    arbitrate between rival rows.
    """
    return f"lb_{strategy_id}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"


def _skip_cache_key(session_id: str, start_date: str, end_date: str, strategy_id: str) -> str:
    return f"{session_id}|{start_date}|{end_date}|{strategy_id}"


def _load_skip_cache() -> Dict[str, str]:
    if not _SKIP_CACHE_PATH.exists():
        return {}
    try:
        with open(_SKIP_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_skip_cache(cache: Dict[str, str]) -> None:
    """Persist the skip cache.

    Best-effort optimization only: this sidecar just avoids re-fetching a
    baseline already known to be uncomputable for a window. Concurrent writers
    race last-write-wins, and a lost write merely costs one extra recompute
    (``_load_skip_cache`` already tolerates a missing/corrupt file). We still
    write to a unique temp file and ``os.replace`` so a crash mid-write can
    never leave a truncated, unparseable JSON file behind for readers.
    """
    # The temp file must sit in the destination's own directory so os.replace
    # is an atomic same-filesystem rename (a cross-device rename raises OSError).
    dest_dir = _SKIP_CACHE_PATH.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest_dir), prefix=".skip_cache_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        os.replace(tmp, _SKIP_CACHE_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _is_skipped(
    session_id: str, start_date: str, end_date: str, strategy_id: str, cache: Dict[str, str]
) -> bool:
    return _skip_cache_key(session_id, start_date, end_date, strategy_id) in cache


def _record_skip(
    session_id: str,
    start_date: str,
    end_date: str,
    strategy_id: str,
    reason: str,
    cache: Dict[str, str],
) -> None:
    """Remember an uncomputable baseline for this window so we don't refetch daily."""
    cache[_skip_cache_key(session_id, start_date, end_date, strategy_id)] = reason
    _save_skip_cache(cache)


def _clear_skips_for_window(
    session_id: str, start_date: str, end_date: str, cache: Dict[str, str]
) -> Dict[str, str]:
    prefix = f"{session_id}|{start_date}|{end_date}|"
    kept = {k: v for k, v in cache.items() if not k.startswith(prefix)}
    if len(kept) != len(cache):
        _save_skip_cache(kept)
    return kept


def _prune_stale_window_skips(
    session_id: str, start_date: str, end_date: str, cache: Dict[str, str]
) -> Dict[str, str]:
    """Drop skip entries for *earlier* windows of this same session.

    The daily board's window rolls every trading day, so yesterday's
    ``leaderboard-daily|<old-window>|...`` keys are dead the moment the window
    advances. Without pruning, the sidecar would gain one entry per failing
    baseline per day forever (a slow leak on a persistent disk). For a
    fixed-window board like the contest there are no other-window keys under
    its ``session_id``, so this is a no-op there. Entries from *other* sessions
    are always preserved.
    """
    session_prefix = f"{session_id}|"
    current_prefix = f"{session_id}|{start_date}|{end_date}|"
    kept = {
        k: v
        for k, v in cache.items()
        if not k.startswith(session_prefix) or k.startswith(current_prefix)
    }
    if len(kept) != len(cache):
        _save_skip_cache(kept)
    return kept


def _wanted_prompt(value: Any) -> Optional[str]:
    """The strategy prompt to compare on, or None to not compare at all."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _run_strategy_prompt(run: Dict[str, Any]) -> Optional[str]:
    """The instruction a published run recorded, or None if it recorded none.

    Written by ``_llm_run_metadata`` into ``agent_runs.metadata`` (PR #366).
    Every row in the committed seed database predates it and has
    ``metadata = NULL``, which is exactly why an unrecorded prompt has to mean
    *unknown* and not *empty*.
    """
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return _wanted_prompt(metadata.get("strategy_prompt"))


def _cache_match_rank(
    run: Dict[str, Any],
    wanted_capital: Optional[float],
    wanted_prompt: Optional[str],
) -> int:
    """How well one cached row answers what the board is asking for.

    ``_CACHE_MATCH`` every recorded dimension agrees, ``_CACHE_UNRECORDED`` the
    row never recorded one of them, ``_CACHE_STALE`` the row recorded one and it
    disagrees. Shared by ``_resolve_cached_run`` and ``_cached_run_index`` so
    the scan and its batched twin cannot drift.

    **Only a recorded disagreement is stale.** The twelve rows in the committed
    seed database carry no ``metadata`` at all, so treating *unrecorded* as a
    mismatch would refuse the entire board — and refusing is not free even
    though it never recomputes, because the entry then vanishes from a public
    page. A dimension the *config* does not set is not compared either: the
    prompt a run records is resolved off the strategy impl, not only off
    ``leaderboard.json``, so a one-sided comparison would drop legitimate rows.
    """
    rank = _CACHE_MATCH
    if wanted_capital is not None:
        seed = _run_seed(run)
        if seed is None:
            rank = max(rank, _CACHE_UNRECORDED)
        elif abs(seed - wanted_capital) > _SEED_MATCH_TOLERANCE:
            return _CACHE_STALE
    if wanted_prompt is not None:
        stored = _run_strategy_prompt(run)
        if stored is None:
            rank = max(rank, _CACHE_UNRECORDED)
        elif stored != wanted_prompt:
            return _CACHE_STALE
    return rank


def _resolve_cached_run(
    strategy_id: str,
    start_date: str,
    end_date: str,
    session_id: str,
    initial_capital: Optional[float] = None,
    strategy_prompt: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """``(cached run, how well it matches)`` for one entry and window.

    ⚠ **The config is a RANKING key here, never a filter, and that is a spend
    control.** A row for this window is always returned if one exists; the rank
    only says whether it was produced under the config the board now publishes.
    Refusing a drifted row would make it *missing*, and missing is not a quiet
    state in this module:

    * ``ensure_leaderboard_runs`` answers missing by recomputing — on a public,
      unauthenticated GET, and on every page load for as long as the config
      disagrees.
    * ``_daily_models_status`` reports missing as **pending**, and a non-zero
      pending count is what ``maybe_schedule_daily_leaderboard_refresh`` acts
      on; ``refresh_daily_leaderboard`` then calls ``deploy_model_run`` for
      **every** configured LLM entry. With ``LEADERBOARD_DAILY_AUTO_DEPLOY``
      armed, one edit to ``leaderboard.json`` would buy a billable re-run of the
      whole board from an anonymous request.

    So a cache miss caused by a config change is not merely undesirable, it is
    unbounded spend — which is why this cannot miss. It is also the policy
    ``_warn_on_feed_drift`` already set for the identical problem one field
    over: warn, never treat as missing, leave recomputing to an operator.
    ``_warn_on_capital_drift`` is the matching signal, and issue #365's third
    criterion asks for exactly that warning — a criterion that would be dead
    code if a drifted row were refused instead of published.

    Preference order: a row matching every recorded dimension, then a row that
    never recorded one, then a row that recorded one and disagrees. With no
    ``initial_capital`` or ``strategy_prompt`` passed, nothing is compared and
    the first row for the window wins, exactly as this has always behaved.
    """
    wanted_capital = _finite_positive(initial_capital)
    wanted_prompt = _wanted_prompt(strategy_prompt)
    best: Optional[Dict[str, Any]] = None
    best_rank = _CACHE_STALE + 1
    for run in db.get_runs_by_session(session_id) or []:
        if not (
            run.get("mode") == LEADERBOARD_MODE
            and run.get("start_date") == start_date
            and run.get("end_date") == end_date
            and run.get("llm_model") == strategy_id
        ):
            continue
        rank = _cache_match_rank(run, wanted_capital, wanted_prompt)
        if rank < best_rank:
            best, best_rank = run, rank
            if rank == _CACHE_MATCH:
                break  # nothing can beat it; stop scanning
    return best, best_rank


def _find_cached_run(
    strategy_id: str,
    start_date: str,
    end_date: str,
    session_id: str,
    initial_capital: Optional[float] = None,
    strategy_prompt: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The cached run for one entry and window, or None. See ``_resolve_cached_run``.

    Passing neither ``initial_capital`` nor ``strategy_prompt`` compares nothing
    and returns the first row for the window, which is what this function has
    always done.
    """
    return _resolve_cached_run(
        strategy_id,
        start_date,
        end_date,
        session_id,
        initial_capital,
        strategy_prompt,
    )[0]


def _symbols_for_config(config: Dict[str, Any]) -> List[str]:
    symbols: set[str] = set()
    for strategy in config.get("strategies", []):
        symbols.update(get_strategy(strategy).required_symbols())
    return sorted(symbols)


def _config_needs_alpaca(config: Dict[str, Any]) -> bool:
    """True when any auto-compute strategy requires Alpaca hourly stock bars."""
    for strategy in config.get("strategies", []):
        if not _auto_compute(strategy):
            continue
        if get_strategy(strategy).required_symbols():
            return True
    return False


def _alpaca_bars_start(config: Dict[str, Any]) -> str:
    """Earliest date for Alpaca fetch — includes prior-month reference when configured."""
    contest_start = config["start_date"]
    if config.get("reference_start_date") or _config_needs_mean_variance(config):
        return reference_start_date(contest_start, config)
    return contest_start


def _config_needs_mean_variance(config: Dict[str, Any]) -> bool:
    for strategy in config.get("strategies", []):
        if not _auto_compute(strategy):
            continue
        if strategy.get("strategy") == "mean_variance":
            return True
    return False


def ensure_leaderboard_runs(
    force_refresh: bool = False,
    period: Optional[str] = "contest",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute and persist leaderboard baselines if missing.

    Successful runs are cached in SQLite. Strategies that fail for a given
    window (empty curve / no bars) are recorded in a small skip cache so a
    broken baseline like mean-variance cannot force an Alpaca refetch on every
    page load — especially important for the daily board ("once per day").
    """
    config = config or resolve_leaderboard_config(period)
    session_id = config["session_id"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    initial_capital = float(config.get("initial_capital", INITIAL_CAPITAL))
    skip_cache = _load_skip_cache()
    # Bound the sidecar: forget skip entries for now-stale windows of this
    # rolling board before doing anything else (no-op for fixed-window boards).
    skip_cache = _prune_stale_window_skips(session_id, start_date, end_date, skip_cache)
    if force_refresh:
        skip_cache = _clear_skips_for_window(session_id, start_date, end_date, skip_cache)

    cached_runs: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for strategy in config.get("strategies", []):
        if not _auto_compute(strategy):
            continue  # LLM models are deployed manually, never block a request
        strategy_id = strategy["id"]
        cached = (
            None
            if force_refresh
            else _find_cached_run(
                strategy_id,
                start_date,
                end_date,
                session_id,
                initial_capital,
                strategy.get("strategy_prompt"),
            )
        )
        if cached:
            # A row that drifted from the config is still a HIT, deliberately:
            # see `_resolve_cached_run`. `_warn_on_capital_drift` below is what
            # reports it; recomputing is an operator action.
            cached_runs.append(cached)
            continue
        if not force_refresh and _is_skipped(session_id, start_date, end_date, strategy_id, skip_cache):
            continue
        missing.append(strategy)

    _warn_on_feed_drift(cached_runs, _configured_feed_or_none())
    _warn_on_capital_drift(cached_runs, initial_capital)

    # Nothing left to compute — serve cached board without touching Alpaca.
    if not missing and not force_refresh:
        return {
            "session_id": session_id,
            "start_date": start_date,
            "end_date": end_date,
            "period": config.get("period", "contest"),
            "created": 0,
            "skipped": 0,
            "refreshed_at": _utcnow_iso(),
            "cache_hit": True,
        }

    needs_fetch = bool(missing) or force_refresh
    bars_by_symbol: Optional[Dict[str, Any]] = None
    if needs_fetch:
        if _config_needs_alpaca(config):
            fetch_start = _alpaca_bars_start(config)
            bars_by_symbol = fetch_hourly_bars(
                _symbols_for_config(config), fetch_start, end_date
            )
            if not bars_by_symbol:
                print(
                    "⚠️ No Alpaca market data — skipping stock-based baselines "
                    "(index lines still use Yahoo Finance)"
                )
                bars_by_symbol = {}
        else:
            bars_by_symbol = {}

    created = 0
    skipped = 0
    # Only iterate strategies that still need work (or everything on force refresh).
    to_run = config.get("strategies", []) if force_refresh else missing
    for strategy in to_run:
        strategy_id = strategy["id"]
        if not _auto_compute(strategy):
            continue  # deployed via deploy_model_run(), not on-demand
        existing = None if force_refresh else _find_cached_run(
            strategy_id,
            start_date,
            end_date,
            session_id,
            initial_capital,
            strategy.get("strategy_prompt"),
        )
        if existing and not force_refresh:
            continue

        strategy_impl = get_strategy(strategy)
        required = strategy_impl.required_symbols()
        if bars_by_symbol is not None:
            bars = bars_by_symbol
        else:
            bars = fetch_hourly_bars(required, start_date, end_date) if required else {}

        if required and not bars:
            print(f"⚠️ Skipping {strategy_id}: no Alpaca bars for contest window")
            _record_skip(
                session_id, start_date, end_date, strategy_id, "no_bars", skip_cache
            )
            skipped += 1
            continue

        curve = strategy_impl.run(bars, start_date, end_date, initial_capital)
        if not curve:
            print(f"⚠️ Skipping {strategy_id}: empty equity curve")
            _record_skip(
                session_id, start_date, end_date, strategy_id, "empty_curve", skip_cache
            )
            skipped += 1
            continue

        metrics = calc_metrics(curve, initial_capital)
        run_id = _run_id(strategy_id, start_date, end_date)

        # Belt-and-suspenders: the auto-compute path is meant for cheap rule-based
        # baselines (LLM entries carry auto_compute=false and deploy manually via
        # deploy_model_run). Guard here too so a misconfigured LLM entry can't
        # slip a rule-based fallback onto the board without the manual override.
        auto_llm_calls = int(getattr(strategy_impl, "llm_calls", 0) or 0)
        auto_llm_decisions = _reported_int(strategy_impl, "llm_decisions")
        _reject_if_llm_fallback(
            strategy_id,
            strategy_impl,
            auto_llm_calls,
            llm_decisions=auto_llm_decisions,
            decision_steps=int(getattr(strategy_impl, "decision_steps", 0) or 0),
            model=strategy.get("model"),
            model_id=getattr(strategy_impl, "model_id", None) or strategy.get("model_id"),
        )

        db.insert_run(
            run_id=run_id,
            session_id=session_id,
            agent_name=strategy["name"],
            mode=LEADERBOARD_MODE,
            start_date=start_date,
            end_date=end_date,
            initial_equity=metrics["initial_equity"],
            final_equity=metrics["final_equity"],
            total_return=metrics["total_return"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            num_trades=strategy_impl.num_trades(),
            llm_model=strategy_id,
            # The numbers the guard immediately above just measured. This path
            # is meant for rule-based baselines, where both are 0 -- but it
            # publishes any entry the guard lets through, and dropping them
            # wrote `llm_model = <entry>` beside `llm_calls = 0`, which is the
            # exact shape `classify_decision_provenance` reads as a rule-based
            # curve wearing that model's name.
            llm_calls=auto_llm_calls,
            llm_decisions=(
                auto_llm_calls if auto_llm_decisions is None else auto_llm_decisions
            ),
            metadata=_with_market_data_provenance(
                _llm_run_metadata(
                    strategy_id,
                    strategy,
                    strategy_impl,
                    model_id=(
                        getattr(strategy_impl, "model_id", None)
                        or strategy.get("model_id")
                    ),
                    initial_capital=initial_capital,
                    start_date=start_date,
                    end_date=end_date,
                ),
                feed_provenance(bars),
            ),
        )
        db.insert_equity_points(run_id, curve)
        created += 1

    return {
        "session_id": session_id,
        "start_date": start_date,
        "end_date": end_date,
        "period": config.get("period", "contest"),
        "created": created,
        "skipped": skipped,
        "refreshed_at": _utcnow_iso(),
        "cache_hit": False,
    }


class LeaderboardFallbackError(RuntimeError):
    """Raised when an LLM leaderboard entry silently fell back to rule-based
    trading, so publishing it would misrepresent a rule-based curve as that
    model's result. Override deliberately with ``allow_fallback=True``."""


def _reported_int(strategy_impl: Any, name: str) -> Optional[int]:
    """Read an int counter a strategy *may* report. Returns ``None`` when the
    attribute is absent so the guard can apply its documented default (e.g.
    ``llm_decisions`` → ``llm_calls``); a present value (including a real 0) is
    coerced to int. Distinguishing absent-from-zero matters: a genuine 0 means
    "the model drove no step" (reject), while absent means "this strategy shape
    doesn't report it" (fall back to llm_calls)."""
    val = getattr(strategy_impl, name, None)
    return None if val is None else int(val)


def _reject_if_llm_fallback(
    entry_id: str,
    strategy_impl: Any,
    llm_calls: int,
    *,
    llm_decisions: Optional[int] = None,
    decision_steps: int = 0,
    model: Optional[str] = None,
    model_id: Optional[str] = None,
    allow_fallback: bool = False,
) -> None:
    """Integrity guard (H6): refuse to publish an LLM entry that silently fell
    back to rule-based trading. Two shapes of fallback are caught:

    - **Total fallback** — no client (missing key/SDK) or a model id the active
      gateway rejected so every call failed (``used_llm`` False or ``llm_calls``
      0). The whole curve is rule-based.
    - **Partial fallback** — the client responded but most steps produced no
      usable decision (``llm_decisions / decision_steps`` below
      ``MIN_LLM_DECISION_COVERAGE``). The curve is *mostly* rule-based, so
      publishing it still misrepresents the model (this is the 1-of-161 run that
      silently topped the board).

    Coverage keys off ``llm_decisions`` — steps the model actually drove — not
    ``llm_calls`` (billed API calls), because a truncated / unparseable response
    is billed yet trades rule-based. A run that returns garbage every step has
    ``llm_calls == decision_steps`` but ``llm_decisions == 0``, and must still be
    refused. ``llm_decisions`` defaults to ``llm_calls`` for callers (or older
    strategy objects) that don't report it separately.

    Rule-based baselines expose no ``used_llm`` (getattr → None) and pass through
    untouched. Applied on BOTH insert paths so an LLM entry can't slip through
    the auto-compute path. Coverage is only checked when ``decision_steps`` is
    known (> 0); a genuine run always reports it."""
    used_llm = getattr(strategy_impl, "used_llm", None)
    if used_llm is None or allow_fallback:
        return
    if llm_decisions is None:
        llm_decisions = llm_calls
    if not used_llm or llm_calls == 0:
        raise LeaderboardFallbackError(
            f"Entry '{entry_id}' produced a rule-based fallback "
            f"(used_llm={used_llm}, llm_calls={llm_calls}); refusing to publish it "
            f"under model '{model}'. Usually the model id '{model_id}' is not valid "
            f"for the active LLM gateway, or the API key is missing. Pass "
            f"allow_fallback=True / --allow-fallback to publish it anyway."
        )
    if decision_steps > 0 and llm_decisions < MIN_LLM_DECISION_COVERAGE * decision_steps:
        coverage = llm_decisions / decision_steps
        raise LeaderboardFallbackError(
            f"Entry '{entry_id}' is a partial rule-based fallback: only "
            f"{llm_decisions}/{decision_steps} steps ({coverage:.1%}) produced a "
            f"usable model decision, below the {MIN_LLM_DECISION_COVERAGE:.0%} "
            f"threshold. Most of the curve is rule-based, so refusing to publish "
            f"it under model '{model}'. Usually the model id '{model_id}' "
            f"intermittently failed for the active LLM gateway (e.g. rate limits "
            f"or output truncated into invalid JSON). Pass allow_fallback=True / "
            f"--allow-fallback to publish it anyway."
        )


def _with_market_data_provenance(
    metadata: Optional[Dict[str, Any]],
    provenance: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Record which Alpaca tape priced this run alongside its config snapshot.

    Without this the board cannot tell a SIP curve from an IEX-fallback one:
    the loader stamps the feed on the dataframes, but frames are transient and
    ``agent_runs`` outlives them. Two curves for the same window computed off
    different tapes are not comparable, and ranking them side by side is the
    failure this guards — so the tape goes in the row, not just in a log line.

    Applies to baselines too (``_llm_run_metadata`` returns ``None`` for those),
    which is why it is a separate wrapper rather than another field in there.
    """
    if not provenance:
        return metadata
    return {**(metadata or {}), **provenance}


def _configured_feed_or_none() -> Optional[str]:
    """``ALPACA_DATA_FEED`` as a canonical name, or ``None`` if it is unusable.

    The fetch path raises on a bad value (see ``resolve_alpaca_data_feed``);
    this read-only comparison must not turn a fully cached board into a 500.
    """
    try:
        return configured_feed_name()
    except MarketDataUnavailableError as exc:
        print(f"WARNING: cannot resolve the Alpaca feed for run provenance: {exc}")
        return None


def _run_market_data_feed(run: Dict[str, Any]) -> Optional[str]:
    """Feed recorded on a cached run, or ``None`` for rows written before it was."""
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        return None
    feed = metadata.get("market_data_feed")
    return feed if isinstance(feed, str) else None


def _warn_on_feed_drift(runs: List[Dict[str, Any]], configured: Optional[str]) -> None:
    """Print when cached rows were priced off a tape we no longer use.

    Deliberately a warning and not an auto-refresh: ``ensure_leaderboard_runs``
    runs on a public, unauthenticated GET, and treating a feed mismatch as
    "missing" would re-fetch Alpaca on every page load for as long as the
    mismatch persists (e.g. while SIP keeps falling back to IEX). Recomputing
    is an operator action — ``POST /api/v1/leaderboard/refresh`` with
    ``force=true``, or ``scripts/refresh_daily_leaderboard.py --remote``.

    Emitted once per distinct drift per process: this runs on the hot cached
    path of a public endpoint, and a line per page load would bury itself.
    """
    if not configured:
        return
    stale = sorted(
        {
            feed
            for run in runs
            if (feed := _run_market_data_feed(run)) and feed != configured
        }
    )
    if not stale:
        return
    seen_key = (configured, tuple(stale))
    if seen_key in _warned_feed_drift:
        return
    _warned_feed_drift.add(seen_key)
    print(
        f"WARNING: leaderboard has cached runs priced off {', '.join(stale)} "
        f"while ALPACA_DATA_FEED resolves to {configured}. Curves from "
        "different tapes are not comparable — force-refresh to recompute."
    )


def _warn_on_capital_drift(
    runs: List[Dict[str, Any]], configured: Optional[float]
) -> None:
    """Print when cached rows were run at seed capital the board no longer publishes.

    The capital twin of ``_warn_on_feed_drift`` directly above, and deliberately
    the same shape for the same reason: this is the established answer in this
    module to "cached rows are not comparable". A warning, once per distinct
    drift per process, never an auto-refresh — ``ensure_leaderboard_runs`` and
    ``get_leaderboard`` both run on a public, unauthenticated GET, and
    recomputing is an operator action (``POST /api/v1/leaderboard/refresh`` with
    ``force=true``).

    Why capital is a comparability problem and not a display one: a $10,000 run
    buys whole shares in a coarser quantum than a $100,000 one and pays
    per-trade costs against a smaller base, so it is a *different run* of the
    same strategy, with different returns. Scaling the dollar levels hides that
    and fixes nothing. A board holding both is ranking two things that were not
    measured the same way (issue #365).
    """
    wanted = _finite_positive(configured)
    if wanted is None:
        return
    stale = sorted(
        {
            seed
            for run in runs
            if (seed := _run_seed(run)) and abs(seed - wanted) > _SEED_MATCH_TOLERANCE
        }
    )
    if not stale:
        return
    seen_key = (wanted, tuple(stale))
    if seen_key in _warned_capital_drift:
        return
    _warned_capital_drift.add(seen_key)
    rendered = ", ".join(f"${seed:,.2f}" for seed in stale)
    print(
        f"WARNING: leaderboard has cached runs seeded at {rendered} while the "
        f"board publishes ${wanted:,.2f}. Curves seeded at different capital "
        "are not comparable — a smaller account trades a coarser share quantum, "
        "so the returns differ, not just the dollar levels. Align initial_capital "
        "in dashboard/config/leaderboard.json to the seed the rows were actually "
        "run at, or re-run BOTH halves of the board. ⚠ Do not force-refresh on "
        "its own: `ensure_leaderboard_runs` recomputes only the auto_compute "
        "baselines, so it moves those to the new seed and strands the LLM "
        "entries at the old one — which mixes the board rather than aligning it, "
        "and drops whichever half lands in the minority. The LLM half is "
        "redeployed by deploy_model_run(force_refresh=True)."
    )


def _llm_run_metadata(
    entry_id: str,
    entry: Dict[str, Any],
    strategy_impl: Any,
    *,
    model_id: Optional[str],
    initial_capital: float,
    start_date: str,
    end_date: str,
) -> Optional[Dict[str, Any]]:
    """Snapshot effective config for a published LLM run (``None`` otherwise)."""
    if entry.get("strategy") != "llm_agent":
        return None
    return {
        "entry_id": entry_id,
        "model_id": model_id,
        "integration": getattr(
            strategy_impl, "integration", entry.get("integration")
        ),
        "temperature": getattr(
            strategy_impl, "temperature", entry.get("temperature")
        ),
        "reasoning_effort": getattr(
            strategy_impl, "reasoning_effort", entry.get("reasoning_effort")
        ),
        # The Open Track's competing variable, and the only effective-config knob
        # that rewrites the model's entire strategy body. Recorded verbatim (it is
        # capped at MAX_STRATEGY_PROMPT_CHARS) so a published curve can be tied
        # back to the instruction that produced it — leaderboard.json is editable,
        # so the config is not a record of what ran.
        #
        # NOTE: this makes the run *auditable*, not *invalidating*.
        # `_find_cached_run` (:615) still keys only on
        # (mode, start_date, end_date, llm_model), so editing an entry's
        # instruction and redeploying returns the cached row rather than
        # recomputing. Same omission as `initial_equity`, tracked in issue #365 —
        # fixing the key belongs with that, not here.
        "strategy_prompt": getattr(
            strategy_impl, "strategy_prompt", entry.get("strategy_prompt")
        ),
        "llm_max_output_tokens": llm_harness.DEFAULT_MAX_OUTPUT_TOKENS,
        "initial_capital": initial_capital,
        "start_date": start_date,
        "end_date": end_date,
    }


def deploy_model_run(
    entry_id: str,
    *,
    force_refresh: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    allow_fallback: bool = False,
    period: Optional[str] = "contest",
) -> Dict[str, Any]:
    """Compute and persist one (expensive) leaderboard model entry.

    Used by scripts/deploy_leaderboard_model.py to "deploy" an LLM model onto the
    leaderboard: it runs the model's hourly backtest over the contest window,
    stores the equity curve + metrics + token cost, and caches it so the web
    leaderboard can display it without recomputing. Pass start/end to test on a
    shorter window (writes a separate cached run for that window). Pass
    ``period="daily"`` to target the rolling daily board window.
    """
    config = resolve_leaderboard_config(period)
    session_id = config["session_id"]
    start_date = start_date or config["start_date"]
    end_date = end_date or config["end_date"]
    initial_capital = float(config.get("initial_capital", INITIAL_CAPITAL))

    entry = next(
        (s for s in config.get("strategies", []) if s.get("id") == entry_id),
        None,
    )
    if entry is None:
        available = [s.get("id") for s in config.get("strategies", [])]
        raise ValueError(f"Unknown leaderboard entry '{entry_id}'. Available: {available}")

    existing, existing_rank = _resolve_cached_run(
        entry_id,
        start_date,
        end_date,
        session_id,
        initial_capital,
        entry.get("strategy_prompt"),
    )
    if existing and not force_refresh:
        # A drifted row short-circuits here exactly like a matching one, and
        # that is the point: this function is reached from
        # `refresh_daily_leaderboard`, which loops over EVERY configured LLM
        # entry, and that loop is reachable from a public unauthenticated GET
        # whenever LEADERBOARD_DAILY_AUTO_DEPLOY is armed. Re-running on a
        # config change would answer one edit to leaderboard.json with a
        # billable re-run of the whole board. `force_refresh=True` is the
        # operator's way to ask for it on purpose.
        if existing_rank == _CACHE_STALE:
            # Both reasons a row can be stale, because only one of them has a
            # board-wide scan behind it. `_warn_on_capital_drift` finds nothing
            # when the disagreement is the prompt, so before the second call a
            # drifted instruction produced no signal anywhere.
            _warn_on_capital_drift([existing], initial_capital)
            _warn_on_prompt_drift(entry_id, existing, entry.get("strategy_prompt"))
        return {
            "entry_id": entry_id,
            "run_id": existing["run_id"],
            "cached": True,
            "model": entry.get("model"),
            "total_return": existing.get("total_return"),
            "sharpe_ratio": existing.get("sharpe_ratio"),
            "max_drawdown": existing.get("max_drawdown"),
            "final_equity": existing.get("final_equity"),
            "num_trades": existing.get("num_trades"),
            "llm_calls": existing.get("llm_calls"),
            "input_tokens": existing.get("input_tokens"),
            "output_tokens": existing.get("output_tokens"),
            "est_cost_usd": existing.get("est_cost_usd"),
        }

    strategy_impl = get_strategy(entry)
    # LLM prompts need indicator lookback; for a 1-day daily window that means
    # fetching prior bars while only trading inside [start_date, end_date].
    bars_start = reference_start_date(start_date, config)
    if bars_start > start_date:
        bars_start = start_date
    bars = fetch_hourly_bars(strategy_impl.required_symbols(), bars_start, end_date)
    if not bars:
        raise RuntimeError(
            f"No market data returned for bars window {bars_start} → {end_date}"
        )
    print(f"  bars fetch: {bars_start} → {end_date} (trade window {start_date} → {end_date})")

    curve = strategy_impl.run(bars, start_date, end_date, initial_capital)
    if not curve:
        raise RuntimeError(f"No equity curve produced for entry '{entry_id}'")

    metrics = calc_metrics(curve, initial_capital)
    run_id = _run_id(entry_id, start_date, end_date)

    input_tokens = int(getattr(strategy_impl, "input_tokens", 0) or 0)
    output_tokens = int(getattr(strategy_impl, "output_tokens", 0) or 0)
    llm_calls = int(getattr(strategy_impl, "llm_calls", 0) or 0)
    llm_decisions = _reported_int(strategy_impl, "llm_decisions")
    decision_steps = int(getattr(strategy_impl, "decision_steps", 0) or 0)
    model_id = getattr(strategy_impl, "model_id", None) or entry.get("model_id")
    est_cost = token_cost.estimate_cost_usd(model_id, input_tokens, output_tokens)

    _reject_if_llm_fallback(
        entry_id,
        strategy_impl,
        llm_calls,
        llm_decisions=llm_decisions,
        decision_steps=decision_steps,
        model=entry.get("model"),
        model_id=model_id,
        allow_fallback=allow_fallback,
    )

    db.insert_run(
        run_id=run_id,
        session_id=session_id,
        agent_name=entry["name"],
        mode=LEADERBOARD_MODE,
        start_date=start_date,
        end_date=end_date,
        initial_equity=metrics["initial_equity"],
        final_equity=metrics["final_equity"],
        total_return=metrics["total_return"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        num_trades=strategy_impl.num_trades(),
        llm_model=entry_id,
        llm_calls=llm_calls,
        # The same number `_reject_if_llm_fallback` just admitted this run on.
        # Persisting only `llm_calls` left the column at its DEFAULT 0 on the
        # rows H6 governs, and the billing counter cannot stand in for it --
        # that substitution is what the two counters were split to prevent.
        #
        # `_reported_int` answers None for a strategy shape that does not
        # report the counter, and the guard's documented response to that is to
        # measure `llm_calls` instead; the row mirrors it rather than writing
        # NULL, which would read back as "nobody counted" on a run that was.
        llm_decisions=llm_calls if llm_decisions is None else llm_decisions,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        est_cost_usd=est_cost,
        metadata=_with_market_data_provenance(
            _llm_run_metadata(
                entry_id,
                entry,
                strategy_impl,
                model_id=model_id,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
            ),
            feed_provenance(bars),
        ),
    )
    db.insert_equity_points(run_id, curve)

    return {
        "entry_id": entry_id,
        "run_id": run_id,
        "cached": False,
        "model": entry.get("model"),
        "model_id": model_id,
        "window": {"start_date": start_date, "end_date": end_date},
        "total_return": metrics["total_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "final_equity": metrics["final_equity"],
        "num_trades": strategy_impl.num_trades(),
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "est_cost_usd": est_cost,
    }


def _rank_sort_key(entry: Dict[str, Any]) -> tuple:
    """Official rank is by final portfolio value (nof1-style); tie-break on return.

    ⚠ **Never rank a return against dollars.** The old fallback was
    ``pv = entry.get("cumulative_return") or 0``, which was harmless only while
    ``portfolio_value`` could not be None — and this change is precisely what
    made it nullable. Reachable, it hands the sort a *fraction* (``0.07``) to
    compare against its neighbours' *dollars* (``107494``), so an entry whose
    final equity nobody recorded ranks **dead last** no matter how well it
    actually did. That is the same defect this module just removed from the
    Value column — an absent number published as a real one — pointing the other
    way, and it lands in the column the board is ranked on.

    So the absent case is put back on the dollar axis instead: the entry carries
    the seed the board publishes and the return the run recorded, and
    ``seed × (1 + return)`` is the final equity those two imply. This value is a
    **sort key only** — ``portfolio_value`` stays ``None`` on the wire and both
    tables still render it as an em dash. Ranking is a total order over the
    board and has to produce *some* position; inventing a displayed dollar
    figure is what must not happen.
    """
    ret = _finite(entry.get("cumulative_return"))
    value = _finite(entry.get("portfolio_value"))
    if value is None:
        seed = _finite_positive(entry.get("initial_equity"))
        if seed is not None and ret is not None:
            value = seed * (1.0 + ret)
    if value is None:
        # Neither dollars nor a seed to rebuild them from. Only partial fixtures
        # reach this — every entry `get_leaderboard` builds sets `initial_equity`
        # to the board's published capital — and there the whole board is
        # dimensionless, so ranking on the return alone compares like with like.
        value = ret if ret is not None else 0.0
    return (-value, -(ret if ret is not None else 0.0))


def _rank_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign official ranks by final portfolio value (higher is better)."""
    if not entries:
        return entries

    entries.sort(key=_rank_sort_key)
    for idx, entry in enumerate(entries):
        entry["rank"] = idx + 1
    return entries


def _scaled_level(value: Any, scale: float) -> Optional[float]:
    """One stored dollar level, scaled — with *absent* preserved as absent.

    A stored NULL means "no observation at this timestamp", which is a
    different fact from "the account held $0". ``float(pt.get("equity") or 0)``
    conflated them, and because ``chart_equity_curve`` prepends an open tick at
    starting capital the series then read as a −100% loss. The damage is not
    one bad marker: ``align_equity_curves`` puts every entry on one shared
    axis, so a single −100% series sets the y-range and visually flattens every
    honest curve on the board (issue #390).

    A real ``0.0`` is still a real ``0.0`` and passes through — an account that
    actually went to zero did lose 100%.
    """
    num = _finite(value)
    return None if num is None else num * scale


def _stored_seed(run: Dict[str, Any], equity_hourly: List[Dict[str, Any]]) -> Optional[float]:
    """The seed the stored curve was actually run at, or None if unknowable.

    Never falls back to the *config's* capital. That fallback was issue #365:
    with ``initial_equity`` NULL it made ``scale`` exactly ``1.0``, so a curve
    genuinely seeded at $100,000 was published unscaled and labelled $10,000 —
    a wrong number that looked like a deliberate one. The curve's own first
    observation is a fact about the run; the config is a fact about the board.
    """
    seed = _run_seed(run)
    if seed is not None:
        return seed
    for point in equity_hourly:
        seed = _finite_positive(point.get("equity"))
        if seed is not None:
            return seed
    return None


def _report_curve_integrity(
    entry_id: str, run_id: str, scaled_hourly: List[Dict[str, Any]]
) -> None:
    """Make an *absent* curve distinguishable from a *broken* one in the log.

    CLAUDE.md's fail-closed-is-not-fail-visible rule: a per-point warning cannot
    report a total contract break, so the wholesale boundary — no points at all,
    or every point unusable — logs ERROR, while a partial gap logs a single
    aggregate WARNING naming the count rather than one line per point.

    **Once per distinct condition per process**, like ``_warn_on_capital_drift``
    and ``_warn_on_seed_mismatch`` beside it, and for the reason those two give:
    this runs on a public unauthenticated GET, once per entry, so an undeduped
    line turns one broken curve into twelve lines every time anybody loads the
    board — and an alert channel has a credibility budget that every repeat
    spends. The key carries the *counts*, not just the run, so a curve that
    degrades further still reports the new state.
    """
    total = len(scaled_hourly)
    missing = sum(1 for point in scaled_hourly if point.get("equity") is None)
    if not total:
        message = (
            f"ERROR: leaderboard entry '{entry_id}' (run {run_id}): the stored "
            "run has no equity points at all — the board can draw only its "
            "opening tick. A cached run with no curve is a broken write, not an "
            "empty window."
        )
    elif missing == total:
        message = (
            f"ERROR: leaderboard entry '{entry_id}' (run {run_id}): all {total} "
            "stored equity points are NULL or unparseable — this curve is "
            "broken, not absent."
        )
    elif missing:
        message = (
            f"WARNING: leaderboard entry '{entry_id}' (run {run_id}): {missing} "
            f"of {total} equity points carry no value; the chart draws them as "
            "gaps."
        )
    else:
        return
    key = (entry_id, run_id, f"{missing}/{total}")
    if key in _warned_curve_integrity:
        return
    _warned_curve_integrity.add(key)
    print(message)


def _warn_on_seed_mismatch(
    entry_id: str, run_id: str, stored_initial: float, display_capital: float
) -> None:
    """Say out loud that a published row was rescaled from a seed nobody recorded.

    Narrower than ``_warn_on_capital_drift``, and complementary: that one scans
    the board for rows whose *recorded* seed disagrees with the config, so it is
    structurally blind to a row with no recorded seed at all. This is the only
    report such a row gets, and its seed came from the curve's own first point
    rather than from anything the run wrote down.

    One line per distinct mismatch, not per request, and never alongside the
    board-level warning for the same row — this runs on a public GET, and an
    alert channel has a credibility budget that every duplicate line spends.
    """
    key = (entry_id, stored_initial, display_capital)
    if key in _warned_seed_mismatch:
        return
    _warned_seed_mismatch.add(key)
    print(
        f"WARNING: leaderboard entry '{entry_id}' (run {run_id}) was run at "
        f"${stored_initial:,.2f} but the board publishes ${display_capital:,.2f}; "
        "dollar levels are rescaled for display. Re-run the board at the "
        "published seed to remove the shim (issue #194)."
    )


def _warn_on_prompt_drift(
    entry_id: str, run: Dict[str, Any], wanted_prompt: Optional[str]
) -> None:
    """Say out loud that a cached curve predates the instruction now configured.

    The prompt twin of ``_warn_on_capital_drift``, and it exists because
    ``_cache_match_rank`` returns ``_CACHE_STALE`` for **two** reasons — a
    recorded seed that disagrees, and a recorded prompt that disagrees — while
    only the first had a reporter. ``deploy_model_run`` short-circuits a stale
    row to the cached curve either way (deliberately: re-running on a config
    edit answers one line of JSON with a billable redeploy of the whole board),
    so a changed instruction reused the old curve in total silence. That is
    CLAUDE.md's fail-closed-is-not-fail-visible shape exactly: "nobody changed
    the prompt" and "the prompt changed and is being ignored" were byte-identical
    from outside.

    Once per (entry, run) per process, for the reason the two warnings above it
    give. The prompt text itself is deliberately **not** printed: it is operator
    config that can run to paragraphs, and a log line nobody can read is one
    more way to spend the channel's credibility.
    """
    wanted = _wanted_prompt(wanted_prompt)
    if wanted is None:
        return
    stored = _run_strategy_prompt(run)
    if stored is None or stored == wanted:
        return
    run_id = str(run.get("run_id") or "")
    key = (entry_id, run_id)
    if key in _warned_prompt_drift:
        return
    _warned_prompt_drift.add(key)
    print(
        f"WARNING: leaderboard entry '{entry_id}' (run {run_id}) was computed "
        "under a different strategy_prompt than dashboard/config/leaderboard.json "
        "now configures; the cached curve is published as-is rather than "
        "answering a config edit with a billable re-run. Redeploy this entry "
        "with force_refresh=True to run it under the current instruction."
    )


def _board_capital_base(
    seeds: List[float], display_capital: float
) -> Optional[float]:
    """The one seed this board will publish, or None when there is nothing to publish.

    **The defect is mixed capital, not capital that disagrees with the config.**
    Twelve entries all seeded at $100,000 under a `$10,000` config are mutually
    comparable — the board is internally consistent and merely mislabelled, and
    refusing to publish it would turn a labelling problem into an outage on the
    Competition Leaderboard, which is the site's acquisition hook. Twelve
    entries across two seeds are *not* comparable at any label, because the
    smaller account trades a coarser share quantum: that is the state that
    corrupts a ranking, and the minority is what must not publish.

    So: the largest group of mutually-equal seeds wins, and everything outside
    it is dropped by the caller. Ties go to the group matching the config, then
    to the first seen, so the choice is deterministic rather than dict-ordered.
    **This can never return None for a non-empty ``seeds``** — the winning group
    is the largest, so it always has at least one member — which is the property
    that makes a one-character config typo unable to empty the board.

    ⚠ ``seeds`` holds only the seeds rows actually **recorded**; the caller
    filters. A derived seed is an estimate read off a curve's first point and
    must not get a vote on what the board's base is — see the caller for why.
    So None here means "no row on this board wrote down its seed", not "no
    rows", and the caller answers it by publishing every row unfiltered.
    """
    groups: List[Tuple[float, int]] = []
    for seed in seeds:
        for index, (base, count) in enumerate(groups):
            if abs(base - seed) <= _SEED_MATCH_TOLERANCE:
                groups[index] = (base, count + 1)
                break
        else:
            groups.append((seed, 1))
    if not groups:
        return None
    best = max(
        range(len(groups)),
        key=lambda i: (
            groups[i][1],
            abs(groups[i][0] - display_capital) <= _SEED_MATCH_TOLERANCE,
            -i,
        ),
    )
    return groups[best][0]


def get_leaderboard(
    force_refresh: bool = False,
    period: Optional[str] = "contest",
) -> Dict[str, Any]:
    """Return ranked leaderboard entries with chart-ready equity curves."""
    config = resolve_leaderboard_config(period)
    meta = ensure_leaderboard_runs(force_refresh=force_refresh, config=config)
    session_id = config["session_id"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    strategy_by_id = {s["id"]: s for s in config.get("strategies", [])}

    entries: List[Dict[str, Any]] = []
    board_runs: List[Dict[str, Any]] = []
    display_capital = float(config.get("initial_capital", INITIAL_CAPITAL))

    # FIRST PASS: resolve every entry's run and the seed it was actually run at.
    # Nothing can be scaled until the whole board is known, because the base to
    # scale onto is a property of the board and not of any one row — see
    # `_board_capital_base`.
    resolved: List[Tuple[Dict[str, Any], Dict[str, Any], float, List[Dict[str, Any]]]] = []
    for strategy in config.get("strategies", []):
        run = _find_cached_run(
            strategy["id"],
            start_date,
            end_date,
            session_id,
            display_capital,
            strategy.get("strategy_prompt"),
        )
        if not run:
            continue
        board_runs.append(run)

        equity_hourly = db.get_equity_curve(run["run_id"]) or []
        stored_initial = _stored_seed(run, equity_hourly)
        if stored_initial is None:
            # A published leaderboard row is a claim, and with no recorded seed
            # and no usable first point there is no honest number to make it
            # with. Skipping costs one row; the old config fallback published a
            # curve at an unverified scale, which costs the board's credibility.
            print(
                f"ERROR: leaderboard entry '{strategy['id']}' (run "
                f"{run['run_id']}) records no seed capital and its curve has no "
                "usable first point — omitted from the board rather than "
                "published at a scale nobody verified (issue #365)."
            )
            continue
        resolved.append((strategy, run, stored_initial, equity_hourly))

    # Criterion 3 of issue #365: one board must not mix seed capital. This is
    # the only seam that sees EVERY entry — `ensure_leaderboard_runs` skips the
    # LLM entries before it looks anything up (they deploy manually), which is
    # precisely the half the mixing lives in.
    _warn_on_capital_drift(board_runs, display_capital)
    # ⚠ ONLY A RECORDED SEED VOTES, AND ONLY A RECORDED SEED CAN BE OUTVOTED.
    # `_stored_seed` falls back to the curve's own first equity point, and that
    # point is the equity at the END of the run's first hour — seed plus one
    # hour of P&L. So a run seeded at exactly $100,000 that moved 0.4% in hour
    # one derives $100,400, and comparing THAT against the board's base at a
    # one-cent tolerance is not a capital check, it is a check on whether the
    # strategy happened to be flat at the open. Every derived row that moved at
    # all failed it and vanished from the public Competition board — the same
    # absent-treated-as-a-measurement defect this change exists to remove, and
    # a strictly worse outcome than the unscaled row that used to publish.
    #
    # The tolerance is right for what it was written for: a *recorded* seed is
    # an exact stored number (100000.00000000003 is a float round-trip, not a
    # different capital), so a penny of disagreement there really does mean two
    # runs at two capitals. A derived seed simply cannot carry that signal.
    # Hence the split: recorded seeds decide the base and are held to it;
    # derived rows publish through the scaling shim and are reported by
    # `_warn_on_seed_mismatch` — which is what that warning was written for and
    # was, before this, unreachable whenever the board agreed with its config.
    recorded_seeds = [
        seed for _, run, seed, _ in resolved if _run_seed(run) is not None
    ]
    capital_base = _board_capital_base(recorded_seeds, display_capital)

    # SECOND PASS: publish the entries that share the board's one capital base.
    for strategy, run, stored_initial, equity_hourly in resolved:
        if (
            capital_base is not None
            and _run_seed(run) is not None
            and abs(stored_initial - capital_base) > _SEED_MATCH_TOLERANCE
        ):
            # An outlier in a board that otherwise agrees. Ranking it against
            # the rest would compare runs that were not measured the same way.
            print(
                f"WARNING: leaderboard entry '{strategy['id']}' (run "
                f"{run['run_id']}) was run at ${stored_initial:,.2f} while the "
                f"rest of this board was run at ${capital_base:,.2f} — omitted "
                "rather than ranked against curves it is not comparable with "
                "(issue #365). Re-run this entry at the board's seed: a "
                "force-refresh does that only for an auto_compute baseline, an "
                "LLM entry needs deploy_model_run(force_refresh=True)."
            )
            continue
        # SCALING IS A COMPATIBILITY SHIM, NOT A RE-RUN, and it is only honest
        # for a scale-free strategy. Position sizing, whole-share/lot quanta and
        # per-trade costs make a $10k run a genuinely *different* run from a
        # $100k one rather than a scaled copy of it — a $10k account cannot buy
        # the same basket in the same proportions, so its returns differ, not
        # just its dollar levels. Returns / Sharpe below come off the stored run
        # untouched, so the shim moves the dollar axis only. Re-running each
        # entry at the published seed is the real fix (issue #194).
        scale = display_capital / stored_initial
        if (
            _run_seed(run) is None
            and abs(stored_initial - display_capital) > _SEED_MATCH_TOLERANCE
        ):
            # ONLY THE DERIVED CASE, and the narrowing is the point. A row
            # that recorded its seed is already reported by
            # `_warn_on_capital_drift` over the whole board above, and two lines
            # for one condition is how a log stops being read: an alert channel
            # has a credibility budget, and every duplicate line spends it, the
            # same way a badge that cries wolf stops being looked at. This one
            # covers what that warning structurally cannot see -- `_run_seed` is
            # None here, so the board's own drift scan skips the row entirely.
            _warn_on_seed_mismatch(
                strategy["id"], run["run_id"], stored_initial, display_capital
            )
        scaled_hourly = [
            {
                **pt,
                "equity": _scaled_level(pt.get("equity"), scale),
                "cash": _scaled_level(pt.get("cash"), scale),
                "positions_value": _scaled_level(pt.get("positions_value"), scale),
            }
            for pt in equity_hourly
        ]
        _report_curve_integrity(strategy["id"], run["run_id"], scaled_hourly)
        equity_curve = chart_equity_curve(
            scaled_hourly,
            initial_equity=display_capital,
            start_date=start_date,
        )
        strat = strategy_by_id.get(strategy["id"], strategy)
        is_model = strat.get("strategy") == "llm_agent" or strat.get("label") == "Model"
        # `= display_capital` here was the same defect as the two this change
        # exists for: a run that recorded no final equity was published as an
        # account that finished exactly level, which is a claim nobody made.
        # `_rank_sort_key` already falls back to `cumulative_return` for a None,
        # and both chart readers now render it as an em dash.
        stored_final = run.get("final_equity")
        scaled_final = _scaled_level(stored_final, scale)
        portfolio_value = scaled_final

        entries.append(
            {
                "entry_id": strategy["id"],
                "team_name": run["agent_name"],
                "team_badge": strat.get("label", "Baseline Strategy"),
                "model": strat.get("model", "Baseline"),
                "entry_type": "baseline",
                "is_model": is_model,
                "initial_equity": display_capital,
                "portfolio_value": portfolio_value,
                "cumulative_return": run.get("total_return") or 0,
                "sharpe_ratio": run.get("sharpe_ratio") or 0,
                "max_drawdown": run.get("max_drawdown") or 0,
                "status": "Model" if is_model else "Baseline",
                "run_id": run["run_id"],
                "llm_calls": run.get("llm_calls") or 0,
                "input_tokens": run.get("input_tokens") or 0,
                "output_tokens": run.get("output_tokens") or 0,
                "est_cost_usd": run.get("est_cost_usd") or 0,
                "equity_curve": equity_curve,
            }
        )

    # Yahoo index hours (:30 UTC) vs Alpaca stock hours (:00) — align every
    # chart series onto one shared axis so the frontend does not sparse-null.
    aligned = align_equity_curves([e["equity_curve"] for e in entries])
    for entry, curve in zip(entries, aligned):
        entry["equity_curve"] = curve

    entries = _rank_entries(entries)
    models = [e for e in entries if e.get("is_model")]
    models.sort(key=lambda e: e.get("cumulative_return") or 0, reverse=True)
    if models:
        leader = models[0].get("model") or models[0].get("team_name") or "—"
    else:
        leader = entries[0]["team_name"] if entries else "—"

    daily_status: Optional[Dict[str, Any]] = None
    if config.get("period") == "daily":
        # Schedule *before* snapshotting: the scheduler sets the in-progress flag
        # synchronously, so taking the status first would report
        # refresh_in_progress=false for a worker that had just started — and the
        # frontend only polls while that flag is true, so the user would sit on
        # a stale board until they reloaded by hand.
        maybe_schedule_daily_leaderboard_refresh()
        daily_status = _daily_models_status(config)

    payload: Dict[str, Any] = {
        "period": config.get("period", "contest"),
        "board_title": config.get("board_title", "Competition Leaderboard"),
        "phase_label": config.get("phase_label", "Preseason"),
        "standings_label": config.get("standings_label", "Ranking"),
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "label": daily_window_label(start_date, end_date)
            if config.get("period") == "daily"
            else (f"{start_date} → {end_date}" if start_date != end_date else start_date),
            "description": config.get("description", ""),
        },
        "updated_at": meta.get("refreshed_at"),
        "total_entries": len(entries),
        "display_capital": display_capital,
        "leader": leader,
        "entries": entries,
    }
    if daily_status is not None:
        payload["daily_status"] = daily_status
    if config.get("period") == "live":
        payload["season"] = build_season_payload(config)
    return payload
