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
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import dashboard.backend.infrastructure.llm.token_cost as token_cost
from dashboard.backend.database import db
import dashboard.backend.domain.leaderboard.baselines as _baselines

# Bound by assignment from the module alias rather than a bare
# `from ... import X`. Five of these are used below, but `downsample_daily` is a
# pure re-export: test_service_move asserts `service.downsample_daily is
# baselines.downsample_daily`, a cross-module contract `py/unused-import`
# (intra-file only) cannot see. One import form for the module, so this does not
# trade that alert for `py/import-and-import-from`.
INITIAL_CAPITAL = _baselines.INITIAL_CAPITAL
align_equity_curves = _baselines.align_equity_curves
calc_metrics = _baselines.calc_metrics
chart_equity_curve = _baselines.chart_equity_curve
downsample_daily = _baselines.downsample_daily
fetch_hourly_bars = _baselines.fetch_hourly_bars
from dashboard.backend.domain.leaderboard.strategies._common import reference_start_date
from dashboard.backend.domain.leaderboard.strategies import get_strategy
from dashboard.backend.infrastructure.llm import backtest_harness as llm_harness
from dashboard.backend.paths import CONFIG_DIR, DATA_DIR

LEADERBOARD_MODE = "leaderboard"
VALID_PERIODS = ("contest", "daily")
_SKIP_CACHE_PATH = DATA_DIR / "leaderboard_skip_cache.json"

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


def daily_window_dates(as_of: Optional[date] = None) -> Tuple[str, str]:
    """Most recently completed weekday (Mon–Fri), used as the daily board window."""
    # UTC "yesterday" is intentionally conservative: for the ~4h between a
    # weekday's 16:00 ET close and 00:00 UTC the board still shows the prior
    # completed session rather than a partial/in-progress one. Same-day
    # freshness is driven by the nightly deploy job, not this boundary.
    d = as_of or datetime.now(timezone.utc).date()
    d = d - timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun → roll back to Friday
        d -= timedelta(days=1)
    iso = d.isoformat()
    return iso, iso


def resolve_leaderboard_config(period: Optional[str] = "contest") -> Dict[str, Any]:
    """Return the effective leaderboard config for ``contest`` or ``daily``.

    Daily reuses the same strategy roster as the contest board, but caches under
    a separate session and a rolling 1-day (last completed weekday) window.
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
                f"Daily leaderboard for {start_date} (last completed US weekday). "
                "Baselines recompute on load; LLM entries appear once the daily deploy job runs."
            ),
            "period": "daily",
            "board_title": "Daily Leaderboard",
            "phase_label": "Daily",
            "standings_label": "Ranking",
        }
    return {
        **base,
        "period": "contest",
        "board_title": "Competition Leaderboard",
        "phase_label": "Preseason",
        "standings_label": "Ranking",
    }


def _run_id(strategy_id: str, start_date: str, end_date: str) -> str:
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


def _find_cached_run(strategy_id: str, start_date: str, end_date: str, session_id: str) -> Optional[Dict[str, Any]]:
    for run in db.get_runs_by_session(session_id) or []:
        if (
            run.get("mode") == LEADERBOARD_MODE
            and run.get("start_date") == start_date
            and run.get("end_date") == end_date
            and run.get("llm_model") == strategy_id
        ):
            return run
    return None


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

    missing: List[Dict[str, Any]] = []
    for strategy in config.get("strategies", []):
        if not _auto_compute(strategy):
            continue  # LLM models are deployed manually, never block a request
        strategy_id = strategy["id"]
        if not force_refresh and _find_cached_run(strategy_id, start_date, end_date, session_id):
            continue
        if not force_refresh and _is_skipped(session_id, start_date, end_date, strategy_id, skip_cache):
            continue
        missing.append(strategy)

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
            strategy_id, start_date, end_date, session_id
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
        _reject_if_llm_fallback(
            strategy_id,
            strategy_impl,
            int(getattr(strategy_impl, "llm_calls", 0) or 0),
            llm_decisions=_reported_int(strategy_impl, "llm_decisions"),
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
            metadata=_llm_run_metadata(
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

    existing = _find_cached_run(entry_id, start_date, end_date, session_id)
    if existing and not force_refresh:
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
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        est_cost_usd=est_cost,
        metadata=_llm_run_metadata(
            entry_id,
            entry,
            strategy_impl,
            model_id=model_id,
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
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
    """Official rank is by final portfolio value (nof1-style); tie-break on return."""
    pv = entry.get("portfolio_value")
    if pv is None:
        # Unit tests / partial fixtures may omit dollars — fall back to return.
        pv = entry.get("cumulative_return") or 0
    return (-float(pv), -(entry.get("cumulative_return") or 0))


def _rank_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign official ranks by final portfolio value (higher is better)."""
    if not entries:
        return entries

    entries.sort(key=_rank_sort_key)
    for idx, entry in enumerate(entries):
        entry["rank"] = idx + 1
    return entries


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
    display_capital = float(config.get("initial_capital", INITIAL_CAPITAL))

    for strategy in config.get("strategies", []):
        run = _find_cached_run(strategy["id"], start_date, end_date, session_id)
        if not run:
            continue

        equity_hourly = db.get_equity_curve(run["run_id"]) or []
        stored_initial = float(
            run.get("initial_equity") or config.get("initial_capital", INITIAL_CAPITAL)
        )
        # Display capital may differ from the stored seed run (e.g. $100k seed
        # shown as $1k). Scale dollar levels; returns / Sharpe stay unchanged.
        scale = (display_capital / stored_initial) if stored_initial else 1.0
        scaled_hourly = [
            {
                **pt,
                "equity": float(pt.get("equity") or 0) * scale,
                "cash": float(pt.get("cash") or 0) * scale,
                "positions_value": float(pt.get("positions_value") or 0) * scale,
            }
            for pt in equity_hourly
        ]
        equity_curve = chart_equity_curve(
            scaled_hourly,
            initial_equity=display_capital,
            start_date=start_date,
        )
        strat = strategy_by_id.get(strategy["id"], strategy)
        is_model = strat.get("strategy") == "llm_agent" or strat.get("label") == "Model"
        final_equity = run.get("final_equity")
        if final_equity is None:
            portfolio_value = display_capital
        else:
            portfolio_value = float(final_equity) * scale

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

    return {
        "period": config.get("period", "contest"),
        "board_title": config.get("board_title", "Competition Leaderboard"),
        "phase_label": config.get("phase_label", "Preseason"),
        "standings_label": config.get("standings_label", "Ranking"),
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "label": f"{start_date} → {end_date}" if start_date != end_date else start_date,
            "description": config.get("description", ""),
        },
        "updated_at": meta.get("refreshed_at"),
        "total_entries": len(entries),
        "display_capital": display_capital,
        "leader": leader,
        "entries": entries,
    }
