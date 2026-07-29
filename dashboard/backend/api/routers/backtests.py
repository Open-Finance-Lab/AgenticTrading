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
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pytz
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
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
from dashboard.backend.infrastructure.llm.providers import (
    LLMProviderConfigurationError,
    ensure_llm_client_available,
)
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
    market_index_baselines_for_run,
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
        ):
            if field in metadata:
                payload[field] = metadata[field]
    return RunMetadata(**payload)


# ============================================================================
# Background backtest state + worker
# ============================================================================

# Global state for background backtest
backtest_status = {
    "running": False,
    "error": None,
    "runs_count": 0,
    "started_at": None,
    "progress_file": None,
    "live_run_id": None,
}
backtest_session_id = None  # Track which session owns the running backtest


def _read_backtest_progress() -> Optional[Dict[str, Any]]:
    """Load incremental equity snapshots written by the backtest subprocess."""
    progress_file = backtest_status.get("progress_file")
    if not progress_file:
        return None
    path = Path(progress_file)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None

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
):
    """Run backtest in background thread."""
    global backtest_status, backtest_session_id

    strategy_prompt_path = None
    pipeline_path = None
    runtime_config_path = None
    progress_file = None
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
        
        backtest_status["running"] = True
        backtest_status["error"] = None
        backtest_status["started_at"] = time.time()
        backtest_session_id = session_id  # Store session for status polling

        if not live_run_id:
            live_run_id = (
                f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            )
        progress_file = str(Path(tempfile.gettempdir()) / f"backtest_progress_{live_run_id}.json")
        backtest_status["live_run_id"] = live_run_id
        backtest_status["progress_file"] = progress_file
        
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

        cmd += ["--run-id", live_run_id, "--progress-file", progress_file]

        # Simulation capital is independent of the agent's portfolio sleeve.
        cmd += ["--initial-capital", str(resolve_initial_capital(initial_capital))]

        if assets:
            cmd += ["--assets", ",".join(assets)]
            print(f"   Assets: {', '.join(assets)}", flush=True)

        print(f"📋 Running: {' '.join(cmd)}", flush=True)
        
        result = subprocess.run(
            cmd,
            cwd=str(DASHBOARD_DIR),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes for LLM backtest (longer than rule-based)
            env=env
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
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            summary = _sanitize_backtest_error(
                error_msg,
                500,
                extra_secret=financial_datasets_api_key,
            )
            backtest_status["error"] = (
                f"Backtest failed with return code {result.returncode}. {summary}"
            )
            print(f"❌ Backtest failed (returncode={result.returncode})", flush=True)
        else:
            runs = db.get_runs_by_mode("backtest")
            backtest_status["runs_count"] = len(runs)
            print(f"✅ Backtest completed. Found {len(runs)} runs in database.", flush=True)
            if len(runs) > 0:
                print(f"   Latest run IDs: {[r['run_id'] for r in runs[:3]]}", flush=True)
            _maybe_writeback_adapted_pipeline(agent_id, live_run_id)
    except Exception as e:
        summary = _sanitize_backtest_error(
            e,
            500,
            extra_secret=financial_datasets_api_key,
        )
        backtest_status["error"] = summary
        print(f"❌ Backtest exception: {summary}", flush=True)
    finally:
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
                pass
        print("✋ Backtest background thread finished", flush=True)


def _redact_credentials(text: object, extra_secret: Optional[str] = None) -> str:
    """Strip credentials from text without dropping any of it.

    Kept separate from truncation on purpose: the subprocess log dump needs
    redaction over its FULL length, while only the operator-facing error
    summary needs a length bound.
    """
    message = str(text)
    token = os.getenv("IFIND_ACCESS_TOKEN", "").strip()
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


def _maybe_writeback_adapted_pipeline(agent_id: Optional[str], run_id: Optional[str]) -> None:
    """Persist post-trade adapted pipeline back onto the agent row."""
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
async def run_backtest_endpoint(
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

    # Guard operator LLM spend BEFORE scheduling anything. Validation first (so a
    # caller correcting a bad request isn't charged rate budget for a typo), then
    # the per-client run budget.
    _validate_backtest_params(start_date, end_date, strategy_prompt, model, pipeline)

    if (
        decision_source_was_explicit
        and resolved_decision_source == LLM_DECISION_SOURCE
        and runtime_type == PIPELINE_RUNTIME_TYPE
    ):
        try:
            ensure_llm_client_available()
        except LLMProviderConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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
    
    if backtest_status["running"]:
        print(f"⚠️ Backtest already running, rejecting request", flush=True)
        return {
            "success": False,
            "error": "Backtest already running. Please wait for it to complete."
        }

    # Mint run id before the worker starts so callers (Discord job watcher)
    # can key notifications on a stable id from the HTTP response.
    live_run_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Publish "running" synchronously — before thread.start() — so a status poll
    # landing in the gap between this return and the worker's first line cannot
    # read a PRIOR run's completed state (running=False, runs_count>0) and report
    # it as this run's result. Clearing runs_count/error retires the previous
    # run's terminal signal; setting live_run_id lets the watcher key on this
    # exact run. (PR #163 completion-detection race.) Item assignment on this
    # global dict is atomic under the GIL, same as the worker's own writes.
    global backtest_session_id
    backtest_status["running"] = True
    backtest_status["error"] = None
    backtest_status["runs_count"] = 0
    backtest_status["started_at"] = time.time()
    backtest_status["live_run_id"] = live_run_id
    backtest_status["progress_file"] = None
    backtest_session_id = session_id

    # Start backtest in background thread
    print(f"🧵 Starting background thread for backtest", flush=True)
    # Keyword args, not positional: this call passes 14 of them and universe /
    # timeframe were inserted mid-signature. By name, a future insertion in the
    # wrong slot is a TypeError instead of a silently shifted argument.
    thread = threading.Thread(
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
        },
        daemon=True
    )
    thread.start()
    
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
    return response

@router.get("/backtest/status")
async def get_backtest_status(request: Request):
    """Get backtest status (running, error, or completed)."""
    session_id = request.state.session_id
    
    if backtest_status["running"]:
        elapsed = 0
        started_at = backtest_status.get("started_at")
        if started_at:
            elapsed = max(0, int(time.time() - started_at))
        progress = _read_backtest_progress()
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
            "live_run_id": backtest_status.get("live_run_id"),
            "session_id": backtest_session_id,
        }
        if progress:
            payload["progress"] = progress
        return payload
    elif backtest_status["error"]:
        return {
            "running": False,
            "error": backtest_status["error"],
            "message": "Backtest failed"
        }
    elif backtest_status["runs_count"] > 0:
        # Verify the completed backtest belongs to this session
        runs = db.get_runs_by_session(session_id)
        if not runs:
            return {
                "running": False,
                "error": "Backtest completed but no runs found for this session",
                "message": "Session mismatch"
            }
        
        return {
            "running": False,
            "success": True,
            "runs_count": backtest_status["runs_count"],
            "session_id": session_id,
            "message": "Backtest completed successfully"
        }
    else:
        return {
            "running": False,
            "message": "No backtest has been run yet"
        }


# ============================================================================
# Backtest Routes
# ============================================================================

@router.get("/api/backtest/runs", response_model=List[RunMetadata])
async def get_backtest_runs(request: Request):
    """Get all backtest runs for this session."""
    session_id = get_session_id_from_request(request)
    runs = db.get_runs_by_session(session_id)
    runs = [r for r in runs if r['mode'] == 'backtest']
    return [_run_metadata_response(run) for run in runs]


# IMPORTANT: Register /compare/latest BEFORE /{run_id} to prevent {run_id} from matching "compare/latest"

@router.get("/api/backtest/compare/latest", response_model=ComparisonResponse)
async def compare_latest_backtests(request: Request):
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
async def get_backtest_chart_data(run_id: str, request: Request):
    """Chart-ready equity series for the Playground backtest page.

    Uses the same DJIA index + Nasdaq-100 baselines and gapless market-hour
    x-axis as ``/runs/{run_id}/plot.png`` (Discord chart).
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
            stored_baselines=(
                _stored_buyhold_baseline(run)
                if not profile.index_baseline_enabled
                else []
            ),
            include_market_indexes=profile.index_baseline_enabled,
            market_timezone=profile.timezone,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return BacktestChartData(**payload)


@router.get("/api/backtest/{run_id}", response_model=EquityCurve)
async def get_backtest_run(run_id: str, request: Request):
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
async def get_latest_metrics(request: Request):
    """Get metrics for the latest Agent backtest run in this session (excludes baselines)."""
    session_id = request.state.session_id
    runs = [r for r in db.get_runs_by_session(session_id) or [] 
            if r['mode'] == 'backtest' and r['agent_name'] == 'Agent']
    if not runs:
        raise HTTPException(status_code=404, detail="No Agent backtest runs found for this session")
    
    latest_run = max(runs, key=lambda r: r['created_at'])
    return _run_metadata_response(latest_run)


@router.get("/runs", response_model=List[RunMetadata])
async def get_runs(request: Request, mode: Optional[str] = None):
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
async def get_run(run_id: str, request: Request):
    """Get metadata for a specific run."""
    session_id = request.state.session_id
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or not yours")
    return _run_metadata_response(run)


@router.get("/runs/{run_id}/equity", response_model=EquityCurve)
async def get_equity_curve(run_id: str, request: Request):
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
async def get_run_trades(run_id: str, request: Request):
    """Trade log for a backtest run owned by this session."""
    session_id = request.state.session_id
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or not yours")
    trades = db.get_trades(run_id)
    return {"run_id": run_id, "trades": trades, "count": len(trades)}


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
    return Response(content=_render_run_plot_png(run_id), media_type="image/png")


@lru_cache(maxsize=128)
def _render_run_plot_png(run_id: str) -> bytes:
    """Render (and memoize) the equity-curve comparison PNG for ``run_id``.

    A run's equity data is immutable once written and run_ids are unique per
    run, so the rendered bytes are reused without re-querying the DB or
    re-rendering. HTTPExceptions (missing run / no equity data) are raised, not
    cached — so data that appears later is still picked up on a retry.
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
    if profile.index_baseline_enabled:
        baselines = market_index_baselines_for_run(
            timestamps,
            run.get("start_date") or "",
            run.get("end_date") or "",
            initial_capital,
        )
    else:
        baselines = [
            (label, baseline_run_id, align_equity(
                timestamps, equity_lookup(curve)
            ))
            for label, baseline_run_id, curve in _stored_buyhold_baseline(run)
        ]

    try:
        return render_backtest_equity_png(
            agent_label=agent_label,
            agent_run_id=run_id,
            timestamps=timestamps,
            agent_values=agent_values,
            baselines=baselines,
            market_timezone=profile.timezone,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/compare", response_model=ComparisonResponse)
async def compare_runs(run_ids: str, request: Request):
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
