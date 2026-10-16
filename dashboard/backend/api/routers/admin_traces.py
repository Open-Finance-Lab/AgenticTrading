"""Admin-only Agent execution trace endpoints."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from dashboard.backend.api.auth import require_admin
from dashboard.backend.domain.traces.repository import trace_store


router = APIRouter(
    prefix="/admin/traces",
    tags=["admin-traces"],
    dependencies=[Depends(require_admin)],
)


def _json_column(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def _public_trace(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["initial_input"] = _json_column(item.pop("initial_input_json", None))
    item["final_output_summary"] = _json_column(item.get("final_output_summary"))
    return item


@router.get("")
def list_traces(
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    status: Optional[str] = Query(default=None, pattern="^(running|completed|failed)$"),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: int = Query(default=0, ge=0),
):
    page = trace_store.list_traces(
        agent_id=agent_id,
        run_id=run_id,
        status=status,
        limit=limit,
        offset=cursor,
    )
    return {
        "items": [_public_trace(item) for item in page["items"]],
        "has_more": page["has_more"],
        "next_cursor": page["next_cursor"],
    }


@router.get("/{trace_id}")
def get_trace(trace_id: str):
    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return _public_trace(trace)


@router.get("/{trace_id}/events")
def list_trace_events(
    trace_id: str,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=100),
):
    if trace_store.get_trace(trace_id) is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace_store.list_events(
        trace_id, after_sequence=after_sequence, limit=limit
    )


@router.get("/{trace_id}/performance")
def get_trace_performance(trace_id: str):
    """Resolve equity by the exact run ID, never by agent or nearest date."""
    import math
    from dashboard.backend.database import db
    from dashboard.backend.domain.backtesting.metrics import calculate_max_drawdown

    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    run_id = trace.get("run_id")
    run = db.get_run(run_id) if run_id else None
    points = []
    if run:
        for row in db.get_equity_curve(run_id):
            equity = row.get("equity")
            if isinstance(equity, (int, float)) and not isinstance(equity, bool) and math.isfinite(equity):
                points.append({"timestamp": row["timestamp"], "equity": equity})
    # Report the run's own stored figures, so this chart and the backtest panel
    # beside it (and the results page) show one return for one run. The stored
    # total_return is measured against initial capital; agent_runs.initial_equity
    # is the first equity mark, which already carries first-bar fees.
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    stored_return = (run or {}).get("total_return")
    stored_drawdown = (run or {}).get("max_drawdown")
    if finite(stored_drawdown):
        drawdown = abs(stored_drawdown)
    elif points:
        # Metrics use the full series; drawing uses a bounded sample including endpoints.
        drawdown = abs(calculate_max_drawdown(points))
    else:
        drawdown = None
    count = len(points)
    metrics = {
        "return_pct": stored_return * 100 if points and finite(stored_return) else None,
        "max_drawdown_pct": drawdown * 100 if points and drawdown is not None else None,
    }
    if count > 1000:
        points = [points[round(i * (count - 1) / 999)] for i in range(1000)]
    return {
        "run_id": run_id, "source": "run_equity" if points else "unavailable",
        "points": points, "point_count": count, "sampled": count > 1000,
        "metrics": metrics, "partial": trace.get("status") != "completed",
    }


@router.get("/{trace_id}/chart-data")
def get_trace_chart_data(trace_id: str):
    """The Backtest panel's chart payload for this trace's exact run.

    Same builder, market-hour filter and benchmarks as the owner-scoped
    ``/api/backtest/{run_id}/chart-data``; only the lookup differs, by run ID
    across owners, because an administrator reads every user's runs. A trace
    whose run has no stored equity answers ``available: False`` rather than
    404, so the panel can still show its decision snapshots.
    """
    from dashboard.backend.api.routers import backtests as bt
    from dashboard.backend.database import db
    from dashboard.backend.equity_plot import build_backtest_chart_data

    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    run_id = trace.get("run_id")
    run = db.get_run(run_id) if run_id else None
    if run is None:
        return {"available": False, "run_id": run_id}
    agent_curve = bt._filter_equity_for_run(run, db.get_equity_curve(run_id))
    if not agent_curve:
        return {"available": False, "run_id": run_id}
    profile = bt._market_profile_for_run(run)
    try:
        payload = build_backtest_chart_data(
            run_id=run_id,
            agent_name=run.get("agent_name") or "Agent",
            llm_model=run.get("llm_model"),
            start_date=run.get("start_date") or "",
            end_date=run.get("end_date") or "",
            initial_capital=bt._run_initial_capital(run, agent_curve[0].get("equity")),
            agent_curve=agent_curve,
            stored_baselines=bt._stored_buyhold_baseline(run),
            include_market_indexes=profile.index_baseline_enabled,
            market_timezone=profile.timezone,
        )
    except ValueError:
        return {"available": False, "run_id": run_id}
    # The comparison model reads these to pick its columns (A-share runs have
    # no US index benchmarks) and to match the stored buy-and-hold series.
    metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
    payload["run"] = {
        "data_source": metadata.get("data_source"),
        "baseline_buyhold_run_id": run.get("baseline_buyhold_run_id"),
        "reporting_currency": metadata.get("reporting_currency"),
    }
    return {"available": True, **payload}


@router.get("/{trace_id}/export")
def export_trace(trace_id: str, format: str = Query(default="json", pattern="^(json|markdown)$")):
    from fastapi.responses import StreamingResponse
    from starlette.background import BackgroundTask
    from dashboard.backend.domain.traces.export import build_export

    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    file = build_export(trace_store, _public_trace(trace), format)
    def chunks():
        try:
            while chunk := file.read(65536):
                yield chunk
        finally:
            file.close()
    extension = 'json' if format == 'json' else 'md'
    return StreamingResponse(chunks(), media_type='application/json' if format == 'json' else 'text/markdown',
                             headers={'Content-Disposition': f'attachment; filename="agent-trace.{extension}"', 'Cache-Control': 'no-store'},
                             background=BackgroundTask(file.close))


@router.get("/{trace_id}/backtest")
def get_trace_backtest(trace_id: str):
    """Administrator view across owners, resolved only by the trace's run ID."""
    from dashboard.backend.database import db
    from dashboard.backend.domain.traces.export import redact
    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    run = db.get_run(trace.get('run_id')) if trace.get('run_id') else None
    if run is None:
        return {'available': False, 'run_id': trace.get('run_id')}
    fields = ('run_id', 'agent_name', 'llm_model', 'mode', 'start_date', 'end_date',
              'initial_equity', 'final_equity', 'total_return', 'max_drawdown', 'sharpe_ratio',
              'num_trades', 'llm_calls', 'llm_decisions', 'input_tokens', 'output_tokens', 'est_cost_usd')
    return {'available': True, 'run_id': trace['run_id'],
            'run': {key: run.get(key) for key in fields},
            'configuration': redact(run.get('metadata') or {})}
