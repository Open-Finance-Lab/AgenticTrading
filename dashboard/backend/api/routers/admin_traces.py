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
    # Metrics use the full series; drawing uses a bounded sample including endpoints.
    peak = None
    drawdown = 0.0
    for point in points:
        peak = point["equity"] if peak is None else max(peak, point["equity"])
        if peak > 0:
            drawdown = max(drawdown, (peak - point["equity"]) / peak)
    count = len(points)
    initial = (run or {}).get("initial_equity")
    baseline = initial if isinstance(initial, (int, float)) and math.isfinite(initial) and initial >= 0 else (points[0]["equity"] if points else 0)
    metrics = {
        "return_pct": ((points[-1]["equity"] / baseline - 1) * 100) if points and baseline > 0 else None,
        "max_drawdown_pct": drawdown * 100 if points else None,
    }
    if count > 1000:
        points = [points[round(i * (count - 1) / 999)] for i in range(1000)]
    return {
        "run_id": run_id, "source": "run_equity" if points else "unavailable",
        "points": points, "point_count": count, "sampled": count > 1000,
        "metrics": metrics, "partial": trace.get("status") != "completed",
    }
