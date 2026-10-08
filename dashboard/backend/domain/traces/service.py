"""Lifecycle adapters that turn existing Agent runs into trace events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dashboard.backend.domain.traces.repository import trace_store


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_trace_for_run(
    *,
    run: Dict[str, Any],
    initial_input: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a trace and its first event for an already-persisted Run."""

    existing = trace_for_run(run["run_id"])
    if existing is not None:
        return existing

    try:
        trace = trace_store.create_trace(
            agent_id=run.get("agent_id"),
            agent_version_id=run.get("agent_version_id"),
            run_id=run["run_id"],
            user_id=run.get("user_id"),
            initial_input=initial_input or run.get("config") or {},
        )
        trace_store.append_event(
            trace_id=trace["trace_id"],
            event_type="run_started",
            actor_type="system",
            payload={
                "environment_id": run.get("environment_id"),
                "environment_type": run.get("environment_type"),
                "config_summary": initial_input or run.get("config") or {},
            },
            idempotency_key=f"run_started:{run['run_id']}",
        )
    except Exception:
        # A concurrent creator may have won the unique run_id race. Preserve
        # the existing trace when possible; callers may decide whether a trace
        # persistence failure should be visible for their lifecycle.
        existing = trace_for_run(run["run_id"])
        if existing is None:
            raise
        return existing
    return trace_store.get_trace(trace["trace_id"]) or trace


def trace_for_run(run_id: str) -> Optional[Dict[str, Any]]:
    return trace_store.get_trace_for_run(run_id)


def record_decision_event(
    *,
    run_id: str,
    step_id: str,
    decision_id: str,
    actions: list[Dict[str, Any]],
    reasoning_summaries: list[str],
    accepted: bool,
    idempotency_key: str,
) -> Optional[Dict[str, Any]]:
    trace = trace_for_run(run_id)
    if trace is None:
        return None
    return trace_store.append_event(
        trace_id=trace["trace_id"],
        event_type="decision_recorded",
        actor_type="agent",
        step_id=step_id,
        decision_id=decision_id,
        payload={
            "actions": actions,
            "reasoning_summaries": reasoning_summaries,
            "accepted": bool(accepted),
        },
        idempotency_key=f"decision:{run_id}:{step_id}:{idempotency_key}",
    )


def record_execution_event(
    *, run_id: str, step_id: str, decision_id: str, result: Dict[str, Any],
    idempotency_key: str,
) -> Optional[Dict[str, Any]]:
    trace = trace_for_run(run_id)
    if trace is None:
        return None
    return trace_store.append_event(
        trace_id=trace["trace_id"],
        event_type="execution_result",
        actor_type="system",
        step_id=step_id,
        decision_id=decision_id,
        payload={
            "accepted": bool(result.get("accepted")),
            "fills": result.get("fills") or [],
            "executed": result.get("executed") or [],
            "rejected": result.get("rejected") or [],
            "validation": result.get("validation") or {},
            "run_status": result.get("run_status"),
        },
        idempotency_key=f"execution:{run_id}:{step_id}:{idempotency_key}",
    )


def complete_trace(run_id: str, result_summary: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    trace = trace_for_run(run_id)
    if trace is None:
        return None
    trace_store.append_event(
        trace_id=trace["trace_id"],
        event_type="run_completed",
        actor_type="system",
        payload={"result_summary": result_summary or {}},
        idempotency_key=f"run_completed:{run_id}",
    )
    return trace_store.update_trace(
        trace["trace_id"], status="completed", final_output_summary=result_summary or {}, ended_at=_utcnow_iso()
    )


def fail_trace(run_id: str, error_code: str = "run_failed") -> Optional[Dict[str, Any]]:
    trace = trace_for_run(run_id)
    if trace is None:
        return None
    trace_store.append_event(
        trace_id=trace["trace_id"],
        event_type="run_failed",
        actor_type="system",
        payload={"error_code": error_code},
        idempotency_key=f"run_failed:{run_id}:{error_code}",
    )
    return trace_store.update_trace(trace["trace_id"], status="failed", ended_at=_utcnow_iso())


def _best_effort_trace_event(
    *,
    run_id: str,
    event_type: str,
    actor_type: str,
    payload: Dict[str, Any],
    idempotency_key: str,
    **links: Any,
) -> Optional[Dict[str, Any]]:
    """Look up and append observational events without affecting a run."""
    try:
        trace = trace_for_run(run_id)
        if trace is None:
            return None
        return trace_store.append_event(
            trace_id=trace["trace_id"],
            event_type=event_type,
            actor_type=actor_type,
            payload=payload,
            idempotency_key=idempotency_key,
            **links,
        )
    except Exception:
        return None


def record_tool_call(
    *, run_id: str, tool_name: str, input_summary: Dict[str, Any],
    idempotency_key: str, actor_type: str = "agent",
) -> Optional[Dict[str, Any]]:
    return _best_effort_trace_event(
        run_id=run_id, event_type="tool_call", actor_type=actor_type,
        payload={"tool_name": tool_name, "input_summary": input_summary},
        idempotency_key=f"tool_call:{run_id}:{idempotency_key}",
    )


def record_tool_result(
    *, run_id: str, tool_name: str, outcome: str, duration_ms: float,
    result_summary: Optional[Dict[str, Any]] = None,
    error_code: Optional[str] = None, idempotency_key: str,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "tool_name": tool_name, "outcome": outcome,
        "duration_ms": max(0.0, round(float(duration_ms), 3)),
        "result_summary": result_summary or {},
    }
    if error_code:
        payload["error_code"] = error_code
    return _best_effort_trace_event(
        run_id=run_id, event_type="tool_result", actor_type="system",
        payload=payload,
        idempotency_key=f"tool_result:{run_id}:{idempotency_key}",
    )


def record_data_retrieval(
    *, run_id: str, source: str, query_summary: Dict[str, Any],
    result_summary: Dict[str, Any], duration_ms: float, idempotency_key: str,
    outcome: str = "success",
) -> Optional[Dict[str, Any]]:
    return _best_effort_trace_event(
        run_id=run_id, event_type="data_retrieval", actor_type="system",
        payload={
            "source": source, "query_summary": query_summary,
            "result_summary": result_summary,
            "duration_ms": max(0.0, round(float(duration_ms), 3)),
            "outcome": outcome,
        },
        idempotency_key=f"data_retrieval:{run_id}:{idempotency_key}",
    )


def ensure_trace_for_run(
    *, run_id: str, initial_input: Dict[str, Any], run: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Create a provider trace when a legacy engine path has no envelope yet."""
    try:
        existing = trace_for_run(run_id)
        if existing is not None:
            return existing
        record = dict(run or {})
        record["run_id"] = run_id
        return start_trace_for_run(run=record, initial_input=initial_input)
    except Exception:
        # Market-data tracing is observational; a missing trace must never
        # prevent a backtest from loading its provider data.
        return None
