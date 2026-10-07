"""Lifecycle adapters that turn existing Agent runs into trace events."""

from __future__ import annotations

from typing import Any, Dict, Optional

from dashboard.backend.domain.traces.repository import trace_store


def start_trace_for_run(
    *,
    run: Dict[str, Any],
    initial_input: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a trace and its first event for an already-persisted Run."""

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
        trace["trace_id"], status="completed", final_output_summary=result_summary or {}
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
    return trace_store.update_trace(trace["trace_id"], status="failed")
