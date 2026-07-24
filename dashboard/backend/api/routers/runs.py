"""Run API — generalized Agent-Environment Protocol execution.

Wraps the existing backtest engine via ``run_service``. External agents
authenticate with their Agent API key (``X-API-Key``); the legacy
session-based ``/api/v1/backtest/*`` endpoints remain available for
backward compatibility.

Canonical location (Phase 3B3). Moved verbatim from
``dashboard/backend/api/runs.py``, which is now a thin compatibility re-export
shim. Endpoint paths, methods, names, prefixes, tags, schemas, status codes,
exception messages, auth dependencies, and Run service calls are unchanged; only
the module location moved.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

import dashboard.backend.domain.runs.service as run_service
from dashboard.backend.api.protocol_auth import resolve_agent_by_key
from dashboard.backend.domain.agents.version_repository import agent_version_store
from dashboard.backend.domain.runs.environment import default_environment_id
from dashboard.backend.domain.runs.protocol import DecisionIn, ProtocolError, error_body
from dashboard.backend.domain.runs.repository import run_store

router = APIRouter(prefix="/v1/runs", tags=["runs"])


class EnvironmentRef(BaseModel):
    type: str = Field(default="backtest")
    environment_id: Optional[str] = None


class CreateRunBody(BaseModel):
    agent_version_id: Optional[str] = None
    environment: EnvironmentRef = Field(default_factory=EnvironmentRef)
    config: Dict[str, Any] = Field(default_factory=dict)


def _handle_protocol_error(exc: ProtocolError):
    raise HTTPException(status_code=exc.status_code, detail=exc.to_body(),
                        headers=exc.headers or None)


def _require_run_owner(run_id: str, agent: Dict[str, Any]) -> Dict[str, Any]:
    record = run_store.get_run(run_id)
    # Fail closed AND no existence oracle: a run owned by another agent — or an
    # orphaned/legacy row with no owner agent_id — answers exactly like a
    # missing run. A 403-vs-404 split would let any key holder enumerate which
    # run ids exist (same convention as create_run's agent_version lookup).
    if (
        not record
        or not record.get("agent_id")
        or record["agent_id"] != agent["agent_id"]
    ):
        raise HTTPException(
            status_code=404, detail=error_body("run_not_found", "Run not found")
        )
    return record


@router.post("")
def create_run(
    body: CreateRunBody,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    agent = resolve_agent_by_key(x_api_key)

    agent_version: Dict[str, Any] = {}
    if body.agent_version_id:
        agent_version = agent_version_store.get_version(body.agent_version_id) or {}
        # Another agent's version answers exactly like a nonexistent one: a
        # 403-vs-404 split would let any key holder enumerate version ids.
        if not agent_version or agent_version["agent_id"] != agent["agent_id"]:
            raise HTTPException(
                status_code=404,
                detail=error_body("agent_version_not_found", "agent_version_id not found"),
            )

    environment_id = body.environment.environment_id or default_environment_id()
    try:
        return run_service.create_run(
            agent=agent,
            agent_version=agent_version,
            environment_id=environment_id,
            config=body.config,
        )
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}")
def get_run(run_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.run_view(run_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}/status")
def get_run_status(run_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.run_status(run_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}/steps/next")
def get_next_step(run_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.get_next_step(run_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}/steps/{step_id}")
def get_step(
    run_id: str,
    step_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.get_step(run_id, step_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.post("/{run_id}/steps/{step_id}/decision")
def submit_step_decision(
    run_id: str,
    step_id: str,
    body: DecisionIn,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    if body.run_id and body.run_id != run_id:
        raise HTTPException(
            status_code=400,
            detail=error_body("run_id_mismatch", "Body run_id does not match path run_id"),
        )
    if body.step_id and body.step_id != step_id:
        raise HTTPException(
            status_code=400,
            detail=error_body("step_id_mismatch", "Body step_id does not match path step_id"),
        )
    try:
        return run_service.submit_decision(run_id, step_id, body)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}/steps")
def list_steps(run_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.list_steps(run_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}/decisions")
def list_decisions(run_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.list_decisions(run_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}/trades")
def list_trades(run_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.list_trades(run_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}/metrics")
def get_metrics(run_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.get_metrics(run_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)


@router.get("/{run_id}/result")
def get_result(run_id: str, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    agent = resolve_agent_by_key(x_api_key)
    _require_run_owner(run_id, agent)
    try:
        return run_service.get_result(run_id)
    except ProtocolError as exc:
        _handle_protocol_error(exc)
