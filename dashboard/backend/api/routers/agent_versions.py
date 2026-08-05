"""AgentVersion API — immutable agent configuration snapshots.

Canonical location (Phase 3A3). Moved verbatim from
``dashboard/backend/api/agent_versions.py``, which is now a thin compatibility
re-export shim. Endpoint paths, prefixes, tags, schemas, status codes,
exception messages, auth behavior, and ``AgentService`` calls are unchanged;
only the module location moved.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from dashboard.backend.domain.agents.service import (
    InvalidVersionFieldError,
    agent_service,
)
from dashboard.backend.api.protocol_auth import require_agent_access

router = APIRouter(prefix="/v1", tags=["agent-versions"])


class CreateVersionBody(BaseModel):
    version: str = Field(default="0.1.0", min_length=1, max_length=50)
    execution_mode: str = Field(default="external")
    architecture: Optional[str] = Field(default=None, max_length=100)
    model_backbones: List[str] = Field(default_factory=list)
    decision_frequency: str = Field(default="1h", max_length=20)
    code_commit: Optional[str] = Field(default=None, max_length=200)
    prompt_hash: Optional[str] = Field(default=None, max_length=128)
    config_hash: Optional[str] = Field(default=None, max_length=128)
    prompt: Optional[str] = Field(default=None, description="Raw prompt; hashed server-side if prompt_hash omitted")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Raw config; hashed if config_hash omitted")
    verification_level: str = Field(default="self_reported")


@router.post("/agents/{agent_id}/versions")
def create_agent_version(
    agent_id: str,
    body: CreateVersionBody,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """Create a new immutable version snapshot for an agent."""
    require_agent_access(
        agent_id, request=request, x_api_key=x_api_key, authorization=authorization
    )
    try:
        version = agent_service.create_version(
            agent_id=agent_id,
            version=body.version.strip(),
            execution_mode=body.execution_mode,
            architecture=body.architecture,
            model_backbones=body.model_backbones,
            decision_frequency=body.decision_frequency,
            code_commit=body.code_commit,
            prompt_hash=body.prompt_hash,
            config_hash=body.config_hash,
            prompt=body.prompt,
            config=body.config,
            verification_level=body.verification_level,
        )
    except InvalidVersionFieldError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"agent_version": version}


@router.get("/agents/{agent_id}/versions")
def list_agent_versions(
    agent_id: str,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """List versions for an agent (newest first)."""
    require_agent_access(
        agent_id, request=request, x_api_key=x_api_key, authorization=authorization
    )
    return {"agent_id": agent_id, "versions": agent_service.list_versions(agent_id)}


@router.get("/agent-versions/{agent_version_id}")
def get_agent_version(
    agent_version_id: str,
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """Fetch a single immutable agent version."""
    version = agent_service.get_version(agent_version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Agent version not found")
    try:
        require_agent_access(
            version["agent_id"], request=request, x_api_key=x_api_key, authorization=authorization
        )
    except HTTPException as exc:
        if exc.status_code in (401, 403):
            # The version id alone doesn't say which agent to authorize
            # against, so the row is looked up before auth. Any auth failure
            # must then collapse into the same 404 a nonexistent id gets —
            # otherwise the 401/403-vs-404 split is an existence oracle for
            # agent-version ids.
            raise HTTPException(status_code=404, detail="Agent version not found")
        raise
    return {"agent_version": version}
