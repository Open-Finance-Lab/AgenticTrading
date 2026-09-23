"""Research-agent routes (design N2/PR2).

The /admin-style analogue of the marketplace for report-output agents: the
catalog lists them (shelf ``research``), "add" is the research clone, and the
workbench's runs proxy to the agent's external service via the N2/PR2 contract
(manifest / runs / status / result — see research-agent-integration/contract.md).

Auth model: everything requires a signed-in user (runs and reports are
per-user; a due-diligence report is not platform-public content). Service-side
calls carry ``X-Service-Token`` from RESEARCH_SERVICE_TOKEN.

Completion side effects happen on the poll that *discovers* completion (the
frontend polls every ~15s while a run is open; there is no background sweeper
in v1). If the user closes the browser, the email fires the next time anyone
fetches that run's status — acceptable for v1, noted in the route docstring.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.domain.agents import marketplace as marketplace_mod
from dashboard.backend.domain.agents import research_store

router = APIRouter(prefix="/v1/research", tags=["research"])

POLL_TIMEOUT_SECONDS = 10.0
MANIFEST_CACHE_SECONDS = 300.0
_manifest_cache: Dict[str, Any] = {}

ARTIFACT_CONTENT_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "markdown": "text/markdown; charset=utf-8",
    "markdown_report": "text/markdown; charset=utf-8",
    "evidence_json": "application/json",
    "evidence": "application/json",
}


def _service_token() -> str:
    return os.getenv("RESEARCH_SERVICE_TOKEN", "dev-token")


def _template_or_404(template_id: str) -> Dict[str, Any]:
    template = marketplace_mod.get_marketplace_template(template_id)
    if not template or not marketplace_mod.shelf_is_research(template):
        raise HTTPException(status_code=404, detail="Research template not found")
    return template


def _service_base(template: Dict[str, Any]) -> str:
    return marketplace_mod.research_service_config(template)["base_url"]


def _agent_id(template: Dict[str, Any]) -> str:
    return (template.get("research") or {}).get("agent_id", "")


def _service_headers() -> Dict[str, str]:
    return {"X-Service-Token": _service_token()}


def _service_error(exc: httpx.HTTPError, action: str) -> HTTPException:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 401:
        return HTTPException(status_code=502, detail="Research service rejected its token")
    if status == 404:
        return HTTPException(status_code=404, detail="Research service does not know this run")
    if status == 422:
        try:
            detail = exc.response.json()
        except Exception:
            detail = {"detail": "validation failed"}
        return HTTPException(status_code=422, detail=detail)
    return HTTPException(status_code=503, detail=f"Research service unavailable while {action}")


def _cached_manifest(template: Dict[str, Any]) -> Dict[str, Any]:
    import time

    cache_key = _agent_id(template)
    cached = _manifest_cache.get(cache_key)
    now = time.time()
    if cached and now - cached["at"] < MANIFEST_CACHE_SECONDS:
        return cached["data"]
    response = httpx.get(
        f"{_service_base(template)}/manifest",
        params={"agent_id": _agent_id(template)},
        headers=_service_headers(),
        timeout=POLL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    _manifest_cache[cache_key] = {"at": now, "data": data}
    return data


def _public_agent_card(template: Dict[str, Any]) -> Dict[str, Any]:
    """Catalog card + added-state for the Community / My Agents shelves."""
    public = marketplace_mod._public_template(template)
    return public


@router.get("/agents")
def list_research_agents(current_user: dict = Depends(get_current_user)):
    added = set(research_store.list_added_template_ids(current_user["id"]))
    items = [
        _public_agent_card(raw)
        for raw in marketplace_mod._load_catalog().values()
        if marketplace_mod.shelf_is_research(raw)
    ]
    for item in items:
        item["added"] = item["template_id"] in added
    return {"agents": items}


@router.post("/agents/{template_id}/add")
def add_research_agent(template_id: str, current_user: dict = Depends(get_current_user)):
    _template_or_404(template_id)
    created = research_store.add_research_agent(current_user["id"], template_id)
    return {"added": True, "created": created}


@router.delete("/agents/{template_id}/add")
def remove_research_agent(template_id: str, current_user: dict = Depends(get_current_user)):
    removed = research_store.remove_research_agent(current_user["id"], template_id)
    return {"removed": removed}


@router.get("/agents/{template_id}/manifest")
def get_manifest(template_id: str, current_user: dict = Depends(get_current_user)):
    template = _template_or_404(template_id)
    try:
        return _cached_manifest(template)
    except httpx.HTTPError as exc:
        raise _service_error(exc, "loading the agent manifest") from None


@router.post("/agents/{template_id}/runs")
def create_research_run(
    template_id: str,
    body: dict,
    current_user: dict = Depends(get_current_user),
):
    template = _template_or_404(template_id)
    settings = body.get("settings")
    if not isinstance(settings, dict):
        raise HTTPException(status_code=422, detail={"detail": "settings object required"})
    try:
        manifest = _cached_manifest(template)
    except httpx.HTTPError as exc:
        raise _service_error(exc, "loading the agent manifest") from None

    # Server-side required-field check so a stale frontend cannot silently
    # submit an incomplete mandate (the service would 422 anyway; this gives
    # the same field_errors shape without depending on its copy).
    field_errors = {
        field["id"]: "required"
        for field in manifest.get("settings_schema", {}).get("fields", [])
        if field.get("required") and not str(settings.get(field["id"]) or "").strip()
    }
    if field_errors:
        raise HTTPException(
            status_code=422,
            detail={"detail": "validation failed", "field_errors": field_errors},
        )

    email_me = bool(body.get("email_me"))
    payload = {"agent_id": _agent_id(template), "settings": settings}
    try:
        response = httpx.post(
            f"{_service_base(template)}/runs",
            json=payload,
            headers=_service_headers(),
            timeout=POLL_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        service_run = response.json()
    except httpx.HTTPError as exc:
        raise _service_error(exc, "submitting the research run") from None

    import uuid

    service_run_id = str(service_run.get("run_id") or "").strip()
    run_id = f"rr_{service_run_id}" if service_run_id else f"rr_{uuid.uuid4().hex[:12]}"
    research_store.create_run(
        run_id=run_id,
        user_id=current_user["id"],
        template_id=template_id,
        service_run_id=service_run_id,
        status=str(service_run.get("status") or "running"),
        settings=settings,
        email_me=email_me,
    )
    return {"run_id": run_id, "status": service_run.get("status") or "running"}


def _service_run_id(run: Dict[str, Any]) -> str:
    return str(run.get("service_run_id") or "")


def _maybe_complete_run(run: Dict[str, Any], template: Dict[str, Any],
                        current_user: dict) -> Dict[str, Any]:
    """Poll the service once for a non-terminal run; on first discovery of
    completion, store artifacts and send the notification email."""
    if run["status"] in ("completed", "failed"):
        return run
    try:
        response = httpx.get(
            f"{_service_base(template)}/runs/{_service_run_id(run)}",
            headers=_service_headers(),
            timeout=POLL_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        service_status = response.json()
    except httpx.HTTPError:
        # A single poll failure is tolerated — the next frontend poll retries.
        return run

    status = str(service_status.get("status") or "running")
    if status == "failed":
        research_store.update_run_status(run["run_id"], "failed",
                                         error=service_status.get("error") or "Research failed",
                                         completed=True)
        run["status"] = "failed"
        return run
    if status != "completed":
        return run

    try:
        result = httpx.get(
            f"{_service_base(template)}/runs/{_service_run_id(run)}/result",
            headers=_service_headers(),
            timeout=POLL_TIMEOUT_SECONDS,
        )
        result.raise_for_status()
        payload = result.json()
    except httpx.HTTPError:
        research_store.update_run_status(run["run_id"], "failed",
                                         error="Result could not be retrieved",
                                         completed=True)
        run["status"] = "failed"
        return run

    research_store.store_artifacts(
        run["run_id"],
        payload.get("artifacts") or {},
        payload.get("evidence"),
        payload.get("report_markdown") or "",
    )
    research_store.update_run_status(run["run_id"], "completed", completed=True)
    run["status"] = "completed"

    # Notification email (link, not attachment — Brevo sender is plain-text v1).
    if run.get("email_me") and not run.get("emailed"):
        from dashboard.backend.infrastructure.email.sender import email_configured, send_email

        if email_configured():
            base = os.getenv("PUBLIC_BASE_URL", "https://agentic-trading-lab.vercel.app")
            link = f"{base}/app?view=research"
            sent = send_email(
                current_user["email"],
                f"[ATL] Your research report is ready — {template['name']}",
                "Your research report has completed.\n\n"
                f"Open it here: {link}\n"
                "(The report page offers Markdown / DOCX / PDF downloads.)\n",
            )
            if sent:
                research_store.mark_emailed(run["run_id"])
    return run


@router.get("/runs")
def list_my_runs(current_user: dict = Depends(get_current_user)):
    runs = research_store.list_runs_for_user(current_user["id"])
    for run in runs:
        try:
            run["settings"] = __import__("json").loads(run.pop("settings_json") or "{}")
        except Exception:
            run["settings"] = {}
    return {"runs": runs}


@router.get("/runs/{run_id}")
def get_run_status(run_id: str, current_user: dict = Depends(get_current_user)):
    run = research_store.get_run(run_id, current_user["id"])
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    template = _template_or_404(run["template_id"])
    run = _maybe_complete_run(run, template, current_user)
    return {
        "run_id": run["run_id"],
        "template_id": run["template_id"],
        "status": run["status"],
        "error": run.get("error"),
        "created_at": run["created_at"],
        "completed_at": run["completed_at"],
    }


def _run_and_template_or_404(run_id: str, user: dict):
    run = research_store.get_run(run_id, user["id"])
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] != "completed":
        raise HTTPException(status_code=409, detail="Run not completed")
    template = _template_or_404(run["template_id"])
    return run, template


@router.get("/runs/{run_id}/report")
def get_run_report(run_id: str, current_user: dict = Depends(get_current_user)):
    run, _template = _run_and_template_or_404(run_id, current_user)
    artifact = research_store.get_artifact(run_id, "markdown")
    if not artifact:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "run_id": run_id,
        "template_id": run["template_id"],
        "report_markdown": artifact["content_base64"],
        "filename": artifact["filename"],
    }


@router.get("/runs/{run_id}/artifacts/{kind}")
def download_artifact(run_id: str, kind: str, current_user: dict = Depends(get_current_user)):
    run, _template = _run_and_template_or_404(run_id, current_user)
    artifact = research_store.get_artifact(run_id, kind)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    content = artifact["content_base64"] or ""
    if artifact["kind"] in ("markdown_report", "evidence_json"):
        # These two are stored as plain text, not base64.
        return Response(
            content=content,
            media_type=ARTIFACT_CONTENT_TYPES[artifact["kind"]],
            headers={"Content-Disposition": f'attachment; filename="{artifact["filename"]}"'},
        )
    try:
        raw = base64.b64decode(content)
    except Exception:
        raise HTTPException(status_code=500, detail="Artifact payload is corrupt")
    return Response(
        content=raw,
        media_type=ARTIFACT_CONTENT_TYPES.get(artifact["kind"], "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{artifact["filename"]}"'},
    )
