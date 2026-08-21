"""Admin-only provider allowlist and encrypted platform credential APIs."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from dashboard.backend.api.auth import require_admin
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.domain.model_providers.models import (
    AdminPlatformCredentialRequest,
    AdminProviderActionRequest,
    AdminProviderRequest,
)
from dashboard.backend.domain.model_providers.repository_common import (
    CredentialConflictError,
    InvalidProviderOriginError,
    ModelProviderStoreError,
    ProviderNotFoundError,
)
from dashboard.backend.domain.model_providers.service import model_provider_service


router = APIRouter(
    prefix="/admin/model-providers",
    tags=["admin-model-providers"],
    dependencies=[Depends(require_admin)],
)

_ADMIN_MODEL_PROVIDER_LIMITER = FixedWindowRateLimiter(
    max_events=60,
    window_seconds=300,
)
_MAX_CREDENTIAL_BODY_BYTES = 16 * 1024


def reset_admin_model_provider_limiter() -> None:
    """Clear the process-local mutation budget for isolated tests."""
    _ADMIN_MODEL_PROVIDER_LIMITER.reset()


def _limit_mutation(admin_user_id: int) -> None:
    key = f"admin-model-providers:{int(admin_user_id)}"
    if _ADMIN_MODEL_PROVIDER_LIMITER.allow(key):
        return
    raise HTTPException(
        status_code=429,
        detail="Too many provider administration requests; please try again later.",
        headers={
            "Retry-After": str(
                _ADMIN_MODEL_PROVIDER_LIMITER.retry_after_seconds(key)
            )
        },
    )


def _raise_admin_provider_http_error(exc: Exception) -> None:
    if isinstance(exc, ProviderNotFoundError):
        raise HTTPException(status_code=404, detail="Provider or credential was not found.") from exc
    if isinstance(exc, InvalidProviderOriginError):
        raise HTTPException(status_code=422, detail="Invalid provider configuration.") from exc
    if isinstance(exc, CredentialConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError) and not isinstance(exc, ModelProviderStoreError):
        raise HTTPException(
            status_code=503,
            detail="Credential encryption is unavailable.",
        ) from exc
    if isinstance(exc, (ModelProviderStoreError, ValueError, KeyError)):
        raise HTTPException(status_code=422, detail="Invalid provider configuration.") from exc
    raise exc


def _provider_payload(provider) -> dict:
    return provider.model_dump(mode="json")


def _platform_credential_payload(credential) -> dict | None:
    return credential.model_dump(mode="json") if credential else None


async def _parse_platform_credential_request(
    request: Request,
) -> AdminPlatformCredentialRequest:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_CREDENTIAL_BODY_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Platform credential request is too large.",
                )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request.") from None
    body = await request.body()
    if len(body) > _MAX_CREDENTIAL_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Platform credential request is too large.",
        )
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return AdminPlatformCredentialRequest.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Invalid platform credential request.",
        ) from None


@router.get("")
async def list_admin_model_providers(
    _admin: dict = Depends(require_admin),
):
    providers = await run_in_threadpool(model_provider_service.list_admin_providers)
    items = []
    for provider in providers:
        credential = await run_in_threadpool(
            model_provider_service.get_platform_credential,
            provider.provider_id,
        )
        items.append(
            {
                **_provider_payload(provider),
                "platform_credential": _platform_credential_payload(credential),
            }
        )
    return {"providers": items}


@router.put("/{provider_id}")
async def upsert_admin_model_provider(
    provider_id: str,
    payload: AdminProviderRequest,
    admin: dict = Depends(require_admin),
):
    admin_user_id = int(admin["id"])
    _limit_mutation(admin_user_id)
    try:
        provider = await run_in_threadpool(
            model_provider_service.upsert_provider,
            admin_user_id,
            provider_id,
            payload,
        )
    except Exception as exc:
        _raise_admin_provider_http_error(exc)
    return {"provider": _provider_payload(provider)}


@router.put("/{provider_id}/platform-credential")
async def set_admin_platform_credential(
    provider_id: str,
    request: Request,
    admin: dict = Depends(require_admin),
):
    admin_user_id = int(admin["id"])
    _limit_mutation(admin_user_id)
    payload = await _parse_platform_credential_request(request)
    try:
        credential = await run_in_threadpool(
            model_provider_service.set_platform_credential,
            admin_user_id,
            provider_id,
            payload,
        )
    except Exception as exc:
        _raise_admin_provider_http_error(exc)
    return {"platform_credential": _platform_credential_payload(credential)}


@router.post("/{provider_id}/platform-credential/verify")
async def reverify_admin_platform_credential(
    provider_id: str,
    payload: AdminProviderActionRequest,
    admin: dict = Depends(require_admin),
):
    admin_user_id = int(admin["id"])
    _limit_mutation(admin_user_id)
    try:
        credential = await run_in_threadpool(
            model_provider_service.reverify_platform_credential,
            admin_user_id,
            provider_id,
            source=payload.source,
            reason=payload.reason,
            idempotency_key=payload.idempotency_key,
        )
    except Exception as exc:
        _raise_admin_provider_http_error(exc)
    return {"platform_credential": _platform_credential_payload(credential)}


@router.delete("/{provider_id}/platform-credential")
async def revoke_admin_platform_credential(
    provider_id: str,
    payload: AdminProviderActionRequest,
    admin: dict = Depends(require_admin),
):
    admin_user_id = int(admin["id"])
    _limit_mutation(admin_user_id)
    try:
        revoked = await run_in_threadpool(
            model_provider_service.revoke_platform_credential,
            admin_user_id,
            provider_id,
            source=payload.source,
            reason=payload.reason,
            idempotency_key=payload.idempotency_key,
        )
    except Exception as exc:
        _raise_admin_provider_http_error(exc)
    return {"revoked": revoked}
