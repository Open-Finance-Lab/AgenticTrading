"""Authenticated APIs for user-owned model provider credentials."""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.domain.model_providers.models import UserCredentialCreate
from dashboard.backend.domain.model_providers.repository_common import (
    CredentialConflictError,
    CredentialNotFoundError,
    CredentialOwnershipError,
    ModelProviderStoreError,
    ProviderNotFoundError,
    validate_provider_id,
)
from dashboard.backend.domain.model_providers.service import (
    ModelProviderService,
    get_model_provider_service,
)


router = APIRouter(tags=["model-credentials"])

_CREDENTIAL_MUTATION_LIMITER = FixedWindowRateLimiter(
    max_events=30,
    window_seconds=300,
)
_MAX_CREATE_BODY_BYTES = 16 * 1024


def _public_credential(credential) -> dict:
    return credential.model_dump(mode="json")


def _public_provider(provider) -> dict:
    return {
        "provider_id": provider.provider_id,
        "display_name": provider.display_name,
        "capabilities": provider.capabilities.model_dump(mode="json"),
    }


def _parse_credential_id(raw_value: str) -> str:
    try:
        return str(UUID(raw_value))
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail="Invalid API key identifier.") from None


def _limit_mutation(user_id: int) -> None:
    key = f"model-credentials:{user_id}"
    if _CREDENTIAL_MUTATION_LIMITER.allow(key):
        return
    raise HTTPException(
        status_code=429,
        detail="Too many API key changes; please try again later.",
        headers={
            "Retry-After": str(
                _CREDENTIAL_MUTATION_LIMITER.retry_after_seconds(key)
            )
        },
    )


def _raise_credential_http_error(exc: Exception) -> None:
    if isinstance(
        exc,
        (CredentialNotFoundError, CredentialOwnershipError, ProviderNotFoundError),
    ):
        raise HTTPException(status_code=404, detail="API key was not found.") from exc
    if isinstance(exc, CredentialConflictError):
        detail = (
            "An API key with this name already exists."
            if "label" in str(exc).lower()
            else str(exc)
        )
        raise HTTPException(status_code=409, detail=detail) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="Credential encryption is unavailable.",
        ) from exc
    if isinstance(exc, (ValueError, KeyError)):
        raise HTTPException(status_code=422, detail="Invalid API key request.") from exc
    raise exc


async def _parse_create_request(request: Request) -> UserCredentialCreate:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_CREATE_BODY_BYTES:
                raise HTTPException(status_code=413, detail="API key request is too large.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request.") from None
    body = await request.body()
    if len(body) > _MAX_CREATE_BODY_BYTES:
        raise HTTPException(status_code=413, detail="API key request is too large.")
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return UserCredentialCreate.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, ValueError):
        # Pydantic validation errors include the rejected input. Never serialize
        # that structure because it can contain the submitted credential.
        raise HTTPException(
            status_code=422,
            detail="Invalid API key request.",
        ) from None


@router.get("/credits/model-providers")
async def list_model_providers(
    _current_user: dict = Depends(get_current_user),
    service: ModelProviderService = Depends(get_model_provider_service),
):
    providers = await run_in_threadpool(service.list_providers)
    return {"providers": [_public_provider(provider) for provider in providers]}


@router.get("/credits/execution-options")
async def list_execution_options(
    current_user: dict = Depends(get_current_user),
    service: ModelProviderService = Depends(get_model_provider_service),
):
    providers = await run_in_threadpool(
        service.list_execution_options,
        int(current_user["id"]),
    )
    return {
        "providers": [
            provider.model_dump(mode="json")
            for provider in providers
        ]
    }


@router.get("/credits/api-keys")
async def list_model_credentials(
    provider_id: str | None = Query(default=None),
    current_user: dict = Depends(get_current_user),
    service: ModelProviderService = Depends(get_model_provider_service),
):
    if provider_id is not None:
        try:
            provider_id = validate_provider_id(provider_id)
        except (ModelProviderStoreError, ValueError):
            raise HTTPException(status_code=422, detail="Invalid provider filter.") from None
    credentials = await run_in_threadpool(
        service.list_credentials,
        int(current_user["id"]),
        provider_id,
    )
    return {"items": [_public_credential(item) for item in credentials]}


@router.post("/credits/api-keys", status_code=201)
async def create_model_credential(
    request: Request,
    current_user: dict = Depends(get_current_user),
    service: ModelProviderService = Depends(get_model_provider_service),
):
    user_id = int(current_user["id"])
    _limit_mutation(user_id)
    payload = await _parse_create_request(request)
    try:
        credential = await run_in_threadpool(
            service.create_credential,
            user_id,
            payload,
        )
    except Exception as exc:
        _raise_credential_http_error(exc)
    return {"credential": _public_credential(credential)}


@router.post("/credits/api-keys/{credential_id}/verify")
async def reverify_model_credential(
    credential_id: str,
    current_user: dict = Depends(get_current_user),
    service: ModelProviderService = Depends(get_model_provider_service),
):
    credential_id = _parse_credential_id(credential_id)
    user_id = int(current_user["id"])
    _limit_mutation(user_id)
    try:
        credential = await run_in_threadpool(
            service.reverify_credential,
            user_id,
            str(credential_id),
        )
    except Exception as exc:
        _raise_credential_http_error(exc)
    return {"credential": _public_credential(credential)}


@router.post("/credits/api-keys/{credential_id}/default")
async def set_default_model_credential(
    credential_id: str,
    current_user: dict = Depends(get_current_user),
    service: ModelProviderService = Depends(get_model_provider_service),
):
    credential_id = _parse_credential_id(credential_id)
    user_id = int(current_user["id"])
    _limit_mutation(user_id)
    try:
        credential = await run_in_threadpool(
            service.set_default_credential,
            user_id,
            str(credential_id),
        )
    except Exception as exc:
        _raise_credential_http_error(exc)
    return {"credential": _public_credential(credential)}


@router.delete("/credits/api-keys/{credential_id}")
async def revoke_model_credential(
    credential_id: str,
    current_user: dict = Depends(get_current_user),
    service: ModelProviderService = Depends(get_model_provider_service),
):
    credential_id = _parse_credential_id(credential_id)
    user_id = int(current_user["id"])
    _limit_mutation(user_id)
    try:
        credential = await run_in_threadpool(
            service.revoke_credential,
            user_id,
            str(credential_id),
        )
    except Exception as exc:
        _raise_credential_http_error(exc)
    return {"credential": _public_credential(credential)}
