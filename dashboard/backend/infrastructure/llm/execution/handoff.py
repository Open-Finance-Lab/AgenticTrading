"""Signed, secret-free execution handoffs for backtest worker processes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dashboard.backend.domain.model_providers.repository_common import (
    validate_provider_id,
)

from dashboard.backend.infrastructure.llm.execution.models import BillingMode
from dashboard.backend.session_tokens import session_hash_secret


_HANDOFF_VERSION = 1
_DEFAULT_TTL_SECONDS = 300
_MAX_HANDOFF_BYTES = 16_384


class ExecutionHandoffError(ValueError):
    """A fixed failure for invalid, expired, or replayed worker input."""

    def __init__(self) -> None:
        super().__init__("invalid execution handoff")


class ExecutionHandoff(BaseModel):
    """The worker-safe execution identity; it contains no credential material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=_HANDOFF_VERSION, ge=_HANDOFF_VERSION, le=_HANDOFF_VERSION)
    user_id: int = Field(gt=0)
    run_id: str = Field(min_length=1, max_length=128)
    billing_mode: BillingMode
    provider_id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9_]+$")
    provider_ids: tuple[str, ...] = Field(default=(), max_length=8)
    model_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._/\-]{0,63}$",
    )
    prompt_digest: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    nonce: str = Field(min_length=16, max_length=128)
    issued_at: int = Field(gt=0)
    expires_at: int = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def populate_provider_candidates(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if "provider_ids" not in payload or payload["provider_ids"] is None:
            payload["provider_ids"] = (payload.get("provider_id"),)
        return payload

    @model_validator(mode="after")
    def validate_provider_candidates(self) -> "ExecutionHandoff":
        try:
            cleaned = tuple(validate_provider_id(value) for value in self.provider_ids)
        except Exception as exc:
            raise ValueError("invalid provider candidates") from exc
        if (
            not cleaned
            or cleaned[0] != self.provider_id
            or len(set(cleaned)) != len(cleaned)
        ):
            raise ValueError("provider candidates must be ordered and unique")
        object.__setattr__(self, "provider_ids", cleaned)
        return self


class HandoffReplayGuard:
    """Process-local one-time nonce guard with expiry-aware cleanup."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._used: dict[str, int] = {}

    def claim(self, nonce: str, expires_at: int, *, now: int) -> None:
        with self._lock:
            expired = [value for value, expiry in self._used.items() if expiry <= now]
            for value in expired:
                del self._used[value]
            if nonce in self._used:
                raise ExecutionHandoffError()
            self._used[nonce] = expires_at


_default_replay_guard = HandoffReplayGuard()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, UnicodeEncodeError) as exc:
        raise ExecutionHandoffError() from exc


def _signing_key() -> bytes:
    """Use a dedicated configured key or derive a scoped session HMAC key."""

    configured = (os.getenv("LLM_EXECUTION_HANDOFF_SECRET") or "").strip()
    if configured:
        return configured.encode("utf-8")
    return hmac.new(
        session_hash_secret(),
        b"atl-llm-execution-handoff-v1",
        hashlib.sha256,
    ).digest()


def _prompt_digest(metadata: Mapping[str, Any] | None) -> str:
    try:
        encoded = json.dumps(
            dict(metadata or {}),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExecutionHandoffError() from exc
    return hashlib.sha256(encoded).hexdigest()


def create_execution_handoff(
    *,
    user_id: int,
    run_id: str,
    billing_mode: BillingMode | str,
    provider_id: str,
    model_id: str,
    provider_ids: tuple[str, ...] | None = None,
    prompt_metadata: Mapping[str, Any] | None = None,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    now: int | None = None,
) -> str:
    """Create an HMAC-signed worker payload with no plaintext credential."""

    issued_at = int(time.time() if now is None else now)
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be a positive integer")
    try:
        handoff = ExecutionHandoff(
            user_id=user_id,
            run_id=run_id.strip(),
            billing_mode=billing_mode,
            provider_id=provider_id.strip(),
            provider_ids=provider_ids or (provider_id.strip(),),
            model_id=model_id.strip(),
            prompt_digest=_prompt_digest(prompt_metadata),
            nonce=secrets.token_urlsafe(24),
            issued_at=issued_at,
            expires_at=issued_at + ttl_seconds,
        )
    except (AttributeError, ValidationError) as exc:
        raise ValueError("invalid execution handoff input") from exc
    payload = json.dumps(
        handoff.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_payload = _b64encode(payload)
    signature = hmac.new(
        _signing_key(), encoded_payload.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{encoded_payload}.{_b64encode(signature)}"


def consume_execution_handoff(
    payload: str,
    *,
    replay_guard: HandoffReplayGuard | None = None,
    now: int | None = None,
) -> ExecutionHandoff:
    """Verify and claim exactly one short-lived worker handoff."""

    if not isinstance(payload, str) or not payload or len(payload) > _MAX_HANDOFF_BYTES:
        raise ExecutionHandoffError()
    try:
        encoded_payload, encoded_signature = payload.split(".", 1)
    except ValueError as exc:
        raise ExecutionHandoffError() from exc
    expected = hmac.new(
        _signing_key(), encoded_payload.encode("ascii"), hashlib.sha256
    ).digest()
    supplied = _b64decode(encoded_signature)
    if not hmac.compare_digest(supplied, expected):
        raise ExecutionHandoffError()
    try:
        raw = json.loads(_b64decode(encoded_payload))
        handoff = ExecutionHandoff.model_validate(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise ExecutionHandoffError() from exc
    current_time = int(time.time() if now is None else now)
    if handoff.expires_at <= current_time or handoff.issued_at > current_time:
        raise ExecutionHandoffError()
    (replay_guard or _default_replay_guard).claim(
        handoff.nonce,
        handoff.expires_at,
        now=current_time,
    )
    return handoff


__all__ = [
    "ExecutionHandoff",
    "ExecutionHandoffError",
    "HandoffReplayGuard",
    "consume_execution_handoff",
    "create_execution_handoff",
]
