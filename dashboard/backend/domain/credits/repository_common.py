"""Shared validation and errors for Credits persistence backends."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone


class CreditsStoreError(RuntimeError):
    """Base class for expected Credits-store failures."""


class OrderConflictError(CreditsStoreError):
    """An idempotent operation was retried with different data."""


class RefundNotAllowedError(CreditsStoreError):
    """A refund would exceed the unused, unrefunded purchase lot."""


class IdempotencyConflictError(CreditsStoreError):
    """An idempotent Grant operation was retried with different data."""


class GrantPoolInsufficientError(CreditsStoreError):
    """A Grant operation would make the pool balance negative."""


class GrantReclaimExceedsAvailableError(CreditsStoreError):
    """A reclaim exceeds the user's available Grant Credits."""


class CreditAccountRestrictedStoreError(CreditsStoreError):
    """A Grant operation targets a restricted credit account."""


class InsufficientCreditsError(CreditsStoreError):
    """A usage reservation would exceed the user's available Credits."""


class LLMReservationConflictError(CreditsStoreError):
    """A reservation replay or state transition conflicts with prior data."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _required_text(value: object, name: str, max_length: int | None = None) -> str:
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must be trimmed")
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")
    return value


def _nonzero_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value == 0:
        raise ValueError(f"{name} must be a non-zero integer")
    return value


def _canonical_digest(parts: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(parts),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_amount_pair(amount_usd_cents: int, credits_micro: int) -> None:
    cents = _positive_integer(amount_usd_cents, "amount_usd_cents")
    credits = _positive_integer(credits_micro, "credits_micro")
    if credits != cents * 10_000:
        raise ValueError("credits_micro must equal amount_usd_cents * 10,000")


def _positive_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("limit must be an integer from 1 through 100")
    return value
