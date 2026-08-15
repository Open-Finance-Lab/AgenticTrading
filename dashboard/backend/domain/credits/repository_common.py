"""Shared validation and errors for Credits persistence backends."""

from __future__ import annotations

from datetime import datetime, timezone


class CreditsStoreError(RuntimeError):
    """Base class for expected Credits-store failures."""


class OrderConflictError(CreditsStoreError):
    """An idempotent operation was retried with different data."""


class RefundNotAllowedError(CreditsStoreError):
    """A refund would exceed the unused, unrefunded purchase lot."""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
