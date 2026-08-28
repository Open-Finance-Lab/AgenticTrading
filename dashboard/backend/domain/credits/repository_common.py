"""Shared validation and errors for Credits persistence backends."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Iterable, Mapping
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


def encode_activity_cursor(
    created_at: str,
    source_kind: str,
    source_id: int,
) -> str:
    """Encode the stable cross-ledger ordering key without exposing SQL ids."""

    if not isinstance(created_at, str) or not created_at or len(created_at) > 64:
        raise ValueError("invalid activity cursor")
    if source_kind not in {"ledger", "llm_usage", "promotion"}:
        raise ValueError("invalid activity cursor")
    try:
        source_id = _positive_integer(source_id, "source_id")
    except ValueError as exc:
        raise ValueError("invalid activity cursor") from exc
    payload = json.dumps(
        [created_at, source_kind, source_id],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_activity_cursor(cursor: str | int) -> tuple[str, str, int] | int:
    """Decode an opaque cursor, retaining decimal legacy ledger cursors."""

    if isinstance(cursor, int) and not isinstance(cursor, bool):
        try:
            return _positive_integer(cursor, "cursor")
        except ValueError as exc:
            raise ValueError("invalid activity cursor") from exc
    if not isinstance(cursor, str) or not cursor or len(cursor) > 256:
        raise ValueError("invalid activity cursor")
    if cursor.isdecimal():
        try:
            return _positive_integer(int(cursor), "cursor")
        except ValueError as exc:
            raise ValueError("invalid activity cursor") from exc
    try:
        payload = base64.b64decode(
            cursor + "=" * (-len(cursor) % 4),
            altchars=b"-_",
            validate=True,
        )
        value = json.loads(payload.decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid activity cursor") from exc
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("invalid activity cursor")
    created_at, source_kind, source_id = value
    if (
        not isinstance(created_at, str)
        or not created_at
        or len(created_at) > 64
        or source_kind not in {"ledger", "llm_usage", "promotion"}
        or isinstance(source_id, bool)
        or not isinstance(source_id, int)
        or source_id <= 0
    ):
        raise ValueError("invalid activity cursor")
    return created_at, source_kind, source_id


def summarize_activity_evidence(values: Iterable[object]) -> dict[str, object]:
    """Reduce private per-call evidence to safe run-level display fields."""

    providers: set[str] = set()
    models: set[str] = set()
    billing_sources: set[str] = set()
    provider_unknown = False
    model_unknown = False
    billing_unknown = False
    for raw in values:
        try:
            evidence = json.loads(raw) if isinstance(raw, str) else {}
        except json.JSONDecodeError:
            evidence = {}
        if not isinstance(evidence, dict):
            evidence = {}
        snapshot = evidence.get("pricing_snapshot")
        if not isinstance(snapshot, dict):
            snapshot = {}
        provider = snapshot.get("provider_id")
        model = snapshot.get("model_id")
        billing = evidence.get("billing_source")
        if isinstance(provider, str) and provider.strip():
            providers.add(provider)
        else:
            provider_unknown = True
        if isinstance(model, str) and model.strip():
            models.add(model)
        else:
            model_unknown = True
        if isinstance(billing, str) and billing.strip():
            billing_sources.add(billing)
        else:
            billing_unknown = True

    return {
        "provider_id": (
            next(iter(providers))
            if len(providers) == 1 and not provider_unknown
            else None
        ),
        "model_id": (
            next(iter(models)) if len(models) == 1 and not model_unknown else None
        ),
        "billing_source": (
            next(iter(billing_sources))
            if len(billing_sources) == 1 and not billing_unknown
            else None
        ),
        "provider_mixed": len(providers) > 1,
        "model_mixed": len(models) > 1,
    }


def normalize_activity_item(
    value: Mapping[str, object],
    *,
    evidence_json_values: Iterable[object] = (),
) -> dict[str, object]:
    """Return one public-safe activity row and discard raw billing evidence."""

    item = dict(value)
    item.pop("evidence_json", None)
    item["id"] = int(item.pop("source_id"))
    item["amount_micro"] = int(item["amount_micro"])
    if item.get("source_kind") != "llm_usage":
        item.pop("model_call_count", None)
        return item
    item.update(
        {
            "entry_type": "backtest_usage",
            "source": "llm_execution",
            "reason": "Backtest usage.",
            "model_call_count": int(item["model_call_count"]),
            **summarize_activity_evidence(evidence_json_values),
        }
    )
    item.pop("reservation_id", None)
    item.pop("call_index", None)
    return item
