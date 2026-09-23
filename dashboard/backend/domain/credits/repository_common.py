"""Shared validation and errors for Credits persistence backends."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta, timezone


def _utc_text(value: datetime, name: str) -> str:
    """ISO-8601 UTC text, the format every ``created_at`` in this ledger uses."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return value.astimezone(timezone.utc).isoformat()


def _day_bounds(day: date) -> tuple[str, str]:
    start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    return _utc_text(start, "day"), _utc_text(start + timedelta(days=1), "day")


def _unique_user_ids(user_ids: Sequence[int]) -> list[int]:
    if not isinstance(user_ids, (list, tuple)):
        raise ValueError("user_ids must be a list or tuple")
    return list(
        dict.fromkeys(_positive_integer(user_id, "user_id") for user_id in user_ids)
    )


def _assemble_commercial_ledger(
    ids: list[int], lifetime_rows, period_rows, usage_rows
) -> dict[int, dict[str, int]]:
    lifetime = {
        int(row["user_id"]): (
            int(row["purchased_micro"] or 0),
            int(row["refunded_micro"] or 0),
        )
        for row in lifetime_rows
    }
    period = {
        int(row["user_id"]): (
            int(row["purchased_micro"] or 0),
            int(row["refunded_micro"] or 0),
            int(row["grant_activity_micro"] or 0),
        )
        for row in period_rows
    }
    usage = {int(row["user_id"]): int(row["consumed_micro"] or 0) for row in usage_rows}
    result: dict[int, dict[str, int]] = {}
    for user_id in ids:
        lifetime_purchased, lifetime_refunded = lifetime.get(user_id, (0, 0))
        purchased, refunded, grant_activity = period.get(user_id, (0, 0, 0))
        result[user_id] = {
            "lifetime_purchased_micro": lifetime_purchased,
            "lifetime_refunded_micro": lifetime_refunded,
            "purchased_micro": purchased,
            "refunded_micro": refunded,
            "grant_activity_micro": grant_activity,
            "consumed_micro": usage.get(user_id, 0),
        }
    return result


def _assemble_ledger_day(usage_rows, lifetime_rows, purchase_rows) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}

    def entry(user_id: int) -> dict[str, Any]:
        return result.setdefault(
            user_id,
            {
                "own_spend_micro": 0,
                "lifetime_net_purchased_micro": 0,
                "last_activity_at": None,
            },
        )

    def later(current: str | None, candidate: Any) -> str | None:
        if candidate is None:
            return current
        text = str(candidate)
        return text if current is None or text > current else current

    for row in usage_rows:
        record = entry(int(row["user_id"]))
        record["own_spend_micro"] = max(0, int(row["consumed_micro"] or 0))
        record["last_activity_at"] = later(record["last_activity_at"], row["last_usage_at"])
    for row in lifetime_rows:
        record = entry(int(row["user_id"]))
        record["lifetime_net_purchased_micro"] = max(
            0, int(row["purchased_micro"] or 0) - int(row["refunded_micro"] or 0)
        )
    for row in purchase_rows:
        record = entry(int(row["user_id"]))
        record["last_activity_at"] = later(
            record["last_activity_at"], row["last_purchase_at"]
        )
    return result


def _assemble_billing_states(account_rows, outstanding_rows) -> dict[int, dict[str, Any]]:
    outstanding = {
        int(row["user_id"]): int(row["outstanding_micro"] or 0) for row in outstanding_rows
    }
    result: dict[int, dict[str, Any]] = {}
    for row in account_rows:
        user_id = int(row["user_id"])
        reason = row["restriction_reason"]
        if row["status"] == "restricted" and reason not in {
            "llm_overage",
            "refund_reconciliation",
        }:
            reason = "refund_reconciliation"
        result[user_id] = {
            "account_status": row["status"],
            "restriction_reason": reason,
            "outstanding_credits_micro": outstanding.get(user_id, 0),
        }
    return result


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