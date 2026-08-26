"""Best-effort server-authoritative Analytics instrumentation helpers.

The source domains call this module only after their authoritative writes
commit.  Every public emitter contains validation and storage failures so
Analytics can never change the source operation's outcome.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from .models import ALLOWED_ERROR_CATEGORIES
from .service import get_analytics_service


_ACCOUNT_EVENTS = {
    "account_signed_up",
    "authenticated_session_started",
}
_CREDENTIAL_EVENTS = {
    "credential_saved",
    "credential_verified",
    "credential_defaulted",
    "credential_reverified",
    "credential_revoked",
}
_AGENT_EVENTS = {
    "agent_created",
    "agent_updated",
    "agent_deleted",
}
_RUN_EVENTS = {
    "backtest_requested",
    "backtest_queued",
    "backtest_started",
    "backtest_completed",
    "backtest_failed",
    "backtest_cancelled",
}
_RESOURCE_EVENTS = {
    "model_usage_recorded",
    "credits_reserved",
    "credits_settled",
    "credits_refunded",
}
SNAPSHOT_RELEVANT_EVENTS = (
    _ACCOUNT_EVENTS
    | _CREDENTIAL_EVENTS
    | _AGENT_EVENTS
    | _RUN_EVENTS
    | _RESOURCE_EVENTS
    | {"safe_error_recorded"}
)
_snapshot_recalculator: Any = None


def register_snapshot_recalculator(callback: Any) -> None:
    global _snapshot_recalculator
    if callback is not None and not callable(callback):
        raise TypeError("snapshot recalculator must be callable")
    _snapshot_recalculator = callback


def _recalculate_snapshot(user_id: int, event_name: str) -> None:
    if event_name not in SNAPSHOT_RELEVANT_EVENTS:
        return
    callback = _snapshot_recalculator
    if callback is None:
        from .states import recalculate_user_snapshot

        callback = recalculate_user_snapshot
    callback(user_id)


def _safe_warning(exc: BaseException) -> None:
    """Log only an exception category, never exception text or identifiers."""

    print(
        "WARNING: analytics.instrumentation_failed "
        f"category={type(exc).__name__[:80]}"
    )


def _occurred_at(value: datetime | None) -> datetime:
    return value or datetime.now(timezone.utc)


def _source_event_id(
    *,
    group: str,
    event_name: str,
    source_record_id: str,
    version: str | int | None,
) -> str:
    record_id = str(source_record_id).strip()
    if not record_id:
        raise ValueError("source_record_id is required")
    parts = [group, event_name, record_id]
    if version is not None:
        normalized_version = str(version).strip()
        if not normalized_version:
            raise ValueError("version must be non-empty when provided")
        parts.append(normalized_version)
    candidate = ":".join(parts)
    if len(candidate) <= 200:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    prefix = f"{group}:{event_name}:"
    return f"{prefix}{digest}"[:200]


def _emit(*, allowed_names: set[str], group: str, **kwargs: Any) -> None:
    try:
        event_name = kwargs.get("event_name")
        if event_name not in allowed_names:
            raise ValueError("unsupported analytics event name")
        source_record_id = kwargs.get("source_record_id")
        if source_record_id is None:
            raise ValueError("source_record_id is required")
        version = kwargs.pop("version", None)
        kwargs["source_event_id"] = _source_event_id(
            group=group,
            event_name=event_name,
            source_record_id=source_record_id,
            version=version,
        )
        result = get_analytics_service().try_record_server_event(**kwargs)
        if result is not None:
            _recalculate_snapshot(int(kwargs["user_id"]), str(event_name))
    except Exception as exc:
        _safe_warning(exc)


def emit_account_event(
    *,
    event_name: str,
    user_id: int,
    source_record_id: str | int,
    occurred_at: datetime | None = None,
    version: str | int | None = None,
) -> None:
    occurred = _occurred_at(occurred_at)
    if version is None and event_name == "authenticated_session_started":
        version = occurred.astimezone(timezone.utc).isoformat()
    _emit(
        allowed_names=_ACCOUNT_EVENTS,
        group="account",
        event_name=event_name,
        user_id=user_id,
        source_record_type="user",
        source_record_id=str(source_record_id),
        properties={},
        occurred_at=occurred,
        version=version,
    )


def emit_credential_event(
    *,
    event_name: str,
    user_id: int,
    credential_id: str,
    provider_id: str | None = None,
    occurred_at: datetime | None = None,
    version: str | int | None = None,
) -> None:
    _emit(
        allowed_names=_CREDENTIAL_EVENTS,
        group="credential",
        event_name=event_name,
        user_id=user_id,
        source_record_type="credential",
        source_record_id=str(credential_id),
        provider_id=provider_id,
        properties={},
        occurred_at=_occurred_at(occurred_at),
        version=version,
    )


def emit_agent_event(
    *,
    event_name: str,
    user_id: int,
    agent_id: str,
    occurred_at: datetime | None = None,
    version: str | int | None = None,
) -> None:
    _emit(
        allowed_names=_AGENT_EVENTS,
        group="agent",
        event_name=event_name,
        user_id=user_id,
        source_record_type="agent",
        source_record_id=str(agent_id),
        properties={},
        occurred_at=_occurred_at(occurred_at),
        version=version,
    )


def emit_run_event(
    *,
    event_name: str,
    user_id: int,
    run_id: str,
    correlation_id: str | None = None,
    outcome: str | None = None,
    error_category: str | None = None,
    occurred_at: datetime | None = None,
    version: str | int | None = None,
) -> None:
    if outcome is None:
        outcome = {
            "backtest_completed": "succeeded",
            "backtest_failed": "failed",
            "backtest_cancelled": "cancelled",
        }.get(event_name)
    _emit(
        allowed_names=_RUN_EVENTS,
        group="run",
        event_name=event_name,
        user_id=user_id,
        source_record_type="run",
        source_record_id=str(run_id),
        correlation_id=correlation_id or str(run_id),
        outcome=outcome,
        error_category=error_category,
        properties={},
        occurred_at=_occurred_at(occurred_at),
        version=version,
    )


def emit_resource_event(
    *,
    event_name: str,
    user_id: int,
    source_record_type: str,
    source_record_id: str,
    properties: dict[str, Any],
    correlation_id: str | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
    billing_mode: str | None = None,
    outcome: str | None = None,
    occurred_at: datetime | None = None,
    version: str | int | None = None,
) -> None:
    _emit(
        allowed_names=_RESOURCE_EVENTS,
        group="resource",
        event_name=event_name,
        user_id=user_id,
        source_record_type=source_record_type,
        source_record_id=str(source_record_id),
        correlation_id=correlation_id,
        provider_id=provider_id,
        model_id=model_id,
        billing_mode=billing_mode,
        outcome=outcome,
        properties=properties,
        occurred_at=_occurred_at(occurred_at),
        version=version,
    )


def classify_safe_error(exc: BaseException) -> str:
    """Reduce arbitrary failures to the fixed Analytics error taxonomy."""

    if isinstance(exc, TimeoutError):
        return "provider_timeout"
    if isinstance(exc, PermissionError):
        return "credential_invalid"
    if isinstance(exc, LookupError):
        return "credential_missing"
    if isinstance(exc, ConnectionError):
        return "provider_unavailable"
    return "internal_error"


def emit_safe_error_event(
    *,
    user_id: int,
    source_record_type: str,
    source_record_id: str,
    exc: BaseException | None = None,
    error_category: str | None = None,
    correlation_id: str | None = None,
    occurred_at: datetime | None = None,
    version: str | int | None = None,
) -> None:
    try:
        category = error_category or classify_safe_error(
            exc or RuntimeError()
        )
        if category not in ALLOWED_ERROR_CATEGORIES:
            raise ValueError("unsupported analytics error category")
        _emit(
            allowed_names={"safe_error_recorded"},
            group="resource",
            event_name="safe_error_recorded",
            user_id=user_id,
            source_record_type=source_record_type,
            source_record_id=str(source_record_id),
            correlation_id=correlation_id,
            error_category=category,
            properties={},
            occurred_at=_occurred_at(occurred_at),
            version=version or category,
        )
    except Exception as facade_exc:
        _safe_warning(facade_exc)
