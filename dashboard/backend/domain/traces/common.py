"""Shared validation and projection helpers for trace stores."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict


_MAX_JSON_BYTES = 64 * 1024
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|cookie|password|secret|token)", re.IGNORECASE
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_event_id() -> str:
    return f"tev_{uuid.uuid4().hex[:16]}"


def _json_text(value: Any, *, field: str) -> str:
    try:
        encoded = json.dumps(
            value if value is not None else {},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError(f"{field} exceeds {_MAX_JSON_BYTES} bytes")
    return encoded


def _reject_sensitive(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _SENSITIVE_KEY_RE.search(str(key)):
                raise ValueError(f"sensitive field is not allowed in {path}.{key}")
            _reject_sensitive(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive(child, f"{path}[{index}]")


def _prepare_batch_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and serialize one ``append_events`` entry, with the same rules
    ``append_event`` applies to a single event."""
    payload = event.get("payload") or {}
    _reject_sensitive(payload)
    return {
        "event_type": event["event_type"],
        "actor_type": event["actor_type"],
        "actor_id": event.get("actor_id"),
        "step_id": event.get("step_id"),
        "decision_id": event.get("decision_id"),
        "artifact_id": event.get("artifact_id"),
        "parent_event_id": event.get("parent_event_id"),
        "payload_json": _json_text(payload, field="payload"),
        "occurred_at": event.get("occurred_at") or _utcnow_iso(),
        "idempotency_key": event.get("idempotency_key"),
        "schema_version": int(event.get("schema_version", 1)),
    }


def _public_trace(row: Any) -> Dict[str, Any]:
    return dict(row)


def _public_event(row: Any) -> Dict[str, Any]:
    data = dict(row)
    try:
        data["payload"] = json.loads(data.pop("payload_json") or "{}")
    except (TypeError, ValueError):
        data["payload"] = {}
    return data

