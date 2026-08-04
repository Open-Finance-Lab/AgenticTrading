"""Dependency-free contracts for the vn.py CTA integration."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from ..models import Order


ARTIFACT_SCHEMA_VERSION = "vnpy-cta-atl-v1"

AUDIT_STATUSES = {
    "atl_rejection",
    "decision_submitted",
    "error_hold",
    "fatal_data_error",
    "local_rejection",
    "partial_submission",
    "run_error",
    "strategy_hold",
    "terminal_bar_skipped",
    "timeout_hold",
    "unsupported_action",
    "warmup_hold",
}

_SENSITIVE_KEYS = {
    "apikey",
    "authorization",
    "credential",
    "credentials",
    "password",
    "passwd",
    "privatekey",
    "secret",
    "token",
}
_KEY_VALUE_RE = re.compile(
    r"(?i)\b([a-z0-9_-]*(?:api[_-]?key|authorization|credential|password|passwd|secret|token))"
    r"['\"]?\s*[:=]\s*['\"]?[^\s,;'\"]+['\"]?"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_URL_CREDENTIAL_RE = re.compile(r"(https?://)[^/@\s]+@", re.IGNORECASE)


class ArtifactValidationError(ValueError):
    """Raised when a vn.py CTA audit artifact violates its schema."""


@dataclass(frozen=True)
class CapturedCtaOrder:
    """A normalized order call captured from a local CtaTemplate."""

    order_id: str
    timestamp: str
    symbol: str
    direction: str
    offset: str
    price: float
    volume: float
    stop: bool = False
    lock: bool = False
    net: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "direction": self.direction,
            "offset": self.offset,
            "price": self.price,
            "volume": self.volume,
            "stop": self.stop,
            "lock": self.lock,
            "net": self.net,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapturedCtaOrder":
        try:
            return cls(
                order_id=str(data["order_id"]),
                timestamp=str(data["timestamp"]),
                symbol=str(data["symbol"]),
                direction=str(data["direction"]),
                offset=str(data["offset"]),
                price=float(data["price"]),
                volume=float(data["volume"]),
                stop=bool(data.get("stop", False)),
                lock=bool(data.get("lock", False)),
                net=bool(data.get("net", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactValidationError("captured order fields are invalid") from exc


@dataclass(frozen=True)
class CtaOrderMapping:
    """Result of translating one captured CTA order into an ATL Order."""

    order: Optional[Order]
    status: str
    reason: Optional[str] = None
    warnings: Tuple[str, ...] = ()


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


_OHLCV_FIELDS = ("open", "high", "low", "close", "volume")


def validate_ohlcv_values(raw: Mapping[str, Any]) -> Dict[str, float]:
    """Extract and sanity-check an OHLCV bar's numeric fields.

    Shared by both the adapter (validating an incoming ATL observation) and
    the runtime (validating a bar payload before building vn.py's BarData) —
    same package, same contract, so one copy instead of two independently
    drifting ones. Raises ``ValueError`` on any missing/non-numeric field or
    contract violation (non-finite, non-positive price, negative volume,
    high/low inconsistent with open/close); callers decide what to do with
    that (reject the bar, hold, etc.) since the right response differs by
    caller.
    """
    try:
        values = {field: float(raw[field]) for field in _OHLCV_FIELDS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("bar is missing numeric OHLCV fields") from exc
    if (
        not all(
            math.isfinite(values[field]) and values[field] > 0
            for field in ("open", "high", "low", "close")
        )
        or not math.isfinite(values["volume"])
        or values["volume"] < 0
        or values["high"] < max(values["open"], values["close"])
        or values["low"] > min(values["open"], values["close"])
    ):
        raise ValueError("bar violates the OHLCV contract")
    return values


def map_captured_order(
    captured: CapturedCtaOrder,
    *,
    symbol: str,
    current_position: int,
) -> CtaOrderMapping:
    """Map the supported long-only CTA subset to ATL's market-order schema."""
    expected_symbol = str(symbol).upper()
    if str(captured.symbol).upper() != expected_symbol:
        return CtaOrderMapping(None, "local_rejection", "symbol_mismatch")

    price = _finite_number(captured.price)
    if price is None or price <= 0:
        return CtaOrderMapping(None, "local_rejection", "invalid_order_price")

    volume = _finite_number(captured.volume)
    if volume is None or volume <= 0 or not volume.is_integer():
        return CtaOrderMapping(None, "local_rejection", "invalid_share_volume")
    quantity = int(volume)

    for enabled, reason in (
        (captured.stop, "stop_not_supported"),
        (captured.lock, "lock_not_supported"),
        (captured.net, "net_not_supported"),
    ):
        if enabled:
            return CtaOrderMapping(None, "unsupported_action", reason)

    direction = str(captured.direction).strip().lower()
    offset = str(captured.offset).strip().lower()
    if direction == "long" and offset == "open":
        side = "buy"
    elif direction == "short" and offset == "close":
        side = "sell"
        if quantity > max(0, int(current_position)):
            return CtaOrderMapping(
                None, "local_rejection", "sell_exceeds_position"
            )
    elif direction == "short" and offset == "open":
        return CtaOrderMapping(None, "unsupported_action", "short_not_supported")
    elif direction == "long" and offset == "close":
        return CtaOrderMapping(None, "unsupported_action", "cover_not_supported")
    else:
        return CtaOrderMapping(
            None,
            "unsupported_action",
            f"unsupported_direction_offset:{direction}:{offset}",
        )

    return CtaOrderMapping(
        order=Order(
            symbol=expected_symbol,
            side=side,
            quantity=quantity,
            quantity_type="shares",
            order_type="market",
        ),
        status="mapped",
        warnings=("limit_price_not_enforced",),
    )


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _sensitive_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    return normalized in _SENSITIVE_KEYS or any(
        normalized.endswith(term) for term in _SENSITIVE_KEYS
    )


def _sanitize_text(value: str) -> str:
    text = _KEY_VALUE_RE.sub(lambda match: f"{match.group(1)}=<redacted>", value)
    text = _BEARER_RE.sub("Bearer <redacted>", text)
    return _URL_CREDENTIAL_RE.sub(r"\1<redacted>@", text)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if _sensitive_key(key) else _sanitize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(str(value))


def build_safe_manifest(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe manifest with credentials redacted recursively."""
    sanitized = _sanitize_value(values)
    if not isinstance(sanitized, dict):  # defensive: Mapping always maps to dict
        raise TypeError("manifest must be a mapping")
    return sanitized


def sanitize_error_message(error: Any, *, max_length: int = 500) -> str:
    """Remove common credential shapes and bound persisted exception text."""
    return _sanitize_text(str(error)).strip()[:max_length]


@dataclass(frozen=True)
class VnpyCtaAuditRecord:
    """One ATL step's local vn.py audit record."""

    sequence: int
    observation_timestamp: str
    status: str
    signal_timestamp: Optional[str] = None
    bar: Dict[str, Any] = field(default_factory=dict)
    signal_bar: Dict[str, Any] = field(default_factory=dict)
    captured_orders: Tuple[CapturedCtaOrder, ...] = ()
    submitted_orders: Tuple[Dict[str, Any], ...] = ()
    execution: Optional[Dict[str, Any]] = None
    diagnostics: Dict[str, int] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or self.sequence < 0:
            raise ArtifactValidationError("record sequence must be a non-negative integer")
        if not str(self.observation_timestamp).strip():
            raise ArtifactValidationError("record observation_timestamp is required")
        if self.status not in AUDIT_STATUSES:
            raise ArtifactValidationError(f"unsupported audit status: {self.status!r}")
        for key, value in self.diagnostics.items():
            if not str(key).strip() or key == "total_records":
                raise ArtifactValidationError("invalid audit diagnostic key")
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ArtifactValidationError("audit diagnostic counts must be non-negative integers")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "observation_timestamp": self.observation_timestamp,
            "signal_timestamp": self.signal_timestamp,
            "status": self.status,
            "bar": _sanitize_value(self.bar),
            "signal_bar": _sanitize_value(self.signal_bar),
            "captured_orders": [order.to_dict() for order in self.captured_orders],
            "submitted_orders": [_sanitize_value(order) for order in self.submitted_orders],
            "execution": _sanitize_value(self.execution),
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
            "error": sanitize_error_message(self.error) if self.error else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VnpyCtaAuditRecord":
        try:
            return cls(
                sequence=int(data["sequence"]),
                observation_timestamp=str(data["observation_timestamp"]),
                signal_timestamp=(
                    str(data["signal_timestamp"])
                    if data.get("signal_timestamp") is not None
                    else None
                ),
                status=str(data["status"]),
                bar=dict(data.get("bar") or {}),
                signal_bar=dict(data.get("signal_bar") or {}),
                captured_orders=tuple(
                    CapturedCtaOrder.from_dict(item)
                    for item in (data.get("captured_orders") or [])
                ),
                submitted_orders=tuple(
                    dict(item) for item in (data.get("submitted_orders") or [])
                ),
                execution=(
                    dict(data["execution"]) if data.get("execution") else None
                ),
                diagnostics={
                    str(key): int(value)
                    for key, value in dict(data.get("diagnostics") or {}).items()
                },
                warnings=tuple(str(item) for item in (data.get("warnings") or [])),
                error=str(data["error"]) if data.get("error") else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactValidationError("audit record fields are invalid") from exc


def _summary(records: Iterable[VnpyCtaAuditRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    total = 0
    for record in records:
        total += 1
        counts[record.status] = counts.get(record.status, 0) + 1
        for key, value in record.diagnostics.items():
            counts[key] = counts.get(key, 0) + value
    counts["total_records"] = total
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class VnpyCtaAuditArtifact:
    """Credential-free, replayable evidence for one vn.py CTA ATL run."""

    schema_version: str
    manifest: Dict[str, Any]
    records: Tuple[VnpyCtaAuditRecord, ...]
    summary: Dict[str, int]

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"unsupported artifact schema: {self.schema_version!r}"
            )
        sequences = [record.sequence for record in self.records]
        if len(sequences) != len(set(sequences)):
            raise ArtifactValidationError("artifact has duplicate sequence values")
        if sequences != sorted(sequences):
            raise ArtifactValidationError("artifact records must be sequence ordered")
        expected_summary = _summary(self.records)
        if dict(self.summary) != expected_summary:
            raise ArtifactValidationError("artifact summary does not match records")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest": build_safe_manifest(self.manifest),
            "records": [record.to_dict() for record in self.records],
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VnpyCtaAuditArtifact":
        try:
            records = tuple(
                VnpyCtaAuditRecord.from_dict(item)
                for item in (data.get("records") or [])
            )
            return cls(
                schema_version=str(data["schema_version"]),
                manifest=build_safe_manifest(dict(data.get("manifest") or {})),
                records=records,
                summary={
                    str(key): int(value)
                    for key, value in dict(data.get("summary") or {}).items()
                },
            )
        except ArtifactValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactValidationError("artifact fields are invalid") from exc


def build_audit_artifact(
    *,
    manifest: Mapping[str, Any],
    records: Iterable[VnpyCtaAuditRecord],
) -> VnpyCtaAuditArtifact:
    ordered = tuple(sorted(records, key=lambda record: record.sequence))
    return VnpyCtaAuditArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        manifest=build_safe_manifest(manifest),
        records=ordered,
        summary=_summary(ordered),
    )


def _artifact_json(artifact: VnpyCtaAuditArtifact) -> str:
    return json.dumps(
        artifact.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def save_audit_artifact(
    artifact: VnpyCtaAuditArtifact,
    path: Path | str,
) -> str:
    """Write an artifact and return the SHA-256 of the exact file bytes."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _artifact_json(artifact)
    destination.write_text(payload, encoding="utf-8")
    return sha256(payload.encode("utf-8")).hexdigest()


def load_audit_artifact(path: Path | str) -> VnpyCtaAuditArtifact:
    """Load and fully validate a vn.py CTA audit artifact."""
    try:
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"artifact is not readable JSON: {sanitize_error_message(exc)}"
        ) from exc
    if not isinstance(raw, dict):
        raise ArtifactValidationError("artifact JSON must be an object")
    return VnpyCtaAuditArtifact.from_dict(raw)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "AUDIT_STATUSES",
    "ArtifactValidationError",
    "CapturedCtaOrder",
    "CtaOrderMapping",
    "VnpyCtaAuditRecord",
    "VnpyCtaAuditArtifact",
    "map_captured_order",
    "build_safe_manifest",
    "sanitize_error_message",
    "build_audit_artifact",
    "save_audit_artifact",
    "load_audit_artifact",
]
