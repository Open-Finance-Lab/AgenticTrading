"""Client-side bridge between TradingAgents decisions and ATL artifacts.

This module deliberately uses only the Python standard library. TradingAgents
is an optional dependency and will be imported lazily by the generator added in
the next implementation loop. Loading and replaying an existing artifact must
remain available without TradingAgents installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

ARTIFACT_SCHEMA_VERSION = "tradingagents-atl-v1"

RATING_TO_ACTION = {
    "buy": "BUY",
    "overweight": "BUY",
    "hold": "HOLD",
    "underweight": "SELL",
    "sell": "SELL",
}

_SAFE_CONFIG_KEYS = (
    "llm_provider",
    "deep_think_llm",
    "quick_think_llm",
    "temperature",
    "max_debate_rounds",
    "max_risk_discuss_rounds",
    "output_language",
    "data_vendors",
    "tool_vendors",
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_-])(api[_-]?)?(?:key|token|secret|password|credentials?)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)[A-Z0-9_]*)"
    r"\s*[=:]\s*[^\s,;]+"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_SK_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]+\b")
_URL_CREDENTIAL_RE = re.compile(r"(https?://)[^\s/:@]+:[^\s@]+@", re.IGNORECASE)
_MAX_ERROR_MESSAGE_LENGTH = 300


class ArtifactValidationError(ValueError):
    """Raised when a TradingAgents decision artifact violates its schema."""


def sha256_text(value: str) -> str:
    """Return a stable SHA-256 hex digest for UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def map_rating(rating: str) -> str:
    """Collapse TradingAgents' five-tier rating into ATL's three directions."""
    normalized = str(rating or "").strip().lower()
    try:
        return RATING_TO_ACTION[normalized]
    except KeyError as exc:
        raise ArtifactValidationError(
            f"unsupported TradingAgents rating: {rating!r}"
        ) from exc


def _canonical_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"analysis_date must be YYYY-MM-DD, got {value!r}"
        ) from exc
    if parsed.isoformat() != value:
        raise ArtifactValidationError(
            f"analysis_date must be canonical YYYY-MM-DD, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class TradingAgentsDecisionRecord:
    """One TradingAgents analysis result for one date."""

    analysis_date: str
    rating: str
    atl_action: str
    status: str
    attempts: int
    raw_final_trade_decision: str
    raw_sha256: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        _canonical_date(self.analysis_date)
        if self.status not in ("valid", "error"):
            raise ArtifactValidationError(
                f"status must be 'valid' or 'error', got {self.status!r}"
            )
        if not isinstance(self.attempts, int) or self.attempts < 1:
            raise ArtifactValidationError("attempts must be a positive integer")
        if self.atl_action not in ("BUY", "HOLD", "SELL"):
            raise ArtifactValidationError(
                f"atl_action must be BUY, HOLD, or SELL, got {self.atl_action!r}"
            )
        expected_digest = sha256_text(self.raw_final_trade_decision)
        if self.raw_sha256 != expected_digest:
            raise ArtifactValidationError(
                "raw_sha256 does not match raw_final_trade_decision"
            )

        if self.status == "valid":
            expected_action = map_rating(self.rating)
            if self.atl_action != expected_action:
                raise ArtifactValidationError(
                    f"rating {self.rating!r} maps to {expected_action}, not {self.atl_action}"
                )
            if self.error_type is not None or self.error_message is not None:
                raise ArtifactValidationError(
                    "valid decision records cannot include error fields"
                )
        else:
            if self.atl_action != "HOLD":
                raise ArtifactValidationError("error decision records must use HOLD")
            if not self.error_type or not self.error_message:
                raise ArtifactValidationError(
                    "error decision records require error_type and error_message"
                )


@dataclass(frozen=True)
class TradingAgentsDecisionArtifact:
    """Versioned, replayable collection of TradingAgents decisions."""

    manifest: Dict[str, Any]
    decisions: Tuple[TradingAgentsDecisionRecord, ...]
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError(
                f"unsupported artifact schema: {self.schema_version!r}"
            )
        if not isinstance(self.manifest, dict):
            raise ArtifactValidationError("manifest must be an object")
        symbol = self.manifest.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ArtifactValidationError("manifest symbol must be non-empty")
        if not self.decisions:
            raise ArtifactValidationError("artifact requires at least one decision")

        dates = [record.analysis_date for record in self.decisions]
        if len(set(dates)) != len(dates):
            raise ArtifactValidationError("analysis dates must be unique")
        if dates != sorted(dates):
            raise ArtifactValidationError("analysis dates must be sorted")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest,
            "decisions": [asdict(record) for record in self.decisions],
        }


def _redact_string(value: str) -> str:
    text = _SECRET_ASSIGNMENT_RE.sub("[REDACTED]", value)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _SK_TOKEN_RE.sub("[REDACTED]", text)
    return _URL_CREDENTIAL_RE.sub(r"\1[REDACTED]@", text)


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(nested)
            for key, nested in value.items()
            if not _SENSITIVE_KEY_RE.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def build_safe_manifest(
    *,
    symbol: str,
    tradingagents_version: str,
    config: Mapping[str, Any],
    selected_analysts: Sequence[str],
    created_at: str,
) -> Dict[str, Any]:
    """Build the credential-free reproducibility metadata stored in artifacts."""
    clean_symbol = str(symbol or "").strip().upper()
    if not clean_symbol:
        raise ArtifactValidationError("symbol must be non-empty")

    safe_config = {
        key: _safe_value(config[key])
        for key in _SAFE_CONFIG_KEYS
        if key in config and not _SENSITIVE_KEY_RE.search(key)
    }
    canonical = json.dumps(
        safe_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    manifest: Dict[str, Any] = {
        "symbol": clean_symbol,
        "created_at": str(created_at),
        "tradingagents_version": str(tradingagents_version),
        "atl_protocol_version": "1.0",
        "selected_analysts": [str(item) for item in selected_analysts],
        "safe_config_sha256": sha256_text(canonical),
    }
    manifest.update(safe_config)
    return manifest


def sanitize_error_message(error: Union[str, BaseException]) -> str:
    """Remove common credential shapes and cap persisted error detail."""
    cleaned = _redact_string(str(error)).replace("\n", " ").replace("\r", " ")
    return cleaned[:_MAX_ERROR_MESSAGE_LENGTH]


def save_decision_artifact(
    artifact: TradingAgentsDecisionArtifact,
    path: Union[str, Path],
) -> str:
    """Write an artifact as UTF-8 JSON and return its file SHA-256."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    destination.write_text(payload, encoding="utf-8")
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_decision_artifact(
    path: Union[str, Path],
) -> TradingAgentsDecisionArtifact:
    """Load and fully validate a TradingAgents decision artifact."""
    source = Path(path).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"artifact is not readable JSON: {sanitize_error_message(exc)}"
        ) from exc

    if not isinstance(payload, dict):
        raise ArtifactValidationError("artifact JSON must be an object")
    try:
        records = tuple(
            TradingAgentsDecisionRecord(**item)
            for item in payload.get("decisions", [])
        )
        return TradingAgentsDecisionArtifact(
            schema_version=payload.get("schema_version", ""),
            manifest=payload.get("manifest"),
            decisions=records,
        )
    except ArtifactValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"artifact fields are invalid: {sanitize_error_message(exc)}"
        ) from exc
