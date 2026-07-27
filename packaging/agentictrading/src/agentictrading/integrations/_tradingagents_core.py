"""Artifact and local-generation core for the TradingAgents ATL bridge.

This module deliberately uses only the Python standard library. TradingAgents
is an optional dependency and is imported lazily only by the generator. Loading and replaying an existing artifact must
remain available without TradingAgents installed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

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


class TradingAgentsDependencyError(RuntimeError):
    """Raised when the optional TradingAgents package is unavailable."""


class TradingAgentsVersionError(RuntimeError):
    """Raised when the installed TradingAgents version is outside the tested line."""


class TradingAgentsGenerationError(RuntimeError):
    """Raised when no usable TradingAgents decision can be generated."""


class TradingAgentsOutputError(ValueError):
    """Raised when TradingAgents returns a response without a valid rating."""


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


def _canonical_date(value: str, *, field: str = "analysis_date") -> date:
    """Parse a strictly canonical ``YYYY-MM-DD`` string into a ``date``.

    ``date.fromisoformat`` also accepts forms ATL never emits, so the round-trip
    check rejects anything whose canonical spelling differs from the input.
    """
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"{field} must be YYYY-MM-DD, got {value!r}"
        ) from exc
    if parsed.isoformat() != value:
        raise ArtifactValidationError(
            f"{field} must be canonical YYYY-MM-DD, got {value!r}"
        )
    return parsed


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
            if self.rating:
                raise ArtifactValidationError(
                    "error decision records cannot carry a rating"
                )
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
        if _safe_value(self.manifest) != self.manifest:
            raise ArtifactValidationError(
                "manifest contains sensitive keys or credential-shaped values"
            )
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


def default_decision_artifact_path(symbol: str) -> Path:
    """Return the conventional on-disk location for ``symbol``'s artifact.

    This is part of the artifact contract, not of any one front-end: a CLI, a
    notebook, and a future integration all have to agree on where decisions
    land, so the convention lives beside save/load rather than in one caller.
    """
    clean_symbol = str(symbol or "").strip().lower()
    if not clean_symbol:
        raise ArtifactValidationError("symbol must be non-empty")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path.home()
        / ".agentictrading"
        / "tradingagents"
        / "decisions"
        / f"{clean_symbol}-{stamp}.json"
    )


def save_decision_artifact(
    artifact: TradingAgentsDecisionArtifact,
    path: Union[str, Path],
) -> str:
    """Write an artifact as UTF-8 JSON and return its file SHA-256.

    Bytes are written verbatim rather than through ``write_text``, whose
    newline translation would give the same logical artifact a different digest
    on Windows. The digest is recorded as ATL run provenance, so it has to
    identify the content rather than the platform that wrote it.
    """
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    data = payload.encode("utf-8")
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_tradingagents_components():
    """Load the optional upstream package only for the generation path."""
    try:
        config_module = importlib.import_module("tradingagents.default_config")
        graph_module = importlib.import_module(
            "tradingagents.graph.trading_graph"
        )
        rating_module = importlib.import_module(
            "tradingagents.agents.utils.rating"
        )
    except ModuleNotFoundError as exc:
        raise TradingAgentsDependencyError(
            "TradingAgents is required to generate decisions. Install it with: "
            "git clone https://github.com/TauricResearch/TradingAgents.git && "
            "cd TradingAgents && pip install ."
        ) from exc

    default_config = config_module.DEFAULT_CONFIG
    graph_cls = graph_module.TradingAgentsGraph
    rating_parser = rating_module.parse_rating

    def graph_factory(*, selected_analysts, config):
        merged_config = copy.deepcopy(default_config)
        merged_config.update(dict(config))
        return graph_cls(
            selected_analysts=tuple(selected_analysts),
            config=merged_config,
        )

    def version_resolver():
        try:
            return importlib.metadata.version("tradingagents")
        except importlib.metadata.PackageNotFoundError as exc:
            raise TradingAgentsDependencyError(
                "TradingAgents is importable but its package version is unavailable; "
                "reinstall v0.3.1 with 'pip install .' from the upstream repository"
            ) from exc

    return graph_factory, rating_parser, version_resolver


def _validate_tradingagents_version(version: str) -> None:
    match = re.fullmatch(r"0\.3\.(\d+)(?:[+.-].*)?", str(version or ""))
    if match is None or int(match.group(1)) < 1:
        raise TradingAgentsVersionError(
            f"TradingAgents v0.3.1-compatible package required; got {version!r}. "
            "This integration currently supports the 0.3.x release line."
        )


def _canonical_rating(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in RATING_TO_ACTION:
        raise TradingAgentsOutputError(
            "TradingAgents final_trade_decision did not contain an explicit "
            "five-tier rating"
        )
    return normalized.capitalize()


class TradingAgentsDecisionGenerator:
    """Generate replayable decisions with a locally installed TradingAgents.

    Dependency seams are injectable so ATL's test suite stays offline and does
    not need the large optional upstream dependency.
    """

    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(
        self,
        *,
        graph_factory: Optional[Callable[..., Any]] = None,
        rating_parser: Optional[Callable[..., str]] = None,
        version_resolver: Optional[Callable[[], str]] = None,
        clock: Optional[Callable[[], str]] = None,
        max_consecutive_failures: Optional[int] = MAX_CONSECUTIVE_FAILURES,
    ) -> None:
        if max_consecutive_failures is not None and max_consecutive_failures < 1:
            raise ArtifactValidationError(
                "max_consecutive_failures must be a positive integer or None"
            )
        self._graph_factory = graph_factory
        self._rating_parser = rating_parser
        self._version_resolver = version_resolver
        self._clock = clock or _utc_now
        self._max_consecutive_failures = max_consecutive_failures

    def _components(self):
        if (
            self._graph_factory is not None
            and self._rating_parser is not None
            and self._version_resolver is not None
        ):
            return (
                self._graph_factory,
                self._rating_parser,
                self._version_resolver,
            )

        default_factory, default_parser, default_version = (
            _load_tradingagents_components()
        )
        return (
            self._graph_factory or default_factory,
            self._rating_parser or default_parser,
            self._version_resolver or default_version,
        )

    def generate(
        self,
        *,
        symbol: str,
        analysis_dates: Sequence[str],
        config: Mapping[str, Any],
        selected_analysts: Sequence[str],
    ) -> TradingAgentsDecisionArtifact:
        clean_symbol = str(symbol or "").strip().upper()
        if not clean_symbol:
            raise ArtifactValidationError("symbol must be non-empty")

        dates = tuple(analysis_dates)
        if not dates:
            raise ArtifactValidationError("at least one analysis_date is required")
        for analysis_date in dates:
            _canonical_date(analysis_date)
        if len(set(dates)) != len(dates):
            raise ArtifactValidationError("analysis dates must be unique")
        if list(dates) != sorted(dates):
            raise ArtifactValidationError("analysis dates must be sorted")

        analysts = tuple(str(item) for item in selected_analysts)
        if not analysts or any(not item for item in analysts):
            raise ArtifactValidationError(
                "selected_analysts must contain at least one analyst"
            )

        graph_factory, rating_parser, version_resolver = self._components()
        version = str(version_resolver())
        _validate_tradingagents_version(version)
        try:
            graph = graph_factory(
                selected_analysts=analysts,
                config=dict(config),
            )
        except Exception as exc:
            raise TradingAgentsGenerationError(
                f"failed to initialize TradingAgents: {sanitize_error_message(exc)}"
            ) from exc

        records = []
        valid_count = 0
        consecutive_failures = 0
        for index, analysis_date in enumerate(dates):
            record = self._generate_date(
                graph=graph,
                rating_parser=rating_parser,
                symbol=clean_symbol,
                analysis_date=analysis_date,
            )
            records.append(record)
            if record.status == "valid":
                valid_count += 1
                consecutive_failures = 0
                continue
            # Each failed date has already burned two full multi-agent runs.
            # A streak of them is a credential, model, or vendor problem that
            # the remaining dates will hit too, so stop paying for it.
            consecutive_failures += 1
            if (
                self._max_consecutive_failures is not None
                and consecutive_failures >= self._max_consecutive_failures
            ):
                unattempted = len(dates) - index - 1
                raise TradingAgentsGenerationError(
                    f"stopped after {consecutive_failures} consecutive failed "
                    f"analysis dates (last {analysis_date}: "
                    f"{record.error_type}: {record.error_message}); "
                    f"{unattempted} later date(s) were not attempted. Repeated "
                    "failures usually mean a credential, model, or data-vendor "
                    "problem rather than a bad date."
                )

        if valid_count == 0:
            raise TradingAgentsGenerationError(
                "all analysis dates failed; refusing to create an ATL run from "
                "an all-error TradingAgents artifact"
            )

        manifest = build_safe_manifest(
            symbol=clean_symbol,
            tradingagents_version=version,
            config=config,
            selected_analysts=analysts,
            created_at=self._clock(),
        )
        return TradingAgentsDecisionArtifact(
            manifest=manifest,
            decisions=tuple(records),
        )

    @staticmethod
    def _generate_date(
        *,
        graph: Any,
        rating_parser: Callable[..., str],
        symbol: str,
        analysis_date: str,
    ) -> TradingAgentsDecisionRecord:
        last_error: Optional[BaseException] = None
        last_raw = ""
        for attempt in (1, 2):
            try:
                state, _processed = graph.propagate(symbol, analysis_date)
                if not isinstance(state, Mapping):
                    raise TradingAgentsOutputError(
                        "TradingAgents propagate() state must be a mapping"
                    )
                raw = state.get("final_trade_decision")
                if not isinstance(raw, str) or not raw.strip():
                    raise TradingAgentsOutputError(
                        "TradingAgents state has no final_trade_decision text"
                    )
                last_raw = raw
                parsed = rating_parser(raw, default="")
                rating = _canonical_rating(parsed)
                return TradingAgentsDecisionRecord(
                    analysis_date=analysis_date,
                    rating=rating,
                    atl_action=map_rating(rating),
                    status="valid",
                    attempts=attempt,
                    raw_final_trade_decision=raw,
                    raw_sha256=sha256_text(raw),
                )
            except Exception as exc:
                last_error = exc

        error = last_error or TradingAgentsOutputError(
            "TradingAgents analysis failed without an exception"
        )
        message = sanitize_error_message(error) or "TradingAgents analysis failed"
        return TradingAgentsDecisionRecord(
            analysis_date=analysis_date,
            rating="",
            atl_action="HOLD",
            status="error",
            attempts=2,
            raw_final_trade_decision=last_raw,
            raw_sha256=sha256_text(last_raw),
            error_type=type(error).__name__,
            error_message=message,
        )
