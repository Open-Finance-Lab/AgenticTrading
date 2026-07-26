"""Optional client-side integrations for external agent frameworks."""

from .tradingagents import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    TradingAgentsDecisionArtifact,
    TradingAgentsDecisionRecord,
    build_safe_manifest,
    load_decision_artifact,
    map_rating,
    sanitize_error_message,
    save_decision_artifact,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactValidationError",
    "TradingAgentsDecisionArtifact",
    "TradingAgentsDecisionRecord",
    "build_safe_manifest",
    "load_decision_artifact",
    "map_rating",
    "sanitize_error_message",
    "save_decision_artifact",
]
