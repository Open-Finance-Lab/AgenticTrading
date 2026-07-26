"""Optional client-side integrations for external agent frameworks."""

from .tradingagents import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    TradingAgentsDecisionArtifact,
    TradingAgentsDecisionGenerator,
    TradingAgentsDecisionRecord,
    TradingAgentsDependencyError,
    TradingAgentsGenerationError,
    TradingAgentsReplayDiagnostics,
    TradingAgentsReplayPlanner,
    TradingAgentsReplayValidationError,
    TradingAgentsVersionError,
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
    "TradingAgentsDecisionGenerator",
    "TradingAgentsDecisionRecord",
    "TradingAgentsDependencyError",
    "TradingAgentsGenerationError",
    "TradingAgentsReplayDiagnostics",
    "TradingAgentsReplayPlanner",
    "TradingAgentsReplayValidationError",
    "TradingAgentsVersionError",
    "build_safe_manifest",
    "load_decision_artifact",
    "map_rating",
    "sanitize_error_message",
    "save_decision_artifact",
]
