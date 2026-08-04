"""Public facade for the optional TradingAgents client integration."""

from ._tradingagents_core import (
    ARTIFACT_SCHEMA_VERSION,
    RATING_TO_ACTION,
    ArtifactValidationError,
    TradingAgentsDecisionArtifact,
    TradingAgentsDecisionGenerator,
    TradingAgentsDecisionRecord,
    TradingAgentsDependencyError,
    TradingAgentsGenerationError,
    TradingAgentsOutputError,
    TradingAgentsVersionError,
    build_safe_manifest,
    default_decision_artifact_path,
    load_decision_artifact,
    map_rating,
    sanitize_error_message,
    save_decision_artifact,
    sha256_text,
)
from ._tradingagents_replay import (
    TradingAgentsReplayDiagnostics,
    TradingAgentsReplayPlanner,
    TradingAgentsReplayValidationError,
)
from ._tradingagents_runner import (
    TradingAgentsATLRunner,
    TradingAgentsATLRunOutcome,
    TradingAgentsReplayIncompleteError,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "RATING_TO_ACTION",
    "ArtifactValidationError",
    "TradingAgentsDecisionArtifact",
    "TradingAgentsDecisionGenerator",
    "TradingAgentsDecisionRecord",
    "TradingAgentsDependencyError",
    "TradingAgentsGenerationError",
    "TradingAgentsOutputError",
    "TradingAgentsVersionError",
    "TradingAgentsReplayDiagnostics",
    "TradingAgentsReplayPlanner",
    "TradingAgentsReplayValidationError",
    "TradingAgentsATLRunner",
    "TradingAgentsATLRunOutcome",
    "TradingAgentsReplayIncompleteError",
    "build_safe_manifest",
    "default_decision_artifact_path",
    "load_decision_artifact",
    "map_rating",
    "sanitize_error_message",
    "save_decision_artifact",
    "sha256_text",
]
