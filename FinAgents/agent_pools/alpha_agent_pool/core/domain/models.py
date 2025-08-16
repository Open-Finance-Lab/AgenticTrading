from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AlphaTask:
    """Immutable task submitted to the orchestrator."""
    task_id: str
    strategy_id: str
    market_ctx: Dict[str, Any]
    time_window: Dict[str, str]
    features_req: List[str] = field(default_factory=list)
    risk_hint: Optional[Dict[str, Any]] = None
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class PlanNode:
    """Single node in the AlphaPlan DAG."""
    node_id: str
    node_type: str  # feature | strategy | validate | attribute | score | synthesize
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AlphaPlan:
    """Immutable execution plan as a DAG of PlanNodes."""
    plan_id: str
    nodes: Tuple[PlanNode, ...]


@dataclass(frozen=True)
class AlphaSignal:
    signal_id: str
    strategy_id: str
    ts: str
    symbol: str
    direction: str
    strength: float
    confidence: float
    features_hash: Optional[str] = None


@dataclass(frozen=True)
class AlphaArtifact:
    artifact_id: str
    kind: str
    ref: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlphaResult:
    task_id: str
    signals: List[AlphaSignal] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[AlphaArtifact] = field(default_factory=list)
    lineage: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Ack:
    status: str
    task_id: str
    idempotency_key: Optional[str] = None

