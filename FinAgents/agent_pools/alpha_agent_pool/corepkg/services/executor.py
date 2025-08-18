from __future__ import annotations

import asyncio
from typing import Dict, Any, List, Optional

from ..domain.models import AlphaPlan, AlphaResult, AlphaSignal
from ..ports.feature import FeaturePort
from ..ports.strategy import StrategyPort
from ..ports.memory import MemoryPort


class Executor:
    """Execute a plan by dispatching nodes to the appropriate ports."""

    def __init__(self, feature_port: FeaturePort, strategy_port: StrategyPort, memory_port: Optional[MemoryPort] = None):
        self.feature_port = feature_port
        self.strategy_port = strategy_port
        self.memory_port = memory_port

    async def run(self, plan: AlphaPlan) -> AlphaResult:
        context: Dict[str, Any] = {}
        lineage: List[str] = []
        signals: List[AlphaSignal] = []

        for node in plan.nodes:
            lineage.append(node.node_id)
            try:
                if node.node_type == "feature":
                    feat = await self.feature_port.fetch({**node.params})
                    context[node.params.get("feature", node.node_id)] = feat
                elif node.node_type == "strategy":
                    node_ctx = {**node.params, "features": context}
                    out = await self.strategy_port.run(node_ctx)
                    if out and "signals" in out:
                        for s in out["signals"]:
                            signals.append(
                                AlphaSignal(
                                    signal_id=s.get("signal_id", ""),
                                    strategy_id=s.get("strategy_id", node.params.get("strategy_id", "unknown")),
                                    ts=s.get("ts", ""),
                                    symbol=s.get("symbol", ""),
                                    direction=s.get("direction", "HOLD"),
                                    strength=float(s.get("strength", 0.0)),
                                    confidence=float(s.get("confidence", 0.0)),
                                    features_hash=s.get("features_hash"),
                                )
                            )
                elif node.node_type == "validate":
                    pass
            except Exception:
                await asyncio.sleep(0.2)
                continue

        return AlphaResult(task_id=plan.plan_id, signals=signals, metrics={}, artifacts=[], lineage=lineage)

