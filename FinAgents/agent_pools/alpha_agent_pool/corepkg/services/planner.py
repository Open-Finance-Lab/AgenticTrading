from __future__ import annotations

import hashlib
from ..domain.models import AlphaTask, AlphaPlan, PlanNode


class Planner:
    """Minimal planner that builds a simple DAG based on strategy_id and features."""

    def plan(self, task: AlphaTask) -> AlphaPlan:
        # Deterministic plan id by hashing task key fields
        plan_key = f"{task.task_id}|{task.strategy_id}|{task.time_window.get('start','')}|{task.time_window.get('end','')}"
        plan_id = hashlib.sha1(plan_key.encode()).hexdigest()[:16]

        nodes = []
        # Feature nodes first
        for idx, feature in enumerate(task.features_req or []):
            nodes.append(
                PlanNode(
                    node_id=f"feature_{idx}",
                    node_type="feature",
                    params={"feature": feature, "market_ctx": task.market_ctx},
                    depends_on=[],
                )
            )

        # Strategy node depends on all features
        nodes.append(
            PlanNode(
                node_id="strategy_main",
                node_type="strategy",
                params={"strategy_id": task.strategy_id, "market_ctx": task.market_ctx},
                depends_on=[n.node_id for n in nodes],
            )
        )

        # Validation node depends on strategy
        nodes.append(
            PlanNode(
                node_id="validate_main",
                node_type="validate",
                params={"rules": ["basic_consistency"]},
                depends_on=["strategy_main"],
            )
        )

        return AlphaPlan(plan_id=plan_id, nodes=tuple(nodes))

