"""GET /api/v2/leaderboard — rank real v2 runs vs baselines (spec §8.4)."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from dashboard.backend.database import db

router = APIRouter(prefix="/v2", tags=["v2-leaderboard"])


def _is_v2_run(run: Dict[str, Any]) -> bool:
    return str(run.get("run_id", "")).startswith("run_")


def build_leaderboard(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    v2_runs = [r for r in runs if _is_v2_run(r)]
    ranked = sorted(v2_runs, key=lambda r: (r.get("total_return") or -1e9), reverse=True)
    board = []
    for i, r in enumerate(ranked, start=1):
        board.append({
            "rank": i,
            "run_id": r.get("run_id"),
            "agent_name": r.get("agent_name"),
            "model": r.get("llm_model"),
            "total_return": r.get("total_return"),
            "sharpe_ratio": r.get("sharpe_ratio"),
            "max_drawdown": r.get("max_drawdown"),
            "num_trades": r.get("num_trades"),
            "final_equity": r.get("final_equity"),
            "baseline_djia_run_id": r.get("baseline_djia_run_id"),
            "baseline_buyhold_run_id": r.get("baseline_buyhold_run_id"),
        })
    return board


@router.get("/leaderboard")
def leaderboard():
    return {"leaderboard": build_leaderboard(db.get_all_runs())}
