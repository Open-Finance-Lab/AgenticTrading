"""Characterization tests for custom-algo prompts (Phase 3D2)."""

import ast
from pathlib import Path

from dashboard.backend.infrastructure.llm.prompts import (
    create_custom_algo_prompt,
    parse_risk_rules,
)

_BACKEND = Path(__file__).resolve().parents[3]


def test_parse_risk_rules_defaults():
    assert parse_risk_rules("") == {
        "stop_loss_pct": 5.0,
        "take_profit_pct": 20.0,
        "daily_stop_pct": 5.0,
    }


def test_parse_risk_rules_extracts_values():
    rules = parse_risk_rules("跌5% 涨20% 单日跌3%")
    assert rules["stop_loss_pct"] == 5.0
    assert rules["take_profit_pct"] == 20.0
    assert rules["daily_stop_pct"] == 3.0


def test_create_custom_algo_prompt_contains_blocks_and_risk():
    blocks = {
        "info_retrieval": "watch news",
        "signal_transfer": "pick winners",
        "trading_algorithm": "aggressive",
        "stop_loss_take_profit": "跌8% 涨25%",
    }
    out = create_custom_algo_prompt({"top_signals": ["AAPL"]}, blocks)
    assert "USER-DEFINED trading strategy" in out
    assert "watch news" in out
    assert "Position stop-loss: 8.0%" in out
    assert "Take-profit reference: 25.0%" in out
    # Exact header line preserved.
    assert out.startswith(
        "You are executing a USER-DEFINED trading strategy on historical DJIA hourly data.\n"
    )


def _imported_modules(path: Path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


def test_prompts_has_no_api_scripts_or_frontend_imports():
    mods = _imported_modules(_BACKEND / "infrastructure" / "llm" / "prompts.py")
    for m in mods:
        assert not m.startswith("dashboard.backend.api"), m
        assert not m.startswith("dashboard.scripts"), m
