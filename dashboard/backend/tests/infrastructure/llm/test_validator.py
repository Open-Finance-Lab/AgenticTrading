"""Characterization tests for the LLM validator (Phase 3D2).

Pure validation/parsing logic; no LLM/provider calls occur. Imports use the
canonical package path.
"""

import ast
import json
from pathlib import Path

from dashboard.backend.infrastructure.llm import validator as validator_mod
from dashboard.backend.infrastructure.llm.validator import (
    DJIA_30,
    TOP_10_STOCKS,
    TradingAction,
    actions_to_executable,
    parse_actions_payload,
    validate_llm_response,
)

_BACKEND = Path(__file__).resolve().parents[3]


def _portfolio(cash=100000):
    return {
        "constraints": {"cash_available": cash, "min_confidence": 0.3},
        "positions": [{"symbol": "AAPL", "shares": 100}],
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_djia_30_and_top10_unchanged():
    assert len(DJIA_30) == 30
    assert DJIA_30[0] == "AAPL"
    assert TOP_10_STOCKS == ["AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "AXP", "HD", "DIS"]


# ---------------------------------------------------------------------------
# validate_llm_response
# ---------------------------------------------------------------------------

def test_valid_buy_accepted():
    raw = json.dumps({
        "action": "buy", "symbol": "AAPL", "confidence": 0.8,
        "reasoning": "strong momentum", "position_size": 10,
    })
    decision = validate_llm_response(raw, _portfolio(), {"AAPL": 100.0})
    assert decision is not None
    assert decision.action == "buy"
    assert decision.symbol == "AAPL"


def test_tool_call_attempt_rejected():
    raw = '{"tool_calls": [{"name": "x"}]}'
    assert validate_llm_response(raw, _portfolio(), {}) is None


def test_invalid_json_rejected():
    assert validate_llm_response("not json", _portfolio(), {}) is None


def test_invalid_symbol_rejected():
    raw = json.dumps({
        "action": "buy", "symbol": "ZZZZ", "confidence": 0.8,
        "reasoning": "bad symbol", "position_size": 10,
    })
    assert validate_llm_response(raw, _portfolio(), {"ZZZZ": 10.0}) is None


def test_low_confidence_becomes_hold():
    raw = json.dumps({
        "action": "buy", "symbol": "AAPL", "confidence": 0.1,
        "reasoning": "weak signal", "position_size": 10,
    })
    decision = validate_llm_response(raw, _portfolio(), {"AAPL": 100.0})
    assert decision is not None
    assert decision.action == TradingAction.HOLD
    assert decision.position_size == 0


def test_insufficient_cash_rejected():
    raw = json.dumps({
        "action": "buy", "symbol": "AAPL", "confidence": 0.9,
        "reasoning": "buy big", "position_size": 1000,
    })
    decision = validate_llm_response(raw, _portfolio(cash=100), {"AAPL": 100.0})
    assert decision is None


# ---------------------------------------------------------------------------
# parse_actions_payload + actions_to_executable
# ---------------------------------------------------------------------------

def test_parse_actions_payload_success():
    payload = {"actions": [{
        "action": "buy", "symbol": "AAPL", "confidence": 0.7,
        "reasoning": "okay reason", "position_size": 5,
    }]}
    decisions, err = parse_actions_payload(payload)
    assert err is None
    assert len(decisions) == 1


def test_parse_actions_payload_missing_actions():
    decisions, err = parse_actions_payload({})
    assert decisions is None
    assert "actions" in err


def test_actions_to_executable_filters_hold_and_low_conf():
    decisions, _ = parse_actions_payload({"actions": [
        {"action": "buy", "symbol": "AAPL", "confidence": 0.8, "reasoning": "go long now", "position_size": 5},
        {"action": "hold", "symbol": "MSFT", "confidence": 0.9, "reasoning": "wait and see", "position_size": 0},
        {"action": "buy", "symbol": "JPM", "confidence": 0.1, "reasoning": "low conf", "position_size": 5},
    ]})
    executable = actions_to_executable(
        decisions, cash=100000, positions={}, current_prices={"AAPL": 100.0, "JPM": 50.0},
    )
    assert len(executable) == 1
    assert executable[0]["symbol"] == "AAPL"
    assert executable[0]["action"] == "buy"


# ---------------------------------------------------------------------------
# Exact prompt preservation (whitespace-sensitive)
# ---------------------------------------------------------------------------

def test_safe_prompt_exact_head_and_tail():
    prompt = validator_mod.SAFE_TRADING_PROMPT
    assert prompt.startswith("You are an active DJIA portfolio trading agent.\n\nGoal:\n")
    assert prompt.endswith("Return ONLY valid JSON.\n")
    # A representative interior block must be byte-identical.
    assert (
        "Output rules:\n"
        "- Return ONLY valid JSON.\n"
        "- No markdown.\n"
        "- No code fences.\n"
        "- No explanations outside JSON.\n"
    ) in prompt


def test_buy_and_hold_prompt_exact_head():
    prompt = validator_mod.BUY_AND_HOLD_PROMPT
    assert prompt.startswith("You are a buy-and-hold backtest agent.\n\n")
    assert "Top 10 DJIA stocks: AAPL, MSFT, JPM, V, JNJ, WMT, PG, AXP, HD, DIS\n" in prompt


def test_create_safe_prompt_injects_symbols():
    out = validator_mod.create_safe_prompt({"x": 1})
    assert ", ".join(DJIA_30) in out


# ---------------------------------------------------------------------------
# Import boundaries
# ---------------------------------------------------------------------------

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


def test_validator_has_no_api_scripts_or_frontend_imports():
    mods = _imported_modules(_BACKEND / "infrastructure" / "llm" / "validator.py")
    for m in mods:
        assert not m.startswith("dashboard.backend.api"), m
        assert not m.startswith("dashboard.scripts"), m


def test_consumers_use_canonical_validator_imports():
    for rel in [
        ("domain", "backtesting", "engine.py"),
        ("domain", "backtesting", "external_run_service.py"),
        ("domain", "backtesting", "portfolio_manager.py"),
        ("domain", "runs", "service.py"),
        ("api", "routers", "external_backtest.py"),
    ]:
        mods = _imported_modules(_BACKEND.joinpath(*rel))
        assert "dashboard.backend.infrastructure.llm.validator" in mods, rel
        assert "dashboard.backend.llm_validator" not in mods, rel
