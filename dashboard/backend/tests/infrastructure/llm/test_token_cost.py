"""Characterization tests for token-cost estimation (Phase 3D2)."""

import ast
from pathlib import Path

from dashboard.backend.infrastructure.llm.execution.models import PricingSnapshot
from dashboard.backend.infrastructure.llm.token_cost import (
    CHARS_PER_TOKEN,
    estimate_cost_usd,
    estimate_tokens,
    is_free_model,
    price_for_model,
    summarize,
)

_BACKEND = Path(__file__).resolve().parents[3]


def test_is_free_model_sentinels_and_empty():
    # Sentinels / rule-based / local markers → free (treated as "no real model").
    for name in ("local-model", "rule-based", "LOCAL-MODEL", " Local-Model ",
                 "none", "baseline", "demo", None, ""):
        assert is_free_model(name) is True, name


def test_is_free_model_real_models():
    # Real hosted model ids are NOT free.
    for name in ("claude-haiku-4-5-20251001", "anthropic/claude-sonnet-4-6",
                 "openai/gpt-5.5", "gemini-3.1-pro"):
        assert is_free_model(name) is False, name


def test_chars_per_token_constant():
    assert CHARS_PER_TOKEN == 3.8


def test_estimate_tokens_string_and_none():
    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    # ceil(len/3.8), min 1
    assert estimate_tokens("a") == 1
    assert estimate_tokens("x" * 38) == 10


def test_estimate_tokens_json_object():
    # json.dumps with compact separators
    assert estimate_tokens({"a": 1}) >= 1


def test_price_for_model_table_and_markers():
    assert price_for_model("claude-opus-4") == (15.0, 75.0)
    assert price_for_model("gpt-4o-mini") == (0.15, 0.60)
    assert price_for_model("rule-based") == (0.0, 0.0)
    assert price_for_model("local-model") == (0.0, 0.0)
    # Unknown but real-looking -> default pricing.
    assert price_for_model("some-unknown-model") == (1.0, 5.0)
    assert price_for_model("") == (1.0, 5.0)
    assert price_for_model(None) == (1.0, 5.0)


def test_gpt_5_5_pricing_snapshot_uses_canonical_catalog_id():
    snapshot = PricingSnapshot.from_model(
        "openai/gpt-5.5",
        "openai",
    )

    assert snapshot.model_id == "openai/gpt-5.5"
    assert snapshot.provider_id == "openai"
    assert snapshot.input_usd_per_million_tokens == 5.0
    assert snapshot.output_usd_per_million_tokens == 30.0


def test_estimate_cost_usd_formula():
    # claude-opus-4: (1_000_000/1e6)*15 + (1_000_000/1e6)*75 = 90.0
    assert estimate_cost_usd("claude-opus-4", 1_000_000, 1_000_000) == 90.0
    assert estimate_cost_usd("rule-based", 1_000_000, 1_000_000) == 0.0


def test_summarize_schema():
    out = summarize("gpt-4o", 100, 50, llm_calls=3)
    assert out == {
        "model": "gpt-4o",
        "llm_calls": 3,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "est_cost_usd": estimate_cost_usd("gpt-4o", 100, 50),
    }


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


def test_token_cost_has_no_api_scripts_or_frontend_imports():
    mods = _imported_modules(_BACKEND / "infrastructure" / "llm" / "token_cost.py")
    for m in mods:
        assert not m.startswith("dashboard.backend.api"), m
        assert not m.startswith("dashboard.scripts"), m


def test_consumers_use_canonical_token_cost():
    for rel in [
        ("domain", "backtesting", "engine.py"),
        ("domain", "backtesting", "external_run_service.py"),
        ("domain", "leaderboard", "service.py"),
    ]:
        mods = _imported_modules(_BACKEND.joinpath(*rel))
        assert "dashboard.backend.infrastructure.llm.token_cost" in mods, rel
        assert "dashboard.backend.token_cost" not in mods, rel
