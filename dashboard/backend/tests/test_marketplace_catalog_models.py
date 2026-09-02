"""Every catalog template must run on a model the platform actually offers.

A template on an unlisted model is invisible trouble: it clones fine, then the
Run Backtest picker cannot represent its model. The only exception is a hosted
runtime, whose model is a property of the runtime rather than a user choice.
"""

import json
import re
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import js_const

_CATALOG = json.loads(
    (Path(__file__).resolve().parents[3] / "dashboard/config/marketplace.json").read_text(
        encoding="utf-8"
    )
)["templates"]

_SUPPORTED_SLUGS = set(re.findall(r"slug:\s*'([^']+)'", js_const("SUPPORTED_MODELS")))

# Nemotron is on the Competition Leaderboard via OpenRouter but is not in
# SUPPORTED_MODELS (the user-facing picker). The supermarket still ships a
# card for it so the catalog matches the board.
_LEADERBOARD_ONLY_SLUGS = {"nvidia/nemotron-3-nano-30b-a3b"}

_EXPECTED_MODELS = {
    "claude-haiku-4-5": ("anthropic/claude-haiku-4-5", "us_stocks"),
    "claude-sonnet-4-6": ("anthropic/claude-sonnet-4-6", "us_stocks"),
    "gpt-5-5": ("openai/gpt-5.5", "us_stocks"),
    "gemini-3-1-pro-preview": ("google/gemini-3.1-pro-preview", "us_stocks"),
    "deepseek-v4-pro": ("deepseek/deepseek-v4-pro", "us_stocks"),
    "qwen3-7-plus": ("qwen/qwen3.7-plus", "us_stocks"),
    "nemotron-3-nano-30b": ("nvidia/nemotron-3-nano-30b-a3b", "us_stocks"),
}

_LEADERBOARD = json.loads(
    (Path(__file__).resolve().parents[3] / "dashboard/config/leaderboard.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("template", _CATALOG, ids=lambda t: t["template_id"])
def test_every_template_runs_a_supported_or_hosted_model(template):
    if template.get("runtime_type"):
        return  # hosted runtime: its model is not user-selectable
    assert template["model_name"] in (_SUPPORTED_SLUGS | _LEADERBOARD_ONLY_SLUGS), (
        f"{template['template_id']} runs {template['model_name']!r}, "
        "which is not in SUPPORTED_MODELS or the leaderboard"
    )


@pytest.mark.parametrize("template_id,expected", sorted(_EXPECTED_MODELS.items()))
def test_leaderboard_model_cards_are_present_with_their_pairings(template_id, expected):
    found = next((t for t in _CATALOG if t["template_id"] == template_id), None)
    assert found is not None, f"{template_id} missing from marketplace.json"
    assert (found["model_name"], found["category"]) == expected


def test_every_leaderboard_llm_has_a_supermarket_card():
    """The supermarket model cards are the board's llm_agent roster, by name.

    A new competition model that ships on the board without a card (or a card
    that outlives its board entry) is otherwise invisible until someone notices.
    AI Hedge Fund is a hosted runtime, not a board entry, and is excluded.
    """
    board_names = {
        entry["name"]
        for entry in _LEADERBOARD["strategies"]
        if entry.get("strategy") == "llm_agent"
    }
    catalog_names = {
        template["name"]
        for template in _CATALOG
        if template.get("shelf") == "llms"
    }
    assert catalog_names == board_names
    # The API payload exposes each entry's *model* string (service.py builds
    # entries with "model", never "name"), and app.js joins cards to entries on
    # it -- pin the join key itself, not just the config's display name.
    board_models = {
        entry["model"]
        for entry in _LEADERBOARD["strategies"]
        if entry.get("strategy") == "llm_agent"
    }
    assert catalog_names == board_models


def test_llm_template_ids_map_to_leaderboard_entry_ids():
    """app.js's fallback join is template_id with '-' -> '_' against entry_id.

    The primary join is the name/model string; this fallback exists for the day
    those drift, so it must actually match -- 'gemini-3-1-pro' silently missed
    'gemini_3_1_pro_preview' until the template_id was renamed.
    """
    board_ids = {
        entry["id"]
        for entry in _LEADERBOARD["strategies"]
        if entry.get("strategy") == "llm_agent"
    }
    catalog_ids = {
        template["template_id"].replace("-", "_")
        for template in _CATALOG
        if template.get("shelf") == "llms"
    }
    assert catalog_ids == board_ids


def test_llm_card_prompts_pin_the_default_starter_instruction():
    """The 7 LLM cards ship the starter instruction verbatim -- a third copy of
    DEFAULT_STARTER_INSTRUCTION (defaults.py already mirrors it to app.js under
    a pin test). Pin this copy too, so tuning the constant cannot silently
    strand the cards on stale wording."""
    from dashboard.backend.domain.agents.defaults import DEFAULT_STARTER_INSTRUCTION

    for template in _CATALOG:
        if template.get("shelf") != "llms":
            continue
        [step] = template["pipeline"]
        assert step["prompt"] == DEFAULT_STARTER_INSTRUCTION, template["template_id"]


def test_restored_strategy_templates_sit_on_the_agents_shelf():
    restored = {
        "balanced-starter",
        "momentum-scout",
        "pipeline-analyst",
        "blue-chip-steady",
        "even-split-dow",
        "ashare-steady-t1",
        "contrarian-dip-buyer",
        "sector-rotator",
        "volatility-guard",
        "ashare-momentum-t1",
    }
    by_id = {template["template_id"]: template for template in _CATALOG}
    assert restored <= set(by_id)
    for template_id in restored:
        assert by_id[template_id].get("shelf") == "open", template_id


def test_catalog_rows_declare_a_supermarket_shelf():
    shelves = {template["template_id"]: template.get("shelf") for template in _CATALOG}
    assert shelves["ai-hedge-fund"] == "open"
    for template_id in _EXPECTED_MODELS:
        assert shelves[template_id] == "llms", template_id


def test_catalog_covers_every_pickable_vendor():
    """The facet is decorative if most of its chips are empty."""
    vendors = {t["model_name"].split("/", 1)[0] for t in _CATALOG}
    assert {"anthropic", "openai", "google", "deepseek", "qwen"} <= vendors


def test_catalog_includes_both_markets():
    """Community chips follow the catalog. A-share templates are back on the
    Agents shelf, so the China A-Share chip ships again without a hardcoded
    chip list.
    """
    assert {t.get("category") for t in _CATALOG} == {"us_stocks", "cn_ashares"}
