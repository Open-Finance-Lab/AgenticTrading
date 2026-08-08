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

_EXPECTED_NEW = {
    "contrarian-dip-buyer": ("openai/gpt-5.5", "us_stocks"),
    "sector-rotator": ("google/gemini-3.1-pro-preview", "us_stocks"),
    "volatility-guard": ("deepseek/deepseek-v4-pro", "us_stocks"),
    "ashare-momentum-t1": ("qwen/qwen3.7-plus", "cn_ashares"),
}


@pytest.mark.parametrize("template", _CATALOG, ids=lambda t: t["template_id"])
def test_every_template_runs_a_supported_or_hosted_model(template):
    if template.get("runtime_type"):
        return  # hosted runtime: its model is not user-selectable
    assert template["model_name"] in _SUPPORTED_SLUGS, (
        f"{template['template_id']} runs {template['model_name']!r}, "
        "which is not in SUPPORTED_MODELS"
    )


@pytest.mark.parametrize("template_id,expected", sorted(_EXPECTED_NEW.items()))
def test_new_templates_are_present_with_their_pairings(template_id, expected):
    found = next((t for t in _CATALOG if t["template_id"] == template_id), None)
    assert found is not None, f"{template_id} missing from marketplace.json"
    assert (found["model_name"], found["category"]) == expected


def test_catalog_covers_every_pickable_vendor():
    """The facet is decorative if most of its chips are empty."""
    vendors = {t["model_name"].split("/", 1)[0] for t in _CATALOG}
    assert {"anthropic", "openai", "google", "deepseek", "qwen"} <= vendors
