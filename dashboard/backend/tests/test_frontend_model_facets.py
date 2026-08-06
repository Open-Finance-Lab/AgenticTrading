"""Guards for the Community model-vendor facet.

The vendor axis is a pure derivation from `model_name` -- no column, no
migration. MODEL_VENDORS is its single source of truth: chip order, display
label and open/closed licence all come from one table, so a badge cannot drift
from the vendor it describes. A wrong badge is a factual claim about someone
else's product.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body, js_const

_CATALOG = json.loads(
    (Path(__file__).resolve().parents[3] / "dashboard/config/marketplace.json").read_text(
        encoding="utf-8"
    )
)["templates"]

EXPECTED_VENDORS = [
    ("anthropic", "anthropic/", "Claude", "closed"),
    ("openai", "openai/", "GPT", "closed"),
    ("google", "google/", "Gemini", "closed"),
    ("deepseek", "deepseek/", "DeepSeek", "open"),
    ("qwen", "qwen/", "Qwen", "open"),
    ("nvidia", "nvidia/nemotron", "NVIDIA Nemotron", "open"),
    ("meta", "meta-llama/", "Llama", "open"),
    ("xai", "x-ai/", "Grok", "closed"),
]


def _vendor_rows():
    return re.findall(
        r"key:\s*'([^']+)',\s*prefix:\s*'([^']+)',\s*label:\s*'([^']+)',\s*licence:\s*'([^']+)'",
        js_const("MODEL_VENDORS"),
    )


def test_vendor_table_is_pinned_including_licence():
    assert _vendor_rows() == EXPECTED_VENDORS


def test_every_catalog_model_matches_a_vendor_prefix():
    """The highest-value guard: a template on an unmatched prefix renders as
    "AI-powered" with no chip and no badge, which is otherwise invisible."""
    prefixes = [row[1] for row in _vendor_rows()]
    for template in _CATALOG:
        model = template["model_name"].lower()
        assert any(model.startswith(p) for p in prefixes), (
            f"{template['template_id']} runs {model!r}, which matches no MODEL_VENDORS prefix"
        )


def test_every_supported_model_matches_a_vendor_prefix():
    prefixes = [row[1] for row in _vendor_rows()]
    for slug in re.findall(r"slug:\s*'([^']+)'", js_const("SUPPORTED_MODELS")):
        assert any(slug.lower().startswith(p) for p in prefixes), slug


def test_supported_model_vendor_fields_agree_with_the_vendor_table():
    """SUPPORTED_MODELS carries its own `vendor` key; it must not drift."""
    by_prefix = {row[1]: row[0] for row in _vendor_rows()}
    pairs = re.findall(
        r"slug:\s*'([^']+)',\s*label:\s*'[^']+',\s*vendor:\s*'([^']+)'",
        js_const("SUPPORTED_MODELS"),
    )
    for slug, vendor in pairs:
        expected = next(k for p, k in by_prefix.items() if slug.lower().startswith(p))
        assert vendor == expected, f"{slug} is tagged {vendor!r}, table says {expected!r}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_provider_label_output_is_unchanged():
    """These six strings ship on cards today. The refactor must not touch them."""
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function formatModelProviderLabel")}
const cases = ['anthropic/claude-haiku-4-5', 'nvidia/nemotron-3-nano-30b-a3b',
               'deepseek/deepseek-v4-pro', 'openai/gpt-5.5',
               'google/gemini-3.1-pro-preview', 'qwen/qwen3.7-plus',
               'totally/unknown', ''];
console.log(JSON.stringify(cases.map(formatModelProviderLabel)));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "Powered by Claude",
        "Powered by NVIDIA Nemotron",
        "Powered by DeepSeek",
        "Powered by GPT",
        "Powered by Gemini",
        "Powered by Qwen",
        "AI-powered",
        "AI-powered",
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_unknown_vendor_resolves_to_empty_string():
    """Same contract as agentMarketKey: unknown stays visible under All and is
    excluded only by an explicit chip -- never hidden, never defaulted."""
    script = f"""
{js_const("MODEL_VENDORS")}
{fn_body("function modelVendorKey")}
{fn_body("function agentVendorKey")}
{fn_body("function modelVendorLicence")}
console.log(JSON.stringify([
  modelVendorKey('totally/unknown'), modelVendorKey(null), modelVendorKey('local-model'),
  agentVendorKey({{model_name: 'qwen/qwen3.7-plus'}}), agentVendorKey(null),
  modelVendorLicence('deepseek/deepseek-v4-pro'),
  modelVendorLicence('anthropic/claude-haiku-4-5'),
  modelVendorLicence('totally/unknown'),
]));
"""
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["", "", "", "qwen", "", "open", "closed", ""]
