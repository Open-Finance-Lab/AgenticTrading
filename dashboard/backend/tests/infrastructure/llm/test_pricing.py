"""Guards for the pricing leaf module (CodeQL #1263 / #1264 ``py/cyclic-import``).

``execution/models.py`` used to import ``token_cost.price_for_model`` inside
``PricingSnapshot.from_model`` while ``token_cost`` imported the pydantic
usage/billing models from ``execution/models`` at module scope — a cycle that
only held together because one side was function-local. The price table now
lives in ``pricing.py``, a leaf that both sides import.
"""

import ast
from pathlib import Path

import pytest

from dashboard.backend.infrastructure.llm import pricing, token_cost
from dashboard.backend.infrastructure.llm.execution.models import PricingSnapshot

_LLM = Path(__file__).resolve().parents[3] / "infrastructure" / "llm"


def _imported_modules(path: Path) -> set[str]:
    # ``ast.walk`` reaches function-local imports too — the cycle hid in one.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


def test_pricing_is_a_leaf_module():
    mods = _imported_modules(_LLM / "pricing.py")
    assert not any(m.startswith("dashboard") for m in mods), mods


def test_execution_models_never_import_token_cost():
    mods = _imported_modules(_LLM / "execution" / "models.py")
    assert "dashboard.backend.infrastructure.llm.token_cost" not in mods


def test_token_cost_reexports_the_pricing_api():
    # ``discord_bot`` imports ``is_free_model`` from ``token_cost`` (pinned by
    # test_discord_wiring); the split must keep one object behind both names.
    assert token_cost.price_for_model is pricing.price_for_model
    assert token_cost.is_free_model is pricing.is_free_model
    assert token_cost.PRICING_SOURCE_VERSION == pricing.PRICING_SOURCE_VERSION


def test_snapshot_default_source_version_follows_the_table():
    # ``from_model`` used to carry its own copy of the version string, so a bump
    # to the table left every snapshot claiming the old version.
    snapshot = PricingSnapshot.from_model("gpt-4o", "openai")
    assert snapshot.source_version == pricing.PRICING_SOURCE_VERSION
    assert (
        snapshot.input_usd_per_million_tokens,
        snapshot.output_usd_per_million_tokens,
    ) == (2.50, 10.0)


def test_haiku_on_commonstack_prices_at_the_listed_rate():
    # CommonStack lists anthropic/claude-haiku-4-5 at $1 / $5 per million
    # (checked 2026-09-23). The claude-haiku-4 entry already matches, so the
    # #535 allowlist addition needed no pricing-table change.
    assert pricing.price_for_model("anthropic/claude-haiku-4-5") == (1.0, 5.0)


def test_listed_price_is_none_where_price_for_model_guesses():
    """``price_for_model`` falls back to $1/$5 so a reservation always has a
    ceiling. ``listed_price_for_model`` backs a published estimate, where that
    guess would be an invented number, so an unlisted model answers None."""
    assert pricing.price_for_model("acme/unlisted-model") == (1.0, 5.0)
    assert pricing.listed_price_for_model("acme/unlisted-model") is None
    assert pricing.listed_price_for_model("") is None
    assert pricing.listed_price_for_model(None) is None
    assert pricing.listed_price_for_model("openai/gpt-5.5") == (5.0, 30.0)


def test_listed_price_of_a_free_variant_is_zero_not_its_paid_siblings():
    # OpenRouter's ":free" variant matches the paid slug's needle; billing keeps
    # that behaviour, the estimate must not.
    assert pricing.price_for_model("openai/gpt-5.5:free") == (5.0, 30.0)
    assert pricing.listed_price_for_model("openai/gpt-5.5:free") == (0.0, 0.0)
    assert pricing.listed_price_for_model("rule-based") == (0.0, 0.0)


def test_listed_price_does_not_borrow_a_sibling_needles_rate():
    """The billing match is by substring, so gpt-4.1-nano reads gpt-4.1's rate
    (~20x its own) and o3-pro o3's (~10x below it). A published estimate lists
    only a name that *is* a table entry's, less provider prefix and snapshot
    suffix."""
    assert pricing.price_for_model("openai/gpt-4.1-nano") == (2.0, 8.0)
    assert pricing.listed_price_for_model("openai/gpt-4.1-nano") is None
    assert pricing.listed_price_for_model("openai/o3-pro") is None
    assert pricing.listed_price_for_model("claude-opus-4-5") is None
    assert pricing.listed_price_for_model("openai/gpt-5.5:nitro") is None
    # A substring free marker is no reason to call a real model free.
    assert pricing.listed_price_for_model("acme/nonexistent") is None


@pytest.mark.parametrize(
    ("model", "price"),
    [
        # The ATL catalog, by catalog id and by the native providers' own id.
        ("anthropic/claude-haiku-4-5", (1.0, 5.0)),
        ("claude-haiku-4-5", (1.0, 5.0)),
        ("anthropic/claude-sonnet-4-6", (3.0, 15.0)),
        ("claude-sonnet-4-6", (3.0, 15.0)),
        ("openai/gpt-5.5", (5.0, 30.0)),
        ("gpt-5.5", (5.0, 30.0)),
        ("google/gemini-3.1-pro-preview", (2.0, 12.0)),
        ("gemini-3.1-pro-preview", (2.0, 12.0)),
        ("deepseek/deepseek-v4-pro", (0.435, 0.87)),
        ("qwen/qwen3.7-plus", (0.40, 1.60)),
        # A dated snapshot is the model it snapshots.
        ("claude-haiku-4-5-20251001", (1.0, 5.0)),
        ("claude-3-5-haiku-20241022", (0.80, 4.0)),
        ("gpt-4o-2024-08-06", (2.50, 10.0)),
        ("gpt-4.1", (2.0, 8.0)),
    ],
)
def test_listed_price_covers_the_catalog_and_its_snapshots(model, price):
    assert pricing.listed_price_for_model(model) == price
