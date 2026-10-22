"""Guards for the pricing leaf module (CodeQL #1263 / #1264 ``py/cyclic-import``).

``execution/models.py`` used to import ``token_cost.price_for_model`` inside
``PricingSnapshot.from_model`` while ``token_cost`` imported the pydantic
usage/billing models from ``execution/models`` at module scope — a cycle that
only held together because one side was function-local. The price table now
lives in ``pricing.py``, a leaf that both sides import.
"""

import ast
import hashlib
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
    "model",
    ["acme/nonexistent", "vendor/model-localized-v1", "acme/demo-7b", "acme/baseline-xl"],
)
def test_a_real_id_containing_a_free_marker_is_not_free(model):
    """#590: ``none``/``local``/``demo``/``baseline`` inside a real id used to
    price it at $0/$0, so a platform call reserved nothing and settled at 0."""
    assert pricing.price_for_model(model) != (0.0, 0.0)
    assert pricing.is_free_model(model) is False


def test_an_unlisted_id_containing_a_free_marker_gets_the_default_price():
    assert pricing.price_for_model("acme/nonexistent") == (1.0, 5.0)


def test_a_free_variant_is_still_a_real_model_to_is_free_model():
    # The Discord bot forwards whatever is_free_model rejects; an OpenRouter
    # ``:free`` id is a real model it must keep forwarding.
    assert pricing.is_free_model("openrouter/some-model:free") is False


@pytest.mark.parametrize(
    "model", [*sorted(pricing.FREE_MODEL_NAMES), "LOCAL-MODEL", " Local-Model "]
)
def test_the_exact_sentinels_are_still_free(model):
    assert pricing.price_for_model(model) == (0.0, 0.0)
    assert pricing.is_free_model(model) is True


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


def test_the_shipped_example_clients_default_name_is_free():
    """``rule-based-demo`` is the example client's default ``--model-name``
    (``dashboard/examples/external_agent_client.py``), so stored agents carry
    it. It was free only by accident of the substring rule; listed by name now,
    or the Discord bot forwards it as a model and its runs show a $1/$5 cost."""
    assert pricing.is_free_model("rule-based-demo") is True
    assert pricing.price_for_model("rule-based-demo") == (0.0, 0.0)
    assert pricing.listed_price_for_model("rule-based-demo") == (0.0, 0.0)


def test_every_rule_based_label_is_a_free_name():
    """Provenance's rule-based labels come from the same constant and are a
    subset of the free names: a run nothing drove made no paid call. Not equal
    sets -- a ``local-model`` run is free, but a self-hosted model drove it."""
    from dashboard.backend.domain.backtesting import provenance

    assert pricing.RULE_BASED_MODEL_LABELS <= pricing.FREE_MODEL_NAMES
    assert provenance._RULE_BASED_MODEL_LABELS == {"", *pricing.RULE_BASED_MODEL_LABELS}
    for label in ("rule_based", "null"):
        assert pricing.price_for_model(label) == (0.0, 0.0)


@pytest.mark.parametrize("model", [None, "", "   "])
def test_no_model_at_all_is_free_wherever_it_is_priced(model):
    """``is_free_model`` and ``price_for_model`` agree that no model costs $0.
    It cannot reach a reservation (``PricingSnapshot.model_id`` has
    ``min_length=1``), so only estimates see it. ``listed_price_for_model``
    still answers None: a published BYOK estimate for an unnamed model reads
    as unpriced, not as a claimed $0."""
    assert pricing.is_free_model(model) is True
    assert pricing.price_for_model(model) == (0.0, 0.0)
    assert pricing.listed_price_for_model(model) is None


@pytest.mark.parametrize(
    "model",
    [
        *sorted(pricing.FREE_MODEL_NAMES),
        "RULE-BASED-DEMO",
        "acme/nonexistent",
        "acme/demo-7b",
        "openrouter/some-model:free",
        "anthropic/claude-haiku-4-5",
        "acme/unlisted",
    ],
)
def test_the_three_functions_agree_on_which_names_are_sentinels(model):
    free = pricing.is_free_model(model)
    assert (pricing.price_for_model(model) == (0.0, 0.0)) is free
    if not model.endswith(":free"):  # listed at $0 but a real model; see above
        assert (pricing.listed_price_for_model(model) == (0.0, 0.0)) is free


# Everything that decides a price, fingerprinted and pinned beside the version
# that stamps it. Change the table, the sentinels or the default and this
# fails until the version and the pin move together.
_PRICED_BY = {
    "pricing-table-2026-10-22": "49fb4f3bccad267f",
}


def _pricing_fingerprint() -> str:
    material = repr(
        (pricing._PRICING_TABLE, sorted(pricing.FREE_MODEL_NAMES), pricing._DEFAULT_PRICING)
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def test_source_version_moves_with_the_rates():
    """A stored ``PricingSnapshot`` names the rule that priced it only if the
    version moves whenever a price does (#648 moved ``acme/demo-7b`` from $0
    to $1/$5). On failure: bump ``PRICING_SOURCE_VERSION``, then re-pin."""
    assert _PRICED_BY.get(pricing.PRICING_SOURCE_VERSION) == _pricing_fingerprint()
