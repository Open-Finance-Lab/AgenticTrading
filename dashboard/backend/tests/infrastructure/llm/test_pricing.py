"""Guards for the pricing leaf module (CodeQL #1263 / #1264 ``py/cyclic-import``).

``execution/models.py`` used to import ``token_cost.price_for_model`` inside
``PricingSnapshot.from_model`` while ``token_cost`` imported the pydantic
usage/billing models from ``execution/models`` at module scope — a cycle that
only held together because one side was function-local. The price table now
lives in ``pricing.py``, a leaf that both sides import.
"""

import ast
from pathlib import Path

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
