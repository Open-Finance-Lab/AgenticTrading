from typing import get_args

import pytest

from dashboard.backend.domain.agents.taxonomy import (
    AGENT_CATEGORIES,
    AgentCategory,
    coerce_category,
    normalize_category,
)


def test_categories_whitelist():
    assert AGENT_CATEGORIES == {"us_stocks", "cn_ashares"}


def test_whitelist_is_derived_from_the_literal():
    """The ``Literal`` is what Pydantic validates against and what FastAPI
    publishes into openapi.json; the frozenset is what the lenient catalog path
    checks. They are one declaration so the two can never disagree."""
    assert AGENT_CATEGORIES == frozenset(get_args(AgentCategory))


# --- normalize_category: the lenient, legacy-catalog boundary ---------------


def test_normalize_valid_passthrough_and_case():
    assert normalize_category("us_stocks") == "us_stocks"
    assert normalize_category(" US_STOCKS ") == "us_stocks"


def test_normalize_unknown_and_legacy_to_none():
    assert normalize_category("Foundation") is None   # legacy marketplace value
    assert normalize_category("Hosted") is None
    assert normalize_category("") is None
    assert normalize_category(None) is None


def test_normalize_never_raises_on_junk():
    """The catalog boundary must degrade, not fail -- a legacy or malformed value
    there must still let a clone through."""
    assert normalize_category(123) is None
    assert normalize_category({"nope": 1}) is None
    assert normalize_category("x" * 5000) is None


# --- coerce_category: the strict, caller-supplied boundary -----------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("us_stocks", "us_stocks"),
        ("  us_stocks  ", "us_stocks"),
        ("US_STOCKS", "us_stocks"),
        ("Cn_AShares", "cn_ashares"),
    ],
)
def test_coerce_folds_case_and_whitespace(raw, expected):
    assert coerce_category(raw) == expected


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_coerce_treats_blank_as_clear_not_error(blank):
    """An unselected HTML <select> posts "", not null. Rejecting it would 422 the
    Configure form's "no shelf" option, so blanks clear the shelf instead."""
    assert coerce_category(blank) is None


def test_coerce_passes_none_through():
    assert coerce_category(None) is None


@pytest.mark.parametrize(
    "bad", ["crypto", "futures", "prompting_llms", "Foundation", "us stocks"]
)
def test_coerce_rejects_unknown(bad):
    with pytest.raises(ValueError, match="unknown category"):
        coerce_category(bad)


def test_coerce_rejects_non_strings():
    with pytest.raises(ValueError, match="must be a string or null"):
        coerce_category(123)
    with pytest.raises(ValueError, match="must be a string or null"):
        coerce_category(["us_stocks"])


def test_coerce_error_does_not_echo_an_unbounded_value():
    """This module's own error text stays bounded, so the log line does.

    Scope note: the 422 *body* is still unbounded, because Pydantic attaches the
    raw input under ``detail[].input`` for every validation error app-wide (same
    for `description`, and for /api/auth/login). Bounding that is a change to the
    RequestValidationError handler and does not belong to this field.
    """
    with pytest.raises(ValueError) as excinfo:
        coerce_category("z" * 100_000)
    assert len(str(excinfo.value)) < 200


# --- locked shelves + catalog migration ------------------------------------


@pytest.mark.parametrize("locked", ["crypto", "futures"])
def test_locked_shelves_have_no_category_slug(locked):
    """Crypto and Futures are inert rows in the frontend only.

    Nothing can be assigned to them -- no bar source, no MarketProfile, no
    engine support -- so they deliberately get no slug. A member here would
    make them selectable in Configure and cloneable from Community while no
    backtest could ever run.
    """
    assert locked not in AGENT_CATEGORIES
    with pytest.raises(ValueError, match="unknown category"):
        coerce_category(locked)


def test_no_template_is_left_on_the_retired_prompting_llms_slug():
    """A template stranded on the retired slug would 422 on any later PATCH and
    render under Community's "General" fallback label rather than a market chip.
    The catalog is `lru_cache`d, so this reloads it before reading.
    """
    import dashboard.backend.domain.agents.marketplace as marketplace_mod

    marketplace_mod.reload_marketplace_catalog()
    slugs = {t.get("category") for t in marketplace_mod.list_marketplace_templates()}
    assert "prompting_llms" not in slugs
    assert slugs <= AGENT_CATEGORIES
