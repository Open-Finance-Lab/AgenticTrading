from typing import get_args

import pytest

from dashboard.backend.domain.agents.taxonomy import (
    AGENT_CATEGORIES,
    AgentCategory,
    coerce_category,
    normalize_category,
)


def test_categories_whitelist():
    assert AGENT_CATEGORIES == {"prompting_llms", "us_stocks", "cn_ashares"}


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


@pytest.mark.parametrize("bad", ["crypto", "futures", "Foundation", "us stocks"])
def test_coerce_rejects_unknown(bad):
    with pytest.raises(ValueError, match="unknown category"):
        coerce_category(bad)


def test_coerce_rejects_non_strings():
    with pytest.raises(ValueError, match="must be a string or null"):
        coerce_category(123)
    with pytest.raises(ValueError, match="must be a string or null"):
        coerce_category(["us_stocks"])


def test_coerce_error_does_not_echo_an_unbounded_value():
    """The rejected value lands in a 422 body and the request log. A caller can
    post a megabyte here; only a bounded prefix may be reflected."""
    with pytest.raises(ValueError) as excinfo:
        coerce_category("z" * 100_000)
    assert len(str(excinfo.value)) < 200
