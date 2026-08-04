from dashboard.backend.domain.agents.taxonomy import AGENT_CATEGORIES, normalize_category

def test_categories_whitelist():
    assert AGENT_CATEGORIES == {"prompting_llms", "us_stocks", "cn_ashares"}

def test_normalize_valid_passthrough_and_case():
    assert normalize_category("us_stocks") == "us_stocks"
    assert normalize_category(" US_STOCKS ") == "us_stocks"

def test_normalize_unknown_and_legacy_to_none():
    assert normalize_category("Foundation") is None   # legacy marketplace value
    assert normalize_category("Hosted") is None
    assert normalize_category("") is None
    assert normalize_category(None) is None
