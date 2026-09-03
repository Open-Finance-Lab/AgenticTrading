"""Pool boundaries, filter-before-limit semantics and reference provenance."""
import csv
import json

import pytest

from dashboard.backend.infrastructure.market_data.strategy_universe import (
    UniverseConfigurationError, resolve_strategy_universe,
    select_strategy_universe, validate_selection,
)


def security(symbol, classification="ordinary_share", **kwargs):
    return {"symbol": symbol, "classification": classification, "status": "active",
            "tradable": "True", "exchange": "NYSE", **kwargs}


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    rows = [security(f"S{i:03}", market_cap_usd=str(20_000_000_000 if i < 35 else 1_000_000_000))
            for i in range(70)]
    rows += [security(f"F{i:03}", "etp", market_cap_usd="") for i in range(40)]
    path = tmp_path / "catalog.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setenv("US_EQUITY_CATALOG", str(path))
    monkeypatch.delenv("US_EQUITY_MARKET_CAPS", raising=False)
    return path, rows


@pytest.mark.parametrize("pool,count", [("ordinary", 70), ("fund", 40), ("large_cap", 35),
                                       ("small_mid_cap", 35), ("all", 110)])
def test_every_category_filters_before_limit(catalog, pool, count):
    full = resolve_strategy_universe(pool, "all")
    limited = resolve_strategy_universe(pool, "top30")
    assert len(full["symbols"]) == count
    assert limited["symbols"] == full["symbols"][:30]
    assert limited["eligible_count"] == full["eligible_count"] == count
    assert limited["selected_count"] == 30
    assert len(full["catalog_sha256"]) == 64
    assert validate_selection(json.loads(json.dumps(full))) == full


def test_cap_boundary_and_smaller_pool():
    rows = [security("LOW", market_cap_usd=9_999_999_999),
            security("EDGE", market_cap_usd=10_000_000_000),
            security("FUND", "etp", market_cap_usd=90_000_000_000)]
    assert select_strategy_universe(rows, "large_cap")["symbols"] == ["EDGE"]
    assert select_strategy_universe(rows, "small_mid_cap")["symbols"] == ["LOW"]


def test_funds_and_all_exclude_non_equities_inactive_and_otc():
    rows = [security("STOCK"), security("ETF", "etp"),
            security("CEF", "other_security_type", figi_security_types="Closed-End Fund"),
            security("ADR", "depositary_receipt"), security("WARRANT", "warrant"),
            security("UNKNOWN", "unresolved"), security("OTC", exchange="OTC"),
            security("HALT", tradable=False), security("DEAD", status="inactive")]
    assert select_strategy_universe(rows, "fund")["symbols"] == ["CEF", "ETF"]
    assert select_strategy_universe(rows, "all")["symbols"] == ["CEF", "ETF", "STOCK"]


@pytest.mark.parametrize("cap", [None, "", "bad", "nan", "inf", 0, -1])
def test_missing_or_invalid_caps_do_not_silently_shrink_all(cap):
    rows = [security("OK", market_cap_usd=20_000_000_000), security("MISSING", market_cap_usd=cap)]
    with pytest.raises(UniverseConfigurationError, match="lack valid market_cap_usd"):
        select_strategy_universe(rows, "large_cap", "all")
    assert len(select_strategy_universe(rows, "ordinary", "all")["symbols"]) == 2


def test_cap_overlay_and_frozen_selection(catalog, tmp_path, monkeypatch):
    path, rows = catalog
    caps = tmp_path / "caps.csv"
    caps.write_text("symbol,market_cap_usd\nS000,100\n", encoding="utf-8")
    monkeypatch.setenv("US_EQUITY_MARKET_CAPS", str(caps))
    selection = resolve_strategy_universe("large_cap", "all")
    assert "S000" not in selection["symbols"]
    assert "market_caps_sha256" in selection
    path.unlink()
    assert validate_selection(selection)["selected_count"] == 34


def test_invalid_empty_and_duplicate_catalogs():
    for pool, mode in [("typo", "all"), ("all", "typo")]:
        with pytest.raises(ValueError):
            select_strategy_universe([security("A")], pool, mode)
    with pytest.raises(ValueError, match="No eligible"):
        select_strategy_universe([security("A")], "fund")
    with pytest.raises(UniverseConfigurationError, match="unique"):
        select_strategy_universe([security("A"), security("A")], "all")


def test_missing_catalog_is_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setenv("US_EQUITY_CATALOG", str(tmp_path / "missing.csv"))
    with pytest.raises(UniverseConfigurationError, match="US_EQUITY_CATALOG"):
        resolve_strategy_universe("all")
