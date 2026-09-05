"""The previewed representative universe is the backend execution universe."""
import json
import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.api.routers import backtests
from dashboard.backend.infrastructure.market_data import strategy_universe as pools


@pytest.mark.parametrize("pool", ["ordinary", "fund", "all"])
def test_preview_and_worker_share_roster_without_workstation_catalog(pool, monkeypatch, tmp_path):
    monkeypatch.setenv("US_EQUITY_CATALOG", str(tmp_path / "missing.csv"))
    monkeypatch.setenv("US_EQUITY_MARKET_CAPS", str(tmp_path / "missing-caps.csv"))
    captured = {}

    class Thread:
        def __init__(self, **kwargs):
            captured.update(kwargs["kwargs"])

        def start(self):
            pass

    monkeypatch.setattr(backtests, "_BackgroundThread", Thread)
    client = TestClient(app)
    options = client.get("/config/stock-pools")
    assert options.status_code == 200
    presets = options.json()["representative_presets"]
    assert {p["stock_pool"] for p in presets} == {"ordinary", "fund", "all"}
    preview = next(p for p in presets if p["stock_pool"] == pool)
    response = client.post("/backtest/run", json={
        "stock_pool": pool, "pool_mode": "representative30", "decision_source": "rule_based",
    }, headers={"X-Session-Id": str(uuid.uuid4())})
    assert response.status_code == 200, response.text
    assert response.json()["universe_selection"] == captured["universe_selection"] == preview
    assert captured["assets"] == preview["symbols"]
    assert len(set(preview["symbols"])) == preview["selected_count"] == 30
    assert preview["selection_order"] == "curated_coverage"
    assert preview["roster_version"]


def test_rosters_cover_sectors_and_asset_classes():
    ordinary, funds, mixed = pools.representative_presets()
    assert len(ordinary["groups"]) == 11
    assert set(ordinary["symbols"]).isdisjoint(funds["symbols"])
    assert len(set(mixed["symbols"]) & set(ordinary["symbols"])) == 15
    assert len(set(mixed["symbols"]) & set(funds["symbols"])) == 15
    assert {"SPY", "TLT", "GLD", "VEA", "VWO"} <= set(funds["symbols"])


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_category"])
def test_invalid_reference_roster_is_not_silently_replaced(monkeypatch, tmp_path, mutation):
    registry = json.loads((pools.CONFIG_DIR / "representative_universes.json").read_text())
    groups = registry["presets"]["ordinary"]["groups"]
    groups[0]["symbols"][0] = "MSFT" if mutation == "duplicate" else "SPY"
    (tmp_path / "representative_universes.json").write_text(json.dumps(registry))
    monkeypatch.setattr(pools, "CONFIG_DIR", tmp_path)
    with pytest.raises(pools.UniverseConfigurationError):
        pools.resolve_strategy_universe("ordinary", "representative30")


@pytest.mark.parametrize("pool", ["large_cap", "small_mid_cap"])
def test_market_cap_categories_have_no_representative_ui_roster(pool):
    with pytest.raises(ValueError, match="supports ordinary, fund and all"):
        pools.resolve_strategy_universe(pool, "representative30")
