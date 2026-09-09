from __future__ import annotations

import pandas as pd
import pytest

from dashboard.backend.infrastructure.market_data.equity_metadata import (
    EquityMetadataUnavailableError,
    configured_dataset_path,
    enrich_decision_bars,
    load_and_enrich_us_equity_bars,
)


def _bars() -> dict[str, pd.DataFrame]:
    return {
        "AAPL": pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=pd.DatetimeIndex(
                ["2025-01-02 15:30:00+00:00", "2025-01-03 15:30:00+00:00"]
            ),
        )
    }


def test_recomputes_intraday_market_cap_and_uses_effective_sic_interval():
    source = _bars()
    market_caps = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "trade_date": ["2025-01-02", "2025-01-03"],
            # These are end-of-day values and must never be injected directly.
            "market_cap_usd": [9_999.0, 9_999.0],
            "market_cap_status": ["available", "stale"],
            "shares_outstanding_effective": [10, 12],
        }
    )
    industries = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "effective_from": ["2020-01-01", "2025-01-03"],
            "effective_to_exclusive": ["2025-01-03", "2026-01-01"],
            "sic": [3571, 7372],
            "sic_description": ["Electronic Computers", "Prepackaged Software"],
            "sic_division_name": ["Manufacturing", "Services"],
        }
    )

    enriched, summary = enrich_decision_bars(
        source,
        market_caps=market_caps,
        industries=industries,
    )

    frame = enriched["AAPL"]
    assert frame["market_cap_usd"].tolist() == [1_000.0, 1_212.0]
    assert frame["market_cap_status"].tolist() == ["available", "stale"]
    assert frame["sic_code"].tolist() == [3571, 7372]
    assert frame["industry"].tolist() == [
        "Electronic Computers",
        "Prepackaged Software",
    ]
    assert frame["sector"].tolist() == ["Manufacturing", "Services"]
    assert "market_cap_usd" not in source["AAPL"].columns
    assert summary["market_cap_rows"] == 2
    assert summary["industry_rows"] == 2


def test_rejected_reference_market_cap_stays_missing():
    caps = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "trade_date": ["2025-01-02"],
            "market_cap_usd": [None],
            "market_cap_status": ["suspect_scale"],
            "shares_outstanding_effective": [10],
        }
    )

    enriched, _ = enrich_decision_bars(_bars(), market_caps=caps)

    assert pd.isna(enriched["AAPL"]["market_cap_usd"].iloc[0])
    assert enriched["AAPL"]["market_cap_status"].iloc[0] == "suspect_scale"


def test_overlapping_industry_intervals_fail_closed():
    industries = pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "effective_from": ["2020-01-01", "2024-01-01"],
            "effective_to_exclusive": ["2025-01-01", "2026-01-01"],
            "sic": [1, 2],
            "sic_description": ["One", "Two"],
            "sic_division_name": ["A", "B"],
        }
    )

    with pytest.raises(EquityMetadataUnavailableError, match="overlapping"):
        enrich_decision_bars(_bars(), industries=industries)


def test_unconfigured_runtime_does_not_guess_a_workstation_path(monkeypatch):
    monkeypatch.delenv("US_EQUITY_DATASET_PATH", raising=False)
    monkeypatch.delenv("US_EQUITY_DATA_ROOT", raising=False)

    source = _bars()
    enriched, summary = load_and_enrich_us_equity_bars(source)

    assert configured_dataset_path() is None
    assert summary["status"] == "not_configured"
    assert enriched["AAPL"] is source["AAPL"]
