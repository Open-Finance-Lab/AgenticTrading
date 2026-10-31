"""Daily Sharpe and its 95% interval -- the number both leaderboards publish.

Every expected value is hand-computed (conformance case M03, design doc
2026-10-07 §4) or derived from the formula's own terms, never read back from
the function under test. P19: one return per US trading day, ``E_0`` the seed,
ddof 1, ``rf_annual / 252`` simple; the interval is the iid standard error.
"""

import json
import math

import pytest

from dashboard.backend.domain.backtesting.metrics import (
    SHARPE_CONFIDENCE,
    SHARPE_MIN_DAYS,
    SHARPE_RF_ANNUAL,
    SHARPE_TRADING_DAYS_PER_YEAR,
    daily_session_closes,
    daily_sharpe_basis,
    daily_sharpe_interval,
    sharpe_standard_error_daily,
)
from dashboard.backend.tests.conformance.cases import M03

# M03: r = [0, 0.02, -0.05, 0.05, 0.01] from E_0 = 100,000.
M03_CLOSES = [100000.0, 102000.0, 96900.0, 101745.0, 102762.45]
SEED = 100000.0


@pytest.fixture(autouse=True)
def _fresh_drift_log(monkeypatch):
    """The drift ERROR fires once per process per break; reset it per test."""
    from dashboard.backend.domain.backtesting import metrics

    monkeypatch.setattr(metrics, "_logged_drift", set())


def _daily(closes, *, first_day=3):
    """One 15:00 ET bar (19:00Z, open-stamped) per August 2026 session."""
    return [
        {"timestamp": f"2026-08-{first_day + i:02d}T19:00:00+00:00", "equity": equity}
        for i, equity in enumerate(closes)
    ]


def test_constants_are_the_p19_rule():
    assert SHARPE_TRADING_DAYS_PER_YEAR == 252
    assert SHARPE_MIN_DAYS == 5
    assert SHARPE_CONFIDENCE == 0.95
    assert SHARPE_RF_ANNUAL == 0.0


def test_m03_anchor_with_its_cash_rate():
    out = daily_sharpe_interval(_daily(M03_CLOSES), SEED, rf_annual=0.0252)
    assert out["status"] == "ok"
    assert out["days"] == 5
    assert out["rf_annual"] == 0.0252
    assert out["sharpe"] == pytest.approx(2.568186, abs=1e-6)
    assert out["ci_low"] == pytest.approx(-11.436927, abs=1e-6)
    assert out["ci_high"] == pytest.approx(16.573299, abs=1e-6)


def test_m03_at_the_boards_cash_rate_of_zero():
    out = daily_sharpe_interval(_daily(M03_CLOSES), SEED)
    assert out["status"] == "ok"
    assert out["rf_annual"] == 0.0
    assert out["sharpe"] == pytest.approx(2.611715, abs=1e-6)
    assert out["ci_low"] == pytest.approx(-11.396490, abs=1e-6)
    assert out["ci_high"] == pytest.approx(16.619919, abs=1e-6)


def test_matches_the_conformance_case_m03_as_stated():
    """The suite's own case object, so a transcription here cannot drift."""
    points = [
        {"timestamp": stamp.isoformat(), "equity": float(value)}
        for stamp, value in M03.expected.equity
    ]
    out = daily_sharpe_interval(
        points, float(M03.initial_cash), rf_annual=float(M03.rf_annual)
    )
    assert out["sharpe"] == pytest.approx(float(M03.expected.sharpe), abs=1e-6)


def test_standard_error_formula():
    assert sharpe_standard_error_daily(1.0, 5) == pytest.approx(0.5477225575, abs=1e-9)
    assert sharpe_standard_error_daily(0.0, 5) == pytest.approx(0.4472135955, abs=1e-9)
    assert sharpe_standard_error_daily(0.0, 20) == pytest.approx(0.2236067977, abs=1e-9)
    with pytest.raises(ValueError):
        sharpe_standard_error_daily(0.0, 0)


def test_five_days_is_the_boundary():
    four = daily_sharpe_interval(_daily(M03_CLOSES[:4]), SEED)
    assert four == {
        "sharpe": None,
        "ci_low": None,
        "ci_high": None,
        "days": 4,
        "status": "insufficient_days",
        "rf_annual": 0.0,
    }
    five = daily_sharpe_interval(_daily(M03_CLOSES), SEED)
    assert five["status"] == "ok"
    assert five["days"] == 5


def test_one_day_and_no_days():
    one = daily_sharpe_interval(_daily([101000.0]), SEED)
    assert one["status"] == "insufficient_days"
    assert one["days"] == 1
    assert one["sharpe"] is None
    empty = daily_sharpe_interval([], SEED)
    assert empty["status"] == "pending"
    assert empty["days"] == 0
    assert empty["sharpe"] is None and empty["ci_low"] is None and empty["ci_high"] is None


def test_a_flat_curve_is_no_variation_not_zero():
    """M03b's rule: undefined is null. ``calculate_sharpe`` answers 0 here."""
    out = daily_sharpe_interval(_daily([SEED] * 6), SEED)
    assert out["status"] == "no_variation"
    assert out["days"] == 6
    assert out["sharpe"] is None
    assert out["ci_low"] is None and out["ci_high"] is None


def test_float_noise_is_not_a_return():
    """One ulp above the seed on every day: r_1 = 2.2e-16 and nothing after.
    Without the noise floor this publishes a Sharpe of 6.48."""
    noisy = math.nextafter(SEED, math.inf)
    out = daily_sharpe_interval(_daily([noisy] * 6), SEED)
    assert out["status"] == "no_variation"
    assert out["sharpe"] is None


def test_intraday_points_collapse_to_the_last_point_of_each_et_day():
    points = _daily(M03_CLOSES)
    # Earlier marks on each day must not count; 01:00Z on Aug 4 is 21:00 ET on
    # Aug 3, so it is Aug 3's close and the 19:00Z Aug 3 mark is not.
    points += [
        {"timestamp": f"2026-08-{d:02d}T14:00:00+00:00", "equity": 1.0}
        for d in range(3, 8)
    ]
    points[0] = {"timestamp": "2026-08-03T19:00:00+00:00", "equity": 55555.0}
    points.append({"timestamp": "2026-08-04T01:00:00+00:00", "equity": M03_CLOSES[0]})
    closes = daily_session_closes(points)
    assert [day.isoformat() for day, _ in closes] == [
        "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07",
    ]
    assert [equity for _, equity in closes] == M03_CLOSES
    out = daily_sharpe_interval(points, SEED)
    assert out["sharpe"] == pytest.approx(2.611715, abs=1e-6)


def test_order_is_by_timestamp_not_list_position():
    out = daily_sharpe_interval(list(reversed(_daily(M03_CLOSES))), SEED)
    assert out["sharpe"] == pytest.approx(2.611715, abs=1e-6)


def test_naive_timestamps_are_read_as_utc():
    # 02:00 UTC on Aug 4 is 22:00 ET on Aug 3. Read as ET it would be Aug 4.
    closes = daily_session_closes([
        {"timestamp": "2026-08-03T14:00:00", "equity": 1.0},
        {"timestamp": "2026-08-04T02:00:00", "equity": 2.0},
    ])
    assert [(day.isoformat(), equity) for day, equity in closes] == [("2026-08-03", 2.0)]


def test_zero_or_nan_equity_never_yields_nan():
    # A stored NULL (issue #390) and a NaN are "no observation", never $0.
    gappy = _daily(M03_CLOSES)
    gappy.insert(2, {"timestamp": "2026-08-04T20:00:00+00:00", "equity": float("nan")})
    gappy.insert(4, {"timestamp": "2026-08-05T20:00:00+00:00", "equity": None})
    skipped = daily_sharpe_interval(gappy, SEED)
    assert skipped["sharpe"] == pytest.approx(2.611715, abs=1e-6)

    # A return off a $0 account is undefined, not "never moved".
    wiped = daily_sharpe_interval(_daily([100000.0, 0.0, 0.0, 0.0, 0.0, 0.0]), SEED)
    assert wiped["status"] == "undefined"
    assert wiped["sharpe"] is None

    for seed in (0.0, -1.0, float("nan"), None, "junk"):
        out = daily_sharpe_interval(_daily(M03_CLOSES), seed)
        assert out["status"] == "undefined", seed

    for out in (skipped, wiped):
        json.dumps(out, allow_nan=False)


def test_a_last_day_wipeout_is_a_real_return():
    """Only a return's denominator must be positive: ending at $0 is -100%."""
    out = daily_sharpe_interval(_daily(M03_CLOSES[:4] + [0.0]), SEED)
    assert out["status"] == "ok"
    assert out["sharpe"] < 0
    json.dumps(out, allow_nan=False)


def test_the_interval_brackets_the_estimate_and_is_symmetric():
    closes = [100400.0, 99800.0, 101300.0, 100900.0, 102100.0, 101700.0, 103000.0, 102600.0]
    out = daily_sharpe_interval(_daily(closes), SEED)
    assert out["status"] == "ok" and out["days"] == 8
    assert out["ci_low"] < out["sharpe"] < out["ci_high"]
    assert out["sharpe"] - out["ci_low"] == pytest.approx(out["ci_high"] - out["sharpe"])
    down = daily_sharpe_interval(_daily([SEED * SEED / c for c in closes]), SEED)
    assert down["sharpe"] < 0
    assert down["ci_low"] < down["sharpe"] < down["ci_high"]


def test_the_basis_block_states_every_number_the_ui_quotes():
    assert daily_sharpe_basis() == {
        "basis": "daily_returns",
        "annualization_days": 252,
        "confidence": 0.95,
        "min_days": 5,
        "rf_annual": 0.0,
        "stderr": "iid",
    }


def test_programmer_errors_raise_rather_than_publish():
    with pytest.raises(ValueError):
        daily_sharpe_interval(_daily(M03_CLOSES), SEED, min_days=1)
    with pytest.raises(ValueError):
        daily_sharpe_interval(_daily(M03_CLOSES), SEED, confidence=1.0)
    with pytest.raises(ValueError):
        daily_sharpe_interval(_daily(M03_CLOSES), SEED, rf_annual=math.inf)


def test_malformed_stored_points_are_skipped_never_raised():
    """The boards call this on their GET path, so bad stored data must not 500."""
    points = _daily(M03_CLOSES) + [
        None,
        "junk",
        {"timestamp": "2026-08-07T20:00:00+00:00", "equity": 10**400},
        {"timestamp": "0001-01-01T00:00:00+00:00", "equity": 1.0},
        {"timestamp": "2026-08-07T20:00:00+00:00", "equity": True},
        {"timestamp": "2026-08-07T20:00:00+00:00", "equity": float("inf")},
        {"timestamp": 1786136400, "equity": 1.0},
    ]
    out = daily_sharpe_interval(points, SEED)
    assert out["status"] == "ok"
    assert out["sharpe"] == pytest.approx(2.611715, abs=1e-6)


def test_a_curve_with_no_usable_close_is_logged_not_silently_pending(caplog):
    """Absent and broken must differ (CLAUDE.md, fail-closed is not fail-visible)."""
    with caplog.at_level("ERROR"):
        assert daily_sharpe_interval([], SEED)["status"] == "pending"
    assert caplog.records == []
    broken = [{"timestamp": "not a time", "equity": None}] * 3
    with caplog.at_level("ERROR"):
        assert daily_sharpe_interval(broken, SEED)["status"] == "pending"
    assert any("no usable daily close" in r.getMessage() for r in caplog.records)


def test_a_bad_seed_is_logged(caplog):
    with caplog.at_level("ERROR"):
        out = daily_sharpe_interval(_daily(M03_CLOSES), None)
    assert out["status"] == "undefined"
    assert any("seed None" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level("ERROR"):
        daily_sharpe_interval(_daily([SEED] * 6), SEED)
    assert caplog.records == []


def test_a_broken_row_is_logged_once_not_on_every_board_get(caplog):
    broken = [{"timestamp": "not a time", "equity": None}] * 3
    with caplog.at_level("ERROR"):
        for _ in range(5):
            daily_sharpe_interval(broken, SEED)
            daily_sharpe_interval(_daily(M03_CLOSES), 0.0)
    messages = [r.getMessage() for r in caplog.records]
    assert sum("no usable daily close" in m for m in messages) == 1
    assert sum("seed 0.0" in m for m in messages) == 1


def test_a_bad_seed_is_reported_from_the_first_close_not_from_day_five(caplog):
    """Below min_days a broken seed used to read "needs more days" with nothing
    logged, then "never moved" from day 5: two healthy-looking stand-ins for a
    break. It is its own status, and logged, on day 1."""
    with caplog.at_level("ERROR"):
        out = daily_sharpe_interval(_daily(M03_CLOSES[:1]), 0.0)
    assert out["status"] == "undefined"
    assert out["days"] == 1
    assert out["sharpe"] is None and out["ci_low"] is None and out["ci_high"] is None
    assert any("seed 0.0" in r.getMessage() for r in caplog.records)
    # A healthy seed below min_days is still only "not enough days yet".
    assert daily_sharpe_interval(_daily(M03_CLOSES[:3]), SEED)["status"] == "insufficient_days"
    # And no closes at all stays pending, whatever the seed.
    assert daily_sharpe_interval([], None)["status"] == "pending"


def test_overflow_is_undefined_not_no_variation():
    out = daily_sharpe_interval(_daily([1e-300, 1e300, 1e-300, 1e300, 1e-300]), 1e-300)
    assert out["status"] == "undefined"
    json.dumps(out, allow_nan=False)
