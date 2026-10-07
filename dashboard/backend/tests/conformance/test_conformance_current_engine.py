"""US conformance suite against the current dashboard engine (adapter D) and
its LLM translator (the DL probe).

Each case is scored by ``scoring.score`` and held to its *measured* outcome:

* PASS runs plainly.
* FAIL is ``xfail(strict=True, raises=ConformanceFailure)``. The test raises
  ``ConformanceFailure`` only when the failing checks are exactly the ones the
  prediction names; failing on a different set is a plain failure, and so is
  passing (strict XPASS). **Fixing an engine defect therefore turns its tests
  red** -- by design: the fix must update the inventory in ``cases.py``
  (``PREDICTIONS_D`` / ``PREDICTIONS_DL``, and ``PREDICTION_DELTAS`` if the
  design doc said otherwise) in the same change.
* N/E (not expressible) is skipped with the design doc's reason, but only
  after the adapter has actually refused the case; an adapter that runs a case
  predicted N/E fails.

The engine runs end to end on synthesized bars with no network, no ``node``
and no optional dependency.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting.bar_aggregation import aggregate_bars_by_symbol

from .adapters.current_engine import CurrentEngineAdapter, synthesize_source_bars
from .cases import ALL_CASES, BY_ID, FAIL, NOT_EXPRESSIBLE, PASS, PREDICTIONS_D, PREDICTIONS_DL
from .model import NotExpressible
from .scoring import ConformanceFailure, report, score


def _param(case, prediction):
    marks = ()
    if prediction.verdict == FAIL:
        reason = f"{', '.join(prediction.defects)}: {prediction.note}"
        marks = (pytest.mark.xfail(strict=True, raises=ConformanceFailure, reason=reason),)
    return pytest.param(case, prediction, id=case.id, marks=marks)


def _assert_outcome(case, prediction, adapter):
    try:
        actual = adapter.run(case)
    except NotExpressible as exc:
        if prediction.verdict != NOT_EXPRESSIBLE:
            pytest.fail(f"{case.id}: predicted {prediction.verdict}, adapter refused it: {exc}")
        pytest.skip(f"N/E -- {prediction.note} (adapter: {exc})")
    if prediction.verdict == NOT_EXPRESSIBLE:
        pytest.fail(f"{case.id}: predicted N/E ({prediction.note}), but the adapter ran it")

    failures = score(case, actual)
    text = report(case.id, failures)
    if prediction.verdict == PASS:
        assert not failures, text
        return
    if not failures:
        pytest.fail(
            f"{case.id}: predicted FAIL ({', '.join(prediction.defects)}) but it now PASSES -- "
            "a defect was fixed; update the prediction in cases.py"
        )
    if set(failures) != prediction.fails:
        pytest.fail(
            f"{case.id}: prediction drift -- predicted failing checks "
            f"{sorted(prediction.fails)}, measured {sorted(failures)}\n{text}"
        )
    raise ConformanceFailure(text)


@pytest.mark.parametrize(
    "case, prediction", [_param(case, PREDICTIONS_D[case.id]) for case in ALL_CASES]
)
def test_dashboard_engine(case, prediction, monkeypatch):
    _assert_outcome(case, prediction, CurrentEngineAdapter(monkeypatch))


@pytest.mark.parametrize(
    "case, prediction",
    [_param(BY_ID[case_id], prediction) for case_id, prediction in PREDICTIONS_DL.items()],
)
def test_dashboard_llm_translator(case, prediction, monkeypatch):
    adapter = CurrentEngineAdapter(monkeypatch, llm=True)
    _assert_outcome(case, prediction, adapter)


def test_dl_probe_never_falls_back_to_rule_based(monkeypatch):
    """The DL probe measures the translator, so every step must be the model's."""
    adapter = CurrentEngineAdapter(monkeypatch, llm=True)
    adapter.run(BY_ID["C01"])
    assert adapter.fallbacks == 0


@pytest.mark.parametrize("case", ALL_CASES, ids=[case.id for case in ALL_CASES])
def test_synthesized_bars_aggregate_back_to_the_case(case):
    """§8.1, pinned: an adapter bug must never be scored as an engine defect.

    The engine's own aggregation of the synthesized 5m bars reproduces every
    hourly case bar exactly, and spans every daily one with its O/H/L/C. The
    one exception is T05's early-close bucket: the engine's session is fixed
    at 16:00 (D19), so a 12:30-13:00 bucket is an incomplete 12:30-13:30 bar
    there -- which is the defect T05 measures, not a synthesis error.
    """
    hourly = aggregate_bars_by_symbol(
        synthesize_source_bars(case), source_timeframe="5m", decision_timeframe="60m"
    )
    for bar in case.bars:
        frame = hourly[bar.symbol]
        close = pd.Timestamp(bar.close_ts).tz_convert("UTC")
        want = [float(bar.open), float(bar.high), float(bar.low), float(bar.close)]
        if bar.minutes == 390:
            day = frame.loc[(frame.index > pd.Timestamp(bar.ts).tz_convert("UTC")) & (frame.index <= close)]
            assert len(day) == 7 and day["is_complete"].all(), bar
            got = [day["open"].iloc[0], day["high"].max(), day["low"].min(), day["close"].iloc[-1]]
        elif bar.minutes == 30 and bar.close_ts.hour != 16:
            continue
        else:
            row = frame.loc[close]
            assert bool(row["is_complete"]), bar
            got = [row["open"], row["high"], row["low"], row["close"]]
            assert row["volume"] == pytest.approx(float(bar.volume)), bar
        assert got == want, f"{case.id} {bar.symbol} {bar.ts}: aggregated {got}, case {want}"


def test_collapsed_rejections_expand_to_one_order_final_per_intent(monkeypatch):
    """The executor folds a repeated pure rejection on one trading day into a
    single event with a ``repeat_count`` (execution.py:197-240); the adapter
    must still give every intent its own ``OrderFinal`` (invariant 7)."""
    from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager

    from .cases import DAY, H, HC, mkt

    pm = PortfolioManager(initial_capital=100.0, allowed_symbols=["AAPL"])
    submitted = []
    for k, oid in enumerate(("o1", "o2")):
        fill_bar = pd.Timestamp(H(DAY, k + 1))
        pm.execute_actions(
            [{"symbol": "AAPL", "action": "buy", "shares": 10, "reason": oid}],
            {"AAPL": {"close": 100.0}},
            fill_bar,
        )
        submitted.append((pd.Timestamp(HC(DAY, k)).tz_convert("UTC"), mkt(oid, H(DAY, k), "AAPL", "buy", "10")))
    assert len(pm.order_events) == 1 and pm.order_events[0]["repeat_count"] == 2

    orders = CurrentEngineAdapter(monkeypatch)._orders(pm, submitted, {"o1", "o2"})
    assert [(o.order_id, o.status, o.reason) for o in orders] == [
        ("o1", "rejected", "insufficient_cash"),
        ("o2", "rejected", "insufficient_cash"),
    ]
