"""US conformance suite against the current dashboard engine (adapter D) and
its LLM translator (the DL probe).

Each case is scored by ``scoring.score`` and held to its *measured* outcome:

* PASS runs plainly.
* FAIL is ``xfail(strict=True, raises=ConformanceFailure)``. The test raises
  ``ConformanceFailure`` only when the failing checks are exactly the ones the
  prediction names *and* every failing line matches the measured baseline in
  ``known_failures.json``; failing on a different set, or with different
  expected/actual values, is a plain failure, and so is passing (strict XPASS).
  The set alone cannot see a known defect getting worse (C04's cash going from
  -500 to -5000 fails the same checks), and this suite is the baseline the
  engine rewrite is measured against. **Changing an engine defect therefore
  turns its tests red** -- by design: the change must update the inventory in
  ``cases.py`` (``PREDICTIONS_D`` / ``PREDICTIONS_DL``, and
  ``PREDICTION_DELTAS`` if the design doc said otherwise) and re-record the
  baseline (``CONFORMANCE_UPDATE_KNOWN_FAILURES=1 pytest <this file>``, then
  review the JSON diff) in the same change. Numbers in the baseline are
  normalized to 10 significant digits, so float noise across library versions
  does not move it.
* N/E (not expressible) is skipped with the design doc's reason, but only
  after the adapter has actually refused the case; an adapter that runs a case
  predicted N/E fails.

The engine runs end to end on synthesized bars with no network, no ``node``
and no optional dependency.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest

from dashboard.backend.domain.backtesting.bar_aggregation import aggregate_bars_by_symbol

from .adapters.current_engine import CurrentEngineAdapter, synthesize_source_bars
from .cases import ALL_CASES, BY_ID, FAIL, NOT_EXPRESSIBLE, PASS, PREDICTIONS_D, PREDICTIONS_DL
from .model import NotExpressible
from .scoring import ConformanceFailure, report, score


KNOWN_FAILURES = Path(__file__).with_name("known_failures.json")
UPDATE_ENV = "CONFORMANCE_UPDATE_KNOWN_FAILURES"
_NUMBER = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+[eE][-+]?\d+")


def _normalized(failures):
    """The failure detail with every decimal number cut to 10 significant
    digits: still pins a value that moved, not float noise in its last bits."""
    def short(match):
        return format(float(match.group(0)), ".10g")

    return {check: [_NUMBER.sub(short, line) for line in lines] for check, lines in sorted(failures.items())}


def _load_known():
    return json.loads(KNOWN_FAILURES.read_text()) if KNOWN_FAILURES.exists() else {}


def _check_known(key, failures, text):
    detail = _normalized(failures)
    if os.environ.get(UPDATE_ENV) == "1":
        known = _load_known()
        known[key] = detail
        KNOWN_FAILURES.write_text(json.dumps(dict(sorted(known.items())), indent=2) + "\n")
        return
    pinned = _load_known().get(key)
    if pinned is None:
        pytest.fail(f"{key}: no baseline in {KNOWN_FAILURES.name}; re-record with {UPDATE_ENV}=1\n{text}")
    if pinned != detail:
        moved = [
            f"  [{check}] pinned {pinned.get(check)}\n  [{check}] now    {detail.get(check)}"
            for check in sorted(set(pinned) | set(detail))
            if pinned.get(check) != detail.get(check)
        ]
        pytest.fail(
            f"{key}: the known failure changed -- review, then re-record with {UPDATE_ENV}=1\n"
            + "\n".join(moved)
        )


def _param(case, prediction):
    marks = ()
    if prediction.verdict == FAIL:
        reason = f"{', '.join(prediction.defects)}: {prediction.note}"
        marks = (pytest.mark.xfail(strict=True, raises=ConformanceFailure, reason=reason),)
    return pytest.param(case, prediction, id=case.id, marks=marks)


def _assert_outcome(case, prediction, adapter, path):
    try:
        actual = adapter.run(case)
    except NotExpressible as exc:
        if prediction.verdict != NOT_EXPRESSIBLE:
            pytest.fail(f"{case.id}: predicted {prediction.verdict}, adapter refused it: {exc}")
        pytest.skip(f"N/E -- {prediction.note} (adapter: {exc})")
        return  # pytest.skip raises; the return keeps `actual` visibly bound below
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
    _check_known(f"{case.id}/{path}", failures, text)
    raise ConformanceFailure(text)


@pytest.mark.parametrize(
    "case, prediction", [_param(case, PREDICTIONS_D[case.id]) for case in ALL_CASES]
)
def test_dashboard_engine(case, prediction, monkeypatch):
    _assert_outcome(case, prediction, CurrentEngineAdapter(monkeypatch), "D")


@pytest.mark.parametrize(
    "case, prediction",
    [_param(BY_ID[case_id], prediction) for case_id, prediction in PREDICTIONS_DL.items()],
)
def test_dashboard_llm_translator(case, prediction, monkeypatch):
    adapter = CurrentEngineAdapter(monkeypatch, llm=True)
    _assert_outcome(case, prediction, adapter, "DL")


def test_known_failures_cover_exactly_the_fail_predictions():
    """A baseline entry per FAIL prediction, and none left for a fixed one."""
    want = {f"{cid}/D" for cid, p in PREDICTIONS_D.items() if p.verdict == FAIL}
    want |= {f"{cid}/DL" for cid, p in PREDICTIONS_DL.items() if p.verdict == FAIL}
    assert set(_load_known()) == want


def test_known_failure_normalization_keeps_magnitude():
    """Float noise collapses; a value that moved does not."""
    noisy = {"cash": ["cash: expected 500, actual -500.00000000000006"]}
    clean = {"cash": ["cash: expected 500, actual -500.0"]}
    worse = {"cash": ["cash: expected 500, actual -5000.0"]}
    assert _normalized(noisy) == _normalized(clean) != _normalized(worse)
    assert _normalized({"x": ["2026-03-02T10:30:00-05:00 qty 1E-15"]}) == {
        "x": ["2026-03-02T10:30:00-05:00 qty 1e-15"]
    }


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


def _rejected_buys(fill_bars):
    """Two unaffordable buys executed at ``fill_bars``; returns the manager and
    the ``executed`` records the adapter's ``execute`` hook would have kept."""
    from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager

    pm = PortfolioManager(initial_capital=100.0, allowed_symbols=["AAPL"])
    executed = []
    for oid, fill_bar in zip(("o1", "o2"), fill_bars):
        stamp = pd.Timestamp(fill_bar)
        pm.execute_actions(
            [{"symbol": "AAPL", "action": "buy", "shares": 10, "reason": oid}],
            {"AAPL": {"close": 100.0}},
            stamp,
        )
        executed.append((stamp.tz_convert("UTC"), "AAPL", "buy", oid))
    return pm, executed


def test_collapsed_rejections_expand_to_one_order_final_per_intent(monkeypatch):
    """The executor folds a repeated pure rejection on one trading day into a
    single event with a ``repeat_count`` (trading/execution.py:207-219); the
    adapter must still give every intent its own ``OrderFinal`` (invariant 7)."""
    from .cases import DAY, H

    pm, executed = _rejected_buys([H(DAY, 1), H(DAY, 2)])
    assert len(pm.order_events) == 1 and pm.order_events[0]["repeat_count"] == 2

    orders = CurrentEngineAdapter(monkeypatch)._orders(pm, executed, {"o1", "o2"})
    assert [(o.order_id, o.status, o.reason) for o in orders] == [
        ("o1", "rejected", "insufficient_cash"),
        ("o2", "rejected", "insufficient_cash"),
    ]


def test_collapsed_rejection_expands_onto_overnight_and_sideless_intents(monkeypatch):
    """The collapse is keyed on the *execution* day and the action's side, so
    the expansion matches on what was executed, never on the intent: a 16:00
    decision executed at the next session's open, and a target-weight intent
    whose side exists only once the translator sizes it, both find their copy.
    Here o2 is decided on DAY's close and executed on the next session, the
    same trading day as o1's execution."""
    from .cases import H

    next_day = "2026-03-03"
    pm, executed = _rejected_buys([H(next_day, 0), H(next_day, 1)])
    assert len(pm.order_events) == 1 and pm.order_events[0]["repeat_count"] == 2

    orders = CurrentEngineAdapter(monkeypatch)._orders(pm, executed, {"o1", "o2"})
    assert [o.order_id for o in orders] == ["o1", "o2"]
