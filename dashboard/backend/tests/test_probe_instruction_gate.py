"""The Phase 0 gate verdict, which decides whether Phase 2 gets built.

`_report` turns a set of measured returns into PASS / FAIL / INCONCLUSIVE. It had
no coverage, and shipped a defect that inverted its own meaning: the >=1pp test
was computed over *all* runs including the control, so a field of identical
seeded instructions plus one far-off control printed "a model separates
instructions by >=1pp" — the dead-axis result the control was added to detect.

Imported as ``dashboard.scripts.probe_instruction_sensitivity`` (the module skips
its direct-execution bootstrap when ``__package__`` is truthy); ``conftest.py``
has already pointed DATABASE_PATH at a temp file by then.
"""

import pytest

from dashboard.scripts import probe_instruction_sensitivity as probe


def _row(return_pct: float, *, valid: bool = True) -> dict:
    """Only the two keys `_report` reads; the rest of a real row is noise here."""
    return {"return_pct": return_pct, "valid": valid}


def _report(rows: dict, failures: list[str] | None = None) -> int:
    return probe._report({"m": rows}, 0.0, failures or [])


def test_identical_seeded_instructions_do_not_pass_on_a_lone_control():
    """The regression this test file exists for.

    Five seeded instructions returning the same number means the instruction axis
    is dead and a Phase 2 board would rank noise. A control landing far away does
    not rescue that — it only shows the model reacts to *having* an instruction.
    The all-runs spread was 3.50pp here, which cleared the old >=1pp check.
    """
    verdict = _report(
        {
            "aggressive_momentum": _row(3.80),
            "defensive_cash": _row(3.80),
            "equal_weight_hold": _row(3.80),
            "contrarian_reversion": _row(3.80),
            "verbose_analytical": _row(3.80),
            "control_nonsense": _row(0.30),
        }
    )
    assert verdict == probe.EXIT_FAIL


def test_separating_seeded_instructions_with_an_outlier_control_pass():
    verdict = _report(
        {
            "aggressive_momentum": _row(5.20),
            "defensive_cash": _row(1.10),
            "control_nonsense": _row(-2.00),
        }
    )
    assert verdict == probe.EXIT_PASS


def test_single_seeded_instruction_still_passes_but_flags_the_untested_axis(capsys):
    """The shape of the leg actually recorded in the write-up.

    One seeded instruction and two controls answers "does an instruction move the
    return" and nothing more, so it must pass the gate as posed *and* say plainly
    that ranking instructions against each other is untested — that is what Phase 2
    rests on.
    """
    verdict = _report(
        {
            "aggressive_momentum": _row(3.83),
            "control_nonsense": _row(0.33),
            "control_nonsense_b": _row(0.13),
        }
    )
    assert verdict == probe.EXIT_PASS
    assert "UNTESTED" in capsys.readouterr().out


def test_one_mid_pack_control_sinks_the_gate_even_when_another_separates():
    """`all` controls, not `any`.

    A nonsense instruction landing inside the seeded range IS the dead-axis
    signal; a second control landing clear does not cancel it out.
    """
    verdict = _report(
        {
            "aggressive_momentum": _row(5.00),
            "defensive_cash": _row(1.00),
            "control_nonsense": _row(3.00),  # mid-pack
            "control_nonsense_b": _row(-4.00),  # separated
        }
    )
    assert verdict == probe.EXIT_FAIL


def test_control_inside_the_seeded_range_fails():
    verdict = _report(
        {
            "aggressive_momentum": _row(5.00),
            "defensive_cash": _row(1.00),
            "control_nonsense": _row(3.00),
        }
    )
    assert verdict == probe.EXIT_FAIL


def test_a_control_barely_clearing_the_band_is_not_separation():
    """0.01pp outside the range is noise wearing the word OUTLIER."""
    verdict = _report(
        {
            "aggressive_momentum": _row(5.00),
            "defensive_cash": _row(1.00),
            "control_nonsense": _row(0.99),
        }
    )
    assert verdict == probe.EXIT_FAIL


def test_a_run_the_model_did_not_drive_is_inconclusive_not_a_failure():
    """Rule-based fallback produces identical curves, i.e. a 0.00pp spread.

    Reading that as FAIL would cancel the design over a billing problem.
    """
    verdict = _report(
        {
            "aggressive_momentum": _row(5.00),
            "defensive_cash": _row(1.00),
            "control_nonsense": _row(-4.00, valid=False),
        }
    )
    assert verdict == probe.EXIT_INCONCLUSIVE


def test_a_failed_run_makes_the_set_incomplete_and_the_verdict_inconclusive():
    verdict = _report(
        {
            "aggressive_momentum": _row(5.00),
            "defensive_cash": _row(1.00),
            "control_nonsense": _row(-4.00),
        },
        failures=["m/contrarian_reversion: TimeoutError: gateway"],
    )
    assert verdict == probe.EXIT_INCONCLUSIVE


def test_a_shard_with_no_control_returns_no_verdict():
    """One instruction per process is a shard, not an answer.

    Printing FAIL here would leave every shard log asserting "do NOT build Phase 2"
    on evidence that cannot support any verdict.
    """
    assert _report({"aggressive_momentum": _row(3.80)}) == probe.EXIT_INCONCLUSIVE


def test_controls_without_any_seeded_run_return_no_verdict():
    assert _report({"control_nonsense": _row(0.30)}) == probe.EXIT_INCONCLUSIVE


def test_zero_completed_runs_does_not_crash_the_report():
    """`--instructions ,` used to reach here and raise out of max()."""
    assert _report({}) == probe.EXIT_INCONCLUSIVE


@pytest.mark.parametrize(
    "capital,expected_problem",
    [(10_000.0, True), (100_000.0, False)],
)
def test_capital_resolution_guard_blocks_the_coarse_base(capital, expected_problem):
    """$10k refuses, $100k allows — the check that invalidated two whole legs."""
    pd = pytest.importorskip("pandas")
    index = pd.date_range("2026-04-15", periods=3, freq="h")
    bars = {
        sym: pd.DataFrame({"close": [price, price, price]}, index=index)
        for sym, price in {"AAA": 45.40, "BBB": 249.40, "CCC": 910.92}.items()
    }
    problem = probe._check_capital_resolution(bars, capital, "2026-04-15")
    assert (problem is not None) is expected_problem


def test_capital_guard_fails_visible_when_no_price_can_be_read():
    """"Guard passed" and "guard could not run" must not look identical.

    Returning None here meant a partial data outage produced a silent log and the
    next line spent real money against an unmeasured instrument.
    """
    pd = pytest.importorskip("pandas")
    bars = {"AAA": pd.DataFrame({"volume": [1, 2]})}
    problem = probe._check_capital_resolution(bars, 100_000.0, "2026-04-15")
    assert problem is not None
    assert "never checked" in problem


def test_capital_guard_samples_the_window_open_not_the_first_fetched_bar():
    """Bars start at `reference_start_date`, a month before the contest window.

    Judging the base off row 0 measures the resolution of a window never run.
    """
    pd = pytest.importorskip("pandas")
    index = pd.to_datetime(["2026-03-15 14:00", "2026-04-15 14:00"], utc=True)
    # Cheap in March, expensive once the window opens: sampling row 0 would read
    # 1.0 (fine at any capital) instead of 5000.0 (far too coarse).
    bars = {"AAA": pd.DataFrame({"close": [1.0, 5_000.0]}, index=index)}
    assert probe._check_capital_resolution(bars, 100_000.0, "2026-04-15") is not None
