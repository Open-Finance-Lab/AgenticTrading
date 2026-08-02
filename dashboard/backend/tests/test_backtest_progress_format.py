"""ETA and staleness formatting for a running backtest.

Both are honesty-constrained rather than precision-constrained:

* an ETA derived from two or three steps is wild, and a number that visibly
  jumps reads as broken -- so it is suppressed early and coarse thereafter;
* a stale progress file means the numbers are old, NOT that the run is stuck.
  An LLM pipeline step can legitimately take minutes. Claiming "stuck" would be
  the same class of error as the fabricated Performance Drivers card.

Two corrections to the first cut are guarded here. The ETA is measured over the
*observed* window rather than from launch, because both elapsed clocks start
before any step exists and folding startup into the per-step rate inflates the
estimate by an order of magnitude. And staleness is computed from a
server-supplied age rather than from an mtime diffed against the browser clock,
because a skewed client is otherwise indistinguishable from a wedged run.
"""

import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import fn_body, js_const

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

_HELPERS = (
    "function formatBacktestEta(",
    "function resolveBacktestEta(",
    "function resolveProgressAgeSeconds(",
    "function formatProgressStaleness(",
    "function formatStartupStaleness(",
    "function resolveRunningNotice(",
)


def _eval(expr: str) -> object:
    script = "\n".join(
        [
            js_const("BACKTEST_STALE_SECONDS"),
            *[fn_body(signature) for signature in _HELPERS],
            f"console.log(JSON.stringify({expr}));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_staleness_threshold_is_two_minutes():
    """Pins the shipped constant, since every staleness case below is scaled to
    it. Lowering it would make the UI cry wolf on ordinary model latency."""
    assert js_const("BACKTEST_STALE_SECONDS") == "const BACKTEST_STALE_SECONDS = 120;"


# --- ETA over the observed window --------------------------------------------


def test_eta_is_suppressed_for_the_first_two_observed_steps():
    assert _eval("formatBacktestEta(4, 1, 239)") is None
    assert _eval("formatBacktestEta(8, 2, 238)") is None


def test_eta_appears_from_the_third_observed_step():
    # 30s for 3 observed steps -> 10s/step -> 240 remaining -> "~40m left"
    assert _eval("formatBacktestEta(30, 3, 240)") == "~40m left"


def test_eta_is_coarse_under_a_minute():
    # 100s for 100 observed steps -> 1s/step -> 30 remaining -> 30s
    assert _eval("formatBacktestEta(100, 100, 30)") == "<1m left"


def test_eta_rounds_to_whole_minutes():
    # 125s for 25 observed steps -> 5s/step -> 25 remaining -> 125s -> ~2m
    assert _eval("formatBacktestEta(125, 25, 25)") == "~2m left"


def test_eta_is_null_without_remaining_work():
    assert _eval("formatBacktestEta(60, 10, 0)") is None
    assert _eval("formatBacktestEta(60, 10, null)") is None
    assert _eval("formatBacktestEta(60, null, 240)") is None


def test_eta_is_measured_from_the_first_observed_step():
    """19 steps observed over 19s -> 1s/step -> 220 remaining -> ~4m."""
    eta = _eval(
        "resolveBacktestEta({step: 20, totalSteps: 240,"
        " firstStep: 1, firstStepAt: Date.now() - 19000})"
    )
    assert eta == "~4m left"


def test_eta_is_withheld_rather_than_inflated_by_launch_cost():
    """The bias the anchor exists to remove.

    Launch costs ~25s (process start, imports, market-data fetch, gateway
    warm-up) and steps run ~1s. Measured from launch, step 3 divides 28s by 3
    and reports "~37m left" for a run that finishes in four minutes -- and the
    later collapse to "~4m left" is itself an "is this broken?" signal. Measured
    from the first observed step there are only two observed steps yet, so the
    honest answer is no estimate at all.
    """
    assert _eval("formatBacktestEta(28, 3, 237)") == "~37m left"  # the old form
    assert (
        _eval(
            "resolveBacktestEta({step: 3, totalSteps: 240,"
            " firstStep: 1, firstStepAt: Date.now() - 2000})"
        )
        is None
    )


def test_eta_needs_an_anchor():
    """No anchor means the poller has not yet seen a step in this run, so there
    is no window to measure over -- and the whole-run fallback is exactly the
    biased form this replaced."""
    assert _eval("resolveBacktestEta({step: 84, totalSteps: 240})") is None


def test_eta_is_null_on_the_final_step():
    """No remaining work to estimate; the completion path takes over."""
    eta = _eval(
        "resolveBacktestEta({step: 240, totalSteps: 240,"
        " firstStep: 1, firstStepAt: Date.now() - 240000})"
    )
    assert eta is None


# --- Progress age, computed on the server ------------------------------------


def test_age_adds_only_the_time_since_this_client_read_it():
    """The server's age plus the local elapsed since the poll. The second term
    is a difference of two Date.now() calls on one machine, so it is skew-free
    in a way `Date.now() - serverTimestamp` is not."""
    age = _eval("resolveProgressAgeSeconds({ageSeconds: 100, ageAt: Date.now() - 5000})")
    assert 104 <= age <= 107


def test_a_null_age_is_not_coerced_to_zero():
    """Number(null) is 0, so a coercing check would report a run with *no age
    reading at all* as perfectly fresh -- indistinguishable from one updated a
    moment ago. Returning null suppresses the notice instead of asserting
    freshness the payload never claimed."""
    assert _eval("resolveProgressAgeSeconds({ageSeconds: null, ageAt: Date.now()})") is None
    assert _eval("resolveProgressAgeSeconds({})") is None


# --- Which notice applies ----------------------------------------------------


def test_staleness_is_silent_below_the_threshold():
    assert _eval("formatProgressStaleness(0)") is None
    assert _eval("formatProgressStaleness(119)") is None


def test_staleness_reports_the_actual_gap_not_the_threshold():
    """A message frozen at '2m' while the real gap grows to ten is worse than
    no message -- it actively misinforms."""
    assert "2m" in _eval("formatProgressStaleness(130)")
    assert "9m" in _eval("formatProgressStaleness(560)")


def test_staleness_wording_does_not_claim_the_run_is_stuck():
    message = _eval("formatProgressStaleness(300)")
    assert "stuck" not in message.lower()
    assert "fail" not in message.lower()


def test_the_notice_depends_only_on_the_server_supplied_age():
    """A client clock offset can neither manufacture nor suppress it, which is
    the whole reason the age is computed server-side."""
    fresh = "{step: 84, totalSteps: 240, ageSeconds: 2, ageAt: Date.now()}"
    wedged = "{step: 84, totalSteps: 240, ageSeconds: 300, ageAt: Date.now()}"
    assert _eval(f"resolveRunningNotice({fresh})") is None
    assert "No progress for 5m" in _eval(f"resolveRunningNotice({wedged})")


def test_a_run_that_never_published_a_step_still_warns():
    """The likeliest wedge -- a subprocess that dies or hangs in imports, the
    market-data fetch or the LLM gateway -- writes no progress file at all, so
    there is no mtime to age and the staleness notice above can never fire. That
    left the exact scenario this feature exists for as the one case with no
    signal on either surface, for the full ten-minute poll ceiling.
    """
    notice = _eval("resolveRunningNotice({elapsedSeconds: 200})")
    assert "no steps" in notice.lower()
    assert "3m" in notice


def test_the_startup_notice_waits_for_the_same_threshold():
    """Every run is stepless for its opening moments; warning about that would
    make the notice meaningless."""
    assert _eval("resolveRunningNotice({elapsedSeconds: 30})") is None
    assert _eval("formatStartupStaleness(119)") is None


def test_the_startup_notice_does_not_claim_the_run_is_dead():
    message = _eval("formatStartupStaleness(300)")
    assert "stuck" not in message.lower()
    assert "fail" not in message.lower()
    assert "dead" not in message.lower()


def test_a_stepping_run_never_gets_the_startup_notice():
    """Once steps are flowing the startup wording would be flatly wrong, even
    if the server stopped supplying an age."""
    assert _eval("resolveRunningNotice({step: 84, totalSteps: 240, elapsedSeconds: 600})") is None
