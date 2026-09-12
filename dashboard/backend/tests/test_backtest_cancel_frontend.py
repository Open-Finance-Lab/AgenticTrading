"""Cancel, on both surfaces, and visibly not a failure.

Issue #273 lists two frontend surfaces in its acceptance criteria — the My
Agents card and the Backtest tab panel — because one updated is a half-shipped
change: a user who launches from a card lands on My Agents, and a user who is
watching the live chart is on the Backtest tab. Either can be the place someone
decides an hour is too long to wait.

/app has no build step and no JS test toolchain, so its contracts are guarded
against the shipped source (the convention set by test_ai_hedge_fund_frontend.py)
and, where the code is a pure string function, by running it under node.

Two of the guards below are the kind only a source-shape assertion can catch.
A test that hands the poller a `{cancelled: true}` payload passes whether or not
`status.cancelled` is checked before `status.error`, because a cancel payload
carries no `error` — and it passes whether or not the card's Cancel button is
ever given a run id, because the fixture supplies one. Both of those are the way
this ships broken.
"""

import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    css_blocks,
    fn_body,
    strip_comments,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

_ERROR_RED = "248, 113, 113"


def _render_actions(running_js: str) -> str:
    script = "\n".join(
        [
            "function escapeHtml(s) { return String(s); }",
            fn_body("function renderAgentRunningActions("),
            "console.log(JSON.stringify(renderAgentRunningActions("
            f"{{agent_id: 'a1'}}, {running_js})));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# ===========================================================================
# My Agents card
# ===========================================================================

def test_running_card_offers_cancel_once_the_run_has_an_id():
    html = _render_actions("{runId: 'agent_20260101_000000_abcd1234'}")

    assert "agent-cancel-backtest-btn" in html
    assert 'data-run-id="agent_20260101_000000_abcd1234"' in html
    # The Cancel button is shown and the pre-launch pill is not.
    cancel = html[html.index("agent-cancel-backtest-btn") : html.index("</button>", html.index("agent-cancel-backtest-btn"))]
    assert " hidden" not in cancel
    pending = html[html.index("data-running-pending") :]
    assert pending.startswith('data-running-pending="a1" hidden')


def test_running_card_hides_cancel_until_the_launch_has_a_run_id():
    """A launch is registered before its POST answers, so for the first tick
    there is no id to cancel. Offering the button anyway would produce a control
    that cannot work — worse than none, at the moment the user most needs to
    believe the feature does."""
    html = _render_actions("{runId: null}")

    assert 'data-run-id=""' in html
    cancel_start = html.index("agent-cancel-backtest-btn")
    assert " hidden>" in html[cancel_start : html.index("</button>", cancel_start)]
    # ...and the neutral "Starting…" pill stands in its place.
    assert "Starting…" in html


def test_card_cancel_button_is_on_the_per_second_patch_path():
    """The guard the fixture-driven test cannot be.

    refreshRunningAgentCards() patches the live DOM every second and does a full
    re-render only when the SET of running agents changes. Promoting a pending
    key to a real run id does not change that set, so a Cancel button whose
    `data-run-id` is written only by the template stays empty and hidden for the
    entire run of every backtest launched from this page — while a test that
    renders the template with a run id in hand passes.
    """
    body = strip_comments(fn_body("function refreshRunningAgentCards("))

    assert "data-running-cancel" in body
    assert "data-running-pending" in body
    assert "el.dataset.runId = entry.runId" in body


# ===========================================================================
# Backtest tab panel
# ===========================================================================

def test_panel_has_a_cancel_control():
    head = APP_HTML[
        APP_HTML.index('class="backtest-run-progress-head"') : APP_HTML.index(
            'id="backtestRunProgressBar"'
        )
    ]
    assert 'id="backtestRunCancel"' in head
    # Hidden by default: armed only once a run id exists.
    assert "hidden" in head[head.index('id="backtestRunCancel"') :]


def test_panel_paints_cancelled_as_its_own_state_not_as_an_error():
    body = strip_comments(fn_body("function showBacktestRunProgress("))

    assert "isCancelled" in body
    assert "'is-cancelled'" in body
    # Distinct classes and distinct titles -- a cancel must not borrow the error
    # panel's "Backtest did not start".
    assert "Backtest cancelled" in body
    assert "Backtest did not start" in body


def test_cancelled_panel_is_not_styled_as_a_failure():
    """The user stopped their own run. Painting it red says they broke it."""
    blocks = css_blocks(".backtest-run-progress.is-cancelled")

    assert blocks, "no .is-cancelled rule ships in styles.css"
    assert not any(_ERROR_RED in block for block in blocks)
    # ...and the error treatment is still red, so this is a real distinction
    # rather than both states having lost their colour.
    assert any(_ERROR_RED in block for block in css_blocks(".backtest-run-progress.is-error"))


def test_poller_checks_cancelled_before_error():
    """The second guard a payload-driven test cannot make.

    A cancelled status carries no `error` key, so feeding one to the poller
    takes the cancelled branch whichever order the two are written in. Order is
    still the contract: reversing them the day a cancel starts carrying an error
    string would silently repaint every cancel as a crash.
    """
    body = strip_comments(fn_body("function ensureBacktestPolling("))

    cancelled_at = body.index("if (status.cancelled)")
    error_at = body.index("} else if (status.error) {")
    assert cancelled_at < error_at


def test_cancelled_branch_paints_no_coverage_badge_and_no_fallback_note():
    """The composition guard against the change this one stacks on.

    #458 gave the completion path a coverage badge (read from the run row's
    `decision_badge`) and a fallback note, and made the panel stay up when the
    model did not drive the run. None of that applies to a run stopped
    mid-flight: there is no verdict to report about steps that never ran. The
    cancelled branch therefore repaints the config with a NULL run -- which is
    what structurally forbids the badge -- and never reaches either helper.
    """
    body = strip_comments(fn_body("function ensureBacktestPolling("))
    branch = body[
        body.index("if (status.cancelled)") : body.index("} else if (status.error) {")
    ]

    assert "renderBacktestRunConfig(null," in branch
    assert "statusLabel: 'Cancelled'" in branch
    assert "formatDecisionProvenance" not in branch
    assert "backtestFellBackFromTheModel" not in branch
    assert "decision_badge" not in branch
    # ...and the config repaint precedes the panel show, because
    # renderBacktestRunConfig() hides the panel outright when it has neither a
    # run nor a launch config.
    assert branch.index("renderBacktestRunConfig(null,") < branch.index(
        "showBacktestRunProgress(true, { isCancelled: true })"
    )


def test_cancel_request_does_not_claim_a_cancel_that_did_not_happen():
    """`cancelled: false` is the server's honest answer to a cancel that raced a
    completion. Reporting it as a success is the fabrication issue #273 warns
    about, one layer up."""
    body = strip_comments(fn_body("async function cancelBacktest("))

    assert "/backtest/cancel" in body
    assert "data.cancelled === false" in body
    assert "already finished" in body
    # A 404 means the run ended between the paint and the click, not an error
    # worth an alert box.
    assert "error.status === 404" in body


def test_terminal_paths_take_the_cancel_button_away():
    """A Cancel offered for a run that has already stopped answers 404."""
    poller = strip_comments(fn_body("function ensureBacktestPolling("))
    launch_failure = strip_comments(fn_body("function showBacktestLaunchFailure("))

    assert "setBacktestCancelTarget(null)" in poller
    assert "setBacktestCancelTarget(null)" in launch_failure
