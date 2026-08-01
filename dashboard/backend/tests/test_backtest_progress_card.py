"""The My Agents card shows step-level progress.

The backend has always emitted step/total_steps (engine.py `_publish_live_progress`,
surfaced by backtests.py `get_backtest_status`), and the Backtest tab has always
had a percentage bar. The card -- the page a user lands on after launching --
threw the data away and rendered an indeterminate bar plus an elapsed timer. A
tester watched it for 3m05s and could not tell running from stuck.

The 2026-07-29 spec called an indeterminate bar deliberate "since no honest
completion estimate exists". That premise was already false when written.
"""

import json
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import css_blocks, fn_body, js_const

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)

_REDUCED_MOTION = "@media (prefers-reduced-motion: reduce)"


def _render(running_js: str) -> str:
    script = "\n".join(
        [
            js_const("BACKTEST_STALE_SECONDS"),
            "function escapeHtml(s) { return String(s); }",
            "function renderAgentAllocatedCapitalHero() { return ''; }",
            "function formatBacktestElapsed(s) { return String(s); }",
            fn_body("function formatBacktestEta("),
            fn_body("function formatProgressStaleness("),
            fn_body("function renderAgentRunningBody("),
            f"console.log(JSON.stringify(renderAgentRunningBody("
            f"{{agent_id: 'a1'}}, {running_js})));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_card_shows_step_and_percent_when_known():
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert "84/240" in html
    assert "35%" in html


def test_card_bar_is_determinate_when_step_is_known():
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert "is-determinate" in html
    assert "width: 35%" in html


def test_card_falls_back_to_indeterminate_before_the_first_step():
    """Not an error state: the progress file does not exist for the opening
    moments of every run."""
    html = _render("{elapsedSeconds: 2}")
    assert "is-determinate" not in html
    assert "Backtesting" in html


def test_card_shows_eta():
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert "left" in html


def test_card_does_not_print_elapsed_twice():
    """The head already carries the timer; repeating it in the detail line one
    row below is the noise this change exists to remove.

    Scoped to the detail node's text: a whole-document search for "elapsed"
    also matches the head's own class name and data attribute, which must both
    stay -- so the obvious assertion would fail against correct output.
    """
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    detail = html.split('data-running-detail="a1">')[1].split("</p>")[0]
    assert "elapsed" not in detail, detail
    assert detail == "35% · ~6m left"
    # The head keeps its timer -- this removes a duplicate, not the value.
    assert "data-running-elapsed" in html


def test_card_detail_is_empty_before_the_first_step():
    """Empty rather than absent: the per-second patch targets this node by
    attribute, and only a change to the *set* of running agents re-renders."""
    html = _render("{elapsedSeconds: 2}")
    assert 'data-running-detail="a1"></p>' in html
    blocks = css_blocks(".agent-card-running-detail:empty")
    assert any("display: none" in block for block in blocks), blocks


def test_card_warns_when_progress_is_stale():
    html = _render(
        "{elapsedSeconds: 600, step: 84, totalSteps: 240, updatedAt: Date.now() - 300000}"
    )
    assert "No progress for 5m" in html


def test_card_is_silent_when_progress_is_fresh():
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert "No progress for" not in html


def test_progressbar_reports_its_value_to_assistive_tech():
    """role=progressbar without aria-valuenow announces as an unlabelled busy
    widget -- the same "is it moving?" question the sighted tester had."""
    html = _render("{elapsedSeconds: 185, step: 84, totalSteps: 240, updatedAt: Date.now()}")
    assert 'aria-valuenow="35"' in html
    assert 'aria-valuemax="100"' in html


def test_indeterminate_bar_omits_aria_valuenow():
    """A progressbar claiming valuenow=0 forever is a false statement; omitting
    it is what tells assistive tech the value is indeterminate."""
    html = _render("{elapsedSeconds: 2}")
    assert "aria-valuenow" not in html


def test_determinate_bar_keeps_a_reduced_motion_fallback():
    """Scoped to the reduced-motion block that names the determinate bar.

    `"is-determinate" in _STYLES` would be satisfied by the plain
    `.agent-card-running-bar.is-determinate` rule alone, so deleting the
    fallback entirely would leave this green -- the same vacuity closed in
    the Phase A guards.
    """
    blocks = css_blocks(_REDUCED_MOTION)
    assert any("agent-card-running-bar.is-determinate" in block for block in blocks), blocks


def test_determinate_bar_suppresses_the_indeterminate_sweep():
    """The width is data once determinate; leaving the keyframe sweep running
    on top of it animates the bar away from the value it is reporting."""
    blocks = css_blocks(".agent-card-running-bar.is-determinate")
    assert any("animation: none" in block for block in blocks), blocks


# --- Task 8: the Backtest tab panel, driven by the same two helpers -----------


def _run_panel(options_js: str) -> dict:
    """Execute updateBacktestRunProgress against three stub elements.

    Executed rather than grepped: `"formatBacktestEta(" in source` passes even
    if the returned value is dropped on the floor, which is precisely the bug
    that would let the two surfaces disagree.
    """
    script = "\n".join(
        [
            js_const("BACKTEST_POLL_MAX_SECONDS"),
            js_const("BACKTEST_STALE_SECONDS"),
            "const els = {",
            "  backtestRunElapsed: { textContent: '' },",
            "  backtestRunProgressMessage: { textContent: '' },",
            "  backtestRunProgressBar: { style: { width: '' } },",
            "};",
            "const document = { getElementById: (id) => els[id] || null };",
            fn_body("function formatBacktestElapsed("),
            fn_body("function formatBacktestEta("),
            fn_body("function formatProgressStaleness("),
            fn_body("function updateBacktestRunProgress("),
            f"updateBacktestRunProgress({options_js});",
            "console.log(JSON.stringify({",
            "  elapsed: els.backtestRunElapsed.textContent,",
            "  message: els.backtestRunProgressMessage.textContent,",
            "  width: els.backtestRunProgressBar.style.width,",
            "}));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_run_panel_shows_the_same_eta_the_card_does():
    """One run, two surfaces. Divergent numbers are worse than one blank
    surface, so both derive the ETA from formatBacktestEta."""
    panel = _run_panel(
        "{elapsedSeconds: 185, message: 'Backtest is running…',"
        " stepPct: 35, step: 84, totalSteps: 240}"
    )
    assert "left" in panel["message"]
    assert panel["message"].startswith("Backtest is running…")


def test_run_panel_reports_staleness():
    panel = _run_panel(
        "{elapsedSeconds: 600, message: 'Backtest is running…', stepPct: 35,"
        " step: 84, totalSteps: 240, updatedAt: Date.now() - 300000}"
    )
    assert "No progress for 5m" in panel["message"]


def test_run_panel_stays_quiet_without_the_new_fields():
    """The six unedited call sites pass none of step/totalSteps/updatedAt. They
    must render exactly what they rendered before -- the message alone."""
    panel = _run_panel("{elapsedSeconds: 42, message: 'Backtest is running…'}")
    assert panel["message"] == "Backtest is running…"


def test_run_panel_prefers_step_percent_over_the_elapsed_guess():
    """The elapsed-based width is a fallback for runs with no step data; a real
    percentage must win, otherwise the bar contradicts the number beside it."""
    panel = _run_panel(
        "{elapsedSeconds: 60, message: 'x', stepPct: 35, step: 84, totalSteps: 240}"
    )
    assert panel["width"] == "35%"


def test_run_panel_falls_back_to_the_elapsed_guess():
    panel = _run_panel("{elapsedSeconds: 60, message: 'x'}")
    assert panel["width"] == "10%"  # 60 / 600


def test_live_poll_passes_the_progress_fields_to_the_panel():
    """The helpers only matter if the live call site actually supplies step,
    totalSteps and updatedAt; every other call site correctly omits them."""
    poller = fn_body("function ensureBacktestPolling(")
    # Anchored to the running branch rather than "the first call in the
    # function": the error, completion and timeout sites are all in here too,
    # and they must NOT gain these fields.
    running_branch = poller[poller.index("const stepPct") :]
    call = running_branch[running_branch.index("updateBacktestRunProgress({") :]
    call = call[: call.index("});") + 3]
    assert "step," in call
    assert "totalSteps: total," in call
    assert "updatedAt: liveBacktestProgress?.updatedAt" in call
