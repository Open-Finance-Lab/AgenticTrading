"""The My Agents card shows step-level progress.

The backend has always emitted step/total_steps (engine.py `_publish_live_progress`,
surfaced by backtests.py `get_backtest_status`), and the Backtest tab has always
had a percentage bar. The card -- the page a user lands on after launching --
threw the data away and rendered an indeterminate bar plus an elapsed timer. A
tester watched it for 3m05s and could not tell running from stuck.

The 2026-07-29 spec called an indeterminate bar deliberate "since no honest
completion estimate exists". That premise was already false when written.

**Two renderers, one card.** renderAgentRunningBody() paints it on a full
re-render; refreshRunningAgentCards() patches it in place every second, because
renderAgentCards() opens with `grid.innerHTML = ''` and doing that once a second
would destroy focus, scroll and any open menu for the whole run. A full
re-render fires only when the *set* of running agents changes -- twice in a
normal run -- so anything the patch path does not touch is frozen at its launch
value. The first cut patched three text nodes and left the bar, its
aria-valuenow and the staleness note behind, which put correct numbers beside a
bar still running the indeterminate sweep and made the staleness warning
unreachable. Both renderers now derive from deriveRunningProgress(), and the
patch cases below drive the real patch path over the real DOM the real template
produced.
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

#: Lifted wholesale rather than re-listed per harness: a helper missing from one
#: list is a ReferenceError, but a helper *stubbed* in one list quietly tests the
#: stub. One list means every harness runs the same shipped code.
_PROGRESS_HELPERS = (
    "function formatBacktestEta(",
    "function resolveBacktestEta(",
    "function resolveProgressAgeSeconds(",
    "function formatProgressStaleness(",
    "function formatStartupStaleness(",
    "function resolveRunningNotice(",
    "function deriveRunningProgress(",
)

#: 84 of 240 steps (35%), 80 of them observed over 184s -> 2.3s/step -> ~6m left.
_LIVE_PROGRESS = (
    "{step: 84, totalSteps: 240, ageSeconds: 1, ageAt: Date.now(),"
    " firstStep: 4, firstStepAt: Date.now() - 184000}"
)


def _node(script: str) -> object:
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _render(running_js: str) -> str:
    script = "\n".join(
        [
            js_const("BACKTEST_STALE_SECONDS"),
            "function escapeHtml(s) { return String(s); }",
            "function renderAgentAllocatedCapitalHero() { return ''; }",
            "function formatBacktestElapsed(s) { return String(s); }",
            *[fn_body(signature) for signature in _PROGRESS_HELPERS],
            fn_body("function renderAgentRunningBody("),
            f"console.log(JSON.stringify(renderAgentRunningBody("
            f"{{agent_id: 'a1'}}, {running_js})));",
        ]
    )
    return _node(script)


def test_card_shows_step_and_percent_when_known():
    html = _render(f"{{elapsedSeconds: 185, ...{_LIVE_PROGRESS}}}")
    assert "84/240" in html
    assert "35%" in html


def test_card_bar_is_determinate_when_step_is_known():
    html = _render(f"{{elapsedSeconds: 185, ...{_LIVE_PROGRESS}}}")
    assert "is-determinate" in html
    assert "width: 35%" in html


def test_card_falls_back_to_indeterminate_before_the_first_step():
    """Not an error state: the progress file does not exist for the opening
    moments of every run."""
    html = _render("{elapsedSeconds: 2}")
    assert "is-determinate" not in html
    assert "Backtesting" in html


def test_card_shows_eta():
    """Pinned to the exact string, not just "left": a harness that accepts any
    output containing the word passes on "0m left" and on an ETA computed from
    the wrong window, which is precisely the bug the anchor exists to prevent.
    """
    html = _render(f"{{elapsedSeconds: 185, ...{_LIVE_PROGRESS}}}")
    assert "~6m left" in html


def test_card_does_not_print_elapsed_twice():
    """The head already carries the timer; repeating it in the detail line one
    row below is the noise this change exists to remove.

    Scoped to the detail node's text: a whole-document search for "elapsed"
    also matches the head's own class name and data attribute, which must both
    stay -- so the obvious assertion would fail against correct output.
    """
    html = _render(f"{{elapsedSeconds: 185, ...{_LIVE_PROGRESS}}}")
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
        "{elapsedSeconds: 600, step: 84, totalSteps: 240,"
        " ageSeconds: 300, ageAt: Date.now()}"
    )
    assert "No progress for 5m" in html


def test_card_is_silent_when_progress_is_fresh():
    html = _render(f"{{elapsedSeconds: 185, ...{_LIVE_PROGRESS}}}")
    assert "No progress for" not in html


def test_card_warns_when_a_run_never_publishes_a_step():
    """The likeliest wedge -- a subprocess that hangs in imports, the
    market-data fetch or the LLM gateway -- writes no progress file at all, so
    the staleness notice above can never fire. Without this the motivating
    scenario is the one case with no signal, for the full poll ceiling."""
    html = _render("{elapsedSeconds: 200}")
    assert "no steps reported after 3m" in html
    assert "is-determinate" not in html


def test_card_stale_node_is_empty_and_hidden_while_healthy():
    """Rendered unconditionally for the same reason the detail node is: the
    patch path fills it by attribute, and a node that appears only once it has
    something to say can never be filled in mid-run."""
    html = _render(f"{{elapsedSeconds: 185, ...{_LIVE_PROGRESS}}}")
    assert 'data-running-stale="a1"></p>' in html
    blocks = css_blocks(".agent-card-running-stale:empty")
    assert any("display: none" in block for block in blocks), blocks


def test_progressbar_reports_its_value_to_assistive_tech():
    """role=progressbar without aria-valuenow announces as an unlabelled busy
    widget -- the same "is it moving?" question the sighted tester had."""
    html = _render(f"{{elapsedSeconds: 185, ...{_LIVE_PROGRESS}}}")
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


# --- The per-second patch path, over the DOM the template really produces -----

#: A minimal DOM: parse the template's own output into nodes the patch path can
#: query and mutate. Hand-built stub nodes would be the vacuity trap here -- they
#: would let the patcher pass against attributes the template never emits, which
#: is the exact class of drift these cases exist to catch.
_DOM_SHIM = r"""
function parseNodes(html) {
    const nodes = [];
    const tagRe = /<(\w+)([^>]*)>([^<]*)/g;
    let tag;
    while ((tag = tagRe.exec(html)) !== null) {
        const attrs = {};
        const attrRe = /([\w-]+)(?:="([^"]*)")?/g;
        let attr;
        while ((attr = attrRe.exec(tag[2])) !== null) {
            attrs[attr[1]] = attr[2] === undefined ? '' : attr[2];
        }
        nodes.push(makeNode(attrs, tag[3]));
    }
    return nodes;
}

function makeNode(attrs, text) {
    const classes = new Set((attrs.class || '').split(/\s+/).filter(Boolean));
    const width = /width:\s*([^;"]+)/.exec(attrs.style || '');
    const owns = (name) => Object.prototype.hasOwnProperty.call(attrs, name);
    return {
        has: owns,
        textContent: text,
        style: { width: width ? width[1].trim() : '' },
        classList: {
            contains: (name) => classes.has(name),
            toggle: (name, force) => { if (force) { classes.add(name); } else { classes.delete(name); } },
        },
        getAttribute: (name) => (owns(name) ? attrs[name] : null),
        setAttribute: (name, value) => { attrs[name] = String(value); },
        removeAttribute: (name) => { delete attrs[name]; },
    };
}

function snapshotOf(nodes) {
    const by = (attr) => nodes.find((node) => node.has(attr));
    const track = by('data-running-track');
    const bar = by('data-running-bar');
    return {
        step: by('data-running-step').textContent,
        detail: by('data-running-detail').textContent,
        stale: by('data-running-stale').textContent,
        elapsed: by('data-running-elapsed').textContent,
        ariaValueNow: track.getAttribute('aria-valuenow'),
        ariaValueMax: track.getAttribute('aria-valuemax'),
        determinate: bar.classList.contains('is-determinate'),
        barWidth: bar.style.width,
    };
}
"""


def _patch(progress_js: str, then_progress_js: str | None = None, elapsed_ms: int = 185000) -> dict:
    """Launch-paint a card, then run the real per-second patch over it.

    The card is rendered in the state every run genuinely starts in -- marked
    running, no progress published yet, so indeterminate -- and then patched
    with what the poller has a moment later. That sequence is what the live UI
    does and what no earlier case covered: every card test called
    renderAgentRunningBody() directly with progress already populated, a state
    the browser reaches only on a re-render.

    Returns the patched snapshot alongside a fresh full render of the same
    state, so the two renderers can be compared field by field.
    """
    second_tick = (
        f"liveBacktestProgress = {then_progress_js};\n"
        f"if ({then_progress_js} === null) delete liveBacktestProgressByRunId['run-1'];\n"
        f"else liveBacktestProgressByRunId['run-1'] = {then_progress_js};\n"
        "refreshRunningAgentCards();"
        if then_progress_js is not None
        else ""
    )
    script = "\n".join(
        [
            js_const("BACKTEST_POLL_MAX_SECONDS"),
            js_const("BACKTEST_STALE_SECONDS"),
            f"const MAP = {{a1: {{runId: 'run-1', startedAt: Date.now() - {elapsed_ms}}}}};",
            "let liveBacktestRunId = 'run-1';",
            "let liveBacktestProgress = null;",
            "const liveBacktestProgressByRunId = Object.create(null);",
            "let lastRenderedRunningKey = null;",
            "let applyCalls = 0;",
            "function readRunningBacktests() { return MAP; }",
            "function clearAgentBacktestRunning(id) { delete MAP[id]; }",
            "function applyAgentFilters() { applyCalls += 1; }",
            "function escapeHtml(s) { return String(s); }",
            "function renderAgentAllocatedCapitalHero() { return ''; }",
            "function formatBacktestElapsed(s) { return String(s); }",
            *[fn_body(signature) for signature in _PROGRESS_HELPERS],
            fn_body("function getAgentBacktestRunning("),
            fn_body("function renderAgentRunningBody("),
            fn_body("function refreshRunningAgentCards("),
            _DOM_SHIM,
            "const launchHtml = renderAgentRunningBody({agent_id: 'a1'}, getAgentBacktestRunning('a1'));",
            "const NODES = parseNodes(launchHtml);",
            "const document = { querySelectorAll: (sel) => "
            + "NODES.filter((node) => node.has(sel.slice(1, -1))) };",
            # The set of running agents is unchanged, which is what forces the
            # patch path instead of a full re-render -- the steady state for all
            # but two ticks of a run.
            "lastRenderedRunningKey = Object.keys(MAP).sort().join(',');",
            f"liveBacktestProgress = {progress_js};",
            f"if ({progress_js} === null) delete liveBacktestProgressByRunId['run-1'];",
            f"else liveBacktestProgressByRunId['run-1'] = {progress_js};",
            "refreshRunningAgentCards();",
            second_tick,
            "console.log(JSON.stringify({",
            "  launch: snapshotOf(parseNodes(launchHtml)),",
            "  patched: snapshotOf(NODES),",
            "  rendered: snapshotOf(parseNodes("
            + "renderAgentRunningBody({agent_id: 'a1'}, getAgentBacktestRunning('a1')))),",
            "  applyCalls,",
            "}));",
        ]
    )
    return _node(script)


def test_the_patch_path_is_what_ran():
    """Guard the guard. If a stray re-render satisfied these cases the patcher
    could be deleted outright and every assertion below would still pass."""
    result = _patch(_LIVE_PROGRESS)
    assert result["applyCalls"] == 0


def test_patch_moves_the_bar_off_the_indeterminate_sweep():
    """The headline defect: the numbers climbed while the bar kept sweeping."""
    result = _patch(_LIVE_PROGRESS)
    assert result["launch"]["determinate"] is False
    assert result["patched"]["determinate"] is True
    assert result["patched"]["barWidth"] == "35%"


def test_patch_updates_the_aria_value():
    """Without this the progressbar announces as an unlabelled busy widget for
    the entire run -- the launch paint is the only one screen readers get."""
    result = _patch(_LIVE_PROGRESS)
    assert result["launch"]["ariaValueNow"] is None
    assert result["patched"]["ariaValueNow"] == "35"
    assert result["patched"]["ariaValueMax"] == "100"


def test_patch_surfaces_the_staleness_notice():
    """The affordance this whole feature exists for. Reachable only on a full
    re-render before, i.e. essentially never during a run.

    The launch paint carries the *startup* notice here -- 185s in with no step
    published yet -- so this also covers the patch replacing existing text
    rather than merely filling a blank node.
    """
    result = _patch(
        "{step: 84, totalSteps: 240, ageSeconds: 300, ageAt: Date.now(),"
        " firstStep: 4, firstStepAt: Date.now() - 184000}"
    )
    assert "no steps reported" in result["launch"]["stale"]
    assert "No progress for 5m" in result["patched"]["stale"]


def test_patch_clears_a_notice_that_no_longer_applies():
    """A card launched slowly enough to earn the startup notice must drop it the
    moment steps start flowing -- a patch that only ever *sets* text would leave
    "still starting up" printed under a moving percentage."""
    result = _patch(_LIVE_PROGRESS)
    assert "no steps reported" in result["launch"]["stale"]
    assert result["patched"]["stale"] == ""


def test_patch_surfaces_the_startup_notice():
    """A run that has published nothing at all: no step ever reaches the store,
    so only the elapsed-based notice can fire."""
    result = _patch("null", elapsed_ms=200000)
    assert "no steps reported after 3m" in result["patched"]["stale"]
    assert result["patched"]["determinate"] is False


def test_patch_updates_the_text_nodes():
    result = _patch(_LIVE_PROGRESS)
    assert result["patched"]["step"] == "84/240"
    assert result["patched"]["detail"] == "35% · ~6m left"


def test_patch_clears_the_numbers_when_progress_vanishes():
    """A tick where the status endpoint reports no progress -- file caught
    mid-rewrite, transient OSError -- must clear the last numbers rather than
    leave "84/240 · 35% · ~6m left" standing as though it were current."""
    result = _patch(_LIVE_PROGRESS, then_progress_js="null")
    assert result["patched"]["step"] == ""
    assert result["patched"]["detail"] == ""


def test_patch_returns_the_bar_to_indeterminate_when_progress_vanishes():
    """Symmetrical with the text: a bar frozen at 35% with a valuenow to match
    is a claim the payload no longer supports."""
    result = _patch(_LIVE_PROGRESS, then_progress_js="null")
    assert result["patched"]["determinate"] is False
    assert result["patched"]["ariaValueNow"] is None
    # Cleared, not zeroed: the stylesheet's 40% is what makes the indeterminate
    # sweep visible, and a 0%-wide bar animates nothing.
    assert result["patched"]["barWidth"] == ""


def test_the_two_renderers_agree():
    """The invariant behind the defect, stated directly.

    renderAgentRunningBody() builds an HTML string and refreshRunningAgentCards()
    mutates live DOM, so they cannot share the emitting code -- only the
    derivation. This compares the patched card against a full re-render of the
    same state, field by field, so the next field added to one and not the other
    fails here rather than in a browser.
    """
    result = _patch(_LIVE_PROGRESS)
    assert result["patched"] == result["rendered"]


def test_the_two_renderers_agree_when_progress_vanishes():
    result = _patch(_LIVE_PROGRESS, then_progress_js="null")
    assert result["patched"] == result["rendered"]


# --- The live progress store --------------------------------------------------


def _advance(previous_js: str, progress_js: str, now: int) -> dict:
    script = "\n".join(
        [
            fn_body("function advanceBacktestProgress("),
            f"console.log(JSON.stringify("
            f"advanceBacktestProgress({previous_js}, {progress_js}, {now})));",
        ]
    )
    return _node(script)


def test_the_store_anchors_on_the_first_step_it_sees():
    stored = _advance("null", "{step: 5, total_steps: 240, progress_age_seconds: 1}", 1000)
    assert stored["firstStep"] == 5
    assert stored["firstStepAt"] == 1000


def test_the_anchor_is_carried_across_ticks():
    """The point of the anchor. Re-stamping it every tick would silently restore
    the launch-biased ETA -- and would still produce a plausible number, so
    nothing else would catch it."""
    first = "{step: 5, firstStep: 5, firstStepAt: 1000}"
    stored = _advance(first, "{step: 9, total_steps: 240}", 5000)
    assert stored["firstStep"] == 5
    assert stored["firstStepAt"] == 1000


def test_the_anchor_resets_when_the_step_count_moves_backwards():
    """Only happens when a fresh run's first tick lands before the previous run
    was cleared. Keeping the old anchor would measure the new run's rate over
    the old run's clock."""
    stale = "{step: 200, firstStep: 5, firstStepAt: 1000}"
    stored = _advance(stale, "{step: 2, total_steps: 240}", 9000)
    assert stored["firstStep"] == 2
    assert stored["firstStepAt"] == 9000


def test_the_store_is_null_before_the_first_step():
    assert _advance("null", "{total_steps: 240}", 1000) is None
    assert _advance("null", "{step: 0, total_steps: 240}", 1000) is None
    assert _advance("null", "null", 1000) is None


def test_the_store_keeps_the_server_age():
    stored = _advance("null", "{step: 5, total_steps: 240, progress_age_seconds: 42.5}", 1000)
    assert stored["ageSeconds"] == 42.5
    assert stored["ageAt"] == 1000


def test_a_missing_server_age_is_stored_as_null_not_zero():
    """Zero would assert freshness the payload never claimed, permanently
    suppressing the staleness notice against an older backend."""
    stored = _advance("null", "{step: 5, total_steps: 240}", 1000)
    assert stored["ageSeconds"] is None


# --- Task 8: the Backtest tab panel, driven by the same two helpers -----------


def _run_panel(options_js: str) -> dict:
    """Execute updateBacktestRunProgress against three stub elements.

    Executed rather than grepped: `"resolveBacktestEta(" in source` passes even
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
            "function formatBacktestElapsed(s) { return String(s); }",
            *[fn_body(signature) for signature in _PROGRESS_HELPERS],
            fn_body("function updateBacktestRunProgress("),
            f"updateBacktestRunProgress({options_js});",
            "console.log(JSON.stringify({",
            "  elapsed: els.backtestRunElapsed.textContent,",
            "  message: els.backtestRunProgressMessage.textContent,",
            "  width: els.backtestRunProgressBar.style.width,",
            "}));",
        ]
    )
    return _node(script)


def test_run_panel_shows_the_same_eta_the_card_does():
    """One run, two surfaces. Divergent numbers are worse than one blank
    surface, so both derive the ETA from the same helper and the same object."""
    panel = _run_panel(
        "{elapsedSeconds: 185, message: 'Backtest is running…',"
        f" stepPct: 35, progress: {_LIVE_PROGRESS}}}"
    )
    assert panel["message"] == "Backtest is running… · ~6m left"


def test_run_panel_reports_staleness():
    panel = _run_panel(
        "{elapsedSeconds: 600, message: 'Backtest is running…', stepPct: 35,"
        " progress: {step: 84, totalSteps: 240, ageSeconds: 300, ageAt: Date.now()}}"
    )
    assert "No progress for 5m" in panel["message"]


def test_run_panel_reports_a_run_that_never_started_stepping():
    """`{}` -- the live branch's stand-in before the first step -- is what opts
    this surface into the startup notice. The subprocess has published nothing,
    so this is the only warning either surface can give."""
    panel = _run_panel(
        "{elapsedSeconds: 200, message: 'Backtest is running…', progress: {}}"
    )
    assert "no steps reported after 3m" in panel["message"]


def test_run_panel_stays_quiet_without_the_new_fields():
    """The terminal call sites pass no progress object at all. They must render
    exactly what they rendered before -- the message alone -- even though their
    elapsed is well past the staleness threshold."""
    panel = _run_panel("{elapsedSeconds: 42, message: 'Backtest is running…'}")
    assert panel["message"] == "Backtest is running…"
    done = _run_panel("{elapsedSeconds: 600, message: 'Completed in 10:00.'}")
    assert done["message"] == "Completed in 10:00."


def test_run_panel_prefers_step_percent_over_the_elapsed_guess():
    """The elapsed-based width is a fallback for runs with no step data; a real
    percentage must win, otherwise the bar contradicts the number beside it."""
    panel = _run_panel(
        f"{{elapsedSeconds: 60, message: 'x', stepPct: 35, progress: {_LIVE_PROGRESS}}}"
    )
    assert panel["width"] == "35%"


def test_run_panel_falls_back_to_the_elapsed_guess():
    panel = _run_panel("{elapsedSeconds: 60, message: 'x'}")
    assert panel["width"] == "2%"  # 60 / 3600


def test_frontend_backtest_observation_window_is_sixty_minutes():
    assert (
        _node(
            js_const("BACKTEST_POLL_MAX_SECONDS")
            + "console.log(JSON.stringify(BACKTEST_POLL_MAX_SECONDS));"
        )
        == 3600
    )


def _resolve_entry(
    map_js: str,
    live_run_id: str,
    progress_js: str,
    agent: str,
    now_ms: int | None = None,
) -> dict:
    """Run the real getAgentBacktestRunning against a stubbed running map."""
    clock = f"Date.now = () => {now_ms};" if now_ms is not None else ""
    script = "\n".join(
        [
            js_const("BACKTEST_POLL_MAX_SECONDS"),
            f"const MAP = {map_js};",
            clock,
            "function readRunningBacktests() { return MAP; }",
            "function clearAgentBacktestRunning(id) { delete MAP[id]; }",
            f"let liveBacktestRunId = {live_run_id};",
            f"let liveBacktestProgress = {progress_js};",
            "const liveBacktestProgressByRunId = Object.create(null);",
            fn_body("function getAgentBacktestRunning("),
            f"console.log(JSON.stringify(getAgentBacktestRunning('{agent}')));",
        ]
    )
    return _node(script)


def _list_entries(map_js: str, now_ms: int) -> dict:
    """Run the real listRunningBacktests helper with a deterministic clock."""
    script = "\n".join(
        [
            js_const("BACKTEST_POLL_MAX_SECONDS"),
            f"const MAP = {map_js};",
            f"Date.now = () => {now_ms};",
            "function readRunningBacktests() { return MAP; }",
            "function writeRunningBacktests(value) { WRITTEN = value; }",
            "let WRITTEN = null;",
            fn_body("function listRunningBacktests("),
            "console.log(JSON.stringify({runs: listRunningBacktests(), map: MAP, written: WRITTEN}));",
        ]
    )
    return _node(script)


_TWO_AGENTS = (
    "{'agent-A': {runId: 'run-1', startedAt: Date.now() - 185000},"
    " 'agent-B': {runId: null, startedAt: Date.now() - 500}}"
)
_PROGRESS = "{step: 45, totalSteps: 50, ageSeconds: 1, ageAt: Date.now()}"


def test_progress_reaches_the_agent_whose_run_is_live():
    entry = _resolve_entry(_TWO_AGENTS, "'run-1'", _PROGRESS, "agent-A")
    assert entry["step"] == 45
    assert entry["totalSteps"] == 50


def test_progress_does_not_bleed_onto_an_unconfirmed_launch():
    """runBacktest() marks an agent running BEFORE its POST resolves, and the
    backend refuses a second concurrent run. So clicking Run on an idle agent
    while another is genuinely in flight leaves both in the map for one
    round-trip. An unconditional spread painted the live agent's 45/50 (90%,
    "<1m left") onto a card whose launch was about to be rejected.
    """
    entry = _resolve_entry(_TWO_AGENTS, "'run-1'", _PROGRESS, "agent-B")
    assert entry.get("step") is None
    assert entry.get("totalSteps") is None
    assert entry["runId"] is None
    # It is still "running" -- just indeterminate, which is honest here.
    assert entry["elapsedSeconds"] >= 0


def test_progress_is_withheld_when_no_run_is_identified():
    """Without a live run id nothing can be attributed, so attribute nothing
    rather than guessing -- an indeterminate bar beats a wrong percentage."""
    entry = _resolve_entry(_TWO_AGENTS, "null", _PROGRESS, "agent-A")
    assert entry.get("step") is None


def test_running_entry_is_retained_until_the_sixty_minute_ceiling():
    entry = _resolve_entry(
        "{'agent-A': {runId: 'run-1', startedAt: 0}}",
        "'run-1'",
        "null",
        "agent-A",
        now_ms=3_599_000,
    )
    assert entry["runId"] == "run-1"
    assert entry["elapsedSeconds"] == 3599


def test_running_entry_is_cleared_after_the_sixty_minute_ceiling():
    entry = _resolve_entry(
        "{'agent-A': {runId: 'run-1', startedAt: 0}}",
        "'run-1'",
        "null",
        "agent-A",
        now_ms=3_600_001,
    )
    assert entry is None


def test_running_list_sweeps_only_entries_past_the_ceiling():
    result = _list_entries(
        "{'keep': {runId: 'run-keep', startedAt: 1},"
        " 'drop': {runId: 'run-drop', startedAt: -1}}",
        now_ms=3_600_000,
    )
    assert [run["runId"] for run in result["runs"]] == ["run-keep"]
    assert "drop" not in result["map"]
    assert result["written"] is not None


def test_progress_store_is_written_before_the_card_repaints():
    """Same poll response, two surfaces. refreshRunningAgentCards() reads
    per-run progress via getAgentBacktestRunning; the Backtest panel is
    handed the focused run's object. Repainting before the assignment made the
    card show the previous tick's numbers while the panel showed this tick's --
    deterministic every tick, not a race.
    """
    poller = fn_body("function ensureBacktestPolling(")
    running_branch = poller[poller.index("if (status.running) {") :]
    # Comments stripped first: the explanatory comment above the assignment
    # names refreshRunningAgentCards(), so a raw text search finds the *comment*
    # earlier than the assignment and reports correct code as broken.
    code = "\n".join(
        line for line in running_branch.splitlines() if not line.lstrip().startswith("//")
    )
    assert code.index("liveBacktestProgressByRunId[") < code.index(
        "refreshRunningAgentCards()"
    ), "per-run progress must be assigned before the card repaints"


def test_the_poller_reads_the_server_computed_age_not_the_mtime():
    """Deriving the age in the browser (`Date.now() - progress_updated_at`)
    makes any client more than the staleness threshold out of step
    indistinguishable from a wedged run: a fast clock pins a permanent
    "No progress for 47m" onto a healthy backtest, a slow one suppresses the
    warning forever. Suspended laptops drift by minutes routinely.
    """
    store = fn_body("function advanceBacktestProgress(")
    assert "progress_age_seconds" in store
    assert "progress_updated_at" not in store
    poller = fn_body("function ensureBacktestPolling(")
    assert "progress_updated_at" not in poller


def test_timeout_branch_clears_the_running_map_and_the_progress_store():
    """The leak that makes one run render another run's numbers.

    Progress is stored per live_run_id. The finished path clears that entry;
    the 60-minute timeout branch must wipe the whole map so an orphaned card
    cannot pick up the NEXT run's step/percent.

    Guarded by source slice rather than execution: reaching the branch takes 3600
    poll ticks. Scoped to the branch itself, because both statements also appear
    in the finished branch a few lines above -- a whole-function search would
    pass with the timeout branch completely untouched.
    """
    branch = _timeout_branch()
    assert "clearAgentBacktestRunning" in branch, branch
    assert "liveBacktestProgressByRunId = Object.create(null)" in branch, branch
    assert "liveBacktestProgress = null" in branch, branch


def test_timeout_branch_repaints_the_cards_it_just_cleared():
    """Clearing the map is invisible on its own: polling has just stopped, so
    refreshRunningAgentCards() never runs again and the card sits on
    "Backtesting…" with a frozen timer until some unrelated re-render happens
    by. The finished branch has always repainted; this one did not."""
    branch = _timeout_branch()
    assert "loadAgents()" in branch, branch


def _timeout_branch() -> str:
    poller = fn_body("function ensureBacktestPolling(")
    start = poller.index("if (attempts >= maxAttempts) {")
    return poller[start : poller.index("\n        } catch (error) {", start)]


def test_live_poll_hands_the_panel_the_shared_progress_object():
    """The helpers only matter if the live call site actually supplies it; every
    terminal call site correctly omits it, and `{}` before the first step is
    what keeps the startup notice reachable."""
    poller = fn_body("function ensureBacktestPolling(")
    # Anchored to the running branch rather than "the first call in the
    # function": the error, completion and timeout sites are all in here too,
    # and they must NOT gain this field.
    running_branch = poller[poller.index("const stepPct") :]
    call = running_branch[running_branch.index("updateBacktestRunProgress({") :]
    call = call[: call.index("});") + 3]
    assert "progress:" in call
    assert "|| {}" in call


def test_poller_queries_each_concurrent_live_run_id():
    """After entitlements allow N concurrent dashboard backtests, a single
    focused liveBacktestRunId poll left every other card on an empty bar.
    The poller must ask /backtest/status?live_run_id= for each in-flight job.
    """
    poller = fn_body("function ensureBacktestPolling(")
    assert "Promise.all" in poller
    assert "live_run_id=" in poller
    assert "liveBacktestProgressByRunId[" in poller


def test_progress_reaches_each_concurrent_agent():
    """Two confirmed runs must each receive their own step numbers — the
    regression behind empty progress on the non-focused card."""
    script = "\n".join(
        [
            js_const("BACKTEST_POLL_MAX_SECONDS"),
            (
                "const MAP = {"
                " 'agent-A': {runId: 'run-1', startedAt: Date.now() - 30000},"
                " 'agent-B': {runId: 'run-2', startedAt: Date.now() - 25000}"
                "};"
            ),
            "function readRunningBacktests() { return MAP; }",
            "function clearAgentBacktestRunning(id) { delete MAP[id]; }",
            "let liveBacktestRunId = 'run-2';",
            "let liveBacktestProgress = null;",
            (
                "const liveBacktestProgressByRunId = {"
                " 'run-1': {step: 3, totalSteps: 21, ageSeconds: 1, ageAt: Date.now()},"
                " 'run-2': {step: 8, totalSteps: 21, ageSeconds: 1, ageAt: Date.now()}"
                "};"
            ),
            fn_body("function getAgentBacktestRunning("),
            (
                "console.log(JSON.stringify({"
                " a: getAgentBacktestRunning('agent-A'),"
                " b: getAgentBacktestRunning('agent-B')"
                "}));"
            ),
        ]
    )
    result = _node(script)
    assert result["a"]["step"] == 3
    assert result["b"]["step"] == 8
