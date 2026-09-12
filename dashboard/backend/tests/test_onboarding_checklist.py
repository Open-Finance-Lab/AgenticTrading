"""My Agents guides a first-time user through one closed loop: run a backtest,
see its results.

**Why nothing here is stored.** The obvious design -- tick boxes in
localStorage, or a `user_onboarding` table -- was rejected because this repo
already has a cautionary instance of it. `defaultAgentProvisionGuardKey()`
(app.js:255) keeps a localStorage flag meaning "we provisioned starters for this
identity", and its own comment records why that is unreliable: user ids recycle
on local SQLite and on Render's ephemeral disk, so the key can name a different
person. Every tick below is instead a live read of data the page already holds,
which cannot disagree with reality and needs no migration.

**"You have an agent" is not progress.** `api/auth.py:534` calls
`provision_starter_agents` at signup (DeepSeek V4 Pro, GPT-5.5, Claude Sonnet
4.6), and `ensureDefaultFoundationAgent()` re-provisions client-side for guests.
`GET /api/v1/agents` is therefore non-empty for every visitor who has ever
loaded the page, signed in or not. A step keyed on agent existence would arrive
pre-ticked for everyone and teach the user the checklist is decorative --
`test_starter_agents_alone_tick_nothing` is the guard against someone adding
one.

**The two steps read different sources, and they have to.** The dashboard engine
calls `db.insert_run` only at completion (engine.py:1650), with `final_equity`
already populated -- there is no in-flight `agent_runs` row. So `run_count > 0`
means "a finished run exists" and would tick BOTH steps in the same instant,
collapsing the checklist into a single on/off. The launch step therefore also
consults `readRunningBacktests()` (app.js:5434), the sessionStorage registry My
Agents already keeps for in-flight runs, which is what produces the middle state
a user actually spends minutes in: launched, still running, no results yet.
"""

import json
import re
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import (
    APP_HTML,
    APP_JS,
    STYLES,
    call_args,
    fn_body,
    strip_comments,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _node(script: str) -> object:
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _derive(agents_js: str, running_js: str = "[]", opts_js: str = "{}") -> dict:
    """Run the shipped deriveOnboardingChecklist() over the given inputs.

    agentRunCount() is lifted from app.js rather than stubbed here: it is the
    page's one definition of "this agent has been backtested", and a harness
    that restated it would keep passing after the shipped helper changed.
    """
    script = "\n".join(
        [
            fn_body("function agentRunCount("),
            fn_body("function deriveOnboardingChecklist("),
            "const out = deriveOnboardingChecklist("
            f"{agents_js}, {running_js}, {opts_js});",
            "console.log(JSON.stringify(out));",
        ]
    )
    return _node(script)


def _step(state: dict, key: str) -> dict:
    match = [s for s in state["steps"] if s["key"] == key]
    assert match, f"no step named {key!r} in {state['steps']}"
    return match[0]


#: The three cards every account is born with. Nothing has been run on them.
_STARTERS = (
    "["
    "{agent_id: 'a1', agent_type: 'builtin', model_name: 'deepseek-v4-pro', run_count: 0},"
    "{agent_id: 'a2', agent_type: 'builtin', model_name: 'gpt-5.5', run_count: 0},"
    "{agent_id: 'a3', agent_type: 'builtin', model_name: 'claude-sonnet-4-6', run_count: 0}"
    "]"
)

_ONE_FINISHED = "[{agent_id: 'a1', agent_type: 'builtin', run_count: 1}]"

#: One live run, in the shape listRunningBacktests() hands back -- a swept
#: list, not the raw sessionStorage map. See
#: test_the_renderer_reads_the_swept_run_list for why the distinction matters.
_IN_FLIGHT = "[{key: 'run-77', agentId: 'a1', runId: 'run-77', startedAt: 1}]"


# --- derivation -------------------------------------------------------------


def test_a_brand_new_owner_sees_both_steps_open():
    state = _derive(_STARTERS)
    assert state["visible"] is True
    assert _step(state, "launch")["done"] is False
    assert _step(state, "results")["done"] is False


def test_starter_agents_alone_tick_nothing():
    """Owning agents is not progress -- signup hands out three of them."""
    state = _derive(_STARTERS)
    assert [s["done"] for s in state["steps"]] == [False, False]


def test_a_run_in_flight_ticks_launch_but_not_results():
    """The middle state. Without readRunningBacktests() this is unreachable,
    because no agent_runs row exists until the run finishes."""
    state = _derive(_STARTERS, _IN_FLIGHT)
    assert _step(state, "launch")["done"] is True
    assert _step(state, "results")["done"] is False
    assert state["visible"] is True


def test_a_finished_run_closes_the_loop_and_hides_the_panel():
    state = _derive(_ONE_FINISHED)
    assert _step(state, "launch")["done"] is True
    assert _step(state, "results")["done"] is True
    assert state["visible"] is False


def test_the_panel_stays_hidden_before_agents_have_loaded():
    """A null roster is "we do not know yet", not "you have never run one".
    Rendering the open checklist during the first fetch would flash a
    to-do list at a user who finished the loop months ago."""
    for empty in ("null", "undefined"):
        state = _derive(empty)
        assert state["visible"] is False, empty
        assert _step(state, "launch")["done"] is False, empty


def test_an_empty_roster_stays_hidden_too():
    """Distinct from the null case above but the same answer: a roster that
    came back empty means the agents call failed or the identity has nothing,
    and instructing someone to run a backtest with no agent to run it on is
    advice they cannot follow."""
    assert _derive("[]")["visible"] is False


def test_a_null_run_count_is_not_a_finished_run():
    """Number(null) is 0 and Number.isFinite(0) is true, so a guard written as
    a finiteness check on the coerced value ticks the step for an agent whose
    run_count never arrived. agentRunCount() coerces and then tests truthiness,
    which lands on the right answer -- pinned because it is the kind of thing a
    later "simplification" of that helper could quietly lose."""
    state = _derive("[{agent_id: 'a1', run_count: null}]")
    assert _step(state, "results")["done"] is False


def test_a_string_run_count_still_counts():
    """JSON from the API is typed, but the mock/demo roster and older cached
    payloads are not -- '3' is three runs, not zero."""
    state = _derive("[{agent_id: 'a1', run_count: '3'}]")
    assert _step(state, "results")["done"] is True


def test_one_agent_with_runs_closes_the_loop_for_the_whole_roster():
    """The loop is per person, not per agent: a user who ran a backtest on one
    starter has seen results and does not need the panel beside the other two."""
    mixed = (
        "[{agent_id: 'a1', run_count: 0},"
        " {agent_id: 'a2', run_count: 2},"
        " {agent_id: 'a3', run_count: 0}]"
    )
    assert _derive(mixed)["visible"] is False


def test_an_empty_running_registry_does_not_tick_launch():
    assert _step(_derive(_STARTERS, "[]"), "launch")["done"] is False


def test_a_run_whose_agent_is_gone_does_not_tick_launch():
    """The registry outlives the agent in two ordinary cases: the agent was
    deleted, and logoutUser() navigates away without clearing sessionStorage,
    so the next account signed in to that tab inherits the previous one's
    entries. renderAgentCards() paints no card for either, so a tick here sends
    the reader to "the card below" and there is no card below."""
    orphan = "[{key: 'run-9', agentId: 'gone', runId: 'run-9', startedAt: 1}]"
    state = _derive(_STARTERS, orphan)
    assert _step(state, "launch")["done"] is False
    assert "card below" not in _step(state, "launch")["hint"]


def test_a_run_history_without_a_run_count_still_closes_the_loop():
    """agentRunCount() falls back to `runs` when `run_count` is absent, and the
    card beside the panel is badged from it. Two definitions of "has been
    backtested" put a BACKTESTED card directly under a checklist saying the
    results step was not done yet."""
    state = _derive("[{agent_id: 'a1', runs: [{run_id: 'r1'}]}]")
    assert _step(state, "results")["done"] is True
    assert state["visible"] is False


def test_a_missing_running_registry_is_survivable():
    """listRunningBacktests() returns [] on a storage failure, but the derive
    must not throw if a caller hands it nothing at all."""
    for absent in ("null", "undefined"):
        state = _derive(_STARTERS, absent)
        assert _step(state, "launch")["done"] is False, absent


def test_demo_mode_never_shows_the_checklist():
    """Demo mode is the no-backend fallback roster (MOCK_AGENTS), and a user
    with no backend cannot run a backtest. Those agents carry run_count > 0, so
    the derive hides the panel rather than handing out advice that cannot be
    followed. Pinned because zeroing the mock counts for some unrelated reason
    would silently start showing it."""
    block = APP_JS[APP_JS.index("const MOCK_AGENTS = [") :]
    block = block[: block.index("\n];")]
    counts = re.findall(r"run_count:\s*(\d+)", block)
    assert counts, "MOCK_AGENTS no longer declares run_count"
    assert any(int(c) > 0 for c in counts)


# --- placement and wiring ---------------------------------------------------


def test_the_panel_sits_above_the_agent_shelves():
    """Above #agentsCategories, not inside a shelf: it describes the whole
    page, and a shelf can be emptied by the search box."""
    panel = APP_HTML.index('id="onboardingChecklist"')
    shelves = APP_HTML.index('id="agentsCategories"')
    assert panel < shelves


def test_the_panel_ships_hidden():
    """First paint happens before the agents call answers. An unhidden panel
    would show the open checklist for that gap to everyone, including users who
    closed the loop long ago -- the same flash test_the_panel_stays_hidden_
    before_agents_have_loaded pins on the derive side."""
    opening = APP_HTML[APP_HTML.index('id="onboardingChecklist"') :]
    assert "hidden" in opening[: opening.index(">")]


def test_the_my_agents_paint_renders_the_checklist():
    body = fn_body("function renderAgentCategories(")
    assert "renderOnboardingChecklist(" in body


def test_the_checklist_reads_the_whole_roster_not_the_filtered_view():
    """renderAgentCategories is called with getFilteredAgents(), which the
    search box and the market chips narrow. The checklist describes the
    account, so it must read allAgents: filtering down to a shelf that excludes
    the agent carrying the runs would otherwise resurrect a panel the user
    already closed by finishing a backtest.

    Asserted over the call's argument list with comments stripped. The call site
    carries a note naming the collection it must not use, so a scan of the raw
    body is satisfied by the prose explaining the bug -- and would stay green
    with the code reverted to the filtered parameter."""
    body = strip_comments(fn_body("function renderAgentCategories("))
    assert call_args(body, "renderOnboardingChecklist") == "(allAgents)"


def test_the_full_paint_is_the_only_render_site():
    """The per-second refresh deliberately does NOT render it. Both events that
    can move a tick -- a launch and a completion -- change the set of running
    agents, and refreshRunningAgentCards answers a changed set by calling
    applyAgentFilters(), which is a full re-render. While that set is unchanged
    no tick can move, so a second render site on the 1Hz path could only ever
    repaint the same two rows. It also cannot be added casually: that function
    is executed under node by test_backtest_progress_card's patch harness,
    which supplies its collaborators by hand.

    Comments stripped for the same reason as above, in the other direction: a
    negative guard over raw source fails on a comment that merely names the
    function."""
    assert "renderOnboardingChecklist(" not in strip_comments(
        fn_body("function refreshRunningAgentCards(")
    )


def test_the_renderer_reads_the_swept_run_list():
    """listRunningBacktests(), never the raw readRunningBacktests() map. The map
    keeps entries for runs that died without a terminal status; the shelves drop
    those as they paint, through getAgentBacktestRunning(). Taken from the map,
    the panel announced a run whose card the same render then declined to
    draw -- and nothing schedules another paint to correct it."""
    body = strip_comments(fn_body("function renderOnboardingChecklist("))
    assert "listRunningBacktests()" in body
    assert "readRunningBacktests" not in body


def test_the_checklist_is_styled():
    assert ".onboarding-checklist" in STYLES


def test_the_launch_hint_stops_giving_instructions_once_the_run_starts():
    """A ticked step whose hint still says "pick an agent and open it in
    Backtest" is telling the reader to do the thing they just did. The step can
    only be done while the panel is visible if a run is in flight -- a finished
    run hides the panel -- so the hint has a true alternative to switch to."""
    idle = _step(_derive(_STARTERS), "launch")
    live = _step(_derive(_STARTERS, _IN_FLIGHT), "launch")
    assert "Pick an agent" in idle["hint"]
    assert live["hint"] != idle["hint"]
    assert "Running now" in live["hint"]


# --- the gap between a finished run and its run_count ------------------------

_AWAITING = "{awaitingResults: true}"


def test_the_checklist_does_not_untick_itself_when_the_run_succeeds():
    """The regression this flag exists for. A run ending clears its registry
    entry, and refreshRunningAgentCards() answers the changed running set with a
    full re-render -- before the roster refresh carrying the new run_count has
    landed. Both of the checklist's sources are false in that window, so the
    panel repainted with BOTH steps unticked and the hint back to "pick an
    agent", at the exact moment the user finished the loop it is celebrating."""
    state = _derive(_STARTERS, "[]", _AWAITING)
    assert _step(state, "launch")["done"] is True
    assert _step(state, "results")["done"] is False
    assert state["visible"] is True


def test_the_awaiting_hint_stops_claiming_the_run_is_still_going():
    """"Its progress is on the card below" is false once the run has ended --
    the card is gone. The three launch hints are three distinct states."""
    idle = _step(_derive(_STARTERS), "launch")["hint"]
    live = _step(_derive(_STARTERS, _IN_FLIGHT), "launch")["hint"]
    settling = _step(_derive(_STARTERS, "[]", _AWAITING), "launch")["hint"]
    assert len({idle, live, settling}) == 3
    assert "card below" not in settling


def test_a_live_run_outranks_the_awaiting_hint():
    """A second run launched while the first settles is the truer statement:
    there IS a card below."""
    hint = _step(_derive(_STARTERS, _IN_FLIGHT, _AWAITING), "launch")["hint"]
    assert "Running now" in hint


def test_the_finished_roster_still_hides_the_panel_while_awaiting():
    """The flag bridges a gap; it cannot reopen a closed loop."""
    assert _derive(_ONE_FINISHED, "[]", _AWAITING)["visible"] is False


def test_the_repaint_guard_can_see_a_hint_change():
    """The launch step moves from "Running now" to "loading your results" with
    both steps' done-ness unchanged. A signature built from the ticks alone
    matched and returned early, leaving the panel claiming a finished run was
    still going."""
    body = strip_comments(fn_body("function renderOnboardingChecklist("))
    signature = body[body.index("const signature") : body.index("panel.dataset")]
    assert "hint" in signature


def test_only_a_successful_run_arms_the_bridge():
    """A failed run leaves no results to wait for, so un-ticking is the honest
    answer -- the reader does need to run another one."""
    body = strip_comments(fn_body("function ensureBacktestPolling("))
    assert "if (status.success) onboardingAwaitingRunCount = true;" in body


def test_every_completion_refreshes_the_roster_that_clears_the_bridge():
    """The refresh used to be gated on the run being the focused one, so a
    background completion never reached it: its results stayed off the page and
    the flag armed above had nothing to clear it. Both must hold -- the arm and
    the refresh are one mechanism, and a flag with no clear is a permanent
    tick."""
    body = strip_comments(fn_body("function ensureBacktestPolling("))
    assert "if (anyFinished) loadAgents();" in body
    assert "anyFinished = true;" in body


def test_the_landed_roster_clears_the_bridge_before_it_paints():
    """Cleared before applyAgentFilters(), not after: the roster is the answer
    the flag stood in for, so the paint it feeds must come from the roster."""
    body = strip_comments(fn_body("async function loadAgentsNow("))
    assign = body.index("allAgents = agents;")
    clear = body.index("onboardingAwaitingRunCount = false;", assign)
    paint = body.index("applyAgentFilters();", assign)
    assert assign < clear < paint
