"""The running-backtest registry, executed rather than pattern-matched.

Issue #258. The `sessionStorage`-backed store added in PR #255 had four guards
in `test_my_agents_card_ui.py`, all of them static source-text assertions
(`assert "sessionStorage" in _APP_JS`). `/app` has no build step and nothing in
CI parses `app.js` as JavaScript, so a string check is the only thing standing
between this store and a behavioural regression -- and two real defects here
were found by reading, not by the suite.

So these lift the real functions out of `app.js` and run them under node
against a fake `sessionStorage` and a fake DOM. The cases are the ones a
reviewer cannot hold in their head: expiry arithmetic at the ceiling, a storage
value that is corrupt or not an object, an accessor that throws, two runs of one
agent, and the re-render-versus-patch decision in
`refreshRunningAgentCards()` -- which is load-bearing, because a full re-render
starts with `grid.innerHTML = ''` and would destroy focus, scroll position and
any open card menu once a second for the length of a run.

A skip is not a pass: the module is skipped wholesale when `node` is absent.
"""

import json
import re
import shutil
import subprocess

import pytest

from dashboard.backend.tests._frontend_source import APP_JS, fn_body, js_const

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _js_let(name: str) -> str:
    """The named top-level `let` declaration, verbatim.

    The sibling of `js_const` for mutable module state. `pendingBacktestSeq` is
    a counter whose only contract is that it keeps increasing, and restating it
    in the harness would test the harness's own seed rather than the shipped
    one.
    """
    match = re.search(rf"^let {re.escape(name)} = [^;]+;", APP_JS, re.MULTILINE)
    assert match, f"{name} is no longer a top-level let in app.js"
    return match.group(0)


_STORE_FUNCTIONS = (
    "function readRunningBacktests(",
    "function writeRunningBacktests(",
    "function markAgentBacktestRunning(",
    "function promoteBacktestRunKey(",
    "function clearAgentBacktestRunning(",
    "function listRunningBacktests(",
    "function getAgentBacktestRunning(",
)

# A frozen clock. The store reads Date.now() twice per lookup -- once to seed
# an entry, once to measure it -- so a case that wants the ceiling *exactly*
# cannot get there with the real clock: the few milliseconds between the two
# reads put elapsed just over, and the test fails one run in N. Freezing it
# makes the boundary a boundary, and makes the same-millisecond key collision a
# certainty instead of a coincidence.
_CLOCK = """
let nowMs = 1700000000000;
Date.now = () => nowMs;
"""

# A storage double, not a mock of the store: getItem/setItem are the only two
# surfaces app.js touches, and `failOn` lets a case make either one throw the
# way a private window or a full quota does.
_SESSION_STORAGE = """
const storageLog = [];
let storageFailOn = null;
const sessionStorage = {
    _data: {},
    getItem(key) {
        if (storageFailOn === 'get') throw new Error('SecurityError');
        return Object.prototype.hasOwnProperty.call(this._data, key)
            ? this._data[key]
            : null;
    },
    setItem(key, value) {
        storageLog.push(value);
        if (storageFailOn === 'set') throw new Error('QuotaExceededError');
        this._data[key] = String(value);
    },
};
function seedStore(value) {
    sessionStorage._data[RUNNING_BACKTESTS_KEY] =
        typeof value === 'string' ? value : JSON.stringify(value);
}
function storedStore() {
    const raw = sessionStorage._data[RUNNING_BACKTESTS_KEY];
    return raw === undefined ? null : JSON.parse(raw);
}
"""

# getAgentBacktestRunning() reads the live-poller globals directly.
_PROGRESS_GLOBALS = """
let liveBacktestProgressByRunId = {};
let liveBacktestRunId = null;
let liveBacktestProgress = null;
"""


def _harness(body: str, *, extra: str = "") -> str:
    return "\n".join(
        [
            _CLOCK,
            js_const("BACKTEST_POLL_MAX_SECONDS"),
            js_const("RUNNING_BACKTESTS_KEY"),
            _js_let("pendingBacktestSeq"),
            _SESSION_STORAGE,
            _PROGRESS_GLOBALS,
            *[fn_body(sig) for sig in _STORE_FUNCTIONS],
            extra,
            body,
        ]
    )


def _run(body: str, *, extra: str = "") -> dict:
    script = _harness(body, extra=extra)
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# ===========================================================================
# A stored value that is not what we wrote
# ===========================================================================


@pytest.mark.parametrize(
    "stored",
    ["{not json", '"a string"', "42", "null", "true"],
    ids=["corrupt", "string", "number", "null", "bool"],
)
def test_a_non_object_store_reads_as_empty(stored):
    """Anything that is not a JSON object has to read as "no runs".

    The card pins itself to "Backtesting…" off this map, so a value that
    survives `JSON.parse` but is not an object must not reach the callers as
    one.
    """
    out = _run(
        f"seedStore({json.dumps(stored)});"
        "console.log(JSON.stringify({map: readRunningBacktests()}));"
    )
    assert out["map"] == {}


def test_a_stored_array_is_rejected_rather_than_swept():
    """`typeof [] === 'object'`, so an array used to pass the shape check.

    Found by this test rather than by reading. An array that reached the sweep
    had its indices deleted; JSON.stringify wrote the holes back as nulls and
    JSON.parse read them as a dense array again, so the next read found the
    same three dead entries and wrote again -- a store that never converged and
    re-wrote sessionStorage on every launch. `Array.isArray` now rejects it at
    the door.
    """
    out = _run(
        'seedStore("[1,2,3]");'
        "const first = listRunningBacktests();"
        "const writes = storageLog.length;"
        "listRunningBacktests();"
        "console.log(JSON.stringify({"
        "runs: first, extraWrites: storageLog.length - writes}));"
    )
    assert out["runs"] == []
    assert out["extraWrites"] == 0


def test_a_throwing_getitem_reads_as_empty():
    """A private window denies storage by throwing, not by returning null."""
    out = _run(
        "storageFailOn = 'get';"
        "console.log(JSON.stringify({map: readRunningBacktests()}));"
    )
    assert out["map"] == {}


def test_a_throwing_setitem_does_not_propagate():
    """A full quota must not take the launch down with it.

    The in-page indicator works off the returned key regardless; only the
    survives-a-refresh half is lost.
    """
    out = _run(
        "storageFailOn = 'set';"
        "const key = markAgentBacktestRunning('a1', 'run-1');"
        "console.log(JSON.stringify({key}));"
    )
    assert out["key"] == "run-1"


# ===========================================================================
# Expiry arithmetic
# ===========================================================================


@pytest.mark.parametrize(
    ("offset_ms", "live"),
    [("0", True), ("1", False)],
    ids=["at-the-ceiling", "one-millisecond-past"],
)
def test_the_expiry_boundary_is_exclusive(offset_ms, live):
    """The comparison is `>`, so the ceiling second itself still counts.

    Both sides are asserted because only the pair pins the operator: a case at
    3600s alone passes with `>=`, and a case at 3601s alone passes with `>`.
    The off-by-one matters because it lands on the run about to finish -- a
    card that goes blank at exactly 3600s is a lost result, not a tidy sweep.
    """
    out = _run(
        "seedStore({'run-1': {agentId: 'a1', runId: 'run-1',"
        " startedAt: nowMs - BACKTEST_POLL_MAX_SECONDS * 1000 - "
        + offset_ms
        + "}});"
        "console.log(JSON.stringify({"
        "live: getAgentBacktestRunning('a1') !== null}));"
    )
    assert out["live"] is live


def test_an_entry_past_the_ceiling_is_dropped_and_swept():
    """A run that died without a terminal status must not pin a card forever."""
    out = _run(
        "seedStore({'run-1': {agentId: 'a1', runId: 'run-1',"
        " startedAt: nowMs - (BACKTEST_POLL_MAX_SECONDS + 1) * 1000}});"
        "const entry = getAgentBacktestRunning('a1');"
        "console.log(JSON.stringify({"
        "entry, stored: storedStore()}));"
    )
    assert out["entry"] is None
    assert out["stored"] == {}


def test_an_entry_with_no_startedat_is_dropped_by_the_card_lookup():
    """`Number(undefined)` is NaN, and NaN fails every comparison silently.

    Without the `Number.isFinite` guard the elapsed check reads `NaN > 3600`,
    which is false -- so a malformed entry would be permanently fresh and pin
    the card to "Backtesting..." for the life of the tab.
    """
    out = _run(
        "seedStore({'run-1': {agentId: 'a1', runId: 'run-1'}});"
        "console.log(JSON.stringify({"
        "entry: getAgentBacktestRunning('a1'), stored: storedStore()}));"
    )
    assert out["entry"] is None
    assert out["stored"] == {}


def test_an_entry_with_no_startedat_is_swept_by_the_listing_too():
    """The same NaN guard, asserted against `listRunningBacktests` alone.

    Deliberately does not call `getAgentBacktestRunning` first: that function
    carries its own copy of the guard and clears the entry, so a combined case
    passes with this one deleted. Found by mutation -- removing
    `!Number.isFinite` from the sweep left the whole module green.

    It matters because this listing is what the concurrency check counts, so a
    permanently-fresh entry refuses every later launch in the tab.
    """
    out = _run(
        "seedStore({'run-1': {agentId: 'a1', runId: 'run-1'}});"
        "console.log(JSON.stringify({"
        "runs: listRunningBacktests(), stored: storedStore()}));"
    )
    assert out["runs"] == []
    assert out["stored"] == {}


def test_the_sweep_is_persisted_once_not_per_entry():
    """listRunningBacktests writes back only when it actually removed something."""
    out = _run(
        "seedStore({"
        "dead1: {agentId: 'a1', startedAt: 1},"
        "dead2: {agentId: 'a2', startedAt: 1},"
        "live: {agentId: 'a3', startedAt: nowMs}});"
        "const runs = listRunningBacktests();"
        "const afterSweep = storageLog.length;"
        "listRunningBacktests();"
        "console.log(JSON.stringify({"
        "ids: runs.map((r) => r.agentId),"
        "writesForSweep: afterSweep,"
        "writesWhenNothingToSweep: storageLog.length - afterSweep}));"
    )
    assert out["ids"] == ["a3"]
    assert out["writesForSweep"] == 1
    assert out["writesWhenNothingToSweep"] == 0


def test_running_backtests_are_listed_oldest_first():
    out = _run(
        "const now = nowMs;"
        "seedStore({"
        "newer: {agentId: 'a2', startedAt: now - 1000},"
        "older: {agentId: 'a1', startedAt: now - 9000}});"
        "console.log(JSON.stringify({"
        "ids: listRunningBacktests().map((r) => r.agentId)}));"
    )
    assert out["ids"] == ["a1", "a2"]


# ===========================================================================
# Keys: legacy entries, two runs of one agent, promotion
# ===========================================================================


def test_a_legacy_entry_keyed_by_agent_id_still_resolves():
    """Entries written before the registry was keyed per run carry no agentId.

    They are read by falling back to the key, which for those entries *is* the
    agent id. A session open across the deploy that shipped per-run keys is the
    whole reason this fallback exists.
    """
    out = _run(
        "seedStore({a1: {startedAt: nowMs}});"
        "const entry = getAgentBacktestRunning('a1');"
        "console.log(JSON.stringify({"
        "found: entry !== null,"
        "listed: listRunningBacktests().map((r) => r.agentId)}));"
    )
    assert out["found"] is True
    assert out["listed"] == ["a1"]


def test_two_runs_of_one_agent_show_the_newest_and_clear_independently():
    """Clearing by agent id deleted whichever entry sat in that slot.

    With two runs of one agent that is the wrong one half the time, which is
    why callers hold a run key and hand it back.
    """
    out = _run(
        "const now = nowMs;"
        "seedStore({"
        "'run-1': {agentId: 'a1', runId: 'run-1', startedAt: now - 9000},"
        "'run-2': {agentId: 'a1', runId: 'run-2', startedAt: now - 1000}});"
        "const shown = getAgentBacktestRunning('a1');"
        "clearAgentBacktestRunning('run-2');"
        "const after = getAgentBacktestRunning('a1');"
        "console.log(JSON.stringify({"
        "shown: shown.runId, after: after.runId,"
        "remaining: Object.keys(storedStore())}));"
    )
    assert out["shown"] == "run-2"
    assert out["after"] == "run-1"
    assert out["remaining"] == ["run-1"]


def test_clearing_an_unknown_key_does_not_rewrite_the_store():
    out = _run(
        "seedStore({'run-1': {agentId: 'a1', startedAt: nowMs}});"
        "const before = storageLog.length;"
        "clearAgentBacktestRunning('run-does-not-exist');"
        "clearAgentBacktestRunning(null);"
        "console.log(JSON.stringify({writes: storageLog.length - before}));"
    )
    assert out["writes"] == 0


def test_two_pending_launches_in_one_millisecond_get_distinct_keys():
    """`pending:${Date.now()}` alone collides on a fast double-click.

    The sequence counter is what separates them, and a collision would make the
    second launch overwrite the first's entry -- the exact overwrite this
    registry exists to prevent.
    """
    out = _run(
        "const first = markAgentBacktestRunning('a1');"
        "const second = markAgentBacktestRunning('a1');"
        "console.log(JSON.stringify({"
        "first, second, entries: Object.keys(storedStore()).length}));"
    )
    assert out["first"] != out["second"]
    assert out["entries"] == 2


def test_promotion_carries_the_click_time_not_the_response_time():
    """The card's elapsed clock runs from the click, not from the POST answer."""
    out = _run(
        "const key = markAgentBacktestRunning('a1');"
        "const store = storedStore();"
        "const clickedAt = store[key].startedAt;"
        "const promoted = promoteBacktestRunKey(key, 'a1', 'run-9');"
        "const after = storedStore();"
        "console.log(JSON.stringify({"
        "promoted,"
        "keys: Object.keys(after),"
        "carried: after['run-9'].startedAt === clickedAt,"
        "agentId: after['run-9'].agentId}));"
    )
    assert out["promoted"] == "run-9"
    assert out["keys"] == ["run-9"]
    assert out["carried"] is True
    assert out["agentId"] == "a1"


def test_promotion_without_a_run_id_leaves_the_pending_entry_alone():
    out = _run(
        "const key = markAgentBacktestRunning('a1');"
        "const promoted = promoteBacktestRunKey(key, 'a1', null);"
        "console.log(JSON.stringify({"
        "same: promoted === key, keys: Object.keys(storedStore())}));"
    )
    assert out["same"] is True
    assert len(out["keys"]) == 1


# ===========================================================================
# refreshRunningAgentCards: patch in place, re-render only on a set change
# ===========================================================================

# Enough DOM for the patch path to run. querySelectorAll answers from a flat
# registry keyed by attribute, which is what the real code queries by.
_FAKE_DOM = """
const rendered = [];
function applyAgentFilters() { rendered.push('re-render'); }
function formatBacktestElapsed(s) { return `${s}s`; }
function deriveRunningProgress(entry) {
    return {
        determinate: true, pct: 42, sparkHtml: '<svg/>', equityPositive: true,
        equityLabel: '$1', stepLabel: '1/2', detail: '42%', notice: '',
    };
}
function fakeEl(attribute, value) {
    return {
        _attrs: {[attribute]: value}, _removed: [], dataset: {}, style: {},
        textContent: null, innerHTML: null, hidden: null,
        classList: {toggle() {}},
        getAttribute(name) { return this._attrs[name] ?? null; },
        setAttribute(name, v) { this._attrs[name] = String(v); },
        removeAttribute(name) { this._removed.push(name); },
    };
}
const domNodes = {};
const document = {
    querySelectorAll(selector) {
        const attribute = selector.slice(1, -1);
        return domNodes[attribute] || [];
    },
};
let lastRenderedRunningKey = null;
"""


def _run_refresh(body: str) -> dict:
    return _run(
        body,
        extra="\n".join(
            [_FAKE_DOM, fn_body("function refreshRunningAgentCards(")]
        ),
    )


def test_an_unchanged_running_set_patches_without_re_rendering():
    """The grid is rebuilt from `innerHTML = ''`, so a per-second re-render
    destroys focus, scroll and any open menu for the length of the run."""
    out = _run_refresh(
        "const elapsed = fakeEl('data-running-elapsed', 'a1');"
        "domNodes['data-running-elapsed'] = [elapsed];"
        "seedStore({'run-1': {agentId: 'a1', runId: 'run-1',"
        " startedAt: nowMs - 5000}});"
        "refreshRunningAgentCards();"
        "const first = rendered.length;"
        "refreshRunningAgentCards();"
        "refreshRunningAgentCards();"
        "console.log(JSON.stringify({"
        "firstCall: first, total: rendered.length,"
        "elapsedText: elapsed.textContent}));"
    )
    # The first call sees the key change from null and re-renders once; the
    # two after it patch.
    assert out["firstCall"] == 1
    assert out["total"] == 1
    assert out["elapsedText"] == "5s"


def test_a_changed_running_set_re_renders_and_returns_before_patching():
    """The early return matters: the nodes on screen belong to the previous
    set, so patching them before the re-render writes one agent's numbers onto
    another agent's card."""
    out = _run_refresh(
        "const elapsed = fakeEl('data-running-elapsed', 'a1');"
        "domNodes['data-running-elapsed'] = [elapsed];"
        "seedStore({'run-1': {agentId: 'a1', startedAt: nowMs}});"
        "refreshRunningAgentCards();"
        "elapsed.textContent = 'untouched';"
        "seedStore({'run-1': {agentId: 'a1', startedAt: nowMs},"
        " 'run-2': {agentId: 'a2', startedAt: nowMs}});"
        "refreshRunningAgentCards();"
        "console.log(JSON.stringify({"
        "renders: rendered.length, elapsedText: elapsed.textContent}));"
    )
    assert out["renders"] == 2
    assert out["elapsedText"] == "untouched"


def test_the_re_render_key_is_order_independent():
    """The key is a sorted join, so two agents arriving in the other order are
    the same set. Unsorted, the key flips every time the map is re-keyed and
    the grid re-renders once a second -- the bug this guard exists for."""
    out = _run_refresh(
        "const now = nowMs;"
        "seedStore({x: {agentId: 'a2', startedAt: now},"
        " y: {agentId: 'a1', startedAt: now}});"
        "refreshRunningAgentCards();"
        "seedStore({p: {agentId: 'a1', startedAt: now},"
        " q: {agentId: 'a2', startedAt: now}});"
        "refreshRunningAgentCards();"
        "console.log(JSON.stringify({renders: rendered.length}));"
    )
    assert out["renders"] == 1


def test_a_second_run_of_one_agent_is_not_a_set_change():
    """Two runs of one agent are two entries but one card, so the set is
    unchanged and the card is patched. Counting entries instead of distinct
    agents would re-render the whole grid on every launch."""
    out = _run_refresh(
        "const now = nowMs;"
        "seedStore({'run-1': {agentId: 'a1', startedAt: now}});"
        "refreshRunningAgentCards();"
        "seedStore({'run-1': {agentId: 'a1', startedAt: now},"
        " 'run-2': {agentId: 'a1', startedAt: now}});"
        "refreshRunningAgentCards();"
        "console.log(JSON.stringify({renders: rendered.length}));"
    )
    assert out["renders"] == 1


def test_the_cancel_button_appears_the_tick_after_the_post_answers():
    """At launch the entry has no run id, and the Cancel button lives in the
    actions block that a patch-only tick never re-renders. Patched here, it
    appears without waiting for a set change that may never come."""
    out = _run_refresh(
        "const cancel = fakeEl('data-running-cancel', 'a1');"
        "const pending = fakeEl('data-running-pending', 'a1');"
        "domNodes['data-running-cancel'] = [cancel];"
        "domNodes['data-running-pending'] = [pending];"
        "const key = markAgentBacktestRunning('a1');"
        "refreshRunningAgentCards();"
        "refreshRunningAgentCards();"
        "const atLaunch = {runId: cancel.dataset.runId, hidden: cancel.hidden,"
        " pendingHidden: pending.hidden};"
        "promoteBacktestRunKey(key, 'a1', 'run-7');"
        "refreshRunningAgentCards();"
        "console.log(JSON.stringify({atLaunch,"
        "after: {runId: cancel.dataset.runId, hidden: cancel.hidden,"
        " pendingHidden: pending.hidden}}));"
    )
    assert out["atLaunch"] == {"runId": "", "hidden": True, "pendingHidden": False}
    assert out["after"] == {"runId": "run-7", "hidden": False, "pendingHidden": True}
