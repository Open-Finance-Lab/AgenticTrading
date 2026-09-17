"""js/admin-live.js under node: the live row reads /api/admin/stats and nothing else."""

import re

from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
CREDIT_FORMAT = source("credit-format.js")
LIVE = source("admin-live.js")
STATS = fixture("admin_stats.json")


def _eval(expression: str, *setup: str) -> object:
    # Promise.resolve(...).then(...) rather than a bare `console.log(JSON.stringify(expr))`:
    # several scenarios below evaluate to a pending Promise (async IIFEs exercising
    # request()/gate()/handleAccessLost()), and stringifying a Promise directly captures
    # it before it settles ("{}"), not its resolved value. Wrapping it uniformly also
    # covers the plain-value scenarios, since Promise.resolve() on a non-thenable just
    # forwards it to .then() unchanged.
    return run_node(
        SHELL, CREDIT_FORMAT, LIVE, *setup,
        f"Promise.resolve({expression}).then((result) => console.log(JSON.stringify(result)));",
    )


def test_tiles_render_users_agents_and_running_with_the_real_ceiling():
    result = _eval(
        "(() => {"
        f"  const root = window.AdminLive.renderTiles({STATS});"
        "  return byClass(root, 'snapshot-tile').map((tile) => ["
        "    byClass(tile, 'snapshot-label')[0].textContent,"
        "    byClass(tile, 'snapshot-value')[0].textContent,"
        "    (byClass(tile, 'snapshot-meta')[0] || {textContent: null}).textContent,"
        "  ]);"
        "})()"
    )
    assert result == [
        ["Total users", "412", None],
        ["Total agents", "4,380", None],
        ["Backtests running · this instance", "2", "of 5 slots on this instance"],
    ]


def test_tiles_show_dashes_for_a_missing_payload():
    result = _eval(
        "(() => { const root = window.AdminLive.renderTiles({}); return texts(byClass(root, 'snapshot-value')).concat(texts(byClass(root, 'snapshot-meta'))); })()"
    )
    assert result == ["—", "—", "—", "—"]


def test_load_paints_the_row_and_marks_the_update_time():
    result = _eval(
        "(async () => {"
        "  register('liveTiles', document.createElement('div'));"
        "  register('liveUpdated', document.createElement('small'));"
        "  const {panel} = panelStub();"
        "  document.querySelector = (selector) => selector === '[data-panel=\"live\"]' ? panel : null;"
        f"  fetchQueue.push({{ok: true, status: 200, body: {STATS}}});"
        "  await window.AdminLive.load();"
        "  return {calls: fetchCalls.map(([url]) => url), values: texts(byClass(document.getElementById('liveTiles'), 'snapshot-value')), updated: document.getElementById('liveUpdated').textContent, busy: panel.getAttribute('aria-busy')};"
        "})()"
    )
    assert result["calls"] == ["/api/admin/stats"]
    assert result["values"] == ["412", "4,380", "2"]
    assert re.fullmatch(r"Updated [A-Z][a-z]{2} \d{1,2}, \d{4}, \d{2}:\d{2} UTC", result["updated"]), result["updated"]
    assert result["busy"] == "false"


def test_load_failure_keeps_the_last_good_row_and_says_so():
    result = _eval(
        "(async () => {"
        "  register('liveTiles', document.createElement('div'));"
        "  register('liveUpdated', document.createElement('small'));"
        "  const {panel, parts} = panelStub();"
        "  document.querySelector = (selector) => selector === '[data-panel=\"live\"]' ? panel : null;"
        f"  fetchQueue.push({{ok: true, status: 200, body: {STATS}}});"
        "  await window.AdminLive.load();"
        "  fetchQueue.push({ok: false, status: 503, body: {}});"
        "  await window.AdminLive.load();"
        "  const stale = {values: texts(byClass(document.getElementById('liveTiles'), 'snapshot-value')), error: parts.errorText.textContent, isStale: panel.classList.contains('is-stale')};"
        "  fetchQueue.push({ok: false, status: 403, body: {}});"
        "  await window.AdminLive.load();"
        "  return {stale, nav};"
        "})()"
    )
    assert result["stale"] == {"values": ["412", "4,380", "2"], "error": "This section is temporarily unavailable.", "isStale": True}
    assert result["nav"] == [["replace", "/app"]]


def test_module_reads_only_the_stats_route():
    assert LIVE.count("/api/admin/") == 1
    assert "/api/admin/stats" in LIVE
    assert "/api/admin/analytics" not in LIVE
    assert "#live" not in LIVE
