"""js/admin-shell.js under node: router, URL state, formatters, guards, dialogs."""

import json

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
CREDIT_FORMAT = source("credit-format.js")


def _eval(expression: str, *setup: str) -> object:
    # Promise.resolve(...).then(...) rather than a bare `console.log(JSON.stringify(expr))`:
    # several scenarios below evaluate to a pending Promise (async IIFEs exercising
    # request()/gate()/handleAccessLost()), and stringifying a Promise directly captures
    # it before it settles ("{}"), not its resolved value. Wrapping it uniformly also
    # covers the plain-value scenarios, since Promise.resolve() on a non-thenable just
    # forwards it to .then() unchanged.
    return run_node(
        SHELL, CREDIT_FORMAT, *setup,
        f"Promise.resolve({expression}).then((result) => console.log(JSON.stringify(result)));",
    )


def test_hash_router_knows_exactly_the_seven_routes_and_the_profile():
    assert _eval("window.AdminShell.ROUTES") == [
        "overview", "sources", "retention", "credits", "lifecycle", "health", "users",
    ]
    assert _eval("window.AdminShell.parseHash('')") == {"route": "overview", "id": None}
    assert _eval("window.AdminShell.parseHash('#health')") == {"route": "health", "id": None}
    assert _eval("window.AdminShell.parseHash('#users')") == {"route": "users", "id": None}
    assert _eval("window.AdminShell.parseHash('#users/42')") == {"route": "users", "id": "42"}
    assert _eval("window.AdminShell.parseHash('#users/abc')") == {"route": "users", "id": None}
    # No #live route (design §8.2, D16) and no orphan routes: unknown → overview.
    for unknown in ("#live", "#usage", "#revenue", "#profiles", "#funnel", "#nonsense"):
        assert _eval(f"window.AdminShell.parseHash('{unknown}')") == {"route": "overview", "id": None}, unknown


def test_range_maps_to_inclusive_utc_dates_within_the_180_day_cap():
    today = "new Date('2026-09-15T13:00:00Z')"
    assert _eval(f"window.AdminShell.rangeDates('1W', {today})") == {"from": "2026-09-09", "to": "2026-09-15"}
    assert _eval(f"window.AdminShell.rangeDates('1M', {today})") == {"from": "2026-08-17", "to": "2026-09-15"}
    # 180 inclusive days: `(end - start).days > MAX_VALUE_RANGE_DAYS` is the server's bound.
    assert _eval(f"window.AdminShell.rangeDates('1Y', {today})") == {"from": "2026-03-20", "to": "2026-09-15"}
    assert _eval(f"window.AdminShell.rangeDates('1D', {today})") == {"from": "2026-09-09", "to": "2026-09-15"}


def test_url_state_round_trips_and_drops_unknown_values():
    parsed = _eval(
        "window.AdminShell.readUrlState('?range=1M&group=organic&segment=bogus&tier=invested&internal=true&q=ada&priority=true')"
    )
    assert parsed == {
        "range": "1M",
        "filters": {"group": "organic", "segment": "", "tier": "invested", "internal": True, "q": "ada", "priority": True},
    }
    assert _eval(
        "window.AdminShell.buildSearch('1M', {group: 'organic', segment: '', tier: 'invested', internal: true, q: 'ada', priority: true})"
    ) == "?range=1M&group=organic&tier=invested&internal=true&q=ada&priority=true"
    assert _eval("window.AdminShell.buildSearch('1W', {group: '', segment: '', tier: '', internal: false, q: '', priority: false})") == ""


def test_request_parameters_use_the_declared_query_names():
    setup = (
        "window.AdminShell.state.range = '1W';"
        "window.AdminShell.state.filters = {group: 'partner', segment: 'core', tier: 'invested', internal: true, q: 'ada', priority: true};"
        "window.AdminShell.today = () => new Date('2026-09-15T00:00:00Z');"
    )
    assert _eval("window.AdminShell.analyticsParams().toString()", setup) == "from=2026-09-09&to=2026-09-15&include_internal=true"
    assert _eval("window.AdminShell.analyticsParams({withGroup: true}).toString()", setup) == "from=2026-09-09&to=2026-09-15&include_internal=true&user_group=partner"
    assert _eval("window.AdminShell.userListParams({offset: 50}).toString()", setup) == (
        "q=ada&user_group=partner&lifecycle_segment=core&commercial_tier=invested&priority=true&include_internal=true&limit=50&offset=50"
    )


def test_formatters_are_display_safe_and_fixed_locale():
    assert _eval("window.AdminShell.formatNumber(1234)") == "1,234"
    assert _eval("window.AdminShell.formatNumber(null)") == "—"
    assert _eval("window.AdminShell.formatPercent(0.6666)") == "66.7%"
    assert _eval("window.AdminShell.formatPercent(null)") == "—"
    assert _eval("window.AdminShell.formatPercent(1.5)") == "—"
    assert _eval("window.AdminShell.formatCredits(4800000)") == "4.800000 Credits"
    assert _eval("window.AdminShell.formatCredits(null)") == "—"
    assert _eval("window.AdminShell.formatDateOnly('2026-09-01')") == "Sep 1, 2026"
    assert _eval("window.AdminShell.formatShortDay('2026-09-04')") == "Sep 4"
    assert _eval("window.AdminShell.formatTimestamp('2026-08-22T11:20:00Z')") == "Aug 22, 2026, 11:20 UTC"
    assert _eval("window.AdminShell.formatTimestamp(null, 'No activity')") == "No activity"
    assert _eval("window.AdminShell.humanize('no_usable_billing_lane')") == "No Usable Billing Lane"


def test_freshness_legend_names_yesterday_utc():
    assert _eval("window.AdminShell.freshnessLegendText(new Date('2026-09-15T02:00:00Z'))") == (
        "Daily figures complete through 2026-09-14 UTC; live tiles are this instance's process state"
    )


def test_availability_incomplete_reads_both_payload_shapes():
    assert _eval("window.AdminShell.availabilityIncomplete({available: false, error_code: 'temporarily_unavailable'})") is True
    assert _eval("window.AdminShell.availabilityIncomplete({available: true, stale: false, status: 'partial'})") is True
    assert _eval("window.AdminShell.availabilityIncomplete({snapshot: {available: true}, growth: {available: false}})") is True
    assert _eval("window.AdminShell.availabilityIncomplete({current: {status: 'ready'}, history: {status: 'partial'}})") is True
    assert _eval("window.AdminShell.availabilityIncomplete({snapshot: {available: true}, growth: {available: true}})") is False
    assert _eval("window.AdminShell.availabilityIncomplete(null)") is False


def test_field_pending_distinguishes_absent_from_served_and_empty():
    """Interim contract: absent (not served yet) renders PENDING; served-and-empty renders the panel's own copy."""
    assert _eval("window.AdminShell.PENDING") == "Awaiting data source"
    assert _eval("window.AdminShell.fieldPending({daily_active_users: []}, 'billing_lane_mix')") is True
    assert _eval("window.AdminShell.fieldPending({billing_lane_mix: []}, 'billing_lane_mix')") is False
    assert _eval("window.AdminShell.fieldPending({billing_lane_mix: null}, 'billing_lane_mix')") is False
    assert _eval("window.AdminShell.fieldPending(null, 'billing_lane_mix')") is False
    assert _eval("window.AdminShell.fieldPending(undefined, 'billing_lane_mix')") is False


def test_rules_dialog_carries_section_15_copy():
    entries = _eval("window.AdminShell.rulesEntries()")
    assert [label for label, _rule in entries] == [
        "New", "Onboarding", "Growing", "Core", "At risk", "Dormant",
        "Blocked", "Needs attention", "Healthy", "Commercial value",
    ]
    assert entries[4][1] == "Last meaningful activity was 8–29 UTC days ago."
    assert entries[9][1] == "Commercial value uses settled purchases minus refunds. Admin Grants do not count as purchases."


def test_request_seq_guard_and_access_loss():
    result = _eval(
        "(async () => {"
        "  const shell = window.AdminShell;"
        "  const a = shell.nextSeq('overview'); const b = shell.nextSeq('overview');"
        "  const stale = shell.isCurrent('overview', a); const fresh = shell.isCurrent('overview', b);"
        "  shell.invalidateAll();"
        "  const afterInvalidate = shell.isCurrent('overview', b);"
        "  const lost = await shell.handleAccessLost({status: 403});"
        "  const kept = await shell.handleAccessLost({status: 500});"
        "  return {stale, fresh, afterInvalidate, lost, kept, nav};"
        "})()"
    )
    assert result == {"stale": False, "fresh": True, "afterInvalidate": False, "lost": True, "kept": False, "nav": [["replace", "/app"]]}


def test_request_is_a_credentialed_get_that_throws_with_status():
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {users: 1}});"
        "  fetchQueue.push({ok: false, status: 503, body: {detail: 'x'}});"
        "  const ok = await window.AdminShell.request('/api/admin/stats');"
        "  let failed = null;"
        "  try { await window.AdminShell.request('/api/admin/analytics/overview'); } catch (error) { failed = error.status; }"
        "  return {ok, failed, calls: fetchCalls.map(([url, options]) => [url, options.method, options.credentials])};"
        "})()"
    )
    assert result == {
        "ok": {"users": 1},
        "failed": 503,
        "calls": [["/api/admin/stats", "GET", "include"], ["/api/admin/analytics/overview", "GET", "include"]],
    }


def test_gate_sends_non_admins_to_the_app():
    for body, ok, expected in (
        ({"user": {"role": "admin"}}, True, {"admin": True, "nav": []}),
        ({"user": {"role": "user"}}, True, {"admin": False, "nav": [["replace", "/app"]]}),
        ({"detail": "Not authenticated"}, False, {"admin": False, "nav": [["replace", "/app"]]}),
    ):
        result = _eval(
            "(async () => {"
            f"  fetchQueue.push({{ok: {str(ok).lower()}, status: {200 if ok else 401}, body: {json.dumps(body)}}});"
            "  const admin = await window.AdminShell.gate();"
            "  return {admin, nav};"
            "})()"
        )
        assert result == expected, body


def test_set_panel_state_covers_loading_empty_error_and_stale():
    result = _eval(
        "(() => {"
        "  const {panel, parts} = panelStub();"
        "  const shell = window.AdminShell;"
        "  shell.setPanelState(panel, {busy: true});"
        "  const loading = [panel.getAttribute('aria-busy'), parts.status.hidden];"
        "  shell.setPanelState(panel, {busy: false, error: shell.SECTION_UNAVAILABLE});"
        "  const errored = [panel.getAttribute('aria-busy'), parts.error.hidden, parts.errorText.textContent];"
        "  shell.setPanelState(panel, {busy: false, stale: true});"
        "  const stale = [panel.classList.contains('is-stale'), parts.status.textContent, parts.status.hidden];"
        "  shell.setPanelState(panel, {busy: false, status: shell.INCOMPLETE});"
        "  const incomplete = [panel.classList.contains('is-stale'), parts.status.textContent, parts.error.hidden];"
        "  return {loading, errored, stale, incomplete};"
        "})()"
    )
    assert result == {
        "loading": ["true", True],
        "errored": ["false", False, "This section is temporarily unavailable."],
        "stale": [True, "Showing the last successful response; refresh failed.", False],
        "incomplete": [False, "Incomplete data", True],
    }


def test_dialogs_return_focus_to_the_opener():
    result = _eval(
        "(() => {"
        "  const dialog = document.createElement('dialog'); dialog.id = 'evidenceDialog';"
        "  dialog.showModal = () => { dialog.open = true; }; dialog.close = () => { dialog.open = false; };"
        "  const opener = document.createElement('button'); globalThis.focused = null;"
        "  window.AdminShell.openDialog(dialog, opener);"
        "  const opened = dialog.open;"
        "  window.AdminShell.closeDialog(dialog);"
        "  return {opened, closed: !dialog.open, returned: globalThis.focused === opener};"
        "})()"
    )
    assert result == {"opened": True, "closed": True, "returned": True}
