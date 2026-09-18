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
        "  shell.setPanelState(panel, {busy: false, empty: true});"
        "  const emptyDefault = {"
        "    body: parts.body.children.map((child) => ({tag: child.tagName, class: child.className, text: child.textContent})),"
        "    ariaBusy: panel.getAttribute('aria-busy'), statusHidden: parts.status.hidden, errorHidden: parts.error.hidden,"
        "  };"
        "  shell.setPanelState(panel, {busy: false, empty: 'No credits usage in this range.'});"
        "  const emptyCustom = parts.body.children.map((child) => ({tag: child.tagName, class: child.className, text: child.textContent}));"
        "  return {loading, errored, stale, incomplete, emptyDefault, emptyCustom};"
        "})()"
    )
    assert result == {
        "loading": ["true", True],
        "errored": ["false", False, "This section is temporarily unavailable."],
        "stale": [True, "Showing the last successful response; refresh failed.", False],
        "incomplete": [False, "Incomplete data", True],
        # `empty` clears the body and replaces it with a single paragraph -- true uses the
        # default copy, a string overrides it -- while the loading/error affordances stay
        # hidden (this call passes neither `error` nor `stale`, both defaulting off).
        "emptyDefault": {
            "body": [{"tag": "P", "class": "panel-empty", "text": "Nothing to show for this range."}],
            "ariaBusy": "false", "statusHidden": True, "errorHidden": True,
        },
        "emptyCustom": [{"tag": "P", "class": "panel-empty", "text": "No credits usage in this range."}],
    }


def test_a_stale_panel_that_is_also_incomplete_says_both():
    """`stale ? STALE_NOTICE : status` published the weaker of two true warnings.

    paint() passes both whenever a panel renders partially-available rollups whose
    sibling endpoint also failed, so the one case where the data on screen is both
    out of date *and* known-incomplete showed only that it was out of date.
    """
    result = _eval(
        "(() => {"
        "  const {panel, parts} = panelStub();"
        "  const shell = window.AdminShell;"
        "  shell.setPanelState(panel, {busy: false, status: shell.INCOMPLETE, stale: true});"
        "  return [parts.status.textContent, parts.status.hidden, panel.classList.contains('is-stale')];"
        "})()"
    )
    assert result == [
        "Showing the last successful response; refresh failed. · Incomplete data",
        False,
        True,
    ]


def test_credits_formatter_survives_a_failed_credit_format_load():
    """js/credit-format.js is a separate <script>; a 404, a CSP block or an
    ad-blocker leaves window.CreditFormat undefined. Unguarded, the dereference
    threw out of whichever renderer called it -- see the panel-containment test
    in test_admin_overview_frontend.py for what that cost."""
    assert _eval("(() => { delete window.CreditFormat; return window.AdminShell.formatCredits(4800000); })()") == "—"


def test_a_half_open_period_end_renders_as_the_last_day_inside_it():
    """`selected_period_end` is `to + 1 day` (admin_analytics.py's
    _value_range_from_values), published verbatim by value_queries.py:1388. Read
    as an inclusive date it makes the 1W range an eight-day week."""
    assert _eval("window.AdminShell.formatLastIncludedDay('2026-09-18')") == "Sep 17, 2026"
    assert _eval("window.AdminShell.formatLastIncludedDay('2026-09-01')") == "Aug 31, 2026"
    assert _eval("window.AdminShell.formatLastIncludedDay(null)") == "—"
    assert _eval("window.AdminShell.formatLastIncludedDay('', 'No period')") == "No period"
    assert _eval("window.AdminShell.formatLastIncludedDay('not-a-date')") == "—"


def test_one_history_traversal_routes_exactly_once():
    """A same-document hash traversal fires popstate *and* hashchange, so binding
    route() to both ran every navigation twice: doubled panel fetches, and a Back
    onto #users/{id} wrote two admin profile-access audit rows for one view
    (_record_access, admin_analytics.py:559). nextSeq/isCurrent guards the render,
    never the request, so this has to be checked at the routing boundary."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'admin'}}});"
        "  let routes = 0;"
        "  document.addEventListener('admin:route', () => { routes += 1; });"
        "  window.location.hash = '#users';"
        "  document.dispatchEvent(new Event('DOMContentLoaded'));"
        "  await new Promise((resolve) => setTimeout(resolve, 0));"
        "  const boot = routes;"
        "  window.location.hash = '#overview';"
        "  winListeners.popstate.forEach((fn) => fn());"
        "  winListeners.hashchange.forEach((fn) => fn());"
        "  const traversal = routes - boot;"
        "  window.location.hash = '#health';"
        "  winListeners.hashchange.forEach((fn) => fn());"
        "  return {boot, traversal, nextNavigation: routes - boot - traversal, route: window.AdminShell.state.route};"
        "})()"
    )
    assert result == {"boot": 1, "traversal": 1, "nextNavigation": 1, "route": "health"}


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


def test_write_sends_the_verb_the_body_and_the_csrf_double_submit_header():
    """The /app copy of this lives in app.js's csrfHeaders(); /admin cannot reach it,
    so the shell carries its own reader. Without the header CsrfMiddleware answers 403
    to every provider write and to logout -- and a fetch stub would never notice."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {saved: true}});"
        "  const saved = await window.AdminShell.write('/api/admin/model-providers/x', "
        "    {method: 'PUT', body: JSON.stringify({display_name: 'X'})});"
        "  const [url, options] = fetchCalls[0];"
        "  return {saved, url, method: options.method, credentials: options.credentials,"
        "          csrf: options.headers['X-CSRF-Token'], type: options.headers['Content-Type'],"
        "          body: options.body};"
        "})()",
        "globalThis.document.cookie = 'other=1; atl_csrf=tok%20en; more=2';",
    )
    assert result == {
        "saved": {"saved": True},
        "url": "/api/admin/model-providers/x",
        "method": "PUT",
        "credentials": "include",
        "csrf": "tok en",
        "type": "application/json",
        "body": '{"display_name":"X"}',
    }


def test_write_reads_the_host_prefixed_csrf_cookie_production_sets():
    """cookie_secure() picks __Host-atl_csrf in prod and atl_csrf in dev
    (backend/csrf.py:51-52). Reading only one name works in exactly one of them."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 204, body: null});"
        "  const body = await window.AdminShell.write('/api/auth/logout', {method: 'POST'});"
        "  return {body, csrf: fetchCalls[0][1].headers['X-CSRF-Token'], sent: 'body' in fetchCalls[0][1]};"
        "})()",
        "globalThis.document.cookie = '__Host-atl_csrf=prodtoken';",
    )
    assert result == {"body": None, "csrf": "prodtoken", "sent": False}


def test_write_without_a_csrf_cookie_omits_the_header_rather_than_sending_empty():
    """An empty header is not the same request as no header: csrf_tokens_match()
    rejects both, but only the absent one lets the middleware fall through to the
    cookie-less agent lane it was written for. Sending '' claims a token we lack."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {}});"
        "  await window.AdminShell.write('/api/admin/x', {method: 'POST', body: '{}'});"
        "  return {present: 'X-CSRF-Token' in fetchCalls[0][1].headers};"
        "})()",
        "globalThis.document.cookie = '';",
    )
    assert result == {"present": False}


def test_write_throws_the_servers_detail_with_its_status():
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: false, status: 403, body: {detail: 'CSRF token missing or invalid'}});"
        "  try { await window.AdminShell.write('/api/admin/x', {method: 'DELETE', body: '{}'}); }"
        "  catch (error) { return {message: error.message, status: error.status}; }"
        "  return 'did not throw';"
        "})()",
        "globalThis.document.cookie = 'atl_csrf=t';",
    )
    assert result == {"message": "CSRF token missing or invalid", "status": 403}


def test_gate_retains_the_user_it_already_fetched():
    """N8: the account menu needs the display name and email, and the page has
    already paid for /api/auth/me. A second call for the same body is the thing
    this pin exists to stop."""
    granted = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'admin',"
        "    display_name: 'Ada Admin', email: 'ada@example.test'}}});"
        "  const admin = await window.AdminShell.gate();"
        "  return {admin, user: window.AdminShell.user(), calls: fetchCalls.length};"
        "})()"
    )
    assert granted == {
        "admin": True,
        "user": {"display_name": "Ada Admin", "email": "ada@example.test"},
        "calls": 1,
    }
    refused = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'user', email: 'bo@example.test'}}});"
        "  await window.AdminShell.gate();"
        "  return window.AdminShell.user();"
        "})()"
    )
    assert refused is None


def test_losing_access_drops_the_retained_user():
    """handleAccessLost redirects, but a redirect is not instantaneous: anything
    that reads user() in the same tick must not still see the identity."""
    result = _eval(
        "(async () => {"
        "  fetchQueue.push({ok: true, status: 200, body: {user: {role: 'admin', display_name: 'A', email: 'a@x.test'}}});"
        "  await window.AdminShell.gate();"
        "  await window.AdminShell.handleAccessLost({status: 401});"
        "  return {user: window.AdminShell.user(), nav};"
        "})()"
    )
    assert result == {"user": None, "nav": [["replace", "/app"]]}
