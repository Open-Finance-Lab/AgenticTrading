"""js/admin-users.js under node: list rows, profile, activity tabs, evidence — against the fixtures."""

from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source, target_fixture

pytestmark = requires_node

SHELL = source("admin-shell.js")
CREDIT_FORMAT = source("credit-format.js")
USERS_JS = source("admin-users.js")
# Target-shape copies (Task 1): today's payloads plus the §9 fields. PR D switches these to `fixture`.
USERS = target_fixture("users.json")
PROFILE = target_fixture("user_detail.json")
SESSIONS = fixture("activity_sessions.json")
USAGE = fixture("activity_usage.json")
RUNS = fixture("activity_runs.json")
TIMELINE = fixture("activity_timeline.json")


def _eval(expression: str, *setup: str) -> object:
    # Wrapped in Promise.resolve(...).then(...) rather than a bare console.log:
    # an async IIFE's top-level `await` does not propagate outward from a nested
    # expression under `node -e`, so `console.log(JSON.stringify(expression))`
    # would stringify the still-pending Promise (`{}`) instead of its resolved
    # value. Harmless for synchronous expressions, so applied unconditionally.
    return run_node(
        SHELL, CREDIT_FORMAT, USERS_JS, *setup,
        f"Promise.resolve({expression}).then((result) => console.log(JSON.stringify(result)));",
    )


def test_user_rows_show_group_lifecycle_operational_and_last_active():
    result = _eval(
        "(() => {"
        f"  const rows = window.AdminUsers.renderUserRows({USERS}).children;"
        "  return rows.map((tr) => tr.children.map((td) => td.textContent));"
        "})()"
    )
    assert result == [
        ["Synthetic Adaada.synthetic@example.test", "invited", "At risk", "Healthy", "Aug 22, 2026, 11:20 UTC", "Review evidence"],
        ["<Synthetic & Grace>grace.synthetic@example.test", "free", "Onboarding", "Blocked", "Sep 1, 2026, 08:00 UTC", "Review evidence"],
    ]


def test_group_badge_is_rendered_verbatim_from_the_server():
    """D11: the server owns the precedence rule; the client prints the string it is given."""
    result = _eval(
        "(() => {"
        "  const badge = window.AdminUsers.groupBadge('zzz-not-a-real-badge');"
        "  const admin = window.AdminUsers.groupBadge('admin');"
        "  return [badge.textContent, badge.className, admin.textContent, admin.className];"
        "})()"
    )
    assert result == ["zzz-not-a-real-badge", "group-badge is-zzz-not-a-real-badge", "admin", "group-badge is-admin"]
    assert "role === 'admin'" not in USERS_JS
    assert "commercial_tier === 'unpaid'" not in USERS_JS


def test_rows_and_profile_without_recut_fields_render_dashes_not_guesses():
    """Interim contract (before PR D): absent group_badge / last_meaningful_activity_at read as —;
    the profile's activity line falls back to today's last_meaningful_activity, the same instant."""
    result = _eval(
        "(() => {"
        f"  const users = {USERS}; users.items.forEach((item) => {{ delete item.user_group; delete item.role; delete item.group_badge; delete item.last_meaningful_activity_at; }});"
        f"  const profile = {PROFILE}; delete profile.user_group; delete profile.role; delete profile.group_badge; delete profile.last_meaningful_activity_at;"
        "  const rows = window.AdminUsers.renderUserRows(users);"
        "  const badge = byClass(rows.children[0], 'group-badge')[0];"
        "  const header = window.AdminUsers.renderProfileHeader(profile);"
        "  return {"
        "    rows: rows.children.map((tr) => tr.children.map((td) => td.textContent)),"
        "    badge: [badge.className, badge.getAttribute('title')],"
        "    identity: byClass(header, 'identity')[0].children[1].children[1].textContent,"
        "    activity: texts(byTag(header, 'small')).find((text) => text.startsWith('Last meaningful activity')),"
        "  };"
        "})()"
    )
    assert result["rows"] == [
        ["Synthetic Adaada.synthetic@example.test", "—", "At risk", "Healthy", "—", "Review evidence"],
        ["<Synthetic & Grace>grace.synthetic@example.test", "—", "Onboarding", "Blocked", "—", "Review evidence"],
    ]
    assert result["badge"] == ["group-badge is-pending", "Awaiting data source"]
    assert result["identity"] == "ada.synthetic@example.test · — · Activation week of Jun 29, 2026"
    assert result["activity"].startswith("Last meaningful activity Aug 22, 2026, 11:20 UTC")


def test_user_rows_link_to_the_profile_hash_and_escape_nothing_by_hand():
    result = _eval(
        "(() => {"
        f"  const rows = window.AdminUsers.renderUserRows({USERS}).children;"
        "  return rows.map((tr) => byTag(tr, 'a')[0].getAttribute('href'));"
        "})()"
    )
    assert result == ["#users/101", "#users/102"]
    assert "innerHTML" not in USERS_JS


def test_empty_list_and_pager_copy():
    assert _eval("window.AdminUsers.renderUserRows({items: []}).children.map((tr) => tr.textContent)") == ["No users match these filters."]
    assert _eval("window.AdminUsers.renderPager({total: 120, offset: 50, items: new Array(50).fill({})})") == "Showing 51–100 of 120"
    assert _eval("window.AdminUsers.renderPager({total: 0, offset: 0, items: []})") == "0 users"


def test_evidence_dialog_content_is_display_safe_reasons_and_evidence():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderEvidence({USERS}.items[1]);"
        "  return {badges: texts(byClass(node, 'badge')), reasons: texts(byTag(node, 'p')), evidence: texts(byTag(node, 'li'))};"
        "})()"
    )
    assert result == {
        "badges": ["Onboarding", "Blocked", "Unpaid"],
        "reasons": ["grace.synthetic@example.test", "The account has not completed a successful backtest.", "The Credits account is restricted from model spending."],
        "evidence": ["no successful backtest recorded", "Credits account restriction is unresolved."],
    }


def test_profile_header_carries_badges_activation_week_and_milestones():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderProfileHeader({PROFILE});"
        "  return {h1: byTag(node, 'h1')[0].textContent, identity: byClass(node, 'identity')[0].children[1].children[1].textContent, badges: texts(byClass(node, 'badge')), group: texts(byClass(node, 'group-badge')), milestones: texts(byClass(node, 'milestone')), tabs: texts(byTag(node, 'button').filter((b) => b.getAttribute('role') === 'tab')), account: byTag(node, 'a').map((a) => a.getAttribute('href'))};"
        "})()"
    )
    assert result["h1"] == "Synthetic Ada"
    assert result["identity"] == "ada.synthetic@example.test · invited · Activation week of Jun 29, 2026"
    assert result["badges"] == ["At risk", "Healthy", "Invested"]
    assert result["group"] == ["invited"]
    assert result["milestones"] == [
        "Account signed upJul 1, 2026, 09:00 UTC", "Credential verifiedJul 1, 2026, 10:00 UTC",
        "Agent createdJul 1, 2026, 11:00 UTC", "Backtest completedJul 2, 2026, 10:00 UTC",
    ]
    assert result["tabs"] == ["Overview", "Timeline", "Runs", "Usage", "Sessions"]
    assert "/app?view=admin&adminTab=users&adminUserQuery=ada.synthetic%40example.test" in result["account"]


def test_activation_week_is_the_utc_monday():
    assert _eval("window.AdminUsers.activationWeekLabel('2026-07-02T10:00:00Z')") == "Activation week of Jun 29, 2026"
    assert _eval("window.AdminUsers.activationWeekLabel('2026-06-29T00:00:00Z')") == "Activation week of Jun 29, 2026"
    assert _eval("window.AdminUsers.activationWeekLabel(null)") == "Not yet activated"


def test_profile_overview_renders_value_facts_and_drops_the_d15_fields():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderProfileOverview({PROFILE});"
        "  const dts = texts(byTag(node, 'dt')); const dds = texts(byTag(node, 'dd'));"
        "  return {facts: dts.map((label, index) => [label, dds[index]]), all: flatten(node).map((n) => n.nodeType === 3 ? n.textContent : (n._text || ''))};"
        "})()"
    )
    facts = dict(result["facts"])
    # The fixture's selected_period_end is 2026-09-01, and that field is a *half-open*
    # boundary (admin_analytics.py sets `end = to + 1 day`). Rendered verbatim it
    # claimed a period one day longer than the one that was queried.
    assert facts["Selected period"] == "Aug 1, 2026 – Aug 31, 2026"
    assert facts["Activated"] == "Jul 2, 2026, 10:00 UTC"
    assert facts["Active days (30d)"] == "2"
    assert facts["Successful backtests (30d)"] == "2"
    assert facts["Inactive UTC days"] == "12"
    assert facts["Lifetime net purchased"] == "7.000000 Credits"
    assert facts["Consumed in period"] == "0.750000 Credits"
    assert facts["Available balance"] == "2.000000 Credits"
    assert facts["Completed"] == "8"
    assert facts["ATL Credits debited"] == "4.250000 Credits"
    assert "Top product page" not in facts
    for leaked in ("US", "desktop", "Chrome"):
        assert leaked not in result["all"], leaked
    for field in ("country_code", "device_category", "browser_family", "top_product_page"):
        assert field not in USERS_JS, field


def test_sessions_table_has_three_columns_and_no_region_device_browser():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderActivityItems('sessions', {SESSIONS}.items);"
        "  return {headers: texts(byTag(node, 'th')), rows: byTag(node, 'tbody')[0].children.map((tr) => texts(tr.children))};"
        "})()"
    )
    assert result == {
        "headers": ["Started", "Events", "Visible time"],
        "rows": [["Aug 26, 2026, 10:30 UTC", "9", "35m 0s"], ["Aug 25, 2026, 08:00 UTC", "4", "8m 0s"]],
    }


def test_usage_table_never_prints_an_atl_charge_for_byok():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderActivityItems('usage', {USAGE}.items);"
        "  return byTag(node, 'tbody')[0].children.map((tr) => texts(tr.children));"
        "})()"
    )
    assert result[0] == ["Aug 26, 2026, 11:02 UTC", "Model usage recorded", "openrouter · openai/gpt-5.5", "Platform Credits", "1,400", "350", "$0.43", "—"]
    assert result[1] == ["Aug 25, 2026, 09:18 UTC", "Model usage recorded", "openrouter · anthropic/claude-sonnet-4", "BYOK — no ATL charge", "900", "210", "—", "—"]
    assert result[2] == ["Aug 26, 2026, 11:03 UTC", "ATL Credits debited", "—", "Platform Credits", "—", "—", "—", "0.430000 Credits"]


def test_runs_and_timeline_renderers_label_events():
    runs = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderActivityItems('runs', {RUNS}.items);"
        "  return byTag(node, 'tbody')[0].children.map((tr) => texts(tr.children));"
        "})()"
    )
    assert runs == [["Aug 26, 2026, 11:02 UTC", "Backtest completed", "Succeeded", "openrouter · openai/gpt-5.5", "Platform Credits", "—"]]
    timeline = _eval(
        "(() => {"
        f"  const node = window.AdminUsers.renderActivityItems('timeline', {TIMELINE}.items);"
        "  return byTag(node, 'li').map((li) => [byTag(li, 'strong')[0].textContent, byTag(li, 'p')[0].textContent]);"
        "})()"
    )
    assert timeline == [["Backtest failed", "Failed · Openrouter · Openai/Gpt-5.5 · Platform Credits · Provider Timeout"]]
    assert _eval("window.AdminUsers.renderActivityItems('timeline', []).textContent") == "No activity in this section."


def test_list_load_uses_the_declared_query_names_and_paints_rows():
    result = _eval(
        "(async () => {"
        "  const body = register('usersBody', document.createElement('tbody'));"
        "  register('usersRange', document.createElement('span'));"
        "  register('usersPrev', document.createElement('button'));"
        "  register('usersNext', document.createElement('button'));"
        "  register('usersView', panelStub().panel);"
        "  window.AdminShell.state.filters = {group: 'invited', segment: '', tier: '', internal: false, q: 'ada', priority: false};"
        f"  fetchQueue.push({{ok: true, status: 200, body: {USERS}}});"
        "  await window.AdminUsers.loadList({offset: 0});"
        "  return {url: fetchCalls[0][0], rows: body.children.length, range: document.getElementById('usersRange').textContent, prev: document.getElementById('usersPrev').disabled, next: document.getElementById('usersNext').disabled};"
        "})()"
    )
    assert result == {
        "url": "/api/admin/analytics/users?q=ada&user_group=invited&include_internal=false&limit=50&offset=0",
        "rows": 2,
        "range": "Showing 1–2 of 2",
        "prev": True,
        "next": True,
    }


def test_profile_sections_fetch_once_and_page_with_the_cursor():
    result = _eval(
        "(async () => {"
        "  const profile = register('profile', document.createElement('section'));"
        f"  fetchQueue.push({{ok: true, status: 200, body: {PROFILE}}});"
        "  await window.AdminUsers.openProfile('101');"
        "  const afterOpen = fetchCalls.map(([url]) => url);"
        f"  fetchQueue.push({{ok: true, status: 200, body: {TIMELINE}}});"
        "  await window.AdminUsers.selectSection('timeline');"
        "  await window.AdminUsers.selectSection('timeline');"
        "  const afterTimeline = fetchCalls.map(([url]) => url);"
        f"  fetchQueue.push({{ok: true, status: 200, body: {{items: [], next_cursor: null}}}});"
        "  await window.AdminUsers.loadSection('timeline', {append: true});"
        "  return {afterOpen, afterTimeline, afterMore: fetchCalls.map(([url]) => url), items: window.AdminUsers.state.profile.sections.timeline.items.length, cursor: window.AdminUsers.state.profile.sections.timeline.nextCursor};"
        "})()"
    )
    assert len(result["afterOpen"]) == 1
    assert result["afterOpen"][0].startswith("/api/admin/analytics/users/101?from=")
    assert "&to=" in result["afterOpen"][0]
    assert result["afterTimeline"][1] == "/api/admin/analytics/users/101/activity?section=timeline&limit=50"
    assert len(result["afterTimeline"]) == 2
    assert result["afterMore"][2] == "/api/admin/analytics/users/101/activity?section=timeline&limit=50&cursor=synthetic-timeline-cursor"
    assert result["items"] == 1
    assert result["cursor"] is None
