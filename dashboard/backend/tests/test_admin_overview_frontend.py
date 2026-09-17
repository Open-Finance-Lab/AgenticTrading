"""js/admin-overview.js under node: every panel renderer against the committed fixtures."""

from dashboard.backend.tests._admin_dom_stub import fixture, requires_node, run_node, source, target_fixture

pytestmark = requires_node

SHELL = source("admin-shell.js")
CREDIT_FORMAT = source("credit-format.js")
OVERVIEW = source("admin-overview.js")
# The four payloads that gain §9 fields come from the target-shape copies (Task 1);
# the absent-field test below deletes those keys again. PR D switches these to `fixture`.
TARGET = ("overview", "overview_partial_error", "operational", "commercial")
F = {name: (target_fixture if name in TARGET else fixture)(f"{name}.json") for name in (
    "overview", "overview_partial_error", "operational", "lifecycle", "retention", "commercial", "groups",
)}


def _eval(expression: str, *setup: str) -> object:
    # Wrapped in Promise.resolve(...).then(...) rather than a bare console.log:
    # an async IIFE's top-level `await` does not propagate outward from a nested
    # expression under `node -e`, so `console.log(JSON.stringify(expression))`
    # would stringify the still-pending Promise (`{}`) instead of its resolved
    # value. Harmless for synchronous expressions, so applied unconditionally.
    return run_node(
        SHELL, CREDIT_FORMAT, OVERVIEW, *setup,
        f"Promise.resolve({expression}).then((result) => console.log(JSON.stringify(result)));",
    )


def test_active_users_bars_and_axis_follow_the_daily_series():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderActiveUsers({F['overview']});"
        "  return {headline: r.headline, bars: texts(byTag(r.body, 'b')), labels: texts(byClass(r.body, 'bar-column').map((c) => byTag(c, 'span')[0])), axis: texts(byClass(r.body, 'chart-y-axis')[0].children), heights: byClass(r.body, 'bar-column').map((c) => byTag(c, 'i')[0].style['--height'])};"
        "})()"
    )
    assert result == {
        "headline": "42",
        "bars": ["31", "36"],
        "labels": ["Aug 25", "Aug 26"],
        "axis": ["36", "24", "12", "0"],
        "heights": ["65.44444444444444%", "76%"],
    }


def test_activation_progress_computes_percent_remaining_from_the_counts():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderActivation({F['overview']});"
        "  return {headline: r.headline, names: texts(byClass(r.body, 'layer-name')), values: texts(byClass(r.body, 'layer-value')), metas: texts(byClass(r.body, 'layer-meta'))};"
        "})()"
    )
    assert result == {
        "headline": "62.5%",
        "names": ["Account created", "Credential verified", "Agent created", "First successful result"],
        "values": ["80", "67", "59", "50"],
        "metas": ["100% of cohort", "83.8% remain", "73.8% remain", "62.5% remain"],
    }


def test_sources_donut_and_legend_cover_the_six_groups():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderSources({F['groups']});"
        "  const donut = byClass(r.body, 'source-donut')[0];"
        "  return {headline: r.headline, total: donut.getAttribute('data-total'), gradient: donut.style.background.startsWith('conic-gradient('), labels: texts(byClass(r.body, 'legend-row').map((row) => byTag(row, 'span')[0])), shares: texts(byClass(r.body, 'legend-row').map((row) => byTag(row, 'b')[0]))};"
        "})()"
    )
    assert result == {
        "headline": "100",
        "total": "100\nusers",
        "gradient": True,
        "labels": ["Internal", "Invited", "Organic", "Competition", "Partner", "Unknown"],
        "shares": ["22%", "18%", "22%", "12%", "18%", "8%"],
    }


def test_retention_heatmap_leaves_immature_cells_empty():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderRetention({F['retention']});"
        "  return {headline: r.headline, rows: texts(byClass(r.body, 'row-label')), cells: byClass(r.body, 'retention-cell').map((c) => [c.textContent, c.classList.contains('empty')])};"
        "})()"
    )
    assert result == {
        "headline": "66.7%",
        "rows": ["Jul 6", "Aug 24"],
        "cells": [["3", False], ["66.7%", False], ["33.3%", False], ["33.3%", False], ["2", False], ["—", True], ["—", True], ["—", True]],
    }


def test_reaching_value_sums_the_groups():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderValue({F['groups']});"
        "  return {headline: r.headline, steps: texts(byClass(r.body, 'value-step').map((s) => byTag(s, 'b')[0]))};"
        "})()"
    )
    assert result == {"headline": "52", "steps": ["48 · 48%", "32 · 32%", "20 · 20%"]}


def test_lifecycle_bar_and_legend_follow_segment_counts():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderLifecycle({F['lifecycle']});"
        "  return {headline: r.headline, shares: byClass(r.body, 'lifecycle-bar')[0].children.map((i) => i.style['--share']), legend: texts(byClass(r.body, 'lifecycle-legend')[0].children.map((d) => byTag(d, 'b')[0]))};"
        "})()"
    )
    assert result == {"headline": "33", "shares": ["3", "6", "7", "8", "5", "4"], "legend": ["3", "6", "7", "8", "5", "4"]}


def test_credits_pairs_platform_and_byok_per_day():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderCredits({F['commercial']}, {F['overview']});"
        "  return {headline: r.headline, stems: texts(byClass(r.body, 'credit-stem').map((s) => byTag(s, 'b')[0])), days: texts(byClass(r.body, 'credit-date')), legend: texts(byClass(r.body, 'credit-legend')[0].children)};"
        "})()"
    )
    assert result == {
        "headline": "4.800000 Credits",
        "stems": ["11", "7", "14", "8"],
        "days": ["Aug 25", "Aug 26"],
        "legend": ["Platform Credits", "BYOK runs"],
    }


def test_revenue_line_is_generated_svg_with_fixed_axis_labels():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderRevenue({F['commercial']});"
        "  const svg = byTag(r.body, 'svg')[0];"
        "  return {headline: r.headline, label: svg.getAttribute('aria-label'), texts: texts(byTag(svg, 'text')), titles: texts(byTag(svg, 'title')), points: byTag(svg, 'circle').length};"
        "})()"
    )
    assert result == {
        "headline": "12.000000 Credits",
        "label": "Purchased Credits revenue trend with 2 data points",
        "texts": ["7", "3.5", "0", "5", "7", "Sep 1", "Sep 2"],
        "titles": ["Sep 1: 5.0 purchased Credits", "Sep 2: 7.0 purchased Credits"],
        "points": 2,
    }


def test_attention_counts_and_top_reason():
    result = _eval(
        "(() => {"
        f"  const r = window.AdminOverview.renderAttention({F['operational']}, {F['lifecycle']});"
        "  return {headline: r.headline, counts: texts(byClass(r.body, 'attention-row').map((row) => byTag(row, 'b')[0])), note: byClass(r.body, 'attention-note')[0].children[0].children[1].textContent, link: byTag(byClass(r.body, 'attention-note')[0], 'a')[0].getAttribute('href')};"
        "})()"
    )
    assert result == {"headline": "11", "counts": ["2", "4", "5"], "note": "No Usable Billing Lane · 2 users", "link": "#health"}


def test_recut_fields_absent_render_awaiting_data_source_not_an_empty_chart():
    """Interim contract: a §9 field the route does not serve yet is absent, which is not the same as empty."""
    result = _eval(
        "(() => {"
        f"  const overview = {F['overview']}; delete overview.billing_lane_mix;"
        f"  const commercial = {F['commercial']}; delete commercial.purchased_by_day;"
        f"  const operational = {F['operational']}; delete operational.top_operational_reasons;"
        "  const credits = window.AdminOverview.renderCredits(commercial, overview);"
        "  const revenue = window.AdminOverview.renderRevenue(commercial);"
        f"  const attention = window.AdminOverview.renderAttention(operational, {F['lifecycle']});"
        "  const health = window.AdminOverview.detailHealth(operational);"
        "  const reasons = byTag(health, 'table')[1];"
        "  const empty = window.AdminOverview.renderRevenue(Object.assign({}, commercial, {purchased_by_day: []}));"
        "  return {"
        "    credits: [credits.headline, texts(byClass(credits.body, 'panel-empty'))],"
        "    revenue: [revenue.headline, texts(byClass(revenue.body, 'panel-empty'))],"
        "    attention: [attention.headline, texts(byClass(attention.body, 'attention-row').map((row) => byTag(row, 'b')[0])), byClass(attention.body, 'attention-note')[0].children[0].children[1].textContent],"
        "    health: byTag(reasons, 'tbody')[0].children.map((tr) => texts(tr.children)),"
        "    empty: texts(byClass(empty.body, 'panel-empty')),"
        "  };"
        "})()"
    )
    assert result["credits"] == ["4.800000 Credits", ["Awaiting data source"]]
    assert result["revenue"] == ["12.000000 Credits", ["Awaiting data source"]]
    assert result["attention"] == ["11", ["2", "4", "5"], "Awaiting data source"]
    assert result["health"] == [["Awaiting data source"]]
    # Served-and-empty keeps the panel's own copy: the two states must never collapse into one.
    assert result["empty"] == ["No settled purchases in this range."]


def test_health_detail_has_no_affected_users_column():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminOverview.detailHealth({F['operational']});"
        "  const tables = byTag(node, 'table');"
        "  return {h1: byTag(node, 'h1')[0].textContent, metrics: texts(byClass(node, 'detail-metric').map((m) => byTag(m, 'strong')[0])), headers: tables.map((t) => texts(byTag(t, 'th'))), rows: tables.map((t) => byTag(t, 'tbody')[0].children.map((tr) => texts(tr.children)))};"
        "})()"
    )
    assert result["h1"] == "System health"
    assert result["metrics"] == ["10", "80%", "40", "2"]
    assert result["headers"] == [["Failure category", "Count"], ["Reason", "State", "Users"]]
    assert result["rows"] == [
        [["Credential Invalid", "4"], ["Provider Timeout", "2"]],
        [["No Usable Billing Lane", "Blocked", "2"], ["Credential Invalid", "Needs attention", "4"]],
    ]
    assert "Affected users" not in OVERVIEW


def test_sources_detail_lists_six_groups_with_activation_per_group():
    result = _eval(
        "(() => {"
        f"  const node = window.AdminOverview.detailSources({F['groups']});"
        "  const table = byTag(node, 'table')[0];"
        "  return {headers: texts(byTag(table, 'th')), rows: byTag(table, 'tbody')[0].children.map((tr) => texts(tr.children))};"
        "})()"
    )
    assert result["headers"] == ["Source", "Users", "Share", "Successful run users", "Repeat users", "Total runs", "ATL cost", "Paid users", "Activation"]
    assert result["rows"][0] == ["Internal", "22", "22%", "14", "8", "120", "$8.40", "3", "63.6%"]
    assert len(result["rows"]) == 6


def test_paint_marks_partial_availability_as_incomplete_and_never_blanks_a_sibling():
    result = _eval(
        "(async () => {"
        "  const stubs = {};"
        "  window.AdminOverview.PANELS.forEach((def) => { stubs[def.id] = panelStub(); register(def.id, stubs[def.id].panel); });"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['overview_partial_error']}}});"  # overview
        "  fetchQueue.push({ok: false, status: 503, body: {}});"  # lifecycle
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['retention']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['commercial']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['operational']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['groups']}}});"
        "  await window.AdminOverview.loadAll(['overview', 'lifecycle', 'retention', 'commercial', 'operational', 'groups']);"
        "  const read = (id) => ({headline: stubs[id].parts.headline.textContent, status: stubs[id].parts.status.textContent, error: stubs[id].parts.errorText.textContent, body: stubs[id].parts.body.children.length});"
        "  return {activeUsers: read('panelActiveUsers'), attention: read('panelAttention'), lifecycle: read('panelLifecycle'), sources: read('panelSources'), calls: fetchCalls.map(([url]) => url.split('?')[0])};"
        "})()"
    )
    assert result["calls"] == [
        "/api/admin/analytics/overview", "/api/admin/analytics/lifecycle", "/api/admin/analytics/retention",
        "/api/admin/analytics/commercial", "/api/admin/analytics/operational", "/api/admin/analytics/groups",
    ]
    # The partial overview still paints (its funnel and headline are present) and says "Incomplete data".
    assert result["activeUsers"]["headline"] == "42"
    assert result["activeUsers"]["status"] == "Incomplete data"
    # Lifecycle failed: the two panels that need it show the error; the others are untouched.
    assert result["lifecycle"] == {"headline": "—", "status": "", "error": "This section is temporarily unavailable.", "body": 0}
    assert result["attention"]["error"] == "This section is temporarily unavailable."
    assert result["sources"] == {"headline": "100", "status": "", "error": "", "body": 1}


def test_one_failing_renderer_does_not_strand_the_panels_behind_it():
    """A bare `PANELS.forEach(paint)` over an unguarded paint() meant any throw
    aborted the loop, leaving every later panel at aria-busy="true" with no body
    and no error -- a spinner that never resolves. The reachable case was
    formatCredits dereferencing a missing window.CreditFormat: renderCredits is
    PANELS index 7, so it stranded panelRevenue behind it."""
    result = _eval(
        "(async () => {"
        "  const stubs = {};"
        "  window.AdminOverview.PANELS.forEach((def) => { stubs[def.id] = panelStub(); register(def.id, stubs[def.id].panel); });"
        "  const credits = window.AdminOverview.PANELS.find((def) => def.id === 'panelCredits');"
        "  credits.render = () => { throw new TypeError('renderer blew up'); };"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['overview']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['lifecycle']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['retention']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['commercial']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['operational']}}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['groups']}}});"
        "  await window.AdminOverview.loadAll(['overview', 'lifecycle', 'retention', 'commercial', 'operational', 'groups']);"
        "  const read = (id) => ({headline: stubs[id].parts.headline.textContent, error: stubs[id].parts.errorText.textContent, busy: stubs[id].panel.getAttribute('aria-busy')});"
        "  return {credits: read('panelCredits'), revenue: read('panelRevenue'), attention: read('panelAttention')};"
        "})()"
    )
    # The panel whose renderer threw reports the failure it actually had...
    assert result["credits"] == {"headline": "—", "error": "This section is temporarily unavailable.", "busy": "false"}
    # ...and the panel *after* it in PANELS still paints, rather than spinning forever.
    assert result["revenue"] == {"headline": "12.000000 Credits", "error": "", "busy": "false"}
    assert result["attention"]["headline"] == "11"


def test_show_detail_clears_the_previous_route_before_it_fetches():
    """route() reveals #detail the moment the hash changes. Clearing only after
    Promise.allSettled resolved left the *previous* detail page -- breadcrumb, h1,
    metric strip, tables -- fully painted under the new route's subnav highlight,
    with no busy state, for the whole request."""
    result = _eval(
        "(async () => {"
        "  const detail = document.createElement('div');"
        "  register('detail', detail);"
        f"  fetchQueue.push({{ok: true, status: 200, body: {F['groups']}}});"
        "  await window.AdminOverview.showDetail('sources');"
        "  const first = {h1: texts(byTag(detail, 'h1')), busy: detail.getAttribute('aria-busy')};"
        "  let release;"
        "  const held = new Promise((resolve) => { release = resolve; });"
        "  globalThis.fetch = () => held;"
        "  const inFlight = window.AdminOverview.showDetail('retention');"
        "  await new Promise((resolve) => setTimeout(resolve, 0));"
        "  const during = {h1: texts(byTag(detail, 'h1')), busy: detail.getAttribute('aria-busy'), text: detail.textContent};"
        f"  release({{ok: true, status: 200, json: () => Promise.resolve({F['retention']})}});"
        "  await inFlight;"
        "  const after = {h1: texts(byTag(detail, 'h1')), busy: detail.getAttribute('aria-busy')};"
        "  return {first, during, after};"
        "})()"
    )
    assert result["first"]["h1"] == ["User sources"]
    # Mid-flight: the old page is gone and the wait is announced, not disguised.
    assert result["during"]["h1"] == []
    assert result["during"]["busy"] == "true"
    assert "Loading" in result["during"]["text"]
    assert result["after"]["h1"] == ["Retention"]
    assert result["after"]["busy"] == "false"


def test_paint_clears_a_stale_headline_and_body_when_its_data_goes_missing():
    """F1 regression: a route/range change wipes `state.data` for a panel whose
    fetch then fails. The very next paint() must not leave the previous range's
    headline and chart on screen at full brightness with only an amber notice --
    verified consequence: 1W's "42" survived a 1M failure with no stale dimming."""
    result = _eval(
        "(() => {"
        "  const {panel, parts} = panelStub();"
        "  register('panelAttention', panel);"
        "  const def = window.AdminOverview.PANELS.find((d) => d.id === 'panelAttention');"
        f"  window.AdminOverview.state.data = {{operational: {F['operational']}, lifecycle: {F['lifecycle']}}};"
        "  window.AdminOverview.state.errors = {};"
        "  window.AdminOverview.paint(def);"
        "  const before = {headline: parts.headline.textContent, body: parts.body.children.length};"
        "  window.AdminOverview.state.data = {};"  # the signature change that drops cached data
        "  window.AdminOverview.paint(def);"
        "  const after = {headline: parts.headline.textContent, body: parts.body.children.length, error: parts.errorText.textContent, stale: panel.classList.contains('is-stale')};"
        "  return {before, after};"
        "})()"
    )
    assert result["before"]["headline"] == "11"
    assert result["before"]["body"] > 0
    assert result["after"] == {"headline": "—", "body": 0, "error": "This section is temporarily unavailable.", "stale": False}
