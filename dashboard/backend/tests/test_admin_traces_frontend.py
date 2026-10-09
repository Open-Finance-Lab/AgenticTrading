"""Admin trace list and timeline renderers under the shared DOM stub."""

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
TRACES = source("admin-traces.js")


def _eval(expression: str) -> object:
    return run_node(
        SHELL,
        source("admin-trace-performance.js"),
        source("admin-trace-timeline.js"),
        TRACES,
        f"Promise.resolve({expression}).then((result) => console.log(JSON.stringify(result)));",
    )


def test_trace_rows_render_status_and_missing_values_without_html_interpretation():
    result = _eval(
        "(() => {"
        "  const node = window.AdminTraces.renderTraceRows({items: ["
        "    {trace_id: '<trace-1>', agent_id: null, run_id: 'run-1', status: 'running', started_at: '2026-10-07T09:00:00Z'},"
        "  ]});"
        "  return {texts: node.children[0].children.map((cell) => cell.textContent), href: byTag(node, 'a')[0].getAttribute('href'), pre: byTag(node, 'a')[0].textContent};"
        "})()"
    )
    assert result == {
        "texts": ["<trace-1>", "—", "run-1", "Running", "Oct 7, 2026, 09:00 UTC", "—"],
        "href": "#traces/%3Ctrace-1%3E",
        "pre": "<trace-1>",
    }


def test_timeline_keeps_order_and_serializes_payload_as_text():
    result = _eval(
        "(() => {"
        "  const node = window.AdminTraces.renderEventTimeline(["
        "    {sequence_no: 1, event_type: 'run_started', actor_type: 'system', occurred_at: '2026-10-07T09:00:00Z', payload: {message: '<safe>'}},"
        "    {sequence_no: 2, event_type: 'decision_recorded', actor_type: 'agent', decision_id: 'd-1', occurred_at: '2026-10-07T09:01:00Z', payload: {accepted: true}},"
        "  ]);"
        "  return {events: node.children.map((item) => item.textContent), pre: byTag(node, 'pre').map((item) => item.textContent)};"
        "})()"
    )
    assert "#1 Run Started" in result["events"][0]
    assert "<safe>" in result["pre"][0]
    assert "#2 Decision Recorded" in result["events"][1]


def test_empty_trace_states_are_explicit():
    assert _eval("window.AdminTraces.renderTraceRows({items: []}).children[0].textContent") == "No traces yet."
    assert _eval("window.AdminTraces.renderEventTimeline([]).children[0].textContent") == "No events recorded yet."


def test_performance_filters_invalid_snapshots_and_uses_market_time():
    result = _eval("window.AdminTracePerformance.series([{event_type:'decision_recorded', occurred_at:'2026-10-09', payload:{decision_at:'2026-05-04T10:00:00Z',state:{equity:100}}}, {event_type:'decision_recorded',payload:{decision_at:'bad',state:{equity:99}}}], null)")
    assert result['snapshots'] is True
    assert len(result['points']) == 1
    assert result['points'][0]['timestamp'] == '2026-05-04T10:00:00Z'


def test_failed_performance_renders_partial_and_links_by_decision_id():
    result = _eval("(() => { const n = window.AdminTracePerformance.render({status:'failed'}, [{sequence_no:1,event_type:'decision_recorded',decision_id:'d',payload:{decision_at:'2026-05-04T10:00:00Z',state:{equity:100}}},{sequence_no:2,event_type:'decision_recorded',payload:{decision_at:'2026-05-04T11:00:00Z',state:{equity:90}}},{sequence_no:3,event_type:'execution_result',decision_id:'d',payload:{fills:[{side:'BUY',symbol:'AAPL'}]}}],null); return {text:n.textContent, circles:byTag(n,'circle').length, buttons:byTag(n,'button').map(b=>b.textContent)}; })()")
    assert 'Run failed' in result['text']
    assert 'Snapshot change: -10.00%' in result['text']
    assert result['circles'] == 1
    assert result['buttons'][1] == 'Execution'


def test_groups_only_by_explicit_links_and_keeps_event_order():
    result = _eval("window.AdminTraceTimeline.groupEvents([{sequence_no:1,event_type:'data_retrieval',step_id:'s'},{sequence_no:2,event_type:'decision_recorded',step_id:'s',decision_id:'d'},{sequence_no:3,event_type:'execution_result',decision_id:'d'},{sequence_no:4,event_type:'run_failed'}]).map(g=>g.events.map(e=>e.sequence_no))")
    assert result == [[1, 2, 3], [4]]


def test_failure_opens_by_default_and_explicit_collapse_is_preserved():
    result = _eval("(() => {const events=[{sequence_no:1,event_type:'run_failed',payload:{error_code:'timeout'}}];return [window.AdminTraceTimeline.render(events).children[0].children[0].open,window.AdminTraceTimeline.render(events,{'event:1':false}).children[0].children[0].open];})()")
    assert result == [True, False]
