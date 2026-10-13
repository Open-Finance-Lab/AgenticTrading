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


# Drives loadDetail through the route event, then one poll, against the stub's
# ordered fetch queue. `settle` lets the awaited request chain drain.
_ROUTE_THEN_POLL = """
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
const host = register('tracesView', document.createElement('div'));
const envelope = (status) => ({ok: true, status: 200, body: {trace_id: 't', run_id: 'r', status}});
const page = (items) => ({ok: true, status: 200, body: {items, has_more: false}});
const performance = {ok: true, status: 200, body: {points: [], point_count: 0, metrics: {}}};
async function openRunningTrace() {
  fetchQueue.push(envelope('running'), page([{sequence_no: 1, event_type: 'decision_recorded', decision_id: 'd'}]), performance);
  document.dispatchEvent(new CustomEvent('admin:route', {detail: {route: 'traces', id: 't'}}));
  for (let i = 0; i < 5; i += 1) await settle();
  window.AdminTraces.stopPolling();
}
"""


def test_poll_reads_the_envelope_before_events_so_a_finished_run_keeps_its_tail():
    """The store appends run_completed before it marks the trace completed. Reading
    events first and the envelope second let a trace finish between the two reads:
    the poll saw 'completed', stopped, and the last flush never loaded."""
    result = _eval(
        "(async () => {" + _ROUTE_THEN_POLL +
        "  await openRunningTrace();"
        "  const before = fetchCalls.length;"
        "  fetchQueue.push(envelope('completed'), page([{sequence_no: 2, event_type: 'execution_result', decision_id: 'd'}, {sequence_no: 3, event_type: 'run_completed'}]), performance);"
        "  await window.AdminTraces.pollEvents();"
        "  window.AdminTraces.stopPolling();"
        "  return {calls: fetchCalls.slice(before).map(([url]) => url), text: host.textContent};"
        "})()"
    )
    assert result["calls"][0] == "/api/admin/traces/t"
    assert result["calls"][1].startswith("/api/admin/traces/t/events?after_sequence=1")
    assert result["calls"][2] == "/api/admin/traces/t/performance"
    assert "#3 Run Completed" in result["text"]


def test_poll_without_changes_keeps_the_rendered_tree():
    result = _eval(
        "(async () => {" + _ROUTE_THEN_POLL +
        "  await openRunningTrace();"
        "  const first = host.children[0];"
        "  fetchQueue.push(envelope('running'), page([]), performance);"
        "  await window.AdminTraces.pollEvents();"
        "  const unchanged = host.children[0] === first;"
        "  fetchQueue.push(envelope('running'), page([{sequence_no: 2, event_type: 'execution_result', decision_id: 'd'}]), performance);"
        "  await window.AdminTraces.pollEvents();"
        "  window.AdminTraces.stopPolling();"
        "  return {unchanged, rerendered: host.children[0] !== first};"
        "})()"
    )
    assert result == {"unchanged": True, "rerendered": True}


def test_an_open_backtest_report_is_fetched_once_per_trace_status():
    result = _eval(
        "(async () => {"
        "  const report = {ok: true, status: 200, body: {available: true, run_id: 'r', run: {llm_model: 'deepseek-v4-pro'}, configuration: {}}};"
        "  async function openReport(trace) {"
        "    const panel = byClass(window.AdminTraces.renderTraceDetail(trace, []), 'trace-backtest-report')[0];"
        "    panel.open = true;"
        "    await panel.listeners.toggle[0]();"
        "    return panel;"
        "  }"
        "  fetchQueue.push(report);"
        "  await openReport({trace_id: 't', status: 'running'});"
        "  const rerendered = await openReport({trace_id: 't', status: 'running'});"
        "  const whileRunning = fetchCalls.filter(([url]) => url.endsWith('/backtest')).length;"
        "  fetchQueue.push(report);"
        "  await openReport({trace_id: 't', status: 'completed'});"
        "  return {whileRunning, afterCompletion: fetchCalls.filter(([url]) => url.endsWith('/backtest')).length, text: rerendered.textContent};"
        "})()"
    )
    assert result["whileRunning"] == 1
    assert result["afterCompletion"] == 2
    assert "Llm Model: deepseek-v4-pro" in result["text"]


def test_trading_log_takes_each_fills_reason_from_its_decision():
    """Tape fills carry no decision_id and decisions record reasoning_summaries
    (a list, one per action), so the log links through the execution event."""
    result = _eval(
        "(() => {"
        "  const events = ["
        "    {sequence_no: 1, event_type: 'decision_recorded', decision_id: 'd1', payload: {"
        "      actions: [{symbol: 'AAPL', action: 'BUY', shares: 10, reason: 'momentum'}, {symbol: 'MSFT', action: 'SELL', shares: 5, reason: 'trim'}],"
        "      reasoning_summaries: ['momentum', 'trim']}},"
        "    {sequence_no: 2, event_type: 'execution_result', decision_id: 'd1', payload: {fills: ["
        "      {timestamp: '2026-05-04T14:00:00Z', symbol: 'MSFT', side: 'SELL', shares: 5, price: 300},"
        "      {timestamp: '2026-05-04T14:00:00Z', symbol: 'AAPL', side: 'BUY', shares: 10, price: 150}]}},"
        "  ];"
        "  const log = byClass(window.AdminTraces.renderTraceDetail({trace_id: 't', status: 'completed'}, events), 'trace-trading-log')[0];"
        "  return byClass(log, 'trace-log-item').map((item) => ["
        "    byClass(item, 'trace-log-order')[0].textContent, (byClass(item, 'trace-reason')[0] || {}).textContent]);"
        "})()"
    )
    # Newest first.
    assert result == [["BUY AAPL", "momentum"], ["SELL MSFT", "trim"]]


def test_trading_log_shows_the_latest_fills_and_the_true_count():
    result = _eval(
        "(() => {"
        "  const fills = Array.from({length: 60}, (_, i) => ({symbol: `S${i}`, side: 'BUY', shares: 1, price: 1}));"
        "  const log = byClass(window.AdminTraces.renderTraceDetail({trace_id: 't', status: 'running'}, [{sequence_no: 1, event_type: 'execution_result', payload: {fills}}]), 'trace-trading-log')[0];"
        "  const items = byClass(log, 'trace-log-item');"
        "  return {head: byClass(log, 'eyebrow')[0].textContent, count: items.length, first: byClass(items[0], 'trace-log-order')[0].textContent};"
        "})()"
    )
    assert result == {"head": "Latest 50 of 60 fills", "count": 50, "first": "BUY S59"}


def test_trace_detail_tolerates_missing_trace_or_events():
    result = _eval(
        "(() => {"
        "  const empty = (node) => byClass(node, 'panel-empty').map((item) => item.textContent);"
        "  return [empty(window.AdminTraces.renderTraceDetail({trace_id: 't', status: 'running'}, null)),"
        "          empty(window.AdminTraces.renderTraceDetail(null, []))];"
        "})()"
    )
    for texts in result:
        assert "No events recorded yet." in texts
