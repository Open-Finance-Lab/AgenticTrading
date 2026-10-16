"""Admin trace list and timeline renderers under the shared DOM stub."""

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
TRACES = source("admin-traces.js")


def _eval(expression: str) -> object:
    return run_node(
        SHELL,
        source("backtest-comparison.js"),
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
    assert "#1Run Started" in result["events"][0]
    assert "<safe>" in result["pre"][0]
    assert "#2Decision Recorded" in result["events"][1]


def test_empty_trace_states_are_explicit():
    assert _eval("window.AdminTraces.renderTraceRows({items: []}).children[0].textContent") == "No traces yet."
    assert _eval("window.AdminTraces.renderEventTimeline([]).children[0].textContent") == "No events recorded yet."


def test_performance_filters_invalid_snapshots_and_uses_market_time():
    result = _eval("window.AdminTracePerformance.series([{event_type:'decision_recorded', occurred_at:'2026-10-09', payload:{decision_at:'2026-05-04T10:00:00Z',state:{equity:100}}}, {event_type:'decision_recorded',payload:{decision_at:'bad',state:{equity:99}}}], null)")
    assert result['snapshots'] is True
    assert len(result['points']) == 1
    assert result['points'][0]['timestamp'] == '2026-05-04T10:00:00Z'


def test_failed_performance_renders_partial_and_marks_trades_by_decision_id():
    """Without the run's chart payload the curve falls back to decision
    snapshots: one agent column, its metrics from those points, and a marker
    only on the decision whose execution produced an order."""
    result = _eval("(() => { const n = window.AdminTracePerformance.render({status:'failed'}, [{sequence_no:1,event_type:'decision_recorded',decision_id:'d',payload:{decision_at:'2026-05-04T10:00:00Z',state:{equity:100}}},{sequence_no:2,event_type:'decision_recorded',payload:{decision_at:'2026-05-04T11:00:00Z',state:{equity:90}}},{sequence_no:3,event_type:'execution_result',decision_id:'d',payload:{fills:[{side:'BUY',symbol:'AAPL'}]}}],null); return {text:n.textContent, circles:byTag(n,'circle').length, legend:byClass(n,'trace-legend-btn').map(b=>b.textContent), cells:byTag(n,'td').map(c=>c.textContent)}; })()")
    assert 'Run failed' in result['text']
    assert result['circles'] == 1
    assert result['legend'] == ['Your Agent']
    assert result['cells'] == ['$90', '-10.00%', '-10.00%']


def test_chart_and_comparison_use_the_backtest_model_when_the_run_has_benchmarks():
    result = _eval("(() => {"
        "  const chart = {available: true, agent_run_id: 'r', timestamps: ['2026-05-04T14:30:00', '2026-05-04T15:30:00', '2026-05-05T14:30:00'],"
        "    series: [{run_id: 'r', label: 'Agent', values: [1000, 1010, 1020], color: '#4FC3F7', dashed: false},"
        "             {run_id: 'index:^DJI', label: 'DJIA index', values: [1000, 1005, 1030], color: '#F5C04A', dashed: true}], run: {}};"
        "  const n = window.AdminTracePerformance.render({status:'completed'}, [], {points: []}, chart);"
        "  return {legend: byClass(n,'trace-legend-btn').map(b=>b.textContent), lines: byTag(n,'polyline').length,"
        "    best: byClass(n,'trace-best').length, deltas: byClass(n,'trace-delta').map(d=>d.textContent), head: byTag(n,'th').map(t=>t.textContent)};"
        "})()")
    assert result['legend'] == ['Your Agent', 'DJIA']
    assert result['lines'] == 2
    assert result['deltas'] == ['DJIA -1.00pp']
    # Sharpe is the engine's to compute (2026-10-14 direction), never this table's.
    assert 'Sharpe Ratio' not in result['head']


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
    assert "#3Run Completed" in result["text"]


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


_TWO_ORDERS = (
    "  const events = ["
    "    {sequence_no: 1, event_type: 'decision_recorded', decision_id: 'd1', payload: {decision_at: '2026-05-04T14:00:00Z',"
    "      actions: [{symbol: 'AAPL', action: 'BUY', shares: 10, reason: 'momentum'}, {symbol: 'MSFT', action: 'SELL', shares: 5, reason: 'trim'}],"
    "      reasoning_summaries: ['momentum', 'trim']}},"
    "    {sequence_no: 2, event_type: 'execution_result', decision_id: 'd1', payload: {fills: ["
    "      {timestamp: '2026-05-04T14:00:00Z', symbol: 'MSFT', side: 'SELL', shares: 5, price: 300}],"
    "      rejected: [{symbol: 'AAPL', side: 'BUY', requested_shares: 10, reason: 'insufficient_cash'}]}},"
    "    {sequence_no: 3, event_type: 'decision_recorded', decision_id: 'd2', payload: {decision_at: '2026-05-04T15:00:00Z',"
    "      actions: [{symbol: 'NVDA', action: 'BUY'}, {symbol: 'AMZN', action: 'BUY'}], reasoning_summaries: ['equal weight', 'equal weight']}},"
    "    {sequence_no: 4, event_type: 'execution_result', decision_id: 'd2', payload: {fills: ["
    "      {symbol: 'NVDA', side: 'BUY', shares: 1, price: 237.05}, {symbol: 'AMZN', side: 'BUY', shares: 1, price: 251.58}]}},"
    "  ];"
)


def test_trading_log_groups_orders_by_decision_with_each_reason_once():
    """An order takes the reason of the action on its symbol and side; a reason
    every order in the group shares prints once, above them."""
    result = _eval(
        "(() => {" + _TWO_ORDERS +
        "  const log = window.AdminTraces.renderTradingLog(events);"
        "  return {summary: byClass(log, 'trace-log-summary')[0].textContent,"
        "    groups: byClass(log, 'trace-log-group').map((g) => ({id: g.getAttribute('id'),"
        "      lines: byClass(g, 'trace-log-line').map((l) => l.textContent),"
        "      reasons: byClass(g, 'trace-log-reason').map((r) => r.textContent),"
        "      rejects: byClass(g, 'trace-log-reject').map((r) => r.textContent)}))};"
        "})()"
    )
    assert result["summary"] == "4 orders · 3 filled · 1 rejected"
    newest, oldest = result["groups"]
    assert newest["id"] == "trace-orders-d2"
    assert newest["lines"] == ["BUYNVDA1 × $237.05Filled", "BUYAMZN1 × $251.58Filled"]
    assert newest["reasons"] == ["equal weight"]
    assert oldest["lines"] == ["SELLMSFT5 × $300.00Filled", "BUYAAPL10 shRejected"]
    assert oldest["reasons"] == ["trim", "momentum"]
    assert oldest["rejects"] == ["Rejected: Insufficient Cash"]


def test_timeline_rows_describe_the_process_and_never_the_orders():
    """The Trading log owns symbols and reasoning; a trade row says how the step
    ran, and its body links to the orders instead of repeating them."""
    result = _eval(
        "(() => {" + _TWO_ORDERS +
        "  let shown = null;"
        "  const list = window.AdminTraceTimeline.render(events, {}, {onShowOrders: (id) => { shown = id; }});"
        "  const rows = byClass(list, 'trace-row');"
        "  byClass(rows[0], 'trace-row-orders')[0].listeners.click[0]();"
        "  const visible = rows.map((row) => byClass(row, 'trace-row-summary')[0].textContent).join(' | ');"
        "  return {visible, shown, kinds: byClass(list, 'trace-kind').map((k) => k.textContent),"
        "    eventLines: byClass(list, 'trace-event-line').map((l) => l.textContent).join(' | ')};"
        "})()"
    )
    assert result["kinds"] == ["Trade", "Trade"]
    assert "2 orders (1 filled, 1 rejected)" in result["visible"]
    for leaked in ("AAPL", "MSFT", "NVDA", "momentum", "equal weight"):
        assert leaked not in result["visible"], leaked
        assert leaked not in result["eventLines"], leaked
    assert result["shown"] == "d1"


def test_quiet_steps_fold_and_the_errors_filter_keeps_a_failed_trade():
    result = _eval(
        "(() => {"
        "  const hold = (n) => [{sequence_no: n, event_type: 'decision_recorded', decision_id: `h${n}`, payload: {decision_at: '2026-05-04T14:00:00Z'}},"
        "                       {sequence_no: n + 1, event_type: 'execution_result', decision_id: `h${n}`, payload: {fills: []}}];"
        "  const events = [...hold(1), ...hold(3), ...hold(5),"
        "    {sequence_no: 7, event_type: 'tool_result', step_id: 's', decision_id: 't', payload: {tool_name: 'news_digest', outcome: 'error', error_code: 'provider_timeout'}},"
        "    {sequence_no: 8, event_type: 'decision_recorded', step_id: 's', decision_id: 't', payload: {decision_at: '2026-05-04T17:00:00Z'}},"
        "    {sequence_no: 9, event_type: 'execution_result', decision_id: 't', payload: {fills: [{side: 'SELL', symbol: 'X', shares: 1, price: 1}]}}];"
        "  const kinds = (list) => byClass(list, 'trace-kind').map((k) => k.textContent);"
        "  return {all: kinds(window.AdminTraceTimeline.render(events)), errors: kinds(window.AdminTraceTimeline.render(events, {}, {filter: 'errors'})),"
        "    flag: byClass(window.AdminTraceTimeline.render(events), 'trace-row-flag').map((f) => f.textContent)};"
        "})()"
    )
    assert result["all"] == ["Hold", "Trade"]
    assert result["errors"] == ["Trade"]
    assert result["flag"] == ["news_digest failed"]


def test_back_follows_the_entry_point_and_escape_uses_it():
    result = _eval(
        "(() => {"
        "  const href = (ctx) => byClass(window.AdminTraces.renderTraceDetail({trace_id: 't', status: 'completed'}, [], ctx), 'trace-back')[0];"
        "  const fromUser = href({user_id: 42});"
        "  return {list: href({}).getAttribute('href'), user: fromUser.getAttribute('href'), label: fromUser.textContent};"
        "})()"
    )
    assert result == {"list": "#traces", "user": "#users/42", "label": "← User analytics"}


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
