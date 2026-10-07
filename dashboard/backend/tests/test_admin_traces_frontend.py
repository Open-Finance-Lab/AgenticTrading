"""Admin trace list and timeline renderers under the shared DOM stub."""

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source

pytestmark = requires_node

SHELL = source("admin-shell.js")
TRACES = source("admin-traces.js")


def _eval(expression: str) -> object:
    return run_node(
        SHELL,
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
    assert result["events"][0].startswith("#1Run Started")
    assert "<safe>" in result["pre"][0]
    assert result["events"][1].startswith("#2Decision Recorded")


def test_empty_trace_states_are_explicit():
    assert _eval("window.AdminTraces.renderTraceRows({items: []}).children[0].textContent") == "No traces yet."
    assert _eval("window.AdminTraces.renderEventTimeline([]).children[0].textContent") == "No events recorded yet."
