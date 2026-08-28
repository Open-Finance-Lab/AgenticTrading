"""Trading Log order-outcome behavior against the shipped vanilla JavaScript."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_APP_JS = _ROOT / "frontend" / "app.js"
_APP_HTML = _ROOT / "frontend" / "app.html"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def _extract_function(source: str, name: str) -> str:
    for marker in (f"async function {name}(", f"function {name}("):
        start = source.find(marker)
        if start != -1:
            break
    else:
        raise AssertionError(f"{name} not found in {_APP_JS.name}")
    paren_depth = 0
    paren_index = source.index("(", start)
    while paren_index < len(source):
        if source[paren_index] == "(":
            paren_depth += 1
        elif source[paren_index] == ")":
            paren_depth -= 1
            if paren_depth == 0:
                break
        paren_index += 1
    depth = 0
    index = source.index("{", paren_index)
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
        index += 1
    raise AssertionError(f"unterminated function {name}")


def _run_node(lines):
    result = subprocess.run(
        ["node", "-e", "\n".join(lines)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_trading_log_markup_is_an_accessible_right_rail_feed():
    html = _APP_HTML.read_text(encoding="utf-8")

    assert "All Orders" in html
    assert "All Trades" not in html
    assert 'id="tradingLogFeed"' in html
    assert 'role="list"' in html
    assert 'tabindex="0"' in html
    assert 'aria-label="Trading Log orders"' in html
    assert 'id="tradingLogCount"' in html
    assert 'id="tradingLogStatusSummary"' in html
    assert 'class="trading-log-table"' not in html
    center_end = html.index('</section>', html.index('class="center-panel"'))
    log_at = html.index('id="tradingLogFeed"')
    right_at = html.index('class="right-panel"')
    assert right_at < log_at
    assert log_at > center_end


def _merge_harness(source: str):
    return [
        _extract_function(source, "orderEventMatchKey"),
        _extract_function(source, "resolveTradingLogRecords"),
        _extract_function(source, "resolveTradingLogTruncation"),
        _extract_function(source, "normalizeOrderRecord"),
    ]


def test_rejections_are_added_to_the_complete_trade_list_not_substituted_for_it():
    """The two server lists are complementary, so neither may shadow the other.

    ``trades`` is every fill and is uncapped; ``order_events`` is only what did
    not fill and is capped. Preferring one wholesale hides real rows either
    way -- and preferring the capped one silently truncated the log.
    """
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_merge_harness(source) + [
        "const rejected = { timestamp: '2026-04-01T11:00:00Z', symbol: '600519.SH',",
        "  side: 'BUY', requested_shares: 100.5, executed_shares: 0,",
        "  unfilled_shares: 100.5, price: 250, executed_value: 0,",
        "  status: 'rejected', reason: 'invalid_lot_size' };",
        "const trade = { timestamp: '2026-04-01T10:00:00Z', symbol: 'AAPL',",
        "  side: 'BUY', quantity: 2, price: 100, value: 200 };",
        "const merged = resolveTradingLogRecords({ order_events: [rejected], trades: [trade] });",
        "const tradesOnly = resolveTradingLogRecords({ trades: [trade] });",
        "const emptyEvents = resolveTradingLogRecords({ order_events: [], trades: [trade] });",
        "console.log(JSON.stringify({",
        "  merged: merged.map(normalizeOrderRecord),",
        "  tradesOnly: tradesOnly.map(normalizeOrderRecord),",
        "  emptyEvents: emptyEvents.map(normalizeOrderRecord),",
        "}));",
    ])

    merged = result["merged"]
    assert len(merged) == 2, "the fill must survive alongside the rejection"
    # Sorted by timestamp, so the 10:00 fill precedes the 11:00 rejection.
    assert merged[0]["symbol"] == "AAPL"
    assert merged[0]["status"] == "filled"
    assert merged[0]["executedShares"] == 2
    assert merged[1]["symbol"] == "600519.SH"
    assert merged[1]["status"] == "rejected"
    assert merged[1]["requestedShares"] == 100.5
    assert merged[1]["executedShares"] == 0
    assert merged[1]["value"] == 0
    assert result["tradesOnly"] == [merged[0]]
    assert result["emptyEvents"] == [merged[0]]


def test_partial_fill_replaces_its_trade_row_instead_of_duplicating_it():
    """A partial appears in both lists; the event is the superset, so it wins."""
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_merge_harness(source) + [
        "const trade = { timestamp: '2026-04-01T10:00:00Z', symbol: '600519.SH',",
        "  side: 'SELL', shares: 100, price: 20, proceeds: 2000 };",
        "const partial = { timestamp: '2026-04-01T10:00:00Z', symbol: '600519.SH',",
        "  side: 'SELL', requested_shares: 200, executed_shares: 100,",
        "  unfilled_shares: 100, price: 20, executed_value: 2000,",
        "  status: 'partial', reason: 't1_frozen' };",
        "const merged = resolveTradingLogRecords({ order_events: [partial], trades: [trade] });",
        "console.log(JSON.stringify(merged.map(normalizeOrderRecord)));",
    ])

    assert len(result) == 1, "one order must not render as two rows"
    assert result[0]["status"] == "partial"
    assert result[0]["requestedShares"] == 200
    assert result[0]["executedShares"] == 100
    assert result[0]["reason"] == "t1_frozen"


def test_truncation_is_derived_from_either_signal():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_merge_harness(source) + [
        "const events = [{ status: 'rejected', executed_shares: 0 }];",
        "console.log(JSON.stringify({",
        "  explicit: resolveTradingLogTruncation({ order_events: events, order_events_truncated: 47 }),",
        "  derivedLive: resolveTradingLogTruncation({ order_events: events, order_events_count: 12 }),",
        "  derivedRun: resolveTradingLogTruncation({ order_events: events, order_event_count: 9 }),",
        "  none: resolveTradingLogTruncation({ order_events: events, order_events_count: 1 }),",
        "  legacy: resolveTradingLogTruncation({ trades: [] }),",
        "}));",
    ])

    assert result["explicit"] == 47
    assert result["derivedLive"] == 11
    assert result["derivedRun"] == 8
    assert result["none"] == 0
    assert result["legacy"] == 0


def _render_harness(source: str):
    return [
        "const tbody = { innerHTML: '' };",
        "const tradingLogCount = { textContent: '' };",
        "const tradingLogStatusSummary = { textContent: '' };",
        "const document = { getElementById: (id) => ({",
        "  tradingLogFeed: tbody, tradingLogCount, tradingLogStatusSummary,",
        "}[id] || tbody) };",
        "const IFIND_ASHARE_UNIVERSES = { demo: { assets: [",
        "  { symbol: '600519.SH', name: 'Kweichow Moutai' },",
        "] } };",
        "const POPULAR_STOCKS = { AAPL: 'Apple Inc.' };",
        "let tradingLogCache = [];",
        "let tradingLogFilter = 'all';",
        "let tradingLogEmptyMessage = 'No orders yet.';",
        "let tradingLogTruncatedCount = 0;",
        _extract_function(source, "escapeHtml"),
        _extract_function(source, "resolveTradingAssetName"),
        _extract_function(source, "formatOrderExecutionReason"),
        _extract_function(source, "normalizeOrderRecord"),
        _extract_function(source, "formatTradingMoney"),
        _extract_function(source, "renderOrderCostAudit"),
        _extract_function(source, "renderMarketRuleAudit"),
        _extract_function(source, "formatTradeTimestamp"),
        _extract_function(source, "renderTradingLogSummary"),
        _extract_function(source, "paintTradingLog"),
        _extract_function(source, "renderTradingLog"),
    ]


def test_rendered_rejection_has_safe_reason_zero_fill_and_english_company():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([{",
        "  timestamp: '2026-04-01T10:00:00+08:00', symbol: '600519.SH', side: 'BUY',",
        "  requested_shares: 50, executed_shares: 0, unfilled_shares: 50,",
        "  price: 250, executed_value: 0, status: 'rejected', reason: 'invalid_lot_size',",
        "}]);",
        "const known = tbody.innerHTML;",
        "renderTradingLog([{ symbol: '600519.SH', side: 'BUY', requested_shares: 100,",
        "  executed_shares: 0, price: 250, executed_value: 0, status: 'rejected',",
        "  reason: '<img src=x onerror=alert(1)>' }]);",
        "console.log(JSON.stringify({ known, unknown: tbody.innerHTML }));",
    ])

    known = result["known"]
    assert "Kweichow Moutai" in known
    assert "0 / 50 shares" in known
    assert "REJECTED" in known
    assert "Invalid lot size" in known
    assert "$250.00" in known
    assert ">--<" in known
    assert "<img" not in result["unknown"]
    assert "Order not executed" in result["unknown"]


def test_rendered_market_rule_rejection_has_english_reason_and_native_close():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([{",
        "  timestamp: '2026-04-01T15:00:00+08:00', symbol: '600519.SH', side: 'BUY',",
        "  requested_shares: 100, executed_shares: 0, price: 200,",
        "  status: 'rejected', reason: 'limit_up_buy_blocked',",
        "  market_rule_date: '2026-04-01', market_rule_suspended: false,",
        "  market_rule_closing_limit_state: 'upper',",
        "  market_rule_official_close: 1400,",
        "  market_rule_closing_gate_effective: true,",
        "}]);",
        "console.log(JSON.stringify(tbody.innerHTML));",
    ])

    assert "Buy blocked at upper limit" in result
    assert "Official close: upper limit" in result
    assert "¥1400.00" in result
    assert "2026-04-01" in result


def test_rendered_partial_uses_actual_value_and_side_filter():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "const rows = [",
        " { symbol: 'AAPL', side: 'BUY', requested_shares: 100, executed_shares: 100,",
        "   price: 10, executed_value: 1000, status: 'filled', reason: '' },",
        " { symbol: '600519.SH', side: 'SELL', requested_shares: 200, executed_shares: 100,",
        "   price: 20, executed_value: 2000, status: 'partial', reason: 't1_frozen' },",
        "];",
        "tradingLogFilter = 'sell';",
        "renderTradingLog(rows);",
        "console.log(JSON.stringify(tbody.innerHTML));",
    ])

    assert "PARTIAL" in result
    assert "100 / 200 shares" in result
    assert "$2,000.00" in result
    assert "T+1 frozen" in result
    assert "Apple Inc." not in result


def test_rendering_updates_order_count_and_status_summary():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([",
        " { symbol: 'AAPL', side: 'BUY', requested_shares: 1, executed_shares: 1,",
        "   price: 10, executed_value: 10, status: 'filled', reason: '' },",
        " { symbol: 'MSFT', side: 'BUY', requested_shares: 2, executed_shares: 1,",
        "   price: 20, executed_value: 20, status: 'partial', reason: '' },",
        " { symbol: '600519.SH', side: 'SELL', requested_shares: 100, executed_shares: 0,",
        "   price: 200, executed_value: 0, status: 'rejected', reason: 'invalid_lot_size' },",
        "]);",
        "console.log(JSON.stringify({ count: tradingLogCount.textContent, summary: tradingLogStatusSummary.textContent }));",
    ])

    assert result == {"count": "3 orders", "summary": "1 filled · 1 partial · 1 rejected"}


def test_capped_sample_says_so_instead_of_ending_early():
    """A truncated log that looks complete is the failure mode, not the cap."""
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([{ timestamp: '2026-04-01T10:00:00Z', symbol: 'AAPL',",
        "  side: 'BUY', requested_shares: 5, executed_shares: 0, price: 10,",
        "  executed_value: 0, status: 'rejected', reason: 'insufficient_cash' }],",
        "  { truncatedCount: 312 });",
        "const shown = tbody.innerHTML;",
        "renderTradingLog([{ timestamp: '2026-04-01T10:00:00Z', symbol: 'AAPL',",
        "  side: 'BUY', requested_shares: 5, executed_shares: 5, price: 10,",
        "  executed_value: 50, status: 'filled', reason: '' }]);",
        "console.log(JSON.stringify({ shown, complete: tbody.innerHTML }));",
    ])

    assert "312 more unfilled orders are not shown" in result["shown"]
    assert "Insufficient cash" in result["shown"]
    # No cap applied => no notice at all, so the row count is the whole story.
    assert "not shown" not in result["complete"]


def test_rendered_a_share_fill_shows_reporting_and_native_cost_breakdown():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([{ timestamp: '2026-04-01T10:00:00Z', symbol: '600519.SH',",
        "  side: 'BUY', requested_shares: 100, executed_shares: 100, price: 14.2929,",
        "  executed_value: 204.14, gross_value: 204.14, status: 'filled', reason: 'Momentum entry',",
        "  commission: 0.7143, stamp_duty: 0, transfer_fee: 0.0143,",
        "  slippage_amount: 0.7143, total_fees: 0.7286, net_cash_impact: -204.8686,",
        "  native_price: 100.05, native_value: 10005, native_commission: 5,",
        "  native_stamp_duty: 0, native_transfer_fee: 0.1,",
        "  native_slippage_amount: 5, native_total_fees: 5.1,",
        "  native_net_cash_impact: -10010.1, fx_rate: 7 }]);",
        "console.log(JSON.stringify(tbody.innerHTML));",
    ])

    assert "Commission $0.71" in result
    assert "Stamp duty $0.00" in result
    assert "Transfer fee $0.01" in result
    assert "Slippage $0.71" in result
    assert "Net cash -$204.87" in result
    assert "CNY native" in result
    assert "Commission ¥5.00" in result
    assert "Net cash -¥10,010.10" in result
    assert "Momentum entry" in result
    assert "Order not executed" not in result


def test_rejected_a_share_order_does_not_render_zero_costs_as_a_charge():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([{ symbol: '600519.SH', side: 'BUY', requested_shares: 50,",
        "  executed_shares: 0, price: 100, executed_value: 0, status: 'rejected',",
        "  commission: 0, stamp_duty: 0, transfer_fee: 0, slippage_amount: 0,",
        "  total_fees: 0, net_cash_impact: 0, reason: 'invalid_lot_size' }]);",
        "console.log(JSON.stringify(tbody.innerHTML));",
    ])

    assert "Commission" not in result
    assert "Net cash" not in result
    assert "Invalid lot size" in result


def test_filtering_preserves_quantities_and_the_truncation_notice():
    """Re-rendering must not push normalized rows back through the normalizer.

    ``tradingLogCache`` holds normalized records (``requestedShares``), not wire
    records (``requested_shares``). Feeding it to the normalizer a second time
    reads keys that are not there and zeroes every quantity -- so the log would
    silently blank out the moment a user touched the filter.
    """
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([{ timestamp: '2026-04-01T10:00:00Z', symbol: 'AAPL',",
        "  side: 'BUY', requested_shares: 200, executed_shares: 150, price: 10,",
        "  executed_value: 1500, status: 'partial', reason: 'insufficient_cash' }],",
        "  { truncatedCount: 4 });",
        "tradingLogFilter = 'buy';",
        "paintTradingLog(tradingLogCache, {",
        "  emptyMessage: tradingLogEmptyMessage,",
        "  truncatedCount: tradingLogTruncatedCount,",
        "});",
        "console.log(JSON.stringify(tbody.innerHTML));",
    ])

    assert "150 / 200 shares" in result
    assert "$1,500.00" in result
    assert "4 more unfilled orders are not shown" in result


def test_repeated_rejection_reports_how_many_times_it_fired():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([{ timestamp: '2026-04-01T10:00:00Z', symbol: '600519.SH',",
        "  side: 'BUY', requested_shares: 100, executed_shares: 0, price: 250,",
        "  executed_value: 0, status: 'rejected',",
        "  reason: 'insufficient_cash_for_lot', repeat_count: 47 }]);",
        "console.log(JSON.stringify(tbody.innerHTML));",
    ])

    assert "Insufficient cash for one lot" in result
    assert "×47 that day" in result


def test_ordinary_a_share_fill_carries_no_market_rule_audit_line():
    """The audit line is for rows a rule spoke to, not every A-share order.

    Every order on a rule-aware run carries the same audit payload, so keying
    the line on "an official close is present" prints a date-and-price row under
    every fill and buries the handful that were actually suspended or gated.
    """
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node(_render_harness(source) + [
        "renderTradingLog([{",
        "  timestamp: '2026-04-01T15:00:00+08:00', symbol: '600519.SH', side: 'BUY',",
        "  requested_shares: 100, executed_shares: 100, price: 13,",
        "  status: 'filled', reason: '',",
        "  market_rule_date: '2026-04-01', market_rule_suspended: false,",
        "  market_rule_closing_limit_state: 'none',",
        "  market_rule_official_close: 13,",
        "  market_rule_closing_gate_effective: false,",
        "}]);",
        "console.log(JSON.stringify(tbody.innerHTML));",
    ])

    assert "¥13.00" not in result
    assert "Official close" not in result
    assert "Official status" not in result


def test_late_trading_log_response_cannot_repaint_a_newer_run():
    source = _APP_JS.read_text(encoding="utf-8")
    function_source = _extract_function(source, "loadTradingLogForRun")
    body = function_source[function_source.index("{") :]
    assert body.count("if (!isCurrent()) return") >= 2
    result = _run_node([
        "const API_BASE = '';",
        "const paints = [];",
        "const API = { get: async () => ({ trades: [{ symbol: 'AAPL' }] }) };",
        "const resolveTradingLogRecords = (payload) => payload.trades;",
        "const resolveTradingLogTruncation = () => 0;",
        "const renderTradingLog = () => paints.push('render');",
        "const clearTradingLog = () => paints.push('clear');",
        function_source,
        "(async () => {",
        "  await loadTradingLogForRun('run-a', { isCurrent: () => false });",
        "  console.log(JSON.stringify(paints));",
        "})();",
    ])
    assert result == []
