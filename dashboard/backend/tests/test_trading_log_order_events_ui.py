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
    depth = 0
    index = source.index("{", start)
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


def test_trading_log_markup_uses_order_language_and_eight_columns():
    html = _APP_HTML.read_text(encoding="utf-8")

    assert "All Orders" in html
    assert "All Trades" not in html
    assert "<th>Status</th>" in html
    assert "<th>Reason</th>" in html
    assert 'colspan="8"' in html


def test_order_events_take_priority_and_legacy_trades_fall_back_as_filled():
    source = _APP_JS.read_text(encoding="utf-8")
    result = _run_node([
        _extract_function(source, "resolveTradingLogRecords"),
        _extract_function(source, "normalizeOrderRecord"),
        "const rejected = { symbol: '600519.SH', side: 'BUY',",
        "  requested_shares: 100.5, executed_shares: 0, unfilled_shares: 100.5,",
        "  price: 250, executed_value: 0, status: 'rejected', reason: 'invalid_lot_size' };",
        "const trade = { symbol: 'AAPL', side: 'BUY', quantity: 2, price: 100, value: 200 };",
        "const preferred = resolveTradingLogRecords({ order_events: [rejected], trades: [trade] });",
        "const fallback = resolveTradingLogRecords({ trades: [trade] });",
        "const emptyEventsFallback = resolveTradingLogRecords({ order_events: [], trades: [trade] });",
        "console.log(JSON.stringify({",
        "  preferred: normalizeOrderRecord(preferred[0]),",
        "  fallback: normalizeOrderRecord(fallback[0]),",
        "  emptyEventsFallback: normalizeOrderRecord(emptyEventsFallback[0]),",
        "}));",
    ])

    assert result["preferred"]["symbol"] == "600519.SH"
    assert result["preferred"]["requestedShares"] == 100.5
    assert result["preferred"]["executedShares"] == 0
    assert result["preferred"]["value"] == 0
    assert result["preferred"]["status"] == "rejected"
    assert result["fallback"]["symbol"] == "AAPL"
    assert result["fallback"]["requestedShares"] == 2
    assert result["fallback"]["executedShares"] == 2
    assert result["fallback"]["status"] == "filled"
    assert result["emptyEventsFallback"] == result["fallback"]


def _render_harness(source: str):
    return [
        "const tbody = { innerHTML: '' };",
        "const document = { getElementById: () => tbody };",
        "const IFIND_ASHARE_UNIVERSES = { demo: { assets: [",
        "  { symbol: '600519.SH', name: 'Kweichow Moutai' },",
        "] } };",
        "const POPULAR_STOCKS = { AAPL: 'Apple Inc.' };",
        "let tradingLogCache = [];",
        "let tradingLogFilter = 'all';",
        "let tradingLogEmptyMessage = 'No orders yet.';",
        _extract_function(source, "escapeHtml"),
        _extract_function(source, "resolveTradingAssetName"),
        _extract_function(source, "formatOrderExecutionReason"),
        _extract_function(source, "normalizeOrderRecord"),
        _extract_function(source, "formatTradeTimestamp"),
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
    assert ">$250.00<" in known
    assert ">--<" in known
    assert "<img" not in result["unknown"]
    assert "Order not executed" in result["unknown"]


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
