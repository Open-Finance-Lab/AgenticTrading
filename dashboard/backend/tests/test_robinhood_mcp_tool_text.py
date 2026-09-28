"""An MCP tool error must reach the live-run gates as an error, not as text."""

from __future__ import annotations

from types import SimpleNamespace

from dashboard.backend.execution import robinhood_live_service as live_service
from dashboard.backend.infrastructure.brokers.robinhood_mcp import _tool_text


def _result(text: str, *, is_error: bool) -> SimpleNamespace:
    return SimpleNamespace(
        structuredContent=None,
        content=[SimpleNamespace(text=text)],
        isError=is_error,
    )


def test_tool_error_text_becomes_an_error_payload():
    assert _tool_text(_result("insufficient buying power", is_error=True)) == {
        "error": "insufficient buying power"
    }


def test_tool_error_blocks_the_pre_trade_review():
    review = _tool_text(_result("Error: order not allowed", is_error=True))
    assert live_service._review_blocks_order(review) == (True, "review_rejected")


def test_tool_error_on_placement_maps_to_rejected():
    placed = _tool_text(_result("market closed", is_error=True))
    assert live_service._execution_status(placed) == "rejected"


def test_successful_text_reply_is_unchanged():
    assert _tool_text(_result('{"approved": true}', is_error=False)) == {"approved": True}
    assert _tool_text(_result("review ok", is_error=False)) == "review ok"
