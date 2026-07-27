"""Robinhood Agentic Trading MCP client."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from dashboard.backend.infrastructure.brokers import robinhood_oauth

logger = logging.getLogger(__name__)

MCP_URL = robinhood_oauth.MCP_URL
MCP_TIMEOUT_SECONDS = float(os.getenv("ROBINHOOD_MCP_TIMEOUT_SECONDS", "12"))


def _tool_text(result: Any) -> Any:
    if result is None:
        return None
    if hasattr(result, "structuredContent") and result.structuredContent is not None:
        return result.structuredContent
    content = getattr(result, "content", None) or []
    texts: List[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    if not texts:
        return None
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return joined


class RobinhoodMCPClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self._headers = {"Authorization": f"Bearer {access_token}"}

    async def _call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        arguments = arguments or {}

        async def _run() -> Any:
            async with streamablehttp_client(MCP_URL, headers=self._headers) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    return _tool_text(result)

        return await asyncio.wait_for(_run(), timeout=MCP_TIMEOUT_SECONDS)

    async def get_accounts(self) -> Any:
        return await self._call_tool("get_accounts")

    async def get_portfolio(self) -> Any:
        return await self._call_tool("get_portfolio")

    async def get_equity_positions(self) -> Any:
        return await self._call_tool("get_equity_positions")

    async def get_equity_quotes(self, symbols: List[str]) -> Any:
        return await self._call_tool("get_equity_quotes", {"symbols": symbols[:20]})

    async def get_equity_orders(self) -> Any:
        return await self._call_tool("get_equity_orders")

    async def review_equity_order(self, payload: Dict[str, Any]) -> Any:
        return await self._call_tool("review_equity_order", payload)

    async def place_equity_order(self, payload: Dict[str, Any]) -> Any:
        return await self._call_tool("place_equity_order", payload)

    async def cancel_equity_order(self, order_id: str) -> Any:
        return await self._call_tool("cancel_equity_order", {"order_id": order_id})


def extract_buying_power(portfolio: Any) -> Optional[float]:
    if not isinstance(portfolio, dict):
        return None
    for key in ("buying_power", "equity_buying_power", "cash_available_for_withdrawal"):
        val = portfolio.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    nested = portfolio.get("portfolio") or portfolio.get("account")
    if isinstance(nested, dict):
        return extract_buying_power(nested)
    return None


def extract_portfolio_value(portfolio: Any) -> Optional[float]:
    if not isinstance(portfolio, dict):
        return None
    for key in ("portfolio_value", "total_value", "equity"):
        val = portfolio.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    nested = portfolio.get("portfolio") or portfolio.get("account")
    if isinstance(nested, dict):
        return extract_portfolio_value(nested)
    return None
