"""The Robinhood MCP client over the real SDK transport, against a fake server.

``test_robinhood_mcp_tool_text.py`` covers result parsing only; nothing else
exercised the transport, so an ``mcp`` bump could change how the bearer token
travels with the suite still green. These pin the properties that matter on a
real-money path: the token reaches the endpoint, no tool call (an order
included) is ever sent off the endpoint's origin by a redirect, and a tool
result is validated without fetching anything the server names.
"""

from __future__ import annotations

import asyncio
import functools
import json

import httpx
import pytest

from dashboard.backend.infrastructure.brokers import robinhood_mcp

TOKEN = "test-access-token"
OTHER_ORIGIN = "https://elsewhere.example/mcp/trading"


class FakeMCPServer:
    """Answers the JSON-RPC a ClientSession sends for initialize + one tool call."""

    def __init__(self, *, tools=None, call_result=None, redirect=None):
        self.requests: list[httpx.Request] = []
        self.tools = tools if tools is not None else []
        self.call_result = call_result or {
            "content": [{"type": "text", "text": '{"buying_power": "1250.50"}'}],
            "isError": False,
        }
        self.redirect = redirect  # (status, location) for the first POST to MCP_URL

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if str(request.url) == robinhood_mcp.MCP_URL and self.redirect is not None:
            status, location = self.redirect
            return httpx.Response(status, headers={"location": location})
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        method = message["method"]
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-robinhood", "version": "0"},
            }
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            result = self.call_result
        else:
            return httpx.Response(400)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})


@pytest.fixture
def serve(monkeypatch):
    real_client = httpx.AsyncClient

    def _serve(server: FakeMCPServer) -> FakeMCPServer:
        # On httpx itself, not a name in robinhood_mcp, so this holds whichever
        # side builds the client (ours, or the SDK's own factory).
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            functools.partial(real_client, transport=httpx.MockTransport(server.handler)),
        )
        return server

    return _serve


def _get_portfolio():
    return asyncio.run(robinhood_mcp.RobinhoodMCPClient(TOKEN).get_portfolio())


def test_tool_call_sends_the_bearer_token_and_parses_the_reply(serve):
    server = serve(FakeMCPServer())

    assert _get_portfolio() == {"buying_power": "1250.50"}
    posts = [r for r in server.requests if r.method == "POST"]
    assert posts and all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in posts)


def test_same_origin_redirect_is_followed(serve):
    server = serve(FakeMCPServer(redirect=(307, robinhood_mcp.MCP_URL + "/")))

    assert _get_portfolio() == {"buying_power": "1250.50"}
    assert any(str(r.url) == robinhood_mcp.MCP_URL + "/" for r in server.requests)


@pytest.mark.parametrize("status", [302, 307, 308])
def test_cross_origin_redirect_is_refused(serve, status):
    # httpx already strips Authorization off-origin; what is at stake is the
    # tool call itself, which mcp<1.30 re-sent to wherever the redirect pointed.
    server = serve(FakeMCPServer(redirect=(status, OTHER_ORIGIN)))

    with pytest.raises(ExceptionGroup) as excinfo:
        _get_portfolio()
    assert excinfo.group_contains(httpx.HTTPStatusError, match="not followed")
    assert {r.url.host for r in server.requests} == {httpx.URL(robinhood_mcp.MCP_URL).host}


def test_output_schema_ref_is_never_fetched(serve):
    # mcp<1.30 validated with jsonschema's default registry, which retrieves an
    # unknown $ref over the network: the server chose a URL and we fetched it.
    serve(
        FakeMCPServer(
            tools=[
                {
                    "name": "get_portfolio",
                    "inputSchema": {"type": "object"},
                    "outputSchema": {"$ref": "https://elsewhere.example/portfolio.json"},
                }
            ],
            call_result={
                "content": [{"type": "text", "text": "{}"}],
                "structuredContent": {"buying_power": "1"},
                "isError": False,
            },
        )
    )

    # Unresolvable is what proves no retrieval happened: the ref is left unresolved.
    with pytest.raises(ExceptionGroup) as excinfo:
        _get_portfolio()
    assert excinfo.group_contains(RuntimeError, match="Unresolvable: https://elsewhere.example/")
