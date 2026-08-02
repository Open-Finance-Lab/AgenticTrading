"""Responses must be gzip-compressed when the client accepts it.

The Vercel frontend fetches multi-hundred-KB JSON payloads (equity curves,
agent lists) from the Render backend; without compression they ship raw.
"""

import uuid

from fastapi.testclient import TestClient

from dashboard.backend.app import app


def test_large_json_response_is_gzipped_when_accepted():
    client = TestClient(app)
    # /openapi.json is not in SessionMiddleware's exempt set, so it needs a
    # session header like any backtest route; without it the request 400s
    # before ever reaching a handler.
    response = client.get(
        "/openapi.json",
        headers={"Accept-Encoding": "gzip", "X-Session-Id": str(uuid.uuid4())},
    )
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"
