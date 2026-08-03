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
    # Caches must not serve a gzipped body to a client that cannot decode it.
    assert "accept-encoding" in response.headers.get("vary", "").lower()


def test_small_response_is_not_gzipped():
    # /health is a handful of bytes: below minimum_size the gzip header would
    # cost more than it saves, so the response must go out uncompressed.
    client = TestClient(app)
    response = client.get("/health", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers.get("content-encoding") is None


def test_compression_level_is_not_starlette_default():
    # Starlette compresses inline on the event loop, so level 9 costs ~4x the
    # stall of level 6 for ~0.7pp better ratio. Pinned because the default is 9
    # and reverting to it silently reintroduces the stall this PR removed.
    gzip_layer = next(
        m for m in app.user_middleware if m.cls.__name__ == "GZipMiddleware"
    )
    assert gzip_layer.kwargs["compresslevel"] == 6
