"""Every response carries anti-framing and anti-sniffing headers."""

from fastapi.testclient import TestClient

from dashboard.backend.app import app


def test_app_page_refuses_framing_and_sniffing():
    resp = TestClient(app).get("/health")
    assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
