"""Analytics request privacy-reduction tests."""

from datetime import datetime, timezone

from starlette.requests import Request

from dashboard.backend.domain.analytics.privacy import (
    monthly_network_hash,
    request_analytics_context,
)


def _request(headers=None, client=("203.0.113.19", 443)):
    raw_headers = [
        (key.lower().encode(), value.encode())
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/analytics/events",
            "headers": raw_headers,
            "client": client,
            "scheme": "https",
            "server": ("testserver", 443),
            "query_string": b"",
        }
    )


def test_missing_or_short_key_omits_network_hash(monkeypatch):
    received_at = datetime(2026, 8, 26, tzinfo=timezone.utc)
    monkeypatch.delenv("ANALYTICS_PSEUDONYMIZATION_KEY", raising=False)
    assert monthly_network_hash("203.0.113.19", received_at) is None

    monkeypatch.setenv("ANALYTICS_PSEUDONYMIZATION_KEY", "too-short")
    assert monthly_network_hash("203.0.113.19", received_at) is None


def test_network_hash_is_month_scoped_and_never_contains_plain_ip(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_PSEUDONYMIZATION_KEY",
        "synthetic-analytics-hmac-key-at-least-32-bytes",
    )
    august = monthly_network_hash(
        "203.0.113.19", datetime(2026, 8, 26, tzinfo=timezone.utc)
    )
    september = monthly_network_hash(
        "203.0.113.19", datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    assert august and september and august != september
    assert "203.0.113.19" not in august
    assert len(august) == 64


def test_request_context_reduces_user_agent_and_country(monkeypatch):
    monkeypatch.setenv(
        "ANALYTICS_PSEUDONYMIZATION_KEY",
        "synthetic-analytics-hmac-key-at-least-32-bytes",
    )
    monkeypatch.setenv("RENDER", "true")
    raw_agent = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
        "AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1 "
        "synthetic-secret-canary"
    )
    context = request_analytics_context(
        _request({"user-agent": raw_agent, "cf-ipcountry": "US"}),
        datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
    assert context.browser_family == "Safari"
    assert context.device_category == "mobile"
    assert context.country_code == "US"
    assert "synthetic-secret-canary" not in repr(context)
    assert "Mozilla" not in repr(context)
    assert "203.0.113.19" not in repr(context)


def test_untrusted_or_invalid_country_headers_are_omitted(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    context = request_analytics_context(
        _request({"cf-ipcountry": "US", "x-vercel-ip-country": "CA"}),
        datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
    assert context.country_code is None

    monkeypatch.setenv("VERCEL", "1")
    context = request_analytics_context(
        _request({"x-vercel-ip-country": "USA"}),
        datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
    assert context.country_code is None


def test_browser_and_device_reducers_are_allowlisted(monkeypatch):
    monkeypatch.delenv("ANALYTICS_PSEUDONYMIZATION_KEY", raising=False)
    cases = [
        ("Mozilla/5.0 Edg/126.0", "Edge", "desktop"),
        ("Mozilla/5.0 Android Mobile Chrome/126.0", "Chrome", "mobile"),
        ("Mozilla/5.0 iPad Version/18.0 Safari/604.1", "Safari", "tablet"),
        ("curl/8.0", "Other", "unknown"),
    ]
    for agent, browser, device in cases:
        context = request_analytics_context(
            _request({"user-agent": agent}),
            datetime(2026, 8, 26, tzinfo=timezone.utc),
        )
        assert context.browser_family == browser
        assert context.device_category == device
        assert context.network_hash is None
