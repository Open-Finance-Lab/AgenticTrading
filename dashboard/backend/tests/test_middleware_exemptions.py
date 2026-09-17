"""The session middleware must let the admin shell and its redirect through.

`SessionMiddleware` demands `X-Session-Id` on every path that is not exempt, an
API route, or a paper-trading route. A page load sends no such header, so a
page path missing from `EXEMPT_PATHS` answers a 400 JSON body instead of HTML
(design §7.1). Nothing pinned `is_exempt('/admin-analytics')` before this file
(§4.6), which is how the live page depended on one string nobody tested.
"""

from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.middleware import EXEMPT_PATHS, is_exempt


def test_admin_shell_and_its_redirect_are_exempt():
    assert "/admin" in EXEMPT_PATHS
    assert is_exempt("/admin") is True
    # Kept until the redirect route itself is removed: an exact-match exemption
    # dropped early would 400 the redirect before it fires.
    assert "/admin-analytics" in EXEMPT_PATHS
    assert is_exempt("/admin-analytics") is True
    # The stylesheet rides the extension rule, not the set.
    assert is_exempt("/admin.css") is True
    # Exact match only: the API surface under /admin/ keeps its own rules.
    assert is_exempt("/admin/runs/1") is False


def test_admin_page_loads_without_a_session_header():
    with TestClient(app) as client:
        response = client.get("/admin")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>ATL Admin</title>" in response.text


def test_admin_css_is_served():
    with TestClient(app) as client:
        response = client.get("/admin.css")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")


def test_admin_analytics_redirects_to_admin_preserving_the_query():
    with TestClient(app) as client:
        bare = client.get("/admin-analytics", follow_redirects=False)
        with_query = client.get("/admin-analytics?range=1M&group=organic", follow_redirects=False)
    assert bare.status_code == 308
    assert bare.headers["location"] == "/admin"
    assert with_query.status_code == 308
    assert with_query.headers["location"] == "/admin?range=1M&group=organic"
