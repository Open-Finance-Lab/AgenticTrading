"""Shared cookie-session helpers for auth-related API tests."""

def _cookie_session_token(client) -> str:
    """Raw session token from the HttpOnly cookie set by login/signup."""
    from dashboard.backend.auth_cookies import session_cookie_name
    token = client.cookies.get(session_cookie_name())
    assert token, f"missing session cookie {session_cookie_name()!r} in {dict(client.cookies)}"
    return token

