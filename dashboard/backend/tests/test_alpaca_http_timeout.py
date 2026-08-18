"""Tests for the default HTTP timeout applied to the Alpaca client's session.

alpaca-py 0.43.2 calls ``self._session.request(method, url, **opts)`` with no
``timeout`` in ``opts``, so a stalled socket blocks ``requests`` forever and
permanently leaks a threadpool thread (binds at concurrency >= 1). These tests
exercise ``_apply_default_timeout`` directly against plain fake objects -- no
real ``AlpacaDataLoader``, no alpaca-py import, no network. See
``test_alpaca_bars.py``'s ``_FakeClient`` for why the no-``_session`` case
matters: that fixture has no ``_session`` attribute, so this function must
tolerate it without raising.
"""

from dashboard.backend.infrastructure.market_data.alpaca_bars import (
    ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS,
    ALPACA_HTTP_TIMEOUT_SECONDS,
    _apply_default_timeout,
)


class _FakeSession:
    """Records every call made through ``request``."""

    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return kwargs


class _FakeClient:
    def __init__(self, session):
        self._session = session


def test_default_timeout_applied_when_absent():
    session = _FakeSession()
    client = _FakeClient(session)

    _apply_default_timeout(client)
    client._session.request("GET", "https://data.alpaca.markets/v2/stocks/bars")

    assert len(session.calls) == 1
    _, _, kwargs = session.calls[0]
    assert kwargs["timeout"] == (
        ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS,
        ALPACA_HTTP_TIMEOUT_SECONDS,
    )


def test_caller_supplied_timeout_is_not_overridden():
    session = _FakeSession()
    client = _FakeClient(session)

    _apply_default_timeout(client)
    client._session.request(
        "GET", "https://data.alpaca.markets/v2/stocks/bars", timeout=(1, 2)
    )

    assert len(session.calls) == 1
    _, _, kwargs = session.calls[0]
    assert kwargs["timeout"] == (1, 2)


def test_applying_twice_does_not_double_wrap():
    session = _FakeSession()
    client = _FakeClient(session)

    _apply_default_timeout(client)
    # Guard attribute must be set on the wrapped request so a second call is a
    # no-op rather than wrapping the wrapper.
    assert getattr(client._session.request, "_atl_default_timeout_applied", False) is True

    # Capture the bound callable *after* the first apply, before any call is
    # made. len(session.calls) and the recorded timeout are both
    # depth-invariant -- stacked *args/**kwargs wrappers still bottom out in
    # exactly one call to the fake, so neither can distinguish one wrapper
    # from three. Identity of session.request is the assertion that actually
    # fails if the early-return guard is removed.
    wrapped = client._session.request
    _apply_default_timeout(client)
    assert client._session.request is wrapped

    client._session.request("GET", "https://data.alpaca.markets/v2/stocks/bars")

    assert len(session.calls) == 1
    _, _, kwargs = session.calls[0]
    assert kwargs["timeout"] == (
        ALPACA_HTTP_CONNECT_TIMEOUT_SECONDS,
        ALPACA_HTTP_TIMEOUT_SECONDS,
    )


class _NoSessionClient:
    """Mirrors the existing test_alpaca_bars.py _FakeClient: no _session at all."""


def test_missing_session_warns_and_does_not_raise(capsys):
    client = _NoSessionClient()

    _apply_default_timeout(client)  # must not raise

    captured = capsys.readouterr()
    assert "_session" in captured.out
