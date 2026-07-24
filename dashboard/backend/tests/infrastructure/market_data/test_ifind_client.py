"""HTTP contract, retry, and secret-safety tests for the iFinD client."""

from __future__ import annotations

from datetime import date

import pytest
import requests

from dashboard.backend.infrastructure.market_data.profiles import (
    A_SHARE_DEMO_6_SYMBOLS,
)


START = date(2026, 4, 1)
END = date(2026, 4, 23)
TOKEN = "private-ifind-token"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, json_error=None):
        self.status_code = status_code
        self._payload = {"errorcode": 0, "tables": []} if payload is None else payload
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_client(session, **kwargs):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindHttpClient,
    )

    options = {
        "session": session,
        "token": TOKEN,
        "base_url": "https://ifind.test",
        "sleep": lambda _seconds: None,
    }
    options.update(kwargs)
    return IFindHttpClient(**options)


def test_builds_official_hourly_request_with_exclusive_end():
    session = FakeSession([FakeResponse()])

    result = make_client(session).fetch_hourly_bars(
        A_SHARE_DEMO_6_SYMBOLS,
        START,
        END,
    )

    assert result == {"errorcode": 0, "tables": []}
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == "https://ifind.test/api/v1/high_frequency"
    assert kwargs["headers"] == {
        "Content-Type": "application/json",
        "access_token": TOKEN,
        "ifindlang": "cn",
    }
    assert kwargs["timeout"] == (3.0, 20.0)
    assert kwargs["json"] == {
        "codes": ",".join(A_SHARE_DEMO_6_SYMBOLS),
        "indicators": "open,high,low,close,volume",
        "starttime": "2026-04-01 09:30:00",
        "endtime": "2026-04-22 15:00:00",
        "functionpara": {
            "Interval": "60",
            "CPS": "forward1",
            "Timeformat": "LocalTime",
            "Limitstart": "09:30:00",
            "Limitend": "15:00:00",
        },
    }
    assert "Fill" not in kwargs["json"]["functionpara"]


def test_uses_environment_defaults_without_exposing_token(monkeypatch):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindHttpClient,
    )

    session = FakeSession([FakeResponse()])
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", TOKEN)
    monkeypatch.setenv("IFIND_BASE_URL", "https://local-ifind.test/")

    IFindHttpClient(session=session).fetch_hourly_bars(["600519.SH"], START, END)

    url, kwargs = session.calls[0]
    assert url == "https://local-ifind.test/api/v1/high_frequency"
    assert kwargs["headers"]["access_token"] == TOKEN


def test_uses_official_base_url_when_no_override_is_configured(monkeypatch):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindHttpClient,
    )

    session = FakeSession([FakeResponse()])
    monkeypatch.delenv("IFIND_BASE_URL", raising=False)

    IFindHttpClient(session=session, token=TOKEN).fetch_hourly_bars(
        ["600519.SH"], START, END
    )

    url, _kwargs = session.calls[0]
    assert url == "https://quantapi.51ifind.com/api/v1/high_frequency"


def test_missing_token_fails_before_http_call(monkeypatch):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindConfigurationError,
        IFindHttpClient,
    )

    session = FakeSession([FakeResponse()])
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)

    with pytest.raises(IFindConfigurationError, match="IFIND_ACCESS_TOKEN"):
        IFindHttpClient(session=session)

    assert session.calls == []


@pytest.mark.parametrize(
    "start,end",
    [
        (END, START),
        (START, START),
    ],
)
def test_rejects_invalid_date_window_before_http_call(start, end):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindRequestError,
    )

    session = FakeSession([FakeResponse()])

    with pytest.raises(IFindRequestError, match="end must be after start"):
        make_client(session).fetch_hourly_bars(["600519.SH"], start, end)

    assert session.calls == []


def test_rejects_empty_symbols_before_http_call():
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindRequestError,
    )

    session = FakeSession([FakeResponse()])

    with pytest.raises(IFindRequestError, match="symbols"):
        make_client(session).fetch_hourly_bars([], START, END)

    assert session.calls == []


@pytest.mark.parametrize(
    "transport_error",
    [
        requests.ConnectionError("socket details must stay private"),
        requests.Timeout("timeout details must stay private"),
    ],
)
def test_retries_connection_failures_twice_then_succeeds(transport_error):
    sleeps = []
    session = FakeSession(
        [transport_error, transport_error, FakeResponse(payload={"errorcode": 0})]
    )

    result = make_client(session, sleep=sleeps.append).fetch_hourly_bars(
        ["600519.SH"], START, END
    )

    assert result == {"errorcode": 0}
    assert len(session.calls) == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retries_retryable_http_statuses_twice_then_succeeds(status_code):
    sleeps = []
    session = FakeSession(
        [
            FakeResponse(status_code=status_code),
            FakeResponse(status_code=status_code),
            FakeResponse(payload={"errorcode": 0}),
        ]
    )

    result = make_client(session, sleep=sleeps.append).fetch_hourly_bars(
        ["600519.SH"], START, END
    )

    assert result == {"errorcode": 0}
    assert len(session.calls) == 3
    assert sleeps == [0.5, 1.0]


def test_persistent_connection_failure_stops_after_three_attempts(caplog):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindTransportError,
    )

    raw_error = "raw-network-detail-must-not-leak"
    session = FakeSession([requests.ConnectionError(raw_error)] * 3)

    with pytest.raises(IFindTransportError) as exc_info:
        make_client(session).fetch_hourly_bars(["600519.SH"], START, END)

    assert len(session.calls) == 3
    combined = str(exc_info.value) + caplog.text
    assert TOKEN not in combined
    assert raw_error not in combined
    assert "symbols=1" in combined
    assert "start=2026-04-01" in combined
    assert "end=2026-04-23" in combined


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_non_retryable_http_errors_fail_once_without_response_leak(
    status_code, caplog
):
    from dashboard.backend.infrastructure.market_data.ifind_client import IFindHttpError

    upstream_secret = "upstream-body-secret"
    session = FakeSession(
        [
            FakeResponse(
                status_code=status_code,
                payload={"errorcode": -1, "errmsg": upstream_secret},
            )
        ]
    )

    with pytest.raises(IFindHttpError) as exc_info:
        make_client(session).fetch_hourly_bars(["600519.SH"], START, END)

    assert len(session.calls) == 1
    assert exc_info.value.status_code == status_code
    combined = str(exc_info.value) + caplog.text
    assert TOKEN not in combined
    assert upstream_secret not in combined


def test_business_error_fails_once_without_errmsg_leak(caplog):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindBusinessError,
    )

    upstream_secret = f"permission denied for {TOKEN}"
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "errorcode": -403,
                    "errmsg": upstream_secret,
                    "tables": [],
                }
            )
        ]
    )

    with pytest.raises(IFindBusinessError) as exc_info:
        make_client(session).fetch_hourly_bars(["600519.SH"], START, END)

    assert len(session.calls) == 1
    assert exc_info.value.errorcode == -403
    combined = str(exc_info.value) + caplog.text
    assert TOKEN not in combined
    assert upstream_secret not in combined


def test_untrusted_business_errorcode_is_not_echoed(caplog):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindBusinessError,
    )

    untrusted_code = f"malicious-{TOKEN}"
    session = FakeSession(
        [FakeResponse(payload={"errorcode": untrusted_code, "tables": []})]
    )

    with pytest.raises(IFindBusinessError) as exc_info:
        make_client(session).fetch_hourly_bars(["600519.SH"], START, END)

    assert exc_info.value.errorcode is None
    combined = str(exc_info.value) + caplog.text
    assert TOKEN not in combined
    assert untrusted_code not in combined


def test_invalid_json_fails_once_without_raw_decoder_message(caplog):
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindResponseError,
    )

    raw_error = "raw-response-fragment-must-not-leak"
    session = FakeSession(
        [FakeResponse(json_error=ValueError(raw_error))]
    )

    with pytest.raises(IFindResponseError) as exc_info:
        make_client(session).fetch_hourly_bars(["600519.SH"], START, END)

    assert len(session.calls) == 1
    combined = str(exc_info.value) + caplog.text
    assert raw_error not in combined
    assert TOKEN not in combined


def test_non_mapping_json_is_rejected():
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindResponseError,
    )

    session = FakeSession([FakeResponse(payload=["not", "an", "object"])])

    with pytest.raises(IFindResponseError, match="JSON object"):
        make_client(session).fetch_hourly_bars(["600519.SH"], START, END)

    assert len(session.calls) == 1
