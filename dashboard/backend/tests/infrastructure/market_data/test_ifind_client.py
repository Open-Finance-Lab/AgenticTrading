"""HTTP contract, retry, and secret-safety tests for the iFinD client."""

from __future__ import annotations

from datetime import date
import logging
import threading

import pytest
import requests

from dashboard.backend.infrastructure.market_data.profiles import (
    A_SHARE_DEMO_6_SYMBOLS,
)


START = date(2026, 4, 1)
END = date(2026, 4, 23)
TOKEN = "private-ifind-token"
REFRESH_TOKEN = "private-ifind-refresh-token"
SHORT_TOKEN = "short-lived-ifind-token"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, json_error=None, headers=None):
        self.status_code = status_code
        self._payload = {"errorcode": 0, "tables": []} if payload is None else payload
        self._json_error = json_error
        self.headers = dict(headers or {})

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
        # Pinned rather than left to default: unset, the client falls back to
        # os.getenv("IFIND_REFRESH_TOKEN"), so a developer's exported prod
        # credential would turn every static-token case below into a
        # refresh-first client and eat the first queued response.
        "refresh_token": "",
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
            "CPS": "no",
            "Timeformat": "LocalTime",
            "Limitstart": "09:30:00",
            "Limitend": "15:00:00",
        },
    }
    assert "Fill" not in kwargs["json"]["functionpara"]


@pytest.mark.parametrize("currency", ["RMB", "MHB"])
def test_builds_official_daily_close_request_with_exclusive_end(currency):
    session = FakeSession([FakeResponse()])

    result = make_client(session).fetch_daily_closes(
        ["600519.SH", "601318.SH"],
        START,
        END,
        currency=currency,
    )

    assert result == {"errorcode": 0, "tables": []}
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == "https://ifind.test/api/v1/cmd_history_quotation"
    assert kwargs["headers"] == {
        "Content-Type": "application/json",
        "access_token": TOKEN,
        "ifindlang": "cn",
    }
    assert kwargs["json"] == {
        "codes": "600519.SH,601318.SH",
        "indicators": "close",
        "startdate": "2026-04-01",
        "enddate": "2026-04-22",
        "functionpara": {
            "Interval": "D",
            "CPS": "1",
            "Currency": currency,
            "Fill": "Blank",
        },
    }


def test_builds_official_daily_market_rule_request_with_verified_indicators():
    session = FakeSession([FakeResponse()])

    result = make_client(session).fetch_daily_market_rules(
        ["600519.SH", "601318.SH"], START, END
    )

    assert result == {"errorcode": 0, "tables": []}
    url, kwargs = session.calls[0]
    assert url == "https://ifind.test/api/v1/cmd_history_quotation"
    assert kwargs["json"] == {
        "codes": "600519.SH,601318.SH",
        "indicators": (
            "close,ths_trading_status_stock,ths_up_and_down_status_stock"
        ),
        "startdate": "2026-04-01",
        "enddate": "2026-04-22",
        "functionpara": {
            "Interval": "D",
            "CPS": "1",
            "Currency": "RMB",
            "Fill": "Blank",
        },
    }


def test_builds_basic_status_supplement_request():
    session = FakeSession([FakeResponse()])

    result = make_client(session).fetch_basic_market_status(
        ["688981.SH"], date(2025, 9, 1)
    )

    assert result == {"errorcode": 0, "tables": []}
    url, kwargs = session.calls[0]
    assert url == "https://ifind.test/api/v1/basic_data_service"
    assert kwargs["json"] == {
        "codes": "688981.SH",
        "indipara": [
            {
                "indicator": "ths_trading_status_stock",
                "indiparams": ["2025-09-01"],
            },
            {
                "indicator": "ths_up_and_down_status_stock",
                "indiparams": ["2025-09-01"],
            },
        ],
    }


def test_rejects_unknown_daily_close_currency_before_http_call():
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindRequestError,
    )

    session = FakeSession([FakeResponse()])

    with pytest.raises(IFindRequestError, match="currency"):
        make_client(session).fetch_daily_closes(
            ["600519.SH"], START, END, currency="USD"
        )

    assert session.calls == []


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


def test_exchanges_refresh_token_before_first_data_request():
    session = FakeSession(
        [
            FakeResponse(
                payload={
                    "errorcode": 0,
                    "data": {"access_token": SHORT_TOKEN},
                }
            ),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
        ]
    )

    result = make_client(
        session,
        token="",
        refresh_token=REFRESH_TOKEN,
    ).fetch_hourly_bars(["600519.SH"], START, END)

    assert result == {"errorcode": 0, "tables": []}
    assert len(session.calls) == 2
    refresh_url, refresh_kwargs = session.calls[0]
    assert refresh_url == "https://ifind.test/api/v1/get_access_token"
    assert refresh_kwargs["headers"] == {
        "Content-Type": "application/json",
        "refresh_token": REFRESH_TOKEN,
    }
    data_url, data_kwargs = session.calls[1]
    assert data_url.endswith("/api/v1/high_frequency")
    assert data_kwargs["headers"]["access_token"] == SHORT_TOKEN


def test_refresh_token_takes_precedence_over_static_token():
    session = FakeSession(
        [
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": SHORT_TOKEN}}
            ),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
        ]
    )

    make_client(
        session,
        token=TOKEN,
        refresh_token=REFRESH_TOKEN,
    ).fetch_hourly_bars(["600519.SH"], START, END)

    assert session.calls[1][1]["headers"]["access_token"] == SHORT_TOKEN


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
    monkeypatch.delenv("IFIND_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("IFIND_ACCESS_TOKEN", raising=False)

    with pytest.raises(IFindConfigurationError, match="IFIND_ACCESS_TOKEN"):
        IFindHttpClient(session=session)

    assert session.calls == []


def test_refreshes_once_after_authentication_failure():
    session = FakeSession(
        [
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": "first-token"}}
            ),
            FakeResponse(status_code=401),
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": "second-token"}}
            ),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
        ]
    )

    result = make_client(
        session,
        token="",
        refresh_token=REFRESH_TOKEN,
    ).fetch_hourly_bars(["600519.SH"], START, END)

    assert result == {"errorcode": 0, "tables": []}
    assert len(session.calls) == 4
    assert session.calls[1][1]["headers"]["access_token"] == "first-token"
    assert session.calls[3][1]["headers"]["access_token"] == "second-token"


def test_refresh_cache_expires_after_six_days():
    now = [0.0]
    session = FakeSession(
        [
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": "token-1"}}
            ),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": "token-2"}}
            ),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
        ]
    )
    client = make_client(
        session,
        token="",
        refresh_token=REFRESH_TOKEN,
        clock=lambda: now[0],
    )

    client.fetch_hourly_bars(["600519.SH"], START, END)
    now[0] = 6 * 24 * 60 * 60 - 1
    client.fetch_hourly_bars(["600519.SH"], START, END)
    now[0] = 6 * 24 * 60 * 60
    client.fetch_hourly_bars(["600519.SH"], START, END)

    assert [url.endswith("/api/v1/get_access_token") for url, _ in session.calls] == [
        True,
        False,
        False,
        True,
        False,
    ]


def test_concurrent_first_requests_exchange_refresh_token_once():
    class ConcurrentSession:
        def __init__(self):
            self.calls = []
            self.exchange_calls = 0
            self._lock = threading.Lock()

        def post(self, url, **kwargs):
            with self._lock:
                self.calls.append((url, kwargs))
                if url.endswith("/api/v1/get_access_token"):
                    self.exchange_calls += 1
            if url.endswith("/api/v1/get_access_token"):
                return FakeResponse(
                    payload={"errorcode": 0, "data": {"access_token": SHORT_TOKEN}}
                )
            return FakeResponse(payload={"errorcode": 0, "tables": []})

    session = ConcurrentSession()
    client = make_client(
        session,
        token="",
        refresh_token=REFRESH_TOKEN,
    )
    errors = []

    def fetch():
        try:
            client.fetch_hourly_bars(["600519.SH"], START, END)
        except Exception as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)

    threads = [threading.Thread(target=fetch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert session.exchange_calls == 1


def test_refresh_failure_is_sanitized():
    session = FakeSession(
        [
            FakeResponse(
                payload={"errorcode": -1, "errmsg": REFRESH_TOKEN}
            )
        ]
    )

    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindTokenRefreshError,
    )

    with pytest.raises(IFindTokenRefreshError) as exc_info:
        make_client(
            session,
            token="",
            refresh_token=REFRESH_TOKEN,
        ).fetch_hourly_bars(["600519.SH"], START, END)

    assert REFRESH_TOKEN not in str(exc_info.value)


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


def test_throttled_retry_honours_retry_after_over_the_fixed_backoff():
    sleeps = []
    session = FakeSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "5"}),
            FakeResponse(payload={"errorcode": 0}),
        ]
    )

    result = make_client(session, sleep=sleeps.append).fetch_hourly_bars(
        ["600519.SH"], START, END
    )

    assert result == {"errorcode": 0}
    assert sleeps == [5.0]


def test_retry_after_is_clamped_so_a_backtest_thread_cannot_be_parked():
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        MAX_RETRY_AFTER_SECONDS,
    )

    sleeps = []
    session = FakeSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "86400"}),
            FakeResponse(payload={"errorcode": 0}),
        ]
    )

    make_client(session, sleep=sleeps.append).fetch_hourly_bars(
        ["600519.SH"], START, END
    )

    assert sleeps == [MAX_RETRY_AFTER_SECONDS]


@pytest.mark.parametrize("raw", ["", "not-a-number", "-1", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_unusable_retry_after_falls_back_to_the_fixed_backoff(raw):
    sleeps = []
    session = FakeSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": raw}),
            FakeResponse(payload={"errorcode": 0}),
        ]
    )

    make_client(session, sleep=sleeps.append).fetch_hourly_bars(
        ["600519.SH"], START, END
    )

    assert sleeps == [0.5]


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


def test_auth_refresh_on_the_final_attempt_still_re_sends():
    """An expired token discovered after the transport retries is recoverable.

    The refresh path re-sends the request, so it must not be the loop's last
    iteration: falling out of the retry loop raises an error the backtest
    engine does not catch.
    """
    session = FakeSession(
        [
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": "first-token"}}
            ),
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
            FakeResponse(status_code=401),
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": "second-token"}}
            ),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
        ]
    )

    result = make_client(
        session,
        token="",
        refresh_token=REFRESH_TOKEN,
    ).fetch_hourly_bars(["600519.SH"], START, END)

    assert result == {"errorcode": 0, "tables": []}
    assert session.calls[-1][1]["headers"]["access_token"] == "second-token"


def test_persistent_auth_failure_raises_a_catchable_client_error():
    """A 401 that survives the refresh must stay inside the client's hierarchy."""
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindClientError,
    )

    session = FakeSession(
        [
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": "first-token"}}
            ),
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
            FakeResponse(status_code=401),
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": "second-token"}}
            ),
            FakeResponse(status_code=401),
        ]
    )

    with pytest.raises(IFindClientError):
        make_client(
            session,
            token="",
            refresh_token=REFRESH_TOKEN,
        ).fetch_hourly_bars(["600519.SH"], START, END)


def test_refresh_accepts_a_response_that_omits_errorcode():
    """Absent `errorcode` means success on the data path; the same must hold here."""
    session = FakeSession(
        [
            FakeResponse(payload={"data": {"access_token": SHORT_TOKEN}}),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
        ]
    )

    result = make_client(
        session,
        token="",
        refresh_token=REFRESH_TOKEN,
    ).fetch_hourly_bars(["600519.SH"], START, END)

    assert result == {"errorcode": 0, "tables": []}
    assert session.calls[1][1]["headers"]["access_token"] == SHORT_TOKEN


def test_refresh_still_rejects_a_non_zero_errorcode_carrying_a_token():
    """Tolerating absence must not become tolerating an explicit failure."""
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindTokenRefreshError,
    )

    session = FakeSession(
        [
            FakeResponse(
                payload={"errorcode": -1, "data": {"access_token": SHORT_TOKEN}}
            )
        ]
    )

    with pytest.raises(IFindTokenRefreshError):
        make_client(
            session,
            token="",
            refresh_token=REFRESH_TOKEN,
        ).fetch_hourly_bars(["600519.SH"], START, END)


def test_refresh_failure_logs_the_errorcode_without_the_secret(caplog):
    """A revoked token and a renamed field must be distinguishable in the log."""
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindTokenRefreshError,
    )

    session = FakeSession(
        [FakeResponse(payload={"errorcode": -1, "errmsg": REFRESH_TOKEN})]
    )

    with caplog.at_level(
        logging.WARNING,
        logger="dashboard.backend.infrastructure.market_data.ifind_client",
    ):
        with pytest.raises(IFindTokenRefreshError):
            make_client(
                session,
                token="",
                refresh_token=REFRESH_TOKEN,
            ).fetch_hourly_bars(["600519.SH"], START, END)

    assert "errorcode=-1" in caplog.text
    assert REFRESH_TOKEN not in caplog.text
    assert TOKEN not in caplog.text


def test_refresh_transport_failure_is_logged_without_the_secret(caplog):
    """The three non-business refresh failures must not be silent either."""
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindTokenRefreshError,
    )

    session = FakeSession(
        [
            requests.ConnectionError(REFRESH_TOKEN),
            requests.ConnectionError(REFRESH_TOKEN),
            requests.ConnectionError(REFRESH_TOKEN),
        ]
    )

    with caplog.at_level(
        logging.WARNING,
        logger="dashboard.backend.infrastructure.market_data.ifind_client",
    ):
        with pytest.raises(IFindTokenRefreshError):
            make_client(
                session,
                token="",
                refresh_token=REFRESH_TOKEN,
            ).fetch_hourly_bars(["600519.SH"], START, END)

    assert "iFinD access token refresh" in caplog.text
    assert REFRESH_TOKEN not in caplog.text


def test_token_exchange_retries_a_transport_failure():
    """A single connection reset on the exchange must not abort the backtest."""
    session = FakeSession(
        [
            requests.ConnectionError("boom"),
            FakeResponse(
                payload={"errorcode": 0, "data": {"access_token": SHORT_TOKEN}}
            ),
            FakeResponse(payload={"errorcode": 0, "tables": []}),
        ]
    )

    result = make_client(
        session,
        token="",
        refresh_token=REFRESH_TOKEN,
    ).fetch_hourly_bars(["600519.SH"], START, END)

    assert result == {"errorcode": 0, "tables": []}
    assert session.calls[2][1]["headers"]["access_token"] == SHORT_TOKEN


def test_concurrent_authentication_failures_exchange_once():
    """One rotation must cost one exchange, not one per in-flight thread."""

    class AuthFailureSession:
        def __init__(self):
            self._lock = threading.Lock()
            self.exchange_calls = 0
            self.data_calls = 0
            # Both threads must be holding a 401 before either is allowed to
            # refresh, or the race this test exists to pin never happens.
            self._both_rejected = threading.Barrier(2, timeout=10)

        def post(self, url, **kwargs):
            if url.endswith("/api/v1/get_access_token"):
                with self._lock:
                    self.exchange_calls += 1
                    issued = self.exchange_calls
                return FakeResponse(
                    payload={
                        "errorcode": 0,
                        "data": {"access_token": f"token-{issued}"},
                    }
                )
            with self._lock:
                self.data_calls += 1
                seen = self.data_calls
            if seen <= 2:
                self._both_rejected.wait()
                return FakeResponse(status_code=401)
            return FakeResponse(payload={"errorcode": 0, "tables": []})

    session = AuthFailureSession()
    client = make_client(session, token="", refresh_token=REFRESH_TOKEN)
    errors: list[Exception] = []

    def fetch():
        try:
            client.fetch_hourly_bars(["600519.SH"], START, END)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=fetch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert errors == []
    # One exchange to open, one to replace the rotated token. A third means
    # each thread refreshed independently -- and clobbered its sibling's token.
    assert session.exchange_calls == 2


class CountingSession:
    """Classifies calls by endpoint so exchanges can be counted, not queued."""

    def __init__(self):
        self.exchange_calls = 0
        self.data_tokens: list[str] = []

    def post(self, url, **kwargs):
        if url.endswith("/api/v1/get_access_token"):
            self.exchange_calls += 1
            return FakeResponse(
                payload={
                    "errorcode": 0,
                    "data": {"access_token": f"token-{self.exchange_calls}"},
                }
            )
        self.data_tokens.append(kwargs["headers"]["access_token"])
        return FakeResponse(payload={"errorcode": 0, "tables": []})


def test_separate_clients_in_one_process_share_one_access_token():
    """The cache is per process, as designed -- not per client instance.

    A client is built per provider, a provider per `create_market_data_provider`
    call, and that runs in every `HourlyBacktester.__init__`. Scoped to the
    instance, an in-process burst of backtests exchanges once per backtest,
    which is the per-worker request the design set out to avoid.
    """
    session = CountingSession()

    make_client(session, token="", refresh_token=REFRESH_TOKEN).fetch_hourly_bars(
        ["600519.SH"], START, END
    )
    make_client(session, token="", refresh_token=REFRESH_TOKEN).fetch_hourly_bars(
        ["600519.SH"], START, END
    )

    assert session.exchange_calls == 1
    assert session.data_tokens == ["token-1", "token-1"]


def test_clients_with_different_credentials_do_not_share_a_token():
    """Sharing is keyed on the credential, so a second account gets its own token."""
    session = CountingSession()

    make_client(session, token="", refresh_token=REFRESH_TOKEN).fetch_hourly_bars(
        ["600519.SH"], START, END
    )
    make_client(
        session, token="", refresh_token="a-second-ifind-refresh-token"
    ).fetch_hourly_bars(["600519.SH"], START, END)

    assert session.exchange_calls == 2
    assert session.data_tokens == ["token-1", "token-2"]


def test_token_exchange_wraps_every_request_exception():
    """Adding retries must not narrow what the exchange catches.

    Only connection resets and timeouts are worth retrying, but every
    ``requests`` failure still has to leave as an ``IFindClientError`` -- a raw
    ``requests`` exception escapes the engine's handlers the same way an
    ``AssertionError`` does.
    """
    from dashboard.backend.infrastructure.market_data.ifind_client import (
        IFindTokenRefreshError,
    )

    session = FakeSession([requests.TooManyRedirects(REFRESH_TOKEN)])

    with pytest.raises(IFindTokenRefreshError):
        make_client(
            session,
            token="",
            refresh_token=REFRESH_TOKEN,
        ).fetch_hourly_bars(["600519.SH"], START, END)
