"""Secret-safe HTTP client for iFinD historical market data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
import hashlib
import logging
import math
import os
import threading
import time
from typing import Any

import requests


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://quantapi.51ifind.com"
HIGH_FREQUENCY_ENDPOINT = "/api/v1/high_frequency"
HISTORY_QUOTATION_ENDPOINT = "/api/v1/cmd_history_quotation"
BASIC_DATA_ENDPOINT = "/api/v1/basic_data_service"
ACCESS_TOKEN_ENDPOINT = "/api/v1/get_access_token"
DEFAULT_TIMEOUT = (3.0, 20.0)
ACCESS_TOKEN_MAX_AGE_SECONDS = 6 * 24 * 60 * 60
_RETRY_DELAYS = (0.5, 1.0)
# One iteration beyond the transport retries, reserved for the single
# permitted re-send after an access-token refresh: an expired token is not a
# transport fault and must not spend a retry slot. Every other `continue` in
# the request loop is bounded by `attempt < len(_RETRY_DELAYS)` and the
# refresh fires at most once per call, so the loop still always terminates.
_MAX_REQUEST_ATTEMPTS = len(_RETRY_DELAYS) + 2
# The exchange runs while `_ACCESS_TOKEN_CACHE_LOCK` is held, so its retry
# budget is deliberately smaller than the data path's: holding the lock is what
# makes one token rotation cost one exchange, and every extra attempt is time
# every other thread spends blocked on it. One retry absorbs a dropped
# connection without turning an iFinD outage into a multi-minute stall for
# each waiting caller.
_TOKEN_EXCHANGE_ATTEMPTS = 2
# Honour a server-supplied Retry-After, but never park a backtest thread on an
# arbitrarily large one — past this we fail fast and let the caller retry.
MAX_RETRY_AFTER_SECONDS = 30.0


class IFindClientError(RuntimeError):
    """Base error for sanitized iFinD client failures."""


class IFindConfigurationError(IFindClientError):
    """Raised when required local client configuration is missing."""


class IFindTokenRefreshError(IFindClientError):
    """Raised when iFinD refuses or fails to exchange a refresh token."""


class IFindRequestError(IFindClientError):
    """Raised when a caller supplies an invalid request window or symbol list."""


class IFindTransportError(IFindClientError):
    """Raised when the request cannot reach iFinD."""


class IFindHttpError(IFindClientError):
    """Raised when iFinD returns a non-success HTTP status."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class IFindResponseError(IFindClientError):
    """Raised when an HTTP success response is not a JSON object."""


class IFindBusinessError(IFindClientError):
    """Raised when iFinD reports a non-zero business error code."""

    def __init__(self, message: str, errorcode: int | None):
        super().__init__(message)
        self.errorcode = errorcode


def _retry_after_seconds(response: object) -> float | None:
    """Read a clamped Retry-After delay, or None when there isn't a usable one.

    Only the delta-seconds form is honoured: the HTTP-date form would need a
    trusted clock, and iFinD does not document sending one.
    """
    headers = getattr(response, "headers", None) or {}
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        delay = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(delay) or delay < 0:
        return None
    return min(delay, MAX_RETRY_AFTER_SECONDS)


# One access token per process, which is what the design specifies ("the first
# iFinD data request in a process", "a process-local lock so concurrent
# backtests perform one exchange rather than a request per worker thread").
# Held per client instance instead, that guarantee did not hold: a client is
# built per provider, a provider per `create_market_data_provider`, and that
# runs in every `HourlyBacktester.__init__`, so an in-process burst exchanged
# once per backtest. Keyed by base URL and a digest of the refresh token so
# two differently-credentialled clients never share an entry, and so the
# credential itself is never a key in a long-lived module-level mapping.
# (Dashboard backtests run in their own subprocess and so still exchange once
# each; sharing across processes would need a store and is out of scope.)
_ACCESS_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
_ACCESS_TOKEN_CACHE_LOCK = threading.Lock()


class IFindHttpClient:
    """Fetch official iFinD responses without interpreting table data."""

    def __init__(
        self,
        *,
        session: Any | None = None,
        token: str | None = None,
        refresh_token: str | None = None,
        base_url: str | None = None,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        resolved_refresh_token = (
            refresh_token
            if refresh_token is not None
            else os.getenv("IFIND_REFRESH_TOKEN", "")
        )
        resolved_token = token if token is not None else os.getenv(
            "IFIND_ACCESS_TOKEN", ""
        )
        self._refresh_token = resolved_refresh_token.strip()
        self._static_token = resolved_token.strip()
        if not self._refresh_token and not self._static_token:
            raise IFindConfigurationError(
                "iFinD credentials are not configured; "
                "set IFIND_REFRESH_TOKEN or IFIND_ACCESS_TOKEN"
            )

        configured_url = base_url
        if configured_url is None:
            configured_url = os.getenv("IFIND_BASE_URL", DEFAULT_BASE_URL)
        configured_url = configured_url.strip() or DEFAULT_BASE_URL

        self._session = session if session is not None else requests.Session()
        self._base_url = configured_url.rstrip("/")
        self._timeout = timeout
        self._sleep = sleep
        self._clock = clock

    def fetch_hourly_bars(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> Mapping[str, object]:
        """Return the decoded official response for a half-open date window."""
        normalized_symbols = self._validate_request(symbols, start, end)
        payload = self._build_hourly_payload(normalized_symbols, start, end)
        return self._request_json(
            HIGH_FREQUENCY_ENDPOINT,
            normalized_symbols,
            start,
            end,
            payload,
        )

    def fetch_daily_closes(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        *,
        currency: str,
    ) -> Mapping[str, object]:
        """Fetch unadjusted daily closes in RMB or iFinD's USD currency code."""
        normalized_symbols = self._validate_request(symbols, start, end)
        normalized_currency = str(currency or "").strip().upper()
        if normalized_currency not in {"RMB", "MHB"}:
            raise IFindRequestError("currency must be RMB or MHB")
        payload = self._build_daily_close_payload(
            normalized_symbols,
            start,
            end,
            normalized_currency,
        )
        return self._request_json(
            HISTORY_QUOTATION_ENDPOINT,
            normalized_symbols,
            start,
            end,
            payload,
        )

    def fetch_daily_market_rules(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> Mapping[str, object]:
        """Fetch official daily status, closing limit state, and close in CNY."""
        normalized_symbols = self._validate_request(symbols, start, end)
        payload = self._build_daily_market_rules_payload(
            normalized_symbols, start, end
        )
        return self._request_json(
            HISTORY_QUOTATION_ENDPOINT,
            normalized_symbols,
            start,
            end,
            payload,
        )

    def fetch_basic_market_status(
        self,
        symbols: Sequence[str],
        trading_date: date,
    ) -> Mapping[str, object]:
        """Fetch official same-date status supplements for blank history rows."""
        if isinstance(symbols, (str, bytes)):
            raise IFindRequestError("symbols must be a non-empty sequence")
        if any(not isinstance(symbol, str) for symbol in symbols):
            raise IFindRequestError("symbols must contain only strings")
        normalized_symbols = tuple(symbol.strip() for symbol in symbols)
        if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
            raise IFindRequestError("symbols must be a non-empty sequence")
        if not isinstance(trading_date, date):
            raise IFindRequestError("trading_date must be a date value")
        payload = self._build_basic_market_status_payload(
            normalized_symbols, trading_date
        )
        return self._request_json(
            BASIC_DATA_ENDPOINT,
            normalized_symbols,
            trading_date,
            trading_date + timedelta(days=1),
            payload,
        )

    def _request_json(
        self,
        endpoint: str,
        symbols: Sequence[str],
        start: date,
        end: date,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        url = f"{self._base_url}{endpoint}"
        refreshed_after_auth_failure = False

        for attempt in range(_MAX_REQUEST_ATTEMPTS):
            headers = self._data_headers()
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt < len(_RETRY_DELAYS):
                    self._sleep(_RETRY_DELAYS[attempt])
                    continue
                message = self._failure_message(
                    endpoint,
                    symbols,
                    start,
                    end,
                    status_code=None,
                    error_type="transport",
                )
                logger.warning(message)
                raise IFindTransportError(message) from None
            except requests.RequestException:
                message = self._failure_message(
                    endpoint,
                    symbols,
                    start,
                    end,
                    status_code=None,
                    error_type="transport",
                )
                logger.warning(message)
                raise IFindTransportError(message) from None

            status_code = int(response.status_code)
            if not 200 <= status_code < 300:
                if (
                    status_code in {401, 403}
                    and self._refresh_token
                    and not refreshed_after_auth_failure
                ):
                    self._refresh_rejected_access_token(
                        headers.get("access_token", "")
                    )
                    refreshed_after_auth_failure = True
                    continue
                retryable = status_code == 429 or 500 <= status_code < 600
                if retryable and attempt < len(_RETRY_DELAYS):
                    # A throttled server knows better than our fixed backoff
                    # how long it wants us gone; a bare 0.5s retry into a 429
                    # just burns the attempt budget.
                    delay = _retry_after_seconds(response)
                    if delay is None:
                        delay = _RETRY_DELAYS[attempt]
                    self._sleep(delay)
                    continue
                message = self._failure_message(
                    endpoint,
                    symbols,
                    start,
                    end,
                    status_code=status_code,
                    error_type="http",
                )
                logger.warning(message)
                raise IFindHttpError(message, status_code) from None

            try:
                decoded = response.json()
            except ValueError:
                message = self._failure_message(
                    endpoint,
                    symbols,
                    start,
                    end,
                    status_code=status_code,
                    error_type="invalid_json",
                )
                logger.warning(message)
                raise IFindResponseError(message) from None

            if not isinstance(decoded, Mapping):
                message = self._failure_message(
                    endpoint,
                    symbols,
                    start,
                    end,
                    status_code=status_code,
                    error_type="non_object_json",
                )
                logger.warning(message)
                raise IFindResponseError(
                    f"{message}; expected a JSON object"
                ) from None

            if "errorcode" in decoded and decoded["errorcode"] != 0:
                raw_errorcode = decoded["errorcode"]
                errorcode = (
                    raw_errorcode
                    if isinstance(raw_errorcode, int)
                    and not isinstance(raw_errorcode, bool)
                    else None
                )
                errorcode_label = (
                    str(errorcode) if errorcode is not None else "unavailable"
                )
                message = self._failure_message(
                    endpoint,
                    symbols,
                    start,
                    end,
                    status_code=status_code,
                    error_type="business",
                )
                logger.warning(message)
                raise IFindBusinessError(
                    f"{message}; errorcode={errorcode_label}", errorcode
                ) from None

            return decoded

        # Unreachable while every `continue` above stays bounded, but raised
        # inside the client's own hierarchy regardless: an AssertionError here
        # slips past the engine's `except IFindClientError` handlers and reaches
        # the user as an unhandled traceback rather than a sanitized
        # market-data failure.
        raise IFindTransportError(
            self._failure_message(
                endpoint,
                symbols,
                start,
                end,
                status_code=None,
                error_type="attempts_exhausted",
            )
        )

    def _data_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "access_token": self._get_access_token(),
            "ifindlang": "cn",
        }

    def _cache_key(self) -> tuple[str, str]:
        digest = hashlib.sha256(self._refresh_token.encode("utf-8")).hexdigest()
        return (self._base_url, digest)

    def _get_access_token(self) -> str:
        if not self._refresh_token:
            return self._static_token

        key = self._cache_key()
        now = self._clock()
        with _ACCESS_TOKEN_CACHE_LOCK:
            cached = _ACCESS_TOKEN_CACHE.get(key)
            if cached is not None:
                token, issued_at = cached
                if token and now - issued_at < ACCESS_TOKEN_MAX_AGE_SECONDS:
                    return token
            access_token = self._exchange_refresh_token()
            _ACCESS_TOKEN_CACHE[key] = (access_token, self._clock())
            return access_token

    def _refresh_rejected_access_token(self, rejected_token: str) -> None:
        """Replace ``rejected_token`` unless a sibling caller already did.

        Invalidating and then force-refreshing took the lock twice with a gap
        between, so every thread holding a 401 from the same rotation forced its
        own exchange -- and each invalidation discarded the token a sibling had
        just fetched, cascading further exchanges against an endpoint meant to be
        called about once a week. One acquisition, comparing against the token
        that actually failed, makes a rotation cost exactly one exchange.
        """
        if not self._refresh_token:
            return

        key = self._cache_key()
        with _ACCESS_TOKEN_CACHE_LOCK:
            cached = _ACCESS_TOKEN_CACHE.get(key)
            if cached is not None and cached[0] and cached[0] != rejected_token:
                return
            _ACCESS_TOKEN_CACHE[key] = (
                self._exchange_refresh_token(),
                self._clock(),
            )

    def _refresh_error(
        self, reason: str, *, detail: str = ""
    ) -> IFindTokenRefreshError:
        """Log one refresh failure and build it, with enough detail to tell them apart.

        Every branch below produced a bare exception and no log line, so "the
        refresh token was revoked", "iFinD renamed data.access_token" and "the
        endpoint returned HTML" were indistinguishable in production. The
        upstream ``errmsg`` is still never echoed -- it is attacker-influenced
        and has carried the credential itself -- but the numeric errorcode and
        HTTP status are safe and are the whole diagnosis.

        Returned rather than raised so each `raise` stays visible at its own
        call site: a helper that raises makes every caller look like it can
        fall off the end, which is both harder to read and a static-analysis
        false positive waiting to happen.
        """
        message = f"iFinD access token refresh {reason}"
        if detail:
            message = f"{message}; {detail}"
        logger.warning(message)
        return IFindTokenRefreshError(message)

    def _post_token_exchange(self) -> Any:
        """POST the refresh token, retrying only the faults worth retrying.

        A connection reset gets a second chance; every other ``requests``
        failure is wrapped immediately. The wrapping is the point -- a raw
        ``requests`` exception escapes the engine's ``except IFindClientError``
        handlers, and its message can carry the refresh token itself.
        """
        url = f"{self._base_url}{ACCESS_TOKEN_ENDPOINT}"
        for attempt in range(_TOKEN_EXCHANGE_ATTEMPTS):
            try:
                return self._session.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "refresh_token": self._refresh_token,
                    },
                    timeout=self._timeout,
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt < _TOKEN_EXCHANGE_ATTEMPTS - 1:
                    self._sleep(_RETRY_DELAYS[attempt])
                    continue
                break
            except requests.RequestException:
                break
        raise self._refresh_error("failed during transport") from None

    def _exchange_refresh_token(self) -> str:
        response = self._post_token_exchange()

        status_code = int(response.status_code)
        if not 200 <= status_code < 300:
            raise self._refresh_error(
                f"failed with HTTP {status_code}"
            ) from None

        try:
            decoded = response.json()
        except ValueError:
            raise self._refresh_error(
                "returned invalid JSON", detail=f"status={status_code}"
            ) from None
        if not isinstance(decoded, Mapping):
            raise self._refresh_error(
                "returned an invalid response", detail=f"status={status_code}"
            ) from None

        raw_errorcode = decoded.get("errorcode")
        errorcode = (
            raw_errorcode
            if isinstance(raw_errorcode, int) and not isinstance(raw_errorcode, bool)
            else None
        )
        errorcode_label = str(errorcode) if errorcode is not None else "unavailable"
        # Absence means success on the data path (see `_request_json`), so it has
        # to mean success here too: rejecting a response that omits `errorcode`
        # while carrying a valid token fails every A-share backtest and points
        # the operator at a credential problem that does not exist.
        reported_failure = "errorcode" in decoded and decoded["errorcode"] != 0
        data = decoded.get("data")
        access_token = data.get("access_token") if isinstance(data, Mapping) else None
        if (
            reported_failure
            or not isinstance(access_token, str)
            or not access_token.strip()
        ):
            raise self._refresh_error(
                "returned no usable access token",
                detail=f"errorcode={errorcode_label}",
            ) from None
        return access_token.strip()

    @staticmethod
    def _validate_request(
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> tuple[str, ...]:
        if isinstance(symbols, (str, bytes)):
            raise IFindRequestError("symbols must be a non-empty sequence")
        if any(not isinstance(symbol, str) for symbol in symbols):
            raise IFindRequestError("symbols must contain only strings")
        normalized_symbols = tuple(symbol.strip() for symbol in symbols)
        if not normalized_symbols or any(not symbol for symbol in normalized_symbols):
            raise IFindRequestError("symbols must be a non-empty sequence")
        if not isinstance(start, date) or not isinstance(end, date):
            raise IFindRequestError("start and end must be date values")
        if end <= start:
            raise IFindRequestError("end must be after start")
        return normalized_symbols

    @staticmethod
    def _build_hourly_payload(
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> dict[str, object]:
        effective_last_day = end - timedelta(days=1)
        return {
            "codes": ",".join(symbols),
            "indicators": "open,high,low,close,volume",
            "starttime": f"{start.isoformat()} 09:30:00",
            "endtime": f"{effective_last_day.isoformat()} 15:00:00",
            "functionpara": {
                "Interval": "60",
                "CPS": "no",
                "Timeformat": "LocalTime",
                "Limitstart": "09:30:00",
                "Limitend": "15:00:00",
            },
        }

    @staticmethod
    def _build_daily_close_payload(
        symbols: Sequence[str],
        start: date,
        end: date,
        currency: str,
    ) -> dict[str, object]:
        effective_last_day = end - timedelta(days=1)
        return {
            "codes": ",".join(symbols),
            "indicators": "close",
            "startdate": start.isoformat(),
            "enddate": effective_last_day.isoformat(),
            "functionpara": {
                "Interval": "D",
                "CPS": "1",
                "Currency": currency,
                "Fill": "Blank",
            },
        }

    @staticmethod
    def _build_daily_market_rules_payload(
        symbols: Sequence[str],
        start: date,
        end: date,
    ) -> dict[str, object]:
        effective_last_day = end - timedelta(days=1)
        return {
            "codes": ",".join(symbols),
            "indicators": (
                "close,ths_trading_status_stock,"
                "ths_up_and_down_status_stock"
            ),
            "startdate": start.isoformat(),
            "enddate": effective_last_day.isoformat(),
            "functionpara": {
                "Interval": "D",
                "CPS": "1",
                "Currency": "RMB",
                "Fill": "Blank",
            },
        }

    @staticmethod
    def _build_basic_market_status_payload(
        symbols: Sequence[str],
        trading_date: date,
    ) -> dict[str, object]:
        date_value = trading_date.isoformat()
        return {
            "codes": ",".join(symbols),
            "indipara": [
                {
                    "indicator": "ths_trading_status_stock",
                    "indiparams": [date_value],
                },
                {
                    "indicator": "ths_up_and_down_status_stock",
                    "indiparams": [date_value],
                },
            ],
        }

    @staticmethod
    def _failure_message(
        endpoint: str,
        symbols: Sequence[str],
        start: date,
        end: date,
        *,
        status_code: int | None,
        error_type: str,
    ) -> str:
        status = "none" if status_code is None else str(status_code)
        return (
            "iFinD request failed "
            f"endpoint={endpoint} "
            f"symbols={len(symbols)} "
            f"start={start.isoformat()} "
            f"end={end.isoformat()} "
            f"status={status} "
            f"error={error_type}"
        )
