"""Secret-safe HTTP client for iFinD historical market data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
import logging
import math
import os
import time
from typing import Any

import requests


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://quantapi.51ifind.com"
HIGH_FREQUENCY_ENDPOINT = "/api/v1/high_frequency"
HISTORY_QUOTATION_ENDPOINT = "/api/v1/cmd_history_quotation"
DEFAULT_TIMEOUT = (3.0, 20.0)
_RETRY_DELAYS = (0.5, 1.0)
# Honour a server-supplied Retry-After, but never park a backtest thread on an
# arbitrarily large one — past this we fail fast and let the caller retry.
MAX_RETRY_AFTER_SECONDS = 30.0


class IFindClientError(RuntimeError):
    """Base error for sanitized iFinD client failures."""


class IFindConfigurationError(IFindClientError):
    """Raised when required local client configuration is missing."""


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


class IFindHttpClient:
    """Fetch official iFinD responses without interpreting table data."""

    def __init__(
        self,
        *,
        session: Any | None = None,
        token: str | None = None,
        base_url: str | None = None,
        timeout: tuple[float, float] = DEFAULT_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_token = token if token is not None else os.getenv(
            "IFIND_ACCESS_TOKEN", ""
        )
        if not resolved_token.strip():
            raise IFindConfigurationError(
                "iFinD credentials are not configured; set IFIND_ACCESS_TOKEN"
            )

        configured_url = base_url
        if configured_url is None:
            configured_url = os.getenv("IFIND_BASE_URL", DEFAULT_BASE_URL)
        configured_url = configured_url.strip() or DEFAULT_BASE_URL

        self._session = session if session is not None else requests.Session()
        self._token = resolved_token.strip()
        self._base_url = configured_url.rstrip("/")
        self._timeout = timeout
        self._sleep = sleep

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

    def _request_json(
        self,
        endpoint: str,
        symbols: Sequence[str],
        start: date,
        end: date,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        url = f"{self._base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "access_token": self._token,
            "ifindlang": "cn",
        }

        for attempt in range(len(_RETRY_DELAYS) + 1):
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

        raise AssertionError("iFinD retry loop ended without a result")

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
                "CPS": "forward1",
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
