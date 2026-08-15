"""Translate verified iFinD status responses into ATL market-rule objects."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

import pandas as pd

from dashboard.backend.domain.backtesting.market_rules import (
    ClosingLimitState,
    DailyMarketRule,
    MarketRuleCalendar,
    MarketRuleDataError,
)


TRADING_STATUS_FIELD = "ths_trading_status_stock"
LIMIT_STATUS_FIELD = "ths_up_and_down_status_stock"

# Wider than any A-share daily band (10% main boards, 20% STAR/ChiNext), so an
# overnight move past it cannot have come from trading. Both the hourly bars and
# these daily closes are requested unadjusted — deliberately, because the audit
# has to show the official close a limit was set against, not a back-adjusted
# one — and nothing in the backend applies dividends or splits. A 除权除息 date
# therefore lands in the equity curve as a real overnight loss that never
# happened. Detecting it does not fix it; it stops it being silent.
_CORPORATE_ACTION_GAP = Decimal("0.21")


def _fail(detail: str) -> MarketRuleDataError:
    return MarketRuleDataError(f"Market rule data unavailable: {detail}")


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "")[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise _fail("invalid rule date") from None


def _values(table: Mapping[str, object], field: str, symbol: str) -> list[object]:
    raw = table.get(field)
    if not isinstance(raw, list):
        raise _fail(f"missing field={field} for symbol={symbol}")
    return raw


def _status_is_suspended(trading_status: object, limit_status: object) -> bool:
    values = (str(trading_status or ""), str(limit_status or ""))
    return any("停牌" in value for value in values)


def _limit_state(value: object, symbol: str, trading_date: date) -> ClosingLimitState:
    text = str(value or "").strip()
    if text == "涨停":
        return ClosingLimitState.UPPER
    if text == "跌停":
        return ClosingLimitState.LOWER
    if text in {"非涨跌停", "停牌"}:
        return ClosingLimitState.NONE
    # Naming the value is the whole diagnosis. The gate is deliberately keyed on
    # exact vendor strings and stays that way — guessing at an unrecognised
    # status is how a limit-locked day gets traded — but an operator has to be
    # able to tell "iFinD added a status" from "this account answers in English"
    # from "the field arrived blank", and a bare message tells them none of it.
    raise _fail(
        f"unknown closing limit status={text[:40]!r} for {symbol} {trading_date}"
    )


def _numeric_close(value: object, symbol: str, trading_date: date) -> Decimal | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        close = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise _fail(f"invalid official close for {symbol} {trading_date}") from None
    if not close.is_finite() or close <= 0:
        raise _fail(f"invalid official close for {symbol} {trading_date}")
    return close


def _table_by_symbol(payload: Mapping[str, object], expected: Sequence[str]) -> dict[str, Mapping[str, object]]:
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise _fail("response tables are malformed")
    result: dict[str, Mapping[str, object]] = {}
    expected_set = set(expected)
    for item in tables:
        if not isinstance(item, Mapping):
            raise _fail("response table is malformed")
        symbol = str(item.get("thscode") or "").strip().upper()
        table = item.get("table")
        if symbol not in expected_set or not isinstance(table, Mapping):
            raise _fail("response contains an unexpected or malformed symbol table")
        if symbol in result:
            raise _fail("duplicate symbol rule response")
        result[symbol] = table
    return result


def _history_rows(
    payload: Mapping[str, object], expected: Sequence[str]
) -> dict[tuple[str, date], dict[str, object]]:
    tables = payload.get("tables")
    if not isinstance(tables, list):
        raise _fail("history response tables are malformed")
    expected_set = set(expected)
    rows: dict[tuple[str, date], dict[str, object]] = {}
    seen_symbols: set[str] = set()
    for item in tables:
        if not isinstance(item, Mapping):
            raise _fail("history response table is malformed")
        symbol = str(item.get("thscode") or "").strip().upper()
        if symbol not in expected_set:
            raise _fail("history response contains an unexpected symbol")
        if symbol in seen_symbols:
            raise _fail("duplicate history symbol response")
        seen_symbols.add(symbol)
        times = item.get("time")
        table = item.get("table")
        if not isinstance(times, list) or not isinstance(table, Mapping):
            raise _fail(f"history response is malformed for symbol={symbol}")
        fields = {
            field: _values(table, field, symbol)
            for field in ("close", TRADING_STATUS_FIELD, LIMIT_STATUS_FIELD)
        }
        lengths = {len(times), *(len(values) for values in fields.values())}
        if len(lengths) != 1:
            raise _fail(f"history field lengths differ for symbol={symbol}")
        for index, raw_date in enumerate(times):
            trading_date = _as_date(raw_date)
            key = (symbol, trading_date)
            if key in rows:
                raise _fail("duplicate symbol-date history response")
            rows[key] = {
                "close": fields["close"][index],
                "trading_status": fields[TRADING_STATUS_FIELD][index],
                "limit_status": fields[LIMIT_STATUS_FIELD][index],
            }
    missing_symbols = expected_set - seen_symbols
    if missing_symbols:
        raise _fail(f"history response is missing symbols={sorted(missing_symbols)!r}")
    return rows


def _basic_rows(
    payload: Mapping[str, object], expected: Sequence[str], trading_date: date
) -> dict[str, dict[str, object]]:
    tables = _table_by_symbol(payload, expected)
    result: dict[str, dict[str, object]] = {}
    for symbol in expected:
        table = tables.get(symbol)
        if table is None:
            raise _fail(f"basic status response is missing symbol={symbol}")
        trading_values = _values(table, TRADING_STATUS_FIELD, symbol)
        limit_values = _values(table, LIMIT_STATUS_FIELD, symbol)
        if len(trading_values) != 1 or len(limit_values) != 1:
            raise _fail(f"basic status response has invalid row count for {symbol}")
        result[symbol] = {
            "close": None,
            "trading_status": trading_values[0],
            "limit_status": limit_values[0],
            "date": trading_date,
        }
    return result


def _final_bar_for_date(frame: pd.DataFrame, trading_date: date) -> datetime | None:
    timestamps = []
    for timestamp in frame.index:
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()
        if not isinstance(timestamp, datetime):
            continue
        if timestamp.date() == trading_date:
            timestamps.append(timestamp)
    if not timestamps:
        return None
    return max(timestamps)


def _bar_close(frame: pd.DataFrame, timestamp: datetime) -> object:
    try:
        value = frame.loc[pd.Timestamp(timestamp), "close"]
    except (KeyError, IndexError, TypeError):
        raise _fail("final hourly bar close is unavailable") from None
    if isinstance(value, pd.Series):
        raise _fail("duplicate final hourly bar")
    return value


def _same_price_tick(left: Decimal, right: Decimal, price_tick: Decimal) -> bool:
    if price_tick <= 0:
        raise _fail("price tick is invalid")
    return (
        (left / price_tick).to_integral_value(rounding=ROUND_HALF_UP)
        == (right / price_tick).to_integral_value(rounding=ROUND_HALF_UP)
    )


def _require_price_tick(
    value: Decimal, price_tick: Decimal, symbol: str, trading_date: date
) -> None:
    if value % price_tick != 0:
        raise _fail(f"price tick mismatch for {symbol} {trading_date}")


def response_to_market_rules(
    payload: Mapping[str, object],
    *,
    expected_symbols: Sequence[str],
    required_dates: Sequence[date],
    bars_by_symbol: Mapping[str, pd.DataFrame],
    fetch_basic_status: Callable[[Sequence[str], date], Mapping[str, object]],
    price_tick: Decimal | float = Decimal("0.01"),
) -> MarketRuleCalendar:
    """Normalize daily history plus official blank-row supplements."""
    expected = tuple(str(symbol).strip().upper() for symbol in expected_symbols)
    dates = tuple(sorted(set(required_dates)))
    if not expected or not dates:
        raise _fail("rule calendar scope is empty")
    history = _history_rows(payload, expected)
    tick = Decimal(str(price_tick))
    if tick <= 0:
        raise _fail("price tick is invalid")

    supplements: dict[tuple[str, date], dict[str, object]] = {}
    for trading_date in dates:
        missing = []
        for symbol in expected:
            row = history.get((symbol, trading_date))
            if row is None or any(
                row.get(field) in (None, "")
                for field in ("trading_status", "limit_status")
            ):
                if row is not None and _status_is_suspended(
                    row.get("trading_status"), row.get("limit_status")
                ):
                    # A suspended security is exactly what leaves these fields
                    # blank, and one field already settles it — the supplement
                    # cannot change the outcome. Skipping it matters because
                    # this fetch is one blocking round trip per date: a universe
                    # with something suspended most days would otherwise open
                    # the run with dozens of sequential vendor calls.
                    continue
                missing.append(symbol)
        if missing:
            supplement_payload = fetch_basic_status(tuple(missing), trading_date)
            supplements_for_date = _basic_rows(
                supplement_payload, missing, trading_date
            )
            supplements.update({
                (symbol, trading_date): row
                for symbol, row in supplements_for_date.items()
            })

    rules: list[DailyMarketRule] = []
    unaligned: list[tuple[str, date]] = []
    price_gaps: list[tuple[str, date, Decimal]] = []
    for symbol in expected:
        frame = bars_by_symbol.get(symbol)
        if frame is None:
            raise _fail(f"hourly bars are missing symbol={symbol}")
        previous_close: Decimal | None = None
        for trading_date in dates:
            row = history.get((symbol, trading_date))
            supplement = supplements.get((symbol, trading_date))
            if row is None:
                row = supplement
            elif any(
                row.get(field) in (None, "")
                for field in ("trading_status", "limit_status")
            ):
                row = {
                    "close": row.get("close"),
                    "trading_status": (
                        row.get("trading_status")
                        if row.get("trading_status") not in (None, "")
                        else (
                            supplement.get("trading_status") if supplement else None
                        )
                    ),
                    "limit_status": (
                        row.get("limit_status")
                        if row.get("limit_status") not in (None, "")
                        else (
                            supplement.get("limit_status") if supplement else None
                        )
                    ),
                }
            if row is None:
                raise _fail(f"missing symbol-date rule for {symbol} {trading_date}")

            trading_status = row.get("trading_status")
            limit_status = row.get("limit_status")
            suspended = _status_is_suspended(trading_status, limit_status)
            if suspended:
                rules.append(
                    DailyMarketRule(
                        symbol=symbol,
                        trading_date=trading_date,
                        suspended=True,
                    )
                )
                # A halt legitimately lets the price gap on resumption, so it
                # breaks the chain rather than producing a false positive.
                previous_close = None
                continue

            state = _limit_state(limit_status, symbol, trading_date)

            official_close = _numeric_close(row.get("close"), symbol, trading_date)
            observed_status = str(trading_status or "").strip()
            if observed_status != "交易":
                # Split from the missing-close case on purpose: these two fail
                # for opposite reasons and want opposite responses, and one
                # message covering both sent operators looking at the data feed
                # when the actual cause was a status string nobody had seen.
                raise _fail(
                    f"unexpected trading status={observed_status[:40]!r} for "
                    f"{symbol} {trading_date}"
                )
            if official_close is None:
                raise _fail(
                    f"active rule has no official close for {symbol} {trading_date}"
                )
            _require_price_tick(official_close, tick, symbol, trading_date)
            if previous_close is not None and previous_close > 0:
                move = abs(official_close - previous_close) / previous_close
                if move > _CORPORATE_ACTION_GAP:
                    price_gaps.append((symbol, trading_date, move))
            previous_close = official_close
            final_bar = _final_bar_for_date(frame, trading_date)
            if final_bar is None:
                # ``required_dates`` is the union across the universe, so this
                # is one symbol's gap on a date its neighbours traded — which
                # the engine already tolerates (50-bar minimum, common-index
                # start, 80%-coverage bar filter). Failing the load here would
                # reject universes that run today. Carry the observation
                # ungated; the symbol's own bar dates below still get the full
                # alignment check, so a wholesale contract break still trips.
                unaligned.append((symbol, trading_date))
                rules.append(
                    DailyMarketRule(
                        symbol=symbol,
                        trading_date=trading_date,
                        suspended=False,
                        closing_limit_state=state,
                        official_close_price=official_close,
                        final_bar_timestamp=None,
                    )
                )
                continue
            bar_close = _numeric_close(_bar_close(frame, final_bar), symbol, trading_date)
            if bar_close is not None:
                _require_price_tick(bar_close, tick, symbol, trading_date)
            if bar_close is None or not _same_price_tick(bar_close, official_close, tick):
                raise _fail(f"daily close does not match final hourly bar for {symbol} {trading_date}")
            rules.append(
                DailyMarketRule(
                    symbol=symbol,
                    trading_date=trading_date,
                    suspended=False,
                    closing_limit_state=state,
                    official_close_price=official_close,
                    final_bar_timestamp=final_bar,
                )
            )
    if price_gaps:
        sample = ", ".join(
            f"{symbol} {trading_date} {move:.0%}"
            for symbol, trading_date, move in price_gaps[:5]
        )
        print(
            f"   ⚠️  {len(price_gaps)} overnight move(s) exceed every A-share "
            f"daily limit band, which trading cannot produce — these prices are "
            f"unadjusted, so a 除权除息 (ex-rights/ex-dividend) date will show in "
            f"the equity curve and in the buy-and-hold baseline as a loss that "
            f"did not happen: {sample}"
            f"{' …' if len(price_gaps) > 5 else ''}"
        )
    if unaligned:
        # Absent is not the same as broken: say so, or a run whose rules never
        # gated anything looks exactly like one whose rules all held.
        sample = ", ".join(
            f"{symbol} {trading_date}" for symbol, trading_date in unaligned[:5]
        )
        print(
            f"   ⚠️  {len(unaligned)} active symbol-date(s) have no hourly bars "
            f"and cannot be closing-gated: {sample}"
            f"{' …' if len(unaligned) > 5 else ''}"
        )
    return MarketRuleCalendar(rules)
