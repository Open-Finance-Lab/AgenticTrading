"""Point-in-time US equity metadata for backtest decision bars.

The runtime keeps this adapter optional: ordinary Alpaca backtests still run when
no local metadata dataset is configured.  Once ``US_EQUITY_DATASET_PATH`` (or
``US_EQUITY_DATA_ROOT``) points at the dataset built by the standalone download
scripts, decision bars receive SEC SIC classification and a no-lookahead market
capitalization calculated from the decision price and then-effective shares.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


DATASET_PATH_ENV = "US_EQUITY_DATASET_PATH"
DATA_ROOT_ENV = "US_EQUITY_DATA_ROOT"
DEFAULT_DATASET_NAME = "us_equities_5min_2016_2025"

_CAP_COLUMNS = [
    "symbol",
    "trade_date",
    "market_cap_usd",
    "market_cap_status",
    "shares_outstanding_effective",
]
_INDUSTRY_COLUMNS = [
    "symbol",
    "effective_from",
    "effective_to_exclusive",
    "sic",
    "sic_description",
    "sic_division_name",
]


class EquityMetadataUnavailableError(RuntimeError):
    """Raised when explicitly configured metadata cannot be read safely."""


def configured_dataset_path(dataset_path: str | Path | None = None) -> Path | None:
    """Resolve the optional external metadata dataset without guessing a host path."""

    value = dataset_path or os.getenv(DATASET_PATH_ENV)
    if value:
        return Path(value).expanduser().resolve()
    root = os.getenv(DATA_ROOT_ENV)
    if root:
        return (Path(root).expanduser() / DEFAULT_DATASET_NAME).resolve()
    return None


def _local_dates(index: pd.Index, timezone: str) -> list[Any]:
    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(timezone)
    else:
        timestamps = timestamps.tz_convert(timezone)
    return list(timestamps.date)


def _valid_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def enrich_decision_bars(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    market_caps: pd.DataFrame | None = None,
    industries: pd.DataFrame | None = None,
    timezone: str = "US/Eastern",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Attach point-in-time market cap and SIC fields to decision bars.

    The daily metadata's own ``market_cap_usd`` uses the final close of that
    session.  It is used only as a validity flag here.  Intraday market cap is
    recomputed as ``decision_close * shares_outstanding_effective`` so an hourly
    agent never sees that later close.
    """

    caps = market_caps if market_caps is not None else pd.DataFrame(columns=_CAP_COLUMNS)
    sectors = industries if industries is not None else pd.DataFrame(columns=_INDUSTRY_COLUMNS)
    cap_lookup: dict[tuple[str, Any], dict[str, Any]] = {}
    if not caps.empty:
        missing = set(_CAP_COLUMNS) - set(caps.columns)
        if missing:
            raise EquityMetadataUnavailableError(
                f"Market-cap metadata is missing columns: {sorted(missing)}"
            )
        normalized = caps.copy()
        normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
        normalized["trade_date"] = pd.to_datetime(
            normalized["trade_date"], errors="coerce"
        ).dt.date
        if normalized.duplicated(["symbol", "trade_date"]).any():
            raise EquityMetadataUnavailableError(
                "Market-cap metadata has duplicate symbol/trade_date rows"
            )
        cap_lookup = {
            (row["symbol"], row["trade_date"]): row
            for row in normalized.to_dict("records")
        }

    industry_lookup: dict[str, list[dict[str, Any]]] = {}
    if not sectors.empty:
        missing = set(_INDUSTRY_COLUMNS) - set(sectors.columns)
        if missing:
            raise EquityMetadataUnavailableError(
                f"Industry metadata is missing columns: {sorted(missing)}"
            )
        normalized = sectors.copy()
        normalized["symbol"] = normalized["symbol"].astype(str).str.upper()
        for field in ("effective_from", "effective_to_exclusive"):
            normalized[field] = pd.to_datetime(
                normalized[field], errors="coerce"
            ).dt.date
        normalized = normalized.sort_values(["symbol", "effective_from"])
        for symbol, rows in normalized.groupby("symbol", sort=False):
            records = rows.to_dict("records")
            prior_end = None
            for record in records:
                start = record["effective_from"]
                end = record["effective_to_exclusive"]
                if pd.isna(start) or pd.isna(end) or start >= end:
                    raise EquityMetadataUnavailableError(
                        f"Industry metadata has an invalid interval for {symbol}"
                    )
                if prior_end is not None and start < prior_end:
                    raise EquityMetadataUnavailableError(
                        f"Industry metadata has overlapping intervals for {symbol}"
                    )
                prior_end = end
            industry_lookup[str(symbol)] = records

    enriched: dict[str, pd.DataFrame] = {}
    market_cap_values = 0
    industry_values = 0
    total_rows = 0
    for raw_symbol, source in bars_by_symbol.items():
        symbol = str(raw_symbol).upper()
        frame = source.copy()
        dates = _local_dates(frame.index, timezone)
        total_rows += len(frame)

        if cap_lookup:
            values = []
            statuses = []
            for decision_date, close in zip(dates, frame["close"]):
                reference = cap_lookup.get((symbol, decision_date))
                status = reference.get("market_cap_status") if reference else None
                shares = (
                    _valid_number(reference.get("shares_outstanding_effective"))
                    if reference
                    else None
                )
                reference_cap = (
                    _valid_number(reference.get("market_cap_usd"))
                    if reference
                    else None
                )
                decision_close = _valid_number(close)
                value = (
                    decision_close * shares
                    if reference_cap is not None
                    and shares is not None
                    and shares > 0
                    and decision_close is not None
                    else None
                )
                values.append(value)
                statuses.append(
                    str(status) if status is not None and pd.notna(status) else None
                )
                market_cap_values += value is not None
            frame["market_cap_usd"] = values
            frame["market_cap_status"] = statuses

        intervals = industry_lookup.get(symbol)
        if intervals:
            sic_values = []
            industry_values_for_rows = []
            sector_values = []
            for decision_date in dates:
                match = next(
                    (
                        row
                        for row in reversed(intervals)
                        if row["effective_from"] <= decision_date
                        < row["effective_to_exclusive"]
                    ),
                    None,
                )
                sic = _valid_number(match.get("sic")) if match else None
                industry = match.get("sic_description") if match else None
                sector = match.get("sic_division_name") if match else None
                sic_values.append(int(sic) if sic is not None else None)
                industry_values_for_rows.append(
                    str(industry) if industry is not None and pd.notna(industry) else None
                )
                sector_values.append(
                    str(sector) if sector is not None and pd.notna(sector) else None
                )
                industry_values += match is not None
            frame["sic_code"] = sic_values
            frame["industry"] = industry_values_for_rows
            frame["sector"] = sector_values

        enriched[str(raw_symbol)] = frame

    return enriched, {
        "status": "available" if cap_lookup or industry_lookup else "empty",
        "classification": "SEC SIC",
        "point_in_time": True,
        "decision_rows": total_rows,
        "market_cap_rows": market_cap_values,
        "industry_rows": industry_values,
    }


def load_and_enrich_us_equity_bars(
    bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    dataset_path: str | Path | None = None,
    timezone: str = "US/Eastern",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load only requested symbols/years from the configured Parquet metadata."""

    resolved = configured_dataset_path(dataset_path)
    if resolved is None:
        return dict(bars_by_symbol), {
            "status": "not_configured",
            "classification": "SEC SIC",
            "point_in_time": True,
        }
    if not resolved.is_dir():
        raise EquityMetadataUnavailableError(
            f"Configured US equity dataset does not exist: {resolved}"
        )

    symbols = sorted({str(symbol).upper() for symbol in bars_by_symbol})
    years = sorted(
        {
            decision_date.year
            for frame in bars_by_symbol.values()
            for decision_date in _local_dates(frame.index, timezone)
        }
    )
    cap_frames = []
    try:
        for year in years:
            path = (
                resolved
                / "metadata"
                / "market_cap"
                / "daily_market_cap"
                / f"year={year}"
                / "data.parquet"
            )
            if path.is_file():
                cap_frames.append(
                    pd.read_parquet(
                        path,
                        columns=_CAP_COLUMNS,
                        filters=[("symbol", "in", symbols)],
                    )
                )

        industry_files = sorted(
            (resolved / "metadata" / "industry").glob(
                "sec_sic_effective_history_*.parquet"
            )
        )
        industries = (
            pd.read_parquet(
                industry_files[-1],
                columns=_INDUSTRY_COLUMNS,
                filters=[("symbol", "in", symbols)],
            )
            if industry_files
            else pd.DataFrame(columns=_INDUSTRY_COLUMNS)
        )
    except Exception as exc:
        raise EquityMetadataUnavailableError(
            "Could not read configured US equity metadata"
        ) from exc

    market_caps = (
        pd.concat(cap_frames, ignore_index=True)
        if cap_frames
        else pd.DataFrame(columns=_CAP_COLUMNS)
    )
    enriched, summary = enrich_decision_bars(
        bars_by_symbol,
        market_caps=market_caps,
        industries=industries,
        timezone=timezone,
    )
    # Persist only a portable dataset name; host filesystem paths are neither
    # useful to agents nor safe provenance to expose through run APIs.
    summary["dataset_name"] = resolved.name
    summary["years"] = years
    return enriched, summary
