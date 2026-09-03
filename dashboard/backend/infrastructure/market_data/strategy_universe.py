"""Backend-owned strategy pools resolved from a frozen security catalog.

Selection happens before scheduling. Workers receive the exact selected symbols
and provenance, so a catalog refresh cannot change an already queued run.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Literal, Mapping

from dashboard.backend.paths import CONFIG_DIR
StockPool = Literal["ordinary", "fund", "large_cap", "small_mid_cap", "all"]
PoolMode = Literal["top30", "all", "representative30"]
STOCK_POOLS = ("ordinary", "fund", "large_cap", "small_mid_cap", "all")
POOL_MODES = ("top30", "all", "representative30")
POOL_LIMIT = 30
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_FUND_TYPES = {"Closed-End Fund", "Open-End Fund", "Mutual Fund", "ETP", "ETF"}


def _is_true(value: Any) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() == "true"
    )


class UniverseConfigurationError(ValueError):
    """Required reference data or server policy is missing or invalid."""


def universe_policy() -> dict[str, Any]:
    try:
        policy = json.loads((CONFIG_DIR / "strategy_universes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UniverseConfigurationError("Cannot read strategy_universes.json") from exc
    if not isinstance(policy, dict):
        raise UniverseConfigurationError("strategy_universes.json must contain an object")
    boundary = policy.get("large_cap_min_usd")
    if isinstance(boundary, bool) or not isinstance(boundary, (int, float)) or not math.isfinite(boundary) or boundary <= 0:
        raise UniverseConfigurationError("large_cap_min_usd must be finite and positive")
    return policy


def _reference_path(policy: Mapping[str, Any], key: str, env: str) -> Path | None:
    value = os.getenv(env) or policy.get(key)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else CONFIG_DIR / path


def _read_catalog(path: Path | None, env: str) -> tuple[list[dict[str, str]], str]:
    if path is None or not path.is_file():
        raise UniverseConfigurationError(f"Security reference data unavailable; configure {env}")
    try:
        content = path.read_bytes()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    except (OSError, UnicodeError) as exc:
        raise UniverseConfigurationError(f"Cannot read {env} CSV") from exc
    if not reader.fieldnames or "symbol" not in reader.fieldnames:
        raise UniverseConfigurationError(f"{env} CSV requires a symbol column")
    records = list(reader)
    seen = set()
    for row in records:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            raise UniverseConfigurationError(f"{env} has empty or duplicate symbols")
        row["symbol"] = symbol
        seen.add(symbol)
    return records, hashlib.sha256(content).hexdigest()


def _market_cap(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return None
    return cap if math.isfinite(cap) and cap > 0 else None


def select_strategy_universe(
    records: list[Mapping[str, Any]],
    stock_pool: str,
    pool_mode: str = "top30",
    *,
    large_cap_min_usd: float = 10_000_000_000,
) -> dict[str, Any]:
    """Filter first, then deterministically take 30 or all eligible symbols.

    All means ordinary shares plus exchange-traded products / recognized funds.
    Missing capitalization is never inferred from price or company name.
    """
    if stock_pool not in STOCK_POOLS:
        raise ValueError(f"Unknown stock_pool {stock_pool!r}; choose from {STOCK_POOLS}")
    if pool_mode not in {"top30", "all"}:
        raise ValueError("Catalog filtering supports top30 or all; representative30 uses the versioned roster")
    if not math.isfinite(large_cap_min_usd) or large_cap_min_usd <= 0:
        raise ValueError("large_cap_min_usd must be finite and positive")
    eligible = []
    missing_caps = 0
    seen = set()
    for row in records:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            raise UniverseConfigurationError("Catalog must contain unique non-empty symbols")
        seen.add(symbol)
        if row.get("status") != "active" or not _is_true(row.get("tradable")) or row.get("exchange") == "OTC":
            continue
        category = row.get("classification")
        ordinary = category == "ordinary_share"
        types = set(str(row.get("figi_security_types") or "").split("|"))
        fund = category in {"fund", "etp"} or (
            category == "other_security_type" and bool(types) and types <= _FUND_TYPES
        )
        if not (ordinary or fund):
            continue
        if stock_pool == "ordinary" and not ordinary:
            continue
        if stock_pool == "fund" and not fund:
            continue
        if stock_pool in {"large_cap", "small_mid_cap"}:
            if not ordinary:
                continue
            cap = _market_cap(row.get("market_cap_usd"))
            if cap is None:
                missing_caps += 1
                continue
            if (cap >= large_cap_min_usd) != (stock_pool == "large_cap"):
                continue
        if not _SYMBOL.fullmatch(symbol):
            raise UniverseConfigurationError(f"Unsupported catalog ticker {symbol!r}")
        eligible.append(symbol)
    # A partial capitalization catalog would make 'all eligible' misleading.
    if missing_caps:
        raise UniverseConfigurationError(
            f"{missing_caps} ordinary shares lack valid market_cap_usd; "
            "complete the catalog or configure US_EQUITY_MARKET_CAPS"
        )
    eligible.sort()
    if not eligible:
        raise ValueError(f"No eligible securities for stock_pool={stock_pool!r}")
    symbols = eligible[:POOL_LIMIT] if pool_mode == "top30" else eligible
    return {
        "stock_pool": stock_pool,
        "pool_mode": pool_mode,
        "symbols": symbols,
        "eligible_count": len(eligible),
        "selected_count": len(symbols),
        "selection_order": "symbol_asc",
        "large_cap_min_usd": large_cap_min_usd,
        "catalog_policy": "current_snapshot_not_point_in_time",
    }


def resolve_strategy_universe(stock_pool: str, pool_mode: str = "top30") -> dict[str, Any]:
    if stock_pool not in STOCK_POOLS or pool_mode not in POOL_MODES:
        raise ValueError("Invalid stock_pool or pool_mode")
    if pool_mode == "representative30":
        return resolve_representative_universe(stock_pool)
    policy = universe_policy()
    records, catalog_hash = _read_catalog(
        _reference_path(policy, "catalog_path", "US_EQUITY_CATALOG"), "US_EQUITY_CATALOG"
    )
    provenance = {"catalog_sha256": catalog_hash}
    if stock_pool in {"large_cap", "small_mid_cap"}:
        cap_path = _reference_path(policy, "market_caps_path", "US_EQUITY_MARKET_CAPS")
        if cap_path is not None:
            caps, cap_hash = _read_catalog(cap_path, "US_EQUITY_MARKET_CAPS")
            by_symbol = {row["symbol"]: row.get("market_cap_usd") for row in caps}
            for row in records:
                if row["symbol"] in by_symbol:
                    row["market_cap_usd"] = by_symbol[row["symbol"]]
            provenance["market_caps_sha256"] = cap_hash
    selection = select_strategy_universe(
        records, stock_pool, pool_mode, large_cap_min_usd=policy["large_cap_min_usd"]
    )
    selection.update(provenance)
    selection["snapshot_asof"] = sorted({row["snapshot_asof"] for row in records if row.get("snapshot_asof")})
    return selection


def representative_presets() -> list[dict[str, Any]]:
    """Expose the same versioned rosters that workers will execute."""
    return [resolve_representative_universe(pool) for pool in ("ordinary", "fund", "all")]


def resolve_representative_universe(stock_pool: str) -> dict[str, Any]:
    if stock_pool not in {"ordinary", "fund", "all"}:
        raise ValueError("representative30 supports ordinary, fund and all")
    try:
        content = (CONFIG_DIR / "representative_universes.json").read_bytes()
        registry = json.loads(content)
        preset = registry["presets"][stock_pool]
        groups = preset["groups"]
        symbols = [symbol for group in groups for symbol in group["symbols"]]
        # Validate the roster against its shipped reference snapshot, so this
        # small preset works on hosts without the full workstation CSV catalog.
        eligible = select_strategy_universe(registry["securities"], stock_pool, "all")
        if len(symbols) != POOL_LIMIT or len(set(symbols)) != POOL_LIMIT:
            raise ValueError("Representative presets require exactly 30 unique symbols")
        if set(symbols) - set(eligible["symbols"]):
            raise ValueError("Representative preset contains symbols outside its category")
        selection = {
            **eligible,
            "pool_mode": "representative30",
            "symbols": symbols,
            "selected_count": len(symbols),
            "name": preset["name"],
            "description": preset["description"],
            "groups": groups,
            "selection_order": "curated_coverage",
            "roster_version": registry["version"],
            "catalog_sha256": hashlib.sha256(content).hexdigest(),
            "source_catalog_sha256": registry["source_catalog_sha256"],
            "catalog_scope": "representative_reference_snapshot",
            "snapshot_asof": [registry["snapshot_asof"]],
        }
        return validate_selection(selection)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise UniverseConfigurationError("Representative stock pools are unavailable or invalid") from exc


def validate_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a worker handoff without resolving a changing catalog again."""
    if not isinstance(selection, Mapping):
        raise ValueError("Universe selection must be an object")
    if selection.get("stock_pool") not in STOCK_POOLS or selection.get("pool_mode") not in POOL_MODES:
        raise ValueError("Invalid universe selection")
    symbols = selection.get("symbols")
    if not isinstance(symbols, list) or not symbols or any(not isinstance(s, str) or not _SYMBOL.fullmatch(s) for s in symbols):
        raise ValueError("Universe selection requires valid symbols")
    if len(set(symbols)) != len(symbols) or selection.get("selected_count") != len(symbols):
        raise ValueError("Universe selection count or symbols are inconsistent")
    eligible_count = selection.get("eligible_count")
    if type(eligible_count) is not int or eligible_count < len(symbols):
        raise ValueError("Universe selection eligible_count is inconsistent")
    if selection["pool_mode"] == "top30" and len(symbols) != min(eligible_count, POOL_LIMIT):
        raise ValueError("top30 universe must contain min(eligible_count, 30) symbols")
    if selection["pool_mode"] == "all" and eligible_count != len(symbols):
        raise ValueError("All mode must include every eligible symbol")
    if selection["pool_mode"] == "representative30" and (
        len(symbols) != POOL_LIMIT or selection["stock_pool"] not in {"ordinary", "fund", "all"}
    ):
        raise ValueError("Representative mode requires a supported category with 30 symbols")
    return {**selection, "symbols": list(symbols)}
