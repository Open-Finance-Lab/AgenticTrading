"""Default-configuration route (Phase 3D4A).

Moved verbatim from ``dashboard/backend/app.py``. The external path
``/config/defaults`` and its behavior are unchanged; registered directly on the app.
"""

from fastapi import APIRouter, HTTPException

from dashboard.backend.infrastructure.market_data.provider import (
    ifind_ashare_enabled,
    vnpy_simulation_enabled,
)
from dashboard.backend.paths import CONFIG_DIR
from dashboard.backend.infrastructure.market_data.strategy_universe import (
    STOCK_POOLS, POOL_MODES, UniverseConfigurationError, representative_presets, universe_policy,
)

router = APIRouter()


@router.get("/config/stock-pools")
def get_stock_pools():
    """Backend options and previewable representative rosters for Run Backtest."""
    try:
        presets = representative_presets()
        policy = universe_policy()
    except UniverseConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "stock_pools": list(STOCK_POOLS),
        "pool_modes": list(POOL_MODES),
        "default_pool_mode": "top30",
        "selection_order": "symbol_asc",
        "large_cap_min_usd": policy["large_cap_min_usd"],
        "legacy_default": "djia_30",
        "representative_presets": presets,
    }


@router.get("/config/features")
def get_features():
    """Return optional dashboard capabilities enabled by configuration."""
    return {
        "vnpy_simulation_enabled": vnpy_simulation_enabled(),
        "ifind_ashare_enabled": ifind_ashare_enabled(),
    }


@router.get("/config/defaults")
def get_defaults():
    """
    Get default configuration for the website.
    
    Returns:
        Default run IDs and settings for initial page load
    """
    defaults_path = CONFIG_DIR / "defaults.json"
    
    if not defaults_path.exists():
        return {
            "error": "No defaults configured",
            "message": "Create dashboard/config/defaults.json to set default runs and settings"
        }
    
    import json
    with open(defaults_path, 'r') as f:
        defaults = json.load(f)
    
    return defaults
