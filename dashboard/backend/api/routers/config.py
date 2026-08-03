"""Default-configuration route (Phase 3D4A).

Moved verbatim from ``dashboard/backend/app.py``. The external path
``/config/defaults`` and its behavior are unchanged; registered directly on the app.
"""

from fastapi import APIRouter

from dashboard.backend.infrastructure.market_data.provider import (
    ifind_ashare_enabled,
    vnpy_simulation_enabled,
)
from dashboard.backend.paths import CONFIG_DIR

router = APIRouter()


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
