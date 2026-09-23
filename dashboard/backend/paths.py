"""Path resolution for the Agentic Trading Lab dashboard application."""

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BACKEND_DIR.parent
REPO_ROOT = DASHBOARD_DIR.parent

DATA_DIR = DASHBOARD_DIR / "storage" / "data"

# On-disk bar cache (see infrastructure/market_data/bar_cache.py). A fresh
# sibling of `cache/`, never that directory: `cache/` still holds nine
# git-tracked orphan CSVs, and gitignore does not untrack what is already
# tracked. `.gitignore` ignores `dashboard/storage/data` wholesale, so nothing
# written here can be staged by accident.
BAR_CACHE_DIR = DATA_DIR / "bar_cache"

BACKUPS_DIR = DASHBOARD_DIR / "storage" / "backups"
CONFIG_DIR = DASHBOARD_DIR / "config"
SCRIPTS_DIR = DASHBOARD_DIR / "scripts"
FRONTEND_DIR = DASHBOARD_DIR / "frontend"
CREDENTIALS_DIR = REPO_ROOT / "credentials"

DEFAULT_DB_PATH = DATA_DIR / "backtest.db"
