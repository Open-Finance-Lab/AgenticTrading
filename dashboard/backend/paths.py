"""Path resolution for the Agentic Trading Lab dashboard application."""

import os
import sys
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BACKEND_DIR.parent
REPO_ROOT = DASHBOARD_DIR.parent
VENV_DIR = REPO_ROOT / ".venv"


def venv_python(venv_dir: Path) -> Optional[Path]:
    """The interpreter inside ``venv_dir``, or None when it has none for this OS.

    The layout is picked by ``os.name`` and then probed, never assumed: a
    checkout shared between WSL and Windows (this repo under OneDrive is one)
    can hold a venv the other OS built, whose ``.venv`` exists but whose
    interpreter for this OS does not -- ``.venv/bin/python3`` on Windows was
    PR #653's ``WinError 2``. ``bin/python.exe`` is MSYS2's layout, which
    reports ``nt`` too. A Windows-built ``Scripts/python.exe`` is never
    returned on POSIX, where it exists as a file but cannot run.
    """
    if os.name == "nt":
        candidates = (
            venv_dir / "Scripts" / "python.exe",
            venv_dir / "bin" / "python.exe",
        )
    else:
        candidates = (venv_dir / "bin" / "python3", venv_dir / "bin" / "python")
    return next((path for path in candidates if path.is_file()), None)


def child_python() -> str:
    """Interpreter for the backtest scripts the backend runs as subprocesses.

    The server's own interpreter, not the repo venv: it has already imported
    everything those scripts need, whether it was launched from a venv, conda
    or the Docker image's system Python. Preferring ``.venv`` ran the child on
    a different environment from the server whenever one was present, and on
    a path that did not exist whenever it had the other OS's layout. The venv
    is only consulted for an embedded host that reports no ``sys.executable``.
    """
    if sys.executable:
        return sys.executable
    fallback = venv_python(VENV_DIR)
    if fallback is None:
        raise RuntimeError(
            "No Python interpreter for backtest subprocesses: sys.executable "
            f"is empty and {VENV_DIR} has none"
        )
    return str(fallback)


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
