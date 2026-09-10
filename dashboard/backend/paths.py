"""Path resolution for the Agentic Trading Lab dashboard application."""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = BACKEND_DIR.parent
REPO_ROOT = DASHBOARD_DIR.parent

DATA_DIR = DASHBOARD_DIR / "storage" / "data"
BACKUPS_DIR = DASHBOARD_DIR / "storage" / "backups"
CONFIG_DIR = DASHBOARD_DIR / "config"
SCRIPTS_DIR = DASHBOARD_DIR / "scripts"
FRONTEND_DIR = DASHBOARD_DIR / "frontend"
CREDENTIALS_DIR = REPO_ROOT / "credentials"

DEFAULT_DB_PATH = DATA_DIR / "backtest.db"


def resolve_python_exe(venv_dir: Path) -> str:
    """Resolve the Python interpreter to run subprocesses with.

    Prefers ``<venv_dir>/Scripts/python.exe`` (Windows venv layout), then
    ``<venv_dir>/bin/python3`` (POSIX venv layout), then falls back to the
    interpreter currently running (``sys.executable``) when neither exists --
    covering both "no venv at all" and "the venv directory exists but is
    empty/broken".

    This module stays a leaf (stdlib-only imports) so every layer of the
    backend can import it without risking a cycle.
    """
    win_py = venv_dir / "Scripts" / "python.exe"
    if win_py.exists():
        return str(win_py)
    unix_py = venv_dir / "bin" / "python3"
    if unix_py.exists():
        return str(unix_py)
    return sys.executable


def resolve_env_path(value: str, *, base: Path = REPO_ROOT) -> Path:
    """Resolve a path that came from an environment variable.

    Expands a leading ``~``, joins a relative result onto ``base`` (an
    absolute path is used as given), then resolves once -- normalising
    ``..`` segments and symlinks the same way for both branches, so a
    relative and an absolute env value that name the same location produce
    identical resolved paths downstream.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
