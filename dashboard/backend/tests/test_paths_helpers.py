"""Unit tests for the shared path-resolution helpers in ``dashboard.backend.paths``.

``resolve_python_exe`` and ``resolve_env_path`` were hoisted out of duplicated
logic in ``api/routers/backtests.py``, ``domain/backtesting/algo_service.py``,
``database.py``, and ``infrastructure/ai_hedge_fund/adapter.py`` -- these tests
pin the behaviour those call sites now share.
"""

import sys

from dashboard.backend.paths import resolve_env_path, resolve_python_exe


class TestResolvePythonExe:
    def test_missing_venv_falls_back_to_sys_executable(self, tmp_path):
        venv_dir = tmp_path / ".venv"  # never created
        assert resolve_python_exe(venv_dir) == sys.executable

    def test_venv_exists_but_empty_falls_back_to_sys_executable(self, tmp_path):
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        assert resolve_python_exe(venv_dir) == sys.executable

    def test_windows_layout_scripts_python_exe(self, tmp_path):
        venv_dir = tmp_path / ".venv"
        win_py = venv_dir / "Scripts" / "python.exe"
        win_py.parent.mkdir(parents=True)
        win_py.write_bytes(b"")
        assert resolve_python_exe(venv_dir) == str(win_py)

    def test_posix_layout_bin_python3(self, tmp_path):
        venv_dir = tmp_path / ".venv"
        unix_py = venv_dir / "bin" / "python3"
        unix_py.parent.mkdir(parents=True)
        unix_py.write_bytes(b"")
        assert resolve_python_exe(venv_dir) == str(unix_py)

    def test_windows_layout_preferred_when_both_present(self, tmp_path):
        venv_dir = tmp_path / ".venv"
        win_py = venv_dir / "Scripts" / "python.exe"
        win_py.parent.mkdir(parents=True)
        win_py.write_bytes(b"")
        unix_py = venv_dir / "bin" / "python3"
        unix_py.parent.mkdir(parents=True)
        unix_py.write_bytes(b"")
        assert resolve_python_exe(venv_dir) == str(win_py)


class TestResolveEnvPath:
    def test_relative_path_joins_onto_base(self, tmp_path):
        base = tmp_path / "repo"
        base.mkdir()
        result = resolve_env_path("storage/data/backtest.db", base=base)
        assert result == (base / "storage" / "data" / "backtest.db").resolve()
        assert result.is_absolute()

    def test_relative_path_normalises_dotdot(self, tmp_path):
        base = tmp_path / "repo" / "dashboard"
        base.mkdir(parents=True)
        result = resolve_env_path("../other/backtest.db", base=base)
        assert result == (tmp_path / "repo" / "other" / "backtest.db")

    def test_absolute_path_passes_through(self, tmp_path):
        absolute = tmp_path / "somewhere" / "backtest.db"
        result = resolve_env_path(str(absolute), base=tmp_path / "unused")
        assert result == absolute.resolve()

    def test_tilde_expands_to_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        # os.path.expanduser (used by Path.expanduser) reads $HOME on POSIX.
        result = resolve_env_path("~/backtest.db", base=tmp_path / "unused")
        assert result == (tmp_path / "backtest.db").resolve()

    def test_default_base_is_repo_root(self):
        from dashboard.backend.paths import REPO_ROOT

        result = resolve_env_path("some/relative/file.db")
        assert result == (REPO_ROOT / "some" / "relative" / "file.db").resolve()
