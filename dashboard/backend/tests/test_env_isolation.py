"""The suite must never see backend-selecting env vars from the developer's shell.

conftest.py pops them at import time -- before any backend module is imported,
which is the only moment that works, since the store singletons are built during
that import.

Scope, honestly: the three `is_stripped` tests below assert the vars are absent
*while the suite runs*. On their own that is close to vacuous -- in CI the vars
are never set, so they pass whether or not conftest pops anything (this file
passes with no conftest at all). Deleting all three pops would keep the whole suite
green while silently unguarding it. So the fourth test re-runs all three in a
subprocess with hostile ambient values, where only real strips are green.

They still cannot prove the pop happens at conftest *import* time rather than in
a fixture; that placement is guarded by the comment at the pop itself.

Why this matters beyond "don't touch prod data": once the store factories exist,
an ambient CONTENT_DATABASE_URL doesn't merely redirect writes -- it swaps the singletons
for Postgres twins that have no .db_path, breaking the existing tests that assert
on it (tests/domain/agents/test_repository_move.py:39-40 and
tests/domain/strategies/test_strategy_store.py:42-43).
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# A value that must never be dialled. If the strip regresses, the subprocess
# below builds Postgres twins against this host and fails loudly rather than
# reaching anything real -- the name is the point.
_HOSTILE_URL = "postgresql://evil:pw@must-never-be-dialled.invalid/prod"


def test_users_database_url_is_stripped_for_the_suite():
    assert "USERS_DATABASE_URL" not in os.environ


def test_content_database_url_is_stripped_for_the_suite():
    assert "CONTENT_DATABASE_URL" not in os.environ


def test_agent_runs_database_url_is_stripped_for_the_suite():
    assert "AGENT_RUNS_DATABASE_URL" not in os.environ


def test_the_pop_survives_a_hostile_ambient_environment():
    """Prove the strips *work*, which the three tests above cannot.

    They only assert absence, and the vars are already absent on every runner --
    so they stay green with the pops deleted. This one sets all three vars to a
    poisoned value and re-runs all three in a subprocess: green only if
    conftest really strips them.

    A subprocess, not monkeypatch: the pops must land at conftest import time,
    because the store singletons are built during backend import. Anything
    in-process is already too late by definition, so only a fresh interpreter
    can test it. It re-runs *named* sibling tests rather than this file, which
    would recurse.
    """
    env = {
        **os.environ,
        "AGENT_RUNS_DATABASE_URL": _HOSTILE_URL,
        "CONTENT_DATABASE_URL": _HOSTILE_URL,
        "USERS_DATABASE_URL": _HOSTILE_URL,
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "dashboard/backend/tests/test_env_isolation.py::"
            + "test_users_database_url_is_stripped_for_the_suite",
            "dashboard/backend/tests/test_env_isolation.py::"
            + "test_content_database_url_is_stripped_for_the_suite",
            "dashboard/backend/tests/test_env_isolation.py::"
            + "test_agent_runs_database_url_is_stripped_for_the_suite",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "conftest did not strip ambient AGENT_RUNS_DATABASE_URL, "
        "CONTENT_DATABASE_URL, or USERS_DATABASE_URL:\n"
        f"{result.stdout}\n{result.stderr}"
    )
