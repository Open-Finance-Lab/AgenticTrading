"""Integrity guard for the committed seed database (``dashboard/storage/data/backtest.db``).

That file is not a fixture — on the live Render service (free tier, no persistent
disk, ``DATABASE_PATH`` unset) it *is* the running database. Whatever is committed
here is what prod serves, and whatever is missing here is missing from prod.

Two opposite mistakes ship from that, and neither shows up anywhere else in CI
because ``conftest.py`` deliberately points ``DATABASE_PATH`` at a temp file so the
suite never touches this one:

* **Deleting rows prod needs.** The seven LLM leaderboard entries carry
  ``"auto_compute": false`` in ``dashboard/config/leaderboard.json``, so
  ``ensure_leaderboard_runs()`` never regenerates them — they exist in prod *only*
  because their ``lb_*`` rows ride along in this file. ``get_leaderboard()`` skips
  a strategy whose run it cannot find, so losing them degrades the board silently:
  no error, no 500, just a shorter list. Same for the three runs
  ``dashboard/config/defaults.json`` names as the dashboard's default comparison.
* **Adding rows prod must not carry.** Running the app locally writes real accounts,
  sessions and agent keys into this same file (``CREATE TABLE IF NOT EXISTS`` against
  ``DATABASE_PATH``), and a later seed refresh commits them. ``auth_sessions`` holds
  session digests (and historically held plaintext bearer tokens), and
  ``broker_connections`` holds Robinhood OAuth credentials — neither belongs in a
  public repository.

Read-only and ``immutable=1`` on purpose: this DB is in WAL mode, so an ordinary
connection would leave ``-wal``/``-shm`` sidecars in the working tree just by
looking at it.
"""

import json
import sqlite3
from typing import Any, Dict, Iterator, List

import pytest

from dashboard.backend.paths import CONFIG_DIR, DEFAULT_DB_PATH

# Tables that only ever hold *user* data. The seed exists to carry backtest runs
# and their equity curves; every one of these is populated by someone using the
# app, never by a seed refresh, so a non-zero count means local state leaked in.
_ACCOUNT_TABLES = (
    "users",
    "auth_sessions",
    "external_agents",
    "agent_versions",
    "user_portfolios",
    "broker_connections",
)


@pytest.fixture(scope="module")
def seed_db() -> Iterator[sqlite3.Connection]:
    if not DEFAULT_DB_PATH.exists():
        pytest.fail(
            f"the committed seed database is missing at {DEFAULT_DB_PATH}.\n"
            "It is prod's database, not a local artifact — restore it with\n"
            "  git checkout -- dashboard/storage/data/backtest.db"
        )
    conn = sqlite3.connect(f"file:{DEFAULT_DB_PATH}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _tables(conn: sqlite3.Connection) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _leaderboard_config() -> Dict[str, Any]:
    return json.loads((CONFIG_DIR / "leaderboard.json").read_text(encoding="utf-8"))


def _manual_deploy_strategies(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entries that ``ensure_leaderboard_runs()`` will not recompute on demand."""
    return [s for s in config.get("strategies", []) if not s.get("auto_compute", True)]


def test_seed_database_carries_no_account_data(seed_db):
    """Local accounts, sessions and API keys must never ride along into git."""
    present = _tables(seed_db)
    populated = {}
    for table in _ACCOUNT_TABLES:
        if table not in present:
            continue
        count = seed_db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count:
            populated[table] = count

    assert not populated, (
        f"the committed seed database holds live account data: {populated}.\n"
        "Running the app locally writes accounts, session tokens and broker "
        "credentials into this file, and committing it publishes them.\n"
        "If this is uncommitted local state, restore the file:\n"
        "  git checkout -- dashboard/storage/data/backtest.db\n"
        "If you are refreshing the seed, clear these tables in an offline copy "
        "first (VACUUM INTO, delete, VACUUM INTO again) — never edit in place."
    )


def test_seed_database_ships_the_default_runs(seed_db):
    """``defaults.json`` names the dashboard's default comparison runs."""
    defaults = json.loads((CONFIG_DIR / "defaults.json").read_text(encoding="utf-8"))
    missing = [
        f"{slot}={run_id}"
        for slot, run_id in defaults.get("defaultRuns", {}).items()
        if seed_db.execute(
            "SELECT 1 FROM agent_runs WHERE run_id = ?", (run_id,)
        ).fetchone() is None
    ]
    assert not missing, (
        f"defaults.json points at runs the seed database does not contain: {missing}.\n"
        "The dashboard loads these on first visit; without them it opens empty. "
        "Either restore the seed or update dashboard/config/defaults.json."
    )


def test_seed_database_ships_the_manual_leaderboard_entries(seed_db):
    """The LLM board is precomputed — nothing regenerates it at request time.

    This also keeps the credential check above from passing vacuously: an empty
    or deleted database trivially has zero account rows.
    """
    config = _leaderboard_config()
    manual = _manual_deploy_strategies(config)
    assert manual, "leaderboard.json declares no manual-deploy entries — check the fixture"

    missing = []
    for strategy in manual:
        # Same lookup ``domain/leaderboard/service.py::_find_cached_run`` performs.
        row = seed_db.execute(
            "SELECT run_id FROM agent_runs WHERE mode = 'leaderboard' AND session_id = ?"
            " AND start_date = ? AND end_date = ? AND llm_model = ?",
            (
                config["session_id"],
                config["start_date"],
                config["end_date"],
                strategy["id"],
            ),
        ).fetchone()
        if row is None:
            missing.append(strategy["id"])
            continue
        points = seed_db.execute(
            "SELECT COUNT(*) FROM equity_timeseries WHERE run_id = ?", (row["run_id"],)
        ).fetchone()[0]
        if not points:
            missing.append(f"{strategy['id']} (run present, equity curve empty)")

    assert not missing, (
        f"the seed database is missing precomputed leaderboard entries: {missing}.\n"
        'These carry "auto_compute": false, so nothing recomputes them on request — '
        "they simply disappear from the board in prod, with no error anywhere.\n"
        "Redeploy them with dashboard/scripts/deploy_leaderboard_model.py, or "
        "restore the seed: git checkout -- dashboard/storage/data/backtest.db"
    )
