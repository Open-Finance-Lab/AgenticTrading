"""Phase 1B guard: backend modules must have a single canonical identity.

After importing the application through the canonical package
(``dashboard.backend.app``), critical backend modules must exist ONLY under
their ``dashboard.backend.*`` names — never also as flat top-level modules
(e.g. both ``database`` and ``dashboard.backend.database``). Two identities for
a module would mean two copies of its import-time singleton state.
"""

import sys

import dashboard.backend.app as _app  # Load full module graph for canonical identity checks

CRITICAL_MODULES = [
    "database",
    "paths",
    "cache",
    # Phase 3A1 moved the agent repositories into the canonical domain package.
    "domain.agents.repository",
    "domain.agents.version_repository",
    # Phase 3B1 moved the run repository/service into the canonical domain package.
    "domain.runs.repository",
    "domain.runs.service",
    "users",
]


def test_no_duplicate_backend_module_identities():
    for name in CRITICAL_MODULES:
        canonical = sys.modules.get(f"dashboard.backend.{name}")
        assert canonical is not None, (
            f"dashboard.backend.{name} was not imported via the canonical package"
        )
        flat = sys.modules.get(name)
        # A flat entry is tolerated only if it is the *same* object; prefer none.
        assert flat is None or flat is canonical, (
            f"module '{name}' is loaded under two identities: "
            f"flat={flat!r} vs dashboard.backend.{name}={canonical!r}"
        )


def test_critical_modules_have_no_flat_entry():
    # Stronger assertion: the flat names should not exist at all.
    leaked = [name for name in CRITICAL_MODULES if name in sys.modules]
    assert not leaked, f"flat module identities leaked into sys.modules: {leaked}"


def test_database_singleton_is_shared():
    import dashboard.backend.database as database_mod
    import dashboard.backend.app as app_mod

    # The db object used by the API layer must be the exact singleton exported
    # by dashboard.backend.database.
    assert app_mod.db is database_mod.db
