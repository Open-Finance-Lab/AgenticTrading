"""Every global a Postgres twin reads must exist in that module.

CI runs the suite on SQLite, and the ``@pg_only`` tier covers only the methods
someone wrote a live test for. So a twin can call a helper it never imported
and stay green: ``value_repository_postgres.py`` used fifteen helpers from its
SQLite sibling without importing them, and from 2026-09-23 every daily-facts
tick in prod failed four steps with ``NameError``.

``symtable`` resolves scopes the way the compiler does, so a name counts only
when a function or class body reads it as a global -- locals, parameters and
comprehension variables never reach the check.
"""

from __future__ import annotations

import builtins
import importlib
import symtable
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND.parents[1]

TWINS = sorted(
    path
    for path in BACKEND.rglob("*_postgres.py")
    if "tests" not in path.relative_to(BACKEND).parts
)


def _globals_read_by_nested_scopes(source: str, filename: str) -> set[str]:
    names: set[str] = set()

    def walk(table: symtable.SymbolTable) -> None:
        for child in table.get_children():
            for symbol in child.get_symbols():
                if symbol.is_referenced() and symbol.is_global():
                    names.add(symbol.get_name())
            walk(child)

    walk(symtable.symtable(source, filename, "exec"))
    return names


def test_twins_are_discovered():
    # An empty parametrization would pass vacuously if the layout moved.
    assert len(TWINS) >= 12


@pytest.mark.parametrize(
    "path", TWINS, ids=[str(p.relative_to(BACKEND)) for p in TWINS]
)
def test_postgres_twin_globals_resolve(path: Path):
    module_name = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
    module = importlib.import_module(module_name)
    unresolved = sorted(
        name
        for name in _globals_read_by_nested_scopes(path.read_text(), str(path))
        if not hasattr(module, name) and not hasattr(builtins, name)
    )
    assert not unresolved, (
        f"{path.name} reads names it never defines or imports: {unresolved}. "
        "They raise NameError only on a Postgres deployment."
    )
