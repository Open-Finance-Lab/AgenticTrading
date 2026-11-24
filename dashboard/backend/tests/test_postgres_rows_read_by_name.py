"""Postgres stores read rows by column name, never by position.

Every pool here hands out ``dict_row`` rows (``db_pool.py``), so ``row[0]`` is a
``KeyError`` on Postgres, while the SQLite twin's ``sqlite3.Row`` accepts both
and passes. That is how the startup reclaimer failed on every prod boot for a
month with the suite green (#601): the SQLite half was tested, the Postgres half
was not, and the bug lived only in the half that was not.

The check is a per-function flow over the names a fetch binds, not a ban on
every ``[0]``: ``rows[-1]`` (the last row of a page) indexes a list and is
legitimate, and the stores use it. It does not follow a row through a helper's
arguments; a positional read there is the one shape it cannot see.
"""

import ast
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

# The one dual-backend store whose Postgres half is not a *_postgres.py file
# (CLAUDE.md, Persistence). Only that class is checked: the module's SQLite half
# reads sqlite3.Row, where a positional read is legal.
_INLINE_POSTGRES_CLASSES = {
    "domain/agents/research_store.py": "_PostgresResearchStore",
}


def _checked_trees():
    for path in sorted(BACKEND.rglob("*_postgres.py")):
        if "tests" in path.relative_to(BACKEND).parts:
            continue
        yield path, ast.parse(path.read_text(encoding="utf-8"))
    for relative, class_name in _INLINE_POSTGRES_CLASSES.items():
        path = BACKEND / relative
        module = ast.parse(path.read_text(encoding="utf-8"))
        [cls] = [
            node for node in ast.walk(module)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        yield path, cls


def _int_index(node):
    index = node.slice
    if isinstance(index, ast.UnaryOp) and isinstance(index.op, ast.USub):
        index = index.operand
    return (
        isinstance(index, ast.Constant)
        and isinstance(index.value, int)
        and not isinstance(index.value, bool)
    )


def _fetch_call(node, *methods):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in methods
    )


def _positional_row_reads(tree):
    """Integer subscripts on a value that is a row, per function.

    A row is ``fetchone()``'s result, an element of ``fetchall()``'s, or a loop
    variable over either (or over the cursor itself). ``rows[-1]`` indexes the
    list of rows and is fine; ``rows[-1][0]`` indexes a row and is not.
    """
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for scope in scopes:
        row_names, rows_names = set(), set()
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    if _fetch_call(node.value, "fetchone"):
                        row_names.add(target.id)
                    elif _fetch_call(node.value, "fetchall", "fetchmany"):
                        rows_names.add(target.id)

        def is_rows(expr):
            return (
                (isinstance(expr, ast.Name) and expr.id in rows_names)
                or _fetch_call(expr, "fetchall", "fetchmany")
                or (isinstance(expr, ast.Name) and expr.id in {"cur", "cursor"})
            )

        for node in ast.walk(scope):
            loops = []
            if isinstance(node, (ast.For, ast.AsyncFor)):
                loops = [(node.target, node.iter)]
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                loops = [(g.target, g.iter) for g in node.generators]
            for target, iterable in loops:
                if isinstance(target, ast.Name) and is_rows(iterable):
                    row_names.add(target.id)

        def is_row(expr):
            return (
                (isinstance(expr, ast.Name) and expr.id in row_names)
                or _fetch_call(expr, "fetchone")
                or (isinstance(expr, ast.Subscript) and _int_index(expr) and is_rows(expr.value))
            )

        for node in ast.walk(scope):
            if isinstance(node, ast.Subscript) and _int_index(node) and is_row(node.value):
                yield node


def test_the_guard_sees_every_postgres_store():
    # 13 twin modules plus the inline research class as of 2026-10-29. A floor,
    # not an equality: a new twin must not need this file edited to be covered.
    assert len(list(_checked_trees())) >= 14


def test_no_postgres_store_reads_a_row_by_position():
    offenders = [
        f"{path.relative_to(BACKEND)}:{node.lineno}: {ast.unparse(node)}"
        for path, tree in _checked_trees()
        for node in _positional_row_reads(tree)
    ]
    assert offenders == [], (
        "dict_row rows have no positions; read the column by name:\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_catches_the_601_shapes_and_spares_lists_of_rows():
    tree = ast.parse(
        "def get(cur):\n"
        "    row = cur.fetchone()\n"
        "    a = row[0]\n"
        "    b = cur.fetchone()[0]\n"
        "    rows = cur.fetchall()\n"
        "    for r in rows:\n"
        "        c = r[1]\n"
        "    d = [x[2] for x in cur.fetchall()]\n"
        "    e = rows[-1][0]\n"
        "    last = rows[-1]\n"
        "    f = last['id']\n"
        "    g = row['payload']\n"
    )
    found = sorted(ast.unparse(n) for n in _positional_row_reads(tree))
    assert found == sorted(
        ["row[0]", "cur.fetchone()[0]", "r[1]", "x[2]", "rows[-1][0]"]
    )
