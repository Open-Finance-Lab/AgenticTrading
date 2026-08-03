"""Same-origin API-base guard for the static frontend.

Production serves ``dashboard/frontend`` on Vercel and proxies backend paths
to Render via ``vercel.json`` rewrites. Scripts must therefore use an empty
API base (root-relative URLs) off localhost — not a hardcoded Render origin,
and not a bare ``window.location.origin`` assignment that would skip the
localhost special-case.

Local uvicorn still uses ``window.location.origin`` so ``localhost:8000`` keeps
working without the Vercel rewrite layer.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"

# ``assets/`` is minified Vite output for the landing page -- its identifiers are
# mangled, so a source-level name scan cannot say anything about it.
_SOURCES = sorted(
    path
    for path in _FRONTEND.rglob("*")
    if path.suffix in {".js", ".html"}
    and "assets" not in path.relative_to(_FRONTEND).parts
)

# Capture the initializer up to its terminating semicolon.
_DEFINITION = re.compile(r"(?:const|let|var)\s+(?:API_BASE|API)\s*=\s*([^;]{0,300})")

_BARE_ORIGIN = re.compile(r"(?:API_BASE|API)\s*=\s*window\.location\.origin\s*;")

# Anchored to the exact quoted literal, not a bare substring: a check like
# ``"localhost" in initializer`` would also pass for a typo'd host such as
# ``'https://evil.example/localhost'`` (CodeQL: py/incomplete-url-substring
# -sanitization). Quote-delimiting the literal makes the match exact.
_LOCALHOST_LITERAL = re.compile(r"""['"]localhost['"]""")
# Matching quote pairs only: ``['"]{2}`` also matches the mixed adjacency `'"`,
# so an initializer that merely abuts two differently-quoted strings would
# false-pass as "uses an empty API base".
_EMPTY_PROD_BASE = re.compile(r"""(?:''|"")""")
_LEGACY_ONRENDER = re.compile(r"""['"]https://agentictrading\.onrender\.com['"]""")


def _definitions():
    for path in _SOURCES:
        rel = path.relative_to(_FRONTEND).as_posix()
        for match in _DEFINITION.finditer(path.read_text(encoding="utf-8")):
            initializer = match.group(1).strip()
            # ``const API = {`` is app.js's fetch-helper object, not a base URL.
            if initializer.startswith("{"):
                continue
            yield rel, initializer


def test_the_known_api_base_definers_are_still_matched():
    """Guard the guard: a rename must fail loudly rather than pass vacuously."""
    definers = {name for name, _ in _definitions()}
    assert {"app.js", "js/agent-editor.js", "index.html", "strategy.html"} <= definers


def test_every_api_base_definition_uses_same_origin_off_localhost():
    for name, initializer in _definitions():
        assert _LOCALHOST_LITERAL.search(initializer), (
            f"{name}: the API base must special-case local development"
        )
        assert _EMPTY_PROD_BASE.search(initializer), (
            f"{name}: off localhost the API base must be '' (Vercel rewrites "
            "backend paths to Render; see vercel.json)"
        )
        assert not _LEGACY_ONRENDER.search(initializer), (
            f"{name}: hardcoded Render origin regresses same-origin cookie auth"
        )


def test_no_source_uses_a_bare_location_origin_as_its_api_base():
    offenders = [
        path.relative_to(_FRONTEND).as_posix()
        for path in _SOURCES
        if _BARE_ORIGIN.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"split-origin regression in: {offenders}"
