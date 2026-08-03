"""Split-origin API-base guard for the static frontend.

In production the frontend and the API live on **different origins** -- Vercel
serves ``dashboard/frontend`` statically, Render serves the FastAPI app -- so a
script that resolves its API base to ``window.location.origin`` addresses every
``/api/...`` call to the static host.  ``vercel.json`` has no ``/api`` rewrite,
so those requests come back as Vercel's *HTML* 404 page and surface as a JSON
parse error (``Unexpected token 'T', "The page c"...``) rather than a clean
404.  That is exactly how ``js/agent-editor.js`` shipped a Configure screen
whose save silently could not reach the backend.

The contract is on every file that *defines* a base; the files that merely read
the global ``API_BASE`` from ``app.js`` inherit a correct value.  ``API`` is
matched too because ``strategy.html`` names its base that.
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
# ``"onrender.com" in initializer`` would also pass for a typo'd host such as
# ``'https://evil.example/onrender.com'`` (CodeQL: py/incomplete-url-substring
# -sanitization). Quote-delimiting the literal makes the match exact.
_LOCALHOST_LITERAL = re.compile(r"""['"]localhost['"]""")
_ONRENDER_HOST_LITERAL = re.compile(r"""['"]https://agentictrading\.onrender\.com['"]""")


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


def test_every_api_base_definition_targets_the_backend_off_localhost():
    for name, initializer in _definitions():
        assert _LOCALHOST_LITERAL.search(initializer), (
            f"{name}: the API base must special-case local development"
        )
        assert _ONRENDER_HOST_LITERAL.search(initializer), (
            f"{name}: the API base must point at the hosted backend when not on "
            "localhost -- the Vercel origin serves no /api routes"
        )


def test_no_source_uses_a_bare_location_origin_as_its_api_base():
    offenders = [
        path.relative_to(_FRONTEND).as_posix()
        for path in _SOURCES
        if _BARE_ORIGIN.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"split-origin regression in: {offenders}"
