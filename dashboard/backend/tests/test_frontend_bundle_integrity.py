"""Integrity guard for the shipped landing bundle (``dashboard/frontend/``).

The landing page at ``/`` is a Vite build whose output is committed **by hand**:
a refresh copies the new content-hashed assets into ``frontend/assets/`` and then
edits ``frontend/index.html`` to point at them, keeping the inline auth layer that
the build cannot produce (recipe + rationale in ``dashboard/landing/README.md``).

Nothing in CI builds the landing source, so the halves of that manual step can
drift apart silently, and either direction ships a blank page to prod:

* a forgotten ``index.html`` edit leaves ``<script src>`` on a hash that no longer
  exists → 404 → React never mounts and ``/`` renders an empty ``<div id="root">``;
* a forgotten deletion leaves a superseded ``index-*.js`` in the tree — dead weight
  that makes the next refresh ambiguous about which bundle is live;
* a forgotten *rebuild* leaves ``index-*.js`` older than ``landing/src`` — the page
  still renders, so nothing looks wrong, but it silently serves the previous CTA
  wiring and every source edit since is simply absent from prod.

These checks are deliberately *not* a build-reproducibility check: Vite's content
hashes move with toolchain versions, so rebuilding and diffing would be flaky in
CI. Filename/reference agreement is the half that actually breaks in practice, and
it is verifiable with nothing but the committed tree. The staleness checks below
hold to that same rule by comparing only *minifier-stable* markers: esbuild mangles
identifiers globally but never rewrites string literals, and a hyphenated property
key like ``data-landing-auth`` can never be emitted unquoted.
"""

import re
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_INDEX_HTML = _FRONTEND / "index.html"
_ASSETS = _FRONTEND / "assets"
_LANDING_SRC = Path(__file__).resolve().parents[2] / "landing" / "src"
_CTA_TS = _LANDING_SRC / "lib" / "cta.ts"

# Root-relative refs into the built output. Anything else in index.html is either
# absolute (fonts, the API host) or an in-page anchor, neither of which is ours.
_LOCAL_REF = re.compile(r'(?:src|href)="(/(?:assets|images)/[^"?#]+)')

# The attribute every landing CTA carries; the inline handler in index.html is its
# only consumer. Hyphenated, so minifiers must keep it a quoted string literal.
_AUTH_HOOK = "data-landing-auth"

# `label: "Start Free"` in landing/src/lib/cta.ts — the single source of CTA copy.
_CTA_LABEL = re.compile(r'label:\s*"([^"]+)"')


def _index_html() -> str:
    return _INDEX_HTML.read_text(encoding="utf-8")


def _referenced_paths(html: str) -> set[Path]:
    return {_FRONTEND / ref.lstrip("/") for ref in _LOCAL_REF.findall(html)}


def _entry_bundle_text() -> str:
    """Concatenated text of every ``/assets/*.js`` index.html actually loads."""
    js = [p for p in _referenced_paths(_index_html()) if p.suffix == ".js" and p.is_file()]
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in js)


def _landing_sources() -> list[Path]:
    """Every landing TS/TSX file. Absence is a failure, not a skip: this source is
    committed, so "not found" only ever means the guard is pointed at the wrong path.
    """
    assert _LANDING_SRC.is_dir(), (
        f"{_LANDING_SRC} does not exist. The landing source moved — re-point "
        "_LANDING_SRC, or these staleness checks guard nothing."
    )
    return sorted(p for p in _LANDING_SRC.rglob("*") if p.suffix in {".ts", ".tsx"})


def test_index_html_references_an_entry_bundle():
    """Guards the other two tests against passing vacuously.

    A mangled index.html with *zero* asset refs would trivially satisfy "every
    reference resolves", so pin that the entry points are actually there.
    """
    refs = {p.name for p in _referenced_paths(_index_html())}
    assert any(n.endswith(".js") for n in refs), (
        f"index.html references no /assets/*.js entry bundle (found: {sorted(refs)})"
    )
    assert any(n.endswith(".css") for n in refs), (
        f"index.html references no /assets/*.css bundle (found: {sorted(refs)})"
    )


def test_every_referenced_asset_exists():
    """A ref pointing at a deleted hash is a white page in prod, not a 500."""
    missing = sorted(
        str(p.relative_to(_FRONTEND)) for p in _referenced_paths(_index_html()) if not p.is_file()
    )
    assert not missing, (
        "index.html references files that do not exist under dashboard/frontend/: "
        f"{missing}. Point the <script>/<link> at the committed asset filenames "
        "(see dashboard/landing/README.md)."
    )


def test_no_orphaned_assets():
    """Every file in assets/ must be reachable from index.html.

    Reachability is transitive: the logo PNG is referenced from *inside* the JS
    bundle, not from index.html, so a check that only read index.html would flag
    it. Content hashes make basename matching sufficient here — a superseded
    ``index-*.js`` is named by nothing, which is exactly the case worth catching.
    """
    if not _ASSETS.is_dir():
        pytest.skip("no built assets committed")

    html = _index_html()
    reachable_text = [html]
    for ref in _referenced_paths(html):
        if ref.is_file() and ref.suffix in {".js", ".css"}:
            reachable_text.append(ref.read_text(encoding="utf-8", errors="replace"))
    haystack = "\n".join(reachable_text)

    orphans = sorted(f.name for f in _ASSETS.iterdir() if f.is_file() and f.name not in haystack)
    assert not orphans, (
        f"unreferenced files left in dashboard/frontend/assets/: {orphans}. "
        "A bundle refresh should delete the superseded index-*.{js,css} it replaces."
    )


def test_hand_written_auth_layer_survives_a_bundle_refresh():
    """The inline auth layer is not reproducible by ``vite build`` — pin it.

    ``dashboard/landing/index.html`` (the Vite template) is 26 lines and contains
    none of this; the shipped file is ~400. Copying ``dist/index.html`` wholesale
    over the shipped one — the obvious way to refresh a bundle — deletes the
    signup modal outright and turns all six ``data-landing-auth`` CTAs into
    buttons that do nothing when clicked, with no console error to notice.
    """
    html = _index_html()
    for marker, what in [
        ("landing-auth-pending", "auth-gate <script> that redirects signed-in visitors to /app"),
        ('id="landing-auth-patch"', "auth-layer <style> block"),
        ('id="landingAuthModal"', "signup/sign-in modal markup"),
        ("[data-landing-auth]", "delegated CTA click handler"),
    ]:
        assert marker in html, (
            f"dashboard/frontend/index.html lost the {what} ({marker!r}). It cannot be "
            "regenerated by `vite build` — see dashboard/landing/README.md."
        )


def test_shipped_bundle_has_one_cta_per_landing_source_emitter():
    """Catches a source edit that was never rebuilt into the committed bundle.

    ``index.html``'s delegated handler only fires for elements carrying
    ``data-landing-auth``, so a CTA that exists in ``landing/src`` but not in the
    shipped bundle is a button prod never renders — invisible to the other checks
    here, which only prove the bundle *loads*, not that it is current.

    Counting is sound because each JSX element compiles to its own props object:
    minifiers rename identifiers, never property-key string literals.
    """
    sources = _landing_sources()
    per_file = {
        p.relative_to(_LANDING_SRC).as_posix(): p.read_text(encoding="utf-8").count(_AUTH_HOOK)
        for p in sources
    }
    emitters = {name: n for name, n in per_file.items() if n}
    in_source = sum(emitters.values())
    in_bundle = _entry_bundle_text().count(_AUTH_HOOK)

    # Non-vacuity: the path exists (asserted above) but emits nothing — e.g. the
    # attribute was renamed everywhere. 0 == 0 would otherwise pass silently.
    assert in_source, (
        f"no {_AUTH_HOOK!r} emitters found under {_LANDING_SRC} — the landing CTAs "
        "moved, or this check is looking in the wrong place."
    )
    assert in_bundle == in_source, (
        f"shipped bundle has {in_bundle} {_AUTH_HOOK!r} CTA(s) but landing/src emits "
        f"{in_source} ({emitters}). The committed bundle predates the source: rebuild "
        "it and re-point index.html (see dashboard/landing/README.md)."
    )


def test_shipped_bundle_carries_the_current_cta_label():
    """Catches CTA *copy* drift, which the count check above cannot see.

    Renaming the shared label in ``lib/cta.ts`` without rebuilding leaves the count
    at six while prod still shows the old wording — a silent regression precisely
    because the page keeps working.
    """
    assert _CTA_TS.is_file(), (
        f"{_CTA_TS} does not exist. The shared CTA copy moved — re-point _CTA_TS, "
        "or this check guards nothing."
    )

    labels = _CTA_LABEL.findall(_CTA_TS.read_text(encoding="utf-8"))
    assert labels, f"no `label: \"...\"` found in {_CTA_TS} — CTA copy moved elsewhere."

    bundle = _entry_bundle_text()
    missing = [label for label in labels if label not in bundle]
    assert not missing, (
        f"CTA label(s) {missing} declared in landing/src/lib/cta.ts are absent from the "
        "shipped bundle, so prod still renders the previous copy. Rebuild the landing "
        "bundle (see dashboard/landing/README.md)."
    )
