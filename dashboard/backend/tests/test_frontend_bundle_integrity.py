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
  that makes the next refresh ambiguous about which bundle is live.

These checks are deliberately *not* a build-reproducibility check: Vite's content
hashes move with toolchain versions, so rebuilding and diffing would be flaky in
CI. Filename/reference agreement is the half that actually breaks in practice, and
it is verifiable with nothing but the committed tree.
"""

import re
from pathlib import Path

import pytest

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_INDEX_HTML = _FRONTEND / "index.html"
_ASSETS = _FRONTEND / "assets"

# Root-relative refs into the built output. Anything else in index.html is either
# absolute (fonts, the API host) or an in-page anchor, neither of which is ours.
_LOCAL_REF = re.compile(r'(?:src|href)="(/(?:assets|images)/[^"?#]+)')


def _index_html() -> str:
    return _INDEX_HTML.read_text(encoding="utf-8")


def _referenced_paths(html: str) -> set[Path]:
    return {_FRONTEND / ref.lstrip("/") for ref in _LOCAL_REF.findall(html)}


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
