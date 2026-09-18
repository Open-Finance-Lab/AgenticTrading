"""The /admin console and /app draw the same top bar.

Two reports, one cause: "i do want the top nav bar to remain the same across all
pages no matter admins are on the admin page or my agents page". `/admin` shipped
with the standalone mock's bespoke header (`.brand` / `.product-nav` /
`.account-btn`), so the console announced itself as a different product.

The fix is a retyped copy of `app.html`'s header, and a copy is only worth
anything while it stays a copy. Neither document has a build step or a module
graph, so nothing in this repo would notice the two drifting apart -- not the
browser, not a linter, not any other test. This file is the mechanism: extract
the `<header>` block from each document, normalise the deltas that app.js's
absence *forces*, and assert what remains is identical.

The normalisations are the contract. Each one is a delta the port is allowed to
have; anything else showing up in the diff is drift, and the test says so by
failing. Adding a normalisation here is how you widen that contract, and it
should be as hard to do accidentally as editing this docstring.
"""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")

_HEADER = re.compile(r"<header class=\"header\">.*?</header>", re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _header_block(markup: str) -> str:
    found = _HEADER.search(markup)
    assert found, "no <header class=\"header\"> block"
    return found.group(0)


def _normalise(block: str) -> str:
    """Strip comments and collapse whitespace. Comments go because the two
    documents explain themselves to different readers; whitespace goes because
    app.html indents with 4 spaces inside a wrapper and admin.html with 2."""
    return re.sub(r"\s+", " ", _COMMENT.sub("", block)).strip()


def _canonicalise(block: str) -> str:
    """Apply the six enumerated deltas, turning either header into the shape
    both share. Deliberately literal, not regex-clever, and every substitution
    has to prove it fired: a normalisation that silently matches nothing is a
    normalisation that has stopped normalising, which is how a parity guard
    keeps passing after the thing it guards has gone."""
    text = _normalise(block)

    # Delta 1+2: /admin's nav entries are anchors with real hrefs and none is
    # .active -- app.js drives /app's buttons and lights the current view, and
    # /admin is none of the four product views. Exactly one form per mode must
    # be present, so a nav entry that vanishes from either page still fails.
    for mode, href in (
        ("home", "/app"),
        ("playground", "/app?view=playground&amp;playgroundTab=agents"),
        ("competition", "/app?view=competition"),
        ("community", "/app?view=community"),
    ):
        text = _sub_one_of(
            text,
            (
                f'<a class="mode-btn" data-mode="{mode}" href="{href}">',
                f'<button class="mode-btn active" data-mode="{mode}">',
                f'<button class="mode-btn" data-mode="{mode}">',
            ),
            f'<NAV data-mode="{mode}">',
        )
    text = text.replace("</button>", "</NAV>").replace("</a>", "</NAV>")

    # Delta 7: /admin's Discord anchor drops #openDiscordBtn and
    # data-discord-link -- app.js's opt-in markers for the account-linking OAuth
    # flow, which does not exist on this page. Normalised here rather than left
    # out, because markup that merely *looks* the same is what this file is for.
    text = _sub_one_of(
        text,
        (
            '<a class="header-discord-btn" href="https://discord.gg/9HnQ6XDG98" target="_blank" rel="noopener noreferrer">',
            '<a id="openDiscordBtn" class="header-discord-btn" data-discord-link href="https://discord.gg/9HnQ6XDG98" target="_blank" rel="noopener noreferrer">',
        ),
        "<DISCORD>",
    )

    # Delta 3: no #authSignInBtn on /admin -- gate() bounces a non-admin before
    # the page paints, so the signed-out branch is unreachable there. One-sided,
    # so it may be absent; `test_the_one_sided_deltas_are_on_the_side_they_claim`
    # pins which side.
    text = _sub_optional(
        text,
        '<button id="authSignInBtn" class="auth-btn auth-btn-primary" type="button">Sign in</NAV>',
    )

    # Delta 4+5: account-menu entries are <a href> on /admin, and its Admin
    # entry is unhidden (only admins reach the page) and points at /admin.
    for label, app_id, href in (
        ("Account", "accountMenuAccountBtn", "/app?view=account"),
        ("Credits &amp; Billing", "accountMenuCreditsBtn", "/app?view=credits"),
        ("Admin", "accountMenuAdminBtn", "/admin"),
    ):
        text = _sub_one_of(
            text,
            (
                f'<a class="account-menu-item" href="{href}">{label}</NAV>',
                f'<button id="{app_id}" class="account-menu-item" type="button" hidden>{label}</NAV>',
                f'<button id="{app_id}" class="account-menu-item" type="button">{label}</NAV>',
            ),
            f"<MENUITEM>{label}</MENUITEM>",
        )

    # Delta 6: #accountMenuLogoutError is /admin-only (PR #488). The console is
    # precisely the page you must not be left sitting on believing you signed
    # out, so it is an addition rather than a divergence -- but it is still a
    # difference, and it is enumerated here rather than tolerated silently.
    text = _sub_optional(
        text,
        '<p id="accountMenuLogoutError" class="account-menu-error" role="alert" hidden></p>',
    )

    return re.sub(r"\s+", " ", text).strip()


def _sub_one_of(text: str, candidates: tuple[str, ...], replacement: str) -> str:
    """Exactly one of `candidates` must be present. Zero means the node is gone
    from this page; more than one means the literals overlap and the delta list
    has stopped describing what it matches."""
    present = [needle for needle in candidates if needle in text]
    assert len(present) == 1, (len(present), candidates)
    return text.replace(present[0], replacement)


def _sub_optional(text: str, needle: str) -> str:
    return text.replace(needle, "")


def test_the_two_headers_are_the_same_header():
    admin = _canonicalise(_header_block(ADMIN_HTML))
    app = _canonicalise(_header_block(APP_HTML))
    assert admin == app


def test_the_deltas_are_real_deltas_and_not_a_vacuous_pass():
    """A normaliser that erased both sides down to nothing would pass the test
    above for any two documents. Pin that the canonical form still carries the
    header's substance."""
    admin = _canonicalise(_header_block(ADMIN_HTML))
    for marker in (
        "header-github-link",
        "<DISCORD>",
        "header-brand",
        "logo-container",
        "Agentic Trading Lab",
        'id="navMenuToggle"',
        'id="primaryNav"',
        'id="authAccountBtn"',
        'id="accountMenuWrap"',
        "<MENUITEM>Admin</MENUITEM>",
    ):
        assert marker in admin, marker
    assert len(admin) > 1200, len(admin)


def test_the_one_sided_deltas_are_on_the_side_they_claim():
    """`_sub_optional` erases two nodes wherever it finds them, which would also
    paper over them appearing on the wrong page. Pin the side."""
    admin, app = _header_block(ADMIN_HTML), _header_block(APP_HTML)
    assert 'id="authSignInBtn"' in app
    assert 'id="authSignInBtn"' not in admin
    assert 'id="accountMenuLogoutError"' in admin
    assert 'id="accountMenuLogoutError"' not in app


def test_the_raw_blocks_are_not_already_identical():
    """If they were, `_canonicalise` would be dead code and its assertions would
    stop guarding anything. They differ today because app.js exists on one page
    and not the other; the day that stops being true, delete the normaliser
    rather than leaving it unexercised."""
    assert _normalise(_header_block(ADMIN_HTML)) != _normalise(_header_block(APP_HTML))


def test_every_icon_the_header_references_is_in_the_admin_sprite():
    """A <use href="#missing"> renders an empty box and raises nothing, so a
    symbol left behind by the port is invisible until someone looks at the page."""
    header = _header_block(ADMIN_HTML)
    referenced = set(re.findall(r'<use href="#([a-z-]+)"', header))
    assert referenced == {"icon-github", "icon-discord", "icon-chevron-right"}, referenced
    declared = set(re.findall(r'<symbol id="([a-z-]+)"', ADMIN_HTML))
    assert referenced <= declared, sorted(referenced - declared)


def test_the_bespoke_mock_header_is_gone_from_both_the_markup_and_the_css():
    """The old bar's classes are generic enough to survive a partial port and
    keep painting -- `.brand` and `.product-nav` styled real nodes, and a bare
    `header{}` rule would restyle the new bar from underneath it."""
    admin_css = (FRONTEND / "admin.css").read_text(encoding="utf-8")
    for gone in ('class="brand"', 'class="product-nav"', 'id="accountBtn"',
                 'id="accountAvatar"', 'id="accountLabel"'):
        assert gone not in ADMIN_HTML, gone
    for gone in (".brand{", ".brand img{", ".product-nav{", "\nheader{"):
        assert gone not in admin_css, gone


def test_the_hamburger_opens_the_same_nav_on_both_pages():
    """Below 900px the ported CSS hides #primaryNav and shows #navMenuToggle, so
    the hamburger is the *only* way to the nav in that band. app.js owns that
    handler on /app and does not load here, so without a copy in admin-shell.js
    the button paints and does nothing -- silently, and only at widths a desktop
    reviewer never opens. Compared against app.js's rather than merely asserted
    to exist, for the same reason the markup is: two copies of one behaviour."""
    handler = re.compile(
        r"getElementById\('navMenuToggle'\)\?\.addEventListener\('click', \(\) => \{(.*?)\n\s*\}\);",
        re.DOTALL,
    )
    app_js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    shell_js = (FRONTEND / "js" / "admin-shell.js").read_text(encoding="utf-8")
    app_body = handler.search(app_js)
    shell_body = handler.search(shell_js)
    assert app_body, "app.js no longer wires #navMenuToggle"
    assert shell_body, "admin-shell.js no longer wires #navMenuToggle"
    assert re.sub(r"\s+", " ", app_body.group(1)).strip() == re.sub(r"\s+", " ", shell_body.group(1)).strip()
    # The CSS half of the same contract: a handler with nothing to toggle, or a
    # toggle with nothing hidden, is the same dead button from the other side.
    admin_css = (FRONTEND / "admin.css").read_text(encoding="utf-8")
    assert ".nav-menu-toggle{display:none" in admin_css
    assert ".nav-menu-toggle{display:inline-flex}" in admin_css
    assert ".primary-nav.open{display:flex}" in admin_css


def test_the_workspace_height_is_derived_from_the_header_not_a_second_literal():
    """`.workspace` subtracts the bar's height. It was `calc(100vh - 80px)`,
    sized to the mock's 80px bar; the ported one is 85px, and a second literal
    is a second owner of one measurement."""
    admin_css = (FRONTEND / "admin.css").read_text(encoding="utf-8")
    assert "--header-height:85px" in admin_css
    assert "calc(100vh - var(--header-height))" in admin_css
    assert "calc(100vh - 80px)" not in admin_css
