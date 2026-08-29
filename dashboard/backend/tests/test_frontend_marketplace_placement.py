"""Guards for the Agent Supermarket living on the Community page.

The marketplace moved out of Playground (where it was a subtab) and became a
section of Community. Nothing about that move is enforceable at runtime -- the
frontend has no JS test harness -- so, following the convention of the other
test_frontend_* files here, these contracts are asserted against the shipped
source directly.

What they are really protecting is the *anti-FOUC* pair: app.html's boot script
decides which page the CSS paints before app.js loads, and app.js decides which
page it renders. If those two ever disagree about where the marketplace lives,
the dashboard paints one page and renders another -- a flash bug no other test
in this repo can see.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = _FRONTEND / "app.html"
_APP_JS = _FRONTEND / "app.js"
_STYLES_CSS = _FRONTEND / "styles.css"


def _community_view() -> str:
    html = _APP_HTML.read_text(encoding="utf-8")
    start = html.index('<div id="communityView"')
    end = html.index('<div id="accountView"')
    assert start < end, "communityView must precede accountView in app.html"
    return html[start:end]


def test_marketplace_markup_lives_inside_the_community_view():
    view = _community_view()
    for marker in (
        'id="marketplaceGrid"',
        'id="marketplaceSearchInput"',
        'id="marketplaceEmptyState"',
        'id="marketplaceErrorState"',
    ):
        assert marker in view, f"{marker} is not inside #communityView"

    # ...and nowhere else. A duplicate id would make document.getElementById
    # return whichever copy came first, silently wiring the search box and the
    # grid to different sections.
    html = _APP_HTML.read_text(encoding="utf-8")
    assert html.count('id="marketplaceGrid"') == 1


def test_the_playground_marketplace_subtab_is_gone():
    html = _APP_HTML.read_text(encoding="utf-8")
    assert "playgroundMarketplacePanel" not in html
    assert 'data-playground-tab="marketplace"' not in html
    # app.js must not reference the removed panel either -- a leftover
    # getElementById would be a silent null, not an error.
    assert "playgroundMarketplacePanel" not in _APP_JS.read_text(encoding="utf-8")


def test_nav_view_map_sends_the_marketplace_slug_to_community():
    """?view=marketplace and #marketplace are load-bearing: they are in shipped
    docs and in any URL a user bookmarked before the move."""
    html = _APP_HTML.read_text(encoding="utf-8")
    match = re.search(r"marketplace:\s*\{([^}]*)\}", html)
    assert match, "NAV_VIEW_MAP has no 'marketplace' key"
    entry = match.group(1)
    assert "page: 'community'" in entry
    # No playgroundTab: leaving one behind would set the module-level tab to a
    # subtab that no longer exists.
    assert "playgroundTab" not in entry


def test_boot_css_reveals_the_community_view():
    """The boot stylesheet is what makes the move flash-free. Without this rule
    a saved 'community' nav state paints nothing until app.js runs."""
    html = _APP_HTML.read_text(encoding="utf-8")
    assert (
        'html[data-nav-boot][data-nav-page="community"] #communityView'
        in html
    )


def test_the_saved_state_migration_is_defined_once_and_shared():
    """Both files restore the SAME localStorage blob.

    Two hand-written copies of the rewrite rule is exactly the divergence the
    NAV_VIEW_MAP comment warns about, so app.html owns it and app.js reads it
    back off `window`.
    """
    html = _APP_HTML.read_text(encoding="utf-8")
    js = _APP_JS.read_text(encoding="utf-8")

    assert "window.migrateSavedNavState = migrateSavedNavState;" in html
    assert "window.migrateSavedNavState" in js
    # app.js must not re-implement it: the boot copy inspects the parsed blob's
    # own playgroundTab, so that expression appearing here means a second copy.
    assert "saved.playgroundTab" not in js


def test_navigating_to_the_retired_subtab_lands_on_community():
    js = _APP_JS.read_text(encoding="utf-8")
    start = js.index("function navigateToPage")
    guard = js[start : start + 1800]
    assert "=== 'marketplace'" in guard, "navigateToPage lost its marketplace redirect"
    assert "page = 'community'" in guard


def test_view_param_never_emits_the_marketplace_alias():
    """viewParamForNavState is NAV_VIEW_MAP's hand-maintained inverse. Emitting
    'marketplace' would write a URL naming a page that no longer exists."""
    js = _APP_JS.read_text(encoding="utf-8")
    start = js.index("function viewParamForNavState")
    body = js[start : js.index("function buildNavigationUrl")]
    assert "return 'marketplace'" not in body
    assert "return 'community'" in body


def test_community_page_header_matches_the_nav_button():
    """The nav says one word and the page must say the same word.

    Before this guard the button read "Community" while the page it opened was
    titled "Agent Marketplace" and held nothing else.
    """
    html = _APP_HTML.read_text(encoding="utf-8")
    nav = re.search(r'<button[^>]*data-mode="community"[^>]*>([^<]+)</button>', html)
    assert nav, "no primary-nav button for the community page"
    label = nav.group(1).strip()

    view = _community_view()
    title = re.search(r'<h2 class="page-title">([^<]+)</h2>', view)
    assert title, "#communityView has no page title"
    assert title.group(1).strip() == label

    # The supermarket is a section *within* that page, so it must not also claim
    # the page-level heading.
    assert 'class="marketplace-section-title">Agent Supermarket<' in view
    assert view.count('class="page-title"') == 1


def test_marketplace_section_title_is_styled():
    """A heading class with no rule renders at the browser default h3 size,
    which is larger than the .page-title above it."""
    css = _STYLES_CSS.read_text(encoding="utf-8")
    assert ".marketplace-section-title {" in css


def test_repeat_visits_do_not_refetch_the_catalog():
    """Community is a top-level page now, so loadMarketplace runs on every nav
    click, every popstate and the initial boot -- not once per session as it did
    behind the Playground subtab."""
    js = _APP_JS.read_text(encoding="utf-8")
    start = js.index("async function loadMarketplace")
    body = js[start : js.index("async function cloneMarketplaceTemplate")]

    assert "if (marketplaceTemplates.length)" in body, "no cache short-circuit"
    assert "marketplaceLoadInFlight" in body, "no in-flight dedup"
    # The cache must be cleared on failure, or one flaky GET pins the error
    # state for the rest of the session.
    assert "marketplaceTemplates = [];" in body


def test_marketplace_repo_url_renders_as_github_button():
    """Open-source templates link out to their GitHub repo from the card meta."""
    js = _APP_JS.read_text(encoding="utf-8")
    css = _STYLES_CSS.read_text(encoding="utf-8")
    catalog = (
        Path(__file__).resolve().parents[2] / "config" / "marketplace.json"
    ).read_text(encoding="utf-8")
    assert "marketplace-repo-btn" in js
    assert "template.repo_url" in js
    assert 'href="#icon-github"' in js
    assert "https://github.com/virattt/ai-hedge-fund" in catalog
    assert ".marketplace-repo-btn" in css
