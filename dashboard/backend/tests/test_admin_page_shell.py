"""Source-shape guard for the /admin shell (design §7.6, D6, D7).

The page must carry no data: every number arrives from a require_admin-gated
endpoint after the gate has run. The mock this page replaces failed exactly
that property -- sixteen inline scripts wrote literals into every panel -- so
the guard checks the *markup*: no inline script, every script pinned, and no
numeric or percentage literal inside any panel region (placeholders are "--").
"""

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
ADMIN_CSS = (FRONTEND / "admin.css").read_text(encoding="utf-8")

EXPECTED_SCRIPTS = [
    "js/admin-shell.js?v=5",
    "js/credit-format.js?v=1",
    # The app chrome's ticker (D5 layout pass) rides after the formatters.
    "js/admin-ticker.js?v=2",
    "js/admin-live.js?v=1",
    "js/admin-overview.js?v=1",
    "js/admin-users.js?v=1",
    "js/admin-providers.js?v=1",
    # The absorbed credits console (design N2/PR2).
    "js/admin-credits.js?v=2",
]

# This guard's whole job is "no inline script anywhere in this page", so a
# spelling of <script> it cannot find is a hole rather than a style nit. CodeQL
# py/bad-tag-filter named three, one per round: tag names are case-insensitive
# (`<SCRIPT>`), an end tag may pad before `>` (`</script >`), and an end tag may
# carry ignored attributes (`</script foo=bar>`). `\b[^>]*>` covers the last two
# at once while still rejecting `</scriptfoo>`, which is not an end tag at all.
# test_the_inline_script_guard_sees_tags_html_allows pins every spelling, because
# admin.html is lowercase and tightly closed -- nothing else here can catch a
# regression.
SCRIPT_TAG = re.compile(r"<script\b([^>]*)>(.*?)</script\b[^>]*>", re.S | re.I)
PANEL_REGION = re.compile(
    r'<section\b[^>]*\bdata-panel="([^"]+)"[^>]*>(.*?)</section\b[^>]*>', re.S | re.I
)
TAG = re.compile(r"<[^>]+>")
# A standalone number: not glued to a letter, underscore, hash, dot or dash on
# the left (so "W1", "#users" and "8-29" never count) and not followed by a
# word character ("1W" is a range label, not a value). Optional decimals and %.
NUMERIC_LITERAL = re.compile(r"(?<![A-Za-z0-9_#.\-])\d+(?:[.,]\d+)?%?(?![A-Za-z0-9_])")


def test_every_script_is_external_and_none_is_inline():
    tags = SCRIPT_TAG.findall(ADMIN_HTML)
    assert len(tags) == len(EXPECTED_SCRIPTS)
    for attrs, body in tags:
        assert 'src="' in attrs, attrs
        assert "defer" in attrs, attrs
        assert body.strip() == "", body


def test_the_inline_script_guard_sees_tags_html_allows():
    """The guard proves a negative -- that no inline script exists -- so any spelling
    of `<script>` it cannot find is a hole. CodeQL named three across three rounds,
    each one only after the previous was fixed, so this walks the pattern over all of
    them at once instead of one per CI cycle. The page is lowercase and tightly
    closed, so nothing else in this file can catch a regression here."""
    for markup in (
        "<script>alert(1)</script>",
        "<SCRIPT>alert(1)</SCRIPT>",
        "<script>alert(1)</script >",
        "<script>alert(1)</script\t\n bar>",
        "<script\t\n bar>alert(1)</script>",
        "<script src='a' defer>alert(1)</script>",
        "<ScRiPt>alert(1)</sCrIpT foo=bar>",
    ):
        tags = SCRIPT_TAG.findall(markup)
        assert tags, markup
        assert tags[0][1] == "alert(1)", (markup, tags)
    # `</scriptfoo>` is a different tag, not a padded end tag: `\b` is what keeps
    # `[^>]*` from swallowing the rest of the name.
    for markup in ("<script>alert(1)</scriptfoo>", "<scriptfoo>alert(1)</script>"):
        assert not SCRIPT_TAG.findall(markup), markup


def test_gate_module_loads_first_and_every_script_is_pinned():
    srcs = re.findall(r'<script src="([^"]+)" defer></script>', ADMIN_HTML)
    assert srcs == EXPECTED_SCRIPTS
    assert srcs[0] == "js/admin-shell.js?v=5"
    for src in srcs:
        assert "?v=" in src, src


def test_stylesheet_is_admin_css_and_the_page_does_not_inherit_styles_css():
    assert '<link rel="stylesheet" href="admin.css?v=4">' in ADMIN_HTML
    assert "styles.css" not in ADMIN_HTML
    assert "@import" not in ADMIN_CSS
    assert "cdn.jsdelivr.net" not in ADMIN_HTML
    assert "chart.js" not in ADMIN_HTML.lower()


def test_title_is_the_admin_console():
    assert "<title>ATL Admin</title>" in ADMIN_HTML


def test_panel_regions_carry_no_numeric_or_percentage_literal():
    regions = PANEL_REGION.findall(ADMIN_HTML)
    names = [name for name, _body in regions]
    assert names == [
        "live", "attention", "active-users", "activation", "sources", "retention",
        "value", "lifecycle", "credits", "revenue", "detail", "users", "profile", "providers",
        # The absorbed credits console's two routes (design N2/PR2).
        "account", "activity",
    ]
    for name, body in regions:
        text = TAG.sub(" ", body)
        match = NUMERIC_LITERAL.search(text)
        assert match is None, (name, match and match.group(0), text.strip()[:200])


def test_every_value_slot_is_a_dash_placeholder():
    for attr in ("data-headline", "data-live"):
        slots = re.findall(rf"<strong[^>]*\b{attr}[^>]*>([^<]*)</strong>", ADMIN_HTML)
        assert slots, attr
        assert set(slots) == {"—"}, (attr, slots)
    metas = re.findall(r'<small[^>]*\bdata-live="slots"[^>]*>([^<]*)</small>', ADMIN_HTML)
    assert metas == ["—"]


def test_live_panel_carries_a_status_node_for_its_stale_notice():
    """F2 regression: the live row is the only panel `setPanelState` targets that
    had no `[data-status]` node, so a stale reading (design D17) rendered no
    notice at all -- a stale counter was pixel-identical to a fresh one. (The
    node's own text-content and numeric-literal rules are still covered by
    test_panel_regions_carry_no_numeric_or_percentage_literal /
    test_every_value_slot_is_a_dash_placeholder above.)"""
    regions = dict(PANEL_REGION.findall(ADMIN_HTML))
    assert "data-status" in regions["live"]


def test_filter_bar_and_range_match_the_survival_table():
    filters_start = ADMIN_HTML.index('<form class="filters"')
    filters_end = ADMIN_HTML.index("</form>", filters_start)
    filters = ADMIN_HTML[filters_start:filters_end]
    assert filters.count("<select") == 3
    for control in ("filterGroup", "filterSegment", "filterTier", "filterInternal"):
        assert f'id="{control}"' in filters, control
    assert "Include internal accounts" in filters
    assert "Cohort" not in filters and "intake" not in filters
    assert "All paid states" not in filters
    for option in ("internal", "invited", "organic", "competition", "partner", "unknown"):
        assert f'value="{option}"' in filters
    for option in ("new", "onboarding", "growing", "core", "at_risk", "dormant"):
        assert f'value="{option}"' in filters
    for option in ("unpaid", "starter", "invested", "high_value"):
        assert f'value="{option}"' in filters
    ranges = re.findall(r'data-range="([^"]+)"', ADMIN_HTML)
    assert ranges == ["1W", "1M", "1Y"]


def test_range_and_filters_live_outside_the_overview_section():
    """They govern every route, so they must not sit inside the one section
    `route()` hides.

    `showView('overview', parsed.route === 'overview')` sets `#overview.hidden`
    on the six other routes. With the range group and the filter form nested
    inside it, a Lifecycle stage picked on Overview went invisible while
    `userListParams` kept sending `lifecycle_segment`: the users table was
    silently narrowed, and its toolbar (search + priority) offers nothing to see
    or clear it with. The detail routes had the matching defect -- they render
    "Selected range · 1W · UTC" as a label for a control the operator could only
    reach by navigating back to Overview.
    """
    controls_start = ADMIN_HTML.index('<div class="page-controls" id="pageControls">')
    overview_start = ADMIN_HTML.index('<section id="overview"')
    assert controls_start < overview_start
    toolbar = ADMIN_HTML[controls_start:overview_start]
    from_overview_on = ADMIN_HTML[overview_start:]
    for control in (
        'id="filters"', 'id="filterGroup"', 'id="filterSegment"',
        'id="filterTier"', 'id="filterInternal"', 'data-range=',
    ):
        assert control in toolbar, control
        assert control not in from_overview_on, control


def test_cut_elements_are_absent():
    for token in (
        "Sample data", "synthetic", "Sample snapshot", "Online now", "Queued",
        'href="#live"', "Layout proposal", "inline-key", "Affected users",
        "September intake", 'data-range="1D"', ">Region<", ">Device<", ">Browser<",
        "Top product page",
    ):
        assert token not in ADMIN_HTML, token


def test_subnav_routes_match_the_shell_router_and_the_aside_links_back():
    start = ADMIN_HTML.index('<nav class="analytics-subnav"')
    end = ADMIN_HTML.index("</nav>", start)
    hrefs = re.findall(r'href="(#[a-z]+)"', ADMIN_HTML[start:end])
    assert hrefs == ["#overview", "#sources", "#retention", "#credits", "#lifecycle", "#health", "#users"]
    assert "#live" not in ADMIN_HTML
    # N2/PR2: no console entry links out any more — Account Management and
    # Activity joined Providers as routes on this page.
    assert "adminTab=" not in ADMIN_HTML
    assert 'data-rail="account" href="#account"' in ADMIN_HTML
    assert 'data-rail="providers" href="#providers"' in ADMIN_HTML
    assert 'data-rail="activity" href="#activity"' in ADMIN_HTML


def test_freshness_legend_replaces_the_sample_notice():
    assert 'id="freshnessLegend"' in ADMIN_HTML
    assert "Daily figures complete through" in ADMIN_HTML
    assert "live tiles are this instance" in ADMIN_HTML


def test_pruned_css_carries_none_of_the_dead_mock_passes():
    for selector in (
        ".funnel", ".activation-river", ".activation-stages", ".live-cell",
        ".live-grid", ".live-composite", ".mini-viz", ".sparkline", ".run-lanes",
        ".blocked-reasons", ".source-bars", ".stack-column", ".inline-key",
        ".sample-label", ".queue-", ".operation-lane", ".blocker-item",
        ".value-donut", ".detail-live-grid", ".snapshot-tile.online",
        ".snapshot-tile.queued", ".snapshot-tile.blocked",
    ):
        assert selector not in ADMIN_CSS, selector
    for selector in (
        ".layered-route", ".source-donut", ".source-legend-six", ".value-steps",
        ".lifecycle-bar", ".retention-chart", ".credits-paired", ".revenue-viz",
        ".snapshot-grid", ".attention-row", ".group-badge", ".panel-error",
        ".freshness-legend", "@media(max-width:1040px)", "@media(max-width:680px)",
        "@media(prefers-reduced-motion:reduce)",
    ):
        assert selector in ADMIN_CSS, selector


RAIL_ICONS = ("icon-chart", "icon-users", "icon-network", "icon-activity")
HEADER_ICONS = ("icon-github", "icon-discord", "icon-chevron-right")
SPRITE_ICONS = (
    RAIL_ICONS
    + ("icon-refresh", "icon-x", "icon-check-circle")
    + HEADER_ICONS
    + ("icon-wallet", "icon-search", "icon-minus")  # absorbed credits console
)


def test_the_rail_is_anchors_with_icons_not_a_tablist():
    """N1: /admin navigates by hash and #users/{id} is a real linkable location,
    so the rail imitates the legacy rail's look and not its ARIA. role="tab" on a
    control that changes the URL misreports the interaction, and an anchor gets
    middle-click, copy-link and keyboard handling with no JavaScript."""
    start = ADMIN_HTML.index('<aside id="adminRail"')
    end = ADMIN_HTML.index("</aside>", start)
    rail = ADMIN_HTML[start:end]
    entries = re.findall(r'<a class="admin-tab[^"]*"[^>]*data-rail="([a-z-]+)"[^>]*>(.*?)</a>', rail, re.S)
    assert [name for name, _ in entries] == ["analytics", "account", "providers", "activity"]
    for name, body in entries:
        assert "<use href=\"#icon-" in body, name          # every entry has an icon
        assert re.search(r"<span>[^<]+</span>", body), name  # ...and a text label
    assert 'role="tab"' not in rail
    assert "aria-selected" not in rail
    # Checked against the four top-level entry bodies, not the whole `rail`
    # blob: the (unchanged) analytics subnav nested inside the same <aside>
    # legitimately contains a "Users" link to the #users route, so a whole-
    # block substring check collides with it. The guard's actual target is
    # the account-management entry's label.
    labels = [re.search(r"<span>([^<]+)</span>", body).group(1) for _, body in entries]
    assert "Account Management" in labels
    assert "Users" not in labels


def test_the_sprite_defines_every_symbol_the_page_references():
    """A <use href="#icon-x"> with no matching <symbol> renders an empty box and
    raises nothing -- the one failure mode a source-shape guard can actually catch."""
    defined = set(re.findall(r'<symbol id="([a-z-]+)"', ADMIN_HTML))
    assert defined == set(SPRITE_ICONS), defined
    referenced = set(re.findall(r'<use href="#([a-z-]+)"', ADMIN_HTML))
    assert referenced <= defined, referenced - defined
    # The sprite is markup, not script, so D6's no-inline-script guard is untouched.
    assert "<script" not in ADMIN_HTML[ADMIN_HTML.index("<svg") : ADMIN_HTML.index("</svg>")]


def test_the_account_menu_exists_and_carries_no_identity_text():
    """The menu renders only after the gate resolves. A name baked into the markup
    would be a different account's name for the duration of the probe."""
    start = ADMIN_HTML.index('<div id="accountMenuWrap"')
    end = ADMIN_HTML.index("</header>", start)
    menu = ADMIN_HTML[start:end]
    for control in ("authAccountBtn", "accountMenu", "accountMenuName", "accountMenuEmail", "accountMenuLogoutBtn"):
        assert f'id="{control}"' in menu, control
    assert '<span id="accountMenuName" class="account-menu-name"></span>' in menu
    assert '<span id="accountMenuEmail" class="account-menu-email"></span>' in menu
    assert 'href="/app?view=account"' in menu
    assert 'href="/app?view=credits"' in menu
    # N7 said no Admin item, on the grounds that it would link to the page it is
    # on. The header port overrides that half: the menu is now a retyped copy of
    # app.html's, and dropping one entry is exactly the kind of drift the parity
    # guard exists to prevent. Self-referential is the lesser cost -- the entry
    # is how an admin gets *back* here from /app, so it has to read the same on
    # both. Unhidden, unlike /app's, because only admins reach this page at all.
    assert '<a class="account-menu-item" href="/admin">Admin</a>' in menu
    # N7's no-ticker half is superseded by the 09-18 layout pass: the operator
    # asked for the app header's strip on this page too (admin-ticker.js).
    assert 'id="tickerTrack"' in ADMIN_HTML


def test_every_styles_css_token_name_used_here_is_also_declared_here():
    """admin.css does not import styles.css, so a `var(--foo)` copy-pasted from
    there is a rule that renders unstyled unless this file declares --foo too.

    This used to ban styles.css's token names outright. The header port made
    that ban wrong in the one direction that matters: the universal top bar is
    retyped verbatim *on styles.css's token names*, deliberately, because
    substituting this file's nearest equivalent (#12172b for #0f1328, #2a3041
    for #1f2937) is precisely how two copies of one header become two headers.
    So the guard moved from the name to the declaration -- which is the property
    the old assertion was really protecting, and is now checked for every token
    rather than the six that were listed."""
    # Set per element at runtime by admin-overview.js's style.setProperty calls,
    # so they are legitimately used-but-not-declared here. Asserted against the
    # JS rather than hardcoded: a renamed property must break one of the two
    # sides, not quietly widen the exemption.
    js_set = set(
        re.findall(
            r"setProperty\('(--[a-z0-9-]+)'",
            (FRONTEND / "js" / "admin-overview.js").read_text(encoding="utf-8"),
        )
    )
    assert js_set, "admin-overview.js sets no custom properties — update the exemption"
    # Comments stripped first: this file explains its tokens in prose, and a
    # `--foo:` inside /* */ would count as a declaration and re-open exactly
    # the copy-paste hole this assertion replaced the old name-ban with.
    code = re.sub(r"/\*.*?\*/", "", ADMIN_CSS, flags=re.DOTALL)
    declared = set(re.findall(r"(--[a-z0-9-]+)\s*:", code)) | js_set
    used = set(re.findall(r"var\((--[a-z0-9-]+)", ADMIN_CSS))
    assert used <= declared, sorted(used - declared)
    # The alias block itself: these carry styles.css's values, not this file's
    # nearest equivalent, and the header is unrecognisable if they drift.
    for token, value in (
        ("--bg-surface", "#0f1328"),
        ("--bg-card", "#131a35"),
        ("--bg-hover", "#1a2047"),
        ("--text-primary", "#e5e7eb"),
        ("--text-secondary", "#9ca3af"),
        ("--border-color", "#1f2937"),
        ("--info-color", "#00bfff"),
    ):
        assert f"{token}:{value}" in ADMIN_CSS, token
    # Never copied: a literal that belongs to no token on either side.
    assert "#67e8f9" not in ADMIN_CSS
    for retyped in (".admin-rail", ".admin-tab", ".admin-tab svg", ".admin-tab.is-active::before",
                    ".account-menu", ".account-menu-item", ".account-menu-item--danger"):
        assert retyped in ADMIN_CSS, retyped
    assert "min-height:43px" in ADMIN_CSS      # the legacy pill height
    assert "font-size:13px;font-weight:600" in ADMIN_CSS
    assert "width:18px;height:18px" in ADMIN_CSS


def test_the_flat_text_list_rules_are_gone():
    """The old aside styled every anchor with one `aside a` rule and shrank the
    labels to font-size:0 on phones. Leaving either behind lets a stale rule win
    over the rail by source order."""
    assert "aside a{" not in ADMIN_CSS
    assert "aside a:hover{" not in ADMIN_CSS
    assert "aside a::first-letter" not in ADMIN_CSS
    assert ".analytics-parent{" not in ADMIN_CSS
