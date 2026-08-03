"""The /app header brand must never be painted over by the side chrome.

`.header` is a three-column grid: side chrome left, brand, side chrome right.
The brand was laid out on a full-span overlapping rail (`grid-column: 1 / -1`)
so it would centre on the *viewport* rather than on the residual gap between the
two side groups. That rail does centre it, but it puts the brand in the same
grid area as `.header-left` and `.header-right` while contributing *zero* width
to the tracks -- so nothing reserves room for it, and the side groups grow
straight through it. With `z-index: 1` against their `z-index: 2` the brand is
the one that loses. Measured on Chromium: the nav covered up to 137px of the
wordmark from 1440px down, and the GitHub/Discord group covered the logo itself
below ~1115px.

Moving the brand into a real track is necessary but not sufficient, and the
arithmetic is the whole reason this file exists. Two `minmax(0, 1fr)` side
tracks are forced *equal*, so the wider group sizes both: the nav side wants
~505px signed out, which reserves ~1010px for chrome and leaves the row needing
~1480px before it fits. Below that something must give, and everything in
`.header-left` and `.header-account` is `flex-shrink: 0`. Restated: equal tracks
cannot be made to fit at 901px even by a brand of zero width, so viewport-
centring is unreachable there and the labels have to collapse instead.

These are source-text guards because /app has no build step and CI has no
browser (the convention set by test_ai_hedge_fund_frontend.py). The geometry was
verified on Chromium at every 10px from 901 to 2560 in both auth states; what is
pinned here is the CSS contract that produced it.
"""

import re

from dashboard.backend.tests._frontend_source import APP_HTML, STYLES, css_blocks

#: Below this the header stops being a three-column row and stacks vertically,
#: and the nav becomes a hamburger -- overlap is structurally impossible there.
_STACK_BREAKPOINT_PX = 900

#: Widest viewport at which the signed-in row still overflowed with the GitHub
#: wordmark shown. Its collapse has to start at least this early.
_GITHUB_NAME_COLLISION_PX = 1440


def _base_blocks(prelude: str) -> list[str]:
    """`css_blocks` restricted to the rules above the first `@media`.

    The desktop rule and its `max-width: 900px` override share a selector, and
    picking `css_blocks(...)[0]` would silently depend on authoring order --
    reorder the file and the guard starts asserting against the override, where
    several claims below happen to be true for the wrong reason.
    """
    cutoff = STYLES.index("@media")
    return [block for block in css_blocks(prelude) if STYLES.index(block) < cutoff]


def _media_block(max_width_px: int) -> str:
    marker = f"@media (max-width: {max_width_px}px) {{"
    start = STYLES.index(marker)
    return STYLES[start : STYLES.index("\n}", start)]


def _hides(selector: str, block: str) -> bool:
    rule = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", block)
    return bool(rule) and "display: none" in rule.group(1)


def _collapse_breakpoint_for(selector: str) -> int:
    """Widest `max-width` media query that hides `selector`.

    Read out of the stylesheet rather than restated here: a guard that hardcodes
    the breakpoint stops covering the shipped value the moment someone edits it.
    """
    widest = 0
    for match in re.finditer(r"@media \(max-width: (\d+)px\) \{", STYLES):
        block = STYLES[match.start() : STYLES.index("\n}", match.start())]
        if _hides(selector, block):
            widest = max(widest, int(match.group(1)))
    return widest


def test_brand_is_not_on_a_full_span_overlapping_rail():
    """The regression itself: a brand spanning every column shares the side
    chrome's grid area, so nothing but z-index keeps them apart."""
    (brand,) = _base_blocks(".header-brand")
    assert "1 / -1" not in brand


def test_each_header_group_owns_its_own_column():
    """Separate tracks are what makes the overlap impossible. Asserted for all
    three: a brand moved to column 2 while a side group still spans the row
    would read as fixed here while overlapping in the browser."""
    (brand,) = _base_blocks(".header-brand")
    (left,) = _base_blocks(".header-left")
    (right,) = _base_blocks(".header-right")
    assert re.search(r"grid-column:\s*1\s*;", left)
    assert re.search(r"grid-column:\s*2\s*;", brand)
    assert re.search(r"grid-column:\s*3\s*;", right)


def test_side_tracks_are_content_sized_not_forced_equal():
    """`minmax(0, 1fr)` either side is what made the row unfittable.

    Two `fr` tracks with a `0` floor are always the same width, so the ~505px
    nav side reserves an equal ~505px of dead space on the GitHub side and the
    row needs ~1480px before the brand fits between them -- wider than the
    laptops this bug was reported on. Content-sized tracks fit from 900px up.
    """
    (header,) = _base_blocks(".header")
    columns = re.search(r"grid-template-columns:([^;]+);", header).group(1)
    assert "minmax(0, 1fr)" not in columns
    assert columns.split() == ["auto", "auto", "auto"]


def test_brand_track_cannot_be_squeezed_under_its_own_content():
    """`min-width: 0` lets a grid item shrink below min-content, so the brand
    would overflow its track and reach the neighbours again -- the same bug via
    a different route. It stays only in the stacked layout, where each group has
    the row to itself and the wordmark is wider than a 320px viewport."""
    (brand,) = _base_blocks(".header-brand")
    assert "min-width: 0" not in brand
    assert "min-width: 0" in _media_block(_STACK_BREAKPOINT_PX)


def test_left_chrome_collapses_before_it_can_reach_the_brand():
    """The other half of the same root cause.

    The GitHub link and the Discord button are both `flex-shrink: 0`, so the
    left group cannot give way -- it overflows its track and lands on the logo.
    Its labels used to collapse only at 700px, which is *below* the 900px point
    where the header stacks, so the collapse could never relieve the overlap
    window it was needed for.
    """
    github = _collapse_breakpoint_for(".header-github-name")
    assert github >= _GITHUB_NAME_COLLISION_PX
    assert github > _STACK_BREAKPOINT_PX


def test_left_chrome_collapses_in_stages():
    """Staging is load-bearing, not tidiness.

    The brand centres in whatever space the two side groups leave, so the
    further apart their widths the further it sits from centre. Collapsing the
    whole left group at the first breakpoint costs ~215px of offset immediately;
    dropping the GitHub wordmark first and the Discord label later holds it near
    ~160px through the 1341-1440px band.
    """
    github = _collapse_breakpoint_for(".header-github-name")
    discord = _collapse_breakpoint_for(".header-discord-btn__label")
    assert discord < github, "the two labels must not collapse at the same width"
    assert discord > _STACK_BREAKPOINT_PX
    assert _collapse_breakpoint_for(".header-discord-btn__arrow") == discord


def test_account_name_is_bounded_and_then_dropped():
    """The account group is the widest thing on the right when signed in, and
    the row is laid out against the *cap*, not against any one name."""
    (label,) = _base_blocks(".auth-user-label")
    cap = int(re.search(r"max-width:\s*(\d+)px", label).group(1))
    assert cap <= 100
    assert _collapse_breakpoint_for(".auth-user-label") > _STACK_BREAKPOINT_PX


def test_stylesheet_cache_buster_was_bumped():
    """Render serves /app's static files with no content hash, so a CSS fix that
    keeps the old `?v=` is invisible to every warm cache in prod. 76 is already
    claimed by two open branches."""
    version = int(re.search(r"styles\.css\?v=(\d+)", APP_HTML).group(1))
    assert version >= 77
