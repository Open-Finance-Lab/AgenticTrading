"""Guards for the sign-in -> My Agents -> configure -> backtest flow (2026-08-03).

Every contract here was found by measuring the *rendered* page in headless
Chromium, not by reading source, and each one reverts silently: a colour that
drops under the contrast floor, a base rule deleted as redundant, or a client
limit that drifts from the server's still render a page that looks fine in a
diff. So they are asserted against the shipped bytes, per this suite's
frontend convention (`_frontend_source`).
"""

import re
from pathlib import Path

from dashboard.backend.api.routers.backtests import MAX_BACKTEST_DAYS
from dashboard.backend.tests._frontend_source import (
    STYLES,
    css_blocks,
    fn_body,
)

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_INDEX_HTML = (_FRONTEND / "index.html").read_text(encoding="utf-8")
_EDITOR_JS = (_FRONTEND / "js" / "agent-editor.js").read_text(encoding="utf-8")

# WCAG 2.1 AA for normal-weight body text.
_AA_CONTRAST = 4.5


def _css_var(name: str) -> str:
    """The first value of a `--custom-property` in styles.css."""
    match = re.search(rf"{re.escape(name)}:\s*(#[0-9a-fA-F]{{3,8}})", STYLES)
    assert match, f"{name} is no longer a hex custom property in styles.css"
    return match.group(1)


def _declaration(block: str, prop: str) -> str:
    """A property's value from inside one already-sliced CSS block.

    Comment-stripped first: three of the blocks below carry a rationale comment
    that names the very colour being replaced (`white on #00bfff measures
    2.12:1`), and a naive search finds that mention instead of the live
    declaration -- the guard would then read the *rejected* value and fail on
    correct code.
    """
    body = re.sub(r"/\*.*?\*/", "", block, flags=re.DOTALL)
    match = re.search(rf"(?<![-\w]){re.escape(prop)}:\s*([^;}}]+)", body)
    assert match, f"no {prop} declaration in block: {body[:120]}"
    return match.group(1).strip()


def _code_only(source: str) -> str:
    """`source` with its JS comments removed.

    Both fixes below are explained in a comment that *names the behaviour being
    replaced* ("used to go to '?view=home'"), so a `not in` assertion over raw
    source reads the rationale and reports a regression that is not there. The
    inverse is the real hazard: an `in` assertion satisfied by a comment passes
    after the code it describes has been deleted.
    """
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL))


def _slice_fn(source: str, signature: str) -> str:
    """`fn_body` for a file other than app.js (index.html's inline script)."""
    start = source.index(signature)
    depth, index = 0, source.index("{", source.index(")", start))
    while True:
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1


def _relative_luminance(hex_colour: str) -> float:
    raw = hex_colour.lstrip("#")
    if len(raw) == 3:
        raw = "".join(channel * 2 for channel in raw)
    channels = []
    for offset in (0, 2, 4):
        value = int(raw[offset : offset + 2], 16) / 255
        channels.append(
            value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def test_contrast_helper_agrees_with_the_wcag_reference_pairs():
    """Anchor the maths, so the ratio assertions below cannot pass vacuously.

    A luminance function with a transposed coefficient or a missing gamma step
    still returns a plausible number for any input, and every threshold check
    built on it would keep passing. Black-on-white is 21:1 exactly, and the
    white-on-#00bfff pair these fixes replace is the documented 2.12:1.
    """
    assert round(_contrast("#000000", "#ffffff"), 2) == 21.0
    assert round(_contrast("#ffffff", "#00bfff"), 2) == 2.12


def test_primary_cyan_buttons_clear_the_aa_contrast_floor():
    """`.auth-submit` paints the submit control at every step of this flow.

    The sign-in modal, the Run Backtest modal and both create-agent modals all
    use it, and it shipped as white on --info-color: 2.12:1, less than half the
    AA floor.
    """
    blocks = css_blocks(".auth-submit-btn")
    assert len(blocks) == 1, "expected one base .auth-submit-btn rule"
    # Pin the fill too: the ratio below is only meaningful while the button is
    # still painted on --info-color.
    assert _declaration(blocks[0], "background") == "var(--info-color)"
    ratio = _contrast(_declaration(blocks[0], "color"), _css_var("--info-color"))
    assert ratio >= _AA_CONTRAST, f"{ratio:.2f}:1"


def test_active_subtab_clears_the_aa_contrast_floor():
    """The My Agents / Backtest / Paper Trading switch -- the most-used control
    in the flow -- had the same white-on-cyan pairing."""
    blocks = css_blocks(".subtab-btn.active")
    assert len(blocks) == 1
    assert _declaration(blocks[0], "background") == "var(--info-color)"
    ratio = _contrast(_declaration(blocks[0], "color"), _css_var("--info-color"))
    assert ratio >= _AA_CONTRAST, f"{ratio:.2f}:1"


def test_form_controls_inherit_the_page_font():
    """Without this base rule the UA stylesheet wins and controls render Arial.

    Measured before the fix: 15 of the 16 buttons on My Agents plus the search
    input painted in Arial while the prose beside them used the app font. The
    cause is an *absence*, so nothing in the file points at it -- which is
    exactly why a later reader can delete the rule as redundant.
    """
    blocks = css_blocks("button,\ninput,\nselect,\ntextarea")
    assert len(blocks) == 1, "the form-control font base rule is gone"
    assert _declaration(blocks[0], "font-family") == "inherit"


def test_the_subtab_bar_is_not_the_smallest_text_on_the_page():
    """It was 11px in a 26px target, under a 14px/32px top nav, while being the
    primary in-product navigation."""
    blocks = css_blocks(".subtab-btn")
    assert blocks, "no .subtab-btn rule"
    assert int(_declaration(blocks[0], "font-size").removesuffix("px")) >= 13


def test_configure_panel_headings_outrank_the_section_card_default():
    """`.section-card h3 { font-size: 12px }` beat these single-class rules.

    All three panel titles in Configure therefore painted *smaller than their
    own body text*. The fix is specificity, so the guard has to check the
    qualified selector still exists -- an unqualified rule with the right
    font-size looks correct and renders at 12px.
    """
    default = css_blocks(".section-card h3")
    assert default, "no .section-card h3 rule to override"
    overridden = int(_declaration(default[0], "font-size").removesuffix("px"))

    for title in (
        "agent-capital-title",
        "agent-editor-history-title",
        "agent-editor-intro-title",
    ):
        blocks = css_blocks(f".section-card .{title}")
        assert len(blocks) == 1, f".{title} is no longer qualified by .section-card"
        size = int(_declaration(blocks[0], "font-size").removesuffix("px"))
        assert size > overridden, f".{title} renders at {size}px"


def test_the_instruction_field_is_sized_in_px_like_the_rest_of_the_app():
    """`rem` resolves against the 16px html root; the app is sized off a 14px
    body, so `0.95rem` painted this field at a fractional 15.2px."""
    blocks = css_blocks(".agent-editor-simple textarea")
    assert blocks, "no .agent-editor-simple textarea rule"
    assert _declaration(blocks[0], "font-size").endswith("px")


def test_run_backtest_submit_is_pinned_in_view():
    """The panel's content runs ~1080px, so the CTA rendered at y=1045 on a
    1440x900 screen -- past the fold, with nothing indicating the panel scrolls."""
    blocks = css_blocks(".run-backtest-modal-panel > #runBacktestModalSubmit")
    assert len(blocks) == 1, "the Run Backtest submit is no longer pinned"
    assert _declaration(blocks[0], "position") == "sticky"


def test_the_flow_controls_have_a_visible_focus_style():
    """styles.css carried eight bare `outline: none` resets and zero
    `:focus-visible` rules, so keyboard users could not see their position."""
    for selector in (".subtab-btn", ".agent-card-cta", ".home-btn"):
        assert f"{selector}:focus-visible" in STYLES, f"{selector} has no focus style"


def test_both_auth_paths_land_the_user_on_my_agents():
    """The two sign-in surfaces are separate code (the landing page has no build
    step and cannot import app.js), so each needs its own assertion.

    Sign-up used to go to Home -- a second marketing hero carrying the button
    the user had just clicked -- and sign-in navigated nowhere at all.
    """
    landing = _code_only(_slice_fn(_INDEX_HTML, "function goToDashboardLoggedIn"))
    assert "?view=agents" in landing
    assert "?view=home" not in landing
    # The '?view=' wins on boot; this write keeps a later bare /app visit off
    # the tab the user was last on while logged out.
    assert "'nav-state'" in landing
    assert "playgroundTab: 'agents'" in landing

    # Only the post-authentication branch, not all of initAuthUI: the sign-*out*
    # handler in the same function navigates Home and rightly so, so a whole-body
    # `not in` would forbid correct code.
    body = _code_only(fn_body("function initAuthUI"))
    prefix = body[: body.index("claimAgentsForUser()")]
    on_success = prefix[prefix.rindex("closeAuthModal();") :]
    assert "navigateToPage('agents')" in on_success
    # The destination used to fork on sign-up vs sign-in; it no longer does.
    assert "authMode === 'signup'" not in on_success


def test_the_client_backtest_span_check_mirrors_the_server_constant():
    """A client limit that drifts from the server's is worse than none: the
    modal closes, then a 422 arrives with the dates no longer on screen."""
    body = fn_body("async function runBacktest")
    match = re.search(r"const MAX_BACKTEST_DAYS = (\d+);", body)
    assert match, "runBacktest no longer range-checks the window client-side"
    assert int(match.group(1)) == MAX_BACKTEST_DAYS


def test_the_period_helper_states_the_limit_rather_than_inviting_the_error():
    from dashboard.backend.tests._frontend_source import APP_HTML

    assert f"up to {MAX_BACKTEST_DAYS} days" in APP_HTML
    assert "any range you have data for" not in APP_HTML


def test_editor_status_messages_also_reach_the_toast():
    """Every caller of showSaveStatus is a click in the editor's sticky header,
    but the element it writes to renders 920px below the fold and has no
    aria-live -- so 'Save changes before Run Backtest' was invisible and the
    primary CTA read as dead. Asserted inside the function so a stray call
    elsewhere in the file cannot satisfy it."""
    start = _EDITOR_JS.index("function showSaveStatus")
    end = _EDITOR_JS.index("\n  function ", start + 1)
    assert "window.showAppToast" in _EDITOR_JS[start:end]


def test_no_untranslated_strings_remain_in_the_served_frontend():
    """My Agents shipped a Chinese pagination footer and legend toggle in an
    otherwise all-English UI. The sweep covers the whole served tree, so it also
    catches a *new* untranslated string rather than only a revert of these two.

    `assets/` is excluded: it is the built landing bundle, whose source of truth
    is dashboard/landing (see test_frontend_bundle_integrity).
    """
    cjk = re.compile(r"[一-鿿]")
    offenders = [
        path.relative_to(_FRONTEND).as_posix()
        for pattern in ("*.js", "*.html", "*.css")
        for path in _FRONTEND.rglob(pattern)
        if "assets" not in path.parts
        and cjk.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []
