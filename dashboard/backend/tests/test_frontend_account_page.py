"""Account-page markup and cascade guards.

The frontend has no JS test harness, and these two contracts are structural
rather than behavioural -- an ordering and a CSS source-order requirement -- so
they are asserted against the shipped source directly.
"""

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
_APP_HTML = _FRONTEND / "app.html"
_STYLES_CSS = _FRONTEND / "styles.css"


def _account_card() -> str:
    html = _APP_HTML.read_text(encoding="utf-8")
    start = html.index('<div id="accountSignedIn"')
    end = html.index('<div id="accountSignedOut"')
    return html[start:end]


def test_logout_button_is_last_in_the_account_card():
    card = _account_card()
    logout_at = card.index('id="authLogoutBtn"')

    for marker in ("id=\"avatarUploadBtn\"", "id=\"changePasswordForm\""):
        assert card.index(marker) < logout_at, f"{marker} must come before Log out"

    # "after those two" is not "last" -- a section appended later would keep the
    # assertions above green. Nothing else in the card may carry an id.
    tail = card[logout_at + len('id="authLogoutBtn"'):]
    assert 'id="' not in tail, f"something with an id follows Log out: {tail!r}"


def test_logout_button_carries_the_danger_class():
    card = _account_card()
    match = re.search(r'<button[^>]*id="authLogoutBtn"[^>]*>', card)
    assert match, "logout button not found in the account card"
    tag = match.group(0)
    # Assert on the button's OWN tag. A substring search over the whole card
    # would pass if "auth-btn-danger" appeared anywhere else, and a fixed-width
    # window before the id cannot see the class at all -- this file's markup
    # puts id= before class=.
    assert "auth-btn-danger" in tag
    assert "auth-btn-secondary" not in tag


def test_header_dropdown_logout_is_untouched():
    # The brief targeted the account-page button only. Removing the dropdown
    # item would also make docs/source/lab/accounts.rst factually wrong.
    html = _APP_HTML.read_text(encoding="utf-8")
    assert 'id="accountMenuLogoutBtn"' in html


def test_auth_btn_danger_is_declared_after_the_generic_hover():
    """.auth-btn:hover and .auth-btn-danger:hover both score (0,2,0).

    With identical specificity, source order alone decides. Declared earlier,
    the logout button reverts to info-blue on hover -- red at rest, wrong on
    mouseover, which a screenshot taken at rest would not catch.
    """
    css = _STYLES_CSS.read_text(encoding="utf-8")
    assert css.index(".auth-btn-danger:hover") > css.index(".auth-btn:hover")
    assert css.index(".auth-btn-danger {") > css.index(".auth-btn {")


def test_account_card_section_order():
    card = _account_card()
    order = [
        'id="accountDisplayName"',      # read-only summary row
        'id="accountEmail"',            # read-only summary row
        'id="accountDisplayNameForm"',  # editor
        'id="accountEmailForm"',        # editor
        'id="avatarUploadBtn"',
        'id="changePasswordForm"',
        'id="authLogoutBtn"',
    ]
    positions = [card.index(marker) for marker in order]
    assert positions == sorted(positions), "account card sections are out of order"


def test_email_change_copy_mentions_the_spam_folder():
    """An unauthenticated single sender has materially degraded inbox placement,
    and a code silently in spam is indistinguishable from one never sent."""
    js = (_FRONTEND / "app.js").read_text(encoding="utf-8")
    assert js.lower().count("spam folder") >= 2  # one line per stage


def test_cache_bust_versions_were_bumped():
    html = _APP_HTML.read_text(encoding="utf-8")
    assert "styles.css?v=65" in html
    assert "app.js?v=48" in html
