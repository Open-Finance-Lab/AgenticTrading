"""Static contracts for the no-build Credits & Billing frontend."""

from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")
STYLES = (FRONTEND / "styles.css").read_text(encoding="utf-8")
CREDITS_JS_PATH = FRONTEND / "js" / "credits.js"


def test_credits_page_is_reachable_from_the_account_menu():
    assert 'id="accountMenuCreditsBtn"' in APP_HTML
    assert 'id="creditsView"' in APP_HTML
    assert "credits: { page: 'credits' }" in APP_HTML
    assert "navigateToPage('credits')" in APP_JS


def test_credits_page_ships_test_mode_and_purchase_controls():
    assert 'id="creditsTestModeBadge"' in APP_HTML
    assert 'id="creditsBalance"' in APP_HTML
    assert 'data-credit-package="usd_5"' in APP_HTML
    assert 'data-credit-package="usd_50"' in APP_HTML
    assert 'id="creditsCustomAmount"' in APP_HTML
    assert 'id="creditsPurchaseBtn"' in APP_HTML
    assert 'id="creditsLedgerList"' in APP_HTML


def test_credits_script_loads_after_shared_api_wrapper():
    assert CREDITS_JS_PATH.exists()
    app_at = APP_HTML.index('src="app.js?')
    credits_at = APP_HTML.index('src="js/credits.js?')
    assert app_at < credits_at


def test_credits_client_keeps_stripe_authoritative():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "crypto.randomUUID()" in source
    assert "client_request_id" in source
    assert "pendingPurchase" in source
    assert "https:" in source
    assert "checkout.stripe.com" in source
    assert "MAX_ORDER_POLLS" in source
    assert "/api/credits/orders/" in source
    assert "Payment confirmation pending" in source
    assert "checkout.session_id" not in source


def test_credits_api_values_never_enter_inner_html():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert ".textContent" in source
    assert ".innerHTML" not in source
    assert "insertAdjacentHTML" not in source
    assert "toLocaleString('en-US'" in source


def test_credits_layout_has_mobile_contract():
    assert ".credits-view" in STYLES
    assert ".credits-package-grid" in STYLES
    assert "@media (max-width: 600px)" in STYLES
    assert '<th aria-label="Action"></th>' in APP_HTML
    assert '<span class="sr-only">Action</span>' not in APP_HTML
