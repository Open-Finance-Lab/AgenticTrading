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
    assert "parsed.protocol ===" in source
    assert "parsed.hostname ===" in source
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


# ---------------------------------------------------------------------------
# The balance must not claim spending power the platform does not have.
# ---------------------------------------------------------------------------

def test_balance_does_not_claim_credits_are_spendable():
    """Purchased Credits currently buy nothing, by design.

    They land in credit_ledger_entries; the only metered surface in this repo
    (POST /backtest/run via domain/entitlements/credits.py) spends
    user_entitlements.credits, which nothing here tops up. This PR ships the
    purchase side and defers consumption, so telling the buyer the balance is
    "available for model runs and backtests" sells spending power that does not
    exist. Delete this test only together with the code that debits this ledger.
    """
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    banned = "Available for model runs and backtests."
    # The claim is also spelled out in a comment above the corrected string, so
    # match on the rendered call rather than raw presence anywhere in the file.
    assert f"setStatus(accountStatus, '{banned}'" not in source
    assert "not enabled yet" in source
    assert ">Available balance<" not in APP_HTML


def test_refund_retry_reuses_its_request_id_only_while_unchanged():
    """The server derives the refund id from client_request_id.

    Reusing it on retry is what makes the retry idempotent instead of stacking a
    second reservation. But reusing it after the admin edits the amount sent the
    OLD amount while the UI showed and validated the new one.
    """
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "state.pendingRefund.amount_usd_cents !== cents" in source
    assert "state.pendingRefund.payment_order_id !== order.order_id" in source


def test_credits_billing_has_three_tabs_and_api_keys_surface():
    assert 'data-credits-tab="overview"' in APP_HTML
    assert '>Credits</button>' in APP_HTML
    assert 'data-credits-tab="top-up"' not in APP_HTML
    assert 'id="creditsPanelTopup"' not in APP_HTML
    assert 'data-credits-tab="api-keys"' in APP_HTML
    assert 'data-credits-tab="activity"' in APP_HTML
    assert 'id="creditsApiKeysPanel"' in APP_HTML
    assert 'id="creditsApiKeyForm"' in APP_HTML
    assert 'id="creditsApiKeyProvider"' in APP_HTML
    assert 'id="creditsApiKeySecret"' in APP_HTML
    assert 'id="creditsApiKeyList"' in APP_HTML


def test_api_keys_client_never_persists_or_renders_full_secret():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "/api/credits/model-providers" in source
    assert "/api/credits/api-keys" in source
    assert "creditsApiKeySecret" in source
    assert "secretInput.value = ''" in source
    assert "localStorage" not in source
    assert ".innerHTML" not in source
    assert "api_key: secret" in source
    assert "Spending Credits on model runs is not enabled yet." in APP_HTML


def test_verified_default_key_can_prepare_a_safe_byok_backtest():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "/api/credits/execution-options" in source
    assert "atlPendingByokBacktest" in source
    assert "sessionStorage.setItem" in source
    assert "billing_mode: 'byok'" in source
    assert "provider_id: credential.provider_id" in source
    assert "model_id: modelId" in source
    assert "expires_at:" in source
    assert "'Run Backtest'" in source
    assert "localStorage" not in source
    assert ".innerHTML" not in source


def test_quick_start_state_never_contains_a_secret():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    start = source.index("function beginByokBacktest")
    boundaries = (
        source.find("\n  function ", start + 1),
        source.find("\n  async function ", start + 1),
    )
    end = min(boundary for boundary in boundaries if boundary >= 0)
    body = source[start:end]
    assert "api_key" not in body
    assert "key_last_four" not in body
    assert "credential_id" not in body
