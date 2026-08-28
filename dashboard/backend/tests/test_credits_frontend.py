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

def test_balance_copy_describes_platform_and_byok_lanes():
    """Credits are available for metered runs while BYOK stays user-funded."""
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    wording = "Available for metered ATL model runs. BYOK runs use your provider account and do not deduct ATL Credits."
    assert "not enabled yet" not in source
    assert wording in source
    assert ">Available balance<" not in APP_HTML


def test_activity_renders_negative_model_usage_with_safe_context():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "entry.entry_type === 'llm_usage'" in source
    assert "'Model usage'" in source
    assert "entry.provider_id" in source
    assert "entry.model_id" in source
    assert "String(entry.run_id).slice(0, 12)" in source


def test_activity_names_the_automatic_welcome_grant():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")

    assert "entry.entry_type === 'system_promotion_grant'" in source
    assert "'Welcome Credits'" in source


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


def test_api_keys_is_the_primary_credits_tab():
    tabs = APP_HTML[APP_HTML.index('<nav id="creditsTabs"'):APP_HTML.index('</nav>', APP_HTML.index('<nav id="creditsTabs"'))]
    assert tabs.index('data-credits-tab="api-keys"') < tabs.index('data-credits-tab="overview"')
    assert 'id="creditsTabApiKeys" class="credits-tab is-active"' in tabs
    assert 'id="creditsTabOverview" class="credits-tab"' in tabs
    assert 'data-credits-panel="api-keys">' in APP_HTML
    assert 'data-credits-panel="overview" hidden>' in APP_HTML


def test_saved_key_launch_controls_have_room_for_model_and_action():
    assert 'minmax(360px, 1.35fr)' in STYLES
    assert 'grid-template-columns: minmax(0, 1fr) max-content;' in STYLES
    assert 'white-space: nowrap;' in STYLES


def test_saved_key_actions_keep_reverify_with_status_and_delete_in_corner():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "credits-key-inline-action" in source
    assert "credits-key-delete" in source
    assert "'Delete'" in source
    assert "'Revoke'" not in source
    assert "mutateCredential(credential, 'revoke')" in source
    assert ".credits-key-inline-action" in STYLES
    assert ".credits-key-delete" in STYLES


def test_api_keys_client_never_persists_or_renders_full_secret():
    source = CREDITS_JS_PATH.read_text(encoding="utf-8")
    assert "/api/credits/model-providers" in source
    assert "/api/credits/api-keys" in source
    assert "creditsApiKeySecret" in source
    assert "secretInput.value = ''" in source
    assert "localStorage" not in source
    assert ".innerHTML" not in source
    assert "api_key: secret" in source
    assert "Available for metered ATL model runs. BYOK runs use your provider account and do not deduct ATL Credits." in APP_HTML


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
