"""Static contracts for the separate Admin provider vault surface."""

from pathlib import Path
import re


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")
APP_JS = (FRONTEND / "app.js").read_text(encoding="utf-8")
ADMIN_JS = (FRONTEND / "js" / "admin-model-providers.js").read_text(encoding="utf-8")
STYLES = (FRONTEND / "styles.css").read_text(encoding="utf-8")


def test_admin_provider_surface_is_separate_from_credits_api_keys():
    admin_start = APP_HTML.index('id="adminView"')
    admin_end = APP_HTML.index('id="creditsRefundDialog"')
    admin_markup = APP_HTML[admin_start:admin_end]
    assert 'id="adminModelProvidersSection"' in admin_markup
    assert 'id="adminProviderList"' in admin_markup
    assert 'id="adminPlatformKeySecret"' in admin_markup
    assert 'id="creditsApiKeySecret"' not in admin_markup


def test_admin_provider_client_uses_admin_routes_and_clears_secret():
    assert "/api/admin/model-providers" in ADMIN_JS
    assert "adminPlatformKeySecret" in ADMIN_JS
    assert "secretInput.value = ''" in ADMIN_JS
    assert "localStorage" not in ADMIN_JS
    assert ".innerHTML" not in ADMIN_JS
    assert "api_key: secret" in ADMIN_JS
    assert "crypto.randomUUID()" in ADMIN_JS


def test_admin_provider_surface_has_responsive_styles():
    assert ".admin-provider-section" in STYLES
    assert ".admin-provider-grid" in STYLES
    assert ".admin-provider-form > .auth-btn" in STYLES
    assert ".admin-platform-key-form > .auth-btn" in STYLES
    assert re.search(
        r"\.admin-provider-form \.auth-btn svg\s*,\s*\.admin-platform-key-form \.auth-btn svg\s*\{[^}]*width:\s*15px;[^}]*height:\s*15px;",
        STYLES,
        re.DOTALL,
    )
    assert "@media (max-width: 600px)" in STYLES


def test_admin_provider_module_is_wired_to_auth_navigation_and_refresh():
    assert "window.AdminModelProviders.syncAuth(user)" in APP_JS
    assert APP_JS.count("window.AdminModelProviders.onEnter()") >= 2
