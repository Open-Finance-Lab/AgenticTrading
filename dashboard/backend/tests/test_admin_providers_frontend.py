"""js/admin-providers.js under node: registry rows, platform-key states, writes.

Replaces test_admin_model_providers_frontend.py, whose four cases all asserted
against artifacts the 09-17 consolidation removes (the ?view=admin markup, the
old module, the app.js wiring, the styles.css families). The assertions that are
still true about the surface -- admin routes only, the secret field cleared on
every exit, no localStorage, no innerHTML -- are carried forward below.
"""

import re
from pathlib import Path

from dashboard.backend.tests._admin_dom_stub import requires_node, run_node, source
from dashboard.backend.tests._frontend_source import fn_body, strip_comments

pytestmark = requires_node

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
ADMIN_HTML = (FRONTEND / "admin.html").read_text(encoding="utf-8")
ADMIN_CSS = (FRONTEND / "admin.css").read_text(encoding="utf-8")
SHELL = source("admin-shell.js")
PROVIDERS_JS = source("admin-providers.js")
# Every source-shape assertion below runs on the stripped copy. This module's
# header comment names what it replaced -- window.API.request,
# getStoredAuthUser -- so a raw scan for those strings would fail on the prose
# explaining their absence, which is the inverse of the trap `strip_comments`
# was written for (a guard satisfied by a comment rather than by the code).
PROVIDERS_CODE = strip_comments(PROVIDERS_JS)

PROVIDERS = """{"providers": [
  {"provider_id": "openrouter", "display_name": "OpenRouter", "adapter_type": "openrouter",
   "approved_base_url": "https://openrouter.ai/api/v1", "status": "enabled",
   "byok_enabled": true, "platform_enabled": true,
   "platform_credential": {"status": "verified", "key_last_four": "9f21"}},
  {"provider_id": "anthropic", "display_name": "Anthropic", "adapter_type": "anthropic",
   "approved_base_url": "https://api.anthropic.com/v1", "status": "disabled",
   "byok_enabled": true, "platform_enabled": false, "platform_credential": null},
  {"provider_id": "gemini", "display_name": "Gemini", "adapter_type": "gemini",
   "approved_base_url": "https://generativelanguage.googleapis.com/v1", "status": "enabled",
   "byok_enabled": false, "platform_enabled": true,
   "platform_credential": {"status": "pending_verification", "key_last_four": "0c4d"}}
]}"""


def _eval(expression: str, *setup: str) -> object:
    return run_node(
        SHELL, PROVIDERS_JS, *setup,
        f"Promise.resolve({expression}).then((result) => console.log(JSON.stringify(result)));",
    )


def test_registry_rows_show_name_state_identity_and_key_status():
    result = _eval(
        "(() => {"
        f"  const list = window.AdminProviders.renderProviderList({PROVIDERS});"
        "  return list.children.map((row) => ["
        "    byTag(row, 'STRONG')[0].textContent,"
        "    byClass(row, 'admin-provider-state')[0].textContent,"
        "    byClass(row, 'admin-provider-row-meta')[0].textContent,"
        "    byClass(row, 'admin-provider-credential')[0].textContent,"
        "  ]);"
        "})()"
    )
    assert result == [
        ["OpenRouter", "enabled", "openrouter · openrouter · https://openrouter.ai/api/v1", "Verified •••• 9f21"],
        ["Anthropic", "disabled", "anthropic · anthropic · https://api.anthropic.com/v1", "No platform key"],
        ["Gemini", "enabled", "gemini · gemini · https://generativelanguage.googleapis.com/v1",
         "pending verification •••• 0c4d"],
    ]


def test_only_a_live_key_offers_reverify_and_revoke():
    """A provider with no key, or a revoked one, must not offer actions that
    would 404 or 409 -- the row is the only place the operator learns which."""
    result = _eval(
        "(() => {"
        f"  const payload = {PROVIDERS};"
        "  payload.providers.push({provider_id: 'dead', display_name: 'Dead', adapter_type: 'openai',"
        "    approved_base_url: 'https://x.test/v1', status: 'disabled',"
        "    platform_credential: {status: 'revoked', key_last_four: 'aaaa'}});"
        "  const list = window.AdminProviders.renderProviderList(payload);"
        "  return list.children.map((row) => byTag(row, 'BUTTON').map((b) => b.textContent));"
        "})()"
    )
    assert result == [
        ["Edit provider", "Reverify key", "Revoke key"],
        ["Edit provider"],
        ["Edit provider", "Reverify key", "Revoke key"],
        ["Edit provider"],
    ]


def test_an_empty_registry_says_so_rather_than_rendering_nothing():
    result = _eval(
        "(() => {"
        "  const list = window.AdminProviders.renderProviderList({providers: []});"
        "  return [list.children.length, list.textContent];"
        "})()"
    )
    assert result == [1, "No providers registered."]


def test_a_non_array_providers_field_is_treated_as_empty_not_crashed_on():
    result = _eval(
        "(() => window.AdminProviders.renderProviderList({providers: null}).textContent)()"
    )
    assert result == "No providers registered."


def test_the_provider_select_keeps_the_operators_choice_across_a_refresh():
    """Reloading the registry after a save must not silently move the platform-key
    form to a different provider -- the next Save would write the key to it.

    This pins the *restore*, not the vanished-provider case: a real <select>
    whose value no longer matches any option falls back to the first one on its
    own, and the node stub models `value` as a plain property, so asserting on
    it there would pin the stub rather than the browser."""
    result = _eval(
        "(() => {"
        "  const select = register('adminPlatformProvider', new Node('select'));"
        "  select.value = 'anthropic';"
        f"  window.AdminProviders.renderProviderOptions({PROVIDERS});"
        "  const kept = select.value;"
        f"  const payload = {PROVIDERS};"
        "  payload.providers = payload.providers.filter((p) => p.provider_id !== 'anthropic');"
        "  window.AdminProviders.renderProviderOptions(payload);"
        "  return {kept, options: select.children.map((o) => o.textContent)};"
        "})()"
    )
    assert result["kept"] == "anthropic"
    assert result["options"] == ["Select a provider", "OpenRouter", "Gemini"]
    # The restore is guarded rather than unconditional; an unguarded
    # `select.value = current` is what would point the form at a gone provider.
    assert "providers.some((provider) => provider.provider_id === current)" in PROVIDERS_CODE


def test_reads_go_through_the_shell_get():
    """Renamed: it only ever exercised load(). The writes are covered by the two
    tests below, which execute them rather than matching `method:` literals."""
    result = _eval(
        "(async () => {"
        f"  fetchQueue.push({{ok: true, status: 200, body: {PROVIDERS}}});"
        "  register('adminProviderList', new Node('div'));"
        "  register('adminPlatformProvider', new Node('select'));"
        "  await window.AdminProviders.load();"
        "  return fetchCalls.map(([url, options]) => [url, options.method]);"
        "})()"
    )
    assert result == [["/api/admin/model-providers", "GET"]]


SECRET = "sk-platform-do-not-leak"


def test_the_platform_key_write_carries_the_secret_only_in_a_json_body():
    """The one call in this module that carries a credential (savePlatformKey).
    Nothing executed it before: the source-shape guard pins the string `api_key`
    to a line that also says `JSON.stringify(`, which cannot tell a body from a
    query string, and the URL and verb were pinned by a `method: 'PUT'` literal
    that would ship green if either were wrong."""
    result = _eval(
        "(async () => {"
        "  const secretInput = register('adminPlatformKeySecret', new Node('input'));"
        "  const select = register('adminPlatformProvider', new Node('select'));"
        "  register('adminPlatformKeyStatus', new Node('p'));"
        "  register('adminProviderList', new Node('div'));"
        f"  secretInput.value = '{SECRET}';"
        "  select.value = 'openrouter';"
        "  fetchQueue.push({ok: true, status: 200, body: {status: 'pending_verification'}});"
        f"  fetchQueue.push({{ok: true, status: 200, body: {PROVIDERS}}});"
        "  await window.AdminProviders.savePlatformKey({preventDefault() {}});"
        "  const [url, options] = fetchCalls[0];"
        "  return {url, method: options.method, credentials: options.credentials,"
        "          csrf: options.headers['X-CSRF-Token'], bodyType: typeof options.body,"
        "          sentKey: JSON.parse(options.body).api_key,"
        "          headerText: Object.values(options.headers).join(' | '),"
        "          left: secretInput.value,"
        "          reload: [fetchCalls[1][0], fetchCalls[1][1].method]};"
        "})()",
        "globalThis.document.cookie = 'atl_csrf=tok';",
    )
    assert result["url"] == "/api/admin/model-providers/openrouter/platform-credential"
    assert result["method"] == "PUT"
    assert result["credentials"] == "include"
    # Only AdminShell.write attaches the double-submit header. A raw fetch() here
    # would be a 403 in production that no fetch-stubbed test could otherwise see.
    assert result["csrf"] == "tok"
    # The secret travels as a JSON body and nowhere a proxy, a referrer or a
    # server access log would record it.
    assert result["bodyType"] == "string"
    assert result["sentKey"] == SECRET
    assert SECRET not in result["url"]
    assert SECRET not in result["headerText"]
    # ...and it does not survive in the live input afterwards.
    assert result["left"] == ""
    assert result["reload"] == ["/api/admin/model-providers", "GET"]


def test_every_write_call_site_reaches_its_own_url_and_verb_through_the_shell():
    """All four re-pointed call sites, driven through the affordances that reach
    them: the registry form, the platform-key form, and the two row buttons. A
    wrong URL or verb in any of them used to ship green, because the only thing
    holding them was a string literal in a source-shape scan."""
    result = _eval(
        "(async () => {"
        "  ['adminProviderId', 'adminProviderDisplayName', 'adminProviderAdapter',"
        "   'adminProviderBaseUrl', 'adminProviderStatus', 'adminProviderReason',"
        "   'adminPlatformKeySecret'].forEach((id) => { register(id, new Node('input')).value = ''; });"
        "  register('adminProviderByok', new Node('input')).checked = true;"
        "  register('adminProviderPlatform', new Node('input')).checked = true;"
        "  ['adminProviderStatusMessage', 'adminPlatformKeyStatus'].forEach("
        "    (id) => register(id, new Node('p')));"
        "  const host = register('adminProviderList', new Node('div'));"
        "  const form = register('adminProviderForm', new Node('form'));"
        "  register('adminPlatformKeyForm', new Node('form'));"
        "  register('adminProviderRefreshBtn', new Node('button'));"
        "  const select = register('adminPlatformProvider', new Node('select'));"
        f"  for (let i = 0; i < 12; i += 1) fetchQueue.push({{ok: true, status: 200, body: {PROVIDERS}}});"
        "  window.AdminShell.state.user = {email: 'ada@example.test'};"
        "  window.AdminProviders.onRoute({route: 'providers'});"
        "  await new Promise((resolve) => setTimeout(resolve, 0));"
        "  elements['adminProviderId'].value = 'openrouter';"
        "  await form.listeners.submit[0]({preventDefault() {}});"
        "  elements['adminPlatformKeySecret'].value = 'sk-sweep';"
        "  select.value = 'openrouter';"
        "  await window.AdminProviders.savePlatformKey({preventDefault() {}});"
        "  const reverify = byTag(host, 'BUTTON').find((b) => b.textContent === 'Reverify key');"
        "  await reverify.listeners.click[0]();"
        "  window.confirm = () => true;"
        "  await window.AdminProviders.confirmRevoke('openrouter', 'OpenRouter');"
        "  return {calls: fetchCalls.map(([url, o]) => [o.method, url]),"
        "          signed: fetchCalls.filter(([, o]) => o.method !== 'GET')"
        "            .every(([, o]) => o.headers['X-CSRF-Token'] === 'tok')};"
        "})()",
        "globalThis.document.cookie = 'atl_csrf=tok';",
    )
    read = ["GET", "/api/admin/model-providers"]
    credential = "/api/admin/model-providers/openrouter/platform-credential"
    assert result["calls"] == [
        read,
        ["PUT", "/api/admin/model-providers/openrouter"], read,
        ["PUT", credential], read,
        ["POST", f"{credential}/verify"], read,
        ["DELETE", credential], read,
    ]
    # Every write, not just the one above, goes through AdminShell.write (N4).
    assert result["signed"] is True


def test_the_platform_key_secret_is_cleared_on_every_exit():
    """Carried forward from test_admin_model_providers_frontend.py. The field is
    cleared on the refusal path as well as the success path: a secret left in a
    live input survives every later navigation on a single-page console."""
    assert PROVIDERS_CODE.count("secretInput.value = ''") >= 2
    refused = _eval(
        "(async () => {"
        "  const secretInput = register('adminPlatformKeySecret', new Node('input'));"
        "  const select = register('adminPlatformProvider', new Node('select'));"
        "  register('adminPlatformKeyStatus', new Node('p'));"
        "  secretInput.value = 'sk-should-not-survive';"
        "  select.value = '';"
        "  await window.AdminProviders.savePlatformKey({preventDefault() {}});"
        "  return {left: secretInput.value, calls: fetchCalls.length};"
        "})()"
    )
    assert refused == {"left": "", "calls": 0}


def test_revoking_asks_first_and_does_nothing_when_the_operator_declines():
    result = _eval(
        "(async () => {"
        "  register('adminPlatformKeyStatus', new Node('p'));"
        "  window.confirm = () => false;"
        "  await window.AdminProviders.confirmRevoke('openrouter', 'OpenRouter');"
        "  return fetchCalls.length;"
        "})()",
        "globalThis.document.cookie = 'atl_csrf=t';",
    )
    assert result == 0


def test_the_module_is_an_iife_with_one_global_and_no_fetch_of_its_own():
    assert PROVIDERS_JS.lstrip().startswith("/**")
    assert "(function () {\n  'use strict';" in PROVIDERS_JS
    assert set(re.findall(r"^\s*window\.(\w+) = ", PROVIDERS_CODE, re.M)) == {"AdminProviders"}
    # Stripped, so the header comment naming the two seams this port replaced
    # cannot satisfy -- or fail -- the assertion that they are gone.
    assert "fetch(" not in PROVIDERS_CODE
    assert "window.API" not in PROVIDERS_CODE
    assert "getStoredAuthUser" not in PROVIDERS_CODE
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "localStorage", "sessionStorage"):
        assert forbidden not in PROVIDERS_CODE, forbidden


def test_every_renderer_is_a_named_function_the_shared_harness_can_slice():
    """§7.6 / §12: the other five admin modules are testable because every
    renderer is a named function `fn_body` can brace-match out. An arrow
    assigned to a const, or a closure inside load(), is not liftable and the
    whole module then has to be exercised through the DOM."""
    for signature in (
        "function renderProviderList(", "function renderProviderOptions(",
        "function platformLabel(", "function fillProviderForm(",
    ):
        body = fn_body(signature, PROVIDERS_JS)
        assert "innerHTML" not in body, signature
    assert "textContent" in fn_body("function setStatus(", PROVIDERS_JS)


def test_the_providers_section_is_flat_so_the_panel_guard_can_see_all_of_it():
    """PANEL_REGION in test_admin_page_shell.py is non-greedy, so an outer
    <section data-panel> wrapping an inner <section> matches only as far as the
    inner closing tag and the numeric-literal guard silently scans a fragment.
    The port flattens the app.html wrapper to a <div> for exactly that reason."""
    start = ADMIN_HTML.index('<section id="providersView"')
    end = ADMIN_HTML.index("</section>", start)
    markup = ADMIN_HTML[start:end]
    assert "<section" not in markup[1:]
    for control in ("adminProviderList", "adminProviderForm", "adminPlatformKeyForm",
                    "adminPlatformKeySecret", "adminProviderRefreshBtn"):
        assert f'id="{control}"' in markup, control
    assert 'type="password"' in markup and 'autocomplete="new-password"' in markup


def test_the_provider_css_is_retyped_here_and_not_imported():
    for family in (".admin-provider-row", ".admin-provider-state", ".admin-provider-grid",
                   ".admin-platform-key-form", ".auth-btn", ".control-select",
                   ".credits-key-field", ".credits-key-action", ".credits-icon-btn"):
        assert family in ADMIN_CSS, family
    assert "@import" not in ADMIN_CSS
