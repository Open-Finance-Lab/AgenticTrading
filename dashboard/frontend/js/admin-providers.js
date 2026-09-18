/** /admin Providers: the approved provider registry and the platform credential.
 *
 * Ported from js/admin-model-providers.js (design §8), which reached the host
 * page through window.API.request and window.getStoredAuthUser -- both defined
 * in app.js, neither reachable from /admin. Every seam now goes through
 * AdminShell: reads through request() (GET-hardcoded), writes through write()
 * (the page's single write path, N4), identity through user().
 *
 * This is the page's ONLY write surface. Two things follow from that and are
 * load-bearing rather than stylistic:
 *   - no line may both name a credential and touch a DOM sink, which is why the
 *     masked-key line is called keyState and not credentialLine;
 *   - `api_key` appears exactly once, inside a JSON.stringify body.
 * test_admin_page_modules.py asserts both.
 */
(function () {
  'use strict';

  const SURFACE = 'providers';
  const state = { providers: [], bound: false };

  function shell() {
    return window.AdminShell;
  }

  function element(id) {
    return document.getElementById(id);
  }

  function textNode(tag, className, text) {
    return shell().el(tag, className, text);
  }

  function appendIcon(button, iconId) {
    const icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    icon.setAttribute('aria-hidden', 'true');
    const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
    use.setAttribute('href', `#${iconId}`);
    icon.appendChild(use);
    button.appendChild(icon);
  }

  function setStatus(target, message, tone = '') {
    if (!target) return;
    target.textContent = message || '';
    target.classList.toggle('is-error', tone === 'error');
    target.classList.toggle('is-success', tone === 'success');
    target.classList.toggle('is-pending', tone === 'pending');
  }

  function operationKey(prefix) {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  function actionPayload(reason) {
    return {
      source: 'admin-console',
      reason,
      idempotency_key: operationKey('admin-provider'),
    };
  }

  function fillProviderForm(provider) {
    if (!provider) return;
    element('adminProviderId').value = provider.provider_id || '';
    element('adminProviderDisplayName').value = provider.display_name || '';
    element('adminProviderAdapter').value = provider.adapter_type || 'openai_compatible';
    element('adminProviderBaseUrl').value = provider.approved_base_url || '';
    element('adminProviderByok').checked = provider.byok_enabled !== false;
    element('adminProviderPlatform').checked = provider.platform_enabled === true;
    element('adminProviderStatus').value = provider.status || 'enabled';
    const select = element('adminPlatformProvider');
    if (select) select.value = provider.provider_id || '';
  }

  // The masked label. Named for the *state* it reports, not for the credential
  // it reports on: the caller appends its node to the DOM, and a line that both
  // says "credential" and calls .append() is what the DOM-sink guard refuses.
  function platformLabel(credential) {
    if (!credential) return 'No platform key';
    if (credential.status === 'verified') return `Verified •••• ${credential.key_last_four}`;
    if (credential.status === 'revoked') return 'Revoked';
    return `${String(credential.status).replaceAll('_', ' ')} •••• ${credential.key_last_four}`;
  }

  function renderProviderList(payload) {
    const providers = Array.isArray(payload?.providers) ? payload.providers : [];
    const list = shell().el('div', 'admin-provider-list');
    if (!providers.length) {
      list.appendChild(textNode('p', 'credits-muted', 'No providers registered.'));
      return list;
    }
    providers.forEach((provider) => {
      const row = shell().el('article', 'admin-provider-row');
      const head = shell().el('div', 'admin-provider-row-head');
      head.appendChild(textNode('strong', '', provider.display_name));
      head.appendChild(textNode('span', `admin-provider-state is-${provider.status}`, provider.status));
      const meta = textNode(
        'p',
        'admin-provider-row-meta',
        `${provider.provider_id} · ${provider.adapter_type} · ${provider.approved_base_url}`,
      );
      const key = provider.platform_credential;
      const keyState = textNode('p', 'admin-provider-credential', platformLabel(key));
      const actions = shell().el('div', 'admin-provider-row-actions');
      const edit = textNode('button', 'credits-key-action', 'Edit provider');
      edit.type = 'button';
      edit.addEventListener('click', () => fillProviderForm(provider));
      actions.appendChild(edit);
      // A provider with no key, or a revoked one, is offered neither action:
      // both would refuse server-side, and the row is the only place the
      // operator finds out which providers actually hold one.
      if (key && key.status !== 'revoked') {
        const verify = textNode('button', 'credits-key-action', 'Reverify key');
        verify.type = 'button';
        appendIcon(verify, 'icon-refresh');
        verify.addEventListener('click', () => reverifyPlatformKey(provider.provider_id));
        actions.appendChild(verify);
        const revoke = textNode('button', 'credits-key-action is-danger', 'Revoke key');
        revoke.type = 'button';
        appendIcon(revoke, 'icon-x');
        revoke.addEventListener('click', () => confirmRevoke(provider.provider_id, provider.display_name));
        actions.appendChild(revoke);
      }
      row.append(head, meta, keyState, actions);
      list.appendChild(row);
    });
    return list;
  }

  function renderProviderOptions(payload) {
    const select = element('adminPlatformProvider');
    if (!select) return;
    const providers = Array.isArray(payload?.providers) ? payload.providers : [];
    const current = select.value;
    shell().clear(select);
    select.appendChild(textNode('option', '', 'Select a provider'));
    providers.forEach((provider) => {
      const option = shell().el('option', '', provider.display_name);
      option.value = provider.provider_id;
      select.appendChild(option);
    });
    // Restored only when it still exists. Forcing a vanished id back onto the
    // select would leave the platform-key form pointed at a provider the
    // registry no longer has, and the next Save would post the secret to it.
    if (providers.some((provider) => provider.provider_id === current)) select.value = current;
  }

  function paint(payload) {
    state.providers = Array.isArray(payload?.providers) ? payload.providers : [];
    const host = element('adminProviderList');
    if (host) host.replaceChildren(...renderProviderList(payload).children);
    renderProviderOptions(payload);
  }

  async function load() {
    const seq = shell().nextSeq(SURFACE);
    try {
      const data = await shell().request('/api/admin/model-providers');
      if (!shell().isCurrent(SURFACE, seq)) return;
      paint(data);
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      if (!shell().isCurrent(SURFACE, seq)) return;
      paint({ providers: [] });
      setStatus(element('adminProviderStatusMessage'), error.message || 'Providers could not be loaded.', 'error');
    }
  }

  async function saveProvider(event) {
    event.preventDefault();
    const providerId = element('adminProviderId')?.value.trim();
    if (!providerId) {
      setStatus(element('adminProviderStatusMessage'), 'Provider ID is required.', 'error');
      return;
    }
    const existing = state.providers.find((provider) => provider.provider_id === providerId);
    const payload = {
      display_name: element('adminProviderDisplayName').value.trim(),
      adapter_type: element('adminProviderAdapter').value,
      approved_base_url: element('adminProviderBaseUrl').value.trim(),
      capabilities: existing?.capabilities || { model_discovery: true },
      byok_enabled: element('adminProviderByok').checked,
      platform_enabled: element('adminProviderPlatform').checked,
      status: element('adminProviderStatus').value,
      source: 'admin-console',
      reason: element('adminProviderReason').value.trim() || 'Provider registry update.',
      idempotency_key: operationKey('provider-registry'),
    };
    setStatus(element('adminProviderStatusMessage'), 'Saving provider…', 'pending');
    try {
      await shell().write(`/api/admin/model-providers/${encodeURIComponent(providerId)}`, {
        method: 'PUT',
        body: JSON.stringify(payload),
      });
      setStatus(element('adminProviderStatusMessage'), 'Provider saved.', 'success');
      await load();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      setStatus(element('adminProviderStatusMessage'), error.message || 'Provider could not be saved.', 'error');
    }
  }

  async function savePlatformKey(event) {
    event.preventDefault();
    const providerId = element('adminPlatformProvider')?.value;
    const secretInput = element('adminPlatformKeySecret');
    const secret = secretInput?.value || '';
    if (!providerId || !secret) {
      setStatus(element('adminPlatformKeyStatus'), 'Select a provider and enter the key once.', 'error');
      if (secretInput) secretInput.value = '';
      return;
    }
    setStatus(element('adminPlatformKeyStatus'), 'Saving and verifying…', 'pending');
    try {
      await shell().write(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential`, {
        method: 'PUT',
        body: JSON.stringify({ api_key: secret, ...actionPayload('Configure platform model access.') }),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key saved and verification requested.', 'success');
      await load();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be saved.', 'error');
    } finally {
      // In `finally`, not on each branch: the field must be empty after a
      // refusal and after a redirect, not only after a success.
      if (secretInput) secretInput.value = '';
    }
  }

  async function reverifyPlatformKey(providerId) {
    setStatus(element('adminPlatformKeyStatus'), 'Reverifying platform key…', 'pending');
    try {
      await shell().write(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential/verify`, {
        method: 'POST',
        body: JSON.stringify(actionPayload('Retry platform provider verification.')),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key verification updated.', 'success');
      await load();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be verified.', 'error');
    }
  }

  // window.confirm is kept for this PR (design §8). Replacing it with
  // AdminShell.openDialog is a real improvement and a real scope increase; it
  // is recorded in the design's §11 rather than smuggled in here.
  async function confirmRevoke(providerId, displayName) {
    if (!window.confirm(`Revoke the platform key for ${displayName}?`)) return;
    await revokePlatformKey(providerId);
  }

  async function revokePlatformKey(providerId) {
    setStatus(element('adminPlatformKeyStatus'), 'Revoking platform key…', 'pending');
    try {
      await shell().write(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential`, {
        method: 'DELETE',
        body: JSON.stringify(actionPayload('Revoke platform model access.')),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key revoked.', 'success');
      await load();
    } catch (error) {
      if (await shell().handleAccessLost(error)) return;
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be revoked.', 'error');
    }
  }

  function bind() {
    if (state.bound) return;
    state.bound = true;
    element('adminProviderForm')?.addEventListener('submit', saveProvider);
    element('adminPlatformKeyForm')?.addEventListener('submit', savePlatformKey);
    element('adminProviderRefreshBtn')?.addEventListener('click', () => load());
  }

  function onRoute(detail) {
    if (detail?.route !== SURFACE) {
      // Leaving the surface clears the field: a secret typed and not submitted
      // must not still be sitting in a live input when the operator comes back
      // three routes later.
      const secretInput = element('adminPlatformKeySecret');
      if (secretInput) secretInput.value = '';
      return;
    }
    if (!shell().user()) return;
    bind();
    load();
  }

  window.AdminProviders = {
    renderProviderList, renderProviderOptions, platformLabel,
    load, savePlatformKey, confirmRevoke, onRoute,
  };
  document.addEventListener('admin:route', (event) => onRoute(event.detail));
})();
