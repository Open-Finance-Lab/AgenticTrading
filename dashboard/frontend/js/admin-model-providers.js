/** Separate Admin provider registry and platform credential controls. */
(function () {
  'use strict';

  const API_BASE = (
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ) ? window.location.origin : '';
  const state = { providers: [], initialized: false };

  function element(id) {
    return document.getElementById(id);
  }

  function apiRequest(path, options = {}) {
    return window.API.request(`${API_BASE}${path}`, options);
  }

  function clearChildren(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function textNode(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    return node;
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

  function renderProviderOptions() {
    const select = element('adminPlatformProvider');
    if (!select) return;
    const current = select.value;
    clearChildren(select);
    select.appendChild(textNode('option', '', 'Select a provider'));
    state.providers.forEach((provider) => {
      const option = document.createElement('option');
      option.value = provider.provider_id;
      option.textContent = provider.display_name;
      select.appendChild(option);
    });
    if (state.providers.some((provider) => provider.provider_id === current)) select.value = current;
  }

  function platformLabel(credential) {
    if (!credential) return 'No platform key';
    if (credential.status === 'verified') return `Verified •••• ${credential.key_last_four}`;
    if (credential.status === 'revoked') return 'Revoked';
    return `${credential.status.replaceAll('_', ' ')} •••• ${credential.key_last_four}`;
  }

  function renderProviderList() {
    const list = element('adminProviderList');
    if (!list) return;
    clearChildren(list);
    if (!state.providers.length) {
      list.appendChild(textNode('p', 'credits-muted', 'No providers registered.'));
      return;
    }
    state.providers.forEach((provider) => {
      const row = document.createElement('article');
      row.className = 'admin-provider-row';
      const head = document.createElement('div');
      head.className = 'admin-provider-row-head';
      head.appendChild(textNode('strong', '', provider.display_name));
      head.appendChild(textNode('span', `admin-provider-state is-${provider.status}`, provider.status));
      const meta = document.createElement('p');
      meta.className = 'admin-provider-row-meta';
      meta.textContent = `${provider.provider_id} · ${provider.adapter_type} · ${provider.approved_base_url}`;
      const credential = provider.platform_credential;
      const credentialLine = textNode('p', 'admin-provider-credential', platformLabel(credential));
      const actions = document.createElement('div');
      actions.className = 'admin-provider-row-actions';
      const edit = textNode('button', 'credits-key-action', 'Edit provider');
      edit.type = 'button';
      edit.addEventListener('click', () => fillProviderForm(provider));
      actions.appendChild(edit);
      if (credential && credential.status !== 'revoked') {
        const verify = textNode('button', 'credits-key-action', 'Reverify key');
        verify.type = 'button';
        appendIcon(verify, 'icon-refresh');
        verify.addEventListener('click', () => reverifyPlatformKey(provider.provider_id));
        actions.appendChild(verify);
        const revoke = textNode('button', 'credits-key-action is-danger', 'Revoke key');
        revoke.type = 'button';
        appendIcon(revoke, 'icon-x');
        revoke.addEventListener('click', () => {
          if (window.confirm(`Revoke the platform key for ${provider.display_name}?`)) revokePlatformKey(provider.provider_id);
        });
        actions.appendChild(revoke);
      }
      row.append(head, meta, credentialLine, actions);
      list.appendChild(row);
    });
  }

  async function loadProviders() {
    try {
      const data = await apiRequest('/api/admin/model-providers');
      state.providers = Array.isArray(data.providers) ? data.providers : [];
      renderProviderList();
      renderProviderOptions();
    } catch (error) {
      state.providers = [];
      renderProviderList();
      renderProviderOptions();
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
      await apiRequest(`/api/admin/model-providers/${encodeURIComponent(providerId)}`, {
        method: 'PUT',
        body: JSON.stringify(payload),
      });
      setStatus(element('adminProviderStatusMessage'), 'Provider saved.', 'success');
      await loadProviders();
    } catch (error) {
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
      await apiRequest(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential`, {
        method: 'PUT',
        body: JSON.stringify({ api_key: secret, ...actionPayload('Configure platform model access.') }),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key saved and verification requested.', 'success');
      await loadProviders();
    } catch (error) {
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be saved.', 'error');
    } finally {
      if (secretInput) secretInput.value = '';
    }
  }

  async function reverifyPlatformKey(providerId) {
    setStatus(element('adminPlatformKeyStatus'), 'Reverifying platform key…', 'pending');
    try {
      await apiRequest(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential/verify`, {
        method: 'POST',
        body: JSON.stringify(actionPayload('Retry platform provider verification.')),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key verification updated.', 'success');
      await loadProviders();
    } catch (error) {
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be verified.', 'error');
    }
  }

  async function revokePlatformKey(providerId) {
    setStatus(element('adminPlatformKeyStatus'), 'Revoking platform key…', 'pending');
    try {
      await apiRequest(`/api/admin/model-providers/${encodeURIComponent(providerId)}/platform-credential`, {
        method: 'DELETE',
        body: JSON.stringify(actionPayload('Revoke platform model access.')),
      });
      setStatus(element('adminPlatformKeyStatus'), 'Platform key revoked.', 'success');
      await loadProviders();
    } catch (error) {
      setStatus(element('adminPlatformKeyStatus'), error.message || 'Platform key could not be revoked.', 'error');
    }
  }

  function syncAuth(user) {
    if (user?.role === 'admin') return;
    const secretInput = element('adminPlatformKeySecret');
    if (secretInput) secretInput.value = '';
  }

  function onEnter() {
    if (!state.initialized) {
      state.initialized = true;
      element('adminProviderForm')?.addEventListener('submit', saveProvider);
      element('adminPlatformKeyForm')?.addEventListener('submit', savePlatformKey);
      element('adminProviderRefreshBtn')?.addEventListener('click', loadProviders);
    }
    if (window.getStoredAuthUser && window.getStoredAuthUser()?.role !== 'admin') return;
    loadProviders();
  }

  window.AdminModelProviders = { onEnter, syncAuth };
  document.addEventListener('DOMContentLoaded', () => {
    if (document.documentElement.dataset.navPage === 'admin') onEnter();
  });
})();
