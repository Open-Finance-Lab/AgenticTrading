/** Credits & Billing page. Server ledgers remain the only source of balance changes. */
(function () {
  'use strict';

  const { formatCredits } = window.CreditFormat;
  const MAX_ORDER_POLLS = 8;
  const ORDER_POLL_DELAYS_MS = [0, 1000, 1500, 2500, 4000, 6000, 8000, 10000];
  const TERMINAL_ORDER_STATUSES = new Set(['paid', 'partially_refunded', 'refunded']);
  const CREDITS_API_BASE = (
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ) ? window.location.origin : '';
  const PENDING_BYOK_STORAGE_KEY = 'atlPendingByokBacktest';
  const PENDING_BYOK_TTL_MS = 10 * 60 * 1000;
  const OFFICIAL_API_KEY_PAGES = Object.freeze({
    openai: Object.freeze({
      displayName: 'OpenAI',
      url: 'https://platform.openai.com/api-keys',
    }),
    openrouter: Object.freeze({
      displayName: 'OpenRouter',
      url: 'https://openrouter.ai/keys',
    }),
    anthropic: Object.freeze({
      displayName: 'Anthropic',
      url: 'https://platform.claude.com/settings/keys',
    }),
    gemini: Object.freeze({
      displayName: 'Google Gemini',
      url: 'https://aistudio.google.com/apikey',
    }),
  });

  const state = {
    initialized: false,
    user: null,
    selection: { kind: 'package', value: 'usd_1' },
    pendingPurchase: null,
    pendingRefund: null,
    selectedAdminOrder: null,
    orderPollToken: 0,
    balanceMicro: 0,
    activeTab: 'api-keys',
    providers: [],
    credentials: [],
    executionOptions: [],
    focusApiKeysOnReady: false,
  };

  function element(id) {
    return document.getElementById(id);
  }

  function apiRequest(path, options = {}) {
    return window.API.request(`${CREDITS_API_BASE}${path}`, options);
  }

  function setStatus(target, message, tone = '') {
    if (!target) return;
    target.textContent = message || '';
    target.classList.toggle('is-error', tone === 'error');
    target.classList.toggle('is-success', tone === 'success');
    target.classList.toggle('is-pending', tone === 'pending');
  }

  function setApiKeyTroubleshooting(visible) {
    const help = element('creditsApiKeyTroubleshooting');
    if (help) help.hidden = !visible;
  }

  function formatUsdCents(cents) {
    if (!Number.isSafeInteger(cents)) return '—';
    const sign = cents < 0 ? '-' : '';
    const absolute = Math.abs(cents);
    return `${sign}$${Math.floor(absolute / 100).toLocaleString('en-US')}.${String(absolute % 100).padStart(2, '0')}`;
  }

  function parseUsdCents(raw) {
    const text = String(raw || '').trim();
    const match = /^(\d{1,3})(?:\.(\d{1,2}))?$/.exec(text);
    if (!match) return null;
    const cents = Number(match[1]) * 100 + Number((match[2] || '').padEnd(2, '0'));
    return Number.isSafeInteger(cents) ? cents : null;
  }

  function formatTimestamp(value) {
    const date = new Date(value);
    if (!value || Number.isNaN(date.getTime())) return 'Unknown time';
    return date.toLocaleString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
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

  function setCreditsTab(tab, { reload = true } = {}) {
    const allowed = new Set(['overview', 'api-keys', 'activity']);
    const next = tab === 'top-up' ? 'overview' : (allowed.has(tab) ? tab : 'overview');
    state.activeTab = next;
    document.querySelectorAll('[data-credits-tab]').forEach((button) => {
      const selected = button.dataset.creditsTab === next;
      button.classList.toggle('is-active', selected);
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('[data-credits-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.creditsPanel !== next;
    });
    if (reload && next === 'api-keys' && state.user) loadApiKeys();
  }

  function focusApiKeysEntry() {
    const provider = element('creditsApiKeyProvider');
    const heading = element('creditsApiKeysHeading');
    const target = provider && !provider.disabled ? provider : heading;
    state.focusApiKeysOnReady = false;
    window.requestAnimationFrame(() => target?.focus());
  }

  function openApiKeys({ focus = false } = {}) {
    setCreditsTab('api-keys', { reload: false });
    if (!focus) return;
    state.focusApiKeysOnReady = true;
    if (state.providers.length || element('creditsApiKeyProvider')?.disabled) {
      focusApiKeysEntry();
    }
  }

  function renderProviderOptions() {
    const select = element('creditsApiKeyProvider');
    if (!select) {
      renderProviderGuide();
      if (state.focusApiKeysOnReady) focusApiKeysEntry();
      return;
    }
    const current = select.value;
    clearChildren(select);
    if (!state.providers.length) {
      select.appendChild(textNode('option', '', 'No approved providers available'));
      select.value = '';
      select.disabled = true;
      renderProviderGuide();
      if (state.focusApiKeysOnReady) focusApiKeysEntry();
      return;
    }
    select.disabled = false;
    state.providers.forEach((provider) => {
      const option = document.createElement('option');
      option.value = provider.provider_id;
      option.textContent = provider.display_name;
      select.appendChild(option);
    });
    if (state.providers.some((provider) => provider.provider_id === current)) {
      select.value = current;
    }
    renderProviderGuide();
    if (state.focusApiKeysOnReady) focusApiKeysEntry();
  }

  function providerDisplayName(providerId) {
    return state.providers.find((provider) => provider.provider_id === providerId)?.display_name || providerId;
  }

  function renderProviderGuide() {
    const guide = element('creditsApiKeyGuide');
    const steps = element('creditsApiKeyGuideSteps');
    const fallback = element('creditsApiKeyGuideFallback');
    const officialLink = element('creditsApiKeyOfficialLink');
    const providerCopy = element('creditsApiKeyGuideProvider');
    const providerId = element('creditsApiKeyProvider')?.value || '';
    const selectedProvider = state.providers.find(
      (provider) => provider.provider_id === providerId,
    ) || null;
    const officialPage = OFFICIAL_API_KEY_PAGES[providerId] || null;

    if (!guide || !steps || !fallback || !officialLink || !providerCopy) return;
    guide.hidden = !selectedProvider;
    steps.hidden = !officialPage;
    fallback.hidden = Boolean(officialPage);
    officialLink.removeAttribute('href');
    officialLink.removeAttribute('aria-label');

    if (!selectedProvider || !officialPage) return;
    officialLink.href = officialPage.url;
    officialLink.setAttribute(
      'aria-label',
      `Open ${officialPage.displayName} official API key page in a new tab`,
    );
    providerCopy.textContent = (
      `Continue on ${officialPage.displayName}, then return to ATL.`
    );
  }

  function onApiKeyProviderChange() {
    setApiKeyTroubleshooting(false);
    renderProviderGuide();
  }

  function credentialStatusLabel(status) {
    if (status === 'verified') return 'Verified';
    if (status === 'invalid') return 'Invalid';
    if (status === 'revoked') return 'Revoked';
    return 'Verification unavailable';
  }

  function executionOption(providerId) {
    return state.executionOptions.find(
      (item) => item.provider_id === providerId,
    ) || null;
  }

  function renderApiKeys(items) {
    const list = element('creditsApiKeyList');
    const count = element('creditsApiKeyCount');
    if (!list) return;
    state.credentials = Array.isArray(items) ? items : [];
    if (count) count.textContent = `${state.credentials.length} saved`;
    clearChildren(list);
    if (!state.credentials.length) {
      list.appendChild(textNode('p', 'credits-muted', 'No API keys saved yet.'));
      return;
    }
    state.credentials.forEach((credential) => {
      const row = document.createElement('article');
      row.className = 'credits-key-row';

      const head = document.createElement('div');
      head.className = 'credits-key-row-head';
      head.appendChild(textNode('strong', 'credits-key-name', credential.label));
      const status = textNode('span', `credits-key-status is-${credential.status}`, credentialStatusLabel(credential.status));
      head.appendChild(status);

      const meta = document.createElement('div');
      meta.className = 'credits-key-meta';
      meta.appendChild(textNode('span', '', providerDisplayName(credential.provider_id)));
      meta.appendChild(textNode('span', 'credits-key-last-four', `•••• ${credential.key_last_four}`));
      if (credential.verification_message) {
        meta.appendChild(textNode('span', 'credits-key-verification-message', credential.verification_message));
      }
      if (credential.is_default) meta.appendChild(textNode('span', 'credits-key-default-badge', 'Default'));

      const actions = document.createElement('div');
      actions.className = 'credits-key-actions';
      if (credential.status !== 'revoked') {
        const verify = textNode(
          'button',
          'credits-key-action credits-key-inline-action',
          'Reverify',
        );
        verify.type = 'button';
        verify.title = 'Verify this key again';
        appendIcon(verify, 'icon-refresh');
        verify.addEventListener('click', () => mutateCredential(credential, 'verify'));
        head.appendChild(verify);
        if (credential.status === 'verified' && !credential.is_default) {
          const makeDefault = textNode('button', 'credits-key-action', 'Set default');
          makeDefault.type = 'button';
          makeDefault.title = 'Set as the verified default';
          appendIcon(makeDefault, 'icon-badge-check');
          makeDefault.addEventListener('click', () => mutateCredential(credential, 'default'));
          actions.appendChild(makeDefault);
        }
        const deleteButton = textNode(
          'button',
          'credits-key-action credits-key-delete is-danger',
          'Delete',
        );
        deleteButton.type = 'button';
        deleteButton.title = 'Delete this key';
        appendIcon(deleteButton, 'icon-x');
        deleteButton.addEventListener('click', () => {
          if (window.confirm(`Delete “${credential.label}”?`)) {
            mutateCredential(credential, 'revoke');
          }
        });
        actions.appendChild(deleteButton);
      }

      row.append(head, meta);
      if (credential.status === 'verified' && credential.is_default) {
        const launch = document.createElement('div');
        launch.className = 'credits-key-launch';

        const modelLabel = textNode(
          'label',
          'credits-key-model-label',
          'Model',
        );
        const modelSelect = document.createElement('select');
        modelSelect.className = 'credits-key-model';
        modelSelect.setAttribute(
          'aria-label',
          `Model for ${credential.label}`,
        );

        const option = executionOption(credential.provider_id);
        (option?.models || []).forEach((model) => {
          const modelOption = document.createElement('option');
          modelOption.value = model.model_id;
          modelOption.textContent = model.label;
          modelSelect.appendChild(modelOption);
        });

        const run = textNode(
          'button',
          'credits-key-action credits-key-run',
          'Run Backtest',
        );
        run.type = 'button';
        run.disabled = (
          !option?.byok_available
          || modelSelect.options.length === 0
        );
        run.addEventListener(
          'click',
          () => beginByokBacktest(credential, modelSelect),
        );

        launch.append(modelLabel, modelSelect, run);
        row.appendChild(launch);
      }
      row.appendChild(actions);
      list.appendChild(row);
    });
  }

  function beginByokBacktest(credential, modelSelect) {
    const modelId = String(modelSelect?.value || '').trim();
    if (
      credential.status !== 'verified'
      || !credential.is_default
      || !modelId
    ) {
      setStatus(
        element('creditsApiKeyStatus'),
        (
          'Choose a verified default key and model '
          + 'before starting a backtest.'
        ),
        'error',
      );
      return;
    }
    sessionStorage.setItem(
      PENDING_BYOK_STORAGE_KEY,
      JSON.stringify({
        billing_mode: 'byok',
        provider_id: credential.provider_id,
        model_id: modelId,
        expires_at: Date.now() + PENDING_BYOK_TTL_MS,
      }),
    );
    const target = new URL(window.location.href);
    target.searchParams.set('view', 'agents');
    window.location.assign(target.href);
  }

  async function loadApiKeys() {
    if (!state.user) return;
    const [
      providersResult,
      credentialsResult,
      executionOptionsResult,
    ] = await Promise.allSettled([
      apiRequest('/api/credits/model-providers'),
      apiRequest('/api/credits/api-keys'),
      apiRequest('/api/credits/execution-options'),
    ]);
    if (providersResult.status === 'fulfilled') {
      state.providers = providersResult.value.providers || [];
      renderProviderOptions();
    } else {
      state.providers = [];
      renderProviderOptions();
      setStatus(element('creditsApiKeyStatus'), 'Approved providers could not be loaded.', 'error');
    }
    if (executionOptionsResult.status === 'fulfilled') {
      state.executionOptions = executionOptionsResult.value.providers || [];
    } else {
      state.executionOptions = [];
      setStatus(element('creditsApiKeyStatus'), 'Backtest options could not be loaded.', 'error');
    }
    if (credentialsResult.status === 'fulfilled') {
      renderApiKeys(credentialsResult.value.items || []);
    } else {
      renderApiKeys([]);
      setStatus(element('creditsApiKeyStatus'), 'Saved API keys could not be loaded.', 'error');
    }
  }

  async function saveApiKey(event) {
    event.preventDefault();
    const provider = element('creditsApiKeyProvider')?.value;
    const labelInput = element('creditsApiKeyLabel');
    const secretInput = element('creditsApiKeySecret');
    const defaultInput = element('creditsApiKeyDefault');
    const save = element('creditsApiKeySave');
    const secret = secretInput?.value || '';
    setApiKeyTroubleshooting(false);
    if (!provider || !labelInput?.value.trim() || !secret) {
      setStatus(element('creditsApiKeyStatus'), 'Choose a provider, name the key, and enter the key once.', 'error');
      if (secretInput) secretInput.value = '';
      return;
    }
    if (save) save.disabled = true;
    setStatus(element('creditsApiKeyStatus'), 'Saving and verifying…', 'pending');
    try {
      const data = await apiRequest('/api/credits/api-keys', {
        method: 'POST',
        body: JSON.stringify({
          provider_id: provider,
          label: labelInput.value.trim(),
          api_key: secret,
          set_default: Boolean(defaultInput?.checked),
        }),
      });
      const status = data.credential?.status;
      setApiKeyTroubleshooting(status !== 'verified');
      setStatus(
        element('creditsApiKeyStatus'),
        status === 'verified' ? 'Key saved and verified.' : 'Key saved. Verification can be retried from the list.',
        status === 'verified' ? 'success' : 'pending',
      );
      labelInput.value = '';
      if (defaultInput) defaultInput.checked = false;
      await loadApiKeys();
    } catch (error) {
      setApiKeyTroubleshooting(true);
      setStatus(element('creditsApiKeyStatus'), error.message || 'The key could not be saved.', 'error');
    } finally {
      // The full secret exists only for this submit lifecycle and is never put
      // in state, browser storage, URL parameters, or a rendered node.
      if (secretInput) secretInput.value = '';
      if (save) save.disabled = false;
    }
  }

  async function mutateCredential(credential, action) {
    const endpoint = action === 'verify'
      ? `/api/credits/api-keys/${encodeURIComponent(credential.credential_id)}/verify`
      : action === 'default'
        ? `/api/credits/api-keys/${encodeURIComponent(credential.credential_id)}/default`
        : `/api/credits/api-keys/${encodeURIComponent(credential.credential_id)}`;
    try {
      setStatus(element('creditsApiKeyStatus'), action === 'revoke' ? 'Revoking key…' : 'Updating key…', 'pending');
      await apiRequest(endpoint, { method: action === 'revoke' ? 'DELETE' : 'POST' });
      setStatus(element('creditsApiKeyStatus'), action === 'revoke' ? 'Key revoked.' : 'Key updated.', 'success');
      await loadApiKeys();
    } catch (error) {
      setStatus(element('creditsApiKeyStatus'), error.message || 'The key could not be updated.', 'error');
    }
  }

  function syncAuth(user) {
    const wasSignedIn = Boolean(state.user);
    state.user = user || null;
    const signedIn = Boolean(state.user);
    const signedInPanel = element('creditsSignedIn');
    const signedOutPanel = element('creditsSignedOut');
    if (signedInPanel) signedInPanel.hidden = !signedIn;
    if (signedOutPanel) signedOutPanel.hidden = signedIn;
    if (!signedIn) {
      state.orderPollToken += 1;
      state.pendingPurchase = null;
      state.providers = [];
      state.credentials = [];
      state.executionOptions = [];
      const secretInput = element('creditsApiKeySecret');
      if (secretInput) secretInput.value = '';
      setApiKeyTroubleshooting(false);
      renderProviderOptions();
      renderProviderGuide();
      renderApiKeys([]);
    } else if (!wasSignedIn && state.initialized && document.documentElement.dataset.navPage === 'credits') {
      loadBalanceAndLedger();
      loadAdminOrders();
    }
  }

  function setPurchaseEnabled(enabled) {
    const button = element('creditsPurchaseBtn');
    if (button) button.disabled = !enabled;
    document.querySelectorAll('[data-credit-package]').forEach((packageButton) => {
      packageButton.disabled = !enabled;
    });
    const custom = element('creditsCustomAmount');
    if (custom) custom.disabled = !enabled;
  }

  function renderBalance(balance) {
    state.balanceMicro = Number.isSafeInteger(balance.balance_micro) ? balance.balance_micro : 0;
    element('creditsBalance').textContent = `${formatCredits(balance.display_credits)} Credits`;
    const accountStatus = element('creditsAccountStatus');
    const restricted = balance.account_status !== 'active';
    const reason = balance.restriction_reason;
    const outstandingMicro = Number(balance.outstanding_credits_micro || 0);
    if (reason === 'llm_overage') {
      setStatus(
        accountStatus,
        `Model usage exceeded the reserved amount. Add at least ${window.CreditFormat.formatCreditsMicro(outstandingMicro)} Credits to restore access.`,
        'error',
      );
    } else if (restricted) {
      setStatus(accountStatus, 'Purchases are paused for a payment refund review. Contact an administrator to restore access.', 'error');
    } else if (!balance.billing_available) {
      setStatus(accountStatus, 'Stripe Test Mode is not configured on this server.', 'error');
    } else {
      setStatus(accountStatus, 'Available for metered ATL model runs. BYOK runs use your provider account and do not deduct ATL Credits.', 'success');
    }
    setPurchaseEnabled((!restricted || reason === 'llm_overage') && balance.billing_available);
  }

  function renderLedger(items) {
    const list = element('creditsLedgerList');
    clearChildren(list);
    if (!items.length) {
      list.appendChild(textNode('p', 'credits-muted', 'No Credit activity yet.'));
      return;
    }
    items.forEach((entry) => {
      const row = document.createElement('div');
      row.className = 'credits-ledger-row';

      const meta = document.createElement('div');
      meta.className = 'credits-ledger-meta';
      const isUsage = entry.entry_type === 'backtest_usage';
      const isWelcomeGrant = entry.entry_type === 'system_promotion_grant';
      const isNegative = isUsage || entry.entry_type === 'refund';
      const title = isUsage
        ? 'Backtest usage'
        : (entry.entry_type === 'refund'
          ? 'Refund'
          : (isWelcomeGrant ? 'Welcome Credits' : 'Credit purchase'));
      meta.appendChild(textNode('strong', '', title));
      const callCount = Number.isSafeInteger(entry.model_call_count)
        && entry.model_call_count > 0
        ? `${entry.model_call_count} model call${entry.model_call_count === 1 ? '' : 's'}`
        : null;
      const usageDetail = isUsage
        ? [
          entry.provider_mixed ? 'Multiple providers' : entry.provider_id,
          entry.model_mixed ? 'Multiple models' : entry.model_id,
          callCount,
          entry.run_id ? `run ${String(entry.run_id).slice(0, 12)}` : null,
        ]
          .filter(Boolean)
          .join(' · ')
        : null;
      meta.appendChild(textNode(
        'span',
        '',
        [usageDetail, formatTimestamp(entry.created_at)].filter(Boolean).join(' · '),
      ));

      const formatted = formatCredits(entry.display_credits);
      const amount = textNode(
        'span',
        `credits-ledger-amount ${isNegative ? 'is-negative' : 'is-positive'}`,
        `${isNegative ? '' : '+'}${formatted}`,
      );
      row.append(meta, amount);
      list.appendChild(row);
    });
  }

  async function loadBalanceAndLedger() {
    const [balanceResult, ledgerResult] = await Promise.allSettled([
      apiRequest('/api/credits/balance'),
      apiRequest('/api/credits/ledger?limit=50'),
    ]);

    if (balanceResult.status === 'fulfilled') {
      renderBalance(balanceResult.value.balance);
    } else {
      element('creditsBalance').textContent = 'Unavailable';
      setStatus(element('creditsAccountStatus'), balanceResult.reason.message, 'error');
      setPurchaseEnabled(false);
    }

    if (ledgerResult.status === 'fulfilled') {
      renderLedger(ledgerResult.value.items || []);
    } else {
      renderLedger([]);
      setStatus(element('creditsPurchaseStatus'), 'Recent activity could not be loaded.', 'error');
    }
  }

  function setSelection(kind, value) {
    state.selection = { kind, value };
    state.pendingPurchase = null;
    const purchaseLabel = element('creditsPurchaseBtn')?.querySelector('span');
    if (purchaseLabel) purchaseLabel.textContent = 'Continue to Stripe';
    document.querySelectorAll('[data-credit-package]').forEach((button) => {
      const selected = kind === 'package' && button.dataset.creditPackage === value;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-checked', selected ? 'true' : 'false');
    });
  }

  function selectedCheckoutPayload() {
    if (state.selection.kind === 'package') {
      return { package_id: state.selection.value };
    }
    const cents = parseUsdCents(element('creditsCustomAmount')?.value);
    if (cents === null || cents < 50 || cents > 500) {
      throw new Error('Enter a custom amount from $0.50 through $5.00.');
    }
    return { custom_amount_usd_cents: cents };
  }

  function trustedStripeCheckoutUrl(rawUrl) {
    let parsed;
    try {
      parsed = new URL(rawUrl);
    } catch (_) {
      return null;
    }
    return parsed.protocol === 'https:' && parsed.hostname === 'checkout.stripe.com'
      ? parsed.href
      : null;
  }

  async function beginPurchase() {
    const button = element('creditsPurchaseBtn');
    const label = button?.querySelector('span');
    setStatus(element('creditsPurchaseStatus'), '');
    try {
      if (!state.pendingPurchase) {
        state.pendingPurchase = {
          client_request_id: crypto.randomUUID(),
          ...selectedCheckoutPayload(),
        };
      }
      if (button) button.disabled = true;
      if (label) label.textContent = 'Opening Stripe…';
      const data = await apiRequest('/api/credits/checkout-sessions', {
        method: 'POST',
        body: JSON.stringify(state.pendingPurchase),
      });
      const checkoutUrl = trustedStripeCheckoutUrl(data.checkout?.checkout_url);
      if (!checkoutUrl) throw new Error('The server returned an untrusted checkout address.');
      window.location.assign(checkoutUrl);
    } catch (error) {
      setStatus(element('creditsPurchaseStatus'), error.message, 'error');
      if (button) button.disabled = false;
      if (label) label.textContent = 'Retry Stripe checkout';
    }
  }

  function cleanCheckoutQuery() {
    const url = new URL(window.location.href);
    url.searchParams.delete('session_id');
    url.searchParams.delete('order_id');
    url.searchParams.delete('payment');
    window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
  }

  async function pollOrder(orderId, attempt, token) {
    if (token !== state.orderPollToken) return;
    if (attempt >= MAX_ORDER_POLLS) {
      setStatus(element('creditsPurchaseStatus'), 'Payment confirmation pending. Refresh this page in a moment.', 'pending');
      cleanCheckoutQuery();
      return;
    }

    const delay = ORDER_POLL_DELAYS_MS[attempt] || 10000;
    window.setTimeout(async () => {
      if (token !== state.orderPollToken) return;
      try {
        const data = await apiRequest(`/api/credits/orders/${encodeURIComponent(orderId)}`);
        const status = data.order?.status;
        if (TERMINAL_ORDER_STATUSES.has(status)) {
          setStatus(element('creditsPurchaseStatus'), 'Payment confirmed. Credits are now available.', 'success');
          cleanCheckoutQuery();
          await loadBalanceAndLedger();
          if (state.user?.role === 'admin') await loadAdminOrders();
          return;
        }
        if (status === 'failed' || status === 'expired') {
          setStatus(element('creditsPurchaseStatus'), 'Payment was not completed. No Credits were added.', 'error');
          cleanCheckoutQuery();
          return;
        }
        setStatus(element('creditsPurchaseStatus'), 'Waiting for Stripe payment confirmation…', 'pending');
      } catch (error) {
        setStatus(element('creditsPurchaseStatus'), 'Checking payment confirmation…', 'pending');
      }
      pollOrder(orderId, attempt + 1, token);
    }, delay);
  }

  function inspectCheckoutReturn() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('payment') === 'cancelled') {
      setStatus(element('creditsPurchaseStatus'), 'Stripe checkout was cancelled. No Credits were added.');
      cleanCheckoutQuery();
      return;
    }
    const orderId = params.get('order_id');
    if (!orderId) return;
    const token = ++state.orderPollToken;
    setStatus(element('creditsPurchaseStatus'), 'Waiting for Stripe payment confirmation…', 'pending');
    pollOrder(orderId, 0, token);
  }

  function orderCell(text, className = '') {
    return textNode('td', className, text);
  }

  function renderAdminOrders(orders) {
    const body = element('creditsAdminOrders');
    clearChildren(body);
    if (!orders.length) {
      const row = document.createElement('tr');
      const cell = orderCell('No payment orders yet.', 'credits-admin-empty');
      cell.colSpan = 6;
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }

    orders.forEach((order) => {
      const row = document.createElement('tr');
      row.appendChild(orderCell(order.order_id));
      row.appendChild(orderCell(String(order.user_id)));
      row.appendChild(orderCell(formatUsdCents(order.amount_usd_cents)));
      row.appendChild(orderCell(String(order.status).replaceAll('_', ' '), 'credits-order-status'));
      row.appendChild(orderCell(formatUsdCents(order.refundable_usd_cents)));

      const actionCell = document.createElement('td');
      if (order.refundable_usd_cents > 0) {
        const refundButton = textNode('button', 'credits-refund-btn', 'Refund');
        refundButton.type = 'button';
        refundButton.addEventListener('click', () => openRefundDialog(order));
        actionCell.appendChild(refundButton);
      }
      row.appendChild(actionCell);
      body.appendChild(row);
    });
  }

  async function loadAdminOrders() {
    const section = element('creditsAdminSection');
    const isAdmin = state.user?.role === 'admin';
    if (section) section.hidden = !isAdmin;
    if (!isAdmin) return;
    try {
      const data = await apiRequest('/api/admin/credits/orders?limit=50');
      renderAdminOrders(data.items || []);
      setStatus(element('creditsAdminStatus'), '');
    } catch (error) {
      renderAdminOrders([]);
      setStatus(element('creditsAdminStatus'), error.message, 'error');
    }
  }

  function openRefundDialog(order) {
    state.selectedAdminOrder = order;
    state.pendingRefund = null;
    element('creditsRefundOrder').textContent = `Order ${order.order_id}`;
    element('creditsRefundLimit').textContent = `Up to ${formatUsdCents(order.refundable_usd_cents)} can be refunded.`;
    element('creditsRefundAmount').value = (order.refundable_usd_cents / 100).toFixed(2);
    setStatus(element('creditsRefundStatus'), '');
    const dialog = element('creditsRefundDialog');
    if (dialog && !dialog.open) dialog.showModal();
  }

  function closeRefundDialog() {
    const dialog = element('creditsRefundDialog');
    if (dialog?.open) dialog.close();
  }

  async function submitRefund(event) {
    event.preventDefault();
    const order = state.selectedAdminOrder;
    const submit = element('creditsRefundSubmit');
    if (!order) return;
    try {
      const cents = parseUsdCents(element('creditsRefundAmount').value);
      if (cents === null || cents <= 0 || cents > order.refundable_usd_cents) {
        throw new Error(`Enter an amount up to ${formatUsdCents(order.refundable_usd_cents)}.`);
      }
      // Reuse the pending request verbatim on a retry: the server derives the
      // refund id from client_request_id, so the same id reuses the existing
      // reservation and Stripe's idempotency key instead of stacking a second
      // reservation against the order's refundable amount.
      //
      // But only while the request is genuinely the same. Editing the amount
      // after a failed attempt and pressing Submit used to validate the new
      // number and then send the old one, so the admin was told a refund they
      // could see on screen had been requested while a different amount went to
      // Stripe. A changed amount is a different request and gets a new id.
      if (
        !state.pendingRefund
        || state.pendingRefund.payment_order_id !== order.order_id
        || state.pendingRefund.amount_usd_cents !== cents
      ) {
        state.pendingRefund = {
          client_request_id: crypto.randomUUID(),
          payment_order_id: order.order_id,
          amount_usd_cents: cents,
        };
      }
      if (submit) submit.disabled = true;
      setStatus(element('creditsRefundStatus'), 'Submitting refund to Stripe…', 'pending');
      await apiRequest('/api/admin/credits/refunds', {
        method: 'POST',
        body: JSON.stringify(state.pendingRefund),
      });
      setStatus(element('creditsRefundStatus'), 'Refund requested. Waiting for Stripe confirmation.', 'success');
      await loadAdminOrders();
      window.setTimeout(closeRefundDialog, 900);
    } catch (error) {
      setStatus(element('creditsRefundStatus'), error.message, 'error');
    } finally {
      if (submit) submit.disabled = false;
    }
  }

  function wireControls() {
    document.querySelectorAll('[data-credit-package]').forEach((button) => {
      button.addEventListener('click', () => {
        const custom = element('creditsCustomAmount');
        if (custom) custom.value = '';
        setSelection('package', button.dataset.creditPackage);
      });
    });
    element('creditsCustomAmount')?.addEventListener('input', (event) => {
      setSelection('custom', event.target.value);
    });
    element('creditsPurchaseBtn')?.addEventListener('click', beginPurchase);
    element('creditsRefreshBtn')?.addEventListener('click', onEnter);
    element('creditsSignInBtn')?.addEventListener('click', () => element('authSignInBtn')?.click());
    element('creditsRefundClose')?.addEventListener('click', closeRefundDialog);
    element('creditsRefundCancel')?.addEventListener('click', closeRefundDialog);
    element('creditsRefundForm')?.addEventListener('submit', submitRefund);
    document.querySelectorAll('[data-credits-tab]').forEach((button) => {
      button.addEventListener('click', () => setCreditsTab(button.dataset.creditsTab));
    });
    element('creditsApiKeyProvider')?.addEventListener('change', onApiKeyProviderChange);
    element('creditsApiKeyForm')?.addEventListener('submit', saveApiKey);
  }

  async function onEnter() {
    if (!state.initialized) {
      state.initialized = true;
      wireControls();
    }
    syncAuth(window.getStoredAuthUser ? window.getStoredAuthUser() : window.AUTH_USER);
    if (!state.user) return;
    setStatus(element('creditsAccountStatus'), 'Loading account…', 'pending');
    await loadBalanceAndLedger();
    await loadApiKeys();
    await loadAdminOrders();
    inspectCheckoutReturn();
  }

  window.CreditsPage = { onEnter, syncAuth, openApiKeys };
})();
