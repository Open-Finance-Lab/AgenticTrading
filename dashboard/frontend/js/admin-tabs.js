/** Admin console tab state. No permissions or data access live here. */
(function () {
  'use strict';

  // Analytics lives on the standalone /admin page (design §7.5). This console
  // keeps Users (account management + grant pool), Providers and Activity until
  // the follow-up port (D5). Nothing here navigates away from /app any more.
  const DEFAULT_TAB = 'users';
  const ALLOWED_TABS = new Set(['users', 'providers', 'activity']);
  let initialized = false;

  function normalizeTab(value) {
    const normalizedValue = value === 'grant-pool' ? 'users' : value;
    return ALLOWED_TABS.has(normalizedValue) ? normalizedValue : DEFAULT_TAB;
  }

  function setTab(value, { updateUrl = true } = {}) {
    const tab = normalizeTab(value);
    const tablist = document.getElementById('adminTabs');
    tablist?.querySelectorAll('[data-admin-tab]').forEach((button) => {
      const selected = button.dataset.adminTab === tab;
      button.classList.toggle('is-active', selected);
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('[data-admin-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.adminPanel !== tab;
    });
    if (updateUrl && window.history?.replaceState) {
      const url = new URL(window.location.href);
      url.searchParams.set('adminTab', tab);
      window.history.replaceState(window.history.state, '', url);
    }
    document.dispatchEvent(new CustomEvent('admin:tabchange', { detail: { tab } }));
    return tab;
  }

  function bind() {
    if (initialized) return;
    initialized = true;
    const tablist = document.getElementById('adminTabs');
    tablist?.querySelectorAll('[data-admin-tab]').forEach((button) => {
      button.addEventListener('click', () => setTab(button.dataset.adminTab));
      button.addEventListener('keydown', (event) => {
        const keys = new Set(['ArrowDown', 'ArrowUp', 'Home', 'End']);
        if (!keys.has(event.key)) return;
        event.preventDefault();
        const buttons = [...tablist.querySelectorAll('[data-admin-tab]')];
        const index = buttons.indexOf(button);
        const next = event.key === 'Home'
          ? buttons[0]
          : event.key === 'End'
            ? buttons[buttons.length - 1]
            : event.key === 'ArrowDown'
              ? buttons[(index + 1) % buttons.length]
              : buttons[(index - 1 + buttons.length) % buttons.length];
        next.focus();
        setTab(next.dataset.adminTab);
      });
    });
    window.addEventListener('popstate', () => {
      const requested = new URL(window.location.href).searchParams.get('adminTab');
      setTab(requested || DEFAULT_TAB, { updateUrl: false });
    });
  }

  function openAccountManagement({ userId, email } = {}) {
    setTab('users');
    const url = new URL(window.location.href);
    url.searchParams.delete('adminUserQuery');
    window.history.replaceState(window.history.state, '', url);
    const input = document.getElementById('adminCreditsUserQuery');
    const form = document.getElementById('adminCreditsUserSearch');
    if (input) input.value = String(email || userId || '');
    form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    input?.focus();
  }

  function onEnter() {
    bind();
    const params = new URL(window.location.href).searchParams;
    setTab(params.get('adminTab') || DEFAULT_TAB);
    // The /admin profile deep-links here with the account to look up, the
    // hand-off its in-process predecessor (the evidence dialog) used to make.
    const query = params.get('adminUserQuery');
    if (query) openAccountManagement({ email: query });
  }

  // Page load only initialises panel state; app.js routes `?view=admin` to
  // navigateToPage('admin') → onEnter.
  function init() {
    bind();
    const requested = new URL(window.location.href).searchParams.get('adminTab');
    setTab(requested || DEFAULT_TAB);
  }

  window.AdminTabs = { onEnter, openAccountManagement, setTab };
  document.addEventListener('DOMContentLoaded', init);
})();
