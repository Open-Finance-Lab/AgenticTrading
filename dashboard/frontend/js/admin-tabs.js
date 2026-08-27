/** Admin console tab state. No permissions or data access live here. */
(function () {
  'use strict';

  const DEFAULT_TAB = 'analytics';
  const ALLOWED_TABS = new Set(['analytics', 'users', 'providers', 'activity']);
  let initialized = false;

  function normalizeTab(value) {
    const normalizedValue = value === 'grant-pool' ? 'users' : value;
    return ALLOWED_TABS.has(normalizedValue) ? normalizedValue : DEFAULT_TAB;
  }

  function setTab(value, { updateUrl = true } = {}) {
    const tab = normalizeTab(value);
    document.querySelectorAll('[data-admin-tab]').forEach((button) => {
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
    document.querySelectorAll('[data-admin-tab]').forEach((button) => {
      button.addEventListener('click', () => setTab(button.dataset.adminTab));
      button.addEventListener('keydown', (event) => {
        const keys = new Set(['ArrowRight', 'ArrowLeft', 'Home', 'End']);
        if (!keys.has(event.key)) return;
        event.preventDefault();
        const buttons = [...document.querySelectorAll('[data-admin-tab]')];
        const index = buttons.indexOf(button);
        const next = event.key === 'Home'
          ? buttons[0]
          : event.key === 'End'
            ? buttons[buttons.length - 1]
            : event.key === 'ArrowRight'
              ? buttons[(index + 1) % buttons.length]
              : buttons[(index - 1 + buttons.length) % buttons.length];
        next.focus();
        setTab(next.dataset.adminTab);
      });
    });
  }

  function onEnter() {
    bind();
    const requested = new URL(window.location.href).searchParams.get('adminTab');
    setTab(requested || DEFAULT_TAB);
  }

  function openAccountManagement({ userId, email } = {}) {
    setTab('users');
    const url = new URL(window.location.href);
    url.searchParams.delete('analyticsUser');
    url.searchParams.delete('analyticsSection');
    window.history.replaceState(window.history.state, '', url);
    const input = document.getElementById('adminCreditsUserQuery');
    const form = document.getElementById('adminCreditsUserSearch');
    if (input) input.value = String(email || userId || '');
    form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    input?.focus();
  }

  window.AdminTabs = { onEnter, openAccountManagement, setTab };
  document.addEventListener('DOMContentLoaded', onEnter);
})();
