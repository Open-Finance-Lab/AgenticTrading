/** Admin console tab state. No permissions or data access live here. */
(function () {
  'use strict';

  const DEFAULT_TAB = 'users';
  const ALLOWED_TABS = new Set(['users', 'providers', 'activity']);
  let initialized = false;

  function setTab(value, { updateUrl = true } = {}) {
    const normalizedValue = value === 'grant-pool' ? DEFAULT_TAB : value;
    const tab = ALLOWED_TABS.has(normalizedValue) ? normalizedValue : DEFAULT_TAB;
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
  }

  function bind() {
    if (initialized) return;
    initialized = true;
    document.querySelectorAll('[data-admin-tab]').forEach((button) => {
      button.addEventListener('click', () => setTab(button.dataset.adminTab));
      button.addEventListener('keydown', (event) => {
        if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
        const buttons = [...document.querySelectorAll('[data-admin-tab]')];
        const index = buttons.indexOf(button);
        const next = event.key === 'ArrowRight'
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

  window.AdminTabs = { onEnter, setTab };
  document.addEventListener('DOMContentLoaded', onEnter);
})();
