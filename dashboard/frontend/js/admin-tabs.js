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

  // `leave` gates the hand-off to the standalone /admin-analytics page. It is
  // true only on admin *intent* — entering the admin view, clicking the tab.
  // The page-load and popstate calls below pass false: they run on every /app
  // load to initialise hidden panel state, and a redirect there bounced every
  // view of the app to /admin-analytics (PR #467 regression).
  function setTab(value, { updateUrl = true, leave = true } = {}) {
    const tab = normalizeTab(value);
    if (tab === 'analytics' && leave) {
      // The Analytics tab now lives on the standalone /admin-analytics page
      // (preview with synthetic data). Leave the in-app panel in place but
      // never show it; providers/users/activity still render here.
      // `replace`, not `assign`: `/app?view=admin` must not stay in history,
      // or Back reloads it, app.js routes to admin, and we redirect forward
      // again — a loop the user cannot escape with the Back button.
      window.location.replace('/admin-analytics');
      return tab;
    }
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
      setTab(requested || DEFAULT_TAB, { updateUrl: false, leave: false });
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
    url.searchParams.delete('analyticsProfile');
    url.searchParams.delete('analyticsSection');
    window.history.replaceState(window.history.state, '', url);
    const input = document.getElementById('adminCreditsUserQuery');
    const form = document.getElementById('adminCreditsUserSearch');
    if (input) input.value = String(email || userId || '');
    form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    input?.focus();
  }

  // Page load only initialises panel state; app.js routes `?view=admin` to
  // navigateToPage('admin') → onEnter, which is where the redirect belongs.
  function init() {
    bind();
    const requested = new URL(window.location.href).searchParams.get('adminTab');
    setTab(requested || DEFAULT_TAB, { leave: false });
  }

  window.AdminTabs = { onEnter, openAccountManagement, setTab };
  document.addEventListener('DOMContentLoaded', init);
})();
