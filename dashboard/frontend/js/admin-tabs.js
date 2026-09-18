/** Admin console tab state. No permissions or data access live here. */
(function () {
  'use strict';

  // Analytics lives on the standalone /admin page (design §7.5), and Providers
  // joined it in the 09-17 consolidation (N2). This console keeps Users (account
  // management + grant pool) and Activity until the follow-up port -- they are
  // one 712-line module and move together.
  const DEFAULT_TAB = 'users';
  const ALLOWED_TABS = new Set(['users', 'activity']);
  // N5: an explicit hop, not a silent fallback. normalizeTab would otherwise
  // coerce this to DEFAULT_TAB and land the operator on Account Management with
  // nothing on screen saying the tab they asked for lives elsewhere now.
  // replace, not assign: a retired URL must not stay a reachable history entry.
  // replace drops the *current* one -- the ?adminTab=providers URL being
  // navigated away from -- and leaves the entry before it, the one the operator
  // actually came from, untouched. assign kept the retired URL in history, so
  // Back either re-ran this redirect with no way out of it, or restored the old
  // console from bfcache still showing Account Management: setTab returns before
  // it paints, so the bfcache copy is the silent fallback this hop exists to
  // prevent.
  const RETIRED_TABS = Object.freeze({ providers: '/admin#providers' });
  let initialized = false;

  function retiredDestination(value) {
    return Object.hasOwn(RETIRED_TABS, value) ? RETIRED_TABS[value] : null;
  }

  function normalizeTab(value) {
    const normalizedValue = value === 'grant-pool' ? 'users' : value;
    return ALLOWED_TABS.has(normalizedValue) ? normalizedValue : DEFAULT_TAB;
  }

  function setTab(value, { updateUrl = true } = {}) {
    const retired = retiredDestination(value);
    if (retired) {
      window.location.replace(retired);
      return value;
    }
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
  //
  // The one thing init must not do is navigate off a page the operator did not
  // ask to leave. It runs off DOMContentLoaded on *every* /app load, so an
  // `adminTab` left in the URL by a shared link or an old bookmark reached the
  // retired-tab hop while the visitor was reading Community -- and, since this
  // fires before any identity is known, it did so for signed-out visitors too,
  // bouncing them to /admin only for gate() to bounce them straight back.
  // Gating on `view=admin` is the same fix PR #468 applied to this file's
  // earlier retired-tab redirect, and for the same reason: the hop is a
  // courtesy owed to someone opening the admin console, not to every /app URL
  // that happens to carry the parameter. Role cannot be the gate here -- this
  // controller has no session at DOMContentLoaded -- but intent can. (Spelling
  // that older destination out is what a pinned guard in
  // test_admin_tabs_redirect.py forbids, so it stays unnamed here.)
  function init() {
    bind();
    const params = new URL(window.location.href).searchParams;
    const requested = params.get('adminTab');
    if (retiredDestination(requested) && params.get('view') !== 'admin') {
      setTab(DEFAULT_TAB);
      return;
    }
    setTab(requested || DEFAULT_TAB);
  }

  window.AdminTabs = { onEnter, openAccountManagement, setTab };
  document.addEventListener('DOMContentLoaded', init);
})();
