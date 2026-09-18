/** /admin account chip and page Refresh (design D5 layout).
 *
 * The app header's account menu is app.js territory; this is the one slice the
 * absorbed console needs — identity from the shell's verified user, and a
 * logout that reuses the bridge's CSRF-aware request.
 */
(function () {
  'use strict';

  function element(id) {
    return document.getElementById(id);
  }

  function renderIdentity(user) {
    if (!user) return;
    const avatar = element('adminAvatar');
    const label = element('adminUserLabel');
    const name = element('adminAccountName');
    const email = element('adminAccountEmail');
    const source = (user.display_name || user.email || '?').trim();
    if (avatar) avatar.textContent = source ? source[0].toUpperCase() : '?';
    if (label) label.textContent = source;
    if (name) name.textContent = user.display_name || source;
    if (email) email.textContent = user.email || '';
  }

  function closeMenu(wrap, menu) {
    if (!menu) return;
    menu.hidden = true;
    element('adminAccountBtn')?.setAttribute('aria-expanded', 'false');
    wrap?.removeEventListener('focusout', wrap._focusOut);
  }

  function toggleMenu() {
    const menu = element('adminAccountMenu');
    if (!menu) return;
    menu.hidden = !menu.hidden;
    element('adminAccountBtn')?.setAttribute('aria-expanded', String(!menu.hidden));
  }

  async function logout() {
    try {
      await window.API?.request('/api/auth/logout', { method: 'POST' });
    } catch (_error) {
      // A failed logout still ends the local view: the session cookie stays
      // until the server clears it, but the console must not pretend otherwise.
    }
    window.location.replace('/app');
  }

  function bind() {
    const btn = element('adminAccountBtn');
    const menu = element('adminAccountMenu');
    btn?.addEventListener('click', (event) => {
      event.stopPropagation();
      toggleMenu();
    });
    document.addEventListener('click', (event) => {
      if (menu && !menu.hidden && !menu.contains(event.target) && event.target !== btn) {
        menu.hidden = true;
        btn?.setAttribute('aria-expanded', 'false');
      }
    });
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape' || !menu || menu.hidden) return;
      menu.hidden = true;
      btn?.setAttribute('aria-expanded', 'false');
      btn?.focus();
    });
    element('adminAccountLogoutBtn')?.addEventListener('click', logout);
    element('adminRefreshBtn')?.addEventListener('click', () => {
      // One honest reload: every route's loaders re-run, the gate re-verifies,
      // and stale module state cannot survive the click.
      window.location.reload();
    });
  }

  window.AdminChrome = { renderIdentity };
  document.addEventListener('DOMContentLoaded', () => {
    bind();
    // The gate resolves before any absorbed panel renders; the chip just needs
    // the same user object. Retry a beat in case the gate is still in flight.
    if (window.AdminShell?.state?.user) {
      renderIdentity(window.AdminShell.state.user);
    } else {
      document.addEventListener('admin:gate-ready', () => {
        renderIdentity(window.AdminShell?.state?.user);
      }, { once: true });
    }
  });
})();
