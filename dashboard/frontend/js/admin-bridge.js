/** Bridge for the old-console modules absorbed into /admin (design D5).
 *
 * admin-credits.js and admin-model-providers.js were written for /app, where
 * app.js provided window.API.request (CSRF + JSON error contract) and
 * window.getStoredAuthUser. admin.html does not load app.js, so this module
 * stands in for both. It must load before those two; every lookup stays lazy
 * (inside the call), so ordering is a formality rather than a load-bearing
 * constraint.
 */
(function () {
  'use strict';

  function readCsrfToken() {
    try {
      const raw = document.cookie || '';
      for (const name of ['atl_csrf', '__Host-atl_csrf']) {
        const match = raw.match(new RegExp('(?:^|; )' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '=([^;]*)'));
        if (match) return decodeURIComponent(match[1]);
      }
    } catch (_error) { /* ignore */ }
    return null;
  }

  // Mirrors app.js's API.request error contract: message from the JSON body's
  // detail/error/message, `.status` set so handleAccessLost can react to
  // 401/403. The x-session-id headers app.js attached only matter to backtest
  // routes, none of which the admin console calls.
  const API = {
    async request(endpoint, options = {}) {
      const token = readCsrfToken();
      const headers = {
        'Content-Type': 'application/json',
        ...(token ? { 'X-CSRF-Token': token } : {}),
        ...options.headers,
      };
      const response = await fetch(endpoint, { ...options, headers, credentials: 'include' });
      const contentType = response.headers.get('content-type');
      let data;
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        const text = await response.text();
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${text.substring(0, 200)}`);
        }
        return text;
      }
      if (!response.ok) {
        const errorMsg = data.detail || data.error || data.message || `HTTP ${response.status}`;
        const error = new Error(typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg));
        error.status = response.status;
        throw error;
      }
      return data;
    },
  };

  // admin-shell.js's gate() has already verified the role against the server
  // before any absorbed module renders; the cache here only answers the same
  // "is the viewer an admin / is this row me?" questions the /app modules ask.
  function getStoredAuthUser() {
    return window.AdminShell?.state?.user || null;
  }

  window.API = API;
  window.getStoredAuthUser = getStoredAuthUser;
})();
