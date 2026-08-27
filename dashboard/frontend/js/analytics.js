/** Privacy-reduced, authenticated browser Analytics lifecycle events. */
(function () {
  'use strict';

  const SESSION_STORAGE_KEY = 'atl-analytics-session-v1';
  const SESSION_TIMEOUT_MS = 30 * 60 * 1000;
  const HEARTBEAT_INTERVAL_MS = 30 * 1000;
  const MAX_VISIBLE_MS = 1_800_000;
  const CANONICAL_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
  const PAGE_VIEW_MAP = Object.freeze({
    home: 'home',
    playground_agents: 'agents',
    agent_editor: 'agent_editor',
    playground_backtest: 'backtest',
    playground_paper: 'paper_trading',
    competition: 'competition',
    community: 'community',
    credits: 'credits',
    account: 'account',
  });

  let memorySession = null;
  let basePageView = null;
  let currentPageView = null;
  let currentViewStartedAt = null;
  let transientPageView = null;
  let transientReturnView = null;
  let viewSuspended = false;
  let deliveryWarningShown = false;

  function signedIn() {
    try {
      return (
        typeof window.getStoredAuthUser === 'function' &&
        Boolean(window.getStoredAuthUser())
      );
    } catch (error) {
      return false;
    }
  }

  function validSession(value, now) {
    if (!value || typeof value !== 'object') return false;
    if (!CANONICAL_UUID.test(value.id || '')) return false;
    if (!Number.isFinite(value.last_activity_at)) return false;
    const age = now - value.last_activity_at;
    return age >= 0 && age < SESSION_TIMEOUT_MS;
  }

  function readSession(now) {
    let raw;
    try {
      raw = sessionStorage.getItem(SESSION_STORAGE_KEY);
    } catch (error) {
      return validSession(memorySession, now)
        ? memorySession
        : { id: crypto.randomUUID(), last_activity_at: now };
    }
    if (raw) {
      try {
        const stored = JSON.parse(raw);
        if (validSession(stored, now)) return stored;
      } catch (error) {
        /* malformed state rotates the tab-scoped session */
      }
    }
    return { id: crypto.randomUUID(), last_activity_at: now };
  }

  function touchSession(now) {
    const session = readSession(now);
    session.last_activity_at = now;
    memorySession = session;
    try {
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
    } catch (error) {
      /* the in-memory tab session still keeps events correlated */
    }
    return session.id;
  }

  function warnDeliveryOnce() {
    if (deliveryWarningShown) return;
    deliveryWarningShown = true;
    console.warn('Analytics event delivery failed.');
  }

  function queueEvent(eventName, pageView, properties) {
    if (!signedIn() || !pageView) return false;
    const now = Date.now();
    const payload = {
      event_id: crypto.randomUUID(),
      schema_version: 1,
      event_name: eventName,
      session_id: touchSession(now),
      occurred_at: new Date(now).toISOString(),
      page_view: pageView,
      properties: properties || {},
    };
    try {
      const csrf = typeof window.csrfHeaders === 'function'
        ? window.csrfHeaders()
        : {};
      void fetch(API_BASE + '/api/analytics/events', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...csrf },
        credentials: 'include',
        keepalive: true,
        body: JSON.stringify(payload),
      }).catch(() => warnDeliveryOnce());
    } catch (error) {
      warnDeliveryOnce();
    }
    return true;
  }

  function visibleMs(now) {
    if (!Number.isFinite(currentViewStartedAt)) return 0;
    return Math.max(
      0,
      Math.min(MAX_VISIBLE_MS, Math.round(now - currentViewStartedAt)),
    );
  }

  function hideCurrentView(now) {
    if (!currentPageView || viewSuspended) return;
    queueEvent('page_hidden', currentPageView, { visible_ms: visibleMs(now) });
    currentViewStartedAt = null;
    viewSuspended = true;
  }

  function showCurrentView(now) {
    if (!currentPageView || !signedIn()) return;
    queueEvent('page_viewed', currentPageView, {});
    currentViewStartedAt = now;
    viewSuspended = false;
  }

  function transitionTo(pageView) {
    if (pageView === currentPageView && !viewSuspended) return;
    const now = Date.now();
    hideCurrentView(now);
    currentPageView = pageView;
    currentViewStartedAt = null;
    viewSuspended = false;
    if (pageView) showCurrentView(now);
  }

  function resolvePageView(page, options) {
    if (page === 'playground') {
      const tab = options.playgroundTab || document.documentElement.dataset.navPlaygroundTab;
      return PAGE_VIEW_MAP[`playground_${tab}`] || null;
    }
    return PAGE_VIEW_MAP[page] || null;
  }

  function resetSignedOutState() {
    currentPageView = null;
    currentViewStartedAt = null;
    viewSuspended = false;
    transientPageView = null;
    transientReturnView = null;
  }

  function recordNavigation(page, options = {}) {
    const pageView = resolvePageView(page, options);
    basePageView = pageView;
    if (!signedIn()) {
      resetSignedOutState();
      return;
    }
    if (transientPageView) {
      transientReturnView = pageView;
      return;
    }
    transitionTo(pageView);
  }

  function enterTransientView(pageView) {
    const approved = PAGE_VIEW_MAP[pageView];
    if (!approved || !signedIn()) return;
    if (transientPageView === approved) return;
    transientReturnView = basePageView || currentPageView;
    transientPageView = approved;
    transitionTo(approved);
  }

  function leaveTransientView() {
    if (!transientPageView) return;
    const returnView = basePageView || transientReturnView;
    transientPageView = null;
    transientReturnView = null;
    if (!signedIn()) {
      resetSignedOutState();
      return;
    }
    transitionTo(returnView);
  }

  function handleVisibilityChange() {
    if (document.visibilityState === 'hidden') {
      hideCurrentView(Date.now());
      return;
    }
    if (document.visibilityState === 'visible' && viewSuspended) {
      showCurrentView(Date.now());
    }
  }

  function sendHeartbeat() {
    if (!signedIn()) return;
    if (document.visibilityState !== 'visible') return;
    if (!currentPageView || viewSuspended) return;
    const now = Date.now();
    if (queueEvent('session_heartbeat', currentPageView, { visible_ms: visibleMs(now) })) {
      currentViewStartedAt = now;
    }
  }

  function handlePageHide() {
    hideCurrentView(Date.now());
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (!signedIn()) return;
    const html = document.documentElement;
    recordNavigation(html.dataset.navPage || 'home', {
      playgroundTab: html.dataset.navPlaygroundTab,
      competitionTab: html.dataset.navCompetitionTab,
    });
  });
  document.addEventListener('visibilitychange', handleVisibilityChange);
  window.addEventListener('pagehide', handlePageHide);
  window.setInterval(sendHeartbeat, HEARTBEAT_INTERVAL_MS);

  window.ATLAnalytics = Object.freeze({
    recordNavigation,
    enterTransientView,
    leaveTransientView,
  });
})();
