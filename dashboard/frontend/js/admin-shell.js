/** /admin console shell: courtesy gate, hash router, URL state, guarded GETs, dialogs. */
(function () {
  'use strict';

  // Design §7.2: one page, seven routes plus the profile (#users/{id}). There is
  // deliberately no live-operations route -- the live row has no detail page (§8.2, D16).
  const ROUTES = ['overview', 'sources', 'retention', 'credits', 'lifecycle', 'health', 'users'];
  const DETAIL_ROUTES = ['sources', 'retention', 'credits', 'lifecycle', 'health'];
  // 1D is cut (§8.2): no cross-user source finer than a day exists. 1Y is 180
  // inclusive days because the value routes reject a window wider than
  // MAX_VALUE_RANGE_DAYS (180) measured as (end - start).days with end = to + 1.
  const RANGE_DAYS = Object.freeze({ '1W': 7, '1M': 30, '1Y': 180 });
  const USER_GROUPS = ['internal', 'invited', 'organic', 'competition', 'partner', 'unknown'];
  const LIFECYCLE_SEGMENTS = ['new', 'onboarding', 'growing', 'core', 'at_risk', 'dormant'];
  const COMMERCIAL_TIERS = ['unpaid', 'starter', 'invested', 'high_value'];
  const LIFECYCLE_LABELS = Object.freeze({
    new: 'New', onboarding: 'Onboarding', growing: 'Growing',
    core: 'Core', at_risk: 'At risk', dormant: 'Dormant',
  });
  const OPERATIONAL_LABELS = Object.freeze({
    blocked: 'Blocked', needs_attention: 'Needs attention', healthy: 'Healthy',
  });
  const COMMERCIAL_LABELS = Object.freeze({
    unpaid: 'Unpaid', starter: 'Starter', invested: 'Invested', high_value: 'High value',
  });
  // Copy is design §15.2-§15.4; harvested from the retired admin-analytics-value.js
  // (§7.4; deleted in PR C, see git history).
  const LIFECYCLE_RULES = Object.freeze({
    new: 'Account is 0–6 UTC days old and has no successful backtest.',
    onboarding: 'No successful backtest yet; the account is no longer New and is not inactive.',
    growing: 'Activated and active in the last 7 UTC days, below the Core repeat-value threshold.',
    core: 'At least 3 active days and 3 successful backtests in 30 UTC days, active in the last 7 days.',
    at_risk: 'Last meaningful activity was 8–29 UTC days ago.',
    dormant: 'Last meaningful activity was at least 30 UTC days ago.',
  });
  const OPERATIONAL_RULES = Object.freeze({
    blocked: 'A current issue prevents a core action, such as an unavailable billing lane.',
    needs_attention: 'A supported issue needs operator review but may not block every action.',
    healthy: 'No supported current blocker or attention condition matched.',
  });
  const COMMERCIAL_RULE = 'Commercial value uses settled purchases minus refunds. Admin Grants do not count as purchases.';
  const SECTION_UNAVAILABLE = 'This section is temporarily unavailable.';
  const STALE_NOTICE = 'Showing the last successful response; refresh failed.';
  const INCOMPLETE = 'Incomplete data';
  const PENDING = 'Awaiting data source';
  const DASH = '—';
  const LOCALE = 'en-US';

  const state = {
    route: 'overview',
    routeId: null,
    range: '1W',
    filters: { group: '', segment: '', tier: '', internal: false, q: '', priority: false },
    admin: false,
    user: null,
    seq: {},
  };
  const returnFocus = new Map();

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clear(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function isoDay(date) {
    return date.toISOString().slice(0, 10);
  }

  function today() {
    return new Date();
  }

  function parseHash(hash) {
    const raw = String(hash || '').replace(/^#/, '');
    const [head, tail] = raw.split('/');
    if (head === 'users') {
      return { route: 'users', id: /^\d+$/.test(tail || '') ? tail : null };
    }
    return { route: ROUTES.includes(head) ? head : 'overview', id: null };
  }

  function rangeDates(range, now) {
    const days = RANGE_DAYS[range] || RANGE_DAYS['1W'];
    const base = now || api.today();
    const end = new Date(Date.UTC(base.getUTCFullYear(), base.getUTCMonth(), base.getUTCDate()));
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - (days - 1));
    return { from: isoDay(start), to: isoDay(end) };
  }

  function readUrlState(search) {
    const params = new URLSearchParams(search || '');
    const range = params.get('range');
    const group = params.get('group') || '';
    const segment = params.get('segment') || '';
    const tier = params.get('tier') || '';
    return {
      range: Object.hasOwn(RANGE_DAYS, range) ? range : '1W',
      filters: {
        group: USER_GROUPS.includes(group) ? group : '',
        segment: LIFECYCLE_SEGMENTS.includes(segment) ? segment : '',
        tier: COMMERCIAL_TIERS.includes(tier) ? tier : '',
        internal: params.get('internal') === 'true',
        q: String(params.get('q') || '').slice(0, 100),
        priority: params.get('priority') === 'true',
      },
    };
  }

  function buildSearch(range, filters) {
    const params = new URLSearchParams();
    if (range && range !== '1W') params.set('range', range);
    if (filters.group) params.set('group', filters.group);
    if (filters.segment) params.set('segment', filters.segment);
    if (filters.tier) params.set('tier', filters.tier);
    if (filters.internal) params.set('internal', 'true');
    if (filters.q) params.set('q', filters.q);
    if (filters.priority) params.set('priority', 'true');
    const text = params.toString();
    return text ? `?${text}` : '';
  }

  function analyticsParams({ withGroup = false } = {}) {
    const { from, to } = rangeDates(state.range);
    const params = new URLSearchParams();
    params.set('from', from);
    params.set('to', to);
    params.set('include_internal', state.filters.internal ? 'true' : 'false');
    if (withGroup && state.filters.group) params.set('user_group', state.filters.group);
    return params;
  }

  function userListParams({ offset = 0, limit = 50 } = {}) {
    const params = new URLSearchParams();
    const filters = state.filters;
    if (filters.q) params.set('q', filters.q);
    if (filters.group) params.set('user_group', filters.group);
    if (filters.segment) params.set('lifecycle_segment', filters.segment);
    if (filters.tier) params.set('commercial_tier', filters.tier);
    if (filters.priority) params.set('priority', 'true');
    params.set('include_internal', filters.internal ? 'true' : 'false');
    params.set('limit', String(limit));
    params.set('offset', String(offset));
    return params;
  }

  function formatNumber(value) {
    if (value == null || value === '') return DASH;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? new Intl.NumberFormat(LOCALE).format(numeric) : DASH;
  }

  function formatPercent(value) {
    if (value == null || value === '') return DASH;
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0 || numeric > 1) return DASH;
    return new Intl.NumberFormat(LOCALE, { style: 'percent', maximumFractionDigits: 1 }).format(numeric);
  }

  // Guarded the way the retired admin-analytics-value.js:292 guarded this same
  // call, and the way app.js's formatBacktestTimeoutMessage now does (d7a8a541):
  // a blocked or 404ing credit-format.js must not throw out of a renderer. DASH
  // rather than a local six-decimal fallback -- this function already owns a "no
  // number" answer, and a second copy of the formatter is how the two drift.
  function formatCredits(value) {
    if (!window.CreditFormat?.formatCreditsMicro) return DASH;
    const formatted = window.CreditFormat.formatCreditsMicro(value);
    return formatted === DASH ? DASH : `${formatted} Credits`;
  }

  function usdFromMicro(value) {
    if (value == null || value === '') return DASH;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return DASH;
    return new Intl.NumberFormat(LOCALE, { style: 'currency', currency: 'USD' }).format(numeric / 1000000);
  }

  function formatDateOnly(value, fallback = DASH) {
    if (!value) return fallback;
    const date = new Date(`${value}T00:00:00Z`);
    return Number.isFinite(date.getTime())
      ? new Intl.DateTimeFormat(LOCALE, { dateStyle: 'medium', timeZone: 'UTC' }).format(date)
      : fallback;
  }

  // `selected_period_end` is a *half-open* boundary: admin_analytics.py's
  // _value_range_from_values sets `end = to + 1 day` and value_queries.py:1388
  // publishes it verbatim, so printing it as written renders an 8-day "week" and
  // a 181-day "year". Converted for display only: the exclusive end is what
  // MAX_VALUE_RANGE_DAYS is measured against, so the contract is right and the
  // rendering was wrong.
  function formatLastIncludedDay(value, fallback = DASH) {
    if (!value) return fallback;
    const date = new Date(`${value}T00:00:00Z`);
    if (!Number.isFinite(date.getTime())) return fallback;
    date.setUTCDate(date.getUTCDate() - 1);
    return new Intl.DateTimeFormat(LOCALE, { dateStyle: 'medium', timeZone: 'UTC' }).format(date);
  }

  function formatShortDay(value, fallback = DASH) {
    if (!value) return fallback;
    const date = new Date(`${value}T00:00:00Z`);
    return Number.isFinite(date.getTime())
      ? new Intl.DateTimeFormat(LOCALE, { month: 'short', day: 'numeric', timeZone: 'UTC' }).format(date)
      : fallback;
  }

  function formatTimestamp(value, fallback = DASH) {
    if (!value) return fallback;
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return fallback;
    const day = new Intl.DateTimeFormat(LOCALE, { dateStyle: 'medium', timeZone: 'UTC' }).format(date);
    const time = new Intl.DateTimeFormat(LOCALE, { hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZone: 'UTC' }).format(date);
    return `${day}, ${time} UTC`;
  }

  function humanize(value) {
    const key = String(value || 'unknown');
    return key.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function incompleteItem(item) {
    if (!item || typeof item !== 'object') return false;
    if (item.available === false) return true;
    return Boolean(item.status && item.status !== 'ready');
  }

  function availabilityIncomplete(availability) {
    if (!availability || typeof availability !== 'object') return false;
    if (incompleteItem(availability)) return true;
    return Object.values(availability).some(incompleteItem);
  }

  function fieldPending(payload, key) {
    // Interim contract (PR C ships before PR D): a §9 field the route does not
    // serve yet is *absent* from the payload, which is not served-and-empty.
    // Absent renders PENDING so the slot is visibly waiting on a data source;
    // after PR D an absent field is a contract bug and reads the same way
    // (fail-visible), never as an empty chart. No payload at all is the
    // caller's loading/error state, not a pending field.
    return Boolean(payload) && typeof payload === 'object' && !(key in payload);
  }

  function freshnessLegendText(now) {
    const base = now || api.today();
    const yesterday = new Date(Date.UTC(base.getUTCFullYear(), base.getUTCMonth(), base.getUTCDate() - 1));
    return `Daily figures complete through ${isoDay(yesterday)} UTC; live tiles are this instance's process state`;
  }

  function rulesEntries() {
    return [
      ...LIFECYCLE_SEGMENTS.map((segment) => [LIFECYCLE_LABELS[segment], LIFECYCLE_RULES[segment]]),
      ...Object.keys(OPERATIONAL_LABELS).map((key) => [OPERATIONAL_LABELS[key], OPERATIONAL_RULES[key]]),
      ['Commercial value', COMMERCIAL_RULE],
    ];
  }

  async function request(path) {
    const response = await fetch(path, {
      method: 'GET',
      credentials: 'include',
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) {
      const error = new Error(`Request failed with status ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  // The /app copy of this is app.js's readCsrfToken(); /admin is a separate
  // document with no access to it, so the two are deliberate twins rather than
  // a shared helper -- linking them would mean importing app.js, which is the
  // 15,000-line inheritance this page exists to avoid. Both cookie names because
  // cookie_secure() picks __Host-atl_csrf in prod and atl_csrf in dev
  // (backend/csrf.py:51-52); reading one name works in exactly one environment.
  // `document.cookie || ''` is load-bearing, not defensive noise: the node test
  // stub has no cookie property at all.
  function readCsrfToken() {
    try {
      const raw = document.cookie || '';
      for (const name of ['atl_csrf', '__Host-atl_csrf']) {
        const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const match = raw.match(new RegExp(`(?:^|; )${escaped}=([^;]*)`));
        if (match) return decodeURIComponent(match[1]);
      }
    } catch (_error) { /* a document with no cookie access is a no-token document */ }
    return null;
  }

  // The page's ONE write path (design N4). Deliberately a second, differently
  // named function rather than an options bag on request(): "does this module
  // write?" has to stay answerable by grep, because that is exactly what
  // test_admin_page_modules.py asserts over the module set. An options bag would
  // make the two indistinguishable in source and leave the guard asserting
  // nothing.
  //
  // CsrfMiddleware requires the double-submit header on every unsafe method that
  // carries a session cookie, and /api/auth/logout is not exempt, so a write
  // without this header is a 403 in production that no fetch-stubbed test can
  // see. The header is omitted rather than sent empty when there is no cookie:
  // an empty token claims one we do not have.
  async function write(path, { method, body } = {}) {
    const token = readCsrfToken();
    const headers = { Accept: 'application/json', 'Content-Type': 'application/json' };
    if (token) headers['X-CSRF-Token'] = token;
    const init = { method, credentials: 'include', headers };
    if (body !== undefined) init.body = body;
    const response = await fetch(path, init);
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      const detail = payload?.detail || payload?.error;
      const error = new Error(
        typeof detail === 'string' ? detail : `Request failed with status ${response.status}`,
      );
      error.status = response.status;
      throw error;
    }
    if (response.status === 204) return null;
    return response.json().catch(() => null);
  }

  function nextSeq(surface) {
    state.seq[surface] = (state.seq[surface] || 0) + 1;
    return state.seq[surface];
  }

  function isCurrent(surface, seq) {
    return state.seq[surface] === seq;
  }

  function invalidateAll() {
    Object.keys(state.seq).forEach((surface) => { state.seq[surface] += 1; });
  }

  // Every call site does `if (await s.handleAccessLost(error)) return;` and then
  // writes to the DOM with no re-check of isCurrent(). That is only safe because
  // this function is `async` but contains no internal `await`: awaiting it costs
  // exactly one microtask tick, and the microtask queue always drains before the
  // next macrotask (hashchange/popstate), so no route change can land in between.
  // Adding any real `await` in here (a token refresh, another fetch) reopens a
  // stale-write window at every call site at once -- fix it here, not there.
  async function handleAccessLost(error) {
    if (error?.status !== 401 && error?.status !== 403) return false;
    invalidateAll();
    state.admin = false;
    state.user = null;
    window.location.replace('/app');
    return true;
  }

  // Courtesy redirect, not the gate (design D7): Vercel serves this HTML as a
  // static file with no session access, so the real gate is require_admin on
  // every /api/admin/* route. A non-admin who defeats this sees empty panels
  // and 403s. Until the probe resolves the shell shows placeholders, never
  // sample numbers.
  async function gate() {
    try {
      const response = await fetch('/api/auth/me', { method: 'GET', credentials: 'include', headers: { Accept: 'application/json' } });
      const body = response.ok ? await response.json() : null;
      const account = body && body.user;
      if (!account || account.role !== 'admin') {
        state.user = null;
        window.location.replace('/app');
        return false;
      }
      state.admin = true;
      // N8: kept, not discarded. The account menu needs the display name and
      // the email, and this request has already been paid for -- the next reader
      // should not add a second /api/auth/me for the same two strings. Only
      // these two fields are retained: nothing on this page renders a role or
      // an id, and a wider copy is a wider thing to leak into a DOM sink.
      state.user = { display_name: account.display_name || '', email: account.email || '' };
      return true;
    } catch (_error) {
      state.user = null;
      window.location.replace('/app');
      return false;
    }
  }

  function user() {
    return state.user;
  }

  function setPanelState(panel, { busy = false, status = '', error = '', stale = false, empty = false } = {}) {
    if (!panel) return;
    panel.setAttribute('aria-busy', busy ? 'true' : 'false');
    panel.classList.toggle('is-stale', Boolean(stale));
    const statusNode = panel.querySelector('[data-status]');
    if (statusNode) {
      // Both notices, not the stronger one: `stale ? STALE_NOTICE : status` made
      // "stale" and "stale *and* known-incomplete" render identically, which is
      // the absent-vs-broken collapse the fail-closed-is-not-fail-visible rule is
      // about. paint() passes the two together whenever a panel renders partial
      // rollups whose sibling endpoint also failed.
      const text = [stale ? STALE_NOTICE : '', status].filter(Boolean).join(' · ');
      statusNode.textContent = text;
      statusNode.hidden = !text;
    }
    const errorNode = panel.querySelector('[data-error]');
    if (errorNode) {
      const textNode = panel.querySelector('[data-error] span');
      if (textNode) textNode.textContent = error || '';
      errorNode.hidden = !error;
    }
    const body = panel.querySelector('[data-body]');
    if (body && empty) {
      clear(body);
      body.appendChild(el('p', 'panel-empty', typeof empty === 'string' ? empty : 'Nothing to show for this range.'));
    }
  }

  function openDialog(dialog, opener) {
    if (!dialog || typeof dialog.showModal !== 'function') return;
    returnFocus.set(dialog.id, opener || document.activeElement);
    dialog.showModal();
    dialog.querySelector('[data-dialog-initial-focus]')?.focus();
  }

  function closeDialog(dialog) {
    if (!dialog?.open) return;
    dialog.close();
    const opener = returnFocus.get(dialog.id);
    returnFocus.delete(dialog.id);
    if (opener?.isConnected) opener.focus();
  }

  function fillRulesDialog() {
    const list = document.getElementById('rulesList');
    if (!list) return;
    clear(list);
    rulesEntries().forEach(([label, rule]) => {
      const wrapper = el('div');
      wrapper.appendChild(el('dt', '', label));
      wrapper.appendChild(el('dd', '', rule));
      list.appendChild(wrapper);
    });
  }

  function openRules(opener) {
    fillRulesDialog();
    openDialog(document.getElementById('rulesDialog'), opener);
  }

  function writeUrl() {
    if (!window.history?.replaceState) return;
    const search = buildSearch(state.range, state.filters);
    window.history.replaceState(window.history.state, '', `${window.location.pathname}${search}${window.location.hash}`);
  }

  function announce() {
    document.dispatchEvent(new CustomEvent('admin:route', {
      detail: { route: state.route, id: state.routeId, range: state.range, filters: { ...state.filters } },
    }));
  }

  function showView(id, visible) {
    const node = document.getElementById(id);
    if (node) node.hidden = !visible;
  }

  // A same-document hash traversal (Back/Forward between two #routes) fires BOTH
  // popstate and hashchange, so binding route() to each ran the whole route twice:
  // every panel fetch doubled, and one Back onto #users/{id} wrote two admin
  // profile-access audit rows (_record_access, admin_analytics.py:559) for a single
  // view. nextSeq/isCurrent suppresses the duplicate *render*, never the duplicate
  // *request*, so the dedupe belongs in front of route(). Keyed on the whole
  // location, not the hash: a traversal restores the entry's query string too, and
  // that is what readUrlIntoState reads. Order-independent by construction --
  // whichever event arrives first does the work and the other one no-ops.
  let renderedLocation = null;

  function locationKey() {
    return `${window.location.pathname}${window.location.search}${window.location.hash}`;
  }

  function handleLocationChange() {
    if (locationKey() === renderedLocation) return;
    readUrlIntoState();
    syncControls();
    route();
  }

  function route() {
    const parsed = parseHash(window.location.hash);
    if (window.location.hash && parsed.route === 'overview' && window.location.hash !== '#overview') {
      window.location.hash = 'overview';
      return;
    }
    // Recorded after the normalisation redirect above, which deliberately leaves
    // the key stale so the hashchange it triggers is not swallowed.
    renderedLocation = locationKey();
    state.route = parsed.route;
    state.routeId = parsed.id;
    showView('overview', parsed.route === 'overview');
    showView('detail', DETAIL_ROUTES.includes(parsed.route));
    showView('usersView', parsed.route === 'users' && !parsed.id);
    showView('profile', parsed.route === 'users' && Boolean(parsed.id));
    document.querySelectorAll('#analyticsSubnav a[data-route]').forEach((link) => {
      link.classList.toggle('active', link.dataset.route === parsed.route);
    });
    window.scrollTo(0, 0);
    announce();
  }

  function syncControls() {
    document.querySelectorAll('.range button[data-range]').forEach((button) => {
      button.setAttribute('aria-pressed', button.dataset.range === state.range ? 'true' : 'false');
    });
    const group = document.getElementById('filterGroup');
    const segment = document.getElementById('filterSegment');
    const tier = document.getElementById('filterTier');
    const internal = document.getElementById('filterInternal');
    if (group) group.value = state.filters.group;
    if (segment) segment.value = state.filters.segment;
    if (tier) tier.value = state.filters.tier;
    if (internal) internal.checked = state.filters.internal;
    document.querySelectorAll('.selected-range').forEach((node) => {
      node.textContent = `Selected range · ${state.range}`;
    });
    const legend = document.getElementById('freshnessLegend');
    if (legend) legend.textContent = freshnessLegendText();
  }

  function readUrlIntoState() {
    const parsed = readUrlState(window.location.search);
    state.range = parsed.range;
    state.filters = parsed.filters;
  }

  function setRange(range) {
    if (!Object.hasOwn(RANGE_DAYS, range)) return;
    state.range = range;
    syncControls();
    writeUrl();
    announce();
  }

  function setFilters(patch) {
    state.filters = { ...state.filters, ...patch };
    syncControls();
    writeUrl();
    announce();
  }

  function bindControls() {
    document.querySelectorAll('.range button[data-range]').forEach((button) => {
      button.addEventListener('click', () => setRange(button.dataset.range));
    });
    document.getElementById('filterGroup')?.addEventListener('change', (event) => setFilters({ group: event.target.value }));
    document.getElementById('filterSegment')?.addEventListener('change', (event) => setFilters({ segment: event.target.value }));
    document.getElementById('filterTier')?.addEventListener('change', (event) => setFilters({ tier: event.target.value }));
    document.getElementById('filterInternal')?.addEventListener('change', (event) => setFilters({ internal: Boolean(event.target.checked) }));
    document.getElementById('filters')?.addEventListener('submit', (event) => event.preventDefault());
    document.getElementById('analyticsParent')?.addEventListener('click', (event) => {
      event.preventDefault();
      const subnav = document.getElementById('analyticsSubnav');
      if (!subnav) return;
      subnav.hidden = !subnav.hidden;
      event.currentTarget.setAttribute('aria-expanded', String(!subnav.hidden));
    });
    document.querySelectorAll('[data-retry]').forEach((button) => {
      button.addEventListener('click', () => {
        const panel = button.closest('[data-panel]');
        document.dispatchEvent(new CustomEvent('admin:retry', { detail: { panel: panel?.dataset.panel || '' } }));
      });
    });
    document.querySelectorAll('dialog').forEach((dialog) => {
      dialog.querySelectorAll('[data-dialog-close]').forEach((button) => {
        button.addEventListener('click', () => closeDialog(dialog));
      });
      dialog.addEventListener('cancel', (event) => { event.preventDefault(); closeDialog(dialog); });
      dialog.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;
        event.preventDefault();
        closeDialog(dialog);
      });
      dialog.addEventListener('click', (event) => { if (event.target === dialog) closeDialog(dialog); });
    });
  }

  async function boot() {
    bindControls();
    const admin = await gate();
    if (!admin) return;
    readUrlIntoState();
    syncControls();
    window.addEventListener('hashchange', handleLocationChange);
    window.addEventListener('popstate', handleLocationChange);
    route();
  }

  const api = {
    ROUTES, RANGE_DAYS, LIFECYCLE_LABELS, OPERATIONAL_LABELS, COMMERCIAL_LABELS,
    LIFECYCLE_RULES, OPERATIONAL_RULES, SECTION_UNAVAILABLE, STALE_NOTICE, INCOMPLETE, PENDING, DASH,
    state, today,
    parseHash, rangeDates, readUrlState, buildSearch, analyticsParams, userListParams,
    formatNumber, formatPercent, formatCredits, usdFromMicro, formatDateOnly, formatLastIncludedDay, formatShortDay, formatTimestamp, humanize,
    availabilityIncomplete, fieldPending, freshnessLegendText, rulesEntries, el, clear,
    request, write, user, nextSeq, isCurrent, invalidateAll, handleAccessLost, gate,
    setPanelState, openDialog, closeDialog, openRules, setFilters,
  };
  window.AdminShell = api;
  document.addEventListener('DOMContentLoaded', boot);
})();
