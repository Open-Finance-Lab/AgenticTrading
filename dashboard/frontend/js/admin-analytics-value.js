/** Read-only Admin user-value analytics overview. */
(function () {
  'use strict';

  const API_BASE = (
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ) ? window.location.origin : '';
  const SECTION_UNAVAILABLE = 'This section is temporarily unavailable.';
  const API_ENDPOINTS = Object.freeze({
    lifecycle: '/api/admin/analytics/lifecycle',
    retention: '/api/admin/analytics/retention',
    commercial: '/api/admin/analytics/commercial',
    operational: '/api/admin/analytics/operational',
    acquisition: '/api/admin/analytics/acquisition',
    users: '/api/admin/analytics/users',
  });
  const LIFECYCLE_SEGMENTS = ['new', 'onboarding', 'growing', 'core', 'at_risk', 'dormant'];
  const LIFECYCLE_LABELS = Object.freeze({
    new: 'New',
    onboarding: 'Onboarding',
    growing: 'Growing',
    core: 'Core',
    at_risk: 'At risk',
    dormant: 'Dormant',
  });
  const OPERATIONAL_LABELS = Object.freeze({
    blocked: 'Blocked',
    needs_attention: 'Needs attention',
    healthy: 'Healthy',
  });
  const COMMERCIAL_LABELS = Object.freeze({
    unpaid: 'Unpaid',
    starter: 'Starter',
    invested: 'Invested',
    high_value: 'High value',
  });
  const ACQUISITION_SOURCE_LABELS = Object.freeze({
    student: 'Student',
    community: 'Community',
    friend: 'Friend',
    competition: 'Competition',
    unknown: 'Unknown',
  });
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
  const MOVEMENT_RANGES = Object.freeze({
    '5d': { label: '5D', title: 'Recent 5-day movement', granularity: 'daily' },
    '1w': { label: '1W', title: '7-day movement', granularity: 'daily' },
    '1m': { label: '1M', title: 'Monthly movement', granularity: 'weekly' },
    '1y': { label: '1Y', title: 'Yearly movement', granularity: 'monthly' },
  });
  const ANALYTICS_RANGES = Object.freeze({
    '1d': { label: '1D', days: 1 },
    '1w': { label: '1W', days: 7 },
    '1m': { label: '1M', days: 30 },
    '1y': { label: '1Y', days: 365 },
  });
  const CHART_COLORS = Object.freeze({
    new: '#94a3b8',
    onboarding: '#38bdf8',
    growing: '#2dd4bf',
    core: '#a3e635',
    at_risk: '#fbbf24',
    dormant: '#fb7185',
  });
  const URL_KEYS = Object.freeze({
    lifecycle: 'analyticsLifecycle',
    operational: 'analyticsOperational',
    commercial: 'analyticsCommercial',
    user: 'analyticsUser',
    profile: 'analyticsProfile',
    movementRange: 'analyticsMovementRange',
    analyticsRange: 'analyticsRange',
    acquisitionSource: 'analyticsAcquisitionSource',
    acquisitionCohort: 'analyticsAcquisitionCohort',
    acquisitionGroupBy: 'analyticsAcquisitionGroupBy',
    acquisitionOpen: 'analyticsAcquisitionOpen',
    acquisitionMetric: 'analyticsAcquisitionMetric',
  });
  const returnFocus = new Map();

  const state = {
    initialized: false,
    active: false,
    requestSeq: 0,
    range: null,
    analyticsRange: '1m',
    movementRange: '5d',
    includeInternal: false,
    acquisition: {
      source: '',
      cohort: '',
      lifecycle: '',
      blocked: '',
      paid: '',
      groupBy: 'source',
      metric: '',
    },
    userFilters: {
      lifecycle: '',
      operational: '',
      commercial: '',
      query: '',
      profile: '',
    },
    openDisclosures: new Set(),
    sections: {
      lifecycle: { loaded: false, data: null, error: null, stale: false },
      users: { loaded: false, data: null, error: null, stale: false },
      retention: { loaded: false, data: null, error: null, stale: false },
      commercial: { loaded: false, data: null, error: null, stale: false },
      operational: { loaded: false, data: null, error: null, stale: false },
      acquisition: { loaded: false, data: null, error: null, stale: false },
    },
    movementChart: null,
    evidenceUser: null,
  };

  function element(id) {
    return document.getElementById(id);
  }

  function node(tag, className, text) {
    const target = document.createElement(tag);
    if (className) target.className = className;
    if (text !== undefined) target.textContent = String(text);
    return target;
  }

  function clear(target) {
    if (!target) return;
    while (target.firstChild) target.removeChild(target.firstChild);
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

  function appendEvidence(target, evidence, fallback) {
    clear(target);
    (evidence || []).forEach((fact) => target.appendChild(node('li', '', fact)));
    if (!target.children.length) target.appendChild(node('li', 'admin-value-empty', fallback));
  }

  function request(path) {
    if (!window.API || typeof window.API.request !== 'function') {
      return Promise.reject(new Error('Admin Analytics API is not ready yet.'));
    }
    return window.API.request(`${API_BASE}${path}`, { method: 'GET' });
  }

  function utcRangeForPreset(preset, now = new Date()) {
    const config = ANALYTICS_RANGES[preset] || ANALYTICS_RANGES['1m'];
    const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - (config.days - 1));
    return {
      start: start.toISOString().slice(0, 10),
      end: end.toISOString().slice(0, 10),
    };
  }

  function readUrlState() {
    const params = new URLSearchParams(window.location.search);
    const analyticsRange = params.get(URL_KEYS.analyticsRange) || '1m';
    state.analyticsRange = Object.hasOwn(ANALYTICS_RANGES, analyticsRange)
      ? analyticsRange : '1m';
    state.range = utcRangeForPreset(state.analyticsRange);
    const movementRange = params.get(URL_KEYS.movementRange) || '5d';
    state.movementRange = Object.hasOwn(MOVEMENT_RANGES, movementRange) ? movementRange : '5d';
    state.includeInternal = params.get('analyticsInternal') === 'true';
    const lifecycle = params.get(URL_KEYS.lifecycle) || '';
    const operational = params.get(URL_KEYS.operational) || '';
    const commercial = params.get(URL_KEYS.commercial) || '';
    state.userFilters.lifecycle = LIFECYCLE_SEGMENTS.includes(lifecycle) ? lifecycle : '';
    state.userFilters.operational = Object.hasOwn(OPERATIONAL_LABELS, operational) ? operational : '';
    state.userFilters.commercial = Object.hasOwn(COMMERCIAL_LABELS, commercial) ? commercial : '';
    state.userFilters.query = params.get('analyticsUserQuery') || '';
    state.userFilters.profile = params.get(URL_KEYS.profile) || params.get(URL_KEYS.user) || '';
    const acquisitionSource = params.get(URL_KEYS.acquisitionSource) || '';
    const acquisitionLifecycle = params.get('analyticsAcquisitionLifecycle') || '';
    const acquisitionBlocked = params.get('analyticsAcquisitionBlocked') || '';
    const acquisitionPaid = params.get('analyticsAcquisitionPaid') || '';
    const acquisitionGroupBy = params.get(URL_KEYS.acquisitionGroupBy) || 'source';
    state.acquisition.source = ['student', 'community', 'friend', 'competition', 'unknown'].includes(acquisitionSource)
      ? acquisitionSource : '';
    state.acquisition.cohort = params.get(URL_KEYS.acquisitionCohort) || '';
    state.acquisition.lifecycle = ['new', 'active', 'at_risk', 'dormant'].includes(acquisitionLifecycle)
      ? acquisitionLifecycle : '';
    state.acquisition.blocked = ['true', 'false'].includes(acquisitionBlocked) ? acquisitionBlocked : '';
    state.acquisition.paid = ['true', 'false'].includes(acquisitionPaid) ? acquisitionPaid : '';
    state.acquisition.groupBy = ['source', 'cohort'].includes(acquisitionGroupBy) ? acquisitionGroupBy : 'source';
    state.acquisition.metric = [
      'active', 'task_completed', 'repeat', 'paid_intent', 'paid',
    ].includes(params.get(URL_KEYS.acquisitionMetric)) ? params.get(URL_KEYS.acquisitionMetric) : '';
    state.openDisclosures = new Set(
      String(params.get('analyticsPanel') || '')
        .split(',')
        .filter((name) => ['retention', 'commercial', 'operational', 'acquisition'].includes(name))
    );
    if (params.get(URL_KEYS.acquisitionOpen) === 'true') state.openDisclosures.add('acquisition');
  }

  function setOrDelete(params, key, value) {
    if (value) params.set(key, value);
    else params.delete(key);
  }

  function writeUrlState() {
    if (!window.history?.replaceState) return;
    const url = new URL(window.location.href);
    url.searchParams.set(URL_KEYS.analyticsRange, state.analyticsRange);
    url.searchParams.delete('analyticsStart');
    url.searchParams.delete('analyticsEnd');
    setOrDelete(url.searchParams, 'analyticsInternal', state.includeInternal ? 'true' : '');
    setOrDelete(url.searchParams, URL_KEYS.lifecycle, state.userFilters.lifecycle);
    setOrDelete(url.searchParams, URL_KEYS.operational, state.userFilters.operational);
    setOrDelete(url.searchParams, URL_KEYS.commercial, state.userFilters.commercial);
    setOrDelete(url.searchParams, 'analyticsUserQuery', state.userFilters.query);
    setOrDelete(url.searchParams, URL_KEYS.profile, state.userFilters.profile);
    url.searchParams.set(URL_KEYS.movementRange, state.movementRange);
    setOrDelete(url.searchParams, URL_KEYS.acquisitionSource, state.acquisition.source);
    setOrDelete(url.searchParams, URL_KEYS.acquisitionCohort, state.acquisition.cohort);
    setOrDelete(url.searchParams, 'analyticsAcquisitionLifecycle', state.acquisition.lifecycle);
    setOrDelete(url.searchParams, 'analyticsAcquisitionBlocked', state.acquisition.blocked);
    setOrDelete(url.searchParams, 'analyticsAcquisitionPaid', state.acquisition.paid);
    url.searchParams.set(URL_KEYS.acquisitionGroupBy, state.acquisition.groupBy);
    setOrDelete(url.searchParams, URL_KEYS.acquisitionMetric, state.acquisition.metric);
    setOrDelete(
      url.searchParams,
      URL_KEYS.acquisitionOpen,
      state.openDisclosures.has('acquisition') ? 'true' : ''
    );
    setOrDelete(url.searchParams, 'analyticsPanel', [...state.openDisclosures].sort().join(','));
    window.history.replaceState(window.history.state, '', url);
  }

  function setControls() {
    element('adminValueInternal').checked = state.includeInternal;
    element('adminPriorityQuery').value = state.userFilters.query;
    element('adminPriorityLifecycle').value = state.userFilters.lifecycle;
    element('adminPriorityOperational').value = state.userFilters.operational;
    element('adminPriorityCommercial').value = state.userFilters.commercial;
    element('adminAcquisitionSource').value = state.acquisition.source;
    element('adminAcquisitionCohort').value = state.acquisition.cohort;
    element('adminAcquisitionLifecycle').value = state.acquisition.lifecycle;
    element('adminAcquisitionBlocked').value = state.acquisition.blocked;
    element('adminAcquisitionPaid').value = state.acquisition.paid;
    element('adminAcquisitionGroupBy').value = state.acquisition.groupBy;
    document.querySelectorAll('[data-movement-range]').forEach((button) => {
      const selected = button.dataset.movementRange === state.movementRange;
      button.setAttribute('aria-pressed', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll('[data-analytics-range]').forEach((button) => {
      const selected = button.dataset.analyticsRange === state.analyticsRange;
      button.setAttribute('aria-pressed', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
  }

  function rangeParams() {
    return new URLSearchParams({
      date_range: state.analyticsRange,
      include_internal: state.includeInternal ? 'true' : 'false',
    });
  }

  function appendAcquisitionFilters(params, { includeMetric = false } = {}) {
    if (state.acquisition.source) params.set('acquisition_source', state.acquisition.source);
    if (state.acquisition.cohort) params.set('acquisition_cohort', state.acquisition.cohort);
    if (state.acquisition.lifecycle) params.set('lifecycle', state.acquisition.lifecycle);
    if (state.acquisition.blocked) params.set('blocked', state.acquisition.blocked);
    if (state.acquisition.paid) params.set('paid', state.acquisition.paid);
    if (includeMetric && state.acquisition.metric) {
      const queryKey = {
        active: 'acquisition_active',
        task_completed: 'acquisition_task_completed',
        repeat: 'acquisition_repeat',
        paid_intent: 'acquisition_paid_intent',
        paid: 'paid',
      }[state.acquisition.metric];
      if (queryKey) params.set(queryKey, 'true');
    }
    return params;
  }

  function acquisitionParams() {
    const params = appendAcquisitionFilters(rangeParams());
    params.set('group_by', state.acquisition.groupBy);
    return params;
  }

  function acquisitionIsFilteringUsers() {
    return Boolean(
      state.acquisition.source
      || state.acquisition.cohort
      || state.acquisition.lifecycle
      || state.acquisition.blocked
      || state.acquisition.paid
      || state.acquisition.metric
    );
  }

  function userParams() {
    const params = new URLSearchParams({
      priority: acquisitionIsFilteringUsers() ? 'false' : 'true',
      include_internal: state.includeInternal ? 'true' : 'false',
      limit: '25',
      offset: '0',
      date_range: state.analyticsRange,
    });
    if (state.userFilters.query) params.set('q', state.userFilters.query);
    if (state.userFilters.lifecycle) params.set('lifecycle_segment', state.userFilters.lifecycle);
    if (state.userFilters.operational) params.set('operational_state', state.userFilters.operational);
    if (state.userFilters.commercial) params.set('commercial_tier', state.userFilters.commercial);
    return appendAcquisitionFilters(params, { includeMetric: true });
  }

  async function handleAccessLost(error) {
    if (error?.status !== 401 && error?.status !== 403) return false;
    if (typeof window.refreshAuthUser === 'function') await window.refreshAuthUser();
    if (typeof window.navigateToPage === 'function') window.navigateToPage('home');
    return true;
  }

  function number(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? new Intl.NumberFormat().format(parsed) : '—';
  }

  function percent(value) {
    if (value == null || value === '') return 'Not mature';
    const parsed = Number(value);
    return Number.isFinite(parsed)
      ? new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 0 }).format(parsed)
      : 'Not mature';
  }

  function credits(value) {
    if (window.CreditFormat?.formatCreditsMicro) {
      return window.CreditFormat.formatCreditsMicro(value);
    }
    const parsed = Number(value);
    return Number.isFinite(parsed)
      ? `${new Intl.NumberFormat(undefined, { minimumFractionDigits: 6, maximumFractionDigits: 6 }).format(parsed / 1000000)} Credits`
      : '—';
  }

  function dollarsFromMicro(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed)
      ? new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' }).format(parsed / 1000000)
      : '—';
  }

  function formatDate(value) {
    const parsed = new Date(`${value}T00:00:00Z`);
    return Number.isFinite(parsed.getTime())
      ? new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeZone: 'UTC' }).format(parsed)
      : '—';
  }

  function metricCard(label, value, detail) {
    const card = node('article', 'admin-value-metric-card');
    card.appendChild(node('span', '', label));
    card.appendChild(node('strong', '', value));
    if (detail) card.appendChild(node('small', '', detail));
    return card;
  }

  function availabilityIncomplete(availability) {
    if (!availability) return false;
    if (availability.status && availability.status !== 'ready') return true;
    return Object.values(availability).some((item) => item?.status && item.status !== 'ready');
  }

  function renderHeadline(payload) {
    const headline = payload?.headline || {};
    const mapping = {
      activated: headline.activated_users,
      core: headline.core_users,
      'at-risk': headline.at_risk_users,
      paid: headline.paid_users,
    };
    Object.entries(mapping).forEach(([name, value]) => {
      const target = document.querySelector(`[data-admin-value-metric="${name}"]`);
      if (target) target.textContent = number(value);
    });
    element('adminAnalyticsHeadline').setAttribute('aria-busy', 'false');
  }

  function renderDistribution(payload) {
    const target = element('adminLifecycleDistribution');
    clear(target);
    LIFECYCLE_SEGMENTS.forEach((segment) => {
      const button = node('button', `admin-lifecycle-segment is-${segment}`);
      button.type = 'button';
      button.dataset.lifecycle = segment;
      button.setAttribute('aria-pressed', state.userFilters.lifecycle === segment ? 'true' : 'false');
      button.appendChild(node('span', '', LIFECYCLE_LABELS[segment]));
      button.appendChild(node('strong', '', number(payload.segment_counts?.[segment] || 0)));
      button.addEventListener('click', () => {
        applyUserFilters({
          lifecycle: state.userFilters.lifecycle === segment ? '' : segment,
        });
      });
      target.appendChild(button);
    });
    const history = payload.availability?.history;
    const coverage = element('adminLifecycleCoverage');
    coverage.textContent = history?.coverage_start && history?.coverage_end
      ? `${formatDate(history.coverage_start)} – ${formatDate(history.coverage_end)}`
      : 'Current snapshot';
  }

  function movementPointDate(point) {
    return point?.period_start || point?.week_start;
  }

  function replaceHiddenMovementRows(series, granularity) {
    const body = element('adminLifecycleMovementTable')?.querySelector('tbody');
    clear(body);
    const table = element('adminLifecycleMovementTable');
    const periodLabel = granularity === 'day' ? 'Day' : granularity === 'month' ? 'Month' : 'Week';
    if (table) {
      const caption = table.querySelector('caption');
      if (caption) caption.textContent = `Lifecycle segment counts by ${periodLabel.toLowerCase()}`;
      const heading = table.querySelector('thead th');
      if (heading) heading.textContent = periodLabel;
    }
    series.forEach((point) => {
      const row = document.createElement('tr');
      row.appendChild(node('th', '', formatDate(movementPointDate(point))));
      row.firstChild.scope = 'row';
      LIFECYCLE_SEGMENTS.forEach((segment) => {
        row.appendChild(node('td', '', number(point.segment_counts?.[segment] || 0)));
      });
      body.appendChild(row);
    });
  }

  function renderLifecycleMovement(payload) {
    const hasMovementContract = Array.isArray(payload?.movement_segments);
    const range = Object.hasOwn(MOVEMENT_RANGES, payload?.movement_range)
      ? payload.movement_range
      : state.movementRange;
    const config = hasMovementContract
      ? MOVEMENT_RANGES[range]
      : { title: 'Weekly movement', granularity: 'weekly' };
    const granularity = payload?.movement_granularity || (
      config.granularity === 'daily' ? 'day' : config.granularity === 'weekly' ? 'week' : 'month'
    );
    const rows = hasMovementContract
      ? payload.movement_segments
      : (Array.isArray(payload?.weekly_segments) ? payload.weekly_segments : []);
    replaceHiddenMovementRows(rows, granularity);
    element('adminLifecycleMovementTitle').textContent = config.title;
    element('adminLifecycleMovementGranularity').textContent = `${config.granularity} snapshots`;
    const quality = element('adminLifecycleQuality');
    quality.hidden = !rows.some((point) => point.data_quality === 'partial');
    const canvas = element('adminLifecycleMovementChart');
    const periodLabel = granularity === 'day' ? 'daily' : granularity === 'month' ? 'monthly' : 'weekly';
    canvas.setAttribute(
      'aria-label',
      rows.length ? `Lifecycle movement across ${rows.length} ${periodLabel} snapshots` : 'No lifecycle movement data available'
    );
    if (state.movementChart) {
      state.movementChart.destroy();
      state.movementChart = null;
    }
    if (!rows.length || typeof window.Chart !== 'function') return;
    state.movementChart = new window.Chart(canvas, {
      type: 'line',
      data: {
        labels: rows.map((point) => formatDate(movementPointDate(point))),
        datasets: LIFECYCLE_SEGMENTS.map((segment) => ({
          label: LIFECYCLE_LABELS[segment],
          data: rows.map((point) => Number(point.segment_counts?.[segment] || 0)),
          borderColor: CHART_COLORS[segment],
          backgroundColor: CHART_COLORS[segment],
          borderWidth: 2,
          pointRadius: 2.5,
          tension: 0.28,
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 240 },
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: { position: 'bottom', labels: { color: '#94a3b8', usePointStyle: true } },
        },
        scales: {
          x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(148, 163, 184, 0.08)' } },
          y: { beginAtZero: true, ticks: { color: '#64748b', precision: 0 }, grid: { color: 'rgba(148, 163, 184, 0.08)' } },
        },
      },
    });
  }

  function renderLifecycle(payload) {
    renderHeadline(payload);
    renderDistribution(payload);
    renderLifecycleMovement(payload);
    const incomplete = availabilityIncomplete(payload.availability);
    element('adminValuePrimaryStatus').textContent = incomplete ? 'Incomplete data · available sections remain current.' : '';
  }

  function badge(kind, value, labels, rules) {
    const target = node('span', `admin-value-badge is-${kind}-${value}`, labels[value] || String(value || 'Unknown'));
    if (rules?.[value]) {
      target.title = rules[value];
      target.setAttribute('aria-label', `${target.textContent}: ${rules[value]}`);
    }
    return target;
  }

  function renderRulesDialog() {
    const list = element('adminAnalyticsRulesList');
    clear(list);
    LIFECYCLE_SEGMENTS.forEach((segment) => {
      const wrapper = document.createElement('div');
      wrapper.appendChild(node('dt', '', LIFECYCLE_LABELS[segment]));
      wrapper.appendChild(node('dd', '', LIFECYCLE_RULES[segment]));
      list.appendChild(wrapper);
    });
    Object.keys(OPERATIONAL_LABELS).forEach((status) => {
      const wrapper = document.createElement('div');
      wrapper.appendChild(node('dt', '', OPERATIONAL_LABELS[status]));
      wrapper.appendChild(node('dd', '', OPERATIONAL_RULES[status]));
      list.appendChild(wrapper);
    });
  }

  function renderEvidenceDialog(user) {
    state.evidenceUser = user;
    element('adminAnalyticsEvidenceTitle').textContent = user.display_name || user.email || `User #${user.user_id}`;
    element('adminAnalyticsEvidenceEmail').textContent = user.email || `User #${user.user_id}`;
    const signals = element('adminAnalyticsEvidenceSignals');
    clear(signals);
    signals.appendChild(badge('lifecycle', user.lifecycle?.segment, LIFECYCLE_LABELS, LIFECYCLE_RULES));
    signals.appendChild(badge('operational', user.operational?.state, OPERATIONAL_LABELS, OPERATIONAL_RULES));
    signals.appendChild(badge('commercial', user.commercial_tier, COMMERCIAL_LABELS));
    element('adminAnalyticsEvidenceLifecycleReason').textContent = user.lifecycle?.reason || 'No lifecycle reason is available.';
    element('adminAnalyticsEvidenceOperationalReason').textContent = user.operational?.reason || 'No operational reason is available.';
    appendEvidence(
      element('adminAnalyticsEvidenceLifecycle'),
      user.lifecycle?.evidence,
      'No lifecycle evidence is available.'
    );
    appendEvidence(
      element('adminAnalyticsEvidenceOperational'),
      user.operational?.evidence,
      'No operational evidence is available.'
    );
  }

  function openEvidenceDialog(user, opener) {
    renderEvidenceDialog(user);
    openDialog(element('adminAnalyticsEvidenceDialog'), opener);
  }

  function profileHref(userId) {
    const url = new URL(window.location.href);
    url.searchParams.set('adminTab', 'analytics');
    url.searchParams.set('analyticsUser', String(userId));
    url.searchParams.set('analyticsProfile', String(userId));
    url.searchParams.set('analyticsSection', 'overview');
    return `${url.pathname}${url.search}`;
  }

  function profileLink(user) {
    const label = user.display_name || user.email || `User #${user.user_id}`;
    const link = node('a', 'admin-priority-profile-link', label);
    link.href = profileHref(user.user_id);
    link.setAttribute('aria-label', `Open analytics profile for ${label}`);
    link.addEventListener('click', (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (typeof window.AdminAnalytics?.openProfile !== 'function') return;
      event.preventDefault();
      window.AdminAnalytics.openProfile(user.user_id);
    });
    return link;
  }

  function renderUsers(payload) {
    const target = element('adminPriorityUsers');
    clear(target);
    const items = Array.isArray(payload?.items) ? payload.items : [];
    items.forEach((user) => {
      const row = node('article', 'admin-priority-user');
      const identity = node('div', 'admin-priority-identity');
      const name = node('strong', '', '');
      name.appendChild(profileLink(user));
      identity.appendChild(name);
      identity.appendChild(node('span', '', user.email || `User #${user.user_id}`));
      row.appendChild(identity);
      const signals = node('div', 'admin-priority-signals');
      signals.appendChild(badge('lifecycle', user.lifecycle?.segment, LIFECYCLE_LABELS, LIFECYCLE_RULES));
      signals.appendChild(badge('operational', user.operational?.state, OPERATIONAL_LABELS, OPERATIONAL_RULES));
      signals.appendChild(badge('commercial', user.commercial_tier, COMMERCIAL_LABELS));
      if (acquisitionIsFilteringUsers() && user.acquisition) {
        signals.appendChild(badge(
          'acquisition',
          user.acquisition.source,
          ACQUISITION_SOURCE_LABELS
        ));
        if (user.acquisition.cohort) {
          signals.appendChild(node('span', 'admin-value-badge is-acquisition-cohort', user.acquisition.cohort));
        }
      }
      row.appendChild(signals);
      row.appendChild(node('p', 'admin-priority-reason', user.operational?.state === 'healthy' ? user.lifecycle?.reason : user.operational?.reason));
      row.appendChild(node('span', 'admin-priority-value', credits(user.lifetime_net_purchased_micro)));
      const open = node('button', 'credits-key-action', 'Review evidence');
      open.type = 'button';
      open.dataset.userId = String(user.user_id);
      open.setAttribute('aria-haspopup', 'dialog');
      open.addEventListener('click', () => openEvidenceDialog(user, open));
      row.appendChild(open);
      target.appendChild(row);
    });
    if (!items.length) target.appendChild(node('p', 'admin-value-empty', 'No users match these filters.'));
    target.setAttribute('aria-busy', 'false');
    element('adminPriorityUsersRange').textContent = payload?.total
      ? `Showing ${items.length} of ${number(payload.total)}`
      : '0 users';
    element('adminPriorityUsersTitle').textContent = acquisitionIsFilteringUsers()
      ? 'Acquisition users'
      : 'Priority users';
  }

  function summaryGrid(entries) {
    const grid = node('div', 'admin-value-summary-grid');
    entries.forEach(([label, value, detail]) => grid.appendChild(metricCard(label, value, detail)));
    return grid;
  }

  function renderRetention(payload, container) {
    clear(container);
    container.appendChild(summaryGrid([
      ['Week 1', percent(payload.summary_week_1?.rate), payload.summary_week_1?.mature ? `${number(payload.summary_week_1.eligible_users)} eligible` : 'Cohorts are still maturing'],
      ['Week 2', percent(payload.summary_week_2?.rate), payload.summary_week_2?.mature ? `${number(payload.summary_week_2.eligible_users)} eligible` : 'Cohorts are still maturing'],
      ['Week 4', percent(payload.summary_week_4?.rate), payload.summary_week_4?.mature ? `${number(payload.summary_week_4.eligible_users)} eligible` : 'Cohorts are still maturing'],
    ]));
    const wrapper = node('div', 'admin-value-table-wrap');
    const table = document.createElement('table');
    table.className = 'admin-value-table';
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    ['Activation week', 'Activated', 'Week 1', 'Week 2', 'Week 4'].forEach((label) => {
      const cell = node('th', '', label);
      cell.scope = 'col';
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    const body = document.createElement('tbody');
    (payload.cohorts || []).forEach((cohort) => {
      const row = document.createElement('tr');
      [
        formatDate(cohort.cohort_week),
        number(cohort.activated_users),
        percent(cohort.week_1?.rate),
        percent(cohort.week_2?.rate),
        percent(cohort.week_4?.rate),
      ].forEach((value) => row.appendChild(node('td', '', value)));
      body.appendChild(row);
    });
    if (!body.children.length) {
      const row = document.createElement('tr');
      const cell = node('td', 'admin-value-empty', 'No activation cohorts in this range.');
      cell.colSpan = 5;
      row.appendChild(cell);
      body.appendChild(row);
    }
    table.appendChild(body);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
  }

  function renderCommercial(payload, container) {
    clear(container);
    const period = payload.selected_period || {};
    const balances = payload.current_balances || {};
    container.appendChild(summaryGrid([
      ['Lifetime net purchased', credits(payload.lifetime_net_purchased_micro), 'Settled purchases minus refunds'],
      ['Purchased in range', credits(period.purchased_micro), 'Revenue signal'],
      ['Consumed in range', credits(period.consumed_micro), 'Model execution'],
      ['Platform model cost', dollarsFromMicro(period.platform_model_cost_micro_usd), 'Platform Credits lane'],
    ]));
    const tiers = node('div', 'admin-commercial-tier-grid');
    Object.keys(COMMERCIAL_LABELS).forEach((tier) => {
      tiers.appendChild(metricCard(COMMERCIAL_LABELS[tier], number(payload.tier_counts?.[tier] || 0), 'users'));
    });
    container.appendChild(tiers);
    container.appendChild(summaryGrid([
      ['Grant balance', credits(balances.grant_available_micro), 'Not revenue'],
      ['Purchased balance', credits(balances.purchased_available_micro), 'Customer-funded'],
      ['Total available', credits(balances.total_available_micro), 'Current spendable balance'],
      ['Admin Grant activity', credits(period.admin_grant_activity_micro), 'Excluded from revenue'],
    ]));
  }

  function renderOperational(payload, container) {
    clear(container);
    container.appendChild(summaryGrid([
      ['Blocked', number(payload.operational_state_counts?.blocked || 0), 'Core action unavailable'],
      ['Needs attention', number(payload.operational_state_counts?.needs_attention || 0), 'Operator review recommended'],
      ['Healthy', number(payload.operational_state_counts?.healthy || 0), 'No supported current issue'],
      ['Backtest success', percent(payload.backtest_success_rate), `${number(payload.completed_runs)} completed · ${number(payload.failed_runs)} failed`],
    ]));
    const wrapper = node('div', 'admin-value-table-wrap');
    const table = document.createElement('table');
    table.className = 'admin-value-table';
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    ['Failure category', 'Affected users'].forEach((label) => {
      const cell = node('th', '', label);
      cell.scope = 'col';
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    const body = document.createElement('tbody');
    (payload.top_failure_categories || []).forEach((failure) => {
      const row = document.createElement('tr');
      row.appendChild(node('td', '', String(failure.error_category || 'Unknown').replaceAll('_', ' ')));
      row.appendChild(node('td', '', number(failure.affected_users)));
      body.appendChild(row);
    });
    if (!body.children.length) {
      const row = document.createElement('tr');
      const cell = node('td', 'admin-value-empty', 'No failure categories in this range.');
      cell.colSpan = 2;
      row.appendChild(cell);
      body.appendChild(row);
    }
    table.appendChild(body);
    wrapper.appendChild(table);
    container.appendChild(wrapper);
  }

  function formatRatio(value) {
    if (value == null || value === '') return '—';
    const parsed = Number(value);
    return Number.isFinite(parsed)
      ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(parsed)
      : '—';
  }

  function acquisitionDrilldownLabel(group, metric) {
    const groupLabel = group?.label || 'Unknown';
    const metricLabel = {
      active: 'active users',
      task_completed: 'users who completed the task',
      repeat: 'repeat users',
      paid_intent: 'users with paid intent',
      paid: 'paid users',
    }[metric];
    return metricLabel ? `Show ${metricLabel} from ${groupLabel}` : `Show users from ${groupLabel}`;
  }

  function openAcquisitionUsers(group, metric = '') {
    if (!group || !['source', 'cohort'].includes(group.kind)) return;
    if (group.kind === 'source') state.acquisition.source = group.value;
    else state.acquisition.cohort = group.value;
    state.acquisition.metric = metric;
    if (metric === 'paid') state.acquisition.paid = 'true';
    setControls();
    writeUrlState();
    state.sections.users.loaded = false;
    element('adminPriorityUsers')?.setAttribute('aria-busy', 'true');
    fetchPriorityUsers()
      .then((payload) => applySettledSection('users', { status: 'fulfilled', value: payload }))
      .catch((error) => applySettledSection('users', { status: 'rejected', reason: error }));
    element('adminPriorityUsersTitle')?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    element('adminPriorityUsersTitle')?.focus({ preventScroll: true });
  }

  function acquisitionLink(group, metric, value) {
    const button = node('button', 'admin-acquisition-drilldown', value);
    button.type = 'button';
    button.setAttribute('aria-label', acquisitionDrilldownLabel(group, metric));
    button.addEventListener('click', () => openAcquisitionUsers(group, metric));
    return button;
  }

  function renderAcquisition(payload, container) {
    const table = container?.querySelector('table');
    const body = table?.querySelector('tbody');
    clear(body);
    const available = payload?.availability?.available !== false;
    const groups = available && Array.isArray(payload?.groups) ? payload.groups : [];
    groups.forEach((item) => {
      const row = document.createElement('tr');
      const group = item.group || { kind: payload.group_by || 'source', value: 'unknown', label: 'Unknown' };
      const groupCell = document.createElement('th');
      groupCell.scope = 'row';
      groupCell.appendChild(acquisitionLink(group, '', group.label || 'Unknown'));
      row.appendChild(groupCell);
      [
        ['', item.users],
        ['active', item.active],
        ['task_completed', item.task_completed],
        ['repeat', item.repeat],
      ].forEach(([metric, value]) => {
        const cell = document.createElement('td');
        cell.appendChild(acquisitionLink(group, metric, number(value)));
        row.appendChild(cell);
      });
      row.appendChild(node('td', 'admin-acquisition-number', formatRatio(item.runs_per_active_user)));
      row.appendChild(node('td', 'admin-acquisition-number', credits(item.atl_credits_settled_micro)));
      ['paid_intent', 'paid'].forEach((metric) => {
        const cell = document.createElement('td');
        cell.appendChild(acquisitionLink(group, metric, number(item[metric])));
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
    if (!body?.children.length) {
      const row = document.createElement('tr');
      const cell = node(
        'td',
        'admin-value-empty',
        available ? 'No acquisition groups match these filters.' : SECTION_UNAVAILABLE
      );
      cell.colSpan = 9;
      row.appendChild(cell);
      body?.appendChild(row);
    }
    const caption = table?.querySelector('caption');
    if (caption) {
      caption.textContent = `Acquisition groups by ${payload?.group_by === 'cohort' ? 'operating cohort' : 'source'}`;
    }
    element('adminAcquisitionMaturity').textContent = payload?.purchase_window_complete === false
      ? 'Purchase window incomplete: recent checkouts have not had seven full days to settle.'
      : '';
    element('adminAcquisitionRange').textContent = `${formatDate(state.range.start)} – ${formatDate(state.range.end)}`;
  }

  function sectionPanel(name) {
    return document.querySelector(`[data-admin-value-panel="${name}"]`);
  }

  function setSectionLoading(name, loading) {
    const panel = sectionPanel(name);
    const status = panel?.querySelector('[data-admin-value-status]');
    if (status) status.textContent = loading ? 'Loading section…' : '';
    panel?.setAttribute('aria-busy', loading ? 'true' : 'false');
  }

  function renderSection(name) {
    const section = state.sections[name];
    const panel = sectionPanel(name);
    const content = panel?.querySelector('[data-admin-value-content]');
    if (!content || !section.data) return;
    if (name === 'retention') renderRetention(section.data, content);
    else if (name === 'commercial') renderCommercial(section.data, content);
    else if (name === 'operational') renderOperational(section.data, content);
    else if (name === 'acquisition') renderAcquisition(section.data, content);
    const status = panel.querySelector('[data-admin-value-status]');
    status.textContent = section.stale
      ? 'Showing the last successful response; refresh failed.'
      : availabilityIncomplete(section.data.availability) ? 'Incomplete data' : '';
  }

  function sectionPath(name) {
    const params = name === 'acquisition' ? acquisitionParams() : rangeParams();
    if (name === 'operational') {
      const billing = element('adminOperationalBilling').value;
      const provider = element('adminOperationalProvider').value.trim();
      const model = element('adminOperationalModel').value.trim();
      if (billing) params.set('billing_mode', billing);
      if (provider) params.set('provider', provider);
      if (model) params.set('model', model);
    }
    return `${API_ENDPOINTS[name]}?${params}`;
  }

  async function loadSection(name, { keepStaleData = true } = {}) {
    const section = state.sections[name];
    if (!section) return;
    const panel = sectionPanel(name);
    const errorTarget = panel?.querySelector('[data-admin-value-error]');
    const retry = panel?.querySelector('[data-admin-value-retry]');
    setSectionLoading(name, true);
    if (errorTarget) errorTarget.hidden = true;
    if (retry) retry.hidden = true;
    try {
      const payload = await request(sectionPath(name));
      section.data = payload;
      section.loaded = true;
      section.error = null;
      section.stale = false;
      renderSection(name);
    } catch (error) {
      if (await handleAccessLost(error)) return;
      section.error = SECTION_UNAVAILABLE;
      section.stale = keepStaleData && Boolean(section.data);
      if (section.stale) renderSection(name);
      if (errorTarget) {
        errorTarget.textContent = SECTION_UNAVAILABLE;
        errorTarget.hidden = false;
      }
      if (retry) retry.hidden = false;
    } finally {
      setSectionLoading(name, false);
    }
  }

  async function ensureDisclosureLoaded(name) {
    const section = state.sections[name];
    if (section.loaded && !section.error) return renderSection(name);
    return loadSection(name, { keepStaleData: true });
  }

  function syncDisclosureControls({ load = true } = {}) {
    document.querySelectorAll('[data-admin-value-disclosure]').forEach((button) => {
      const name = button.dataset.adminValueDisclosure;
      const expanded = state.openDisclosures.has(name);
      button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
      const panel = sectionPanel(name);
      if (panel) panel.hidden = !expanded;
      if (expanded && load) ensureDisclosureLoaded(name);
    });
  }

  function fetchLifecycle() {
    const params = rangeParams();
    params.set('movement_range', state.movementRange);
    return request(`${API_ENDPOINTS.lifecycle}?${params}`);
  }

  function fetchPriorityUsers() {
    return request(`${API_ENDPOINTS.users}?${userParams()}`);
  }

  async function applySettledSection(name, result) {
    const section = state.sections[name];
    if (result.status === 'fulfilled') {
      section.data = result.value;
      section.loaded = true;
      section.error = null;
      section.stale = false;
      if (name === 'lifecycle') renderLifecycle(result.value);
      else renderUsers(result.value);
      return;
    }
    if (await handleAccessLost(result.reason)) return;
    section.error = SECTION_UNAVAILABLE;
    section.stale = Boolean(section.data);
    const target = name === 'lifecycle' ? element('adminValuePrimaryError') : element('adminPriorityError');
    if (target) {
      target.textContent = SECTION_UNAVAILABLE;
      target.hidden = false;
    }
    if (section.stale) {
      if (name === 'lifecycle') renderLifecycle(section.data);
      else renderUsers(section.data);
    }
  }

  async function refreshPrimary() {
    const requestSeq = ++state.requestSeq;
    element('adminAnalyticsHeadline').setAttribute('aria-busy', 'true');
    element('adminPriorityUsers').setAttribute('aria-busy', 'true');
    element('adminValuePrimaryError').hidden = true;
    element('adminPriorityError').hidden = true;
    element('adminValuePrimaryStatus').textContent = 'Refreshing user value analytics…';
    const results = await Promise.allSettled([fetchLifecycle(), fetchPriorityUsers()]);
    if (requestSeq !== state.requestSeq) return;
    await applySettledSection('lifecycle', results[0]);
    await applySettledSection('users', results[1]);
    element('adminValuePrimaryStatus').textContent = state.sections.lifecycle.stale
      ? 'Showing the last successful response; refresh failed.'
      : element('adminValuePrimaryStatus').textContent.replace('Refreshing user value analytics…', '');
  }

  function applyUserFilters(next = {}) {
    Object.assign(state.userFilters, next);
    setControls();
    writeUrlState();
    document.querySelectorAll('[data-lifecycle]').forEach((button) => {
      button.setAttribute('aria-pressed', button.dataset.lifecycle === state.userFilters.lifecycle ? 'true' : 'false');
    });
    state.sections.users.loaded = false;
    fetchPriorityUsers()
      .then((payload) => applySettledSection('users', { status: 'fulfilled', value: payload }))
      .catch((error) => applySettledSection('users', { status: 'rejected', reason: error }));
  }

  function setMovementRange(value) {
    if (!Object.hasOwn(MOVEMENT_RANGES, value) || value === state.movementRange) return;
    state.movementRange = value;
    setControls();
    writeUrlState();
    state.sections.lifecycle.loaded = false;
    refreshPrimary();
  }

  function setAnalyticsRange(value) {
    if (!Object.hasOwn(ANALYTICS_RANGES, value)) return;
    state.analyticsRange = value;
    state.range = utcRangeForPreset(value);
    setControls();
  }

  function bindEvents() {
    const analyticsRangeKeys = Object.keys(ANALYTICS_RANGES);
    document.querySelectorAll('[data-analytics-range]').forEach((button) => {
      button.addEventListener('click', () => setAnalyticsRange(button.dataset.analyticsRange));
      button.addEventListener('keydown', (event) => {
        if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const index = analyticsRangeKeys.indexOf(button.dataset.analyticsRange);
        const next = event.key === 'Home'
          ? analyticsRangeKeys[0]
          : event.key === 'End'
            ? analyticsRangeKeys[analyticsRangeKeys.length - 1]
            : analyticsRangeKeys[(index + (event.key === 'ArrowRight' ? 1 : -1) + analyticsRangeKeys.length) % analyticsRangeKeys.length];
        setAnalyticsRange(next);
        document.querySelector(`[data-analytics-range="${next}"]`)?.focus();
      });
    });
    const movementRangeKeys = Object.keys(MOVEMENT_RANGES);
    document.querySelectorAll('[data-movement-range]').forEach((button) => {
      button.addEventListener('click', () => setMovementRange(button.dataset.movementRange));
      button.addEventListener('keydown', (event) => {
        if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const index = movementRangeKeys.indexOf(button.dataset.movementRange);
        const next = event.key === 'Home'
          ? movementRangeKeys[0]
          : event.key === 'End'
            ? movementRangeKeys[movementRangeKeys.length - 1]
            : movementRangeKeys[(index + (event.key === 'ArrowRight' ? 1 : -1) + movementRangeKeys.length) % movementRangeKeys.length];
        setMovementRange(next);
        element(`adminLifecycleMovementRange${next}`).focus();
      });
    });
    element('adminAnalyticsRulesOpen')?.addEventListener('click', (event) => {
      renderRulesDialog();
      openDialog(element('adminAnalyticsRulesDialog'), event.currentTarget);
    });
    document.querySelectorAll('.admin-value-dialog').forEach((dialog) => {
      dialog.querySelectorAll('[data-admin-value-dialog-close]').forEach((button) => {
        button.addEventListener('click', () => closeDialog(dialog));
      });
      dialog.addEventListener('cancel', (event) => {
        event.preventDefault();
        closeDialog(dialog);
      });
      dialog.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;
        event.preventDefault();
        closeDialog(dialog);
      });
      dialog.addEventListener('click', (event) => {
        if (event.target === dialog) closeDialog(dialog);
      });
    });
    element('adminAnalyticsEvidenceProfile')?.addEventListener('click', () => {
      const user = state.evidenceUser;
      if (!user) return;
      closeDialog(element('adminAnalyticsEvidenceDialog'));
      state.userFilters.profile = String(user.user_id);
      writeUrlState();
      window.AdminAnalytics?.openProfile(user.user_id);
    });
    element('adminAnalyticsEvidenceAccount')?.addEventListener('click', () => {
      const user = state.evidenceUser;
      if (!user) return;
      closeDialog(element('adminAnalyticsEvidenceDialog'));
      state.userFilters.profile = '';
      writeUrlState();
      window.AdminTabs?.openAccountManagement({ userId: user.user_id, email: user.email });
    });
    element('adminAnalyticsValueFilters')?.addEventListener('submit', (event) => {
      event.preventDefault();
      const error = element('adminValueFilterError');
      try {
        state.includeInternal = element('adminValueInternal').checked;
        state.acquisition.source = element('adminAcquisitionSource').value;
        state.acquisition.cohort = element('adminAcquisitionCohort').value.trim();
        state.acquisition.lifecycle = element('adminAcquisitionLifecycle').value;
        state.acquisition.blocked = element('adminAcquisitionBlocked').value;
        state.acquisition.paid = element('adminAcquisitionPaid').value;
        state.acquisition.groupBy = element('adminAcquisitionGroupBy').value;
        state.acquisition.metric = '';
        error.hidden = true;
        writeUrlState();
        refresh();
      } catch (validationError) {
        error.textContent = validationError.message;
        error.hidden = false;
        error.focus();
      }
    });
    element('adminAnalyticsValueRefresh')?.addEventListener('click', refresh);
    element('adminPriorityFilters')?.addEventListener('submit', (event) => {
      event.preventDefault();
      applyUserFilters({
        query: element('adminPriorityQuery').value.trim(),
        lifecycle: element('adminPriorityLifecycle').value,
        operational: element('adminPriorityOperational').value,
        commercial: element('adminPriorityCommercial').value,
      });
    });
    document.querySelectorAll('[data-admin-value-disclosure]').forEach((button) => {
      button.addEventListener('click', () => {
        const name = button.dataset.adminValueDisclosure;
        if (state.openDisclosures.has(name)) state.openDisclosures.delete(name);
        else state.openDisclosures.add(name);
        writeUrlState();
        syncDisclosureControls();
      });
    });
    document.querySelectorAll('[data-admin-value-retry]').forEach((button) => {
      button.addEventListener('click', () => {
        const name = button.closest('[data-admin-value-panel]')?.dataset.adminValuePanel;
        if (name) loadSection(name, { keepStaleData: true });
      });
    });
    element('adminOperationalFilters')?.addEventListener('submit', (event) => {
      event.preventDefault();
      loadSection('operational', { keepStaleData: true });
    });
    document.addEventListener('admin:tabchange', (event) => {
      state.active = event.detail?.tab === 'analytics';
      if (state.active) onEnter();
    });
    window.addEventListener('popstate', () => {
      if (new URL(window.location.href).searchParams.get('adminTab') !== 'analytics') return;
      readUrlState();
      setControls();
      syncDisclosureControls();
      refreshPrimary();
    });
  }

  async function refresh() {
    if (!state.active) return;
    await refreshPrimary();
    const expanded = [...document.querySelectorAll('[data-admin-value-disclosure][aria-expanded="true"]')];
    await Promise.allSettled(expanded.map((button) => loadSection(button.dataset.adminValueDisclosure, { keepStaleData: true })));
  }

  function onEnter() {
    if (!state.initialized) {
      state.initialized = true;
      readUrlState();
      setControls();
      bindEvents();
      syncDisclosureControls();
    }
    const tab = new URL(window.location.href).searchParams.get('adminTab') || 'analytics';
    state.active = tab === 'analytics';
    if (!state.active) return;
    // The profile controller owns deep-link restoration. Avoid fetching the
    // overview behind an already-open profile on a direct URL or browser back.
    const profileRequested = /^\d+$/.test(state.userFilters.profile);
    if (!profileRequested && (!state.sections.lifecycle.loaded || !state.sections.users.loaded)) refreshPrimary();
  }

  function syncAuth(user) {
    if (user?.role === 'admin') return;
    state.active = false;
    state.requestSeq += 1;
    Object.values(state.sections).forEach((section) => {
      section.loaded = false;
      section.data = null;
      section.error = null;
      section.stale = false;
    });
    if (state.movementChart) {
      state.movementChart.destroy();
      state.movementChart = null;
    }
  }

  function getRange() {
    return state.range ? { ...state.range, dateRange: state.analyticsRange } : null;
  }

  window.AdminAnalyticsValue = { onEnter, refresh, syncAuth, applyUserFilters, getRange };
  document.addEventListener('DOMContentLoaded', () => {
    if (document.documentElement.dataset.navPage === 'admin') onEnter();
  });
})();
