/** Read-only Admin user-value analytics overview. */
(function () {
  'use strict';

  const API_BASE = (
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ) ? window.location.origin : '';
  const SECTION_UNAVAILABLE = 'This section is temporarily unavailable.';
  const API_ENDPOINTS = Object.freeze({
    overview: '/api/admin/analytics/overview',
    retention: '/api/admin/analytics/retention',
    commercial: '/api/admin/analytics/commercial',
    operational: '/api/admin/analytics/operational',
    acquisition: '/api/admin/analytics/acquisition',
    users: '/api/admin/analytics/users',
  });
  const ANALYTICS_RANGES = Object.freeze({
    '1d': { label: '1D', days: 1 },
    '1w': { label: '1W', days: 7 },
    '1m': { label: '1M', days: 30 },
    '1y': { label: '1Y', days: 365 },
  });
  const LIFECYCLE_LABELS = Object.freeze({
    new: 'New',
    onboarding: 'New',
    growing: 'Active',
    core: 'Active',
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
    competition: 'Competition team',
    unknown: 'Unknown',
  });
  const URL_KEYS = Object.freeze({
    analyticsRange: 'analyticsRange',
    acquisitionSource: 'analyticsAcquisitionSource',
    acquisitionCohort: 'analyticsAcquisitionCohort',
    acquisitionLifecycle: 'analyticsAcquisitionLifecycle',
    acquisitionPaid: 'analyticsAcquisitionPaid',
    acquisitionOpen: 'analyticsAcquisitionOpen',
    acquisitionMetric: 'analyticsAcquisitionMetric',
    usersView: 'analyticsUsersView',
    usersOffset: 'analyticsUsersOffset',
    profile: 'analyticsProfile',
  });

  const state = {
    initialized: false,
    active: false,
    requestSeq: 0,
    range: null,
    analyticsRange: '1w',
    acquisitionOpen: true,
    includeInternal: false,
    acquisition: { source: '', cohort: '', lifecycle: '', paid: '', metric: '' },
    directory: { active: false, offset: 0, limit: 50, total: 0, opener: null },
    openDisclosures: new Set(),
    deepOpen: false,
    sections: {
      overview: { loaded: false, data: null, error: null, stale: false },
      acquisition: { loaded: false, data: null, error: null, stale: false },
      users: { loaded: false, data: null, error: null, stale: false },
      retention: { loaded: false, data: null, error: null, stale: false },
      commercial: { loaded: false, data: null, error: null, stale: false },
      operational: { loaded: false, data: null, error: null, stale: false },
    },
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

  function request(path) {
    if (!window.API || typeof window.API.request !== 'function') {
      return Promise.reject(new Error('Admin Analytics API is not ready yet.'));
    }
    return window.API.request(API_BASE + path, { method: 'GET' });
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
      ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 6 }).format(parsed / 1000000) + ' Credits'
      : '—';
  }

  function dollarsFromMicro(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed)
      ? new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' }).format(parsed / 1000000)
      : '—';
  }

  function formatRatio(value) {
    if (value == null || value === '') return '—';
    const parsed = Number(value);
    return Number.isFinite(parsed)
      ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(parsed)
      : '—';
  }

  function formatDate(value) {
    const parsed = new Date(String(value).length === 10 ? value + 'T00:00:00Z' : value);
    return Number.isFinite(parsed.getTime())
      ? new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeZone: 'UTC' }).format(parsed)
      : '—';
  }

  function formatTimestamp(value) {
    const parsed = new Date(value);
    return Number.isFinite(parsed.getTime())
      ? new Intl.DateTimeFormat(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short',
        timeZone: 'UTC',
      }).format(parsed)
      : '—';
  }

  function relativeTime(value, now = Date.now()) {
    const timestamp = new Date(value).getTime();
    if (!Number.isFinite(timestamp)) return 'No activity';
    const elapsed = Math.max(0, now - timestamp);
    const minutes = Math.floor(elapsed / 60000);
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return String(minutes) + 'm ago';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return String(hours) + 'h ago';
    return String(Math.floor(hours / 24)) + 'd ago';
  }

  function availabilityIncomplete(availability) {
    if (!availability) return false;
    if (availability.status && availability.status !== 'ready') return true;
    return Object.values(availability).some(
      (item) => item?.status && item.status !== 'ready'
    );
  }

  function utcRangeForPreset(preset, now = new Date()) {
    const config = ANALYTICS_RANGES[preset] || ANALYTICS_RANGES['1w'];
    const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - (config.days - 1));
    return {
      start: start.toISOString().slice(0, 10),
      end: end.toISOString().slice(0, 10),
    };
  }

  function setOrDelete(params, key, value) {
    if (value) params.set(key, value);
    else params.delete(key);
  }

  function readUrlState() {
    const params = new URLSearchParams(window.location.search);
    const requestedRange = params.get(URL_KEYS.analyticsRange) || '1w';
    state.analyticsRange = Object.hasOwn(ANALYTICS_RANGES, requestedRange)
      ? requestedRange : '1w';
    state.range = utcRangeForPreset(state.analyticsRange);
    state.includeInternal = params.get('analyticsInternal') === 'true';

    const source = params.get(URL_KEYS.acquisitionSource) || '';
    const lifecycle = params.get(URL_KEYS.acquisitionLifecycle) || '';
    const paid = params.get(URL_KEYS.acquisitionPaid) || '';
    state.acquisition.source = Object.hasOwn(ACQUISITION_SOURCE_LABELS, source)
      ? source : '';
    state.acquisition.cohort = params.get(URL_KEYS.acquisitionCohort) || '';
    state.acquisition.lifecycle = ['new', 'active', 'at_risk', 'dormant'].includes(lifecycle)
      ? lifecycle : '';
    state.acquisition.paid = ['true', 'false'].includes(paid) ? paid : '';
    state.acquisition.metric = [
      'active', 'task_completed', 'repeat', 'paid_intent', 'paid',
    ].includes(params.get(URL_KEYS.acquisitionMetric))
      ? params.get(URL_KEYS.acquisitionMetric) : '';

    state.acquisitionOpen = params.get(URL_KEYS.acquisitionOpen) !== 'false';
    state.directory.active = params.get(URL_KEYS.usersView) === 'true';
    const offset = Number(params.get(URL_KEYS.usersOffset) || 0);
    state.directory.offset = Number.isSafeInteger(offset) && offset >= 0 ? offset : 0;
    state.openDisclosures = new Set(
      String(params.get('analyticsPanel') || '')
        .split(',')
        .filter((name) => ['retention', 'commercial', 'operational'].includes(name))
    );
    state.deepOpen = state.openDisclosures.size > 0;
  }

  function writeUrlState() {
    if (!window.history?.replaceState) return;
    const url = new URL(window.location.href);
    url.searchParams.set(URL_KEYS.analyticsRange, state.analyticsRange);
    [
      'analyticsStart',
      'analyticsEnd',
      'analyticsMovementRange',
      'analyticsOperational',
      'analyticsCommercial',
      'analyticsAcquisitionBlocked',
      'analyticsAcquisitionGroupBy',
    ].forEach((key) => url.searchParams.delete(key));
    setOrDelete(url.searchParams, 'analyticsInternal', state.includeInternal ? 'true' : '');
    setOrDelete(url.searchParams, URL_KEYS.acquisitionSource, state.acquisition.source);
    setOrDelete(url.searchParams, URL_KEYS.acquisitionCohort, state.acquisition.cohort);
    setOrDelete(url.searchParams, URL_KEYS.acquisitionLifecycle, state.acquisition.lifecycle);
    setOrDelete(url.searchParams, URL_KEYS.acquisitionPaid, state.acquisition.paid);
    setOrDelete(url.searchParams, URL_KEYS.acquisitionMetric, state.acquisition.metric);
    setOrDelete(
      url.searchParams,
      URL_KEYS.acquisitionOpen,
      state.acquisitionOpen ? '' : 'false'
    );
    setOrDelete(
      url.searchParams,
      URL_KEYS.usersView,
      state.directory.active ? 'true' : ''
    );
    setOrDelete(
      url.searchParams,
      URL_KEYS.usersOffset,
      state.directory.active && state.directory.offset
        ? String(state.directory.offset) : ''
    );
    setOrDelete(
      url.searchParams,
      'analyticsPanel',
      [...state.openDisclosures].sort().join(',')
    );
    window.history.replaceState(window.history.state, '', url);
  }

  function setControls() {
    const values = {
      adminValueInternal: state.includeInternal,
      adminAcquisitionSource: state.acquisition.source,
      adminAcquisitionCohort: state.acquisition.cohort,
      adminAcquisitionLifecycle: state.acquisition.lifecycle,
      adminAcquisitionPaid: state.acquisition.paid,
    };
    Object.entries(values).forEach(([id, value]) => {
      const control = element(id);
      if (!control) return;
      if (control.type === 'checkbox') control.checked = Boolean(value);
      else control.value = value;
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

  function overviewParams() {
    return rangeParams();
  }

  function appendAcquisitionFilters(params, { includeMetric = false } = {}) {
    if (state.acquisition.source) {
      params.set('acquisition_source', state.acquisition.source);
    }
    if (state.acquisition.cohort) {
      params.set('acquisition_cohort', state.acquisition.cohort);
    }
    if (state.acquisition.lifecycle) {
      params.set('lifecycle', state.acquisition.lifecycle);
    }
    if (state.acquisition.paid) {
      params.set('paid', state.acquisition.paid);
    }
    if (includeMetric && state.acquisition.metric) {
      const key = {
        active: 'acquisition_active',
        task_completed: 'acquisition_task_completed',
        repeat: 'acquisition_repeat',
        paid_intent: 'acquisition_paid_intent',
        paid: 'paid',
      }[state.acquisition.metric];
      if (key) params.set(key, 'true');
    }
    return params;
  }

  function acquisitionParams() {
    const params = appendAcquisitionFilters(rangeParams());
    params.set('group_by', 'source');
    return params;
  }

  function userParams({ priority = true, limit = 25, offset = 0 } = {}) {
    const params = new URLSearchParams({
      priority: priority ? 'true' : 'false',
      include_internal: state.includeInternal ? 'true' : 'false',
      limit: String(limit),
      offset: String(offset),
      date_range: state.analyticsRange,
    });
    return appendAcquisitionFilters(params, { includeMetric: true });
  }

  function summarizeAcquisition(payload) {
    return (Array.isArray(payload?.groups) ? payload.groups : []).reduce(
      (total, group) => ({
        users: total.users + Number(group.users || 0),
        activeUsers: total.activeUsers + Number(group.active || 0),
        taskCompleted: total.taskCompleted + Number(group.task_completed || 0),
        repeatUsers: total.repeatUsers + Number(group.repeat || 0),
        runs: total.runs + Number(group.runs || 0),
        creditsSettledMicro:
          total.creditsSettledMicro + Number(group.atl_credits_settled_micro || 0),
        paidIntent: total.paidIntent + Number(group.paid_intent || 0),
        paid: total.paid + Number(group.paid || 0),
      }),
      {
        users: 0,
        activeUsers: 0,
        taskCompleted: 0,
        repeatUsers: 0,
        runs: 0,
        creditsSettledMicro: 0,
        paidIntent: 0,
        paid: 0,
      }
    );
  }

  function setPulseMetric(name, value) {
    const target = document.querySelector(
      '[data-admin-value-metric="' + name + '"]'
    );
    if (target) target.textContent = value;
  }

  function renderPulse(summary) {
    setPulseMetric('active-users', number(summary.activeUsers));
    setPulseMetric('task-completed', number(summary.taskCompleted));
    setPulseMetric('repeat-users', number(summary.repeatUsers));
    setPulseMetric('credits-settled', credits(summary.creditsSettledMicro));
    element('adminAnalyticsHeadline')?.setAttribute('aria-busy', 'false');
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
    return metricLabel
      ? 'Show ' + metricLabel + ' from ' + groupLabel
      : 'Show users from ' + groupLabel;
  }

  function acquisitionLink(group, metric, value) {
    const button = node('button', 'admin-acquisition-drilldown', value);
    button.type = 'button';
    button.setAttribute('aria-label', acquisitionDrilldownLabel(group, metric));
    button.addEventListener('click', (event) => {
      openAnalyticsUsers({ group, metric, opener: event.currentTarget });
    });
    return button;
  }

  function renderAcquisition(payload) {
    const container = element('adminAcquisitionPanel');
    const table = container?.querySelector('table');
    const body = table?.querySelector('tbody');
    clear(body);
    const available = payload?.availability?.available !== false;
    const groups = available && Array.isArray(payload?.groups) ? payload.groups : [];
    groups.forEach((item) => {
      const row = document.createElement('tr');
      const group = item.group || {
        kind: 'source',
        value: 'unknown',
        label: 'Unknown',
      };
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
      row.appendChild(
        node('td', 'admin-analytics-number', formatRatio(item.runs_per_active_user))
      );
      row.appendChild(
        node('td', 'admin-analytics-number', credits(item.atl_credits_settled_micro))
      );
      ['paid_intent', 'paid'].forEach((metric) => {
        const cell = document.createElement('td');
        cell.appendChild(acquisitionLink(group, metric, number(item[metric])));
        row.appendChild(cell);
      });
      body?.appendChild(row);
    });
    if (!body?.children.length) {
      const row = document.createElement('tr');
      const cell = node(
        'td',
        'admin-value-empty',
        available
          ? 'No acquisition groups match these filters.'
          : SECTION_UNAVAILABLE
      );
      cell.colSpan = 9;
      row.appendChild(cell);
      body?.appendChild(row);
    }
    const caption = table?.querySelector('caption');
    if (caption) caption.textContent = 'Acquisition groups by source';
    const maturity = element('adminAcquisitionMaturity');
    if (maturity) {
      maturity.textContent = payload?.purchase_window_complete === false
        ? 'Purchase window incomplete: recent checkouts have not had seven full days to settle.'
        : '';
    }
    const range = element('adminAcquisitionRange');
    if (range) {
      range.textContent =
        formatDate(state.range.start) + ' – ' + formatDate(state.range.end);
    }
  }

  function renderReturnPanel(payload) {
    const target = element('adminAnalyticsReturnContent');
    clear(target);
    const acquisitionScoped = Boolean(
      state.acquisition.source
      || state.acquisition.cohort
      || state.acquisition.lifecycle
      || state.acquisition.paid
    );
    if (acquisitionScoped) {
      target?.appendChild(
        node('p', 'admin-value-empty', 'Unavailable for acquisition-filtered slices.')
      );
      return;
    }
    const daily = Object.entries(payload?.daily_active_users || {}).sort(
      ([left], [right]) => left.localeCompare(right)
    );
    const chart = node('div', 'admin-analytics-return-chart');
    const max = Math.max(1, ...daily.map(([, value]) => Number(value || 0)));
    daily.forEach(([day, value]) => {
      const column = node('div', 'admin-analytics-return-column');
      const bar = node('span', 'admin-analytics-return-bar');
      bar.style.setProperty('--bar-ratio', String(Number(value || 0) / max));
      bar.setAttribute(
        'aria-label',
        formatDate(day) + ': ' + number(value) + ' active users'
      );
      const weekday = new Intl.DateTimeFormat(undefined, {
        weekday: 'short',
        timeZone: 'UTC',
      }).format(new Date(day + 'T00:00:00Z'));
      column.append(bar, node('small', '', weekday));
      chart.appendChild(column);
    });
    target?.appendChild(
      daily.length
        ? chart
        : node('p', 'admin-value-empty', 'No activity in this range.')
    );
    target?.appendChild(
      node(
        'p',
        'admin-analytics-legend',
        'Active users · Returning-user series unavailable'
      )
    );
  }

  function metricList(entries) {
    const list = node('ul', 'admin-analytics-signal-list');
    entries.forEach(([label, value]) => {
      const item = document.createElement('li');
      item.append(node('span', '', label), node('strong', '', value));
      list.appendChild(item);
    });
    return list;
  }

  function funnelList(entries) {
    const list = node('ol', 'admin-analytics-funnel');
    entries.forEach(([label, value]) => {
      const item = document.createElement('li');
      item.append(node('span', '', label), node('strong', '', value));
      list.appendChild(item);
    });
    return list;
  }

  function renderValueExchange(summary) {
    const target = element('adminAnalyticsValueExchangeContent');
    clear(target);
    const runsPerActive = summary.activeUsers
      ? summary.runs / summary.activeUsers : null;
    const creditsPerUser = summary.users
      ? summary.creditsSettledMicro / summary.users : null;
    target?.appendChild(metricList([
      ['Runs per active user', formatRatio(runsPerActive)],
      [
        'Credits settled / user',
        creditsPerUser == null ? '—' : credits(creditsPerUser),
      ],
      ['Checkout starters', number(summary.paidIntent)],
    ]));
    target?.appendChild(funnelList([
      ['Credits page viewed', 'Unavailable'],
      ['Checkout started', number(summary.paidIntent)],
      ['Purchase settled', number(summary.paid)],
    ]));
  }

  function profileHref(userId) {
    const url = new URL(window.location.href);
    url.searchParams.set('adminTab', 'analytics');
    url.searchParams.set('analyticsUser', String(userId));
    url.searchParams.set(URL_KEYS.profile, String(userId));
    url.searchParams.set('analyticsSection', 'overview');
    url.searchParams.delete(URL_KEYS.usersView);
    url.searchParams.delete(URL_KEYS.usersOffset);
    return url.pathname + url.search;
  }

  function profileLink(user, { iconOnly = false } = {}) {
    const label = user.display_name || user.email || 'User #' + user.user_id;
    const link = node(
      'a',
      'admin-priority-profile-link' + (iconOnly ? ' is-icon-only' : '')
    );
    link.href = profileHref(user.user_id);
    link.setAttribute('aria-label', 'Open analytics profile for ' + label);
    if (iconOnly) {
      link.title = 'Open analytics profile for ' + label;
      const icon = document.createElement('svg');
      icon.setAttribute('aria-hidden', 'true');
      const use = document.createElement('use');
      use.setAttribute('href', '#icon-chevron-right');
      icon.appendChild(use);
      link.append(icon, node('span', 'sr-only', 'Open ' + label));
    } else {
      link.textContent = label;
    }
    link.addEventListener('click', (event) => {
      if (
        event.button !== 0
        || event.metaKey
        || event.ctrlKey
        || event.shiftKey
        || event.altKey
      ) return;
      if (typeof window.AdminAnalytics?.openProfile !== 'function') return;
      event.preventDefault();
      window.AdminAnalytics.openProfile(user.user_id);
    });
    return link;
  }

  function actionStatus(user) {
    if (user.operational?.state !== 'healthy') {
      return OPERATIONAL_LABELS[user.operational?.state] || 'Needs attention';
    }
    const segment = user.lifecycle?.segment;
    if (segment === 'at_risk') return 'At risk';
    if (segment === 'dormant') return 'Dormant';
    if (
      Number(user.accepted_runs_in_range || 0) > 0
      || ['growing', 'core'].includes(segment)
    ) return 'Active';
    return 'New';
  }

  function renderActionQueue(payload) {
    const body = element('adminPriorityUsers');
    clear(body);
    const items = Array.isArray(payload?.items) ? payload.items : [];
    items.slice(0, 12).forEach((user) => {
      const row = document.createElement('tr');
      const identity = document.createElement('th');
      identity.scope = 'row';
      identity.append(
        profileLink(user),
        node('span', 'admin-analytics-user-email', user.email)
      );
      row.appendChild(identity);
      row.appendChild(
        node(
          'td',
          '',
          ACQUISITION_SOURCE_LABELS[user.acquisition?.source] || 'Unknown'
        )
      );
      row.appendChild(node('td', '', actionStatus(user)));
      row.appendChild(
        node(
          'td',
          '',
          user.operational?.state === 'healthy'
            ? user.lifecycle?.reason
            : user.operational?.reason
        )
      );
      row.appendChild(
        node(
          'td',
          '',
          relativeTime(user.lifecycle?.last_meaningful_activity_at)
        )
      );
      row.appendChild(
        node(
          'td',
          'admin-analytics-number',
          number(user.accepted_runs_in_range)
        )
      );
      const action = document.createElement('td');
      action.appendChild(profileLink(user, { iconOnly: true }));
      row.appendChild(action);
      body?.appendChild(row);
    });
    if (!items.length) {
      const row = document.createElement('tr');
      const cell = node(
        'td',
        'admin-value-empty',
        'No users need attention for these filters.'
      );
      cell.colSpan = 7;
      row.appendChild(cell);
      body?.appendChild(row);
    }
    body?.setAttribute('aria-busy', 'false');
    const range = element('adminPriorityUsersRange');
    if (range) range.textContent = number(payload?.total || 0) + ' users';
  }

  function renderUsersDirectory(payload) {
    const body = element('adminAnalyticsUsersDirectoryBody');
    clear(body);
    const items = Array.isArray(payload?.items) ? payload.items : [];
    items.forEach((user) => {
      const row = document.createElement('tr');
      const identity = document.createElement('th');
      identity.scope = 'row';
      identity.append(
        profileLink(user),
        node('span', 'admin-analytics-user-email', user.email)
      );
      row.appendChild(identity);
      row.appendChild(
        node(
          'td',
          '',
          ACQUISITION_SOURCE_LABELS[user.acquisition?.source] || 'Unknown'
        )
      );
      row.appendChild(
        node(
          'td',
          '',
          LIFECYCLE_LABELS[user.lifecycle?.segment] || 'Unknown'
        )
      );
      row.appendChild(
        node(
          'td',
          '',
          OPERATIONAL_LABELS[user.operational?.state] || 'Unknown'
        )
      );
      row.appendChild(
        node(
          'td',
          '',
          relativeTime(user.lifecycle?.last_meaningful_activity_at)
        )
      );
      row.appendChild(
        node(
          'td',
          'admin-analytics-number',
          number(user.accepted_runs_in_range)
        )
      );
      const action = document.createElement('td');
      const manage = node('button', 'credits-key-action', 'Manage');
      manage.type = 'button';
      manage.setAttribute(
        'aria-label',
        'Manage ' + (user.display_name || user.email)
      );
      manage.addEventListener('click', () => {
        closeUsersDirectory({ updateUrl: false, focus: false });
        window.AdminTabs?.openAccountManagement({
          userId: user.user_id,
          email: user.email,
        });
      });
      action.appendChild(manage);
      row.appendChild(action);
      body?.appendChild(row);
    });
    if (!items.length) {
      const row = document.createElement('tr');
      const cell = node(
        'td',
        'admin-value-empty',
        'No users match these Analytics filters.'
      );
      cell.colSpan = 7;
      row.appendChild(cell);
      body?.appendChild(row);
    }
    state.directory.total = Number(payload?.total || 0);
    const start = state.directory.total ? state.directory.offset + 1 : 0;
    const end = Math.min(
      state.directory.offset + items.length,
      state.directory.total
    );
    const range = element('adminAnalyticsUsersDirectoryRange');
    if (range) {
      range.textContent =
        'Showing ' + start + '–' + end + ' of ' + state.directory.total;
    }
    const previous = element('adminAnalyticsUsersDirectoryPrev');
    const next = element('adminAnalyticsUsersDirectoryNext');
    if (previous) previous.disabled = state.directory.offset <= 0;
    if (next) next.disabled = end >= state.directory.total;
  }

  async function refreshUsersDirectory() {
    const status = element('adminAnalyticsUsersDirectoryStatus');
    const error = element('adminAnalyticsUsersDirectoryError');
    const retry = element('adminAnalyticsUsersDirectoryRetry');
    if (status) status.textContent = 'Loading filtered users...';
    if (error) error.hidden = true;
    if (retry) retry.hidden = true;
    try {
      const params = userParams({
        priority: false,
        limit: state.directory.limit,
        offset: state.directory.offset,
      });
      const payload = await request(API_ENDPOINTS.users + '?' + params);
      renderUsersDirectory(payload);
      if (status) status.textContent = '';
    } catch (requestError) {
      if (await handleAccessLost(requestError)) return;
      if (status) status.textContent = '';
      if (error) {
        error.textContent = SECTION_UNAVAILABLE;
        error.hidden = false;
      }
      if (retry) retry.hidden = false;
    }
  }

  function updateScopeCopy() {
    const source = state.acquisition.source
      ? (
        ACQUISITION_SOURCE_LABELS[state.acquisition.source]
        || state.acquisition.source
      )
      : 'All acquisition sources';
    const cohort = state.acquisition.cohort || 'all cohorts';
    const lifecycle = state.acquisition.lifecycle
      ? (
        LIFECYCLE_LABELS[state.acquisition.lifecycle]
        || state.acquisition.lifecycle
      )
      : 'all lifecycle stages';
    const paid = state.acquisition.paid === 'true'
      ? 'paid users'
      : state.acquisition.paid === 'false'
        ? 'unpaid users'
        : 'all paid states';
    const scope =
      source + ' · ' + cohort + ' · ' + lifecycle + ' · ' + paid;
    const context = element('adminAnalyticsContext');
    if (context) {
      clear(context);
      context.append(
        node('span', '', scope),
        node(
          'span',
          '',
          'Progress is lifetime · activity is '
            + ANALYTICS_RANGES[state.analyticsRange].label
        )
      );
    }
    const directoryScope = element('adminAnalyticsUsersDirectoryScope');
    if (directoryScope) directoryScope.textContent = scope;
    const selectedRange = element('adminAnalyticsSelectedRange');
    if (selectedRange) {
      selectedRange.textContent =
        'Selected range: ' + ANALYTICS_RANGES[state.analyticsRange].label;
    }
    const railState = element('adminRailInternalState');
    if (railState) {
      railState.textContent = state.includeInternal
        ? 'Internal accounts included'
        : 'Internal accounts hidden';
    }
  }

  function openAnalyticsUsers({
    group = null,
    metric = '',
    opener = document.activeElement,
  } = {}) {
    if (group?.kind === 'source') state.acquisition.source = group.value;
    if (group?.kind === 'cohort') state.acquisition.cohort = group.value;
    state.acquisition.metric = metric;
    if (metric === 'paid') state.acquisition.paid = 'true';
    state.directory.active = true;
    state.directory.offset = 0;
    state.directory.opener = opener;
    setControls();
    updateScopeCopy();
    writeUrlState();
    window.AdminTabs?.setTab('users');
    syncUsersDirectory();
  }

  function closeUsersDirectory({ updateUrl = true, focus = true } = {}) {
    state.directory.active = false;
    state.directory.offset = 0;
    const directory = element('adminAnalyticsUsersDirectory');
    if (directory) directory.hidden = true;
    if (updateUrl) writeUrlState();
    if (focus) element('adminCreditsUserQuery')?.focus();
  }

  function syncUsersDirectory() {
    const tab = new URL(window.location.href).searchParams.get('adminTab');
    const visible = state.directory.active && tab === 'users';
    const directory = element('adminAnalyticsUsersDirectory');
    if (directory) directory.hidden = !visible;
    if (!visible) return;
    updateScopeCopy();
    refreshUsersDirectory();
    element('adminAnalyticsUsersDirectoryTitle')?.focus();
  }

  function metricCard(label, value, detail) {
    const card = node('article', 'admin-value-metric-card');
    card.appendChild(node('span', '', label));
    card.appendChild(node('strong', '', value));
    if (detail) card.appendChild(node('small', '', detail));
    return card;
  }

  function summaryGrid(entries) {
    const grid = node('div', 'admin-value-summary-grid');
    entries.forEach(([label, value, detail]) => {
      grid.appendChild(metricCard(label, value, detail));
    });
    return grid;
  }

  function renderRetention(payload, container) {
    clear(container);
    container?.appendChild(summaryGrid([
      [
        'Week 1',
        percent(payload.summary_week_1?.rate),
        payload.summary_week_1?.mature
          ? number(payload.summary_week_1.eligible_users) + ' eligible'
          : 'Cohorts are still maturing',
      ],
      [
        'Week 2',
        percent(payload.summary_week_2?.rate),
        payload.summary_week_2?.mature
          ? number(payload.summary_week_2.eligible_users) + ' eligible'
          : 'Cohorts are still maturing',
      ],
      [
        'Week 4',
        percent(payload.summary_week_4?.rate),
        payload.summary_week_4?.mature
          ? number(payload.summary_week_4.eligible_users) + ' eligible'
          : 'Cohorts are still maturing',
      ],
    ]));
    const wrapper = node('div', 'admin-analytics-table-wrap');
    const table = document.createElement('table');
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    ['Activation week', 'Activated', 'Week 1', 'Week 2', 'Week 4'].forEach(
      (label) => {
        const cell = node('th', '', label);
        cell.scope = 'col';
        headRow.appendChild(cell);
      }
    );
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
      const cell = node(
        'td',
        'admin-value-empty',
        'No activation cohorts in this range.'
      );
      cell.colSpan = 5;
      row.appendChild(cell);
      body.appendChild(row);
    }
    table.appendChild(body);
    wrapper.appendChild(table);
    container?.appendChild(wrapper);
  }

  function renderCommercial(payload, container) {
    clear(container);
    const period = payload.selected_period || {};
    const balances = payload.current_balances || {};
    container?.appendChild(summaryGrid([
      [
        'Lifetime net purchased',
        credits(payload.lifetime_net_purchased_micro),
        'Settled purchases minus refunds',
      ],
      ['Purchased in range', credits(period.purchased_micro), 'Revenue signal'],
      ['Consumed in range', credits(period.consumed_micro), 'Model execution'],
      [
        'Platform model cost',
        dollarsFromMicro(period.platform_model_cost_micro_usd),
        'Platform Credits lane',
      ],
    ]));
    const tiers = node('div', 'admin-commercial-tier-grid');
    Object.keys(COMMERCIAL_LABELS).forEach((tier) => {
      tiers.appendChild(
        metricCard(
          COMMERCIAL_LABELS[tier],
          number(payload.tier_counts?.[tier] || 0),
          'users'
        )
      );
    });
    container?.appendChild(tiers);
    container?.appendChild(summaryGrid([
      [
        'Grant balance',
        credits(balances.grant_available_micro),
        'Not revenue',
      ],
      [
        'Purchased balance',
        credits(balances.purchased_available_micro),
        'Customer-funded',
      ],
      [
        'Total available',
        credits(balances.total_available_micro),
        'Current spendable balance',
      ],
      [
        'Admin Grant activity',
        credits(period.admin_grant_activity_micro),
        'Excluded from revenue',
      ],
    ]));
  }

  function renderOperational(payload, container) {
    clear(container);
    container?.appendChild(summaryGrid([
      [
        'Blocked',
        number(payload.operational_state_counts?.blocked || 0),
        'Core action unavailable',
      ],
      [
        'Needs attention',
        number(payload.operational_state_counts?.needs_attention || 0),
        'Operator review recommended',
      ],
      [
        'Healthy',
        number(payload.operational_state_counts?.healthy || 0),
        'No supported current issue',
      ],
      [
        'Backtest success',
        percent(payload.backtest_success_rate),
        number(payload.completed_runs) + ' completed · '
          + number(payload.failed_runs) + ' failed',
      ],
    ]));
    const wrapper = node('div', 'admin-analytics-table-wrap');
    const table = document.createElement('table');
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
      row.appendChild(
        node(
          'td',
          '',
          String(failure.error_category || 'Unknown').replaceAll('_', ' ')
        )
      );
      row.appendChild(node('td', '', number(failure.affected_users)));
      body.appendChild(row);
    });
    if (!body.children.length) {
      const row = document.createElement('tr');
      const cell = node(
        'td',
        'admin-value-empty',
        'No failure categories in this range.'
      );
      cell.colSpan = 2;
      row.appendChild(cell);
      body.appendChild(row);
    }
    table.appendChild(body);
    wrapper.appendChild(table);
    container?.appendChild(wrapper);
  }

  function sectionPanel(name) {
    return document.querySelector('[data-admin-value-panel="' + name + '"]');
  }

  function sectionPath(name) {
    return API_ENDPOINTS[name] + '?' + rangeParams();
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
    if (name === 'commercial') renderCommercial(section.data, content);
    if (name === 'operational') renderOperational(section.data, content);
    const status = panel.querySelector('[data-admin-value-status]');
    if (status) {
      status.textContent = section.stale
        ? 'Showing the last successful response; refresh failed.'
        : availabilityIncomplete(section.data.availability)
          ? 'Incomplete data'
          : '';
    }
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
    document.querySelectorAll('[data-admin-value-disclosure]').forEach(
      (button) => {
        const name = button.dataset.adminValueDisclosure;
        const expanded = state.openDisclosures.has(name);
        button.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        const panel = sectionPanel(name);
        if (panel) panel.hidden = !expanded;
        if (expanded && load) ensureDisclosureLoaded(name);
      }
    );
    const overview = element('adminAnalyticsOverview');
    const deep = element('adminAnalyticsDeepSections');
    if (overview) overview.hidden = state.deepOpen;
    if (deep) deep.hidden = !state.deepOpen;
  }

  function openDeepAnalysis(name, opener) {
    if (!['retention', 'commercial', 'operational'].includes(name)) return;
    state.openDisclosures.add(name);
    state.deepOpen = true;
    state.directory.opener = opener;
    writeUrlState();
    syncDisclosureControls();
    element('adminAnalyticsDeepTitle')?.focus();
  }

  function closeDeepAnalysis() {
    state.deepOpen = false;
    state.openDisclosures.clear();
    writeUrlState();
    syncDisclosureControls({ load: false });
    const opener = document.querySelector('[data-open-analysis="retention"]');
    opener?.focus();
  }

  function setAcquisitionExpanded(expanded, { writeUrl = true } = {}) {
    state.acquisitionOpen = Boolean(expanded);
    const panel = element('adminAcquisitionPanel');
    const toggle = element('adminAcquisitionToggle');
    if (panel) panel.hidden = !state.acquisitionOpen;
    if (toggle) {
      toggle.setAttribute('aria-expanded', String(state.acquisitionOpen));
      toggle.setAttribute(
        'aria-label',
        state.acquisitionOpen
          ? 'Collapse acquisition groups'
          : 'Expand acquisition groups'
      );
      toggle.classList.toggle('is-collapsed', !state.acquisitionOpen);
    }
    if (writeUrl) writeUrlState();
  }

  async function applyPrimaryResult(name, result) {
    const section = state.sections[name];
    if (result.status === 'fulfilled') {
      section.data = result.value;
      section.loaded = true;
      section.error = null;
      section.stale = false;
      if (name === 'overview' && result.value?.last_updated) {
        const updated = element('adminAnalyticsLastUpdated');
        if (updated) {
          updated.dateTime = String(result.value.last_updated);
          updated.textContent = formatTimestamp(result.value.last_updated);
        }
      }
      return;
    }
    if (await handleAccessLost(result.reason)) return;
    section.error = SECTION_UNAVAILABLE;
    section.stale = Boolean(section.data);
  }

  function setPrimaryLoading(loading) {
    element('adminAnalyticsHeadline')?.setAttribute(
      'aria-busy',
      String(loading)
    );
    element('adminAcquisitionPanel')?.setAttribute(
      'aria-busy',
      String(loading)
    );
    element('adminPriorityUsers')?.setAttribute(
      'aria-busy',
      String(loading)
    );
    const status = element('adminValuePrimaryStatus');
    if (status) {
      status.textContent = loading ? 'Refreshing analytics...' : '';
    }
  }

  function setPrimaryRetry(name, visible) {
    document.querySelectorAll(
      '[data-admin-primary-retry="' + name + '"]'
    ).forEach((button) => {
      button.hidden = !visible;
    });
  }

  function renderPrimaryErrors() {
    const mappings = [
      [
        'acquisition',
        element('adminAcquisitionPanel')?.querySelector(
          '[data-admin-value-error]'
        ),
      ],
      [
        'overview',
        element('adminAnalyticsReturn')?.querySelector(
          '[data-admin-value-error]'
        ),
      ],
      ['users', element('adminPriorityError')],
    ];
    mappings.forEach(([name, target]) => {
      if (!target) return;
      target.textContent = state.sections[name].error || '';
      target.hidden = !state.sections[name].error;
    });
    const pulseError = element('adminValuePrimaryError');
    if (pulseError) {
      pulseError.textContent = state.sections.acquisition.error || '';
      pulseError.hidden = !state.sections.acquisition.error;
    }
    const exchangeError = element('adminValueExchangeError');
    if (exchangeError) {
      exchangeError.textContent = state.sections.acquisition.error || '';
      exchangeError.hidden = !state.sections.acquisition.error;
    }
  }

  function renderCombinedOverview() {
    const acquisition = state.sections.acquisition;
    const overview = state.sections.overview;
    const users = state.sections.users;
    if (acquisition.data) {
      const summary = summarizeAcquisition(acquisition.data);
      renderPulse(summary);
      renderAcquisition(acquisition.data);
      renderValueExchange(summary);
    }
    if (overview.data) renderReturnPanel(overview.data);
    if (users.data) renderActionQueue(users.data);
    renderPrimaryErrors();
    setPrimaryRetry('acquisition', Boolean(acquisition.error));
    setPrimaryRetry('overview', Boolean(overview.error));
    setPrimaryRetry('users', Boolean(users.error));
    updateScopeCopy();
  }

  async function refreshPrimary() {
    const requestSeq = ++state.requestSeq;
    setPrimaryLoading(true);
    const results = await Promise.allSettled([
      request(API_ENDPOINTS.acquisition + '?' + acquisitionParams()),
      request(API_ENDPOINTS.overview + '?' + overviewParams()),
      request(API_ENDPOINTS.users + '?' + userParams()),
    ]);
    if (requestSeq !== state.requestSeq) return;
    await applyPrimaryResult('acquisition', results[0]);
    await applyPrimaryResult('overview', results[1]);
    await applyPrimaryResult('users', results[2]);
    renderCombinedOverview();
    setPrimaryLoading(false);
  }

  async function refreshPrimarySection(name) {
    const path = name === 'acquisition'
      ? API_ENDPOINTS.acquisition + '?' + acquisitionParams()
      : name === 'overview'
        ? API_ENDPOINTS.overview + '?' + overviewParams()
        : API_ENDPOINTS.users + '?' + userParams();
    const result = await Promise.allSettled([request(path)]);
    await applyPrimaryResult(name, result[0]);
    renderCombinedOverview();
  }

  function applyOverviewFilters() {
    const form = element('adminAnalyticsValueFilters');
    if (!form?.reportValidity()) return;
    state.includeInternal = Boolean(element('adminValueInternal')?.checked);
    state.acquisition.source = element('adminAcquisitionSource')?.value || '';
    state.acquisition.cohort =
      element('adminAcquisitionCohort')?.value.trim() || '';
    state.acquisition.lifecycle =
      element('adminAcquisitionLifecycle')?.value || '';
    state.acquisition.paid = element('adminAcquisitionPaid')?.value || '';
    state.acquisition.metric = '';
    writeUrlState();
    refresh();
  }

  let filterTimer = null;
  function scheduleFilterRefresh() {
    window.clearTimeout(filterTimer);
    filterTimer = window.setTimeout(applyOverviewFilters, 350);
  }

  function setAnalyticsRange(value) {
    if (!Object.hasOwn(ANALYTICS_RANGES, value)) return;
    state.analyticsRange = value;
    state.range = utcRangeForPreset(value);
    setControls();
    writeUrlState();
    refresh();
  }

  function bindEvents() {
    const rangeKeys = Object.keys(ANALYTICS_RANGES);
    document.querySelectorAll('[data-analytics-range]').forEach((button) => {
      button.addEventListener(
        'click',
        () => setAnalyticsRange(button.dataset.analyticsRange)
      );
      button.addEventListener('keydown', (event) => {
        if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) {
          return;
        }
        event.preventDefault();
        const index = rangeKeys.indexOf(button.dataset.analyticsRange);
        const next = event.key === 'Home'
          ? rangeKeys[0]
          : event.key === 'End'
            ? rangeKeys[rangeKeys.length - 1]
            : rangeKeys[
              (
                index
                + (event.key === 'ArrowRight' ? 1 : -1)
                + rangeKeys.length
              ) % rangeKeys.length
            ];
        setAnalyticsRange(next);
        document.querySelector(
          '[data-analytics-range="' + next + '"]'
        )?.focus();
      });
    });

    const form = element('adminAnalyticsValueFilters');
    form?.addEventListener('submit', (event) => {
      event.preventDefault();
      applyOverviewFilters();
    });
    [
      'adminAcquisitionSource',
      'adminAcquisitionLifecycle',
      'adminAcquisitionPaid',
      'adminValueInternal',
    ].forEach((id) => {
      element(id)?.addEventListener('change', applyOverviewFilters);
    });
    element('adminAcquisitionCohort')?.addEventListener(
      'input',
      scheduleFilterRefresh
    );
    element('adminAcquisitionCohort')?.addEventListener(
      'keydown',
      (event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        window.clearTimeout(filterTimer);
        applyOverviewFilters();
      }
    );

    element('adminAnalyticsValueRefresh')?.addEventListener('click', refresh);
    element('adminAcquisitionToggle')?.addEventListener('click', () => {
      setAcquisitionExpanded(!state.acquisitionOpen);
    });
    document.querySelectorAll('[data-admin-primary-retry]').forEach((button) => {
      button.addEventListener('click', () => {
        refreshPrimarySection(button.dataset.adminPrimaryRetry);
      });
    });
    document.querySelectorAll('[data-open-analysis]').forEach((button) => {
      button.addEventListener('click', (event) => {
        openDeepAnalysis(
          button.dataset.openAnalysis,
          event.currentTarget
        );
      });
    });
    element('adminAnalyticsDeepBack')?.addEventListener(
      'click',
      closeDeepAnalysis
    );
    document.querySelectorAll('[data-admin-value-disclosure]').forEach(
      (button) => {
        button.addEventListener('click', () => {
          const name = button.dataset.adminValueDisclosure;
          if (state.openDisclosures.has(name)) {
            state.openDisclosures.delete(name);
          } else {
            state.openDisclosures.add(name);
          }
          writeUrlState();
          syncDisclosureControls();
        });
      }
    );
    document.querySelectorAll('[data-admin-value-retry]').forEach((button) => {
      button.addEventListener('click', () => {
        const name = button.closest(
          '[data-admin-value-panel]'
        )?.dataset.adminValuePanel;
        if (name) loadSection(name, { keepStaleData: true });
      });
    });

    element('adminAnalyticsOpenAllUsers')?.addEventListener(
      'click',
      (event) => {
        openAnalyticsUsers({ opener: event.currentTarget });
      }
    );
    element('adminAnalyticsUsersDirectoryClose')?.addEventListener(
      'click',
      () => closeUsersDirectory()
    );
    element('adminAnalyticsUsersDirectoryRetry')?.addEventListener(
      'click',
      refreshUsersDirectory
    );
    element('adminAnalyticsUsersDirectoryPrev')?.addEventListener(
      'click',
      () => {
        state.directory.offset = Math.max(
          0,
          state.directory.offset - state.directory.limit
        );
        writeUrlState();
        refreshUsersDirectory();
      }
    );
    element('adminAnalyticsUsersDirectoryNext')?.addEventListener(
      'click',
      () => {
        state.directory.offset += state.directory.limit;
        writeUrlState();
        refreshUsersDirectory();
      }
    );

    document.addEventListener('admin:tabchange', (event) => {
      const tab = event.detail?.tab;
      state.active = tab === 'analytics';
      if (state.active) onEnter();
      if (tab === 'users') syncUsersDirectory();
    });
    window.addEventListener('popstate', () => {
      readUrlState();
      setControls();
      setAcquisitionExpanded(state.acquisitionOpen, { writeUrl: false });
      syncDisclosureControls();
      if (state.directory.active) syncUsersDirectory();
      if (
        new URL(window.location.href).searchParams.get('adminTab')
        === 'analytics'
      ) refreshPrimary();
    });
  }

  async function refresh() {
    if (!state.active) return;
    await refreshPrimary();
    const expanded = [
      ...document.querySelectorAll(
        '[data-admin-value-disclosure][aria-expanded="true"]'
      ),
    ];
    await Promise.allSettled(
      expanded.map((button) => {
        return loadSection(
          button.dataset.adminValueDisclosure,
          { keepStaleData: true }
        );
      })
    );
  }

  function onEnter() {
    if (!state.initialized) {
      state.initialized = true;
      readUrlState();
      setControls();
      bindEvents();
      setAcquisitionExpanded(state.acquisitionOpen, { writeUrl: false });
      syncDisclosureControls();
    }
    const tab =
      new URL(window.location.href).searchParams.get('adminTab')
      || 'analytics';
    state.active = tab === 'analytics';
    if (!state.active) {
      if (tab === 'users') syncUsersDirectory();
      return;
    }
    const profileRequested = /^\d+$/.test(
      new URL(window.location.href).searchParams.get(URL_KEYS.profile) || ''
    );
    if (
      !profileRequested
      && (
        !state.sections.acquisition.loaded
        || !state.sections.overview.loaded
        || !state.sections.users.loaded
      )
    ) refreshPrimary();
  }

  function syncAuth(user) {
    if (user?.role === 'admin') return;
    state.active = false;
    state.requestSeq += 1;
    state.directory.active = false;
    Object.values(state.sections).forEach((section) => {
      section.loaded = false;
      section.data = null;
      section.error = null;
      section.stale = false;
    });
  }

  function getRange() {
    return state.range
      ? { ...state.range, dateRange: state.analyticsRange }
      : null;
  }

  window.AdminAnalyticsValue = {
    onEnter,
    refresh,
    syncAuth,
    getRange,
    openAnalyticsUsers,
    closeUsersDirectory,
  };

  document.addEventListener('DOMContentLoaded', () => {
    if (document.documentElement.dataset.navPage === 'admin') onEnter();
  });
})();
