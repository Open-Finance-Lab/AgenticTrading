/** Read-only Admin Analytics overview and user profile. */
(function () {
  'use strict';

  const API_BASE = (
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ) ? window.location.origin : '';
  const TEMPORARY_UNAVAILABLE = 'This metric is temporarily unavailable.';
  const PROFILE_UNAVAILABLE = 'User analytics are temporarily unavailable.';
  const SECTION_UNAVAILABLE = 'This section is temporarily unavailable.';
  const MORE_UNAVAILABLE = 'More activity is temporarily unavailable.';
  const PROFILE_SECTIONS = ['overview', 'timeline', 'runs', 'usage', 'sessions'];
  const USER_STATES = ['blocked', 'needs_attention', 'dormant', 'onboarding', 'active'];
  const USER_SORTS = new Set(['last_activity', 'joined_at', 'recent_runs', 'recent_failures']);
  const BILLING_MODES = new Set(['all', 'byok', 'platform_credits']);
  const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
  const STATE_LABELS = {
    blocked: 'Blocked',
    needs_attention: 'Needs Attention',
    dormant: 'Dormant',
    onboarding: 'Onboarding',
    active: 'Active',
  };
  const EVENT_LABELS = {
    account_signed_up: 'Account signed up',
    credential_verified: 'Credential verified',
    agent_created: 'Agent created',
    backtest_requested: 'Backtest requested',
    backtest_started: 'Backtest started',
    backtest_completed: 'Backtest completed',
    backtest_failed: 'Backtest failed',
    backtest_cancelled: 'Backtest cancelled',
    model_usage_recorded: 'Model usage recorded',
    credits_reserved: 'ATL Credits reserved',
    credits_settled: 'ATL Credits debited',
    credits_refunded: 'ATL Credits refunded',
    page_viewed: 'Product page viewed',
    session: 'Product session',
  };

  const state = {
    initialized: false,
    active: false,
    overviewLoaded: false,
    refreshing: false,
    overviewRequestSeq: 0,
    usersRequestSeq: 0,
    userRequestSeq: 0,
    filters: null,
    attention: {
      items: [],
      total: 0,
      limit: 25,
      offset: 0,
      query: '',
      status: 'all',
      sort: 'recent_failures',
    },
    profile: { userId: null, detail: null, section: 'overview', sections: {} },
    trendChart: null,
  };

  function element(id) {
    return document.getElementById(id);
  }

  function request(path) {
    if (!window.API || typeof window.API.request !== 'function') {
      return Promise.reject(new Error('Admin Analytics API is not ready yet.'));
    }
    return window.API.request(`${API_BASE}${path}`, { method: 'GET' });
  }

  function clearChildren(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function textNode(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    return node;
  }

  function appendDefinition(list, label, value) {
    const wrapper = document.createElement('div');
    wrapper.appendChild(textNode('dt', '', label));
    wrapper.appendChild(textNode('dd', '', value));
    list.appendChild(wrapper);
  }

  function defaultDateRange(now = new Date()) {
    const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - 29);
    return {
      start: start.toISOString().slice(0, 10),
      end: end.toISOString().slice(0, 10),
    };
  }

  function validDate(value) {
    if (!DATE_PATTERN.test(String(value || ''))) return false;
    const parsed = new Date(`${value}T00:00:00Z`);
    return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
  }

  function readUrlFilters() {
    const params = new URLSearchParams(window.location.search);
    const defaults = defaultDateRange();
    const start = params.get('analyticsStart');
    const end = params.get('analyticsEnd');
    const billing = params.get('analyticsBilling');
    return {
      start: validDate(start) ? start : defaults.start,
      end: validDate(end) ? end : defaults.end,
      billingMode: BILLING_MODES.has(billing) ? billing : 'all',
      provider: String(params.get('analyticsProvider') || '').trim(),
      model: String(params.get('analyticsModel') || '').trim(),
      includeInternal: params.get('analyticsInternal') === 'true',
    };
  }

  function setFilterControls(filters) {
    element('adminAnalyticsStart').value = filters.start;
    element('adminAnalyticsEnd').value = filters.end;
    element('adminAnalyticsBilling').value = filters.billingMode;
    element('adminAnalyticsProvider').value = filters.provider;
    element('adminAnalyticsModel').value = filters.model;
    element('adminAnalyticsInternal').checked = filters.includeInternal;
  }

  function readFilterControls() {
    const filters = {
      start: element('adminAnalyticsStart').value,
      end: element('adminAnalyticsEnd').value,
      billingMode: element('adminAnalyticsBilling').value,
      provider: element('adminAnalyticsProvider').value.trim(),
      model: element('adminAnalyticsModel').value.trim(),
      includeInternal: element('adminAnalyticsInternal').checked,
    };
    if (!validDate(filters.start) || !validDate(filters.end)) {
      throw new Error('Choose a valid start and end date.');
    }
    if (filters.start > filters.end) {
      throw new Error('Start date cannot be after end date.');
    }
    return filters;
  }

  function setOptionalParam(url, key, value) {
    if (value == null || value === '') url.searchParams.delete(key);
    else url.searchParams.set(key, String(value));
  }

  function replaceAnalyticsUrl({ userId = state.profile.userId, section = state.profile.section } = {}) {
    if (!state.filters) return;
    const url = new URL(window.location.href);
    url.searchParams.set('adminTab', 'analytics');
    url.searchParams.set('analyticsStart', state.filters.start);
    url.searchParams.set('analyticsEnd', state.filters.end);
    url.searchParams.set('analyticsBilling', state.filters.billingMode);
    setOptionalParam(url, 'analyticsProvider', state.filters.provider);
    setOptionalParam(url, 'analyticsModel', state.filters.model);
    if (state.filters.includeInternal) url.searchParams.set('analyticsInternal', 'true');
    else url.searchParams.delete('analyticsInternal');
    setOptionalParam(url, 'analyticsUser', userId);
    if (userId) url.searchParams.set('analyticsSection', section || 'overview');
    else url.searchParams.delete('analyticsSection');
    window.history.replaceState(window.history.state, '', url);
  }

  async function handleAccessLost(error) {
    if (error?.status !== 401 && error?.status !== 403) return false;
    if (typeof window.refreshAuthUser === 'function') await window.refreshAuthUser();
    if (typeof window.navigateToPage === 'function') window.navigateToPage('home');
    return true;
  }

  function setPanelBusy(name, busy) {
    document.querySelector(`[data-analytics-panel="${name}"]`)?.setAttribute(
      'aria-busy', busy ? 'true' : 'false'
    );
  }

  function setPanelError(name, message) {
    const panel = document.querySelector(`[data-analytics-panel="${name}"]`);
    const error = panel?.querySelector('[data-panel-error]');
    if (!error) return;
    error.textContent = message || '';
    error.hidden = !message;
  }

  function availability(payload, name) {
    return payload?.availability?.[name]?.available !== false;
  }

  function numberOrDash(value) {
    if (value == null || value === '') return '—';
    return Number.isFinite(Number(value)) ? new Intl.NumberFormat().format(Number(value)) : '—';
  }

  function formatPercent(value) {
    if (value == null || value === '') return '—';
    const numeric = Number(value);
    if (!Number.isFinite(numeric) || numeric < 0 || numeric > 1) return '—';
    return new Intl.NumberFormat(undefined, { style: 'percent', maximumFractionDigits: 1 }).format(numeric);
  }

  function formatMoney(value) {
    if (value == null || value === '') return '—';
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '—';
    return new Intl.NumberFormat(undefined, { style: 'currency', currency: 'USD' }).format(numeric);
  }

  function formatCreditsMicro(value) {
    const formatted = window.CreditFormat.formatCreditsMicro(value);
    return formatted === '—' ? '—' : `${formatted} Credits`;
  }

  function formatTimestamp(value, fallback = '—') {
    if (!value) return fallback;
    const date = new Date(value);
    return Number.isFinite(date.getTime()) ? date.toLocaleString() : fallback;
  }

  function makeTime(value, fallback = '—') {
    const time = document.createElement('time');
    if (value) time.dateTime = String(value);
    time.textContent = formatTimestamp(value, fallback);
    return time;
  }

  function humanizeIdentifier(value) {
    const key = String(value || 'unknown');
    return key.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function eventLabel(value) {
    return EVENT_LABELS[value] || humanizeIdentifier(value);
  }

  function renderSnapshot(payload) {
    const available = availability(payload, 'snapshot');
    setPanelError('snapshot', available ? '' : TEMPORARY_UNAVAILABLE);
    const values = {
      'active-users': payload.active_users_7d,
      'first-success': formatPercent(payload.first_success_conversion),
      'success-rate': formatPercent(payload.backtest_success_rate),
      'platform-cost': formatMoney(payload.platform_model_cost_usd),
      'completed-runs': payload.completed_runs,
      'failed-runs': payload.failed_runs,
      'repeat-rate': formatPercent(payload.repeat_run_rate),
      'input-tokens': payload.input_tokens,
      'output-tokens': payload.output_tokens,
    };
    Object.entries(values).forEach(([key, value]) => {
      const target = document.querySelector(`[data-analytics-metric="${key}"]`);
      if (!target) return;
      target.textContent = typeof value === 'string' ? value : numberOrDash(value);
    });
  }

  function renderTrendTable(payload, dates) {
    const body = element('adminAnalyticsTrendTable')?.querySelector('tbody');
    clearChildren(body);
    dates.forEach((date) => {
      const row = document.createElement('tr');
      row.appendChild(textNode('th', '', date));
      row.lastChild.scope = 'row';
      row.appendChild(textNode('td', '', numberOrDash(payload.daily_active_users?.[date])));
      row.appendChild(textNode('td', '', numberOrDash(payload.daily_completed_runs?.[date])));
      body.appendChild(row);
    });
  }

  function renderTrend(payload) {
    const growthError = document.querySelector('[data-growth-error]');
    const growthAvailable = availability(payload, 'growth');
    growthError.textContent = growthAvailable ? '' : TEMPORARY_UNAVAILABLE;
    growthError.hidden = growthAvailable;
    const dates = [...new Set([
      ...Object.keys(payload.daily_active_users || {}),
      ...Object.keys(payload.daily_completed_runs || {}),
    ])].sort();
    renderTrendTable(payload, dates);
    const canvas = element('adminAnalyticsTrendChart');
    const fallback = element('adminAnalyticsChartFallback');
    if (state.trendChart) {
      state.trendChart.destroy();
      state.trendChart = null;
    }
    if (!growthAvailable) return;
    if (!window.Chart || !canvas) {
      fallback.textContent = 'Trend chart is unavailable; values are listed in the table.';
      fallback.hidden = false;
      return;
    }
    fallback.hidden = true;
    canvas.setAttribute('aria-label', dates.length
      ? `Daily active users and completed runs from ${dates[0]} to ${dates[dates.length - 1]}`
      : 'Daily active users and completed runs; no rows in this period');
    state.trendChart = new window.Chart(canvas, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          {
            label: 'Active users',
            data: dates.map((date) => Number(payload.daily_active_users?.[date] || 0)),
            borderColor: '#67e8f9',
            backgroundColor: 'rgba(103, 232, 249, 0.12)',
            tension: 0.25,
          },
          {
            label: 'Completed runs',
            data: dates.map((date) => Number(payload.daily_completed_runs?.[date] || 0)),
            borderColor: '#a3e635',
            backgroundColor: 'rgba(163, 230, 53, 0.10)',
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? false : undefined,
        plugins: { legend: { labels: { color: '#cbd5e1' } } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.10)' } },
          y: { beginAtZero: true, ticks: { color: '#94a3b8', precision: 0 }, grid: { color: 'rgba(148, 163, 184, 0.10)' } },
        },
      },
    });
  }

  function renderFunnel(payload) {
    const error = document.querySelector('[data-funnel-error]');
    const available = availability(payload, 'funnel');
    error.textContent = available ? '' : TEMPORARY_UNAVAILABLE;
    error.hidden = available;
    const list = element('adminAnalyticsFunnelList');
    clearChildren(list);
    if (!available) return;
    const entries = Object.entries(payload.activation_funnel || {});
    const baseline = Number(entries[0]?.[1] || 0);
    entries.forEach(([key, count]) => {
      const item = document.createElement('li');
      const rate = baseline > 0 ? Math.max(0, Math.min(1, Number(count) / baseline)) : 0;
      item.style.setProperty('--analytics-progress', `${Math.round(rate * 100)}%`);
      item.appendChild(textNode('span', '', eventLabel(key)));
      item.appendChild(textNode('strong', '', numberOrDash(count)));
      item.appendChild(textNode('small', '', formatPercent(rate)));
      list.appendChild(item);
    });
    if (!entries.length) list.appendChild(textNode('li', 'admin-analytics-empty-row', 'No activation events in this period.'));
  }

  function renderHealth(payload) {
    const available = availability(payload, 'friction');
    setPanelError('health', available ? '' : TEMPORARY_UNAVAILABLE);
    const states = element('adminAnalyticsStateList');
    const failures = element('adminAnalyticsFrictionBody');
    clearChildren(states);
    clearChildren(failures);
    if (!available) return;
    USER_STATES.forEach((status) => {
      const item = document.createElement('li');
      item.appendChild(textNode('span', `admin-analytics-state-badge is-${status}`, STATE_LABELS[status]));
      item.appendChild(textNode('strong', '', numberOrDash(payload.user_state_counts?.[status] || 0)));
      states.appendChild(item);
    });
    const categories = Array.isArray(payload.top_failure_categories) ? payload.top_failure_categories : [];
    categories.forEach((category) => {
      const row = document.createElement('tr');
      row.appendChild(textNode('td', '', humanizeIdentifier(category.error_category)));
      row.appendChild(textNode('td', '', numberOrDash(category.affected_users)));
      failures.appendChild(row);
    });
    if (!categories.length) {
      const row = document.createElement('tr');
      const cell = textNode('td', 'admin-empty', 'No actionable failure categories in this period.');
      cell.colSpan = 2;
      row.appendChild(cell);
      failures.appendChild(row);
    }
  }

  function renderLastUpdated(value) {
    const target = element('adminAnalyticsLastUpdated');
    if (!target) return;
    if (value) target.dateTime = String(value);
    else target.removeAttribute('datetime');
    target.textContent = formatTimestamp(value);
  }

  function markOverviewUnavailable() {
    ['snapshot', 'engagement', 'health'].forEach((name) => {
      setPanelBusy(name, false);
      if (name === 'engagement') {
        const growth = document.querySelector('[data-growth-error]');
        const funnel = document.querySelector('[data-funnel-error]');
        [growth, funnel].forEach((error) => {
          if (!error) return;
          error.textContent = TEMPORARY_UNAVAILABLE;
          error.hidden = false;
        });
      } else setPanelError(name, TEMPORARY_UNAVAILABLE);
    });
  }

  function overviewQuery() {
    const params = new URLSearchParams();
    params.set('from', state.filters.start);
    params.set('to', state.filters.end);
    params.set('include_internal', state.filters.includeInternal ? 'true' : 'false');
    if (state.filters.billingMode !== 'all') params.set('billing_mode', state.filters.billingMode);
    if (state.filters.provider) params.set('provider', state.filters.provider);
    if (state.filters.model) params.set('model', state.filters.model);
    return params;
  }

  async function loadOverview() {
    const requestSeq = ++state.overviewRequestSeq;
    ['snapshot', 'engagement', 'health'].forEach((name) => setPanelBusy(name, true));
    try {
      const payload = await request(`/api/admin/analytics/overview?${overviewQuery()}`);
      if (requestSeq !== state.overviewRequestSeq) return;
      renderSnapshot(payload);
      renderTrend(payload);
      renderFunnel(payload);
      renderHealth(payload);
      renderLastUpdated(payload.last_updated);
      state.overviewLoaded = true;
    } catch (error) {
      if (requestSeq !== state.overviewRequestSeq) return;
      if (await handleAccessLost(error)) return;
      markOverviewUnavailable();
    } finally {
      if (requestSeq === state.overviewRequestSeq) {
        ['snapshot', 'engagement', 'health'].forEach((name) => setPanelBusy(name, false));
      }
    }
  }

  function displayIdentity(user) {
    return String(user?.display_name || user?.email || `User #${user?.user_id ?? 'unknown'}`);
  }

  function appendAccountCell(row, user) {
    const cell = document.createElement('td');
    const account = textNode('div', 'admin-analytics-account', '');
    account.appendChild(textNode('strong', '', displayIdentity(user)));
    account.appendChild(textNode('span', '', String(user.email || '—')));
    account.appendChild(textNode('small', '', `User #${user.user_id}`));
    cell.appendChild(account);
    row.appendChild(cell);
  }

  function renderAttention(payload) {
    const body = element('adminAnalyticsAttentionBody');
    clearChildren(body);
    const items = Array.isArray(payload.items) ? payload.items : [];
    items.forEach((user) => {
      const row = document.createElement('tr');
      appendAccountCell(row, user);
      const stateCell = document.createElement('td');
      stateCell.appendChild(textNode('span', `admin-analytics-state-badge is-${user.status}`, STATE_LABELS[user.status] || humanizeIdentifier(user.status)));
      stateCell.appendChild(textNode('p', 'admin-analytics-reason', String(user.human_readable_reason || '—')));
      row.appendChild(stateCell);
      const activityCell = document.createElement('td');
      activityCell.appendChild(makeTime(user.last_meaningful_activity, 'No activity'));
      row.appendChild(activityCell);
      row.appendChild(textNode('td', 'admin-analytics-number', numberOrDash(user.recent_runs)));
      row.appendChild(textNode('td', 'admin-analytics-number', numberOrDash(user.recent_failures)));
      const actionCell = document.createElement('td');
      const button = textNode('button', 'credits-key-action', 'View analytics');
      button.type = 'button';
      button.dataset.analyticsUserId = String(user.user_id);
      button.setAttribute('aria-label', `View analytics for ${displayIdentity(user)}`);
      actionCell.appendChild(button);
      row.appendChild(actionCell);
      body.appendChild(row);
    });
    if (!items.length) {
      const row = document.createElement('tr');
      const cell = textNode('td', 'admin-empty', 'No users match these filters.');
      cell.colSpan = 6;
      row.appendChild(cell);
      body.appendChild(row);
    }
    state.attention.items = items;
    state.attention.total = Number(payload.total) || 0;
    state.attention.limit = Number(payload.limit) || 25;
    state.attention.offset = Number(payload.offset) || 0;
    const start = state.attention.total ? state.attention.offset + 1 : 0;
    const end = Math.min(state.attention.offset + items.length, state.attention.total);
    element('adminAnalyticsUserRange').textContent = `Showing ${start}–${end} of ${state.attention.total}`;
    element('adminAnalyticsUsersPrev').disabled = state.attention.offset <= 0;
    element('adminAnalyticsUsersNext').disabled = state.attention.offset + state.attention.limit >= state.attention.total;
  }

  function attentionQuery(offset) {
    const params = new URLSearchParams();
    params.set('limit', String(state.attention.limit));
    params.set('offset', String(Math.max(0, offset)));
    params.set('last_activity_from', state.filters.start);
    params.set('last_activity_to', state.filters.end);
    params.set('sort', state.attention.sort);
    params.set('order', 'desc');
    params.set('include_internal', state.filters.includeInternal ? 'true' : 'false');
    if (state.attention.query) params.set('q', state.attention.query);
    if (state.attention.status !== 'all') params.set('status', state.attention.status);
    return params;
  }

  async function loadAttention({ reset = false, offset = state.attention.offset } = {}) {
    if (reset) offset = 0;
    const previousOffset = state.attention.offset;
    const requestSeq = ++state.usersRequestSeq;
    setPanelBusy('attention', true);
    try {
      const payload = await request(`/api/admin/analytics/users?${attentionQuery(offset)}`);
      if (requestSeq !== state.usersRequestSeq) return;
      setPanelError('attention', '');
      renderAttention(payload);
    } catch (error) {
      if (requestSeq !== state.usersRequestSeq) return;
      if (await handleAccessLost(error)) return;
      state.attention.offset = previousOffset;
      setPanelError('attention', offset === 0
        ? TEMPORARY_UNAVAILABLE
        : 'The requested user page is temporarily unavailable.');
    } finally {
      if (requestSeq === state.usersRequestSeq) setPanelBusy('attention', false);
    }
  }

  async function refresh() {
    if (!state.active || !state.filters || state.refreshing) return;
    state.refreshing = true;
    try {
      await Promise.allSettled([loadOverview(), loadAttention({ reset: true })]);
    } finally {
      state.refreshing = false;
    }
  }

  function emptySectionState() {
    return { items: [], nextCursor: null, loading: false, loaded: false, error: null, requestSeq: 0 };
  }

  function resetProfileSections() {
    state.profile.sections = {
      timeline: emptySectionState(),
      runs: emptySectionState(),
      usage: emptySectionState(),
      sessions: emptySectionState(),
    };
    ['timeline', 'runs', 'usage', 'sessions'].forEach((section) => {
      const panel = document.querySelector(`[data-analytics-section-panel="${section}"]`);
      clearChildren(panel?.querySelector('[data-section-items]'));
      const error = panel?.querySelector('[data-section-error]');
      if (error) {
        error.textContent = '';
        error.hidden = true;
      }
      const more = panel?.querySelector('[data-section-more]');
      if (more) more.hidden = true;
    });
  }

  function setProfileText(id, value, fallback = '—') {
    const target = element(id);
    if (target) target.textContent = value == null || value === '' ? fallback : String(value);
  }

  function setProfileTime(id, value, fallback = '—') {
    const target = element(id);
    if (!target) return;
    clearChildren(target);
    target.appendChild(makeTime(value, fallback));
  }

  function renderEvidence(ids) {
    const list = element('adminAnalyticsProfileEvidence')?.querySelector('ul');
    clearChildren(list);
    (Array.isArray(ids) ? ids : []).forEach((id) => list.appendChild(textNode('li', '', String(id))));
    if (!list.children.length) list.appendChild(textNode('li', '', 'No evidence event IDs available.'));
  }

  function renderMilestones(profile) {
    const list = element('adminAnalyticsMilestones');
    clearChildren(list);
    const entries = Object.entries(profile.activation_milestones || {}).sort(
      (left, right) => new Date(left[1]) - new Date(right[1])
    );
    entries.forEach(([name, occurredAt]) => {
      const item = document.createElement('li');
      item.appendChild(textNode('strong', '', eventLabel(name)));
      item.appendChild(makeTime(occurredAt));
      list.appendChild(item);
    });
    if (!entries.length) list.appendChild(textNode('li', 'admin-analytics-empty-row', 'No activation milestones recorded.'));
  }

  function renderRunSummary(profile) {
    const list = element('adminAnalyticsRunSummary');
    clearChildren(list);
    Object.entries(profile.run_summary || {}).forEach(([key, value]) => {
      appendDefinition(list, humanizeIdentifier(key), numberOrDash(value));
    });
  }

  function renderUsageSummary(profile) {
    const list = element('adminAnalyticsUsageSummary');
    clearChildren(list);
    appendDefinition(list, 'Input tokens', numberOrDash(profile.input_tokens));
    appendDefinition(list, 'Output tokens', numberOrDash(profile.output_tokens));
    const total = Number(profile.input_tokens) + Number(profile.output_tokens);
    appendDefinition(list, 'Total tokens', Number.isFinite(total) ? numberOrDash(total) : '—');
    appendDefinition(list, 'ATL platform model cost', formatMoney(profile.platform_model_cost_usd));
    appendDefinition(list, 'ATL Credits debited', formatCreditsMicro(profile.credits_debited_micro));
    appendDefinition(list, 'Top product page', profile.top_product_page ? humanizeIdentifier(profile.top_product_page) : '—');
    const lanes = element('adminAnalyticsBillingMix');
    clearChildren(lanes);
    Object.entries(profile.billing_lane_mix || {}).forEach(([lane, count]) => {
      const label = lane === 'byok' ? 'BYOK usage — no ATL Credits debit' : humanizeIdentifier(lane);
      const item = document.createElement('li');
      item.appendChild(textNode('span', '', label));
      item.appendChild(textNode('strong', '', numberOrDash(count)));
      lanes.appendChild(item);
    });
  }

  function renderFootprint(profile) {
    const list = element('adminAnalyticsRecentFootprint');
    clearChildren(list);
    (profile.recent_footprint || []).forEach((item) => {
      const row = document.createElement('li');
      const head = document.createElement('div');
      head.appendChild(textNode('strong', '', eventLabel(item.event_name)));
      head.appendChild(makeTime(item.occurred_at));
      row.appendChild(head);
      const details = [item.page_view, item.provider_id, item.model_id, item.billing_mode, item.outcome, item.error_category]
        .filter(Boolean).map(humanizeIdentifier).join(' · ');
      if (details) row.appendChild(textNode('p', '', details));
      list.appendChild(row);
    });
    if (!list.children.length) list.appendChild(textNode('li', 'admin-analytics-empty-row', 'No recent footprint events.'));
  }

  function renderProfile(profile) {
    state.profile.detail = profile;
    setProfileText('adminAnalyticsProfileTitle', profile.display_name || profile.email || `User #${profile.user_id}`);
    setProfileText('adminAnalyticsProfileEmail', profile.email);
    setProfileText('adminAnalyticsProfileUserId', profile.user_id);
    setProfileTime('adminAnalyticsProfileJoined', profile.joined_at);
    setProfileTime('adminAnalyticsProfileLastActivity', profile.last_meaningful_activity, 'No activity');
    const stateTarget = element('adminAnalyticsProfileState');
    clearChildren(stateTarget);
    stateTarget.appendChild(textNode('span', `admin-analytics-state-badge is-${profile.state?.status}`, STATE_LABELS[profile.state?.status] || humanizeIdentifier(profile.state?.status)));
    setProfileText('adminAnalyticsProfileReason', profile.state?.human_readable_reason);
    setProfileText('adminAnalyticsProfileBilling', profile.primary_billing_lane ? humanizeIdentifier(profile.primary_billing_lane) : null);
    setProfileText('adminAnalyticsProfileProvider', profile.default_provider);
    setProfileText('adminAnalyticsProfileRegion', profile.country_code, 'Unknown');
    setProfileText('adminAnalyticsProfileDevice', profile.device_category ? humanizeIdentifier(profile.device_category) : null, 'Unknown');
    setProfileText('adminAnalyticsProfileBrowser', profile.browser_family, 'Unknown');
    renderEvidence(profile.state?.evidence_event_ids);
    renderMilestones(profile);
    renderRunSummary(profile);
    renderUsageSummary(profile);
    renderFootprint(profile);
    const overviewPanel = document.querySelector('[data-analytics-section-panel="overview"]');
    const status = overviewPanel?.querySelector('[data-section-status]');
    if (status) status.textContent = '';
  }

  async function loadProfile(userId) {
    const requestSeq = ++state.userRequestSeq;
    try {
      const profile = await request(`/api/admin/analytics/users/${encodeURIComponent(String(userId))}`);
      if (requestSeq !== state.userRequestSeq || String(state.profile.userId) !== String(userId)) return;
      element('adminAnalyticsProfileError').hidden = true;
      renderProfile(profile);
      if (state.profile.section !== 'overview') selectProfileSection(state.profile.section, { focus: false });
    } catch (error) {
      if (requestSeq !== state.userRequestSeq) return;
      if (await handleAccessLost(error)) return;
      const target = element('adminAnalyticsProfileError');
      target.textContent = PROFILE_UNAVAILABLE;
      target.hidden = false;
      const status = document.querySelector('[data-analytics-section-panel="overview"] [data-section-status]');
      if (status) status.textContent = '';
    }
  }

  function validUserId(value) {
    return /^\d+$/.test(String(value || '')) && Number(value) > 0;
  }

  function openProfile(userId, { section = 'overview', focus = true } = {}) {
    if (!validUserId(userId)) return;
    state.userRequestSeq += 1;
    state.profile.userId = String(userId);
    state.profile.detail = null;
    state.profile.section = PROFILE_SECTIONS.includes(section) ? section : 'overview';
    resetProfileSections();
    element('adminAnalyticsOverview').hidden = true;
    element('adminAnalyticsProfile').hidden = false;
    element('adminAnalyticsProfileError').hidden = true;
    setProfileText('adminAnalyticsProfileTitle', 'Loading user analytics');
    setProfileText('adminAnalyticsProfileEmail', '—');
    selectProfileSection(state.profile.section, { focus: false, load: false });
    replaceAnalyticsUrl();
    if (focus) element('adminAnalyticsProfileTitle')?.focus();
    loadProfile(state.profile.userId);
  }

  function closeProfile({ focus = true } = {}) {
    state.userRequestSeq += 1;
    state.profile.userId = null;
    state.profile.detail = null;
    state.profile.section = 'overview';
    resetProfileSections();
    element('adminAnalyticsProfile').hidden = true;
    element('adminAnalyticsOverview').hidden = false;
    replaceAnalyticsUrl({ userId: null, section: 'overview' });
    if (focus) element('adminAnalyticsAttentionHeading')?.focus();
  }

  function sectionPanel(section) {
    return document.querySelector(`[data-analytics-section-panel="${section}"]`);
  }

  function selectProfileSection(value, { focus = false, load = true } = {}) {
    const section = PROFILE_SECTIONS.includes(value) ? value : 'overview';
    state.profile.section = section;
    document.querySelectorAll('[data-analytics-section-tab]').forEach((button) => {
      const selected = button.dataset.analyticsSectionTab === section;
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
      button.classList.toggle('is-active', selected);
      if (selected && focus) button.focus();
    });
    document.querySelectorAll('[data-analytics-section-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.analyticsSectionPanel !== section;
    });
    replaceAnalyticsUrl();
    if (load && section !== 'overview' && !state.profile.sections[section]?.loaded) {
      loadProfileSection(section, { append: false });
    }
  }

  function appendTimeCell(row, value, fallback = '—') {
    const cell = document.createElement('td');
    cell.appendChild(makeTime(value, fallback));
    row.appendChild(cell);
  }

  function makeActivityTable(headers) {
    const wrapper = textNode('div', 'admin-analytics-table-wrap', '');
    const table = textNode('table', 'admin-analytics-activity-table', '');
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    headers.forEach((header) => {
      const cell = textNode('th', '', header);
      cell.scope = 'col';
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    const body = document.createElement('tbody');
    table.appendChild(body);
    wrapper.appendChild(table);
    return { wrapper, body };
  }

  function renderTimeline(items, container) {
    const list = textNode('ol', 'admin-analytics-timeline', '');
    items.forEach((item) => {
      const row = document.createElement('li');
      const head = document.createElement('div');
      head.appendChild(textNode('strong', '', eventLabel(item.event_name)));
      head.appendChild(makeTime(item.occurred_at));
      row.appendChild(head);
      const details = [item.outcome, item.provider_id, item.model_id, item.billing_mode, item.error_category]
        .filter(Boolean).map(humanizeIdentifier).join(' · ');
      row.appendChild(textNode('p', '', details || 'No additional display-safe details.'));
      list.appendChild(row);
    });
    container.appendChild(list);
  }

  function renderRuns(items, container) {
    const { wrapper, body } = makeActivityTable(['Time', 'Run event', 'Outcome', 'Provider / model', 'Billing lane', 'Error category']);
    items.forEach((item) => {
      const row = document.createElement('tr');
      appendTimeCell(row, item.occurred_at);
      row.appendChild(textNode('td', '', eventLabel(item.event_name)));
      row.appendChild(textNode('td', '', item.outcome ? humanizeIdentifier(item.outcome) : '—'));
      row.appendChild(textNode('td', '', [item.provider_id, item.model_id].filter(Boolean).join(' · ') || '—'));
      row.appendChild(textNode('td', '', item.billing_mode ? humanizeIdentifier(item.billing_mode) : '—'));
      row.appendChild(textNode('td', '', item.error_category ? humanizeIdentifier(item.error_category) : '—'));
      body.appendChild(row);
    });
    container.appendChild(wrapper);
  }

  function renderUsage(items, container) {
    const { wrapper, body } = makeActivityTable(['Time', 'Usage event', 'Provider / model', 'Billing lane', 'Input', 'Output', 'ATL cost', 'ATL Credits debited']);
    items.forEach((item) => {
      const row = document.createElement('tr');
      appendTimeCell(row, item.occurred_at);
      row.appendChild(textNode('td', '', eventLabel(item.event_name)));
      row.appendChild(textNode('td', '', [item.provider_id, item.model_id].filter(Boolean).join(' · ') || '—'));
      const byok = item.billing_mode === 'byok';
      row.appendChild(textNode('td', '', byok ? 'BYOK — no ATL charge' : (item.billing_mode ? humanizeIdentifier(item.billing_mode) : '—')));
      row.appendChild(textNode('td', 'admin-analytics-number', numberOrDash(item.input_tokens)));
      row.appendChild(textNode('td', 'admin-analytics-number', numberOrDash(item.output_tokens)));
      row.appendChild(textNode('td', 'admin-analytics-number', byok || item.cost_micro_usd == null ? '—' : formatMoney(Number(item.cost_micro_usd) / 1000000)));
      row.appendChild(textNode('td', 'admin-analytics-number', byok || item.amount_micro == null ? '—' : formatCreditsMicro(item.amount_micro)));
      body.appendChild(row);
    });
    container.appendChild(wrapper);
  }

  function formatVisibleTime(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '—';
    const seconds = Math.round(numeric / 1000);
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  }

  function renderSessions(items, container) {
    const { wrapper, body } = makeActivityTable(['Started', 'Events', 'Visible time', 'Region', 'Device', 'Browser']);
    items.forEach((item) => {
      const row = document.createElement('tr');
      appendTimeCell(row, item.occurred_at);
      row.appendChild(textNode('td', 'admin-analytics-number', numberOrDash(item.session_event_count)));
      row.appendChild(textNode('td', '', formatVisibleTime(item.visible_ms)));
      row.appendChild(textNode('td', '', item.country_code || 'Unknown'));
      row.appendChild(textNode('td', '', item.device_category ? humanizeIdentifier(item.device_category) : 'Unknown'));
      row.appendChild(textNode('td', '', item.browser_family || 'Unknown'));
      body.appendChild(row);
    });
    container.appendChild(wrapper);
  }

  function renderSection(section) {
    const sectionState = state.profile.sections[section];
    const panel = sectionPanel(section);
    const container = panel?.querySelector('[data-section-items]');
    clearChildren(container);
    if (!sectionState.items.length) {
      container.appendChild(textNode('p', 'admin-analytics-empty-row', 'No activity in this section.'));
    } else if (section === 'timeline') renderTimeline(sectionState.items, container);
    else if (section === 'runs') renderRuns(sectionState.items, container);
    else if (section === 'usage') renderUsage(sectionState.items, container);
    else renderSessions(sectionState.items, container);
    const more = panel?.querySelector('[data-section-more]');
    if (more) {
      more.hidden = !sectionState.nextCursor;
      more.disabled = sectionState.loading;
    }
  }

  async function loadProfileSection(section, { append = false } = {}) {
    const sectionState = state.profile.sections[section];
    const userId = state.profile.userId;
    if (!sectionState || !userId || sectionState.loading) return;
    if (append && !sectionState.nextCursor) return;
    sectionState.loading = true;
    sectionState.error = null;
    const requestSeq = ++sectionState.requestSeq;
    const panel = sectionPanel(section);
    const status = panel?.querySelector('[data-section-status]');
    const errorTarget = panel?.querySelector('[data-section-error]');
    if (status) status.textContent = append ? 'Loading more activity…' : 'Loading activity…';
    if (errorTarget) errorTarget.hidden = true;
    const params = new URLSearchParams({ section, limit: '50' });
    if (append) params.set('cursor', sectionState.nextCursor);
    try {
      const payload = await request(`/api/admin/analytics/users/${encodeURIComponent(String(userId))}/activity?${params}`);
      if (String(state.profile.userId) !== String(userId) || requestSeq !== sectionState.requestSeq) return;
      const nextItems = Array.isArray(payload.items) ? payload.items : [];
      sectionState.items = append ? sectionState.items.concat(nextItems) : nextItems;
      sectionState.nextCursor = payload.next_cursor || null;
      sectionState.loaded = true;
      renderSection(section);
    } catch (error) {
      if (requestSeq !== sectionState.requestSeq) return;
      if (await handleAccessLost(error)) return;
      sectionState.error = append ? MORE_UNAVAILABLE : SECTION_UNAVAILABLE;
      if (errorTarget) {
        errorTarget.textContent = sectionState.error;
        errorTarget.hidden = false;
      }
    } finally {
      if (requestSeq === sectionState.requestSeq) {
        sectionState.loading = false;
        if (status) status.textContent = '';
        const more = panel?.querySelector('[data-section-more]');
        if (more) more.disabled = false;
      }
    }
  }

  function restoreAnalyticsUrlState() {
    state.filters = readUrlFilters();
    setFilterControls(state.filters);
    const params = new URLSearchParams(window.location.search);
    const userId = params.get('analyticsUser');
    const section = params.get('analyticsSection') || 'overview';
    if (validUserId(userId)) openProfile(userId, { section, focus: false });
    else if (state.profile.userId) closeProfile({ focus: false });
  }

  function bindEvents() {
    element('adminAnalyticsFilters')?.addEventListener('submit', (event) => {
      event.preventDefault();
      const error = element('adminAnalyticsFilterError');
      try {
        state.filters = readFilterControls();
        error.hidden = true;
        state.attention.offset = 0;
        replaceAnalyticsUrl();
        refresh();
      } catch (validationError) {
        error.textContent = validationError.message;
        error.hidden = false;
      }
    });
    element('adminAnalyticsRefreshBtn')?.addEventListener('click', refresh);
    element('adminAnalyticsUserFilters')?.addEventListener('submit', (event) => {
      event.preventDefault();
      state.attention.query = element('adminAnalyticsUserQuery').value.trim();
      state.attention.status = element('adminAnalyticsUserStatus').value;
      const sort = element('adminAnalyticsUserSort').value;
      state.attention.sort = USER_SORTS.has(sort) ? sort : 'recent_failures';
      loadAttention({ reset: true });
    });
    element('adminAnalyticsUsersPrev')?.addEventListener('click', () => {
      loadAttention({ offset: Math.max(0, state.attention.offset - state.attention.limit) });
    });
    element('adminAnalyticsUsersNext')?.addEventListener('click', () => {
      loadAttention({ offset: state.attention.offset + state.attention.limit });
    });
    element('adminAnalyticsAttentionBody')?.addEventListener('click', (event) => {
      const button = event.target.closest('[data-analytics-user-id]');
      if (button) openProfile(button.dataset.analyticsUserId);
    });
    element('adminAnalyticsProfileBack')?.addEventListener('click', () => closeProfile());
    document.querySelectorAll('[data-analytics-section-tab]').forEach((button) => {
      button.addEventListener('click', () => selectProfileSection(button.dataset.analyticsSectionTab));
      button.addEventListener('keydown', (event) => {
        const keys = new Set(['ArrowRight', 'ArrowLeft', 'Home', 'End']);
        if (!keys.has(event.key)) return;
        event.preventDefault();
        const index = PROFILE_SECTIONS.indexOf(button.dataset.analyticsSectionTab);
        const section = event.key === 'Home'
          ? PROFILE_SECTIONS[0]
          : event.key === 'End'
            ? PROFILE_SECTIONS[PROFILE_SECTIONS.length - 1]
            : event.key === 'ArrowRight'
              ? PROFILE_SECTIONS[(index + 1) % PROFILE_SECTIONS.length]
              : PROFILE_SECTIONS[(index - 1 + PROFILE_SECTIONS.length) % PROFILE_SECTIONS.length];
        selectProfileSection(section, { focus: true });
      });
    });
    document.querySelectorAll('[data-section-more]').forEach((button) => {
      button.addEventListener('click', () => {
        const section = button.closest('[data-analytics-section-panel]')?.dataset.analyticsSectionPanel;
        if (section) loadProfileSection(section, { append: true });
      });
    });
    element('adminAnalyticsOpenAccount')?.addEventListener('click', (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const user = state.profile.detail;
      if (!user) return;
      closeProfile({ focus: false });
      window.AdminTabs?.openAccountManagement({ userId: user.user_id, email: user.email });
    });
    document.addEventListener('admin:tabchange', (event) => {
      state.active = event.detail?.tab === 'analytics';
      if (!state.active) return;
      restoreAnalyticsUrlState();
      if (!state.profile.userId) refresh();
    });
    window.addEventListener('popstate', () => {
      const params = new URLSearchParams(window.location.search);
      state.active = params.get('adminTab') === 'analytics';
      if (state.active) restoreAnalyticsUrlState();
    });
  }

  function onEnter() {
    if (!state.initialized) {
      state.initialized = true;
      state.filters = readUrlFilters();
      setFilterControls(state.filters);
      resetProfileSections();
      bindEvents();
    }
    const requestedTab = new URLSearchParams(window.location.search).get('adminTab') || 'analytics';
    state.active = requestedTab === 'analytics';
    if (!state.active) return;
    restoreAnalyticsUrlState();
    if (!state.profile.userId) refresh();
  }

  function syncAuth(user) {
    if (user?.role === 'admin') return;
    state.active = false;
    state.overviewLoaded = false;
    state.refreshing = false;
    state.overviewRequestSeq += 1;
    state.usersRequestSeq += 1;
    state.userRequestSeq += 1;
    state.attention.items = [];
    state.profile.userId = null;
    state.profile.detail = null;
    resetProfileSections();
    if (state.trendChart) {
      state.trendChart.destroy();
      state.trendChart = null;
    }
  }

  window.AdminAnalytics = { onEnter, refresh, syncAuth };
  document.addEventListener('DOMContentLoaded', () => {
    if (document.documentElement.dataset.navPage === 'admin') onEnter();
  });
})();
