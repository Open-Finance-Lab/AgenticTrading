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
  });

  const state = {
    initialized: false,
    active: false,
    requestSeq: 0,
    range: null,
    includeInternal: false,
    userFilters: {
      lifecycle: '',
      operational: '',
      commercial: '',
      query: '',
      profile: '',
    },
    sections: {
      lifecycle: { loaded: false, data: null, error: null, stale: false },
      users: { loaded: false, data: null, error: null, stale: false },
      retention: { loaded: false, data: null, error: null, stale: false },
      commercial: { loaded: false, data: null, error: null, stale: false },
      operational: { loaded: false, data: null, error: null, stale: false },
    },
    movementChart: null,
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
    return window.API.request(`${API_BASE}${path}`, { method: 'GET' });
  }

  function defaultUtcRange(now = new Date()) {
    const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    const start = new Date(end);
    start.setUTCDate(start.getUTCDate() - 55);
    return {
      start: start.toISOString().slice(0, 10),
      end: end.toISOString().slice(0, 10),
    };
  }

  function validDate(value) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(String(value || ''))) return false;
    const parsed = new Date(`${value}T00:00:00Z`);
    return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
  }

  function readUrlState() {
    const params = new URLSearchParams(window.location.search);
    const defaults = defaultUtcRange();
    const start = params.get('analyticsStart');
    const end = params.get('analyticsEnd');
    state.range = {
      start: validDate(start) ? start : defaults.start,
      end: validDate(end) ? end : defaults.end,
    };
    state.includeInternal = params.get('analyticsInternal') === 'true';
    const lifecycle = params.get(URL_KEYS.lifecycle) || '';
    const operational = params.get(URL_KEYS.operational) || '';
    const commercial = params.get(URL_KEYS.commercial) || '';
    state.userFilters.lifecycle = LIFECYCLE_SEGMENTS.includes(lifecycle) ? lifecycle : '';
    state.userFilters.operational = Object.hasOwn(OPERATIONAL_LABELS, operational) ? operational : '';
    state.userFilters.commercial = Object.hasOwn(COMMERCIAL_LABELS, commercial) ? commercial : '';
    state.userFilters.query = params.get('analyticsUserQuery') || '';
    state.userFilters.profile = params.get(URL_KEYS.profile) || params.get(URL_KEYS.user) || '';
  }

  function setOrDelete(params, key, value) {
    if (value) params.set(key, value);
    else params.delete(key);
  }

  function writeUrlState() {
    if (!window.history?.replaceState) return;
    const url = new URL(window.location.href);
    url.searchParams.set('analyticsStart', state.range.start);
    url.searchParams.set('analyticsEnd', state.range.end);
    setOrDelete(url.searchParams, 'analyticsInternal', state.includeInternal ? 'true' : '');
    setOrDelete(url.searchParams, URL_KEYS.lifecycle, state.userFilters.lifecycle);
    setOrDelete(url.searchParams, URL_KEYS.operational, state.userFilters.operational);
    setOrDelete(url.searchParams, URL_KEYS.commercial, state.userFilters.commercial);
    setOrDelete(url.searchParams, 'analyticsUserQuery', state.userFilters.query);
    setOrDelete(url.searchParams, URL_KEYS.profile, state.userFilters.profile);
    window.history.replaceState(window.history.state, '', url);
  }

  function setControls() {
    element('adminValueStart').value = state.range.start;
    element('adminValueEnd').value = state.range.end;
    element('adminValueInternal').checked = state.includeInternal;
    element('adminPriorityQuery').value = state.userFilters.query;
    element('adminPriorityLifecycle').value = state.userFilters.lifecycle;
    element('adminPriorityOperational').value = state.userFilters.operational;
    element('adminPriorityCommercial').value = state.userFilters.commercial;
  }

  function rangeParams() {
    return new URLSearchParams({
      from: state.range.start,
      to: state.range.end,
      include_internal: state.includeInternal ? 'true' : 'false',
    });
  }

  function userParams() {
    const params = new URLSearchParams({
      priority: 'true',
      include_internal: state.includeInternal ? 'true' : 'false',
      limit: '25',
      offset: '0',
    });
    if (state.userFilters.query) params.set('q', state.userFilters.query);
    if (state.userFilters.lifecycle) params.set('lifecycle_segment', state.userFilters.lifecycle);
    if (state.userFilters.operational) params.set('operational_state', state.userFilters.operational);
    if (state.userFilters.commercial) params.set('commercial_tier', state.userFilters.commercial);
    return params;
  }

  async function handleAccessLost(error) {
    if (error?.status !== 401 && error?.status !== 403) return false;
    if (typeof window.refreshAuthUser === 'function') await window.refreshAuthUser();
    if (typeof window.navigateToPage === 'function') window.navigateToPage('home');
    return true;
  }

  function number(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed.toLocaleString() : '—';
  }

  function percent(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? `${Math.round(parsed * 100)}%` : 'Not mature';
  }

  function credits(value) {
    if (window.CreditFormat?.formatCreditsMicro) {
      return window.CreditFormat.formatCreditsMicro(value);
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? `${(parsed / 1000000).toFixed(6)} Credits` : '—';
  }

  function dollarsFromMicro(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? `$${(parsed / 1000000).toFixed(2)}` : '—';
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
      ? `${history.coverage_start} – ${history.coverage_end}`
      : 'Current snapshot';
  }

  function replaceHiddenMovementRows(series) {
    const body = element('adminLifecycleMovementTable')?.querySelector('tbody');
    clear(body);
    series.forEach((week) => {
      const row = document.createElement('tr');
      row.appendChild(node('th', '', week.week_start));
      row.firstChild.scope = 'row';
      LIFECYCLE_SEGMENTS.forEach((segment) => {
        row.appendChild(node('td', '', number(week.segment_counts?.[segment] || 0)));
      });
      body.appendChild(row);
    });
  }

  function renderLifecycleMovement(series) {
    const rows = Array.isArray(series) ? series : [];
    replaceHiddenMovementRows(rows);
    const quality = element('adminLifecycleQuality');
    quality.hidden = !rows.some((week) => week.data_quality === 'partial');
    const canvas = element('adminLifecycleMovementChart');
    canvas.setAttribute(
      'aria-label',
      rows.length ? `Lifecycle movement across ${rows.length} weekly snapshots` : 'No lifecycle movement data available'
    );
    if (state.movementChart) {
      state.movementChart.destroy();
      state.movementChart = null;
    }
    if (!rows.length || typeof window.Chart !== 'function') return;
    state.movementChart = new window.Chart(canvas, {
      type: 'line',
      data: {
        labels: rows.map((week) => week.week_start),
        datasets: LIFECYCLE_SEGMENTS.map((segment) => ({
          label: LIFECYCLE_LABELS[segment],
          data: rows.map((week) => Number(week.segment_counts?.[segment] || 0)),
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
    renderLifecycleMovement(payload.weekly_segments);
    const incomplete = availabilityIncomplete(payload.availability);
    element('adminValuePrimaryStatus').textContent = incomplete ? 'Incomplete data · available sections remain current.' : '';
  }

  function badge(kind, value, labels) {
    return node('span', `admin-value-badge is-${kind}-${value}`, labels[value] || String(value || 'Unknown'));
  }

  function renderUsers(payload) {
    const target = element('adminPriorityUsers');
    clear(target);
    const items = Array.isArray(payload?.items) ? payload.items : [];
    items.forEach((user) => {
      const row = node('article', 'admin-priority-user');
      const identity = node('div', 'admin-priority-identity');
      identity.appendChild(node('strong', '', user.display_name || user.email || `User #${user.user_id}`));
      identity.appendChild(node('span', '', user.email || `User #${user.user_id}`));
      row.appendChild(identity);
      const signals = node('div', 'admin-priority-signals');
      signals.appendChild(badge('lifecycle', user.lifecycle?.segment, LIFECYCLE_LABELS));
      signals.appendChild(badge('operational', user.operational?.state, OPERATIONAL_LABELS));
      signals.appendChild(badge('commercial', user.commercial_tier, COMMERCIAL_LABELS));
      row.appendChild(signals);
      row.appendChild(node('p', 'admin-priority-reason', user.operational?.state === 'healthy' ? user.lifecycle?.reason : user.operational?.reason));
      row.appendChild(node('span', 'admin-priority-value', credits(user.lifetime_net_purchased_micro)));
      const open = node('button', 'credits-key-action', 'Open profile');
      open.type = 'button';
      open.dataset.userId = String(user.user_id);
      open.addEventListener('click', () => window.AdminAnalytics?.openProfile(user.user_id));
      row.appendChild(open);
      target.appendChild(row);
    });
    if (!items.length) target.appendChild(node('p', 'admin-value-empty', 'No users match these filters.'));
    target.setAttribute('aria-busy', 'false');
    element('adminPriorityUsersRange').textContent = payload?.total
      ? `Showing ${items.length} of ${number(payload.total)}`
      : '0 users';
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
        cohort.cohort_week,
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
    const status = panel.querySelector('[data-admin-value-status]');
    status.textContent = section.stale
      ? 'Showing the last successful response; refresh failed.'
      : availabilityIncomplete(section.data.availability) ? 'Incomplete data' : '';
  }

  function sectionPath(name) {
    const params = rangeParams();
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

  function fetchLifecycle() {
    return request(`${API_ENDPOINTS.lifecycle}?${rangeParams()}`);
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

  function validateRange(start, end) {
    if (!validDate(start) || !validDate(end) || end < start) {
      throw new Error('Choose a valid UTC date range.');
    }
    const days = Math.round((Date.parse(`${end}T00:00:00Z`) - Date.parse(`${start}T00:00:00Z`)) / 86400000) + 1;
    if (days > 180) throw new Error('Choose no more than 180 UTC dates.');
  }

  function bindEvents() {
    element('adminAnalyticsValueFilters')?.addEventListener('submit', (event) => {
      event.preventDefault();
      const error = element('adminValueFilterError');
      try {
        const start = element('adminValueStart').value;
        const end = element('adminValueEnd').value;
        validateRange(start, end);
        state.range = { start, end };
        state.includeInternal = element('adminValueInternal').checked;
        error.hidden = true;
        writeUrlState();
        refresh();
      } catch (validationError) {
        error.textContent = validationError.message;
        error.hidden = false;
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
        const panel = sectionPanel(name);
        const expanded = button.getAttribute('aria-expanded') === 'true';
        button.setAttribute('aria-expanded', expanded ? 'false' : 'true');
        panel.hidden = expanded;
        if (!expanded) ensureDisclosureLoaded(name);
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
    }
    const tab = new URL(window.location.href).searchParams.get('adminTab') || 'analytics';
    state.active = tab === 'analytics';
    if (!state.active) return;
    if (!state.sections.lifecycle.loaded || !state.sections.users.loaded) refreshPrimary();
    if (/^\d+$/.test(state.userFilters.profile)) {
      window.AdminAnalytics?.openProfile(state.userFilters.profile, { focus: false });
    }
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

  window.AdminAnalyticsValue = { onEnter, refresh, syncAuth, applyUserFilters };
  document.addEventListener('DOMContentLoaded', () => {
    if (document.documentElement.dataset.navPage === 'admin') onEnter();
  });
})();
