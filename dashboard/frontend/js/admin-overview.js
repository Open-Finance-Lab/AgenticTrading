/** /admin overview: nine panels and five detail routes from the analytics endpoints (design §7.2, §8.2). */
(function () {
  'use strict';

  const ENDPOINTS = Object.freeze({
    overview: '/api/admin/analytics/overview',
    lifecycle: '/api/admin/analytics/lifecycle',
    retention: '/api/admin/analytics/retention',
    commercial: '/api/admin/analytics/commercial',
    operational: '/api/admin/analytics/operational',
    groups: '/api/admin/analytics/groups',
  });
  const ALL = Object.keys(ENDPOINTS);
  const FUNNEL_LABELS = Object.freeze({
    account_signed_up: 'Account created',
    credential_verified: 'Credential verified',
    agent_created: 'Agent created',
    backtest_requested: 'Backtest attempted',
    backtest_completed: 'First successful result',
  });
  const GROUP_CLASSES = Object.freeze({
    internal: '', invited: 'source-green', organic: 'source-violet',
    competition: 'source-amber', partner: 'source-red', unknown: 'source-steel',
  });
  const GROUP_COLORS = Object.freeze({
    internal: 'var(--accent)', invited: 'var(--green)', organic: 'var(--violet)',
    competition: 'var(--amber)', partner: 'var(--red)', unknown: 'var(--steel)',
  });
  const SEGMENT_ORDER = ['new', 'onboarding', 'growing', 'core', 'at_risk', 'dormant'];
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const DETAIL_NEEDS = Object.freeze({
    sources: ['groups'], retention: ['retention'], credits: ['commercial'],
    lifecycle: ['lifecycle'], health: ['operational'],
  });

  const state = { data: {}, errors: {}, signature: '' };

  function shell() {
    return window.AdminShell;
  }

  function sum(rows, key) {
    return (rows || []).reduce((total, row) => total + (Number(row?.[key]) || 0), 0);
  }

  function ratio(part, whole) {
    return whole ? (Number(part) || 0) / whole : null;
  }

  function emptyBody(text) {
    const body = shell().el('div');
    body.appendChild(shell().el('p', 'panel-empty', text));
    return body;
  }

  // ---------------------------------------------------------------- panels

  function renderAttention(operational, lifecycle) {
    const s = shell();
    const counts = operational?.operational_state_counts || {};
    const blocked = Number(counts.blocked) || 0;
    const needs = Number(counts.needs_attention) || 0;
    const atRisk = Number(lifecycle?.segment_counts?.at_risk) || 0;
    const body = s.el('div');
    const list = s.el('div', 'attention-list');
    [
      ['severity', 'Blocked', blocked, 'A current issue prevents a core action'],
      ['severity warn', 'Needs attention', needs, 'A supported issue needs operator review'],
      ['severity muted', 'At risk', atRisk, 'Inactive, not yet dormant'],
    ].forEach(([tone, label, count, detail]) => {
      const row = s.el('div', 'attention-row');
      row.appendChild(s.el('i', tone));
      row.appendChild(s.el('span', '', label));
      row.appendChild(s.el('b', '', s.formatNumber(count)));
      row.appendChild(s.el('small', '', detail));
      list.appendChild(row);
    });
    body.appendChild(list);
    const reasons = Array.isArray(operational?.top_operational_reasons) ? operational.top_operational_reasons : [];
    const note = s.el('div', 'attention-note');
    const text = s.el('div');
    text.appendChild(s.el('span', '', 'Blocking signal'));
    // Absent (not served until PR D) is not empty: the note says which.
    text.appendChild(s.el('strong', '', s.fieldPending(operational, 'top_operational_reasons') ? s.PENDING : reasons.length
      ? `${s.humanize(reasons[0].reason_code)} · ${s.formatNumber(reasons[0].users)} users`
      : 'No blocking signal recorded'));
    note.appendChild(text);
    const link = s.el('a', '', 'View reason details');
    link.setAttribute('href', '#health');
    note.appendChild(link);
    body.appendChild(note);
    return { headline: s.formatNumber(blocked + needs + atRisk), body };
  }

  function renderActiveUsers(overview) {
    const s = shell();
    const series = Object.entries(overview?.daily_active_users || {}).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
    const headline = s.formatNumber(overview?.active_users_7d);
    if (!series.length) return { headline, body: emptyBody('No daily activity recorded for this range.') };
    const body = s.el('div');
    const chart = s.el('div', 'bar-chart with-axis chart-with-axis');
    chart.setAttribute('role', 'img');
    const max = Math.max(1, ...series.map(([, value]) => Number(value) || 0));
    const axis = s.el('div', 'chart-y-axis');
    [max, Math.round(max * 2 / 3), Math.round(max / 3), 0].forEach((tick) => axis.appendChild(s.el('span', '', s.formatNumber(tick))));
    chart.appendChild(axis);
    series.forEach(([day, value]) => {
      const column = s.el('div', 'bar-column');
      column.appendChild(s.el('b', '', s.formatNumber(value)));
      const bar = s.el('i');
      bar.style.setProperty('--height', `${Math.max(4, (Number(value) || 0) * 76 / max)}%`);
      column.appendChild(bar);
      column.appendChild(s.el('span', '', s.formatShortDay(day)));
      chart.appendChild(column);
    });
    chart.setAttribute('aria-label', `Active users ${series.map(([day, value]) => `${s.formatShortDay(day)}: ${s.formatNumber(value)}`).join(', ')}`);
    body.appendChild(chart);
    return { headline, body };
  }

  function renderActivation(overview) {
    const s = shell();
    const stages = Object.entries(overview?.activation_funnel || {});
    const headline = s.formatPercent(overview?.first_success_conversion);
    if (!stages.length) return { headline, body: emptyBody('No activation events recorded for this range.') };
    const body = s.el('div');
    const route = s.el('div', 'layered-route');
    route.setAttribute('role', 'img');
    const first = Number(stages[0][1]) || 0;
    const depths = [72, 58, 46, 37, 30];
    stages.forEach(([key, value], index) => {
      const layer = s.el('div', 'layer');
      layer.style.setProperty('--depth', `${depths[Math.min(index, depths.length - 1)]}%`);
      layer.appendChild(s.el('span', 'layer-name', FUNNEL_LABELS[key] || s.humanize(key)));
      layer.appendChild(s.el('strong', 'layer-value', s.formatNumber(value)));
      const meta = s.el('small', 'layer-meta');
      if (index === 0) {
        meta.textContent = '100% of cohort';
      } else {
        meta.appendChild(s.el('strong', '', s.formatPercent(ratio(value, first))));
        meta.append(' remain');
      }
      layer.appendChild(meta);
      route.appendChild(layer);
    });
    route.setAttribute('aria-label', `Activation progress: ${stages.map(([key, value]) => `${s.formatNumber(value)} ${(FUNNEL_LABELS[key] || s.humanize(key)).toLowerCase()}`).join(', ')}`);
    body.appendChild(route);
    return { headline, body };
  }

  function renderSources(groups) {
    const s = shell();
    const rows = Array.isArray(groups?.groups) ? groups.groups : [];
    const total = sum(rows, 'users');
    const headline = s.formatNumber(total);
    if (!rows.length) return { headline, body: emptyBody('No account sources recorded.') };
    const body = s.el('div');
    const wrap = s.el('div', 'source-wrap');
    const donut = s.el('div', 'source-donut');
    donut.setAttribute('role', 'img');
    donut.setAttribute('data-total', `${s.formatNumber(total)}\nusers`);
    let cursor = 0;
    const stops = rows.map((row) => {
      const share = total ? (Number(row.users) || 0) / total * 100 : 0;
      const start = cursor;
      cursor += share;
      return `${GROUP_COLORS[row.group] || 'var(--steel)'} ${start.toFixed(2)}% ${cursor.toFixed(2)}%`;
    });
    donut.style.background = total ? `conic-gradient(${stops.join(',')})` : 'var(--line)';
    donut.setAttribute('aria-label', `User sources: ${rows.map((row) => `${row.label || s.humanize(row.group)} ${s.formatPercent(ratio(row.users, total))}`).join(', ')}`);
    wrap.appendChild(donut);
    const legend = s.el('div', 'source-legend-six');
    rows.forEach((row) => {
      const item = s.el('div', 'legend-row');
      item.appendChild(s.el('i', GROUP_CLASSES[row.group] ?? 'source-steel'));
      item.appendChild(s.el('span', '', row.label || s.humanize(row.group)));
      item.appendChild(s.el('b', '', s.formatPercent(ratio(row.users, total))));
      legend.appendChild(item);
    });
    wrap.appendChild(legend);
    body.appendChild(wrap);
    return { headline, body };
  }

  function renderRetention(retention) {
    const s = shell();
    const cohorts = Array.isArray(retention?.cohorts) ? retention.cohorts : [];
    const headline = s.formatPercent(retention?.summary_week_1?.rate);
    if (!cohorts.length) return { headline, body: emptyBody('No activation cohorts in this range.') };
    const body = s.el('div');
    const chart = s.el('div', 'retention-chart');
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', 'Retention by activation week');
    ['', 'W0', 'W1', 'W2', 'W4'].forEach((label) => chart.appendChild(s.el('span', '', label)));
    cohorts.forEach((cohort) => {
      chart.appendChild(s.el('span', 'row-label', s.formatShortDay(cohort.cohort_week)));
      const activated = s.el('div', 'retention-cell', s.formatNumber(cohort.activated_users));
      activated.style.setProperty('--alpha', '.55');
      chart.appendChild(activated);
      [cohort.week_1, cohort.week_2, cohort.week_4].forEach((cell) => {
        const mature = Boolean(cell?.mature) && cell.rate != null;
        const node = s.el('div', mature ? 'retention-cell' : 'retention-cell empty', mature ? s.formatPercent(cell.rate) : s.DASH);
        if (mature) node.style.setProperty('--alpha', (Number(cell.rate) * 0.55).toFixed(2));
        if (cell?.data_quality === 'partial') node.setAttribute('title', s.INCOMPLETE);
        chart.appendChild(node);
      });
    });
    body.appendChild(chart);
    return { headline, body };
  }

  function renderValue(groups) {
    const s = shell();
    const rows = Array.isArray(groups?.groups) ? groups.groups : [];
    const users = sum(rows, 'users');
    const successful = sum(rows, 'successful_run_users');
    const repeat = sum(rows, 'repeat_users');
    const headline = s.formatNumber(successful);
    if (!rows.length) return { headline, body: emptyBody('No users recorded.') };
    const steps = [
      ['Not yet successful', Math.max(0, users - successful)],
      ['Successful once', Math.max(0, successful - repeat)],
      ['Repeat users', repeat],
    ];
    const body = s.el('div');
    const list = s.el('div', 'value-steps');
    list.setAttribute('role', 'img');
    list.setAttribute('aria-label', `Value progression: ${steps.map(([label, count]) => `${s.formatNumber(count)} ${label.toLowerCase()}`).join(', ')}`);
    steps.forEach(([label, count]) => {
      const step = s.el('div', 'value-step');
      step.appendChild(s.el('span', '', label));
      const track = s.el('div', 'value-track');
      const fill = s.el('i');
      fill.style.setProperty('--share', s.formatPercent(ratio(count, users)) === s.DASH ? '0%' : s.formatPercent(ratio(count, users)));
      track.appendChild(fill);
      step.appendChild(track);
      step.appendChild(s.el('b', '', `${s.formatNumber(count)} · ${s.formatPercent(ratio(count, users))}`));
      list.appendChild(step);
    });
    body.appendChild(list);
    return { headline, body };
  }

  function renderLifecycle(lifecycle) {
    const s = shell();
    const counts = lifecycle?.segment_counts || {};
    const total = SEGMENT_ORDER.reduce((acc, key) => acc + (Number(counts[key]) || 0), 0);
    const headline = s.formatNumber(total);
    if (!lifecycle?.segment_counts) return { headline, body: emptyBody('No lifecycle distribution recorded.') };
    const body = s.el('div');
    const bar = s.el('div', 'lifecycle-bar');
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', `Lifecycle distribution: ${SEGMENT_ORDER.map((key) => `${s.LIFECYCLE_LABELS[key]} ${s.formatNumber(counts[key] || 0)}`).join(', ')}`);
    SEGMENT_ORDER.forEach((key) => {
      const segment = s.el('i');
      segment.style.setProperty('--share', String(total ? (Number(counts[key]) || 0) : 1));
      bar.appendChild(segment);
    });
    body.appendChild(bar);
    const legend = s.el('div', 'lifecycle-legend');
    SEGMENT_ORDER.forEach((key) => {
      const item = s.el('div');
      item.appendChild(s.el('i'));
      item.appendChild(s.el('span', '', s.LIFECYCLE_LABELS[key]));
      item.appendChild(s.el('b', '', s.formatNumber(counts[key] || 0)));
      legend.appendChild(item);
    });
    body.appendChild(legend);
    return { headline, body };
  }

  function renderCredits(commercial, overview) {
    const s = shell();
    const headline = s.formatCredits(commercial?.selected_period?.consumed_micro);
    const days = Array.isArray(overview?.billing_lane_mix) ? overview.billing_lane_mix : [];
    if (s.fieldPending(overview, 'billing_lane_mix')) return { headline, body: emptyBody(s.PENDING) };
    if (!days.length) return { headline, body: emptyBody('No run activity by billing lane in this range.') };
    const body = s.el('div');
    const chart = s.el('div', 'credits-paired');
    chart.setAttribute('role', 'img');
    chart.setAttribute('aria-label', `Platform Credits and BYOK runs by date for ${s.state.range}`);
    chart.style.gridTemplateColumns = `repeat(${days.length},minmax(0,1fr))`;
    chart.style.height = days.length > 8 ? '184px' : '164px';
    const axis = s.el('div', 'credit-axis');
    axis.appendChild(s.el('span', '', 'Higher usage'));
    axis.appendChild(s.el('span', '', `Selected range · ${s.state.range}`));
    chart.appendChild(axis);
    const max = Math.max(1, ...days.flatMap((day) => [Number(day.platform_credits) || 0, Number(day.byok) || 0]));
    days.forEach((day) => {
      const column = s.el('div', 'credit-day');
      [[day.platform_credits, 'credit-stem', 12], [day.byok, 'credit-stem violet', 10]].forEach(([value, className, floor]) => {
        const stem = s.el('span', className);
        stem.style.height = `${Math.max(floor, (Number(value) || 0) / max * 124)}px`;
        stem.appendChild(s.el('b', '', s.formatNumber(value)));
        column.appendChild(stem);
      });
      column.appendChild(s.el('span', 'credit-date', s.formatShortDay(day.day)));
      chart.appendChild(column);
    });
    const legend = s.el('div', 'credit-legend');
    [['Platform Credits', ''], ['BYOK runs', 'violet']].forEach(([label, className]) => {
      const item = s.el('span');
      item.appendChild(s.el('i', className));
      item.append(label);
      legend.appendChild(item);
    });
    chart.appendChild(legend);
    body.appendChild(chart);
    return { headline, body };
  }

  function svgNode(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text != null) node.textContent = String(text);
    return node;
  }

  function axisLabel(value) {
    return value >= 100 ? String(Math.round(value)) : value.toFixed(1).replace(/\.0$/, '');
  }

  function renderRevenue(commercial) {
    const s = shell();
    const headline = s.formatCredits(commercial?.selected_period?.purchased_micro);
    const series = Array.isArray(commercial?.purchased_by_day) ? commercial.purchased_by_day : [];
    if (s.fieldPending(commercial, 'purchased_by_day')) return { headline, body: emptyBody(s.PENDING) };
    if (!series.length) return { headline, body: emptyBody('No settled purchases in this range.') };
    const values = series.map((row) => (Number(row.amount_micro) || 0) / 1000000);
    const labels = series.map((row) => s.formatShortDay(row.day));
    const width = 520, height = 180, left = 42, right = 508, top = 19, bottom = 141;
    const plotWidth = right - left, plotHeight = bottom - top;
    const max = Math.max(1, ...values);
    const points = values.map((value, index) => ({
      x: left + (values.length === 1 ? 0 : index / (values.length - 1) * plotWidth),
      y: bottom - (value / max) * plotHeight,
      value,
      label: labels[index],
    }));
    const svg = svgNode('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `Purchased Credits revenue trend with ${values.length} data points` });
    [0, 0.5, 1].forEach((fraction) => svg.appendChild(svgNode('line', { class: 'revenue-grid', x1: left, y1: bottom - fraction * plotHeight, x2: right, y2: bottom - fraction * plotHeight })));
    svg.appendChild(svgNode('line', { class: 'revenue-axis', x1: left, y1: top, x2: left, y2: bottom }));
    svg.appendChild(svgNode('line', { class: 'revenue-axis', x1: left, y1: bottom, x2: right, y2: bottom }));
    [max, max / 2, 0].forEach((value, index) => svg.appendChild(svgNode('text', { class: 'revenue-axis-label', x: 5, y: [top + 3, (top + bottom) / 2 + 3, bottom + 3][index] }, axisLabel(value))));
    const path = points.map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(' ');
    svg.appendChild(svgNode('path', { class: 'revenue-area', d: `${path} L ${right} ${bottom} L ${left} ${bottom} Z` }));
    svg.appendChild(svgNode('path', { class: 'revenue-line', d: path }));
    points.forEach((point) => {
      const circle = svgNode('circle', { class: 'revenue-point', cx: point.x, cy: point.y, r: 4, tabindex: '0' });
      circle.appendChild(svgNode('title', {}, `${point.label}: ${point.value.toFixed(1)} purchased Credits`));
      svg.appendChild(circle);
      svg.appendChild(svgNode('text', { class: 'revenue-point-label', x: point.x, y: Math.max(top + 11, point.y - 9) }, axisLabel(point.value)));
    });
    const shown = [0, Math.floor((labels.length - 1) / 2), labels.length - 1].filter((index, position, list) => list.indexOf(index) === position);
    shown.forEach((index) => svg.appendChild(svgNode('text', { class: 'revenue-x-label', x: points[index].x, y: 164, 'text-anchor': index === 0 ? 'start' : index === labels.length - 1 ? 'end' : 'middle' }, labels[index])));
    const body = s.el('div');
    const wrap = s.el('div', 'revenue-viz');
    wrap.appendChild(svg);
    body.appendChild(wrap);
    return { headline, body };
  }

  // --------------------------------------------------------- detail views

  function table(headers, rows, emptyText) {
    const s = shell();
    const wrap = s.el('div', 'table-wrap');
    const node = s.el('table');
    const head = s.el('thead');
    const headRow = s.el('tr');
    headers.forEach((label) => {
      const cell = s.el('th', '', label);
      cell.setAttribute('scope', 'col');
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    node.appendChild(head);
    const body = s.el('tbody');
    rows.forEach((cells) => {
      const row = s.el('tr');
      cells.forEach((cell) => row.appendChild(typeof cell === 'string' ? s.el('td', '', cell) : (() => { const td = s.el('td'); td.appendChild(cell); return td; })()));
      body.appendChild(row);
    });
    if (!rows.length) {
      const row = s.el('tr');
      const cell = s.el('td', 'panel-empty', emptyText);
      cell.setAttribute('colspan', String(headers.length));
      row.appendChild(cell);
      body.appendChild(row);
    }
    node.appendChild(body);
    wrap.appendChild(node);
    return wrap;
  }

  function detailShell(title, description, metrics) {
    const s = shell();
    const root = s.el('div');
    const crumb = s.el('nav', 'breadcrumb');
    crumb.setAttribute('aria-label', 'Breadcrumb');
    const parent = s.el('a', '', 'Analytics');
    parent.setAttribute('href', '#overview');
    crumb.appendChild(parent);
    crumb.appendChild(s.el('span', '', '/'));
    crumb.appendChild(s.el('span', '', title));
    root.appendChild(crumb);
    const head = s.el('div', 'page-head');
    const text = s.el('div');
    const heading = s.el('h1', '', title);
    heading.setAttribute('tabindex', '-1');
    text.appendChild(heading);
    text.appendChild(s.el('p', 'muted', description));
    head.appendChild(text);
    head.appendChild(s.el('small', 'muted', `Selected range · ${s.state.range} · UTC`));
    root.appendChild(head);
    const strip = s.el('div', 'metric-strip');
    metrics.forEach(([label, value]) => {
      const metric = s.el('div', 'detail-metric');
      metric.appendChild(s.el('span', '', label));
      metric.appendChild(s.el('strong', '', value));
      strip.appendChild(metric);
    });
    root.appendChild(strip);
    return root;
  }

  function section(root, title, meta, content) {
    const s = shell();
    const node = s.el('section', 'detail-section');
    const head = s.el('div', 'section-head');
    const text = s.el('div');
    text.appendChild(s.el('h2', '', title));
    if (meta) text.appendChild(s.el('small', '', meta));
    head.appendChild(text);
    node.appendChild(head);
    node.appendChild(content);
    root.appendChild(node);
    return node;
  }

  function detailSources(groups) {
    const s = shell();
    const rows = Array.isArray(groups?.groups) ? groups.groups : [];
    const total = sum(rows, 'users');
    const successful = sum(rows, 'successful_run_users');
    const largest = rows.slice().sort((a, b) => (Number(b.users) || 0) - (Number(a.users) || 0))[0];
    const root = detailShell('User sources', 'Compare the six account-source groups and see where users progress toward a first successful result.', [
      ['Identified users', s.formatNumber(total)],
      ['Largest source', largest ? `${largest.label || s.humanize(largest.group)} · ${s.formatPercent(ratio(largest.users, total))}` : s.DASH],
      ['Users with a result', s.formatNumber(successful)],
      ['Activation', s.formatPercent(ratio(successful, total))],
    ]);
    section(root, 'Source groups', 'Users, results and cost by account source', table(
      ['Source', 'Users', 'Share', 'Successful run users', 'Repeat users', 'Total runs', 'ATL cost', 'Paid users', 'Activation'],
      rows.map((row) => [
        row.label || s.humanize(row.group), s.formatNumber(row.users), s.formatPercent(ratio(row.users, total)),
        s.formatNumber(row.successful_run_users), s.formatNumber(row.repeat_users), s.formatNumber(row.total_runs),
        s.usdFromMicro(row.atl_cost_micro_usd), s.formatNumber(row.paid_users), s.formatPercent(ratio(row.successful_run_users, row.users)),
      ]),
      'No account sources recorded.'
    ));
    return root;
  }

  function retentionCell(cell) {
    const s = shell();
    if (!cell?.mature || cell.rate == null) return 'Not mature';
    return `${s.formatNumber(cell.retained_users)} / ${s.formatNumber(cell.eligible_users)} · ${s.formatPercent(cell.rate)}`;
  }

  function detailRetention(retention) {
    const s = shell();
    const summary = (cell) => (cell?.mature ? s.formatPercent(cell.rate) : 'Not mature');
    const root = detailShell('Retention', 'See whether users return after reaching their first successful result.', [
      ['W1 return', summary(retention?.summary_week_1)],
      ['W2 return', summary(retention?.summary_week_2)],
      ['W4 return', summary(retention?.summary_week_4)],
      ['Eligible users', s.formatNumber(retention?.summary_week_1?.eligible_users)],
    ]);
    section(root, 'Activation cohorts', 'UTC Monday-to-Sunday week of the first successful backtest', table(
      ['Activation week', 'Activated', 'W1', 'W2', 'W4'],
      (retention?.cohorts || []).map((cohort) => [
        s.formatDateOnly(cohort.cohort_week), s.formatNumber(cohort.activated_users),
        retentionCell(cohort.week_1), retentionCell(cohort.week_2), retentionCell(cohort.week_4),
      ]),
      'No activation cohorts in this range.'
    ));
    return root;
  }

  function detailCredits(commercial) {
    const s = shell();
    const period = commercial?.selected_period || {};
    const balances = commercial?.current_balances || {};
    const tiers = commercial?.tier_counts || {};
    const root = detailShell('Credits & revenue', 'Review platform Credits usage and settled purchases together.', [
      ['ATL Credits settled', s.formatCredits(period.consumed_micro)],
      ['Purchased Credits', s.formatCredits(period.purchased_micro)],
      ['Refunds', s.formatCredits(period.refunded_micro)],
      ['Admin Grants', s.formatCredits(period.admin_grant_activity_micro)],
    ]);
    section(root, 'Selected period', 'Ledger movement in the selected range', table(
      ['Measure', 'Value', 'Scope'],
      [
        ['ATL Credits settled', s.formatCredits(period.consumed_micro), 'Selected period'],
        ['Purchased Credits', s.formatCredits(period.purchased_micro), 'Selected period'],
        ['Refunds', s.formatCredits(period.refunded_micro), 'Selected period'],
        ['Admin Grants', s.formatCredits(period.admin_grant_activity_micro), 'Excluded from revenue'],
        ['Platform model cost', s.usdFromMicro(period.platform_model_cost_micro_usd), 'Platform Credits lane'],
        ['Lifetime net purchased', s.formatCredits(commercial?.lifetime_net_purchased_micro), 'Lifetime'],
      ],
      'No ledger activity.'
    ));
    section(root, 'Commercial tiers', 'Lifetime net purchase per user', table(
      ['Tier', 'Users'],
      Object.keys(s.COMMERCIAL_LABELS).map((tier) => [s.COMMERCIAL_LABELS[tier], s.formatNumber(tiers[tier] || 0)]),
      'No users recorded.'
    ));
    section(root, 'Current balances', 'Spendable Credits right now', table(
      ['Balance', 'Value', 'Note'],
      [
        ['Grant balance', s.formatCredits(balances.grant_available_micro), 'Not revenue'],
        ['Purchased balance', s.formatCredits(balances.purchased_available_micro), 'Customer-funded'],
        ['Total available', s.formatCredits(balances.total_available_micro), 'Current spendable balance'],
      ],
      'No balances recorded.'
    ));
    return root;
  }

  function detailLifecycle(lifecycle) {
    const s = shell();
    const counts = lifecycle?.segment_counts || {};
    const root = detailShell('User lifecycle', 'See current user maturity and inactivity without mixing it with operational blockers.', [
      ['Growing', s.formatNumber(counts.growing || 0)],
      ['Core', s.formatNumber(counts.core || 0)],
      ['At risk', s.formatNumber(counts.at_risk || 0)],
      ['Dormant', s.formatNumber(counts.dormant || 0)],
    ]);
    section(root, 'Segments', 'Rules from the lifecycle definition', table(
      ['Stage', 'Users', 'Rule'],
      SEGMENT_ORDER.map((key) => [s.LIFECYCLE_LABELS[key], s.formatNumber(counts[key] || 0), s.LIFECYCLE_RULES[key]]),
      'No segments recorded.'
    ));
    section(root, 'Recent movement', 'Segment transitions in the selected range', table(
      ['From', 'To', 'Users', 'Period'],
      (lifecycle?.transitions || []).map((transition) => [
        s.LIFECYCLE_LABELS[transition.from_segment] || s.humanize(transition.from_segment),
        s.LIFECYCLE_LABELS[transition.to_segment] || s.humanize(transition.to_segment),
        s.formatNumber(transition.users),
        `${s.formatDateOnly(transition.period_start)} – ${s.formatDateOnly(transition.period_end)}${transition.data_quality === 'partial' ? ` · ${s.INCOMPLETE}` : ''}`,
      ]),
      'No lifecycle transitions in this range.'
    ));
    return root;
  }

  function detailHealth(operational) {
    const s = shell();
    const counts = operational?.operational_state_counts || {};
    const root = detailShell('System health', 'Track reliability and the failure reasons that prevent users from receiving results.', [
      ['Failed runs', s.formatNumber(operational?.failed_runs)],
      ['Success rate', s.formatPercent(operational?.backtest_success_rate)],
      ['Completed runs', s.formatNumber(operational?.completed_runs)],
      ['Blocked users', s.formatNumber(counts.blocked || 0)],
    ]);
    // Rollups are anonymous (design D13, §8.2): the failure table carries only
    // a category and its count, with no per-user breakdown column.
    section(root, 'Failure categories', 'Display-safe categories from the daily rollups', table(
      ['Failure category', 'Count'],
      (operational?.top_failure_categories || []).map((failure) => [s.humanize(failure.error_category), s.formatNumber(failure.affected_users)]),
      'No failure categories in this range.'
    ));
    section(root, 'What is blocking users?', 'Operational reasons as of yesterday UTC', table(
      ['Reason', 'State', 'Users'],
      (operational?.top_operational_reasons || []).map((reason) => [s.humanize(reason.reason_code), s.OPERATIONAL_LABELS[reason.state] || s.humanize(reason.state), s.formatNumber(reason.users)]),
      s.fieldPending(operational, 'top_operational_reasons') ? s.PENDING : 'No blocking reasons recorded.'
    ));
    return root;
  }

  // ------------------------------------------------------------ loading

  const PANELS = [
    { id: 'panelAttention', name: 'attention', needs: ['operational', 'lifecycle'], render: (d) => renderAttention(d.operational, d.lifecycle) },
    { id: 'panelActiveUsers', name: 'active-users', needs: ['overview'], render: (d) => renderActiveUsers(d.overview) },
    { id: 'panelActivation', name: 'activation', needs: ['overview'], render: (d) => renderActivation(d.overview) },
    { id: 'panelSources', name: 'sources', needs: ['groups'], render: (d) => renderSources(d.groups) },
    { id: 'panelRetention', name: 'retention', needs: ['retention'], render: (d) => renderRetention(d.retention) },
    { id: 'panelValue', name: 'value', needs: ['groups'], render: (d) => renderValue(d.groups) },
    { id: 'panelLifecycle', name: 'lifecycle', needs: ['lifecycle'], render: (d) => renderLifecycle(d.lifecycle) },
    { id: 'panelCredits', name: 'credits', needs: ['commercial', 'overview'], render: (d) => renderCredits(d.commercial, d.overview) },
    { id: 'panelRevenue', name: 'revenue', needs: ['commercial'], render: (d) => renderRevenue(d.commercial) },
  ];

  function pathFor(name) {
    const s = shell();
    return `${ENDPOINTS[name]}?${s.analyticsParams({ withGroup: name === 'groups' })}`;
  }

  function paint(def) {
    const s = shell();
    const panel = document.getElementById(def.id);
    if (!panel) return;
    const missing = def.needs.filter((name) => !state.data[name]);
    const failed = def.needs.some((name) => state.errors[name]);
    if (missing.length) {
      const headline = panel.querySelector('[data-headline]');
      if (headline) headline.textContent = s.DASH;
      const body = panel.querySelector('[data-body]');
      if (body) s.clear(body);
      s.setPanelState(panel, { busy: false, error: s.SECTION_UNAVAILABLE });
      return;
    }
    const result = def.render(state.data);
    const headline = panel.querySelector('[data-headline]');
    if (headline) headline.textContent = result.headline;
    const body = panel.querySelector('[data-body]');
    if (body) {
      s.clear(body);
      body.appendChild(result.body);
    }
    const incomplete = def.needs.some((name) => s.availabilityIncomplete(state.data[name]?.availability));
    s.setPanelState(panel, { busy: false, status: incomplete ? s.INCOMPLETE : '', error: failed ? s.SECTION_UNAVAILABLE : '', stale: failed });
  }

  async function loadAll(names) {
    const s = shell();
    const seq = s.nextSeq('overview');
    PANELS.forEach((def) => s.setPanelState(document.getElementById(def.id), { busy: true }));
    const results = await Promise.allSettled(names.map((name) => s.request(pathFor(name))));
    if (!s.isCurrent('overview', seq)) return;
    for (const [index, name] of names.entries()) {
      const result = results[index];
      if (result.status === 'fulfilled') {
        state.data[name] = result.value;
        state.errors[name] = null;
      } else {
        if (await s.handleAccessLost(result.reason)) return;
        state.errors[name] = s.SECTION_UNAVAILABLE;
      }
    }
    PANELS.forEach((def) => paint(def));
  }

  const DETAILS = Object.freeze({
    sources: (d) => detailSources(d.groups),
    retention: (d) => detailRetention(d.retention),
    credits: (d) => detailCredits(d.commercial),
    lifecycle: (d) => detailLifecycle(d.lifecycle),
    health: (d) => detailHealth(d.operational),
  });

  async function showDetail(route) {
    const s = shell();
    const target = document.getElementById('detail');
    if (!target || !DETAILS[route]) return;
    const seq = s.nextSeq('detail');
    const needs = DETAIL_NEEDS[route];
    const missing = needs.filter((name) => !state.data[name]);
    if (missing.length) {
      const results = await Promise.allSettled(missing.map((name) => s.request(pathFor(name))));
      if (!s.isCurrent('detail', seq)) return;
      for (const [index, name] of missing.entries()) {
        const result = results[index];
        if (result.status === 'fulfilled') { state.data[name] = result.value; state.errors[name] = null; }
        else {
          if (await s.handleAccessLost(result.reason)) return;
          state.errors[name] = s.SECTION_UNAVAILABLE;
        }
      }
    }
    s.clear(target);
    if (needs.some((name) => !state.data[name])) {
      target.appendChild(s.el('p', 'panel-error', s.SECTION_UNAVAILABLE));
      return;
    }
    target.appendChild(DETAILS[route](state.data));
    if (needs.some((name) => s.availabilityIncomplete(state.data[name]?.availability))) {
      target.appendChild(s.el('p', 'panel-status muted', s.INCOMPLETE));
    }
    target.querySelector('h1')?.focus?.({ preventScroll: true });
  }

  function signatureOf(detail) {
    return JSON.stringify([detail.range, detail.filters.group, detail.filters.internal]);
  }

  document.addEventListener('admin:route', (event) => {
    const detail = event.detail || {};
    const signature = signatureOf(detail);
    if (signature !== state.signature) {
      state.signature = signature;
      state.data = {};
      state.errors = {};
    }
    if (detail.route === 'overview') loadAll(ALL);
    else if (DETAILS[detail.route]) showDetail(detail.route);
  });
  document.addEventListener('admin:retry', (event) => {
    const def = PANELS.find((panel) => panel.name === event.detail?.panel);
    if (def) loadAll(def.needs);
  });

  window.AdminOverview = {
    PANELS, DETAIL_NEEDS, state,
    renderAttention, renderActiveUsers, renderActivation, renderSources, renderRetention,
    renderValue, renderLifecycle, renderCredits, renderRevenue,
    detailSources, detailRetention, detailCredits, detailLifecycle, detailHealth,
    paint, loadAll, showDetail,
  };
})();
