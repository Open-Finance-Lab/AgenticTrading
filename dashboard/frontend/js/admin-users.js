/** /admin users: the SQL-filtered list and the lazy, cursor-paged profile (design §7.2, D5, D11, D15). */
(function () {
  'use strict';

  const USERS_PATH = '/api/admin/analytics/users';
  const PAGE_SIZE = 50;
  const SECTIONS = ['overview', 'timeline', 'runs', 'usage', 'sessions'];
  const SECTION_LABELS = Object.freeze({ overview: 'Overview', timeline: 'Timeline', runs: 'Runs', usage: 'Usage', sessions: 'Sessions' });
  // Harvested from the retired admin-analytics.js:35-50 (deleted in PR C; see git history).
  const EVENT_LABELS = Object.freeze({
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
  });
  const BADGE_TONE = Object.freeze({
    lifecycle: { at_risk: 'warn', dormant: 'bad', core: 'good' },
    operational: { blocked: 'bad', needs_attention: 'warn', healthy: 'good' },
    commercial: { high_value: 'good', invested: 'good' },
  });

  function emptySection() {
    return { items: [], nextCursor: null, loading: false, loaded: false, error: null, requestSeq: 0 };
  }

  const state = {
    list: { offset: 0, total: 0, items: [], loaded: false },
    profile: { userId: null, detail: null, section: 'overview', sections: {} },
  };

  function shell() {
    return window.AdminShell;
  }

  function eventLabel(value) {
    return EVENT_LABELS[value] || shell().humanize(value);
  }

  function labelFor(kind, value) {
    const s = shell();
    const labels = kind === 'lifecycle' ? s.LIFECYCLE_LABELS : kind === 'operational' ? s.OPERATIONAL_LABELS : s.COMMERCIAL_LABELS;
    return labels[value] || s.humanize(value);
  }

  // D11: the badge string is the server's `group_badge`; nothing here recomputes it.
  function groupBadge(badge) {
    const s = shell();
    if (badge == null) {
      // Interim contract: `group_badge` is absent until PR D. A dash, never a
      // guessed group — the server owns the precedence rule (D11).
      const pending = s.el('span', 'group-badge is-pending', s.DASH);
      pending.setAttribute('title', s.PENDING);
      return pending;
    }
    const text = String(badge);
    return s.el('span', `group-badge is-${text}`, text);
  }

  function signalBadge(kind, value) {
    const s = shell();
    const tone = BADGE_TONE[kind]?.[value];
    const node = s.el('span', tone ? `badge ${tone}` : 'badge', labelFor(kind, value));
    const rules = kind === 'lifecycle' ? s.LIFECYCLE_RULES : kind === 'operational' ? s.OPERATIONAL_RULES : null;
    if (rules?.[value]) {
      node.setAttribute('title', rules[value]);
      node.setAttribute('aria-label', `${labelFor(kind, value)}: ${rules[value]}`);
    }
    return node;
  }

  function formatVisibleTime(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return shell().DASH;
    const seconds = Math.round(numeric / 1000);
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  }

  function td(content, className) {
    const s = shell();
    const cell = s.el('td', className);
    if (typeof content === 'string') cell.textContent = content;
    else cell.appendChild(content);
    return cell;
  }

  // ------------------------------------------------------------ list

  function renderUserRows(payload) {
    const s = shell();
    const fragment = document.createDocumentFragment();
    const items = Array.isArray(payload?.items) ? payload.items : [];
    if (!items.length) {
      const row = s.el('tr');
      const cell = s.el('td', 'panel-empty', 'No users match these filters.');
      cell.setAttribute('colspan', '6');
      row.appendChild(cell);
      fragment.appendChild(row);
      return fragment;
    }
    items.forEach((user) => {
      const row = s.el('tr');
      const who = s.el('div', 'who');
      const link = s.el('a', '', user.display_name || user.email || `User #${user.user_id}`);
      link.setAttribute('href', `#users/${encodeURIComponent(String(user.user_id))}`);
      who.appendChild(link);
      who.appendChild(s.el('small', '', user.email || ''));
      row.appendChild(td(who));
      row.appendChild(td(groupBadge(user.group_badge)));
      row.appendChild(td(signalBadge('lifecycle', user.lifecycle?.segment)));
      row.appendChild(td(signalBadge('operational', user.operational?.state)));
      row.appendChild(td(s.fieldPending(user, 'last_meaningful_activity_at') ? s.DASH : s.formatTimestamp(user.last_meaningful_activity_at, 'No activity')));
      const button = s.el('button', 'row-action', 'Review evidence');
      button.setAttribute('type', 'button');
      button.setAttribute('aria-haspopup', 'dialog');
      button.addEventListener('click', () => openEvidence(user, button));
      row.appendChild(td(button));
      fragment.appendChild(row);
    });
    return fragment;
  }

  function renderPager(payload) {
    const s = shell();
    const total = Number(payload?.total) || 0;
    const offset = Number(payload?.offset) || 0;
    const shown = Array.isArray(payload?.items) ? payload.items.length : 0;
    if (!total) return '0 users';
    return `Showing ${s.formatNumber(offset + 1)}–${s.formatNumber(offset + shown)} of ${s.formatNumber(total)}`;
  }

  async function loadList({ offset = 0 } = {}) {
    const s = shell();
    const view = document.getElementById('usersView');
    const body = document.getElementById('usersBody');
    const seq = s.nextSeq('users');
    s.setPanelState(view, { busy: true });
    try {
      const payload = await s.request(`${USERS_PATH}?${s.userListParams({ offset, limit: PAGE_SIZE })}`);
      if (!s.isCurrent('users', seq)) return;
      state.list = { offset, total: Number(payload.total) || 0, items: payload.items || [], loaded: true };
      if (body) {
        s.clear(body);
        body.appendChild(renderUserRows(payload));
      }
      const range = document.getElementById('usersRange');
      if (range) range.textContent = renderPager(payload);
      const prev = document.getElementById('usersPrev');
      const next = document.getElementById('usersNext');
      if (prev) prev.disabled = offset <= 0;
      if (next) next.disabled = offset + PAGE_SIZE >= state.list.total;
      s.setPanelState(view, { busy: false });
    } catch (error) {
      if (!s.isCurrent('users', seq)) return;
      if (await s.handleAccessLost(error)) return;
      s.setPanelState(view, { busy: false, error: s.SECTION_UNAVAILABLE, stale: state.list.loaded });
    }
  }

  // ------------------------------------------------------- evidence

  function evidenceList(values, fallback) {
    const s = shell();
    const list = s.el('ul', 'evidence-list');
    (values || []).forEach((fact) => list.appendChild(s.el('li', '', fact)));
    if (!list.children.length) list.appendChild(s.el('li', 'muted', fallback));
    return list;
  }

  function renderEvidence(user) {
    const s = shell();
    const root = s.el('div');
    root.appendChild(s.el('p', 'muted', user.email || `User #${user.user_id}`));
    const signals = s.el('div', 'signals');
    signals.appendChild(signalBadge('lifecycle', user.lifecycle?.segment));
    signals.appendChild(signalBadge('operational', user.operational?.state));
    signals.appendChild(signalBadge('commercial', user.commercial_tier ?? user.commercial?.commercial_tier));
    root.appendChild(signals);
    const lifecycle = s.el('section');
    lifecycle.appendChild(s.el('h3', '', 'Lifecycle evidence'));
    lifecycle.appendChild(s.el('p', '', user.lifecycle?.reason || 'No lifecycle reason is available.'));
    lifecycle.appendChild(evidenceList(user.lifecycle?.evidence, 'No lifecycle evidence is available.'));
    root.appendChild(lifecycle);
    const operational = s.el('section');
    operational.appendChild(s.el('h3', '', 'Operational evidence'));
    operational.appendChild(s.el('p', '', user.operational?.reason || 'No operational reason is available.'));
    operational.appendChild(evidenceList(user.operational?.evidence, 'No operational evidence is available.'));
    root.appendChild(operational);
    return root;
  }

  function accountManagementHref(user) {
    const query = user?.email || (user?.user_id != null ? String(user.user_id) : '');
    // Account management lives on this page since D5; the ?user= hash query is
    // the hand-off that used to ride to /app as adminUserQuery.
    return `#account${query ? `?user=${encodeURIComponent(query)}` : ''}`;
  }

  function openEvidence(user, opener) {
    const s = shell();
    const body = document.getElementById('evidenceBody');
    if (body) {
      s.clear(body);
      body.appendChild(renderEvidence(user));
    }
    const title = document.getElementById('evidenceTitle');
    if (title) title.textContent = user.display_name || user.email || `User #${user.user_id}`;
    document.getElementById('evidenceProfile')?.setAttribute('href', `#users/${encodeURIComponent(String(user.user_id))}`);
    document.getElementById('evidenceAccount')?.setAttribute('href', accountManagementHref(user));
    s.openDialog(document.getElementById('evidenceDialog'), opener);
  }

  // -------------------------------------------------------- profile

  function activationWeekLabel(activatedAt) {
    const s = shell();
    if (!activatedAt) return 'Not yet activated';
    const date = new Date(activatedAt);
    if (!Number.isFinite(date.getTime())) return 'Not yet activated';
    const monday = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
    monday.setUTCDate(monday.getUTCDate() - ((monday.getUTCDay() + 6) % 7));
    return `Activation week of ${s.formatDateOnly(monday.toISOString().slice(0, 10))}`;
  }

  function renderProfileHeader(profile) {
    const s = shell();
    const root = s.el('div');
    const crumb = s.el('nav', 'breadcrumb');
    crumb.setAttribute('aria-label', 'Breadcrumb');
    const parent = s.el('a', '', 'Users');
    parent.setAttribute('href', '#users');
    crumb.appendChild(parent);
    crumb.appendChild(s.el('span', '', '/'));
    crumb.appendChild(s.el('span', '', profile.display_name || profile.email || `User #${profile.user_id}`));
    root.appendChild(crumb);

    const head = s.el('div', 'page-head profile-head');
    const identity = s.el('div', 'identity');
    identity.appendChild(s.el('div', 'avatar', String(profile.display_name || profile.email || '?').trim().charAt(0).toUpperCase()));
    const text = s.el('div');
    const heading = s.el('h1', '', profile.display_name || profile.email || `User #${profile.user_id}`);
    heading.setAttribute('tabindex', '-1');
    text.appendChild(heading);
    const line = s.el('p', 'muted');
    line.append(`${profile.email || ''} · `);
    line.appendChild(groupBadge(profile.group_badge));
    line.append(` · ${activationWeekLabel(profile.lifecycle?.activated_at)}`);
    text.appendChild(line);
    identity.appendChild(text);
    head.appendChild(identity);
    const back = s.el('a', 'module-link', 'Back to users');
    back.setAttribute('href', '#users');
    head.appendChild(back);
    root.appendChild(head);

    const panel = s.el('section', 'identity-panel');
    panel.appendChild(s.el('h2', '', 'Current identity'));
    const help = s.el('button', 'help-button', '?');
    help.setAttribute('type', 'button');
    help.setAttribute('aria-label', 'How states are determined');
    help.setAttribute('title', 'How states are determined');
    help.addEventListener('click', () => s.openRules(help));
    panel.appendChild(help);
    const badges = s.el('p');
    badges.appendChild(signalBadge('lifecycle', profile.lifecycle?.segment));
    badges.append(' ');
    badges.appendChild(signalBadge('operational', profile.operational?.state));
    badges.append(' ');
    badges.appendChild(signalBadge('commercial', profile.commercial?.commercial_tier));
    panel.appendChild(badges);
    const reason = profile.operational?.state === 'healthy' ? profile.lifecycle?.reason : profile.operational?.reason;
    const evaluated = profile.operational?.calculated_at || profile.lifecycle?.calculated_at;
    panel.appendChild(s.el('p', '', `${reason || 'No reason recorded'} · last evaluated ${s.formatTimestamp(evaluated)}`));
    panel.appendChild(s.el('small', '', `Last meaningful activity ${s.formatTimestamp(profile.last_meaningful_activity_at ?? profile.last_meaningful_activity, 'none')} · First successful result ${s.formatTimestamp(profile.lifecycle?.activated_at, 'none')}`));
    const account = s.el('a', 'module-link', 'Open account management');
    account.setAttribute('href', accountManagementHref(profile));
    panel.appendChild(account);
    root.appendChild(panel);

    const milestones = s.el('div', 'milestones');
    Object.entries(profile.activation_milestones || {})
      .sort((left, right) => new Date(left[1]) - new Date(right[1]))
      .forEach(([name, occurredAt]) => {
        const item = s.el('div', 'milestone');
        item.append(eventLabel(name));
        item.appendChild(s.el('br'));
        item.appendChild(s.el('small', '', s.formatTimestamp(occurredAt)));
        milestones.appendChild(item);
      });
    if (!milestones.children.length) milestones.appendChild(s.el('p', 'panel-empty', 'No activation milestones recorded.'));
    root.appendChild(milestones);

    const tabs = s.el('div', 'tabs');
    tabs.setAttribute('role', 'tablist');
    tabs.setAttribute('aria-label', 'User analytics sections');
    SECTIONS.forEach((section) => {
      const button = s.el('button', '', SECTION_LABELS[section]);
      button.setAttribute('type', 'button');
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-selected', section === 'overview' ? 'true' : 'false');
      button.setAttribute('aria-controls', `profileSection-${section}`);
      button.dataset.section = section;
      button.addEventListener('click', () => selectSection(section));
      button.addEventListener('keydown', (event) => {
        const keys = { ArrowRight: 1, ArrowLeft: -1, Home: 0, End: SECTIONS.length - 1 };
        if (!(event.key in keys)) return;
        event.preventDefault();
        const index = SECTIONS.indexOf(section);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? SECTIONS.length - 1 : (index + keys[event.key] + SECTIONS.length) % SECTIONS.length;
        selectSection(SECTIONS[next]);
      });
      tabs.appendChild(button);
    });
    root.appendChild(tabs);
    return root;
  }

  function definitions(entries) {
    const s = shell();
    const list = s.el('dl', 'profile-facts');
    entries.forEach(([label, value]) => {
      list.appendChild(s.el('dt', '', label));
      list.appendChild(s.el('dd', '', value));
    });
    return list;
  }

  function renderProfileOverview(profile) {
    const s = shell();
    const lifecycle = profile.lifecycle || {};
    const commercial = profile.commercial || {};
    const root = s.el('div');
    const value = s.el('section', 'detail-section');
    value.appendChild(s.el('h3', '', 'User value summary'));
    value.appendChild(definitions([
      // `selected_period_end` is exclusive (end = to + 1 day); formatLastIncludedDay
      // renders the last day actually in the window, so 1W reads as seven days.
      ['Selected period', `${s.formatDateOnly(profile.selected_period_start)} – ${s.formatLastIncludedDay(profile.selected_period_end)}`],
      ['Activated', s.formatTimestamp(lifecycle.activated_at, 'Not activated')],
      ['Active days (30d)', s.formatNumber(lifecycle.active_days_30d)],
      ['Successful backtests (30d)', s.formatNumber(lifecycle.successful_backtests_30d)],
      ['Inactive UTC days', s.formatNumber(lifecycle.inactive_days)],
      ['Lifetime net purchased', s.formatCredits(commercial.lifetime_net_purchased_micro)],
      ['Consumed in period', s.formatCredits(commercial.consumed_micro)],
      ['Available balance', s.formatCredits(commercial.total_available_micro)],
    ]));
    root.appendChild(value);

    const evidence = s.el('section', 'detail-section');
    evidence.appendChild(s.el('h3', '', 'Lifecycle evidence'));
    evidence.appendChild(evidenceList(lifecycle.evidence, 'No lifecycle evidence is available.'));
    evidence.appendChild(s.el('h3', '', 'Operational evidence'));
    evidence.appendChild(evidenceList(profile.operational?.evidence, 'No operational evidence is available.'));
    root.appendChild(evidence);

    const runs = s.el('section', 'detail-section');
    runs.appendChild(s.el('h3', '', 'Run summary'));
    runs.appendChild(definitions(Object.entries(profile.run_summary || {}).map(([key, count]) => [s.humanize(key), s.formatNumber(count)])));
    root.appendChild(runs);

    const usage = s.el('section', 'detail-section');
    usage.appendChild(s.el('h3', '', 'Billing and usage'));
    const totalTokens = Number(profile.input_tokens) + Number(profile.output_tokens);
    usage.appendChild(definitions([
      ['Input tokens', s.formatNumber(profile.input_tokens)],
      ['Output tokens', s.formatNumber(profile.output_tokens)],
      ['Total tokens', Number.isFinite(totalTokens) ? s.formatNumber(totalTokens) : s.DASH],
      ['ATL platform model cost', s.usdFromMicro(Number(profile.platform_model_cost_usd) * 1000000)],
      ['ATL Credits debited', s.formatCredits(profile.credits_debited_micro)],
      ...Object.entries(profile.billing_lane_mix || {}).map(([lane, count]) => [
        lane === 'byok' ? 'BYOK usage — no ATL Credits debit' : s.humanize(lane), s.formatNumber(count),
      ]),
    ]));
    root.appendChild(usage);

    const movement = s.el('section', 'detail-section');
    movement.appendChild(s.el('h3', '', 'Recent lifecycle movement'));
    const transitions = s.el('ol', 'timeline');
    (profile.recent_lifecycle_transitions || []).forEach((transition) => {
      const item = s.el('li');
      const head = s.el('div');
      head.appendChild(s.el('strong', '', `${labelFor('lifecycle', transition.from_segment)} → ${labelFor('lifecycle', transition.to_segment)}`));
      head.appendChild(s.el('span', '', `${s.formatNumber(transition.users)} user${Number(transition.users) === 1 ? '' : 's'}`));
      item.appendChild(head);
      item.appendChild(s.el('p', '', `${s.formatDateOnly(transition.period_start)} – ${s.formatDateOnly(transition.period_end)}${transition.data_quality === 'partial' ? ` · ${s.INCOMPLETE}` : ''}`));
      transitions.appendChild(item);
    });
    if (!transitions.children.length) transitions.appendChild(s.el('li', 'panel-empty', 'No lifecycle transitions in this range.'));
    movement.appendChild(transitions);
    root.appendChild(movement);

    const footprint = s.el('section', 'detail-section');
    footprint.appendChild(s.el('h3', '', 'Recent footprint'));
    const list = s.el('ol', 'timeline');
    (profile.recent_footprint || []).forEach((item) => {
      const row = s.el('li');
      const head = s.el('div');
      head.appendChild(s.el('strong', '', eventLabel(item.event_name)));
      head.appendChild(s.el('span', '', s.formatTimestamp(item.occurred_at)));
      row.appendChild(head);
      const details = [item.page_view, item.provider_id, item.model_id, item.billing_mode, item.outcome, item.error_category].filter(Boolean).map(s.humanize).join(' · ');
      if (details) row.appendChild(s.el('p', '', details));
      list.appendChild(row);
    });
    if (!list.children.length) list.appendChild(s.el('li', 'panel-empty', 'No recent footprint events.'));
    footprint.appendChild(list);
    root.appendChild(footprint);
    return root;
  }

  function activityTable(headers, rows) {
    const s = shell();
    const wrap = s.el('div', 'table-wrap');
    const table = s.el('table');
    const head = s.el('thead');
    const headRow = s.el('tr');
    headers.forEach((label) => {
      const cell = s.el('th', '', label);
      cell.setAttribute('scope', 'col');
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    table.appendChild(head);
    const body = s.el('tbody');
    rows.forEach((cells) => {
      const row = s.el('tr');
      cells.forEach((cell) => row.appendChild(td(cell)));
      body.appendChild(row);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  function renderActivityItems(section, items) {
    const s = shell();
    const rows = Array.isArray(items) ? items : [];
    if (!rows.length) return s.el('p', 'panel-empty', 'No activity in this section.');
    if (section === 'timeline') {
      const list = s.el('ol', 'timeline');
      rows.forEach((item) => {
        const row = s.el('li');
        const head = s.el('div');
        head.appendChild(s.el('strong', '', eventLabel(item.event_name)));
        head.appendChild(s.el('span', '', s.formatTimestamp(item.occurred_at)));
        row.appendChild(head);
        const details = [item.outcome, item.provider_id, item.model_id, item.billing_mode, item.error_category].filter(Boolean).map(s.humanize).join(' · ');
        row.appendChild(s.el('p', '', details || 'No additional display-safe details.'));
        list.appendChild(row);
      });
      return list;
    }
    if (section === 'runs') {
      return activityTable(['Time', 'Run event', 'Outcome', 'Provider / model', 'Billing lane', 'Error category'], rows.map((item) => [
        s.formatTimestamp(item.occurred_at), eventLabel(item.event_name),
        item.outcome ? s.humanize(item.outcome) : s.DASH,
        [item.provider_id, item.model_id].filter(Boolean).join(' · ') || s.DASH,
        item.billing_mode ? s.humanize(item.billing_mode) : s.DASH,
        item.error_category ? s.humanize(item.error_category) : s.DASH,
      ]));
    }
    if (section === 'usage') {
      return activityTable(['Time', 'Usage event', 'Provider / model', 'Billing lane', 'Input', 'Output', 'ATL cost', 'ATL Credits debited'], rows.map((item) => {
        const byok = item.billing_mode === 'byok';
        return [
          s.formatTimestamp(item.occurred_at), eventLabel(item.event_name),
          [item.provider_id, item.model_id].filter(Boolean).join(' · ') || s.DASH,
          byok ? 'BYOK — no ATL charge' : (item.billing_mode ? s.humanize(item.billing_mode) : s.DASH),
          s.formatNumber(item.input_tokens), s.formatNumber(item.output_tokens),
          byok || item.cost_micro_usd == null ? s.DASH : s.usdFromMicro(item.cost_micro_usd),
          byok || item.amount_micro == null ? s.DASH : s.formatCredits(item.amount_micro),
        ];
      }));
    }
    // Sessions: Started · Events · Visible time. Region / Device / Browser are
    // collected but not displayed (design D15); the columns are gone, not hidden.
    return activityTable(['Started', 'Events', 'Visible time'], rows.map((item) => [
      s.formatTimestamp(item.occurred_at), s.formatNumber(item.session_event_count), formatVisibleTime(item.visible_ms),
    ]));
  }

  function sectionPanel(section) {
    return document.getElementById(`profileSection-${section}`);
  }

  function renderSectionPanel(section) {
    const s = shell();
    const panel = sectionPanel(section);
    if (!panel) return;
    const sectionState = state.profile.sections[section];
    s.clear(panel);
    if (section === 'overview') {
      panel.appendChild(renderProfileOverview(state.profile.detail));
      return;
    }
    if (sectionState.error) {
      panel.appendChild(s.el('p', 'panel-error', sectionState.error));
    }
    if (sectionState.loaded) panel.appendChild(renderActivityItems(section, sectionState.items));
    else if (sectionState.loading) panel.appendChild(s.el('p', 'panel-status muted', 'Loading activity…'));
    if (sectionState.nextCursor) {
      const more = s.el('button', 'load-more', `Load more ${SECTION_LABELS[section].toLowerCase()}`);
      more.setAttribute('type', 'button');
      more.disabled = sectionState.loading;
      more.addEventListener('click', () => loadSection(section, { append: true }));
      panel.appendChild(more);
    }
  }

  async function loadSection(section, { append = false } = {}) {
    const s = shell();
    const sectionState = state.profile.sections[section];
    const userId = state.profile.userId;
    if (!sectionState || !userId || sectionState.loading) return;
    if (append && !sectionState.nextCursor) return;
    sectionState.loading = true;
    sectionState.error = null;
    const seq = ++sectionState.requestSeq;
    renderSectionPanel(section);
    const params = new URLSearchParams({ section, limit: String(PAGE_SIZE) });
    if (append) params.set('cursor', sectionState.nextCursor);
    try {
      const payload = await s.request(`${USERS_PATH}/${encodeURIComponent(String(userId))}/activity?${params}`);
      if (String(state.profile.userId) !== String(userId) || seq !== sectionState.requestSeq) return;
      const next = Array.isArray(payload.items) ? payload.items : [];
      sectionState.items = append ? sectionState.items.concat(next) : next;
      sectionState.nextCursor = payload.next_cursor || null;
      sectionState.loaded = true;
    } catch (error) {
      if (seq !== sectionState.requestSeq) return;
      if (await s.handleAccessLost(error)) return;
      sectionState.error = append ? 'More activity is temporarily unavailable.' : s.SECTION_UNAVAILABLE;
    } finally {
      if (seq === sectionState.requestSeq) {
        sectionState.loading = false;
        renderSectionPanel(section);
      }
    }
  }

  async function selectSection(value) {
    const section = SECTIONS.includes(value) ? value : 'overview';
    state.profile.section = section;
    const root = document.getElementById('profile');
    root?.querySelectorAll('[role="tab"]').forEach((button) => {
      const selected = button.dataset.section === section;
      button.setAttribute('aria-selected', selected ? 'true' : 'false');
      button.tabIndex = selected ? 0 : -1;
    });
    SECTIONS.forEach((name) => {
      const panel = sectionPanel(name);
      if (panel) panel.hidden = name !== section;
    });
    if (section !== 'overview' && !state.profile.sections[section]?.loaded) {
      await loadSection(section, { append: false });
    }
  }

  function paintProfile(profile) {
    const s = shell();
    const root = document.getElementById('profile');
    if (!root) return;
    s.clear(root);
    root.appendChild(renderProfileHeader(profile));
    SECTIONS.forEach((section) => {
      const panel = s.el('section', 'detail-section');
      panel.id = `profileSection-${section}`;
      panel.setAttribute('role', 'tabpanel');
      panel.hidden = section !== 'overview';
      root.appendChild(panel);
    });
    renderSectionPanel('overview');
    root.querySelector('h1')?.focus?.({ preventScroll: true });
  }

  async function openProfile(userId) {
    const s = shell();
    if (!/^\d+$/.test(String(userId || ''))) return;
    state.profile = {
      userId: String(userId),
      detail: null,
      section: 'overview',
      sections: Object.fromEntries(SECTIONS.filter((name) => name !== 'overview').map((name) => [name, emptySection()])),
    };
    const seq = s.nextSeq('profile');
    const root = document.getElementById('profile');
    if (root) {
      s.clear(root);
      root.appendChild(s.el('p', 'panel-status muted', 'Loading user analytics…'));
    }
    try {
      const { from, to } = s.rangeDates(s.state.range);
      const params = new URLSearchParams({ from, to });
      const profile = await s.request(`${USERS_PATH}/${encodeURIComponent(String(userId))}?${params}`);
      if (!s.isCurrent('profile', seq) || String(state.profile.userId) !== String(userId)) return;
      state.profile.detail = profile;
      paintProfile(profile);
    } catch (error) {
      if (!s.isCurrent('profile', seq)) return;
      if (await s.handleAccessLost(error)) return;
      if (root) {
        s.clear(root);
        root.appendChild(s.el('p', 'panel-error', error?.status === 404 ? 'Analytics user was not found.' : 'User analytics are temporarily unavailable.'));
      }
    }
  }

  function bind() {
    const s = shell();
    document.getElementById('usersSearch')?.addEventListener('submit', (event) => {
      event.preventDefault();
      const input = document.getElementById('usersQuery');
      s.setFilters({ q: String(input?.value || '').trim().slice(0, 100) });
    });
    document.getElementById('usersPriority')?.addEventListener('change', (event) => {
      s.setFilters({ priority: Boolean(event.target.checked) });
    });
    document.getElementById('usersPrev')?.addEventListener('click', () => loadList({ offset: Math.max(0, state.list.offset - PAGE_SIZE) }));
    document.getElementById('usersNext')?.addEventListener('click', () => loadList({ offset: state.list.offset + PAGE_SIZE }));
  }

  document.addEventListener('DOMContentLoaded', bind);
  document.addEventListener('admin:route', (event) => {
    const detail = event.detail || {};
    if (detail.route !== 'users') return;
    const query = document.getElementById('usersQuery');
    if (query) query.value = detail.filters?.q || '';
    const priority = document.getElementById('usersPriority');
    if (priority) priority.checked = Boolean(detail.filters?.priority);
    if (detail.id) openProfile(detail.id);
    else loadList({ offset: 0 });
  });
  document.addEventListener('admin:retry', (event) => {
    if (event.detail?.panel === 'users') loadList({ offset: state.list.offset });
  });

  window.AdminUsers = {
    state,
    groupBadge, signalBadge, renderUserRows, renderPager, renderEvidence, activationWeekLabel,
    renderProfileHeader, renderProfileOverview, renderActivityItems, accountManagementHref,
    loadList, openProfile, selectSection, loadSection,
  };
})();
