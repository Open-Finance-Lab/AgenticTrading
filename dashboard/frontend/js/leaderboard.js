// ============================================================================
// LEADERBOARD (Competition tab — real baselines from API)
// ============================================================================

let leaderboardPayload = null;
let currentLeaderboardSort = 'rank'; // 'rank' | 'value' | 'return' | 'sharpe' | 'dd'
let currentLeaderboardSortDir = 'asc'; // rank: asc (1 best); metrics usually start desc
let selectedLeaderboardEntry = null;
let equityCurvesData = null;
let equityCurvesChartInstance = null;
let currentChartView = 'absolute'; // default: money ($). 'cumulative' = % return
let leaderboardListenersInitialized = false;
let dailyLeaderboardPollTimer = null;

// Chart visual-hierarchy state
let hiddenSeries = new Set();
let hiddenInitialized = false;
let hoveredDatasetIndex = null;
let canvasLeaveBound = false;
const selectedBenchmarkLabel = 'SPY';
/** Expanded groups in the Show-on-Chart picker: model | baseline | index */
let curvePickerExpanded = new Set();
let curvePickerOutsideBound = false;

// Stable per-series style presets. `kind` drives width/opacity hierarchy:
//   benchmark -> neutral gray, dotted / dash-dot, understated
//   strategy  -> colored, long-dashed, secondary
//   team      -> solid, prominent (colors assigned stably, never by rank)
const LEADERBOARD_STYLES = {
  SPY: { color: '#CBD5E1', kind: 'benchmark', dash: [2, 4] },
  DJIA: { color: '#94A3B8', kind: 'benchmark', dash: [8, 4, 2, 4] },
  'Buy & Hold': { color: '#38BDF8', kind: 'strategy', dash: [10, 6] },
  'Mean-Variance': { color: '#C084FC', kind: 'strategy', dash: [10, 6] },
  'Equal-Weight': { color: '#4ADE80', kind: 'strategy', dash: [10, 6] },
};

// Visual hierarchy: teams are boldest, provided models prominent (solid),
// strategy baselines secondary (dashed), market indices the most understated.
const KIND_WIDTH = { team: 2.25, model: 2.0, strategy: 1.6, benchmark: 1.1 };
const KIND_ALPHA = { team: 1.0, model: 0.95, strategy: 0.7, benchmark: 0.5 };
const EMPHASIS_WIDTH = 3;

// Stable bright palette for actual competition teams (assigned first-seen).
const TEAM_COLOR_PALETTE = [
  '#F97316', '#EAB308', '#EC4899', '#14B8A6', '#A855F7',
  '#EF4444', '#06B6D4', '#84CC16', '#F43F5E', '#8B5CF6',
];
const teamColorMap = {};

// Provided LLM models get their own warm, distinct palette (solid lines) so
// they read as a separate category from rule-based strategy baselines.
const MODEL_COLOR_PALETTE = [
  '#FBBF24', '#FB923C', '#F472B6', '#A78BFA', '#34D399',
];
const modelColorMap = {};

function formatEntryBadge(badge) {
  const raw = String(badge || '').trim();
  if (!raw || raw === 'Baseline') return 'Baseline Strategy';
  if (raw === 'Index') return 'Market Index';
  if (raw === 'Strategy') return 'Baseline Strategy';
  return raw;
}

function isModelEntry(entry) {
  return !!(entry && (entry.is_model || entry.team_badge === 'Model'));
}

function getModelColor(stableId) {
  const key = String(stableId);
  if (!modelColorMap[key]) {
    const idx = Object.keys(modelColorMap).length % MODEL_COLOR_PALETTE.length;
    modelColorMap[key] = MODEL_COLOR_PALETTE[idx];
  }
  return modelColorMap[key];
}

function shortName(label) {
  // Model names are already canonical/short; only guard against long team names.
  return label && label.length > 18 ? `${label.slice(0, 17)}…` : (label || '');
}

function hexToRgba(hex, alpha) {
  const h = String(hex || '').replace('#', '');
  const r = parseInt(h.slice(0, 2), 16) || 0;
  const g = parseInt(h.slice(2, 4), 16) || 0;
  const b = parseInt(h.slice(4, 6), 16) || 0;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function getTeamColor(stableId) {
  const key = String(stableId);
  if (!teamColorMap[key]) {
    const idx = Object.keys(teamColorMap).length % TEAM_COLOR_PALETTE.length;
    teamColorMap[key] = TEAM_COLOR_PALETTE[idx];
  }
  return teamColorMap[key];
}

function getSeriesStyle(label, entry) {
  if (isModelEntry(entry)) {
    return { color: getModelColor(entry?.entry_id || label), kind: 'model', dash: [] };
  }
  const preset = LEADERBOARD_STYLES[label];
  if (preset) return { ...preset };
  return { color: getTeamColor(entry?.entry_id || label), kind: 'team', dash: [] };
}

function getEntryKind(entry) {
  if (entry.entry_type && entry.entry_type !== 'baseline') return 'team';
  if (isModelEntry(entry)) return 'model';
  const label = entry.model || entry.team_name;
  const preset = LEADERBOARD_STYLES[label];
  return preset ? preset.kind : 'strategy';
}

/** Map series kind → chart-picker group id. */
function getFilterCategory(entry) {
  const kind = getEntryKind(entry);
  if (kind === 'model' || kind === 'team') return 'model';
  if (kind === 'benchmark') return 'index';
  return 'baseline'; // strategy baselines
}

const CURVE_PICKER_GROUPS = [
  { id: 'model', title: 'Models' },
  { id: 'baseline', title: 'Baseline Strategies' },
  { id: 'index', title: 'Market Indices' },
];

function entrySeriesLabel(entry) {
  return entry.model || entry.team_name || entry.entry_id || '';
}

function getCurvePickerGroups(entries) {
  const buckets = { model: [], baseline: [], index: [] };
  (entries || []).forEach((entry) => {
    const cat = getFilterCategory(entry);
    if (buckets[cat]) buckets[cat].push(entry);
  });
  // Models: sort by return desc for a familiar board order
  buckets.model.sort(
    (a, b) => (Number(b.cumulative_return) || 0) - (Number(a.cumulative_return) || 0)
  );
  return CURVE_PICKER_GROUPS.map((g) => ({
    ...g,
    entries: buckets[g.id] || [],
  })).filter((g) => g.entries.length > 0);
}

function seriesVisible(label) {
  return !hiddenSeries.has(label);
}

function countVisibleSeries(entries) {
  return (entries || []).filter((e) => seriesVisible(entrySeriesLabel(e))).length;
}

function setGroupVisibility(groupEntries, visible) {
  groupEntries.forEach((entry) => {
    const label = entrySeriesLabel(entry);
    if (!label) return;
    if (visible) hiddenSeries.delete(label);
    else hiddenSeries.add(label);
  });
}

function groupCheckState(groupEntries) {
  const total = groupEntries.length;
  if (!total) return 'none';
  const visible = groupEntries.filter((e) => seriesVisible(entrySeriesLabel(e))).length;
  if (visible === 0) return 'none';
  if (visible === total) return 'all';
  return 'partial';
}

function updateCurvePickerCount() {
  const el = document.getElementById('curvePickerCount');
  if (!el) return;
  const entries = leaderboardPayload?.entries || [];
  const n = countVisibleSeries(entries);
  const total = entries.length;
  el.textContent = total ? `${n} selected` : '0 selected';
}

function renderCurvePicker() {
  const body = document.getElementById('curvePickerBody');
  if (!body) return;
  const groups = getCurvePickerGroups(leaderboardPayload?.entries || []);

  body.innerHTML = groups.map((group) => {
    const state = groupCheckState(group.entries);
    const visibleN = group.entries.filter((e) => seriesVisible(entrySeriesLabel(e))).length;
    const expanded = curvePickerExpanded.has(group.id);
    const checkClass =
      state === 'all' ? 'is-checked' : state === 'partial' ? 'is-partial' : 'is-unchecked';
    // Always show selected/total so collapsed groups aren't mistaken for "all".
    const countLabel = `${visibleN}/${group.entries.length}`;

    const children = group.entries.map((entry, idx) => {
      const label = entrySeriesLabel(entry);
      const on = seriesVisible(label);
      const safeName = String(label)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/"/g, '&quot;');
      return `
        <label class="curve-picker-item">
          <input type="checkbox" data-group-id="${group.id}" data-entry-idx="${idx}" ${on ? 'checked' : ''} />
          <span class="curve-picker-item-name">${safeName}</span>
        </label>`;
    }).join('');

    return `
      <div class="curve-picker-group" data-group="${group.id}">
        <div class="curve-picker-group-row">
          <button type="button" class="curve-picker-group-check ${checkClass}"
            data-group-toggle="${group.id}" aria-label="Toggle ${group.title}"
            aria-checked="${state === 'all' ? 'true' : state === 'none' ? 'false' : 'mixed'}"></button>
          <button type="button" class="curve-picker-group-expand" data-group-expand="${group.id}"
            aria-expanded="${expanded ? 'true' : 'false'}">
            <span class="curve-picker-group-title">${group.title} (${countLabel})</span>
            <span class="curve-picker-group-chevron" aria-hidden="true">${expanded ? '▾' : '›'}</span>
          </button>
        </div>
        <div class="curve-picker-children${expanded ? '' : ' is-collapsed'}">${children}</div>
      </div>`;
  }).join('');

  updateCurvePickerCount();
}

function applyChartVisibilityChange() {
  updateCurvePickerCount();
  const open = document.getElementById('curvePickerTrigger')?.getAttribute('aria-expanded') === 'true';
  if (open) renderCurvePicker();
  else updateCurvePickerCount();
  renderEquityCurvesChart();
}

function setCurvePickerOpen(open) {
  const menu = document.getElementById('curvePickerMenu');
  const trigger = document.getElementById('curvePickerTrigger');
  const root = document.getElementById('curvePicker');
  if (!menu || !trigger) return;
  menu.hidden = !open;
  trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
  root?.classList.toggle('is-open', open);
  if (open) renderCurvePicker();
}

function updateLeaderboardHeader(payload) {
  const entries = payload.entries || [];
  // Best LLM / provided model by cumulative return (not overall board rank).
  const models = entries
    .filter((e) => isModelEntry(e))
    .slice()
    .sort((a, b) => (Number(b.cumulative_return) || 0) - (Number(a.cumulative_return) || 0));

  const totalEl = document.getElementById('totalTeams');
  const windowEl = document.getElementById('tradingWindow');
  const updatedEl = document.getElementById('lastUpdate');
  const leaderEl = document.getElementById('leaderTeam');
  const standingsEl = document.getElementById('leaderboardStandingsTitle');
  const subtitleEl = document.querySelector('#leaderboardView .contest-subtitle');

  if (totalEl) totalEl.textContent = payload.phase_label || (payload.period === 'daily' ? 'Daily' : 'Preseason');
  if (windowEl) windowEl.textContent = payload.window?.label || '—';
  if (updatedEl) {
    updatedEl.textContent = payload.updated_at
      ? new Date(payload.updated_at).toLocaleString()
      : '—';
  }
  if (leaderEl) {
    const top = models[0];
    leaderEl.textContent = top
      ? (top.model || top.team_name)
      : (payload.leader || '—');
  }
  if (standingsEl) {
    standingsEl.textContent = payload.standings_label || 'Ranking';
  }
  if (subtitleEl) {
    subtitleEl.textContent = payload.period === 'daily'
      ? formatDailyBoardSubtitle(payload)
      : 'Sep 1 – Oct 30, 2026';
  }
  renderDailyLeaderboardNotice(payload);
  updateCurvePickerCount();
}

function formatDailyBoardSubtitle(payload) {
  const date = payload.daily_status?.trading_date || payload.window?.start_date;
  if (date) {
    return `Daily window · ${date} (last completed US weekday)`;
  }
  return 'Daily window · last completed US weekday';
}

function renderDailyLeaderboardNotice(payload) {
  const host = document.getElementById('leaderboardDailyNotice');
  if (!host) return;
  if (payload.period !== 'daily') {
    host.hidden = true;
    host.textContent = '';
    return;
  }
  const status = payload.daily_status || {};
  const pending = Number(status.models_pending) || 0;
  const total = Number(status.models_total) || 0;
  const cached = Number(status.models_cached) || 0;
  const inProgress = Boolean(status.refresh_in_progress);

  if (inProgress || pending > 0) {
    host.hidden = false;
    const date = status.trading_date || payload.window?.start_date || 'today';
    if (inProgress) {
      host.textContent = `Running competition models for ${date}… (${cached}/${total} ready). This page refreshes automatically.`;
    } else {
      host.textContent = `${pending} competition model curve(s) pending for ${date}. Baselines are live; models appear once the daily job finishes.`;
    }
    return;
  }
  if (total > 0 && cached === total) {
    host.hidden = false;
    host.textContent = `All ${total} competition models loaded for ${status.trading_date || payload.window?.start_date}.`;
    return;
  }
  host.hidden = true;
  host.textContent = '';
}

function scheduleDailyLeaderboardPoll(payload) {
  if (dailyLeaderboardPollTimer) {
    clearTimeout(dailyLeaderboardPollTimer);
    dailyLeaderboardPollTimer = null;
  }
  if (payload.period !== 'daily') return;
  const status = payload.daily_status || {};
  // Only poll while a refresh worker is running. Pending curves with no worker
  // wait for the nightly cron — polling forever just hammers the API.
  if (!status.refresh_in_progress) return;
  dailyLeaderboardPollTimer = setTimeout(() => {
    loadLeaderboardData('daily');
  }, 30000);
}

async function loadLeaderboardData(period = 'contest') {
  console.log('Loading leaderboard from API...', period);
  const boardPeriod = period === 'daily' ? 'daily' : 'contest';

  try {
    const url = `${API_BASE}/api/v1/leaderboard?period=${encodeURIComponent(boardPeriod)}&t=${Date.now()}`;
    leaderboardPayload = await API.get(url);
    const entries = leaderboardPayload.entries || [];
    equityCurvesData = buildEquityCurvesFromEntries(entries);

    // Reset chart visibility when switching boards so daily/contest don't share hide state.
    hiddenSeries = new Set();
    hiddenInitialized = true;

    updateLeaderboardHeader(leaderboardPayload);
    populateLeaderboardTable();
    renderCurvePicker();

    if (!leaderboardListenersInitialized) {
      initLeaderboardListeners();
      leaderboardListenersInitialized = true;
    }

    await renderEquityCurvesChart();
    scheduleDailyLeaderboardPoll(leaderboardPayload);
  } catch (error) {
    console.error('Error loading leaderboard:', error);
    displayLeaderboardError(error.message);
  }
}

function initLeaderboardListeners() {
  const trigger = document.getElementById('curvePickerTrigger');
  const menu = document.getElementById('curvePickerMenu');
  const body = document.getElementById('curvePickerBody');
  const clearBtn = document.getElementById('curvePickerClear');

  // Keep the menu open while interacting: stop bubbles, and close only on
  // outside pointerdown (avoids the classic "rerender detaches target →
  // document click thinks it's outside" bug).
  menu?.addEventListener('click', (e) => e.stopPropagation());
  menu?.addEventListener('pointerdown', (e) => e.stopPropagation());

  trigger?.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = trigger.getAttribute('aria-expanded') === 'true';
    setCurvePickerOpen(!open);
  });

  clearBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    const entries = leaderboardPayload?.entries || [];
    entries.forEach((entry) => {
      const label = entrySeriesLabel(entry);
      if (label) hiddenSeries.add(label);
    });
    applyChartVisibilityChange();
  });

  body?.addEventListener('click', (e) => {
    e.stopPropagation();
    const expandBtn = e.target.closest('[data-group-expand]');
    if (expandBtn) {
      e.preventDefault();
      const id = expandBtn.dataset.groupExpand;
      if (curvePickerExpanded.has(id)) curvePickerExpanded.delete(id);
      else curvePickerExpanded.add(id);
      renderCurvePicker();
      return;
    }

    const groupToggle = e.target.closest('[data-group-toggle]');
    if (groupToggle) {
      e.preventDefault();
      const id = groupToggle.dataset.groupToggle;
      const group = getCurvePickerGroups(leaderboardPayload?.entries || [])
        .find((g) => g.id === id);
      if (!group) return;
      const state = groupCheckState(group.entries);
      // All on → turn off; partial or none → turn all on.
      setGroupVisibility(group.entries, state !== 'all');
      applyChartVisibilityChange();
    }
  });

  body?.addEventListener('change', (e) => {
    e.stopPropagation();
    const input = e.target;
    if (!(input instanceof HTMLInputElement) || input.dataset.entryIdx == null) return;
    const groupId = input.dataset.groupId;
    const idx = Number(input.dataset.entryIdx);
    const group = getCurvePickerGroups(leaderboardPayload?.entries || [])
      .find((g) => g.id === groupId);
    const entry = group?.entries?.[idx];
    if (!entry) return;
    const label = entrySeriesLabel(entry);
    if (input.checked) hiddenSeries.delete(label);
    else hiddenSeries.add(label);
    applyChartVisibilityChange();
  });

  if (!curvePickerOutsideBound) {
    document.addEventListener('pointerdown', (e) => {
      const root = document.getElementById('curvePicker');
      if (!root?.classList.contains('is-open')) return;
      if (e.target instanceof Node && root.contains(e.target)) return;
      setCurvePickerOpen(false);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') setCurvePickerOpen(false);
    });
    curvePickerOutsideBound = true;
  }

  document.querySelectorAll('.leaderboard-table th.sortable').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (!key) return;
      if (currentLeaderboardSort === key) {
        currentLeaderboardSortDir = currentLeaderboardSortDir === 'asc' ? 'desc' : 'asc';
      } else {
        currentLeaderboardSort = key;
        // Rank: 1 first. Value/Return/Sharpe: high first. Max DD: low first.
        currentLeaderboardSortDir = key === 'dd' || key === 'rank' ? 'asc' : 'desc';
      }
      updateLeaderboardSortHeaders();
      populateLeaderboardTable();
    });
  });
  updateLeaderboardSortHeaders();

  document.querySelectorAll('.view-toggle-btn').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      document.querySelectorAll('.view-toggle-btn').forEach((b) => b.classList.remove('active'));
      e.target.classList.add('active');
      currentChartView = e.target.dataset.view === 'absolute' ? 'absolute' : 'cumulative';
      await renderEquityCurvesChart();
    });
  });

  document.querySelectorAll('.chart-view-btn').forEach((btn) => {
    btn.addEventListener('click', async (e) => {
      document.querySelectorAll('.chart-view-btn').forEach((b) => b.classList.remove('active'));
      e.target.classList.add('active');
      currentChartView = e.target.dataset.view === 'absolute' ? 'absolute' : 'cumulative';
      await renderEquityCurvesChart();
    });
  });
}

function formatLeaderboardNumber(num) {
  return Number(num || 0).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function formatShortDate(isoDay) {
  if (!isoDay) return '';
  const d = new Date(String(isoDay).includes('T') ? isoDay : `${isoDay}T00:00:00`);
  if (Number.isNaN(d.getTime())) return isoDay;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatChartTooltipLabel(ts) {
  if (!ts) return '';
  const raw = String(ts);
  const d = new Date(raw.includes('T') ? raw : `${raw}T00:00:00`);
  if (Number.isNaN(d.getTime())) return raw;
  // Hourly series: show date + hour so the open tick and intraday points differ.
  if (raw.includes('T')) {
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    });
  }
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/** Normalize equity timestamps to hour precision for shared-axis alignment. */
function chartTimeKey(ts) {
  const s = String(ts || '');
  if (!s) return '';
  // 2026-04-15T14:00:00+00:00 → 2026-04-15T14:00
  if (s.length >= 16 && s[10] === 'T') return s.slice(0, 16);
  if (s.length >= 10) return s.slice(0, 10);
  return s;
}

function buildEquityCurvesFromEntries(entries) {
  // Align every series on a shared, sorted hourly axis rather than by
  // positional index or calendar-day bucket, so the open tick and each market
  // hour render at their true x-positions.
  const timeSet = new Set();
  const perEntry = [];

  entries.forEach((entry) => {
    const points = entry.equity_curve || [];
    if (!points.length) return;
    const byTime = {};
    points.forEach((pt) => {
      const key = chartTimeKey(pt.timestamp);
      if (!key) return;
      byTime[key] = Number(pt.equity) || 0;
      timeSet.add(key);
    });
    perEntry.push({ entry, seriesLabel: entry.model || entry.team_name, byTime });
  });

  const times = Array.from(timeSet).sort();
  const curves = {};
  const trajectories = {};
  const initials = {};

  perEntry.forEach(({ entry, seriesLabel, byTime }) => {
    const values = times.map((t) => (t in byTime ? byTime[t] : null));
    const firstReal = values.find((v) => v != null);
    curves[seriesLabel] = values;
    initials[seriesLabel] = Number(entry.initial_equity) || firstReal || 10000;
    trajectories[seriesLabel] = getSeriesStyle(seriesLabel, entry);
  });

  // Keep `days` alias so any older callers still unpack a familiar key.
  return { times, days: times, curves, trajectories, initials };
}

// Default chart visibility: top 5 teams + benchmarks. Strategy baselines are
// hidden by default (selectable via legend) — unless there are no teams yet,
// in which case we show them so the chart isn't just two gray lines.
function computeDefaultHidden(entries) {
  const hidden = new Set();
  const teams = entries.filter((e) => getEntryKind(e) === 'team');
  const hasTeams = teams.length > 0;
  const visibleTeamIds = new Set(teams.slice(0, 5).map((e) => e.entry_id));

  entries.forEach((entry) => {
    const label = entry.model || entry.team_name;
    const kind = getEntryKind(entry);
    if (kind === 'team' && !visibleTeamIds.has(entry.entry_id)) hidden.add(label);
    if (kind === 'strategy' && hasTeams) hidden.add(label);
  });
  return hidden;
}

function updateLeaderboardSortHeaders() {
  document.querySelectorAll('.leaderboard-table th.sortable').forEach((th) => {
    const active = th.dataset.sort === currentLeaderboardSort;
    th.classList.toggle('is-sorted', active);
    th.setAttribute('aria-sort', active
      ? (currentLeaderboardSortDir === 'asc' ? 'ascending' : 'descending')
      : 'none');
    const arrow = th.querySelector('.sort-arrow');
    if (arrow) {
      arrow.textContent = active ? (currentLeaderboardSortDir === 'asc' ? '↑' : '↓') : '';
    }
  });
}

function getFilteredLeaderboardEntries() {
  // Official rankings table always lists every entry; chart visibility is separate.
  // Left Rank stays the official portfolio-value rank; this only reorders rows.
  const entries = (leaderboardPayload?.entries || []).slice();
  const dir = currentLeaderboardSortDir === 'asc' ? 1 : -1;
  const num = (v) => Number(v) || 0;
  entries.sort((a, b) => {
    let cmp = 0;
    switch (currentLeaderboardSort) {
      case 'rank':
        cmp = num(a.rank) - num(b.rank);
        break;
      case 'value':
        cmp = num(a.portfolio_value) - num(b.portfolio_value);
        break;
      case 'return':
        cmp = num(a.cumulative_return) - num(b.cumulative_return);
        break;
      case 'sharpe':
        cmp = num(a.sharpe_ratio) - num(b.sharpe_ratio);
        break;
      case 'dd':
        cmp = Math.abs(num(a.max_drawdown)) - Math.abs(num(b.max_drawdown));
        break;
      default:
        cmp = num(a.rank) - num(b.rank);
    }
    return cmp * dir;
  });
  return entries;
}

// `entry.model` / `entry.team_name` are user-registered agent names, so every
// string field must go through `escapeHtml` (a global from app.js, loaded
// first). The onclick id additionally needs JS-string escaping — backslash
// before quote, or a trailing "\" would un-escape the closing quote — and the
// JS-escaped result is then HTML-escaped for the attribute context.
function renderLeaderboardRowHtml(entry) {
  const safeId = escapeHtml(
    String(entry.entry_id || entry.team_name).replace(/\\/g, '\\\\').replace(/'/g, "\\'")
  );
  const entryLabel = escapeHtml(entry.model || entry.team_name || '—');
  const ret = Number(entry.cumulative_return || 0);
  const retClass = ret >= 0 ? 'return-positive' : 'return-negative';
  const ddRaw = Number(entry.max_drawdown || 0);
  const dd = (Math.abs(ddRaw) * 100).toFixed(2);

  return `
      <tr onclick="selectLeaderboardTeam('${safeId}')">
        <td class="rank-cell">${escapeHtml(entry.rank)}</td>
        <td>
          <div class="team-name-badge">
            <span>${entryLabel}</span>
            <span class="team-badge">${escapeHtml(formatEntryBadge(entry.team_badge))}</span>
          </div>
        </td>
        <td style="text-align: right; font-family: var(--font-mono);">$${formatLeaderboardNumber(entry.portfolio_value)}</td>
        <td style="text-align: right;" class="${retClass}">
          <span class="metric-value-text">${(ret * 100).toFixed(2)}%</span>
        </td>
        <td style="text-align: right; font-family: var(--font-mono);">${Number(entry.sharpe_ratio || 0).toFixed(2)}</td>
        <td style="text-align: right; font-family: var(--font-mono);">${dd}%</td>
      </tr>
    `;
}

function populateLeaderboardTable() {
  const tbody = document.getElementById('leaderboardTableBody');
  if (!tbody) return;

  const filtered = getFilteredLeaderboardEntries();
  if (!filtered.length) {
    const period = leaderboardPayload?.period;
    const msg = period === 'daily'
      ? 'No daily entries yet. Baselines compute on first load; competition models deploy automatically (local dev) or via the nightly refresh job.'
      : 'No leaderboard entries yet. Baselines compute on first load (requires market data).';
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text-secondary);">${msg}</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(renderLeaderboardRowHtml).join('');
}

function renderLeaderboardDetailHtml(entry, totalEntries) {
  const ret = Number(entry.cumulative_return || 0);
  const retColor = ret >= 0 ? 'var(--success-color)' : 'var(--danger-color)';
  const entryLabel = escapeHtml(entry.model || entry.team_name || '—');
  return `
      <div class="team-detail-row">
        <span class="team-detail-label">Entry</span>
        <span class="team-detail-value">${entryLabel}</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Type</span>
        <span class="team-detail-value">${escapeHtml(formatEntryBadge(entry.team_badge || (entry.entry_type === 'baseline' ? 'Baseline Strategy' : 'Agent')))}</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Value</span>
        <span class="team-detail-value">$${formatLeaderboardNumber(entry.portfolio_value)}</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Return</span>
        <span class="team-detail-value" style="color: ${retColor};">${(ret * 100).toFixed(2)}%</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Sharpe</span>
        <span class="team-detail-value">${Number(entry.sharpe_ratio || 0).toFixed(2)}</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Max Drawdown</span>
        <span class="team-detail-value">${(Math.abs(Number(entry.max_drawdown || 0)) * 100).toFixed(2)}%</span>
      </div>
      <div class="team-detail-row">
        <span class="team-detail-label">Rank</span>
        <span class="team-detail-value">${escapeHtml(entry.rank)} / ${escapeHtml(totalEntries || '—')}</span>
      </div>
    `;
}

function selectLeaderboardTeam(entryId) {
  const entries = leaderboardPayload?.entries || [];
  selectedLeaderboardEntry =
    entries.find((e) => String(e.entry_id) === String(entryId)) ||
    entries.find((e) => e.team_name === entryId);
  if (!selectedLeaderboardEntry) return;

  const entry = selectedLeaderboardEntry;
  const detailPanel = document.getElementById('selectedTeamDetail');
  if (detailPanel) {
    detailPanel.innerHTML = renderLeaderboardDetailHtml(entry, leaderboardPayload?.total_entries);
  }

  // Selecting an entry also forces it visible and re-emphasizes it on the chart.
  const label = entry.model || entry.team_name;
  hiddenSeries.delete(label);
  renderEquityCurvesChart();
}

function getEmphasisLabel(orderedEntries) {
  if (selectedLeaderboardEntry) {
    return selectedLeaderboardEntry.model || selectedLeaderboardEntry.team_name;
  }
  const leadTeam = orderedEntries.find((e) => getEntryKind(e) === 'team');
  return leadTeam ? (leadTeam.model || leadTeam.team_name) : null;
}

function styleDatasets(chart) {
  const emphasisLabel = chart.$emphasisLabel;
  chart.data.datasets.forEach((ds, i) => {
    const st = ds._style || { kind: 'team' };
    const baseW = KIND_WIDTH[st.kind] || 2;
    let alpha;
    let width;
    if (hoveredDatasetIndex != null) {
      if (i === hoveredDatasetIndex) {
        alpha = 1.0;
        width = Math.max(baseW + 0.75, 2.5);
      } else {
        alpha = 0.25;
        width = baseW;
      }
    } else {
      const emph = ds.label === emphasisLabel;
      alpha = emph ? 1.0 : (KIND_ALPHA[st.kind] ?? 1.0);
      width = emph ? EMPHASIS_WIDTH : baseW;
    }
    ds.borderColor = hexToRgba(st.color, alpha);
    ds.borderWidth = width;
  });
}

// Subtle glow only on the emphasized (selected/leading) curve.
const selectedGlowPlugin = {
  id: 'selectedGlow',
  beforeDatasetDraw(chart, args) {
    const ds = chart.data.datasets[args.index];
    if (ds && ds.label === chart.$emphasisLabel && hoveredDatasetIndex == null) {
      const { ctx } = chart;
      ctx.save();
      ctx.shadowColor = hexToRgba((ds._style && ds._style.color) || '#ffffff', 0.45);
      ctx.shadowBlur = 6;
    }
  },
  afterDatasetDraw(chart, args) {
    const ds = chart.data.datasets[args.index];
    if (ds && ds.label === chart.$emphasisLabel && hoveredDatasetIndex == null) {
      chart.ctx.restore();
    }
  },
};

// Right-endpoint labels: name + latest return for each visible curve, with
// vertical collision avoidance and subtle leader lines to displaced labels.
const endpointLabelPlugin = {
  id: 'endpointLabels',
  afterDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    const labels = [];

    chart.data.datasets.forEach((ds, i) => {
      const meta = chart.getDatasetMeta(i);
      if (meta.hidden || ds.hidden || !ds._raw || !meta.data || !meta.data.length) return;
      let lastIdx = -1;
      for (let k = ds._raw.length - 1; k >= 0; k -= 1) {
        if (ds._raw[k] != null) { lastIdx = k; break; }
      }
      if (lastIdx < 0 || !meta.data[lastIdx]) return;

      const ret = (ds._entry && ds._entry.cumulative_return != null)
        ? Number(ds._entry.cumulative_return)
        : (ds._raw[lastIdx] - ds._initial) / (ds._initial || 1);

      labels.push({
        i,
        anchorX: meta.data[lastIdx].x,
        anchorY: meta.data[lastIdx].y,
        y: meta.data[lastIdx].y,
        color: (ds._style && ds._style.color) || '#e5e7eb',
        text: `${shortName(ds.label)}  ${(ret * 100).toFixed(2)}%`,
      });
    });
    if (!labels.length) return;

    // Stagger overlapping labels downward, keeping >= GAP px spacing. Each
    // label keeps its original endpoint y (anchorY) so we can tell later
    // whether collision-avoidance actually moved it.
    const GAP = 20;
    // Only draw a leader line once a label has been displaced enough that the
    // connection is genuinely ambiguous; otherwise it just leaves a tiny stub.
    const LEADER_MIN_DISPLACEMENT = 7;
    labels.sort((a, b) => a.y - b.y);
    for (let k = 1; k < labels.length; k += 1) {
      if (labels[k].y - labels[k - 1].y < GAP) labels[k].y = labels[k - 1].y + GAP;
    }
    // If the stack overflows the plot bottom, shift the whole stack up.
    const overflow = labels[labels.length - 1].y - chartArea.bottom;
    if (overflow > 0) labels.forEach((lab) => { lab.y -= overflow; });

    // Labels live entirely inside the reserved right gutter so the plotted
    // line paths (which end at chartArea.right) never leave stubs under them.
    const gutterStart = chartArea.right + 6;
    const labelX = chartArea.right + 12;
    ctx.save();
    ctx.font = '600 11px Inter, system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    labels.forEach((lab) => {
      const faded = hoveredDatasetIndex != null && lab.i !== hoveredDatasetIndex;
      const displaced = Math.abs(lab.y - lab.anchorY) > LEADER_MIN_DISPLACEMENT;
      if (displaced) {
        // Subtle dotted leader, drawn only within the gutter (never over the
        // data line endpoint), connecting the displaced label to its curve.
        ctx.setLineDash([1, 3]);
        ctx.strokeStyle = hexToRgba(lab.color, faded ? 0.12 : 0.35);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(gutterStart, lab.anchorY);
        ctx.lineTo(labelX - 3, lab.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.fillStyle = hexToRgba(lab.color, faded ? 0.3 : 1);
      ctx.fillText(lab.text, labelX, lab.y);
    });
    ctx.restore();
  },
};

function buildCustomLegend(chart) {
  const container = document.getElementById('equityCurvesLegend');
  if (!container) return;

  const order = { team: 0, model: 1, strategy: 2, benchmark: 3 };
  const items = chart.data.datasets
    .map((ds, i) => ({ ds, i }))
    .sort((a, b) => (order[a.ds._style.kind] ?? 9) - (order[b.ds._style.kind] ?? 9));

  container.innerHTML = items.map(({ ds }) => {
    const st = ds._style;
    const hidden = hiddenSeries.has(ds.label);
    const w = Math.min(KIND_WIDTH[st.kind] || 2, 2.4);
    const dash = (st.dash && st.dash.length) ? st.dash.join(',') : '';
    const stroke = hidden ? 'rgba(148,163,184,0.4)' : st.color;
    return `
      <button class="legend-item${hidden ? ' legend-hidden' : ''}" data-label="${ds.label.replace(/"/g, '&quot;')}">
        <svg class="legend-sample" width="26" height="10" viewBox="0 0 26 10">
          <line x1="1" y1="5" x2="25" y2="5" stroke="${stroke}" stroke-width="${w}"
            ${dash ? `stroke-dasharray="${dash}"` : ''} stroke-linecap="round" />
        </svg>
        <span class="legend-label">${shortName(ds.label)}</span>
      </button>`;
  }).join('');

  container.querySelectorAll('.legend-item').forEach((el) => {
    el.addEventListener('click', () => {
      const label = el.dataset.label;
      if (hiddenSeries.has(label)) hiddenSeries.delete(label);
      else hiddenSeries.add(label);
      updateCurvePickerCount();
      if (document.getElementById('curvePickerTrigger')?.getAttribute('aria-expanded') === 'true') {
        renderCurvePicker();
      }
      renderEquityCurvesChart();
    });
  });
}

async function renderEquityCurvesChart() {
  if (!equityCurvesData) return;

  const canvas = document.getElementById('equityCurvesChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const { times, days, curves, initials } = equityCurvesData;
  const axisLabels = times || days;
  const orderedEntries = (leaderboardPayload?.entries || []);
  const emphasisLabel = getEmphasisLabel(orderedEntries);

  const datasets = [];
  orderedEntries.forEach((entry) => {
    const label = entry.model || entry.team_name;
    const raw = curves[label];
    if (!raw || !raw.length) return;
    const initial = initials[label] || raw[0] || 10000;
    const style = getSeriesStyle(label, entry);

    datasets.push({
      label,
      data: transformLeaderboardChartData(raw, currentChartView, initial),
      _raw: raw,
      _initial: initial,
      _entry: entry,
      _style: style,
      borderColor: style.color,
      backgroundColor: 'transparent',
      borderDash: style.dash || [],
      borderCapStyle: 'round',
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.1,
      fill: false,
      // Series use different hour grids (e.g. SPY :30 vs LLM :00). On a shared
      // axis that leaves many nulls; span across them so each curve still draws.
      spanGaps: true,
      hidden: hiddenSeries.has(label),
    });
  });

  const baseInitial = (datasets[0] && datasets[0]._initial) || 10000;
  const isMoney = currentChartView === 'absolute';

  if (equityCurvesChartInstance) {
    equityCurvesChartInstance.destroy();
  }

  equityCurvesChartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels: axisLabels, datasets },
    plugins: [selectedGlowPlugin, endpointLabelPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { right: 120, top: 8 } },
      // 'nearest' across BOTH axes (no axis:'x') so hover targets the single
      // line under the cursor instead of every series sharing that x-position.
      interaction: { mode: 'nearest', intersect: false },
      onHover(event, _els, chart) {
        let idx = null;
        const hits = chart.getElementsAtEventForMode(event, 'nearest', { intersect: false }, true);
        if (hits.length) idx = hits[0].datasetIndex;
        if (idx !== hoveredDatasetIndex) {
          hoveredDatasetIndex = idx;
          styleDatasets(chart);
          chart.update('none');
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'nearest',
          intersect: false,
          position: 'nearest',
          backgroundColor: 'rgba(15, 23, 42, 0.96)',
          borderColor: 'rgba(148, 163, 184, 0.25)',
          borderWidth: 1,
          titleColor: '#e5e7eb',
          bodyColor: '#cbd5e1',
          padding: 10,
          displayColors: false,
          callbacks: {
            title(items) {
              if (!items.length) return '';
              return formatChartTooltipLabel(items[0].label);
            },
            label(context) {
              const ds = context.dataset;
              const idx = context.dataIndex;
              const equity = (ds._raw && ds._raw[idx]) || 0;
              const ret = (equity - ds._initial) / (ds._initial || 1);
              const entry = ds._entry || {};
              const lines = [
                ds.label,
                `Return: ${(ret * 100).toFixed(2)}%`,
                `Value: $${formatLeaderboardNumber(equity)}`,
                `Rank: ${entry.rank ?? '—'} / ${leaderboardPayload?.total_entries || '—'}`,
              ];

              const benchDs = context.chart.data.datasets.find((d) => d.label === selectedBenchmarkLabel);
              if (benchDs && benchDs.label !== ds.label && benchDs._raw && benchDs._raw[idx] != null) {
                const benchRet = (benchDs._raw[idx] - benchDs._initial) / (benchDs._initial || 1);
                const diff = (ret - benchRet) * 100;
                const sign = diff >= 0 ? '+' : '';
                lines.push(`vs ${shortName(selectedBenchmarkLabel)}: ${sign}${diff.toFixed(2)}%`);
              }
              return lines;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#6b7280',
            maxRotation: 0,
            minRotation: 0,
            autoSkip: true,
            maxTicksLimit: 7,
            callback(value) {
              return formatShortDate(this.getLabelForValue(value));
            },
          },
          grid: { color: 'rgba(148, 163, 184, 0.05)', drawTicks: false },
        },
        y: {
          ticks: {
            color: '#9ca3af',
            callback(value) {
              if (isMoney) return `$${formatLeaderboardNumber(value)}`;
              return `${(value * 100).toFixed(1)}%`;
            },
          },
          grid: {
            color(c) {
              const v = c.tick.value;
              const isRef = isMoney ? Math.abs(v - baseInitial) < 1 : Math.abs(v) < 1e-9;
              return isRef ? 'rgba(148, 163, 184, 0.45)' : 'rgba(148, 163, 184, 0.08)';
            },
            lineWidth(c) {
              const v = c.tick.value;
              const isRef = isMoney ? Math.abs(v - baseInitial) < 1 : Math.abs(v) < 1e-9;
              return isRef ? 1.4 : 1;
            },
          },
        },
      },
    },
  });

  equityCurvesChartInstance.$emphasisLabel = emphasisLabel;
  styleDatasets(equityCurvesChartInstance);
  equityCurvesChartInstance.update('none');

  if (!canvasLeaveBound) {
    canvas.addEventListener('mouseleave', () => {
      if (hoveredDatasetIndex != null && equityCurvesChartInstance) {
        hoveredDatasetIndex = null;
        styleDatasets(equityCurvesChartInstance);
        equityCurvesChartInstance.update('none');
      }
    });
    canvasLeaveBound = true;
  }

  buildCustomLegend(equityCurvesChartInstance);
}

function transformLeaderboardChartData(curveValues, viewType, initialValue) {
  const base = initialValue || 10000;
  if (viewType === 'absolute') {
    return curveValues.slice();
  }
  return curveValues.map((v) => (v == null ? null : (v - base) / base));
}

function displayLeaderboardError(message) {
  const tbody = document.getElementById('leaderboardTableBody');
  if (tbody) {
    // `escapeHtml` is a global from app.js, which app.html loads first — the same
    // contract js/leaderboard.js already relies on for API/API_BASE.
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; padding: 30px; color: var(--danger-color);">Error: ${escapeHtml(message)}</td></tr>`;
  }
}

window.loadLeaderboardData = loadLeaderboardData;
window.selectLeaderboardTeam = selectLeaderboardTeam;
