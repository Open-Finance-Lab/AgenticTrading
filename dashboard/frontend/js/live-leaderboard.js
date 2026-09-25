// ============================================================================
// LIVE TRADING LEADERBOARD (calendar-month board — Phase 0)
// Isolated from contest/daily: own panel, own chart, own payload.
// ============================================================================

let livePayload = null;
let liveChartInstance = null;
let liveChartView = 'absolute';
let liveHiddenSeries = new Set();
let liveListenersInitialized = false;
let liveSortKey = 'rank';
let liveSortDir = 'asc';
let liveSelectedId = null;
let livePickerExpanded = new Set();

const LIVE_STYLES = {
  SPY: { color: '#CBD5E1', kind: 'benchmark', dash: [2, 4] },
  DJIA: { color: '#94A3B8', kind: 'benchmark', dash: [8, 4, 2, 4] },
  'Buy & Hold': { color: '#38BDF8', kind: 'strategy', dash: [10, 6] },
  'Mean-Variance': { color: '#C084FC', kind: 'strategy', dash: [10, 6] },
  'Equal-Weight': { color: '#4ADE80', kind: 'strategy', dash: [10, 6] },
};
const LIVE_MODEL_PALETTE = ['#FBBF24', '#FB923C', '#F472B6', '#A78BFA', '#34D399', '#F87171', '#22D3EE'];
const liveModelColorMap = {};

function liveIsModel(entry) {
  return !!(entry && (entry.is_model || entry.team_badge === 'Model'));
}

function liveSeriesLabel(entry) {
  return entry.model || entry.team_name || entry.entry_id || '';
}

function liveModelColor(id) {
  const key = String(id);
  if (!liveModelColorMap[key]) {
    const idx = Object.keys(liveModelColorMap).length % LIVE_MODEL_PALETTE.length;
    liveModelColorMap[key] = LIVE_MODEL_PALETTE[idx];
  }
  return liveModelColorMap[key];
}

function liveSeriesStyle(entry) {
  const label = liveSeriesLabel(entry);
  if (liveIsModel(entry)) {
    return { color: liveModelColor(entry.entry_id || label), kind: 'model', dash: [] };
  }
  return LIVE_STYLES[label] || { color: '#94A3B8', kind: 'strategy', dash: [8, 4] };
}

function liveKind(entry) {
  if (liveIsModel(entry)) return 'model';
  const preset = LIVE_STYLES[liveSeriesLabel(entry)];
  return preset ? preset.kind : 'strategy';
}

function liveFilterCategory(entry) {
  const kind = liveKind(entry);
  if (kind === 'model') return 'model';
  if (kind === 'benchmark') return 'index';
  return 'baseline';
}

function liveFormatBadge(badge) {
  const raw = String(badge || '').trim();
  if (!raw || raw === 'Baseline' || raw === 'Strategy') return 'Baseline Strategy';
  if (raw === 'Index') return 'Market Index';
  return raw;
}

function liveFormatMoney(num) {
  return Number(num || 0).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function liveTimeKey(ts) {
  const s = String(ts || '');
  if (s.length >= 16 && s[10] === 'T') return s.slice(0, 16);
  if (s.length >= 10) return s.slice(0, 10);
  return s;
}

function liveAxisTick(label) {
  const m = String(label).match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (!m) return '';
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${months[Number(m[2]) - 1]} ${Number(m[3])}`;
}

function liveTooltipLabel(ts) {
  const m = String(ts).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!m) return String(ts || '');
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const hour = Number(m[4]);
  const ampm = hour >= 12 ? 'PM' : 'AM';
  const h12 = hour % 12 || 12;
  return `${months[Number(m[2]) - 1]} ${Number(m[3])}, ${h12}:00 ${ampm} ET`;
}

function liveSessionCopy(state) {
  if (state === 'rth') return 'Cash session open';
  if (state === 'preopen') return 'Before the open';
  if (state === 'weekend') return 'Weekend';
  return 'Cash session closed';
}

function isLiveBoardVisible() {
  const view = document.getElementById('liveLeaderboardView');
  if (!view || view.style.display === 'none' || view.offsetParent === null) return false;
  const activeTab = document.querySelector('.competition-subtabs .subtab-btn.active');
  return activeTab?.dataset.competitionTab === 'live';
}

async function loadLiveLeaderboardData() {
  const tbody = document.getElementById('liveLeaderboardTableBody');
  try {
    const url = `${API_BASE}/api/v1/leaderboard?period=live&t=${Date.now()}`;
    livePayload = await API.get(url);
    liveHiddenSeries = new Set();
    (livePayload.entries || []).forEach((entry) => {
      const has = (entry.equity_curve || []).some((pt) => pt && pt.equity != null);
      if (!has) liveHiddenSeries.add(liveSeriesLabel(entry));
    });
    updateLiveHeader(livePayload);
    populateLiveTable();
    renderLiveCurvePicker();
    if (!liveListenersInitialized) {
      initLiveLeaderboardListeners();
      liveListenersInitialized = true;
    }
    renderLiveChart();
  } catch (error) {
    console.error('Error loading live leaderboard:', error);
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--danger-color);">Error: ${escapeHtml(error.message)}</td></tr>`;
    }
  }
}

function updateLiveHeader(payload) {
  const status = payload.live_status || {};
  const window = payload.window || {};
  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };

  setText('liveBoardTitle', payload.board_title || 'Live Trading Leaderboard');
  setText('livePhaseLabel', payload.phase_label || 'Season 0');
  setText('liveWindowLabel', window.label || '—');
  setText('liveUpdatedLabel', payload.updated_at
    ? new Date(payload.updated_at).toLocaleString()
    : '—');
  setText('liveLeaderLabel', payload.leader || '—');
  const elapsed = Number(status.trading_days_elapsed) || 0;
  const total = Number(status.trading_days_total) || 0;
  setText('liveProgressLabel', total ? `Day ${elapsed} of ${total}` : '—');
  setText('liveStandingsTitle', payload.standings_label || 'Ranking');

  const subtitle = document.getElementById('liveBoardSubtitle');
  if (subtitle) {
    const month = status.month || '';
    subtitle.textContent = month
      ? `${month} · paper trading · hourly US cash session`
      : 'Paper trading · hourly US cash session';
  }

  const badge = document.getElementById('liveStatusBadge');
  if (badge) {
    badge.textContent = liveSessionCopy(status.session_state);
    badge.classList.toggle('live', status.session_state === 'rth');
    badge.classList.toggle('upcoming', status.session_state !== 'rth');
  }

  const banner = document.getElementById('livePhaseBanner');
  if (banner) {
    if (status.freeze_error) {
      banner.hidden = false;
      banner.textContent = `Could not refresh freeze-window bars (${status.frozen_through || 'T-1'}). Showing cached live-month runs if any exist.`;
    } else if (!status.has_prints) {
      banner.hidden = false;
      banner.textContent = (
        `Waiting on freeze-window bars through ${status.frozen_through || 'T-1'}. ` +
        `The axis is ${window.label || 'this month'}; curves appear after Alpaca hourly data is available.`
      );
    } else if ((Number(status.models_cached) || 0) === 0) {
      banner.hidden = false;
      banner.textContent = (
        `Baselines frozen through ${status.frozen_through}. ` +
        `Competition model curves load from this month's freeze snapshot; they are not computed on page load.`
      );
    } else {
      banner.hidden = true;
      banner.textContent = '';
    }
  }

  const zone = document.getElementById('liveZoneCaption');
  if (zone) {
    const liveDay = status.live_day ? `live tail ${status.live_day}` : 'no live tail';
    zone.textContent = `Frozen through ${status.frozen_through || '—'} · ${liveDay} · future days have no points`;
  }

  const countEl = document.getElementById('liveCurvePickerCount');
  if (countEl) {
    const n = (payload.entries || []).length;
    countEl.textContent = n ? `${n} selected` : '0 selected';
  }
}

function liveSortedEntries() {
  const entries = (livePayload?.entries || []).slice();
  const dir = liveSortDir === 'asc' ? 1 : -1;
  const num = (v) => Number(v) || 0;
  entries.sort((a, b) => {
    let cmp = 0;
    switch (liveSortKey) {
      case 'value': cmp = num(a.portfolio_value) - num(b.portfolio_value); break;
      case 'return': cmp = num(a.cumulative_return) - num(b.cumulative_return); break;
      case 'sharpe': cmp = num(a.sharpe_ratio) - num(b.sharpe_ratio); break;
      case 'dd': cmp = Math.abs(num(a.max_drawdown)) - Math.abs(num(b.max_drawdown)); break;
      default: cmp = num(a.rank) - num(b.rank);
    }
    return cmp * dir;
  });
  return entries;
}

function populateLiveTable() {
  const tbody = document.getElementById('liveLeaderboardTableBody');
  if (!tbody) return;
  const rows = liveSortedEntries();
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text-secondary);">No live-month entries configured.</td></tr>`;
    return;
  }
  const hasPrints = Boolean(livePayload?.live_status?.has_prints);
  tbody.innerHTML = rows.map((entry) => {
    const safeId = escapeHtml(String(entry.entry_id || '').replace(/\\/g, '\\\\').replace(/'/g, "\\'"));
    const label = escapeHtml(liveSeriesLabel(entry));
    const ret = Number(entry.cumulative_return || 0);
    const retClass = ret >= 0 ? 'return-positive' : 'return-negative';
    const dd = (Math.abs(Number(entry.max_drawdown || 0)) * 100).toFixed(2);
    const valueCell = `$${liveFormatMoney(hasPrints ? entry.portfolio_value : entry.initial_equity)}`;
    const selected = liveSelectedId === entry.entry_id ? ' is-selected' : '';
    return `
      <tr class="${selected}" data-live-entry="${safeId}">
        <td class="rank-cell">${hasPrints ? escapeHtml(entry.rank) : '—'}</td>
        <td>
          <div class="team-name-badge">
            <span>${label}</span>
            <span class="team-badge">${escapeHtml(liveFormatBadge(entry.team_badge))}</span>
          </div>
        </td>
        <td style="text-align:right;font-family:var(--font-mono);">${valueCell}</td>
        <td style="text-align:right;" class="${retClass}">${hasPrints ? `${(ret * 100).toFixed(2)}%` : '—'}</td>
        <td style="text-align:right;font-family:var(--font-mono);">${hasPrints ? Number(entry.sharpe_ratio || 0).toFixed(2) : '—'}</td>
        <td style="text-align:right;font-family:var(--font-mono);">${hasPrints ? `${dd}%` : '—'}</td>
      </tr>`;
  }).join('');

  document.querySelectorAll('#liveLeaderboardView th.sortable').forEach((th) => {
    const active = th.dataset.sort === liveSortKey;
    th.classList.toggle('is-sorted', active);
    const arrow = th.querySelector('.sort-arrow');
    if (arrow) arrow.textContent = active ? (liveSortDir === 'asc' ? '↑' : '↓') : '';
  });
}

function renderLiveDetail(entry) {
  const host = document.getElementById('liveSelectedDetail');
  if (!host) return;
  if (!entry) {
    host.innerHTML = '<div class="no-selection">Click a row</div>';
    return;
  }
  const hasPrints = Boolean(livePayload?.live_status?.has_prints);
  const ret = Number(entry.cumulative_return || 0);
  const retColor = ret >= 0 ? 'var(--success-color)' : 'var(--danger-color)';
  host.innerHTML = `
    <div class="team-detail-row"><span class="team-detail-label">Entry</span><span class="team-detail-value">${escapeHtml(liveSeriesLabel(entry))}</span></div>
    <div class="team-detail-row"><span class="team-detail-label">Type</span><span class="team-detail-value">${escapeHtml(liveFormatBadge(entry.team_badge))}</span></div>
    <div class="team-detail-row"><span class="team-detail-label">Value</span><span class="team-detail-value">${hasPrints ? `$${liveFormatMoney(entry.portfolio_value)}` : 'Awaiting freeze'}</span></div>
    <div class="team-detail-row"><span class="team-detail-label">Return</span><span class="team-detail-value" style="color:${retColor};">${hasPrints ? `${(ret * 100).toFixed(2)}%` : '—'}</span></div>
    <div class="team-detail-row"><span class="team-detail-label">Printed hours</span><span class="team-detail-value">${livePayload?.live_status?.printed_count ?? 0}</span></div>
  `;
}

function livePickerGroups(entries) {
  const buckets = { model: [], baseline: [], index: [] };
  (entries || []).forEach((entry) => {
    const cat = liveFilterCategory(entry);
    if (buckets[cat]) buckets[cat].push(entry);
  });
  return [
    { id: 'model', title: 'Models', entries: buckets.model },
    { id: 'baseline', title: 'Baseline Strategies', entries: buckets.baseline },
    { id: 'index', title: 'Market Indices', entries: buckets.index },
  ].filter((g) => g.entries.length);
}

function renderLiveCurvePicker() {
  const body = document.getElementById('liveCurvePickerBody');
  const countEl = document.getElementById('liveCurvePickerCount');
  if (!body) return;
  const entries = livePayload?.entries || [];
  const groups = livePickerGroups(entries);
  const visible = entries.filter((e) => !liveHiddenSeries.has(liveSeriesLabel(e))).length;
  if (countEl) countEl.textContent = entries.length ? `${visible} selected` : '0 selected';

  body.innerHTML = groups.map((group) => {
    const expanded = livePickerExpanded.has(group.id);
    const vis = group.entries.filter((e) => !liveHiddenSeries.has(liveSeriesLabel(e))).length;
    const children = group.entries.map((entry, idx) => {
      const label = liveSeriesLabel(entry);
      const checked = !liveHiddenSeries.has(label);
      const style = liveSeriesStyle(entry);
      return `
        <label class="curve-picker-item">
          <input type="checkbox" data-live-group="${group.id}" data-live-idx="${idx}" ${checked ? 'checked' : ''}>
          <span class="curve-picker-swatch" style="background:${style.color}"></span>
          <span>${escapeHtml(label)}</span>
        </label>`;
    }).join('');
    return `
      <div class="curve-picker-group">
        <div class="curve-picker-group-head">
          <button type="button" class="curve-picker-group-toggle" data-live-group-toggle="${group.id}">${vis === group.entries.length ? 'All' : vis ? 'Some' : 'None'}</button>
          <button type="button" class="curve-picker-group-title-btn" data-live-group-expand="${group.id}">
            <span class="curve-picker-group-title">${escapeHtml(group.title)} (${vis}/${group.entries.length})</span>
            <span class="curve-picker-group-chevron">${expanded ? '▾' : '›'}</span>
          </button>
        </div>
        <div class="curve-picker-children${expanded ? '' : ' is-collapsed'}">${children}</div>
      </div>`;
  }).join('');
}

function liveBuildSeries() {
  const axis = (livePayload?.chart_axis || []).map(liveTimeKey);
  const entries = livePayload?.entries || [];
  const curves = {};
  const initials = {};
  entries.forEach((entry) => {
    const label = liveSeriesLabel(entry);
    const byTime = {};
    (entry.equity_curve || []).forEach((pt) => {
      const key = liveTimeKey(pt.timestamp);
      if (!key || pt.equity == null) return;
      byTime[key] = Number(pt.equity);
    });
    curves[label] = axis.map((t) => (t in byTime ? byTime[t] : null));
    initials[label] = Number(entry.initial_equity) || 10000;
  });
  return { axis, curves, initials };
}

const liveNowCursorPlugin = {
  id: 'liveNowCursor',
  afterDraw(chart) {
    const idx = chart.$nowIndex;
    if (idx == null || idx < 0) return;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    if (!xScale || !yScale) return;
    const x = xScale.getPixelForValue(idx);
    const { ctx } = chart;
    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = 'rgba(96, 165, 250, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.moveTo(x, yScale.top);
    ctx.lineTo(x, yScale.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

const liveZonePlugin = {
  id: 'liveZones',
  beforeDraw(chart) {
    const zones = chart.$zones;
    if (!zones) return;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    if (!xScale || !yScale) return;
    const { ctx } = chart;
    const paint = (from, to, color) => {
      if (from == null || to == null || to < from) return;
      const x1 = xScale.getPixelForValue(from);
      const x2 = xScale.getPixelForValue(to);
      ctx.fillStyle = color;
      ctx.fillRect(Math.min(x1, x2), yScale.top, Math.abs(x2 - x1), yScale.bottom - yScale.top);
    };
    ctx.save();
    paint(zones.frozenFrom, zones.frozenTo, 'rgba(148, 163, 184, 0.06)');
    paint(zones.liveFrom, zones.liveTo, 'rgba(56, 189, 248, 0.08)');
    paint(zones.futureFrom, zones.futureTo, 'rgba(15, 23, 42, 0.35)');
    ctx.restore();
  },
};

function liveZoneIndices(axis, status) {
  const frozenDay = status.frozen_through || '';
  const liveDay = status.live_day || '';
  let frozenFrom = null;
  let frozenTo = null;
  let liveFrom = null;
  let liveTo = null;
  let futureFrom = null;
  axis.forEach((ts, i) => {
    const day = String(ts).slice(0, 10);
    if (liveDay && day === liveDay) {
      if (liveFrom == null) liveFrom = i;
      liveTo = i;
    } else if (frozenDay && day <= frozenDay) {
      if (frozenFrom == null) frozenFrom = i;
      frozenTo = i;
    } else if (futureFrom == null) {
      futureFrom = i;
    }
  });
  return {
    frozenFrom,
    frozenTo,
    liveFrom,
    liveTo,
    futureFrom,
    futureTo: futureFrom == null ? null : axis.length - 1,
  };
}

function renderLiveChart() {
  const canvas = document.getElementById('liveEquityCurvesChart');
  if (!canvas || typeof Chart === 'undefined' || !livePayload) return;
  const ctx = canvas.getContext('2d');
  const { axis, curves, initials } = liveBuildSeries();
  const status = livePayload.live_status || {};
  const entries = livePayload.entries || [];
  const capital = Number(livePayload.display_capital) || 10000;

  const datasets = entries.map((entry) => {
    const label = liveSeriesLabel(entry);
    const style = liveSeriesStyle(entry);
    const raw = curves[label] || axis.map(() => null);
    const initial = initials[label] || capital;
    const data = liveChartView === 'absolute'
      ? raw.slice()
      : raw.map((v) => (v == null ? null : (v - initial) / initial));
    return {
      label,
      data,
      _raw: raw,
      borderColor: style.color,
      backgroundColor: 'transparent',
      borderDash: style.dash || [],
      borderCapStyle: 'round',
      pointRadius: 0,
      pointHoverRadius: 3,
      tension: 0.08,
      fill: false,
      spanGaps: true,
      hidden: liveHiddenSeries.has(label),
    };
  });

  if (liveChartInstance) liveChartInstance.destroy();

  const isMoney = liveChartView === 'absolute';
  liveChartInstance = new Chart(ctx, {
    type: 'line',
    data: { labels: axis, datasets },
    plugins: [liveZonePlugin, liveNowCursorPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { right: 16, top: 8 } },
      interaction: { mode: 'nearest', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.96)',
          borderColor: 'rgba(148, 163, 184, 0.25)',
          borderWidth: 1,
          displayColors: false,
          callbacks: {
            title(items) {
              return items.length ? liveTooltipLabel(items[0].label) : '';
            },
            label(context) {
              const raw = context.dataset._raw?.[context.dataIndex];
              if (raw == null) return `${context.dataset.label}: no print`;
              return `${context.dataset.label}: $${liveFormatMoney(raw)}`;
            },
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#6b7280',
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 8,
            callback(value) {
              return liveAxisTick(this.getLabelForValue(value));
            },
          },
          grid: { color: 'rgba(148, 163, 184, 0.05)', drawTicks: false },
        },
        y: {
          suggestedMin: isMoney ? capital * 0.9 : -0.05,
          suggestedMax: isMoney ? capital * 1.1 : 0.05,
          ticks: {
            color: '#9ca3af',
            callback(value) {
              if (isMoney) return `$${liveFormatMoney(value)}`;
              return `${(value * 100).toFixed(1)}%`;
            },
          },
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
        },
      },
    },
  });
  liveChartInstance.$nowIndex = status.now_index;
  liveChartInstance.$zones = liveZoneIndices(axis, status);
  liveChartInstance.update('none');
  buildLiveLegend();
}

function buildLiveLegend() {
  const host = document.getElementById('liveEquityCurvesLegend');
  if (!host || !liveChartInstance) return;
  host.innerHTML = (livePayload?.entries || []).map((entry, i) => {
    const label = liveSeriesLabel(entry);
    const style = liveSeriesStyle(entry);
    const hidden = liveHiddenSeries.has(label);
    return `<button type="button" class="legend-item${hidden ? ' legend-hidden' : ''}" data-live-legend="${i}">
      <span class="legend-swatch" style="background:${style.color};opacity:${hidden ? 0.3 : 1}"></span>
      <span>${escapeHtml(label)}</span>
    </button>`;
  }).join('');
}

function initLiveLeaderboardListeners() {
  const root = document.getElementById('liveLeaderboardView');
  if (!root) return;

  root.querySelectorAll('.live-view-toggle-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      root.querySelectorAll('.live-view-toggle-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      liveChartView = btn.dataset.view === 'absolute' ? 'absolute' : 'cumulative';
      renderLiveChart();
    });
  });

  root.querySelectorAll('th.sortable').forEach((th) => {
    th.addEventListener('click', () => {
      const key = th.dataset.sort;
      if (liveSortKey === key) liveSortDir = liveSortDir === 'asc' ? 'desc' : 'asc';
      else {
        liveSortKey = key;
        liveSortDir = key === 'dd' || key === 'rank' ? 'asc' : 'desc';
      }
      populateLiveTable();
    });
  });

  document.getElementById('liveLeaderboardTableBody')?.addEventListener('click', (e) => {
    const row = e.target.closest('tr[data-live-entry]');
    if (!row) return;
    liveSelectedId = row.getAttribute('data-live-entry');
    populateLiveTable();
    const entry = (livePayload?.entries || []).find((en) => en.entry_id === liveSelectedId);
    renderLiveDetail(entry);
  });

  const trigger = document.getElementById('liveCurvePickerTrigger');
  const menu = document.getElementById('liveCurvePickerMenu');
  trigger?.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = trigger.getAttribute('aria-expanded') === 'true';
    if (menu) menu.hidden = open;
    trigger.setAttribute('aria-expanded', open ? 'false' : 'true');
    document.getElementById('liveCurvePicker')?.classList.toggle('is-open', !open);
    if (!open) renderLiveCurvePicker();
  });
  menu?.addEventListener('click', (e) => e.stopPropagation());
  document.getElementById('liveCurvePickerClear')?.addEventListener('click', (e) => {
    e.stopPropagation();
    (livePayload?.entries || []).forEach((entry) => liveHiddenSeries.add(liveSeriesLabel(entry)));
    renderLiveCurvePicker();
    renderLiveChart();
  });
  document.getElementById('liveCurvePickerBody')?.addEventListener('click', (e) => {
    const expand = e.target.closest('[data-live-group-expand]');
    if (expand) {
      const id = expand.getAttribute('data-live-group-expand');
      if (livePickerExpanded.has(id)) livePickerExpanded.delete(id);
      else livePickerExpanded.add(id);
      renderLiveCurvePicker();
      return;
    }
    const toggle = e.target.closest('[data-live-group-toggle]');
    if (toggle) {
      const id = toggle.getAttribute('data-live-group-toggle');
      const group = livePickerGroups(livePayload?.entries || []).find((g) => g.id === id);
      if (!group) return;
      const allOn = group.entries.every((en) => !liveHiddenSeries.has(liveSeriesLabel(en)));
      group.entries.forEach((en) => {
        const label = liveSeriesLabel(en);
        if (allOn) liveHiddenSeries.add(label);
        else liveHiddenSeries.delete(label);
      });
      renderLiveCurvePicker();
      renderLiveChart();
    }
  });
  document.getElementById('liveCurvePickerBody')?.addEventListener('change', (e) => {
    const input = e.target;
    if (!(input instanceof HTMLInputElement) || input.dataset.liveIdx == null) return;
    const group = livePickerGroups(livePayload?.entries || []).find((g) => g.id === input.dataset.liveGroup);
    const entry = group?.entries?.[Number(input.dataset.liveIdx)];
    if (!entry) return;
    const label = liveSeriesLabel(entry);
    if (input.checked) liveHiddenSeries.delete(label);
    else liveHiddenSeries.add(label);
    renderLiveCurvePicker();
    renderLiveChart();
  });

  document.getElementById('liveEquityCurvesLegend')?.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-live-legend]');
    if (!btn) return;
    const entry = (livePayload?.entries || [])[Number(btn.getAttribute('data-live-legend'))];
    if (!entry) return;
    const label = liveSeriesLabel(entry);
    if (liveHiddenSeries.has(label)) liveHiddenSeries.delete(label);
    else liveHiddenSeries.add(label);
    renderLiveCurvePicker();
    renderLiveChart();
  });

  document.addEventListener('pointerdown', (e) => {
    const picker = document.getElementById('liveCurvePicker');
    if (!picker || picker.contains(e.target)) return;
    const t = document.getElementById('liveCurvePickerTrigger');
    const m = document.getElementById('liveCurvePickerMenu');
    if (m) m.hidden = true;
    t?.setAttribute('aria-expanded', 'false');
    picker.classList.remove('is-open');
  });
}

window.loadLiveLeaderboardData = loadLiveLeaderboardData;
window.isLiveBoardVisible = isLiveBoardVisible;
