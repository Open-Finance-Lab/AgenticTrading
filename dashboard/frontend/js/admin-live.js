/** /admin live-operations row. Renders the live stats and nothing else (design D16, D17). */
(function () {
  'use strict';

  const STATS_PATH = '/api/admin/stats';
  // Two caveats the row must carry (design D17): the slot ledger is per-process,
  // and the ceiling is the parsed MAX_ACTIVE_DASHBOARD_BACKTESTS, not a fiction.
  const CAVEAT = "Counters are this instance's process state; a second replica would under-report them.";
  const state = { data: null };

  function shell() {
    return window.AdminShell;
  }

  function tile(label, value, meta, extraClass) {
    const s = shell();
    const article = s.el('article', extraClass ? `snapshot-tile ${extraClass}` : 'snapshot-tile');
    article.appendChild(s.el('span', 'snapshot-label', label));
    article.appendChild(s.el('strong', 'snapshot-value', value));
    if (meta !== undefined) article.appendChild(s.el('small', 'snapshot-meta', meta));
    return article;
  }

  function renderTiles(stats) {
    const s = shell();
    const ceiling = Number(stats?.max_active_dashboard_backtests);
    const root = s.el('div');
    root.appendChild(tile('Total users', s.formatNumber(stats?.users)));
    root.appendChild(tile('Total agents', s.formatNumber(stats?.agents)));
    root.appendChild(tile(
      'Backtests running · this instance',
      s.formatNumber(stats?.active_dashboard_backtests),
      Number.isFinite(ceiling) ? `of ${s.formatNumber(ceiling)} slots on this instance` : s.DASH,
      'runs'
    ));
    return root;
  }

  function paint(stats) {
    const s = shell();
    const tiles = document.getElementById('liveTiles');
    if (tiles) {
      s.clear(tiles);
      Array.from(renderTiles(stats).children).forEach((node) => tiles.appendChild(node));
    }
    const updated = document.getElementById('liveUpdated');
    if (updated) updated.textContent = `Updated ${s.formatTimestamp(new Date().toISOString())}`;
  }

  async function load() {
    const s = shell();
    const panel = document.querySelector('[data-panel="live"]');
    const seq = s.nextSeq('live');
    s.setPanelState(panel, { busy: true });
    try {
      const stats = await s.request(STATS_PATH);
      if (!s.isCurrent('live', seq)) return;
      state.data = stats;
      paint(stats);
      s.setPanelState(panel, { busy: false });
    } catch (error) {
      if (!s.isCurrent('live', seq)) return;
      if (await s.handleAccessLost(error)) return;
      s.setPanelState(panel, { busy: false, error: s.SECTION_UNAVAILABLE, stale: Boolean(state.data) });
    }
  }

  document.addEventListener('admin:route', (event) => {
    if (event.detail?.route === 'overview') load();
  });
  document.addEventListener('admin:retry', (event) => {
    if (event.detail?.panel === 'live') load();
  });

  window.AdminLive = { renderTiles, load, STATS_PATH, CAVEAT };
})();
