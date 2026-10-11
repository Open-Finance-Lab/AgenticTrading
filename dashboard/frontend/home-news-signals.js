// Home panel: FinSearch latest news (left) + identified signals (right).
// Data: GET /api/news/signals (server-side proxy; 420s TTL — above this 300s
// poll, so a re-poll lands in-cache). Poll only while the Home view is visible.
// Reuses app.js globals: escapeHtml (attribute-safe), API (fetch wrapper),
// API_BASE; and market-events/marketEventRelativeTime.js for relative times.
// Load order: this script tag must come after app.js and market-events/*.
(function () {
  const POLL_MS = 5 * 60 * 1000;
  let timer = null;

  // escapeHtml (app.js) neutralizes quotes/angle-brackets but does NOT scheme-validate
  // a URL — a `javascript:` value would survive it unchanged and execute on click.
  // The producer's clean_text does not validate scheme either, so this is the only
  // guard: gate every outbound link to http(s) BEFORE it reaches an href.
  function safeUrl(raw) {
    const s = String(raw == null ? '' : raw).trim();
    return /^https?:\/\//i.test(s) ? s : '#';
  }

  function relTime(publishedEpochSeconds) {
    // Shared helper takes ms/Date-parseable input; producer sends epoch seconds.
    return window.formatMarketEventRelativeTime
      ? window.formatMarketEventRelativeTime(publishedEpochSeconds * 1000, new Date())
      : '';
  }

  // A frozen demo snapshot (`demo: true`, api/routers/news.py) gets a fixed
  // date: a relative time keeps growing on a paused feed. Mirrors
  // home-page.js homeFeedTime.
  function feedTime(publishedEpochSeconds, demo) {
    if (!demo) return relTime(publishedEpochSeconds);
    const ms = Number(publishedEpochSeconds) * 1000;
    if (!Number.isFinite(ms) || ms <= 0) return '';
    return new Date(ms).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }

  function render(payload) {
    const updated = document.getElementById('nnsUpdated');
    const badge = document.getElementById('nnsStatusBadge');
    const overview = document.getElementById('nnsOverview');
    const feedList = document.getElementById('nnsFeedList');
    const sigList = document.getElementById('nnsSignalsList');
    if (!feedList || !sigList) return;

    if (payload.status === 'unavailable') {
      overview.textContent = 'News signals are currently unavailable.';
      feedList.innerHTML = sigList.innerHTML = '';
      updated.textContent = '';
      badge.hidden = true;
      return;
    }
    // Coerce like the sentiment score below: the value is passed through from
    // the producer unvalidated, so a string/non-number must not throw and mask
    // an otherwise-valid payload as "unavailable". null stays "" as before.
    const staleness = Number(payload.staleness_hours);
    // `demo`: a frozen snapshot served while FinSearch is paused. It must say
    // so — status 'ok' and real stories otherwise read as a live feed.
    if (payload.demo) {
      updated.textContent = 'Demo news';
    } else {
      updated.textContent = (payload.staleness_hours != null && Number.isFinite(staleness))
        ? `Updated ${staleness.toFixed(1)}h ago` : '';
    }
    badge.hidden = payload.status !== 'degraded';
    if (!badge.hidden) badge.textContent = `degraded: ${payload.status_reason || 'partial data'}`;
    overview.textContent = payload.news_overview || '';

    feedList.innerHTML = (payload.feed || []).map(item => {
      // Phase B's raw-items feed can carry a null ticker (general-market story
      // with no symbol); Phase A's signals-derived feed always had one. Build
      // the meta line from only the non-empty segments so a missing ticker (or
      // an absent relTime helper) collapses cleanly instead of leaving a
      // dangling " · " separator. relTime is a controlled string, not producer
      // data, so it stays unescaped like before.
      const meta = [escapeHtml(item.source), escapeHtml(item.ticker), feedTime(item.published, payload.demo)]
        .filter(Boolean).join(' · ');
      return `
      <li class="nns-item">
        <a href="${escapeHtml(safeUrl(item.url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.headline)}</a>
        <span class="nns-meta">${meta}</span>
      </li>`;
    }).join('') || '<li class="nns-empty">No qualifying news today.</li>';

    sigList.innerHTML = Object.entries(payload.signals || {}).map(([sym, s]) => `
      <li class="nns-item nns-signal nns-${escapeHtml(s.sentiment)}">
        <span class="nns-chip">${escapeHtml(s.sentiment)}</span>
        <strong>${escapeHtml(sym)}</strong> <span class="nns-score">${Number(s.sentiment_score).toFixed(2)}</span>
        <span class="nns-rationale">${escapeHtml(s.rationale || '')}</span>
        <a class="nns-src" href="${escapeHtml(safeUrl(s.url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(s.source)}</a>
      </li>`).join('') || '<li class="nns-empty">No directional reads.</li>';
  }

  async function load() {
    try {
      render(await API.get(`${API_BASE}/api/news/signals`));
    } catch (e) {
      render({ status: 'unavailable' });
    }
  }

  window.newsSignalsPanel = {
    onShow() { load(); if (!timer) timer = setInterval(load, POLL_MS); },
    onHide() { if (timer) { clearInterval(timer); timer = null; } },
  };
})();
