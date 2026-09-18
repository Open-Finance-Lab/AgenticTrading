/** /admin live ticker — the same strip the app header carries (design D5 layout).
 *
 * Ported from app.js's MAG-7 marquee so the absorbed console keeps the chrome
 * users see on /app. Self-contained on purpose: it owns its fetch, its render
 * and its scroll loop, and shares nothing with the analytics modules.
 */
(function () {
  'use strict';

  const MAG7_TICKER_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'];
  const TICKER_SCROLL_PX_PER_SEC = 55;
  const TICKER_ESTIMATED_ITEM_WIDTH = 140;
  const TICKER_TIMEOUT_MS = 45000;

  const API_BASE = (
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ) ? window.location.origin : '';

  const scroll = { raf: null, offset: 0, setWidth: 0, lastTime: 0, paused: false, controlsBound: false, resizeTimer: null };
  let latestQuotes = [];

  function element(id) {
    return document.getElementById(id);
  }

  function marqueeWidth() {
    return element('tickerMarquee')?.clientWidth || window.innerWidth;
  }

  function sortQuotes(quotes) {
    const order = new Map(MAG7_TICKER_SYMBOLS.map((symbol, index) => [symbol, index]));
    return [...quotes].sort((a, b) => (order.get(a.symbol) ?? 99) - (order.get(b.symbol) ?? 99));
  }

  function quoteFields(quote) {
    let changeDisplay = '--';
    let changeClass = '';
    let tooltip = 'Data unavailable';
    let sparkPath = 'M0,8 L5,6 L10,7 L15,4 L20,5 L25,3 L30,5';

    if (quote.changePercent !== null && quote.changePercent !== undefined) {
      const changeSign = quote.changePercent >= 0 ? '+' : '';
      changeDisplay = `${changeSign}${quote.changePercent.toFixed(2)}%`;
      changeClass = quote.changePercent >= 0 ? 'positive' : 'negative';
      tooltip = 'Change vs previous close';
      sparkPath = quote.changePercent >= 0
        ? 'M0,10 L5,8 L10,9 L15,6 L20,7 L25,4 L30,3'
        : 'M0,3 L5,5 L10,4 L15,7 L20,6 L25,9 L30,10';
    }

    const price = quote.price != null
      ? quote.price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
      : '--';

    return { price, changeDisplay, changeClass, tooltip, sparkPath };
  }

  function buildSet(quotes, repeats) {
    const itemHtml = sortQuotes(quotes).map((quote) => {
      const fields = quoteFields(quote);
      return `
        <div class="ticker-item" data-symbol="${quote.symbol}">
          <span class="symbol">${quote.symbol}</span>
          <span class="price">${fields.price}</span>
          <span class="change ${fields.changeClass}" title="${fields.tooltip}">${fields.changeDisplay}</span>
          <svg class="ticker-chart ${fields.changeClass}" viewBox="0 0 30 12" aria-hidden="true">
            <path d="${fields.sparkPath}" stroke="currentColor" fill="none" stroke-width="1"/>
          </svg>
        </div>`;
    }).join('');
    return Array(Math.max(1, repeats)).fill(itemHtml).join('');
  }

  function setWidth(track) {
    return track.querySelector('.ticker-set')?.offsetWidth || 0;
  }

  function stopScroll() {
    if (scroll.raf !== null) {
      cancelAnimationFrame(scroll.raf);
      scroll.raf = null;
    }
  }

  function frame(now) {
    const track = element('tickerTrack');
    if (!track || track.dataset.tickerReady !== '1') {
      stopScroll();
      return;
    }
    if (!scroll.setWidth) {
      scroll.setWidth = setWidth(track);
      if (!scroll.setWidth) {
        scroll.raf = requestAnimationFrame(frame);
        return;
      }
    }
    if (!scroll.lastTime) scroll.lastTime = now;
    if (!scroll.paused) {
      const dt = Math.min(0.05, (now - scroll.lastTime) / 1000);
      scroll.offset -= TICKER_SCROLL_PX_PER_SEC * dt;
      if (scroll.offset <= -scroll.setWidth) scroll.offset += scroll.setWidth;
      track.style.transform = `translate3d(${scroll.offset}px, 0, 0)`;
    }
    scroll.lastTime = now;
    scroll.raf = requestAnimationFrame(frame);
  }

  function startScroll() {
    stopScroll();
    const track = element('tickerTrack');
    if (!track || track.dataset.tickerReady !== '1') return;
    scroll.offset = 0;
    scroll.setWidth = 0;
    scroll.lastTime = 0;
    track.style.transform = 'translate3d(0, 0, 0)';
    scroll.raf = requestAnimationFrame(frame);
  }

  function scheduleStart() {
    stopScroll();
    requestAnimationFrame(() => {
      requestAnimationFrame(() => startScroll());
    });
  }

  function bindControls() {
    if (scroll.controlsBound) return;
    scroll.controlsBound = true;
    element('tickerMarquee')?.addEventListener('mouseenter', () => { scroll.paused = true; });
    element('tickerMarquee')?.addEventListener('mouseleave', () => {
      scroll.paused = false;
      scroll.lastTime = 0;
    });
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        stopScroll();
        return;
      }
      if (element('tickerTrack')?.dataset.tickerReady === '1') scheduleStart();
    });
    window.addEventListener('resize', () => {
      clearTimeout(scroll.resizeTimer);
      scroll.resizeTimer = setTimeout(() => {
        const track = element('tickerTrack');
        if (!track || track.dataset.tickerReady !== '1') return;
        const firstSet = track.querySelector('.ticker-set');
        if (!firstSet || firstSet.offsetWidth < marqueeWidth() + 40) {
          const source = latestQuotes.length
            ? latestQuotes
            : MAG7_TICKER_SYMBOLS.map((symbol) => ({ symbol, price: null, changePercent: null }));
          track.dataset.tickerReady = '0';
          stopScroll();
          render(source);
        } else {
          scroll.setWidth = setWidth(track);
        }
      }, 200);
    });
  }

  // Patch existing tiles in place when the marquee is already running; the
  // scroll loop reads offsetWidth, so a full re-render would stutter it.
  function patchQuotes(quotes) {
    const track = element('tickerTrack');
    if (!track || track.dataset.tickerReady !== '1') return false;
    const bySymbol = new Map(quotes.map((quote) => [quote.symbol, quote]));
    track.querySelectorAll('.ticker-item[data-symbol]').forEach((item) => {
      const quote = bySymbol.get(item.dataset.symbol);
      if (!quote) return;
      const fields = quoteFields(quote);
      const priceEl = item.querySelector('.price');
      const changeEl = item.querySelector('.change');
      const chartEl = item.querySelector('.ticker-chart');
      const pathEl = item.querySelector('.ticker-chart path');
      if (priceEl) priceEl.textContent = fields.price;
      if (changeEl) {
        changeEl.textContent = fields.changeDisplay;
        changeEl.className = `change ${fields.changeClass}`.trim();
        changeEl.title = fields.tooltip;
      }
      if (chartEl) chartEl.className = `ticker-chart ${fields.changeClass}`.trim();
      if (pathEl) pathEl.setAttribute('d', fields.sparkPath);
    });
    return true;
  }

  function render(quotes) {
    const track = element('tickerTrack');
    if (!track) return;
    stopScroll();
    const width = marqueeWidth();
    // One repeat ≈ quotes.length tiles of ~140px; the set must be wider than
    // the marquee plus a seam. Division LAST: `(w+80)/n*140` reads the same
    // but multiplies after dividing — that bug once asked for 27,000 repeats
    // and shipped ~380,000 live DOM nodes to the page.
    const singlePassWidth = Math.max(quotes.length, 1) * TICKER_ESTIMATED_ITEM_WIDTH;
    let repeats = Math.max(3, Math.ceil((width + 80) / singlePassWidth));
    let setHtml = buildSet(quotes, repeats);
    track.innerHTML =
      `<div class="ticker-set">${setHtml}</div>` +
      `<div class="ticker-set" aria-hidden="true">${setHtml}</div>`;
    // Re-query each pass: the captured node is detached by innerHTML and its
    // offsetWidth reads 0 forever, which would run the loop to the cap.
    let firstSet = track.querySelector('.ticker-set');
    while (firstSet && firstSet.offsetWidth < width + 40 && repeats < 24) {
      repeats += 1;
      setHtml = buildSet(quotes, repeats);
      track.innerHTML =
        `<div class="ticker-set">${setHtml}</div>` +
        `<div class="ticker-set" aria-hidden="true">${setHtml}</div>`;
      firstSet = track.querySelector('.ticker-set');
    }
    track.dataset.tickerReady = '1';
    scheduleStart();
  }

  function showStatus(message) {
    const track = element('tickerTrack');
    if (!track || track.dataset.tickerReady === '1') return;
    stopScroll();
    track.dataset.tickerReady = '0';
    track.style.transform = 'none';
    track.textContent = '';
    const placeholder = document.createElement('div');
    placeholder.className = 'ticker-placeholder';
    placeholder.textContent = message;
    track.appendChild(placeholder);
  }

  // US equity regular session: Mon–Fri 09:30–16:00 America/New_York.
  // Holidays are not modeled; closed on weekends and outside RTH.
  function isUsEquityMarketOpen(now = new Date()) {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      weekday: 'short',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).formatToParts(now);
    const get = (type) => parts.find((p) => p.type === type)?.value;
    const weekday = get('weekday');
    if (weekday === 'Sat' || weekday === 'Sun') return false;
    let hour = Number(get('hour'));
    const minute = Number(get('minute'));
    if (hour === 24) hour = 0;
    const mins = hour * 60 + minute;
    return mins >= 9 * 60 + 30 && mins < 16 * 60;
  }

  function updateMarketsStatus() {
    const el = element('tickerMarketsStatus');
    if (!el) return;
    const label = el.querySelector('.ticker-markets-label');
    const open = isUsEquityMarketOpen();
    el.classList.toggle('is-closed', !open);
    el.classList.toggle('ticker-markets-open', true);
    if (label) label.textContent = open ? 'Markets open' : 'Markets closed';
    el.setAttribute('aria-label', open ? 'US equity markets are open' : 'US equity markets are closed');
  }

  async function load() {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TICKER_TIMEOUT_MS);
    try {
      const response = await fetch(`${API_BASE}/ticker?symbols=${MAG7_TICKER_SYMBOLS.join(',')}`, {
        signal: controller.signal,
      });
      const data = await response.json().catch(() => ({}));
      if (data.quotes && data.quotes.length > 0) {
        latestQuotes = data.quotes;
        if (!patchQuotes(data.quotes)) render(data.quotes);
        return;
      }
      showStatus(data.error
        || (response.ok ? 'Market data temporarily unavailable' : `Market data unavailable (HTTP ${response.status})`));
    } catch (error) {
      showStatus(error.name === 'AbortError'
        ? 'Market data is taking longer than expected — retrying…'
        : 'Could not load market data');
    } finally {
      clearTimeout(timeoutId);
    }
  }

  function boot() {
    if (!element('tickerTrack')) return;
    bindControls();
    updateMarketsStatus();
    load();
  }

  window.AdminTicker = { load, isUsEquityMarketOpen, updateMarketsStatus };
  document.addEventListener('DOMContentLoaded', boot);
})();
