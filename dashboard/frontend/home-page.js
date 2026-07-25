/**
 * Home page mock live events (frontend only).
 * Replace useMockLiveEvents with a real event source when backend is ready.
 */
const ENABLE_MOCK_LIVE_EVENTS = true;

const INITIAL_EVENTS = [
    {
        id: 'initial-decision',
        type: 'decision_generated',
        agent: 'FinAgent Alpha',
        action: 'BUY',
        symbol: 'NVDA',
        price: 921.43,
        confidence: 0.74,
        rationale: 'Momentum strengthened with above-average volume.',
        createdAt: Date.now() - 2 * 60 * 1000,
    },
    {
        id: 'initial-trade-tsla',
        type: 'trade_executed',
        agent: 'QuantNova',
        action: 'SELL',
        symbol: 'TSLA',
        price: 410.22,
        rationale: 'Taking profits after resistance.',
        createdAt: Date.now() - 5 * 60 * 1000,
    },
    {
        id: 'initial-trade-aapl',
        type: 'trade_executed',
        agent: 'MacroMind',
        action: 'BUY',
        symbol: 'AAPL',
        price: 195.63,
        rationale: 'Positive earnings momentum.',
        createdAt: Date.now() - 8 * 60 * 1000,
    },
    {
        id: 'initial-backtest',
        type: 'backtest_completed',
        agent: 'SignalScout',
        strategy: 'Mean Reversion v2',
        returnValue: 4.2,
        sharpe: 1.42,
        createdAt: Date.now() - 12 * 60 * 1000,
    },
    {
        id: 'initial-risk',
        type: 'risk_check_passed',
        agent: 'RiskGuardian',
        message: 'Portfolio risk remains within configured limits.',
        createdAt: Date.now() - 15 * 60 * 1000,
    },
    {
        id: 'initial-rank',
        type: 'rank_changed',
        agent: 'FinAgent Alpha',
        previousRank: 7,
        newRank: 6,
        createdAt: Date.now() - 18 * 60 * 1000,
    },
];

const MOCK_EVENTS = [
    {
        id: 'mock-decision-1',
        type: 'decision_generated',
        agent: 'FinAgent Alpha',
        action: 'BUY',
        symbol: 'NVDA',
        price: 921.43,
        confidence: 0.74,
        rationale: 'Momentum strengthened with above-average volume.',
    },
    {
        id: 'mock-trade-1',
        type: 'trade_executed',
        agent: 'QuantNova',
        action: 'SELL',
        symbol: 'TSLA',
        price: 410.22,
        rationale: 'Taking profits after resistance.',
    },
    {
        id: 'mock-trade-2',
        type: 'trade_executed',
        agent: 'MacroMind',
        action: 'BUY',
        symbol: 'AAPL',
        price: 195.63,
        rationale: 'Positive earnings momentum.',
    },
    {
        id: 'mock-backtest-1',
        type: 'backtest_completed',
        agent: 'SignalScout',
        strategy: 'Mean Reversion v2',
        returnValue: 4.2,
        sharpe: 1.42,
    },
    {
        id: 'mock-risk-1',
        type: 'risk_check_passed',
        agent: 'RiskGuardian',
        message: 'Portfolio risk remains within configured limits.',
    },
    {
        id: 'mock-rank-1',
        type: 'rank_changed',
        agent: 'FinAgent Alpha',
        previousRank: 7,
        newRank: 6,
    },
];

const HOME_MARKET_PULSE_DATA = {
    traded: [
        { ticker: 'NVDA', count: 28, change: '+3.42%', up: true },
        { ticker: 'AAPL', count: 21, change: '+1.66%', up: true },
        { ticker: 'TSLA', count: 18, change: '+1.24%', up: true },
        { ticker: 'META', count: 16, change: '+4.25%', up: true },
        { ticker: 'AMZN', count: 15, change: '+3.69%', up: true },
    ],
    discussed: [
        { ticker: 'NVDA', count: 42, up: true },
        { ticker: 'TSLA', count: 35, up: false },
        { ticker: 'AAPL', count: 29, up: true },
        { ticker: 'MSFT', count: 24, up: true },
        { ticker: 'META', count: 19, up: true },
    ],
    trending: [
        { ticker: 'META', change: '+4.25%', up: true },
        { ticker: 'NVDA', change: '+3.42%', up: true },
        { ticker: 'AMZN', change: '+3.09%', up: true },
        { ticker: 'MSFT', change: '+2.32%', up: true },
        { ticker: 'AAPL', change: '+1.66%', up: true },
    ],
};

const EVENT_META = {
    decision_generated: { label: 'NEW DECISION', tone: 'cyan', icon: 'icon-brain' },
    trade_executed: { label: 'TRADE EXECUTED', tone: 'trade', icon: 'icon-chart' },
    backtest_completed: { label: 'BACKTEST COMPLETE', tone: 'cyan', icon: 'icon-flask' },
    risk_check_passed: { label: 'RISK CHECK PASSED', tone: 'green', icon: 'icon-shield-check' },
    rank_changed: { label: 'RANK UPDATE', tone: 'blue', icon: 'icon-trending-up' },
};

let homeMockTimer = null;
let homeMockIndex = 0;
let homeToastTimer = null;
let homeActivePulseTab = 'traded';
let homeFeedHovered = false;
let homeMetricValues = { agents: 28, decisions: 147, trades: 37, backtests: 126 };
let homeEvents = [];
let homeLatestEvent = null;

function homeIcon(name) {
    return `<svg class="ui-icon" aria-hidden="true"><use href="#${name}"/></svg>`;
}

function homeSparkline(up) {
    const points = up
        ? '0,14 8,10 16,12 24,6 32,8 40,2'
        : '0,4 8,8 16,6 24,12 32,10 40,14';
    const color = up ? '#22c55e' : '#ef4444';
    return `<svg class="home-sparkline" viewBox="0 0 40 16" aria-hidden="true"><polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

function formatPrice(price) {
    return `$${Number(price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatRelativeTime(createdAt) {
    const diffMs = Math.max(0, Date.now() - createdAt);
    const mins = Math.floor(diffMs / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
}

function eventToActivity(event, timeLabel) {
    const meta = EVENT_META[event.type] || { tone: 'cyan', icon: 'icon-activity' };
    const time = timeLabel || formatRelativeTime(event.createdAt || Date.now());
    const base = { id: event.id, time };

    switch (event.type) {
        case 'decision_generated':
            return {
                ...base,
                agent: event.agent,
                headline: 'generated a decision',
                action: `${event.action} ${event.symbol}`,
                context: event.rationale,
                tone: event.action === 'BUY' ? 'green' : event.action === 'SELL' ? 'red' : 'cyan',
                icon: meta.icon,
            };
        case 'trade_executed':
            return {
                ...base,
                agent: event.agent,
                headline: 'executed a trade',
                action: `${event.action} ${event.symbol} at ${formatPrice(event.price)}`,
                context: event.rationale,
                tone: event.action === 'SELL' ? 'red' : 'green',
                icon: meta.icon,
            };
        case 'backtest_completed':
            return {
                ...base,
                agent: event.agent,
                headline: 'completed a backtest',
                action: event.strategy,
                context: `Return +${event.returnValue}% · Sharpe ${event.sharpe}`,
                tone: 'amber',
                icon: meta.icon,
            };
        case 'risk_check_passed':
            return {
                ...base,
                agent: event.agent,
                headline: 'passed a risk check',
                action: event.message,
                context: '',
                tone: 'green',
                icon: meta.icon,
            };
        case 'rank_changed':
            return {
                ...base,
                agent: event.agent,
                headline: 'changed rank',
                action: `Moved from #${event.previousRank} to #${event.newRank}.`,
                context: '',
                tone: 'blue',
                icon: meta.icon,
            };
        default:
            return null;
    }
}

function eventToToast(event) {
    const meta = EVENT_META[event.type] || { label: 'LIVE EVENT', tone: 'cyan' };
    let actionText = '';
    let tone = meta.tone;

    switch (event.type) {
        case 'decision_generated':
            actionText = `${event.action} ${event.symbol} at ${formatPrice(event.price)}`;
            tone = event.action === 'BUY' ? 'green' : event.action === 'SELL' ? 'red' : 'cyan';
            break;
        case 'trade_executed':
            actionText = `${event.action} ${event.symbol} at ${formatPrice(event.price)}`;
            tone = event.action === 'SELL' ? 'red' : 'green';
            break;
        case 'backtest_completed':
            actionText = `${event.strategy} · Return +${event.returnValue}%`;
            tone = 'cyan';
            break;
        case 'risk_check_passed':
            actionText = event.message;
            tone = 'green';
            break;
        case 'rank_changed':
            actionText = `Rank #${event.previousRank} → #${event.newRank}`;
            tone = 'blue';
            break;
        default:
            actionText = 'New activity';
    }

    return {
        label: meta.label,
        time: 'just now',
        agent: event.agent,
        action: actionText,
        rationale: event.rationale || event.message || '',
        tone,
        icon: meta.icon,
    };
}

function renderActivityItem(item, isNew) {
    const cls = isNew ? ' home-timeline-item--new' : '';
    const context = item.context
        ? `<p class="home-timeline-context">${item.context}</p>`
        : '';
    return `
        <article class="home-timeline-item home-timeline-item--${item.tone}${cls}" data-event-id="${item.id || ''}">
            <div class="home-timeline-rail" aria-hidden="true">
                <span class="home-timeline-dot"></span>
            </div>
            <div class="home-timeline-body">
                <div class="home-timeline-head">
                    <span class="home-timeline-time">${item.time}</span>
                    <span class="home-timeline-icon">${homeIcon(item.icon)}</span>
                </div>
                <p class="home-timeline-text"><strong>${item.agent}</strong> ${item.headline}</p>
                <p class="home-timeline-action">${item.action}</p>
                ${context}
            </div>
        </article>
    `;
}

function renderActivityFeed(items) {
    const feed = document.getElementById('homeActivityFeed');
    if (!feed) return;

    if (!items.length) {
        feed.innerHTML = '<p class="home-timeline-fallback">Waiting for the next agent event…</p>';
        return;
    }

    feed.innerHTML = items.map((item) => renderActivityItem(item, false)).join('');
}

function prependActivity(item) {
    const feed = document.getElementById('homeActivityFeed');
    if (!feed) return;

    const fallback = feed.querySelector('.home-timeline-fallback');
    if (fallback) fallback.remove();

    feed.insertAdjacentHTML('afterbegin', renderActivityItem(item, true));
    const first = feed.firstElementChild;
    if (first && !homeFeedHovered) {
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                first.classList.remove('home-timeline-item--new');
                first.classList.add('home-timeline-item--highlight');
                window.setTimeout(() => first.classList.remove('home-timeline-item--highlight'), 700);
            });
        });
    } else if (first) {
        first.classList.remove('home-timeline-item--new');
    }

    while (feed.children.length > 6) {
        feed.lastElementChild?.remove();
    }
}

function flashMetric(key, delta) {
    if (homeMetricValues[key] === undefined) return;
    homeMetricValues[key] = Math.max(0, homeMetricValues[key] + delta);
    const map = {
        agents: 'homeMetricAgents',
        decisions: 'homeMetricDecisions',
        trades: 'homeMetricTrades',
        backtests: 'homeMetricBacktests',
    };
    const el = document.getElementById(map[key]);
    if (!el) return;
    el.classList.add('home-metric-value--flash');
    el.textContent = String(homeMetricValues[key]);
    window.setTimeout(() => el.classList.remove('home-metric-value--flash'), 650);
}

function applyEventMetrics(event) {
    switch (event.type) {
        case 'decision_generated':
            flashMetric('decisions', 1);
            break;
        case 'trade_executed':
            flashMetric('trades', 1);
            break;
        case 'backtest_started':
            flashMetric('backtests', 1);
            break;
        case 'backtest_completed':
            flashMetric('backtests', -1);
            break;
        case 'agent_online':
            flashMetric('agents', 1);
            break;
        case 'agent_offline':
            flashMetric('agents', -1);
            break;
        default:
            break;
    }
}

function flashField(el) {
    if (!el) return;
    el.classList.add('home-field--flash');
    window.setTimeout(() => el.classList.remove('home-field--flash'), 650);
}

function updateSpotlightFromEvent(event) {
    if (event.agent !== 'FinAgent Alpha') return;

    const actionMain = document.getElementById('homeSpotlightActionMain');
    const priceEl = document.getElementById('homeSpotlightPrice');
    const timeEl = document.getElementById('homeSpotlightActionTime');
    const rationaleEl = document.getElementById('homeSpotlightRationale');
    const activeEl = document.getElementById('homeSpotlightLastActive');
    const sparkline = document.getElementById('homeSpotlightSparkline');
    const actionBlock = actionMain?.closest('.home-spotlight-action-block');

    if (event.type === 'decision_generated' || event.type === 'trade_executed') {
        const verbEl = actionMain?.querySelector('.home-spotlight-action-verb');
        if (verbEl) {
            verbEl.textContent = event.action;
            verbEl.className = `home-spotlight-action-verb ${event.action === 'SELL' ? 'home-highlight-negative' : 'home-highlight-positive'}`;
            flashField(verbEl);
        }
        if (priceEl) {
            priceEl.textContent = formatPrice(event.price);
            flashField(priceEl);
        }
        if (actionMain) flashField(actionMain);
        if (timeEl) {
            timeEl.textContent = 'just now';
            flashField(timeEl);
        }
        if (rationaleEl && event.rationale) {
            rationaleEl.textContent = event.rationale;
            flashField(rationaleEl);
        }
        if (activeEl) {
            activeEl.textContent = 'just now';
            flashField(activeEl);
        }
        if (sparkline) {
            const line = sparkline.querySelector('polyline');
            if (line) line.setAttribute('points', '0,16 6,12 12,14 18,8 24,10 30,4 36,6 40,2');
            flashField(sparkline.closest('.home-spotlight-stat'));
        }
        if (actionBlock) {
            actionBlock.classList.add('home-spotlight-action-block--flash');
            window.setTimeout(() => actionBlock.classList.remove('home-spotlight-action-block--flash'), 700);
        }
    } else if (event.type === 'rank_changed') {
        if (activeEl) {
            activeEl.textContent = 'just now';
            flashField(activeEl);
        }
    }
}

function highlightPulseSymbol(symbol) {
    if (!symbol) return;
    document.querySelectorAll('.home-pulse-row').forEach((row) => {
        if (row.dataset.ticker === symbol) {
            row.classList.add('home-pulse-row--flash');
            window.setTimeout(() => row.classList.remove('home-pulse-row--flash'), 900);
        }
    });
}

function renderMarketPulseTab(tab) {
    homeActivePulseTab = tab;
    const list = document.getElementById('homeMarketPulseList');
    if (!list) return;

    let rowsHtml = '';

    if (tab === 'traded') {
        rowsHtml = HOME_MARKET_PULSE_DATA.traded.map((row) => `
            <div class="home-pulse-row" data-ticker="${row.ticker}">
                <div class="home-pulse-ticker">${row.ticker}</div>
                <div class="home-pulse-agents tabular-nums">${row.count} agents</div>
                <div class="home-pulse-change tabular-nums positive">${row.change}</div>
                ${homeSparkline(row.up)}
            </div>
        `).join('');
    } else if (tab === 'discussed') {
        rowsHtml = HOME_MARKET_PULSE_DATA.discussed.map((row) => `
            <div class="home-pulse-row" data-ticker="${row.ticker}">
                <div class="home-pulse-ticker">${row.ticker}</div>
                <div class="home-pulse-agents tabular-nums">${row.count} mentions</div>
                <div class="home-pulse-change tabular-nums home-pulse-muted">—</div>
                ${homeSparkline(row.up !== false)}
            </div>
        `).join('');
    } else {
        rowsHtml = HOME_MARKET_PULSE_DATA.trending.map((row) => `
            <div class="home-pulse-row" data-ticker="${row.ticker}">
                <div class="home-pulse-ticker">${row.ticker}</div>
                <div class="home-pulse-agents tabular-nums home-pulse-muted">trending</div>
                <div class="home-pulse-change tabular-nums positive">${row.change}</div>
                ${homeSparkline(row.up)}
            </div>
        `).join('');
    }

    list.innerHTML = rowsHtml || '<p class="home-pulse-fallback">No agent market activity available.</p>';
}

function hideLiveToast() {
    // Live toast removed from Home.
}

function showLiveToast(_toastData) {
    // Live toast removed from Home.
}

function pushMockEvent(event) {
    const stamped = { ...event, createdAt: Date.now() };
    homeEvents = [stamped, ...homeEvents].slice(0, 6);
    homeLatestEvent = stamped;

    const activity = eventToActivity(stamped);
    if (activity) prependActivity(activity);

    applyEventMetrics(stamped);
    updateSpotlightFromEvent(stamped);

    if (stamped.symbol) highlightPulseSymbol(stamped.symbol);

    showLiveToast(eventToToast(stamped));
}

function emitMockLiveEvent() {
    const event = MOCK_EVENTS[homeMockIndex % MOCK_EVENTS.length];
    homeMockIndex += 1;
    pushMockEvent({ ...event, id: `${event.id}-${homeMockIndex}` });
}

function scheduleNextMockEvent() {
    if (!ENABLE_MOCK_LIVE_EVENTS) return;
    const delay = 6000 + Math.floor(Math.random() * 4000);
    homeMockTimer = window.setTimeout(() => {
        if (document.getElementById('homeView')?.style.display !== 'none') {
            emitMockLiveEvent();
        }
        scheduleNextMockEvent();
    }, delay);
}

function stopHomeMockEvents() {
    window.clearTimeout(homeMockTimer);
    homeMockTimer = null;
}

function dismissLatestEvent() {
    hideLiveToast();
}

function useMockLiveEvents() {
    return {
        events: homeEvents,
        latestEvent: homeLatestEvent,
        metrics: { ...homeMetricValues },
        spotlight: null,
        pulseData: HOME_MARKET_PULSE_DATA,
        dismissLatestEvent,
        start() {
            stopHomeMockEvents();
            if (!ENABLE_MOCK_LIVE_EVENTS) return;
            homeMockTimer = window.setTimeout(() => {
                if (document.getElementById('homeView')?.style.display !== 'none') {
                    emitMockLiveEvent();
                }
                scheduleNextMockEvent();
            }, 3000);
        },
        stop: stopHomeMockEvents,
    };
}

let homeMockLive = null;

function initMarketPulseTabs() {
    document.querySelectorAll('[data-pulse-tab]').forEach((btn) => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.pulseTab;
            document.querySelectorAll('[data-pulse-tab]').forEach((b) => {
                b.classList.toggle('active', b.dataset.pulseTab === tab);
            });
            renderMarketPulseTab(tab);
        });
    });
    renderMarketPulseTab('traded');
}

function initActivityFeedHover() {
    const feed = document.getElementById('homeActivityFeed');
    if (!feed) return;
    feed.addEventListener('mouseenter', () => { homeFeedHovered = true; });
    feed.addEventListener('mouseleave', () => { homeFeedHovered = false; });
}

function navigateToLeaderboard() {
    if (typeof navigateToPage === 'function') {
        navigateToPage('competition', { competitionTab: 'leaderboard' });
    }
}



function initLandingPlaygroundChat() {
    const root = document.getElementById('homePlaygroundChat');
    if (!root || root.dataset.simStarted === '1') return;
    root.dataset.simStarted = '1';

    const steps = Array.from(root.querySelectorAll('.home-chat-step'));
    let step = 0;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function revealThrough(n) {
        steps.forEach((el) => {
            const s = Number(el.dataset.step || 0);
            if (s <= n) el.hidden = false;
        });
        root.dataset.step = String(n);
        // Keep latest agent bubble in view as steps advance.
        const latest = steps.find((el) => Number(el.dataset.step || 0) === n);
        if (latest && typeof latest.scrollIntoView === 'function') {
            latest.scrollIntoView({ block: 'nearest', behavior: reduceMotion ? 'auto' : 'smooth' });
        }
    }

    if (reduceMotion) {
        revealThrough(4);
        return;
    }

    revealThrough(0);
    const timer = setInterval(() => {
        step += 1;
        revealThrough(step);
        if (step >= 4) clearInterval(timer);
    }, 2200);
}

function initHomeGetStarted() {
    document.getElementById('homeGetStartedBtn')?.addEventListener('click', () => {
        if (typeof navigateToPage === 'function') {
            navigateToPage('playground', { playgroundTab: 'agents' });
            return;
        }
        document.getElementById('homeLiveSection')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
}

function measureAppChromeHeight() {
    const header = document.querySelector('.header');
    const ticker = document.querySelector('.ticker-bar');
    const tickerInfo = document.querySelector('.ticker-info');
    let h = 0;
    if (header) h += header.getBoundingClientRect().height;
    if (ticker) h += ticker.getBoundingClientRect().height;
    if (tickerInfo) h += tickerInfo.getBoundingClientRect().height;
    // homeView starts after these siblings; include a small safety gap
    const measured = Math.max(120, Math.round(h));
    document.documentElement.style.setProperty('--app-chrome-height', `${measured}px`);
    return measured;
}

function homePrefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function refreshHomeModulesWhenReady() {
    const hasAgents = typeof allAgents !== 'undefined' && Array.isArray(allAgents) && allAgents.length > 0;
    if (!hasAgents && typeof loadAgents === 'function') {
        Promise.resolve(loadAgents()).catch(() => {
            refreshHomeModules();
        });
    } else {
        refreshHomeModules();
    }
}

/** @param {0|1|number} page @param {{ instant?: boolean }} [opts] */
function setHomePagerPage(page, opts = {}) {
    const track = document.getElementById('homePagerTrack');
    const hint = document.getElementById('homeScrollHint');
    const landing = document.getElementById('homeScreenLanding');
    const dashboard = document.getElementById('homeScreenDashboard');
    if (!track) return;
    const next = page === 1 ? 1 : 0;
    const target = next === 1 ? dashboard : landing;
    track.dataset.page = String(next);
    if (hint) hint.classList.toggle('is-hidden', next === 1);
    if (!target) return;

    const behavior = opts.instant || homePrefersReducedMotion() ? 'auto' : 'smooth';
    // Prefer scrollTop so we don't fight nested scroll containers.
    const top = next === 1 ? track.clientHeight : 0;
    if (typeof track.scrollTo === 'function') {
        track.scrollTo({ top, behavior });
    } else {
        track.scrollTop = top;
    }
}

function homeEscape(value) {
    if (typeof escapeHtml === 'function') return escapeHtml(value);
    return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function homeSafeUrl(raw) {
    const s = String(raw == null ? '' : raw).trim();
    return /^https?:\/\//i.test(s) ? s : '#';
}

function homeInitials(name) {
    const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
    if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return '?';
}

function homeFormatMoney(value, digits = 0) {
    const n = Number(value);
    if (!Number.isFinite(n)) return digits ? '$10,000.00' : '$10,000';
    return `$${n.toLocaleString('en-US', {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    })}`;
}

function homeFormatReturnPct(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    const pct = n * 100;
    const sign = pct > 0 ? '+' : '';
    return `${sign}${pct.toFixed(2)}%`;
}

function homeSparkPolyline(values, width = 52, height = 18) {
    const nums = (values || []).map(Number).filter(Number.isFinite);
    if (nums.length < 2) {
        return `0,${height / 2} ${width},${height / 2}`;
    }
    const min = Math.min(...nums);
    const max = Math.max(...nums);
    const span = max - min || 1;
    return nums.map((v, i) => {
        const x = (i / (nums.length - 1)) * width;
        const y = height - ((v - min) / span) * (height - 2) - 1;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
}

function getHomeAuthUser() {
    if (typeof getStoredAuthUser === 'function') return getStoredAuthUser();
    return window.AUTH_USER || null;
}

function isHomeSignedIn() {
    const token = localStorage.getItem(typeof AUTH_TOKEN_KEY === 'string' ? AUTH_TOKEN_KEY : 'auth-token');
    return !!(token && getHomeAuthUser());
}

function openHomeCreateAgent() {
    if (!isHomeSignedIn()) {
        if (typeof openAuthModal === 'function') openAuthModal('signup');
        return;
    }
    if (typeof navigateToPage === 'function') {
        navigateToPage('playground', { playgroundTab: 'agents' });
    }
    if (typeof openAddAgentModal === 'function') openAddAgentModal();
}

/** Demo / placeholder portfolio used by the home module for guests. */
const HOME_PORTFOLIO = {
    equity: 10000,
    dayPnl: 0,
    alloc: { cash: 6200, stocks: 2800, crypto: 1000 },
};

/** @type {null | { equity: number, cash_available: number, allocated: number }} */
let homePortfolioLive = null;
let homePortRange = '1D';
let homePortChartState = null;

function homePortRangeMeta(range) {
    switch (range) {
        case '7D': return { points: 8, xLabels: ['6d', '4d', '2d', 'Now'] };
        case '1M': return { points: 13, xLabels: ['4w', '3w', '2w', '1w', 'Now'] };
        case '3M': return { points: 14, xLabels: ['3m', '2m', '1m', 'Now'] };
        case '1Y': return { points: 13, xLabels: ['12m', '9m', '6m', '3m', 'Now'] };
        case 'All': return { points: 13, xLabels: ['Start', 'Mid', 'Now'] };
        case '1D':
        default: return { points: 13, xLabels: ['9:30', '12:00', '15:00', 'Now'] };
    }
}

function homePortTimestamps(range, count) {
    const now = Date.now();
    const labels = [];
    if (range === '1D') {
        // Regular session 09:30 → 16:00
        const start = new Date();
        start.setHours(9, 30, 0, 0);
        const end = new Date();
        end.setHours(16, 0, 0, 0);
        for (let i = 0; i < count; i += 1) {
            const t = start.getTime() + ((end.getTime() - start.getTime()) * i) / (count - 1);
            labels.push(new Date(t));
        }
        return labels;
    }
    const spans = {
        '7D': 7 * 86400000,
        '1M': 30 * 86400000,
        '3M': 90 * 86400000,
        '1Y': 365 * 86400000,
        All: 365 * 86400000,
    };
    const span = spans[range] || spans['7D'];
    for (let i = 0; i < count; i += 1) {
        labels.push(new Date(now - span + (span * i) / (count - 1)));
    }
    return labels;
}

function homePortSeries(equity, dayPnl, range) {
    const { points, xLabels } = homePortRangeMeta(range);
    const end = Number(equity) || 10000;
    const pnl = Number(dayPnl) || 0;
    const values = [];
    const times = homePortTimestamps(range, points);

    // When there is no real change, keep an intentional flat series — do not invent drift.
    if (Math.abs(pnl) < 1e-9) {
        for (let i = 0; i < points; i += 1) values.push(end);
    } else if (range === '1D') {
        const start = end - pnl;
        for (let i = 0; i < points; i += 1) {
            const t = i / (points - 1);
            values.push(start + (end - start) * t);
        }
    } else {
        const start = end - pnl;
        for (let i = 0; i < points; i += 1) {
            const t = i / (points - 1);
            values.push(start + (end - start) * t);
        }
        values[values.length - 1] = end;
    }

    const isFlat = values.every((v) => Math.abs(v - values[0]) < 1e-6);
    return { values, xLabels, times, startValue: values[0], endValue: values[values.length - 1], isFlat };
}

function homeNiceStep(rough) {
    const abs = Math.max(Math.abs(rough), 1e-9);
    const exp = Math.floor(Math.log10(abs));
    const mag = 10 ** exp;
    const norm = abs / mag;
    let nice;
    if (norm <= 1) nice = 1;
    else if (norm <= 2) nice = 2;
    else if (norm <= 5) nice = 5;
    else nice = 10;
    return nice * mag;
}

/** Three adaptive Y ticks; domain is not forced through zero. Not used for flat charts. */
function homeAdaptiveYScale(values) {
    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);
    const mid = (dataMin + dataMax) / 2;
    const rawSpan = Math.max(dataMax - dataMin, 1e-9);
    let half = Math.max(rawSpan * 0.7, rawSpan / 2 + 25);
    let lo = mid - half;
    let hi = mid + half;
    lo = Math.min(lo, dataMin);
    hi = Math.max(hi, dataMax);

    const step = homeNiceStep((hi - lo) / 2);
    let midTick = Math.round(mid / step) * step;
    let top = midTick + step;
    let bot = midTick - step;
    while (dataMax > top) top += step;
    while (dataMin < bot) bot -= step;
    if (top === bot) {
        top = midTick + step;
        bot = midTick - step;
    }
    return { min: bot, max: top, ticks: [top, midTick, bot] };
}

function homeFormatAxisMoney(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    const rounded = Math.round(n);
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(rounded);
}

function homeFormatExactMoney(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(n);
}

function homeFormatSignedMoney(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    const body = homeFormatExactMoney(Math.abs(n));
    if (n > 0) return `+${body}`;
    if (n < 0) return `−${body}`;
    return body;
}

function homeFormatHoverTime(date, range) {
    if (!(date instanceof Date) || Number.isNaN(date.getTime())) return '—';
    if (range === '1D') {
        return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
    }
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function bindHomePortfolioChartHover() {
    const figure = document.getElementById('homePortfolioFigure');
    const svg = document.getElementById('homePortfolioChart');
    const tip = document.getElementById('homePortfolioTooltip');
    if (!figure || !svg || !tip || figure.dataset.hoverBound === '1') return;
    figure.dataset.hoverBound = '1';

    const hide = () => {
        tip.hidden = true;
        const cross = svg.querySelector('.hm-port-crosshair');
        const focus = svg.querySelector('.hm-port-focus');
        if (cross) cross.setAttribute('opacity', '0');
        if (focus) focus.setAttribute('opacity', '0');
    };

    figure.addEventListener('mouseleave', hide);
    figure.addEventListener('mousemove', (event) => {
        const state = homePortChartState;
        if (!state?.pts?.length) return;
        const rect = svg.getBoundingClientRect();
        const xSvg = ((event.clientX - rect.left) / rect.width) * state.W;
        let best = 0;
        let bestDist = Infinity;
        state.pts.forEach((p, i) => {
            const d = Math.abs(p[0] - xSvg);
            if (d < bestDist) {
                bestDist = d;
                best = i;
            }
        });
        const pt = state.pts[best];
        const value = state.values[best];
        const start = state.startValue;
        const chg = value - start;
        const chgClass = chg > 0.005 ? 'is-pos' : chg < -0.005 ? 'is-neg' : 'is-flat';
        tip.innerHTML = `
          <span>${homeEscape(homeFormatHoverTime(state.times[best], state.range))}</span>
          <strong>${homeEscape(homeFormatExactMoney(value))}</strong>
          <span class="hm-port-tip-chg ${chgClass}">${homeEscape(homeFormatSignedMoney(chg))} vs period start</span>
        `;
        tip.hidden = false;
        const tipX = (pt[0] / state.W) * rect.width;
        const tipY = (pt[1] / state.H) * rect.height;
        tip.style.left = `${tipX}px`;
        tip.style.top = `${tipY}px`;

        const cross = svg.querySelector('.hm-port-crosshair');
        const focus = svg.querySelector('.hm-port-focus');
        if (cross) {
            cross.setAttribute('x1', pt[0].toFixed(1));
            cross.setAttribute('x2', pt[0].toFixed(1));
            cross.setAttribute('opacity', '1');
        }
        if (focus) {
            focus.setAttribute('cx', pt[0].toFixed(1));
            focus.setAttribute('cy', pt[1].toFixed(1));
            focus.setAttribute('opacity', '1');
        }
    });
}

function renderHomePortfolioChart(equity = HOME_PORTFOLIO.equity, dayPnl = HOME_PORTFOLIO.dayPnl, range = homePortRange) {
    const svg = document.getElementById('homePortfolioChart');
    const tip = document.getElementById('homePortfolioTooltip');
    if (!svg) return;
    if (tip) tip.hidden = true;

    const series = homePortSeries(equity, dayPnl, range);
    const { values, xLabels, times, startValue, isFlat } = series;
    const W = 360;
    const H = 148;

    // Flat charts reclaim Y-axis width; varied charts keep a compact left gutter.
    const pad = isFlat
        ? { top: 14, right: 12, bottom: 24, left: 12 }
        : { top: 12, right: 12, bottom: 24, left: 46 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    let minV;
    let maxV;
    let yTicks = [];
    if (isFlat) {
        // Keep a tiny visual band so the flat line sits mid-plot without fake ticks.
        minV = values[0] - 1;
        maxV = values[0] + 1;
    } else {
        const scale = homeAdaptiveYScale(values);
        minV = scale.min;
        maxV = scale.max;
        yTicks = scale.ticks;
    }

    const xAt = (i) => pad.left + (i / Math.max(values.length - 1, 1)) * plotW;
    const yAt = (v) => pad.top + ((maxV - v) / (maxV - minV || 1)) * plotH;
    const pts = values.map((v, i) => [xAt(i), yAt(v)]);
    const line = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');

    const xTickIdx = xLabels.map((_, i) => {
        if (xLabels.length === 1) return 0;
        return Math.round((i / (xLabels.length - 1)) * (values.length - 1));
    });
    const xAxis = xLabels.map((label, i) => {
        const x = xAt(xTickIdx[i]);
        return `<text x="${x.toFixed(1)}" y="${H - 6}" text-anchor="middle" fill="rgba(148,163,184,0.65)" font-size="9" font-family="ui-sans-serif, system-ui, sans-serif">${label}</text>`;
    }).join('');

    let plotBody = '';
    const area = `${line} L${pts[pts.length - 1][0].toFixed(1)},${(pad.top + plotH).toFixed(1)} L${pts[0][0].toFixed(1)},${(pad.top + plotH).toFixed(1)} Z`;
    const endPt = pts[pts.length - 1];
    if (isFlat) {
        // Flat: cyan line + restrained area fill, no Y-axis labels, no endpoint label.
        plotBody = `
          <defs>
            <linearGradient id="hmPortFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="rgba(34,211,238,0.28)"/>
              <stop offset="100%" stop-color="rgba(34,211,238,0)"/>
            </linearGradient>
          </defs>
          <path d="${area}" fill="url(#hmPortFill)"/>
          <path d="${line}" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <line class="hm-port-crosshair" x1="${pts[0][0]}" y1="${pad.top}" x2="${pts[0][0]}" y2="${pad.top + plotH}" stroke="rgba(34,211,238,0.35)" stroke-width="1" stroke-dasharray="3 3" opacity="0"/>
          <circle class="hm-port-focus" cx="${pts[0][0]}" cy="${pts[0][1]}" r="3.2" fill="#22d3ee" opacity="0"/>
        `;
    } else {
        const yGrid = yTicks.map((v) => {
            const y = yAt(v);
            return `
              <line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${W - pad.right}" y2="${y.toFixed(1)}" stroke="rgba(148,163,184,0.12)" stroke-width="1"/>
              <text x="${pad.left - 7}" y="${(y + 3).toFixed(1)}" text-anchor="end" fill="rgba(148,163,184,0.7)" font-size="9" font-family="ui-sans-serif, system-ui, sans-serif">${homeFormatAxisMoney(v)}</text>`;
        }).join('');
        plotBody = `
          <defs>
            <linearGradient id="hmPortFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="rgba(34,211,238,0.28)"/>
              <stop offset="100%" stop-color="rgba(34,211,238,0)"/>
            </linearGradient>
          </defs>
          ${yGrid}
          <path d="${area}" fill="url(#hmPortFill)"/>
          <path d="${line}" fill="none" stroke="#22d3ee" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <line class="hm-port-crosshair" x1="${endPt[0]}" y1="${pad.top}" x2="${endPt[0]}" y2="${pad.top + plotH}" stroke="rgba(34,211,238,0.35)" stroke-width="1" stroke-dasharray="3 3" opacity="0"/>
          <circle class="hm-port-focus" cx="${endPt[0]}" cy="${endPt[1]}" r="3.4" fill="#22d3ee" opacity="0"/>
          <circle cx="${endPt[0].toFixed(1)}" cy="${endPt[1].toFixed(1)}" r="2.6" fill="#22d3ee"/>
        `;
    }

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = `
      <line x1="${pad.left}" y1="${pad.top + plotH}" x2="${W - pad.right}" y2="${pad.top + plotH}" stroke="rgba(148,163,184,0.22)" stroke-width="1"/>
      ${plotBody}
      ${xAxis}
    `;

    homePortChartState = {
        W, H, pad, pts, values, times, startValue, range, plotH, isFlat,
    };
    bindHomePortfolioChartHover();
}

function homePortfolioEquity() {
    if (homePortfolioLive && Number.isFinite(Number(homePortfolioLive.equity))) {
        return Number(homePortfolioLive.equity);
    }
    return HOME_PORTFOLIO.equity;
}

function homePortfolioDayPnl() {
    // Paper trading P&L not wired yet — keep flat zero for live + demo.
    return 0;
}

async function loadHomePortfolioLedger() {
    if (!isHomeSignedIn() || typeof API === 'undefined' || typeof API_BASE === 'undefined') {
        homePortfolioLive = null;
        return null;
    }
    try {
        const data = await API.get(`${API_BASE}/api/v1/portfolio`);
        const portfolio = data && data.portfolio;
        if (!portfolio) {
            homePortfolioLive = null;
            return null;
        }
        homePortfolioLive = {
            equity: Number(portfolio.equity) || 0,
            cash_available: Number(portfolio.cash_available) || 0,
            allocated: Number(portfolio.allocated) || 0,
        };
        return homePortfolioLive;
    } catch (error) {
        console.warn('Home portfolio API unavailable:', error?.message || error);
        homePortfolioLive = null;
        return null;
    }
}

async function updateHomePortfolioModule() {
    const user = getHomeAuthUser();
    const signedIn = isHomeSignedIn();
    const avatar = document.getElementById('homePortfolioAvatar');
    const nameEl = document.getElementById('homePortfolioName');
    const equityEl = document.getElementById('homePortfolioEquity');
    const labelEl = document.getElementById('homePortfolioEquityLabel');
    const btn = document.getElementById('homeModulePortfolioBtn');
    const pnl = document.getElementById('homeMetricPnl');
    const demoBadge = document.getElementById('homePortfolioDemoBadge');

    if (signedIn && user) {
        await loadHomePortfolioLedger();
    } else {
        homePortfolioLive = null;
    }

    const equity = homePortfolioEquity();
    const dayPnl = homePortfolioDayPnl();
    const dayPct = equity ? (dayPnl / equity) * 100 : 0;
    const live = signedIn && !!homePortfolioLive;

    if (!signedIn || !user) {
        if (avatar) avatar.textContent = 'G';
        if (nameEl) nameEl.textContent = 'Guest Account';
        if (labelEl) labelEl.textContent = 'Demo Portfolio · Total Equity';
        if (btn) {
            btn.hidden = false;
            btn.textContent = 'Sign in';
        }
        if (demoBadge) demoBadge.hidden = false;
    } else {
        const label = user.display_name || user.email || 'Trader';
        if (avatar) avatar.textContent = homeInitials(label);
        if (nameEl) nameEl.textContent = label;
        if (labelEl) labelEl.textContent = live ? 'Total Equity' : 'Demo Portfolio · Total Equity';
        if (btn) btn.hidden = true;
        if (demoBadge) demoBadge.hidden = live;
    }

    if (equityEl) equityEl.textContent = homeFormatMoney(equity, 2);
    if (pnl) {
        const pctText = `(${dayPct === 0 ? '0.00' : `${dayPct > 0 ? '+' : ''}${dayPct.toFixed(2)}`}%)`;
        pnl.innerHTML = `${homeEscape(homeFormatMoney(dayPnl, 2))} <em id="homeMetricPnlPct">${homeEscape(pctText)}</em>`;
    }
    renderHomePortfolioChart(equity, dayPnl, homePortRange);
}

function homeListUserAgents() {
    if (typeof allAgents === 'undefined' || !Array.isArray(allAgents)) return [];
    return allAgents.filter(
        (a) => a?.agent_id && !(typeof isDemoAgent === 'function' && isDemoAgent(a.agent_id)),
    );
}

function homeAgentIsPaperTrading(agent) {
    if (typeof resolveAgentStatusBadge === 'function') {
        return resolveAgentStatusBadge(agent).key === 'paper';
    }
    const deployment = String(agent?.deployment_status || '').toLowerCase();
    return agent?.is_live === true || deployment === 'live' || deployment === 'paper';
}

function homeParseActivityTime(raw) {
    if (!raw) return NaN;
    const t = new Date(String(raw).replace(' ', 'T')).getTime();
    return Number.isFinite(t) ? t : NaN;
}

function homeActivityRelTime(iso) {
    const t = homeParseActivityTime(iso);
    if (!Number.isFinite(t)) return '';
    const mins = Math.round((Date.now() - t) / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins} min ago`;
    const hours = Math.round(mins / 60);
    if (hours < 24) return `${hours} hr ago`;
    const startToday = new Date();
    startToday.setHours(0, 0, 0, 0);
    const startYesterday = new Date(startToday);
    startYesterday.setDate(startYesterday.getDate() - 1);
    if (t >= startYesterday.getTime() && t < startToday.getTime()) return 'Yesterday';
    if (typeof formatRelativeTime === 'function') return formatRelativeTime(iso);
    return `${Math.round(hours / 24)}d ago`;
}

function homeFormatReturnBadge(run) {
    const retRaw = Number(run?.total_return);
    let frac = null;
    if (Number.isFinite(retRaw)) {
        frac = Math.abs(retRaw) <= 1 ? retRaw : retRaw / 100;
    } else {
        const initial = Number(run?.initial_equity);
        const final = Number(run?.final_equity);
        if (Number.isFinite(initial) && initial > 0 && Number.isFinite(final)) {
            frac = (final - initial) / initial;
        }
    }
    if (frac == null || !Number.isFinite(frac)) return null;
    const pct = frac * 100;
    const sign = pct > 0 ? '+' : '';
    return {
        text: `${sign}${pct.toFixed(1)}%`,
        tone: pct >= 0 ? 'pos' : 'neg',
    };
}

/** Build up to 3 recent real activities from agent records (no demo filler). */
function collectHomeAgentActivities(agents) {
    const events = [];
    for (const agent of agents) {
        const agentId = agent.agent_id;
        const agentName = agent.name || 'Untitled agent';

        if (agent.created_at) {
            events.push({
                at: agent.created_at,
                agentId,
                agentName,
                kind: 'created',
                description: 'was created',
                badge: null,
                tone: 'neutral',
                target: 'configure',
                runId: null,
            });
        }

        const runs = Array.isArray(agent.runs) && agent.runs.length
            ? agent.runs
            : (agent.latest_run && (agent.latest_run.run_id || agent.latest_run.created_at)
                ? [agent.latest_run]
                : []);
        const seenRuns = new Set();
        for (const run of runs) {
            const runId = run?.run_id || null;
            const dedupe = runId || `${agentId}:${run?.created_at || ''}`;
            if (seenRuns.has(dedupe)) continue;
            seenRuns.add(dedupe);
            if (!run?.created_at && !runId) continue;
            const badge = homeFormatReturnBadge(run);
            events.push({
                at: run.created_at || agent.created_at,
                agentId,
                agentName,
                kind: 'backtest',
                description: 'completed a backtest',
                badge: badge?.text || null,
                tone: badge?.tone || 'neutral',
                target: 'backtest',
                runId,
            });
        }

        if (homeAgentIsPaperTrading(agent)) {
            const paperAt = agent.paper_updated_at || agent.last_used_at || agent.created_at;
            if (paperAt) {
                events.push({
                    at: paperAt,
                    agentId,
                    agentName,
                    kind: 'paper',
                    description: 'started paper trading',
                    badge: null,
                    tone: 'cyan',
                    target: 'paper',
                    runId: null,
                });
            }
        }
    }

    return events
        .filter((e) => Number.isFinite(homeParseActivityTime(e.at)))
        .sort((a, b) => homeParseActivityTime(b.at) - homeParseActivityTime(a.at))
        .slice(0, 3);
}

function renderHomeAgentOverview(agents) {
    const esc = (typeof escapeHtml === 'function') ? escapeHtml : (v) => String(v ?? '');
    const total = agents.length;
    const paper = agents.filter(homeAgentIsPaperTrading).length;
    const notTrading = Math.max(0, total - paper);
    const activities = collectHomeAgentActivities(agents);

    const activityHtml = activities.length
        ? `<ul class="hm-agent-activity-list" role="list">
            ${activities.map((ev) => {
                const badge = ev.badge
                    ? `<span class="hm-agent-activity-badge is-${esc(ev.tone)}">${esc(ev.badge)}</span>`
                    : '';
                return `<li>
                  <button type="button" class="hm-agent-activity-row"
                    data-agent-id="${esc(ev.agentId)}"
                    data-target="${esc(ev.target)}"
                    data-run-id="${esc(ev.runId || '')}"
                    data-kind="${esc(ev.kind)}">
                    <span class="hm-agent-activity-dot is-${esc(ev.tone || ev.kind)}" aria-hidden="true"></span>
                    <span class="hm-agent-activity-main">
                      <span class="hm-agent-activity-name">${esc(ev.agentName)}</span>
                      <span class="hm-agent-activity-desc">${esc(ev.description)}</span>
                      ${badge}
                    </span>
                    <time class="hm-agent-activity-time tabular-nums" datetime="${esc(ev.at)}">${esc(homeActivityRelTime(ev.at))}</time>
                  </button>
                </li>`;
            }).join('')}
          </ul>`
        : `<p class="hm-agent-activity-empty">No recent agent activity.</p>`;

    return `
      <div class="hm-agent-overview" aria-label="Agent team overview">
        <div class="hm-agent-stats" role="group" aria-label="Agent status overview">
          <div class="hm-agent-stat">
            <span class="hm-agent-stat-value tabular-nums">${esc(String(total))}</span>
            <span class="hm-agent-stat-label">Total Agents</span>
          </div>
          <div class="hm-agent-stat hm-agent-stat--paper">
            <span class="hm-agent-stat-value tabular-nums">${esc(String(paper))}</span>
            <span class="hm-agent-stat-label">Paper Trading</span>
          </div>
          <div class="hm-agent-stat hm-agent-stat--idle">
            <span class="hm-agent-stat-value tabular-nums">${esc(String(notTrading))}</span>
            <span class="hm-agent-stat-label">Not Trading</span>
          </div>
        </div>
        <div class="hm-agent-activity">
          <p class="hm-agent-activity-label">Agent Activity</p>
          ${activityHtml}
        </div>
      </div>`;
}

async function openHomeAgentActivity(event) {
    const row = event.currentTarget;
    const agentId = row?.dataset?.agentId;
    if (!agentId) return;
    const agents = homeListUserAgents();
    const agent = agents.find((a) => a.agent_id === agentId);
    if (!agent) return;

    const target = row.dataset.target || 'configure';
    const runId = row.dataset.runId || null;

    if (target === 'backtest' && typeof openAgentInBacktest === 'function') {
        await openAgentInBacktest(agent, runId || undefined);
        return;
    }
    if (target === 'paper' && typeof openAgentInPaper === 'function') {
        await openAgentInPaper(agent);
        return;
    }
    if (typeof navigateToPage === 'function') {
        navigateToPage('playground', { playgroundTab: 'agents' });
    }
    if (typeof showPlaygroundPanel === 'function') {
        showPlaygroundPanel('agents');
    }
    if (window.AgentEditor?.open) {
        window.AgentEditor.open(agent);
    }
}

function updateHomeAgentModule() {
    const empty = document.getElementById('homeAgentEmpty');
    const filled = document.getElementById('homeAgentFilled');
    const agents = homeListUserAgents();

    if (!agents.length) {
        if (empty) empty.hidden = false;
        if (filled) {
            filled.hidden = true;
            filled.innerHTML = '';
        }
        return;
    }

    if (empty) empty.hidden = true;
    if (!filled) return;
    filled.hidden = false;
    filled.innerHTML = renderHomeAgentOverview(agents);

    filled.querySelectorAll('.hm-agent-activity-row').forEach((btn) => {
        btn.addEventListener('click', (event) => {
            event.preventDefault();
            openHomeAgentActivity(event);
        });
    });
}

async function loadHomeLeaderboardModule() {
    const list = document.getElementById('homeModuleRankList');
    if (!list) return;

    // Home module shows LLM model performance only (no baselines / indices).
    const HOME_MOCK_LEADERBOARD = [
        { rank: 1, model: 'DeepSeek V4 Pro', is_model: true, cumulative_return: 0.0749, sharpe_ratio: 5.01, portfolio_value: 107490 },
        { rank: 2, model: 'Claude Sonnet 4.6', is_model: true, cumulative_return: 0.0312, sharpe_ratio: 1.18, portfolio_value: 103120 },
        { rank: 3, model: 'GPT-5.5', is_model: true, cumulative_return: 0.0281, sharpe_ratio: 0.94, portfolio_value: 102810 },
        { rank: 4, model: 'Qwen3.7 Plus', is_model: true, cumulative_return: 0.0249, sharpe_ratio: 0.72, portfolio_value: 102490 },
        { rank: 5, model: 'Gemini 3.1 Pro', is_model: true, cumulative_return: 0.0156, sharpe_ratio: 0.41, portfolio_value: 101560 },
    ];

    function isHomeModelEntry(entry) {
        return !!(entry && (entry.is_model || entry.team_badge === 'Model'));
    }

    function homeModelEntries(entries) {
        return (entries || [])
            .filter(isHomeModelEntry)
            .slice()
            .sort((a, b) => {
                const ra = Number(a.rank);
                const rb = Number(b.rank);
                if (Number.isFinite(ra) && Number.isFinite(rb) && ra !== rb) return ra - rb;
                return Number(b.cumulative_return || 0) - Number(a.cumulative_return || 0);
            })
            .map((entry, index) => ({ ...entry, rank: index + 1 }));
    }

    function homeFormatPortfolioValue(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return '—';
        if (n >= 1000) {
            return `$${Math.round(n).toLocaleString('en-US')}`;
        }
        return homeFormatMoney(n, 0);
    }

    function renderEntries(entries) {
        if (!entries.length) {
            list.innerHTML = '<li class="home-module-rank-empty">No model rankings yet.</li>';
            return;
        }
        list.innerHTML = entries.map((entry) => {
            const rank = Number(entry.rank) || 0;
            const rankClass = rank >= 1 && rank <= 3 ? ` home-module-rank--${rank}` : '';
            const label = entry.model || entry.team_name || '—';
            const ret = Number(entry.cumulative_return || 0);
            const retClass = ret >= 0 ? 'positive' : 'negative';
            const sharpe = Number(entry.sharpe_ratio || 0);
            const value = homeFormatPortfolioValue(entry.portfolio_value);
            return `<li>
                <span class="home-module-rank${rankClass}">${homeEscape(rank || '—')}</span>
                <span class="hm-rank-entry">
                    <span class="home-module-rank-name">${homeEscape(label)}</span>
                </span>
                <span class="hm-rank-value tabular-nums">${homeEscape(value)}</span>
                <span class="hm-rank-ret ${retClass} tabular-nums">${homeEscape(homeFormatReturnPct(ret))}</span>
                <span class="hm-rank-sharpe tabular-nums">${homeEscape(sharpe.toFixed(2))}</span>
            </li>`;
        }).join('');
    }

    try {
        if (typeof API === 'undefined' || typeof API_BASE === 'undefined') {
            renderEntries(homeModelEntries(HOME_MOCK_LEADERBOARD));
            return;
        }
        const payload = await API.get(`${API_BASE}/api/v1/leaderboard?t=${Date.now()}`);
        const models = homeModelEntries(payload.entries || []);

        if (!models.length) {
            renderEntries(homeModelEntries(HOME_MOCK_LEADERBOARD));
            return;
        }
        renderEntries(models);
    } catch (error) {
        console.warn('Home leaderboard module failed:', error.message);
        renderEntries(homeModelEntries(HOME_MOCK_LEADERBOARD));
    }
}

function homeSentimentClass(raw) {
    const s = String(raw || '').toLowerCase();
    if (s.includes('bull')) return 'bullish';
    if (s.includes('bear')) return 'bearish';
    return 'neutral';
}

function homeRelTime(publishedEpochSeconds) {
    if (window.formatMarketEventRelativeTime) {
        return window.formatMarketEventRelativeTime(publishedEpochSeconds * 1000, new Date());
    }
    return '';
}

const HOME_MOCK_NEWS = {
    status: 'ok',
    _mock: true,
    feed: [
        {
            ticker: 'AAPL', category: 'Earnings', sentiment: 'bullish', source: 'FinSearch',
            headline: 'Apple beats expectations as services revenue climbs again',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 12 * 60,
        },
        {
            ticker: 'NVDA', category: 'Markets', sentiment: 'bullish', source: 'FinSearch',
            headline: 'Nvidia demand remains firm as AI capex cycle extends',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 28 * 60,
        },
        {
            ticker: 'SPY', category: 'Economy', sentiment: 'neutral', source: 'FinSearch',
            headline: 'Fed speakers lean cautious ahead of next policy decision',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 46 * 60,
        },
        {
            ticker: 'TSLA', category: 'Markets', sentiment: 'bearish', source: 'FinSearch',
            headline: 'Tesla wobbles as delivery outlook stays uncertain',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 71 * 60,
        },
        {
            ticker: 'MSFT', category: 'Markets', sentiment: 'bullish', source: 'FinSearch',
            headline: 'Microsoft cloud growth steadies enterprise spending outlook',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 95 * 60,
        },
        {
            ticker: 'AMZN', category: 'Earnings', sentiment: 'bullish', source: 'FinSearch',
            headline: 'Amazon ads and AWS margins keep profit momentum intact',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 110 * 60,
        },
        {
            ticker: 'JPM', category: 'Economy', sentiment: 'neutral', source: 'FinSearch',
            headline: 'Banks brace for mixed credit trends into the next quarter',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 140 * 60,
        },
        {
            ticker: 'META', category: 'Markets', sentiment: 'bullish', source: 'FinSearch',
            headline: 'Meta ad pricing firms as engagement stays elevated',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 165 * 60,
        },
        {
            ticker: 'BA', category: 'Markets', sentiment: 'bearish', source: 'FinSearch',
            headline: 'Boeing delivery cadence remains under pressure',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 190 * 60,
        },
        {
            ticker: 'XOM', category: 'Economy', sentiment: 'neutral', source: 'FinSearch',
            headline: 'Energy majors track oil range as inventories stabilize',
            url: 'https://agenticfinsearch.org/', published: Math.floor(Date.now() / 1000) - 220 * 60,
        },
    ],
    signals: {
        AAPL: { sentiment: 'bullish', score: 0.72, rationale: 'Positive earnings impulse', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        NVDA: { sentiment: 'bullish', score: 0.68, rationale: 'AI demand remains elevated', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        TSLA: { sentiment: 'bearish', score: -0.41, rationale: 'Delivery outlook softness', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        SPY: { sentiment: 'neutral', score: 0.08, rationale: 'Macro tone mixed', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        MSFT: { sentiment: 'bullish', score: 0.55, rationale: 'Cloud demand resilient', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        AMZN: { sentiment: 'bullish', score: 0.49, rationale: 'Ads + AWS margin support', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        META: { sentiment: 'bullish', score: 0.61, rationale: 'Ad pricing firms', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        BA: { sentiment: 'bearish', score: -0.33, rationale: 'Delivery cadence pressure', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        JPM: { sentiment: 'neutral', score: 0.05, rationale: 'Credit trends mixed', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
        XOM: { sentiment: 'neutral', score: -0.02, rationale: 'Oil range-bound', source: 'FinSearch', url: 'https://agenticfinsearch.org/' },
    },
};

let homeMarketPayload = null;

function setHomeMarketDemoBadge(isMock) {
    const badge = document.getElementById('homeMarketDemoBadge');
    if (badge) badge.hidden = !isMock;
}

function renderHomeMarketNews(payload) {
    const list = document.getElementById('homeModuleNewsList');
    const status = document.getElementById('homeMarketStatus');
    if (!list) return;

    const live = payload
        && payload.status !== 'unavailable'
        && Array.isArray(payload.feed)
        && payload.feed.length > 0
        && !payload._mock;
    const data = live ? payload : HOME_MOCK_NEWS;
    setHomeMarketDemoBadge(!live);

    if (status) {
        if (live && Number.isFinite(Number(data.staleness_hours))) {
            status.hidden = false;
            status.textContent = `Updated ${Number(data.staleness_hours).toFixed(1)}h ago`;
        } else {
            status.hidden = true;
            status.textContent = '';
        }
    }

    const items = (data.feed || []).slice(0, 40);
    list.innerHTML = items.map((item) => {
        const sent = homeSentimentClass(item.sentiment || item.category);
        const logo = (item.ticker || item.source || '?').toString().slice(0, 2).toUpperCase();
        const meta = [item.ticker, item.category || item.source || 'FinSearch', homeRelTime(item.published)]
            .filter(Boolean).join(' · ');
        const label = sent === 'bullish' ? 'Bullish' : sent === 'bearish' ? 'Bearish' : 'Neutral';
        return `<li>
            <span class="hm-news-logo hm-news-logo--${sent === 'bullish' ? 'bull' : sent === 'bearish' ? 'bear' : 'neut'}">${homeEscape(logo)}</span>
            <div class="hm-news-main">
                <a href="${homeEscape(homeSafeUrl(item.url || 'https://agenticfinsearch.org/'))}" target="_blank" rel="noopener noreferrer">${homeEscape(item.headline || 'Untitled')}</a>
                <span class="hm-news-meta">${homeEscape(meta)}</span>
            </div>
            <span class="hm-sent hm-sent--${sent}">${label}</span>
        </li>`;
    }).join('');
}

function renderHomeMarketSignals(payload) {
    const list = document.getElementById('homeModuleSignalsList');
    const status = document.getElementById('homeMarketSignalsStatus');
    if (!list) return;

    const live = payload
        && payload.status !== 'unavailable'
        && payload.signals
        && Object.keys(payload.signals).length > 0
        && !payload._mock;
    const data = live ? payload : HOME_MOCK_NEWS;
    const signals = Object.entries(data.signals || {});
    setHomeMarketDemoBadge(!live);

    if (status) {
        if (live) {
            status.hidden = false;
            status.textContent = `${signals.length} signal${signals.length === 1 ? '' : 's'}`;
        } else {
            status.hidden = true;
            status.textContent = '';
        }
    }

    list.innerHTML = signals.slice(0, 40).map(([sym, s]) => {
        const sent = homeSentimentClass(s.sentiment);
        const label = sent === 'bullish' ? 'Bullish' : sent === 'bearish' ? 'Bearish' : 'Neutral';
        return `<li>
            <span class="hm-news-logo hm-news-logo--${sent === 'bullish' ? 'bull' : sent === 'bearish' ? 'bear' : 'neut'}">${homeEscape(String(sym).slice(0, 2))}</span>
            <div class="hm-news-main">
                <a href="${homeEscape(homeSafeUrl(s.url || 'https://agenticfinsearch.org/'))}" target="_blank" rel="noopener noreferrer">${homeEscape(sym)} · score ${Number(s.score || 0).toFixed(2)}</a>
                <span class="hm-news-meta">${homeEscape(s.rationale || s.source || 'FinSearch')}</span>
            </div>
            <span class="hm-sent hm-sent--${sent}">${label}</span>
        </li>`;
    }).join('');
}

async function loadHomeMarketNewsModule() {
    try {
        if (typeof API === 'undefined' || typeof API_BASE === 'undefined') {
            homeMarketPayload = HOME_MOCK_NEWS;
            renderHomeMarketNews(homeMarketPayload);
            renderHomeMarketSignals(homeMarketPayload);
            return;
        }
        homeMarketPayload = await API.get(`${API_BASE}/api/news/signals`);
        renderHomeMarketNews(homeMarketPayload);
        renderHomeMarketSignals(homeMarketPayload);
    } catch (error) {
        console.warn('FinSearch news/signals unavailable:', error?.message || error);
        homeMarketPayload = HOME_MOCK_NEWS;
        renderHomeMarketNews(homeMarketPayload);
        renderHomeMarketSignals(homeMarketPayload);
    }
}

function setHomeMarketTab(tab) {
    document.querySelectorAll('[data-market-tab]').forEach((btn) => {
        const on = btn.dataset.marketTab === tab;
        btn.classList.toggle('is-active', on);
        btn.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    const news = document.getElementById('homeMarketNewsPane');
    const signals = document.getElementById('homeMarketSignalsPane');
    if (news) news.hidden = tab !== 'news';
    if (signals) signals.hidden = tab !== 'signals';
    const title = document.getElementById('homeModuleMarketTitle');
    if (title) title.textContent = tab === 'signals' ? 'Market Signals' : 'Market News';
}

function refreshHomeModules() {
    Promise.resolve(updateHomePortfolioModule()).catch((error) => {
        console.warn('Home portfolio refresh failed:', error?.message || error);
    });
    updateHomeAgentModule();
    loadHomeLeaderboardModule();
    loadHomeMarketNewsModule();
}

function initHomeSnapScroll() {
    const view = document.getElementById('homeView');
    const track = document.getElementById('homePagerTrack');
    const hint = document.getElementById('homeScrollHint');
    const dashboard = document.getElementById('homeScreenDashboard');
    if (!view || !track || view.dataset.snapBound === '1') return;
    view.dataset.snapBound = '1';

    measureAppChromeHeight();
    window.addEventListener('resize', () => {
        measureAppChromeHeight();
        // Re-snap after chrome height changes so pages stay full-viewport.
        const page = track.dataset.page === '1' ? 1 : 0;
        setHomePagerPage(page, { instant: true });
    });

    hint?.addEventListener('click', () => setHomePagerPage(1));

    let scrollRaf = 0;
    track.addEventListener('scroll', () => {
        if (scrollRaf) return;
        scrollRaf = window.requestAnimationFrame(() => {
            scrollRaf = 0;
            const page = track.scrollTop >= track.clientHeight * 0.45 ? 1 : 0;
            track.dataset.page = String(page);
            hint?.classList.toggle('is-hidden', page === 1);
        });
    }, { passive: true });

    // Refresh dashboard modules once the second screen is mostly on-screen.
    if (dashboard && dashboard.dataset.refreshObserved !== '1') {
        dashboard.dataset.refreshObserved = '1';
        let lastRefreshAt = 0;
        const io = new IntersectionObserver(
            (entries) => {
                for (const entry of entries) {
                    if (!entry.isIntersecting || entry.intersectionRatio < 0.55) continue;
                    const now = Date.now();
                    if (now - lastRefreshAt < 800) continue;
                    lastRefreshAt = now;
                    refreshHomeModulesWhenReady();
                }
            },
            { root: track, threshold: [0.55, 0.75] },
        );
        io.observe(dashboard);
    }

    setHomePagerPage(0, { instant: true });
}

function initHomeModules() {
    document.getElementById('homeModulePortfolioBtn')?.addEventListener('click', () => {
        if (!isHomeSignedIn()) {
            if (typeof openAuthModal === 'function') openAuthModal('login');
        }
    });
    document.getElementById('homeModuleViewPortfolioBtn')?.addEventListener('click', () => {
        if (typeof navigateToPage === 'function') {
            navigateToPage('playground', { playgroundTab: 'agents' });
        }
        window.requestAnimationFrame(() => {
            document.querySelector('#playgroundAgentsPanel .page-header')?.scrollIntoView({
                block: 'start',
                behavior: 'smooth',
            });
        });
    });
    document.getElementById('homeModuleCreateAgentEmpty')?.addEventListener('click', openHomeCreateAgent);
    document.getElementById('homeModuleViewAgentsBtn')?.addEventListener('click', () => {
        if (typeof navigateToPage === 'function') {
            navigateToPage('playground', { playgroundTab: 'agents' });
        }
    });
    document.getElementById('homeModuleRankingBtn')?.addEventListener('click', navigateToLeaderboard);
    document.getElementById('homeViewLeaderboardBtn')?.addEventListener('click', navigateToLeaderboard);
    const openFinSearch = () => window.open('https://agenticfinsearch.org/', '_blank', 'noopener,noreferrer');
    document.getElementById('homeModuleMarketBtn')?.addEventListener('click', openFinSearch);
    document.getElementById('homeModuleMarketBtn')?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            openFinSearch();
        }
    });
    document.getElementById('homeModuleCommunityBtn')?.addEventListener('click', (event) => {
        if (typeof openDiscordWithAccount === 'function') {
            openDiscordWithAccount(event);
            return;
        }
        window.open('https://discord.gg/9HnQ6XDG98', '_blank', 'noopener,noreferrer');
    });

    document.querySelectorAll('[data-market-tab]').forEach((btn) => {
        btn.addEventListener('click', () => setHomeMarketTab(btn.dataset.marketTab));
    });
    document.querySelectorAll('[data-port-range]').forEach((btn) => {
        btn.addEventListener('click', () => {
            homePortRange = btn.dataset.portRange || '1D';
            document.querySelectorAll('[data-port-range]').forEach((b) => {
                b.classList.toggle('is-active', b === btn);
            });
            renderHomePortfolioChart(homePortfolioEquity(), homePortfolioDayPnl(), homePortRange);
        });
    });

    refreshHomeModules();
}

function initHomePage() {
    homeEvents = INITIAL_EVENTS.map((event) => ({ ...event }));
    if (document.getElementById('homeActivityFeed')) {
        renderActivityFeed(homeEvents.map((event) => eventToActivity(event)));
    }
    if (document.getElementById('homeMarketPulseList')) {
        initMarketPulseTabs();
    }
    initActivityFeedHover();
    initLandingPlaygroundChat();
    initHomeGetStarted();
    initHomeSnapScroll();
    initHomeModules();

    document.getElementById('homeResourceLeaderboardBtn')?.addEventListener('click', navigateToLeaderboard);

    document.getElementById('homeActivityViewAll')?.addEventListener('click', (e) => {
        e.preventDefault();
        if (typeof navigateToPage === 'function') {
            navigateToPage('playground', { playgroundTab: 'agents' });
        }
    });

    homeMockLive = useMockLiveEvents();
    if (document.getElementById('homeView')?.style.display !== 'none') {
        homeMockLive.start();
    }
}

function onHomePageShow() {
    if (!homeMockLive) homeMockLive = useMockLiveEvents();
    homeMockLive.start();
    window.newsSignalsPanel && window.newsSignalsPanel.onShow();
    measureAppChromeHeight();
    initHomeSnapScroll();
    setHomePagerPage(0, { instant: true });
    refreshHomeModulesWhenReady();
}

function onHomePageHide() {
    homeMockLive?.stop();
    hideLiveToast();
    window.newsSignalsPanel && window.newsSignalsPanel.onHide();
}

window.initHomePage = initHomePage;
window.onHomePageShow = onHomePageShow;
window.onHomePageHide = onHomePageHide;
window.useMockLiveEvents = useMockLiveEvents;
window.refreshHomeModules = refreshHomeModules;
