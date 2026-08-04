/*
 * portfolio.js — "My Portfolio" section for the My Agents page.
 *
 * Layout: left Portfolio Overview + right Capital Allocation pie.
 * Signed-in users load GET /api/v1/portfolio; guests keep SAMPLE DATA mock.
 */

const PORTFOLIO_MOCK = {
    summary: {
        totalValue: 10000,
        cashAvailable: 2000,
        allocated: 8000,
    },
};

/** @type {null | { equity: number, cash_available: number, allocated: number }} */
let livePortfolio = null;
let portfolioRenderSeq = 0;
/** In-flight GET /portfolio — shared so boot prefetch + loadAgents don't double-fetch. */
let portfolioFetchPromise = null;
const PORTFOLIO_CACHE_KEY = 'portfolio-live-cache';

const PF_WALLET_ICON =
    '<path d="M19 7V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-2"/><path d="M16 12h5v4h-5a2 2 0 0 1 0-4Z"/>';

function pfMoney(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(Number(value) || 0);
}

function portfolioPct(value, totalPortfolioValue) {
    const total = Number(totalPortfolioValue) || 0;
    if (total <= 0) return 0;
    return Math.round((Number(value) / total) * 1000) / 10;
}

function getTotalPortfolioValue() {
    if (livePortfolio) return Number(livePortfolio.equity) || 0;
    return Number(PORTFOLIO_MOCK.summary.totalValue) || 0;
}

function setPortfolioSampleBadgeVisible(visible) {
    const badge = document.getElementById('portfolioSampleBadge');
    if (!badge) return;
    badge.style.display = visible ? 'inline-block' : 'none';
}

function isPortfolioSignedIn() {
    try {
        if (typeof isSignedIn === 'function') return isSignedIn();
        if (typeof getStoredAuthUser === 'function') return !!getStoredAuthUser();
        return !!JSON.parse(localStorage.getItem('auth-user') || 'null');
    } catch (_) {
        return false;
    }
}

function normalizeSummary(summary) {
    const total = Number(summary.totalValue) || 0;
    const available = Number(summary.cashAvailable) || 0;
    const allocated = summary.allocated != null
        ? Number(summary.allocated)
        : Math.max(total - available, 0);
    return {
        totalValue: total,
        cashAvailable: available,
        allocated,
        pnlUnavailable: !!summary.pnlUnavailable,
    };
}

function summaryFromLivePortfolio(portfolio) {
    const equity = Number(portfolio.equity) || 0;
    const cash = Number(portfolio.cash_available) || 0;
    const allocated = portfolio.allocated != null
        ? Number(portfolio.allocated)
        : Math.max(equity - cash, 0);
    return {
        totalValue: equity,
        cashAvailable: cash,
        allocated,
        // The ledger tracks cash only — no marks, no history, so there is no
        // P/L to report. Flagged rather than left at 0 so the overview can
        // show "—" instead of rendering a fabricated "$0.00" that reads as a
        // real, flat day.
        pnlUnavailable: true,
    };
}

// ---------------------------------------------------------------------------
// Portfolio Overview (left card)
// ---------------------------------------------------------------------------
function renderPortfolioOverview(summary) {
    const root = document.getElementById('portfolioOverviewCard');
    if (!root) return;
    const s = normalizeSummary(summary);
    const allocPct = portfolioPct(s.allocated, s.totalValue);
    const availPct = portfolioPct(s.cashAvailable, s.totalValue);
    const barAlloc = Math.min(Math.max(allocPct, 0), 100);
    const barAvail = Math.min(Math.max(100 - barAlloc, 0), 100);

    root.innerHTML = `
        <div class="pf-overview-head">
            <span class="pf-overview-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">${PF_WALLET_ICON}</svg>
            </span>
            <h3 class="pf-overview-title">Portfolio Overview</h3>
        </div>
        <div class="pf-overview-hero">
            <div class="pf-overview-hero-col">
                <span class="pf-overview-label">Total Portfolio Value</span>
                <span class="pf-overview-total">${pfMoney(s.totalValue)}</span>
            </div>
            <div class="pf-overview-hero-col pf-overview-hero-col--right">
                <span class="pf-overview-label">Today's P&amp;L</span>
                <span class="pf-overview-pnl">${s.pnlUnavailable ? '—' : pfMoney(0)}</span>
            </div>
        </div>
        <div class="pf-overview-split">
            <div class="pf-overview-split-col">
                <span class="pf-overview-label">Unallocated Cash</span>
                <span class="pf-overview-split-row">
                    <span class="pf-overview-split-value">${pfMoney(s.cashAvailable)}</span>
                    <span class="pf-overview-pill pf-overview-pill--avail">${availPct}%</span>
                </span>
            </div>
            <div class="pf-overview-split-col pf-overview-split-col--right">
                <span class="pf-overview-label">Allocated to Agents</span>
                <span class="pf-overview-split-row">
                    <span class="pf-overview-split-value">${pfMoney(s.allocated)}</span>
                    <span class="pf-overview-pill pf-overview-pill--alloc">${allocPct}%</span>
                </span>
            </div>
        </div>
        <div class="pf-overview-bar" role="img" aria-label="Unallocated ${availPct} percent, allocated ${allocPct} percent">
            <span class="pf-overview-bar-avail" style="width:${barAvail}%"></span>
            <span class="pf-overview-bar-alloc" style="width:${barAlloc}%"></span>
        </div>
    `;
}

// ---------------------------------------------------------------------------
// Capital Allocation donut (right card)
// ---------------------------------------------------------------------------
const AGENT_SLICE_COLORS = ['#22d3ee', '#a855f7', '#34d399', '#fbbf24', '#f87171', '#c084fc', '#38bdf8', '#2dd4bf'];
const AVAILABLE_SLICE_COLOR = '#64748b';
const pfChartInstances = {};
/** Compact scroll height; expand raises the cap. All agents stay in the list either way. */
const ALLOCATION_LEGEND_COLLAPSED_LIMIT = 6;
const ALLOCATION_LEGEND_SCROLL_MAX_PX = 220;
const ALLOCATION_LEGEND_EXPANDED_MAX_PX = 420;
let allocationLegendExpanded = false;
let allocationLegendBound = false;

function buildAgentAllocationData(agents, totalPortfolioValue) {
    const total = Number(totalPortfolioValue) || 0;
    const assignedAgents = (agents || []).filter(
        (agent) => agent.cash_allocation != null && Number(agent.cash_allocation) > 0,
    );

    if (total <= 0) {
        return {
            total: 0,
            slices: [{ label: 'Unallocated', value: 0, pct: 0, color: AVAILABLE_SLICE_COLOR }],
        };
    }

    if (!assignedAgents.length) {
        return {
            total,
            slices: [{
                label: 'Unallocated',
                value: total,
                pct: 100,
                color: AVAILABLE_SLICE_COLOR,
            }],
        };
    }

    const assignedTotal = assignedAgents.reduce(
        (sum, agent) => sum + Number(agent.cash_allocation),
        0,
    );
    const available = Math.max(total - assignedTotal, 0);
    const overAllocated = assignedTotal > total;
    const chartScale = overAllocated && assignedTotal > 0 ? total / assignedTotal : 1;
    const slices = [];

    // Agents first, Unallocated last (legend + pie order).
    assignedAgents.forEach((agent, index) => {
        const assignedCapital = Number(agent.cash_allocation);
        slices.push({
            label: agent.name || 'Agent',
            value: assignedCapital * chartScale,
            assignedCapital,
            pct: portfolioPct(assignedCapital, total),
            color: AGENT_SLICE_COLORS[index % AGENT_SLICE_COLORS.length],
        });
    });

    if (available > 0) {
        slices.push({
            label: 'Unallocated',
            value: available,
            pct: portfolioPct(available, total),
            color: AVAILABLE_SLICE_COLOR,
        });
    }

    if (!slices.length) {
        slices.push({
            label: 'Unallocated',
            value: total,
            pct: 100,
            color: AVAILABLE_SLICE_COLOR,
        });
    }

    return { total, slices, overAllocated };
}

function pfAllocationSignature(data) {
    return (data.slices || [])
        .map((s) => `${s.label}:${Number(s.value) || 0}:${s.color}`)
        .join('|');
}

function renderAllocationChart(key, data, settleRetries = 60) {
    const canvas = document.getElementById(`${key}AllocationChart`);
    const legendEl = document.getElementById(`${key}AllocationLegend`);
    if (!canvas || typeof Chart === 'undefined') return;

    // Avoid first paint at 0×0 (layout not settled) — that causes a size jump.
    // Bounded: a hidden panel never gains width, so an unbounded retry would
    // spin at 60fps forever. After ~1s create the chart anyway and let
    // Chart.js's responsive resize take over once the panel is visible.
    const wrap = canvas.parentElement;
    if (!pfChartInstances[key] && wrap && wrap.clientWidth < 8 && settleRetries > 0) {
        requestAnimationFrame(() => renderAllocationChart(key, data, settleRetries - 1));
        return;
    }

    const labels = data.slices.map((s) => s.label);
    const values = data.slices.map((s) => s.value);
    const colors = data.slices.map((s) => s.color);
    const signature = pfAllocationSignature(data);
    canvas._pfSliceData = data;

    if (pfChartInstances[key]) {
        if (canvas._pfSignature === signature) {
            // Same slices — refresh legend only, do not re-animate.
        } else {
            const chart = pfChartInstances[key];
            chart.data.labels = labels;
            chart.data.datasets[0].data = values;
            chart.data.datasets[0].backgroundColor = colors;
            canvas._pfSignature = signature;
            // Animate data morph; skip resize animation so layout settle doesn't twitch.
            chart.update();
        }
    } else {
        canvas._pfSignature = signature;
        pfChartInstances[key] = new Chart(canvas.getContext('2d'), {
            type: 'pie',
            data: {
                labels,
                datasets: [
                    {
                        data: values,
                        backgroundColor: colors,
                        borderColor: 'rgba(10, 14, 39, 0.95)',
                        borderWidth: 2,
                        hoverOffset: 4,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                resizeDelay: 80,
                animation: {
                    duration: 650,
                    easing: 'easeOutQuart',
                    animateRotate: true,
                    animateScale: true,
                },
                transitions: {
                    // Prevent layout/resize from replaying the entrance animation.
                    resize: { animation: { duration: 0 } },
                    show: { animations: { colors: false, numbers: { duration: 650 } } },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const sliceData = canvas._pfSliceData || data;
                                const slice = sliceData.slices[ctx.dataIndex];
                                if (!slice) return '';
                                const amount = slice.assignedCapital != null
                                    ? slice.assignedCapital
                                    : slice.value;
                                return `${slice.label}: ${slice.pct}% · ${pfMoney(amount)}`;
                            },
                        },
                    },
                },
            },
        });
    }

    if (legendEl) {
        renderAllocationLegend(legendEl, data.slices);
    }
}

function allocationLegendRowHtml(slice) {
    const displayValue = slice.assignedCapital != null ? slice.assignedCapital : slice.value;
    return `
            <li class="allocation-legend-row">
                <span class="allocation-legend-name">
                    <span class="allocation-legend-dot" style="background:${slice.color}"></span>
                    ${escapeHtml(slice.label)}
                </span>
                <span class="allocation-legend-pct">${slice.pct}%</span>
                <span class="allocation-legend-value">${pfMoney(displayValue)}</span>
            </li>`;
}

function sortAllocationLegendSlices(slices) {
    const unallocated = slices.filter((s) => s.label === 'Unallocated' || s.label === 'Loading');
    const agents = slices.filter((s) => s.label !== 'Unallocated' && s.label !== 'Loading');
    agents.sort((a, b) => {
        const av = a.assignedCapital != null ? a.assignedCapital : a.value;
        const bv = b.assignedCapital != null ? b.assignedCapital : b.value;
        return bv - av;
    });
    return [...agents, ...unallocated];
}

function renderAllocationLegend(legendEl, slices) {
    const sorted = sortAllocationLegendSlices(slices || []);
    const agentCount = sorted.filter((s) => s.label !== 'Unallocated' && s.label !== 'Loading').length;
    const needsExpand = agentCount > ALLOCATION_LEGEND_COLLAPSED_LIMIT;
    const expanded = allocationLegendExpanded && needsExpand;
    if (!needsExpand) allocationLegendExpanded = false;

    const rowsHtml = sorted.map(allocationLegendRowHtml).join('');
    const maxHeight = expanded ? ALLOCATION_LEGEND_EXPANDED_MAX_PX : ALLOCATION_LEGEND_SCROLL_MAX_PX;
    const toggleHtml = needsExpand
        ? `<div class="allocation-legend-toggle-row">
            <button type="button" class="allocation-legend-toggle" data-allocation-legend-toggle aria-expanded="${expanded ? 'true' : 'false'}">
                ${expanded ? 'Show less' : `Show all (${agentCount})`}
            </button>
           </div>`
        : '';

    legendEl.innerHTML =
        `<div class="allocation-legend-scroll${expanded ? ' allocation-legend-scroll--expanded' : ''}" style="max-height:${maxHeight}px">` +
        `<ul class="allocation-legend-list">${rowsHtml}</ul></div>${toggleHtml}`;

    if (!allocationLegendBound) {
        allocationLegendBound = true;
        legendEl.closest('.allocation-detail')?.addEventListener('click', (event) => {
            const btn = event.target.closest('[data-allocation-legend-toggle]');
            if (!btn) return;
            allocationLegendExpanded = btn.getAttribute('aria-expanded') !== 'true';
            const canvas = document.getElementById('agentAllocationChart');
            const sliceData = canvas?._pfSliceData;
            if (sliceData?.slices) {
                renderAllocationLegend(legendEl, sliceData.slices);
            }
        });
    }
}

function updateAgentAllocationFromAgents(agents) {
    renderAllocationChart('agent', buildAgentAllocationData(agents, getTotalPortfolioValue()));
}

/** Placeholder pie while agents are still loading — avoids a false "all Unallocated" flash. */
function renderAllocationLoading() {
    const legendEl = document.getElementById('agentAllocationLegend');
    if (legendEl) {
        legendEl.innerHTML =
            '<ul class="allocation-legend-list">' +
            '<li class="allocation-legend-item allocation-legend-item--loading">' +
            '<span class="allocation-legend-label">Loading agents…</span></li></ul>';
    }
    renderAllocationChart('agent', {
        total: getTotalPortfolioValue(),
        slices: [{
            label: 'Loading',
            value: Math.max(getTotalPortfolioValue(), 1),
            pct: 100,
            color: AVAILABLE_SLICE_COLOR,
        }],
    });
}

function readCachedPortfolio() {
    try {
        const raw = sessionStorage.getItem(PORTFOLIO_CACHE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return null;
        return {
            equity: Number(parsed.equity) || 0,
            cash_available: Number(parsed.cash_available) || 0,
            allocated: Number(parsed.allocated) || 0,
        };
    } catch (_) {
        return null;
    }
}

function writeCachedPortfolio(portfolio) {
    try {
        sessionStorage.setItem(
            PORTFOLIO_CACHE_KEY,
            JSON.stringify({
                equity: Number(portfolio.equity) || 0,
                cash_available: Number(portfolio.cash_available) || 0,
                allocated: Number(portfolio.allocated) || 0,
            }),
        );
    } catch (_) {
        /* sessionStorage unavailable — ignore */
    }
}

function clearCachedPortfolio() {
    try {
        sessionStorage.removeItem(PORTFOLIO_CACHE_KEY);
    } catch (_) {
        /* ignore */
    }
}

function renderPortfolioFromMock(agents, options) {
    options = options || {};
    livePortfolio = null;
    clearCachedPortfolio();
    setPortfolioSampleBadgeVisible(true);
    renderPortfolioOverview(PORTFOLIO_MOCK.summary);
    if (options.deferAllocation) {
        renderAllocationLoading();
        return;
    }
    renderAllocationChart('agent', buildAgentAllocationData(agents, getTotalPortfolioValue()));
}

function renderPortfolioFromLive(portfolio, agents, options) {
    options = options || {};
    livePortfolio = {
        equity: Number(portfolio.equity) || 0,
        cash_available: Number(portfolio.cash_available) || 0,
        allocated: Number(portfolio.allocated) || 0,
    };
    writeCachedPortfolio(livePortfolio);
    setPortfolioSampleBadgeVisible(false);
    renderPortfolioOverview(summaryFromLivePortfolio(livePortfolio));
    if (options.deferAllocation) {
        renderAllocationLoading();
        return;
    }
    updateAgentAllocationFromAgents(agents);
}

function fetchLivePortfolio() {
    if (portfolioFetchPromise) return portfolioFetchPromise;
    portfolioFetchPromise = (async () => {
        try {
            const data = await API.get(`${API_BASE}/api/v1/portfolio`);
            return data && data.portfolio ? data.portfolio : null;
        } finally {
            portfolioFetchPromise = null;
        }
    })();
    return portfolioFetchPromise;
}

// ---------------------------------------------------------------------------
// Public entry points — My Agents tab + boot prefetch.
// ---------------------------------------------------------------------------

// Renders are async and callers do not await them, so two can be in flight at
// once (boot prefetch, then loadAgents). Responses are not guaranteed to
// arrive in request order, so without this sequence guard a slower earlier
// request can repaint the panel with stale agents after the newer one landed.
async function renderPortfolio(agents, options) {
    options = options || {};
    const list = agents || [];
    const deferAllocation = !!options.deferAllocation;
    const seq = ++portfolioRenderSeq;
    if (!isPortfolioSignedIn() || typeof API === 'undefined' || typeof API_BASE === 'undefined') {
        if (seq !== portfolioRenderSeq) return;
        renderPortfolioFromMock(list, { deferAllocation: deferAllocation });
        return;
    }
    // Instant paint from memory / sessionStorage before the network round-trip.
    const cached = livePortfolio || readCachedPortfolio();
    if (cached) {
        renderPortfolioFromLive(cached, list, { deferAllocation: deferAllocation });
    } else if (deferAllocation) {
        renderAllocationLoading();
    }
    try {
        const portfolio = await fetchLivePortfolio();
        if (seq !== portfolioRenderSeq) return;
        if (!portfolio) {
            if (!livePortfolio) renderPortfolioFromMock(list, { deferAllocation: deferAllocation });
            return;
        }
        renderPortfolioFromLive(portfolio, list, { deferAllocation: deferAllocation });
    } catch (error) {
        if (seq !== portfolioRenderSeq) return;
        console.warn('Portfolio API unavailable; showing sample data:', error?.message || error);
        // Keep last-known figures if we already painted them; only fall back to
        // SAMPLE DATA when the panel would otherwise stay blank.
        if (!livePortfolio) renderPortfolioFromMock(list, { deferAllocation: deferAllocation });
    }
}

/**
 * Repaint the panel from memory or sessionStorage — no network.
 *
 * Lets hard refresh and My Agents tab revisit paint instantly while the
 * authoritative refresh rides along with prefetchPortfolio / loadAgents.
 */
function repaintPortfolioFromCache(agents, options) {
    options = options || {};
    const list = agents || [];
    const deferAllocation = !!options.deferAllocation;
    if (livePortfolio) {
        renderPortfolioFromLive(livePortfolio, list, { deferAllocation: deferAllocation });
        return;
    }
    if (!isPortfolioSignedIn()) {
        renderPortfolioFromMock(list, { deferAllocation: deferAllocation });
        return;
    }
    const cached = readCachedPortfolio();
    if (cached) {
        renderPortfolioFromLive(cached, list, { deferAllocation: deferAllocation });
        return;
    }
    if (deferAllocation) {
        renderAllocationLoading();
    }
    // Signed in but nothing cached yet: leave it be — prefetch / loadAgents
    // is about to paint the real figures.
}

/** Boot helper: overview from cache; defer pie until loadAgents has the list. */
function paintPortfolioBoot(agents) {
    repaintPortfolioFromCache(agents, { deferAllocation: true });
}

/** Boot helper: fetch portfolio totals without painting a false empty-agent pie. */
async function prefetchPortfolio(agents) {
    return renderPortfolio(agents || [], { deferAllocation: true });
}

window.renderPortfolio = renderPortfolio;
window.repaintPortfolioFromCache = repaintPortfolioFromCache;
window.paintPortfolioBoot = paintPortfolioBoot;
window.prefetchPortfolio = prefetchPortfolio;
window.updateAgentAllocationFromAgents = updateAgentAllocationFromAgents;
