/**
 * agent-editor.js — Fullscreen agent Configure screen.
 * Pipeline agents edit capital + one plain-language instruction + model.
 * AI Hedge Fund agents edit capital + analyst composition and can place their
 * Financial Datasets API key into server-side encrypted credential storage.
 *
 * The multi-step "Advanced" sub-agent editor was removed. Multi-step pipelines
 * still EXECUTE server-side (infrastructure/llm/pipeline_runner.py) and the
 * marketplace still ships a 3-step template, so an agent may legitimately hold a
 * pipeline this screen cannot author. Such a pipeline is carried opaquely in
 * `subAgents` and re-sent when the user has an instruction; an EMPTY
 * instruction deliberately clears it instead, so the backend falls back to
 * its platform default -- save() guards that with a confirm when the
 * existing pipeline is multi-step. See `sendPipeline` in getEditorState().
 */
(function () {
  'use strict';

  const STORAGE_PREFIX = 'agent-pipeline-config:';
  const NAME_OVERRIDE_PREFIX = 'agent-name-override:';
  const CASH_OVERRIDE_PREFIX = 'agent-cash-allocation:';
  const AI_HEDGE_FUND_RUNTIME = 'ai_hedge_fund';
  const AI_HEDGE_FUND_ANALYSTS = [
    {
      id: 'aswath_damodaran',
      label: 'Aswath Damodaran',
      description: 'Estimates intrinsic value by connecting company narratives with growth, reinvestment, risk, cash flow, and relative valuation.',
    },
    {
      id: 'ben_graham',
      label: 'Ben Graham',
      description: 'Uses value-investing principles and a margin of safety to identify financially strong, potentially undervalued companies.',
    },
    {
      id: 'bill_ackman',
      label: 'Bill Ackman',
      description: 'Seeks high-quality, cash-generative businesses where financial discipline, catalysts, or activism can unlock value.',
    },
    {
      id: 'cathie_wood',
      label: 'Cathie Wood',
      description: 'Looks for disruptive, innovation-led companies with exponential growth potential, large markets, and long time horizons.',
    },
    {
      id: 'charlie_munger',
      label: 'Charlie Munger',
      description: 'Favors predictable, high-quality businesses with durable moats and strong management at fair valuations.',
    },
    {
      id: 'michael_burry',
      label: 'Michael Burry',
      description: 'Hunts for contrarian deep-value opportunities using free cash flow, EV/EBIT, balance-sheet risk, and hard catalysts.',
    },
    {
      id: 'mohnish_pabrai',
      label: 'Mohnish Pabrai',
      description: 'Seeks simple, durable businesses with protected downside, high free-cash-flow yields, low leverage, and doubling potential.',
    },
    {
      id: 'nassim_taleb',
      label: 'Nassim Taleb',
      description: 'Evaluates tail risk, antifragility, fragility, and convexity to find asymmetric opportunities with limited downside.',
    },
    {
      id: 'peter_lynch',
      label: 'Peter Lynch',
      description: 'Looks for understandable businesses with steady growth and ten-bagger potential at reasonable PEG-based valuations.',
    },
    {
      id: 'phil_fisher',
      label: 'Phil Fisher',
      description: 'Seeks long-term growers with strong management, R&D investment, durable margins, and thorough scuttlebutt-style research.',
    },
    {
      id: 'rakesh_jhunjhunwala',
      label: 'Rakesh Jhunjhunwala',
      description: 'Seeks understandable, financially strong businesses with durable moats, quality management, consistent growth, and a margin of safety.',
    },
    {
      id: 'stanley_druckenmiller',
      label: 'Stanley Druckenmiller',
      description: 'Targets asymmetric opportunities with strong growth, price momentum, and sentiment while controlling drawdown, leverage, and volatility risk.',
    },
    {
      id: 'warren_buffett',
      label: 'Warren Buffett',
      description: 'Seeks businesses with durable moats, strong management, predictable fundamentals, and intrinsic value above market price.',
    },
    {
      id: 'technical_analyst',
      label: 'Technical',
      description: 'Analyzes price trends and technical indicators to identify trading signals.',
    },
    {
      id: 'fundamentals_analyst',
      label: 'Fundamentals',
      description: 'Evaluates company financial health using profitability, growth, leverage, and other fundamental metrics.',
    },
    {
      id: 'growth_analyst',
      label: 'Growth',
      description: 'Identifies opportunities using revenue, earnings, and cash-flow growth, valuation, margins, insider activity, and financial health.',
    },
    {
      id: 'news_sentiment_analyst',
      label: 'News Sentiment',
      description: 'Analyzes company-news sentiment, classifying recent headlines when needed, and aggregates it into a trading signal.',
    },
    {
      id: 'sentiment_analyst',
      label: 'Sentiment',
      description: 'Combines company-news sentiment with insider trading activity to gauge market sentiment and generate a signal.',
    },
    {
      id: 'valuation_analyst',
      label: 'Valuation',
      description: 'Estimates intrinsic value using DCF, owner earnings, EV/EBITDA, and residual-income models, then compares it with market value.',
    },
  ];
  // Match app.js: same-origin locally, hosted backend everywhere else. In
  // Same-origin API base. Local uvicorn serves the backend; production Vercel
  // rewrites API paths to Render (see vercel.json). Empty string = root-relative.
  const API_BASE =
    window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
      ? window.location.origin
      : '';

  // The simple-instruction contract (preset key + trading-actions output format)
  // has a single source of truth in app.js, published on `window`. app.js loads
  // AFTER this file, so read lazily at call time — every use below is inside an
  // event-driven function, by which point window.* is populated. The fallbacks
  // only matter if app.js somehow failed to load (whole app is broken anyway).
  function simplePresetKey() {
    return (
      (typeof window !== 'undefined' && window.SIMPLE_INSTRUCTION_PRESET_KEY) ||
      'simple_instruction'
    );
  }
  function simpleOutputFormat() {
    return (
      (typeof window !== 'undefined' && window.SIMPLE_INSTRUCTION_OUTPUT_FORMAT) || ''
    );
  }
  function defaultStarterInstruction() {
    return (typeof window !== 'undefined' && window.DEFAULT_STARTER_INSTRUCTION) || '';
  }

  // Demo/mock agents (see MOCK_AGENTS in app.js) only exist in the frontend —
  // they have no database row, so PATCH would 404. We persist their edits
  // locally instead so the rename is still reflected in the UI.
  function isDemoAgent(agentId) {
    return typeof agentId === 'string' && agentId.startsWith('mock-');
  }

  let currentAgent = null;
  let subAgents = [];
  let saveStatusTimer = null;
  let isDirty = false;
  let savedSnapshot = '';
  let financialDatasetsConfigured = false;
  let activeCpTab = 'overview';
  let cachedRuns = [];
  // True when Market shows US Stocks only as a UI default for an uncategorized
  // agent (DB category still null). Cleared when the user touches the select so
  // Save of unrelated fields cannot silently shelf the agent.
  let categoryUiDefaulted = false;

  function isAiHedgeFundAgent(agent = currentAgent) {
    return (agent?.runtime_type || 'pipeline') === AI_HEDGE_FUND_RUNTIME;
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function storageKey(agentId) {
    return `${STORAGE_PREFIX}${agentId}`;
  }

  function newSubAgentId() {
    return `sub_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  }

  // Structural only: this screen no longer knows about sub-agent presets, it
  // just round-trips whatever shape is already stored.
  function normalizeLoadedSubAgent(item) {
    return {
      id: item.id || newSubAgentId(),
      presetKey: item.presetKey || 'custom',
      label: item.label || 'Sub-agent',
      prompt: item.prompt || '',
      outputFormat: item.outputFormat || '',
    };
  }

  function isSimplePipeline(pipeline) {
    return (
      !Array.isArray(pipeline) ||
      pipeline.length === 0 ||
      (pipeline.length === 1 && pipeline[0].presetKey === simplePresetKey())
    );
  }

  // The pipeline the agent ACTUALLY has (backend row, then local cache). Returns
  // [] when it has none -- an empty pipeline is a supported state (the platform
  // default), not something open() backfills any more.
  function loadStoredPipeline(agent) {
    if (Array.isArray(agent.pipeline) && agent.pipeline.length) {
      return agent.pipeline.map(normalizeLoadedSubAgent);
    }
    try {
      const raw = localStorage.getItem(storageKey(agent.agent_id));
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed.subAgents) && parsed.subAgents.length) {
          return parsed.subAgents.map(normalizeLoadedSubAgent);
        }
      }
    } catch {
      /* fall through to empty */
    }
    return [];
  }

  function savePipelineLocal(agentId, agents) {
    localStorage.setItem(
      storageKey(agentId),
      JSON.stringify({ subAgents: agents, updatedAt: new Date().toISOString() })
    );
  }

  function readCsrfToken() {
    try {
      const raw = document.cookie || '';
      for (const name of ['atl_csrf', '__Host-atl_csrf']) {
        const match = raw.match(new RegExp('(?:^|; )' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '=([^;]*)'));
        if (match) return decodeURIComponent(match[1]);
      }
    } catch (_) { /* ignore */ }
    return null;
  }

  function authHeaders() {
    const headers = { 'x-session-id': window.SESSION_ID };
    const csrf = (typeof window.csrfHeaders === 'function')
      ? window.csrfHeaders()
      : (readCsrfToken() ? { 'X-CSRF-Token': readCsrfToken() } : {});
    return { ...headers, ...csrf };
  }

  function isEditorSignedIn() {
    if (typeof window.getStoredAuthUser === 'function') return !!window.getStoredAuthUser();
    try {
      return !!JSON.parse(localStorage.getItem('auth-user') || 'null');
    } catch (_) {
      return false;
    }
  }

  function positionAiHedgeFundTooltip(option) {
    const tooltip = option?.querySelector('.agent-editor-analyst-tooltip');
    if (!tooltip) return;

    const optionRect = option.getBoundingClientRect();
    const margin = 12;
    const gap = 8;
    const visualViewport = window.visualViewport;
    const viewportLeft = visualViewport?.offsetLeft || 0;
    const viewportTop = visualViewport?.offsetTop || 0;
    const viewportWidth = Math.min(
      visualViewport?.width || window.innerWidth,
      document.documentElement.clientWidth || window.innerWidth,
    );
    const viewportHeight = Math.min(
      visualViewport?.height || window.innerHeight,
      document.documentElement.clientHeight || window.innerHeight,
    );
    const viewportRight = viewportLeft + viewportWidth;
    const viewportBottom = viewportTop + viewportHeight;
    const editorScrollRect = option.closest('.agent-cp-main, .agent-editor-body')?.getBoundingClientRect();
    const safeTop = Math.max(
      viewportTop + margin,
      (editorScrollRect?.top || viewportTop) + margin,
    );
    const safeBottom = Math.min(
      viewportBottom - margin,
      (editorScrollRect?.bottom || viewportBottom) - margin,
    );

    if (optionRect.bottom <= safeTop || optionRect.top >= safeBottom) {
      tooltip.hidden = true;
      option.classList.remove('is-tooltip-visible');
      if (activeAiHedgeFundTooltipOption === option) {
        activeAiHedgeFundTooltipOption = null;
      }
      return;
    }

    tooltip.hidden = false;
    tooltip.style.setProperty(
      '--analyst-tooltip-max-width',
      `${Math.max(1, viewportWidth - margin * 2)}px`,
    );
    const tooltipRect = tooltip.getBoundingClientRect();

    const preferredLeft = optionRect.left + (optionRect.width - tooltipRect.width) / 2;
    const safeLeft = viewportLeft + margin;
    const maxLeft = Math.max(safeLeft, viewportRight - tooltipRect.width - margin);
    const left = Math.min(Math.max(preferredLeft, safeLeft), maxLeft);

    const above = optionRect.top - tooltipRect.height - gap;
    const below = optionRect.bottom + gap;
    const preferredTop = above >= safeTop ? above : below;
    const maxTop = Math.max(safeTop, safeBottom - tooltipRect.height);
    const top = Math.min(Math.max(preferredTop, safeTop), maxTop);

    tooltip.style.setProperty('--analyst-tooltip-left', `${Math.round(left)}px`);
    tooltip.style.setProperty('--analyst-tooltip-top', `${Math.round(top)}px`);
  }

  let activeAiHedgeFundTooltipOption = null;
  let analystTooltipPositionFrame = null;

  function showAiHedgeFundTooltip(option) {
    if (activeAiHedgeFundTooltipOption !== option) {
      activeAiHedgeFundTooltipOption?.classList.remove('is-tooltip-visible');
      activeAiHedgeFundTooltipOption = option;
    }
    positionAiHedgeFundTooltip(option);
    option.classList.add('is-tooltip-visible');
  }

  function hideAiHedgeFundTooltip(option) {
    option?.classList.remove('is-tooltip-visible');
    if (activeAiHedgeFundTooltipOption === option) {
      activeAiHedgeFundTooltipOption = null;
    }
  }

  function restoreKeyboardFocusedAiHedgeFundTooltip() {
    const focusedInput = document.querySelector(
      'input[name="agentEditorAiHedgeFundAnalyst"]:focus-visible',
    );
    const focusedOption = focusedInput?.closest('.agent-editor-analyst-option');
    if (focusedOption) showAiHedgeFundTooltip(focusedOption);
  }

  function scheduleActiveAiHedgeFundTooltipPosition() {
    if (
      analystTooltipPositionFrame !== null
      || !activeAiHedgeFundTooltipOption?.isConnected
    ) return;
    analystTooltipPositionFrame = window.requestAnimationFrame(() => {
      analystTooltipPositionFrame = null;
      if (activeAiHedgeFundTooltipOption?.isConnected) {
        positionAiHedgeFundTooltip(activeAiHedgeFundTooltipOption);
      }
    });
  }

  function renderAiHedgeFundAnalysts(agent) {
    const grid = document.getElementById('agentEditorAiHedgeFundAnalysts');
    if (!grid) return;
    hideAiHedgeFundTooltip(activeAiHedgeFundTooltipOption);
    const selected = new Set(agent?.runtime_config?.analysts || []);
    grid.innerHTML = AI_HEDGE_FUND_ANALYSTS.map(({ id, label, description }) => {
      const labelId = `agentEditorAiHedgeFundAnalyst-${id}-label`;
      const tooltipId = `agentEditorAiHedgeFundAnalyst-${id}-tooltip`;
      return `
        <label class="agent-editor-analyst-option">
          <input type="checkbox" name="agentEditorAiHedgeFundAnalyst" value="${escapeHtml(id)}" aria-labelledby="${escapeHtml(labelId)}" aria-describedby="${escapeHtml(tooltipId)}" ${selected.has(id) ? 'checked' : ''}>
          <span id="${escapeHtml(labelId)}">${escapeHtml(label)}</span>
          <span id="${escapeHtml(tooltipId)}" class="agent-editor-analyst-tooltip" role="tooltip">${escapeHtml(description)}</span>
        </label>`;
    }).join('');
    grid.querySelectorAll('.agent-editor-analyst-option').forEach((option) => {
      option.addEventListener('mouseenter', () => showAiHedgeFundTooltip(option));
      option.addEventListener('mouseleave', () => {
        hideAiHedgeFundTooltip(option);
        restoreKeyboardFocusedAiHedgeFundTooltip();
      });
      option.addEventListener('focusin', () => {
        window.requestAnimationFrame(() => {
          if (option.querySelector('input:focus-visible')) {
            showAiHedgeFundTooltip(option);
          }
        });
      });
      option.addEventListener('focusout', (event) => {
        if (!option.contains(event.relatedTarget)) hideAiHedgeFundTooltip(option);
      });
    });
  }

  function selectedAiHedgeFundAnalysts() {
    return Array.from(
      document.querySelectorAll('input[name="agentEditorAiHedgeFundAnalyst"]:checked')
    ).map((input) => input.value);
  }

  function setFinancialDatasetsStatus(configured, message) {
    financialDatasetsConfigured = Boolean(configured);
    const status = document.getElementById('agentEditorFinancialDatasetsStatus');
    if (status) {
      status.textContent = message || (
        financialDatasetsConfigured
          ? 'Credential configured — enter a new key only to replace it.'
          : 'Credential not configured — required before Run Backtest.'
      );
    }
    // Without this the DELETE route has no UI path at all: a user could store a
    // third-party key and never remove it.
    const removeBtn = document.getElementById('agentEditorFinancialDatasetsRemove');
    if (removeBtn) removeBtn.hidden = !financialDatasetsConfigured;
  }

  async function removeFinancialDatasetsCredential() {
    const agent = currentAgent;
    if (!agent || !financialDatasetsConfigured) return;
    const removeBtn = document.getElementById('agentEditorFinancialDatasetsRemove');
    if (removeBtn) removeBtn.disabled = true;
    try {
      await credentialRequest(agent, 'DELETE');
      if (currentAgent?.agent_id !== agent.agent_id) return;
      const keyInput = document.getElementById('agentEditorFinancialDatasetsKey');
      if (keyInput) keyInput.value = '';
      setFinancialDatasetsStatus(false, 'Stored key removed.');
    } catch (error) {
      if (currentAgent?.agent_id !== agent.agent_id) return;
      setFinancialDatasetsStatus(
        financialDatasetsConfigured,
        `Could not remove the stored key: ${error.message}`,
      );
    } finally {
      if (removeBtn) removeBtn.disabled = false;
    }
  }

  async function credentialRequest(agent, method, body) {
    const endpoint = `${API_BASE}/api/v1/agents/${encodeURIComponent(agent.agent_id)}/credentials/financial-datasets`;

    async function send(extraHeaders) {
      const headers = {
        'Content-Type': 'application/json',
        ...authHeaders(),
        ...extraHeaders,
      };
      const response = await fetch(endpoint, {
        method,
        headers,
        credentials: 'include',
        body: body == null ? undefined : JSON.stringify(body),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail || data.message || `HTTP ${response.status}`;
        const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
        error.status = response.status;
        throw error;
      }
      return data;
    }

    try {
      return await send({
        'x-browser-id': window.BROWSER_OWNER_ID,
        'x-session-id': agent.session_id || window.SESSION_ID,
      });
    } catch (error) {
      if (error.status !== 403 || !agent?.session_id) throw error;
      return send({ 'x-session-id': agent.session_id, 'x-browser-id': '' });
    }
  }

  async function refreshFinancialDatasetsStatus(agent) {
    if (!isAiHedgeFundAgent(agent)) return;
    setFinancialDatasetsStatus(false, 'Checking credential…');
    try {
      const status = await credentialRequest(agent, 'GET');
      if (currentAgent?.agent_id !== agent.agent_id) return;
      setFinancialDatasetsStatus(status.configured);
    } catch (error) {
      if (currentAgent?.agent_id !== agent.agent_id) return;
      setFinancialDatasetsStatus(false, `Credential status unavailable: ${error.message}`);
    }
  }

  function formatUsd(value) {
    if (value == null || Number.isNaN(Number(value))) return '—';
    return `$${Number(value).toLocaleString(undefined, {
      maximumFractionDigits: Number(value) % 1 === 0 ? 0 : 2,
    })}`;
  }

  function formatReturnPct(run) {
    if (typeof window.formatBacktestRunReturn === 'function') {
      return window.formatBacktestRunReturn(run);
    }
    if (run?.total_return == null) return '—';
    const pct = Math.abs(run.total_return) <= 1 ? run.total_return * 100 : run.total_return;
    const sign = pct >= 0 ? '+' : '';
    return `${sign}${pct.toFixed(1)}%`;
  }

  function returnToneClass(run) {
    if (run?.total_return == null || Number.isNaN(Number(run.total_return))) return '';
    return Number(run.total_return) >= 0 ? 'is-positive' : 'is-negative';
  }

  function formatRelativeTime(iso) {
    if (typeof window.formatRelativeTime === 'function') return window.formatRelativeTime(iso);
    if (!iso) return '';
    const t = new Date(iso).getTime();
    if (!Number.isFinite(t)) return '';
    const mins = Math.round((Date.now() - t) / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins} min ago`;
    const hours = Math.round(mins / 60);
    if (hours < 48) return `${hours}h ago`;
    return `${Math.round(hours / 24)}d ago`;
  }

  function formatShortDateRange(start, end) {
    if (typeof window.formatShortDateRange === 'function') {
      return window.formatShortDateRange(start, end);
    }
    const parts = [start, end].filter(Boolean);
    return parts.length ? parts.join(' – ') : '—';
  }

  function marketLabelForAgent(agent) {
    const labels = shelfLabels();
    const slug = String(agent?.category || '').trim().toLowerCase();
    if (labels[slug]) return labels[slug];
    if (agent?.agent_type === 'builtin') return 'Not set';
    return 'External';
  }

  function modelLabelForAgent(agent) {
    if (isAiHedgeFundAgent(agent)) {
      return document.getElementById('agentEditorManagedModelValue')?.textContent?.trim()
        || 'Hosted Nemotron';
    }
    const select = document.getElementById('agentEditorModelSelect');
    if (select?.selectedOptions?.[0]?.textContent) return select.selectedOptions[0].textContent.trim();
    return agent?.model_name || '—';
  }

  function resolveStatusBadge(agent) {
    if (typeof window.resolveAgentStatusBadge === 'function') {
      return window.resolveAgentStatusBadge(agent);
    }
    const runCount = Number(agent?.run_count) || (Array.isArray(agent?.runs) ? agent.runs.length : 0);
    if (runCount > 0 || agent?.latest_run?.run_id) {
      return { key: 'backtested', label: 'Backtested', className: 'idle' };
    }
    return { key: 'draft', label: 'Draft', className: 'draft' };
  }

  function setActiveTab(tab) {
    const allowed = new Set(['overview', 'settings', 'backtest', 'paper', 'live']);
    activeCpTab = allowed.has(tab) ? tab : 'overview';

    document.querySelectorAll('[data-agent-cp-tab]').forEach((btn) => {
      const on = btn.getAttribute('data-agent-cp-tab') === activeCpTab;
      btn.classList.toggle('is-active', on);
      btn.setAttribute('aria-selected', on ? 'true' : 'false');
    });

    document.querySelectorAll('[data-agent-cp-panel]').forEach((panel) => {
      const on = panel.getAttribute('data-agent-cp-panel') === activeCpTab;
      panel.hidden = !on;
      panel.classList.toggle('is-active', on);
    });
  }

  function syncHeroDescription() {
    const hero = document.getElementById('agentEditorHeroDescription');
    if (!hero) return;
    const text = String(currentAgent?.description || '').trim();
    hero.textContent = text
      || 'Researches market data and identifies trading opportunities using a configurable LLM strategy.';
  }

  function updateStatusChrome(agent = currentAgent) {
    if (!agent) return;
    const badge = document.getElementById('agentEditorStatusBadge');
    const status = resolveStatusBadge(agent);
    if (badge) {
      const raw = status.label || (status.key === 'draft' ? 'Draft' : 'Draft');
      badge.textContent = raw
        .toLowerCase()
        .replace(/\b\w/g, (ch) => ch.toUpperCase())
        .replace('Paper Trading', 'Paper');
      badge.className = 'agent-cp-status-badge';
      if (status.className === 'idle') badge.classList.add('is-idle');
      if (status.className === 'paper' || status.key === 'paper') badge.classList.add('is-paper');
    }

    const updated = agent.updated_at || agent.created_at;
    const updatedLabel = document.getElementById('agentEditorUpdatedLabel');
    if (updatedLabel) {
      updatedLabel.textContent = updated
        ? `Updated ${new Date(updated).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`
        : 'Updated —';
    }

    const version = agent.version || agent.current_version || 'v1';
    const versionLabel = String(version).startsWith('v') ? String(version) : `v${version}`;
    const versionEls = [
      document.getElementById('agentEditorVersionLabel'),
      document.getElementById('agentEditorOverviewVersionBadge'),
      document.getElementById('agentBacktestConfigVersion'),
    ];
    versionEls.forEach((el) => {
      if (!el) return;
      el.textContent = el.id === 'agentEditorOverviewVersionBadge'
        ? `Agent ${versionLabel}`
        : versionLabel;
    });
  }

  function renderOverview(runs = cachedRuns) {
    const agent = currentAgent;
    if (!agent) return;

    const configPrimary = document.getElementById('agentOverviewConfigPrimary');
    const configSecondary = document.getElementById('agentOverviewConfigSecondary');
    if (configPrimary) configPrimary.textContent = modelLabelForAgent(agent);
    if (configSecondary) {
      configSecondary.textContent = `${marketLabelForAgent(agent)} · allocated ${formatUsd(
        agent.backtest_allocation ?? agent.cash_allocation ?? 1000,
      )}`;
    }

    const sorted = [...(runs || [])].sort(
      (a, b) => (b.created_at || '').localeCompare(a.created_at || ''),
    );
    const latest = sorted[0] || agent.latest_run || null;
    const btPrimary = document.getElementById('agentOverviewBacktestPrimary');
    const btSecondary = document.getElementById('agentOverviewBacktestSecondary');
    if (btPrimary && btSecondary) {
      if (latest && latest.total_return != null) {
        btPrimary.textContent = formatReturnPct(latest);
        btPrimary.className = `agent-cp-status-primary ${returnToneClass(latest)}`;
        const sharpe = latest.sharpe_ratio != null && Number.isFinite(Number(latest.sharpe_ratio))
          ? ` · Sharpe ${Number(latest.sharpe_ratio).toFixed(2)}`
          : '';
        btSecondary.textContent = `${formatShortDateRange(latest.start_date, latest.end_date)}${sharpe}`;
      } else {
        btPrimary.textContent = sorted.length ? 'Completed' : 'No runs yet';
        btPrimary.className = 'agent-cp-status-primary';
        btSecondary.textContent = sorted.length
          ? formatShortDateRange(latest?.start_date, latest?.end_date)
          : 'Run a backtest to see results';
      }
    }

    const capitalEl = document.getElementById('agentBacktestConfigCapital');
    if (capitalEl) {
      capitalEl.textContent = formatUsd(agent.backtest_allocation ?? agent.cash_allocation ?? 1000);
    }
    const marketEl = document.getElementById('agentBacktestConfigMarket');
    if (marketEl) marketEl.textContent = marketLabelForAgent(agent);

    const activity = document.getElementById('agentEditorActivityList');
    if (!activity) return;
    const items = [];
    sorted.slice(0, 5).forEach((run) => {
      items.push({
        title: 'Backtest completed',
        sub: `${formatShortDateRange(run.start_date, run.end_date)} · ${formatReturnPct(run)}`,
        time: formatRelativeTime(run.created_at) || formatRunSecondary(run),
        icon: 'check',
      });
    });
    if (agent.updated_at && agent.updated_at !== agent.created_at) {
      items.push({
        title: 'Agent settings saved',
        sub: 'Configuration updated',
        time: formatRelativeTime(agent.updated_at),
        icon: 'save',
      });
    }
    items.push({
      title: 'Agent created',
      sub: `${agent.name || 'Agent'} ${agent.version || 'v1'}`,
      time: formatRelativeTime(agent.created_at) || (agent.created_at
        ? new Date(agent.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        : '—'),
      icon: 'plus',
    });

    const iconSvg = {
      check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg>',
      save: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8"/><path d="M7 3v5h8"/></svg>',
      plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14"/><path d="M5 12h14"/></svg>',
    };

    activity.innerHTML = items.slice(0, 6).map((item) => `
      <div class="agent-cp-activity-item">
        <span class="agent-cp-activity-icon" aria-hidden="true">${iconSvg[item.icon] || iconSvg.plus}</span>
        <div>
          <p class="agent-cp-activity-title-text">${escapeHtml(item.title)}</p>
          <p class="agent-cp-activity-sub">${escapeHtml(item.sub)}</p>
        </div>
        <span class="agent-cp-activity-time">${escapeHtml(item.time || '')}</span>
      </div>`).join('');
  }

  function setBrokerMessage(message, isError) {
    const resultEl = document.getElementById('agentEditorLiveRunResult');
    if (!resultEl) return;
    resultEl.hidden = !message;
    resultEl.textContent = message || '';
    resultEl.classList.toggle('agent-editor-live-result--error', !!isError);
  }

  function fetchWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    return fetch(url, { credentials: 'include', ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
  }

  /**
   * Run Live is the loudest button on the screen, so it must not look ready
   * while the broker it needs is unconnected — demote it until then.
   */
  function setRunLiveProminence(connected) {
    const runLiveBtn = document.getElementById('agentEditorRunLiveBtn');
    if (!runLiveBtn) return;
    runLiveBtn.disabled = !connected;
    runLiveBtn.className = connected
      ? 'home-btn home-btn-primary'
      : 'home-btn home-btn-secondary';
    runLiveBtn.title = connected ? '' : 'Connect Robinhood first';
  }

  async function refreshRobinhoodStatus() {
    const statusEl = document.getElementById('agentEditorRobinhoodStatus');
    const metaEl = document.getElementById('agentEditorRobinhoodMeta');
    const connectBtn = document.getElementById('agentEditorConnectRobinhoodBtn');
    const disconnectBtn = document.getElementById('agentEditorDisconnectRobinhoodBtn');
    const liveToggle = document.getElementById('agentEditorLiveTradingEnabled');
    if (!isEditorSignedIn()) {
      if (statusEl) {
        statusEl.textContent = 'Sign in required';
        statusEl.className = 'agent-editor-broker-status agent-editor-broker-status--warn';
      }
      if (connectBtn) {
        connectBtn.hidden = false;
        connectBtn.disabled = false;
        connectBtn.textContent = 'Sign in to connect';
      }
      if (disconnectBtn) disconnectBtn.hidden = true;
      if (metaEl) metaEl.hidden = true;
      setRunLiveProminence(false);
      setBrokerMessage('Log in to your ATL account, then click Connect Robinhood.', true);
      return;
    }

    if (connectBtn) {
      connectBtn.hidden = false;
      connectBtn.disabled = false;
      connectBtn.textContent = 'Connect Robinhood';
    }

    if (statusEl) {
      statusEl.textContent = 'Checking…';
      statusEl.className = 'agent-editor-broker-status';
    }

    try {
      const response = await fetchWithTimeout(
        `${API_BASE}/api/v1/robinhood/status`,
        { headers: authHeaders() },
        8000,
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Status check failed');

      const connected = Boolean(data.connected);
      if (statusEl) {
        statusEl.textContent = connected ? 'Connected' : 'Not connected';
        statusEl.className = connected
          ? 'agent-editor-broker-status agent-editor-broker-status--ok'
          : 'agent-editor-broker-status agent-editor-broker-status--warn';
      }
      if (connectBtn) {
        connectBtn.hidden = connected;
        connectBtn.disabled = false;
        connectBtn.textContent = 'Connect Robinhood';
      }
      if (disconnectBtn) disconnectBtn.hidden = !connected;
      setRunLiveProminence(connected);
      if (metaEl) {
        if (connected) {
          metaEl.hidden = false;
          const parts = [];
          if (data.buying_power != null) parts.push(`Buying power: ${formatUsd(data.buying_power)}`);
          if (data.portfolio_value != null) parts.push(`Portfolio: ${formatUsd(data.portfolio_value)}`);
          parts.push(`Execute switch: ${data.execute_enabled ? 'ON' : 'OFF (review only)'}`);
          metaEl.textContent = parts.join(' · ') || 'Connected — click Connect again to refresh account details.';
        } else {
          metaEl.hidden = true;
        }
      }
      const livePrimary = document.getElementById('agentOverviewLivePrimary');
      if (livePrimary) livePrimary.textContent = connected ? 'Connected' : 'Not connected';
      if (!connected) {
        setBrokerMessage('Not connected yet. Click Connect Robinhood to authorize.');
      } else {
        setBrokerMessage('');
      }
    } catch (error) {
      const timedOut = error?.name === 'AbortError';
      if (statusEl) {
        statusEl.textContent = timedOut ? 'Not connected' : 'Unavailable';
        statusEl.className = 'agent-editor-broker-status agent-editor-broker-status--warn';
      }
      if (connectBtn) {
        connectBtn.hidden = false;
        connectBtn.disabled = false;
        connectBtn.textContent = 'Connect Robinhood';
      }
      if (disconnectBtn) disconnectBtn.hidden = true;
      setRunLiveProminence(false);
      setBrokerMessage(
        timedOut
          ? 'Status check timed out. You can still click Connect Robinhood.'
          : (error.message || 'Could not check Robinhood status.'),
        true,
      );
      console.warn('Robinhood status failed:', error);
    }

    if (liveToggle && currentAgent) {
      liveToggle.checked = Boolean(currentAgent.live_trading_enabled);
    }
  }

  async function connectRobinhood() {
    if (!isEditorSignedIn()) {
      setBrokerMessage('Please sign in first.', true);
      if (typeof window.openAuthModal === 'function') window.openAuthModal('login');
      else alert('Please sign in to your ATL account first.');
      return;
    }
    if (!currentAgent?.agent_id) {
      setBrokerMessage('Open a saved agent before connecting Robinhood.', true);
      return;
    }
    if (isDemoAgent(currentAgent.agent_id)) {
      setBrokerMessage('Create a real agent in My Agents first (demo agents cannot connect).', true);
      return;
    }

    const connectBtn = document.getElementById('agentEditorConnectRobinhoodBtn');
    if (connectBtn) {
      connectBtn.disabled = true;
      connectBtn.textContent = 'Connecting…';
    }
    setBrokerMessage('Starting Robinhood authorization…');

    try {
      const response = await fetch(`${API_BASE}/api/auth/robinhood/start`, {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ agent_id: currentAgent.agent_id }),
      });
      let data = {};
      try {
        data = await response.json();
      } catch (_) {
        data = {};
      }
      if (!response.ok) {
        const detail = data.detail;
        const msg = typeof detail === 'string' ? detail : 'Could not start Robinhood OAuth';
        throw new Error(msg);
      }
      if (data.already_linked) {
        setBrokerMessage('Robinhood is already connected for your account.');
        await refreshRobinhoodStatus();
        return;
      }
      if (data.authorize_url) {
        setBrokerMessage('Redirecting to Robinhood…');
        window.location.href = data.authorize_url;
        return;
      }
      throw new Error('No authorize URL returned from server');
    } catch (error) {
      const msg = error.message || 'Robinhood connect failed';
      setBrokerMessage(msg, true);
      showSaveStatus(msg, true);
      if (/failed to fetch|networkerror/i.test(msg)) {
        alert('Cannot reach the backend. Start it with: uvicorn dashboard.backend.app:app --reload');
      }
    } finally {
      if (connectBtn) {
        connectBtn.disabled = false;
        connectBtn.textContent = 'Connect Robinhood';
      }
    }
  }

  async function disconnectRobinhood() {
    if (!window.confirm('Disconnect Robinhood from your ATL account?')) return;
    try {
      const response = await fetch(`${API_BASE}/api/v1/robinhood/disconnect`, {
        method: 'DELETE',
        headers: authHeaders(),
        credentials: 'include',
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Disconnect failed');
      }
      await refreshRobinhoodStatus();
      showSaveStatus('Robinhood disconnected');
    } catch (error) {
      showSaveStatus(error.message || 'Disconnect failed', true);
    }
  }

  async function runLive() {
    if (!currentAgent?.agent_id || isDemoAgent(currentAgent.agent_id)) {
      showSaveStatus('Save a real agent before running live', true);
      return;
    }
    const liveToggle = document.getElementById('agentEditorLiveTradingEnabled');
    if (!liveToggle?.checked) {
      showSaveStatus('Enable live trading for this agent first', true);
      return;
    }

    if (isDirty) {
      showSaveStatus('Save changes before Run Live', true);
      return;
    }

    const confirmMsg =
      'Run this agent against your Robinhood Agentic account? Real orders may be placed if the server execute switch is ON.';
    if (!window.confirm(confirmMsg)) return;

    const resultEl = document.getElementById('agentEditorLiveRunResult');
    const runBtn = document.getElementById('agentEditorRunLiveBtn');
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.textContent = 'Running…';
    }
    if (resultEl) {
      resultEl.hidden = false;
      resultEl.textContent = 'Running live cycle…';
    }

    try {
      const response = await fetch(
        `${API_BASE}/api/v1/robinhood/agents/${encodeURIComponent(currentAgent.agent_id)}/live-run`,
        {
          method: 'POST',
          headers: { ...authHeaders(), 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ dry_run: false }),
        }
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Live run failed');
      const execCount = Array.isArray(data.executions) ? data.executions.length : 0;
      const submitted = (data.executions || []).filter((e) => e.status === 'submitted').length;
      if (resultEl) {
        resultEl.textContent = data.dry_run
          ? `Dry run ${data.run_id}: reviewed ${execCount} order(s), none submitted (ROBINHOOD_EXECUTE is off).`
          : `Live run ${data.run_id}: ${submitted} order(s) submitted, ${execCount} total reviewed.`;
      }
      showSaveStatus(data.dry_run ? 'Live review completed (execute off)' : 'Live run completed');
    } catch (error) {
      if (resultEl) {
        resultEl.hidden = false;
        resultEl.textContent = error.message || 'Live run failed';
      }
      showSaveStatus(error.message || 'Live run failed', true);
    } finally {
      if (runBtn) {
        runBtn.disabled = false;
        runBtn.textContent = 'Run Live';
      }
    }
  }

  // My Agents' shelf slugs. Read from app.js at call time rather than copied:
  // app.js derives SHELF_LABELS from AGENT_SHELVES, so a shelf renamed or added
  // there reaches this <select> without a second edit. The literal is only a
  // floor for the case where app.js failed to load -- agent-editor.js is loaded
  // first, so this must never be read at module-init time.
  const SHELF_LABELS_FALLBACK = {
    us_stocks: 'US Stocks',
    cn_ashares: 'China A-Share',
  };

  function shelfLabels() {
    const exported = window.AGENT_SHELF_LABELS;
    return exported && Object.keys(exported).length ? exported : SHELF_LABELS_FALLBACK;
  }

  /** Only built-in agents are shelved by category; connected agents always
   * render under "For Developers", so the picker is meaningless for them.
   * Demo agents are excluded too -- they have no backend row, so the save path
   * skips the PATCH entirely and the choice would silently not stick. */
  function categoryFieldApplies(agent) {
    const target = agent || currentAgent;
    if (!target || target.agent_type !== 'builtin') return false;
    return !isDemoAgent(target.agent_id);
  }

  function fillCategorySelect(agent) {
    const select = document.getElementById('agentEditorCategorySelect');
    if (!select) return;
    const labels = shelfLabels();
    // "" remains a saveable choice (backend folds it to NULL / un-shelves).
    // Uncategorized agents show US Stocks as the suggested default, but
    // categoryUiDefaulted keeps that from being PATCH'd until the user
    // touches Market (see getEditorState).
    const options = [['', 'Not set']].concat(
      Object.keys(labels).map((slug) => [slug, labels[slug]]),
    );
    select.innerHTML = '';
    options.forEach(([value, label]) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    });
    const current = String(agent?.category || '').trim().toLowerCase();
    if (labels[current]) {
      select.value = current;
      categoryUiDefaulted = false;
    } else {
      select.value = labels.us_stocks ? 'us_stocks' : '';
      categoryUiDefaulted = Boolean(labels.us_stocks);
    }
  }

  function getEditorState() {
    const hostedAiHedgeFund = isAiHedgeFundAgent();
    const categorySelect = document.getElementById('agentEditorCategorySelect');
    // null means "omit the key", which the PATCH route reads as "leave alone".
    // "" means "clear the shelf" and must still be sent.
    // categoryUiDefaulted → null so a capital/name save does not invent a market.
    let category = null;
    if (categoryFieldApplies() && categorySelect) {
      category = categoryUiDefaulted ? null : String(categorySelect.value || '');
    }
    const nameInput = document.getElementById('agentEditorNameInput');
    const cashInput = document.getElementById('agentEditorCashAllocation');
    // Description was removed from Configure — it is not shown on agent cards
    // and was only acting as a private note. Preserve whatever is already stored.
    let cash_allocation = null;
    if (cashInput && cashInput.value !== '') {
      const value = Number(cashInput.value);
      if (!Number.isFinite(value) || value < 0) {
        throw new Error('Paper Trading Allocated Capital must be zero or greater.');
      }
      if (value > 3000) {
        throw new Error('Paper Trading Allocated Capital cannot exceed $3,000.');
      }
      cash_allocation = Math.round(value);
    } else {
      cash_allocation = 1000;
    }
    const backtestInput = document.getElementById('agentEditorBacktestAllocation');
    let backtest_allocation = null;
    if (backtestInput && backtestInput.value !== '') {
      const value = Number(backtestInput.value);
      if (!Number.isFinite(value) || value < 1) {
        throw new Error('Backtest Allocated Capital must be at least $1.');
      }
      if (value > 3000) {
        throw new Error('Backtest Allocated Capital cannot exceed $3,000.');
      }
      backtest_allocation = Math.round(value);
    } else {
      // Non-positive counts as absent: cash_allocation is legally 0 (a $0 paper
      // sleeve), but backtest capital is >= 1 server-side, so 0 must fall
      // through to the default rather than becoming an unsaveable value.
      backtest_allocation =
        Number.isFinite(Number(cash_allocation)) && Number(cash_allocation) > 0
          ? Math.min(Math.round(Number(cash_allocation)), 3000)
          : 1000;
    }
    const modelSelect = document.getElementById('agentEditorModelSelect');
    const instruction = hostedAiHedgeFund ? '' : (
      document.getElementById('agentEditorSimpleInstruction')?.value || ''
    ).trim();
    let subAgentsOut;
    let sendPipeline = !hostedAiHedgeFund;
    if (hostedAiHedgeFund) {
      // Hosted runtimes do not consume or mutate the legacy prompt pipeline.
      subAgentsOut = subAgents;
    } else if (instruction) {
      const existing =
        subAgents.length === 1 && subAgents[0].presetKey === simplePresetKey()
          ? subAgents[0]
          : null;
      subAgentsOut = [
        {
          id: existing ? existing.id : newSubAgentId(),
          presetKey: simplePresetKey(),
          label: 'Trading instruction',
          prompt: instruction,
          outputFormat: simpleOutputFormat(),
        },
      ];
      sendPipeline = true;
    } else {
      // Empty means "use the platform default": clear the pipeline so the
      // backend takes its create_prompt branch. The multi-step pipeline this
      // screen cannot author is protected by a confirm in save(), not by
      // silently refusing to send -- which used to make an empty save a no-op
      // that still reported success.
      subAgentsOut = [];
      sendPipeline = true;
    }
    const liveToggle = document.getElementById('agentEditorLiveTradingEnabled');
    const credentialInput = document.getElementById('agentEditorFinancialDatasetsKey');
    return {
      name: nameInput ? nameInput.value.trim() : '',
      description: String(currentAgent?.description || '').trim(),
      category,
      cash_allocation,
      backtest_allocation,
      model_name: hostedAiHedgeFund
        ? ''
        : (modelSelect ? modelSelect.value : ''),
      live_trading_enabled: Boolean(liveToggle?.checked),
      runtime_config: hostedAiHedgeFund
        ? { analysts: selectedAiHedgeFundAnalysts() }
        : null,
      financial_datasets_api_key: hostedAiHedgeFund
        ? (credentialInput?.value || '').trim()
        : '',
      subAgents: subAgentsOut,
      sendPipeline,
    };
  }

  function snapshotState() {
    const state = getEditorState();
    // Never copy credential plaintext into snapshots or browser storage. Its
    // presence is enough to make the editor dirty until a successful save.
    state.financial_datasets_api_key = Boolean(state.financial_datasets_api_key);
    return JSON.stringify(state);
  }

  function setDirty(dirty) {
    isDirty = dirty;
    const badge = document.getElementById('agentEditorDirtyBadge');
    if (badge) badge.hidden = !dirty;
  }

  function markDirtyFromInput() {
    setDirty(snapshotState() !== savedSnapshot);
  }

  function captureSavedSnapshot() {
    savedSnapshot = snapshotState();
    setDirty(false);
  }

  function showSaveStatus(message, isError) {
    // Toast as well as write the inline note. #agentEditorSaveStatus lives at
    // the bottom of the editor's left column, but every one of this function's
    // callers is a click on a control in the sticky header (Save, Run Backtest,
    // Run Live, Connect/Disconnect Robinhood). Measured live at 1440x900: the
    // header button sits at y=14 and this element renders at y=934 -- 920px
    // below the fold in a 900px viewport, and it carries no aria-live. So
    // "Save changes before Run Backtest" and "Agent name is required" both
    // landed somewhere nobody was looking, and the button read as dead.
    // #appToast is role="status" aria-live="polite", so this reaches sighted
    // and screen-reader users both. Done here rather than at the 16 call sites
    // so no future message can regress to inline-only.
    if (typeof window.showAppToast === 'function') {
      window.showAppToast(message);
    }
    const el = document.getElementById('agentEditorSaveStatus');
    if (!el) return;
    el.hidden = false;
    el.textContent = message;
    el.classList.toggle('agent-editor-save-status--error', !!isError);
    clearTimeout(saveStatusTimer);
    saveStatusTimer = setTimeout(() => {
      el.hidden = true;
    }, 3000);
  }

  // Warn only when the agent holds a pipeline this screen cannot author, since
  // saving an instruction would replace it.
  function updateSimpleReplaceNote() {
    const note = document.getElementById('agentEditorSimpleReplaceNote');
    if (note) note.hidden = isSimplePipeline(subAgents);
  }

  function fillHeader(agent) {
    const nameInput = document.getElementById('agentEditorNameInput');
    const cashInput = document.getElementById('agentEditorCashAllocation');
    const meta = document.getElementById('agentEditorMeta');

    if (nameInput) nameInput.value = agent.name || '';
    if (cashInput) {
      cashInput.value = agent.cash_allocation != null ? String(agent.cash_allocation) : '';
    }
    const backtestInput = document.getElementById('agentEditorBacktestAllocation');
    if (backtestInput) {
      // Non-positive counts as absent: cash_allocation is legally 0 (a $0 paper
      // sleeve), but backtest capital is >= 1 server-side, so 0 must fall
      // through to the default rather than becoming an unsaveable value.
      const candidates = [agent.backtest_allocation, agent.cash_allocation];
      let resolved = 1000;
      for (const raw of candidates) {
        const value = Number(raw);
        if (Number.isFinite(value) && value > 0) { resolved = value; break; }
      }
      backtestInput.value = String(Math.min(Math.round(resolved), 3000));
    }
    if (meta) {
      meta.textContent = agent.agent_type === 'builtin' ? 'Built-in agent' : 'External agent';
      meta.hidden = true;
    }
    const categoryField = document.getElementById('agentEditorCategoryField');
    if (categoryField) categoryField.hidden = !categoryFieldApplies(agent);
    fillCategorySelect(agent);
    const liveToggle = document.getElementById('agentEditorLiveTradingEnabled');
    if (liveToggle) liveToggle.checked = Boolean(agent.live_trading_enabled);
    syncHeroDescription();
    updateStatusChrome(agent);
  }

  function configureEditorMode(agent) {
    const hostedAiHedgeFund = isAiHedgeFundAgent(agent);
    const modelField = document.getElementById('agentEditorModelField');
    const managedModelField = document.getElementById('agentEditorManagedModelField');
    const simplePanel = document.getElementById('agentEditorSimplePanel');
    const hedgeFundPanel = document.getElementById('agentEditorAiHedgeFundPanel');
    if (modelField) modelField.hidden = hostedAiHedgeFund;
    if (managedModelField) managedModelField.hidden = !hostedAiHedgeFund;
    if (simplePanel) simplePanel.hidden = hostedAiHedgeFund;
    if (hedgeFundPanel) hedgeFundPanel.hidden = !hostedAiHedgeFund;
    if (hostedAiHedgeFund) {
      renderAiHedgeFundAnalysts(agent);
      const keyInput = document.getElementById('agentEditorFinancialDatasetsKey');
      if (keyInput) keyInput.value = '';
      setFinancialDatasetsStatus(false, 'Checking credential…');
    }
  }

  function populateModelSelect(agent) {
    const select = document.getElementById('agentEditorModelSelect');
    if (!select) return;
    select.innerHTML = '';
    const seen = new Set();
    const source = document.getElementById('builtinAgentModel');
    if (source) {
      Array.from(source.options).forEach((opt) => {
        const clone = document.createElement('option');
        clone.value = opt.value;
        clone.textContent = opt.textContent;
        select.appendChild(clone);
        seen.add(opt.value);
      });
    }
    const current = agent.model_name || 'local-model';
    if (!seen.has(current)) {
      // External / legacy models aren't in the curated list — keep them selectable.
      const opt = document.createElement('option');
      opt.value = current;
      opt.textContent = current;
      select.insertBefore(opt, select.firstChild);
    }
    select.value = current;
  }

  function serializePipeline(steps) {
    return steps.map((sub) => ({
      id: sub.id,
      presetKey: sub.presetKey,
      label: sub.label,
      prompt: sub.prompt,
      outputFormat: sub.outputFormat,
    }));
  }

  async function patchAgent(
    agent,
    name,
    description,
    pipeline,
    cash_allocation,
    backtest_allocation,
    model_name,
    live_trading_enabled,
    runtimeConfig,
    category,
  ) {
    const payload = {
      name,
      description: description || null,
      cash_allocation,
      backtest_allocation,
      live_trading_enabled: Boolean(live_trading_enabled),
    };
    // Presence of the key is the signal, so the falsy-but-meaningful "" (clear
    // the shelf) must still be sent; only null/undefined means "leave alone".
    if (category !== null && category !== undefined) payload.category = category;
    if (pipeline) payload.pipeline = serializePipeline(pipeline);
    if (model_name) payload.model_name = model_name;
    if (runtimeConfig) payload.runtime_config = runtimeConfig;
    const endpoint = `${API_BASE}/api/v1/agents/${encodeURIComponent(agent.agent_id)}`;

    async function requestWithHeaders(extraHeaders) {
      if (window.API?.patch) {
        return window.API.patch(endpoint, payload, {
          ...authHeaders(),
          ...extraHeaders,
        });
      }
      // window.API is a const in app.js and is not on `window` unless exported —
      // this fallback is the live prod path. It must send X-CSRF-Token whenever
      // the session cookie is present (#285), same as Robinhood / credential calls.
      const headers = {
        'Content-Type': 'application/json',
        ...authHeaders(),
        ...extraHeaders,
      };
      const response = await fetch(endpoint, {
        method: 'PATCH',
        headers,
        credentials: 'include',
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail || data.message || `HTTP ${response.status}`;
        const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
        error.status = response.status;
        throw error;
      }
      return data;
    }

    try {
      const data = await requestWithHeaders({
        'x-browser-id': window.BROWSER_OWNER_ID,
        'x-session-id': agent.session_id || window.SESSION_ID,
      });
      return data.agent;
    } catch (error) {
      // Legacy/imported agents may store owner_browser_session = session_id.
      // Retry with session-only ownership (omit X-Browser-Id) when denied.
      if (error.status !== 403 || !agent?.session_id) throw error;
      const data = await requestWithHeaders({
        'x-session-id': agent.session_id,
        'x-browser-id': '',
      });
      return data.agent;
    }
  }

  function formatRunPrimary(run) {
    if (typeof window.formatBacktestRunPrimary === 'function') {
      return window.formatBacktestRunPrimary(run);
    }
    const dates = [run.start_date, run.end_date].filter(Boolean).join(' → ');
    return dates || run.run_id || 'Backtest run';
  }

  function formatRunSecondary(run) {
    if (typeof window.formatBacktestRunSecondary === 'function') {
      return window.formatBacktestRunSecondary(run);
    }
    return run.created_at ? new Date(run.created_at).toLocaleString() : '';
  }

  function formatRunMeta(run) {
    const parts = [];
    if (run.llm_model) parts.push(run.llm_model);
    const tokens = Number(run.input_tokens || 0) + Number(run.output_tokens || 0);
    if (tokens > 0) parts.push(`${tokens.toLocaleString()} tokens`);
    if (run.num_trades != null) parts.push(`${run.num_trades} trades`);
    return parts.join(' · ');
  }

  function renderRunHistory(runs) {
    const container = document.getElementById('agentEditorRunHistory');
    const countEl = document.getElementById('agentEditorRunCount');
    const navCountEl = document.getElementById('agentEditorNavRunCount');
    if (!container) return;

    const sorted = [...(runs || [])].sort(
      (a, b) => (b.created_at || '').localeCompare(a.created_at || ''),
    );
    cachedRuns = sorted;

    if (countEl) {
      countEl.textContent = `${sorted.length} run${sorted.length === 1 ? '' : 's'}`;
    }
    if (navCountEl) navCountEl.textContent = String(sorted.length);

    if (!sorted.length) {
      container.innerHTML = '<p class="agent-editor-run-empty">No backtest runs yet. Run a backtest from this agent to see history here.</p>';
      renderOverview([]);
      return;
    }

    container.innerHTML = `
      <table class="agent-cp-run-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Period</th>
            <th>Initial capital</th>
            <th>Return</th>
            <th>Sharpe</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${sorted.map((run) => {
            const ret = formatReturnPct(run);
            const tone = returnToneClass(run);
            const sharpe = run.sharpe_ratio != null && Number.isFinite(Number(run.sharpe_ratio))
              ? Number(run.sharpe_ratio).toFixed(2)
              : '—';
            const capital = formatUsd(run.initial_equity ?? run.initial_capital ?? currentAgent?.backtest_allocation);
            const created = run.created_at
              ? new Date(run.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
              : '';
            const runLabel = run.strategy_name || run.universe_label || run.agent_name || currentAgent?.name || 'Backtest';
            return `
              <tr data-run-id="${escapeHtml(run.run_id)}" role="listitem">
                <td>
                  <span class="agent-cp-run-name">${escapeHtml(runLabel)}</span>
                  <span class="agent-cp-run-date">${escapeHtml(created)}</span>
                </td>
                <td>${escapeHtml(formatShortDateRange(run.start_date, run.end_date))}</td>
                <td>${escapeHtml(capital)}</td>
                <td class="agent-cp-run-return ${tone}">${escapeHtml(ret)}</td>
                <td>${escapeHtml(sharpe)}</td>
                <td><span class="agent-cp-run-status">Complete</span></td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>`;

    container.querySelectorAll('tr[data-run-id]').forEach((row) => {
      row.addEventListener('click', () => {
        const runId = row.dataset.runId;
        if (!currentAgent || !runId) return;
        window.dispatchEvent(
          new CustomEvent('agent-editor-open-run', {
            detail: { agent: currentAgent, runId },
          }),
        );
      });
    });

    renderOverview(sorted);
  }

  async function refreshRunHistory(agent) {
    if (!agent?.agent_id) {
      renderRunHistory([]);
      return;
    }

    if (isDemoAgent(agent.agent_id)) {
      renderRunHistory(agent.runs || []);
      return;
    }

    renderRunHistory(agent.runs || []);

    try {
      const headers = { 'x-session-id': window.SESSION_ID };

      const response = await fetch(
        `${API_BASE}/api/v1/agents/${encodeURIComponent(agent.agent_id)}`,
        { headers, credentials: 'include' },
      );
      if (!response.ok) return;
      const data = await response.json();
      const fresh = data.agent;
      if (!fresh) return;
      currentAgent = { ...currentAgent, ...fresh };
      renderRunHistory(fresh.runs || []);
    } catch (error) {
      console.warn('Could not refresh backtest history:', error);
    }
  }

  function open(agent) {
    if (!agent || !agent.agent_id) return;

    currentAgent = { ...agent };
    subAgents = loadStoredPipeline(agent);
    fillHeader(agent);
    populateModelSelect(agent);
    configureEditorMode(agent);
    setActiveTab('overview');

    const instructionEl = document.getElementById('agentEditorSimpleInstruction');
    const defaultText = document.getElementById('agentEditorDefaultInstructionText');
    if (defaultText) defaultText.textContent = defaultStarterInstruction();
    const simpleStep =
      subAgents.length === 1 && subAgents[0].presetKey === simplePresetKey()
        ? subAgents[0]
        : null;
    if (instructionEl) instructionEl.value = simpleStep ? simpleStep.prompt : '';
    updateSimpleReplaceNote();
    refreshRunHistory(currentAgent);
    refreshRobinhoodStatus();
    if (isAiHedgeFundAgent(agent)) {
      refreshFinancialDatasetsStatus(agent);
    }
    renderOverview(currentAgent.runs || []);

    const view = document.getElementById('agentEditorView');
    if (view) {
      view.hidden = false;
      document.body.classList.add('agent-editor-open');
    }

    const playgroundView = document.getElementById('playgroundView');
    if (playgroundView) playgroundView.setAttribute('aria-hidden', 'true');

    // Baseline the stored state so the dirty badge only fires on real edits.
    captureSavedSnapshot();

    document.getElementById('agentEditorNameInput')?.focus();
  }

  function close(force) {
    if (!force && isDirty) {
      if (!window.confirm('Discard unsaved changes?')) return;
    }

    hideAiHedgeFundTooltip(activeAiHedgeFundTooltipOption);
    const view = document.getElementById('agentEditorView');
    if (view) view.hidden = true;
    document.body.classList.remove('agent-editor-open');

    const playgroundView = document.getElementById('playgroundView');
    if (playgroundView) playgroundView.removeAttribute('aria-hidden');

    currentAgent = null;
    setDirty(false);
  }

  async function save() {
    if (!currentAgent) return;

    let state;
    try {
      state = getEditorState();
    } catch (error) {
      showSaveStatus(error.message, true);
      document.getElementById('agentEditorCashAllocation')?.focus();
      return;
    }
    if (!state.name) {
      showSaveStatus('Agent name is required', true);
      document.getElementById('agentEditorNameInput')?.focus();
      return;
    }
    if (isAiHedgeFundAgent() && !state.runtime_config.analysts.length) {
      showSaveStatus('Select at least one AI Hedge Fund analyst', true);
      return;
    }

    const clearingToDefault = state.sendPipeline && state.subAgents.length === 0;
    if (clearingToDefault && !isSimplePipeline(subAgents) && subAgents.length) {
      const ok = window.confirm(
        'This agent uses a custom multi-step pipeline. Saving an empty '
        + 'instruction replaces it with the platform default. Continue?',
      );
      if (!ok) return;
    }

    subAgents = state.subAgents;
    updateSimpleReplaceNote();
    const saveBtn = document.getElementById('agentEditorSaveBtn');
    if (saveBtn) {
      saveBtn.disabled = true;
      saveBtn.textContent = 'Saving…';
    }

    // Demo agents have no backend row: persist name/description locally and skip
    // the PATCH (which would 404) so the rename still sticks in the UI.
    if (isDemoAgent(currentAgent.agent_id)) {
      try {
        localStorage.setItem(
          `${NAME_OVERRIDE_PREFIX}${currentAgent.agent_id}`,
          JSON.stringify({ name: state.name, description: state.description })
        );
        if (state.cash_allocation != null) {
          localStorage.setItem(`${CASH_OVERRIDE_PREFIX}${currentAgent.agent_id}`, String(state.cash_allocation));
        } else {
          localStorage.removeItem(`${CASH_OVERRIDE_PREFIX}${currentAgent.agent_id}`);
        }
        currentAgent = {
          ...currentAgent,
          name: state.name,
          description: state.description,
          cash_allocation: state.cash_allocation,
        };
        if (state.sendPipeline) savePipelineLocal(currentAgent.agent_id, subAgents);
        if (localStorage.getItem('active-agent-id') === currentAgent.agent_id) {
          localStorage.setItem('active-agent-name', state.name);
        }
        captureSavedSnapshot();
        showSaveStatus('Saved (demo agent — stored locally)');
        window.dispatchEvent(
          new CustomEvent('agent-editor-saved', { detail: { agent: currentAgent } })
        );
      } finally {
        if (saveBtn) {
          saveBtn.disabled = false;
          saveBtn.textContent = 'Save changes';
        }
      }
      return;
    }

    let credentialSavePending = false;
    try {
      const updated = await patchAgent(
        currentAgent,
        state.name,
        state.description,
        state.sendPipeline ? subAgents : null,
        state.cash_allocation,
        state.backtest_allocation,
        state.model_name,
        state.live_trading_enabled,
        state.runtime_config,
        state.category,
      );
      currentAgent = state.sendPipeline
        ? { ...currentAgent, ...updated, pipeline: subAgents }
        : { ...currentAgent, ...updated };
      if (state.financial_datasets_api_key) {
        credentialSavePending = true;
        await credentialRequest(currentAgent, 'PUT', {
          api_key: state.financial_datasets_api_key,
        });
        credentialSavePending = false;
        const keyInput = document.getElementById('agentEditorFinancialDatasetsKey');
        if (keyInput) keyInput.value = '';
        setFinancialDatasetsStatus(true);
        state.financial_datasets_api_key = '';
      }
      if (state.sendPipeline) savePipelineLocal(currentAgent.agent_id, subAgents);
      localStorage.removeItem(`${NAME_OVERRIDE_PREFIX}${currentAgent.agent_id}`);

      if (localStorage.getItem('active-agent-id') === currentAgent.agent_id) {
        localStorage.setItem('active-agent-name', state.name);
      }

      fillHeader(currentAgent);
      renderOverview(cachedRuns);
      captureSavedSnapshot();
      showSaveStatus(
        clearingToDefault
          ? 'Saved — using the default trading instruction.'
          : 'Saved successfully',
      );
      window.dispatchEvent(
        new CustomEvent('agent-editor-saved', { detail: { agent: currentAgent } })
      );
    } catch (error) {
      if (credentialSavePending) {
        fillHeader(currentAgent);
        setDirty(true);
        showSaveStatus(
          `Agent saved, but credential storage failed: ${error.message}`,
          true
        );
        window.dispatchEvent(
          new CustomEvent('agent-editor-saved', { detail: { agent: currentAgent } })
        );
        return;
      }
      if (state.sendPipeline) savePipelineLocal(currentAgent.agent_id, subAgents);
      localStorage.setItem(
        `${NAME_OVERRIDE_PREFIX}${currentAgent.agent_id}`,
        JSON.stringify({ name: state.name, description: state.description })
      );
      currentAgent = { ...currentAgent, name: state.name, description: state.description };
      fillHeader(currentAgent);
      captureSavedSnapshot();
      showSaveStatus(
        `Saved locally; server update failed: ${error.message}`,
        true
      );
      window.dispatchEvent(
        new CustomEvent('agent-editor-saved', { detail: { agent: currentAgent } })
      );
    } finally {
      if (saveBtn) {
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save changes';
      }
    }
  }

  function openRunBacktestFromEditor() {
    if (!currentAgent) return;
    // The modal reads the last-saved agent, so an unsaved edit would run the
    // old instruction while the preview shows the new one. Same guard as Run Live.
    if (isDirty) {
      showSaveStatus('Save changes before Run Backtest', true);
      setActiveTab('settings');
      return;
    }
    if (typeof window.openRunBacktestModal === 'function') {
      window.openRunBacktestModal(currentAgent);
    }
  }

  function bindEvents() {
    document.getElementById('agentEditorBackBtn')?.addEventListener('click', () => close(false));
    document.getElementById('agentEditorSaveBtn')?.addEventListener('click', () => save());
    document.getElementById('agentEditorConnectRobinhoodBtn')?.addEventListener('click', connectRobinhood);
    document.getElementById('agentEditorDisconnectRobinhoodBtn')?.addEventListener('click', disconnectRobinhood);
    document.getElementById('agentEditorFinancialDatasetsRemove')?.addEventListener('click', removeFinancialDatasetsCredential);
    document.getElementById('agentEditorRunLiveBtn')?.addEventListener('click', runLive);
    document.getElementById('agentEditorRunBacktestBtn')?.addEventListener('click', openRunBacktestFromEditor);
    document.getElementById('agentEditorNewBacktestBtn')?.addEventListener('click', openRunBacktestFromEditor);
    document.getElementById('agentEditorTalkBtn')?.addEventListener('click', () => {
      showSaveStatus('Talk to agent is coming soon');
    });

    document.querySelectorAll('[data-agent-cp-tab]').forEach((btn) => {
      btn.addEventListener('click', () => setActiveTab(btn.getAttribute('data-agent-cp-tab')));
    });
    document.querySelectorAll('[data-agent-cp-goto]').forEach((btn) => {
      btn.addEventListener('click', () => setActiveTab(btn.getAttribute('data-agent-cp-goto')));
    });

    const body = document.getElementById('agentEditorView');
    body?.addEventListener('input', (event) => {
      if (event.target?.id === 'agentEditorCategorySelect') {
        categoryUiDefaulted = false;
      }
      markDirtyFromInput();
      if (
        event.target?.id === 'agentEditorModelSelect'
        || event.target?.id === 'agentEditorCategorySelect'
        || event.target?.id === 'agentEditorCashAllocation'
        || event.target?.id === 'agentEditorBacktestAllocation'
      ) {
        renderOverview(cachedRuns);
      }
    });
    body?.addEventListener('change', (event) => {
      if (event.target?.id === 'agentEditorCategorySelect') {
        categoryUiDefaulted = false;
      }
      markDirtyFromInput();
      if (
        event.target?.id === 'agentEditorModelSelect'
        || event.target?.id === 'agentEditorCategorySelect'
      ) {
        renderOverview(cachedRuns);
      }
    });
    // Control Plane scrolls .agent-cp-main; legacy editor used .agent-editor-body.
    document.querySelector('.agent-cp-main, .agent-editor-body')?.addEventListener(
      'scroll',
      scheduleActiveAiHedgeFundTooltipPosition,
      { passive: true },
    );
    window.addEventListener('resize', scheduleActiveAiHedgeFundTooltipPosition);
    window.visualViewport?.addEventListener(
      'resize',
      scheduleActiveAiHedgeFundTooltipPosition,
    );
    window.visualViewport?.addEventListener(
      'scroll',
      scheduleActiveAiHedgeFundTooltipPosition,
    );

    document.addEventListener('keydown', (event) => {
      const view = document.getElementById('agentEditorView');
      if (view?.hidden) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        close(false);
      }
      if ((event.ctrlKey || event.metaKey) && event.key === 's') {
        event.preventDefault();
        save();
      }
    });

    window.addEventListener('beforeunload', (event) => {
      if (isDirty && !document.getElementById('agentEditorView')?.hidden) {
        event.preventDefault();
        event.returnValue = '';
      }
    });
  }

  bindEvents();

  window.AgentEditor = { open, close, save };
})();
