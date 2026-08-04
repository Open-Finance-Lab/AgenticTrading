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
    ['aswath_damodaran', 'Aswath Damodaran'],
    ['ben_graham', 'Ben Graham'],
    ['bill_ackman', 'Bill Ackman'],
    ['cathie_wood', 'Cathie Wood'],
    ['charlie_munger', 'Charlie Munger'],
    ['michael_burry', 'Michael Burry'],
    ['mohnish_pabrai', 'Mohnish Pabrai'],
    ['nassim_taleb', 'Nassim Taleb'],
    ['peter_lynch', 'Peter Lynch'],
    ['phil_fisher', 'Phil Fisher'],
    ['rakesh_jhunjhunwala', 'Rakesh Jhunjhunwala'],
    ['stanley_druckenmiller', 'Stanley Druckenmiller'],
    ['warren_buffett', 'Warren Buffett'],
    ['technical_analyst', 'Technical'],
    ['fundamentals_analyst', 'Fundamentals'],
    ['growth_analyst', 'Growth'],
    ['news_sentiment_analyst', 'News Sentiment'],
    ['sentiment_analyst', 'Sentiment'],
    ['valuation_analyst', 'Valuation'],
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

  function authHeaders() {
    return { 'x-session-id': window.SESSION_ID };
  }

  function isEditorSignedIn() {
    if (typeof window.getStoredAuthUser === 'function') return !!window.getStoredAuthUser();
    try {
      return !!JSON.parse(localStorage.getItem('auth-user') || 'null');
    } catch (_) {
      return false;
    }
  }

  function renderAiHedgeFundAnalysts(agent) {
    const grid = document.getElementById('agentEditorAiHedgeFundAnalysts');
    if (!grid) return;
    const selected = new Set(agent?.runtime_config?.analysts || []);
    grid.innerHTML = AI_HEDGE_FUND_ANALYSTS.map(([id, label]) => `
      <label class="agent-editor-analyst-option">
        <input type="checkbox" name="agentEditorAiHedgeFundAnalyst" value="${escapeHtml(id)}" ${selected.has(id) ? 'checked' : ''}>
        <span>${escapeHtml(label)}</span>
      </label>`).join('');
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
    return `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
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

  function getEditorState() {
    const hostedAiHedgeFund = isAiHedgeFundAgent();
    const nameInput = document.getElementById('agentEditorNameInput');
    const descInput = document.getElementById('agentEditorDescription');
    const cashInput = document.getElementById('agentEditorCashAllocation');
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
      if (value > 10000) {
        throw new Error('Backtest Allocated Capital cannot exceed $10,000.');
      }
      backtest_allocation = Math.round(value);
    } else {
      // Non-positive counts as absent: cash_allocation is legally 0 (a $0 paper
      // sleeve), but backtest capital is >= 1 server-side, so 0 must fall
      // through to the default rather than becoming an unsaveable value.
      backtest_allocation =
        Number.isFinite(Number(cash_allocation)) && Number(cash_allocation) > 0
          ? Math.min(Math.round(Number(cash_allocation)), 10000)
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
      description: descInput ? descInput.value.trim() : '',
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
    const descInput = document.getElementById('agentEditorDescription');
    const cashInput = document.getElementById('agentEditorCashAllocation');
    const meta = document.getElementById('agentEditorMeta');

    if (nameInput) nameInput.value = agent.name || '';
    if (descInput) descInput.value = agent.description || '';
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
      backtestInput.value = String(Math.min(Math.round(resolved), 10000));
    }
    if (meta) {
      meta.textContent = agent.agent_type === 'builtin' ? 'Built-in agent' : 'External agent';
    }
    const liveToggle = document.getElementById('agentEditorLiveTradingEnabled');
    if (liveToggle) liveToggle.checked = Boolean(agent.live_trading_enabled);
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
  ) {
    const payload = {
      name,
      description: description || null,
      cash_allocation,
      backtest_allocation,
      live_trading_enabled: Boolean(live_trading_enabled),
    };
    if (pipeline) payload.pipeline = serializePipeline(pipeline);
    if (model_name) payload.model_name = model_name;
    if (runtimeConfig) payload.runtime_config = runtimeConfig;
    const endpoint = `${API_BASE}/api/v1/agents/${encodeURIComponent(agent.agent_id)}`;

    async function requestWithHeaders(extraHeaders) {
      if (window.API?.patch) {
        return window.API.patch(endpoint, payload, extraHeaders);
      }
      const headers = {
        'Content-Type': 'application/json',
        'x-session-id': window.SESSION_ID,
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
    if (!container) return;

    const sorted = [...(runs || [])].sort(
      (a, b) => (b.created_at || '').localeCompare(a.created_at || ''),
    );

    if (countEl) {
      countEl.textContent = `${sorted.length} run${sorted.length === 1 ? '' : 's'}`;
    }

    if (!sorted.length) {
      container.innerHTML = '<p class="agent-editor-run-empty">No backtest runs yet. Run a backtest from this agent to see history here.</p>';
      return;
    }

    container.innerHTML = sorted
      .map(
        (run) => `
          <button type="button" class="agent-editor-run-item" data-run-id="${escapeHtml(run.run_id)}" role="listitem">
            <span class="agent-editor-run-primary">${escapeHtml(formatRunPrimary(run))}</span>
            <span class="agent-editor-run-secondary">${escapeHtml(formatRunSecondary(run))}</span>
            ${formatRunMeta(run) ? `<span class="agent-editor-run-meta">${escapeHtml(formatRunMeta(run))}</span>` : ''}
          </button>`,
      )
      .join('');

    container.querySelectorAll('.agent-editor-run-item').forEach((btn) => {
      btn.addEventListener('click', () => {
        const runId = btn.dataset.runId;
        if (!currentAgent || !runId) return;
        window.dispatchEvent(
          new CustomEvent('agent-editor-open-run', {
            detail: { agent: currentAgent, runId },
          }),
        );
      });
    });
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
          saveBtn.textContent = 'Save';
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
        saveBtn.textContent = 'Save';
      }
    }
  }

  function bindEvents() {
    document.getElementById('agentEditorBackBtn')?.addEventListener('click', () => close(false));
    document.getElementById('agentEditorSaveBtn')?.addEventListener('click', () => save());
    document.getElementById('agentEditorConnectRobinhoodBtn')?.addEventListener('click', connectRobinhood);
    document.getElementById('agentEditorDisconnectRobinhoodBtn')?.addEventListener('click', disconnectRobinhood);
    document.getElementById('agentEditorFinancialDatasetsRemove')?.addEventListener('click', removeFinancialDatasetsCredential);
    document.getElementById('agentEditorRunLiveBtn')?.addEventListener('click', runLive);
    document.getElementById('agentEditorRunBacktestBtn')?.addEventListener('click', () => {
      if (!currentAgent) return;
      // The modal reads the last-saved agent, so an unsaved edit would run the
      // old instruction while the preview shows the new one. Same guard as Run Live.
      if (isDirty) {
        showSaveStatus('Save changes before Run Backtest', true);
        return;
      }
      if (typeof window.openRunBacktestModal === 'function') {
        window.openRunBacktestModal(currentAgent);
      }
    });

    const body = document.getElementById('agentEditorView');
    body?.addEventListener('input', markDirtyFromInput);
    body?.addEventListener('change', markDirtyFromInput);

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
