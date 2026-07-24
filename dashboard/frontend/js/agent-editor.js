/**
 * agent-editor.js — Fullscreen agent Configure screen.
 * Simple mode: capital + one plain-language instruction (stored as a 1-step
 * pipeline) + model. Advanced mode: the multi-step sub-agent chain.
 * Agent fields: PATCH /api/v1/agents/{id}. Pipeline cache: localStorage.
 */
(function () {
  'use strict';

  const STORAGE_PREFIX = 'agent-pipeline-config:';
  const NAME_OVERRIDE_PREFIX = 'agent-name-override:';
  const CASH_OVERRIDE_PREFIX = 'agent-cash-allocation:';
  const API_BASE = window.location.origin;

  // The Simple-mode contract (preset key + trading-actions output format) has a
  // single source of truth in app.js, published on `window`. app.js loads AFTER
  // this file, so read lazily at call time — every use below is inside an
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

  // Demo/mock agents (see MOCK_AGENTS in app.js) only exist in the frontend —
  // they have no database row, so PATCH would 404. We persist their edits
  // locally instead so the rename is still reflected in the UI.
  function isDemoAgent(agentId) {
    return typeof agentId === 'string' && agentId.startsWith('mock-');
  }

  const SUB_AGENT_PRESETS = [
    {
      presetKey: 'info_gather',
      label: 'Information Gathering',
      defaultPrompt:
        'You are the information-gathering sub-agent. Collect key facts relevant to trading decisions from market data, news, and macro events. Filter noise and keep high-confidence facts and indicator changes.',
      defaultOutputFormat:
        'JSON: { "timestamp": "ISO8601", "symbols": ["..."], "facts": [{ "source": "...", "summary": "...", "impact": "bullish|bearish|neutral" }], "confidence": 0.0-1.0 }',
    },
    {
      presetKey: 'info_to_signal',
      label: 'Information to Signal',
      defaultPrompt:
        'You are the signal-generation sub-agent. Based on upstream information-gathering output, convert facts and indicators into executable trading signals (direction, strength, time horizon).',
      defaultOutputFormat:
        'JSON: { "signals": [{ "symbol": "...", "direction": "long|short|flat", "strength": 0.0-1.0, "horizon": "1h|4h|1d", "rationale": "..." }] }',
    },
    {
      presetKey: 'signal_to_execution',
      label: 'Signal to Execution',
      defaultPrompt:
        'You are the trade-execution sub-agent. Turn signals into concrete order instructions, respecting position limits, liquidity, and slippage. Output a submit-ready buy/sell plan.',
      defaultOutputFormat:
        'JSON: { "orders": [{ "symbol": "...", "side": "buy|sell|hold", "qty": number, "order_type": "market|limit", "limit_price": number|null, "reason": "..." }] }',
    },
    {
      presetKey: 'global_strategy',
      label: 'Global Trading Strategy',
      defaultPrompt:
        'You are the global strategy sub-agent. Coordinate asset allocation, risk budget, and conflicting signals so the overall portfolio stays within strategy constraints and target return/risk profile.',
      defaultOutputFormat:
        'JSON: { "portfolio_targets": [{ "symbol": "...", "target_weight": 0.0-1.0 }], "risk_budget": { "max_drawdown": number, "max_single_position": number }, "strategy_notes": "..." }',
    },
    {
      presetKey: 'stop_loss_take_profit',
      label: 'Stop Loss / Take Profit',
      defaultPrompt:
        'You are the risk-management sub-agent. Set stop-loss and take-profit rules for open positions, monitor unrealized P/L, and output close or reduce instructions when triggers fire.',
      defaultOutputFormat:
        'JSON: { "risk_actions": [{ "symbol": "...", "action": "stop_loss|take_profit|trail|hold", "trigger_price": number, "size_pct": 0.0-1.0, "reason": "..." }] }',
    },
    {
      presetKey: 'post_trade_analysis',
      label: 'Post-trade Analysis',
      defaultPrompt:
        'You are the post-trade analysis sub-agent. Once per trading day, review that day\'s trades, equity change, and the current decision-step prompts. Identify what went wrong in those prompts (missed risk, bad signal rules, weak execution constraints). Then propose revised prompts for the upstream sub-agents so the next trading day can improve. Do not invent market facts beyond the episode context you are given.',
      defaultOutputFormat:
        'JSON: { "summary": "...", "prompt_problems": [{ "step_id": "...", "presetKey": "...", "issue": "..." }], "prompt_patches": [{ "step_id": "...", "presetKey": "...", "new_prompt": "...", "change_rationale": "..." }] }',
    },
    {
      presetKey: 'custom',
      label: 'Custom Sub-agent',
      defaultPrompt: 'Describe this sub-agent\'s responsibilities and decision boundaries.',
      defaultOutputFormat: 'JSON: { "output": "..." }',
    },
  ];

  const presetByKey = Object.fromEntries(SUB_AGENT_PRESETS.map((p) => [p.presetKey, p]));

  let currentAgent = null;
  let subAgents = [];
  let saveStatusTimer = null;
  let isDirty = false;
  let savedSnapshot = '';
  let editorMode = 'simple';

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

  function createSubAgentFromPreset(presetKey, customLabel) {
    const preset = presetByKey[presetKey] || presetByKey.custom;
    const label =
      presetKey === 'custom' && customLabel
        ? customLabel.trim()
        : preset.label;
    return {
      id: newSubAgentId(),
      presetKey,
      label,
      prompt: preset.defaultPrompt,
      outputFormat: preset.defaultOutputFormat,
    };
  }

  function defaultPipeline() {
    return SUB_AGENT_PRESETS.filter((p) => p.presetKey !== 'custom').map((preset) => ({
      id: newSubAgentId(),
      presetKey: preset.presetKey,
      label: preset.label,
      prompt: preset.defaultPrompt,
      outputFormat: preset.defaultOutputFormat,
    }));
  }

  function normalizeLoadedSubAgent(item) {
    const presetKey = item.presetKey || 'custom';
    const preset = presetByKey[presetKey];
    const isCustom = presetKey === 'custom' || !preset;
    return {
      id: item.id || newSubAgentId(),
      presetKey,
      label: isCustom ? (item.label || 'Sub-agent') : preset.label,
      prompt: item.prompt || (preset ? preset.defaultPrompt : ''),
      outputFormat: item.outputFormat || (preset ? preset.defaultOutputFormat : ''),
    };
  }

  // For real agents the backend-stored pipeline is the source of truth; fall
  // back to any locally cached / default pipeline when the agent has none yet.
  // Demo (mock) agents have no backend row, so they always use localStorage.
  function resolvePipeline(agent) {
    if (!isDemoAgent(agent.agent_id) && Array.isArray(agent.pipeline) && agent.pipeline.length) {
      return agent.pipeline.map(normalizeLoadedSubAgent);
    }
    return loadPipeline(agent.agent_id);
  }

  function isSimplePipeline(pipeline) {
    return (
      !Array.isArray(pipeline) ||
      pipeline.length === 0 ||
      (pipeline.length === 1 && pipeline[0].presetKey === simplePresetKey())
    );
  }

  // The pipeline the agent ACTUALLY has (backend row, then local cache) — with
  // NO default-5-step substitution, so a fresh agent opens in Simple mode.
  // Demo agents keep resolvePipeline's default behavior.
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

  function loadPipeline(agentId) {
    try {
      const raw = localStorage.getItem(storageKey(agentId));
      if (!raw) return defaultPipeline();
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed.subAgents) || !parsed.subAgents.length) {
        return defaultPipeline();
      }
      return parsed.subAgents.map(normalizeLoadedSubAgent);
    } catch {
      return defaultPipeline();
    }
  }

  function savePipelineLocal(agentId, agents) {
    localStorage.setItem(
      storageKey(agentId),
      JSON.stringify({ subAgents: agents, updatedAt: new Date().toISOString() })
    );
  }

  function getEditorState() {
    const nameInput = document.getElementById('agentEditorNameInput');
    const descInput = document.getElementById('agentEditorDescription');
    const cashInput = document.getElementById('agentEditorCashAllocation');
    let cash_allocation = null;
    if (cashInput && cashInput.value !== '') {
      const value = Number(cashInput.value);
      if (!Number.isFinite(value) || value < 0) {
        throw new Error('Allocated capital must be zero or greater.');
      }
      if (value > 3000) {
        throw new Error('Allocated capital cannot exceed $3,000.');
      }
      cash_allocation = Math.round(value);
    } else {
      cash_allocation = 1000;
    }
    const modelSelect = document.getElementById('agentEditorModelSelect');
    let subAgentsOut;
    let sendPipeline;
    if (editorMode === 'simple') {
      const instruction = (
        document.getElementById('agentEditorSimpleInstruction')?.value || ''
      ).trim();
      if (instruction) {
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
        // Empty instruction never touches the stored pipeline: not sent to the
        // server, not cached locally, not folded into currentAgent.
        subAgentsOut = subAgents;
        sendPipeline = false;
      }
    } else {
      subAgentsOut = collectPipelineFromDom();
      sendPipeline = true;
    }
    return {
      name: nameInput ? nameInput.value.trim() : '',
      description: descInput ? descInput.value.trim() : '',
      cash_allocation,
      model_name: modelSelect ? modelSelect.value : '',
      subAgents: subAgentsOut,
      sendPipeline,
    };
  }

  function snapshotState() {
    return JSON.stringify(getEditorState());
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

  function collectPipelineFromDom() {
    const pipeline = document.getElementById('agentEditorPipeline');
    if (!pipeline) return subAgents;

    return subAgents.map((sub) => {
      const card = pipeline.querySelector(`[data-sub-id="${sub.id}"]`);
      if (!card) return sub;
      const labelInput = card.querySelector('[data-field="label"]');
      const promptInput = card.querySelector('[data-field="prompt"]');
      const outputInput = card.querySelector('[data-field="outputFormat"]');
      return {
        ...sub,
        label: labelInput ? labelInput.value.trim() || sub.label : sub.label,
        prompt: promptInput ? promptInput.value : sub.prompt,
        outputFormat: outputInput ? outputInput.value : sub.outputFormat,
      };
    });
  }

  function showSaveStatus(message, isError) {
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

  function updateStepCount() {
    const el = document.getElementById('agentEditorStepCount');
    if (!el) return;
    const n = subAgents.length;
    el.textContent = `${n} step${n === 1 ? '' : 's'}`;
  }

  function refreshAddSelect() {
    const select = document.getElementById('agentEditorAddSelect');
    const customName = document.getElementById('agentEditorCustomName');
    if (!select) return;

    const usedPresetKeys = new Set(
      subAgents.filter((s) => s.presetKey !== 'custom').map((s) => s.presetKey)
    );

    select.innerHTML = '<option value="">— Select type —</option>';
    SUB_AGENT_PRESETS.forEach((preset) => {
      if (preset.presetKey !== 'custom' && usedPresetKeys.has(preset.presetKey)) return;
      const opt = document.createElement('option');
      opt.value = preset.presetKey;
      opt.textContent = preset.label;
      select.appendChild(opt);
    });

    const customOpt = document.createElement('option');
    customOpt.value = 'custom';
    customOpt.textContent = 'Custom Sub-agent';
    select.appendChild(customOpt);

    if (customName) {
      customName.hidden = select.value !== 'custom';
    }
  }

  function moveSubAgent(index, delta) {
    const next = index + delta;
    if (next < 0 || next >= subAgents.length) return;
    subAgents = collectPipelineFromDom();
    const tmp = subAgents[index];
    subAgents[index] = subAgents[next];
    subAgents[next] = tmp;
    renderPipeline();
    markDirtyFromInput();
  }

  function duplicateSubAgent(subId) {
    subAgents = collectPipelineFromDom();
    const idx = subAgents.findIndex((s) => s.id === subId);
    if (idx < 0) return;
    const src = subAgents[idx];
    const copy = {
      ...src,
      id: newSubAgentId(),
      presetKey: 'custom',
      label: `${src.label} (copy)`,
    };
    subAgents.splice(idx + 1, 0, copy);
    renderPipeline();
    markDirtyFromInput();
    showSaveStatus('Sub-agent duplicated');
  }

  function restoreSubAgentDefaults(subId) {
    subAgents = collectPipelineFromDom();
    const sub = subAgents.find((s) => s.id === subId);
    if (!sub) return;
    const preset = presetByKey[sub.presetKey];
    if (!preset || sub.presetKey === 'custom') return;
    sub.prompt = preset.defaultPrompt;
    sub.outputFormat = preset.defaultOutputFormat;
    renderPipeline();
    markDirtyFromInput();
    showSaveStatus('Restored default prompt for this step');
  }

  function renderPipeline() {
    const pipeline = document.getElementById('agentEditorPipeline');
    if (!pipeline) return;

    pipeline.innerHTML = '';
    updateStepCount();

    subAgents.forEach((sub, index) => {
      const card = document.createElement('article');
      card.className = 'section-card agent-sub-card';
      card.setAttribute('role', 'listitem');
      card.dataset.subId = sub.id;

      const isCustom = sub.presetKey === 'custom';
      const isFirst = index === 0;
      const isLast = index === subAgents.length - 1;
      const canRestore = !isCustom && presetByKey[sub.presetKey];

      const isPostTrade = sub.presetKey === 'post_trade_analysis';
      const nextIsPostTrade =
        !isLast && subAgents[index + 1] && subAgents[index + 1].presetKey === 'post_trade_analysis';
      const labelField = isCustom
        ? `<input class="agent-sub-label-input" type="text" data-field="label" value="${escapeHtml(sub.label)}" placeholder="Sub-agent name" aria-label="Sub-agent name">`
        : `<span class="agent-sub-preset-label">${escapeHtml(sub.label)}${
            isPostTrade
              ? '<span class="agent-sub-freq-badge" title="Runs once per trading day after trades">daily after trades</span>'
              : ''
          }</span>`;

      card.innerHTML = `
        <div class="agent-sub-head">
          <div class="agent-sub-head-left">
            <span class="agent-sub-index" aria-hidden="true">${index + 1}</span>
            ${labelField}
          </div>
          <div class="agent-sub-actions">
            <div class="agent-sub-reorder" role="group" aria-label="Reorder sub-agent">
              <button class="agent-sub-move-btn" type="button" data-action="up" data-sub-id="${escapeHtml(sub.id)}" aria-label="Move up" ${isFirst ? 'disabled' : ''}>↑</button>
              <button class="agent-sub-move-btn" type="button" data-action="down" data-sub-id="${escapeHtml(sub.id)}" aria-label="Move down" ${isLast ? 'disabled' : ''}>↓</button>
            </div>
            <button class="agent-sub-icon-btn" type="button" data-action="duplicate" data-sub-id="${escapeHtml(sub.id)}" title="Duplicate">⧉</button>
            ${canRestore ? `<button class="agent-sub-icon-btn" type="button" data-action="restore" data-sub-id="${escapeHtml(sub.id)}" title="Restore defaults">↺</button>` : ''}
            <button class="agent-sub-remove-btn" type="button" data-action="remove" data-sub-id="${escapeHtml(sub.id)}" aria-label="Remove sub-agent">Remove</button>
          </div>
        </div>
        <div class="agent-sub-fields">
          <label class="agent-sub-field">
            <span class="agent-sub-field-label">Task prompt</span>
            <span class="agent-sub-field-hint">What this sub-agent should do</span>
            <textarea data-field="prompt" rows="5" placeholder="Describe this sub-agent's role…">${escapeHtml(sub.prompt)}</textarea>
          </label>
          <label class="agent-sub-field">
            <span class="agent-sub-field-label">Output format</span>
            <span class="agent-sub-field-hint">Structure passed to the next model</span>
            <textarea data-field="outputFormat" rows="4" placeholder="JSON or structured text…">${escapeHtml(sub.outputFormat)}</textarea>
          </label>
        </div>
        ${
          !isLast
            ? `<div class="agent-sub-connector" aria-hidden="true"><span>${
                nextIsPostTrade
                  ? '↓ Then post-trade analysis (once per day)'
                  : '↓ Output to next sub-agent'
              }</span></div>`
            : ''
        }
      `;

      pipeline.appendChild(card);
    });

    pipeline.querySelectorAll('[data-action]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        const subId = btn.dataset.subId;
        const idx = subAgents.findIndex((s) => s.id === subId);
        if (action === 'up') moveSubAgent(idx, -1);
        else if (action === 'down') moveSubAgent(idx, 1);
        else if (action === 'duplicate') duplicateSubAgent(subId);
        else if (action === 'restore') restoreSubAgentDefaults(subId);
        else if (action === 'remove') {
          if (subAgents.length <= 1) {
            showSaveStatus('Keep at least one sub-agent', true);
            return;
          }
          subAgents = collectPipelineFromDom().filter((s) => s.id !== subId);
          renderPipeline();
          refreshAddSelect();
          markDirtyFromInput();
        }
      });
    });

    refreshAddSelect();
  }

  function updateSimpleReplaceNote() {
    const note = document.getElementById('agentEditorSimpleReplaceNote');
    if (note) note.hidden = !(editorMode === 'simple' && !isSimplePipeline(subAgents));
  }

  function setEditorMode(mode) {
    editorMode = mode === 'advanced' ? 'advanced' : 'simple';
    const simplePanel = document.getElementById('agentEditorSimplePanel');
    const advancedPanel = document.getElementById('agentEditorAdvancedPanel');
    const resetBtn = document.getElementById('agentEditorResetBtn');
    if (simplePanel) simplePanel.hidden = editorMode !== 'simple';
    if (advancedPanel) advancedPanel.hidden = editorMode !== 'advanced';
    if (resetBtn) resetBtn.hidden = editorMode !== 'advanced';
    const modeSimpleBtn = document.getElementById('agentEditorModeSimple');
    const modeAdvancedBtn = document.getElementById('agentEditorModeAdvanced');
    modeSimpleBtn?.classList.toggle('active', editorMode === 'simple');
    modeAdvancedBtn?.classList.toggle('active', editorMode === 'advanced');
    modeSimpleBtn?.setAttribute('aria-pressed', editorMode === 'simple' ? 'true' : 'false');
    modeAdvancedBtn?.setAttribute('aria-pressed', editorMode === 'advanced' ? 'true' : 'false');
    if (editorMode === 'advanced' && subAgents.length === 0) {
      // First look at Advanced on a fresh agent: start from the default chain.
      subAgents = defaultPipeline();
      renderPipeline();
    }
    updateSimpleReplaceNote();
    markDirtyFromInput();
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
    if (meta) {
      meta.textContent = agent.agent_type === 'builtin' ? 'Built-in agent' : 'External agent';
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

  async function patchAgent(agent, name, description, pipeline, cash_allocation, model_name) {
    const payload = {
      name,
      description: description || null,
      cash_allocation,
    };
    if (pipeline) payload.pipeline = serializePipeline(pipeline);
    if (model_name) payload.model_name = model_name;
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
      const token = localStorage.getItem('auth-token');
      if (token) headers.Authorization = `Bearer ${token}`;

      const response = await fetch(endpoint, {
        method: 'PATCH',
        headers,
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
      const token = localStorage.getItem('auth-token');
      if (token) headers.Authorization = `Bearer ${token}`;

      const response = await fetch(
        `${API_BASE}/api/v1/agents/${encodeURIComponent(agent.agent_id)}`,
        { headers },
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
    if (isDemoAgent(agent.agent_id)) {
      subAgents = resolvePipeline(agent); // demo agents keep the legacy default chain
    } else {
      subAgents = loadStoredPipeline(agent);
    }
    fillHeader(agent);
    populateModelSelect(agent);
    const instructionEl = document.getElementById('agentEditorSimpleInstruction');
    if (instructionEl) {
      const simpleStep =
        subAgents.length === 1 && subAgents[0].presetKey === simplePresetKey()
          ? subAgents[0]
          : null;
      instructionEl.value = simpleStep ? simpleStep.prompt : '';
    }
    renderPipeline();
    setEditorMode(isSimplePipeline(subAgents) ? 'simple' : 'advanced');
    refreshRunHistory(currentAgent);

    const view = document.getElementById('agentEditorView');
    if (view) {
      view.hidden = false;
      document.body.classList.add('agent-editor-open');
    }

    const playgroundView = document.getElementById('playgroundView');
    if (playgroundView) playgroundView.setAttribute('aria-hidden', 'true');

    captureSavedSnapshot();
    document.getElementById('agentEditorNameInput')?.focus();
  }

  function close(force) {
    if (!force && isDirty) {
      if (!window.confirm('Discard unsaved changes?')) return;
    }

    subAgents = collectPipelineFromDom();

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

    subAgents = state.subAgents;
    renderPipeline();
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

    try {
      const updated = await patchAgent(
        currentAgent,
        state.name,
        state.description,
        state.sendPipeline ? subAgents : null,
        state.cash_allocation,
        state.model_name
      );
      currentAgent = state.sendPipeline
        ? { ...currentAgent, ...updated, pipeline: subAgents }
        : { ...currentAgent, ...updated };
      if (state.sendPipeline) savePipelineLocal(currentAgent.agent_id, subAgents);
      localStorage.removeItem(`${NAME_OVERRIDE_PREFIX}${currentAgent.agent_id}`);

      if (localStorage.getItem('active-agent-id') === currentAgent.agent_id) {
        localStorage.setItem('active-agent-name', state.name);
      }

      fillHeader(currentAgent);
      captureSavedSnapshot();
      showSaveStatus('Saved successfully');
      window.dispatchEvent(
        new CustomEvent('agent-editor-saved', { detail: { agent: currentAgent } })
      );
    } catch (error) {
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

  function resetPipeline() {
    if (!window.confirm('Reset the pipeline to the default 5 sub-agents? Unsaved prompt edits will be lost.')) {
      return;
    }
    subAgents = defaultPipeline();
    renderPipeline();
    markDirtyFromInput();
    showSaveStatus('Pipeline reset to defaults');
  }

  function addSubAgent() {
    const select = document.getElementById('agentEditorAddSelect');
    const customNameEl = document.getElementById('agentEditorCustomName');
    const presetKey = select ? select.value : '';
    if (!presetKey) {
      showSaveStatus('Select a sub-agent type to add', true);
      return;
    }

    subAgents = collectPipelineFromDom();

    let customLabel = null;
    if (presetKey === 'custom' && customNameEl) {
      customLabel = customNameEl.value.trim() || 'Custom Sub-agent';
    }

    subAgents.push(createSubAgentFromPreset(presetKey, customLabel));
    renderPipeline();
    markDirtyFromInput();
    showSaveStatus('Sub-agent added');

    if (select) select.value = '';
    if (customNameEl) {
      customNameEl.value = '';
      customNameEl.hidden = true;
    }

    const pipeline = document.getElementById('agentEditorPipeline');
    if (pipeline && pipeline.lastElementChild) {
      pipeline.lastElementChild.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function bindEvents() {
    document.getElementById('agentEditorBackBtn')?.addEventListener('click', () => close(false));
    document.getElementById('agentEditorSaveBtn')?.addEventListener('click', () => save());
    document.getElementById('agentEditorAddBtn')?.addEventListener('click', addSubAgent);
    document.getElementById('agentEditorResetBtn')?.addEventListener('click', resetPipeline);
    document.getElementById('agentEditorModeSimple')?.addEventListener('click', () => setEditorMode('simple'));
    document.getElementById('agentEditorModeAdvanced')?.addEventListener('click', () => setEditorMode('advanced'));
    document.getElementById('agentEditorRunBacktestBtn')?.addEventListener('click', () => {
      if (!currentAgent) return;
      if (typeof window.openRunBacktestModal === 'function') {
        window.openRunBacktestModal(currentAgent);
      }
    });

    document.getElementById('agentEditorAddSelect')?.addEventListener('change', (e) => {
      const customName = document.getElementById('agentEditorCustomName');
      if (customName) customName.hidden = e.target.value !== 'custom';
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

  window.AgentEditor = { open, close, save, loadPipeline };
})();
