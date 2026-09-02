/**
 * Agentic Trading Lab - Frontend Application
 * Connects to backend API for real data
 */

// ============================================================================
// Session Management (Anonymous Browser Isolation)
// ============================================================================

// Initialize anonymous session on first load
const ACTIVE_AGENT_KEY = 'active-agent-id';
const ACTIVE_AGENT_NAME_KEY = 'active-agent-name';
const BROWSER_OWNER_KEY = 'browser-owner-id';
const HIDDEN_DEMO_AGENTS_KEY = 'hidden-demo-agent-ids';
const SELECTED_BACKTEST_RUN_KEY = 'selected-backtest-run-id';
// index.html's goToDashboardLoggedIn() writes this same key as a bare string
// literal (no build step to share this constant across the landing/app split).
const NAV_STATE_KEY = 'nav-state';
const DISCORD_SERVER_URL = 'https://discord.gg/9HnQ6XDG98';
const BACKTEST_POLL_MAX_SECONDS = 3600; // 60 minutes at 1-second polling intervals

function initSession() {
  // Stable browser identity — never changes when switching agents.
  // Bootstrap from trading-session-id so legacy agents whose
  // owner_browser_session equals their session id keep working.
  let browserOwnerId = localStorage.getItem(BROWSER_OWNER_KEY);
  if (!browserOwnerId) {
    browserOwnerId = localStorage.getItem('trading-session-id') || crypto.randomUUID();
    localStorage.setItem(BROWSER_OWNER_KEY, browserOwnerId);
  }
  window.BROWSER_OWNER_ID = browserOwnerId;

  // Trading session — switches per active agent (backtest data scope)
  let sessionId = localStorage.getItem('trading-session-id');
  if (!sessionId) {
    sessionId = browserOwnerId;
    localStorage.setItem('trading-session-id', sessionId);
    console.log('New trading session:', sessionId);
  } else {
    console.log('Restored trading session:', sessionId);
  }
  window.SESSION_ID = sessionId;
}

async function restoreActiveAgentSession() {
  const agentId = localStorage.getItem(ACTIVE_AGENT_KEY);
  if (!agentId) return;

  try {
    const data = await API.get(`${API_BASE}/api/v1/agents/${agentId}`);
    const agent = data.agent;
    if (!agent?.session_id) return;
    applyActiveAgent(agent, { persistActiveId: false });
    try {
      await API.post(`${API_BASE}/api/v1/agents/${agent.agent_id}/activate`, {});
    } catch (claimError) {
      console.warn('Agent claim on restore failed:', claimError.message);
    }
    console.log('Restored active agent:', agent.name, agent.session_id);
  } catch (error) {
    console.warn('Could not restore active agent:', error.message);
    // Only drop saved agent if it was deleted server-side
    if (String(error.message || '').includes('404') || String(error.message || '').includes('not found')) {
      localStorage.removeItem(ACTIVE_AGENT_KEY);
      localStorage.removeItem(ACTIVE_AGENT_NAME_KEY);
    }
  }
}

function applyActiveAgent(agent, options = {}) {
  if (!agent?.session_id) return;
  const previousSession = window.SESSION_ID;
  localStorage.setItem('trading-session-id', agent.session_id);
  if (options.persistActiveId !== false) {
    localStorage.setItem(ACTIVE_AGENT_KEY, agent.agent_id);
    localStorage.setItem(ACTIVE_AGENT_NAME_KEY, agent.name || '');
  }
  window.SESSION_ID = agent.session_id;
  window.ACTIVE_AGENT = agent;
  // Only drop the selected run when the trading session actually changes.
  // Re-activating the same agent must not wipe a run id we just pinned for navigation.
  if (
    options.clearSelectedRun === true ||
    (options.clearSelectedRun !== false && previousSession && previousSession !== agent.session_id)
  ) {
    localStorage.removeItem(SELECTED_BACKTEST_RUN_KEY);
  }

  const nameEl = document.getElementById('playgroundAgentName');
  if (nameEl) nameEl.textContent = agent.name || 'External Agent';

  const statusEl = document.getElementById('playgroundAgentStatus');
  if (statusEl) {
    statusEl.textContent = 'External';
    statusEl.className = 'status-badge baseline';
  }

  const discordEl = document.getElementById('playgroundAgentDiscord');
  if (discordEl) {
    discordEl.textContent = `Session ${agent.session_id.slice(0, 8)}…`;
    discordEl.className = 'agent-discord connected';
  }
}

async function activateAgent(agent) {
  applyActiveAgent(agent);
  try {
    await API.post(`${API_BASE}/api/v1/agents/${agent.agent_id}/activate`, {});
  } catch (error) {
    console.warn('Agent activate ping failed:', error.message);
  }
}

function formatAgentReturn(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  const pct = Number(value) * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(1)}%`;
}

function formatUsd(value) {
  const num = Number(value);
  if (value == null || Number.isNaN(num)) return null;
  if (num === 0) return '$0';
  if (num < 0.01) return `$${num.toFixed(4)}`;
  return `$${num.toFixed(num < 1 ? 3 : 2)}`;
}

function formatTokenCount(value) {
  const num = Number(value);
  if (!num || Number.isNaN(num)) return '0';
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}k`;
  return String(num);
}

let appToastTimer = null;

const APP_TOAST_VISIBLE_MS = 4000;
const APP_TOAST_FADE_MS = 240;
// How long the head boot script's /health warmup may stay pending before the
// boot handler tells the user the free-tier server is waking up. A warm
// server answers well under this; a cold start takes 30-60s.
const SLOW_BOOT_NOTICE_MS = 3000;

/**
 * Non-blocking confirmation channel for /app.
 *
 * The pre-existing convention here is alert(), which is modal: acceptable for a
 * launch-time refusal the user must acknowledge, wrong for a success they only
 * need to notice. Text (not innerHTML) -- callers pass agent names.
 *
 * The container is never `hidden`, and this function must never set it. `hidden`
 * is display:none, which takes the node out of the render tree, and a live
 * region is only monitored for mutations while it is rendered -- so writing the
 * message first and unhiding after (the obvious order) announces nothing on the
 * screen readers this role="status" exists for. .app-toast hides itself with
 * opacity + pointer-events, so `hidden` was buying no visual behaviour either.
 * Emptying the text on the way out is what replaces it: the region stays
 * registered from page load, and no stale message is left for a browsing user.
 */
function showAppToast(message) {
  const el = document.getElementById('appToast');
  if (!el) return;
  el.classList.remove('is-visible');
  // Force a reflow so re-showing an already-visible toast replays the transition.
  void el.offsetWidth;
  el.textContent = String(message);
  el.classList.add('is-visible');
  if (appToastTimer) clearTimeout(appToastTimer);
  appToastTimer = setTimeout(() => {
    el.classList.remove('is-visible');
    appToastTimer = setTimeout(() => { el.textContent = ''; }, APP_TOAST_FADE_MS);
  }, APP_TOAST_VISIBLE_MS);
}

// ============================================================================
// Local mock agents — fallback used when the backend returns no agents (or is
// unavailable). Lets the redesigned My Agents page render without a backend.
// TODO: Replace mock agent data with backend API data later.
// ============================================================================
const MAX_AGENT_CASH_ALLOCATION = 3000;
const DEFAULT_AGENT_CASH_ALLOCATION = 1000;
/** Simulated cash ceiling for a single backtest run — unrelated to the paper sleeve above. */
const MAX_BACKTEST_ALLOCATED_CAPITAL = 3000;
const DEFAULT_PORTFOLIO_EQUITY = 10000;
const AGENT_CASH_OVERRIDE_PREFIX = 'agent-cash-allocation:';

const DEFAULT_AGENT_KEY_PREFIX = 'default-agent-id:';

function defaultAgentKey() {
  return `${DEFAULT_AGENT_KEY_PREFIX}${window.BROWSER_OWNER_ID || 'anon'}`;
}

function getDefaultAgentId() {
  try {
    return localStorage.getItem(defaultAgentKey());
  } catch (e) {
    return null;
  }
}

function setDefaultAgentId(agentId) {
  try {
    localStorage.setItem(defaultAgentKey(), agentId);
  } catch (e) {
    /* storage unavailable — badge simply won't persist */
  }
}

const DEFAULT_AGENT_PROVISION_GUARD_PREFIX = 'default-agent-provisioned:';
const STARTER_AGENTS = [
  {
    name: 'DeepSeek V4 Pro',
    model_name: 'deepseek/deepseek-v4-pro',
    description: 'A DeepSeek V4 Pro starter — open it to edit the trading instruction and run a backtest.',
  },
  {
    name: 'GPT-5.5',
    model_name: 'openai/gpt-5.5',
    description: 'A GPT-5.5 starter — open it to edit the trading instruction and run a backtest.',
  },
  {
    name: 'Claude Sonnet 4.6',
    model_name: 'anthropic/claude-sonnet-4-6',
    description: 'A Claude Sonnet 4.6 starter — open it to edit the trading instruction and run a backtest.',
  },
];
const DEFAULT_FOUNDATION_MODEL = 'deepseek/deepseek-v4-pro';
const DEFAULT_STARTER_AGENT_NAME = 'DeepSeek V4 Pro';
const DEFAULT_STARTER_AGENT_DESCRIPTION =
  'A DeepSeek V4 Pro starter — open it to edit the trading instruction and run a backtest.';
const SIMPLE_INSTRUCTION_PRESET_KEY = 'simple_instruction';
const SIMPLE_INSTRUCTION_OUTPUT_FORMAT =
  'JSON: { "orders": [{ "symbol": "...", "side": "buy|sell|hold", "qty": number, "order_type": "market|limit", "limit_price": number|null, "reason": "..." }] }';
// Single source of truth for the Simple-mode trading-actions contract. Published
// on `window` so agent-editor.js (which loads after this file) reads the exact
// same preset key + output format at call time instead of keeping its own copy.
window.SIMPLE_INSTRUCTION_PRESET_KEY = SIMPLE_INSTRUCTION_PRESET_KEY;
window.SIMPLE_INSTRUCTION_OUTPUT_FORMAT = SIMPLE_INSTRUCTION_OUTPUT_FORMAT;
// Mirrors DEFAULT_STARTER_INSTRUCTION in dashboard/backend/domain/agents/defaults.py,
// which is what actually seeds new agents. The copy here populates the
// "See the default instruction" disclosure in Configure's empty-instruction
// state, so the editor can show what an agent falls back to without a pipeline.
// tests/test_agent_starter_defaults.py pins the two copies together.
const DEFAULT_STARTER_INSTRUCTION =
  'Spread the money across a few of the strongest available stocks. Buy on meaningful dips, take profits after strong run-ups, and never put everything into one stock.';
window.DEFAULT_STARTER_INSTRUCTION = DEFAULT_STARTER_INSTRUCTION;

function defaultAgentProvisionGuardKey() {
  // Prefer the signed-in account so a brand-new user on a browser that already
  // provisioned (or deleted) a guest starter still gets their own default.
  // Include created_at: local SQLite (and Render's ephemeral disk) recycle
  // user ids, so `u:1` from a wiped DB would skip provisioning for the next
  // account that lands on id 1. Guests keep the browser-scoped key so
  // logout→login claim can find it.
  const user = typeof getStoredAuthUser === 'function' ? getStoredAuthUser() : null;
  if (user?.id != null) {
    const created = String(user.created_at || '').trim();
    return created
      ? `${DEFAULT_AGENT_PROVISION_GUARD_PREFIX}u:${user.id}:${created}`
      : `${DEFAULT_AGENT_PROVISION_GUARD_PREFIX}u:${user.id}`;
  }
  return `${DEFAULT_AGENT_PROVISION_GUARD_PREFIX}b:${window.BROWSER_OWNER_ID || 'anon'}`;
}

function hasDefaultAgentProvisionGuard() {
  try {
    const key = defaultAgentProvisionGuardKey();
    if (localStorage.getItem(key)) return true;
    // Pre-fix legacy key (no u:/b: prefix). Honor it for guests only so we
    // do not duplicate a starter that was already provisioned; signed-in users
    // intentionally ignore it so a new account still gets onboarding.
    const user = typeof getStoredAuthUser === 'function' ? getStoredAuthUser() : null;
    if (user?.id != null) return false;
    const legacy = `${DEFAULT_AGENT_PROVISION_GUARD_PREFIX}${window.BROWSER_OWNER_ID || 'anon'}`;
    return Boolean(localStorage.getItem(legacy));
  } catch (e) {
    return true; // no storage → cannot guard → do not provision
  }
}

function formatAgentCashAllocation(value) {
  if (value == null || value === '') return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(Number(value));
}

function parseAgentCashAllocationInput(raw) {
  if (raw === '' || raw == null) {
    return DEFAULT_AGENT_CASH_ALLOCATION;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Paper Trading Allocated Capital must be between $0 and $${MAX_AGENT_CASH_ALLOCATION.toLocaleString()}.`);
  }
  if (value > MAX_AGENT_CASH_ALLOCATION) {
    throw new Error(`Paper Trading Allocated Capital cannot exceed $${MAX_AGENT_CASH_ALLOCATION.toLocaleString()}.`);
  }
  return Math.round(value);
}

/**
 * Native number spinners sometimes step by 1 on the first click even when a
 * larger step is configured. Intercept arrows and snap ±1 glitches so each
 * capital input follows its configured increment.
 */
const CASH_STEP_INPUT_IDS = [
  'externalAgentCashAllocation',
  'builtinAgentCashAllocation',
  'agentEditorCashAllocation',
  'agentEditorBacktestAllocation',
];

function cashStepMeta(input) {
  const step = Math.max(1, Number(input.step) || 100);
  const min = Number(input.min);
  const max = Number(input.max);
  return {
    step,
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : Number.POSITIVE_INFINITY,
  };
}

function snapCashStepValue(input, raw) {
  const { step, min, max } = cashStepMeta(input);
  let value = Number(raw);
  if (!Number.isFinite(value)) value = min;
  value = Math.round(value / step) * step;
  return Math.min(max, Math.max(min, value));
}

function nudgeCashStepInput(input, direction) {
  const { step, min, max } = cashStepMeta(input);
  const current = snapCashStepValue(input, input.value === '' ? min : input.value);
  const next = Math.min(max, Math.max(min, current + direction * step));
  input.value = String(next);
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
}

function bindCashStepInput(input) {
  if (!input || input.dataset.cashStepBound === '1') return;
  input.dataset.cashStepBound = '1';
  if (!input.step || input.step === 'any') input.step = '100';

  let lastValue = snapCashStepValue(input, input.value === '' ? 0 : input.value);

  input.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      nudgeCashStepInput(input, 1);
      lastValue = Number(input.value);
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      nudgeCashStepInput(input, -1);
      lastValue = Number(input.value);
    }
  });

  input.addEventListener('input', () => {
    const value = Number(input.value);
    if (!Number.isFinite(value)) return;
    const diff = value - lastValue;
    // Spinner glitch: first click often lands on ±1 instead of ±step.
    if (Math.abs(diff) === 1) {
      const step = Number(input.step) || 100;
      const corrected = snapCashStepValue(
        input,
        lastValue + (diff > 0 ? step : -step),
      );
      input.value = String(corrected);
      lastValue = corrected;
      return;
    }
    lastValue = value;
  });

  input.addEventListener('change', () => {
    if (input.value === '') return;
    const snapped = snapCashStepValue(input, input.value);
    if (String(snapped) !== input.value) input.value = String(snapped);
    lastValue = snapped;
  });
}

function bindCashStepInputs() {
  CASH_STEP_INPUT_IDS.forEach((id) => {
    bindCashStepInput(document.getElementById(id));
  });
}

function applyAgentCashAllocationOverride(agent) {
  if (!agent?.agent_id) return agent;
  if (agent.cash_allocation != null) return agent;
  try {
    const raw = localStorage.getItem(`${AGENT_CASH_OVERRIDE_PREFIX}${agent.agent_id}`);
    if (raw == null) return agent;
    const value = Number(raw);
    if (!Number.isFinite(value)) return agent;
    return { ...agent, cash_allocation: value };
  } catch (e) {
    return agent;
  }
}

function decorateAgent(agent) {
  return applyAgentCashAllocationOverride(applyAgentNameOverride(agent));
}

const MOCK_AGENTS = [
  {
    agent_id: 'mock-momentum-scout', name: 'Momentum Scout', agent_type: 'builtin',
    model_name: 'GPT-5.5', is_live: true, cash_allocation: 3000,
    paper_equity: 12480.32, paper_day_pnl: 184.2, paper_day_pnl_pct: 1.5,
    paper_buying_power: 4820, paper_open_positions: 6,
    paper_last_activity: 'Bought 4 NVDA · 18 min ago',
    paper_updated_at: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
    run_count: 2, latest_run: { total_return: 0.084, start_date: '2026-06-01', end_date: '2026-06-30', initial_equity: 10000, final_equity: 10842.5 },
    total_input_tokens: 41000, total_output_tokens: 21500, total_est_cost_usd: 0.085, runs: [],
  },
  {
    agent_id: 'mock-test-agent-2', name: 'test agent 2', agent_type: 'builtin',
    model_name: 'anthropic/claude-haiku-4-5', run_count: 1, cash_allocation: 3000,
    latest_run: {
      total_return: 0.08425, sharpe_ratio: 2.67,
      start_date: '2026-06-01', end_date: '2026-06-30',
      initial_equity: 10000, final_equity: 10842.5,
      created_at: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
    },
    total_input_tokens: 41000, total_output_tokens: 21500, total_est_cost_usd: 0.085, runs: [],
  },
  {
    agent_id: 'mock-test-agent', name: 'test agent', agent_type: 'builtin', is_active: true,
    model_name: 'anthropic/claude-haiku-4-5', run_count: 1,
    latest_run: { total_return: -0.004, sharpe_ratio: -16.84, start_date: '2026-05-01', end_date: '2026-05-31', initial_equity: 10000, final_equity: 9960 },
    total_input_tokens: 30000, total_output_tokens: 17500, total_est_cost_usd: 0.064, runs: [],
  },
  {
    agent_id: 'mock-draft-alpha', name: 'Alpha Draft', agent_type: 'builtin',
    model_name: 'anthropic/claude-haiku-4-5', run_count: 0, cash_allocation: 1000,
    latest_run: {}, total_input_tokens: 0, total_output_tokens: 0, runs: [],
  },
  {
    agent_id: 'mock-test', name: 'test', agent_type: 'external',
    model_name: 'local-model', run_count: 0, cash_allocation: 1000,
    latest_run: {}, total_input_tokens: 0, total_output_tokens: 0, runs: [],
  },
  {
    agent_id: 'mock-sdk-1', name: 'sdk-selftest-agent', agent_type: 'external',
    model_name: 'rule-based', run_count: 1,
    latest_run: { total_return: 0.02, sharpe_ratio: 4.25, start_date: '2026-06-01', end_date: '2026-06-30', initial_equity: 10000, final_equity: 10200 },
    total_input_tokens: 44800, total_output_tokens: 20000, total_est_cost_usd: 0.0, runs: [],
  },
  {
    agent_id: 'mock-sdk-2', name: 'sdk-selftest-agent', agent_type: 'external',
    model_name: 'rule-based', run_count: 1,
    latest_run: { total_return: 0.022, sharpe_ratio: 8.89 },
    total_input_tokens: 28400, total_output_tokens: 0, total_est_cost_usd: 0.0, runs: [],
  },
  {
    agent_id: 'mock-sdk-3', name: 'sdk-selftest-agent', agent_type: 'external',
    model_name: 'rule-based', run_count: 1,
    latest_run: { total_return: 0.022, sharpe_ratio: 8.89 },
    total_input_tokens: 21000, total_output_tokens: 0, total_est_cost_usd: 0.0, runs: [],
  },
  {
    agent_id: 'mock-sdk-4', name: 'sdk-selftest-agent', agent_type: 'external',
    model_name: 'rule-based', run_count: 1,
    latest_run: { total_return: 0.012, sharpe_ratio: 8.89 },
    total_input_tokens: 28400, total_output_tokens: 0, total_est_cost_usd: 0.0, runs: [],
  },
  {
    agent_id: 'mock-protocol-demo', name: 'protocol-demo', agent_type: 'external',
    model_name: 'rule-based-demo', run_count: 2,
    latest_run: { total_return: 0.06, sharpe_ratio: 7.38 },
    total_input_tokens: 0, total_output_tokens: 0, total_est_cost_usd: 0.0, runs: [],
  },
  {
    agent_id: 'mock-test-2', name: 'test', agent_type: 'external',
    model_name: 'local-model', run_count: 1,
    latest_run: { total_return: 0.081, sharpe_ratio: 25.66 },
    total_input_tokens: 7400, total_output_tokens: 0, total_est_cost_usd: 0.0, runs: [],
  },
];

// Holds the most recently loaded agents so the toolbar can re-filter without refetching.
let allAgents = [];
let agentViewMode = 'grid';
const AGENT_GRID_PAGE_SIZE = 5;

// Legacy runtime -> market, for an uncategorized agent whose runtime already
// implies one. Every agent cloned before shelving shipped carries
// `category: null`, and the hosted AI Hedge Fund runtime is a U.S. stock
// strategy. Keyed on `runtime_type` rather than backfilled in SQL because the
// fallback also covers rows written by an older backend that doesn't send
// `category` at all, which a one-shot migration cannot. New clones stamp the
// column and never reach this table.
const LEGACY_RUNTIME_MARKET = { ai_hedge_fund: 'us_stocks' };

/** Category slug -> market display name. The single place these strings are
 * written: the Prompted Models shelf's market chips, the Community category
 * chips, the agent-card submeta and the Configure picker all read this map, so
 * renaming a market is one edit. Key order is chip order and mirrors the
 * AgentCategory Literal's declaration order in
 * dashboard/backend/domain/agents/taxonomy.py.
 *
 * Markets, not asset classes: Prompted Models still filters by what an agent
 * trades, and equities are the only asset class the engine can backtest, so
 * both entries here live under that shelf. */
const MARKET_LABELS = {
  us_stocks: 'U.S.',
  cn_ashares: 'China A-Share',
};

// Exported for js/agent-editor.js, which builds the Configure screen's market
// <select> from this rather than a second hardcoded option list. agent-editor.js
// is loaded *before* app.js, so it must read this at call time (when the editor
// opens), never at its own module-init time -- the same rule window.API follows.
window.AGENT_SHELF_LABELS = MARKET_LABELS;

/** Every model a user can actually pick and run here. The single source for
 * both model <select> elements: the Run Backtest picker (#modelSelect, live
 * only on the iFinD A-share path) and the Create Built-in picker
 * (#builtinAgentModel, which the Configure editor clones its own options from).
 *
 * These lists were hand-maintained separately and drifted: the backtest picker
 * offered six models this platform does not run and omitted four it does, and
 * an agent on an unlisted model silently submitted the *previous* agent's
 * selection (see syncModelSelectFromAgent). Declaration order is display order.
 *
 * The AI Hedge Fund runtime's Nemotron is deliberately absent: it is a property
 * of a hosted runtime, not a user choice, and syncBacktestModelFieldMode
 * already renders that case as "AI Hedge Fund — hosted runtime". */
const SUPPORTED_MODELS = [
  { slug: 'anthropic/claude-haiku-4-5', label: 'Claude Haiku 4.5', vendor: 'anthropic' },
  { slug: 'anthropic/claude-sonnet-4-6', label: 'Claude Sonnet 4.6', vendor: 'anthropic' },
  { slug: 'openai/gpt-5.5', label: 'GPT-5.5', vendor: 'openai' },
  { slug: 'google/gemini-3.1-pro-preview', label: 'Gemini 3.1 Pro Preview', vendor: 'google' },
  { slug: 'deepseek/deepseek-v4-pro', label: 'DeepSeek V4 Pro', vendor: 'deepseek' },
  { slug: 'qwen/qwen3.7-plus', label: 'Qwen3.7 Plus', vendor: 'qwen' },
];

/** Pure: no DOM, so the guards can run it under node. */
function modelOptionsHtml(models) {
  return models
    .map((model) => `<option value="${escapeHtml(model.slug)}">${escapeHtml(model.label)}</option>`)
    .join('');
}

/** Fill both model pickers. Runs once, in the pure-DOM boot block, which is
 * before syncIFindModelControl can prepend #modelSelect's "Rule-based" option
 * -- calling this again later would wipe that option out. */
function populateSupportedModelSelects() {
  const html = modelOptionsHtml(SUPPORTED_MODELS);
  const backtestPicker = document.getElementById('modelSelect');
  if (backtestPicker) backtestPicker.innerHTML = html;
  const createPicker = document.getElementById('builtinAgentModel');
  if (createPicker) createPicker.innerHTML = html;
}

// My Agents' JS-driven sections, in display order. `match` delegates to
// agentShelfKey so every agent resolves to exactly one shelf by construction
// rather than by predicates staying mutually exclusive as they're edited.
//
// Crypto and Futures are deliberately NOT here. They are locked, inert rows in
// app.html with no grid, footer or empty-state element, so nothing in this file
// may try to address them: listing them would force a `locked` filter at every
// site that iterates this array, and one missed filter trips
// renderAgentCategories' "some grid is missing" guard, silently aborting the
// entire My Agents render. Their order is their order in app.html.
const AGENT_SHELVES = [
  { key: 'prompted', title: 'LLMs',
    match: (a) => agentShelfKey(a) === 'prompted' },
  { key: 'open', title: 'Open Agents',
    match: (a) => agentShelfKey(a) === 'open' },
  { key: 'external', title: 'For Developers: Connected Agents',
    match: (a) => agentShelfKey(a) === 'external' },
];

/** The single shelf an agent renders under. Exactly one value per agent, so no
 * agent can be double-counted or dropped off every shelf.
 *
 * Built-ins split on how they decide: a prompt-and-model pipeline lands on
 * Prompted Models; a hosted runtime (AI Hedge Fund today) lands on Open
 * Agents. Connected agents split off by `agent_type`. The market an agent
 * trades is a separate axis -- see agentMarketKey.
 *
 * runtime_type is always present and truthy (server-defaulted to 'pipeline'),
 * so the hosted check MUST be an inequality against 'pipeline', never a
 * truthiness test. */
function agentShelfKey(agent) {
  if (!agent || agent.agent_type !== 'builtin') return 'external';
  if ((agent.runtime_type || 'pipeline') !== 'pipeline') return 'open';
  return 'prompted';
}

/** Market a built-in agent trades, or '' when the platform genuinely doesn't
 * know -- a NULL/blank category, or a slug from a newer or older backend.
 *
 * '' is not a bug and must never hide the agent: those agents stay on
 * Prompted Models under the All chip and are excluded only by an explicit
 * market filter, which is the honest outcome when the market is unknown. */
function agentMarketKey(agent) {
  const slug = String(agent?.category || '').trim().toLowerCase();
  if (MARKET_LABELS[slug]) return slug;
  return LEGACY_RUNTIME_MARKET[String(agent?.runtime_type || '').trim().toLowerCase()] || '';
}

/** 'all' or one of MARKET_LABELS' keys. Narrows the Prompted Models shelf's
 * grid only -- never its count pill, which reports what the shelf holds. */
let agentMarketFilter = 'all';

/** 'us_stocks' -> 'UsStocks' -- app.html's per-shelf element id suffix (agentsGrid<Suffix> etc). */
function shelfIdSuffix(shelfKey) {
  return String(shelfKey)
    .split('_')
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join('');
}

/** Per-shelf page index (0-based), keyed by AGENT_SHELVES' `key`. Reset on search change. */
let agentGridPage = Object.fromEntries(AGENT_SHELVES.map((shelf) => [shelf.key, 0]));

function agentGridPageCount(total) {
  return Math.max(1, Math.ceil(total / AGENT_GRID_PAGE_SIZE));
}

function normalizeAgentGridPage(categoryKey, total) {
  const maxPage = agentGridPageCount(total) - 1;
  const page = agentGridPage[categoryKey] || 0;
  agentGridPage[categoryKey] = Math.min(Math.max(page, 0), maxPage);
  return agentGridPage[categoryKey];
}

/** @returns {{ key: 'paper'|'backtested'|'draft', label: string, className: string }} */
function resolveAgentStatusBadge(agent) {
  const deployment = String(agent.deployment_status || '').toLowerCase();
  if (
    agent.is_live === true ||
    deployment === 'live' ||
    deployment === 'paper'
  ) {
    return { key: 'paper', label: 'PAPER TRADING', className: 'paper' };
  }
  const runCount = Number(agent.run_count) || (Array.isArray(agent.runs) ? agent.runs.length : 0);
  if (runCount > 0 || agent.latest_run?.run_id || agent.latest_run?.total_return != null) {
    return { key: 'backtested', label: 'BACKTESTED', className: 'idle' };
  }
  // Not "DRAFT": the agent is saved and its capital is already reserved from
  // My Portfolio. The only thing missing is a run.
  return { key: 'draft', label: 'READY', className: 'draft' };
}

function formatAgentMoney(value, { cents = true } = {}) {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: cents ? 2 : 0,
    maximumFractionDigits: cents ? 2 : 0,
  }).format(Number(value));
}

function formatSignedMoney(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  const body = formatAgentMoney(Math.abs(n));
  if (n > 0) return `+${body}`;
  if (n < 0) return `−${body}`;
  return body;
}

function formatRelativeTime(iso) {
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
  const fmt = (raw, withYear = false) => {
    if (!raw) return '';
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) {
      const s = String(raw);
      return s.length >= 10 ? s.slice(5, 10) : s;
    }
    return dt.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      ...(withYear ? { year: 'numeric' } : {}),
    });
  };
  const a = fmt(start);
  const b = fmt(end, true);
  if (a && b) return `${a} — ${b}`;
  return a || b || '—';
}

function agentRunCount(agent) {
  return Number(agent.run_count) || (Array.isArray(agent.runs) ? agent.runs.length : 0);
}

function renderAgentRunsLink(agent) {
  const count = agentRunCount(agent);
  const label = `${count} backtest${count === 1 ? '' : 's'}`;
  return `
    <button class="agent-card-runs-link agent-view-runs-btn" type="button" data-agent-id="${escapeHtml(agent.agent_id)}">
      <span class="agent-card-runs-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 19V5"/><path d="M4 19h16"/><path d="M7 15l3-3 3 2 5-6"/></svg>
      </span>
      <span>${escapeHtml(label)}</span>
      <span class="agent-card-runs-chevron" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6"/></svg>
      </span>
    </button>`;
}

function hashStringSeed(str) {
  let h = 0;
  const s = String(str || '');
  for (let i = 0; i < s.length; i += 1) h = (h * 31 + s.charCodeAt(i)) | 0;
  return Math.abs(h) || 1;
}

/** Render card sparkline from real equity samples (or a 2-point start→end fallback). */
function renderAgentSparklineFromValues(values, positive = true, seed = 'spark') {
  const nums = (Array.isArray(values) ? values : [])
    .map(Number)
    .filter((v) => Number.isFinite(v));
  const color = positive ? '#4ade80' : '#ff6b6b';
  const fillId = `agSpark-${hashStringSeed(seed)}`;
  const w = 80;
  const h = 36;
  const top = 4;
  const bottom = 4;
  const plotH = h - top - bottom;

  if (nums.length < 2) {
    return `
    <svg class="agent-card-sparkline agent-card-sparkline--empty" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
      <path d="M4,${(h / 2).toFixed(1)} H${w - 4}" fill="none" stroke="rgba(148,163,184,0.35)" stroke-width="1.5" stroke-linecap="round" stroke-dasharray="3 3"/>
    </svg>`;
  }

  const min = Math.min(...nums);
  const max = Math.max(...nums);
  // Keep tiny PnL readable without inventing fake volatility.
  const span = max - min;
  const pad = span > 0 ? span * 0.18 : Math.max(Math.abs(max) * 0.004, 1);
  const lo = min - pad;
  const hi = max + pad;
  const range = hi - lo || 1;
  const pts = nums.map((v, i) => {
    const x = (i / (nums.length - 1)) * w;
    const y = top + (1 - (v - lo) / range) * plotH;
    return [x, y];
  });
  const line = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const area = `${line} L${w},${h} L0,${h} Z`;
  return `
    <svg class="agent-card-sparkline" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
      <defs>
        <linearGradient id="${fillId}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${color}" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <path d="${area}" fill="url(#${fillId})"/>
      <path d="${line}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`;
}

function resolveAgentSparklineValues(agent, metrics = {}) {
  const fromAgent = agent?.equity_sparkline;
  if (Array.isArray(fromAgent) && fromAgent.length >= 2) return fromAgent;
  const fromRun = agent?.latest_run?.equity_sparkline;
  if (Array.isArray(fromRun) && fromRun.length >= 2) return fromRun;
  const ending = Number(metrics.ending);
  const pnl = Number(metrics.pnl);
  if (Number.isFinite(ending) && Number.isFinite(pnl)) {
    return [ending - pnl, ending];
  }
  return null;
}

function renderAgentSparkline(agent, positive = true, metrics = {}) {
  const values = resolveAgentSparklineValues(agent, metrics);
  const seed = agent?.agent_id || agent?.name || 'spark';
  return renderAgentSparklineFromValues(values, positive, seed);
}

/** Human-readable model label from provider paths like anthropic/claude-haiku-4-5. */
function formatAgentModelLabel(modelName) {
  const raw = String(modelName || '').trim();
  if (!raw) return 'Local model';
  const known = {
    'anthropic/claude-haiku-4-5': 'Claude Haiku 4.5',
    'anthropic/claude-sonnet-4-6': 'Claude Sonnet 4.6',
    'claude-haiku-4.5': 'Claude Haiku 4.5',
    'claude-sonnet-4.6': 'Claude Sonnet 4.6',
    'gpt-5.5': 'GPT-5.5',
    'openai/gpt-5.5': 'GPT-5.5',
    'deepseek/deepseek-v4-pro': 'DeepSeek V4 Pro',
    'deepseek-v4-pro': 'DeepSeek V4 Pro',
    'local-model': 'Local model',
    'rule-based': 'Rule-based',
    'rule-based-demo': 'Rule-based',
  };
  if (known[raw]) return known[raw];
  try {
    const escaped = (typeof CSS !== 'undefined' && CSS.escape) ? CSS.escape(raw) : raw.replace(/"/g, '\\"');
    const option = document.querySelector(`option[value="${escaped}"]`);
    if (option?.textContent?.trim() && option.textContent.trim() !== raw) {
      return option.textContent.trim();
    }
  } catch (_) { /* ignore selector errors */ }
  let label = raw.includes('/') ? raw.split('/').pop() : raw;
  label = label.replace(/[-_]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  label = label.replace(/\b(\d)\s+(\d)\b/g, '$1.$2');
  return label;
}

function catalogModelLabels() {
  return new Set([
    ...SUPPORTED_MODELS.map((model) => model.label),
    ...STARTER_AGENTS.map((spec) => spec.name),
  ]);
}

/** Card/editor title for a prompted-model agent.

 * If the stored name is already a catalog model label (or empty), it is bound
 * to the current model — so a Claude card cannot keep showing "DeepSeek V4 Pro"
 * after the model field moved. Custom titles ("My dip buyer") stay as stored.
 */
function agentDisplayName(agent) {
  const stored = String(agent?.name || '').trim();
  if ((agent?.agent_type || '') !== 'builtin') return stored || 'Agent';
  if ((agent?.runtime_type || 'pipeline') !== 'pipeline') return stored || 'Agent';
  const modelLabel = formatAgentModelLabel(agent?.model_name);
  if (!stored || stored === modelLabel || catalogModelLabels().has(stored)) {
    return modelLabel;
  }
  return stored;
}

function agentRobotIcon() {
  return `<span class="agent-card-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
      <rect x="5" y="9" width="14" height="10" rx="3"/>
      <path d="M12 5v4"/><circle cx="12" cy="4" r="1"/>
      <circle cx="9" cy="14" r="1.1" fill="currentColor" stroke="none"/>
      <circle cx="15" cy="14" r="1.1" fill="currentColor" stroke="none"/>
    </svg>
  </span>`;
}

function resolvePaperCardMetrics(agent) {
  const cash = Number(agent.cash_allocation);
  const fallback = Number.isFinite(cash) ? cash : 10000;
  const equity = Number(agent.paper_equity ?? agent.paper_portfolio_value);
  const hasLive = Number.isFinite(equity);
  const dayPnl = Number(agent.paper_day_pnl);
  const dayPnlPct = Number(agent.paper_day_pnl_pct);
  const buyingPower = Number(agent.paper_buying_power);
  const openPositions = Number(agent.paper_open_positions);
  return {
    equity: hasLive ? equity : fallback,
    dayPnl: Number.isFinite(dayPnl) ? dayPnl : null,
    dayPnlPct: Number.isFinite(dayPnlPct) ? dayPnlPct : null,
    buyingPower: Number.isFinite(buyingPower) ? buyingPower : fallback,
    openPositions: Number.isFinite(openPositions) ? openPositions : 0,
    lastActivity: agent.paper_last_activity || null,
    updatedAt: agent.paper_updated_at || null,
    hasLive,
  };
}

/** Most recent backtest run on an agent card (prefers latest_run, else runs[]). */
function resolveLatestAgentRun(agent) {
  const latest = agent?.latest_run;
  if (latest && (latest.run_id || latest.total_return != null || latest.final_equity != null)) {
    return latest;
  }
  const runs = Array.isArray(agent?.runs) ? agent.runs : [];
  if (!runs.length) return null;
  return [...runs].sort(
    (a, b) => String(b.created_at || '').localeCompare(String(a.created_at || '')),
  )[0];
}

function resolveLatestAgentRunId(agent) {
  const run = resolveLatestAgentRun(agent);
  return run?.run_id || null;
}

function resolveBacktestCardMetrics(agent) {
  const run = resolveLatestAgentRun(agent);
  const cash = Number(agent.cash_allocation);
  const initial = Number(run?.initial_equity);
  const startEquity = Number.isFinite(initial)
    ? initial
    : Number.isFinite(cash)
      ? cash
      : 10000;
  const final = Number(run?.final_equity);
  let ending = Number.isFinite(final) ? final : null;
  const retRaw = Number(run?.total_return);
  const retFrac = Number.isFinite(retRaw)
    ? Math.abs(retRaw) <= 1
      ? retRaw
      : retRaw / 100
    : null;
  if (ending == null && retFrac != null) ending = startEquity * (1 + retFrac);
  if (ending == null) ending = startEquity;
  const pnl = ending - startEquity;
  const returnPct = retFrac != null
    ? retFrac
    : (startEquity ? pnl / startEquity : 0);
  return {
    ending,
    pnl,
    returnPct,
    positive: pnl >= 0,
    period: formatShortDateRange(run?.start_date, run?.end_date),
    universe: run?.universe || run?.index || 'DJIA',
    createdAt: run?.created_at || null,
    runId: run?.run_id || null,
  };
}

function formatSignedReturnPct(frac) {
  const n = Number(frac);
  if (!Number.isFinite(n)) return '—';
  const pct = n * 100;
  const body = `${Math.abs(pct).toFixed(1)}%`;
  if (pct > 0) return `+${body}`;
  if (pct < 0) return `−${body}`;
  return body;
}

/**
 * Saved simulated capital for an agent's backtests.
 *
 * Mirrors the backend fallback chain exactly: an agent created before
 * `backtest_allocation` existed has a NULL column and must keep behaving as it
 * did, i.e. starting from its paper sleeve.
 */
function resolveBacktestCapital(agent) {
  const candidates = [agent?.backtest_allocation, agent?.cash_allocation];
  for (const raw of candidates) {
    const value = Number(raw);
    if (Number.isFinite(value) && value > 0) {
      return Math.min(Math.round(value), MAX_BACKTEST_ALLOCATED_CAPITAL);
    }
  }
  return DEFAULT_AGENT_CASH_ALLOCATION;
}

/** Shared top block: both capital figures, equal weight (draft + backtested). */
function renderAgentAllocatedCapitalHero(agent) {
  const paper =
    agent.cash_allocation != null
      ? formatAgentCashAllocation(agent.cash_allocation)
      : '$1,000';
  const backtest = formatAgentCashAllocation(resolveBacktestCapital(agent));
  return `
    <div class="agent-card-capitals">
      <div class="agent-card-capital">
        <span class="agent-card-metric-label">Paper Trading</span>
        <p class="agent-card-metric-value">${escapeHtml(paper)}</p>
        <p class="agent-card-capital-note">From My Portfolio</p>
      </div>
      <div class="agent-card-capital">
        <span class="agent-card-metric-label">Backtesting</span>
        <p class="agent-card-metric-value">${escapeHtml(backtest)}</p>
        <p class="agent-card-capital-note">Simulated</p>
      </div>
    </div>`;
}

/**
 * Card body for an agent with a backtest in flight.
 *
 * The bar is determinate whenever the engine has published a step: engine.py's
 * `_publish_live_progress` writes step/total_steps every step and the status
 * endpoint surfaces them. (The 2026-07-29 spec specified an indeterminate bar
 * "since no honest completion estimate exists" -- that was already untrue; see
 * the 2026-08-01 spec.) It falls back to indeterminate before the first step,
 * which is a normal state on every run, not an error.
 */
function renderAgentRunningBody(agent, running) {
  // Every value below comes from deriveRunningProgress, which the per-second
  // patch path reads too -- see refreshRunningAgentCards().
  const view = deriveRunningProgress(running);
  // Every dynamic node carries a data-running-* hook, including the ones that
  // are empty right now: the patch path finds nodes by attribute, and a node
  // rendered only when it has content can never be filled in later.
  const id = escapeHtml(agent.agent_id);

  return `
    <div class="agent-card-running">
      <div class="agent-card-running-head">
        <span class="agent-card-running-dot" aria-hidden="true"></span>
        <span class="agent-card-running-label">Backtesting…</span>
        <span class="agent-card-running-step" data-running-step="${id}">${escapeHtml(view.stepLabel)}</span>
        <span class="agent-card-running-elapsed" data-running-elapsed="${id}">${escapeHtml(formatBacktestElapsed(running.elapsedSeconds))}</span>
      </div>
      <div class="agent-card-running-track" role="progressbar" aria-label="Backtest in progress" data-running-track="${id}"${view.determinate ? ` aria-valuenow="${view.pct}" aria-valuemin="0" aria-valuemax="100"` : ''}>
        <div class="agent-card-running-bar${view.determinate ? ' is-determinate' : ''}" data-running-bar="${id}"${view.determinate ? ` style="width: ${view.pct}%"` : ''}></div>
      </div>
      <p class="agent-card-running-detail" data-running-detail="${id}">${escapeHtml(view.detail)}</p>
      <p class="agent-card-running-stale" data-running-stale="${id}">${escapeHtml(view.notice)}</p>
    </div>
    ${renderAgentAllocatedCapitalHero(agent)}`;
}

function renderAgentCardBody(agent, statusKey) {
  if (statusKey === 'paper') {
    const m = resolvePaperCardMetrics(agent);
    const positive = m.dayPnl == null ? true : m.dayPnl >= 0;
    let changeHtml = '';
    if (m.dayPnl != null) {
      const pct =
        m.dayPnlPct != null
          ? ` (${m.dayPnlPct >= 0 ? '+' : ''}${m.dayPnlPct.toFixed(2)}%)`
          : '';
      changeHtml = `<p class="agent-card-change ${positive ? 'is-pos' : 'is-neg'}">${escapeHtml(formatSignedMoney(m.dayPnl))}${escapeHtml(pct)} today</p>`;
    } else if (!m.hasLive) {
      changeHtml = `<p class="agent-card-change is-muted">Paper Trading Allocated Capital · session not live yet</p>`;
    }
    const activity = m.lastActivity
      ? escapeHtml(m.lastActivity)
      : m.hasLive
        ? 'Paper trading active'
        : 'Ready for paper trading';
    const updated = m.updatedAt
      ? `Updated ${formatRelativeTime(m.updatedAt)}`
      : '';
    return `
      <div class="agent-card-hero">
        <div class="agent-card-hero-text">
          <div class="agent-card-metric-head">
            <span class="agent-card-mode-chip">PAPER</span>
            <span class="agent-card-metric-label">Portfolio Value</span>
          </div>
          <p class="agent-card-metric-value">${escapeHtml(formatAgentMoney(m.equity))}</p>
          ${changeHtml}
        </div>
        ${renderAgentSparkline(agent, positive, { ending: m.equity, pnl: m.dayPnl ?? 0 })}
      </div>
      <div class="agent-card-divider"></div>
      <div class="agent-card-stats">
        <div class="agent-card-stat">
          <span class="agent-card-stat-label">Buying Power</span>
          <span class="agent-card-stat-value">${escapeHtml(formatAgentMoney(m.buyingPower, { cents: false }))}</span>
        </div>
        <div class="agent-card-stat">
          <span class="agent-card-stat-label">Open Positions</span>
          <span class="agent-card-stat-value">${escapeHtml(String(m.openPositions))}</span>
        </div>
      </div>
      ${renderAgentRunsLink(agent)}
      <div class="agent-card-divider"></div>
      <div class="agent-card-activity">
        <span class="agent-card-activity-icon agent-card-activity-icon--buy" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="9" cy="20" r="1"/><circle cx="17" cy="20" r="1"/><path d="M3 4h2l2.4 11.2a2 2 0 0 0 2 1.6h7.4a2 2 0 0 0 2-1.5L21 8H7"/></svg>
        </span>
        <div class="agent-card-activity-text">
          <span>${activity}</span>
          ${updated ? `<span class="agent-card-activity-sub">${escapeHtml(updated)}</span>` : ''}
        </div>
      </div>`;
  }

  if (statusKey === 'backtested') {
    const m = resolveBacktestCardMetrics(agent);
    const endingLabel = formatAgentMoney(m.ending, { cents: false });
    const metaParts = [`Ending Value ${endingLabel}`];
    if (m.period && m.period !== '—') metaParts.push(m.period);
    return `
      ${renderAgentAllocatedCapitalHero(agent)}
      <div class="agent-card-divider"></div>
      <div class="agent-card-latest">
        <div class="agent-card-latest-head">
          <span class="agent-card-metric-label">Latest Backtest</span>
          <span class="agent-card-mode-chip agent-card-mode-chip--simulation">Simulation</span>
        </div>
        <div class="agent-card-latest-row">
          <p class="agent-card-latest-return ${m.positive ? 'is-pos' : 'is-neg'}">${escapeHtml(formatSignedReturnPct(m.returnPct))}</p>
          ${renderAgentSparkline(agent, m.positive, m)}
        </div>
        <p class="agent-card-latest-meta">${escapeHtml(metaParts.join(' · '))}</p>
        ${renderAgentRunsLink(agent)}
      </div>`;
  }

  return `
    ${renderAgentAllocatedCapitalHero(agent)}
    <div class="agent-card-empty">
      <span class="agent-card-empty-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 16l4-4 3 3 5-6 4 4"/><path d="M4 20h16"/></svg>
      </span>
      <strong>No backtests yet</strong>
      <span>Test this agent on historical market data.</span>
    </div>`;
}

function renderAgentCardActions(agent, statusKey) {
  const id = escapeHtml(agent.agent_id);
  let primary = '';
  if (statusKey === 'paper') {
    primary = `<button class="agent-card-cta agent-open-btn" type="button" data-agent-id="${id}">Open Agent</button>`;
  } else {
    // Paper trading is Phase B (execution/paper_backend.py is a stub). Ship the
    // affordance disabled *with a reason* -- an unexplained grey button reads as
    // a bug, and its absence hides that the two capital figures above map onto
    // two different things you can eventually run.
    primary = `
      <button class="agent-card-cta agent-run-backtest-btn" type="button" data-agent-id="${id}">Run Backtest</button>
      <button class="agent-card-cta agent-card-cta--disabled" type="button" disabled aria-disabled="true" title="Paper trading is coming soon" aria-label="Run Paper Trading — Paper trading is coming soon">Run Paper Trading</button>`;
  }
  const configure = `<button class="agent-card-cta agent-card-cta--configure agent-configure-btn" type="button" data-agent-id="${id}">Configure</button>`;
  const rotate =
    agent.agent_type === 'builtin'
      ? ''
      : `<button class="agent-menu-item agent-rotate-key-btn" type="button" data-agent-id="${id}">New access key</button>`;
  // Only once the user has actually run this agent: "try it on another model"
  // is a follow-on offer, not a first action. Built-in only -- duplicating an
  // external agent would mint an API key (see the backend's duplicate route).
  // Also excludes hosted runtimes (runtime_type !== 'pipeline'): ai_hedge_fund
  // hardcodes its own model and never reads the stored value, so duplicating
  // it onto a chosen model would display a model that isn't actually running.
  // runtime_type is always present and truthy (server-defaulted to
  // 'pipeline'), so this MUST be an equality check, never a truthiness test.
  const duplicate =
    agent.agent_type === 'builtin' &&
    agent.runtime_type === 'pipeline' &&
    (statusKey === 'backtested' || statusKey === 'paper')
      ? `<button class="agent-menu-item agent-duplicate-model-btn" type="button" data-agent-id="${id}">Run on another model</button>`
      : '';
  return `
    <div class="agent-card-actions agent-card-actions--status">
      ${configure}
      ${primary}
      <div class="agent-card-menu">
        <button class="agent-menu-toggle" type="button" aria-label="More actions" aria-expanded="false" data-agent-id="${id}">
          <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="6" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="18" cy="12" r="1.6"/></svg>
        </button>
        <div class="agent-menu-dropdown" hidden>
          <button class="agent-menu-item agent-set-default-btn" type="button" data-agent-id="${id}">Set as default</button>
          ${rotate}
          ${duplicate}
          <button class="agent-menu-item agent-menu-item--danger agent-delete-btn" type="button" data-agent-id="${id}">Delete</button>
        </div>
      </div>
    </div>`;
}

/** SUPPORTED_MODELS minus the agent's current model -- an entry that duplicates
 * an agent onto the model it already runs is a no-op the user has to reason
 * about. A legacy or hosted-runtime model isn't in the list, so nothing is
 * filtered out and the full six are offered. */
function duplicateModelChoices(agent) {
  const current = String(agent?.model_name || '').trim().toLowerCase();
  return SUPPORTED_MODELS.filter((model) => model.slug.toLowerCase() !== current).map(
    (model) => ({ slug: model.slug, label: model.label }),
  );
}

/** "Momentum Alpha (DeepSeek)". Collides freely: two copies onto DeepSeek read
 * the same. Names are not unique anywhere else in this product, and
 * de-duplicating would mean a lookup for a cosmetic gain. */
function duplicateAgentName(agent, modelSlug) {
  const vendor = MODEL_VENDORS.find((entry) => entry.key === modelVendorKey(modelSlug));
  const suffix = ` (${vendor?.label || 'new model'})`;
  // 100 mirrors DuplicateAgentBody.name's max_length and both name inputs'
  // maxlength. A 95-char agent would otherwise generate an over-length copy,
  // and API.request JSON.stringify's the non-string 422 `detail`, so the raw
  // Pydantic array renders in the modal's error line. Inlined rather than a
  // module constant because the guards lift this function body into node on
  // its own -- an outside reference would be undefined there.
  // Trim the base, never the suffix: the vendor is the point of the name.
  const base = String(agent?.name || 'Agent').slice(0, 100 - suffix.length).trimEnd();
  return `${base}${suffix}`;
}

let duplicateAgentSource = null;

function openDuplicateAgentModal(agent) {
  const modal = document.getElementById('duplicateAgentModal');
  const select = document.getElementById('duplicateAgentModel');
  const error = document.getElementById('duplicateAgentError');
  if (!modal || !select || !agent) return;
  duplicateAgentSource = agent;
  select.innerHTML = duplicateModelChoices(agent)
    .map((model) => `<option value="${escapeHtml(model.slug)}">${escapeHtml(model.label)}</option>`)
    .join('');
  if (error) { error.hidden = true; error.textContent = ''; }
  modal.hidden = false;
}

function closeDuplicateAgentModal() {
  const modal = document.getElementById('duplicateAgentModal');
  if (modal) modal.hidden = true;
  duplicateAgentSource = null;
}

/** Lands the user on the new agent with Run primed. Deliberately does NOT start
 * a backtest: auto-firing would spend LLM credits on a click the user framed as
 * "make a copy". */
async function submitDuplicateAgent() {
  const agent = duplicateAgentSource;
  const select = document.getElementById('duplicateAgentModel');
  const error = document.getElementById('duplicateAgentError');
  const submit = document.getElementById('duplicateAgentSubmit');
  if (!agent || !select?.value) return;
  if (submit) submit.disabled = true;
  try {
    const data = await API.post(
      `${API_BASE}/api/v1/agents/${encodeURIComponent(agent.agent_id)}/duplicate`,
      { model_name: select.value, name: duplicateAgentName(agent, select.value) },
    );
    const created = data?.agent;
    if (!created?.agent_id) throw new Error('Copy failed — no agent returned');
    closeDuplicateAgentModal();
    applyActiveAgent(created);
    await loadAgents();
    showAppToast(`${created.name} is ready. Press Run Backtest to compare them.`);
    highlightAgentCard(created.agent_id);
  } catch (err) {
    if (error) {
      error.textContent = err.message || `Couldn't create the copy. Please try again.`;
      error.hidden = false;
    }
  } finally {
    if (submit) submit.disabled = false;
  }
}

function renderAgentRunningActions(agent) {
  const id = escapeHtml(agent.agent_id);
  // Configure stays available while a backtest runs: the request that started
  // the job carried its own copy of the pipeline, model and window, so a later
  // save in the editor cannot reach, change or cancel it — and the run declines
  // its own pipeline write-back when the agent changed while it was in flight
  // (_maybe_writeback_adapted_pipeline, api/routers/backtests.py), so the edit
  // does not lose a race against the run either.
  //
  // The card's Run button is replaced by this status pill, but that is NOT a
  // lock on launching — the editor's own "Run Backtest" reaches the same modal
  // through window.openRunBacktestModal (js/agent-editor.js). Starting another
  // run is deliberately allowed (the dashboard runner takes several concurrent
  // backtests per owner); openRunBacktestModal() is the single funnel both
  // buttons go through, and it refuses the click once this browser is at the
  // limit instead of firing a request the server would reject.
  return `
    <div class="agent-card-actions agent-card-actions--status">
      <button class="agent-card-cta agent-card-cta--configure agent-configure-btn" type="button" data-agent-id="${id}">Configure</button>
      <button class="agent-card-cta agent-card-cta--disabled" type="button" disabled aria-disabled="true">Running…</button>
    </div>`;
}

// Demo/mock agents (MOCK_AGENTS) have no database row, so renames made in the editor
// are stored locally under `agent-name-override:{id}`. Real agents use the same key
// only when a server PATCH fails, so the edited name still shows in the UI.
function applyAgentNameOverride(agent) {
  if (!agent || !agent.agent_id) return agent;
  try {
    const raw = localStorage.getItem(`agent-name-override:${agent.agent_id}`);
    if (!raw) return agent;
    const override = JSON.parse(raw);
    return {
      ...agent,
      name: override.name || agent.name,
      description: override.description ?? agent.description,
    };
  } catch (e) {
    return agent;
  }
}

function getFilteredAgents() {
  const query = (document.getElementById('agentSearchInput')?.value || '').trim().toLowerCase();
  let list = allAgents.map(decorateAgent);
  if (query) {
    list = list.filter(
      (a) =>
        String(a.name || '').toLowerCase().includes(query) ||
        String(a.model_name || '').toLowerCase().includes(query),
    );
  }
  return list;
}

function applyAgentFilters(resetPagination = true) {
  if (resetPagination) {
    agentGridPage = Object.fromEntries(AGENT_SHELVES.map((shelf) => [shelf.key, 0]));
  }
  renderAgentCategories(getFilteredAgents());
}

function setAgentViewMode(mode) {
  agentViewMode = mode === 'list' ? 'list' : 'grid';
  document.querySelectorAll('.agents-section .agents-grid').forEach((grid) => {
    grid.classList.toggle('agents-grid--list', agentViewMode === 'list');
  });
  document.getElementById('agentViewGrid')?.classList.toggle('active', agentViewMode === 'grid');
  document.getElementById('agentViewList')?.classList.toggle('active', agentViewMode === 'list');
}

function isDemoAgent(agentId) {
  return typeof agentId === 'string' && agentId.startsWith('mock-');
}

function getHiddenDemoAgentIds() {
  try {
    const raw = localStorage.getItem(HIDDEN_DEMO_AGENTS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function hideDemoAgent(agentId) {
  const hidden = getHiddenDemoAgentIds();
  if (!hidden.includes(agentId)) {
    hidden.push(agentId);
    localStorage.setItem(HIDDEN_DEMO_AGENTS_KEY, JSON.stringify(hidden));
  }
}

function visibleMockAgents() {
  const hidden = new Set(getHiddenDemoAgentIds());
  return MOCK_AGENTS.filter((agent) => !hidden.has(agent.agent_id));
}

// Demo mode is opt-in via ?demo=1 so local development does not show fake agents
// that cannot be deleted from the database.
function isDemoMode() {
  try {
    const params = new URLSearchParams(window.location.search);
    return params.get('demo') === '1';
  } catch (e) {
    return false;
  }
}

// Distinct error-state shown when the agents API is unreachable — never mask a
// backend outage by rendering fake data.
function renderAgentsError() {
  const errorEl = document.getElementById('agentsErrorState');
  document.querySelectorAll('.agents-section .agents-grid').forEach((grid) => {
    grid.innerHTML = '';
  });
  AGENT_SHELVES.forEach((shelf) => {
    const suffix = shelfIdSuffix(shelf.key);
    const footer = document.getElementById(`agentsGridFooter${suffix}`);
    if (footer) {
      footer.hidden = true;
      footer.innerHTML = '';
    }
    const emptyEl = document.getElementById(`agentsEmpty${suffix}`);
    if (emptyEl) emptyEl.hidden = true;
  });
  if (errorEl) errorEl.hidden = false;
}

async function openAgentInBacktest(agent, runId = null) {
  if (!agent) return;
  // Navigate immediately — never block on the activate ping (cold API starts
  // left the user stuck on My Agents). Pin the latest run after session switch.
  applyActiveAgent(agent);
  const resolvedRunId = runId || resolveLatestAgentRunId(agent);
  if (resolvedRunId) {
    localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, resolvedRunId);
  } else {
    localStorage.removeItem(SELECTED_BACKTEST_RUN_KEY);
  }
  navigateToPage('playground', { playgroundTab: 'backtest' });
  currentMode = 'backtest';
  // applyActiveAgent ran once above for immediate navigation; activateAgent
  // re-applies it (idempotent) and fires the server ping we deliberately did
  // NOT await. The same-session re-apply must not clear SELECTED_BACKTEST_RUN_KEY
  // (see applyActiveAgent's previousSession guard).
  activateAgent(agent);
  await loadData();
}

async function openAgentInPaper(agent) {
  if (!agent) return;
  // Navigate immediately; activateAgent below re-applies (idempotent) and pings
  // the server fire-and-forget, so a cold API start never blocks the UI.
  applyActiveAgent(agent);
  navigateToPage('playground', { playgroundTab: 'paper' });
  currentMode = 'paper';
  activateAgent(agent);
  await loadData();
}

function bindAgentCardMenus(grid) {
  grid.querySelectorAll('.agent-menu-toggle').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      event.stopPropagation();
      const menu = btn.closest('.agent-card-menu');
      const dropdown = menu?.querySelector('.agent-menu-dropdown');
      if (!dropdown) return;
      const willOpen = dropdown.hidden;
      grid.querySelectorAll('.agent-menu-dropdown').forEach((el) => {
        el.hidden = true;
      });
      grid.querySelectorAll('.agent-menu-toggle').forEach((el) => {
        el.setAttribute('aria-expanded', 'false');
      });
      dropdown.hidden = !willOpen;
      btn.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
    });
  });
}

function renderAgentGridFooter(categoryKey, total, page, pageCount) {
  const footerId = `agentsGridFooter${shelfIdSuffix(categoryKey)}`;
  const footer = document.getElementById(footerId);
  if (!footer) return;
  if (pageCount <= 1) {
    footer.hidden = true;
    footer.innerHTML = '';
    return;
  }
  footer.hidden = false;
  const atStart = page <= 0;
  const atEnd = page >= pageCount - 1;
  footer.innerHTML = `
    <button type="button" class="agents-grid-footer-btn agents-grid-footer-btn--nav" data-agent-grid-prev="${categoryKey}" aria-label="Previous page" ${atStart ? 'disabled' : ''}>←</button>
    <span class="agents-grid-footer-count">Page ${page + 1} of ${pageCount} · ${total} total</span>
    <button type="button" class="agents-grid-footer-btn agents-grid-footer-btn--nav" data-agent-grid-next="${categoryKey}" aria-label="Next page" ${atEnd ? 'disabled' : ''}>→</button>`;
}

function renderAgentCards(grid, agents, categoryKey) {
  grid.innerHTML = '';
  const total = agents.length;
  const pageCount = agentGridPageCount(total);
  const page = normalizeAgentGridPage(categoryKey, total);
  const start = page * AGENT_GRID_PAGE_SIZE;
  const visibleAgents = agents.slice(start, start + AGENT_GRID_PAGE_SIZE);

  const defaultId = getDefaultAgentId();

  visibleAgents.forEach((agent) => {
    const isBuiltin = agent.agent_type === 'builtin';
    const statusBadge = resolveAgentStatusBadge(agent);
    const card = document.createElement('div');
    card.className = `section-card agent-card agent-card--status agent-card--${statusBadge.key}${isBuiltin ? ' agent-card-builtin' : ''}`;
    card.setAttribute('data-agent-id', agent.agent_id);
    // Title already names the model. Decision-type copy repeated it. Under
    // the All chip this shelf still mixes markets, so keep that when known.
    const market = MARKET_LABELS[agentMarketKey(agent)];
    const submeta = market || '';
    const running = getAgentBacktestRunning(agent.agent_id);
    if (running) card.classList.add('agent-card--running');

    card.innerHTML = `
      <div class="agent-card-top">
        <div class="agent-card-identity">
          ${agentRobotIcon()}
          <div class="agent-card-identity-text">
            <h3 class="agent-name">${escapeHtml(agentDisplayName(agent))}${agent.agent_id === defaultId ? ' <span class="agent-default-badge">Default</span>' : ''}</h3>
            ${submeta ? `<p class="agent-card-submeta" title="${escapeHtml(submeta)}">${escapeHtml(submeta)}</p>` : ''}
          </div>
        </div>
        <span class="status-badge ${statusBadge.className}"><span class="status-badge-dot" aria-hidden="true"></span>${statusBadge.label}</span>
      </div>
      ${running ? renderAgentRunningBody(agent, running) : renderAgentCardBody(agent, statusBadge.key)}
      ${running ? renderAgentRunningActions(agent) : renderAgentCardActions(agent, statusBadge.key)}
    `;
    const identity = card.querySelector('.agent-card-identity');
    if (identity) {
      identity.setAttribute('role', 'button');
      identity.setAttribute('tabindex', '0');
      identity.setAttribute('title', 'Open to edit instructions');
      const openEditor = (event) => {
        event.preventDefault();
        if (!window.AgentEditor) return;
        navigateToPage('playground', { playgroundTab: 'agents' });
        showPlaygroundPanel('agents');
        window.AgentEditor.open(agent);
      };
      identity.addEventListener('click', openEditor);
      identity.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') openEditor(event);
      });
    }
    grid.appendChild(card);
  });

  bindAgentCardMenus(grid);

  renderAgentGridFooter(categoryKey, total, page, pageCount);

  grid.querySelectorAll('.agent-configure-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const agent = visibleAgents.find((a) => a.agent_id === btn.dataset.agentId);
      if (!agent || !window.AgentEditor) return;
      navigateToPage('playground', { playgroundTab: 'agents' });
      showPlaygroundPanel('agents');
      window.AgentEditor.open(agent);
    });
  });

  grid.querySelectorAll('.agent-set-default-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      setDefaultAgentId(btn.dataset.agentId);
      applyAgentFilters(); // re-render: badge + pin move to the new default
    });
  });

  grid.querySelectorAll('.agent-open-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const agent = visibleAgents.find((a) => a.agent_id === btn.dataset.agentId);
      await openAgentInPaper(agent);
    });
  });

  grid.querySelectorAll('.agent-view-runs-btn').forEach((btn) => {
    btn.addEventListener('click', async (event) => {
      event.preventDefault();
      event.stopPropagation();
      const agent =
        agents.find((a) => a.agent_id === btn.dataset.agentId) ||
        allAgents.find((a) => a.agent_id === btn.dataset.agentId);
      if (!agent) {
        console.warn('View runs: agent not found', btn.dataset.agentId);
        return;
      }
      await openAgentInBacktest(agent, resolveLatestAgentRunId(agent));
    });
  });

  grid.querySelectorAll('.agent-run-backtest-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const agent = visibleAgents.find((a) => a.agent_id === btn.dataset.agentId);
      if (!agent) return;
      openRunBacktestModal(agent);
    });
  });

  grid.querySelectorAll('.agent-rotate-key-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const agent = visibleAgents.find((a) => a.agent_id === btn.dataset.agentId);
      if (!agent) return;
      if (!confirm(`Create a new access key for "${agent.name}"? The current key stops working right away — any connected program must switch to the new key.`)) {
        return;
      }
      btn.disabled = true;
      try {
        await rotateAgentApiKey(agent);
      } catch (error) {
        alert(error.message || `Couldn't create a new access key. Please try again.`);
      } finally {
        btn.disabled = false;
      }
    });
  });

  grid.querySelectorAll('.agent-duplicate-model-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const agent = visibleAgents.find((a) => a.agent_id === btn.dataset.agentId);
      if (!agent) return;
      openDuplicateAgentModal(agent);
    });
  });

  grid.querySelectorAll('.agent-delete-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const agentId = btn.dataset.agentId;
      if (!agentId || !confirm('Delete this agent? Backtest history stays in the database.')) return;
      try {
        if (isDemoAgent(agentId)) {
          hideDemoAgent(agentId);
          if (localStorage.getItem(ACTIVE_AGENT_KEY) === agentId) {
            localStorage.removeItem(ACTIVE_AGENT_KEY);
            localStorage.removeItem(ACTIVE_AGENT_NAME_KEY);
          }
          await loadAgents();
          return;
        }
        await API.request(`${API_BASE}/api/v1/agents/${agentId}`, { method: 'DELETE' });
        if (localStorage.getItem(ACTIVE_AGENT_KEY) === agentId) {
          localStorage.removeItem(ACTIVE_AGENT_KEY);
          localStorage.removeItem(ACTIVE_AGENT_NAME_KEY);
        }
        await loadAgents();
      } catch (error) {
        alert(error.message || `Couldn't delete the agent. Please try again.`);
      }
    });
  });
}

// Empty-state HTML for the Prompted Models shelf. Three cases, deliberately
// worded apart: a live search hiding everything, a market chip with nothing on
// it yet, and a genuinely empty shelf. Collapsing them would tell a searching
// or filtering user they own no agents. Prompted Models is the onboarding
// surface (the auto-provisioned DeepSeek card lands here), so the true-empty
// case keeps the create-your-first voice rather than the Community-upsell voice
// used by Open Agents.
//
// External renders a placeholder CARD instead (renderExternalPlaceholderCard),
// so it has no entry here.
function promptedEmptyHtml({ searching, marketFilter }) {
  if (searching) return 'No agents match your search.';
  if (marketFilter !== 'all') {
    const label = escapeHtml(MARKET_LABELS[marketFilter] || '');
    return `No ${label} agents yet. Add a ready-made ${label} strategy from ${communityShelfButtonHtml(marketFilter)}.`;
  }
  return `You don't have any agents yet. Create one and test your first trading idea, or browse ready-made strategies in ${communityShelfButtonHtml('all')}.`;
}

function openAgentsEmptyHtml({ searching }) {
  if (searching) return 'No agents match your search.';
  return `No open agents yet. Add a ready-made strategy like AI Hedge Fund from ${communityShelfButtonHtml('all')}.`;
}

// A real <button>, not an <a href="#">: this is the primary path off an empty
// shelf, and as an anchor it matched no CSS rule anywhere in styles.css, so it
// inherited plain link styling and did not read as actionable.
//
// data-community-category is read by #agentsCategories' delegated click
// handler, which routes it through navigateToPage's options so the matching
// Community chip is pre-selected. 'all' is a valid value there -- navigateToPage
// falls it back to 'all' because it isn't a MARKET_LABELS key.
function communityShelfButtonHtml(category) {
  return `<button type="button" class="agents-empty-community-btn" data-community-category="${escapeHtml(category)}">Community</button>`;
}

/** The Prompted Models shelf's market filter row: 'All' plus one chip per
 * MARKET_LABELS entry, reusing the Community chip classes so the same taxonomy
 * looks the same on both surfaces.
 *
 * Built once, then only toggled. This runs from renderAgentCategories, which is
 * bound to the search box's `input` event -- rebuilding innerHTML per keystroke
 * would blow away the focused chip on every character typed. */
function renderAgentMarketChips() {
  const container = document.getElementById('agentsMarketChips');
  if (!container) return;
  const chips = [
    { key: 'all', label: 'All' },
    ...Object.entries(MARKET_LABELS).map(([key, label]) => ({ key, label })),
  ];
  const existing = container.querySelectorAll('[data-agent-market]');
  if (existing.length !== chips.length) {
    container.innerHTML = chips
      .map((chip) => `<button type="button" class="marketplace-category-chip" data-agent-market="${escapeHtml(chip.key)}" aria-pressed="false">${escapeHtml(chip.label)}</button>`)
      .join('');
  }
  container.querySelectorAll('[data-agent-market]').forEach((button) => {
    const active = button.dataset.agentMarket === agentMarketFilter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

/** Select a market chip and re-render. Resets pagination: the page index is
 * per-shelf, so a page-3 position under 'All' would land past the end of a
 * narrower market's single page -- an empty grid under a "Page 3 of 1" footer.
 * An unrecognized market falls back to 'all' rather than filtering to a chip
 * that doesn't exist. */
function setAgentMarketFilter(market) {
  agentMarketFilter = MARKET_LABELS[market] ? market : 'all';
  applyAgentFilters();
}

function renderAgentCategories(agents) {
  const errorEl = document.getElementById('agentsErrorState');
  const shelves = AGENT_SHELVES.map((shelf) => {
    const suffix = shelfIdSuffix(shelf.key);
    return {
      shelf,
      grid: document.getElementById(`agentsGrid${suffix}`),
      emptyEl: document.getElementById(`agentsEmpty${suffix}`),
      countEl: document.getElementById(`agentsCount${suffix}`),
    };
  });
  if (shelves.some(({ grid }) => !grid)) return;

  if (errorEl) errorEl.hidden = true; // a successful render clears any prior error

  renderAgentMarketChips();

  const defaultId = getDefaultAgentId();
  const pinDefaultFirst = (list) =>
    [...list].sort((a, b) => (b.agent_id === defaultId) - (a.agent_id === defaultId));

  // A live search narrows every shelf: distinguish "no agents at all"
  // (onboarding / Community upsell) from "none match your search" so we
  // never mis-say a shelf is empty when a search term is just hiding its
  // agents, and never surface the External onboarding card as a search result.
  const searching = !!(document.getElementById('agentSearchInput')?.value || '').trim();

  shelves.forEach(({ shelf, grid, emptyEl, countEl }) => {
    // The pill counts what the shelf HOLDS, read from the unfiltered roster --
    // not what is currently on screen. A number that moved while you typed or
    // clicked a chip would read as agents disappearing.
    if (countEl) {
      const held = allAgents.filter(shelf.match).length;
      countEl.hidden = held === 0;
      countEl.textContent = `${held} agent${held === 1 ? '' : 's'}`;
    }

    let matched = pinDefaultFirst(agents.filter(shelf.match));
    if (shelf.key === 'prompted' && agentMarketFilter !== 'all') {
      // agentMarketKey returns '' for a NULL/blank/unknown category, so those
      // agents match no chip and appear under All only -- visible, but never
      // filed under a market the platform can't actually vouch for.
      matched = matched.filter((a) => agentMarketKey(a) === agentMarketFilter);
    }
    renderAgentCards(grid, matched, shelf.key);

    if (shelf.key === 'external') {
      if (matched.length > 0) {
        if (emptyEl) emptyEl.hidden = true;
      } else if (searching) {
        if (emptyEl) {
          emptyEl.hidden = false;
          emptyEl.textContent = 'No agents match your search.';
        }
      } else {
        if (emptyEl) emptyEl.hidden = true;
        renderExternalPlaceholderCard(grid);
      }
      return;
    }

    if (!emptyEl) return;
    emptyEl.hidden = matched.length > 0;
    if (matched.length === 0) {
      emptyEl.innerHTML = shelf.key === 'open'
        ? openAgentsEmptyHtml({ searching })
        : promptedEmptyHtml({ searching, marketFilter: agentMarketFilter });
    }
  });
}

// Reserved entry point for connect-your-own agents: the connection mechanism
// is still an open team decision, so this opens the existing creation flow.
function renderExternalPlaceholderCard(grid) {
  const card = document.createElement('div');
  card.className = 'section-card agent-card agent-card--placeholder';
  card.innerHTML = `
    <div class="agent-card-identity-text">
      <h3 class="agent-name">Connect your own trading program</h3>
      <p class="agent-card-submeta">For developers: run your own trading program against our backtests using an access key.</p>
    </div>
    <button class="agent-card-cta agent-card-cta--outline" type="button">Connect agent</button>`;
  card.querySelector('button')?.addEventListener('click', openCreateExternalAgentModal);
  grid.appendChild(card);
}

document.addEventListener('click', (event) => {
  if (event.target.closest?.('.agent-card-menu')) return;
  document.querySelectorAll('.agents-grid .agent-menu-dropdown').forEach((el) => {
    el.hidden = true;
  });
  document.querySelectorAll('.agents-grid .agent-menu-toggle').forEach((el) => {
    el.setAttribute('aria-expanded', 'false');
  });
});

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderAgentTokenCost(agent) {
  const totalTokens =
    Number(agent.total_input_tokens || 0) + Number(agent.total_output_tokens || 0);
  if (!totalTokens) return '';
  const cost = formatUsd(agent.total_est_cost_usd);
  const costLabel = cost ? `${cost} est. AI cost` : '';
  return `<span title="Estimated from market context served and decisions returned">${formatTokenCount(totalTokens)} tokens${costLabel ? ` · ${costLabel}` : ''}</span>`;
}

function renderAgentRunList(agent) {
  const runs = (agent.runs || []).slice(0, 3);
  if (!runs.length) return '';
  const items = runs
    .map(
      (run) => `
        <button type="button" class="agent-run-link" data-agent-id="${escapeHtml(agent.agent_id)}" data-run-id="${escapeHtml(run.run_id)}">
          <span class="agent-run-primary">${escapeHtml(formatBacktestRunPrimary(run))}</span>
          <span class="agent-run-secondary">${escapeHtml(formatBacktestRunSecondary(run))}</span>
        </button>`,
    )
    .join('');
  return `<div class="agent-run-list">${items}</div>`;
}

function listBacktestableAgents() {
  return (allAgents || []).filter((agent) => agent?.agent_id && !isDemoAgent(agent.agent_id));
}

function populateBacktestAgentSelect() {
  const select = document.getElementById('backtestAgentSelect');
  if (!select) return;

  const agents = listBacktestableAgents();
  const activeId = localStorage.getItem(ACTIVE_AGENT_KEY);

  if (!agents.length) {
    select.innerHTML = '<option value="">No agents yet — create one in My Agents</option>';
    select.disabled = true;
    return;
  }

  select.disabled = false;
  select.innerHTML = agents
    .map((agent) => {
      const type = agent.agent_type === 'builtin' ? 'Built-in' : 'External';
      const model = agent.model_name || 'local-model';
      const label = `${agentDisplayName(agent)} · ${model} · ${type}`;
      return `<option value="${escapeHtml(agent.agent_id)}">${escapeHtml(label)}</option>`;
    })
    .join('');

  const selectedId =
    activeId && agents.some((agent) => agent.agent_id === activeId)
      ? activeId
      : agents[0].agent_id;
  select.value = selectedId;
}

function normalizeBacktestModelId(modelName) {
  const raw = String(modelName || '').trim().toLowerCase();
  const providerless = raw.includes('/') ? raw.split('/').pop() : raw;
  return providerless
    .replace(/_/g, '-')
    .replace(/-(\d+)-(\d+)(?=-|$)/g, '-$1.$2');
}

function findBacktestModelOption(modelSelect, modelName) {
  const normalized = normalizeBacktestModelId(modelName);
  if (!normalized) return null;
  return Array.from(modelSelect?.options || []).find(
    (option) => normalizeBacktestModelId(option.value) === normalized,
  ) || null;
}

/** The picker is live for every pipeline runtime that can use an LLM. */
function backtestModelPickerIsLiveControl() {
  const source = document.getElementById('marketDataSourceSelect')?.value || 'alpaca';
  return (
    (runBacktestModalAgent?.runtime_type || 'pipeline') === 'pipeline'
    && source !== 'vnpy_simulation'
  );
}

function resolveBacktestModelRequest(modelSelect, agent) {
  const selectedModel = modelSelect?.value || '';
  const agentOption = findBacktestModelOption(modelSelect, agent?.model_name);
  if (agentOption?.value === selectedModel && agent?.model_name) {
    return agent.model_name;
  }
  // Belt-and-braces since syncModelSelectFromAgent started injecting an option
  // for unrepresentable models: on the hidden path the agent's saved model wins
  // outright, whatever the select happens to hold.
  if (agent?.model_name && !backtestModelPickerIsLiveControl()) {
    return agent.model_name;
  }
  return selectedModel || agent?.model_name || 'claude-haiku-4.5';
}

/** Show execution controls only for pipeline LLM runs. */
function syncBacktestModelFieldMode() {
  const modelSelect = document.getElementById('modelSelect');
  const readonly = document.getElementById('runBacktestModelReadonly');
  const billingGroup = document.getElementById('runBacktestBillingGroup');
  const source = document.getElementById('marketDataSourceSelect')?.value || 'alpaca';
  const isHostedRuntime = (
    runBacktestModalAgent?.runtime_type || 'pipeline'
  ) !== 'pipeline';
  const isSimulation = source === 'vnpy_simulation';
  const isIFind = source === IFIND_ASHARE_SOURCE;
  const isRuleBased = (
    isSimulation
    || (isIFind && modelSelect?.value === RULE_BASED_DECISION_SOURCE)
  );
  const modelIsEditable = !isHostedRuntime && !isSimulation;
  if (modelSelect) modelSelect.hidden = !modelIsEditable;
  if (readonly) {
    readonly.hidden = modelIsEditable;
    readonly.textContent = isHostedRuntime
      ? 'AI Hedge Fund — hosted runtime'
      : 'Rule-based — simulated practice data, no AI involved';
  }
  if (billingGroup) billingGroup.hidden = isHostedRuntime || isRuleBased;
  syncRunBacktestSubmitAvailability();
}

/**
 * Point the picker at this agent's model.
 *
 * A model the curated list cannot represent (a legacy value like 'gpt-5.2' or
 * 'local-model') is INJECTED as its own option rather than left unmatched.
 * Leaving it unmatched is a silent-wrong-value bug, not a cosmetic one: on the
 * live iFinD path resolveBacktestModelRequest returns the select's current
 * value, so the run would submit whatever the previously-selected agent left
 * there, recorded under this agent's name. js/agent-editor.js does the same
 * thing for the Configure picker.
 */
function syncModelSelectFromAgent(agent) {
  const modelSelect = document.getElementById('modelSelect');
  if (!modelSelect || !agent?.model_name) return;
  // Drop the previous agent's injected option first, so injections cannot pile
  // up across agent switches and cannot be matched as if they were curated.
  modelSelect.querySelectorAll('option[data-injected-model]').forEach((option) => option.remove());
  const option = findBacktestModelOption(modelSelect, agent.model_name);
  if (option) {
    modelSelect.value = option.value;
    return;
  }
  const injected = document.createElement('option');
  injected.value = agent.model_name;
  injected.textContent = formatAgentModelLabel(agent.model_name);
  injected.dataset.injectedModel = 'true';
  modelSelect.appendChild(injected);
  modelSelect.value = agent.model_name;
}

function getSelectedBacktestAgent() {
  const select = document.getElementById('backtestAgentSelect');
  if (select?.value) {
    const agent = allAgents.find((item) => item.agent_id === select.value);
    if (agent) return agent;
  }
  return resolveActiveAgentForBacktest();
}

async function onBacktestAgentSelectChange() {
  const select = document.getElementById('backtestAgentSelect');
  if (!select?.value) return;
  const agent = allAgents.find((item) => item.agent_id === select.value);
  if (!agent) return;

  await activateAgent(agent);
  syncModelSelectFromAgent(agent);
  localStorage.removeItem(SELECTED_BACKTEST_RUN_KEY);
  if (currentMode === 'backtest') {
    await loadData();
  }
}

// First-visit onboarding: a brand-new owner gets the Prompted Models starters
// (DeepSeek V4 Pro, GPT-5.5, Claude Sonnet 4.6). The guard key means "we
// provisioned the set for this identity" — deleting every starter must NOT
// resurrect them. Missing models in a non-empty list are still filled so an
// account that only received the original DeepSeek card gets the other two.
let defaultAgentProvisionInFlight = null;

function stampDefaultAgentProvisionGuard(agentId) {
  try {
    const guardKey = defaultAgentProvisionGuardKey();
    if (!localStorage.getItem(guardKey)) {
      localStorage.setItem(guardKey, agentId || '1');
    }
  } catch (e) {
    /* storage unavailable — delete-guard simply won't persist */
  }
}

async function ensureDefaultFoundationAgent(agents) {
  if (isDemoMode()) return false;
  const builtins = agents.filter((a) => a.agent_type === 'builtin');
  const present = new Set(builtins.map((a) => String(a.model_name || '')));
  const missing = STARTER_AGENTS.filter((spec) => !present.has(spec.model_name));
  if (!missing.length) {
    // A builtin visible only via the unclaimed-browser-session fallback (#235)
    // is not proof claim-account actually landed — owner_user_id is still null
    // server-side. Stamping the guard against it would permanently mark this
    // identity "onboarded" for an agent it may never end up owning.
    const user = typeof getStoredAuthUser === 'function' ? getStoredAuthUser() : null;
    const owned = user?.id != null
      ? builtins.find((a) => a.owner_user_id === user.id)
      : builtins[0];
    if (owned) stampDefaultAgentProvisionGuard(owned.agent_id);
    return false;
  }
  if (!builtins.length && hasDefaultAgentProvisionGuard()) return false;
  if (defaultAgentProvisionInFlight) {
    // Another loadAgents is already creating starters — wait for it so a
    // signup race does not skip provisioning and leave My Agents empty.
    try {
      return await defaultAgentProvisionInFlight;
    } catch (e) {
      return false;
    }
  }
  defaultAgentProvisionInFlight = (async () => {
    let createdAny = false;
    let firstId = null;
    for (const spec of missing) {
      try {
        const data = await API.post(`${API_BASE}/api/v1/agents`, {
          name: spec.name,
          model_name: spec.model_name,
          agent_type: 'builtin',
          description: spec.description,
          cash_allocation: DEFAULT_AGENT_CASH_ALLOCATION,
        });
        const agent = data?.agent;
        if (!agent?.agent_id) continue;
        createdAny = true;
        firstId = firstId || agent.agent_id;
        // The starter instruction is seeded server-side by AgentService.create_agent
        // for every builtin agent. It used to be a follow-up PATCH from here, which
        // failed silently in prod for months (PATCH was missing from the CORS
        // allow_methods, so the preflight 400'd) and left every default agent with
        // an empty pipeline. Seeding in the same call that creates the row means it
        // cannot half-succeed.
      } catch (error) {
        // Non-fatal: the row falls back to its empty state with the Add Agent CTA.
        console.warn('Default agent provisioning skipped:', error.message);
      }
    }
    if (createdAny) {
      stampDefaultAgentProvisionGuard(firstId);
      if (!getDefaultAgentId()) setDefaultAgentId(firstId);
    }
    return createdAny;
  })();
  try {
    return await defaultAgentProvisionInFlight;
  } finally {
    defaultAgentProvisionInFlight = null;
  }
}

async function alignStarterAgentNames(agents) {
  // Persist the card-title binding: a Claude starter whose stored name is
  // still "DeepSeek V4 Pro" (model changed, or a copy) is rewritten so the
  // editor and the grid cannot disagree after the next fetch.
  if (isDemoMode() || !Array.isArray(agents) || !agents.length) return false;
  let changed = false;
  for (const agent of agents) {
    if ((agent.agent_type || '') !== 'builtin') continue;
    if ((agent.runtime_type || 'pipeline') !== 'pipeline') continue;
    const nextName = agentDisplayName(agent);
    if (!nextName || nextName === String(agent.name || '').trim()) continue;
    try {
      await API.patch(
        `${API_BASE}/api/v1/agents/${encodeURIComponent(agent.agent_id)}`,
        { name: nextName },
      );
      agent.name = nextName;
      changed = true;
    } catch (error) {
      console.warn('Starter name align skipped:', error.message);
    }
  }
  return changed;
}

// Auth boot gate: nav goes live before the boot's auth awaits, so a My Agents
// click can arrive while refreshAuthUser → claimAgentsForUser is still in
// flight. Fetching agents at that moment misses the guest Foundation agent a
// landing signup is about to claim, and ensureDefaultFoundationAgent would
// provision a duplicate starter. Every external caller therefore waits here;
// the DOMContentLoaded handler opens the gate once the claim phase settles.
let openAuthBootGate;
const authBootGate = new Promise((resolve) => {
  openAuthBootGate = resolve;
});
let agentsLoadInFlight = null;

async function loadAgents() {
  // Coalesce concurrent callers: several early clicks during a cold boot must
  // share one fetch, not stack identical requests behind the gate. Sequential
  // calls still refetch (re-clicking the subtab is the user's refresh).
  if (agentsLoadInFlight) return agentsLoadInFlight;
  agentsLoadInFlight = (async () => {
    try {
      await authBootGate;
      return await loadAgentsNow();
    } finally {
      agentsLoadInFlight = null;
    }
  })();
  return agentsLoadInFlight;
}

// The ungated loader: only for callers already ordered after the account
// claim (claimAgentsForUser itself, which runs inside the gated section and
// would deadlock on the gate above).
async function loadAgentsNow() {
  try {
    let data = await API.get(`${API_BASE}/api/v1/agents`);
    let agents = data.agents || [];

    // Fallback: fetch saved active agent directly (survives owner/session mismatch)
    const activeId = localStorage.getItem(ACTIVE_AGENT_KEY);
    if (activeId && !agents.some((a) => a.agent_id === activeId)) {
      try {
        const one = await API.get(`${API_BASE}/api/v1/agents/${activeId}`);
        if (one?.agent) {
          agents = [one.agent, ...agents];
        }
      } catch (fallbackError) {
        console.warn('Active agent fallback failed:', fallbackError.message);
      }
    }

    if (!agents.length) {
      try {
        const runs = await API.get(`${API_BASE}/api/backtest/runs?t=${Date.now()}`);
        const hasExt = (runs || []).some((r) => r.run_id && String(r.run_id).startsWith('ext_'));
        if (hasExt) {
          const imported = await API.post(`${API_BASE}/api/v1/agents/import-session`, {});
          if (imported?.agent) {
            agents = [imported.agent];
            applyActiveAgent(imported.agent);
          }
        }
      } catch (importError) {
        console.warn('Session import skipped:', importError.message);
      }
    }

    // Demo only: seed illustrative agents so the page has content without a
    // backend. Real users get the genuine empty-state (rendered by
    // renderAgentCategories) instead of fabricated agents.
    if (!agents.length && isDemoMode()) {
      agents = visibleMockAgents();
    }

    if (await ensureDefaultFoundationAgent(agents)) {
      try {
        const refreshed = await API.get(`${API_BASE}/api/v1/agents`);
        agents = refreshed.agents || agents;
      } catch (refreshError) {
        console.warn('Refresh after default-agent provisioning failed:', refreshError.message);
      }
    }
    await alignStarterAgentNames(agents);

    allAgents = agents;
    applyAgentFilters();
    populateBacktestAgentSelect();
    if (typeof window.renderPortfolio === 'function') {
      Promise.resolve(window.renderPortfolio(allAgents.map(decorateAgent))).catch((error) => {
        console.warn('renderPortfolio after loadAgents failed:', error?.message || error);
      });
    } else if (typeof window.updateAgentAllocationFromAgents === 'function') {
      window.updateAgentAllocationFromAgents(allAgents.map(decorateAgent));
    }
    if (typeof window.refreshHomeModules === 'function') {
      window.refreshHomeModules();
    }
  } catch (error) {
    console.warn('Failed to load agents:', error.message);
    if (isDemoMode()) {
      allAgents = visibleMockAgents();
      applyAgentFilters();
      populateBacktestAgentSelect();
    } else {
      // Real backend outage: show a distinct error-state, never fake data.
      allAgents = [];
      renderAgentsError();
      populateBacktestAgentSelect();
    }
    if (typeof window.refreshHomeModules === 'function') {
      window.refreshHomeModules();
    }
  }
}

let marketplaceTemplates = [];
let marketplaceCloneInFlight = false;
let marketplaceLoadInFlight = null;
/** 'all' or one of MARKET_LABELS' keys. Set by the chip row and by the Prompted
 * Models shelf's empty-state Community button (via navigateToPage's options). */
let marketplaceCategoryFilter = 'all';

/** 'all' or one of MODEL_VENDORS' keys. ANDs with marketplaceCategoryFilter. */
let marketplaceVendorFilter = 'all';

/** The model-vendor axis: who makes a model, and how it is licensed.
 *
 * Promoted from a submeta label lookup into the source of truth for the whole
 * axis -- Community's vendor chips, the open-source badge and the card submeta
 * all derive from this one table, so a badge cannot drift from the vendor it
 * describes. A wrong badge is a factual claim about someone else's product.
 *
 * Matched by PREFIX, not exact slug, so a new model version under a known
 * vendor needs no entry here. Declaration order is chip order, mirroring how
 * MARKET_LABELS' key order mirrors the AgentCategory Literal.
 *
 * All eight are listed even though only six are pickable: a card whose model
 * matches nothing renders as the generic "AI-powered" with no chip and no
 * badge, which is invisible until someone notices. The chip ROW is still
 * derived from what the loaded catalog actually contains (see
 * renderMarketplaceVendorChips), so listing a vendor here never ships an
 * empty chip. */
const MODEL_VENDORS = [
  { key: 'anthropic', prefix: 'anthropic/', label: 'Claude', licence: 'closed' },
  { key: 'openai', prefix: 'openai/', label: 'GPT', licence: 'closed' },
  { key: 'google', prefix: 'google/', label: 'Gemini', licence: 'closed' },
  { key: 'deepseek', prefix: 'deepseek/', label: 'DeepSeek', licence: 'open' },
  { key: 'qwen', prefix: 'qwen/', label: 'Qwen', licence: 'open' },
  // "NVIDIA Nemotron", not "Nemotron": this label also feeds
  // formatModelProviderLabel, whose shipped output must not change.
  { key: 'nvidia', prefix: 'nvidia/nemotron', label: 'NVIDIA Nemotron', licence: 'open' },
  { key: 'meta', prefix: 'meta-llama/', label: 'Llama', licence: 'open' },
  { key: 'xai', prefix: 'x-ai/', label: 'Grok', licence: 'closed' },
];

/** Vendor key for a model slug, or '' when the platform genuinely doesn't know.
 *
 * '' is not a bug and must never hide the template: it stays visible under the
 * All chip and is excluded only by an explicit vendor chip -- the same contract
 * agentMarketKey documents for markets. */
function modelVendorKey(modelName) {
  const raw = String(modelName || '').trim().toLowerCase();
  if (!raw) return '';
  return (MODEL_VENDORS.find((vendor) => raw.startsWith(vendor.prefix)) || {}).key || '';
}

/** modelVendorKey for an agent record. The agent-facing twin of agentMarketKey. */
function agentVendorKey(agent) {
  return modelVendorKey(agent?.model_name);
}

/** 'open' | 'closed' | '' -- '' when the vendor is unknown. */
function modelVendorLicence(modelName) {
  const key = modelVendorKey(modelName);
  return (MODEL_VENDORS.find((vendor) => vendor.key === key) || {}).licence || '';
}

function formatModelProviderLabel(modelName) {
  const key = modelVendorKey(modelName);
  const vendor = MODEL_VENDORS.find((entry) => entry.key === key);
  return vendor ? `Powered by ${vendor.label}` : 'AI-powered';
}

/** Select a Community category chip and re-render, without a route or API
 * change -- this is in-memory UI state, not navigation. Used by the chip
 * row's own click handler for in-page filtering while already on Community.
 * (Pre-selecting a chip on *entry* to Community -- e.g. from C3's My Agents
 * empty-shelf links -- goes through navigateToPage's `communityCategory`
 * option instead, which is also the one place that resets the filter to
 * 'all' on a plain Community nav-tab entry; calling this function directly
 * from a pre-navigation hook would set the filter just before that reset
 * overwrote it back to 'all'.) An unrecognized category falls back to 'all'
 * rather than filtering to a chip that doesn't exist. */
function setMarketplaceCategoryFilter(category) {
  marketplaceCategoryFilter = MARKET_LABELS[category] ? category : 'all';
  renderMarketplaceGrid();
}

/** Select a vendor chip and re-render. Mirrors setMarketplaceCategoryFilter,
 * including the reset-to-'all' fallback for an unrecognized key. */
function setMarketplaceVendorFilter(vendorKey) {
  marketplaceVendorFilter = MODEL_VENDORS.some((vendor) => vendor.key === vendorKey)
    ? vendorKey
    : 'all';
  renderMarketplaceGrid();
}

/** Chip row above the marketplace grid: 'All' plus one chip per market, built
 * from MARKET_LABELS rather than a second hardcoded list. Built from the label
 * map rather than AGENT_SHELVES because Community filters templates by
 * *market*, and Prompted Models holds both markets -- the shelf
 * list and the chip list are different things. */
function renderMarketplaceCategoryChips() {
  const container = document.getElementById('marketplaceCategoryChips');
  if (!container) return;
  const chips = [
    { key: 'all', label: 'All' },
    ...Object.entries(MARKET_LABELS).map(([key, label]) => ({ key, label })),
  ];
  // Build once, then only toggle state. This runs from renderMarketplaceGrid,
  // which is bound to the search box's `input` event -- rebuilding innerHTML
  // per keystroke would blow away the focused chip on every character typed.
  const existing = container.querySelectorAll('[data-marketplace-category]');
  if (existing.length !== chips.length) {
    container.innerHTML = chips
      .map((chip) => `<button type="button" class="marketplace-category-chip" data-marketplace-category="${escapeHtml(chip.key)}" aria-pressed="false">${escapeHtml(chip.label)}</button>`)
      .join('');
  }
  container.querySelectorAll('[data-marketplace-category]').forEach((button) => {
    const active = button.dataset.marketplaceCategory === marketplaceCategoryFilter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

/** Second chip row: 'All' plus one chip per vendor PRESENT IN THE CATALOG.
 *
 * Deliberately asymmetric with the market row, which is hardcoded from
 * MARKET_LABELS: markets are a closed, backend-validated enum, vendors are
 * open-ended. Hardcoding all of MODEL_VENDORS would ship chips that can never
 * match anything. Order still comes from MODEL_VENDORS, not from catalog order,
 * so the row does not reshuffle when a template is added. */
function renderMarketplaceVendorChips() {
  const container = document.getElementById('marketplaceVendorChips');
  if (!container) return;
  const present = new Set(marketplaceTemplates.map((t) => modelVendorKey(t.model_name)));
  const chips = [
    { key: 'all', label: 'All models' },
    ...MODEL_VENDORS.filter((vendor) => present.has(vendor.key)).map((vendor) => ({
      key: vendor.key,
      label: vendor.label,
    })),
  ];
  // Build once, then only toggle state -- same reason as the market row: this
  // runs from renderMarketplaceGrid, which is bound to the search box's `input`.
  const existing = container.querySelectorAll('[data-marketplace-vendor]');
  if (existing.length !== chips.length) {
    container.innerHTML = chips
      .map((chip) => `<button type="button" class="marketplace-category-chip" data-marketplace-vendor="${escapeHtml(chip.key)}" aria-pressed="false">${escapeHtml(chip.label)}</button>`)
      .join('');
  }
  container.querySelectorAll('[data-marketplace-vendor]').forEach((button) => {
    const active = button.dataset.marketplaceVendor === marketplaceVendorFilter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
}

/** Empty-state copy. Three cases, deliberately worded apart -- the same concern
 * promptedEmptyHtml records for My Agents.
 *
 * A typed query wins over the facet case: when a search is what emptied the
 * grid, offering "Clear filters" sends the user to fix the wrong thing. */
function marketplaceEmptyHtml({ searching, categoryFilter, vendorFilter }) {
  if (searching) return 'No templates match your search.';
  if (categoryFilter !== 'all' && vendorFilter !== 'all') {
    return `No templates match both filters. <button type="button" class="marketplace-clear-filters">Clear filters</button>`;
  }
  if (categoryFilter !== 'all') {
    return `No ${escapeHtml(MARKET_LABELS[categoryFilter] || '')} templates yet.`;
  }
  if (vendorFilter !== 'all') {
    const vendor = MODEL_VENDORS.find((entry) => entry.key === vendorFilter);
    return `No ${escapeHtml(vendor?.label || '')} templates yet.`;
  }
  return 'No templates match your search.';
}

function getFilteredMarketplaceTemplates() {
  const query = (document.getElementById('marketplaceSearchInput')?.value || '').trim().toLowerCase();
  let list = marketplaceTemplates.slice();
  if (marketplaceCategoryFilter !== 'all') {
    list = list.filter((template) => String(template.category || '').toLowerCase() === marketplaceCategoryFilter);
  }
  if (marketplaceVendorFilter !== 'all') {
    list = list.filter((template) => modelVendorKey(template.model_name) === marketplaceVendorFilter);
  }
  if (query) {
    list = list.filter((template) => {
      const haystack = [
        template.name,
        template.description,
        template.category,
        template.author,
        ...(template.tags || []),
        template.model_name,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return haystack.includes(query);
    });
  }
  return list;
}

function renderMarketplaceGrid() {
  const grid = document.getElementById('marketplaceGrid');
  const emptyEl = document.getElementById('marketplaceEmptyState');
  const errorEl = document.getElementById('marketplaceErrorState');
  if (!grid) return;

  renderMarketplaceCategoryChips();
  renderMarketplaceVendorChips();
  if (errorEl) errorEl.hidden = true;
  const templates = getFilteredMarketplaceTemplates();
  grid.innerHTML = '';

  if (!templates.length) {
    // Keep it hidden before the first load, so it doesn't flash while
    // marketplaceTemplates is still empty.
    if (emptyEl) {
      emptyEl.hidden = marketplaceTemplates.length === 0;
      emptyEl.innerHTML = marketplaceEmptyHtml({
        searching: Boolean((document.getElementById('marketplaceSearchInput')?.value || '').trim()),
        categoryFilter: marketplaceCategoryFilter,
        vendorFilter: marketplaceVendorFilter,
      });
      emptyEl.querySelector('.marketplace-clear-filters')?.addEventListener('click', () => {
        marketplaceCategoryFilter = 'all';
        marketplaceVendorFilter = 'all';
        renderMarketplaceGrid();
      });
    }
    return;
  }
  if (emptyEl) emptyEl.hidden = true;

  templates.forEach((template) => {
    const card = document.createElement('div');
    card.className = 'section-card agent-card marketplace-card';
    const modeLabel = template.mode === 'runtime'
      ? 'Hosted'
      : (template.mode === 'pipeline' ? 'Multi-step strategy' : 'Simple instruction');
    const cloneLabel = 'Add to My Agents';
    const categoryLabel = MARKET_LABELS[String(template.category || '').toLowerCase()] || 'General';
    const modelLabel = formatModelProviderLabel(template.model_name);
    // Open weights get a badge; closed models get nothing. Licence comes from
    // MODEL_VENDORS so it cannot drift from the vendor it describes.
    const licenceBadge = modelVendorLicence(template.model_name) === 'open'
      ? '<span class="marketplace-licence-badge">Open-source model</span>'
      : '';
    const tags = (template.tags || [])
      .slice(0, 3)
      .map((tag) => `<span class="marketplace-tag">${escapeHtml(tag)}</span>`)
      .join('');
    const repoLabel = (() => {
      if (!template.repo_url) return '';
      try {
        const path = new URL(template.repo_url).pathname.replace(/^\/+|\/+$/g, '');
        return path || template.author || 'GitHub';
      } catch {
        return template.author || 'GitHub';
      }
    })();
    const authorMeta = template.repo_url
      ? `<a class="marketplace-repo-btn" href="${escapeHtml(template.repo_url)}" target="_blank" rel="noopener noreferrer" aria-label="Open ${escapeHtml(repoLabel)} on GitHub">
            <svg class="ui-icon marketplace-repo-icon" aria-hidden="true"><use href="#icon-github"></use></svg>
            <span>${escapeHtml(repoLabel)}</span>
          </a>`
      : `<span>By ${escapeHtml(template.author || 'Community')}</span>`;
    card.innerHTML = `
      <div class="agent-card-top">
        <div class="agent-card-identity">
          ${agentRobotIcon()}
          <div class="agent-card-identity-text">
            <h3 class="agent-name">${escapeHtml(template.name)}</h3>
            <p class="agent-card-submeta" title="${escapeHtml(`${modelLabel} · ${categoryLabel}`)}">${escapeHtml(modelLabel)} · ${escapeHtml(categoryLabel)}</p>
          </div>
        </div>
        <span class="marketplace-mode-chip">${escapeHtml(modeLabel)}</span>
      </div>
      <div class="marketplace-card-body">
        <p class="marketplace-card-description">${escapeHtml(template.description || 'No description provided yet.')}</p>
        <div class="marketplace-card-meta">
          ${authorMeta}
          ${template.step_count ? `<span>${template.step_count} step${template.step_count === 1 ? '' : 's'}</span>` : ''}
        </div>
        ${(licenceBadge || tags) ? `<div class="marketplace-tag-row">${licenceBadge}${tags}</div>` : ''}
      </div>
      <div class="agent-card-actions agent-card-actions--status">
        <div class="marketplace-clone-split">
          <button class="agent-card-cta marketplace-clone-btn" type="button" data-template-id="${escapeHtml(template.template_id)}">${cloneLabel}</button>
          ${template.runtime_type === 'pipeline' ? `
          <button class="agent-card-cta marketplace-clone-model-btn" type="button" data-template-id="${escapeHtml(template.template_id)}" aria-haspopup="true" aria-expanded="false" aria-label="Choose model — add this template on a different model">Choose model ▾</button>
          <div class="marketplace-model-menu" hidden>
            ${SUPPORTED_MODELS.map((model) => `<button type="button" class="agent-menu-item marketplace-model-option" data-template-id="${escapeHtml(template.template_id)}" data-model-slug="${escapeHtml(model.slug)}"${normalizeBacktestModelId(model.slug) === normalizeBacktestModelId(template.model_name) ? ' aria-current="true"' : ''}>${escapeHtml(model.label)}</button>`).join('')}
          </div>` : ''}
        </div>
      </div>`;
    grid.appendChild(card);
  });

  grid.querySelectorAll('.marketplace-clone-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const templateId = btn.dataset.templateId;
      const template = marketplaceTemplates.find((item) => item.template_id === templateId);
      if (!template || marketplaceCloneInFlight) return;
      marketplaceCloneInFlight = true;
      btn.disabled = true;
      const prevLabel = btn.textContent;
      btn.textContent = 'Adding…';
      try {
        await cloneMarketplaceTemplate(template);
      } catch (error) {
        alert(error.message || `Couldn't add this template. Please try again.`);
      } finally {
        marketplaceCloneInFlight = false;
        btn.disabled = false;
        btn.textContent = prevLabel;
      }
    });
  });

  grid.querySelectorAll('.marketplace-clone-model-btn').forEach((btn) => {
    btn.addEventListener('click', (event) => {
      event.stopPropagation();
      const menu = btn.parentElement?.querySelector('.marketplace-model-menu');
      if (!menu) return;
      const opening = menu.hidden;
      // Close every other card's menu first: two open menus overlap.
      grid.querySelectorAll('.marketplace-model-menu').forEach((el) => { el.hidden = true; });
      grid.querySelectorAll('.marketplace-clone-model-btn').forEach((el) => el.setAttribute('aria-expanded', 'false'));
      menu.hidden = !opening;
      btn.setAttribute('aria-expanded', String(opening));
    });
  });

  grid.querySelectorAll('.marketplace-model-option').forEach((option) => {
    option.addEventListener('click', async () => {
      const template = marketplaceTemplates.find((item) => item.template_id === option.dataset.templateId);
      if (!template || marketplaceCloneInFlight) return;
      marketplaceCloneInFlight = true;
      option.disabled = true;
      try {
        await cloneMarketplaceTemplate(template, option.dataset.modelSlug);
      } catch (error) {
        alert(error.message || `Couldn't add this template. Please try again.`);
      } finally {
        marketplaceCloneInFlight = false;
        option.disabled = false;
      }
    });
  });
}

function renderMarketplaceError() {
  const grid = document.getElementById('marketplaceGrid');
  const emptyEl = document.getElementById('marketplaceEmptyState');
  const errorEl = document.getElementById('marketplaceErrorState');
  if (grid) grid.innerHTML = '';
  if (emptyEl) emptyEl.hidden = true;
  if (errorEl) errorEl.hidden = false;
}

/**
 * Fetch the template catalog, at most once per page load.
 *
 * Community is a top-level page now, so this runs on every nav click, every
 * Back/Forward and the initial boot -- where it used to run once, when the
 * Playground marketplace subtab was opened. The catalog is static config the
 * server already caches in-process, so repeat visits repaint from memory and
 * skip the network entirely. A failure clears the cache, so the next visit
 * retries rather than showing the error forever.
 */
async function loadMarketplace() {
  if (marketplaceTemplates.length) {
    renderMarketplaceGrid();
    return;
  }
  // Concurrent callers share one request (boot + a fast nav click can overlap).
  if (marketplaceLoadInFlight) return marketplaceLoadInFlight;
  marketplaceLoadInFlight = (async () => {
    try {
      const data = await API.get(`${API_BASE}/api/v1/agents/marketplace`);
      marketplaceTemplates = data.templates || [];
      renderMarketplaceGrid();
    } catch (error) {
      console.warn('Failed to load marketplace:', error.message);
      marketplaceTemplates = [];
      renderMarketplaceError();
    } finally {
      marketplaceLoadInFlight = null;
    }
  })();
  return marketplaceLoadInFlight;
}

/** `modelName` omitted means the template's own model -- the primary CTA's
 * path, whose behaviour is deliberately unchanged. */
async function cloneMarketplaceTemplate(template, modelName) {
  const data = await API.post(
    `${API_BASE}/api/v1/agents/marketplace/${encodeURIComponent(template.template_id)}/clone`,
    modelName ? { model_name: modelName } : {},
  );
  const agent = data?.agent;
  if (!agent?.agent_id) {
    throw new Error('Add failed — no agent returned');
  }
  applyActiveAgent(agent);
  await loadAgents();
  switchPlaygroundTab('agents');
  if (window.AgentEditor) {
    window.AgentEditor.open(agent);
  }
}

function openCreateExternalAgentModal() {
  closeAddAgentModal();
  const modal = document.getElementById('createExternalAgentModal');
  const errorEl = document.getElementById('createExternalAgentError');
  const form = document.getElementById('createExternalAgentForm');
  if (errorEl) errorEl.hidden = true;
  if (form) form.reset();
  if (modal) modal.hidden = false;
}

function closeCreateExternalAgentModal() {
  const modal = document.getElementById('createExternalAgentModal');
  if (modal) modal.hidden = true;
}

/**
 * Lock a submit button and say what it is doing.
 *
 * disabled alone is nearly invisible in this theme, which is why a create that
 * already set it still read as a dead click.
 */
function setButtonPending(btn, label) {
  if (!btn) return;
  if (btn.dataset.idleLabel === undefined) btn.dataset.idleLabel = btn.textContent;
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  btn.classList.add('is-pending');
  btn.textContent = label;
}

function restoreButton(btn) {
  if (!btn) return;
  btn.disabled = false;
  btn.removeAttribute('aria-busy');
  btn.classList.remove('is-pending');
  if (btn.dataset.idleLabel !== undefined) btn.textContent = btn.dataset.idleLabel;
}

function openCreateBuiltinAgentModal() {
  closeAddAgentModal();
  const modal = document.getElementById('createBuiltinAgentModal');
  const errorEl = document.getElementById('createBuiltinAgentError');
  const form = document.getElementById('createBuiltinAgentForm');
  if (errorEl) errorEl.hidden = true;
  if (form) form.reset();
  if (modal) modal.hidden = false;
}

function closeCreateBuiltinAgentModal() {
  const modal = document.getElementById('createBuiltinAgentModal');
  if (modal) modal.hidden = true;
}

async function submitCreateBuiltinAgent(event) {
  event.preventDefault();
  const nameInput = document.getElementById('builtinAgentName');
  const modelInput = document.getElementById('builtinAgentModel');
  const descInput = document.getElementById('builtinAgentDescription');
  const errorEl = document.getElementById('createBuiltinAgentError');
  const submitBtn = document.getElementById('createBuiltinAgentSubmit');

  const name = nameInput?.value?.trim();
  const model_name = modelInput?.value?.trim() || 'anthropic/claude-haiku-4-5';
  const description = descInput?.value?.trim() || null;
  const cashInput = document.getElementById('builtinAgentCashAllocation');
  if (!name) return;

  let cash_allocation;
  try {
    cash_allocation = parseAgentCashAllocationInput(cashInput?.value);
  } catch (error) {
    if (errorEl) {
      errorEl.textContent = error.message;
      errorEl.hidden = false;
    }
    return;
  }

  if (errorEl) errorEl.hidden = true;
  setButtonPending(submitBtn, 'Creating…');

  try {
    const data = await API.post(`${API_BASE}/api/v1/agents`, {
      name,
      model_name,
      agent_type: 'builtin',
      description,
      cash_allocation,
    });
    // Confirm on the POST result, not after loadAgents(): that is a second
    // round trip, and gating the toast on it reinstates most of the delay.
    closeCreateBuiltinAgentModal();
    showAppToast(`"${name}" created`);
    if (data.agent) applyActiveAgent(data.agent);
    await loadAgents();
    if (data.agent) highlightAgentCard(data.agent.agent_id);
  } catch (error) {
    if (errorEl) {
      errorEl.textContent = error.message;
      errorEl.hidden = false;
    }
  } finally {
    restoreButton(submitBtn);
  }
}

function showAgentCredentials(apiKey, options = {}) {
  const modal = document.getElementById('agentCredentialsModal');
  const titleEl = document.getElementById('agentCredentialsModalTitle');
  const subtitleEl = document.getElementById('agentCredentialsModalSubtitle');
  const apiInput = document.getElementById('agentCredentialApiKey');
  const copyBtn = document.getElementById('agentCredentialCopyBtn');
  const doneBtn = document.getElementById('agentCredentialDoneBtn');

  if (titleEl) {
    titleEl.textContent = options.title || 'Agent created';
  }
  if (subtitleEl) {
    subtitleEl.textContent =
      options.subtitle ||
      'Your agent is ready. Use the access key below to connect your own program to Agentic Trading Lab. (This is the API key in the SDK and docs.)';
  }
  if (apiInput) apiInput.value = apiKey;
  if (copyBtn) {
    copyBtn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(apiKey);
        const prev = copyBtn.textContent;
        copyBtn.textContent = 'Copied';
        setTimeout(() => {
          copyBtn.textContent = prev;
        }, 1500);
      } catch (error) {
        apiInput?.select();
        document.execCommand?.('copy');
        copyBtn.textContent = 'Copied';
      }
    };
  }
  if (doneBtn) {
    doneBtn.onclick = () => closeAgentCredentialsModal();
  }
  if (modal) modal.hidden = false;
}

async function rotateAgentApiKey(agent) {
  const data = await API.post(
    `${API_BASE}/api/v1/agents/${agent.agent_id}/rotate-api-key`,
    {},
  );
  await loadAgents();
  showAgentCredentials(data.api_key, {
    title: 'New access key created',
    subtitle: `A new key was issued for "${agent.name}". Update your program — the old key no longer works.`,
  });
  return data;
}

function closeAgentCredentialsModal() {
  const modal = document.getElementById('agentCredentialsModal');
  if (modal) modal.hidden = true;
}

async function submitCreateExternalAgent(event) {
  event.preventDefault();
  const nameInput = document.getElementById('externalAgentName');
  const modelInput = document.getElementById('externalAgentModel');
  const errorEl = document.getElementById('createExternalAgentError');
  const submitBtn = document.getElementById('createExternalAgentSubmit');

  const name = nameInput?.value?.trim();
  const model_name = modelInput?.value?.trim() || 'local-model';
  const cashInput = document.getElementById('externalAgentCashAllocation');
  if (!name) return;

  let cash_allocation;
  try {
    cash_allocation = parseAgentCashAllocationInput(cashInput?.value);
  } catch (error) {
    if (errorEl) {
      errorEl.textContent = error.message;
      errorEl.hidden = false;
    }
    return;
  }

  if (errorEl) errorEl.hidden = true;
  setButtonPending(submitBtn, 'Creating…');

  try {
    const data = await API.post(`${API_BASE}/api/v1/agents`, { name, model_name, cash_allocation });
    // Same POST, same round trip as the built-in flow, so the same rule: confirm
    // on the response. The API key is shown once and exists only in this
    // response, so gating it on loadAgents() delays the one thing the user has
    // to copy before it is unrecoverable.
    closeCreateExternalAgentModal();
    showAgentCredentials(data.api_key);
    applyActiveAgent(data.agent);
    await loadAgents();
  } catch (error) {
    if (errorEl) {
      errorEl.textContent = error.message;
      errorEl.hidden = false;
    }
  } finally {
    restoreButton(submitBtn);
  }
}

// Load default configuration from backend
async function loadDefaults() {
  try {
    const defaultsUrl = `${API_BASE}/config/defaults`;
    
    console.log('📥 Fetching defaults from:', defaultsUrl);
    
    const response = await fetch(defaultsUrl);
    console.log('🔍 Response status:', response.status, response.statusText);
    
    if (!response.ok) {
      console.warn('⚠️  Failed to fetch defaults:', response.status, response.statusText);
      return;
    }
    
    const defaults = await response.json();
    console.log('📋 Raw defaults response:', defaults);
    
    if (!defaults || defaults.error) {
      console.log('⚠️  Error in defaults:', defaults?.error || 'Unknown error');
      console.log('⚠️  No defaults configured, using URL params instead');
      return;
    }
    
    console.log('✅ Loaded defaults:', defaults);
    
    // Apply defaults to UI
    if (defaults.defaultSettings) {
      const settings = defaults.defaultSettings;
      
      // Set date inputs (using correct ID selectors)
      if (settings.startDate) {
        const startInput = document.getElementById('startDate');
        if (startInput) {
          startInput.value = settings.startDate;
          console.log('✅ Set startDate to:', settings.startDate);
        } else {
          console.warn('⚠️  Could not find #startDate input');
        }
      }
      
      if (settings.endDate) {
        const endInput = document.getElementById('endDate');
        if (endInput) {
          endInput.value = settings.endDate;
          console.log('✅ Set endDate to:', settings.endDate);
        } else {
          console.warn('⚠️  Could not find #endDate input');
        }
      }
      
      // Set asset universe
      if (settings.assetList && settings.assetList.length > 0) {
        if (settings.assetList.length === 7 && settings.assetList.includes('AAPL') && settings.assetList.includes('NVDA')) {
          selectPreset('mag7');
          console.log('✅ Selected Magnificent 7 preset');
        }
      }
      
      console.log('✅ Applied default settings to UI');
    }
    
    // Store defaults globally
    window.DEFAULT_RUNS = defaults.defaultRuns || {};
    console.log('📋 Default run IDs:', window.DEFAULT_RUNS);
    
  } catch (error) {
    console.warn('⚠️  Failed to load defaults:', error.message);
  }
}

async function loadMarketDataFeatures() {
  const select = document.getElementById('marketDataSourceSelect');
  if (!select) return;

  try {
    const features = await API.get(`${API_BASE}/config/features`);
    window.VNPY_SIMULATION_ENABLED = features.vnpy_simulation_enabled === true;
    window.IFIND_ASHARE_ENABLED = features.ifind_ashare_enabled === true;
  } catch (error) {
    window.VNPY_SIMULATION_ENABLED = false;
    window.IFIND_ASHARE_ENABLED = false;
    console.warn('Could not load optional market-data features:', error.message);
  }

  const existing = select.querySelector('option[value="vnpy_simulation"]');
  if (window.VNPY_SIMULATION_ENABLED && !existing) {
    const option = document.createElement('option');
    option.value = 'vnpy_simulation';
    option.textContent = 'vn.py simulated data';
    select.appendChild(option);
  } else if (!window.VNPY_SIMULATION_ENABLED && existing) {
    existing.remove();
    if (select.value === 'vnpy_simulation') select.value = 'alpaca';
  }

  const existingIFind = select.querySelector('option[value="ifind_ashare"]');
  if (window.IFIND_ASHARE_ENABLED && !existingIFind) {
    const option = document.createElement('option');
    option.value = 'ifind_ashare';
    option.textContent = 'iFinD China A-Shares (60 min)';
    select.appendChild(option);
  } else if (!window.IFIND_ASHARE_ENABLED && existingIFind) {
    existingIFind.remove();
    if (select.value === 'ifind_ashare') select.value = 'alpaca';
  }

  syncMarketDataSourceUI();
}

function syncMarketDataSourceUI(options = {}) {
  const select = document.getElementById('marketDataSourceSelect');
  const modelSelect = document.getElementById('modelSelect');
  const startDateInput = document.getElementById('startDate');
  const endDateInput = document.getElementById('endDate');
  const modelSelectHint = document.getElementById('modelSelectHint');
  const notice = document.getElementById('vnpySimulationNotice');
  const ifindNotice = document.getElementById('ifindAshareNotice');
  const ifindUniverse = document.getElementById('ifindAshareUniverse');
  const universeTabs = document.getElementById('universeTabs');
  const builtinTab = document.getElementById('builtinTab');
  const customTab = document.getElementById('customTab');
  const isSimulation = select?.value === 'vnpy_simulation';
  const isIFind = select?.value === 'ifind_ashare';
  const resetIFindDecisionSource = options?.resetIFindDecisionSource === true;
  const enteringIFind = isIFind && !window.IFIND_PREVIOUS_UI_STATE;

  if (enteringIFind) {
    const activeTab = document.querySelector('.universe-tab.active');
    window.IFIND_PREVIOUS_UI_STATE = {
      previousUniverse: selectedUniverse,
      previousModel: modelSelect?.value || '',
      previousTab: activeTab?.dataset.tab || 'builtin',
      previousStartDate: startDateInput?.value || '',
      previousEndDate: endDateInput?.value || '',
    };
    if (startDateInput) startDateInput.value = IFIND_ASHARE_START_DATE;
    if (endDateInput) endDateInput.value = IFIND_ASHARE_END_DATE;
  }

  if (ifindUniverse) ifindUniverse.hidden = !isIFind;
  if (universeTabs) universeTabs.hidden = isIFind;

  if (isIFind) {
    renderIFindAshareUniverse({
      resetDecisionSource: enteringIFind || resetIFindDecisionSource,
    });
    if (builtinTab) {
      builtinTab.classList.remove('active');
      builtinTab.style.display = 'none';
    }
    if (customTab) {
      customTab.classList.remove('active');
      customTab.style.display = 'none';
    }
  } else if (window.IFIND_PREVIOUS_UI_STATE) {
    const {
      previousUniverse,
      previousModel,
      previousTab,
      previousStartDate,
      previousEndDate,
    } = window.IFIND_PREVIOUS_UI_STATE;
    const tab = document.querySelector(`.universe-tab[data-tab="${previousTab}"]`);
    if (tab) handleUniverseTabSwitch(tab);
    selectPreset(previousUniverse);
    if (modelSelect) {
      modelSelect.querySelector('option[value="rule_based"]')?.remove();
      if (previousModel) modelSelect.value = previousModel;
    }
    if (startDateInput) startDateInput.value = previousStartDate;
    if (endDateInput) endDateInput.value = previousEndDate;
    window.IFIND_PREVIOUS_UI_STATE = null;
  }

  if (modelSelect && !isIFind) {
    modelSelect.disabled = isSimulation;
    modelSelect.setAttribute('aria-disabled', String(isSimulation));
  }
  if (modelSelectHint && !isIFind) {
    modelSelectHint.textContent = isSimulation
      ? 'vn.py simulation uses rule-based decisions.'
      : 'Choose a provider-compatible model for this run.';
  }
  if (notice) notice.hidden = !isSimulation;
  if (ifindNotice) ifindNotice.hidden = !isIFind;
  syncBacktestModelFieldMode();
}

function renderBacktestDataSourceBadge(run) {
  const badge = document.getElementById('backtestDataSourceBadge');
  if (!badge) return;
  if (!run) {
    badge.hidden = true;
    return;
  }

  const isSimulation = run.data_source === 'vnpy_simulation';
  const isIFind = run.data_source === 'ifind_ashare';
  badge.textContent = isIFind
    ? 'iFinD China A-Shares · 60m'
    : (isSimulation ? 'vn.py simulated data' : 'Alpaca data');
  badge.className = `data-source-badge ${isIFind ? 'is-ifind' : (isSimulation ? 'is-simulated' : 'is-alpaca')}`;
  badge.hidden = false;
}

// Parse URL config for TensorFlow Playground-style sharing
function loadConfigFromURL() {
  const params = new URLSearchParams(window.location.search);
  return {
    assets: params.get('assets') || 'AAPL,MSFT',
    startDate: params.get('startDate') || '2024-01-01',
    endDate: params.get('endDate') || '2024-12-31',
    agent: params.get('agent') || 'claude',
    benchmark: params.get('benchmark') || 'djia',
    slippage: parseFloat(params.get('slippage') || '0.001'),
    txCost: parseFloat(params.get('txCost') || '10'),
  };
}

// Generate shareable URL with current config
function generateShareURL(config) {
  const params = new URLSearchParams(config);
  return `${window.location.origin}${window.location.pathname}?${params.toString()}`;
}

// ============================================================================
// Robust API Wrapper (auto-attaches X-Session-Id for backtest routes)
// ============================================================================

const API = {
  async request(endpoint, options = {}) {
    const headers = {
      'Content-Type': 'application/json',
      'x-session-id': window.SESSION_ID,
      'x-browser-id': window.BROWSER_OWNER_ID,
      ...csrfHeaders(),
      ...options.headers,
    };
    try {
      const response = await fetch(endpoint, { 
        ...options, 
        headers,
        credentials: 'include',
      });
      
      const contentType = response.headers.get('content-type');
      let data;
      
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        const text = await response.text();
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${text.substring(0, 200)}`);
        }
        return text;
      }
      
      if (!response.ok) {
        const errorMsg = data.detail || data.error || data.message || `HTTP ${response.status}`;
        const error = new Error(typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg));
        error.status = response.status;
        throw error;
      }
      
      return data;
    } catch (error) {
      console.error(`❌ API Error [${endpoint}]:`, error.message);
      throw error;
    }
  },
  
  get(endpoint) {
    return this.request(endpoint, { method: 'GET' });
  },
  
  post(endpoint, data) {
    return this.request(endpoint, { method: 'POST', body: JSON.stringify(data) });
  },

  patch(endpoint, data, extraHeaders = {}) {
    return this.request(endpoint, {
      method: 'PATCH',
      body: JSON.stringify(data),
      headers: extraHeaders,
    });
  },
};

// ============================================================================
// Use production URL on Vercel, localhost for local development
// ============================================================================

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? window.location.origin
    : '';

// Legacy localStorage key — cleared on sign-in/out; never written for new sessions.
// Session identity lives in an HttpOnly cookie (credentials: 'include').
const AUTH_TOKEN_KEY = 'auth-token';
const AUTH_USER_KEY = 'auth-user';

function isSignedIn() {
  return !!getStoredAuthUser();
}

function clearLegacyAuthToken() {
  try { localStorage.removeItem(AUTH_TOKEN_KEY); } catch (_) { /* ignore */ }
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

function csrfHeaders() {
  const token = readCsrfToken();
  return token ? { 'X-CSRF-Token': token } : {};
}
window.csrfHeaders = csrfHeaders;
// Classic-script `const API` is not a window property; agent-editor and others
// look up window.API.patch for credentialed mutating calls.
window.API = API;


const AuthAPI = {
  async request(path, options = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...csrfHeaders(),
      ...options.headers,
    };
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
      credentials: 'include',
    });

    const contentType = response.headers.get('content-type');
    const data = contentType && contentType.includes('application/json')
      ? await response.json()
      : null;

    if (!response.ok) {
      const message = data?.detail || data?.error || `HTTP ${response.status}`;
      const error = new Error(typeof message === 'string' ? message : JSON.stringify(message));
      // Callers need to tell "session is gone" (401) apart from "the server is
      // cold/broken" (5xx, network) -- the message alone cannot carry that.
      // The admin console reads the same field for 403: a refusal there means
      // the cached role is stale, not that the request was malformed.
      error.status = response.status;
      throw error;
    }

    return data;
  },

  signup(email, displayName, password) {
    return this.request('/api/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ email, display_name: displayName, password }),
    });
  },

  login(email, password) {
    return this.request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  me() {
    // Migration bridge: a session issued before the HttpOnly-cookie change
    // exists only in localStorage. Send it once as Bearer; the backend
    // answers with Set-Cookie, and refreshAuthUser then clears the legacy key.
    const legacyToken = localStorage.getItem(AUTH_TOKEN_KEY);
    return this.request('/api/auth/me', {
      method: 'GET',
      ...(legacyToken ? { headers: { Authorization: `Bearer ${legacyToken}` } } : {}),
    });
  },

  logout() {
    return this.request('/api/auth/logout', { method: 'POST' });
  },

  changePassword(currentPassword, newPassword) {
    return this.request('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });
  },

  setAvatar(dataUri) {
    return this.request('/api/auth/avatar', {
      method: 'PUT',
      body: JSON.stringify({ avatar: dataUri }),
    });
  },

  removeAvatar() {
    return this.request('/api/auth/avatar', { method: 'DELETE' });
  },

  updateDisplayName(displayName) {
    return this.request('/api/auth/display-name', {
      method: 'PUT',
      body: JSON.stringify({ display_name: displayName }),
    });
  },

  requestEmailChange(currentPassword, newEmail) {
    return this.request('/api/auth/email-change', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_email: newEmail }),
    });
  },

  verifyEmailChange(code) {
    return this.request('/api/auth/email-change/verify', {
      method: 'POST',
      body: JSON.stringify({ code }),
    });
  },

  emailChangeStatus() {
    return this.request('/api/auth/email-change', { method: 'GET' });
  },

  cancelEmailChange() {
    return this.request('/api/auth/email-change', { method: 'DELETE' });
  },

  requestPasswordReset(email) {
    return this.request('/api/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  resetPassword(email, code, newPassword) {
    return this.request('/api/auth/reset-password', {
      method: 'POST',
      body: JSON.stringify({ email, code, new_password: newPassword }),
    });
  },

  discordStart() {
    return this.request('/api/auth/discord/start', { method: 'POST' });
  },
};

const ADMIN_USERS_PAGE_SIZE = 50;
const adminUsersPage = { offset: 0, total: 0, limit: ADMIN_USERS_PAGE_SIZE };

const AdminAPI = {
  listUsers({ limit = ADMIN_USERS_PAGE_SIZE, offset = 0 } = {}) {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    return AuthAPI.request(`/api/admin/users?${params}`);
  },
  stats() {
    return AuthAPI.request('/api/admin/stats');
  },
  patchUser(userId, patch) {
    return AuthAPI.request(`/api/admin/users/${userId}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
  },
};

let authMode = 'login';

function getStoredAuthUser() {
  try {
    const raw = localStorage.getItem(AUTH_USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    console.warn('Invalid stored auth user:', error);
    return null;
  }
}
window.getStoredAuthUser = getStoredAuthUser;

function setAuthState(user) {
  clearLegacyAuthToken();
  localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
  window.AUTH_USER = user;
  updateAuthUI();
}

async function claimAgentsForUser({ reload = true } = {}) {
  if (!getStoredAuthUser()) return;
  try {
    await API.post(`${API_BASE}/api/v1/agents/claim-account`, {});
  } catch (error) {
    console.warn('Agent account claim skipped:', error.message);
  }
  if (reload) {
    // Ungated on purpose: this call IS the claim-then-load ordering the auth
    // boot gate exists to protect, and it runs before the gate opens.
    await loadAgentsNow();
  }
}

// Drop the previous account's active agent so logout / the next login does not
// keep sending that agent's trading session_id (list/activate used to treat it
// as enough to surface or reclaim another user's agents).
//
// Deliberately NOT part of clearAuthState(): refreshAuthUser() funnels *every*
// /api/auth/me failure through that function, including a free-tier cold start
// or a first-request-after-idle 500. Wiping the agent selection there would
// silently undo the restoreActiveAgentSession() that ran moments earlier on
// boot. Only a real sign-out (logout, or a 401 that proves the session is gone)
// should reach this.
function clearActiveAgentSession() {
  localStorage.removeItem(ACTIVE_AGENT_KEY);
  localStorage.removeItem(ACTIVE_AGENT_NAME_KEY);
  window.ACTIVE_AGENT = null;
  const browserOwnerId = localStorage.getItem(BROWSER_OWNER_KEY) || window.BROWSER_OWNER_ID;
  if (browserOwnerId) {
    localStorage.setItem('trading-session-id', browserOwnerId);
    window.SESSION_ID = browserOwnerId;
  }
}

function clearAuthState() {
  clearLegacyAuthToken();
  localStorage.removeItem(AUTH_USER_KEY);
  window.AUTH_USER = null;
  // The email-change form keeps its stage in a closure keyed to nobody: left
  // alone, the next user to sign in on this tab resumes the previous user's
  // half-finished change. Reset here -- every sign-out path (logout button,
  // missing token, expired session) funnels through clearAuthState.
  resetEmailChangeForm();
  // Same closure hazard for the login modal's password-reset stage.
  resetPasswordResetForm();
  updateAuthUI();
}

function updateAccountPage() {
  const user = getStoredAuthUser();
  const signedIn = document.getElementById('accountSignedIn');
  const signedOut = document.getElementById('accountSignedOut');
  const nameEl = document.getElementById('accountDisplayName');
  const emailEl = document.getElementById('accountEmail');
  const identityName = document.getElementById('accountIdentityHeading');
  const identityEmail = document.getElementById('accountIdentityEmail');
  const roleEl = document.getElementById('accountRole');
  if (!signedIn || !signedOut) return;

  if (user) {
    signedIn.hidden = false;
    signedOut.hidden = true;
    if (nameEl) nameEl.textContent = user.display_name || '—';
    if (emailEl) emailEl.textContent = user.email || '—';
    if (identityName) identityName.textContent = user.display_name || '—';
    if (identityEmail) identityEmail.textContent = user.email || '—';
    if (roleEl) roleEl.textContent = user.role === 'admin' ? 'Administrator' : 'Member';
    const nameInput = document.getElementById('displayNameInput');
    // Skip while focused so a re-render mid-edit does not stomp what is typed.
    if (nameInput && document.activeElement !== nameInput) {
      nameInput.value = user.display_name || '';
    }
    renderAvatar(document.getElementById('accountAvatarPreview'), user);
    const removeBtn = document.getElementById('avatarRemoveBtn');
    if (removeBtn) removeBtn.hidden = !user.avatar;
  } else {
    signedIn.hidden = true;
    signedOut.hidden = false;
  }
}

// Client-side mirror of MAX_CONCURRENT_BACKTESTS_CAP / MAX_CREDITS_CAP in
// dashboard/backend/users.py — the API 422s outside these bounds regardless;
// holding them in one place here just keeps the four spots that render or
// validate them from drifting apart.
const ADMIN_QUOTA_BOUNDS = {
  // min 0, not 1: 0 is "suspended". A floor equal to the default quota let an
  // admin meter an account but never stop one.
  max_concurrent_backtests: { min: 0, max: 20 },
  credits: { min: 0, max: 1000000 },
};

// Email as it goes into a confirm() dialog. escapeHtml is the wrong tool for a
// native dialog — it has no markup to escape — and the risk there is line
// forgery, not injection: the prompts below are multi-line, so an address
// carrying its own newlines writes extra sentences into the box an admin reads
// before granting admin. The backend now rejects those addresses at signup
// (api/auth.py::_normalize_email); this collapses any that predate that rule,
// and bounds the length so a 200-char address cannot push the real question
// off the dialog.
function _adminConfirmEmail(value) {
  return String(value || '').replace(/\s+/g, ' ').trim().slice(0, 120);
}

function _setAdminFlash(kind, message) {
  const errorEl = document.getElementById('adminError');
  const successEl = document.getElementById('adminSuccess');
  if (errorEl) {
    errorEl.hidden = kind !== 'error';
    if (kind === 'error') errorEl.textContent = message || '';
  }
  if (successEl) {
    successEl.hidden = kind !== 'success';
    if (kind === 'success') successEl.textContent = message || '';
  }
}

// Say whether the Credits column binds anything. Metering is a backend env
// var, so the console cannot infer it — a hardcoded "(not enforced yet)" would
// keep claiming that after an operator armed it, and dropping the note
// entirely would let an admin read a stored number as an enforced budget.
// Three states, because "stats failed" must not read as "metering off".
function setAdminCreditsNote(stats) {
  const note = document.getElementById('adminCreditsNote');
  if (!note) return;
  if (!stats || typeof stats.credits_metering_enabled !== 'boolean') {
    note.textContent = '(status unavailable)';
    return;
  }
  if (!stats.credits_metering_enabled) {
    note.textContent = '(metering off)';
    return;
  }
  const fallback = Number(stats.default_credits);
  note.textContent = Number.isFinite(fallback)
    ? `(1 per LLM backtest; default ${fallback})`
    : '(1 per LLM backtest)';
}

async function loadAdminStats() {
  const root = document.getElementById('adminStats');
  if (!root) return;
  try {
    const data = await AdminAPI.stats();
    root.querySelectorAll('[data-stat]').forEach((el) => {
      const key = el.getAttribute('data-stat');
      const value = data?.[key];
      // Strict: the API sends numbers. Number(null) is 0 and
      // Number.isFinite(0) is true, so the old coerce-then-check rendered a
      // literal "null" for a missing counter instead of the dash.
      el.textContent = typeof value === 'number' && Number.isFinite(value)
        ? String(value)
        : '—';
    });
    setAdminCreditsNote(data);
  } catch (error) {
    // Dashes alone make "stats endpoint down" identical to "no data yet";
    // keep the failure visible somewhere an admin can find it.
    console.warn('Admin stats failed to load:', error);
    root.querySelectorAll('[data-stat]').forEach((el) => {
      el.textContent = '—';
    });
    setAdminCreditsNote(null);
  }
}

function _renderAdminPager() {
  const rangeEl = document.getElementById('adminUsersRange');
  const prevBtn = document.getElementById('adminPrevBtn');
  const nextBtn = document.getElementById('adminNextBtn');
  const { offset, total, limit } = adminUsersPage;
  const shown = Math.min(limit, Math.max(0, total - offset));
  if (rangeEl) {
    rangeEl.textContent = total
      ? `Showing ${offset + 1}–${offset + shown} of ${total}`
      : '';
  }
  if (prevBtn) prevBtn.disabled = offset <= 0;
  if (nextBtn) nextBtn.disabled = offset + limit >= total;
}

// A 403 from an admin route means the cached role is stale — someone demoted
// this account since the last /me. Re-read the server's answer so the menu
// entry and the page disappear instead of sitting there erroring.
async function _handleAdminAccessLost() {
  try {
    const refreshed = await AuthAPI.me();
    if (refreshed?.user) applyUpdatedUser(refreshed.user);
  } catch (_error) {
    clearAuthState();
  }
  if (currentPage === 'admin') navigateToPage('home');
}

// Monotonic ticket for loadAdminUsers: pager clicks can overlap, responses
// land in any order, and only the newest request may own the table —
// otherwise a slow page 1 arriving late repaints over page 2 while the pager
// still says page 2.
let _adminUsersRequestSeq = 0;

async function loadAdminUsers({ offset } = {}) {
  const body = document.getElementById('adminUsersBody');
  if (!body) return;
  // Ticket taken before ANY paint: the "Admin access required." branch owns
  // the table too, and must invalidate a slower authorized fetch still in
  // flight — otherwise its late success repaints a live user table over the
  // denial (reachable via a cross-tab logout/demotion, since AUTH_USER_KEY
  // is shared localStorage and nothing listens for storage events).
  const seq = ++_adminUsersRequestSeq;
  const user = getStoredAuthUser();
  if (!user || user.role !== 'admin') {
    body.innerHTML = '<tr><td colspan="6" class="admin-empty">Admin access required.</td></tr>';
    return;
  }
  if (Number.isFinite(offset)) adminUsersPage.offset = Math.max(0, offset);
  body.innerHTML = '<tr><td colspan="6" class="admin-empty">Loading…</td></tr>';
  _setAdminFlash(null);
  try {
    const data = await AdminAPI.listUsers({
      limit: adminUsersPage.limit,
      offset: adminUsersPage.offset,
    });
    if (seq !== _adminUsersRequestSeq) return;
    const users = Array.isArray(data?.users) ? data.users : [];
    adminUsersPage.total = Number(data?.total) || users.length;
    // A page can go out of range when accounts are deleted between requests.
    if (!users.length && adminUsersPage.offset > 0) {
      return loadAdminUsers({ offset: 0 });
    }
    _renderAdminPager();
    if (!users.length) {
      body.innerHTML = '<tr><td colspan="6" class="admin-empty">No users yet.</td></tr>';
      return;
    }
    const maxBounds = ADMIN_QUOTA_BOUNDS.max_concurrent_backtests;
    const creditBounds = ADMIN_QUOTA_BOUNDS.credits;
    body.innerHTML = users.map((row) => {
      const entitlements = row.entitlements || {};
      const maxConcurrent = Number(entitlements.max_concurrent_backtests ?? 1);
      const credits = Number(entitlements.credits ?? 0);
      const role = row.role === 'admin' ? 'admin' : 'user';
      const isSelf = Boolean(user && Number(user.id) === Number(row.id));
      const roleControl = isSelf
        ? `<span class="admin-role-locked" title="You cannot demote yourself">${escapeHtml(role)} (you)</span>`
        : `<select data-field="role" aria-label="Role for ${escapeHtml(row.email)}">
            <option value="user"${role === 'user' ? ' selected' : ''}>user</option>
            <option value="admin"${role === 'admin' ? ' selected' : ''}>admin</option>
          </select>`;
      // data-server-*: the last value the server confirmed for this row.
      // "Save quotas" diffs the inputs against these and sends only what the
      // admin actually changed, so saving an untouched field can never revert
      // a concurrent admin's edit with this page's stale copy.
      return `<tr data-user-id="${escapeHtml(row.id)}" data-current-role="${escapeHtml(role)}"
        data-server-max="${escapeHtml(maxConcurrent)}" data-server-credits="${escapeHtml(credits)}">
        <td class="admin-email">${escapeHtml(row.email)}</td>
        <td>${escapeHtml(row.display_name || '—')}</td>
        <td>${roleControl}</td>
        <td>
          <input data-field="max_concurrent_backtests" type="number" min="${maxBounds.min}" max="${maxBounds.max}"
            value="${escapeHtml(maxConcurrent)}"
            aria-label="Max concurrent backtests for ${escapeHtml(row.email)}">
        </td>
        <td>
          <input data-field="credits" type="number" min="${creditBounds.min}" max="${creditBounds.max}"
            value="${escapeHtml(credits)}"
            aria-label="Credits for ${escapeHtml(row.email)}">
        </td>
        <td>
          <button type="button" class="auth-btn auth-btn-primary admin-save-btn" data-admin-save>Save quotas</button>
        </td>
      </tr>`;
    }).join('');
  } catch (error) {
    if (seq !== _adminUsersRequestSeq) return;
    if (error?.status === 403 || error?.status === 401) {
      body.innerHTML = '<tr><td colspan="6" class="admin-empty">Admin access required.</td></tr>';
      await _handleAdminAccessLost();
      return;
    }
    body.innerHTML = `<tr><td colspan="6" class="admin-empty">${escapeHtml(error.message || 'Failed to load users')}</td></tr>`;
    _setAdminFlash('error', error.message || 'Failed to load users');
  }
}

async function saveAdminUserRole(rowEl, nextRole) {
  if (!rowEl) return;
  const userId = Number(rowEl.getAttribute('data-user-id'));
  const prevRole = rowEl.getAttribute('data-current-role') || 'user';
  const roleSelect = rowEl.querySelector('[data-field="role"]');
  const email = _adminConfirmEmail(rowEl.querySelector('.admin-email')?.textContent) || `user #${userId}`;

  if (nextRole === prevRole) return;

  if (nextRole === 'admin') {
    const ok = window.confirm(
      `Promote ${email} to admin?\n\nThey will see Admin in their profile menu and can manage all accounts.`
    );
    if (!ok) {
      if (roleSelect) roleSelect.value = prevRole;
      return;
    }
  } else if (prevRole === 'admin') {
    const ok = window.confirm(
      `Demote ${email} to user?\n\nThey will lose Admin access immediately.`
    );
    if (!ok) {
      if (roleSelect) roleSelect.value = prevRole;
      return;
    }
  }

  if (roleSelect) roleSelect.disabled = true;
  _setAdminFlash(null);
  try {
    const data = await AdminAPI.patchUser(userId, { role: nextRole });
    rowEl.setAttribute('data-current-role', nextRole);
    _applyAdminRowFromUser(rowEl, data?.user);
    _setAdminFlash('success', `${email} is now ${nextRole}`);
    loadAdminStats();
    const me = getStoredAuthUser();
    if (me && Number(me.id) === userId && data?.user) {
      applyUpdatedUser({
        ...me,
        ...data.user,
        entitlements: data.user.entitlements || me.entitlements,
      });
    }
  } catch (error) {
    if (roleSelect) roleSelect.value = prevRole;
    _setAdminFlash('error', error.message || 'Role update failed');
    if (error?.status === 403 || error?.status === 401) await _handleAdminAccessLost();
  } finally {
    if (roleSelect) roleSelect.disabled = false;
  }
}

// A blank <input type="number"> reads as '' and Number('') is 0 while
// Number(' ') is 0 too — but the old code's Number(undefined) path produced
// NaN, which JSON.stringify writes as null, which Pydantic reads as "field
// omitted". The save then succeeded, changed nothing, and flashed "Updated
// quotas". Refuse the submit instead of sending a value we cannot represent.
function _readAdminQuota(rowEl, field, label, { min, max }) {
  const el = rowEl.querySelector(`[data-field="${field}"]`);
  const raw = String(el?.value ?? '').trim();
  if (raw === '') return { error: `${label} cannot be blank` };
  const value = Number(raw);
  if (!Number.isInteger(value)) return { error: `${label} must be a whole number` };
  if (value < min || value > max) {
    return { error: `${label} must be between ${min} and ${max}` };
  }
  return { value };
}

// Server truth after a PATCH: push the returned row back into the inputs and
// the data-server-* baseline. Skips any input the admin is mid-typing in
// (same focused-element rule updateAccountPage uses) so a concurrent-save
// repaint never stomps a keystroke.
function _applyAdminRowFromUser(rowEl, userPayload) {
  if (!rowEl || !userPayload) return;
  const role = userPayload.role === 'admin' ? 'admin' : 'user';
  rowEl.setAttribute('data-current-role', role);
  const roleSelect = rowEl.querySelector('select[data-field="role"]');
  if (roleSelect) roleSelect.value = role;
  const entitlements = userPayload.entitlements;
  if (!entitlements) return;
  const apply = (field, attr, value) => {
    if (value == null) return;
    rowEl.setAttribute(attr, String(value));
    const input = rowEl.querySelector(`[data-field="${field}"]`);
    if (input && document.activeElement !== input) input.value = String(value);
  };
  apply('max_concurrent_backtests', 'data-server-max', entitlements.max_concurrent_backtests);
  apply('credits', 'data-server-credits', entitlements.credits);
}

async function saveAdminUserRow(rowEl) {
  if (!rowEl) return;
  const userId = Number(rowEl.getAttribute('data-user-id'));
  const maxField = _readAdminQuota(rowEl, 'max_concurrent_backtests', 'Max concurrent backtests', ADMIN_QUOTA_BOUNDS.max_concurrent_backtests);
  const creditsField = _readAdminQuota(rowEl, 'credits', 'Credits', ADMIN_QUOTA_BOUNDS.credits);
  const btn = rowEl.querySelector('[data-admin-save]');
  const email = _adminConfirmEmail(rowEl.querySelector('.admin-email')?.textContent) || `user #${userId}`;
  const invalid = maxField.error || creditsField.error;
  if (invalid) {
    _setAdminFlash('error', `${email}: ${invalid}`);
    return;
  }
  // Send only what this admin changed relative to the last server-confirmed
  // values. The backend upsert COALESCEs omitted fields, so an untouched
  // input stays whatever the database holds now — including an edit another
  // admin committed after this page rendered — instead of being silently
  // reverted to this page's stale copy.
  const patch = {};
  if (String(maxField.value) !== rowEl.getAttribute('data-server-max')) {
    patch.max_concurrent_backtests = maxField.value;
  }
  if (String(creditsField.value) !== rowEl.getAttribute('data-server-credits')) {
    patch.credits = creditsField.value;
  }
  if (!Object.keys(patch).length) {
    _setAdminFlash('success', `No quota changes for ${email}`);
    return;
  }
  if (btn) btn.disabled = true;
  _setAdminFlash(null);
  try {
    const data = await AdminAPI.patchUser(userId, patch);
    _applyAdminRowFromUser(rowEl, data?.user);
    _setAdminFlash('success', `Updated quotas for ${email}`);
    const me = getStoredAuthUser();
    if (me && Number(me.id) === userId && data?.user) {
      // The PATCH response carries the fresh entitlements; spreading over the
      // stored user keeps the avatar the admin projection omits on purpose.
      applyUpdatedUser({
        ...me,
        ...data.user,
        entitlements: data.user.entitlements || me.entitlements,
      });
    }
  } catch (error) {
    _setAdminFlash('error', error.message || 'Save failed');
    if (error?.status === 403 || error?.status === 401) await _handleAdminAccessLost();
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderAvatar(el, user) {
  if (!el) return;
  el.innerHTML = '';
  if (user && user.avatar) {
    const img = document.createElement('img');
    img.src = user.avatar;   // server-validated data: URI
    img.alt = '';
    el.appendChild(img);
  } else {
    const source = ((user && (user.display_name || user.email)) || '?').trim();
    el.textContent = source ? source[0].toUpperCase() : '?';
  }
}

const AVATAR_MAX_INPUT_BYTES = 10 * 1024 * 1024;
const AVATAR_MAX_OUTPUT_BYTES = 100 * 1024;

async function compressAvatar(file) {
  if (file.size > AVATAR_MAX_INPUT_BYTES) {
    throw new Error('Image is too large (max 10 MB).');
  }
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch (error) {
    // createImageBitmap rejects with a developer-facing DOMException ("The source
    // image could not be decoded") for anything the browser cannot decode: a
    // truncated download, or a non-image renamed to .png. Show copy the user can
    // act on and keep the original in the console for debugging.
    console.warn('Avatar decode failed:', error);
    throw new Error('That file could not be read as an image. Try a JPG, PNG, or WebP.');
  }
  const MAX_DIM = 256;
  const scale = Math.min(1, MAX_DIM / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(bitmap, 0, 0, width, height);
  for (const quality of [0.85, 0.6]) {
    const dataUri = canvas.toDataURL('image/jpeg', quality);
    const base64 = dataUri.slice(dataUri.indexOf(',') + 1);
    const decodedBytes = Math.floor(base64.length * 3 / 4);
    if (decodedBytes <= AVATAR_MAX_OUTPUT_BYTES) return dataUri;
  }
  throw new Error('Could not compress the image under 100 KB. Try a simpler image.');
}

function applyUpdatedUser(user) {
  localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
  window.AUTH_USER = user;
  updateAuthUI();
}

function initAvatarControls() {
  const fileInput = document.getElementById('avatarFileInput');
  const uploadBtn = document.getElementById('avatarUploadBtn');
  const removeBtn = document.getElementById('avatarRemoveBtn');
  const errorEl = document.getElementById('avatarError');
  if (!fileInput || !uploadBtn) return;

  uploadBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files && fileInput.files[0];
    fileInput.value = '';
    if (!file) return;
    if (errorEl) errorEl.hidden = true;
    uploadBtn.disabled = true;
    try {
      const dataUri = await compressAvatar(file);
      const data = await AuthAPI.setAvatar(dataUri);
      applyUpdatedUser(data.user);
    } catch (error) {
      if (errorEl) {
        errorEl.textContent = error.message;
        errorEl.hidden = false;
      }
    } finally {
      uploadBtn.disabled = false;
    }
  });

  removeBtn?.addEventListener('click', async () => {
    if (errorEl) errorEl.hidden = true;
    removeBtn.disabled = true;
    try {
      const data = await AuthAPI.removeAvatar();
      applyUpdatedUser(data.user);
    } catch (error) {
      if (errorEl) {
        errorEl.textContent = error.message;
        errorEl.hidden = false;
      }
    } finally {
      removeBtn.disabled = false;
    }
  });
}

// Mirrors password_policy.py's length + email rules for live feedback.
// The blocklist rule is server-only; its violation surfaces on submit.
function localPasswordViolations(password, email) {
  const violations = [];
  if (password.length < 8) violations.push('At least 8 characters.');
  if (password.length > 128) violations.push('At most 128 characters.');
  const localPart = (email || '').split('@')[0].trim().toLowerCase();
  if (localPart.length >= 3 && password.toLowerCase().includes(localPart)) {
    violations.push('Must not contain your email name.');
  }
  return violations;
}

function renderPolicyHints(listEl, violations) {
  if (!listEl) return;
  listEl.innerHTML = '';
  if (!violations.length) {
    listEl.hidden = true;
    return;
  }
  violations.forEach((text) => {
    const li = document.createElement('li');
    li.textContent = text;
    listEl.appendChild(li);
  });
  listEl.hidden = false;
}

function initChangePasswordForm() {
  const form = document.getElementById('changePasswordForm');
  if (!form) return;
  const newInput = document.getElementById('newPasswordInput');
  const hints = document.getElementById('passwordPolicyHints');
  const errorEl = document.getElementById('changePasswordError');
  const successEl = document.getElementById('changePasswordSuccess');

  newInput?.addEventListener('input', () => {
    const user = getStoredAuthUser();
    renderPolicyHints(hints, localPasswordViolations(newInput.value, user?.email));
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const current = document.getElementById('currentPasswordInput')?.value;
    const next = newInput?.value;
    const confirmValue = document.getElementById('confirmPasswordInput')?.value;
    const submitBtn = form.querySelector('button[type="submit"]');
    if (errorEl) errorEl.hidden = true;
    if (successEl) successEl.hidden = true;

    if (next !== confirmValue) {
      if (errorEl) {
        errorEl.textContent = 'New password and confirmation do not match.';
        errorEl.hidden = false;
      }
      return;
    }
    if (submitBtn) submitBtn.disabled = true;
    try {
      await AuthAPI.changePassword(current, next);
      form.reset();
      renderPolicyHints(hints, []);
      if (successEl) successEl.hidden = false;
    } catch (error) {
      if (errorEl) {
        errorEl.textContent = error.message;
        errorEl.hidden = false;
      }
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}

function initDisplayNameForm() {
  const form = document.getElementById('accountDisplayNameForm');
  if (!form) return;
  const input = document.getElementById('displayNameInput');
  const errorEl = document.getElementById('displayNameError');
  const successEl = document.getElementById('displayNameSuccess');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitBtn = form.querySelector('button[type="submit"]');
    if (errorEl) errorEl.hidden = true;
    if (successEl) successEl.hidden = true;

    const value = (input?.value || '').trim();
    if (!value) {
      if (errorEl) {
        errorEl.textContent = 'Display name cannot be empty.';
        errorEl.hidden = false;
      }
      return;
    }

    if (submitBtn) submitBtn.disabled = true;
    try {
      const data = await AuthAPI.updateDisplayName(value);
      applyUpdatedUser(data.user);   // cascades into updateAuthUI() -> updateAccountPage()
      if (successEl) successEl.hidden = false;
    } catch (error) {
      if (errorEl) {
        errorEl.textContent = error.message;
        errorEl.hidden = false;
      }
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });
}

function renderEmailChangeState(state) {
  const idle = document.getElementById('emailChangeIdle');
  const codeStep = document.getElementById('emailChangeCodeStep');
  const copy = document.getElementById('emailChangeStepCopy');
  const submitBtn = document.getElementById('emailChangeSubmitBtn');
  const cancelBtn = document.getElementById('emailChangeCancelBtn');
  if (!idle || !codeStep) return;

  const pending = Boolean(state && state.pending);
  idle.hidden = pending;
  codeStep.hidden = !pending;
  if (cancelBtn) cancelBtn.hidden = !pending;

  if (!pending) {
    if (submitBtn) submitBtn.textContent = 'Send code';
    return;
  }

  const user = getStoredAuthUser();
  if (copy) {
    // textContent, never innerHTML: new_email is user-supplied.
    copy.textContent = state.stage === 'new'
      ? `Code sent to ${state.new_email}. Enter it to finish — check your spam folder if it doesn't arrive.`
      : `We sent a 6-character code to ${user?.email || 'your current address'}. Check your spam folder if it doesn't arrive.`;
  }
  if (submitBtn) submitBtn.textContent = state.stage === 'new' ? 'Confirm' : 'Verify';
}

// Rebound to the form's real reset by initEmailChangeForm(); the no-op covers
// clearAuthState() firing before init (e.g. token expiry on page load).
let resetEmailChangeForm = () => {};

// Same pattern for the login modal's password-reset flow: rebound by
// initAuthUI(), called from clearAuthState() so a second user on the same tab
// never resumes a half-finished reset, and from setAuthMode() so any mode
// switch restarts the flow at stage 1.
let resetPasswordResetForm = () => {};

// Masks the user's OWN typed input for the stage-2 reassurance copy. Masking
// stored account data pre-submission would be an enumeration oracle; masking
// their own input is pure reassurance and never touches the server.
function maskEmailForDisplay(email) {
  const [local = '', domain = ''] = String(email).split('@');
  const keep = local.length <= 3 ? 1 : 3;
  return `${local.slice(0, keep)}•••@${domain}`;
}

function initEmailChangeForm() {
  const form = document.getElementById('accountEmailForm');
  if (!form) return;
  const errorEl = document.getElementById('emailChangeError');
  const successEl = document.getElementById('emailChangeSuccess');
  const codeInput = document.getElementById('emailChangeCodeInput');
  const cancelBtn = document.getElementById('emailChangeCancelBtn');
  let stage = null;

  const showError = (message) => {
    if (errorEl) {
      errorEl.textContent = message;
      errorEl.hidden = false;
    }
  };

  const reset = () => {
    stage = null;
    form.reset();
    if (errorEl) errorEl.hidden = true;
    if (successEl) successEl.hidden = true;
    renderEmailChangeState({ pending: false });
  };
  resetEmailChangeForm = reset;

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitBtn = document.getElementById('emailChangeSubmitBtn');
    if (errorEl) errorEl.hidden = true;
    if (successEl) successEl.hidden = true;
    if (submitBtn) submitBtn.disabled = true;
    try {
      if (!stage) {
        const newEmail = (document.getElementById('newEmailInput')?.value || '').trim();
        // Emptiness is checked on the trimmed value, but the RAW password is what
        // gets sent -- leading/trailing whitespace can be meaningful in a password,
        // and the sibling change-password form reads its field raw too.
        const password = document.getElementById('emailChangePasswordInput')?.value || '';
        if (!newEmail || !password.trim()) {
          showError('Enter a new email address and your current password.');
          return;
        }
        const state = await AuthAPI.requestEmailChange(password, newEmail);
        stage = state.stage;
        renderEmailChangeState({ pending: true, ...state });
        const pwInput = document.getElementById('emailChangePasswordInput');
        if (pwInput) pwInput.value = '';
      } else {
        const code = (codeInput?.value || '').trim();
        if (!code) {
          showError('Enter the 6-character code from your email.');
          return;
        }
        const data = await AuthAPI.verifyEmailChange(code);
        if (data.status === 'ok') {
          applyUpdatedUser(data.user);   // cascades into updateAuthUI() -> updateAccountPage()
          reset();
          if (successEl) successEl.hidden = false;
        } else {
          // Stage advanced: a fresh code just went to the new address.
          stage = data.stage;
          if (codeInput) codeInput.value = '';
          renderEmailChangeState({ pending: true, ...data });
        }
      }
    } catch (error) {
      showError(error.message);
      // A failed verify can mean the server tore the whole request down --
      // it cancels on the 5th wrong code and on a commit-time 409. The client
      // only learns `stage` from successful responses, so re-read the
      // authoritative state instead of leaving a dead code box on screen.
      if (stage) {
        try {
          const state = await AuthAPI.emailChangeStatus();
          stage = state.pending ? state.stage : null;
          // Clear the code only when the request is actually gone. On a
          // stage-two send failure the backend deliberately leaves stage 'old'
          // intact so the code the user already holds stays valid -- wiping the
          // box would force a needless retype of a code that still works.
          if (!state.pending && codeInput) codeInput.value = '';
          renderEmailChangeState(state);
        } catch (statusError) {
          // Keep the current view; the error above already told the user.
        }
      }
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  });

  cancelBtn?.addEventListener('click', async () => {
    if (errorEl) errorEl.hidden = true;
    try {
      await AuthAPI.cancelEmailChange();
    } catch (error) {
      showError(error.message);
      return;
    }
    reset();
  });

  // Re-entering the page mid-flow must not strand the user on the idle form.
  if (getStoredAuthUser()) {
    AuthAPI.emailChangeStatus()
      .then((state) => {
        stage = state.pending ? state.stage : null;
        renderEmailChangeState(state);
      })
      .catch(() => {
        // Fail-closed, and deliberately not fail-visible: a failed status check
        // is indistinguishable here from "nothing pending", and we show the idle
        // form rather than blocking the page. If a change really was in flight,
        // the next submit either hits the 60s cooldown (429) or replaces it --
        // self-healing, but the user is not told which happened. Accepted
        // tradeoff; see the fail-closed-is-not-fail-visible note in CLAUDE.md.
        renderEmailChangeState({ pending: false });
      });
  }
}

function toggleAccountMenu(force) {
  const menu = document.getElementById('accountMenu');
  const btn = document.getElementById('authAccountBtn');
  if (!menu || !btn) return;
  const open = force !== undefined ? force : menu.hidden;
  menu.hidden = !open;
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function closeAccountMenu() {
  toggleAccountMenu(false);
}

function syncHeaderBrand(signedIn) {
  const brand = document.querySelector('.header-brand');
  if (!brand) return;
  if (signedIn) {
    brand.setAttribute('href', '/app?view=home');
    brand.setAttribute('aria-label', 'Agentic Trading Lab dashboard');
  } else {
    brand.setAttribute('href', '/');
    brand.setAttribute('aria-label', 'Agentic Trading Lab home');
  }
}

function updateAuthUI() {
  const user = getStoredAuthUser();
  const label = document.getElementById('authUserLabel');
  const signInBtn = document.getElementById('authSignInBtn');
  const menuWrap = document.getElementById('accountMenuWrap');
  const adminMenuBtn = document.getElementById('accountMenuAdminBtn');
  if (!signInBtn || !menuWrap) {
    return;
  }

  // Profile-dropdown only — never a primary-nav tab. Ordinary users must not
  // see this entry at all (CSS also forces [hidden] because .account-menu-item
  // sets display:block and otherwise overrides the UA rule).
  const isAdmin = Boolean(user && user.role === 'admin');
  if (adminMenuBtn) {
    // [hidden] alone is enough: styles.css's .account-menu-item[hidden]
    // !important guard exists for exactly this toggle, and a second inline
    // display write would just hide the coupling it documents.
    adminMenuBtn.hidden = !isAdmin;
  }
  if (!isAdmin && currentPage === 'admin') {
    navigateToPage('home');
  }

  if (user) {
    if (label) label.textContent = user.display_name || user.email;
    signInBtn.hidden = true;
    menuWrap.hidden = false;
    renderAvatar(document.getElementById('authAvatar'), user);
    const nameEl = document.getElementById('accountMenuName');
    const emailEl = document.getElementById('accountMenuEmail');
    if (nameEl) nameEl.textContent = user.display_name || '—';
    if (emailEl) emailEl.textContent = user.email || '';
  } else {
    if (label) label.textContent = '';
    signInBtn.hidden = false;
    menuWrap.hidden = true;
    closeAccountMenu();
  }

  syncHeaderBrand(Boolean(user));

  updateAccountPage();

  if (window.CreditsPage) {
    window.CreditsPage.syncAuth(user);
  }

  if (window.AdminModelProviders) {
    window.AdminModelProviders.syncAuth(user);
  }

  if (window.AdminCredits) {
    window.AdminCredits.syncAuth(user);
  }

  if (window.AdminAnalytics) {
    window.AdminAnalytics.syncAuth(user);
  }

  if (typeof window.refreshHomeModules === 'function') {
    window.refreshHomeModules();
  }
}

async function logoutUser() {
  try {
    await AuthAPI.logout();
  } catch (error) {
    console.warn('Logout request failed:', error.message);
  } finally {
    clearAuthState();
    clearActiveAgentSession();
    // Signed out, home is the landing page again. index.html sends a visit
    // carrying a cached auth-user straight to /app ("Landing is for first-time
    // / logged-out visitors only"); this is the return trip, which was never
    // built. Without it logout leaves the user on the signed-in shell, whose
    // home re-renders as the "Guest Account" demo portfolio -- the screen a
    // never-signed-in visitor gets -- so the only sign anything happened is the
    // header swapping to "Sign in".
    //
    // Ordering is load-bearing: the two clears above must run first, or the
    // landing sees a cached auth-user and bounces straight back to /app.
    // replace() rather than href so Back cannot restore the shell just left,
    // and it matches the verb the landing's own redirect uses.
    //
    // Nothing follows it: the old loadAgents() re-fetch and account/admin page
    // hop both dressed a page that is being torn down, and awaiting a request
    // first only delays the one action the user asked for.
    window.location.replace('/');
  }
}

function setAuthMode(mode) {
  authMode = mode;
  const title = document.getElementById('authModalTitle');
  const subtitle = document.getElementById('authModalSubtitle');
  const submitBtn = document.getElementById('authSubmitBtn');
  const switchBtn = document.getElementById('authSwitchBtn');
  const passwordInput = document.getElementById('authPassword');
  const errorEl = document.getElementById('authError');
  const displayNameField = document.getElementById('authDisplayNameField');
  const displayNameInput = document.getElementById('authDisplayName');

  const passwordField = document.getElementById('authPasswordField');
  const forgotBtn = document.getElementById('authForgotPasswordBtn');

  if (title) {
    title.textContent = mode === 'signup' ? 'Sign up' : mode === 'reset' ? 'Reset password' : 'Sign in';
  }
  if (subtitle) {
    subtitle.textContent = mode === 'reset'
      ? "Enter your account email and we'll send a 6-character reset code."
      : 'Optional — backtest and paper trading work without an account.';
  }
  if (submitBtn) {
    submitBtn.textContent = mode === 'signup' ? 'Create account' : mode === 'reset' ? 'Send code' : 'Sign in';
  }
  if (switchBtn) {
    switchBtn.textContent = mode === 'signup'
      ? 'Already have an account? Sign in'
      : mode === 'reset'
        ? 'Back to sign in'
        : 'Need an account? Sign up';
  }
  if (passwordInput) {
    passwordInput.autocomplete = mode === 'signup' ? 'new-password' : 'current-password';
    // required must drop with the field: a hidden required input fails native
    // form validation silently, so stage-1 submits would no-op forever.
    passwordInput.required = mode !== 'reset';
    if (mode === 'reset') passwordInput.value = '';
  }
  if (passwordField) passwordField.hidden = mode === 'reset';
  if (forgotBtn) forgotBtn.hidden = mode !== 'login';
  if (displayNameField) {
    displayNameField.hidden = mode !== 'signup';
  }
  if (displayNameInput) {
    displayNameInput.required = mode === 'signup';
    if (mode !== 'signup') {
      displayNameInput.value = '';
    }
  }
  if (errorEl) errorEl.hidden = true;
  renderPolicyHints(document.getElementById('authPasswordHints'), []);
  // Any mode switch restarts the reset flow at stage 1 (closure state must
  // not survive leaving and re-entering reset mode).
  resetPasswordResetForm();
  updateAuthUI();
}

function openAuthModal(mode = 'login') {
  const modal = document.getElementById('authModal');
  if (!modal) return;
  setAuthMode(mode);
  modal.hidden = false;
}

/** Open auth modal from landing-page links (?auth=login|signup). */
function openAuthFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const auth = (params.get('auth') || '').toLowerCase();
  if (auth !== 'login' && auth !== 'signup' && auth !== 'reset') return;

  // Already signed in — stay on the dashboard, no modal.
  if (isSignedIn()) {
    params.delete('auth');
    const clean = params.toString();
    const next = `${window.location.pathname}${clean ? `?${clean}` : ''}${window.location.hash}`;
    window.history.replaceState(getNavigationState(), '', next);
    return;
  }

  openAuthModal(auth === 'signup' ? 'signup' : auth === 'reset' ? 'reset' : 'login');
  params.delete('auth');
  const clean = params.toString();
  const next = `${window.location.pathname}${clean ? `?${clean}` : ''}${window.location.hash}`;
  window.history.replaceState(getNavigationState(), '', next);
}

function closeAuthModal() {
  const modal = document.getElementById('authModal');
  const form = document.getElementById('authForm');
  const errorEl = document.getElementById('authError');
  if (modal) modal.hidden = true;
  if (form) form.reset();
  if (errorEl) errorEl.hidden = true;
  setAuthMode('login');
}

/**
 * Open Discord with the current website account.
 * Not logged in → login modal.
 * Logged in, not linked → Discord OAuth.
 * Already linked → open the guild/channel URL.
 */
async function openDiscordWithAccount(event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }

  if (!isSignedIn()) {
    openAuthModal('login');
    return;
  }

  try {
    const data = await AuthAPI.discordStart();
    const discordUrl = data.discord_url || DISCORD_SERVER_URL;
    if (data.already_linked) {
      window.open(discordUrl, '_blank', 'noopener,noreferrer');
      return;
    }
    if (data.authorize_url) {
      window.location.href = data.authorize_url;
      return;
    }
    window.open(discordUrl, '_blank', 'noopener,noreferrer');
  } catch (error) {
    console.warn('Discord link start failed:', error.message);
    alert(error.message || `Couldn't start Discord linking. Please sign in and try again.`);
  }
}

/** Shared success handling once a Robinhood link is confirmed (reopen editor + confirm). */
async function finishRobinhoodLinkSuccess(agentId) {
  if (agentId && window.AgentEditor?.open) {
    try {
      const headers = { 'x-session-id': SESSION_ID, ...csrfHeaders() };
      const response = await fetch(`${API_BASE}/api/v1/agents/${encodeURIComponent(agentId)}`, {
        headers,
        credentials: 'include',
      });
      if (response.ok) {
        const data = await response.json();
        if (data.agent) window.AgentEditor.open(data.agent);
      }
    } catch (error) {
      console.warn('Could not reopen agent editor after Robinhood link:', error);
    }
  }
  alert('Robinhood connected. Enable live trading, save, then Run Live.');
}

/** Handle /app?robinhood=linked|pending|error after OAuth callback. */
async function handleRobinhoodOAuthReturn() {
  const params = new URLSearchParams(window.location.search);
  const robinhood = (params.get('robinhood') || '').toLowerCase();
  if (!robinhood) return;

  const agentId = params.get('agent_id');
  const linkCode = params.get('link_code');
  // Read before the delete below strips it. The alert deliberately says nothing
  // about `reason` -- it's an upstream error code, not something a user can act
  // on -- but it's the only signal that separates one failure mode from
  // another, and backend logging is not visible in this deployment, so the
  // console is where support has to be able to find it.
  //
  // Narrowed to the shape an error code actually has before it reaches a log
  // sink: this value arrives on the query string, so anyone can choose it, and
  // an unfiltered one could forge console lines with embedded newlines.
  const failureReason =
    (params.get('reason') || '').replace(/[^A-Za-z0-9._-]/g, '').slice(0, 64) || 'oauth_failed';
  params.delete('robinhood');
  params.delete('agent_id');
  params.delete('reason');
  params.delete('link_code');
  const clean = params.toString();
  const next = `${window.location.pathname}${clean ? `?${clean}` : ''}${window.location.hash}`;
  window.history.replaceState(getNavigationState(), '', next);

  if (robinhood === 'linked') {
    await finishRobinhoodLinkSuccess(agentId);
    return;
  }

  if (robinhood === 'pending') {
    try {
      const headers = { 'Content-Type': 'application/json', 'x-session-id': SESSION_ID, ...csrfHeaders() };
      const response = await fetch(`${API_BASE}/api/auth/robinhood/complete`, {
        method: 'POST',
        headers,
        credentials: 'include',
        body: JSON.stringify({ link_code: linkCode }),
      });
      if (response.ok) {
        await finishRobinhoodLinkSuccess(agentId);
        return;
      }
      let detail = null;
      try {
        const data = await response.json();
        detail = data && data.detail;
      } catch (parseError) {
        // Non-JSON error body - fall back to the generic messages below.
      }
      if (response.status === 403) {
        alert(detail || 'Robinhood link was started from a different account.');
      } else if (response.status === 400) {
        alert(detail || 'Robinhood link expired - please connect again.');
      } else {
        alert(detail || 'Could not complete Robinhood link. Please try again.');
      }
    } catch (error) {
      console.warn('Robinhood link completion failed:', error);
      alert('Could not complete Robinhood link. Please try again.');
    }
    return;
  }

  if (robinhood === 'error') {
    console.warn('Robinhood OAuth failed:', failureReason);
    alert('Robinhood connection failed. Connecting only works on a desktop computer, on the address you started from.');
  }
}

/** Handle /app?discord=linked|error after OAuth callback. */
async function handleDiscordOAuthReturn() {
  const params = new URLSearchParams(window.location.search);
  const discord = (params.get('discord') || '').toLowerCase();
  if (!discord) return;

  const reason = params.get('reason') || '';
  params.delete('discord');
  params.delete('reason');
  const clean = params.toString();
  const next = `${window.location.pathname}${clean ? `?${clean}` : ''}${window.location.hash}`;
  window.history.replaceState(getNavigationState(), '', next);

  if (discord === 'linked') {
    try {
      await refreshAuthUser();
    } catch (error) {
      console.warn('Auth refresh after Discord link failed:', error.message);
    }
    try {
      const data = await AuthAPI.discordStart();
      window.open(data.discord_url || DISCORD_SERVER_URL, '_blank', 'noopener,noreferrer');
    } catch (error) {
      window.open(DISCORD_SERVER_URL, '_blank', 'noopener,noreferrer');
    }
    return;
  }

  if (discord === 'error') {
    const messages = {
      missing_params: 'Discord linking failed (missing OAuth params).',
      invalid_state: 'Discord linking expired. Please try Open Discord again.',
      discord_already_linked: 'That Discord account is already linked to another user.',
      oauth_failed: 'Discord authorization failed. Please try again.',
      link_failed: 'Could not link Discord to your account.',
    };
    alert(messages[reason] || `Discord linking failed${reason ? ` (${reason})` : ''}.`);
  }
}

function wireDiscordAccountButtons() {
  // Opt-in only: account-linking buttons carry data-discord-link. A plain
  // "Join Discord" community invite (no marker) stays an ordinary link so
  // logged-out visitors reach the server instead of a login modal.
  document.querySelectorAll('[data-discord-link]').forEach((el) => {
    el.addEventListener('click', openDiscordWithAccount);
  });
}

async function refreshAuthUser() {
  // Probe the cookie session. Guests get 401; signed-in users refresh the
  // cached auth-user profile. A stale auth-user alone must not skip this.
  try {
    const data = await AuthAPI.me();
    clearLegacyAuthToken();
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user));
    window.AUTH_USER = data.user;
    updateAuthUI();
    await claimAgentsForUser();
  } catch (error) {
    if (getStoredAuthUser()) {
      console.warn('Auth session expired:', error.message);
    }
    clearAuthState();
    // Only a 401 proves the session is really gone. A network error or a 5xx
    // cold start must not cost the user their active agent selection.
    if (error?.status === 401) {
      clearActiveAgentSession();
    }
  }
}

function initAuthUI(options = {}) {
  const { refresh = true } = options;
  const signInBtn = document.getElementById('authSignInBtn');
  const accountBtn = document.getElementById('authAccountBtn');
  const accountSignInBtn = document.getElementById('accountSignInBtn');
  const logoutBtn = document.getElementById('authLogoutBtn');
  const closeBtn = document.getElementById('authModalClose');
  const backdrop = document.getElementById('authModalBackdrop');
  const switchBtn = document.getElementById('authSwitchBtn');
  const form = document.getElementById('authForm');

  signInBtn?.addEventListener('click', () => openAuthModal('login'));
  accountSignInBtn?.addEventListener('click', () => openAuthModal('login'));
  accountBtn?.addEventListener('click', (event) => {
    event.stopPropagation();
    toggleAccountMenu();
  });
  document.getElementById('accountMenuAccountBtn')?.addEventListener('click', () => {
    closeAccountMenu();
    navigateToPage('account');
  });
  document.getElementById('accountMenuCreditsBtn')?.addEventListener('click', () => {
    closeAccountMenu();
    navigateToPage('credits');
  });
  document.getElementById('accountMenuAdminBtn')?.addEventListener('click', () => {
    closeAccountMenu();
    navigateToPage('admin');
  });
  document.getElementById('adminRefreshBtn')?.addEventListener('click', () => {
    loadAdminStats();
    loadAdminUsers();
    if (window.AdminAnalytics) {
      window.AdminAnalytics.refresh();
    }
    if (window.AdminModelProviders) {
      window.AdminModelProviders.onEnter();
    }
    if (window.AdminCredits) {
      window.AdminCredits.onEnter();
    }
  });
  document.getElementById('adminPrevBtn')?.addEventListener('click', () => {
    loadAdminUsers({ offset: adminUsersPage.offset - adminUsersPage.limit });
  });
  document.getElementById('adminNextBtn')?.addEventListener('click', () => {
    loadAdminUsers({ offset: adminUsersPage.offset + adminUsersPage.limit });
  });
  document.getElementById('adminUsersBody')?.addEventListener('click', (event) => {
    const btn = event.target.closest('[data-admin-save]');
    if (!btn) return;
    saveAdminUserRow(btn.closest('tr'));
  });
  // Role changes save immediately (SaaS members-table pattern). Quotas still
  // use the row Save button so typing a number does not fire mid-edit.
  document.getElementById('adminUsersBody')?.addEventListener('change', (event) => {
    const select = event.target.closest('select[data-field="role"]');
    if (!select) return;
    saveAdminUserRole(select.closest('tr'), select.value);
  });
  document.getElementById('accountMenuLogoutBtn')?.addEventListener('click', () => {
    closeAccountMenu();
    logoutUser();
  });
  document.querySelector('.header-brand')?.addEventListener('click', (event) => {
    if (!getStoredAuthUser()) return;
    event.preventDefault();
    navigateToPage('home');
  });
  document.addEventListener('click', (event) => {
    const wrap = document.getElementById('accountMenuWrap');
    if (wrap && !wrap.hidden && !wrap.contains(event.target)) {
      closeAccountMenu();
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeAccountMenu();
    }
  });
  logoutBtn?.addEventListener('click', () => {
    logoutUser();
  });
  closeBtn?.addEventListener('click', closeAuthModal);
  backdrop?.addEventListener('click', closeAuthModal);
  switchBtn?.addEventListener('click', () => {
    // In reset mode the button reads "Back to sign in", so both non-login
    // modes route back to login; login still toggles to signup.
    setAuthMode(authMode === 'signup' || authMode === 'reset' ? 'login' : 'signup');
  });

  // Password-reset mode: two stages in one form, stage held in a closure
  // (the email-change pattern).
  let resetStage = 1;
  const resetCodeStep = document.getElementById('resetCodeStep');
  const resetSentCopy = document.getElementById('resetSentCopy');
  const resetCodeInput = document.getElementById('resetCodeInput');
  const resetNewPassword = document.getElementById('resetNewPassword');
  const resetHints = document.getElementById('resetPasswordHints');

  resetPasswordResetForm = () => {
    resetStage = 1;
    if (resetCodeStep) resetCodeStep.hidden = true;
    if (resetSentCopy) resetSentCopy.textContent = '';
    if (resetCodeInput) resetCodeInput.value = '';
    if (resetNewPassword) resetNewPassword.value = '';
    renderPolicyHints(resetHints, []);
  };

  document.getElementById('authForgotPasswordBtn')?.addEventListener('click', () => {
    setAuthMode('reset');
  });

  // The existing #authPassword hint listener is hard-gated to signup mode;
  // the reset flow's new-password field gets its own listener instead of
  // widening that gate.
  resetNewPassword?.addEventListener('input', () => {
    const email = document.getElementById('authEmail')?.value || '';
    renderPolicyHints(resetHints, localPasswordViolations(resetNewPassword.value, email));
  });

  document.getElementById('authPassword')?.addEventListener('input', (event) => {
    if (authMode !== 'signup') return;
    const email = document.getElementById('authEmail')?.value || '';
    let hints = document.getElementById('authPasswordHints');
    if (!hints) {
      hints = document.createElement('ul');
      hints.id = 'authPasswordHints';
      hints.className = 'password-policy-hints';
      event.target.closest('.auth-field')?.after(hints);
    }
    renderPolicyHints(hints, localPasswordViolations(event.target.value, email));
  });

  form?.addEventListener('submit', async (event) => {
    event.preventDefault();
    const email = document.getElementById('authEmail')?.value.trim();
    const displayName = document.getElementById('authDisplayName')?.value.trim();
    const password = document.getElementById('authPassword')?.value;
    const errorEl = document.getElementById('authError');
    const submitBtn = document.getElementById('authSubmitBtn');

    if (authMode === 'reset') {
      // Before the shared email/password guard below (reset mode has no
      // password field, so that guard would silently no-op stage 1) and fully
      // separate from the login/signup success path — none of the signed-in
      // bookkeeping may run for a reset.
      if (!email) return;
      submitBtn.disabled = true;
      if (errorEl) errorEl.hidden = true;
      try {
        if (resetStage === 1) {
          await AuthAPI.requestPasswordReset(email);
          resetStage = 2;
          if (resetSentCopy) {
            // textContent, never innerHTML: the address is user-typed.
            resetSentCopy.textContent = `We sent a 6-character code to ${maskEmailForDisplay(email)} — it expires in 15 minutes. Check your spam folder too.`;
          }
          if (resetCodeStep) resetCodeStep.hidden = false;
          submitBtn.textContent = 'Reset password';
          resetCodeInput?.focus();
        } else {
          const code = (resetCodeInput?.value || '').trim();
          const newPassword = resetNewPassword?.value || '';
          if (!code || !newPassword) {
            if (errorEl) {
              errorEl.textContent = 'Enter the 6-character code and a new password.';
              errorEl.hidden = false;
            }
            return;
          }
          await AuthAPI.resetPassword(email, code, newPassword);
          showAppToast('Password reset. Sign in with your new password.');
          setAuthMode('login');
          // setAuthMode reset the stage closure; re-prefill the email so the
          // user signs straight in with the password they just set.
          const emailInput = document.getElementById('authEmail');
          if (emailInput) emailInput.value = email;
        }
      } catch (error) {
        if (errorEl) {
          errorEl.textContent = error.message;
          errorEl.hidden = false;
        }
      } finally {
        submitBtn.disabled = false;
      }
      return;
    }

    if (!email || !password) {
      return;
    }

    if (authMode === 'signup' && !displayName) {
      if (errorEl) {
        errorEl.textContent = 'Display name is required for sign up.';
        errorEl.hidden = false;
      }
      return;
    }

    submitBtn.disabled = true;
    if (errorEl) errorEl.hidden = true;

    try {
      const data = authMode === 'signup'
        ? await AuthAPI.signup(email, displayName, password)
        : await AuthAPI.login(email, password);
      setAuthState(data.user);
      // Authentication is complete here, so dismiss now. Everything below is
      // post-sign-in housekeeping and must not hold the modal open — a slow or
      // hung backend used to leave the popup up over an already-signed-in UI.
      closeAuthModal();
      // Land on My Agents after either sign-up or sign-in, matching the landing
      // page's goToDashboardLoggedIn. Sign-up used to go to Home (a second
      // marketing hero); sign-in used to navigate nowhere at all, so the only
      // confirmation it had worked was the header avatar swapping in, and the
      // user was left on whatever page they happened to be reading.
      // navigateToPage maps 'agents' → playground + the 'agents' subtab itself.
      navigateToPage('agents');
      showAppToast(`Signed in as ${data.user?.display_name || data.user?.email || 'your account'}`);
      claimAgentsForUser()
        .then(() => {
          // If we arrived here from a Discord deep link that needed this account
          // (params were kept), retry it now that the owner is signed in. This
          // waits on the claim: until it lands the account does not own the
          // agent yet and the deep link's fetch 403s.
          // The params were parked in sessionStorage, not left in the URL, so
          // that they could not leak into every later history entry.
          if (readPendingDeepLink()) {
            applyAgentRunDeepLink();
          }
        })
        .catch((error) => {
          // Sign-in itself succeeded, so this must not reach the form's error
          // slot; agents reload on the next refresh. Not named for the claim:
          // claimAgentsForUser swallows the claim POST's own failure, so what
          // lands here came from the reload leg after it.
          console.warn('Post-sign-in agent reload failed:', error.message);
        });
    } catch (error) {
      if (errorEl) {
        errorEl.textContent = error.message;
        errorEl.hidden = false;
      }
    } finally {
      submitBtn.disabled = false;
    }
  });

  window.AUTH_USER = getStoredAuthUser();
  updateAuthUI();
  openAuthFromUrl();
  handleDiscordOAuthReturn();
  handleRobinhoodOAuthReturn();
  wireDiscordAccountButtons();
  initChangePasswordForm();
  initDisplayNameForm();
  initEmailChangeForm();
  initAvatarControls();
  // Boot claims + loads agents itself so landing signup → /app does not race
  // a fire-and-forget refresh against the first My Agents paint.
  if (refresh) {
    refreshAuthUser();
  }
}

// Store default run IDs
window.DEFAULT_RUNS = {};

let chartInstance = null;
let liveBacktestChartActive = false;
/** When set, Backtest view is pinned to this in-flight run (blocks history chart paint). */
let liveBacktestRunId = null;
/** Per-run progress for concurrent dashboard backtests (keyed by live_run_id).
 *  The Backtest panel still follows one focused run (`liveBacktestRunId`); My
 *  Agents cards read their own entry here so every in-flight job can show
 *  step/percent instead of an empty indeterminate bar. */
let liveBacktestProgressByRunId = Object.create(null);
/** Focused-run progress mirror for the Backtest panel + older single-run
 *  harnesses. Kept in sync with liveBacktestProgressByRunId[liveBacktestRunId]. */
let liveBacktestProgress = null;
let liveBacktestLaunchPending = false;
let liveBacktestLaunchError = false;
/** Active status-poll timer id (so dropdown can re-attach to a running job). */
let backtestPollTimer = null;
/** Consecutive failed status polls, keyed by live_run_id. A poll that throws
 *  (offline, a 502 from a cold instance) reports nothing about whether the run
 *  ended, so the run is carried as running-but-unknown until the budget below is
 *  spent -- reading one dropped request as "finished" used to stop the poller
 *  for every other in-flight run too. */
let backtestPollFailures = Object.create(null);
const BACKTEST_POLL_FAILURE_BUDGET = 5;

// Backtests in flight, so My Agents can show them. Keyed by the run's own id --
// the only identity that is unique per run -- so a second backtest for the SAME
// agent is a second entry instead of an overwrite that stranded the first card.
// Mirrored to sessionStorage: a refresh mid-run must not silently drop the
// indicator and make a running backtest look like it never started.
//
// Entry shape: { agentId, runId, startedAt }. A launch has no run id until its
// POST answers, so it is filed under a local `pending:` key and re-filed under
// the real live_run_id by promoteBacktestRunKey(). Entries written by an earlier
// build were keyed by agent id and carry no `agentId` field; every read below
// falls back to the key, so a reload mid-run across a deploy keeps its card.
const RUNNING_BACKTESTS_KEY = 'running-backtests';
// Paired with a timestamp in the key below: the counter alone restarts at 0 on
// reload, and sessionStorage survives a reload, so a fresh launch would land on
// a stale placeholder's key — the overwrite this registry exists to prevent.
let pendingBacktestSeq = 0;

function readRunningBacktests() {
    try {
        const raw = sessionStorage.getItem(RUNNING_BACKTESTS_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (error) {
        return {};
    }
}

function writeRunningBacktests(map) {
    try {
        sessionStorage.setItem(RUNNING_BACKTESTS_KEY, JSON.stringify(map));
    } catch (error) {
        /* sessionStorage unavailable — the in-page indicator still works */
    }
}

/**
 * Register a backtest as in flight and return its registry key.
 *
 * Callers keep that key and hand it back to clearAgentBacktestRunning() /
 * promoteBacktestRunKey(), so a launch only ever touches its own entry. Clearing
 * by agent id deleted whichever entry happened to sit in that agent's slot,
 * which with two runs of one agent is the wrong one.
 */
function markAgentBacktestRunning(agentId, runId) {
    if (!agentId) return null;
    const map = readRunningBacktests();
    const key = runId || `pending:${Date.now()}:${(pendingBacktestSeq += 1)}`;
    map[key] = { agentId, runId: runId || null, startedAt: Date.now() };
    writeRunningBacktests(map);
    return key;
}

/**
 * Re-file a pending launch under the live_run_id the server just issued, and
 * return the new key.
 *
 * `startedAt` is carried over rather than reset: the card's elapsed clock runs
 * from the click, not from when the POST came back.
 */
function promoteBacktestRunKey(key, agentId, runId) {
    if (!runId) return key;
    const map = readRunningBacktests();
    const previous = key ? map[key] : null;
    if (key && key !== runId) delete map[key];
    map[runId] = {
        agentId: (previous && previous.agentId) || agentId || null,
        runId,
        startedAt: Number(previous && previous.startedAt) || Date.now(),
    };
    writeRunningBacktests(map);
    return runId;
}

/**
 * Drop one run from the registry.
 *
 * `runKey` is that run's live_run_id, or the `pending:` key of a launch whose
 * POST has not answered yet -- never an agent id, which cannot identify a run.
 */
function clearAgentBacktestRunning(runKey) {
    if (!runKey) return;
    const map = readRunningBacktests();
    if (!(runKey in map)) return;
    delete map[runKey];
    writeRunningBacktests(map);
}

/**
 * Every run still in flight, oldest first, with dead entries swept.
 *
 * Entries older than the poll ceiling are discarded here as well as in
 * getAgentBacktestRunning(): a run that died without a terminal status would
 * otherwise count against the concurrency check below forever.
 */
function listRunningBacktests() {
    const map = readRunningBacktests();
    const runs = [];
    let swept = false;
    Object.keys(map).forEach((key) => {
        const entry = map[key];
        const elapsed = (Date.now() - Number(entry && entry.startedAt)) / 1000;
        if (!entry || !Number.isFinite(elapsed) || elapsed > BACKTEST_POLL_MAX_SECONDS) {
            delete map[key];
            swept = true;
            return;
        }
        runs.push({
            key,
            // Legacy entries were keyed by agent id and carry no agentId field.
            agentId: entry.agentId || key,
            runId: entry.runId || null,
            startedAt: Number(entry.startedAt) || 0,
        });
    });
    if (swept) writeRunningBacktests(map);
    runs.sort((a, b) => a.startedAt - b.startedAt);
    return runs;
}

/**
 * Refusal message when this browser is already at its concurrent-backtest
 * limit, or null when there is room.
 *
 * The server is the authority -- `_try_acquire_backtest_slot`
 * (api/routers/backtests.py) counts an owner's runs across every tab and device,
 * which this cannot see. This only turns the refusal this browser can already
 * predict into an immediate message, instead of a round-trip whose error lands
 * after the modal has closed. A signed-out caller gets 1, matching the anonymous
 * branch of the server's `_max_concurrent_for_user`; a stored session with no
 * entitlements attached is left to the server rather than guessed at.
 *
 * A limit of 0 is NOT "unknown". It is the admin console's suspension value and
 * the server refuses on it (`_count_active_for_owner(...) >= 0` is always true),
 * so lumping it in with missing/unparseable would wave through exactly the one
 * account that must never launch.
 */
function backtestConcurrencyRefusal() {
    const user = getStoredAuthUser();
    const raw = user ? user.entitlements?.max_concurrent_backtests : 1;
    const limit = Number(raw);
    if (raw === null || raw === undefined || !Number.isFinite(limit) || limit < 0) {
        return null;
    }
    if (limit === 0) {
        return 'Backtests are disabled for this account. Contact an administrator.';
    }
    if (listRunningBacktests().length < limit) return null;
    return limit === 1
        ? 'A backtest is already running. Wait for it to finish before starting another.'
        : `You already have ${limit} backtests running. Wait for one to finish.`;
}

/**
 * Running entry for an agent, or null.
 *
 * The registry is keyed per run, so one agent can hold several. The card has one
 * indicator, so it reports the most recent launch -- the run the user just
 * started; the others keep their own entries and clear themselves.
 *
 * Entries older than the poll ceiling are discarded: a run that died without a
 * terminal status would otherwise pin a card to "Backtesting…" forever.
 */
function getAgentBacktestRunning(agentId) {
    if (!agentId) return null;
    const map = readRunningBacktests();
    let entryKey = null;
    let entry = null;
    Object.keys(map).forEach((key) => {
        const candidate = map[key];
        // Legacy entries were keyed by agent id and carry no agentId field.
        if (!candidate || (candidate.agentId || key) !== agentId) return;
        if (!entry || Number(candidate.startedAt || 0) >= Number(entry.startedAt || 0)) {
            entryKey = key;
            entry = candidate;
        }
    });
    if (!entry) return null;
    const elapsed = (Date.now() - Number(entry.startedAt || 0)) / 1000;
    if (!Number.isFinite(elapsed) || elapsed > BACKTEST_POLL_MAX_SECONDS) {
        clearAgentBacktestRunning(entryKey);
        return null;
    }
    // Attribute progress by this card's runId. An entry with no runId yet is
    // pre-confirmation (POST still in flight) and must stay indeterminate —
    // spreading the focused run's numbers onto it used to paint a false
    // step/percent on a launch that was about to be refused.
    let progress = null;
    if (entry.runId) {
        progress = liveBacktestProgressByRunId[entry.runId] || null;
        if (!progress && entry.runId === liveBacktestRunId) {
            progress = liveBacktestProgress;
        }
    }
    return {
        ...entry,
        ...(progress || {}),
        elapsedSeconds: Math.floor(elapsed),
    };
}

let lastRenderedRunningKey = null;

/**
 * Per-second refresh for running cards.
 *
 * Patches the elapsed timer in place rather than re-rendering the grid:
 * renderAgentCards() starts with `grid.innerHTML = ''`, so doing that once a
 * second would destroy focus, scroll position and any open card menu for the
 * whole duration of a run. A full re-render happens only when the set of
 * running agents changes.
 */
function refreshRunningAgentCards() {
    const running = readRunningBacktests();
    // The registry is keyed per run, so two runs of one agent are two entries;
    // the cards to patch are the distinct agents behind them. (Legacy entries
    // were keyed by agent id and carry no agentId field.)
    const agentIds = [];
    Object.keys(running).forEach((runKey) => {
        const agentId = (running[runKey] && running[runKey].agentId) || runKey;
        if (agentId && !agentIds.includes(agentId)) agentIds.push(agentId);
    });
    const key = agentIds.slice().sort().join(',');
    if (key !== lastRenderedRunningKey) {
        lastRenderedRunningKey = key;
        applyAgentFilters(false);
        return;
    }
    // Query by attribute presence and compare values in JS rather than
    // interpolating an agent id into a selector string: no escaping, no
    // CSS.escape feature detection, and nothing to get wrong later.
    //
    // EVERY field renderAgentRunningBody() paints is patched here, not just the
    // text ones. A full re-render fires only when the *set* of running agents
    // changes -- twice in a normal run -- so anything missing from this list is
    // frozen at its launch value for the whole run. That is how the bar, its
    // aria-valuenow and the staleness note previously never moved while the
    // numbers beside them climbed: the card showed "84/240 · 35%" next to a bar
    // still running the indeterminate sweep, and the staleness warning this
    // feature exists for was unreachable outside a re-render.
    const nodes = {
        elapsed: document.querySelectorAll('[data-running-elapsed]'),
        step: document.querySelectorAll('[data-running-step]'),
        detail: document.querySelectorAll('[data-running-detail]'),
        stale: document.querySelectorAll('[data-running-stale]'),
        track: document.querySelectorAll('[data-running-track]'),
        bar: document.querySelectorAll('[data-running-bar]'),
    };
    const patch = (list, attribute, agentId, apply) => {
        list.forEach((el) => {
            if (el.getAttribute(attribute) !== agentId) return;
            apply(el);
        });
    };
    agentIds.forEach((agentId) => {
        const entry = getAgentBacktestRunning(agentId);
        if (!entry) return;
        // Same derivation the full render uses, so the two cannot drift.
        const view = deriveRunningProgress(entry);
        patch(nodes.elapsed, 'data-running-elapsed', agentId, (el) => {
            el.textContent = formatBacktestElapsed(entry.elapsedSeconds);
        });
        // Assigned unconditionally, empty string included: a tick where the
        // status endpoint reports no progress (file caught mid-rewrite, a
        // transient OSError) must clear the last numbers rather than leave them
        // on screen looking current.
        patch(nodes.step, 'data-running-step', agentId, (el) => {
            el.textContent = view.stepLabel;
        });
        patch(nodes.detail, 'data-running-detail', agentId, (el) => {
            el.textContent = view.detail;
        });
        patch(nodes.stale, 'data-running-stale', agentId, (el) => {
            el.textContent = view.notice;
        });
        patch(nodes.track, 'data-running-track', agentId, (el) => {
            if (!view.determinate) {
                // Removed, not zeroed: a progressbar reporting valuenow=0
                // forever is a false statement, whereas the absent attribute is
                // exactly what tells assistive tech the value is indeterminate.
                el.removeAttribute('aria-valuenow');
                el.removeAttribute('aria-valuemin');
                el.removeAttribute('aria-valuemax');
                return;
            }
            el.setAttribute('aria-valuenow', String(view.pct));
            el.setAttribute('aria-valuemin', '0');
            el.setAttribute('aria-valuemax', '100');
        });
        patch(nodes.bar, 'data-running-bar', agentId, (el) => {
            el.classList.toggle('is-determinate', view.determinate);
            // Cleared rather than set to 0%: the stylesheet's 40% width is what
            // makes the indeterminate sweep visible, and a 0%-wide bar would
            // animate nothing across the track.
            el.style.width = view.determinate ? `${view.pct}%` : '';
        });
    });
}

/**
 * Scroll the named agent's card into view and flash it.
 *
 * Attribute lookup then compare in JS -- no escaping, no CSS.escape feature
 * detection -- matching refreshRunningAgentCards() above.
 *
 * Scoped to .agent-card: every card also contains 5-8 buttons carrying the same
 * data-agent-id, and the unscoped selector would scroll to each of them in turn.
 */
function highlightAgentCard(agentId) {
  if (!agentId) return;
  document.querySelectorAll('.agent-card[data-agent-id]').forEach((card) => {
    if (card.getAttribute('data-agent-id') !== agentId) return;
    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    card.classList.add('is-just-created');
    setTimeout(() => card.classList.remove('is-just-created'), 2400);
  });
}
let liveBacktestChartMeta = { timestamps: [] };
let tradingLogCache = [];
let tradingLogFilter = 'all';
let tradingLogEmptyMessage = 'No orders yet.';
// Survives filter re-renders: the "N more not shown" notice must not disappear
// just because the user switched to BUY-only.
let tradingLogTruncatedCount = 0;
let currentMode = "home";
let currentPage = "home";
let playgroundTab = "agents";
let competitionTab = "leaderboard";
// True once the user explicitly navigates (any history:'push' navigation).
// Nav is wired before boot's auth awaits, so applyInitialNavigation may run
// AFTER a real click — restoring the saved page then would yank the page out
// from under the user.
let userHasNavigated = false;
let allRuns = [];
let comparisonData = null;
let backtestChartData = null;
let backtestSurfaceRequestSeq = 0;
let defaultConfig = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize session FIRST (before any API calls)
    initSession();
    // Defer refreshAuthUser: claim must finish before the first loadAgents so a
    // landing signup → /app handoff does not miss the guest Foundation agent.
    initAuthUI({ refresh: false });
    bindCashStepInputs();

    // ---- Pure-DOM wiring before ANY network await. On a cold backend start
    // every fetch below this block can hang for tens of seconds, and nothing
    // here needs one: nav must respond to clicks immediately. Data loads an
    // early click triggers are held by authBootGate until the account-claim
    // phase settles, so wiring early cannot reorder the claim invariant. ----
    initNavigation();
    setupTickerResizeHandler();
    setupTickerScrollControls();
    populateSupportedModelSelects();

    // Setup time period buttons
    document.querySelectorAll('.time-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            updateTimePeriod(e.target);
        });
    });

    // Setup run backtest modal
    document.getElementById('runBacktestModalClose')?.addEventListener('click', closeRunBacktestModal);
    document.getElementById('runBacktestModalBackdrop')?.addEventListener('click', closeRunBacktestModal);
    document.getElementById('runBacktestApiKeysBtn')?.addEventListener('click', goToApiKeys);
    document.getElementById('runBacktestModalSubmit')?.addEventListener('click', () => {
        runBacktest();
    });
    document
        .querySelectorAll('#runBacktestBillingGroup [data-billing-mode]')
        .forEach((button) => {
            button.addEventListener('click', () => {
                setRunBacktestBillingMode(button.dataset.billingMode);
            });
        });
    document.getElementById('runBacktestProviderSelect')?.addEventListener(
        'change',
        () => syncRunBacktestModelOptions(),
    );
    document.getElementById('modelSelect')?.addEventListener('change', () => {
        syncBacktestModelFieldMode();
        syncRunBacktestSubmitAvailability();
    });
    document.getElementById('runBacktestEditCapitalBtn')?.addEventListener('click', () => {
        const agent = runBacktestModalAgent;
        closeRunBacktestModal();
        if (agent && window.AgentEditor?.open) window.AgentEditor.open(agent);
    });
    document.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape') return;
        const modal = document.getElementById('runBacktestModal');
        if (modal && !modal.hidden) closeRunBacktestModal();
    });

    const backtestRunSelect = document.getElementById('backtestRunSelect');
    if (backtestRunSelect) {
        backtestRunSelect.addEventListener('change', async () => {
            liveBacktestLaunchPending = false;
            liveBacktestLaunchError = false;
            const runId = backtestRunSelect.value;
            if (runId) {
                localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, runId);
            } else {
                localStorage.removeItem(SELECTED_BACKTEST_RUN_KEY);
            }
            await loadData();
        });
    }

    const tradingLogFilterSelect = document.getElementById('tradingLogFilter');
    if (tradingLogFilterSelect) {
        tradingLogFilterSelect.addEventListener('change', () => {
            tradingLogFilter = tradingLogFilterSelect.value || 'all';
            paintTradingLog(tradingLogCache, {
                emptyMessage: tradingLogCache.length
                    ? 'No orders match this filter.'
                    : tradingLogEmptyMessage,
                truncatedCount: tradingLogTruncatedCount,
            });
        });
    }

    const backtestAgentSelect = document.getElementById('backtestAgentSelect');
    if (backtestAgentSelect) {
        backtestAgentSelect.addEventListener('change', () => {
            onBacktestAgentSelectChange();
        });
    }

    const marketDataSourceSelect = document.getElementById('marketDataSourceSelect');
    if (marketDataSourceSelect) {
        marketDataSourceSelect.addEventListener('change', syncMarketDataSourceUI);
    }
    document.getElementById('ifindAshareUniverseSelect')?.addEventListener(
        'change',
        () => renderIFindAshareUniverse(),
    );

    // Setup universe tabs
    document.querySelectorAll('.universe-tab').forEach(tab => {
        tab.addEventListener('click', (e) => handleUniverseTabSwitch(e.target));
    });

    // Setup preset cards
    document.getElementById('djiaCard')?.addEventListener('click', () => selectPreset('djia'));
    document.getElementById('mag7Card')?.addEventListener('click', () => selectPreset('mag7'));

    // Setup custom universe builder
    setupAssetSearch();

    const addAssetBtn = document.querySelector('.add-asset-btn');
    if (addAssetBtn) {
        addAssetBtn.addEventListener('click', handleAddAsset);
    }

    const searchInput = document.getElementById('assetSearchInput');
    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleAddAsset();
        });
    }

    // Setup chip removal
    document.querySelectorAll('.chip-remove').forEach(btn => {
        btn.addEventListener('click', (e) => removeChip(e.target.closest('.chip')));
    });

    // Ticker immediately: it is the page's de-facto liveness signal and
    // depends on nothing above.
    loadMarketTicker();
    setInterval(loadMarketTicker, 30000);
    updateMarketsOpenStatus();
    setInterval(updateMarketsOpenStatus, 60000);

    // Config fetches in parallel, off the boot critical path; awaited at the
    // end of boot so "Dashboard ready" still means fully configured.
    const configReady = Promise.all([
        loadDefaults().catch((error) => {
            console.warn('Failed to load defaults:', error);
        }),
        loadMarketDataFeatures(),
    ]);

    // If the head boot script's warmup ping is still pending after a beat,
    // say so — a free-tier cold start otherwise looks like a broken page.
    if (window.API_WARMUP) {
        let warmupSettled = false;
        window.API_WARMUP.then(() => { warmupSettled = true; });
        setTimeout(() => {
            if (!warmupSettled) {
                showAppToast('Waking up the server — the first load can take up to a minute on our free hosting.');
            }
        }, SLOW_BOOT_NOTICE_MS);
    }

    await restoreActiveAgentSession();
    // The HttpOnly session cookie is invisible to JS, so the boot signal is
    // the cached auth-user (written on every cookie sign-in) or a pre-cookie
    // legacy localStorage token (upgraded to a cookie by the /me bridge).
    if (localStorage.getItem(AUTH_TOKEN_KEY) || getStoredAuthUser()) {
        try {
            await refreshAuthUser();
        } catch (error) {
            console.warn('Boot refreshAuthUser failed:', error?.message || error);
        }
    }
    // Claim phase settled (or there was nothing to claim): gated loadAgents
    // callers queued by early clicks may fetch now.
    openAuthBootGate();
    // Portfolio overview must not wait on the agents waterfall. Paint any
    // sessionStorage snapshot immediately, kick GET /portfolio in parallel,
    // then show the page while loadAgents continues in the background.
    if (typeof window.paintPortfolioBoot === 'function') {
        try {
            window.paintPortfolioBoot(
                Array.isArray(allAgents) ? allAgents.map(decorateAgent) : [],
            );
        } catch (error) {
            console.warn('Portfolio boot paint failed:', error?.message || error);
        }
    }
    if (typeof window.prefetchPortfolio === 'function') {
        Promise.resolve(
            window.prefetchPortfolio(
                Array.isArray(allAgents) ? allAgents.map(decorateAgent) : [],
            ),
        ).catch((error) => {
            console.warn('Portfolio prefetch failed:', error?.message || error);
        });
    }
    // refreshAuthUser → claimAgentsForUser already loadAgents when signed in.
    const agentsReady = isSignedIn()
        ? Promise.resolve()
        : loadAgents().catch((error) => {
            console.warn('Initial loadAgents failed:', error.message);
        });
    applyInitialNavigation();
    // Home modules / agent cards catch up once the list lands; do not block
    // first navigation on that wait.
    await agentsReady;
    window.addEventListener('agent-editor-saved', async (event) => {
        const agent = event.detail?.agent;
        if (agent?.agent_id) {
            const idx = allAgents.findIndex((a) => a.agent_id === agent.agent_id);
            if (idx >= 0) {
                allAgents[idx] = { ...allAgents[idx], ...agent };
            }
            applyAgentFilters();
        }
        if (agent?.agent_id === localStorage.getItem(ACTIVE_AGENT_KEY)) {
            localStorage.setItem(ACTIVE_AGENT_NAME_KEY, agent.name || '');
            const nameEl = document.getElementById('playgroundAgentName');
            if (nameEl) nameEl.textContent = agent.name || 'Agent';
        }
        await loadAgents();
    });
    window.addEventListener('agent-editor-open-run', async (event) => {
        const { agent, runId } = event.detail || {};
        if (!agent || !runId) return;
        if (window.AgentEditor) window.AgentEditor.close(true);
        await activateAgent(agent);
        localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, runId);
        navigateToPage('playground', { playgroundTab: 'backtest' });
        currentMode = 'backtest';
        await loadData();
    });
    // Guarded: this awaits loadData() internally, and an unhandled rejection here
    // used to abort the rest of boot — including initNavigation(), which wires
    // every nav button. A failed deep link must not cost the user navigation.
    try {
        await applyAgentRunDeepLink();
    } catch (error) {
        console.warn('Deep link failed:', error.message);
    }
    const config = loadConfigFromURL();
    window.CURRENT_CONFIG = config;
    console.log('⚙️ Experiment config:', config);
    console.log('Session ID:', window.SESSION_ID);
    
    console.log('Dashboard initializing...');

    // Kicked off before the auth awaits; settled before boot reports ready.
    await configReady;

    console.log('🎯 Dashboard ready. Default runs:', window.DEFAULT_RUNS || 'None configured');
});

/**
 * US equity regular session: Mon–Fri 09:30–16:00 America/New_York.
 * Holidays are not modeled; closed on weekends and outside RTH.
 */
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
    // Some engines emit "24" for midnight.
    if (hour === 24) hour = 0;
    const mins = hour * 60 + minute;
    return mins >= 9 * 60 + 30 && mins < 16 * 60;
}

function updateMarketsOpenStatus() {
    const el = document.getElementById('tickerMarketsStatus');
    if (!el) return;
    const label = el.querySelector('.ticker-markets-label');
    const open = isUsEquityMarketOpen();
    el.classList.toggle('is-closed', !open);
    el.classList.toggle('ticker-markets-open', true);
    if (label) label.textContent = open ? 'Markets open' : 'Markets closed';
    el.setAttribute('aria-label', open ? 'US equity markets are open' : 'US equity markets are closed');
}

window.updateMarketsOpenStatus = updateMarketsOpenStatus;
window.isUsEquityMarketOpen = isUsEquityMarketOpen;

function formatComparisonMetric(metric, value, currency = 'USD') {
    if (!Number.isFinite(value)) return '—';
    if (metric.kind === 'currency') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency,
            maximumFractionDigits: 0,
        }).format(value);
    }
    if (metric.kind === 'percent') {
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    }
    return value.toFixed(2);
}

function createComparisonDelta(value, label) {
    if (!Number.isFinite(value)) return null;
    const delta = document.createElement('span');
    const state = value > 0 ? 'is-positive' : value < 0 ? 'is-negative' : 'is-neutral';
    const sign = value > 0 ? '+' : value < 0 ? '-' : '';
    delta.className = `performance-delta ${state}`;
    delta.textContent = `${label} ${sign}${Math.abs(value).toFixed(2)}pp`;
    return delta;
}

function setPerformanceComparisonState(state, message = '') {
    const region = document.getElementById('performanceComparison');
    const status = document.getElementById('performanceComparisonStatus');
    if (!region || !status) return;
    region.dataset.state = state;
    status.textContent = message;
}

function clearPerformanceComparison(state = 'empty', message = '') {
    document.getElementById('performanceComparisonHead')?.replaceChildren();
    document.getElementById('performanceComparisonBody')?.replaceChildren();
    renderPerformanceLegend({ columns: [] });
    setPerformanceComparisonState(state, message);
}

function renderPerformanceComparison(payload, run) {
    const head = document.getElementById('performanceComparisonHead');
    const body = document.getElementById('performanceComparisonBody');
    if (!head || !body || !window.BacktestComparison) return;

    const model = window.BacktestComparison.buildModel(payload, run);
    const currency = run?.reporting_currency
        || run?.metadata?.reporting_currency
        || 'USD';
    const headerRow = document.createElement('tr');
    const metricHeader = document.createElement('th');
    metricHeader.scope = 'col';
    metricHeader.textContent = 'Metric';
    headerRow.appendChild(metricHeader);

    for (const column of model.columns) {
        const header = document.createElement('th');
        header.scope = 'col';
        header.dataset.seriesKey = column.key;
        const swatch = document.createElement('span');
        swatch.className = 'performance-series-swatch';
        swatch.style.backgroundColor = column.color;
        swatch.setAttribute('aria-hidden', 'true');
        header.append(swatch, document.createTextNode(column.label));
        headerRow.appendChild(header);
    }
    head.replaceChildren(headerRow);

    const rows = window.BacktestComparison.METRICS.map((metric) => {
        const row = document.createElement('tr');
        const rowHeader = document.createElement('th');
        rowHeader.scope = 'row';
        rowHeader.textContent = metric.label;
        row.appendChild(rowHeader);

        for (const column of model.columns) {
            const cell = document.createElement('td');
            const value = column.metrics[metric.key];
            cell.dataset.seriesKey = column.key;
            const formatted = document.createElement('span');
            formatted.className = 'performance-metric-value';
            formatted.textContent = formatComparisonMetric(metric, value, currency);
            cell.appendChild(formatted);

            if (model.bestByMetric[metric.key].includes(column.key)) {
                cell.classList.add('performance-best');
                const best = document.createElement('span');
                best.className = 'performance-best-label';
                best.textContent = 'Best';
                cell.appendChild(best);
            }

            if (metric.key === 'totalReturn' && column.key === 'agent') {
                const deltas = document.createElement('span');
                deltas.className = 'performance-deltas';
                for (const benchmark of model.columns.filter((item) => item.key !== 'agent')) {
                    const delta = createComparisonDelta(
                        model.agentDeltas[benchmark.key],
                        benchmark.label,
                    );
                    if (delta) deltas.appendChild(delta);
                }
                cell.appendChild(deltas);
            }
            row.appendChild(cell);
        }
        return row;
    });
    body.replaceChildren(...rows);

    const missing = model.columns.filter((column) => !column.available);
    const message = missing.length
        ? `${missing.map((column) => column.label).join(', ')} unavailable for this run.`
        : '';
    setPerformanceComparisonState(missing.length ? 'partial' : 'ready', message);
    return model;
}

function renderPerformanceLegend(model, { disabled = false } = {}) {
    const host = document.getElementById('performanceLegend');
    if (!host) return;
    const available = (model?.columns || []).filter((column) => column.available);
    const buttons = available.map((column, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'performance-legend-button';
        button.dataset.datasetIndex = String(index);
        button.setAttribute('aria-pressed', 'true');
        button.disabled = disabled;

        const swatch = document.createElement('span');
        swatch.className = 'performance-series-swatch';
        swatch.style.backgroundColor = column.color;
        swatch.setAttribute('aria-hidden', 'true');
        button.append(swatch, document.createTextNode(column.label));
        button.addEventListener('click', () => {
            if (!chartInstance) return;
            const visible = button.getAttribute('aria-pressed') === 'true';
            chartInstance.setDatasetVisibility(index, !visible);
            button.setAttribute('aria-pressed', String(!visible));
            chartInstance.update();
        });
        return button;
    });
    host.replaceChildren(...buttons);
}

const MAG7_TICKER_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META'];
const TICKER_SCROLL_PX_PER_SEC = 55;
const TICKER_ESTIMATED_ITEM_WIDTH = 140;
let tickerResizeTimer = null;
let latestTickerQuotes = [];
let tickerScrollRaf = null;
let tickerScrollOffset = 0;
let tickerScrollSetWidth = 0;
let tickerScrollLastTime = 0;
let tickerScrollPaused = false;
let tickerScrollControlsBound = false;

function sortTickerQuotes(quotes) {
    const order = new Map(MAG7_TICKER_SYMBOLS.map((symbol, index) => [symbol, index]));
    return [...quotes].sort(
        (a, b) => (order.get(a.symbol) ?? 99) - (order.get(b.symbol) ?? 99)
    );
}

function getTickerMarqueeWidth() {
    const marquee = document.getElementById('tickerMarquee');
    return marquee?.clientWidth || window.innerWidth;
}

function getTickerQuoteFields(quote) {
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

function buildTickerItemHtml(quote) {
    const fields = getTickerQuoteFields(quote);

    return `
        <div class="ticker-item" data-symbol="${quote.symbol}">
            <span class="symbol">${quote.symbol}</span>
            <span class="price">${fields.price}</span>
            <span class="change ${fields.changeClass}" title="${fields.tooltip}">${fields.changeDisplay}</span>
            <svg class="ticker-chart ${fields.changeClass}" viewBox="0 0 30 12" aria-hidden="true">
                <path d="${fields.sparkPath}" stroke="currentColor" fill="none" stroke-width="1"/>
            </svg>
        </div>
    `;
}

function buildTickerSetHtml(quotes, repeats) {
    const sortedQuotes = sortTickerQuotes(quotes);
    const itemHtml = sortedQuotes.map(buildTickerItemHtml).join('');
    return Array(Math.max(1, repeats)).fill(itemHtml).join('');
}

function stopTickerScroll() {
    if (tickerScrollRaf !== null) {
        cancelAnimationFrame(tickerScrollRaf);
        tickerScrollRaf = null;
    }
}

function getTickerSetWidth(tickerTrack) {
    return tickerTrack.querySelector('.ticker-set')?.offsetWidth || 0;
}

function tickerScrollFrame(now) {
    const tickerTrack = document.getElementById('tickerTrack');
    if (!tickerTrack || tickerTrack.dataset.tickerReady !== '1') {
        stopTickerScroll();
        return;
    }

    if (!tickerScrollSetWidth) {
        tickerScrollSetWidth = getTickerSetWidth(tickerTrack);
        if (!tickerScrollSetWidth) {
            tickerScrollRaf = requestAnimationFrame(tickerScrollFrame);
            return;
        }
    }

    if (!tickerScrollLastTime) {
        tickerScrollLastTime = now;
    }

    if (!tickerScrollPaused) {
        const dt = Math.min(0.05, (now - tickerScrollLastTime) / 1000);
        tickerScrollOffset -= TICKER_SCROLL_PX_PER_SEC * dt;
        if (tickerScrollOffset <= -tickerScrollSetWidth) {
            tickerScrollOffset += tickerScrollSetWidth;
        }
        tickerTrack.style.transform = `translate3d(${tickerScrollOffset}px, 0, 0)`;
    }

    tickerScrollLastTime = now;
    tickerScrollRaf = requestAnimationFrame(tickerScrollFrame);
}

function startTickerScroll() {
    stopTickerScroll();

    const tickerTrack = document.getElementById('tickerTrack');
    if (!tickerTrack || tickerTrack.dataset.tickerReady !== '1') {
        return;
    }

    tickerScrollOffset = 0;
    tickerScrollSetWidth = 0;
    tickerScrollLastTime = 0;
    tickerTrack.style.transform = 'translate3d(0, 0, 0)';
    tickerScrollRaf = requestAnimationFrame(tickerScrollFrame);
}

function scheduleTickerScrollStart() {
    stopTickerScroll();
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            startTickerScroll();
        });
    });
}

function setupTickerScrollControls() {
    if (tickerScrollControlsBound) {
        return;
    }
    tickerScrollControlsBound = true;

    const marquee = document.getElementById('tickerMarquee');
    marquee?.addEventListener('mouseenter', () => {
        tickerScrollPaused = true;
    });
    marquee?.addEventListener('mouseleave', () => {
        tickerScrollPaused = false;
        tickerScrollLastTime = 0;
    });

    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            stopTickerScroll();
            return;
        }
        if (document.getElementById('tickerTrack')?.dataset.tickerReady === '1') {
            scheduleTickerScrollStart();
        }
    });
}

function patchTickerItemElement(item, quote) {
    const fields = getTickerQuoteFields(quote);
    const priceEl = item.querySelector('.price');
    const changeEl = item.querySelector('.change');
    const chartEl = item.querySelector('.ticker-chart');
    const pathEl = item.querySelector('.ticker-chart path');

    if (priceEl) {
        priceEl.textContent = fields.price;
    }
    if (changeEl) {
        changeEl.textContent = fields.changeDisplay;
        changeEl.className = `change ${fields.changeClass}`.trim();
        changeEl.title = fields.tooltip;
    }
    if (chartEl) {
        chartEl.className = `ticker-chart ${fields.changeClass}`.trim();
    }
    if (pathEl) {
        pathEl.setAttribute('d', fields.sparkPath);
    }
}

function patchTickerQuotes(quotes) {
    const tickerTrack = document.getElementById('tickerTrack');
    if (!tickerTrack || tickerTrack.dataset.tickerReady !== '1') {
        return false;
    }

    const quoteBySymbol = new Map(quotes.map((quote) => [quote.symbol, quote]));
    tickerTrack.querySelectorAll('.ticker-item[data-symbol]').forEach((item) => {
        const quote = quoteBySymbol.get(item.dataset.symbol);
        if (quote) {
            patchTickerItemElement(item, quote);
        }
    });
    return true;
}

function estimateTickerRepeats(quotes, marqueeWidth) {
    const minSetWidth = marqueeWidth + 80;
    const singlePassWidth = Math.max(quotes.length, 1) * TICKER_ESTIMATED_ITEM_WIDTH;
    return Math.max(3, Math.ceil(minSetWidth / singlePassWidth));
}

function renderTickerTrack(quotes) {
    const tickerTrack = document.getElementById('tickerTrack');
    const marqueeWidth = getTickerMarqueeWidth();
    if (!tickerTrack) {
        return;
    }

    stopTickerScroll();
    let repeats = estimateTickerRepeats(quotes, marqueeWidth);
    let setHtml = buildTickerSetHtml(quotes, repeats);

    tickerTrack.innerHTML =
        `<div class="ticker-set">${setHtml}</div>` +
        `<div class="ticker-set" aria-hidden="true">${setHtml}</div>`;

    const firstSet = tickerTrack.querySelector('.ticker-set');
    while (firstSet && firstSet.offsetWidth < marqueeWidth + 40 && repeats < 24) {
        repeats += 1;
        setHtml = buildTickerSetHtml(quotes, repeats);
        tickerTrack.innerHTML =
            `<div class="ticker-set">${setHtml}</div>` +
            `<div class="ticker-set" aria-hidden="true">${setHtml}</div>`;
    }

    tickerTrack.dataset.tickerReady = '1';
    scheduleTickerScrollStart();
}

/**
 * Update ticker bar with real market data (tiled for seamless scroll)
 */
function updateTickerDisplay(quotes) {
    latestTickerQuotes = quotes;
    if (patchTickerQuotes(quotes)) {
        return;
    }
    renderTickerTrack(quotes);
}

function setupTickerResizeHandler() {
    window.addEventListener('resize', () => {
        if (tickerResizeTimer) {
            clearTimeout(tickerResizeTimer);
        }

        tickerResizeTimer = setTimeout(() => {
            const tickerTrack = document.getElementById('tickerTrack');
            if (!tickerTrack || tickerTrack.dataset.tickerReady !== '1') {
                return;
            }

            const firstSet = tickerTrack.querySelector('.ticker-set');
            const marqueeWidth = getTickerMarqueeWidth();
            if (!firstSet || firstSet.offsetWidth < marqueeWidth + 40) {
                const sourceQuotes = latestTickerQuotes.length
                    ? latestTickerQuotes
                    : MAG7_TICKER_SYMBOLS.map((symbol) => ({ symbol, price: null, changePercent: null }));
                tickerTrack.dataset.tickerReady = '0';
                stopTickerScroll();
                renderTickerTrack(sourceQuotes);
            } else {
                tickerScrollSetWidth = getTickerSetWidth(tickerTrack);
            }
        }, 200);
    });
}

/**
 * Load live market data from Alpaca API (Magnificent 7)
 */
async function loadMarketTicker() {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    try {
        const symbols = MAG7_TICKER_SYMBOLS.join(',');
        const response = await fetch(`${API_BASE}/ticker?symbols=${symbols}`, {
            signal: controller.signal,
        });
        const data = await response.json().catch(() => ({}));

        if (data.quotes && data.quotes.length > 0) {
            updateTickerDisplay(data.quotes);
            console.log('✅ Market ticker updated:', data.quotes.length, 'symbols');
            return;
        }

        const message = data.error
            || (response.ok ? 'Market data temporarily unavailable' : `Market data unavailable (HTTP ${response.status})`);
        showTickerStatus(message);
        console.warn('Market ticker returned no quotes:', message);
    } catch (error) {
        const message = error.name === 'AbortError'
            ? 'Market data is taking longer than expected — retrying…'
            : 'Could not load market data';
        showTickerStatus(message);
        console.warn('Could not fetch market ticker:', error.message);
    } finally {
        clearTimeout(timeoutId);
    }
}

function showTickerStatus(message) {
    const tickerTrack = document.getElementById('tickerTrack');
    if (!tickerTrack || tickerTrack.dataset.tickerReady === '1') {
        return;
    }
    stopTickerScroll();
    tickerTrack.dataset.tickerReady = '0';
    tickerTrack.style.transform = 'none';
    tickerTrack.innerHTML = `<div class="ticker-placeholder">${escapeHtml(message)}</div>`;
}

/**
 * Update time period selection
 */
function updateTimePeriod(btn) {
    document.querySelectorAll('.time-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    console.log('Time period changed:', btn.textContent);
}


/**
 * Asset Universe Builder - Preset & Custom
 */

// Asset universe definitions
const ASSET_UNIVERSES = {
    djia: {
        name: 'DJIA 30',
        // Canonical Dow-30 — must mirror backend validator.DJIA_30
        // (pinned by dashboard/backend/tests/test_djia30_universe.py).
        assets: ['AAPL', 'AMGN', 'AMZN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
                 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
                 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'SHW', 'TRV', 'UNH', 'V', 'WMT']
    },
    mag7: {
        name: 'Magnificent 7',
        assets: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
    }
};

const IFIND_ASHARE_SOURCE = 'ifind_ashare';
const IFIND_ASHARE_TIMEFRAME = '60m';
const IFIND_ASHARE_START_DATE = '2026-04-01';
const IFIND_ASHARE_END_DATE = '2026-05-01';
const IFIND_ASHARE_DEFAULT_UNIVERSE = 'a_share_demo_6';
const RULE_BASED_DECISION_SOURCE = 'rule_based';
const LLM_DECISION_SOURCE = 'llm';
const IFIND_ASHARE_UNIVERSES = {
    a_share_demo_6: {
        name: 'A-Share Demo 6',
        allowedDecisionSources: ['rule_based', 'llm'],
        assets: [
            { symbol: '600519.SH', name: 'Kweichow Moutai' },
            { symbol: '601318.SH', name: 'Ping An Insurance' },
            { symbol: '600036.SH', name: 'China Merchants Bank' },
            { symbol: '000001.SZ', name: 'Ping An Bank' },
            { symbol: '000858.SZ', name: 'Wuliangye Yibin' },
            { symbol: '300750.SZ', name: 'CATL' },
        ],
    },
    csi300_sample_20_2026h2: {
        name: 'CSI 300 Sample 20 (2026 H2)',
        allowedDecisionSources: ['rule_based', 'llm'],
        assets: [
            { symbol: '600519.SH', name: 'Kweichow Moutai' },
            { symbol: '601318.SH', name: 'Ping An Insurance' },
            { symbol: '600036.SH', name: 'China Merchants Bank' },
            { symbol: '300750.SZ', name: 'CATL' },
            { symbol: '000333.SZ', name: 'Midea Group' },
            { symbol: '002594.SZ', name: 'BYD' },
            { symbol: '600276.SH', name: 'Hengrui Medicine' },
            { symbol: '300760.SZ', name: 'Mindray' },
            { symbol: '688981.SH', name: 'SMIC' },
            { symbol: '002415.SZ', name: 'Hikvision' },
            { symbol: '601766.SH', name: 'CRRC' },
            { symbol: '600309.SH', name: 'Wanhua Chemical' },
            { symbol: '601899.SH', name: 'Zijin Mining' },
            { symbol: '601857.SH', name: 'PetroChina' },
            { symbol: '600900.SH', name: 'China Yangtze Power' },
            { symbol: '600050.SH', name: 'China Unicom' },
            { symbol: '000725.SZ', name: 'BOE Technology' },
            { symbol: '600030.SH', name: 'CITIC Securities' },
            { symbol: '600887.SH', name: 'Yili' },
            { symbol: '600048.SH', name: 'Poly Developments' },
        ],
    },
};

function getSelectedIFindUniverse() {
    const value = document.getElementById('ifindAshareUniverseSelect')?.value;
    return IFIND_ASHARE_UNIVERSES[value]
        ? value
        : IFIND_ASHARE_DEFAULT_UNIVERSE;
}

function getIFindUniverseProfile(universe = getSelectedIFindUniverse()) {
    return IFIND_ASHARE_UNIVERSES[universe]
        || IFIND_ASHARE_UNIVERSES[IFIND_ASHARE_DEFAULT_UNIVERSE];
}

function syncIFindModelControl({ resetDecisionSource = false } = {}) {
    const modelSelect = document.getElementById('modelSelect');
    const modelSelectHint = document.getElementById('modelSelectHint');
    if (!modelSelect) return;

    const previousValue = modelSelect.value;
    let ruleOption = modelSelect.querySelector(
        `option[value="${RULE_BASED_DECISION_SOURCE}"]`,
    );
    if (!ruleOption) {
        ruleOption = document.createElement('option');
        ruleOption.value = RULE_BASED_DECISION_SOURCE;
        ruleOption.textContent = 'Rule-based';
        modelSelect.insertBefore(ruleOption, modelSelect.firstChild);
    }

    const profile = getIFindUniverseProfile();
    const allowsLLM = profile.allowedDecisionSources.includes(LLM_DECISION_SOURCE);
    if (!allowsLLM) {
        modelSelect.value = RULE_BASED_DECISION_SOURCE;
    } else if (resetDecisionSource) {
        const preferredModel = runBacktestModalAgent?.model_name || previousValue;
        const llmOptions = Array.from(modelSelect.options).filter(
            (option) => option.value !== RULE_BASED_DECISION_SOURCE,
        );
        const selectedOption = findBacktestModelOption(modelSelect, preferredModel)
            || llmOptions[0];
        if (selectedOption) modelSelect.value = selectedOption.value;
    }
    modelSelect.disabled = !allowsLLM;
    modelSelect.setAttribute('aria-disabled', String(!allowsLLM));
    if (modelSelectHint) {
        modelSelectHint.textContent = allowsLLM
            ? "Uses this agent's AI model by default. Choose Rule-based for repeatable decisions without AI."
            : 'This universe supports rule-based decisions only.';
    }
}

function renderIFindAshareUniverse({ resetDecisionSource = false } = {}) {
    const universe = getSelectedIFindUniverse();
    const profile = getIFindUniverseProfile(universe);
    const select = document.getElementById('ifindAshareUniverseSelect');
    const title = document.getElementById('ifindAshareUniverseTitle');
    const grid = document.getElementById('ifindAshareSymbolGrid');
    if (select) select.value = universe;
    if (title) title.textContent = `${profile.name} · ${profile.assets.length} stocks`;
    if (grid) {
        grid.innerHTML = profile.assets.map(({ symbol, name }) => `
            <div class="ifind-symbol-item" title="${escapeHtml(name)} (${escapeHtml(symbol)})">
                <span>${escapeHtml(symbol)}</span>
                <small>${escapeHtml(name)}</small>
            </div>
        `).join('');
        grid.setAttribute('aria-label', `${profile.name}, ${profile.assets.length} stocks`);
    }
    syncIFindModelControl({ resetDecisionSource });
}

// Popular stocks for autocomplete
// S&P 100 stocks
const POPULAR_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corp.',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon Inc.',
    'NVDA': 'NVIDIA Corp.',
    'TSLA': 'Tesla Inc.',
    'META': 'Meta Platforms',
    'BRK.B': 'Berkshire Hathaway',
    'JPM': 'JPMorgan Chase',
    'JNJ': 'Johnson & Johnson',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.',
    'PG': 'Procter & Gamble',
    'UNH': 'UnitedHealth Group',
    'HD': 'Home Depot',
    'MA': 'Mastercard',
    'DIS': 'Walt Disney',
    'PYPL': 'PayPal Inc.',
    'ADBE': 'Adobe Inc.',
    'CRM': 'Salesforce Inc.',
    'NFLX': 'Netflix Inc.',
    'BA': 'Boeing Co.',
    'KO': 'Coca-Cola Co.',
    'IBM': 'IBM Corp.',
    'INTC': 'Intel Corp.',
    'AMD': 'Advanced Micro Devices',
    'CSCO': 'Cisco Systems',
    'QCOM': 'Qualcomm',
    'VZ': 'Verizon Communications',
    'T': 'AT&T Inc.',
    'CAT': 'Caterpillar Inc.',
    'HON': 'Honeywell International',
    'MMM': '3M Company',
    'GE': 'General Electric',
    'AXP': 'American Express',
    'MCD': 'McDonalds Corp.',
    'PEP': 'PepsiCo Inc.',
    'KMB': 'Kimberly-Clark',
    'CL': 'Colgate-Palmolive',
    'SYK': 'Stryker Corporation',
    'LMT': 'Lockheed Martin',
    'PLD': 'Prologis Inc.',
    'AMT': 'American Tower',
    'PSA': 'Public Storage',
    'O': 'Realty Income',
    'DUK': 'Duke Energy',
    'SO': 'Southern Company',
    'NEE': 'NextEra Energy',
    'SCHW': 'Charles Schwab',
    'SPGI': 'S&P Global',
    'MCK': 'McKesson Corp.',
    'BX': 'Blackstone Inc.',
    'AIG': 'American International Group',
    'GD': 'General Dynamics',
    'LUV': 'Southwest Airlines',
    'UAL': 'United Airlines',
    'DAL': 'Delta Air Lines',
    'AAL': 'American Airlines',
    'COST': 'Costco Wholesale',
    'ABBV': 'AbbVie Inc.',
    'GILD': 'Gilead Sciences',
    'ISRG': 'Intuitive Surgical',
    'VEEV': 'Veeva Systems',
    'CRWD': 'CrowdStrike',
    'MU': 'Micron Technology',
    'AVGO': 'Broadcom Inc.',
    'INTU': 'Intuit Inc.',
    'AMAT': 'Applied Materials',
    'LRCX': 'Lam Research',
    'SNPS': 'Synopsys',
    'CDNS': 'Cadence Design',
    'NOW': 'ServiceNow',
    'SPLK': 'Splunk',
    'OKTA': 'Okta Inc.',
    'ZM': 'Zoom Video',
    'DOCU': 'DocuSign',
    'TWLO': 'Twilio',
    'DDOG': 'Datadog',
    'SNOW': 'Snowflake Inc.',
};

let selectedUniverse = 'djia'; // Default

function handleUniverseTabSwitch(tab) {
    const tabName = tab.dataset.tab;
    
    // Update tab buttons
    document.querySelectorAll('.universe-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    
    // Update content visibility explicitly
    const builtinTab = document.getElementById('builtinTab');
    const customTab = document.getElementById('customTab');
    
    if (tabName === 'builtin') {
        builtinTab.classList.add('active');
        builtinTab.style.display = 'block';
        customTab.classList.remove('active');
        customTab.style.display = 'none';
    } else {
        builtinTab.classList.remove('active');
        builtinTab.style.display = 'none';
        customTab.classList.add('active');
        customTab.style.display = 'block';
    }
    
    console.log(`Switched to ${tabName} universe tab`);
    notifyAssetUniverseChanged();
}

function selectPreset(preset) {
    if (!ASSET_UNIVERSES[preset]) {
        preset = 'djia';
    }

    selectedUniverse = preset;

    const djiaCard = document.getElementById('djiaCard');
    const mag7Card = document.getElementById('mag7Card');
    if (!djiaCard || !mag7Card) return;

    djiaCard.classList.remove('selected');
    mag7Card.classList.remove('selected');

    if (preset === 'djia') {
        djiaCard.classList.add('selected');
        djiaCard.querySelector('.preset-btn').textContent = 'Selected';
        mag7Card.querySelector('.preset-btn').textContent = 'Select';
    } else if (preset === 'mag7') {
        mag7Card.classList.add('selected');
        mag7Card.querySelector('.preset-btn').textContent = 'Selected';
        djiaCard.querySelector('.preset-btn').textContent = 'Select';
    }

    const universeData = ASSET_UNIVERSES[preset];
    console.log(`✅ Selected preset: ${universeData.name}`);
    notifyAssetUniverseChanged();
}

function handleAddAsset() {
    const input = document.getElementById('assetSearchInput');
    const ticker = input.value.trim().toUpperCase();
    
    if (!ticker) return;
    
    // Validate ticker (only alphanumeric, 1-5 chars)
    if (!/^[A-Z0-9]{1,5}$/.test(ticker)) {
        console.warn(`⚠️ Invalid ticker: ${ticker}`);
        return;
    }
    
    // Check if already added
    if (document.querySelector(`[data-ticker="${ticker}"]`)) {
        console.warn(`⚠️ ${ticker} already in custom universe`);
        input.value = '';
        return;
    }
    
    // Create chip
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.dataset.ticker = ticker;
    const companyName = POPULAR_STOCKS[ticker] || ticker;
    chip.innerHTML = `<span class="chip-ticker">${ticker}</span> <span class="chip-remove">×</span>`;
    chip.title = companyName;
    
    // Add remove listener
    chip.querySelector('.chip-remove').addEventListener('click', () => removeChip(chip));
    
    // Add to container
    document.getElementById('selectedChips').appendChild(chip);
    input.value = '';
    
    console.log(`✅ Added ${ticker} to custom universe`);
    notifyAssetUniverseChanged();
}

function removeChip(chipEl) {
    const ticker = chipEl.dataset.ticker;
    chipEl.remove();
    console.log(`❌ Removed ${ticker} from custom universe`);
    notifyAssetUniverseChanged();
}

function notifyAssetUniverseChanged() {
    document.dispatchEvent(new CustomEvent('asset-universe-changed'));
}

/**
 * Show autocomplete suggestions as user types
 */
function setupAssetSearch() {
    const searchInput = document.getElementById('assetSearchInput');
    let autocompleteDiv = null;
    
    if (!searchInput) return;
    
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim().toUpperCase();
        
        // Remove existing autocomplete
        if (autocompleteDiv) autocompleteDiv.remove();
        
        if (query.length === 0) return;
        
        // Filter matching stocks
        const matches = Object.entries(POPULAR_STOCKS)
            .filter(([ticker, name]) => 
                ticker.includes(query) || name.toUpperCase().includes(query)
            )
            .slice(0, 5); // Limit to 5 suggestions
        
        if (matches.length === 0) return;
        
        // Create autocomplete dropdown
        autocompleteDiv = document.createElement('div');
        autocompleteDiv.className = 'asset-autocomplete';
        
        matches.forEach(([ticker, name]) => {
            const option = document.createElement('div');
            option.className = 'autocomplete-option';
            option.innerHTML = `<strong>${ticker}</strong> - ${name}`;
            option.addEventListener('click', () => {
                searchInput.value = ticker;
                handleAddAsset();
                if (autocompleteDiv) autocompleteDiv.remove();
            });
            autocompleteDiv.appendChild(option);
        });
        
        const inputGroup = searchInput.closest('.search-input-group');
        inputGroup.appendChild(autocompleteDiv);
    });
    
    // Hide autocomplete when clicking elsewhere
    document.addEventListener('click', (e) => {
        if (e.target !== searchInput && autocompleteDiv) {
            autocompleteDiv.remove();
            autocompleteDiv = null;
        }
    });
}

/**
 * Run backtest
 */
/**
 * Get selected assets based on Preset or Custom tab
 */
function getSelectedAssets() {
    const dataSource = document.getElementById('marketDataSourceSelect')?.value;
    if (dataSource === IFIND_ASHARE_SOURCE) {
        return getIFindUniverseProfile().assets.map(({ symbol }) => symbol);
    }

    const builtinTab = document.getElementById('builtinTab');
    const isBuiltin = builtinTab?.classList.contains('active');
    
    if (!isBuiltin) {
        // Get chips from custom universe
        const chips = document.querySelectorAll('#selectedChips .chip');
        const assets = Array.from(chips).map(chip => chip.dataset.ticker);
        return assets.length > 0 ? assets : ['AAPL']; // Default fallback
    } else {
        // Get assets from selected built-in universe
        return ASSET_UNIVERSES[selectedUniverse].assets;
    }
}

function formatBacktestError(error, dataSource = null) {
    const source = dataSource || window.ACTIVE_BACKTEST_DATA_SOURCE || 'alpaca';
    const raw = String(error?.message || error?.detail || error || 'Backtest failed.');
    if (source !== IFIND_ASHARE_SOURCE) return raw;

    const status = Number(error?.status || 0);
    const lower = raw.toLowerCase();
    if (status === 403) return 'iFinD A-share access is disabled (403). Ask the server operator to enable it.';
    if (lower.includes('llm provider client is unavailable') || lower.includes('llm client is unavailable')) {
        return 'The selected AI provider is not configured. Configure the provider or choose Rule-based.';
    }
    if (status === 503) return 'iFinD A-share access is not configured (503). Ask the server operator to finish API setup.';
    if (status === 429 || lower.includes('429')) return 'iFinD is rate limited (429). Wait briefly, then run again.';
    if (
        lower.includes('50 bars')
        || lower.includes('fewer than 50')
        || lower.includes('at least 50')
        || lower.includes('minimum=50')
        || lower.includes('valid bars')
    ) {
        return 'iFinD returned fewer than 50 valid bars. Use a wider date range (about one month) or check data permissions.';
    }
    if (lower.includes('authentication') || lower.includes('credential') || lower.includes('permission') || lower.includes('token')) {
        return 'iFinD authentication or data permission failed. Ask the server operator to check the account.';
    }
    if (lower.includes('response') || lower.includes('tables') || lower.includes('structure') || lower.includes('format')) {
        return 'The iFinD response format was not recognized. Check the backend log for the sanitized summary.';
    }
    return 'The iFinD backtest failed. Check the backend log for the sanitized error summary.';
}

/**
 * Load the saved sub-agent pipeline for an agent (backend or localStorage).
 */
function loadAgentPipelineForBacktest(agent) {
    if (!agent) return null;
    if ((agent.runtime_type || 'pipeline') !== 'pipeline') return null;
    if (Array.isArray(agent.pipeline) && agent.pipeline.length) {
        return agent.pipeline;
    }
    if (!agent.agent_id || typeof agent.agent_id !== 'string') return null;
    try {
        const raw = localStorage.getItem(`agent-pipeline-config:${agent.agent_id}`);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed.subAgents) && parsed.subAgents.length) {
            return parsed.subAgents.map((sub) => ({
                id: sub.id,
                presetKey: sub.presetKey,
                label: sub.label,
                prompt: sub.prompt,
                outputFormat: sub.outputFormat,
            }));
        }
    } catch (error) {
        console.warn('Could not load local pipeline config:', error);
    }
    return null;
}

/**
 * Resolve the active agent object for backtest (API-backed or mock list).
 */
function resolveActiveAgentForBacktest() {
    if (window.ACTIVE_AGENT?.agent_id) {
        return window.ACTIVE_AGENT;
    }
    const activeId = localStorage.getItem(ACTIVE_AGENT_KEY);
    if (!activeId) return null;
    if (typeof allAgents !== 'undefined' && Array.isArray(allAgents)) {
        const found = allAgents.find((a) => a.agent_id === activeId);
        if (found) return found;
    }
    return null;
}

function formatBacktestElapsed(seconds) {
    const total = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(total / 60);
    const secs = total % 60;
    return `${minutes}:${String(secs).padStart(2, '0')}`;
}

/** A progress file older than this is reported as stale (seconds). */
const BACKTEST_STALE_SECONDS = 120;

/**
 * Coarse remaining-time estimate, or null when no honest one exists.
 *
 * Measured over an *observed* window -- seconds and steps counted from the first
 * step this client saw -- rather than over the whole run. Both elapsed clocks
 * start before any step exists (the card's at the click that fires the POST, the
 * panel's at the server's run start), so dividing the full span by the step
 * count folds process start, imports, the market-data fetch and gateway warm-up
 * into the per-step rate. With ~25s of startup and ~1s steps that reports
 * "~37m left" for a run that finishes in four, and the later collapse to
 * "~4m left" is itself an "is this broken?" signal.
 *
 * Suppressed below three observed steps: the first estimates swing wildly, and
 * a number that visibly jumps reads as broken. Coarse buckets thereafter -- a
 * precise-looking ETA that drifts is worse than an obviously approximate one.
 */
function formatBacktestEta(observedSeconds, observedSteps, remainingSteps) {
    const seconds = Number(observedSeconds);
    const done = Number(observedSteps);
    const left = Number(remainingSteps);
    if (!Number.isFinite(seconds) || seconds <= 0) return null;
    if (!Number.isFinite(done) || done < 3) return null;
    if (!Number.isFinite(left) || left <= 0) return null;
    const remaining = (seconds / done) * left;
    if (!Number.isFinite(remaining) || remaining <= 0) return null;
    if (remaining < 60) return '<1m left';
    return `~${Math.round(remaining / 60)}m left`;
}

/**
 * ETA for a running entry, anchored to the step this client first observed.
 *
 * `firstStep`/`firstStepAt` are stamped by the poller the first time a run
 * reports a step and then carried forward untouched, so launch cost never
 * enters the per-step rate. Both ends of the elapsed subtraction are Date.now()
 * reads on this machine, so -- unlike the server's elapsed_seconds -- it
 * carries no clock skew.
 */
function resolveBacktestEta(running) {
    const step = Number(running.step);
    const total = Number(running.totalSteps);
    const anchorStep = Number(running.firstStep);
    const anchorAt = Number(running.firstStepAt);
    if (!Number.isFinite(step) || !Number.isFinite(total) || total <= 0) return null;
    if (step <= 0 || step >= total) return null;
    if (!Number.isFinite(anchorStep) || !Number.isFinite(anchorAt)) return null;
    return formatBacktestEta((Date.now() - anchorAt) / 1000, step - anchorStep, total - step);
}

/**
 * Seconds since the progress file was last written, or null when unknown.
 *
 * The age arrives already computed from the server (`progress_age_seconds`)
 * rather than being derived from the mtime here. Differencing a server
 * timestamp against the browser clock makes any client more than
 * BACKTEST_STALE_SECONDS out of step indistinguishable from a wedged run: a
 * fast clock pins a permanent "No progress for 47m" onto a healthy backtest, a
 * slow one suppresses the warning forever. Only the time since *this* client
 * took the reading is added, which is a difference of two local Date.now()
 * calls and so skew-free.
 */
function resolveProgressAgeSeconds(running) {
    const age = running.ageSeconds;
    // typeof, not Number(): Number(null) is 0, so a coercing check would report
    // a run with no age reading at all as perfectly fresh.
    if (typeof age !== 'number' || !Number.isFinite(age) || age < 0) return null;
    const takenAt = Number(running.ageAt);
    const local = Number.isFinite(takenAt) ? Math.max(0, (Date.now() - takenAt) / 1000) : 0;
    return age + local;
}

/**
 * Staleness notice, or null while progress is fresh.
 *
 * Reports the *actual* gap, never the threshold: a message frozen at "2m" while
 * the real gap grows to ten actively misinforms. Deliberately does not say
 * "stuck" -- we know the file is old, not that the run died, and a long model
 * step looks exactly like this.
 */
function formatProgressStaleness(secondsSinceUpdate) {
    const gap = Number(secondsSinceUpdate);
    if (!Number.isFinite(gap) || gap < BACKTEST_STALE_SECONDS) return null;
    const minutes = Math.floor(gap / 60);
    return `No progress for ${minutes}m — long model steps can do this.`;
}

/**
 * Startup-phase counterpart of formatProgressStaleness.
 *
 * The likeliest wedge publishes *no* progress file at all: a subprocess that
 * dies or hangs in imports, the market-data fetch or the LLM gateway never
 * writes a step, so there is no mtime to age and the notice above can never
 * fire. That left the exact scenario this feature exists for -- "watched it and
 * could not tell running from stuck" -- as the one case with no signal on either
 * surface, for the full ten-minute poll ceiling.
 *
 * Same honesty constraint as the other notice: reports what is known (no steps
 * yet), not a diagnosis (dead).
 */
function formatStartupStaleness(elapsedSeconds) {
    const elapsed = Number(elapsedSeconds);
    if (!Number.isFinite(elapsed) || elapsed < BACKTEST_STALE_SECONDS) return null;
    const minutes = Math.floor(elapsed / 60);
    return `Still starting up — no steps reported after ${minutes}m.`;
}

/** Whichever staleness notice applies to this running entry, or null. */
function resolveRunningNotice(running) {
    const step = Number(running.step);
    if (!Number.isFinite(step) || step <= 0) {
        return formatStartupStaleness(running.elapsedSeconds);
    }
    const age = resolveProgressAgeSeconds(running);
    return age === null ? null : formatProgressStaleness(age);
}

/**
 * Fold a poll's `progress` payload into the shared live-progress store.
 *
 * `firstStep`/`firstStepAt` anchor the ETA to the first step this client saw and
 * are then carried forward untouched, so process start, imports, the
 * market-data fetch and gateway warm-up never enter the per-step rate. The
 * anchor resets with the store itself at every terminal branch, and again here
 * if a step count moves backwards -- which only happens when a fresh run's
 * first tick lands before the previous run was cleared.
 *
 * Split out of ensureBacktestPolling() so it can be exercised directly: an
 * anchor accidentally re-stamped on every tick would quietly restore the
 * launch-biased ETA while every other assertion stayed green.
 */
function advanceBacktestProgress(previous, progress, now) {
    const step = Number(progress?.step);
    const total = Number(progress?.total_steps);
    if (!Number.isFinite(step) || step <= 0) return null;
    const anchorStep = previous ? Number(previous.firstStep) : NaN;
    const anchorAt = previous ? Number(previous.firstStepAt) : NaN;
    const keepAnchor =
        Number.isFinite(anchorStep) && Number.isFinite(anchorAt) && anchorStep <= step;
    const age = Number(progress?.progress_age_seconds);
    return {
        step,
        totalSteps: total,
        // Server-computed (see resolveProgressAgeSeconds), and null when a
        // backend omits it -- which suppresses the staleness notice rather than
        // guessing at a value the payload never claimed.
        ageSeconds: Number.isFinite(age) ? age : null,
        ageAt: now,
        firstStep: keepAnchor ? anchorStep : step,
        firstStepAt: keepAnchor ? anchorAt : now,
    };
}

/**
 * Everything a running card reports, derived once for both renderers.
 *
 * renderAgentRunningBody() builds an HTML string and refreshRunningAgentCards()
 * mutates the live DOM, so they cannot share the emitting code -- but they must
 * never disagree about *what* to emit. Deriving here is what stops the next
 * added field from reaching only one of them, which is exactly how the bar,
 * aria-valuenow and the staleness note came to repaint on a full re-render and
 * never on the per-second patch that runs for the rest of the run.
 *
 * Text is returned as '' rather than null so the patch path can assign it
 * unconditionally -- a tick with no progress must *clear* the last numbers, not
 * leave them standing as though current. :empty hides the emptied nodes.
 */
function deriveRunningProgress(running) {
    const step = Number(running.step);
    const total = Number(running.totalSteps);
    const determinate =
        Number.isFinite(step) && Number.isFinite(total) && total > 0 && step > 0;
    const pct = determinate ? Math.min(99, Math.round((100 * step) / total)) : null;
    const eta = determinate ? resolveBacktestEta(running) : null;
    return {
        determinate,
        pct,
        eta,
        stepLabel: determinate ? `${step}/${total}` : '',
        // Deliberately excludes elapsed: the head already renders it one line
        // above, and printing "3:05" beside "3:05 elapsed" is the kind of noise
        // this change exists to remove. Built from raw values; escaping happens
        // once at each interpolation site.
        detail: [determinate ? `${pct}%` : null, eta].filter(Boolean).join(' · '),
        notice: resolveRunningNotice(running) || '',
    };
}

function showBacktestRunProgress(show, { isError = false } = {}) {
    const panel = document.getElementById('backtestRunProgress');
    if (!panel) return;
    panel.hidden = !show;
    panel.classList.toggle('is-error', !!isError);
    const title = panel.querySelector('.backtest-run-progress-title');
    const elapsed = panel.querySelector('.backtest-run-elapsed');
    const track = panel.querySelector('.backtest-run-progress-track');
    const hint = panel.querySelector('.backtest-run-progress-hint');
    if (title) title.textContent = isError ? 'Backtest did not start' : 'Backtest in progress';
    if (elapsed) elapsed.hidden = !!isError;
    if (track) track.hidden = !!isError;
    if (hint) hint.hidden = !!isError;
}

/**
 * Repaint the Backtest tab's run panel.
 *
 * `progress` is the live poller's shared progress object -- the same one the My
 * Agents card reads -- or null. Null at the five terminal call sites (launch
 * error, backtest error, completion, timeout): those render their own message
 * alone and must not gain an ETA or a "still starting up" notice. The running
 * branch always passes an object, `{}` included, which is what opts it into the
 * startup-staleness notice before any step exists.
 */
function updateBacktestRunProgress({
    elapsedSeconds,
    message = '',
    maxSeconds = BACKTEST_POLL_MAX_SECONDS,
    stepPct = null,
    progress = null,
} = {}) {
    const elapsedEl = document.getElementById('backtestRunElapsed');
    const messageEl = document.getElementById('backtestRunProgressMessage');
    const barEl = document.getElementById('backtestRunProgressBar');

    if (elapsedEl && elapsedSeconds !== undefined && elapsedSeconds !== null) {
        const elapsed = Math.max(0, Number(elapsedSeconds) || 0);
        elapsedEl.textContent = formatBacktestElapsed(elapsed);
    }
    if (messageEl && message) {
        // Same two derived facts the card shows, from the same helper fed the
        // same object -- so the ETA and the staleness notice cannot diverge
        // between the two surfaces. Elapsed still differs (the card's is
        // client-side from startedAt, this one is the server's elapsed_seconds)
        // but nothing derived from it does: the ETA is measured from the
        // poller's own step anchor, not from either elapsed clock.
        const view = progress
            ? deriveRunningProgress({ ...progress, elapsedSeconds })
            : null;
        messageEl.textContent = [message, view?.eta, view?.notice]
            .filter(Boolean)
            .join(' · ');
    }
    if (barEl) {
        const pct = Number.isFinite(stepPct)
            ? Math.min(99, Math.round(stepPct))
            : (elapsedSeconds !== undefined && elapsedSeconds !== null
                ? Math.min(95, Math.round((Math.max(0, Number(elapsedSeconds) || 0) / maxSeconds) * 100))
                : null);
        if (pct != null) barEl.style.width = `${pct}%`;
    }
}

function getPerformanceChartOptions(timestampMeta) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                display: false,
            },
            tooltip: {
                enabled: true,
                backgroundColor: 'rgba(0, 0, 0, 0.9)',
                titleColor: '#e5e7eb',
                bodyColor: '#e5e7eb',
                borderColor: '#1f2937',
                borderWidth: 1,
                padding: 12,
                displayColors: true,
                callbacks: {
                    title(context) {
                        if (context.length > 0) {
                            const dataIndex = context[0].dataIndex;
                            const timestamp = timestampMeta.timestamps[dataIndex];
                            try {
                                const date = new Date(timestamp);
                                const month = date.toLocaleString('en-US', { month: 'short' });
                                const day = date.getDate();
                                const hour = String(date.getHours()).padStart(2, '0');
                                return `${month} ${day} ${hour}:00`;
                            } catch (e) {
                                return timestamp;
                            }
                        }
                        return '';
                    },
                    label(context) {
                        const value = context.parsed.y;
                        return `${context.dataset.label}: $${value.toFixed(0)}`;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: false,
                ticks: {
                    color: '#e5e7eb',
                    font: { size: 11, weight: '500' },
                    callback(value) {
                        return '$' + value.toLocaleString();
                    }
                },
                grid: {
                    color: '#1f2937',
                    drawBorder: false,
                },
            },
            x: {
                ticks: {
                    color: '#e5e7eb',
                    font: { size: 11, weight: '500' }
                },
                grid: {
                    display: false,
                    drawBorder: false,
                }
            }
        }
    };
}

function initLiveBacktestChart() {
    const perfCtx = document.getElementById('performanceChart');
    if (!perfCtx || !perfCtx.getContext) return;

    if (chartInstance) {
        chartInstance.destroy();
        chartInstance = null;
    }

    liveBacktestChartMeta = { timestamps: [] };
    liveBacktestChartActive = true;
    backtestChartData = null;
    window.SELECTED_RUN = null;
    const ctx = perfCtx.getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Agent (live)',
                data: [],
                borderColor: '#4FC3F7',
                backgroundColor: 'transparent',
                borderWidth: 2.5,
                tension: 0,
                fill: false,
                pointRadius: 0,
                pointHoverRadius: 5,
            }],
        },
        options: getPerformanceChartOptions(liveBacktestChartMeta),
    });
    renderPerformanceLegend({
        columns: [{
            key: 'agent',
            label: 'Your Agent',
            color: '#4FC3F7',
            available: true,
        }],
    }, { disabled: true });
    setPerformanceComparisonState(
        'live',
        'Benchmark metrics will appear when this run finishes.',
    );
}

/** Clear history chart/metrics/log and pin the view to a soon-to-start live run. */
function prepareLiveBacktestView(launchConfig = null) {
    backtestSurfaceRequestSeq += 1;
    liveBacktestChartActive = true;
    liveBacktestRunId = null;
    liveBacktestLaunchPending = true;
    liveBacktestLaunchError = false;
    localStorage.removeItem(SELECTED_BACKTEST_RUN_KEY);
    const runSelect = document.getElementById('backtestRunSelect');
    if (runSelect) runSelect.value = '';
    clearPerformanceComparison(
        'live',
        'Benchmark metrics will appear when this run finishes.',
    );
    clearTradingLog('Backtest running… orders will appear here.');
    initLiveBacktestChart();
    renderBacktestRunConfig(null, { running: true, launchConfig });
    showBacktestRunProgress(true);
}

/** Switch the Backtest surface onto an in-flight run (chart + log + config). */
function attachToLiveBacktest(runId, progress = null, launchConfig = null) {
    if (!runId) return;
    backtestSurfaceRequestSeq += 1;
    liveBacktestLaunchPending = false;
    liveBacktestLaunchError = false;
    const alreadyLive =
        liveBacktestChartActive &&
        liveBacktestRunId === runId &&
        chartInstance &&
        chartInstance.data?.datasets?.[0]?.label === 'Agent (live)';

    liveBacktestRunId = runId;
    liveBacktestChartActive = true;
    localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, runId);
    const runSelect = document.getElementById('backtestRunSelect');
    if (runSelect) {
        // Ensure the Running option exists / is selected even if not in DB yet.
        if (![...runSelect.options].some((opt) => opt.value === runId)) {
            const cfg = launchConfig || getBacktestLaunchConfig(runId);
            const label = `Running… · ${cfg?.agentName || 'Agent'}`;
            const opt = document.createElement('option');
            opt.value = runId;
            opt.textContent = label;
            runSelect.insertBefore(opt, runSelect.firstChild);
        }
        runSelect.value = runId;
        runSelect.hidden = false;
    }
    clearPerformanceComparison(
        'live',
        'Benchmark metrics will appear when this run finishes.',
    );
    if (!alreadyLive) {
        initLiveBacktestChart();
    }
    renderBacktestRunConfig(
        { run_id: runId },
        { running: true, launchConfig: launchConfig || getBacktestLaunchConfig(runId) },
    );
    showBacktestRunProgress(true);
    if (progress) {
        updateLiveBacktestChart(progress);
        updateLiveTradingLog(progress);
        const step = Number(progress.step);
        const total = Number(progress.total_steps);
        const stepPct = Number.isFinite(step) && Number.isFinite(total) && total > 0
            ? (100 * step / total)
            : null;
        updateBacktestRunProgress({
            message: stepPct != null
                ? `Backtest running… step ${step}/${total} (${Math.round(stepPct)}%)`
                : 'Backtest is running…',
            stepPct,
        });
    } else if (!alreadyLive) {
        clearTradingLog('Backtest running… orders will appear here.');
    }
    ensureBacktestPolling();
}

/**
 * Paint a launch that was refused or failed.
 *
 * `runKey` is the registry key markAgentBacktestRunning() returned for THIS
 * launch, and is the only entry dropped: clearing by agent id used to delete an
 * earlier, genuinely running backtest for the same agent whenever a later launch
 * was refused.
 */
function showBacktestLaunchFailure(message, launchConfig, runKey = null) {
    if (runKey) {
        clearAgentBacktestRunning(runKey);
        applyAgentFilters(false);
    }
    backtestSurfaceRequestSeq += 1;
    liveBacktestChartActive = false;
    liveBacktestRunId = null;
    liveBacktestLaunchPending = false;
    liveBacktestLaunchError = true;
    renderBacktestRunConfig(null, { launchConfig, statusLabel: 'Failed' });
    clearPerformanceComparison('error', 'Backtest did not start.');
    clearTradingLog('Backtest did not start.');
    showBacktestRunProgress(true, { isError: true });
    updateBacktestRunProgress({ elapsedSeconds: 0, message });
    // The panel painted above lives under the Backtest tab, which is hidden
    // when the user is standing on My Agents -- the landing page after a
    // launch. Surface the failure where they actually are, using this
    // file's existing alert() convention for launch-time refusals (see
    // openRunBacktestModal / runBacktest) rather than inventing a new one.
    if (playgroundTab === 'agents' && currentPage === 'playground') {
        alert(message);
    }
}

function stopBacktestPolling() {
    if (backtestPollTimer) {
        clearInterval(backtestPollTimer);
        backtestPollTimer = null;
    }
    // Reset with the timer: counts left standing would spend a later run's
    // budget on failures that belong to a poller which is no longer attached.
    backtestPollFailures = Object.create(null);
}

function isViewingLiveBacktest(liveId = liveBacktestRunId) {
    if (!liveId) return false;
    return localStorage.getItem(SELECTED_BACKTEST_RUN_KEY) === liveId;
}

function ensureBacktestPolling() {
    if (backtestPollTimer) return;
    const maxAttempts = BACKTEST_POLL_MAX_SECONDS;
    let attempts = 0;

    backtestPollTimer = setInterval(async () => {
        attempts += 1;
        try {
            const jobs = [];
            const seen = new Set();
            listRunningBacktests().forEach((run) => {
                // A pending launch has no run id to poll yet; it is still in the
                // registry, so the stop check at the bottom keeps polling alive
                // until its POST answers.
                if (!run.runId || seen.has(run.runId)) return;
                seen.add(run.runId);
                jobs.push({ key: run.key, agentId: run.agentId, runId: run.runId });
            });
            if (liveBacktestRunId && !seen.has(liveBacktestRunId)) {
                jobs.push({ key: liveBacktestRunId, agentId: null, runId: liveBacktestRunId });
            }
            if (!jobs.length) {
                const statusUrl = `${API_BASE}/backtest/status`;
                const status = await API.get(statusUrl);
                if (!status?.running) {
                    stopBacktestPolling();
                    return;
                }
                jobs.push({ key: null, agentId: null, runId: status.live_run_id || null });
            }

            const snapshots = await Promise.all(jobs.map(async (job) => {
                const statusUrl = job.runId
                    ? `${API_BASE}/backtest/status?live_run_id=${encodeURIComponent(job.runId)}`
                    : `${API_BASE}/backtest/status`;
                try {
                    return { job, status: await API.get(statusUrl), failed: false };
                } catch (error) {
                    // Carried as an explicit failure rather than a bare null: a
                    // request that never answered says nothing about whether the
                    // run ended, and only the terminal branch below may act as
                    // though it did.
                    console.error('Error polling backtest status:', error);
                    return { job, status: null, failed: true };
                }
            }));

            let anyRunning = false;
            let finishedFocused = null;

            for (const { job, status, failed } of snapshots) {
                if (failed || !status) {
                    const failureKey = job.runId || job.key || '';
                    const misses = (backtestPollFailures[failureKey] || 0) + 1;
                    backtestPollFailures[failureKey] = misses;
                    if (misses < BACKTEST_POLL_FAILURE_BUDGET) {
                        // Still running as far as anyone knows: hold the card and
                        // keep the poller attached so a blip on one job cannot end
                        // polling for the healthy ones.
                        anyRunning = true;
                        continue;
                    }
                    // Budget spent -- give up on this run, and say so. Dropping the
                    // card silently is how a backtest that is still going
                    // server-side comes to look like one that never started.
                    delete backtestPollFailures[failureKey];
                    const wasViewed = isViewingLiveBacktest(job.runId);
                    if (job.key) clearAgentBacktestRunning(job.key);
                    if (job.runId) delete liveBacktestProgressByRunId[job.runId];
                    if (job.runId && job.runId === liveBacktestRunId) {
                        liveBacktestRunId = null;
                        liveBacktestProgress = null;
                        liveBacktestChartActive = false;
                    }
                    const lostMessage = 'Lost contact with the backtest — it may still be running. Reload to check.';
                    if (wasViewed) {
                        showBacktestRunProgress(true, { isError: true });
                        updateBacktestRunProgress({ elapsedSeconds: attempts, message: lostMessage });
                    } else {
                        showAppToast(lostMessage);
                    }
                    continue;
                }
                delete backtestPollFailures[job.runId || job.key || ''];
                const liveId = status.live_run_id || job.runId;
                const serverElapsed = Number(status.elapsed_seconds);
                const displayElapsed = Number.isFinite(serverElapsed) && serverElapsed > 0
                    ? serverElapsed
                    : attempts;
                const viewingLive = isViewingLiveBacktest(liveId);

                if (status.running) {
                    anyRunning = true;
                    // Adopt an unfocused run only when the Backtest panel is not
                    // pinned to some other run. Adopting regardless meant that
                    // after run A finished and loaded its results, run B became
                    // the focused run and took the panel over the moment it
                    // completed -- replacing the results the user was reading.
                    const pinnedRunId = localStorage.getItem(SELECTED_BACKTEST_RUN_KEY);
                    if (liveId && !liveBacktestRunId && (!pinnedRunId || pinnedRunId === liveId)) {
                        liveBacktestRunId = liveId;
                    }
                    const step = Number(status.progress?.step);
                    const total = Number(status.progress?.total_steps);
                    const stepPct = Number.isFinite(step) && Number.isFinite(total) && total > 0
                        ? (100 * step / total)
                        : null;
                    // Assigned BEFORE refreshRunningAgentCards() below, which reads
                    // it through getAgentBacktestRunning(). Painting first would
                    // show the previous tick's step on the card while the Backtest
                    // panel — handed the same object a few lines down — showed this
                    // tick's: two surfaces disagreeing by one poll.
                    if (liveId) {
                        liveBacktestProgressByRunId[liveId] = advanceBacktestProgress(
                            liveBacktestProgressByRunId[liveId] || null,
                            status.progress,
                            Date.now(),
                        );
                    }

                    if (viewingLive) {
                        liveBacktestChartActive = true;
                        if (status.progress) {
                            updateLiveBacktestChart(status.progress);
                            updateLiveTradingLog(status.progress);
                        }
                        updateBacktestRunProgress({
                            elapsedSeconds: displayElapsed,
                            message: status.message || 'Backtest is running…',
                            stepPct,
                            // `{}` rather than null before the first step: an empty
                            // object still opts this surface into the startup
                            // staleness notice, which is the only warning available
                            // while the subprocess has published nothing.
                            progress: liveBacktestProgressByRunId[liveId] || liveBacktestProgress || {},
                        });
                        showBacktestRunProgress(true);
                        renderBacktestRunConfig(
                            { run_id: liveId },
                            { running: true, launchConfig: getBacktestLaunchConfig(liveId) },
                        );
                    }
                } else {
                    if (liveId) delete liveBacktestProgressByRunId[liveId];
                    // Only this run's entry: the registry is keyed by run, so a
                    // sibling run of the same agent keeps its card. `liveId` is
                    // the key for anything this build filed; job.key also clears
                    // an entry a previous build left filed under its agent id.
                    if (liveId) clearAgentBacktestRunning(liveId);
                    if (job.key && job.key !== liveId) clearAgentBacktestRunning(job.key);
                    if (viewingLive || liveId === liveBacktestRunId) {
                        finishedFocused = {
                            status,
                            liveId,
                            displayElapsed,
                            finishedId: liveId || liveBacktestRunId,
                        };
                    }
                }
            }

            // Focused-run mirror for the Backtest panel + single-run harnesses.
            liveBacktestProgress = liveBacktestRunId
                ? (liveBacktestProgressByRunId[liveBacktestRunId] || null)
                : null;

            // Repaint My Agents cards even when the user is not on the Backtest
            // tab — that page is the landing page after launch.
            if (playgroundTab === 'agents' && currentPage === 'playground') {
                refreshRunningAgentCards();
            }

            if (finishedFocused) {
                const { status, liveId, displayElapsed, finishedId } = finishedFocused;
                liveBacktestChartActive = false;
                if (liveBacktestRunId === finishedId) {
                    liveBacktestRunId = null;
                    liveBacktestProgress = null;
                }
                lastRenderedRunningKey = null;
                if (playgroundTab === 'agents' && currentPage === 'playground') {
                    loadAgents();
                }

                if (status.error) {
                    const source = getBacktestLaunchConfig(liveId || finishedId)?.dataSource;
                    const message = formatBacktestError(status.error, source);
                    showBacktestRunProgress(true, { isError: true });
                    updateBacktestRunProgress({
                        elapsedSeconds: displayElapsed,
                        message,
                    });
                } else if (status.success) {
                    updateBacktestRunProgress({
                        elapsedSeconds: displayElapsed,
                        message: `Completed in ${formatBacktestElapsed(displayElapsed)}.`,
                    });
                    if (finishedId) {
                        localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, finishedId);
                    } else {
                        localStorage.removeItem(SELECTED_BACKTEST_RUN_KEY);
                    }
                    await loadData();
                    setTimeout(() => showBacktestRunProgress(false), 2500);
                } else {
                    showBacktestRunProgress(false);
                }

                // Deliberately does NOT re-attach the view to another still-running
                // run. attachToLiveBacktest() rewrites SELECTED_BACKTEST_RUN_KEY
                // and repaints the panel into live mode, which threw away the
                // finished run's results loaded a few lines above -- the ones the
                // user had been waiting for. The other runs keep polling and their
                // cards keep updating; the run dropdown is how a user follows a
                // different one, on purpose.
            }

            // Stop only when there is nothing left to watch. A run whose poll
            // failed still counts as running above, so a transient network error
            // cannot end polling for the healthy jobs; a pending launch has no
            // status to report yet but is still in the registry.
            if (!anyRunning && !listRunningBacktests().length) {
                stopBacktestPolling();
            }

            if (attempts >= maxAttempts) {
                stopBacktestPolling();
                if (isViewingLiveBacktest(liveBacktestRunId)) {
                    showBacktestRunProgress(true, { isError: true });
                    updateBacktestRunProgress({
                        elapsedSeconds: maxAttempts,
                        message: 'Timed out after 60 minutes. The backtest may still be running in the background.',
                    });
                }
                liveBacktestChartActive = false;
                liveBacktestRunId = null;
                // The finished branch above clears finished runs; this one must
                // clear every orphan so a stale card cannot pick up the NEXT
                // run's step/percent from a leftover map entry.
                Object.keys(readRunningBacktests()).forEach(clearAgentBacktestRunning);
                liveBacktestProgressByRunId = Object.create(null);
                liveBacktestProgress = null;
                lastRenderedRunningKey = null;
                // Clearing the map is not visible on its own: polling has just
                // stopped, so refreshRunningAgentCards() will never run again
                // and the card would sit on "Backtesting…" with a frozen timer
                // until some unrelated re-render happened by. Same repaint the
                // finished branch does.
                if (playgroundTab === 'agents' && currentPage === 'playground') {
                    loadAgents();
                }
            }
        } catch (error) {
            console.error('Error polling backtest status:', error);
        }
    }, 1000);
}

function updateLiveBacktestChart(progress) {
    if (!liveBacktestChartActive || !chartInstance || !progress) return;

    const curve = progress.equity_curve;
    if (!Array.isArray(curve) || curve.length === 0) return;

    liveBacktestChartMeta.timestamps = curve.map((point) => point.timestamp);
    chartInstance.data.labels = formatTimestamps(liveBacktestChartMeta.timestamps);
    chartInstance.data.datasets[0].data = curve.map((point) => point.equity);
    chartInstance.update('none');
}

function orderEventMatchKey(record) {
    const side = String(record?.side || record?.action || '').toUpperCase();
    return `${record?.timestamp ?? ''}|${record?.symbol ?? ''}|${side}`;
}

/**
 * Reassemble the run's full order history from the two complementary lists.
 *
 * `trades` is every fill, uncapped, straight from the trades table.
 * `order_events` carries only the orders that did NOT fill cleanly, because
 * duplicating fills into the bounded metadata sample is what would make that
 * sample lossy (see `engine._unfilled_order_events`). Preferring one list over
 * the other — as this function first did — therefore hides real rows either
 * way: take `order_events` alone and every fill disappears; take `trades`
 * alone and every rejection does.
 *
 * A partial fill is the one order that appears in both: the trade row records
 * what executed, the order event records the shortfall and its reason. The
 * event is a strict superset, so it replaces its trade rather than adding a
 * second row for the same order.
 */
function resolveTradingLogRecords(payload) {
    const trades = Array.isArray(payload?.trades) ? payload.trades : [];
    const orderEvents = Array.isArray(payload?.order_events) ? payload.order_events : [];
    if (orderEvents.length === 0) return trades;

    const partialsByKey = new Map();
    const standalone = [];
    for (const event of orderEvents) {
        // Executed nothing => no trade row exists to merge with.
        if (Number(event?.executed_shares ?? 0) > 0) {
            const key = orderEventMatchKey(event);
            if (!partialsByKey.has(key)) partialsByKey.set(key, []);
            partialsByKey.get(key).push(event);
        } else {
            standalone.push(event);
        }
    }

    const merged = trades.map((trade) => {
        const queue = partialsByKey.get(orderEventMatchKey(trade));
        return queue && queue.length ? queue.shift() : trade;
    });
    // Any partial with no matching trade still belongs in the log — dropping it
    // would be the same silent loss this merge exists to prevent.
    for (const queue of partialsByKey.values()) merged.push(...queue);
    merged.push(...standalone);

    return merged.sort((a, b) => {
        const left = Date.parse(a?.timestamp) || 0;
        const right = Date.parse(b?.timestamp) || 0;
        return left - right;
    });
}

/** How many non-filled orders the server had to drop from its bounded sample. */
function resolveTradingLogTruncation(payload) {
    const returned = Array.isArray(payload?.order_events) ? payload.order_events.length : 0;
    const explicit = Number(payload?.order_events_truncated);
    if (Number.isFinite(explicit) && explicit > 0) return Math.trunc(explicit);
    const total = Number(payload?.order_events_count ?? payload?.order_event_count);
    if (Number.isFinite(total) && total > returned) return Math.trunc(total - returned);
    return 0;
}

function resolveTradingAssetName(symbol) {
    for (const profile of Object.values(IFIND_ASHARE_UNIVERSES)) {
        const asset = profile.assets.find((item) => item.symbol === symbol);
        if (asset) return asset.name;
    }
    return POPULAR_STOCKS[symbol] || '';
}

function formatOrderExecutionReason(reason, strategyReason = '') {
    const labels = {
        invalid_lot_size: 'Invalid lot size',
        insufficient_cash_for_lot: 'Insufficient cash for one lot',
        insufficient_cash: 'Insufficient cash',
        t1_frozen: 'T+1 frozen',
        insufficient_position: 'Insufficient position',
        suspended: 'Suspended',
        limit_up_buy_blocked: 'Buy blocked at upper limit',
        limit_down_sell_blocked: 'Sell blocked at lower limit',
        market_rule_unavailable: 'No market rule for this symbol',
    };
    const code = String(reason || '').trim();
    if (labels[code]) return labels[code];
    if (code) return 'Order not executed';
    return String(strategyReason || '').trim() || '--';
}

function normalizeOrderRecord(record) {
    const optionalNumber = (value) => {
        if (value == null || value === '') return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    };
    const side = String(record?.side || record?.action || '').toUpperCase();
    const legacyQuantity = Number(record?.quantity ?? record?.shares ?? 0);
    const requestedValue = Number(record?.requested_shares ?? legacyQuantity);
    const executedValue = Number(record?.executed_shares ?? legacyQuantity);
    const requestedShares = Number.isFinite(requestedValue) ? requestedValue : 0;
    const executedShares = Number.isFinite(executedValue) ? executedValue : 0;
    const rawPrice = Number(record?.price ?? 0);
    const price = Number.isFinite(rawPrice) ? rawPrice : 0;
    const rawValue = Number(
        record?.executed_value
        ?? record?.value
        ?? record?.total_value
        ?? record?.cost
        ?? record?.proceeds
        ?? executedShares * price
    );
    const value = Number.isFinite(rawValue) ? rawValue : 0;
    const rawStatus = String(record?.status || '').toLowerCase();
    const status = ['filled', 'partial', 'rejected'].includes(rawStatus)
        ? rawStatus
        : rawStatus
            ? 'rejected'
            : 'filled';
    return {
        timestamp: record?.timestamp,
        side,
        symbol: record?.symbol || '--',
        requestedShares,
        executedShares,
        price,
        value,
        status,
        reason: status === 'filled' ? '' : (record?.reason || ''),
        strategyReason: record?.strategy_reason
            || (status === 'filled' ? (record?.reason || '') : ''),
        repeatCount: Math.max(Math.trunc(Number(record?.repeat_count) || 1), 1),
        nativePrice: record?.native_price == null ? null : Number(record.native_price),
        nativeValue: record?.native_value == null ? null : Number(record.native_value),
        fxRate: record?.fx_rate == null ? null : Number(record.fx_rate),
        referencePrice: optionalNumber(record?.reference_price),
        grossValue: optionalNumber(record?.gross_value),
        slippageAmount: optionalNumber(record?.slippage_amount),
        commission: optionalNumber(record?.commission),
        stampDuty: optionalNumber(record?.stamp_duty),
        transferFee: optionalNumber(record?.transfer_fee),
        totalFees: optionalNumber(record?.total_fees),
        netCashImpact: optionalNumber(record?.net_cash_impact),
        nativeReferencePrice: optionalNumber(record?.native_reference_price),
        nativeGrossValue: optionalNumber(record?.native_gross_value),
        nativeSlippageAmount: optionalNumber(record?.native_slippage_amount),
        nativeCommission: optionalNumber(record?.native_commission),
        nativeStampDuty: optionalNumber(record?.native_stamp_duty),
        nativeTransferFee: optionalNumber(record?.native_transfer_fee),
        nativeTotalFees: optionalNumber(record?.native_total_fees),
        nativeNetCashImpact: optionalNumber(record?.native_net_cash_impact),
        marketRuleDate: record?.market_rule_date || null,
        marketRuleSuspended: record?.market_rule_suspended === true
            || record?.market_rule_suspended === 1,
        marketRuleClosingLimitState: record?.market_rule_closing_limit_state || null,
        marketRuleOfficialClose: optionalNumber(record?.market_rule_official_close),
        marketRuleClosingGateEffective:
            record?.market_rule_closing_gate_effective === true
            || record?.market_rule_closing_gate_effective === 1,
    };
}

// Only rows a rule actually spoke to. An ordinary fill on an ordinary day
// carries the same audit payload as a blocked one, so rendering whenever the
// official close is present puts a date-and-price line under every single
// A-share order and buries the handful that mean something.
function renderMarketRuleAudit(order) {
    if (!order.marketRuleDate) return '';
    const details = [];
    if (order.marketRuleSuspended) {
        details.push('Official status: suspended');
    } else if (order.marketRuleClosingLimitState
        && order.marketRuleClosingLimitState !== 'none') {
        const side = order.marketRuleClosingLimitState === 'upper' ? 'upper' : 'lower';
        details.push(`Official close: ${side} limit`);
    }
    if (!details.length) return '';
    if (Number.isFinite(order.marketRuleOfficialClose)) {
        details.push(`¥${order.marketRuleOfficialClose.toFixed(2)}`);
    }
    return `<div class="trading-log-native">${escapeHtml(order.marketRuleDate)} · ${escapeHtml(details.join(' · '))}</div>`;
}

function formatTradingMoney(value, symbol) {
    const amount = Number(value);
    if (!Number.isFinite(amount)) return '--';
    const sign = amount < 0 ? '-' : '';
    return `${sign}${symbol}${Math.abs(amount).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    })}`;
}

function renderOrderCostAudit(order) {
    if (order.status === 'rejected' || order.executedShares <= 0) return '';
    const reportingCosts = [
        ['Commission', order.commission],
        ['Stamp duty', order.stampDuty],
        ['Transfer fee', order.transferFee],
        ['Slippage', order.slippageAmount],
        ['Net cash', order.netCashImpact],
    ];
    if (!reportingCosts.some(([, value]) => Number.isFinite(value))) return '';

    const reportingLine = reportingCosts
        .filter(([, value]) => Number.isFinite(value))
        .map(([label, value]) => `<span>${label} ${formatTradingMoney(value, '$')}</span>`)
        .join('');
    const nativeCosts = [
        ['Commission', order.nativeCommission],
        ['Stamp duty', order.nativeStampDuty],
        ['Transfer fee', order.nativeTransferFee],
        ['Slippage', order.nativeSlippageAmount],
        ['Net cash', order.nativeNetCashImpact],
    ];
    const nativeLine = nativeCosts.some(([, value]) => Number.isFinite(value))
        ? `<div class="trading-log-native trading-log-native-costs"><strong>CNY native</strong>${nativeCosts
            .filter(([, value]) => Number.isFinite(value))
            .map(([label, value]) => `<span>${label} ${formatTradingMoney(value, '¥')}</span>`)
            .join('')}</div>`
        : '';
    return `<div class="trading-log-costs">${reportingLine}</div>${nativeLine}`;
}

function formatTradeTimestamp(ts) {
    if (!ts) return '--';
    try {
        const date = new Date(ts);
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
        });
    } catch (e) {
        return String(ts);
    }
}

/**
 * Render already-normalized rows.
 *
 * Split out from `renderTradingLog` because the filter control re-renders from
 * `tradingLogCache`, which holds normalized records. Feeding those back through
 * `normalizeOrderRecord` would re-read wire-format keys (`requested_shares`,
 * `native_price`, …) that a normalized record does not carry, silently zeroing
 * every quantity and dropping the currency audit the moment a user filters.
 */
function paintTradingLog(normalizedRecords, options) {
    const feed = document.getElementById('tradingLogFeed');
    if (!feed) return;
    options = options || {};
    const emptyMessage = options.emptyMessage || 'No orders yet.';

    let filtered = normalizedRecords;
    if (tradingLogFilter === 'buy') {
        filtered = normalizedRecords.filter((trade) => trade.side === 'BUY');
    } else if (tradingLogFilter === 'sell') {
        filtered = normalizedRecords.filter((trade) => trade.side === 'SELL');
    }

    renderTradingLogSummary(filtered);

    if (filtered.length === 0) {
        feed.innerHTML = `<p class="trading-log-empty">${escapeHtml(emptyMessage)}</p>`;
        return;
    }

    feed.innerHTML = filtered.map((order) => {
        const actionClass = order.side === 'SELL' ? 'action-sell' : 'action-buy';
        const actionLabel = order.side === 'SELL' ? 'SELL' : 'BUY';
        const statusLabel = order.status.toUpperCase();
        const hasNativeAudit = Number.isFinite(order.nativePrice)
            && Number.isFinite(order.nativeValue)
            && Number.isFinite(order.fxRate);
        const priceAudit = hasNativeAudit
            ? `<div class="trading-log-native">¥${order.nativePrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>`
            : '';
        const valueAudit = hasNativeAudit && order.executedShares > 0
            ? `<div class="trading-log-native">¥${order.nativeValue.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} · FX ${order.fxRate.toFixed(4)}</div>`
            : '';
        const costAudit = renderOrderCostAudit(order);
        const assetName = resolveTradingAssetName(order.symbol);
        const quantity = `${order.executedShares.toLocaleString('en-US')} / ${order.requestedShares.toLocaleString('en-US')} shares`;
        const reason = formatOrderExecutionReason(order.reason, order.strategyReason);
        const marketRuleAudit = renderMarketRuleAudit(order);
        // A rejection the agent re-issued on every bar of a day is stored once,
        // with its tally. Showing the tally is the difference between "this
        // blocked one order" and "this blocked the strategy all day".
        const repeatNote = order.repeatCount > 1
            ? `<div class="trading-log-native">×${order.repeatCount} that day</div>`
            : '';
        return `<article class="trading-log-event" role="listitem">
            <div class="trading-log-event-head">
                <span class="trading-log-action ${actionClass}">${actionLabel}</span>
                <div class="trading-log-asset">
                    <strong>${escapeHtml(order.symbol)}</strong>
                    ${assetName ? `<small>${escapeHtml(assetName)}</small>` : ''}
                    <time datetime="${escapeHtml(order.timestamp || '')}">${escapeHtml(formatTradeTimestamp(order.timestamp))}</time>
                </div>
                <span class="order-status order-status-${order.status}" aria-label="Order status: ${statusLabel}">${statusLabel}</span>
            </div>
            <div class="trading-log-event-meta">
                <span><small>Filled / requested</small>${escapeHtml(quantity)}</span>
                <span><small>Execution price</small>$${order.price.toFixed(2)}${priceAudit}</span>
                <span><small>Filled value</small>${order.executedShares > 0 ? `${formatTradingMoney(order.value, '$')}${valueAudit}` : '--'}</span>
            </div>
            ${costAudit}
            <p class="trading-log-reason"><span class="trading-log-reason-label">Reason</span>${escapeHtml(reason)}${marketRuleAudit}${repeatNote}</p>
        </article>`;
    }).join('');

    // Never let a capped list read like a complete one. The server bounds the
    // non-filled sample, so when it drops records the table must say so rather
    // than quietly ending early.
    const truncated = Math.max(Math.trunc(Number(options.truncatedCount) || 0), 0);
    if (truncated > 0) {
        const note = `${truncated.toLocaleString('en-US')} more unfilled `
            + `${truncated === 1 ? 'order is' : 'orders are'} not shown `
            + '(audit sample capped).';
        feed.innerHTML += `<p class="trading-log-empty trading-log-truncation">${escapeHtml(note)}</p>`;
    }
}

function renderTradingLogSummary(filtered) {
    const countEl = document.getElementById('tradingLogCount');
    const summaryEl = document.getElementById('tradingLogStatusSummary');
    const records = Array.isArray(filtered) ? filtered : [];
    const count = records.length;
    if (countEl) countEl.textContent = `${count.toLocaleString('en-US')} ${count === 1 ? 'order' : 'orders'}`;
    if (summaryEl) {
        const counts = records.reduce((summary, order) => {
            if (order.status === 'filled') summary.filled += 1;
            else if (order.status === 'partial') summary.partial += 1;
            else if (order.status === 'rejected') summary.rejected += 1;
            return summary;
        }, { filled: 0, partial: 0, rejected: 0 });
        summaryEl.textContent = `${counts.filled} filled · ${counts.partial} partial · ${counts.rejected} rejected`;
    }
}

function renderTradingLog(records, options) {
    options = options || {};
    tradingLogCache = Array.isArray(records) ? records.map(normalizeOrderRecord) : [];
    tradingLogEmptyMessage = options.emptyMessage || 'No orders yet.';
    tradingLogTruncatedCount = Math.max(
        Math.trunc(Number(options.truncatedCount) || 0), 0
    );
    paintTradingLog(tradingLogCache, {
        emptyMessage: tradingLogEmptyMessage,
        truncatedCount: tradingLogTruncatedCount,
    });
}

function clearTradingLog(message = 'Waiting for orders…') {
    renderTradingLog([], { emptyMessage: message });
}

function updateLiveTradingLog(progress) {
    if (!Array.isArray(progress?.order_events) && !Array.isArray(progress?.trades)) return;
    renderTradingLog(resolveTradingLogRecords(progress), {
        truncatedCount: resolveTradingLogTruncation(progress),
    });
}

async function loadTradingLogForRun(runId, { isCurrent = () => true } = {}) {
    if (!runId) {
        if (isCurrent()) clearTradingLog('Run a backtest to see orders here.');
        return;
    }
    try {
        const data = await API.get(`${API_BASE}/runs/${encodeURIComponent(runId)}/trades?t=${Date.now()}`);
        if (!isCurrent()) return;
        renderTradingLog(resolveTradingLogRecords(data), {
            emptyMessage: 'No orders were submitted by the selected strategy.',
            truncatedCount: resolveTradingLogTruncation(data),
        });
    } catch (error) {
        if (!isCurrent()) return;
        console.warn('Could not load orders:', error.message);
        clearTradingLog('Order log unavailable for this run.');
    }
}

const BACKTEST_LAUNCH_CONFIG_KEY = 'backtest-launch-configs';
const PENDING_BYOK_STORAGE_KEY = 'atlPendingByokBacktest';
/** @type {null | object} */
let runBacktestModalAgent = null;
let runBacktestExecutionOptions = [];
let runBacktestBillingMode = null;
let runBacktestOptionsReady = false;

function readPendingByokBacktest() {
    let parsed = null;
    try {
        parsed = JSON.parse(
            sessionStorage.getItem(PENDING_BYOK_STORAGE_KEY) || 'null',
        );
    } catch (_error) {
        parsed = null;
    }
    const valid = (
        parsed
        && parsed.billing_mode === 'byok'
        && /^[a-z0-9_]{2,64}$/.test(String(parsed.provider_id || ''))
        && /^[A-Za-z0-9][A-Za-z0-9._\/-]{0,63}$/.test(
            String(parsed.model_id || ''),
        )
        && Number.isFinite(Number(parsed.expires_at))
        && Number(parsed.expires_at) > Date.now()
    );
    if (!valid) {
        clearPendingByokBacktest();
        return null;
    }
    return parsed;
}

function clearPendingByokBacktest() {
    try {
        sessionStorage.removeItem(PENDING_BYOK_STORAGE_KEY);
    } catch (_error) {
        /* Browser storage may be unavailable in hardened contexts. */
    }
}

function runBacktestExecutionOption(providerId) {
    return runBacktestExecutionOptions.find(
        (option) => option.provider_id === providerId,
    ) || null;
}

function runBacktestLaneAvailable(option, billingMode) {
    if (!option || !Array.isArray(option.models) || option.models.length === 0) {
        return false;
    }
    return billingMode === 'byok'
        ? option.byok_available === true
        : (
            billingMode === 'platform_credits'
            && option.platform_credits_available === true
        );
}

function findRunBacktestExecutionModel(option, modelId) {
    const normalized = normalizeBacktestModelId(modelId);
    if (!normalized) return null;
    return (option?.models || []).find(
        (model) => normalizeBacktestModelId(model.model_id) === normalized,
    ) || null;
}

function availableRunBacktestProviders(billingMode) {
    return runBacktestExecutionOptions.filter(
        (option) => runBacktestLaneAvailable(option, billingMode),
    );
}

function clearSelectOptions(select) {
    if (!select) return;
    while (select.firstChild) select.removeChild(select.firstChild);
}

function syncRunBacktestSubmitAvailability() {
    const submit = document.getElementById('runBacktestModalSubmit');
    if (!submit) return;
    const agent = runBacktestModalAgent;
    const isHostedRuntime = (agent?.runtime_type || 'pipeline') !== 'pipeline';
    const dataSource = document.getElementById('marketDataSourceSelect')?.value || 'alpaca';
    const modelId = document.getElementById('modelSelect')?.value || '';
    const isRuleBased = (
        dataSource === 'vnpy_simulation'
        || (
            dataSource === IFIND_ASHARE_SOURCE
            && modelId === RULE_BASED_DECISION_SOURCE
        )
    );
    if (isHostedRuntime || isRuleBased) {
        submit.disabled = false;
        return;
    }
    const providerId = document.getElementById('runBacktestProviderSelect')?.value || '';
    const option = runBacktestExecutionOption(providerId);
    const platformModelAvailable = availableRunBacktestProviders('platform_credits')
        .some((provider) => Boolean(findRunBacktestExecutionModel(provider, modelId)));
    const laneModelAvailable = runBacktestBillingMode === 'platform_credits'
        ? platformModelAvailable
        : runBacktestLaneAvailable(option, runBacktestBillingMode)
            && Boolean(findRunBacktestExecutionModel(option, modelId));
    submit.disabled = !(
        runBacktestOptionsReady
        && runBacktestBillingMode
        && modelId
        && modelId !== RULE_BASED_DECISION_SOURCE
        && laneModelAvailable
        && (
            runBacktestBillingMode === 'platform_credits'
            || (
                providerId
                && runBacktestLaneAvailable(option, runBacktestBillingMode)
            )
        )
    );
}

function syncRunBacktestModelOptions(preferredModelId = '') {
    const providerSelect = document.getElementById('runBacktestProviderSelect');
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect) return;
    const previousRuleBased = modelSelect.value === RULE_BASED_DECISION_SOURCE;
    const isPlatformCredits = runBacktestBillingMode === 'platform_credits';
    const option = runBacktestExecutionOption(providerSelect?.value || '');
    const providers = isPlatformCredits
        ? availableRunBacktestProviders('platform_credits')
        : (option ? [option] : []);
    clearSelectOptions(modelSelect);
    const seenModels = new Set();
    providers.flatMap((provider) => provider.models || []).forEach((model) => {
        const normalizedId = normalizeBacktestModelId(model.model_id);
        if (!normalizedId || seenModels.has(normalizedId)) return;
        seenModels.add(normalizedId);
        const modelOption = document.createElement('option');
        modelOption.value = model.model_id;
        modelOption.textContent = model.label;
        modelSelect.appendChild(modelOption);
    });
    const findModel = (modelId) => providers
        .map((provider) => findRunBacktestExecutionModel(provider, modelId))
        .find(Boolean) || null;
    const preferred = findModel(preferredModelId)
        || findModel(runBacktestModalAgent?.model_name)
        || providers[0]?.models?.[0]
        || null;
    if (preferred) modelSelect.value = preferred.model_id;
    if (document.getElementById('marketDataSourceSelect')?.value === IFIND_ASHARE_SOURCE) {
        syncIFindModelControl();
        if (previousRuleBased) modelSelect.value = RULE_BASED_DECISION_SOURCE;
    }
    syncBacktestModelFieldMode();
    syncRunBacktestSubmitAvailability();
}

function syncRunBacktestProviderVisibility() {
    const control = document.getElementById('runBacktestProviderControl');
    if (!control) return;
    control.hidden = runBacktestBillingMode !== 'byok';
}

function setRunBacktestBillingMode(
    billingMode,
    { providerId = '', modelId = '' } = {},
) {
    setRunBacktestApiKeysRecovery(false);
    const supported = new Set(['byok', 'platform_credits']);
    const providers = supported.has(billingMode)
        ? availableRunBacktestProviders(billingMode)
        : [];
    runBacktestBillingMode = providers.length ? billingMode : null;
    document
        .querySelectorAll('#runBacktestBillingGroup [data-billing-mode]')
        .forEach((button) => {
            const selected = button.dataset.billingMode === runBacktestBillingMode;
            button.setAttribute('aria-checked', selected ? 'true' : 'false');
            button.classList.toggle('is-selected', selected);
        });

    const providerSelect = document.getElementById('runBacktestProviderSelect');
    clearSelectOptions(providerSelect);
    if (billingMode === 'byok') {
        providers.forEach((provider) => {
            const option = document.createElement('option');
            option.value = provider.provider_id;
            option.textContent = provider.display_name;
            providerSelect?.appendChild(option);
        });
    }
    if (providerSelect && billingMode === 'byok' && providers.length) {
        providerSelect.value = providers.some(
            (provider) => provider.provider_id === providerId,
        ) ? providerId : providers[0].provider_id;
    }
    syncRunBacktestProviderVisibility();
    syncRunBacktestModelOptions(modelId);

    const hint = document.getElementById('runBacktestBillingHint');
    if (hint) {
        hint.textContent = runBacktestBillingMode === 'byok'
            ? 'Provider charges go directly to your API key. ATL Credits are not deducted.'
            : (runBacktestBillingMode === 'platform_credits'
                ? 'ATL Credits automatically use OpenRouter first, then CommonStack if needed.'
                : 'Choose an available AI billing method.');
    }
}

function setRunBacktestApiKeysRecovery(visible) {
    const button = document.getElementById('runBacktestApiKeysBtn');
    if (button) button.hidden = !visible;
}

function setRunBacktestExecutionUnavailable(
    message,
    { showApiKeysRecovery = false } = {},
) {
    runBacktestBillingMode = null;
    setRunBacktestApiKeysRecovery(showApiKeysRecovery);
    clearSelectOptions(document.getElementById('runBacktestProviderSelect'));
    syncRunBacktestProviderVisibility();
    const modelSelect = document.getElementById('modelSelect');
    clearSelectOptions(modelSelect);
    if (
        modelSelect
        && document.getElementById('marketDataSourceSelect')?.value
            === IFIND_ASHARE_SOURCE
    ) {
        const ruleOption = document.createElement('option');
        ruleOption.value = RULE_BASED_DECISION_SOURCE;
        ruleOption.textContent = 'Rule-based';
        modelSelect.appendChild(ruleOption);
        modelSelect.value = RULE_BASED_DECISION_SOURCE;
    }
    document
        .querySelectorAll('#runBacktestBillingGroup [data-billing-mode]')
        .forEach((button) => {
            button.setAttribute('aria-checked', 'false');
            button.classList.remove('is-selected');
        });
    const hint = document.getElementById('runBacktestBillingHint');
    if (hint) hint.textContent = message;
    syncBacktestModelFieldMode();
    syncRunBacktestSubmitAvailability();
}

async function loadRunBacktestExecutionOptions(agent) {
    const pending = readPendingByokBacktest();
    if (pending) clearPendingByokBacktest();
    runBacktestOptionsReady = false;
    syncRunBacktestSubmitAvailability();
    try {
        const data = await API.request(`${API_BASE}/api/credits/execution-options`);
        if (runBacktestModalAgent?.agent_id !== agent?.agent_id) return;
        runBacktestExecutionOptions = Array.isArray(data?.providers)
            ? data.providers
            : [];
        runBacktestOptionsReady = true;
    } catch (_error) {
        if (runBacktestModalAgent?.agent_id !== agent?.agent_id) return;
        runBacktestExecutionOptions = [];
        runBacktestOptionsReady = false;
        setRunBacktestExecutionUnavailable('Backtest execution options could not be loaded.');
        return;
    }

    if (pending) {
        const pendingProvider = runBacktestExecutionOption(pending.provider_id);
        const pendingModel = findRunBacktestExecutionModel(
            pendingProvider,
            pending.model_id,
        );
        if (
            runBacktestLaneAvailable(pendingProvider, 'byok')
            && pendingModel
        ) {
            setRunBacktestBillingMode('byok', {
                providerId: pending.provider_id,
                modelId: pendingModel.model_id,
            });
            return;
        }
    }

    for (const billingMode of ['byok', 'platform_credits']) {
        const provider = availableRunBacktestProviders(billingMode).find(
            (option) => Boolean(
                findRunBacktestExecutionModel(option, agent?.model_name),
            ),
        );
        if (provider) {
            setRunBacktestBillingMode(billingMode, {
                providerId: provider.provider_id,
                modelId: agent?.model_name || '',
            });
            return;
        }
    }

    for (const billingMode of ['byok', 'platform_credits']) {
        const provider = availableRunBacktestProviders(billingMode).find(
            (option) => Array.isArray(option.models) && option.models.length > 0,
        );
        const fallbackModel = provider?.models?.[0] || null;
        if (provider && fallbackModel) {
            setRunBacktestBillingMode(billingMode, {
                providerId: provider.provider_id,
                modelId: fallbackModel.model_id,
            });
            const hint = document.getElementById('runBacktestBillingHint');
            if (hint) {
                hint.textContent = `Saved model is unavailable; this run will use ${fallbackModel.label || fallbackModel.model_id} instead.`;
            }
            return;
        }
    }

    setRunBacktestExecutionUnavailable(
        'Add and verify a default API key, or ask an administrator to enable a platform provider.',
        { showApiKeysRecovery: true },
    );
}

function readBacktestLaunchConfigMap() {
    try {
        const raw = localStorage.getItem(BACKTEST_LAUNCH_CONFIG_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (_error) {
        return {};
    }
}

function stashBacktestLaunchConfig(runId, config) {
    if (!runId || !config) return;
    const map = readBacktestLaunchConfigMap();
    map[runId] = { ...config, savedAt: new Date().toISOString() };
    const keys = Object.keys(map).sort(
        (a, b) => String(map[a].savedAt || '').localeCompare(String(map[b].savedAt || '')),
    );
    while (keys.length > 40) {
        delete map[keys.shift()];
    }
    try {
        localStorage.setItem(BACKTEST_LAUNCH_CONFIG_KEY, JSON.stringify(map));
    } catch (_error) {
        /* ignore quota */
    }
}

function getBacktestLaunchConfig(runId) {
    if (!runId) return null;
    return readBacktestLaunchConfigMap()[runId] || null;
}

function formatPromptFromPipeline(pipeline) {
    if (!Array.isArray(pipeline) || !pipeline.length) return null;
    if (pipeline.length === 1) {
        const prompt = String(pipeline[0]?.prompt || '').trim();
        return prompt || null;
    }
    return pipeline
        .map((step, index) => {
            const label = step.label || step.presetKey || `Step ${index + 1}`;
            const prompt = String(step.prompt || '').trim();
            return prompt ? `• ${label}: ${prompt}` : `• ${label}`;
        })
        .join('\n');
}

function describeUniverseFromAssets(assets) {
    if (!Array.isArray(assets) || !assets.length) return null;
    const sorted = [...assets].map(String).sort().join(',');
    for (const uni of Object.values(ASSET_UNIVERSES)) {
        if ([...uni.assets].map(String).sort().join(',') === sorted) {
            return uni.name;
        }
    }
    if (assets.length <= 8) return assets.join(', ');
    return `${assets.slice(0, 6).join(', ')} +${assets.length - 6} more`;
}

function setBacktestConfigText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function formatTransactionCostProfile(profile) {
    if (!profile || typeof profile !== 'object') return '—';
    const percentage = (value) => `${(Number(value) * 100).toLocaleString('en-US', {
        maximumFractionDigits: 4,
    })}%`;
    const minimumCommission = Number(profile.minimum_commission);
    const priceTick = Number(profile.price_tick);
    return [
        `Commission ${percentage(profile.commission_rate)}`
            + (Number.isFinite(minimumCommission) ? ` (min ¥${minimumCommission.toFixed(2)})` : ''),
        `Sell stamp duty ${percentage(profile.stamp_duty_sell_rate)}`,
        `Transfer fee ${percentage(profile.transfer_fee_rate)}`,
        `Slippage ${percentage(profile.buy_slippage_rate)} each side`,
        Number.isFinite(priceTick) ? `Price tick ¥${priceTick.toFixed(2)}` : null,
    ].filter(Boolean).join(' · ');
}

function formatTransactionCostTotals(totals) {
    if (!totals || typeof totals !== 'object') return 'No filled orders';
    const totalFees = Number(totals.total_fees);
    const slippage = Number(totals.slippage_amount);
    if (!Number.isFinite(totalFees) && !Number.isFinite(slippage)) {
        return 'No filled orders';
    }
    return [
        Number.isFinite(totalFees) ? `Fees ${formatTradingMoney(totalFees, '¥')}` : null,
        Number.isFinite(slippage) ? `Slippage ${formatTradingMoney(slippage, '¥')}` : null,
        'CNY native',
    ].filter(Boolean).join(' · ');
}

function renderBacktestRunConfig(
    run,
    {
        running = false,
        launchConfig = null,
        statusLabel = null,
        baselineRun = null,
    } = {},
) {
    const empty = document.getElementById('backtestConfigEmpty');
    const list = document.getElementById('backtestConfigList');
    const cfg = launchConfig || (run?.run_id ? getBacktestLaunchConfig(run.run_id) : null);

    if (!run && !cfg) {
        if (empty) empty.hidden = false;
        if (list) list.hidden = true;
        if (!running) showBacktestRunProgress(false);
        return;
    }

    if (empty) empty.hidden = true;
    if (list) list.hidden = false;

    const metadata = run?.metadata && typeof run.metadata === 'object'
        ? run.metadata
        : {};
    const llmExecution = run?.llm_execution && typeof run.llm_execution === 'object'
        ? run.llm_execution
        : (metadata.llm_execution && typeof metadata.llm_execution === 'object'
            ? metadata.llm_execution
            : null);
    const completedExecution = !running ? llmExecution : null;
    const dataSource = cfg?.dataSource || metadata.data_source || run?.data_source || null;
    const universeKey = cfg?.universeKey || metadata.universe || run?.universe || null;
    const runSymbols = cfg?.assets || metadata.symbols || run?.symbols || null;
    const ifindProfile = dataSource === IFIND_ASHARE_SOURCE
        ? getIFindUniverseProfile(universeKey)
        : null;
    const agentName = cfg?.agentName || run?.agent_name || '—';
    const model = completedExecution?.model_id || cfg?.model || run?.llm_model || '—';
    const capital = cfg?.initialCapital ?? run?.initial_equity;
    const billingMode = completedExecution?.billing_mode
        || cfg?.billingMode
        || metadata.billing_mode
        || run?.billing_mode
        || null;
    const billingProvider = completedExecution?.provider_id
        || cfg?.providerId
        || metadata.provider_id
        || run?.provider_id
        || null;
    let billingLabel = billingMode === 'byok'
        ? 'BYOK' + (billingProvider ? ' · ' + billingProvider : '')
        : (billingMode === 'platform_credits' ? 'ATL Credits' : '—');
    if (completedExecution && completedExecution.usage_available === false) {
        billingLabel += ' · Usage unavailable';
    }
    const nativeInitialCapital = metadata.native_initial_capital
        ?? run?.native_initial_capital;
    const startFxRate = metadata.fx_start_rate ?? run?.fx_start_rate;
    const rawFxSource = metadata.fx_source ?? run?.fx_source;
    const transactionCostProfile = metadata.transaction_cost_profile
        ?? run?.transaction_cost_profile;
    const transactionCostTotals = metadata.transaction_cost_totals
        ?? run?.transaction_cost_totals;
    const marketRuleProfile = metadata.market_rule_profile
        ?? run?.market_rule_profile;
    const marketRuleRejections = metadata.market_rule_rejections
        ?? run?.market_rule_rejections;
    const baselineMetadata = baselineRun?.metadata
        && typeof baselineRun.metadata === 'object'
        ? baselineRun.metadata
        : {};
    const baselineAllocation = baselineMetadata.baseline_allocation
        ?? baselineRun?.baseline_allocation
        ?? metadata.baseline_allocation
        ?? run?.baseline_allocation;
    // Absent (older runs) reads as applied; only an explicit false marks a
    // curve that carries the market's cost rules without ever paying them.
    const transactionCostsApplied = (metadata.transaction_costs_applied
        ?? run?.transaction_costs_applied) !== false;
    const start = cfg?.startDate || run?.start_date;
    const end = cfg?.endDate || run?.end_date;
    const universe = cfg?.universeLabel
        || ifindProfile?.name
        || describeUniverseFromAssets(runSymbols)
        || universeKey
        || '—';
    const symbolCount = cfg?.symbolCount
        ?? (Array.isArray(runSymbols) ? runSymbols.length : null);
    const timeframe = cfg?.timeframe || metadata.timeframe || run?.timeframe || '60m';
    const decisionSource = cfg?.decisionSource
        || metadata.decision_source
        || run?.decision_source
        || (dataSource === 'vnpy_simulation' ? 'rule_based' : null);
    const decisionSourceLabel = decisionSource === LLM_DECISION_SOURCE
        ? formatAgentModelLabel(model)
        : (decisionSource === RULE_BASED_DECISION_SOURCE
            ? 'Rule-based'
            : (decisionSource || 'AI / Rule-based'));
    const marketData = cfg?.marketDataLabel
        || (dataSource === IFIND_ASHARE_SOURCE
            ? 'iFinD A-Share'
            : (dataSource === 'vnpy_simulation' ? 'vn.py Simulation' : 'Alpaca'));
    const started = run?.created_at
        ? new Date(String(run.created_at).replace(' ', 'T')).toLocaleString()
        : (cfg?.startedAt
            ? new Date(cfg.startedAt).toLocaleString()
            : (running ? 'Just now' : '—'));
    const prompt = cfg?.prompt || null;

    setBacktestConfigText('backtestConfigAgent', agentName);
    setBacktestConfigText('backtestConfigModel', model || '—');
    setBacktestConfigText('backtestConfigBilling', billingLabel);
    setBacktestConfigText('backtestConfigStarted', started);
    setBacktestConfigText(
        'backtestConfigCapital',
        Number.isFinite(Number(capital)) ? `$${Number(capital).toLocaleString()}` : '—',
    );
    setBacktestConfigText('backtestConfigMarketData', marketData);
    setBacktestConfigText('backtestConfigMarketDataMeta', marketData);
    const showFx = dataSource === IFIND_ASHARE_SOURCE
        && Number.isFinite(Number(nativeInitialCapital))
        && Number.isFinite(Number(startFxRate));
    const nativeCapitalRow = document.getElementById('backtestConfigNativeCapitalRow');
    const fxSourceRow = document.getElementById('backtestConfigFxSourceRow');
    const fxRateRow = document.getElementById('backtestConfigFxRateRow');
    if (nativeCapitalRow) nativeCapitalRow.hidden = !showFx;
    if (fxSourceRow) fxSourceRow.hidden = !showFx;
    if (fxRateRow) fxRateRow.hidden = !showFx;
    if (showFx) {
        setBacktestConfigText(
            'backtestConfigNativeCapital',
            `¥${Number(nativeInitialCapital).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
        );
        setBacktestConfigText(
            'backtestConfigFxSource',
            rawFxSource === 'ifind_history_currency_conversion'
                ? 'iFinD Historical Conversion Rate'
                : (rawFxSource || 'iFinD Historical Conversion Rate'),
        );
        setBacktestConfigText('backtestConfigFxRate', Number(startFxRate).toFixed(4));
    }
    const showTransactionCosts = transactionCostProfile
        && typeof transactionCostProfile === 'object';
    const transactionCostRow = document.getElementById('backtestConfigTransactionCostsRow');
    const costProfileRow = document.getElementById('backtestConfigCostProfileRow');
    const totalCostsRow = document.getElementById('backtestConfigTotalCostsRow');
    if (transactionCostRow) transactionCostRow.hidden = !showTransactionCosts;
    if (costProfileRow) costProfileRow.hidden = !showTransactionCosts;
    if (totalCostsRow) totalCostsRow.hidden = !showTransactionCosts;
    if (showTransactionCosts) {
        setBacktestConfigText(
            'backtestConfigTransactionCosts',
            transactionCostsApplied
                ? 'Charged · CNY native ledger'
                : 'Market rules shown · not charged on this curve',
        );
        setBacktestConfigText(
            'backtestConfigCostProfile',
            formatTransactionCostProfile(transactionCostProfile),
        );
        setBacktestConfigText(
            'backtestConfigTotalCosts',
            transactionCostsApplied
                ? formatTransactionCostTotals(transactionCostTotals)
                : 'Not applicable — reference price curve',
        );
    }
    const showMarketRules = marketRuleProfile?.enabled === true;
    const marketRulesRow = document.getElementById('backtestConfigMarketRulesRow');
    if (marketRulesRow) marketRulesRow.hidden = !showMarketRules;
    if (showMarketRules) {
        setBacktestConfigText('backtestConfigMarketRules', 'Enabled');
    }
    const rejectionLabels = {
        suspended: 'Suspended',
        limit_up_buy_blocked: 'Upper-limit buys',
        limit_down_sell_blocked: 'Lower-limit sells',
        market_rule_unavailable: 'Missing rule',
    };
    const ruleRejectionParts = Object.entries(marketRuleRejections || {})
        .filter(([, count]) => Number(count) > 0)
        .map(([reason, count]) => `${rejectionLabels[reason] || reason} ${Number(count)}`);
    const ruleRejectionsRow = document.getElementById('backtestConfigRuleRejectionsRow');
    if (ruleRejectionsRow) ruleRejectionsRow.hidden = ruleRejectionParts.length === 0;
    if (ruleRejectionParts.length) {
        setBacktestConfigText('backtestConfigRuleRejections', ruleRejectionParts.join(' · '));
    }
    const delayed = Number(baselineAllocation?.symbols_delayed || 0);
    const unfilled = Number(baselineAllocation?.symbols_unfilled || 0);
    const showBaselineRules = delayed > 0 || unfilled > 0;
    const baselineRulesRow = document.getElementById('backtestConfigBaselineRulesRow');
    if (baselineRulesRow) baselineRulesRow.hidden = !showBaselineRules;
    if (showBaselineRules) {
        setBacktestConfigText(
            'backtestConfigBaselineRules',
            `Delayed ${delayed} · Unfilled ${unfilled}`,
        );
    }
    setBacktestConfigText('backtestConfigUniverse', universe);
    setBacktestConfigText(
        'backtestConfigSymbols',
        Number.isFinite(Number(symbolCount)) ? String(symbolCount) : '—',
    );
    setBacktestConfigText('backtestConfigTimeframe', timeframe);
    setBacktestConfigText(
        'backtestConfigDecisionSource',
        decisionSourceLabel,
    );
    setBacktestConfigText(
        'backtestConfigWindow',
        start && end ? `${start} → ${end}` : '—',
    );
    setBacktestConfigText(
        'backtestConfigStatus',
        statusLabel || (running ? 'Running' : 'Completed'),
    );

    const promptRow = document.getElementById('backtestConfigPromptRow');
    const promptEl = document.getElementById('backtestConfigPrompt');
    const promptDetails = document.getElementById('backtestConfigInstructionDetails');
    if (prompt) {
        if (promptRow) promptRow.hidden = false;
        if (promptDetails) promptDetails.hidden = false;
        if (promptEl) promptEl.textContent = prompt;
    } else if (promptRow) {
        promptRow.hidden = true;
        if (promptDetails) promptDetails.hidden = true;
    }
}

function closeRunBacktestModal() {
    const modal = document.getElementById('runBacktestModal');
    if (modal) modal.hidden = true;
    setRunBacktestApiKeysRecovery(false);
    runBacktestModalAgent = null;
    runBacktestExecutionOptions = [];
    runBacktestBillingMode = null;
    runBacktestOptionsReady = false;
    const err = document.getElementById('runBacktestModalError');
    if (err) {
        err.hidden = true;
        err.textContent = '';
    }
}

async function openRunBacktestModal(agent) {
    if (!agent?.agent_id) {
        alert('Please create or select an agent first.');
        return;
    }
    if (isDemoAgent(agent.agent_id)) {
        alert('Demo agents cannot run backtests. Create your own agent first.');
        return;
    }
    // Both ways into a launch land here — this card's Run button and the agent
    // editor's, which is reachable while a backtest is running (see
    // renderAgentRunningActions). Refuse here rather than opening a modal whose
    // submit the server would reject, using this file's alert() convention for
    // launch-time refusals (see showBacktestLaunchFailure).
    const concurrencyRefusal = backtestConcurrencyRefusal();
    if (concurrencyRefusal) {
        alert(concurrencyRefusal);
        return;
    }

    runBacktestModalAgent = agent;
    runBacktestExecutionOptions = [];
    runBacktestBillingMode = null;
    runBacktestOptionsReady = false;
    setRunBacktestApiKeysRecovery(false);
    const isHostedRuntime = (agent.runtime_type || 'pipeline') !== 'pipeline';
    populateBacktestAgentSelect();
    const select = document.getElementById('backtestAgentSelect');
    if (select) select.value = agent.agent_id;

    const nameEl = document.getElementById('runBacktestAgentName');
    if (nameEl) nameEl.textContent = agent.name || agent.agent_id;

    const sleeve = Number(agent.cash_allocation);
    const hint = document.getElementById('runBacktestCapitalHint');
    if (hint) {
        hint.textContent = Number.isFinite(sleeve)
            ? `Does not change Paper Trading Allocated Capital ($${sleeve.toLocaleString()}).`
            : 'Does not change Paper Trading Allocated Capital.';
    }

    const capitalValue = document.getElementById('runBacktestCapitalValue');
    if (capitalValue) {
        capitalValue.textContent = `$${resolveBacktestCapital(agent).toLocaleString()}`;
    }

    syncModelSelectFromAgent(agent);
    const marketDataSourceSelect = document.getElementById('marketDataSourceSelect');
    if (marketDataSourceSelect) {
        // The hosted adapter consumes the upstream project's US-equity data
        // contract. Keep the modal on the one ATL market profile it supports.
        if (isHostedRuntime) marketDataSourceSelect.value = 'alpaca';
        marketDataSourceSelect.disabled = isHostedRuntime;
        marketDataSourceSelect.setAttribute(
            'aria-disabled',
            String(isHostedRuntime),
        );
    }
    selectPreset('djia');
    const builtinTabBtn = document.querySelector('#runBacktestModal .universe-tab[data-tab="builtin"]');
    if (builtinTabBtn) handleUniverseTabSwitch(builtinTabBtn);
    syncMarketDataSourceUI({ resetIFindDecisionSource: true });

    const pipeline = loadAgentPipelineForBacktest(agent);
    const prompt = formatPromptFromPipeline(pipeline);
    const promptGroup = document.getElementById('runBacktestPromptGroup');
    const promptPreview = document.getElementById('runBacktestPromptPreview');
    if (prompt) {
        if (promptGroup) promptGroup.hidden = false;
        if (promptPreview) promptPreview.textContent = prompt;
    } else if (promptGroup) {
        promptGroup.hidden = true;
    }

    const err = document.getElementById('runBacktestModalError');
    if (err) {
        err.hidden = true;
        err.textContent = '';
    }
    const submit = document.getElementById('runBacktestModalSubmit');
    if (submit) {
        submit.disabled = true;
        submit.textContent = 'Loading execution options…';
    }

    const modal = document.getElementById('runBacktestModal');
    if (modal) modal.hidden = false;
    await loadRunBacktestExecutionOptions(agent);
    if (runBacktestModalAgent?.agent_id !== agent.agent_id) return;
    if (submit) submit.textContent = '▶ Run Backtest';
    syncBacktestModelFieldMode();
}

window.openRunBacktestModal = openRunBacktestModal;
window.closeRunBacktestModal = closeRunBacktestModal;

function goToApiKeys() {
    clearPendingByokBacktest();
    closeRunBacktestModal();
    navigateToPage('credits');
    window.CreditsPage?.openApiKeys({ focus: true });
}

async function runBacktest() {
    // Get dates from form
    const startDateInput = document.getElementById('startDate');
    const endDateInput = document.getElementById('endDate');
    
    if (!startDateInput || !endDateInput) {
        console.error('Date inputs not found');
        return;
    }
    
    const startDate = startDateInput.value;
    const endDate = endDateInput.value;
    
    const showModalError = (msg) => {
        const err = document.getElementById('runBacktestModalError');
        if (err && !document.getElementById('runBacktestModal')?.hidden) {
            err.textContent = msg;
            err.hidden = false;
        } else {
            console.warn(msg);
        }
    };

    if (!startDate || !endDate) {
        showModalError('Please select both start and end dates.');
        return;
    }

    // Mirror the server's MAX_BACKTEST_DAYS (api/routers/backtests.py) here so an
    // over-long window is caught while the modal is still open and the dates are
    // still on screen. Without this the only feedback is a 422 that arrives after
    // the modal has closed, and the helper copy used to actively invite the
    // mistake ("Change it to any range you have data for").
    const MAX_BACKTEST_DAYS = 31;
    const spanDays = Math.round(
        (Date.parse(`${endDate}T00:00:00Z`) - Date.parse(`${startDate}T00:00:00Z`)) / 86400000,
    );
    if (Number.isFinite(spanDays) && spanDays < 0) {
        showModalError('The end date must be on or after the start date.');
        return;
    }
    if (Number.isFinite(spanDays) && spanDays > MAX_BACKTEST_DAYS) {
        showModalError(
            `Pick a window of ${MAX_BACKTEST_DAYS} days or fewer — that range is ${spanDays} days.`,
        );
        return;
    }

    const assets = getSelectedAssets();
    const modelSelect = document.getElementById('modelSelect');
    const marketDataSourceSelect = document.getElementById('marketDataSourceSelect');
    const dataSource = marketDataSourceSelect?.value || 'alpaca';
    const isSimulation = dataSource === 'vnpy_simulation';
    const isIFind = dataSource === IFIND_ASHARE_SOURCE;
    const selectedIFindUniverse = isIFind ? getSelectedIFindUniverse() : null;
    const selectedIFindProfile = isIFind
        ? getIFindUniverseProfile(selectedIFindUniverse)
        : null;
    const selectedModel = modelSelect?.value || '';
    const ifindAllowsLLM = selectedIFindProfile
        ?.allowedDecisionSources.includes(LLM_DECISION_SOURCE) === true;
    const decisionSource = isSimulation
        ? RULE_BASED_DECISION_SOURCE
        : (isIFind
            ? (ifindAllowsLLM && selectedModel !== RULE_BASED_DECISION_SOURCE
                ? LLM_DECISION_SOURCE
                : RULE_BASED_DECISION_SOURCE)
            : LLM_DECISION_SOURCE);
    const isRuleBasedDecision = decisionSource === RULE_BASED_DECISION_SOURCE;
    const activeAgent = runBacktestModalAgent || getSelectedBacktestAgent();
    if (!activeAgent) {
        alert('Please create or select an agent first.');
        return;
    }

    await activateAgent(activeAgent);
    const isHostedRuntime = (activeAgent.runtime_type || 'pipeline') !== 'pipeline';
    const pipeline = isRuleBasedDecision
        ? null
        : (isHostedRuntime ? null : loadAgentPipelineForBacktest(activeAgent));
    const model = isRuleBasedDecision
        ? null
        : (isHostedRuntime ? null : resolveBacktestModelRequest(modelSelect, activeAgent));

    let selectedProviderId = '';
    let selectedBillingMode = null;
    if (
        decisionSource === LLM_DECISION_SOURCE
        && !isHostedRuntime
    ) {
        selectedBillingMode = runBacktestBillingMode;
        if (selectedBillingMode === 'byok') {
            selectedProviderId = (
                document.getElementById('runBacktestProviderSelect')?.value || ''
            );
        }
        const providerOption = runBacktestExecutionOption(selectedProviderId);
        const platformModelAvailable = availableRunBacktestProviders(
            'platform_credits',
        ).some((provider) => Boolean(findRunBacktestExecutionModel(provider, model)));
        const modelAvailable = selectedBillingMode === 'platform_credits'
            ? platformModelAvailable
            : runBacktestLaneAvailable(providerOption, selectedBillingMode)
                && Boolean(findRunBacktestExecutionModel(providerOption, model));
        if (
            !selectedBillingMode
            || !model
            || !modelAvailable
            || (selectedBillingMode === 'byok' && !selectedProviderId)
        ) {
            showModalError(
                selectedBillingMode === 'platform_credits'
                    ? 'Choose an AI billing method and model.'
                    : 'Choose an AI billing method, provider, and model.',
            );
            return;
        }
    }

    const initialCapital = resolveBacktestCapital(activeAgent);

    const promptSummary = formatPromptFromPipeline(pipeline);
    const universeLabel = isIFind
        ? selectedIFindProfile.name
        : (document.getElementById('builtinTab')?.classList.contains('active')
            ? (ASSET_UNIVERSES[selectedUniverse]?.name || selectedUniverse)
            : describeUniverseFromAssets(assets));
    
    console.log(`Running backtest: ${startDate} to ${endDate}`);
    console.log(`Assets: ${assets.join(', ')}`);
    console.log(`Market data: ${dataSource}`);
    console.log(`Initial capital (simulation): $${initialCapital}`);
    console.log(`Model: ${model || 'rule-based'}`);
    if (activeAgent?.agent_id) {
        console.log(`Agent: ${activeAgent.name} (${activeAgent.agent_id})`);
    }
    if (pipeline?.length) {
        console.log(`Sub-agent pipeline: ${pipeline.length} step(s)`);
    }
    
    const btn = document.getElementById('runBacktestModalSubmit');
    if (btn) {
        btn.textContent = '⏳ Running...';
        btn.disabled = true;
    }

    const launchConfigBase = {
        agentId: activeAgent.agent_id,
        agentName: activeAgent.name,
        model: isHostedRuntime
            ? 'AI Hedge Fund (hosted)'
            : (isRuleBasedDecision ? 'Rule-based' : (model || null)),
        prompt: promptSummary,
        initialCapital,
        startDate,
        endDate,
        assets: [...assets],
        universeLabel,
        universeKey: selectedIFindUniverse,
        symbolCount: assets.length,
        timeframe: isIFind ? IFIND_ASHARE_TIMEFRAME : '60m',
        decisionSource,
        billingMode: isRuleBasedDecision ? null : selectedBillingMode,
        providerId: isRuleBasedDecision || selectedBillingMode === 'platform_credits'
            ? null
            : selectedProviderId,
        marketDataLabel: isIFind
            ? 'iFinD A-Share'
            : (isSimulation ? 'vn.py Simulation' : 'Alpaca'),
        dataSource,
        startedAt: new Date().toISOString(),
    };

    window.ACTIVE_BACKTEST_DATA_SOURCE = dataSource;
    renderBacktestDataSourceBadge({
        data_source: dataSource,
        timeframe: isIFind ? IFIND_ASHARE_TIMEFRAME : null,
    });

    // Pin live view BEFORE navigateToPage → showPlaygroundPanel → loadData(),
    // otherwise the async history load paints the previous run over the chart.
    closeRunBacktestModal();
    // The agent editor is a fullscreen overlay (z-index 1200) and the run modal
    // sits above it — without this, a run launched from inside the editor
    // repaints My Agents invisibly underneath the settings page.
    if (window.AgentEditor?.close) window.AgentEditor.close(true);
    prepareLiveBacktestView(launchConfigBase);
    // Keyed per run, so this launch can clear or promote exactly its own entry —
    // a concurrent run of the same agent keeps its card either way. The key is a
    // `pending:` placeholder until the POST below hands back a live_run_id.
    let runKey = markAgentBacktestRunning(activeAgent.agent_id, null);
    // A synchronous throw anywhere in here would otherwise leave the agent
    // marked running with no poller ever attached to clear it — narrow
    // try/catch (not the outer one below, which governs the API call) so we
    // can clear the mark and rethrow rather than swallow the failure.
    try {
        navigateToPage('playground', { playgroundTab: 'agents' });
        currentMode = 'backtest';
        applyAgentFilters(false);
        updateBacktestRunProgress({
            elapsedSeconds: 0,
            message: isHostedRuntime
                ? 'Running hosted AI Hedge Fund…'
                : (pipeline?.length
                    ? `Running ${pipeline.length}-step agent pipeline…`
                    : 'Starting backtest…'),
        });
    } catch (error) {
        clearAgentBacktestRunning(runKey);
        throw error;
    }

    try {
        // Call API with session ID, assets, and model
        const params = new URLSearchParams({
            start_date: startDate,
            end_date: endDate,
            assets: assets.join(','),
            data_source: dataSource,
        });
        const payload = {
            start_date: startDate,
            end_date: endDate,
            data_source: dataSource,
            initial_capital: initialCapital,
            // Body is authoritative; query `assets` kept for older callers/logs.
            assets: [...assets],
        };
        params.set('decision_source', decisionSource);
        payload.decision_source = decisionSource;
        if (isIFind) {
            params.set('universe', selectedIFindUniverse);
            params.set('timeframe', IFIND_ASHARE_TIMEFRAME);
            payload.universe = selectedIFindUniverse;
            payload.timeframe = '60m';
        }
        if (
            decisionSource === LLM_DECISION_SOURCE
            && !isHostedRuntime
        ) {
            params.set('model', model);
            params.set('billing_mode', selectedBillingMode);
            payload.billing_mode = selectedBillingMode;
            payload.model = model;
            if (selectedBillingMode === 'byok') {
                params.set('provider_id', selectedProviderId);
                payload.provider_id = selectedProviderId;
            }
        }
        if (activeAgent?.agent_id && !String(activeAgent.agent_id).startsWith('mock-')) {
            payload.agent_id = activeAgent.agent_id;
        }
        if (decisionSource === LLM_DECISION_SOURCE && pipeline?.length) {
            payload.pipeline = pipeline;
        }
        const data = await API.post(`${API_BASE}/backtest/run?${params.toString()}`, payload);
        
        if (!data.success) {
            const message = formatBacktestError(data.error || data.message, dataSource);
            console.error('❌ Backtest failed:', message);
            showBacktestLaunchFailure(message, launchConfigBase, runKey);
            return;
        }

        const liveRunId = data.live_run_id || data.run_id;
        if (liveRunId) {
            stashBacktestLaunchConfig(liveRunId, launchConfigBase);
            // Re-file the placeholder under the id the server issued rather than
            // registering a second entry for the same run.
            runKey = promoteBacktestRunKey(runKey, activeAgent.agent_id, liveRunId);
            attachToLiveBacktest(liveRunId, null, launchConfigBase);
        }
        
        console.log('✅ Backtest started:', data.message);
        await pollBacktestStatus(null);
        
    } catch (error) {
        const message = formatBacktestError(error, dataSource);
        console.error('❌ Error starting backtest:', message);
        showBacktestLaunchFailure(message, launchConfigBase, runKey);
    }
}

/**
 * Poll backtest status until complete
 */
async function pollBacktestStatus(btn) {
    ensureBacktestPolling();
    // Legacy callers awaited this; keep a lightweight wait until the poller stops
    // or the run leaves "running" (max ~60 min).
    const maxAttempts = BACKTEST_POLL_MAX_SECONDS;
    for (let i = 0; i < maxAttempts; i += 1) {
        if (!backtestPollTimer) {
            if (btn) {
                btn.disabled = false;
                btn.textContent = '▶ Run Backtest';
            }
            return;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    if (btn) {
        btn.disabled = false;
        btn.textContent = '▶ Run Backtest';
    }
}

/**
 * Get selected symbols from checkboxes
 */
function getSelectedSymbols() {
    const symbols = [];
    document.querySelectorAll('.checkbox-item input:checked').forEach(cb => {
        const symbol = cb.nextElementSibling.textContent.trim();
        symbols.push(symbol);
    });
    return symbols;
}

/**
 * Resolve page from URL for legacy deep links + sync History API so browser
 * Back/Forward undo in-app navigation (see #178).
 */
// Defined by the anti-FOUC boot script in app.html's <head> (it needs the map
// before this file loads) and read back here so the two can never drift: the
// boot copy picks which page CSS paints, this copy picks which page renders.
// A divergence would paint one page and render another — a flash bug no test
// in this repo can catch, so there is deliberately only ever one object.
const NAV_VIEW_MAP = window.NAV_VIEW_MAP;
// Same deal, same reason: both files restore the same saved nav blob, so the
// rule that rewrites a pre-move one lives in exactly one place.
const migrateSavedNavState = window.migrateSavedNavState;

// Persist the current tab so a page refresh restores it instead of going home.
function persistNavigation() {
    try {
        localStorage.setItem(
            NAV_STATE_KEY,
            JSON.stringify(getNavigationState()),
        );
    } catch (error) {
        /* localStorage unavailable — ignore */
    }
}

function getNavigationState() {
    return {
        page: currentPage,
        playgroundTab,
        competitionTab,
    };
}

function navigationStatesEqual(a, b) {
    if (!a || !b) return false;
    return a.page === b.page
        && (a.playgroundTab || 'agents') === (b.playgroundTab || 'agents')
        && (a.competitionTab || 'leaderboard') === (b.competitionTab || 'leaderboard');
}

/**
 * Inverse of NAV_VIEW_MAP: nav state -> the ?view= slug that restores it.
 *
 * INVARIANT: every slug returned here must be a key of NAV_VIEW_MAP, and every
 * distinct state NAV_VIEW_MAP can produce must be reachable from some return
 * below. Break the first half and the URL this writes won't restore on refresh
 * or Back; break the second and a page becomes unlinkable. The two are
 * hand-maintained inverses — change one, check the other.
 *
 * Several NAV_VIEW_MAP keys are read-only aliases that this never emits
 * ('contest', 'competition', 'playground', 'my-algo', 'marketplace'); old links
 * keep working, new URLs get the canonical slug. 'marketplace' joined that list
 * when the catalog moved to Community: ?view=marketplace still opens it, but a
 * URL written from that page now says ?view=community.
 */
function viewParamForNavState(state) {
    if (state.page === 'home') return 'home';
    if (state.page === 'community') return 'community';
    if (state.page === 'account') return 'account';
    if (state.page === 'credits') return 'credits';
    if (state.page === 'admin') return 'admin';
    if (state.page === 'playground') {
        if (state.playgroundTab === 'backtest') return 'backtest';
        if (state.playgroundTab === 'paper') return 'paper';
        return 'agents';
    }
    if (state.page === 'competition') {
        if (state.competitionTab === 'live') return 'live';
        if (state.competitionTab === 'participants') return 'participants';
        if (state.competitionTab === 'about') return 'about';
        return 'leaderboard';
    }
    return state.page;
}

function buildNavigationUrl(state) {
    const params = new URLSearchParams(window.location.search);
    params.set('view', viewParamForNavState(state));
    params.delete('mode');
    const clean = params.toString();
    return `${window.location.pathname}${clean ? `?${clean}` : ''}${window.location.hash}`;
}

/**
 * Keep the History stack in sync with the visible page/subtab.
 * @param {{ replace?: boolean }} [options]
 */
function syncNavigationHistory({ replace = false } = {}) {
    const state = getNavigationState();
    const url = buildNavigationUrl(state);
    const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (!replace && url === current && navigationStatesEqual(window.history.state, state)) {
        return;
    }
    if (replace) {
        window.history.replaceState(state, '', url);
    } else {
        window.history.pushState(state, '', url);
    }
}

function clearNavBootState() {
    const html = document.documentElement;
    html.removeAttribute('data-nav-boot');
    // Keep data-nav-page / tab attrs as the live navigation signal (home snap
    // scroll and other page-scoped CSS depend on them after boot).
}

function applyInitialNavigation() {
    // Registered here, not in initNavigation(): initNavigation runs at the very
    // end of a long async boot, behind several awaits that can reject and abort
    // the whole DOMContentLoaded handler. Back is not worth hanging off that
    // single point of failure when the listener is free and order-independent.
    window.addEventListener('popstate', onNavigationPopState);

    // Skip the restore once the user has already clicked somewhere: nav is
    // live during boot's auth awaits, and stomping an explicit navigation
    // with the saved page reads as the app fighting the user.
    if (!userHasNavigated) {
        const initial = resolveInitialNavigation();
        navigateToPage(initial.page, {
            playgroundTab: initial.playgroundTab || 'agents',
            competitionTab: initial.competitionTab || 'leaderboard',
            history: 'replace',
        });
    }
    if (typeof initHomePage === 'function') {
        initHomePage();
    }
}

function resolveInitialNavigation() {
    const params = new URLSearchParams(window.location.search);
    const view = params.get('view') || params.get('mode');
    const hash = window.location.hash.replace('#', '');
    const legacy = view || hash;

    // Discord / share deep links land on the backtest playground.
    if (params.get('agent_id') || params.get('run_id')) {
        return { page: 'playground', playgroundTab: 'backtest' };
    }

    // An explicit URL view/hash always wins.
    if (legacy && NAV_VIEW_MAP[legacy]) {
        return { ...NAV_VIEW_MAP[legacy] };
    }

    // Otherwise restore the last visited tab across refreshes.
    try {
        // Migrated here as well as in navigateToPage's redirect: this function
        // is documented to return a *current* nav state, and popstate reads it
        // directly. Returning a page/subtab pair that no longer exists would be
        // a live trap for the next caller that does not route through
        // navigateToPage.
        const saved = migrateSavedNavState(
            JSON.parse(localStorage.getItem(NAV_STATE_KEY) || 'null'),
        );
        const validPages = ['home', 'playground', 'competition', 'community', 'account', 'credits'];
        if (saved && validPages.includes(saved.page)) {
            return saved;
        }
    } catch (error) {
        /* corrupt/unavailable state — fall through to home */
    }

    return { page: 'home' };
}

function onNavigationPopState(event) {
    const fromState = event.state;
    const target = (fromState && fromState.page)
        ? fromState
        : resolveInitialNavigation();
    navigateToPage(target.page, {
        playgroundTab: target.playgroundTab || 'agents',
        competitionTab: target.competitionTab || 'leaderboard',
        history: 'none',
    });
}

/**
 * A deep link that needs sign-in is parked here rather than left in the URL.
 *
 * The URL is the wrong place to hold it now that every navigation rewrites the
 * query string: buildNavigationUrl copies window.location.search wholesale, so
 * agent_id/run_id would ride along on every later pushState, and
 * resolveInitialNavigation checks them *before* ?view= — so a refresh from any
 * page would snap back to the backtest tab. sessionStorage (not localStorage)
 * because a stale pending link must not outlive the tab and hijack a later visit.
 */
const PENDING_DEEP_LINK_KEY = 'pending-agent-run-deep-link';

function readPendingDeepLink() {
    try {
        const saved = JSON.parse(sessionStorage.getItem(PENDING_DEEP_LINK_KEY) || 'null');
        if (saved && (saved.agentId || saved.runId)) return saved;
    } catch (error) {
        /* corrupt/unavailable — treat as no pending link */
    }
    return null;
}

function savePendingDeepLink(link) {
    try {
        sessionStorage.setItem(PENDING_DEEP_LINK_KEY, JSON.stringify(link));
    } catch (error) {
        /* sessionStorage unavailable — the post-sign-in retry is best effort */
    }
}

function clearPendingDeepLink() {
    try {
        sessionStorage.removeItem(PENDING_DEEP_LINK_KEY);
    } catch (error) {
        /* ignore */
    }
}

/** Drop agent_id/run_id from the visible URL without touching the history stack. */
function stripDeepLinkParamsFromUrl() {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('agent_id') && !params.has('run_id')) return;
    params.delete('agent_id');
    params.delete('run_id');
    const clean = params.toString();
    const next = `${window.location.pathname}${clean ? `?${clean}` : ''}${window.location.hash}`;
    window.history.replaceState(getNavigationState(), '', next);
}

/**
 * Open a specific agent + backtest run from ?agent_id=&run_id= (Discord links),
 * or from a link parked by a previous signed-out attempt.
 */
async function applyAgentRunDeepLink() {
    const params = new URLSearchParams(window.location.search);
    const pending = readPendingDeepLink();
    const agentId = (params.get('agent_id') || pending?.agentId || '').trim();
    const runId = (params.get('run_id') || pending?.runId || '').trim();
    if (!agentId && !runId) return;

    // Consume the link up front so it cannot leak into later history entries.
    // Everything below works off the locals; the signed-out branch re-parks it.
    stripDeepLinkParamsFromUrl();
    clearPendingDeepLink();

    try {
        await loadAgents();
    } catch (error) {
        console.warn('Deep link: loadAgents failed:', error.message);
    }

    let agent = agentId
        ? (allAgents || []).find((a) => a.agent_id === agentId)
        : null;
    let agentAuthError = false;
    if (!agent && agentId) {
        try {
            const data = await API.get(`${API_BASE}/api/v1/agents/${encodeURIComponent(agentId)}`);
            agent = data?.agent || null;
        } catch (error) {
            // The agent card is owner-gated (403). A Discord deep link is often
            // opened on a different device/browser than the one that owns the
            // agent, so surface it instead of silently landing on an empty session.
            agentAuthError = error.status === 401 || error.status === 403;
            console.warn('Deep link: agent not accessible:', error.message);
        }
    }

    // agentAuthError can only be set inside the `!agent && agentId` branch above,
    // and only from the catch — where the `agent = …` assignment never ran. So it
    // already implies both operands; re-testing them added nothing but made the
    // guard read as if the URL-supplied agentId decided the outcome. The access
    // decision is the server's 401/403, not this condition.
    if (agentAuthError) {
        const signedIn = isSignedIn();
        if (!signedIn) {
            // Park it so a successful sign-in retries — see PENDING_DEEP_LINK_KEY.
            savePendingDeepLink({ agentId, runId });
            alert('Sign in with the account that owns this agent to open its backtest from Discord.');
            openAuthModal('login');
            return;
        }
        alert('This agent belongs to a different account. Sign in with the account that owns it to open its backtest.');
    }

    if (agent) {
        try {
            await activateAgent(agent);
        } catch (error) {
            console.warn('Deep link: activateAgent failed:', error.message);
        }
    }

    if (runId) {
        localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, runId);
    }

    navigateToPage('playground', { playgroundTab: 'backtest', history: 'replace' });
    currentMode = 'backtest';
    await loadData();
    // No URL cleanup left to do: the deep-link params were stripped on entry and
    // navigateToPage's 'replace' already wrote ?view=backtest onto this entry.
}

function updatePlaygroundSubtabs() {
    document.querySelectorAll('[data-playground-tab]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.playgroundTab === playgroundTab);
    });
}

function updateCompetitionSubtabs() {
    document.querySelectorAll('[data-competition-tab]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.competitionTab === competitionTab);
    });
}

function showPlaygroundPanel(tab) {
    // Belt and braces: navigateToPage already redirects the retired subtab, so
    // nothing in-tree reaches this. It stays because this function is also the
    // direct target of the subtab click handler, where a stray
    // data-playground-tab="marketplace" would otherwise blank the page -- every
    // panel hidden and none shown.
    if (tab === 'marketplace') {
        navigateToPage('community');
        return;
    }

    playgroundTab = tab;
    updatePlaygroundSubtabs();

    const agents = document.getElementById('playgroundAgentsPanel');
    const backtest = document.querySelector('.playground-backtest-panel')
      || document.querySelector('.main-container');
    const paper = document.getElementById('paperTradingView');

    if (agents) agents.style.display = tab === 'agents' ? 'block' : 'none';
    if (backtest) backtest.style.display = tab === 'backtest' ? 'grid' : 'none';
    if (paper) paper.style.display = tab === 'paper' ? 'block' : 'none';

    if (tab === 'backtest') {
        currentMode = 'backtest';
        populateBacktestAgentSelect();
        if (!allAgents.length) loadAgents();
        loadData();
    } else if (tab === 'paper') {
        currentMode = 'paper';
        loadPaperTradingData();
    } else {
        currentMode = 'agents';
        // Cache-only repaint so the panel is not blank while agents load;
        // loadAgents() below does the authoritative fetch-and-render.
        if (typeof window.repaintPortfolioFromCache === 'function') {
            window.repaintPortfolioFromCache(allAgents.map(decorateAgent));
        }
        loadAgents();
        // A refresh mid-run restores the sessionStorage running marks but
        // drops the poller that would ever clear them -- reattach it here so
        // the card doesn't strand at "Backtesting…" for up to
        // BACKTEST_POLL_MAX_SECONDS. ensureBacktestPolling() is a no-op if a
        // poller is already attached.
        if (Object.keys(readRunningBacktests()).length) ensureBacktestPolling();
    }

    persistNavigation();
}

function showCompetitionPanel(tab) {
    // The Daily Leaderboard was replaced by the Live Trading board. A saved
    // nav state or a cached boot script can still hand us either retired key
    // ('daily', or 'season' from an earlier build of this branch), and an
    // unrecognised tab here shows no panel at all — a blank Competition page.
    if (tab === 'daily' || tab === 'season') tab = 'live';

    competitionTab = tab;
    updateCompetitionSubtabs();

    const leaderboard = document.getElementById('leaderboardView');
    const participants = document.getElementById('competitionParticipantsPanel');
    const about = document.getElementById('competitionAboutPanel');
    const showBoard = tab === 'leaderboard' || tab === 'live';

    if (leaderboard) leaderboard.style.display = showBoard ? 'flex' : 'none';
    if (participants) participants.style.display = tab === 'participants' ? 'block' : 'none';
    if (about) about.style.display = tab === 'about' ? 'block' : 'none';

    if (showBoard) {
        currentMode = 'contest';
        loadLeaderboardData(tab === 'live' ? 'live' : 'contest');
    } else {
        currentMode = tab;
    }

    persistNavigation();
}

function navigateToPage(page, options = {}) {
    console.log('Navigating to page:', page, options);

    // "My Agents" now lives as a Playground subtab; redirect legacy links.
    if (page === 'agents') {
        page = 'playground';
        options = { ...options, playgroundTab: options.playgroundTab || 'agents' };
    }
    // Marketplace moved from Playground → Community. This is the choke point
    // every navigation funnels through, so the redirect belongs here rather than
    // at each call site. Reads the module-level playgroundTab too, so a session
    // that entered this page load holding the retired subtab cannot land back on
    // it, and rewrites the tab to 'agents' rather than clearing it -- leaving it
    // set to 'marketplace' would bounce the *next* Playground visit as well.
    if (page === 'playground' && (options.playgroundTab || playgroundTab) === 'marketplace') {
        page = 'community';
        options = { ...options, playgroundTab: 'agents' };
    }
    // (The PR #335 redirect that sent competitionTab 'daily' to 'leaderboard'
    // stood here. It ran ahead of the alias normalisation below and undid it:
    // every path where `options.competitionTab` was absent and the module-level
    // `competitionTab` still held 'daily' — a cached app.html boot script, a
    // caller restoring raw saved state — landed on Competition instead of the
    // successor board. That is the silent fall-through to the wrong data the
    // aliases exist to stop, so the redirect is gone rather than reordered.)
    // Role-gate the admin shell in the UI too, not only its APIs: without
    // this, anyone landing on ?view=admin saw the empty console chrome until
    // the deferred boot /me settled — tens of seconds on a cold free-tier
    // start. The cached role decides; a stale cached admin still gets bounced
    // by the APIs' 403 via _handleAdminAccessLost.
    if (page === 'admin') {
        const authUser = getStoredAuthUser();
        if (!authUser || authUser.role !== 'admin') page = 'home';
    }

    const historyMode = options.history || 'push';
    if (historyMode === 'push') userHasNavigated = true;
    const prevState = getNavigationState();

    currentPage = page;

    if (options.playgroundTab) playgroundTab = options.playgroundTab;
    if (options.competitionTab) competitionTab = options.competitionTab;
    // Normalise the retired Daily keys here rather than only in
    // showCompetitionPanel: the boot stylesheet keys off
    // data-nav-competition-tab, so a 'daily' written to the attribute below
    // leaves the board hidden through first paint even though the panel is
    // shown a tick later.
    //
    // Applied to the resolved value, not to `options.competitionTab`: the
    // retired key reaches here just as often *without* an option — a caller
    // restoring saved state, or a cached app.html boot script writing the old
    // key — and a normaliser that only reads the argument misses exactly those.
    if (competitionTab === 'daily' || competitionTab === 'season') competitionTab = 'live';

    const html = document.documentElement;
    html.setAttribute('data-nav-page', page);
    html.setAttribute('data-nav-playground-tab', playgroundTab);
    html.setAttribute('data-nav-competition-tab', competitionTab);

    document.querySelectorAll('.primary-nav .mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === page);
    });

    const homeView = document.getElementById('homeView');
    const playgroundView = document.getElementById('playgroundView');
    const competitionView = document.getElementById('competitionView');
    const communityView = document.getElementById('communityView');
    const accountView = document.getElementById('accountView');
    const creditsView = document.getElementById('creditsView');
    const adminView = document.getElementById('adminView');
    const backtestPanel = document.querySelector('.playground-backtest-panel')
      || document.querySelector('.main-container');
    const paperView = document.getElementById('paperTradingView');
    const myAlgoView = document.getElementById('myTradingAlgoView');
    const leaderboardView = document.getElementById('leaderboardView');

    const hide = (el) => {
        if (el) el.style.display = 'none';
    };

    hide(homeView);
    hide(playgroundView);
    hide(competitionView);
    hide(communityView);
    hide(accountView);
    hide(creditsView);
    hide(adminView);
    hide(backtestPanel);
    hide(paperView);
    hide(myAlgoView);
    hide(leaderboardView);
    hide(document.getElementById('playgroundAgentsPanel'));
    hide(document.getElementById('competitionParticipantsPanel'));
    hide(document.getElementById('competitionAboutPanel'));

    if (page === 'home') {
        currentMode = 'home';
        if (homeView) homeView.style.display = 'block';
        if (typeof onHomePageShow === 'function') onHomePageShow();
    } else {
        if (typeof onHomePageHide === 'function') onHomePageHide();
        if (page === 'playground') {
            if (playgroundView) playgroundView.style.display = 'block';
            showPlaygroundPanel(playgroundTab);
        } else if (page === 'competition') {
            if (competitionView) competitionView.style.display = 'block';
            showCompetitionPanel(competitionTab);
        } else if (page === 'community') {
            currentMode = 'community';
            // Every entry to Community resets the chip filter to 'all' unless
            // an explicit category rides in via options.communityCategory (the
            // My Agents empty-shelf "Community" links) -- otherwise a category
            // set on one visit would leak into the next, unrelated visit made
            // through the plain nav tab, the most common entry path.
            marketplaceCategoryFilter = MARKET_LABELS[options.communityCategory] ? options.communityCategory : 'all';
            // The vendor chip resets for the same reason, and nothing rides in
            // via options: a vendor left selected on one visit would AND with an
            // incoming category and strand the empty-shelf deep links on an
            // empty grid.
            marketplaceVendorFilter = 'all';
            if (communityView) communityView.style.display = 'block';
            loadMarketplace();
        } else if (page === 'account') {
            currentMode = 'account';
            if (accountView) accountView.style.display = 'block';
            updateAccountPage();
        } else if (page === 'credits') {
            currentMode = 'credits';
            if (creditsView) creditsView.style.display = 'block';
            if (window.CreditsPage) window.CreditsPage.onEnter();
        } else if (page === 'admin') {
            currentMode = 'admin';
            if (adminView) adminView.style.display = 'block';
            // Stats load on entry and on explicit refresh — not on every
            // pager click, which only changes the user page.
            loadAdminStats();
            loadAdminUsers();
            if (window.AdminTabs) {
                window.AdminTabs.onEnter();
            }
            if (window.AdminAnalytics) {
                window.AdminAnalytics.onEnter();
            }
            if (window.AdminModelProviders) {
                window.AdminModelProviders.onEnter();
            }
            if (window.AdminCredits) {
                window.AdminCredits.onEnter();
            }
        }
    }

    const nav = document.getElementById('primaryNav');
    const menuToggle = document.getElementById('navMenuToggle');
    if (nav) nav.classList.remove('open');
    if (menuToggle) menuToggle.setAttribute('aria-expanded', 'false');

    clearNavBootState();
    persistNavigation();
    window.ATLAnalytics?.recordNavigation(page, { playgroundTab, competitionTab });

    if (historyMode === 'none') return;
    const nextState = getNavigationState();
    if (historyMode === 'push' && navigationStatesEqual(prevState, nextState)) return;
    syncNavigationHistory({ replace: historyMode === 'replace' });
}

function switchPlaygroundTab(tab) {
    if (currentPage !== 'playground') {
        navigateToPage('playground', { playgroundTab: tab });
        return;
    }
    // Re-clicking the active subtab deliberately falls through: showPlaygroundPanel
    // re-runs its loaders, which users rely on as a refresh. It cannot double-push
    // history — syncNavigationHistory early-returns when the URL and state are
    // already the current entry, which is exactly this case.
    showPlaygroundPanel(tab);
    // showPlaygroundPanel can redirect a retired tab to Community. Only record
    // the Playground view after the panel has updated and that redirect did not
    // occur, so heartbeats and visibility events keep the correct page_view.
    if (currentPage === 'playground') {
        window.ATLAnalytics?.recordNavigation('playground', { playgroundTab });
    }
    syncNavigationHistory({ replace: false });
}

function switchCompetitionTab(tab) {
    if (currentPage !== 'competition') {
        navigateToPage('competition', { competitionTab: tab });
        return;
    }
    // Falls through on a re-click for the same reason as switchPlaygroundTab.
    showCompetitionPanel(tab);
    syncNavigationHistory({ replace: false });
}

function openAddAgentModal() {
    const modal = document.getElementById('addAgentModal');
    if (modal) modal.hidden = false;
}

function closeAddAgentModal() {
    const modal = document.getElementById('addAgentModal');
    if (modal) modal.hidden = true;
}

function initNavigation() {
    document.querySelectorAll('.primary-nav .mode-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const mode = e.currentTarget.dataset.mode;
            if (mode === 'competition' && currentPage !== 'competition') {
                navigateToPage('competition', { competitionTab: 'leaderboard' });
                return;
            }
            navigateToPage(mode);
        });
    });

    document.querySelectorAll('[data-playground-tab]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchPlaygroundTab(e.currentTarget.dataset.playgroundTab);
        });
    });

    document.querySelectorAll('[data-competition-tab]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            switchCompetitionTab(e.currentTarget.dataset.competitionTab);
        });
    });

    document.getElementById('homeOpenPlaygroundBtn')?.addEventListener('click', () => {
        navigateToPage('playground', { playgroundTab: 'agents' });
    });

    document.getElementById('homeViewCompetitionBtn')?.addEventListener('click', () => {
        navigateToPage('competition', { competitionTab: 'leaderboard' });
    });

    document.getElementById('homeViewMarketPulseBtn')?.addEventListener('click', () => {
        navigateToPage('playground', { playgroundTab: 'agents' });
    });

    document.querySelectorAll('[data-home-nav]').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.homeNav;
            if (target === 'agents') {
                navigateToPage('playground', { playgroundTab: 'agents' });
            } else if (target === 'playground') {
                navigateToPage('playground', { playgroundTab: 'agents' });
            } else if (target === 'discord') {
                openDiscordWithAccount();
            }
        });
    });

    document.querySelectorAll('.agent-view-playground').forEach(btn => {
        btn.addEventListener('click', () => {
            navigateToPage('playground', { playgroundTab: 'agents' });
        });
    });

    document.getElementById('agentSearchInput')?.addEventListener('input', applyAgentFilters);
    document.getElementById('marketplaceSearchInput')?.addEventListener('input', renderMarketplaceGrid);
    document.getElementById('marketplaceCategoryChips')?.addEventListener('click', (event) => {
      const chipBtn = event.target.closest('[data-marketplace-category]');
      if (!chipBtn) return;
      setMarketplaceCategoryFilter(chipBtn.dataset.marketplaceCategory);
    });
    document.getElementById('marketplaceVendorChips')?.addEventListener('click', (event) => {
        const chip = event.target.closest('[data-marketplace-vendor]');
        if (!chip) return;
        setMarketplaceVendorFilter(chip.dataset.marketplaceVendor);
    });
    document.getElementById('agentsCategories')?.addEventListener('click', (event) => {
      const marketChip = event.target.closest('[data-agent-market]');
      if (marketChip) {
        setAgentMarketFilter(marketChip.dataset.agentMarket);
        return;
      }
      const communityLink = event.target.closest('[data-community-category]');
      if (communityLink) {
        event.preventDefault();
        // Routed through navigateToPage's options rather than a separate
        // setMarketplaceCategoryFilter call -- navigateToPage is the one
        // place that resets the filter to 'all' on a plain Community entry,
        // so the explicit category has to ride the same call to survive it.
        navigateToPage('community', { communityCategory: communityLink.dataset.communityCategory });
        return;
      }
      const prevBtn = event.target.closest('[data-agent-grid-prev]');
      const nextBtn = event.target.closest('[data-agent-grid-next]');
      const key = prevBtn?.dataset.agentGridPrev || nextBtn?.dataset.agentGridNext;
      if (!key) return;
      const page = agentGridPage[key] || 0;
      agentGridPage[key] = prevBtn ? Math.max(0, page - 1) : page + 1;
      applyAgentFilters(false);
    });
    document.getElementById('agentViewGrid')?.addEventListener('click', () => setAgentViewMode('grid'));
    document.getElementById('agentViewList')?.addEventListener('click', () => setAgentViewMode('list'));

    document.getElementById('addAgentBtnToolbar')?.addEventListener('click', openAddAgentModal);
    document.getElementById('addAgentModalClose')?.addEventListener('click', closeAddAgentModal);
    document.getElementById('addAgentModalBackdrop')?.addEventListener('click', closeAddAgentModal);
    document.getElementById('connectExternalAgentBtn')?.addEventListener('click', openCreateExternalAgentModal);
    document.getElementById('createExternalAgentModalClose')?.addEventListener('click', closeCreateExternalAgentModal);
    document.getElementById('createExternalAgentModalBackdrop')?.addEventListener('click', closeCreateExternalAgentModal);
    document.getElementById('createExternalAgentForm')?.addEventListener('submit', submitCreateExternalAgent);
    document.getElementById('createBuiltinAgentBtn')?.addEventListener('click', openCreateBuiltinAgentModal);
    document.getElementById('createBuiltinAgentModalClose')?.addEventListener('click', closeCreateBuiltinAgentModal);
    document.getElementById('createBuiltinAgentModalBackdrop')?.addEventListener('click', closeCreateBuiltinAgentModal);
    document.getElementById('createBuiltinAgentForm')?.addEventListener('submit', submitCreateBuiltinAgent);
    document.getElementById('duplicateAgentModalClose')?.addEventListener('click', closeDuplicateAgentModal);
    document.getElementById('duplicateAgentModalBackdrop')?.addEventListener('click', closeDuplicateAgentModal);
    document.getElementById('duplicateAgentForm')?.addEventListener('submit', (event) => {
        event.preventDefault();
        submitDuplicateAgent();
    });
    document.getElementById('agentCredentialsModalClose')?.addEventListener('click', closeAgentCredentialsModal);
    document.getElementById('agentCredentialsModalBackdrop')?.addEventListener('click', closeAgentCredentialsModal);

    document.getElementById('competitionRulesBtn')?.addEventListener('click', () => {
        if (currentPage !== 'competition') {
            navigateToPage('competition', { competitionTab: 'about' });
        } else {
            switchCompetitionTab('about');
        }
    });

    document.getElementById('navMenuToggle')?.addEventListener('click', () => {
        const nav = document.getElementById('primaryNav');
        const toggle = document.getElementById('navMenuToggle');
        if (!nav || !toggle) return;
        const isOpen = nav.classList.toggle('open');
        toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
    });

}

/**
 * Switch between modes (legacy compatibility)
 */
function switchMode(mode) {
    console.log('Switching to mode:', mode);

    // Shares NAV_VIEW_MAP rather than keeping a third copy of the same table;
    // it is a superset of the slugs this used to carry, so nothing regresses.
    const target = NAV_VIEW_MAP[mode] || { page: mode };
    navigateToPage(target.page, {
        playgroundTab: target.playgroundTab,
        competitionTab: target.competitionTab,
    });
}

function isMyAlgoRun(run) {
    return run && run.run_id && String(run.run_id).startsWith('algo_');
}

function isExternalAgentRun(run) {
    return run && run.run_id && String(run.run_id).startsWith('ext_');
}

function latestRun(runs) {
    if (!runs || !runs.length) return null;
    return runs.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))[0];
}

function scopedExternalRuns(sessionRuns, activeName) {
    const externalRuns = sessionRuns.filter(isExternalAgentRun);
    if (!activeName) return externalRuns;
    const scoped = externalRuns.filter((r) => r.agent_name === activeName);
    return scoped.length ? scoped : externalRuns;
}

function formatBacktestRunReturn(run) {
    if (run.total_return == null) return '—';
    const pct = Math.abs(run.total_return) <= 1 ? run.total_return * 100 : run.total_return;
    const sign = pct >= 0 ? '+' : '';
    return `${sign}${pct.toFixed(2)}%`;
}

function formatBacktestRunPrimary(run) {
    const dates = [run.start_date, run.end_date].filter(Boolean).join(' → ');
    return `${dates || run.run_id} · ${formatBacktestRunReturn(run)}`;
}

function formatBacktestRunSecondary(run) {
    const when = run.created_at ? new Date(run.created_at).toLocaleString() : '';
    const cost = formatUsd(run.est_cost_usd);
    const costLabel = cost && Number(run.est_cost_usd) > 0 ? cost : '';
    const sourceLabel = run.data_source === 'ifind_ashare'
        ? 'iFinD China A-Shares · 60m'
        : (run.data_source === 'vnpy_simulation' ? 'vn.py simulated' : '');
    return [sourceLabel, costLabel, when].filter(Boolean).join(' · ');
}

// Belt-and-braces: the backend already omits US market indexes for iFinD runs
// (include_market_indexes=False). Match on the structural run_id the backend
// mints for them ("index:^DJI", "index:^NDX") rather than the display label —
// a renamed label would silently disable a label-only filter.
const MARKET_INDEX_RUN_ID_PREFIX = 'index:';
const US_INDEX_SERIES_LABELS = new Set(['DJIA index', 'Nasdaq-100']);

function isUsMarketIndexSeries(entry) {
    if (typeof entry?.run_id === 'string' && entry.run_id.startsWith(MARKET_INDEX_RUN_ID_PREFIX)) {
        return true;
    }
    return US_INDEX_SERIES_LABELS.has(entry?.label);
}

function filterIfindChartSeries(series, run = window.SELECTED_RUN) {
    if (run?.data_source !== IFIND_ASHARE_SOURCE) return series;
    return series.filter((entry) => !isUsMarketIndexSeries(entry));
}

function formatBacktestRunLabel(run) {
    return [formatBacktestRunPrimary(run), formatBacktestRunSecondary(run)].filter(Boolean).join(' · ');
}

window.formatBacktestRunPrimary = formatBacktestRunPrimary;
window.formatBacktestRunSecondary = formatBacktestRunSecondary;
window.formatBacktestRunLabel = formatBacktestRunLabel;

function resolveSelectedExternalRun(externalRuns) {
    const selectedId = localStorage.getItem(SELECTED_BACKTEST_RUN_KEY);
    if (selectedId) {
        const match = externalRuns.find((r) => r.run_id === selectedId);
        if (match) return match;
    }
    return latestRun([...externalRuns]);
}

function populateBacktestRunSelector(externalRuns, { runningId = null } = {}) {
    const select = document.getElementById('backtestRunSelect');
    if (!select) return;

    const sorted = [...externalRuns].sort(
        (a, b) => (b.created_at || '').localeCompare(a.created_at || ''),
    );

    if (runningId && !sorted.some((run) => run.run_id === runningId)) {
        const cfg = getBacktestLaunchConfig(runningId);
        sorted.unshift({
            run_id: runningId,
            agent_name: cfg?.agentName || 'Agent',
            created_at: cfg?.startedAt || '',
            _running: true,
        });
    }

    if (!sorted.length) {
        select.innerHTML = '';
        select.hidden = true;
        return;
    }

    select.hidden = false;
    const previous = select.value || localStorage.getItem(SELECTED_BACKTEST_RUN_KEY);
    select.innerHTML = sorted
        .map((run) => {
            const isRunning = run._running || run.run_id === runningId;
            const label = isRunning
                ? `Running… · ${formatBacktestRunPrimary(run)}`
                : formatBacktestRunLabel(run);
            return `<option value="${escapeHtml(run.run_id)}">${escapeHtml(label)}</option>`;
        })
        .join('');

    const selectedId =
        (runningId && sorted.some((r) => r.run_id === runningId) && (!previous || previous === runningId))
            ? runningId
            : (previous && sorted.some((r) => r.run_id === previous)
                ? previous
                : sorted[0].run_id);
    select.value = selectedId;
    localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, selectedId);
}

function resolveBaselineRunIds(extRun, sessionRuns) {
    if (!extRun) return { djia: null, buyhold: null };

    let djia = extRun.baseline_djia_run_id || null;
    let buyhold = extRun.baseline_buyhold_run_id || null;
    if (djia && buyhold) {
        return { djia, buyhold };
    }

    const extCreated = extRun.created_at || '';
    const { start_date: startDate, end_date: endDate } = extRun;
    const extRuns = sessionRuns
        .filter(isExternalAgentRun)
        .sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));
    const extIdx = extRuns.findIndex((r) => r.run_id === extRun.run_id);
    const nextExtCreated =
        extIdx >= 0 && extIdx < extRuns.length - 1
            ? extRuns[extIdx + 1].created_at
            : null;

    function pick(agentName) {
        const candidates = sessionRuns
            .filter(
                (r) =>
                    r.agent_name === agentName &&
                    r.start_date === startDate &&
                    r.end_date === endDate &&
                    (r.created_at || '') >= extCreated &&
                    (!nextExtCreated || (r.created_at || '') < nextExtCreated),
            )
            .sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));
        return candidates[0]?.run_id || null;
    }

    return {
        djia: djia || pick('DJIA'),
        buyhold: buyhold || pick('buy-and-hold'),
    };
}

function findLatestRunByAgent(runs, agentName) {
    const matched = runs.filter(r => r.agent_name === agentName);
    if (!matched.length) return null;
    return matched.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))[0];
}

// Baseline comparison series. They appear on the plot but are never listed or
// selectable as standalone runs.
const BASELINE_AGENT_NAMES = ['DJIA', 'buy-and-hold'];

function isBaselineRun(run) {
    return !!run && BASELINE_AGENT_NAMES.includes(run.agent_name);
}

function _runTime(value) {
    return new Date(String(value || '').replace(' ', 'T')).getTime() || 0;
}

// The selected run drives the whole backtest view. Built-in and external agents
// take the same path: prefer the explicitly clicked/selected run_id, else the
// agent's most recent (non-baseline) run.
function resolveSelectedRun(sessionRuns) {
    const realRuns = (sessionRuns || []).filter(r => !isBaselineRun(r));
    if (!realRuns.length) return null;
    const selectedId = localStorage.getItem(SELECTED_BACKTEST_RUN_KEY);
    if (selectedId) {
        const match = realRuns.find(r => r.run_id === selectedId);
        if (match) return match;
    }
    return latestRun(realRuns);
}

function beginBacktestSurfaceRequest(runId) {
    return { seq: ++backtestSurfaceRequestSeq, runId };
}

function isCurrentBacktestSurfaceRequest(token) {
    return token?.seq === backtestSurfaceRequestSeq
        && token.runId === window.SELECTED_RUN?.run_id
        && !liveBacktestChartActive;
}

async function loadHistoricalBacktestSurfaces(selectedRun) {
    const token = beginBacktestSurfaceRequest(selectedRun.run_id);
    clearPerformanceComparison('loading', 'Loading performance comparison...');
    clearTradingLog('Loading orders...');
    const chartUrl = `${API_BASE}/api/backtest/${encodeURIComponent(selectedRun.run_id)}/chart-data?t=${Date.now()}`;

    const chartRequest = API.get(chartUrl).then((payload) => {
        if (!isCurrentBacktestSurfaceRequest(token)) return;
        backtestChartData = payload;
        initializeCharts();
    }).catch((error) => {
        if (!isCurrentBacktestSurfaceRequest(token)) return;
        backtestChartData = null;
        if (chartInstance) {
            chartInstance.destroy();
            chartInstance = null;
        }
        const notice = document.getElementById('chartBaselineNotice');
        if (notice) notice.hidden = true;
        renderPerformanceLegend({ columns: [] });
        setPerformanceComparisonState(
            'error',
            'Performance comparison is unavailable. Reload to retry.',
        );
        console.warn('Could not load performance comparison:', error.message);
    });

    const logRequest = loadTradingLogForRun(selectedRun.run_id, {
        isCurrent: () => isCurrentBacktestSurfaceRequest(token),
    });
    await Promise.allSettled([chartRequest, logRequest]);
}

// Find the DJIA / buy-and-hold runs that belong to a given run: same session,
// same date window, created closest in time to the run (baselines are written
// seconds apart from the agent run).
function resolveBaselinesForRun(run, sessionRuns) {
    if (!run) return { djia: null, buyhold: null };
    const anchor = _runTime(run.created_at);
    function pick(agentName, explicitId) {
        if (explicitId) return explicitId;
        const candidates = (sessionRuns || []).filter(r =>
            r.agent_name === agentName &&
            r.start_date === run.start_date &&
            r.end_date === run.end_date);
        if (!candidates.length) return null;
        candidates.sort((a, b) =>
            Math.abs(_runTime(a.created_at) - anchor) - Math.abs(_runTime(b.created_at) - anchor));
        return candidates[0].run_id;
    }
    return {
        djia: pick('DJIA', run.baseline_djia_run_id),
        buyhold: pick('buy-and-hold', run.baseline_buyhold_run_id),
    };
}

/**
 * Load dashboard data from backend API
 */
async function loadData() {
    try {
        console.log('Loading data for mode:', currentMode);
        
        if (currentMode === 'backtest') {
            let sessionRuns = [];
            try {
                sessionRuns = await API.get(`${API_BASE}/api/backtest/runs?t=${Date.now()}`);
            } catch (e) {
                console.warn('Session runs unavailable:', e.message);
            }

            let runningId = liveBacktestRunId || null;
            let statusProgress = null;
            try {
                const status = await API.get(`${API_BASE}/backtest/status`);
                if (status?.running && status.live_run_id) {
                    runningId = status.live_run_id;
                    liveBacktestRunId = runningId;
                    statusProgress = status.progress || null;
                    ensureBacktestPolling();
                } else if (!status?.running) {
                    liveBacktestRunId = null;
                }
            } catch (_statusError) {
                /* status optional while browsing history */
            }

            if (!runningId && (liveBacktestLaunchPending || liveBacktestLaunchError)) {
                return;
            }

            const selectableRuns = sessionRuns.filter(r => !isBaselineRun(r));
            populateBacktestRunSelector(selectableRuns, { runningId });

            const selectedId = localStorage.getItem(SELECTED_BACKTEST_RUN_KEY);

            // Dropdown (or deep-link) onto the in-flight run — always attach live
            // surface even if the run is not in DB yet (synthetic selector option).
            if (runningId && selectedId === runningId) {
                attachToLiveBacktest(
                    runningId,
                    statusProgress,
                    getBacktestLaunchConfig(runningId),
                );
                return;
            }

            // Viewing a finished run while another job may still be running.
            liveBacktestChartActive = false;
            showBacktestRunProgress(false);

            const selectedRun = resolveSelectedRun(sessionRuns);

            window.SELECTED_RUN = selectedRun;
            window.MY_ALGO_RUN_ID = isMyAlgoRun(selectedRun) ? selectedRun.run_id : null;
            window.EXTERNAL_AGENT_RUN_ID = isExternalAgentRun(selectedRun) ? selectedRun.run_id : null;
            renderBacktestDataSourceBadge(selectedRun);

            if (!selectedRun) {
                console.warn('No backtest runs for this session');
                comparisonData = null;
                backtestChartData = null;
                clearPerformanceComparison(
                    'empty',
                    runningId
                        ? 'Select the Running run to watch live progress.'
                        : 'No completed backtests yet.',
                );
                clearTradingLog(
                    runningId
                        ? 'Select the Running run to watch live progress.'
                        : 'No backtests yet. Run one from My Agents.',
                );
                if (runningId) {
                    renderBacktestRunConfig(
                        { run_id: runningId },
                        { running: true, launchConfig: getBacktestLaunchConfig(runningId) },
                    );
                } else {
                    renderBacktestRunConfig(null);
                }
                return;
            }

            localStorage.setItem(SELECTED_BACKTEST_RUN_KEY, selectedRun.run_id);
            const baselineIds = resolveBaselinesForRun(selectedRun, sessionRuns);
            const selectedBuyholdRun = sessionRuns.find(
                run => run.run_id === baselineIds.buyhold,
            ) || null;
            renderBacktestRunConfig(selectedRun, {
                running: false,
                launchConfig: getBacktestLaunchConfig(selectedRun.run_id),
                baselineRun: selectedBuyholdRun,
            });

            if (isViewingLiveBacktest(runningId)) return;
            await loadHistoricalBacktestSurfaces(selectedRun);
        }
        
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

/**
 * Initialize charts with real data from backend.
 * Agent vs DJIA index + Nasdaq-100 (same baselines as Discord plot.png).
 */
function initializeCharts() {
    if (liveBacktestChartActive) {
        console.log('Skipping historical chart paint — live backtest view is active');
        return;
    }
    if (!backtestChartData || !backtestChartData.series || !backtestChartData.series.length) {
        console.warn('No backtest chart data available');
        return;
    }

    // Missing index benchmarks are only visible as *fewer lines*, which reads as
    // "this agent has no benchmark". Say which it is. Older payloads omit the
    // flag entirely, so only an explicit false shows the notice.
    const baselineNotice = document.getElementById('chartBaselineNotice');
    if (baselineNotice) {
        baselineNotice.hidden = backtestChartData.index_baselines_ok !== false;
    }

    const perfCtx = document.getElementById('performanceChart');
    if (perfCtx && perfCtx.getContext) {
        if (chartInstance) {
            chartInstance.destroy();
        }

        const ctx = perfCtx.getContext('2d');
        const { timestamps, x_labels: xLabels, series } = backtestChartData;
        const visibleSeries = filterIfindChartSeries(series);
        const comparisonPayload = { ...backtestChartData, series: visibleSeries };
        const model = window.BacktestComparison.buildModel(
            comparisonPayload,
            window.SELECTED_RUN,
        );
        const chartColumns = model.columns.filter((column) => column.available);
        const datasets = chartColumns.map((column) => ({
            label: column.label,
            comparisonKey: column.key,
            data: column.values,
            borderColor: column.color,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            borderDash: column.dashed ? [6, 4] : [],
            tension: 0,
            fill: false,
            pointRadius: 0,
            pointHoverRadius: 5,
        }));

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: xLabels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: false,
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        titleColor: '#e5e7eb',
                        bodyColor: '#e5e7eb',
                        borderColor: '#1f2937',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {
                            title: function(context) {
                                if (context.length > 0) {
                                    const dataIndex = context[0].dataIndex;
                                    const timestamp = timestamps[dataIndex];
                                    try {
                                        const date = new Date(timestamp);
                                        const month = date.toLocaleString('en-US', { month: 'short' });
                                        const day = date.getDate();
                                        const hour = String(date.getHours()).padStart(2, '0');
                                        return `${month} ${day} ${hour}:00`;
                                    } catch (e) {
                                        return timestamp;
                                    }
                                }
                                return '';
                            },
                            label: function(context) {
                                const value = context.parsed.y;
                                return context.dataset.label + ': $' + value.toFixed(0);
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            color: '#e5e7eb',
                            font: { size: 11, weight: '500' },
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        },
                        grid: {
                            color: '#1f2937',
                            drawBorder: false,
                        },
                    },
                    x: {
                        ticks: {
                            color: '#e5e7eb',
                            font: { size: 11, weight: '500' },
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 8,
                            callback: function(_value, index) {
                                const label = xLabels[index];
                                return label || undefined;
                            },
                        },
                        grid: {
                            display: false,
                            drawBorder: false,
                        }
                    }
                }
            }
        });

        renderPerformanceComparison(comparisonPayload, window.SELECTED_RUN);
        renderPerformanceLegend(model);
        liveBacktestChartActive = false;
        console.log('✅ Chart initialized -', chartColumns.map((column) => column.label).join(', '));
    }
}

/**
 * Format agent label for display
 */
function formatAgentLabel(agentName) {
    const labels = {
        'Agent': 'Selected Agent (Claude)',
        'buy-and-hold': 'Market Baseline (SPY)',
        'equal-weight': 'Equal-Weight Baseline',
        'deepseek': 'DeepSeek Agent'
    };
    return labels[agentName] || agentName;
}

/**
 * Format timestamps for chart labels
 */
function formatTimestamps(timestamps) {
    if (!timestamps || timestamps.length === 0) {
        return generateDateLabels(8);
    }
    
    return timestamps.map(ts => {
        try {
            const date = new Date(ts);
            const month = date.toLocaleString('en-US', { month: 'short' });
            const day = date.getDate();
            return `${month} ${day}`;
        } catch (e) {
            return ts;
        }
    });
}

/**
 * Generate date labels (fallback)
 */
function generateDateLabels(days) {
    const labels = [];
    const startDate = new Date(2026, 3, 15);
    
    for (let i = 0; i < days; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);
        const month = date.toLocaleString('en-US', { month: 'short' });
        const day = date.getDate();
        labels.push(`${month} ${day}`);
    }
    
    return labels;
}

/**
 * Format currency
 */
function formatCurrency(value) {
    return '$' + value.toLocaleString('en-US', { 
        minimumFractionDigits: 2,
        maximumFractionDigits: 2 
    });
}

/**
 * Format percentage
 */
function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

/**
 * ============================================================================
 * PAPER TRADING MODE
 * ============================================================================
 */

/**
 * Load all paper trading data in parallel
 */
async function loadPaperTradingData() {
    console.log('Loading paper trading data...');
    
    try {
        // Fetch all data in parallel
        const [accountRes, positionsRes, historyRes, tradesRes] = await Promise.all([
            fetch(`${API_BASE}/paper/account?t=${Date.now()}`),
            fetch(`${API_BASE}/paper/positions?t=${Date.now()}`),
            fetch(`${API_BASE}/paper/portfolio-history?t=${Date.now()}`),
            fetch(`${API_BASE}/paper/trades?t=${Date.now()}`)
        ]);
        
        // Parse responses
        const accountData = accountRes.ok ? await accountRes.json() : null;
        const positionsData = positionsRes.ok ? await positionsRes.json() : null;
        const historyData = historyRes.ok ? await historyRes.json() : null;
        const tradesData = tradesRes.ok ? await tradesRes.json() : null;
        
        console.log('✅ All paper trading data loaded');
        console.log('  Account:', accountData?.account);
        console.log('  Positions:', positionsData?.positions?.length || 0);
        console.log('  Equity curve points:', historyData?.equity_curve?.length || 0);
        console.log('  Recent trades:', tradesData?.trades?.length || 0);
        
        // Display account metrics
        if (accountData?.success && accountData?.account) {
            displayAccountMetrics(accountData.account);
        }
        
        // Display positions
        if (positionsData?.success && positionsData?.positions) {
            displayPositions(positionsData.positions);
        }
        
        // Display equity curve
        if (historyData?.success && historyData?.equity_curve) {
            await displayEquityCurve(historyData.equity_curve);
        }
        
        // Display trades
        if (tradesData?.success && tradesData?.trades) {
            displayTrades(tradesData.trades);
        }
        
    } catch (error) {
        console.error('Error loading paper trading data:', error);
        displayPaperError('Failed to load paper trading data: ' + error.message);
    }
}

/**
 * Display account metrics
 */
function displayAccountMetrics(account) {
    console.log('Displaying account metrics:', account);
    
    // Portfolio Value (use equity)
    const portfolioEl = document.getElementById('portfolioValue');
    if (portfolioEl) {
        const equity = parseFloat(account.equity) || parseFloat(account.portfolio_value) || 0;
        portfolioEl.textContent = formatCurrency(equity);
        portfolioEl.className = 'paper-value';
    }
    
    // Cash
    const cashEl = document.getElementById('cashValue');
    if (cashEl) {
        const cash = parseFloat(account.cash) || 0;
        cashEl.textContent = formatCurrency(cash);
        cashEl.className = 'paper-value';
    }
    
    // Buying Power
    const buyingPowerEl = document.getElementById('buyingPowerValue');
    if (buyingPowerEl) {
        const buyingPower = parseFloat(account.buying_power) || 0;
        buyingPowerEl.textContent = formatCurrency(buyingPower);
        buyingPowerEl.className = 'paper-value';
    }
    
    // Day P&L (try to get from account, fallback to 0)
    const dayPnLEl = document.getElementById('dayPnL');
    if (dayPnLEl) {
        const dayPnL = parseFloat(account.day_pnl) || 0;
        dayPnLEl.textContent = (dayPnL >= 0 ? '+' : '') + formatCurrency(dayPnL);
        dayPnLEl.className = 'paper-value ' + (dayPnL >= 0 ? 'positive' : 'negative');
    }
}

/**
 * Display positions list
 */
function displayPositions(positions) {
    console.log('Displaying positions:', positions.length);
    
    const positionsList = document.getElementById('positionsList');
    if (!positionsList) return;
    
    if (!positions || positions.length === 0) {
        positionsList.innerHTML = '<div class="loading">No open positions</div>';
        return;
    }
    
    positionsList.innerHTML = positions.map(pos => {
        const qty = parseFloat(pos.qty) || 0;
        const currentPrice = parseFloat(pos.current_price) || 0;
        const unrealizedPnL = parseFloat(pos.unrealized_pl) || 0;
        const unrealizedPnLPercent = parseFloat(pos.unrealized_plpc) || 0;
        const isPositive = unrealizedPnL >= 0;
        
        return `
            <div class="position-item">
                <div style="flex: 1;">
                    <div class="position-symbol">${pos.symbol}</div>
                    <div class="position-qty">${Math.abs(qty)} @ $${currentPrice.toFixed(2)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="position-pnl ${isPositive ? 'positive' : 'negative'}">
                        ${isPositive ? '+' : ''}$${unrealizedPnL.toFixed(2)}
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted);">
                        ${isPositive ? '+' : ''}${(unrealizedPnLPercent * 100).toFixed(2)}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Display equity curve chart
 */
async function displayEquityCurve(equityCurve) {
    console.log('Displaying equity curve with', equityCurve.length, 'points');
    
    const canvas = document.getElementById('paperEquityChart');
    if (!canvas) return;
    
    // Destroy existing chart if any
    if (window.paperChartInstance) {
        window.paperChartInstance.destroy();
    }
    
    const ctx = canvas.getContext('2d');
    
    // Extract timestamps and equity values
    const timestamps = equityCurve.map(point => {
        const date = new Date(point.timestamp);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    });
    
    const equityValues = equityCurve.map(point => parseFloat(point.equity) || 0);
    
    // Fetch DJIA baseline
    let djiaValues = [];
    try {
        const response = await fetch(`${API_BASE}/paper/baselines?t=${Date.now()}`);
        if (response.ok) {
            const data = await response.json();
            if (data.baselines && data.baselines.djia) {
                djiaValues = data.baselines.djia.map(point => parseFloat(point.equity) || 0);
                console.log('✅ DJIA baseline loaded:', djiaValues.length, 'points');
            }
        }
    } catch (error) {
        console.warn('Could not fetch DJIA baseline:', error.message);
    }
    
    // Build datasets
    const datasets = [{
        label: 'Your Portfolio',
        data: equityValues,
        borderColor: '#4FC3F7',
        backgroundColor: 'transparent',
        borderWidth: 2.5,
        fill: false,
        tension: 0,
        pointRadius: 0,
        pointHoverRadius: 5
    }];
    
    // Add DJIA if available
    if (djiaValues.length === equityValues.length) {
        datasets.push({
            label: 'DJIA Index',
            data: djiaValues,
            borderColor: '#F5C04A',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            fill: false,
            tension: 0,
            pointRadius: 0,
            pointHoverRadius: 5
        });
    }
    
    // Create chart
    window.paperChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#e5e7eb',
                        font: { size: 12, weight: '600' },
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'line',
                        boxWidth: 12,
                        boxHeight: 2,
                    }
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(0, 0, 0, 0.9)',
                    titleColor: '#e5e7eb',
                    bodyColor: '#e5e7eb',
                    borderColor: '#1f2937',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            return context.dataset.label + ': $' + value.toFixed(0);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        color: '#e5e7eb',
                        font: { size: 11, weight: '500' },
                        callback: (value) => formatCurrency(value)
                    },
                    grid: {
                        color: '#1f2937',
                        drawBorder: false
                    }
                },
                x: {
                    ticks: {
                        color: '#e5e7eb',
                        font: { size: 11, weight: '500' },
                        maxRotation: 45,
                        minRotation: 0
                    },
                    grid: {
                        display: false,
                        drawBorder: false
                    }
                }
            }
        }
    });
}

/**
 * Display recent trades
 */
function displayTrades(trades) {
    console.log('Displaying trades:', trades.length);
    
    const tradesList = document.getElementById('tradesList');
    if (!tradesList) return;
    
    if (!trades || trades.length === 0) {
        tradesList.innerHTML = '<div class="loading">No recent trades</div>';
        return;
    }
    
    // Show latest 20 trades
    const recentTrades = trades.slice(0, 20);
    
    tradesList.innerHTML = recentTrades.map(trade => {
        // Parse timestamp from trade ID or use current time as fallback
        let timeStr = '--:--';
        if (trade.timestamp) {
            const date = new Date(trade.timestamp);
            timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        } else if (trade.id) {
            // Extract timestamp from ID format like "20260430093148799"
            const idParts = trade.id.split('::');
            if (idParts[0].length >= 14) {
                const ts = idParts[0];
                const hour = parseInt(ts.substring(8, 10));
                const minute = parseInt(ts.substring(10, 12));
                timeStr = `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
            }
        }
        
        const side = (trade.side || 'hold').toLowerCase();
        const qty = Math.abs(parseFloat(trade.qty) || 0);
        const price = parseFloat(trade.price) || 0;
        
        return `
            <div class="trade-item">
                <div style="flex: 1;">
                    <div class="trade-symbol">${trade.symbol}</div>
                    <div class="trade-qty">${qty} @ $${price.toFixed(2)}</div>
                </div>
                <div style="text-align: right;">
                    <div class="trade-side ${side}">${side.toUpperCase()}</div>
                    <div class="trade-time">${timeStr}</div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Refresh paper trading data
 */
async function refreshPaperData() {
    const btn = document.querySelector('.paper-refresh-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = '⏳ Refreshing...';
    }
    
    await loadPaperTradingData();
    
    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Refresh';
    }
}

/**
 * Display error message in paper trading view
 */
function displayPaperError(message) {
    console.error('Paper trading error:', message);
    
    const positionsList = document.getElementById('positionsList');
    if (positionsList) {
        positionsList.innerHTML = `<div class="loading" style="color: var(--danger-color);">Error: ${escapeHtml(message)}</div>`;
    }
}

// ============================================================================
// My Trading Algo
// ============================================================================

const ALGO_BLOCK_FIELDS = {
    info_retrieval: 'blockInfoRetrieval',
    signal_transfer: 'blockSignalTransfer',
    trading_algorithm: 'blockTradingAlgorithm',
    stop_loss_take_profit: 'blockStopLoss',
};

const DEFAULT_ALGO_BLOCKS = {
    info_retrieval: "Monitor Trump's Twitter / X feed; capture tweets and sentiment signals",
    signal_transfer: 'AI auto-selects target stocks (single name or basket); map tickers from tweet semantics',
    trading_algorithm: 'No execution algo: buy whatever Trump mentions (immediate market follow)',
    stop_loss_take_profit: 'Stop loss: exit if position down 5%; take profit: hold after +20%; daily stop: exit if down 5% intraday',
};

function getAlgoBlocksFromUI() {
    return {
        info_retrieval: document.getElementById('blockInfoRetrieval')?.value?.trim() || '',
        signal_transfer: document.getElementById('blockSignalTransfer')?.value?.trim() || '',
        trading_algorithm: document.getElementById('blockTradingAlgorithm')?.value?.trim() || '',
        stop_loss_take_profit: document.getElementById('blockStopLoss')?.value?.trim() || '',
    };
}

function setAlgoBlocksToUI(blocks) {
    for (const [key, fieldId] of Object.entries(ALGO_BLOCK_FIELDS)) {
        const el = document.getElementById(fieldId);
        if (el && blocks[key] !== undefined) {
            el.value = blocks[key];
        }
    }
}

function highlightAlgoBlocks(updatedKeys) {
    document.querySelectorAll('.algo-block-card').forEach(card => card.classList.remove('highlight'));
    if (!updatedKeys?.length) return;
    for (const key of updatedKeys) {
        const card = document.querySelector(`.algo-block-card[data-block="${key}"]`);
        if (card) card.classList.add('highlight');
    }
    setTimeout(() => {
        document.querySelectorAll('.algo-block-card').forEach(card => card.classList.remove('highlight'));
    }, 2500);
}

/**
 * Render a chat bubble's text as HTML, supporting only `**bold**`.
 *
 * Escape first, then add the markup: every caller passes text the server
 * controls (an `err.message` carrying a backend `detail` or a backtest job's
 * stderr tail, the LLM's `reply`, the echoed `team_name`), so the raw string
 * must never reach `innerHTML`. Escaping leaves `*` alone, so the bold markers
 * still survive; the only live tags are the ones we generate here.
 */
function renderAlgoChatHtml(text) {
    return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

function appendAlgoChatMessage(text, role = 'bot') {
    const container = document.getElementById('algoChatMessages');
    if (!container) return;
    const row = document.createElement('div');
    row.className = `algo-chat-msg ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'algo-chat-bubble';
    bubble.innerHTML = renderAlgoChatHtml(text);
    row.appendChild(bubble);
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
}

async function loadMyTradingAlgoPage() {
    if (!myAlgoInitialized) {
        initMyTradingAlgoUI();
        myAlgoInitialized = true;
    }
    try {
        const res = await API.get(`${API_BASE}/api/algo/defaults`);
        if (res.blocks) {
            setAlgoBlocksToUI(res.blocks);
        }
        if (res.backtest_window) {
            window.ALGO_BACKTEST_WINDOW = res.backtest_window;
            const statusEl = document.getElementById('algoExecuteStatus');
        if (statusEl) {
            statusEl.hidden = false;
            statusEl.className = 'algo-execute-status';
                statusEl.textContent =
                `Example strategy (edit before Execute). Backtest window: ${res.backtest_window.start_date} → ${res.backtest_window.end_date}`;
        }

        try {
            const setup = await API.get(`${API_BASE}/api/algo/setup`);
            renderAlgoSetupStatus(setup);
        } catch (setupErr) {
            renderAlgoSetupStatus(null, setupErr.message);
        }
        }
    } catch {
        setAlgoBlocksToUI(DEFAULT_ALGO_BLOCKS);
    }
}

function initMyTradingAlgoUI() {
    setAlgoBlocksToUI(DEFAULT_ALGO_BLOCKS);

    const sendBtn = document.getElementById('algoChatSendBtn');
    const input = document.getElementById('algoChatInput');
    const executeBtn = document.getElementById('executeAlgoBtn');

    const sendChat = async () => {
        const message = input?.value?.trim();
        if (!message) return;
        appendAlgoChatMessage(message, 'user');
        input.value = '';
        sendBtn.disabled = true;
        appendAlgoChatMessage('Thinking…', 'bot');

        try {
            const data = await API.post(`${API_BASE}/api/algo/chat`, {
                message,
                blocks: getAlgoBlocksFromUI(),
            });
            const msgs = document.getElementById('algoChatMessages');
            if (msgs && msgs.lastElementChild?.textContent === 'Thinking…') {
                msgs.removeChild(msgs.lastElementChild);
            }
            setAlgoBlocksToUI(data.blocks);
            syncAlgoTeamNameFromBlocks(data.blocks);
            highlightAlgoBlocks(data.updated_blocks);
            appendAlgoChatMessage(data.reply, 'bot');
        } catch (err) {
            appendAlgoChatMessage(`Error: ${err.message}`, 'bot');
        } finally {
            sendBtn.disabled = false;
            input.focus();
        }
    };

    sendBtn?.addEventListener('click', sendChat);
    input?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendChat();
        }
    });

    executeBtn?.addEventListener('click', executeMyTradingAlgo);
}

function syncAlgoTeamNameFromBlocks(blocks) {
    const nameInput = document.getElementById('algoTeamName');
    if (!nameInput) return;
    const info = (blocks.info_retrieval || '').toLowerCase();
    if (info.includes('musk') || (blocks.info_retrieval || '').toLowerCase().includes('musk')) {
        nameInput.value = 'Elon Musk Twitter Algo';
    } else if (info.includes('trump')) {
        nameInput.value = 'Trump Twitter Algo';
    }
}

function renderAlgoSetupStatus(setup, errorMsg) {
    let el = document.getElementById('algoSetupStatus');
    if (!el) {
        el = document.createElement('div');
        el.id = 'algoSetupStatus';
        el.className = 'algo-setup-status';
        const panel = document.querySelector('.algo-blocks-panel');
        if (panel) panel.appendChild(el);
    }
    el.hidden = false;

    if (errorMsg || !setup) {
        el.className = 'algo-setup-status error';
        el.innerHTML =
            '⚠️ Cannot reach My Trading Algo API (HTTP 404). <strong>Restart the backend</strong>: ' +
            '<code>python backend/app.py</code>, then open <code>http://localhost:8000</code>';
        return;
    }

    if (setup.ready) {
        el.className = 'algo-setup-status success';
        el.textContent = '✅ API keys configured. Edit your strategy, then Execute for a real backtest.';
        return;
    }

    const missing = [];
    if (!setup.anthropic_configured) missing.push('ANTHROPIC_API_KEY');
    if (!setup.alpaca_configured) missing.push('Alpaca (credentials/alpaca.json or env vars)');
    el.className = 'algo-setup-status error';
    el.textContent = `⚠️ Missing: ${missing.join(', ')}. Configure .env and restart the backend.`;
}

async function pollAlgoBacktestStatus() {
    const maxAttempts = 360;
    for (let i = 0; i < maxAttempts; i++) {
        let status;
        try {
            status = await API.get(`${API_BASE}/api/algo/status`);
        } catch (err) {
            if (String(err.message).includes('404')) {
                throw new Error(
                    'Backend missing /api/algo/status (old version). Stop with Ctrl+C and run: python backend/app.py'
                );
            }
            throw err;
        }
        const statusEl = document.getElementById('algoExecuteStatus');
        const btn = document.getElementById('executeAlgoBtn');

        if (status.running) {
            if (statusEl) {
                statusEl.textContent = status.progress || `Backtest running… (${i + 1}/${maxAttempts})`;
            }
            if (btn) btn.textContent = `⏳ Running… ${Math.floor(i * 5 / 60)}m`;
            await new Promise(r => setTimeout(r, 5000));
            continue;
        }

        if (status.error) {
            throw new Error(status.error);
        }

        if (status.result) {
            return status.result;
        }

        await new Promise(r => setTimeout(r, 3000));
    }
    throw new Error('Backtest timed out. Check the Backtest tab later.');
}

async function executeMyTradingAlgo() {
    const btn = document.getElementById('executeAlgoBtn');
    const statusEl = document.getElementById('algoExecuteStatus');
    const teamName = document.getElementById('algoTeamName')?.value?.trim();
    const blocks = getAlgoBlocksFromUI();

    const isDefault = Object.keys(DEFAULT_ALGO_BLOCKS).every(
        k => (blocks[k] || '').trim() === (DEFAULT_ALGO_BLOCKS[k] || '').trim()
    );
    if (isDefault) {
        if (statusEl) {
            statusEl.hidden = false;
            statusEl.className = 'algo-execute-status error';
            statusEl.textContent = 'Edit the strategy (chat or blocks) before Execute. The example config does not run a real backtest.';
        }
        appendAlgoChatMessage(
            'Edit all four modules before Execute. Leaderboard teams are mock; only your customized strategy uses real data on Backtest.',
            'bot'
        );
        return;
    }

    btn.disabled = true;
    btn.textContent = '⏳ Starting…';
    if (statusEl) {
        statusEl.hidden = false;
        statusEl.className = 'algo-execute-status';
        statusEl.textContent = 'Submitting backtest — real market data + AI…';
    }

    try {
        const job = await API.post(`${API_BASE}/api/algo/execute`, {
            blocks,
            team_name: teamName || undefined,
        });

        if (statusEl) {
            statusEl.textContent = job.message || 'Backtest started. Please wait…';
        }

        const result = await pollAlgoBacktestStatus();
        const m = result.metrics;

        if (statusEl) {
            statusEl.className = 'algo-execute-status success';
            statusEl.textContent = `✅ ${result.message} Opening Backtest…`;
        }

        const retPct = (m.cumulative_return * 100).toFixed(2);
        appendAlgoChatMessage(
            `Backtest complete: "${result.team_name}" (${result.start_date} → ${result.end_date}).\n` +
            `Return ${retPct}%, Sharpe ${m.sharpe_ratio}, ${result.num_trades} trades.\n` +
            `Switched to Backtest to view your MY ALGO curve (vs DJIA / Buy-and-Hold).`,
            'bot'
        );

        if (result.run_id) {
            window.MY_ALGO_RUN_ID = result.run_id;
        }
        switchMode('backtest');
    } catch (err) {
        if (statusEl) {
            statusEl.className = 'algo-execute-status error';
            statusEl.textContent = `Execution failed: ${err.message}`;
        }
        appendAlgoChatMessage(`Backtest failed: ${err.message}`, 'bot');
    } finally {
        btn.disabled = false;
        btn.textContent = '▶ Execute Algo';
    }
}

console.log('Frontend loaded - connecting to API at ' + API_BASE);
