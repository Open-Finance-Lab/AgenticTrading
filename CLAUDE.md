# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Packaging contract.** The backend is the **`dashboard.backend` Python package**
> (restructured from flat top-level modules in PR #67, hardened by PR #71 — both long
> since merged; this packaged layout is the only one that exists now). Import by full
> package path; run via the app's import string, never by file path. See
> `docs/architecture/dashboard-target-structure.md` for the full layout.

## What this repo is

**Agentic Trading Lab** — an open-source platform for LLM-powered trading agents (backtests, live paper trading, decision-log inspection, leaderboards). The live app is `dashboard/`; everything else is supporting (docs, a PyPI client, and an imported research framework).

The repo is **two loosely-coupled subsystems**:

- **`dashboard/`** — the shipping product. FastAPI backend + static frontend + backtest CLIs. This is where almost all day-to-day work happens.
- **`orchestration/`** — the *FinAgent Orchestration Framework* (imported research code from the NeurIPS 2025 paper). Largely self-contained, hardcoded absolute conda paths, **not** wired into the dashboard. Treat it as a separate project unless explicitly asked.

## Critical: the backend is the `dashboard.backend` package

`dashboard/backend/` is a proper Python package: modules import each other by **full package path** — `from dashboard.backend.database import db`, `from dashboard.backend.app import app`, `from dashboard.backend.paths import ...`. The repo root must be on `sys.path` (it is, when you run from the repo root or via `uvicorn`/`python -m`).

Consequences:
- **Run the app** with the app referenced by its import string, from the repo root — never by running the file directly. `python dashboard/backend/app.py` does **not** work (the top-level `dashboard.backend.*` imports fail without the repo root on `sys.path`).
- `app.py` has a `__main__` block that calls `uvicorn.run("dashboard.backend.app:app", …)` — a real `python -m dashboard.backend.app` entrypoint that references the app by canonical import string so the reloader keeps one module identity.
- Domain logic lives under `dashboard/backend/domain/<area>/` and must **not** import `api/` or `app.py` (enforced by `tests/test_architecture_boundaries.py`).

## Common commands

Run from the **repo root** unless noted.

```bash
# Install deps (the real dependency file — NOT root pyproject.toml)
pip install -r requirements.txt

# Run the backend + dashboard locally (serves frontend at http://localhost:8000)
uvicorn dashboard.backend.app:app --reload          # canonical
python -m dashboard.backend.app                       # equivalent module entrypoint

# Run tests (pytest; install it first — not in requirements.txt)
pytest dashboard/backend/tests/ -v
pytest dashboard/backend/tests/test_protocol_api.py -v            # single file
# The PyPI SDK has its own suite:
pytest packaging/agentictrading/tests/ -v

# Backtest CLIs (from the repo root)
python dashboard/scripts/backtest_hourly_agent.py     # main hourly agent backtest
```

`dashboard/backend/tests/conftest.py` points `DATABASE_PATH` at a temp file before any backend import, so tests never touch the committed `dashboard/storage/data/backtest.db`. The suite is green end-to-end (the old "5 pre-existing failures" were retired in PR #71) — a red test on a fresh run is a real regression.

## Environment & credentials

- `app.py` loads `.env` from **`dashboard/.env`**, not the repo root. `.env.example` (repo root) lists the keys: `ALPACA_API_KEY`/`ALPACA_SECRET_KEY` (paper API) and optionally `ANTHROPIC_API_KEY`/`COMMONSTACK_API_KEY`/`OPENAI_API_KEY`/`DEEPSEEK_API_KEY`.
- `DATABASE_PATH` overrides the SQLite location (defaults to `dashboard/storage/data/backtest.db`; Render mounts a persistent disk at `/data`). Protocol runs (`domain/runs/repository.py`: `protocol_runs`/`protocol_steps`) and the hot `idempotency_keys` table always live in this DB — there is no Postgres option for them. Backtest run history no longer does by default; see `AGENT_RUNS_DATABASE_URL` below.
- `USERS_DATABASE_URL` (optional): when set, `dashboard/backend/users.py` stores accounts/sessions in this Postgres database instead of the local SQLite `DB_PATH`. See the Gotchas entry below for why this exists.
- `CONTENT_DATABASE_URL` (optional): when set, agents (`external_agents`), agent versions, strategies, and user portfolios (`user_portfolios`) are stored in this Postgres database instead of `DATABASE_PATH` SQLite (factories in each store module, cloned from `_build_user_store()`; Postgres twins in `*_postgres.py` siblings). It covers *user-created content* only; accounts have their own `USERS_DATABASE_URL` and the two never fall back to each other — a fully durable deployment sets both, pointed at the same Neon DB, pooled (`-pooler`) URL. **Not** named `DATABASE_URL` on purpose: that is the Heroku-convention name managed-Postgres add-ons inject and unrelated projects export, and an ambient value would silently bind the app to the wrong database (nothing can protect a local `uvicorn` run from it — reading the env var *is* the feature). Set it in the **Render dashboard before merging** anything that depends on it — unset silently selects ephemeral SQLite, which is why each factory logs its choice at startup: `<store> backend: postgres (<host>/<db>)` or `<store> backend: sqlite (ephemeral on Render)`. The line names the host/db rather than a bare "postgres" so a typo'd or staging URL is visible too (`db_url.py::describe_database_url`, which never emits credentials). Leave unset for local dev/tests — `tests/conftest.py` strips it so the suite always runs on SQLite.
- `AGENT_RUNS_DATABASE_URL` (optional): when set, `database.py`'s `_build_backtest_db()` stores backtest run history — `agent_runs`, `equity_timeseries`, `trades`, `backtest_decisions`, `run_manifest` — in this Postgres database instead of `DATABASE_PATH` SQLite (`PostgresBacktestDatabase` in `database_postgres.py`, the twin of `BacktestDatabase`). The hot per-step table, `idempotency_keys`, deliberately stays on local SQLite regardless — the twin delegates `get_idempotency`/`put_idempotency` to an embedded plain `BacktestDatabase` so an agent's decision submission never gains a network round-trip. It covers run history only; it does not fall back to or from `CONTENT_DATABASE_URL` (agents/versions/strategies) or `USERS_DATABASE_URL` (accounts), nor either of those to it — a fully durable deployment sets all three, but **not** at the same database as `CONTENT_DATABASE_URL`/`USERS_DATABASE_URL`: run history lives in its own dedicated Neon project (`ATL-runs-main`), whose separate free-tier storage/compute allotment isolates the largest, hottest tables' growth from the auth-critical users/content database — no code path joins run tables with content/users tables, so co-locating would buy nothing. Still a pooled (`-pooler`) URL. **Not** named `DATABASE_URL`, for the same reason as `CONTENT_DATABASE_URL` above. Set it in the **Render dashboard before merging** anything that depends on it — unset silently selects ephemeral SQLite, which is why `_build_backtest_db()` logs its choice at startup: `run history backend: postgres (<host>/<db>)` or `run history backend: sqlite (ephemeral on Render)`. Leave unset for local dev/tests — `tests/conftest.py` strips it so the suite always runs on SQLite.
- `BREVO_API_KEY` / `ACCOUNT_EMAIL_FROM` (both required together) and optional `ACCOUNT_EMAIL_FROM_NAME` (defaults to `Agentic Trading Lab`): transactional email credentials for the two-code account email-change flow (`infrastructure/email/sender.py`). Leave either required key unset and the two steps that actually send mail return **503**: `POST /api/auth/email-change` (code to the current address) and the `verify` call that advances stage `old`→`new` (code to the new address). The final `verify` — the one that commits the change — sends nothing and succeeds regardless, so rotating credentials mid-flight does not strand a user already holding the second code. Display-name editing and logout are unaffected. Set both in the **Render dashboard** for prod — `render.yaml` is documentation, not the deploy mechanism (see the Prod deploy reality gotcha below).
- `LEADERBOARD_DAILY_REFRESH_SECRET` (optional): shared secret for `POST /api/v1/leaderboard/daily/refresh`, the cron hook that refreshes the rolling **Daily Leaderboard** (`?period=daily`). The caller sends it as `X-Leaderboard-Refresh-Secret`; the endpoint returns **202 Accepted** and runs the work on a background thread, so a Render/Actions HTTP timeout can never abort a multi-hour model deploy. Unset, the route answers **401 to everyone** — deliberately *not* 503, so an anonymous caller cannot learn whether it is armed (the operator signal goes to the server log instead). It is also rate-limited per client (`FixedWindowRateLimiter`, 20/hour) because one shared secret with unlimited attempts is an open guessing budget. Driven by `.github/workflows/daily-leaderboard.yml` (22:30 UTC Mon–Fri, after the 16:00 ET cash close) and by `dashboard/scripts/refresh_daily_leaderboard.py --remote`. Set it in the **Render dashboard** *and* as a repository secret before enabling the workflow.
- `ADMIN_BOOTSTRAP_SECRET` (optional): shared secret for `POST /api/admin/bootstrap`, which promotes the **signed-in** caller to `role=admin`. One-shot: it refuses once any admin account exists (break-glass after that is SQL). Unset, the route answers **503**. Wrong guesses are rate-limited three ways — per user, per client key (5 / 15 min each) and a server-wide 20 / 15 min ceiling. The per-user counter alone was **not** a bound: signup is open, so an attacker exhausting one account's budget just registers another; the per-client and global counters are what cap guessing. The secret is compared via SHA-256 + `secrets.compare_digest` — the hash is load-bearing because `compare_digest` raises `TypeError` on a non-ASCII `str` (a JSON body can send one) and runs in time proportional to the shorter operand, leaking the expected length. It does *not* raise on a length mismatch; unequal buffers just compare false. The JSON body requires 8–256 characters, so a shorter env value can never match. Set it in the **Render dashboard** for a fresh deploy, then leave it (it is inert after the first admin) or unset it. `tests/conftest.py` strips it so the suite sees a known-unset baseline.
- `LEADERBOARD_DAILY_AUTO_DEPLOY` (optional, **strict opt-in — this one spends money**): when truthy, serving the *public, unauthenticated* `GET /api/v1/leaderboard?period=daily` may start a background thread that runs `deploy_model_run` for every competition LLM entry — real billable API calls initiated by an anonymous request. It is therefore off unless explicitly set (`1`/`true`/`yes`/`on`); do **not** restore the old "on whenever `RENDER` is unset" default, which armed it on the Docker image, every self-host and fork, and the test suite (`tests/conftest.py` deliberately strips `RENDER`, and strips this var too). Prod uses the nightly refresh job instead. Note the in-progress guard and the per-window state file (`storage/data/leaderboard_daily_refresh.json`) are **per-process**: fine for the current single-instance Render deploy, but a multi-replica deploy would duplicate model deploys.
- Alpaca paper-trading credentials also live in `credentials/alpaca.json` (gitignored; see `credentials/alpaca.json.example`).

## Architecture

Pipeline is **backtest → SQLite → API → dashboard**. The backend is layered (see `docs/architecture/dashboard-target-structure.md`):

- **`api/`** — FastAPI surface. Business routers live in `api/routers/*` and are mounted by `api/router.py` under `/api`; the canonical agent contract is `api/v2/*` (see "Agent API v2" below). **Paper-trading routes stay outside `/api`** (registered directly on the app), so `/paper/*` is the external contract. `app.py` is the composition root (creates the app, middleware, startup hooks, serves both frontends).
- **`domain/`** — business logic by area: `runs/` (Agent-Environment Protocol: Run/Step/Decision), `agents/`, `leaderboard/` (contest + baseline strategies registry + the H6 integrity guard), `backtesting/` (engine, `external_run_service`, portfolio manager, `baselines/` subpackage), `strategies/` (free-form strategy store), `portfolios/` (the account-bound $10k cash ledger behind `GET /api/v1/portfolio` — distinct from `trading/portfolio` and `backtesting`'s portfolio manager, which track *positions* rather than a per-account balance), `chat/`, `trading/` (live paper trading: `paper_session`, `execution`, `portfolio`). Domain must not import `api/`/`app.py`.
- **`execution/`** — v2 execution backends binding domain engines to the `/api/v2` contract: `base.py` (interface), `backtest_backend.py` (implemented), `paper_backend.py` (**stub** — raises `NotImplementedError`; Phase B not built). Deliberately at the backend root (not `domain/`) so it can bridge domain→API without tripping the `domain/`→`api/` import ban.
- **`infrastructure/`** — `llm/` (the `validator` security boundary, `token_cost`, `backtest_harness`/gateway client), `market_data/` (Alpaca bars), and `brokers/` (`alpaca_paper.py`, the isolated Alpaca paper-trading HTTP adapter).
- **Backend-root modules** — `middleware.py` (session enforcement + CSP), `users.py` (auth/bcrypt/session store), `cache.py` (TTL cache for paper-trading responses), `baseline_generator.py`/`baseline_resolver.py`/`baselines_endpoint.py` (shared baseline equity-curve generation + DJIA/buy-hold baselines for backtests and paper trading), `llm_integration_example.py` (reference safe-LLM pattern). (`engines/` and `services/` are **not** packages — they were pre-refactor compatibility shims, deleted once their code moved under `domain/`; `test_architecture_boundaries.py`'s `_DELETED_SHIMS` list asserts they stay non-importable.)
- **Persistence** (`database.py` + per-store repositories like `domain/runs/repository.py`, `domain/strategies/repository.py`): thin SQLite wrappers over `DATABASE_PATH` in **WAL journal mode** (readers aren't blocked by finalize's heavy writes); schema is created lazily and self-migrates. `agent_runs` carries a JSON `metadata` column recording the effective `LLM_MAX_OUTPUT_TOKENS` per run.
- **Frontend** — `dashboard/frontend/` is the served static root and holds **both** UIs: the **landing page** (`index.html` + `assets/`) served at **`/`**, and the vanilla-JS + Chart.js **dashboard** (`app.html`, `app.js`, `styles.css`, no build step) served at **`/app`**. The landing page is a Vite/React marketing site whose **source** lives in `dashboard/landing/` (Replit-exported, de-monorepo'd; `npm run build`); its build output ships as `frontend/index.html` + `frontend/assets/`. `app.py` adds a `/app/`→`/app` 308 redirect so the dashboard's relative asset paths resolve. Vercel deploys the static `dashboard/frontend`.
- **Paths** (`dashboard/backend/paths.py`): single source of truth for on-disk locations.

### Baseline strategies (registry pattern)

`dashboard/backend/domain/leaderboard/strategies/` holds benchmark strategies (`buy_hold`, `equal_weight_index`, `market_index`, `mean_variance`, `llm_agent`, …). To add one: subclass `BaselineStrategy` (`base.py`), give it a `key`, add the class to `_STRATEGY_CLASSES` in `registry.py`. `get_strategy(config)` resolves by `strategy`/`type` key.

**H6 leaderboard integrity guard.** An LLM-backed entry can only publish if the model actually drove ≥95% of its steps (`MIN_LLM_DECISION_COVERAGE = 0.95`). The guard (`domain/leaderboard/service.py`) keys on `PortfolioManager.llm_decisions` — steps the model genuinely drove, incremented only at the *success exit* of the decision path — **not** `llm_calls` (a pure billing counter that also ticks on truncated/unparseable responses that then silently fall back to rule-based). This stops a rule-based fallback curve from being published under an LLM's name. See the memory note `leaderboard-h6-integrity-model` for the full rationale. All 7 LLM entries currently on the board (Claude Haiku 4.5, Sonnet 4.6, GPT-5.5, Gemini 3.1 Pro, Qwen3.7 Plus, DeepSeek V4 Pro, Nemotron 3 Nano 30B) cleared it; only DeepSeek beat the passive baselines.

### LLM safety boundary

`infrastructure/llm/validator.py` is a hard security boundary: LLM trading responses must be JSON-only matching the trading schema — `tool_calls`/`function_calls` are rejected, portfolio constraints enforced, decisions logged. Do not loosen this to allow tool/web access from agent responses.

### External agents & Agent-Environment Protocol (`/api/v1`)

- **Protocol Run API** (`api/routers/runs.py` → `domain/runs/*`): an external agent authenticates with its Agent API key (`X-API-Key`) and drives a backtest step-by-step (`POST /api/v1/runs`, poll steps, submit decisions). Each step has a decision deadline (default 60s); a late decision auto-holds that step rather than failing the run. A server-wide active-run backstop (`MAX_ACTIVE_RUNS_GLOBAL`, default 100; 0 disables) rejects creates on both surfaces with 429 + `Retry-After` once at capacity.
- **External backtest engine** (`domain/backtesting/external_run_service.py`): the hour-by-hour session behind both the protocol and the legacy `/api/v1/backtest/*` routes.
- **PyPI client** (`packaging/agentictrading/`): stdlib-only Python SDK + `AgentRunner`. Published via `.github/workflows/publish-pypi.yml`.

### Agent API v2 (`/api/v2`) — the canonical agent-facing contract

Two step-driven agent surfaces coexist; they are **not peers**:

- **`/api/v2` is canonical** (`api/v2/*` routers + `execution/` backends over the same domain engines): typed Pydantic contract, per-agent scopes + token-bucket rate limits, canonical `run_id`, DB-backed idempotency (`(run_id, idem_key)`), `context_ref` provenance, self-describing `GET /api/v2/schema`. Spec/plan: `docs/superpowers/{specs,plans}/2026-06-23-agent-api-foundation-*`. New agent-facing features land here. **Phase B** (paper/live via `ExecutionBackend`) and **Phase C** (MCP façade) are **not built yet** — `execution/paper_backend.py` is a stub.
- **`/api/v1` is the compatibility surface** for the shipping SDK (`packaging/agentictrading`), Discord bot, and built-in agents. Keep it working; do not grow it. Migrating the SDK to v2 is the gate for publishing `agentictrading` 0.2.0.
- **Unified run lifecycle (v1 + v2).** The two surfaces share one active-run cap ledger (under a single lock), one reaper sweep (`register_reaper_sweep()` reaches v2 runs), and multi-worker heartbeat recovery (`owner_instance`/`heartbeat_at` columns, `RUN_HEARTBEAT_STALE_SECONDS`). Terminal v2 runs are swapped for a DB-backed `ArchivedBacktestBackend` tombstone; step/idempotency state persists across process restarts; v2 `cancel`/`status` report the true terminal status (not always "closed").
- `execution/` sits at the backend root (not `domain/`) deliberately: the backends bind domain engines to the v2 API contract, and `test_architecture_boundaries` forbids `domain/` → `api/` imports.

## Deployment

- **Backend** → Render (`render.yaml`): `uvicorn dashboard.backend.app:app`, persistent disk at `/data`, health check `/health`.
- **Frontend** → Vercel (`dashboard/frontend/vercel.json`): static `dashboard/frontend`. The Vercel project's Root Directory is `dashboard/frontend`, so the config **must** live there — a repo-root `vercel.json` is silently ignored (issue #301).
- **Container** (`Dockerfile`): `WORKDIR /app`; `uvicorn dashboard.backend.app:app`.

## Merge & branch discipline

**`main` has no branch protection, no required checks, and no CODEOWNERS.** Nothing gates a merge. Any collaborator can merge any open PR at any moment, and the observed norm is that they do — unreviewed, and over red CI. Merging to `main` also auto-deploys prod (see the Deployment gotcha). Treat every open PR as merge-able *right now* by someone who has not read your plan.

- **Never push follow-up work to a branch whose PR is already merged.** Cut a new branch. GitHub gives **no notification, no reopening, and no warning** when commits land behind a merged PR — they orphan silently, and the only signal is a human noticing the branch is ahead of the PR that consumed it. (This is exactly how PR #107 shipped without the fix that was meant to be part of it; the follow-ups had to be re-landed as #110 off the same ref.) Check before pushing: `gh pr list --head <branch> --state all`.
- **If a PR must not merge yet, publish that where GitHub shows or enforces it** — **open it as a draft**, or add a `blocked` label, and put the gate as an imperative in the *first line of the body* ("DO NOT MERGE until X ships"). A comment is not a gate, and a body that explains why the change is *safe* to land early ("depends on X, but falls back transparently until then") reads as *please merge me*. A gating instruction posted after the merge is worthless — intent that only exists in a local worktree or an agent session's memory does not exist.
- **Never record in notes/memory that a merge was sequenced deliberately unless a session actually verified and pressed the button.** Check `gh api repos/Open-Finance-Lab/AgenticTrading/pulls/N --jq '.merged_by.login'`. Writing down a gate that nobody applied teaches every later reader that the gate works.

### Fail-closed is not fail-visible

The FinSearch news adapter (`dashboard/backend/integrations/news_sentiment.py`) is the cautionary case. `get_latest_panel_payload`'s `if not feed:` fallback to the Phase-A representative feed makes **"the upstream endpoint isn't deployed"** and **"the endpoint is live and every story is being silently rejected"** produce a byte-identical `status: ok` HTTP 200. The 404 path logs *nothing* (a bare `pass`). A field rename upstream therefore degraded prod for hours with no error, no metric, and a green test suite.

- When adding a fallback, ask **what distinguishes *absent* from *broken***. If nothing does, log ERROR at the wholesale-drift boundary (a per-item warning cannot report a total contract break).
- **Never build an upstream's fixture from your own adapter's field names.** Fixtures written that way test the mapper against itself and drift with the code, so a producer rename stays green forever. Pin the shape from a real recorded response (`dashboard/backend/tests/fixtures/items-wire-fixture.json`).
- **Mocked coverage cannot detect a cross-repo producer rename** — the producer is mocked. Only a canary against the live endpoint can. Don't mistake more mock tests for coverage of this seam.

## Gotchas

- Root `pyproject.toml` (`finagent-orchestration`) is for the **orchestration** subsystem, not the dashboard — edit `requirements.txt` for dashboard deps.
- `README.md`'s "File Structure" diagram is idealized; the real layout nests everything under `dashboard/`.
- The committed `dashboard/storage/data/backtest.db` holds seed runs referenced by `dashboard/config/defaults.json`. Importing a store module runs `CREATE TABLE IF NOT EXISTS` against `DATABASE_PATH`, so running the app locally can add empty tables to that file — don't commit those mutations. If you regenerate the DB, update `defaults.json`.
- Pytest is not in `requirements.txt`; install it separately.
- `discord.py` (for `integrations/discord_bot.py`) is an **optional** dep declared in `requirements-discord.txt` (like `requirements-sphinx.txt` for the docs build), not core `requirements.txt` — run `pip install -r requirements-discord.txt` to run the bot. It's kept out of core so web/API/backtest installs stay lean; its tests `importorskip('discord')`.
- `vnpy` (for the `vnpy_simulation` market-data source) is the third **optional** dep, in `requirements-vnpy.txt`. CI installs core `requirements.txt` only, so **any test that imports `vnpy` at module scope must `importorskip('vnpy')`** — an unguarded import raises during *collection*, and a collection error doesn't fail one module, it aborts the whole pytest session (0 tests run, and the deploy hook that gates on backend tests never fires). Gate individual cases with `skipif(importlib.util.find_spec("vnpy") is None)` when the module itself is import-safe. Same rule for any future optional dep.
- **Prod deploy reality vs `render.yaml`.** The live Render service runs on the **free tier** with **no persistent `/data` disk**, so the local SQLite file at `DATABASE_PATH` resets to the committed seed `backtest.db` on every redeploy, and the disk/plan in `render.yaml` is aspirational. That reset genuinely destroys `protocol_runs`/`protocol_steps` and `idempotency_keys` — no Postgres option for either (see the `DATABASE_PATH` bullet above). It no longer touches accounts (`USERS_DATABASE_URL`), agents/versions/strategies/portfolios (`CONTENT_DATABASE_URL`), or backtest run history (`AGENT_RUNS_DATABASE_URL`) — each is durable once its var is set in the Render dashboard, independently of the others. **Merging to Open-Finance-Lab `main` auto-deploys prod**: a CI job hits the Render Deploy Hook once backend tests pass on `main` (PR #95, live since 2026-07-11) — no manual trigger or fork-sync needed. (Render's own branch-tracking/autoDeploy is inert but irrelevant; the CI hook drives every deploy.)
- **Phantom `test_deleted_shim_is_not_importable` failures = stale bytecode, not a regression.** If those cases fail locally with `DID NOT RAISE ModuleNotFoundError`, it's leftover `dashboard/backend/{engines,services}/__pycache__/*.pyc` from the pre-refactor layout, which Python resolves as a PEP-420 namespace package. The dirs are untracked so CI is green; `rm -rf dashboard/backend/engines dashboard/backend/services` clears it.
- **User accounts were silently lost on every prod redeploy until 2026-07 (see `docs/superpowers/plans/2026-07-08-user-account-persistence-fix.md`).** `users.py` originally shared `DB_PATH` with backtest data; on the live Render service (free tier, `disk: null`, `DATABASE_PATH` unset) that file resets to the git-committed seed DB on every deploy, deleting the `users`/`auth_sessions` tables with no error surfaced anywhere. The fix is an optional Postgres backend selected via `USERS_DATABASE_URL` — set it in prod (see the **Prod deploy reality** bullet above); leave it unset for local dev/tests, which keep using SQLite exactly as before. The same fix was extended to agents, agent versions, and strategies in 2026-07 via `CONTENT_DATABASE_URL` (see `docs/superpowers/specs/2026-07-15-agent-strategy-persistence-design.md`) — before that, every registered agent and issued API key died on each deploy, breaking all SDK/Discord integrations, with `resolve_api_key()` as the sole auth path for `/api/v1` and `/api/v2`.
