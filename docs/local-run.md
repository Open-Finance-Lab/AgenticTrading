# Running Agentic Trading Lab locally

This guide gets the dashboard, the API and a working backtest running on your
own machine **with no API keys and no money at risk**, then shows how to run the
test suites. Every command below was run end to end on Linux (Python 3.13,
Node 22).

All commands run from the **repository root**.

## Contents

1. [Requirements](#1-requirements)
2. [Install](#2-install)
3. [Start the app safely](#3-start-the-app-safely)
4. [Run a backtest](#4-run-a-backtest)
5. [Run the tests](#5-run-the-tests)
6. [Adding credentials later](#6-adding-credentials-later)
7. [Troubleshooting](#7-troubleshooting)
8. [What "safe" does and does not cover](#8-what-safe-does-and-does-not-cover)

---

## 1. Requirements

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.13 (see `.python-version`) | backend, backtests, tests |
| [uv](https://docs.astral.sh/uv/) | any recent | virtualenv + fast installs (plain `pip` also works) |
| Node.js | any current LTS | optional: ~two dozen frontend-behaviour tests run JS under `node`; without it they **skip**, they do not fail |
| SQLite | bundled with Python | nothing to install |

No Docker, Postgres or Redis is needed for local work. The `Dockerfile` mirrors
production (Render) and adds an isolated AI Hedge Fund venv, which is not
needed for local runs.

## 2. Install

```bash
uv venv --python 3.13 .venv
UV_HTTP_TIMEOUT=180 uv pip install --python .venv/bin/python \
    -r requirements.txt -r requirements-vnpy.txt pytest pytest-timeout
```

- `requirements.txt` is the real dashboard dependency list. The root
  `pyproject.toml` / `uv.lock` belong to the separate `orchestration/` project.
- `requirements-vnpy.txt` adds **vn.py**, which provides simulated market data,
  so backtests work without an Alpaca key.
- `pytest` is not in `requirements.txt`, so install it separately.
- `UV_HTTP_TIMEOUT=180` matters on slow connections. uv's default 30 s timeout
  fails with `Failed to fetch … operation timed out`.

If `uv venv` asks *"A virtual environment already exists. Replace it?"*,
answer **no** unless you mean to reinstall everything. Replacing it deletes
every installed package.

## 3. Start the app safely

```bash
dashboard/scripts/run_local_safe.sh          # default port 8765
dashboard/scripts/run_local_safe.sh 9000     # or pick a port
```

Then open **http://127.0.0.1:8765/app**. The landing page is at `/`, and
`/health` returns `{"status":"ok"}`.

The script:

- copies the seed database `dashboard/storage/data/backtest.db` to
  `~/.cache/atl-local/backtest.db` on first run, and points `DATABASE_PATH`
  at that copy. The committed seed DB is never modified. Delete the copy to
  reset to seed data. Set `ATL_LOCAL_DIR` to use a different folder.
- blanks every broker and LLM key (Alpaca, Anthropic, OpenAI, DeepSeek,
  CommonStack, OpenRouter) for the process, even if they are in your shell.
- forces `ROBINHOOD_EXECUTE=false`, which turns off the only code path that
  can place a real-money order.
- turns off the two background jobs that can spend money on their own:
  `ATL_BAR_CACHE_WARM=0` and `LEADERBOARD_DAILY_AUTO_DEPLOY=0`.
- enables `ENABLE_VNPY_SIMULATION=true`, which adds the simulated data source.

On startup the log prints one `… backend: sqlite (…)` line per store. They
should all say `sqlite`. A `postgres` line means a `*_DATABASE_URL` variable
leaked in from your environment.

To run the server by hand instead, the canonical command is
`uvicorn dashboard.backend.app:app` from the repo root. Never run
`python dashboard/backend/app.py`: the package imports fail that way. Without
the script's variables you are running against the committed DB and whatever
keys are in `dashboard/.env`.

## 4. Run a backtest

**From the dashboard:** use the Backtest panel and set *Market Data* to
**vn.py simulated data**, with a rule-based decision source. LLM decisions need
sign-in and a model key, and vn.py data forces the LLM off anyway.

**From the API:**

```bash
SID=$(python3 -c "import uuid; print(uuid.uuid4())")
curl -s -X POST http://127.0.0.1:8765/backtest/run \
  -H "X-Session-Id: $SID" -H "Content-Type: application/json" \
  -d '{"start_date":"2025-03-03","end_date":"2025-03-07","data_source":"vnpy_simulation","decision_source":"rule_based"}'
# poll until "running": false
curl -s http://127.0.0.1:8765/backtest/status -H "X-Session-Id: $SID"
# list this session's runs (agent + buy-and-hold + DJIA baselines)
curl -s http://127.0.0.1:8765/api/backtest/runs -H "X-Session-Id: $SID"
```

`X-Session-Id` must be a UUID. Windows are capped at 14 days. For reference,
5-, 10- and 14-day simulated runs took about 8–14 s, and the server plus one
backtest worker peaked at about 480 MB RSS (about 275 MB idle).

**Simulated data is synthetic.** It is useful for checking that the pipeline
works, not for judging a strategy. Results on real bars need an Alpaca key
(section 6).

## 5. Run the tests

```bash
# Backend: ~5,600 tests, ~15-17 min, ~580 MB peak RSS
.venv/bin/python -m pytest dashboard/backend/tests -q

# One file
.venv/bin/python -m pytest dashboard/backend/tests/test_protocol_api.py -q

# PyPI SDK: its own suite. Run it separately; mixing it with the backend
# run fails at collection because the SDK package is not installed.
(cd packaging/agentictrading && PYTHONPATH=. ../../.venv/bin/python -m pytest tests -q)
```

The backend suite is hermetic. `dashboard/backend/tests/conftest.py` points
`DATABASE_PATH` at a temp file and strips credential and cost-related variables,
so it never touches the seed DB or spends money. A red test on a fresh run is a
real regression.

Expected result: all passed, roughly 170 skipped. Skips are optional-dependency
guards (vn.py, `node`, Discord) and are not failures.

## 6. Adding credentials later

Only do this once the no-key setup works.

- The app loads **`dashboard/.env`**, not the repo root. Copy the template
  `.env.example` from the repo root to `dashboard/.env` and fill in only what
  you need. `.env` and `credentials/*` are gitignored. Never commit keys.
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`: real historical bars and the
  `/paper/*` account views. The Alpaca code is **paper-only and read-only**:
  the trading host is hardcoded to `paper-api.alpaca.markets` and nothing places
  Alpaca orders. `ALPACA_BASE_URL` in the template is not read by the backend.
- LLM keys (`ANTHROPIC_API_KEY`, …) are **billable**. LLM backtests on the
  dashboard require sign-in and a billing mode (BYOK or credits).
- **Do not set `ROBINHOOD_EXECUTE`.** It is a single server-wide switch that
  allows real-money orders for every linked account.
- `run_local_safe.sh` blanks all of these keys on purpose. To use real keys,
  start uvicorn yourself with the same `DATABASE_PATH` / `ATL_BAR_CACHE_DIR`
  exports and leave `ROBINHOOD_EXECUTE=false`.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to fetch https://pypi.org/... operation timed out` | slow network, uv's 30 s default timeout | prefix the install with `UV_HTTP_TIMEOUT=180` and retry |
| `No module named uvicorn` | venv empty or replaced | rerun the install from section 2 |
| `[Errno 98] address already in use` | another process has the port | `ss -ltnp \| grep :8765`, or pass a different port to the script |
| `curl localhost:8000` answers with a different API's JSON (e.g. a different `openapi.json` title) | another local app owns port 8000 and uvicorn silently failed to bind | use the script's default 8765, or check the `/openapi.json` title |
| `env: ' ': No such file or directory` / `event not found` after pasting | a multi-line command lost its `\` line endings, or picked up stray text | use the script, or paste the command as one line |
| `/paper/account` → `success: false` | no Alpaca keys | expected without keys |
| `401` on `/api/v1/research/agents` in the browser console | signed-out visitor | expected; that endpoint is sign-in-only |
| `test_deleted_shim_is_not_importable` fails with `DID NOT RAISE` | stale `__pycache__` from the old layout | `rm -rf dashboard/backend/engines dashboard/backend/services` |

## 8. What "safe" does and does not cover

**Covered by this setup:** no broker order can be placed, no LLM call can be
billed, and no persistent data outside `~/.cache/atl-local` is written.

**Not covered.** Keep these in mind before trusting results or enabling keys:

- **Backtest realism.** US runs model no commissions or slippage, fill hourly
  decisions at the same bar's close, and use unadjusted prices, so a split
  inside the window looks like a crash. The "DJIA" baseline is an equal-weight
  average of today's Dow members. A positive backtest is not evidence of an
  edge.
- **Outbound calls without keys.** The home-page ticker (`/ticker`) still
  fetches public quotes (Yahoo Finance), and the news panel may call the
  FinSearch endpoint.
- **Live trading.** The Robinhood path has per-order caps but no per-run or
  per-day limit. Leave it disabled.
