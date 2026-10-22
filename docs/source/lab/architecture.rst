Architecture
============

The lab is a static web frontend in front of one API process. A dashboard
backtest runs in its own worker process, and its results are stored and then
served back through the API. Market data comes from Alpaca for US stocks and
iFinD for A-shares.

System diagram
--------------

.. code-block:: text

   Browser
     │
   ┌─────────────────────────────────────────────────────────────┐
   │ Vercel: static frontend (dashboard/frontend/)               │
   │ ├─ /     landing page (index.html, built from landing/)     │
   │ ├─ /app  dashboard (app.html, app.js, js/, styles.css)      │
   │ └─ forwards /api, /backtest, /runs, /paper, ... to the API  │
   └─┬───────────────────────────────────────────────────────────┘
     │ same-origin requests
   ┌─▼───────────────────────────────────────────────────────────┐
   │ API: one FastAPI process on Render (dashboard/backend/)     │
   │ ├─ /api/*: accounts, Credits, agents, leaderboards, v1, v2  │
   │ ├─ /backtest/*, /runs, /compare, /ticker, /config/defaults  │
   │ ├─ market data: Alpaca (US stocks), iFinD (A-shares)        │
   │ └─ My Agents LLM runs: your own key (BYOK) or ATL Credits   │
   └─┬───────────────────────────────────────────────────────────┘
     │ one subprocess per dashboard backtest (5 at once by default)
   ┌─▼───────────────────────────────────────────────────────────┐
   │ Backtest worker: scripts/backtest_hourly_agent.py           │
   │ ├─ fetch bars, warm up indicators, run the agent per bar    │
   │ ├─ price the baselines: buy-and-hold, and the Dow for US    │
   │ └─ write the run, its equity curve and its trades           │
   └─┬───────────────────────────────────────────────────────────┘
     │
   ┌─▼───────────────────────────────────────────────────────────┐
   │ Storage: SQLite by default, Postgres per data group         │
   └─────────────────────────────────────────────────────────────┘

Separately from user backtests, a scheduled job advances the live leaderboard
once each weekday after the US close: the models and baselines on its roster
trade the session that just closed, on simulated fills with no real money.

Storage
-------

A fresh install keeps its application tables in one SQLite file,
``dashboard/storage/data/backtest.db``. Three settings move groups of data to
Postgres:

* ``USERS_DATABASE_URL``: accounts, the ATL Credits ledger, stored provider keys
  and product analytics.
* ``CONTENT_DATABASE_URL``: agents and their versions, strategies, portfolios
  and research-agent runs.
* ``AGENT_RUNS_DATABASE_URL``: backtest run history and agent traces.

Step-by-step runs from the :doc:`agent_api` always stay in the SQLite file. On a
deployment without a persistent disk that file resets on every redeploy, so an
external-agent run that is in progress when a new version ships is interrupted.

API surface (summary)
---------------------

+---------------------------+----------------------------------------------+
| Endpoint                  | Purpose                                      |
+===========================+==============================================+
| ``GET /health``           | Health check                                 |
+---------------------------+----------------------------------------------+
| ``GET /runs``             | List backtest runs                           |
+---------------------------+----------------------------------------------+
| ``GET /runs/{id}/equity`` | Equity curve for a run                       |
+---------------------------+----------------------------------------------+
| ``GET /compare``          | Compare multiple runs                        |
+---------------------------+----------------------------------------------+
| ``POST /backtest/run``    | Start a backtest                             |
+---------------------------+----------------------------------------------+
| ``GET /backtest/status``  | Poll backtest job status                     |
+---------------------------+----------------------------------------------+
| ``POST /backtest/cancel`` | Cancel a running backtest                    |
+---------------------------+----------------------------------------------+
| ``GET /ticker``           | Market quote data                            |
+---------------------------+----------------------------------------------+
| ``/paper/*``              | Alpaca paper-account reads and session       |
|                           | tracking (the dashboard's tab is off)        |
+---------------------------+----------------------------------------------+
| ``GET /config/defaults``  | Default UI / run configuration               |
+---------------------------+----------------------------------------------+
| ``/api/auth/*``           | Accounts and sessions (:doc:`accounts`)      |
+---------------------------+----------------------------------------------+
| ``/api/credits/*``        | ATL Credits (:doc:`credits_billing`)         |
+---------------------------+----------------------------------------------+
| ``/api/v1/*``             | Agents, runs, leaderboards, portfolio        |
+---------------------------+----------------------------------------------+
| ``/api/v1/research/*``    | Research agents (:doc:`research_agents`)     |
+---------------------------+----------------------------------------------+
| ``/api/v2/*``             | Agent API (:doc:`agent_api`)                 |
+---------------------------+----------------------------------------------+

LLM integration example code lives in ``dashboard/backend/llm_integration_example.py`` (reference only, not wired into the main app path).

Related documentation
---------------------

* Discord → backtest → dashboard workflow:
  ``docs/architecture/discord-to-backtest.md``
* Multi-agent orchestration, agent pools, and DAG workflows:
  :doc:`../orchestration/index`.
