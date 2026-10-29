Architecture
============

The lab is a static web frontend in front of one API process. A dashboard
backtest runs in its own worker process, and its results are stored and then
served back through the API, which also reads and writes most of the stored
data itself. Market data comes from Alpaca for US stocks and iFinD for
A-shares. Yahoo Finance supplies index levels such as ``^DJI`` for the
leaderboards' market-index baselines and the backtest chart's index lines, and
the ticker's quotes (Alpaca fills in any it misses). A Yahoo outage or format
change breaks those index baselines and lines; it does not stop a backtest,
whose own Dow baseline is priced from Alpaca bars.

System diagram
--------------

.. code-block:: text

   Browser
     │
   ┌─▼───────────────────────────────────────────────────────────┐
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
   │ ├─ market data: Alpaca (US stocks), iFinD (A-shares),       │
   │ │  Yahoo Finance (index levels, ticker quotes)              │
   │ └─ My Agents LLM runs: your own key (BYOK) or ATL Credits   │
   └─┬────┬──────────────────────────────────────────────────────┘
     │    │ one subprocess per dashboard backtest (5 at once by default)
     │  ┌─▼──────────────────────────────────────────────────────┐
     │  │ Backtest worker: scripts/backtest_hourly_agent.py      │
     │  │ ├─ fetch bars, warm indicators, run the agent per bar  │
     │  │ ├─ price baselines: buy-and-hold, and the Dow for US   │
     │  │ └─ write the run, its equity curve and its trades      │
     │  └─┬──────────────────────────────────────────────────────┘
     │    │
   ┌─▼────▼──────────────────────────────────────────────────────┐
   │ Storage: SQLite by default, Postgres per data group         │
   │ ├─ API: accounts, Credits, agents, protocol and v2 runs,    │
   │ │  analytics; reads run history back for the dashboard      │
   │ └─ worker: a backtest's run history, trace, Credits usage   │
   └─────────────────────────────────────────────────────────────┘

Separately from user backtests, a scheduled job advances the live leaderboard
once each weekday after the US close: the models and baselines on its roster
trade the session that just closed, on simulated fills with no real money.

Storage
-------

A fresh install keeps its application tables in one SQLite file,
``dashboard/storage/data/backtest.db``. Three settings move groups of data to
Postgres:

* ``USERS_DATABASE_URL``: accounts, the ATL Credits ledger (purchase records
  included), stored provider keys, product analytics and Robinhood broker
  connections.
* ``CONTENT_DATABASE_URL``: agents and their versions, strategies, portfolios,
  hosted-runtime agent credentials and research-agent runs. Robinhood
  connections fall back here when ``USERS_DATABASE_URL`` is unset.
* ``AGENT_RUNS_DATABASE_URL``: backtest run history and agent traces.

Stored secrets therefore sit in two groups: provider keys and broker
connections follow ``USERS_DATABASE_URL``, and agent runtime credentials follow
``CONTENT_DATABASE_URL``. Set both: with only ``USERS_DATABASE_URL``, agent
runtime credentials stay in the SQLite file, and with only
``CONTENT_DATABASE_URL``, stored provider keys do.

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
| ``/api/v1/robinhood/*``   | Robinhood live trading (:doc:`live_trading`) |
|                           | Switched off: status and disconnect only.    |
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
