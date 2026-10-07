Architecture
============

The lab stacks a backtest engine, REST API, and web dashboard. Market data (Alpaca for US stocks, iFinD for A-shares) flows through backtests into a database (SQLite by default), then through the API to the frontend.

System diagram
--------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │ Backtest Engine (dashboard/scripts/backtest_hourly_agent.py)│
   │ ├─ Fetch bars: Alpaca 5m -> 60m (US), iFinD (A-share)       │
   │ ├─ Run agent + baseline logic (bar cache, optional vnpy)    │
   │ ├─ Write runs: agent, buy-and-hold, DJIA (US profiles only) │
   │ └─ Store in dashboard/storage/data/backtest.db (SQLite)     │
   └────────────────┬────────────────────────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────────────────────┐
   │ REST API (dashboard/backend/app.py composes the routers)    │
   │ ├─ GET  /health                                             │
   │ ├─ GET  /runs, /runs/{id}/equity, /compare                  │
   │ ├─ POST /backtest/run, /backtest/cancel                     │
   │ ├─ GET  /backtest/status, /ticker, /config/defaults         │
   │ ├─ /api/* (auth, credits, leaderboard, /api/v1, /api/v2)    │
   │ └─ /paper/* (Alpaca paper account data; no UI yet)          │
   └────────────────┬────────────────────────────────────────────┘
                    │
   ┌────────────────▼────────────────────────────────────────────┐
   │ Web (dashboard/frontend/)                                   │
   │ ├─ index.html: landing page, served at /                    │
   │ ├─ app.html, app.js, js/, styles.css: dashboard, at /app    │
   │ └─ admin.html, strategy.html, assets/, images/              │
   └─────────────────────────────────────────────────────────────┘

SQLite is the default store. Set ``AGENT_RUNS_DATABASE_URL``, ``USERS_DATABASE_URL`` and ``CONTENT_DATABASE_URL`` to keep run history, accounts and agent content in Postgres instead.

API surface (summary)
---------------------

+---------------------------+------------------------------------------+
| Endpoint                  | Purpose                                  |
+===========================+==========================================+
| ``GET /health``           | Health check                             |
+---------------------------+------------------------------------------+
| ``GET /runs``             | List backtest runs                       |
+---------------------------+------------------------------------------+
| ``GET /runs/{id}/equity`` | Equity curve for a run                   |
+---------------------------+------------------------------------------+
| ``GET /compare``          | Compare multiple runs                    |
+---------------------------+------------------------------------------+
| ``POST /backtest/run``    | Start a backtest                         |
+---------------------------+------------------------------------------+
| ``GET /backtest/status``  | Poll backtest job status                 |
+---------------------------+------------------------------------------+
| ``POST /backtest/cancel`` | Cancel a running backtest                |
+---------------------------+------------------------------------------+
| ``GET /ticker``           | Market quote data                        |
+---------------------------+------------------------------------------+
| ``/paper/*``              | Alpaca paper-account data (no UI yet)    |
+---------------------------+------------------------------------------+
| ``GET /config/defaults``  | Default UI / run configuration           |
+---------------------------+------------------------------------------+
| ``/api/auth/*``           | Accounts and sessions (:doc:`accounts`)  |
+---------------------------+------------------------------------------+
| ``/api/credits/*``        | ATL Credits (:doc:`credits_billing`)     |
+---------------------------+------------------------------------------+
| ``/api/v1/*``             | Agents, runs, leaderboards, portfolio    |
+---------------------------+------------------------------------------+
| ``/api/v2/*``             | Agent API (:doc:`agent_api`)             |
+---------------------------+------------------------------------------+

LLM integration example code lives in ``dashboard/backend/llm_integration_example.py`` (reference only, not wired into the main app path).

Related documentation
---------------------

* Discord → backtest → dashboard workflow:
  ``docs/architecture/discord-to-backtest.md``
* Multi-agent orchestration, agent pools, and DAG workflows:
  :doc:`../orchestration/index`.
