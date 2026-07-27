Getting Started
===============

Run a backtest in the dashboard
--------------------------------

1. Open `agentic-trading-lab.vercel.app <https://agentic-trading-lab.vercel.app/>`_ or `http://localhost:8000/ <http://localhost:8000/>`_ when running locally. Stay on the **Backtest** tab.
2. Set the date range, assets, and model in the left sidebar.
3. Click **Run Backtest**.
4. Wait for completion—the UI polls ``/backtest/status`` and reloads equity charts when done.

Results appear in **Trading Performance** (agent vs. buy-and-hold vs. DJIA).

Start from a template
---------------------

Rather than writing an agent from scratch, open **Playground → Agent
Marketplace** and copy a ready-made template into **My Agents**, then edit its
prompts and backtest it. See :doc:`marketplace`.

Accounts (optional)
-------------------

Backtests and paper trading work without signing in. Creating an account
persists the agents you register, attributes leaderboard runs to them, and lets
you link Discord. See :doc:`accounts` to sign up and manage your profile.

CLI backtest (optional)
-----------------------

For headless or scripted runs:

.. code-block:: bash

   python3 dashboard/scripts/backtest_hourly_agent.py --start 2026-03-01 --end 2026-03-31
   python3 dashboard/scripts/backtest_hourly_agent.py --mode buy_and_hold

Inspect results in the dashboard after a CLI run, or call ``POST /backtest/run`` with the same parameters the UI sends.

Local deployment
----------------

Install dependencies
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

Configure Alpaca credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use **either** environment variables **or** a local credentials file.

**Option A — ``.env`` (recommended for deploy):**

.. code-block:: bash

   cp .env.example .env
   # ALPACA_API_KEY=...
   # ALPACA_SECRET_KEY=...
   # ALPACA_BASE_URL=https://paper-api.alpaca.markets

**Option B — credentials file (CLI and local API fallback):**

.. code-block:: bash

   cp credentials/alpaca.json.example credentials/alpaca.json

The ``credentials/`` directory is not tracked in git. See ``credentials/README.md``.

Configure Robinhood live trading (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Live trading against a real brokerage account is off unless you configure it,
and orders are never sent unless you also set ``ROBINHOOD_EXECUTE=true``. See
:ref:`robinhood-config` for the full variable list.

Start the API server
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # from the repository root (the backend is the ``dashboard.backend`` package)
   uvicorn dashboard.backend.app:app --reload

   # equivalent module entrypoint:
   python3 -m dashboard.backend.app

Open the dashboard at `http://localhost:8000/ <http://localhost:8000/>`_.
