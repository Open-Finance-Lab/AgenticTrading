Getting Started
===============

Run a backtest in the dashboard
--------------------------------

1. Open `agentic-trading-lab.vercel.app <https://agentic-trading-lab.vercel.app/>`_ or `http://localhost:8000/ <http://localhost:8000/>`_ when running locally, then go to the **My Agents** tab.
2. On an agent's card click **Run Backtest**. Use one of the **Foundation Agents**, or click **Add Agent +** to create your own.
3. In the dialog set the **Period**, **Asset Universe**, and **Backtest Allocated Capital**, then click **Run Backtest**.
4. Wait for completion—the UI polls ``/backtest/status`` and reloads equity charts when done.

Results appear in **Trading Performance** (agent vs. buy-and-hold vs. DJIA).

The backtest always runs the model saved on the agent; change it from the
agent's **Configure** screen rather than at run time. If you have edited the
agent, save first — **Run Backtest** refuses to start on unsaved changes so a
run never uses an instruction you can no longer see.

.. _allocated-capital:

Two kinds of allocated capital
------------------------------

The dashboard keeps these separate, and they have their own limits:

**Paper Trading Allocated Capital**
   Cash reserved from **My Portfolio** for one agent's paper trading, set when
   you create the agent and editable in **Configure**. Maximum **$3,000**.

**Backtest Allocated Capital**
   Simulated starting cash for a single backtest, set in the **Run Backtest**
   dialog. It defaults to that agent's Paper Trading Allocated Capital, but a
   backtest never spends real portfolio cash and never changes it. Maximum
   **$10,000**.

Start from a template
---------------------

Rather than writing an agent from scratch, open **Community → Agent
Marketplace** and add a ready-made template to **My Agents**, then edit its
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
