Getting Started
===============

Run a backtest in the dashboard
--------------------------------

1. Open `agentic-trading-lab.vercel.app <https://agentic-trading-lab.vercel.app/>`_ or `http://localhost:8000/ <http://localhost:8000/>`_ when running locally, then go to the **My Agents** tab.
2. On an agent's card click **Run Backtest**. Ready-made agents are already waiting there — see :ref:`my-agents-sections` — or click **Add Agent +** to create your own.
3. In the dialog set the **Period** and **Asset Universe**, then click **Run Backtest**. **Allocated Capital** is shown read-only — it is a saved setting on the agent; **Edit in Configure** opens the editor to change it.
4. You stay on **My Agents**. The agent's card switches to a live ``Backtesting…`` state with an elapsed timer, and flips to the finished result when the run ends.

Open the **Backtest** tab for the full run — **Trading Performance** charts the
agent against buy-and-hold and DJIA, next to the trades and the hour-by-hour
decision log.

The backtest always runs the model and the capital saved on the agent; change
either from the agent's **Configure** screen rather than at run time. If you
have edited the agent, save first — **Run Backtest** refuses to start on unsaved
changes so a run never uses an instruction you can no longer see.

Leaving **Trading instruction** empty in **Configure** is a supported state, not
an error: the agent then trades on the platform's built-in default strategy.
Expand **See the default instruction** in the editor to read exactly what that
is. Clearing the box on an agent that uses a custom multi-step pipeline asks for
confirmation first, because saving replaces that pipeline.

.. _my-agents-sections:

Sections on My Agents
---------------------

**My Agents** groups your agents into four sections, so what an agent trades is
visible without opening it:

**Prompting LLMs**
   Agents you steer with a written instruction. Every agent you create with
   **Add Agent +** starts here — there is no section to pick while creating one;
   file it elsewhere afterwards.

**U.S. Stock Trading**
   Ready-made strategies for U.S. blue-chip stocks, tested hour by hour on real
   market data.

**China A-Share Trading**
   Strategies for Chinese A-shares, following that market's own next-day (T+1)
   rule — shares bought today cannot be sold until the next trading day.

**For Developers: Connected Agents**
   Your own trading program, running anywhere and driving a Lab backtest over
   the API. Needs an access key — see :doc:`external_agents`.

The first three are a label you control: open an agent's **Configure** screen
and pick a different **Section**, or leave it unset to keep the agent under
Prompting LLMs. The section changes only where the agent is filed; it does not
change how it trades, what it can buy, or any of its settings. Connected agents
are placed by what they are, so they have no **Section** picker; neither do the
sample agents shown before you create one of your own.

Sections are always shown, empty or not. An empty **U.S. Stock Trading** or
**China A-Share Trading** section links straight to the matching filter in
**Community**, which is where ready-made agents for that market come from.

.. _allocated-capital:

Two kinds of allocated capital
------------------------------

Both live in the **Allocated Capital** card on an agent's **Configure** screen,
but they are different kinds of money and have their own limits:

**Paper Trading Allocated Capital** (the card's *Paper Trading* field)
   Real cash reserved from **My Portfolio** for that one agent's paper trading,
   set when you create the agent and editable afterwards. Maximum **$3,000**.

**Backtest Allocated Capital** (the card's *Backtesting* field)
   Simulated starting cash for that agent's backtests. It is a saved per-agent
   setting rather than a per-run choice, so the **Run Backtest** dialog shows it
   read-only. Leave the field blank and it follows the agent's Paper Trading
   Allocated Capital. A backtest never spends real portfolio cash and never
   changes it. Minimum **$1**, maximum **$10,000**.

Paper trading is not switched on yet, so the first of those two currently has
nothing to spend. **Run Paper Trading** sits beside **Run Backtest** on every
agent card but is permanently disabled (*Paper trading is coming soon*), and the
**Playground → Paper Trading** tab is a read-only view of a connected Alpaca
paper account rather than somewhere your agent runs. Backtesting is the only way
to run an agent today.

The reservation is real in the meantime: Paper Trading Allocated Capital leaves
**My Portfolio** the moment you set it and reduces the cash available to your
other agents. You can drop it to **$0** on an agent you only plan to backtest —
give that agent its own **Backtesting** amount first, though, or its backtests
fall back to $1,000 along with it.

Start from a template
---------------------

Rather than writing an agent from scratch, open **Community → Agent
Supermarket** and add a ready-made template to **My Agents**, then edit its
prompts and backtest it. The section chips there filter templates by market, so
you can jump straight to U.S. stocks or A-shares. See :doc:`marketplace`.

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
