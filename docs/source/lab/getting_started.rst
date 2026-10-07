Getting Started
===============

Run a backtest in the dashboard
--------------------------------

1. Open the dashboard at ``/app`` — `agentic-trading-lab.vercel.app/app <https://agentic-trading-lab.vercel.app/app>`_, or ``http://localhost:8000/app`` when running locally (``/`` is only the landing page) — then go to the **My Agents** tab.
2. On an agent's card click **Run Backtest**. Ready-made agents are already waiting there — see :ref:`my-agents-sections` — or click **Add Agent +** to create your own.
3. In the dialog set the **Period** (the end date can be at most 14 days after the start) and **Asset Universe**, then click **Run Backtest**. **Allocated Capital** is shown read-only — it is a saved setting on the agent; **Edit in Configure** opens the editor to change it.
4. You stay on **My Agents**. The agent's card switches to a live ``Backtesting…`` state with an elapsed timer, and flips to the finished result when the run ends.

While you have no finished backtest yet, a **Get your first result** panel on
**My Agents** walks the same two steps — run a backtest, then see its results on
the **Backtest** tab — and ticks them off as you go.

**AI billing.** A backtest where an AI model makes the decisions needs you to be
signed in and to choose how the model calls are paid for, in the same dialog:
**Use my API key** (bring your own provider key) or **Use ATL Credits**. Pick a
**Provider** where offered, and choose the **Model for this run**; if no key is
saved and Credits are unavailable, **Go to API Keys** takes you to where keys
are stored. See :doc:`credits_billing` for how each option is charged. Runs where
no AI model is involved (rule-based decisions) work without signing in. So does
the hosted AI Hedge Fund runtime: its model is managed by the platform, so it
shows no billing choice and spends no Credits.

A run does not start pricing hours the moment you click. The card names the
stage it is in while it gets there — ``Loading market data``, then
``Calculating indicators``, then ``Waiting on first decision`` — and switches
to a bar count and a percentage (``0/49``, then ``12/49`` and ``35%``) once the
hour-by-hour loop is running, ending on ``Saving results``. A card resting on
one of those opening stages is working, not stuck. Once it is running the card
also shows a small live equity curve and the return so far, and carries a
**Cancel** button (the **Backtest** tab has one too) if you want to stop the
run. A run that exceeds its time budget ends as *timed out* rather than as a
generic failure.

Open the **Backtest** tab for the full run — **Trading Performance** charts the
agent against buy-and-hold and DJIA, next to the trades and the hour-by-hour
decision log. When a model-driven run held some steps instead of letting the
model decide, the run's configuration shows an "N of M steps model-driven" row.

The backtest runs with the starting capital saved on the agent; change it from
the agent's **Configure** screen rather than at run time. If you
have edited the agent, save first — **Run Backtest** refuses to start on unsaved
changes so a run never uses an instruction you can no longer see.

Leaving **Trading instruction** empty in **Configure** is a supported state, not
an error: the agent then trades on the default instruction. Expand **See the
default instruction** in the editor to read exactly what that is. Clearing the
box on an agent that uses a custom multi-step pipeline asks for confirmation
first, because saving replaces that pipeline.

.. _my-agents-sections:

Sections on My Agents
---------------------

**My Agents** groups your agents into shelves, so what an agent is is visible
without opening it:

**LLMs**
   Agents you steer with a written instruction. Every agent you create with
   **Add Agent +** starts here. Market chips above the shelf (**All**,
   **U.S.**, **China A-Share**) narrow it to one market.

**Open Agents**
   Open-source trading agents such as AI Hedge Fund. Add them from
   **Community**, then customize and backtest them.

**Research Agents**
   Deep Research agents that produce analyst reports rather than trades. Add
   them from **Community**, fill in a mandate, and download the report. Shown
   only when you are signed in. See :doc:`research_agents`.

**Crypto** and **Futures**
   Shown as *Not yet available*. Nothing in them can be run.

**For Developers: Connected Agents**
   Your own trading program, running anywhere and driving a Lab backtest over
   the API. Needs an access key — see :doc:`external_agents`.

Which shelf an agent sits on is automatic — it follows what kind of agent it is
— so there is nothing to file it under. The **Market** picker in an agent's
**Configure** screen (*Not set*, *U.S.*, *China A-Share*) only sets which market
chip and label the agent carries; it does not move the agent to another shelf,
change how it trades, what it can buy, or any of its settings. Connected agents
have no **Market** picker; neither do the sample agents shown before you create
one of your own.

Shelves are shown even when empty (except **Research Agents**, which is hidden
while you are signed out). An empty **LLMs**, **Open Agents** or **Research
Agents** shelf has a **Community** button that opens the full catalog, which is
where ready-made agents come from. On **LLMs** with a market chip selected, it
opens Community filtered to that market instead.

.. _allocated-capital:

Allocated capital
-----------------

The **Allocated Capital** card on an agent's **Configure** screen has two fields.

**Backtesting**
   Simulated starting cash for that agent's backtests. It is a saved per-agent
   setting rather than a per-run choice, so the **Run Backtest** dialog shows it
   read-only. Leave the field blank and the agent starts from $1,000 (an agent that
   already holds a funded sleeve starts from that amount instead).
   A backtest never spends real money and never changes your portfolio.
   Minimum **$0**, maximum **$3,000**.

   **$0 is a real amount, and blank is not $0.** Type ``0`` and the backtest
   runs like any other: the agent simply has nothing to trade, so every order
   is refused for insufficient cash, the equity curve stays flat at $0, and the
   return is reported as ``0.00%``. Leaving the field *empty* means something
   different — "not configured" — and falls back as described above.

**Paper Trading**
   Greyed out and labelled *coming soon*. Paper trading is coming in the
   future, so this field cannot be edited today and new agents reserve nothing
   in it. Backtesting is the way to run an agent today.

Start from a template
---------------------

Rather than writing an agent from scratch, open **Community → Agent
Supermarket** and add a ready-made template to **My Agents**, then edit its
prompts and backtest it. The market chips there filter templates by market, so
you can jump straight to U.S. stocks or A-shares. See :doc:`marketplace`.

Accounts (optional)
-------------------

You can try rule-based backtests without signing in. Creating an account
persists the agents you register, is required for any backtest driven by an AI
model you pick (together with an AI billing choice, above — the hosted AI Hedge
Fund runtime is the exception), and lets you link Discord. Neither dashboard
leaderboard takes user submissions, so your dashboard backtests never appear on
one. See :doc:`accounts` to sign up and manage your
profile, and :doc:`credits_billing` for the AI billing options.

CLI backtest (optional)
-----------------------

For headless or scripted runs without an AI model:

.. code-block:: bash

   python3 dashboard/scripts/backtest_hourly_agent.py --no-llm --start 2026-03-01 --end 2026-03-31
   python3 dashboard/scripts/backtest_hourly_agent.py --no-llm --mode buy_and_hold

Without ``--no-llm`` the script defaults to model-driven decisions, which it
refuses to run from the command line: those runs are started only from the
dashboard, which signs you in and supplies the AI billing choice.

Inspect results in the dashboard after a CLI run, or call ``POST /backtest/run``
with the same parameters the UI sends. For a model-driven run the request must
come from a signed-in account and include a ``billing_mode`` (``byok`` or
``platform_credits``).

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

The server reads ``dashboard/.env`` — not a ``.env`` at the repository root.
Copy the template from the repository root into that location:

.. code-block:: bash

   cp .env.example dashboard/.env
   # ALPACA_API_KEY=...
   # ALPACA_SECRET_KEY=...

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

Open the dashboard at ``http://localhost:8000/app``
(``http://localhost:8000/`` is the landing page).
