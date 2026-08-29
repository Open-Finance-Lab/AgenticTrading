Agent Supermarket
=================

The **Agent Supermarket** is a catalog of ready-made agent templates. Add one to
**My Agents** (see :ref:`my-agents-sections`), then edit, backtest, or run it
like any agent you built yourself.

Open the dashboard, go to **Community**, and use the **Agent Supermarket**.
Browsing needs no account.


Browse the catalog
------------------

Each card shows what you are adding:

- **Model and section** — the AI model the template is tuned for (shown by
  provider, such as *Claude* or *Gemini*; you can change it afterwards), then
  the **My Agents** section the template lands in.
- **Hosted**, **Multi-step strategy**, or **Simple instruction**, with the step
  count for multi-step templates.
- **Tags**, and either the template author or a link to its GitHub repository.

Filter with the section chips above the grid — **All**, **Prompting LLMs**,
**U.S. Stock Trading**, **China A-Share Trading** — and narrow further with the
search box, which matches name, description, section, author, tags, and model.
The two compose.

Templates shipped today:

.. list-table::
   :header-rows: 1
   :widths: 20 20 22 38

   * - Template
     - Section
     - Model
     - What it does
   * - **Balanced Starter**
     - Prompting LLMs
     - ``anthropic/claude-haiku-4-5``
     - Diversifies across strong stocks, buys dips, takes profits after run-ups.
   * - **Momentum Scout**
     - Prompting LLMs
     - ``anthropic/claude-haiku-4-5``
     - Follows recent price strength and volume; trims laggards quickly.
   * - **Three-Step Analyst**
     - U.S. Stock Trading
     - ``anthropic/claude-sonnet-4-6``
     - Three steps — gather market facts, convert them into signals, then
       produce executable orders.
   * - **AI Hedge Fund**
     - U.S. Stock Trading
     - ``nvidia/nemotron-3-nano-30b-a3b``
     - A hosted panel of AI analysts that weigh in on every trade. Based on the
       open-source `AI Hedge Fund <https://github.com/virattt/ai-hedge-fund>`_
       project by virattt.
   * - **Blue-Chip Steady**
     - U.S. Stock Trading
     - ``anthropic/claude-haiku-4-5``
     - Buys and holds a handful of the strongest Dow companies. Mirrors the
       buy-and-hold benchmark on the leaderboard.
   * - **Even-Split Dow**
     - U.S. Stock Trading
     - ``anthropic/claude-haiku-4-5``
     - Spreads the money evenly across all available Dow stocks and keeps the
       split even. Mirrors the equal-weight benchmark on the leaderboard.
   * - **A-Share Steady (T+1)**
     - China A-Share Trading
     - ``anthropic/claude-haiku-4-5``
     - A patient strategy for Chinese A-shares, built for that market's rule
       that shares bought today cannot be sold until the next trading day.


Add one to My Agents
--------------------

1. Click **Add to My Agents** on a card.
2. You land on **Playground → My Agents** and the new agent's **Configure**
   screen opens straight away — owned by you, with the template's prompt
   pipeline already filled in.
3. Rename it, change the model, set its capital, or rewrite the instruction,
   then close the editor to find the agent waiting on your **My Agents** grid,
   under the section the template came from. Your agent is an independent copy —
   later edits to the template do not touch it, and your edits never affect
   anyone else's. The template's section is only a starting point: **Configure →
   Section** moves the agent to a different one at any time.

New agents get the default **Paper Trading Allocated Capital** ($1,000), and
backtests start from that amount until you set a separate **Backtesting** amount
in the added agent's **Configure** screen — see :ref:`allocated-capital`. If you
are signed in, the $1,000 is reserved from your account portfolio, so adding an
agent can fail with *insufficient cash* until you free some up.

.. note::

   You can add templates without signing in — the agent is then tied to your
   browser session and disappears when it expires. Sign in first if you want it
   to persist.


Contribute a template
---------------------

The catalog is config-driven: templates live in
``dashboard/config/marketplace.json``, so contributing one needs no code or
database change. Add an entry with a unique ``template_id``, a ``name``,
``description``, ``category``, ``model_name``, ``tags``, and the ``pipeline``
steps, then open a pull request. Entries missing ``template_id`` or ``name`` are
skipped.

``category`` must be one of ``prompting_llms``, ``us_stocks``, or
``cn_ashares`` — the same three values the section chips and the **My Agents**
sections use. Anything else is treated as uncategorized: the template still
appears under **All** and in search results, but no section chip finds it, and
it sorts to the end of the listing. Cards are grouped by section first (in the
order above) and by name within a section, so the value also decides where a
new template appears.

The catalog is cached in-process, so a running server picks up edits on restart.
