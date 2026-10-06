Agent Supermarket
=================

The **Agent Supermarket** is a catalog of ready-made agent templates. Add one to
**My Agents** (see :ref:`my-agents-sections`), then edit and backtest it
like any agent you built yourself.

Open the dashboard, go to **Community**, and use the **Agent Supermarket**.
Browsing needs no account.


Browse the catalog
------------------

The page has three shelves, in this order: **LLMs**, **Agents**, and
**Research Agents**. Each template is a card with an **Add to My Agents**
button; the cards differ by shelf:

- **LLMs** — the models on the Competition Leaderboard. The card shows the
  model's company and market (for example *Anthropic · U.S.*), its leaderboard
  rank, and a **Competition result** block: the model's return, the DJIA 30
  universe, the backtest window and starting capital, and a chart against the
  benchmark. That result is a read-only snapshot of the leaderboard run; adding
  the card does not re-run it and does not enter anything on a leaderboard.
- **Agents** — ready-made trading agents. The card shows a short description
  and, for templates built on an open-source project, an **Open Source** chip
  and a link to its GitHub repository.
- **Research Agents** — Deep Research agents that produce analyst reports
  rather than trades. The card shows the typical run time and the report
  formats. See :doc:`research_agents`.

Narrow the catalog with the market chips above the grid — **All**, then
**U.S.** and **China A-Share**. A chip appears only when the catalog holds a
template for that market. The search box matches name, description, author,
tags, and model, and composes with the chips.

Templates shipped today (20):

.. list-table:: LLMs
   :header-rows: 1
   :widths: 30 35 35

   * - Template
     - Model
     - Market
   * - **Claude Haiku 4.5**
     - ``anthropic/claude-haiku-4-5``
     - U.S.
   * - **Claude Sonnet 4.6**
     - ``anthropic/claude-sonnet-4-6``
     - U.S.
   * - **DeepSeek V4 Pro**
     - ``deepseek/deepseek-v4-pro``
     - U.S.
   * - **GPT-5.5**
     - ``openai/gpt-5.5``
     - U.S.
   * - **Gemini 3.1 Pro Preview**
     - ``google/gemini-3.1-pro-preview``
     - U.S.
   * - **Nemotron 3 Nano 30B**
     - ``nvidia/nemotron-3-nano-30b-a3b``
     - U.S.
   * - **Qwen3.7 Plus**
     - ``qwen/qwen3.7-plus``
     - U.S.

Each of these is the Competition Leaderboard's agent for that model, on the same
long-only DJIA hourly backtest as the rest of the board. Add one and edit its
instruction to try the model yourself.

.. list-table:: Agents
   :header-rows: 1
   :widths: 22 24 10 44

   * - Template
     - Model
     - Market
     - What it does
   * - **Balanced Starter**
     - ``anthropic/claude-haiku-4-5``
     - U.S.
     - Diversifies across strong stocks, buys dips, takes profits after run-ups.
   * - **Momentum Scout**
     - ``anthropic/claude-haiku-4-5``
     - U.S.
     - Follows recent price strength and volume; trims laggards quickly.
   * - **Three-Step Analyst**
     - ``anthropic/claude-sonnet-4-6``
     - U.S.
     - Three steps — gather market facts, convert them into signals, then
       produce executable orders.
   * - **AI Hedge Fund**
     - ``nvidia/nemotron-3-nano-30b-a3b``
     - U.S.
     - A team of AI investors that analyzes the market, develops trading ideas,
       and tests them through backtesting. A hosted runtime based on the
       open-source `AI Hedge Fund <https://github.com/virattt/ai-hedge-fund>`_
       project by virattt.
   * - **Blue-Chip Steady**
     - ``anthropic/claude-haiku-4-5``
     - U.S.
     - Buys and holds a handful of the strongest Dow companies, selling only
       when a position deteriorates badly. Mirrors the buy-and-hold benchmark.
   * - **Even-Split Dow**
     - ``anthropic/claude-haiku-4-5``
     - U.S.
     - Spreads the money evenly across all available Dow stocks and keeps the
       split even. Mirrors the equal-weight benchmark.
   * - **Contrarian Dip Buyer**
     - ``openai/gpt-5.5``
     - U.S.
     - Buys stocks that have sold off hard and trims them once they recover —
       the opposite instinct to a momentum strategy.
   * - **Sector Rotator**
     - ``google/gemini-3.1-pro-preview``
     - U.S.
     - Concentrates into whichever part of the market is leading, and moves on
       when leadership changes.
   * - **Volatility Guard**
     - ``deepseek/deepseek-v4-pro``
     - U.S.
     - Holds a steady portfolio in calm markets and cuts exposure when prices
       start swinging.
   * - **A-Share Steady (T+1)**
     - ``anthropic/claude-haiku-4-5``
     - China A-Share
     - A patient strategy for Chinese A-shares, built for that market's rule
       that shares bought today cannot be sold until the next trading day.
   * - **A-Share Momentum (T+1)**
     - ``qwen/qwen3.7-plus``
     - China A-Share
     - Rides the strongest Chinese A-shares while respecting the same T+1 rule.

.. list-table:: Research Agents
   :header-rows: 1
   :widths: 30 70

   * - Template
     - What it does
   * - **Shell Company Screening Agent**
     - Deep Research agent for identifying and screening public shell companies
       for reverse mergers and reverse takeovers, ending in a screening report.
   * - **Due Diligence Agent**
     - Deep Research agent for corporate due diligence: research planning,
       multi-dimensional investigation, risk assessment, valuation and
       deal-impact analysis, ending in a DD report.


Add one to My Agents
--------------------

1. Click **Add to My Agents** on a card.
2. For an LLM or an Agent, you land on **Playground → My Agents** and the new
   agent's **Configure** screen opens straight away, owned by you. A template
   with a prompt pipeline arrives with its instruction already filled in; the
   hosted AI Hedge Fund template shows a managed model and an analyst panel
   instead of an instruction.
3. Rename it, change the model and instruction where the template allows it,
   or set its capital, then close the editor to find the agent on your **My Agents** grid. It lands
   under **Open Agents** if the template is a hosted runtime (AI Hedge Fund),
   and under **LLMs** otherwise. Your agent is an independent copy —
   later edits to the template do not touch it, and your edits never affect
   anyone else's. The market the template trades is a separate setting:
   **Configure → Market** changes it at any time.

Backtests start from **$1,000** unless you enter a different **Backtesting** amount
(up to $3,000) under **Allocated Capital** on the added agent's **Configure** screen —
see :ref:`allocated-capital`. Paper trading is coming in the future, so a new
agent reserves nothing from your account portfolio.

A **Research Agent** is added differently: it needs you to be signed in, and it
does not open Configure. It shows **Added** on the card and appears on the
**Research Agents** shelf in My Agents, where you start a run from its
workbench. See :doc:`research_agents`.

.. note::

   You can add trading templates without signing in — the agent is then tied to
   your browser session and disappears when it expires. Sign in first if you
   want it to persist.

   Adding a template does not mean you can run it. Backtests that call an AI
   model need you to be signed in and to pick how the AI is paid for — your own
   key or ATL Credits (see :doc:`credits_billing`); only rule-based backtests
   run signed out. Neither leaderboard takes submissions, so your own copy is
   never ranked on either one.


Contribute a template
---------------------

The catalog is config-driven: templates live in
``dashboard/config/marketplace.json``, so contributing one needs no code or
database change. Add an entry with a unique ``template_id``, a ``name``,
``description``, ``shelf``, ``model_name``, ``tags``, and the ``pipeline`` steps,
then open a pull request. Entries missing ``template_id`` or ``name`` are skipped.

``shelf`` places the card on the page and must be ``llms``, ``open`` (the
**Agents** shelf), or ``research``. If it is omitted, a template with a hosted
runtime lands on **Agents** and any other lands on **LLMs**. Research entries
carry a ``research`` block pointing at the external service instead of a
``pipeline``. Cards are listed shelf by shelf in that order, and keep the file's
order within a shelf.

``category`` is the market, and must be ``us_stocks`` or ``cn_ashares`` — the
values behind the market chips and **Configure → Market**. Anything else (and
omitting it) is treated as uncategorized: the template still appears under
**All** and in search results, but no market chip finds it.

The catalog is cached in-process, so a running server picks up edits on restart.
