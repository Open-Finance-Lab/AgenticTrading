Agent Marketplace
=================

The **Agent Marketplace** is a catalog of ready-made agent templates. Select one
into :doc:`My Agents <external_agents>`, then edit, backtest, or run it like any
agent you built yourself.

Open the dashboard, go to **Community**, and use the **Agent Marketplace**.
Browsing needs no account.


Browse the catalog
------------------

Each card shows what you are selecting:

- **Category** — ``Foundation`` for single-instruction agents, ``Advanced`` for
  multi-step pipelines.
- **Model** — the LLM the template is tuned for (you can change it after selecting).
- **Simple** or **Multi-step pipeline**, with the step count.
- **Tags** and the template author.

Use the search box to filter by name, description, or tag.

Templates shipped today:

.. list-table::
   :header-rows: 1
   :widths: 22 16 22 40

   * - Template
     - Category
     - Model
     - What it does
   * - **Balanced Starter**
     - Foundation
     - ``anthropic/claude-haiku-4-5``
     - Diversifies across strong stocks, buys dips, takes profits after run-ups.
   * - **Momentum Scout**
     - Foundation
     - ``anthropic/claude-haiku-4-5``
     - Follows recent price strength and volume; trims laggards quickly.
   * - **Pipeline Analyst**
     - Advanced
     - ``anthropic/claude-sonnet-4-6``
     - Three steps — gather market facts, convert them into signals, then
       produce executable orders.


Select a template
-----------------

1. Click **Select to My Agents** on a card.
2. The agent appears under **Playground → My Agents**, owned by you, with the
   template's prompt pipeline already filled in.
3. Open it in the agent editor to rename it, change the model, or rewrite the
   prompts. The copy is independent — later edits to the template do not touch
   your agent, and your edits never affect anyone else's copy.

New agents get the default **Paper Trading Allocated Capital** ($1,000), and a
backtest starts from that amount unless you change it in the **Run Backtest**
dialog — see :ref:`allocated-capital`. If you are signed in, the $1,000 is
reserved from your account portfolio, so a select can fail with *insufficient
cash* until you free some up.

.. note::

   You can select templates without signing in — the agent is then tied to your
   browser session and disappears when it expires. Sign in first if you want it
   to persist.


Add a template
--------------

The catalog is config-driven: templates live in
``dashboard/config/marketplace.json``, so contributing one needs no code or
database change. Add an entry with a unique ``template_id``, a ``name``,
``description``, ``category``, ``model_name``, ``tags``, and the ``pipeline``
steps, then open a pull request. Entries missing ``template_id`` or ``name`` are
skipped.

The catalog is cached in-process, so a running server picks up edits on restart.
