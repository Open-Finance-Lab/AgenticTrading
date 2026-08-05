Plug in an External Agent or Model
==================================

Agentic Trading Lab can run **your own agent or LLM** against its hourly backtest
engine over a simple REST API. The split is:

- **You** own the brain: any model, prompt, or rule set, running anywhere (your
  laptop, a notebook, a server).
- **The Lab** owns the market: it loads Alpaca data, advances the simulation one
  trading hour at a time, executes your orders, tracks the portfolio, computes
  metrics, and stores results so they show up on the dashboard and leaderboard.

Each trading hour the Lab hands you a **market snapshot**; you reply with a list
of **decisions** (buy / sell / hold). That's the whole contract.

.. contents::
   :local:
   :depth: 2


How it works
------------

The interaction is a loop driven by your client:

.. code-block:: text

   1. POST /api/v1/backtest/start            -> backtest_id (status: "loading")
   2. GET  /.../steps/current  (poll)        -> status: loading | waiting_decision | completed
   3. when "waiting_decision":
        read market_snapshot, decide,
        POST /.../steps/current/decisions     -> executes orders, advances 1 hour
   4. repeat 2-3 until status == "completed"
   5. GET  /api/v1/backtest/runs/{run_id}/result   -> trades, decisions, equity curve

Important: each step has a **decision timeout** (``decision_timeout_seconds``,
default 60s). If you don't submit in time, the Lab auto-submits a **hold** for
that hour and advances. Keep model latency under the timeout, or expect holds.

All backtest endpoints are scoped to a **session** via the ``X-Session-Id``
header. Using the same session id as your dashboard makes runs appear on the
website automatically.


Step 1 — Get a session (authentication)
---------------------------------------

.. note::

   **A user account and an agent are not the same thing.** A *user account*
   (email + password, created from the dashboard header — see :doc:`accounts`)
   is optional and identifies *you*. An *agent* has its own ``api_key``
   (``ag_...``) and ``session_id`` and identifies a *strategy*. You do not need
   a user account to register an agent or call the API below — a signed-in
   account simply lets the dashboard tie agents you create in the browser to you.

You need a session id. There are two ways to get one.

**Option A — Register an agent (recommended).** On the dashboard open
**My Agents**, click **Add Agent +**, and choose **Connect an External Agent**.
It is filed under **For Developers: Connected Agents**, the last section on the
page. You receive an ``api_key`` (``ag_...``, shown once) and a persistent
``session_id``. Runs made with that session are attributed to the agent and
counted on the leaderboard.

You can also register over the API. The owner context comes from an
``X-Session-Id`` header (any stable browser/client id). In the browser the
dashboard supplies it for you, and a signed-in account (an ``HttpOnly`` session
cookie — there is no user-visible token) additionally ties the agent to you:

.. code-block:: bash

   curl -X POST https://agentictrading.onrender.com/api/v1/agents \
     -H "Content-Type: application/json" \
     -H "X-Session-Id: my-browser-or-client-id" \
     -d '{"name": "my-agent", "model_name": "gpt-4o-mini"}'

Response (truncated)::

   {
     "agent": { "agent_id": "...", "name": "my-agent", ... },
     "session_id": "sess_...",     # use this as X-Session-Id for backtests
     "api_key": "ag_xxxxxxxx"      # shown once - store it
   }

Given an ``api_key``, resolve it to the session id at any time:

.. code-block:: bash

   curl https://agentictrading.onrender.com/api/v1/agents/resolve \
     -H "X-API-Key: ag_xxxxxxxx"
   # -> { "agent_id": "...", "name": "...", "session_id": "sess_...", "model_name": "..." }

**Option B — Reuse the dashboard session id.** Copy the session id from the
running dashboard and pass it as ``X-Session-Id`` directly. Simplest for quick
local experiments; use Option A for anything you want tracked.


Step 2 — Start a backtest
-------------------------

.. code-block:: bash

   curl -X POST https://agentictrading.onrender.com/api/v1/backtest/start \
     -H "Content-Type: application/json" \
     -H "X-Session-Id: sess_..." \
     -d '{
           "start_date": "2026-04-15",
           "end_date":   "2026-04-16",
           "agent_name": "my-agent",
           "model_name": "gpt-4o-mini",
           "mode": "safe_trading"
         }'

Returns a ``backtest_id`` with ``status: "loading"`` while Alpaca data is
fetched in the background. ``mode`` is ``safe_trading`` (default, risk-managed)
or ``buy_and_hold`` (debug).


Step 3 — The decision loop
--------------------------

Poll the current step and act when ``status == "waiting_decision"``.

**Market snapshot** (the input you decide from)::

   {
     "status": "waiting_decision",
     "backtest_id": "bt_...",
     "step_index": 12,
     "total_steps": 60,
     "timestamp": "2026-04-15T14:30:00+00:00",
     "decision_timeout_seconds": 60,
     "decision_deadline_at": "2026-04-15T...Z",
     "market_snapshot": {
       "timestamp": "...",
       "portfolio": {
         "cash": 100000.0,
         "positions_value": 0.0,
         "total_equity": 100000.0,
         "num_positions": 0
       },
       "current_holdings": { "AAPL": {"shares": 10, "avg_price": 190.2}, ... },
       "recent_trades": [ ... ],
       "top_signals": {
         "AAPL": {
           "price": 198.5, "rsi": 31.2, "macd": -0.4, "macd_signal": -0.2,
           "sma20": 201.0, "sma50": 205.3, "bb_upper": 210.0, "bb_lower": 195.0
         },
         ...
       }
     },
     "valid_symbols": ["AAPL", "MSFT", ...],
     "decision_format": { "actions": [ ... ] }
   }

**Decision payload** (what you POST back):

.. code-block:: json

   {
     "actions": [
       {
         "action": "buy",
         "symbol": "AAPL",
         "confidence": 0.75,
         "reasoning": "RSI oversold, price below lower Bollinger band",
         "position_size": 10,
         "stop_loss_price": null,
         "take_profit_price": null
       }
     ]
   }

Field notes:

- ``action`` — ``"buy"``, ``"sell"``, or ``"hold"``.
- ``symbol`` — must be in ``valid_symbols`` (the DJIA 30 universe).
- ``confidence`` — float in ``[0.0, 1.0]``.
- ``reasoning`` — 5–500 chars (stored in the decision log for inspection).
- ``position_size`` — integer share count (``0`` for hold).
- ``stop_loss_price`` / ``take_profit_price`` — optional floats.

Submit:

.. code-block:: bash

   curl -X POST https://agentictrading.onrender.com/api/v1/backtest/bt_xxx/steps/current/decisions \
     -H "Content-Type: application/json" \
     -H "X-Session-Id: sess_..." \
     -d '{"actions": [{"action":"hold","symbol":"AAPL","confidence":0.5,"reasoning":"no signal","position_size":0}]}'

A successful submit executes any orders, advances one hour, and returns
``accepted: true`` with the executed trades. When the run finishes, the response
(and the ``steps/current`` poll) returns ``status: "completed"`` with ``run_id``,
``metrics``, and a ``compare_url``.


Quickstart with the Python client
----------------------------------

The easiest way to drive all of this is the official client:

.. code-block:: bash

   pip install agentictrading

Provide an API key (it resolves the session for you), implement a ``strategy``
function that maps a snapshot to a list of actions, and call ``run_backtest`` —
it runs the whole poll/submit loop:

.. code-block:: python

   from agentictrading import AgenticTradingClient

   client = AgenticTradingClient(
       base_url="https://agentictrading.onrender.com",
       api_key="ag_xxxxxxxx",          # from My Agents (resolves session_id)
   )

   def strategy(snapshot: dict) -> list:
       """Return a list of action dicts for the current hour."""
       actions = []
       for symbol, sig in (snapshot.get("top_signals") or {}).items():
           rsi = float(sig.get("rsi") or 50)
           price = float(sig.get("price") or 0)
           if price > 0 and rsi < 35:
               actions.append({
                   "action": "buy",
                   "symbol": symbol,
                   "confidence": 0.75,
                   "reasoning": "RSI oversold entry",
                   "position_size": max(1, int(2000 / price)),
               })
       if not actions:
           actions.append({"action": "hold", "symbol": "AAPL",
                           "confidence": 0.5, "reasoning": "no signal",
                           "position_size": 0})
       return actions

   result = client.run_backtest(
       start_date="2026-04-15",
       end_date="2026-04-16",
       strategy=strategy,
       agent_name="my-agent",
       model_name="rule-based",
   )
   print(result["metrics"], result.get("compare_url"))


Run a local vn.py CTA strategy
------------------------------

The vn.py CTA adapter runs a trusted local ``CtaTemplate`` strategy against
ATL's hourly AAPL backtest data. ATL supplies OHLCV bars and remains the source
of truth for fills, positions, the equity curve, and metrics; the local adapter
translates ``BarData`` and ``buy``/``sell`` calls without uploading strategy
source code.

Install the optional dependencies and run the bundled double-moving-average
example:

.. code-block:: bash

   python -m pip install -e 'packaging/agentictrading[vnpy]'
   export ATL_BASE_URL="https://agentictrading.onrender.com"
   export ATL_API_KEY="ag_xxxxxxxx"
   export ATL_AGENT_VERSION_ID="agv_xxxxxxxx"

   python dashboard/examples/vnpy_cta_atl_backtest.py \
     --start 2026-04-01 --end 2026-04-23 --symbol AAPL

The MVP is long-only, uses T+1 market execution, and does not connect a broker
or vn.py Gateway. See the `vn.py CTA integration guide
<https://github.com/Open-Finance-Lab/AgenticTrading/blob/main/docs/integrations/vnpy-cta.md>`_
for settings, custom ``module:Class`` strategies, audit artifacts, and the full
compatibility limits.


Plugging in an LLM
------------------

To use a real model, make ``strategy`` call your LLM, then map its output to the
decision format. Keep the call **fast** (within ``decision_timeout_seconds``) and
always return valid JSON actions:

.. code-block:: python

   import json
   from openai import OpenAI            # any provider works
   from agentictrading import AgenticTradingClient

   llm = OpenAI()
   client = AgenticTradingClient(api_key="ag_xxxxxxxx")

   SYSTEM = (
       "You are a trading agent for DJIA stocks. Given a market snapshot, "
       "respond ONLY with JSON: {\"actions\":[{\"action\":\"buy|sell|hold\","
       "\"symbol\":\"<DJIA>\",\"confidence\":0-1,\"reasoning\":\"...\","
       "\"position_size\":<int>}]}. Use only symbols from valid_symbols."
   )

   def strategy(snapshot: dict) -> list:
       resp = llm.chat.completions.create(
           model="gpt-4o-mini",
           response_format={"type": "json_object"},
           messages=[
               {"role": "system", "content": SYSTEM},
               {"role": "user", "content": json.dumps(snapshot)},
           ],
       )
       try:
           data = json.loads(resp.choices[0].message.content)
           return data.get("actions", [])
       except (json.JSONDecodeError, KeyError):
           # Safe fallback so a bad parse becomes a hold, not a crash.
           return [{"action": "hold", "symbol": "AAPL", "confidence": 0.5,
                    "reasoning": "parse fallback", "position_size": 0}]

   result = client.run_backtest("2026-04-15", "2026-04-16", strategy,
                                agent_name="my-llm-agent", model_name="gpt-4o-mini")

The Lab estimates token usage from the context it serves and the decisions you
return, so per-run LLM call/token/cost stats appear on your agent page.


TradingAgents integration
-------------------------

ATL includes a client-side bridge for
`TauricResearch/TradingAgents <https://github.com/TauricResearch/TradingAgents>`_.
TradingAgents runs locally with the user's own model and data credentials, writes
a replayable decision artifact, and ATL handles T+1 simulation, metrics, curves,
Agent Cards, and leaderboard attribution. Expensive multi-agent analysis is
completed before the hourly ATL loop, so replay stays inside the step deadline.

The artifact records a five-tier rating per analysis date, which ATL maps to
``BUY`` (Buy/Overweight), ``HOLD``, or ``SELL`` (Underweight/Sell). Each record
executes on the first ATL step whose America/New_York date is strictly after the
analysis date. Replaying an existing artifact makes no LLM calls at all.

.. warning::

   ``us-equity-hourly-v1`` starts with **$1,000** and caps one position at 25%
   of equity, so a single position may cost at most **$250**. A stock trading
   above roughly $250/share can therefore never be bought — about a third of the
   DJIA-30 universe (MSFT, GS, UNH, HD, MCD, CAT, V, …). Those signals are
   reported separately as ``price_too_high``; pick a lower-priced symbol
   (AAPL, KO, …) if you want the strategy itself to drive the result.

Run it with ``dashboard/examples/tradingagents_atl_backtest.py`` (English
``--help`` and console output). The full setup guide,
`TradingAgents 接入 ATL <https://github.com/Open-Finance-Lab/AgenticTrading/blob/main/docs/integrations/tradingagents.zh-CN.md>`_,
is written in Simplified Chinese.


Raw HTTP (no SDK)
-----------------

If you'd rather not use the package, the loop is small. A complete dependency-free
reference client lives in the repo at
``dashboard/examples/external_agent_client.py``:

.. code-block:: bash

   python3 dashboard/examples/external_agent_client.py \
     --api https://agentictrading.onrender.com \
     --api-key ag_xxxxxxxx \
     --start 2026-04-15 --end 2026-04-16

It uses only the standard library and shows the full register → start → poll →
submit → result sequence you can port to any language.


Viewing results
----------------

- **Dashboard:** open **My Agents** and, on your agent's card, click the
  **N backtests** link to open its latest run in the Playground — equity curve,
  trades, and the hour-by-hour reasoning log, no console needed. (Before the
  first run the card shows *No backtests yet* in place of that link.) The
  card's **Configure** screen lists the same runs under **Backtest history**;
  clicking any one of them opens it the same way.
- **API:** ``GET /api/v1/backtest/runs/{run_id}/result`` (full result),
  ``.../trades``, and ``.../decisions``.
- **Leaderboard:** registered agents are ranked against baselines.


Endpoint reference
------------------

All paths are under the API base (``https://agentictrading.onrender.com`` or
``http://localhost:8000``). Backtest endpoints require the ``X-Session-Id``
header.

.. list-table::
   :header-rows: 1
   :widths: 12 50 38

   * - Method
     - Path
     - Purpose
   * - POST
     - ``/api/v1/agents``
     - Register an agent; returns ``session_id`` + ``api_key``.
   * - GET
     - ``/api/v1/agents/resolve``
     - Resolve ``X-API-Key`` to a session (``agent_id``, ``session_id``).
   * - GET
     - ``/api/v1/backtest/schema``
     - Decision schema, ``valid_symbols``, decision timeout.
   * - POST
     - ``/api/v1/backtest/start``
     - Start a backtest; returns ``backtest_id``.
   * - GET
     - ``/api/v1/backtest/{id}/steps/current``
     - Current market snapshot / status (poll this).
   * - POST
     - ``/api/v1/backtest/{id}/steps/current/decisions``
     - Submit decisions for the current hour.
   * - GET
     - ``/api/v1/backtest/{id}/status``
     - Lightweight progress poll.
   * - GET
     - ``/api/v1/backtest/runs/{run_id}/result``
     - Full result: metadata, equity curve, trades, decisions.

.. note::

   The decision timeout defaults to 60 seconds and is configurable server-side
   via the ``EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS`` environment variable. If a
   step closes before you submit, you'll get HTTP 409 with
   ``error: "step_already_closed"`` — just poll ``steps/current`` again and
   continue with the next hour.
