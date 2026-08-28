Live Trading with Robinhood
===========================

Agents can be connected to a **real Robinhood brokerage account** and asked to
propose and place orders against it. This is separate from :doc:`backtesting
<getting_started>` (historical data) and from Alpaca paper trading (simulated
money): here the account, the positions, and the orders are real.

.. warning::

   Live trading places orders with real money in your own brokerage account.
   The agents here are LLM-driven research tools, not investment advice, and
   nothing in the Lab predicts market outcomes. Connect an account only if you
   accept the full risk of the trades an agent may place, and start with the
   review-only mode described below.

Live trading is **off unless the deployment turns it on**. On a server started
with the default configuration, a live run does everything except send the
orders — see :ref:`the two gates <live-two-gates>`.


Before you start
----------------

You need:

- an **ATL account** (see :doc:`accounts`) — the Robinhood link is stored
  against it, so live trading is not available to signed-out browser sessions;
- a **real agent** under **Playground → My Agents**. The demo agents on the
  dashboard cannot be connected; create one, or add one from the
  :doc:`Agent Supermarket <marketplace>` first;
- a **Robinhood account** eligible for Robinhood's Agentic Trading (MCP) access;
- a deployment configured for Robinhood (:ref:`robinhood-config`). If it is not,
  connecting fails with *Robinhood OAuth is not configured*.


Connect your brokerage account
------------------------------

1. Open the agent in the agent editor and find the **Robinhood live trading**
   section.
2. Click **Connect Robinhood**. You are sent to Robinhood to sign in and approve
   access.
3. Robinhood returns you to the dashboard, which finishes the link against the
   account you are signed in as.

The status line then reads **Connected**. Clicking **Connect Robinhood** again
refreshes the account details rather than starting a second link.

.. note::

   The link is completed in two steps on purpose. The page Robinhood redirects
   to cannot prove who you are, so it never stores credentials — it hands the
   dashboard a short-lived, single-use code, and the dashboard redeems it under
   your signed-in session. A link started from one account can never be
   redeemed by another.

**Disconnect** revokes access upstream where Robinhood supports it and deletes
the stored tokens. Tokens are encrypted at rest.


Run an agent live
-----------------

1. Tick **Enable live trading for this agent**. This is per agent, so connecting
   your brokerage account does not arm every agent you own.
2. Click **Run Live**. The button stays disabled until a Robinhood account is
   connected, so it never looks ready before the broker behind it is.

The run then:

- reads your Robinhood portfolio and buying power;
- quotes the tradable universe — the DJIA 30 plus anything you already hold;
- asks the agent's model for orders, given that snapshot;
- puts every order through the risk gate below;
- reviews each surviving order with Robinhood, and places it only if the
  deployment allows execution.

The result panel reports what was proposed, what was rejected and why, and what
was actually submitted.

.. _live-two-gates:

The two gates
~~~~~~~~~~~~~

Orders reach the market only when **both** are true:

1. The agent has **live trading enabled** (the per-agent tick box), and
2. the server has execution turned on (``ROBINHOOD_EXECUTE=true``).

The second gate is a deployment setting, not a UI control. With it off — the
default — the run still fetches your real portfolio, asks the model, and reviews
each order with Robinhood, but every order is recorded as ``skipped`` instead of
being submitted. The panel says *Live review completed (execute off)*, and the
response carries ``"execute_enabled": false``. This is the intended way to watch
what an agent would do with your real account before letting it act.

Risk gate
~~~~~~~~~

Every proposed order is clamped or rejected before it reaches the broker, on
**both** the buy and the sell side:

- **No usable quote → rejected.** An order that cannot be priced cannot be
  sized against the USD cap, so it is never passed through.
- **Notional cap.** Quantity is clamped to ``ROBINHOOD_MAX_ORDER_USD / price``,
  default **$25 per order**. The cap is read per order, so it can be tightened
  on a running server.
- **Share cap.** No single order exceeds 10,000 shares.
- **Sells are clamped to what you hold**, and rejected outright if you hold none
  of that symbol — an agent can never open a short.
- Quantities are rounded **down**, so rounding cannot push an order back over
  the cap. A residual below the minimum order quantity is rejected.

One live run at a time is allowed per account; a second request while one is in
flight returns *A live run is already in progress*. Supply an
``idempotency_key`` when driving the API directly to make a retry safe.


API
---

All routes act as your signed-in ATL account (see :doc:`accounts`). The browser
sends the ``HttpOnly`` session cookie automatically — there is no user-visible
token. A script must sign in with ``POST /api/auth/login`` using a cookie jar,
and on mutating requests echo the CSRF cookie's value (``__Host-atl_csrf``, or
``atl_csrf`` on plain-HTTP dev servers) in an ``X-CSRF-Token`` header.

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Endpoint
     - Purpose
   * - ``GET /api/v1/robinhood/status``
     - Connection state. ``?portfolio=true`` includes account details.
   * - ``DELETE /api/v1/robinhood/disconnect``
     - Revoke upstream where supported and delete stored tokens.
   * - ``POST /api/v1/robinhood/agents/{agent_id}/live-run``
     - Run one live cycle. Body: ``{"dry_run": true, "idempotency_key": "..."}``.
   * - ``POST /api/auth/robinhood/start``
     - Begin the OAuth link; returns ``authorize_url``.
   * - ``POST /api/auth/robinhood/complete``
     - Redeem the link code returned by the OAuth callback.

``dry_run: true`` forces a review-only cycle even on a server with execution
enabled. ``dry_run: false`` defers to ``ROBINHOOD_EXECUTE``.


Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Message
     - What to do
   * - *Robinhood OAuth is not configured*
     - The deployment has no Robinhood settings. See :ref:`robinhood-config`.
   * - *Create a real agent in My Agents first*
     - You are on a demo agent. Create one, or add one from the Agent
       Supermarket, then retry.
   * - *Connect Robinhood first*
     - No brokerage account is linked to your ATL account.
   * - *Enable live trading for this agent*
     - Tick the per-agent box; connecting alone does not arm an agent.
   * - *This Robinhood link was started from a different account*
     - The link was begun while signed in as someone else. Start again from the
       account you want to link.
   * - *Link expired — please connect again*
     - The one-time link code lapsed (10 minutes) or was already used.
   * - *A live run is already in progress*
     - Wait for the current run to finish.
   * - *No LLM provider is configured*
     - The server has no LLM API key, so no decision could be made. Nothing was
       traded.


.. _robinhood-config:

Configuration (self-hosting)
----------------------------

Set these in ``dashboard/.env`` (see ``.env.example``):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Meaning
   * - ``BROKER_TOKEN_ENCRYPTION_KEY``
     - **Required.** Fernet key encrypting broker tokens at rest. Without it the
       store refuses to read or write credentials — it deliberately does not
       fall back to a derived key.
   * - ``ROBINHOOD_MCP_URL``
     - Robinhood Agentic Trading endpoint.
   * - ``ROBINHOOD_REDIRECT_URI``
     - OAuth callback. Must match the URL registered with Robinhood.
   * - ``ROBINHOOD_OAUTH_STATE_SECRET``
     - Signs the OAuth ``state``. Unset generates a random per-process key,
       which is safe but breaks in-flight links across restarts and across
       multiple workers. Set it in production.
   * - ``ROBINHOOD_EXECUTE``
     - ``false`` by default. Only ``true`` lets orders reach the market.
   * - ``ROBINHOOD_MAX_ORDER_USD``
     - Per-order notional ceiling, both sides. Defaults to ``25``; an
       unparseable or non-positive value falls back to ``25`` rather than
       disabling the cap.

Generate an encryption key with:

.. code-block:: bash

   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Linked accounts are stored in the ``CONTENT_DATABASE_URL`` Postgres database
when one is configured. Leave that unset only for local development — on an
ephemeral deployment, links do not survive a redeploy.
