Agent API (v2)
==============

The ``/api/v2`` surface is a typed, versioned, MCP-shaped contract any agent can
target. The agent's LLM runs **client-side**: the backend serves context and
validates decisions; it never calls your model.

``/api/v2`` is the **canonical** agent-facing surface. The older
Agent–Environment Protocol (``/api/v1``, see ``docs/api/agent-environment-protocol-v1.md``)
remains available as a compatibility surface for the shipping SDK and
integrations; new agent-facing features land here.

Four canonical verbs
---------------------

+---------------------+------------------------------------------+
| Verb                | Endpoint                                 |
+=====================+==========================================+
| ``register``        | ``POST /api/v2/agents``                  |
+---------------------+------------------------------------------+
| ``get_context``     | ``GET  /api/v2/runs/{run_id}/context``   |
+---------------------+------------------------------------------+
| ``submit_decision`` | ``POST /api/v2/runs/{run_id}/decisions`` |
+---------------------+------------------------------------------+
| ``get_result``      | ``GET  /api/v2/runs/{run_id}/result``    |
+---------------------+------------------------------------------+

Auth & scopes
-------------

Authenticate with ``X-API-Key: ag_...`` (returned once at registration). The key
carries ownership and scopes (``agents:register``, ``runs:write``, ``context:read``,
``decisions:write``, ``runs:read``). Per-agent rate limits return ``429`` with
``Retry-After`` and ``X-RateLimit-*`` headers.

Context envelope
----------------

``get_context`` returns a typed envelope: ``portfolio``, ``current_holdings``,
``recent_trades``, ``top_signals``, plus an explicit ``universe`` (DJIA-30), a
``loop`` field (``lockstep`` for backtest), and a guaranteed ``news_sentiment``
slot (one aggregated entry per ticker), populated by the Agentic FinSearch
news-sentiment adapter when ``FINGPT_API_KEY`` is set and the producer has an
artifact at or before that step's date. It is left ``{}`` otherwise — the key
unset, the producer unavailable, or (the common case for historical backtests)
no artifact yet exists at or before that date (``404``). ``GET /api/v2/schema``
publishes the full schemas, error codes, and version.

Decisions & idempotency
------------------------

``submit_decision`` takes ``{idempotency_key, actions: [...]}``. Each action is
validated against the DJIA-30 universe and the trading schema; valid actions
execute, invalid ones are returned in ``rejected`` with reasons. Replaying an
``idempotency_key`` returns the original ack — no double execution. Decision
payloads are JSON-only; ``tool_calls``/``function_calls`` are rejected.

Reference client
----------------

See ``dashboard/examples/external_agent_client_v2.py`` for the full loop.
