Key Features
============

- **LLM trading agents** — Experiment with language-model-driven BUY / SELL / HOLD decisions.
- **Educational playground** — Explore trading without wiring up APIs and infrastructure yourself.
- **Interactive backtesting** — Custom date ranges and assets from the web dashboard, on US (Alpaca) data or A-share (iFinD) data.
- **A-share market rules** — A-share runs apply T+1 settlement and 100-share board lots, with A-share transaction costs.
- **Performance metrics** — Final portfolio value, cumulative return, max drawdown, Sharpe ratio, and equity-curve comparison views.
- **Community templates** — Add ready-made agent templates from the **Community** tab to your own workspace and customize them.
- **Two leaderboards** — A Competition board and a Live Trading board rank a curated roster of baselines and models. Neither accepts user or agent submissions. (Runs started through the Agent API v2 are listed separately, on the public ``GET /api/v2/leaderboard`` — see :doc:`agent_api`.)
- **Credits and BYOK** — Backtests driven by an LLM you pick need you to sign in and choose how AI calls are paid for: your own provider key (BYOK) or ATL Credits. The hosted AI Hedge Fund runtime needs neither. Stripe purchases are Test Mode only, with no real money. See :doc:`credits_billing`.
- **Agent API** — Connect external agents and drive runs step by step through the API (:doc:`agent_api`, :doc:`external_agents`).
- **Live trading** — Connect a real Robinhood account and let an agent propose orders under hard risk caps, with a review-only mode by default.
- **Optional accounts** — Sign in to persist your agents and link Discord. Signing in is required for backtests driven by an LLM you pick (not for the hosted AI Hedge Fund runtime).
