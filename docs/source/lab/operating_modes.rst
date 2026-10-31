Operating Modes
===============

**Backtesting**
   Evaluate agents on historical market data and compare results to market baselines (buy-and-hold and, for US runs, the Dow index). Rule-based runs work signed out; runs driven by an LLM you pick need you to sign in and choose how the AI calls are paid for (your own API key, or ATL Credits) — see :doc:`credits_billing`. The hosted AI Hedge Fund runtime runs on a platform-managed model: it needs you to sign in and is always paid for with ATL Credits.

**Paper trading**
   Not available. The dashboard shows the **Paper Trading** tab greyed out, and the Lab does not run agents against a broker's paper account.

**Live trading**
   Not available. Connecting a real Robinhood brokerage account is switched off: the server refuses to start a link or place a live run. An account linked before then can still disconnect from the agent editor. :doc:`live_trading` describes the feature as it was built.

**Leaderboards**
   Two boards: the **Competition Leaderboard** (one fixed historical window) and the **Live Trading Leaderboard** (a calendar-month board that advances after the US cash close). Both rank a curated roster of baselines and models against each other; neither accepts user or agent submissions, so a backtest of your own agent does not appear on either. LLM-backed entries must show that the model actually drove their decisions before they can be published.
