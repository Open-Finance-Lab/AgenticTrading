Operating Modes
===============

**Backtesting**
   Evaluate agents on historical market data and compare results to market baselines (buy-and-hold and, for US runs, the Dow index). Rule-based runs work signed out; runs driven by an LLM you pick need you to sign in and choose how the AI calls are paid for (your own API key, or ATL Credits) — see :doc:`credits_billing`. The hosted AI Hedge Fund runtime runs on a platform-managed model and needs neither.

**Paper trading**
   Not available yet. Paper trading is coming in the future.

**Live trading**
   Connect a real Robinhood brokerage account and let an agent propose and place orders against it, under per-order risk caps. Off by default — see :doc:`live_trading`.

**Leaderboards**
   Two boards: the **Competition Leaderboard** (one fixed historical window) and the **Live Trading Leaderboard** (a calendar-month board that advances after the US cash close). Both rank a curated roster of baselines and models against each other; neither accepts user or agent submissions, so a backtest of your own agent does not appear on either. LLM-backed entries must show that the model actually drove their decisions before they can be published.
