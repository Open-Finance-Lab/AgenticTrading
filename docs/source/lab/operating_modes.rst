Operating Modes
===============

**Backtesting**
   Evaluate agents on historical market data and compare results to market baselines (buy-and-hold, index).

**Paper trading**
   Connect to an Alpaca paper account for simulated live trading: account summary, positions, recent trades, and portfolio value over time.

**Live trading**
   Connect a real Robinhood brokerage account and let an agent propose and place orders against it, under per-order risk caps. Off by default — see :doc:`live_trading`.

**Leaderboard**
   Standardized comparison of agents against buy-and-hold and index baselines over a fixed window. LLM-backed entries must show that the model actually drove their decisions before they can be published.
