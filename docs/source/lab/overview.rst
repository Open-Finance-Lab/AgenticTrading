Overview
==========

Agentic Trading Lab is a research and educational environment for studying trading systems powered by large language models. It is built alongside a survey of 130+ agentic trading papers and is designed to make that research accessible: customize agents, evaluate performance, and observe behavior under realistic market constraints.

**Live app:** `agentic-trading-lab.vercel.app <https://agentic-trading-lab.vercel.app/>`_ · **Community:** `Discord <https://discord.gg/9HnQ6XDG98>`_

Goals
-----

The platform bridges alpha-seeking research and deployable trading workflows. Beyond asking whether an agent can generate profitable signals, the lab surfaces the full pipeline—data handling, agent decisions, backtesting, paper trading, execution constraints, risk, and governance.

See also :doc:`operating_modes`, :doc:`key_features`, :doc:`getting_started`, :doc:`accounts`, :doc:`marketplace`, :doc:`live_trading`, and :doc:`architecture`. For the multi-agent framework, see :doc:`../orchestration/index`.

Repository layout
-----------------

.. code-block:: text

   AgenticTrading/
   ├── dashboard/       Agentic Trading Lab web application
   │   ├── backend/     FastAPI app, SQLite, paper trading, LLM validator
   │   ├── frontend/    Dashboard (served at http://localhost:8000 when local)
   │   ├── config/      Default run IDs and date ranges
   │   ├── scripts/     CLI backtests (e.g. backtest_hourly_agent.py)
   │   └── storage/
   │       ├── data/    SQLite backtest results (backtest.db)
   │       └── backups/ Database backups
   ├── credentials/     Local only — not in git
   ├── docs/            This documentation
   └── orchestration/   FinAgent multi-agent framework (separate subsystem)
