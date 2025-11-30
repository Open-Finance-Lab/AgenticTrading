Orchestration
=============

The **Orchestrator** is the central nervous system of the FinAgent framework. It functions as a hierarchical planning and scheduling system, managing the lifecycle of the entire trading pipeline, coordinating communication between specialized agents, and ensuring that data flows correctly from one stage to the next in a DAG (Directed Acyclic Graph) structure.

Architecture
------------

The Orchestrator is designed to support two primary modes of operation:

1.  **Pipeline Mode (Hardcoded Workflow)**: A sequential execution of the standard trading loop (Data -> Alpha -> Risk -> Portfolio -> Execution/Backtest). This deterministic mode is optimized for high-throughput backtesting and training.
2.  **Agentic Mode (Autonomous)**: A flexible, LLM-driven mode where a "Manager Agent" decides the sequence of operations based on a high-level user request (e.g., "Analyze AAPL for the last 6 months and propose a portfolio"). This mode allows for dynamic branching and error recovery.

Pipeline Architecture and Scheduling
------------------------------------

The Orchestrator functions similarly to a modern computer pipeline or a DAG scheduler. It treats each agent pool as a specialized processing unit or node, coordinating the flow of data and control messages to ensure optimal execution order and resource utilization.

.. figure:: orchestration_1.pdf
   :align: center
   :width: 100%
   :alt: FinAgent Orchestration Pipeline Diagram

   **Figure 1: FinAgent Orchestration Pipeline.** This diagram illustrates the system's architecture as a directed workflow. The **Orchestrator** (central node) schedules tasks across specialized agent pools (Data, Alpha, Risk, Portfolio, Execution). Each stage operates like a pipeline stage in a CPU or a node in a DAG:
   
   1.  **Instruction Fetch (Planner)**: The Planner breaks down high-level user objectives into a sequence of executable tasks (the Plan).
   2.  **Execution (Agent Pools)**: Tasks are dispatched to specific pools. For instance, the **Alpha Pool** processes data features to generate signals, while the **Risk Pool** validates these signals against safety constraints.
   3.  **Write Back (Memory)**: Results and state changes are written to the **Memory Agent**, ensuring persistence and allowing downstream agents to access prior context.
   4.  **Control Flow**: The **MCP (Model Context Protocol)** acts as the control bus, managing task assignment, status tracking, and error handling.

This design ensures separation of concerns, where the Orchestrator manages the *control flow* (scheduling, dependencies) while the Agent Pools manage the *data flow* (computation, reasoning).

Scheduling Triggers
~~~~~~~~~~~~~~~~~~~

To accommodate diverse trading strategies—from high-frequency trading to long-term investing—the Orchestrator supports two distinct triggering mechanisms for its pipeline execution:

1.  **Event-Driven Triggering**:
    
    *   **Mechanism**: Execution is initiated by specific market events or data arrivals (e.g., a new price tick, a news alert, or a completed trade confirmation).
    *   **Use Case**: Ideal for high-frequency trading, news-based arbitrage, or reactive risk management.
    *   **Flow**: `Market Data Source` -> `Event Bus` -> `Orchestrator (Wake up)` -> `Alpha Agent` -> ...
    *   **Latency Sensitivity**: High. The system aims to minimize the time between event detection and order submission.

2.  **Time-Scheduled Triggering**:
    
    *   **Mechanism**: Execution is triggered at fixed time intervals (e.g., every minute, hourly, daily at market close).
    *   **Use Case**: Suitable for portfolio rebalancing, end-of-day reporting, or strategies relying on standard bar data (OHLCV).
    *   **Flow**: `Cron/Scheduler` -> `Orchestrator (Wake up)` -> `Fetch Batch Data` -> `Pipeline Execution`.
    *   **Consistency**: Ensures that portfolio state is synchronized with periodic benchmarks.

The **World Model Inference** (described in the Workflow section) typically uses a Time-Scheduled approach (e.g., daily or weekly rolling steps), while live deployment often employs a hybrid model where the core loop is time-scheduled but risk checks are event-driven.

Implementation Details
----------------------

The core logic resides in ``FinAgents/orchestrator_demo/orchestrator.py``.

Initialization
~~~~~~~~~~~~~~

The orchestrator initializes all sub-agents and data clients upon startup. It allows for mixing "Agent" objects (which use LLMs) and "Tools" (deterministic functions).

.. code-block:: python

    class Orchestrator:
        def __init__(self):
            # Initialize specialized agents
            self.alpha_agent = AlphaSignalAgent(model="gpt-4o")
            self.risk_agent = RiskSignalAgent(model="gpt-4o")
            self.portfolio_agent = PortfolioAgent(model="gpt-4o")
            
            # Initialize Manager Agent for Agentic Mode
            self._initialize_manager_agent()

Pipeline Mode
~~~~~~~~~~~~~

The ``run_pipeline`` method executes the standard workflow. This is used heavily during the **Training** and **Backtesting** phases.

1.  **Data Fetching**: Retrieves historical data (with support for rolling windows).
2.  **Alpha Generation**: calls ``alpha_agent.generate_signals_from_data``.
3.  **Risk Analysis**: calls ``risk_agent.generate_risk_signals_from_data``.
4.  **Backtest**: calls ``backtest_agent.run_simple_backtest_paper_interface``.

Agentic Mode
~~~~~~~~~~~~

The ``run_agentic_pipeline`` method employs the **Agent-as-a-Tool** pattern. The specialized agents (Alpha, Risk, Portfolio) are wrapped as tools and provided to a **Manager Agent**.

*   **Manager Agent Instructions**: "You are a trading strategy manager. You use the tools given to you to execute the pipeline..."
*   **Tools**: 
    *   ``ask_alpha_agent``
    *   ``ask_risk_agent``
    *   ``ask_portfolio_agent``
    *   ``fetch_market_data``

This allows for dynamic workflows, such as "Fetch data, check alpha, but if risk is high, skip portfolio construction."

Optimization Loop
-----------------

The Orchestrator also houses the **Meta-Agent Optimizer** logic (``optimize_agent_prompts``). This function:

1.  Takes an underperforming agent (e.g., Alpha Agent) and a metric (e.g., Sharpe Ratio < 1.0).
2.  Uses an LLM to analyze the current instructions and the performance gap.
3.  Rewrites the agent's instructions to improve future performance.
4.  Updates the agent in-place for the next training iteration.

.. code-block:: python

    new_instruction = orchestrator.optimize_agent_prompts(
        agent_name="Alpha",
        performance_metric="Sharpe Ratio",
        current_value=0.5,
        target_value=1.5
    )

This self-improvement loop is a key differentiator of the FinAgent framework.
