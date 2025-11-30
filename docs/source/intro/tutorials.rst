Tutorials & Instructions
========================

This section provides step-by-step instructions for reproducing the results of the FinAgent Orchestration system. These tutorials cover the full lifecycle from training agents to running out-of-sample inference, serving as a reference implementation for researchers and practitioners.

Prerequisites
-------------

Ensure you have the necessary environment variables set:

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."
   export ALPACA_API_KEY="..."     # Optional, for real data
   export ALPACA_SECRET_KEY="..."  # Optional, for real data

Phase 1: Training with Prompt Optimization
------------------------------------------

The primary objective of this phase is to "train" the agents not by updating neural network weights, but by evolving their system prompts based on backtest performance metrics.

**Script**: ``examples/1_backtest_training.ipynb``

**Steps:**

1.  **Initialize the Orchestrator**:
    Load the ``Orchestrator`` class, which sets up the Alpha, Risk, and Portfolio agents.

    .. code-block:: python

       from FinAgents.orchestrator_demo.orchestrator import Orchestrator
       orchestrator = Orchestrator()

2.  **Define the Training Loop**:
    Iterate through historical years (e.g., 2010-2023). For each year:
    
    *   Run a standard backtest pipeline.
    *   Evaluate performance (Sharpe Ratio).
    *   If performance is below threshold (e.g., Sharpe < 1.0), invoke the **Meta-Agent**.

3.  **Optimize Prompts**:
    The Meta-Agent analyzes the failure and rewrites the instructions for the underperforming agent.

    .. code-block:: python

       new_instruction = orchestrator.optimize_agent_prompts(
           agent_name="Alpha",
           performance_metric="Sharpe Ratio",
           current_value=current_sharpe,
           target_value=1.5
       )

4.  **Save Results**:
    At the end of the training period, save the evolved prompts to a JSON file (e.g., ``optimized_prompts.json``). These "frozen" prompts represent the learned strategy.

Phase 2: Out-of-Sample Inference (World Model)
----------------------------------------------

The goal of this phase is to validate the agents on unseen data (e.g., 2024-2025) using a rigorous "World Model" approach to prevent look-ahead bias.

**Script**: ``examples/2_out_of_sample_inference.ipynb``

**Steps:**

1.  **Load Optimized Prompts**:
    Inject the instructions saved in Phase 1 into the agent instances.

    .. code-block:: python

       with open("optimized_prompts.json", "r") as f:
           prompts = json.load(f)
       orchestrator.alpha_agent.agent.instructions = prompts["Alpha"]

2.  **Configure the World Model**:
    Set up a rolling window simulation (e.g., weekly rebalancing). This ensures the agent only sees data up to time $T$ when making decisions for time $T+1$.

3.  **Run Inference**:
    Execute the ``run_inference_rolling_week`` method.

    .. code-block:: python

       symbol = ['AAPL', 'MSFT', 'NVDA']
       result = orchestrator.run_inference_rolling_week(
           symbol, 
           start_date="2024-01-01", 
           end_date="2025-01-01"
       )

4.  **Analyze Results**:
    Review the Out-of-Sample performance metrics (Total Return, Sharpe, Max Drawdown). Since prompts were frozen, this performance is indicative of how the system would behave in production.

Agent Integration Demos
-----------------------

Specific demos for individual agent pools can be found in the ``FinAgents/agent_pools/`` directory. These demos serve as unit tests and standalone examples for each component.

*   **Alpha Agent Demo**: ``FinAgents/agent_pools/alpha_agent_demo/``
    
    *   **Features**: Qlib Factor Construction, Technical Indicators (RSI, MACD), ML Inference (LightGBM, Linear).
    *   **Key Files**:
        *   ``README.md``: Usage guide and architecture overview.
        *   ``example_usage.py``: Basic signal generation pipeline.
        *   ``test_with_real_data.py``: Validation using real market data.

*   **Risk Agent Demo**: ``FinAgents/agent_pools/risk_agent_demo/``
    
    *   **Features**: Volatility, VaR, CVaR, Max Drawdown calculation, and LLM-based risk assessment.
    *   **Key Files**:
        *   ``README.md``: Metrics explanation and output format.
        *   ``risk_signal_agent.py``: Core agent logic.
        *   ``test_with_real_data.py``: comprehensive risk analysis test.

*   **Execution Agent Demo**: ``FinAgents/agent_pools/execution_agent_demo/``
    
    *   **Features**: Alpaca Paper Trading integration, Batch Order execution, Mock Mode fallback.
    *   **Key Files**:
        *   ``README.md``: Setup instructions for Alpaca API.
        *   ``execution_agent.py``: The execution agent implementation.
        *   ``run_live_trading.py``: A standalone script simulating a live trading loop (Data -> Signal -> Portfolio -> Execution).

For detailed running instructions and dependency requirements, please refer to the ``README.md`` file within each respective directory.
