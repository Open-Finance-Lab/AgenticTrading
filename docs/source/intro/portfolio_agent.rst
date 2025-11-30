Portfolio Agent
===============

The **Portfolio Agent** is responsible for constructing the final target portfolio. It synthesizes the predictive signals from the **Alpha Agent** with the safety constraints from the **Risk Agent** to determine optimal asset weights.

Overview
--------

*   **Role**: Asset Allocation, Weight Optimization, Constraint Management.
*   **Inputs**: Alpha Signals (Scores/Predictions), Risk Assessment, Capital.
*   **Outputs**: Target Portfolio Weights vector $W = [w_1, w_2, ..., w_n]$ where $\sum w_i = 1$ (or $\le 1$ if cash is allowed).

Construction Logic
------------------

The Portfolio Agent (``PortfolioAgent``) implements a flexible allocation framework that can be adapted via LLM instructions. Currently, it supports a **Constraint-Satisfaction** approach:

1.  **Signal Ingestion**: Receives a list of assets and their alpha scores $S_i$.
2.  **Risk Adjustment**: Checks the "Risk Level" provided by the Risk Agent to determine the maximum gross exposure $E_{max}$.
    *   **LOW Risk**: $E_{max} = 1.0$ (100% Allocation).
    *   **MODERATE Risk**: $E_{max} = 0.8$ (80% Allocation).
    *   **HIGH Risk**: $E_{max} = 0.5$ (50% Allocation / Defensive mode).
3.  **Weighting Scheme**:
    *   Selects the top $K$ assets (e.g., Top 5) based on alpha scores.
    *   Applies an **Equal Weight** or **Score-Weighted** scheme within the allocation cap.
    *   *(Extensibility)*: The architecture allows for integration of Mean-Variance Optimization (MVO) or Black-Litterman models via external tool calls.

Tools
-----

*   **``run_portfolio_pipeline`` (Macro Tool)**:
    Runs the standard end-to-end construction logic: Filter Assets -> Apply Risk Cap -> Assign Weights.

*   **``construct_portfolio_tool``**:
    Allows the agent to manually select assets and assign weights based on custom reasoning (e.g., "Overweight tech due to momentum").

*   **``submit_portfolio_tool``**:
    Finalizes the portfolio and returns the target weights to the Orchestrator for execution.

Integration with Orchestrator
-----------------------------

The Portfolio Agent sits at the convergence point of the pipeline:

.. code-block:: python

    # In Orchestrator
    
    # 1. Alpha
    alpha_signals = alpha_agent.predict(...)
    
    # 2. Risk
    risk_signals = risk_agent.assess(...)
    
    # 3. Portfolio
    portfolio = portfolio_agent.inference(
        alpha_signals=alpha_signals,
        risk_signals=risk_signals,
        total_capital=100000
    )

The output is a dictionary of target weights that is passed directly to the **Execution Agent** (for live trading) or the **Backtest Agent** (for simulation).
