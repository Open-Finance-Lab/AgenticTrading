Risk Agent
==========

The **Risk Agent** acts as the guardian of the trading system. It evaluates market conditions and portfolio exposures to ensure that trading activities remain within safety bounds. It operates downstream of the Alpha Agent and upstream of the Portfolio Agent.

Overview
--------

*   **Role**: Risk Assessment, Volatility Calculation, Gatekeeping.
*   **Inputs**: Market Data, Portfolio State.
*   **Outputs**: Risk Signals (HIGH/LOW), Risk Score, Approved/Rejected Signals.

Architecture
------------

The Risk Agent (``RiskSignalAgent``) is built using the **ReAct** (Reasoning + Acting) pattern, utilizing the OpenAI Agent SDK. It supports two distinct execution paths:

1.  **Deterministic Evaluation ("Fast Path")**: A direct, code-driven pipeline for standard risk checks (Volatility, VaR). This ensures speed and reproducibility for critical safety checks.
2.  **Semantic Reasoning ("Custom Path")**: An LLM-driven exploration where the agent can choose specific tools to analyze unique market situations, interpret news sentiment, or handle edge cases not covered by the standard pipeline.

Tools and Capabilities
----------------------

The agent is equipped with specific tools to perform quantitative analysis:

*   **``run_risk_pipeline`` (Macro Tool)**:
    Executes the standard suite of risk checks in one go.
    
    *   **Volatility**: Annualized standard deviation of returns.
    *   **Value at Risk (VaR)**:
        
        .. math::
           VaR_{\alpha}(X) = \inf \{ x \in \mathbb{R} : F_X(x) \ge \alpha \}
        
        Calculated at 95% confidence.
    *   **Risk Level Aggregation**: Combines metrics into a discrete level (HIGH, MODERATE, LOW).

*   **``calculate_volatility_tool``**:
    Computes rolling volatility for a specific window (default 20 days).

*   **``submit_risk_assessment_tool``**:
    Finalizes the risk assessment and pushes the results to the shared context.

Logic and Scoring
-----------------

The internal logic (``_generate_risk_signals``) converts raw metrics into actionable signals based on predefined thresholds:

.. code-block:: python

    if vol > 0.3:
        signals['volatility'] = 'HIGH'
        score += 0.5
    
    if var < -0.03: # Loss exceeds 3%
        signals['var'] = 'HIGH'
        score += 0.5
        
    overall_level = "HIGH" if score >= 0.5 else "LOW"

This score is then used by the **Portfolio Agent** to adjust position sizing (e.g., reducing allocation by 50% if Risk is HIGH).

Usage
-----

**Standalone Usage:**

.. code-block:: python

    risk_agent = RiskSignalAgent(model="gpt-4o")
    result = risk_agent.generate_risk_signals_from_data(dataframe)

**Agentic Usage:**

The Orchestrator can invoke the Risk Agent via natural language:

.. code-block:: text

    "Assess the risk of the current market data provided in context."

The agent will then decide whether to run the full pipeline or investigate specific metrics based on the prompt.
