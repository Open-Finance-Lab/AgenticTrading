Training and Testing Workflow
=============================

The FinAgent Orchestration framework employs a rigorous two-phase workflow to ensure robust agent performance: **Training (Backtest with Optimization)** and **Testing (Out-of-Sample Inference)**. This approach minimizes overfitting and validates the agent's adaptability to unseen market conditions.

Phase 1: Training (Backtest with Prompt Optimization)
-----------------------------------------------------

The primary objective of the training phase is to iteratively refine the agents' instructions (prompts) based on historical performance metrics. Unlike traditional machine learning which updates model weights, this framework optimizes the *context* and *instructions* provided to the LLM-based agents, effectively performing **Prompt Engineering via Reinforcement**.

**Process Flow:**

.. code-block:: text

    +-------------------+       +--------------------------+       +---------------------+
    |                   |       |                          |       |                     |
    |  Define Period    |------>|  Run Backtest (1 Year)   |------>|  Evaluate Metrics   |
    |  (e.g., 2010)     |       |  (Alpha -> Risk -> Port) |       |  (Sharpe, Return)   |
    |                   |       |                          |       |                     |
    +-------------------+       +--------------------------+       +----------+----------+
                                                                              |
                                                                              v
                                                                    +---------------------+
                                                                    |  Check Threshold    |
                                                                    |  (e.g., Sharpe > 1) |
                                                                    +----------+----------+
                                                                               |
                                      No (Optimize) <--------------------------+-----------------> Yes (Next Year)
                                            |
                                            v
                                  +--------------------------+
                                  |  Meta-Agent Optimizer    |
                                  |  (Refines Instructions)  |
                                  +--------------------------+
                                            |
                                            v
                                  +--------------------------+
                                  |  Update Agent Prompts    |
                                  |  & Save for Next Loop    |
                                  +--------------------------+

**Optimization Formulation:**

Let $P_t$ be the set of prompts at iteration $t$, and $M(P_t)$ be the performance metric (e.g., Sharpe Ratio) resulting from the backtest. The Meta-Agent aims to find a prompt update $\Delta P$ such that:

.. math::

   P_{t+1} = P_t + \Delta P \quad \text{s.t.} \quad \mathbb{E}[M(P_{t+1})] > M(P_t)

This is achieved by providing the Meta-Agent with the *history of performance*, the *current prompt*, and the *failure modes* (e.g., "High drawdown in Q3").

**Key Steps:**

1.  **Yearly Simulation**: The orchestrator runs the pipeline year by year (e.g., 2010 to 2023).
2.  **Performance Evaluation**: At the end of each year, metrics like Sharpe Ratio and Total Return are calculated.
3.  **Meta-Agent Optimization**: If performance falls below a defined threshold (e.g., Sharpe < 1.0), a "Meta-Agent" (another LLM instance) analyzes the results and the current instructions. It generates refined instructions to improve the agent's decision-making logic (e.g., "Focus more on volatility" or "Be more aggressive in uptrends").
4.  **Persistence**: The optimized prompts are carried forward to the next year and saved at the end of the training phase.

Phase 2: Testing (Out-of-Sample Inference)
------------------------------------------

The testing phase validates the optimized agents on a strictly held-out dataset (e.g., 2024-2025). This phase mimics a real-world production environment using a **World Model** approach.

**World Model Inference:**

To prevent **look-ahead bias**, the inference engine reveals market data step-by-step (e.g., daily or weekly).

.. code-block:: text

    +-----------------------+       +-----------------------+       +-----------------------+
    |                       |       |                       |       |                       |
    |  Load Optimized       |------>|  Initialize           |------>|  Fetch Initial Data   |
    |  Prompts              |       |  Orchestrator         |       |  (e.g., until T-1)    |
    |                       |       |                       |       |                       |
    +-----------------------+       +-----------------------+       +-----------+-----------+
                                                                                |
           +--------------------------------------------------------------------+
           |
           v
    +-----------------------+       +-----------------------+       +-----------------------+
    |  Step T (Today)       |       |                       |       |                       |
    |  - Reveal Data(T)     |------>|  Agents Generate      |------>|  Execute & Record     |
    |  - No Future Data     |       |  Signals & Orders     |       |  Results              |
    |                       |       |                       |       |                       |
    +-----------------------+       +-----------------------+       +-----------+-----------+
           ^                                                                    |
           |                                                                    |
           +---------------------------(Increment T)----------------------------+

**Key Features:**

*   **Zero Look-Ahead Bias**: Agents only access data available up to the current simulation timestamp $T$.
*   **Frozen Prompts**: Instructions are not updated during this phase; the agents must rely on the generalized strategies learned during training.
*   **Rolling Window**: Data is often processed in rolling windows (e.g., lookback 20 days) to simulate continuous operation.

Data Leakage Prevention & Safety Protocols
------------------------------------------

To ensure the validity of backtest results and prevent common quantitative research pitfalls, the framework enforces strict protocols on data access and agent communication.

Context Protocols
^^^^^^^^^^^^^^^^^

The orchestration system uses structured context messages to pass information between agents. All contexts are serialized as JSON and follow a strict schema:

.. math::

   C = \{ \text{task\_id}, \text{agent\_role}, \text{run\_mode}, \text{time\_window}, \text{universe}, \text{inputs}, \text{tool\_outputs}, \text{diagnostics}, \text{uuid} \}

Crucially, the context message **excludes**:

*   Any **raw price or return series** from the test period.
*   Any **labels or targets** from future timestamps ($t > T$).
*   Any **optimization objective** tied directly to the evaluation window (e.g., Sharpe ratio on the test set).

Numerical arrays (like price history) are not sent directly in the prompt. Instead, they are stored in data files (e.g., Parquet/CSV) and referred to by **identifiers** (data paths or dataset IDs). 

Agent Access Control
^^^^^^^^^^^^^^^^^^^^

Specific constraints are applied to each agent role to prevent information leakage:

*   **Alpha/Risk/Portfolio Agents**: These agents **never** receive evaluation-window returns, prices, or labels. They operate solely on "admissible" data (data available up to time $T$).
*   **Backtest Agent**: This is the **only** component allowed to access evaluation-window returns and realized P&L. It computes aggregated metrics (Vol, Sharpe, MaxDD) and returns them as summaries, never exposing per-timestamp P&L to other LLM agents.

Memory Integration & UUIDs
^^^^^^^^^^^^^^^^^^^^^^^^^^

The **Memory Agent** stores long-term states indexed by deterministic UUIDs to ensure reproducibility and isolation between training and testing environments.

.. math::

   \text{UUID} = \text{SHA256}(\text{role} \parallel \text{task} \parallel \text{params} \parallel \text{time})

This UUID design supports:

1.  **Immutability**: Memory entries are referred to only by their hash ID.
2.  **Isolation**: Training and evaluation memories use separate UUID namespaces to avoid mixing information.
3.  **Safe Retrieval**: Downstream agents query by UUID and receive only summarized metadata, not raw test-set values.

By enforcing these protocols, the framework guarantees that the "intelligence" exhibited by the agents is due to learned heuristics and reasoning, not access to future data.
