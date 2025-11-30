=============================
Overview
=============================

.. figure:: FinProtocol.png
   :align: center
   :width: 90%

   System architecture of the protocol-driven multi-agent orchestration framework.

This project introduces a protocol-driven, modular architecture for orchestrating multi-agent systems in algorithmic trading. The design integrates dynamically generated execution graphs, functionally specialized agent pools, and memory-based feedback to support adaptive, interpretable, and scalable decision-making.

Inspired by agentic planning frameworks and distributed control theory, the system decomposes the trading lifecycle into reusable components—ranging from data ingestion and alpha modeling to portfolio construction and post-trade attribution. These agents are composed at runtime via a DAG Planner Agent and coordinated by a central Orchestrator.

Communication across the system is mediated by two protocol layers:

- **MCP** (Multi-agent Control Protocol): task dispatch and execution management
- **A2A** (Agent-to-Agent Protocol): peer-level data flow along DAG edges

By enabling dynamic task routing and memory-informed learning, the framework provides a foundation for research and deployment of intelligent, multi-agent financial systems capable of long-term strategic adaptation.

The **FinAgent Orchestration System** is a protocol-driven, modular, and autonomous agent-based framework designed for the next generation of algorithmic trading infrastructures. It leverages a **multi-agent architecture** augmented by layered communication protocols, dynamic orchestration, and memory-augmented decision processes. The system is engineered to support **self-organizing, adaptive, and explainable financial strategies** across heterogeneous agent pools.

Design Philosophy
-----------------

Our architecture is guided by foundational principles derived from both **systems engineering** and **philosophical frameworks of coordination**, particularly:

- **Emergent Order**: Inspired by self-organizing phenomena in economic systems and swarm intelligence, FinAgent promotes bottom-up coordination among agents, where global strategy emerges from local interactions without a centralized controller.
- **Subsidiarity**: Following the principle that decisions should be handled at the most immediate level capable of resolving them, each agent in the system maintains autonomy over local tasks, escalating only when higher-level orchestration is necessary.
- **Distributed Causality**: Rather than attributing behavior to individual agents, the system treats causality as co-constructed, allowing context and intent to be propagated and shared dynamically.
- **Protocol Duality**: FinAgent supports dual communication protocols—Multi-agent Control Protocol (MCP) for orchestrated scheduling, and Agent-to-Agent Protocol (A2A) for peer-level cooperation—enabling both centralized task dispatch and decentralized collaboration.

This architectural philosophy allows for **scalable cooperation**, **resilient decision making**, and **context-sensitive planning**, which are essential characteristics in volatile, data-intensive financial environments.
