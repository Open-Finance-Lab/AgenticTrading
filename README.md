# FinAgent-Orchestration

**FINAGENT-ORCHESTRATION** is a modular, protocol-driven, and learning-augmented orchestration framework designed for autonomous multi-agent systems in algorithmic trading. The system provides a dynamic, DAG-based execution graph controller, layered communication protocols, and long-term memory integration to facilitate adaptive, composable, and explainable trading workflows.

## Motivation

Contemporary algorithmic trading systems are often constrained by static rule-based pipelines, which limit their ability to adapt to evolving market regimes. This framework introduces a novel agentic architecture that enables runtime reconfiguration, agent selection, and control signal propagation based on high-level strategy objectives and market feedback.

---

## 📐 System Overview

The architecture is composed of vertically layered modules reflecting a self-contained intelligent system:

1. **Raw Market Data Layer**  
   Ingests real-time or historical financial data including prices, orderbooks, macroeconomic indicators, and news streams.

2. **Memory & Learning Layer**  
   Contains a long-term memory agent (`MemoryAgent`) and an optional DRL-based policy agent that evolves orchestration DAGs over time via reward-guided structural learning.

3. **Task Execution Layer**  
   Includes specialized agent pools:  
   - `DataAgents`: Preprocessing and feature enrichment  
   - `AlphaAgents`: Signal and factor generation  
   - `RiskAgents`: Risk modeling and constraint enforcement  
   - `CostAgents`: Transaction cost estimation  
   - `ExecutionAgents`: Order execution and market interaction  
   - `AnalysisAgents`: Post-trade performance attribution

4. **Control & Orchestration Layer**  
   - `Orchestration Core`: Executes task DAGs and propagates intermediate states  
   - `DAGControllerAgent`: Constructs and optimizes the DAG topology  
   - `RegistrationBus`: Tracks agent registration, health, and metadata

---

## 🔗 Communication Protocols

Four primary inter-agent communication protocols govern system operations:

| Protocol | Role |
|----------|------|
| `MCP` (Multi-agent Control Protocol) | Task scheduling and lifecycle control |
| `ACP` (Agent Communication Protocol) | Agent feedback and status reporting |
| `A2A` (Agent-to-Agent Protocol) | Peer-to-peer DAG-executed subtask coordination |
| `ANP` (Agent Notification Protocol) | Event-driven alerts and system-wide state propagation |

---

## 📁 Project Structure
├── docs/                 Documentation and Sphinx sources
├── orchestration/        DAG Controller, Orchestration engine, Bus, Protocols
├── agents/               Modular agent pools (Alpha, Risk, Execution, etc.)
├── memory/               Memory Agent and DRL policy learner
├── config/               YAML-based system configuration
├── examples/             Strategy simulation and demo DAG runs
├── tests/                Unit and integration testing modules
├── Papers/               Whitepapers and system documentation
├── README.md             Project overview and guide
├── requirements.txt      Python dependency list
└── readthedocs.yml       Build configuration for ReadTheDocs

---

## 🧪 Use Cases

- **Research-grade multi-agent trading experiments**  
- **Live or simulated trading strategy orchestration**
- **Benchmarking RL-based DAG optimizers in financial pipelines**
- **Self-adaptive strategy composition under varying market dynamics**

---

## 📚 Documentation

Complete documentation is available in the [`docs/`](docs/) directory and online at:

📘 https://finagent-orchestration.readthedocs.io

---

## 📝 Citation

If you use this system or build upon it, please cite: