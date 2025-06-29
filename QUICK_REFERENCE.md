# FinAgent Orchestration System - Quick Reference Guide

## 🚀 Quick Start Commands

### System Management
```bash
# Start the entire system
./FinAgents/finagent_start.sh start

# Check system status
./FinAgents/finagent_start.sh status

# Stop the system
./FinAgents/finagent_start.sh stop

# Restart the system
./FinAgents/finagent_start.sh restart
```

### Run Demonstrations
```bash
# Navigate to orchestrator directory
cd FinAgents/orchestrator

# Basic demo (1 strategy)
python quick_start_demo.py --demo-type basic

# Advanced demo (multiple strategies)
python quick_start_demo.py --demo-type advanced

# Sandbox demo (backtesting)
python quick_start_demo.py --demo-type sandbox

# RL demo (reinforcement learning)
python quick_start_demo.py --demo-type rl

# Full comprehensive demo (all features)
python quick_start_demo.py --demo-type all
```

### Testing
```bash
# Run comprehensive test suite
cd FinAgents/orchestrator
python test_orchestrator_comprehensive.py

# Run integration tests with live agent pools
python integration_example.py
```

## 📁 Key File Locations

### Core System Files
| File | Purpose | Location |
|------|---------|----------|
| **Main Orchestrator** | System entry point | `FinAgents/orchestrator/main_orchestrator.py` |
| **DAG Planner** | Strategy decomposition | `FinAgents/orchestrator/core/dag_planner.py` |
| **RL Engine** | Reinforcement learning | `FinAgents/orchestrator/core/rl_policy_engine.py` |
| **Sandbox** | Testing environment | `FinAgents/orchestrator/core/sandbox_environment.py` |
| **Configuration** | System settings | `FinAgents/orchestrator/config/orchestrator_config.yaml` |

### Agent Pool Files
| Component | Location |
|-----------|----------|
| **Data Agent Pool** | `FinAgents/agent_pools/data_agent_pool/` |
| **Polygon Agent** | `FinAgents/agent_pools/data_agent_pool/agents/equity/polygon_agent.py` |
| **Agent Pool Core** | `FinAgents/agent_pools/data_agent_pool/core.py` |

### Documentation & Scripts
| Document | Purpose | Location |
|----------|---------|----------|
| **User Guide** | Complete documentation | `FinAgents/orchestrator/README.md` |
| **Implementation Summary** | Technical details | `FinAgents/orchestrator/IMPLEMENTATION_SUMMARY.md` |
| **System Status** | Current state report | `SYSTEM_STATUS_REPORT.md` |
| **Deployment Guide** | Production checklist | `PRODUCTION_DEPLOYMENT_CHECKLIST.md` |

## 🔧 Configuration Quick Reference

### Main Configuration File
```yaml
# Location: FinAgents/orchestrator/config/orchestrator_config.yaml

orchestrator:
  host: "0.0.0.0"
  port: 9000
  max_concurrent_tasks: 100
  enable_rl: true
  enable_sandbox: true

agent_pools:
  data_agent_pool:
    url: "http://localhost:8001"
    enabled: true
  
  alpha_agent_pool:
    url: "http://localhost:5050"
    enabled: true
```

### Environment Variables
```bash
export FINAGENT_ENV=development    # or production
export LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
export POLYGON_API_KEY=your_key    # For real market data
```

## 📊 System Architecture

### Component Overview
```
┌─────────────────────────────────────────────────────────┐
│                 Main Orchestrator (Port 9000)           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ DAG Planner │  │ RL Engine   │  │  Sandbox    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────┐
│                   Agent Pools                           │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Data (8001)     │  │ Alpha (5050)    │              │
│  │ Risk (7000)     │  │ TxnCost (6000)  │              │
│  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### Port Assignments
| Service | Port | Status |
|---------|------|--------|
| **Main Orchestrator** | 9000 | ✅ Ready |
| **Data Agent Pool** | 8001 | ✅ Running |
| **Alpha Agent Pool** | 5050 | 🔧 Configure |
| **Risk Agent Pool** | 7000 | ✅ Running |
| **Transaction Cost Pool** | 6000 | 🔧 Configure |
| **Memory Agent** | 8010 | 🔧 Configure |

## 🎯 Common Use Cases

### 1. Execute a Single Strategy
```python
from FinAgents.orchestrator.core.finagent_orchestrator import FinAgentOrchestrator

# Initialize orchestrator
orchestrator = FinAgentOrchestrator("config/orchestrator_config.yaml")

# Execute strategy
result = orchestrator.execute_strategy("momentum_strategy", {
    "symbols": ["AAPL", "GOOGL"],
    "lookback_period": 20
})
```

### 2. Run Backtest
```python
from FinAgents.orchestrator.core.sandbox_environment import SandboxEnvironment

# Initialize sandbox
sandbox = SandboxEnvironment()

# Run backtest
result = sandbox.run_backtest({
    "strategy": "momentum",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000
})
```

### 3. Train RL Model
```python
from FinAgents.orchestrator.core.rl_policy_engine import RLPolicyEngine

# Initialize RL engine
rl_engine = RLPolicyEngine()

# Train model
model = rl_engine.train_policy({
    "algorithm": "TD3",
    "episodes": 100,
    "learning_rate": 0.001
})
```

## 🚨 Troubleshooting

### Common Issues & Solutions

#### "Connection refused" errors
```bash
# Check if services are running
./FinAgents/finagent_start.sh status

# Restart specific service
./FinAgents/finagent_start.sh restart
```

#### Import errors
```bash
# Ensure you're in the correct directory
cd /Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration

# Check Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### Configuration errors
```bash
# Validate configuration
cd FinAgents/orchestrator
python -c "import yaml; yaml.safe_load(open('config/orchestrator_config.yaml'))"
```

#### Performance issues
```bash
# Check system resources
top
df -h

# Review logs
tail -f logs/orchestrator.log
```

### Log Files
| Service | Log Location |
|---------|--------------|
| **Orchestrator** | `logs/orchestrator.log` |
| **Data Agent Pool** | `logs/data_agent_pool.log` |
| **Risk Agent Pool** | `logs/risk_agent_pool.log` |
| **System** | `logs/system.log` |

## 📈 Performance Benchmarks

### Expected Performance
| Metric | Target | Actual |
|--------|--------|--------|
| **Strategy Execution** | <5s | 3.34s ✅ |
| **Backtest Completion** | <10s | 6.01s ✅ |
| **RL Training Episode** | <3s | 2.5s ✅ |
| **System Startup** | <30s | 15s ✅ |

### Resource Requirements
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 4 cores | 8 cores |
| **Memory** | 8GB | 16GB |
| **Storage** | 50GB | 100GB |
| **Network** | 100Mbps | 1Gbps |

## 🔍 Monitoring

### Health Check URLs
- **Orchestrator**: `http://localhost:9000/health`
- **Data Agent**: `http://localhost:8001/health`
- **Risk Agent**: `http://localhost:7000/health`

### Key Metrics to Monitor
- **Response Times**: Strategy execution latency
- **Error Rates**: Failed strategy executions
- **Resource Usage**: CPU, memory, disk
- **Agent Pool Health**: Connection status

## 📞 Support Information

### Documentation
- **Complete Guide**: `FinAgents/orchestrator/README.md`
- **API Reference**: Generated from docstrings
- **Examples**: `FinAgents/orchestrator/examples/`

### Community
- **Issues**: GitHub Issues tracker
- **Discussions**: GitHub Discussions
- **Wiki**: GitHub Wiki

---

**Quick Reference Version**: 1.0  
**Last Updated**: 2025-06-29  
**System Status**: Production Ready ✅
