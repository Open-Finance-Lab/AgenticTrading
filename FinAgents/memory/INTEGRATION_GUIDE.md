# External Memory Agent Integration Guide

## Overview

The External Memory Agent provides a unified, industrial-grade logging and event storage system for all agent pools in the FinAgent-Orchestration system. This guide explains how to integrate existing agent pools with the new External Memory Agent.

## Key Benefits

- **Unified Logging**: Single interface for all agent pools to log events and data
- **Efficient Storage**: File-based storage with automatic indexing and date partitioning
- **Flexible Querying**: Rich filtering capabilities for event retrieval
- **Real-time Processing**: Hook system for immediate event processing
- **Scalable Architecture**: Designed to handle high-volume logging scenarios
- **Future-Ready**: Extensible design for backtesting and reinforcement learning workflows

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FinAgent-Orchestration System                │
├─────────────────────────────────────────────────────────────────┤
│  Agent Pool 1     │  Agent Pool 2     │  Agent Pool N           │
│  ┌─────────────┐  │  ┌─────────────┐  │  ┌─────────────┐       │
│  │   Agents    │  │  │   Agents    │  │  │   Agents    │       │
│  └──────┬──────┘  │  └──────┬──────┘  │  └──────┬──────┘       │
│         │         │         │         │         │               │
├─────────┼─────────┼─────────┼─────────┼─────────┼───────────────┤
│         └─────────┼─────────┼─────────┼─────────┘               │
│                   │         │         │                         │
│              ┌────▼─────────▼─────────▼────┐                    │
│              │   External Memory Agent     │                    │
│              │  ┌─────────────────────────┐ │                    │
│              │  │  Event Processing       │ │                    │
│              │  │  - Logging              │ │                    │
│              │  │  - Filtering            │ │                    │
│              │  │  - Real-time Hooks      │ │                    │
│              │  └─────────────────────────┘ │                    │
│              │  ┌─────────────────────────┐ │                    │
│              │  │  Storage Backend        │ │                    │
│              │  │  - File-based Storage   │ │                    │
│              │  │  - Date Partitioning    │ │                    │
│              │  │  - Indexing             │ │                    │
│              │  └─────────────────────────┘ │                    │
│              └─────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start Integration

### 1. Basic Setup

```python
from FinAgents.memory.external_memory_interface import (
    create_memory_agent,
    EventType,
    LogLevel
)

# Create memory agent instance for your agent pool
memory_agent = create_memory_agent("your_agent_pool_storage")
```

### 2. Logging Events

```python
# Log a simple event
event_id = memory_agent.log_event(
    event_type=EventType.TRANSACTION,
    log_level=LogLevel.INFO,
    source_agent_pool="your_agent_pool_name",
    source_agent_id="your_agent_id",
    title="Event Title",
    content="Detailed event description",
    tags={"tag1", "tag2"},
    metadata={"key": "value"},
    session_id="optional_session_id"
)
```

### 3. Querying Events

```python
from FinAgents.memory.external_memory_interface import QueryFilter

# Query recent events from your pool
recent_events = memory_agent.get_recent_events(
    source_agent_pool="your_agent_pool_name",
    hours=24,
    limit=100
)

# Advanced filtering
query_filter = QueryFilter(
    event_types=[EventType.TRANSACTION, EventType.OPTIMIZATION],
    source_agent_pools=["your_agent_pool_name"],
    tags={"important"},
    limit=50
)
results = memory_agent.query_events(query_filter)
```

## Integration Examples

### Transaction Cost Agent Pool Integration

Here's how to integrate the transaction cost agent pool with the External Memory Agent:

#### 1. Update the Agent Pool Constructor

```python
# In FinAgents/agent_pools/transaction_cost_agent_pool/__init__.py

from FinAgents.memory.external_memory_interface import create_memory_agent

class TransactionCostAgentPool:
    def __init__(self, pool_config):
        # ... existing initialization ...
        
        # Initialize memory agent
        self.memory_agent = create_memory_agent("transaction_cost_pool_storage")
        self.pool_name = "transaction_cost_agent_pool"
        self.session_id = f"tc_session_{int(time.time())}"
    
    def _log_event(self, event_type, log_level, agent_id, title, content, **kwargs):
        """Helper method for consistent event logging."""
        return self.memory_agent.log_event(
            event_type=event_type,
            log_level=log_level,
            source_agent_pool=self.pool_name,
            source_agent_id=agent_id,
            title=title,
            content=content,
            session_id=self.session_id,
            **kwargs
        )
```

#### 2. Update Individual Agents

```python
# In agent implementation files

from FinAgents.memory.external_memory_interface import EventType, LogLevel

class CostOptimizer:
    def __init__(self, agent_pool):
        self.agent_pool = agent_pool
        self.agent_id = "cost_optimizer_001"
    
    def optimize_transaction_costs(self, portfolio_data):
        """Optimize transaction costs and log the process."""
        
        # Log optimization start
        self.agent_pool._log_event(
            event_type=EventType.OPTIMIZATION,
            log_level=LogLevel.INFO,
            agent_id=self.agent_id,
            title="Cost Optimization Started",
            content=f"Starting cost optimization for {len(portfolio_data)} positions",
            tags={"optimization", "start"},
            metadata={"position_count": len(portfolio_data)}
        )
        
        try:
            # Perform optimization logic
            optimization_result = self._perform_optimization(portfolio_data)
            
            # Log successful completion
            self.agent_pool._log_event(
                event_type=EventType.OPTIMIZATION,
                log_level=LogLevel.INFO,
                agent_id=self.agent_id,
                title="Cost Optimization Completed",
                content=f"Optimization completed with score: {optimization_result['score']:.4f}",
                tags={"optimization", "completed", "success"},
                metadata={
                    "optimization_score": optimization_result['score'],
                    "iterations": optimization_result['iterations'],
                    "execution_time": optimization_result['time']
                }
            )
            
            return optimization_result
            
        except Exception as e:
            # Log error
            self.agent_pool._log_event(
                event_type=EventType.ERROR,
                log_level=LogLevel.ERROR,
                agent_id=self.agent_id,
                title="Cost Optimization Failed",
                content=f"Optimization failed: {str(e)}",
                tags={"optimization", "error", "failed"},
                metadata={"error_type": type(e).__name__, "error_message": str(e)}
            )
            raise
```

#### 3. Log Transactions

```python
from FinAgents.memory.external_memory_interface import create_transaction_event

class TransactionExecutor:
    def __init__(self, agent_pool):
        self.agent_pool = agent_pool
        self.agent_id = "transaction_executor_001"
    
    def execute_transaction(self, symbol, action, quantity, price):
        """Execute transaction and log it."""
        
        try:
            # Execute the transaction
            transaction_result = self._execute_trade(symbol, action, quantity, price)
            
            # Create and log transaction event using helper function
            transaction_event = create_transaction_event(
                agent_pool=self.agent_pool.pool_name,
                agent_id=self.agent_id,
                transaction_type=action,
                symbol=symbol,
                quantity=quantity,
                price=price,
                cost=quantity * price,
                session_id=self.agent_pool.session_id
            )
            
            # Add execution details to metadata
            transaction_event['metadata'].update({
                'execution_id': transaction_result['execution_id'],
                'execution_time': transaction_result['timestamp'],
                'market_impact': transaction_result.get('market_impact', 0.0)
            })
            
            # Log using batch method for consistency
            self.agent_pool.memory_agent.log_events_batch([transaction_event])
            
            return transaction_result
            
        except Exception as e:
            # Log transaction error
            self.agent_pool._log_event(
                event_type=EventType.ERROR,
                log_level=LogLevel.ERROR,
                agent_id=self.agent_id,
                title=f"Transaction Failed: {action.upper()} {symbol}",
                content=f"Failed to execute {action} order for {quantity} shares of {symbol}: {str(e)}",
                tags={"transaction", "error", action, symbol},
                metadata={
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "price": price,
                    "error_type": type(e).__name__
                }
            )
            raise
```

### Portfolio Agent Pool Integration

```python
# Example for portfolio management agent pool

from FinAgents.memory.external_memory_interface import EventType, LogLevel

class PortfolioManager:
    def __init__(self):
        self.memory_agent = create_memory_agent("portfolio_pool_storage")
        self.pool_name = "portfolio_agent_pool"
        self.agent_id = "portfolio_manager_001"
    
    def rebalance_portfolio(self, portfolio):
        """Rebalance portfolio and log the process."""
        
        correlation_id = f"rebalance_{int(time.time())}"
        
        # Log rebalancing start
        self.memory_agent.log_event(
            event_type=EventType.PORTFOLIO_UPDATE,
            log_level=LogLevel.INFO,
            source_agent_pool=self.pool_name,
            source_agent_id=self.agent_id,
            title="Portfolio Rebalancing Started",
            content=f"Starting portfolio rebalancing for {len(portfolio.positions)} positions",
            tags={"portfolio", "rebalancing", "start"},
            metadata={
                "position_count": len(portfolio.positions),
                "total_value": portfolio.total_value,
                "target_allocation": portfolio.target_allocation
            },
            correlation_id=correlation_id
        )
        
        # Log each rebalancing step...
        # (similar pattern as transaction cost example)
```

## Best Practices

### 1. Consistent Agent Pool Naming

Use descriptive, consistent names for agent pools:
```python
# Good
"transaction_cost_agent_pool"
"portfolio_optimization_pool"
"market_data_collection_pool"

# Avoid
"tc_pool"
"pool1"
"my_agents"
```

### 2. Structured Metadata

Always include relevant metadata for future analysis:
```python
metadata = {
    "symbol": "AAPL",
    "quantity": 100,
    "price": 150.25,
    "total_cost": 15025.0,
    "execution_venue": "NYSE",
    "order_type": "MARKET",
    "timestamp": datetime.now().isoformat()
}
```

### 3. Meaningful Tags

Use tags for efficient filtering and categorization:
```python
tags = {
    "transaction",      # Event category
    "buy",             # Action type
    "AAPL",            # Symbol
    "large_order",     # Size classification
    "urgent"           # Priority level
}
```

### 4. Error Handling and Logging

Always log errors with sufficient context:
```python
try:
    result = risky_operation()
except Exception as e:
    memory_agent.log_event(
        event_type=EventType.ERROR,
        log_level=LogLevel.ERROR,
        source_agent_pool=pool_name,
        source_agent_id=agent_id,
        title=f"Operation Failed: {operation_name}",
        content=f"Detailed error description: {str(e)}",
        tags={"error", operation_name},
        metadata={
            "error_type": type(e).__name__,
            "error_message": str(e),
            "operation_context": operation_context,
            "input_parameters": input_params
        }
    )
    raise
```

### 5. Session and Correlation Management

Use session IDs and correlation IDs to track related events:
```python
class AgentPool:
    def __init__(self):
        self.session_id = f"session_{int(time.time())}"
    
    def start_workflow(self, workflow_name):
        correlation_id = f"{workflow_name}_{int(time.time())}"
        
        # All events in this workflow use the same correlation_id
        self._log_workflow_event(correlation_id, "Workflow Started")
        return correlation_id
```

## Real-time Processing Hooks

Add real-time processing capabilities:

```python
def setup_real_time_monitoring(memory_agent):
    """Set up real-time monitoring hooks."""
    
    def transaction_monitor(event):
        if event.event_type == EventType.TRANSACTION:
            cost = event.metadata.get('total_cost', 0)
            if cost > 100000:  # Large transaction threshold
                send_alert(f"Large transaction: {event.title} - ${cost:,.2f}")
    
    def error_monitor(event):
        if event.log_level == LogLevel.ERROR:
            send_error_notification(event)
    
    def batch_analyzer(events):
        transaction_volume = sum(
            e.metadata.get('total_cost', 0) 
            for e in events 
            if e.event_type == EventType.TRANSACTION
        )
        if transaction_volume > 1000000:  # High volume threshold
            send_volume_alert(f"High trading volume: ${transaction_volume:,.2f}")
    
    memory_agent.add_event_hook(transaction_monitor)
    memory_agent.add_event_hook(error_monitor)
    memory_agent.add_batch_hook(batch_analyzer)
```

## Querying and Analysis

### Basic Queries

```python
# Get all events from a specific agent pool
pool_events = memory_agent.get_recent_events(
    source_agent_pool="transaction_cost_agent_pool",
    hours=24
)

# Get all error events
error_filter = QueryFilter(
    log_levels=[LogLevel.ERROR, LogLevel.CRITICAL],
    limit=100
)
error_events = memory_agent.query_events(error_filter)

# Search for specific content
search_filter = QueryFilter(
    content_search="AAPL",
    event_types=[EventType.TRANSACTION]
)
aapl_transactions = memory_agent.query_events(search_filter)
```

### Advanced Analysis

```python
def analyze_trading_performance(memory_agent, symbol, days=7):
    """Analyze trading performance for a specific symbol."""
    
    start_time = datetime.now() - timedelta(days=days)
    
    # Get all transaction events for the symbol
    query_filter = QueryFilter(
        start_time=start_time,
        event_types=[EventType.TRANSACTION],
        content_search=symbol,
        limit=1000
    )
    
    transactions = memory_agent.query_events(query_filter)
    
    # Analyze the transactions
    total_volume = 0
    total_cost = 0
    buy_count = 0
    sell_count = 0
    
    for event in transactions.events:
        metadata = event.metadata
        if metadata.get('symbol') == symbol:
            total_volume += metadata.get('quantity', 0)
            total_cost += metadata.get('total_cost', 0)
            
            if metadata.get('transaction_type') == 'buy':
                buy_count += 1
            else:
                sell_count += 1
    
    return {
        'symbol': symbol,
        'period_days': days,
        'total_transactions': len(transactions.events),
        'buy_transactions': buy_count,
        'sell_transactions': sell_count,
        'total_volume': total_volume,
        'total_cost': total_cost,
        'average_price': total_cost / total_volume if total_volume > 0 else 0
    }
```

## Migration from Existing Memory Systems

### 1. Gradual Migration

Start by running both systems in parallel:

```python
class TransactionCostAgentPool:
    def __init__(self, config):
        # Keep existing memory system temporarily
        self.old_memory_bridge = MemoryBridge(config)
        
        # Add new external memory agent
        self.memory_agent = create_memory_agent("transaction_cost_pool_storage")
        
        # Migration flag
        self.use_new_memory = config.get('use_new_memory', False)
    
    def log_transaction(self, transaction_data):
        if self.use_new_memory:
            # Use new memory agent
            self._log_to_new_memory(transaction_data)
        else:
            # Use old system
            self.old_memory_bridge.log_transaction(transaction_data)
            
            # Also log to new system for comparison
            self._log_to_new_memory(transaction_data)
```

### 2. Data Migration Script

```python
def migrate_existing_data(old_storage_path, memory_agent):
    """Migrate data from old storage format to new External Memory Agent."""
    
    # Read existing data
    with open(old_storage_path, 'r') as f:
        old_data = json.load(f)
    
    # Convert to new format
    events_to_migrate = []
    
    for old_event in old_data:
        new_event = {
            'event_type': map_old_type_to_new(old_event['type']),
            'log_level': LogLevel.INFO.value,
            'source_agent_pool': old_event.get('agent_pool', 'unknown'),
            'source_agent_id': old_event.get('agent_id', 'unknown'),
            'title': old_event.get('title', 'Migrated Event'),
            'content': old_event.get('content', ''),
            'tags': old_event.get('tags', []),
            'metadata': old_event.get('metadata', {})
        }
        events_to_migrate.append(new_event)
    
    # Batch migrate
    memory_agent.log_events_batch(events_to_migrate)
    print(f"Migrated {len(events_to_migrate)} events")
```

## Testing Integration

Create comprehensive tests for your integration:

```python
import unittest
from FinAgents.memory.external_memory_interface import *

class TestTransactionCostPoolIntegration(unittest.TestCase):
    def setUp(self):
        self.memory_agent = create_memory_agent("test_storage")
        self.pool_name = "transaction_cost_agent_pool"
    
    def test_transaction_logging(self):
        """Test transaction event logging."""
        event_id = self.memory_agent.log_event(
            event_type=EventType.TRANSACTION,
            log_level=LogLevel.INFO,
            source_agent_pool=self.pool_name,
            source_agent_id="test_agent",
            title="Test Transaction",
            content="Test transaction content",
            tags={"test", "transaction"},
            metadata={"symbol": "TEST", "quantity": 100}
        )
        
        self.assertIsNotNone(event_id)
        
        # Verify event can be retrieved
        events = self.memory_agent.get_recent_events(
            source_agent_pool=self.pool_name,
            hours=1
        )
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].title, "Test Transaction")
    
    def test_batch_logging(self):
        """Test batch event logging."""
        events_data = [
            create_transaction_event(
                self.pool_name, "test_agent", "buy", "AAPL", 100, 150.0, 15000.0
            ),
            create_transaction_event(
                self.pool_name, "test_agent", "sell", "GOOGL", 50, 2750.0, 137500.0
            )
        ]
        
        event_ids = self.memory_agent.log_events_batch(events_data)
        self.assertEqual(len(event_ids), 2)
    
    def test_error_handling(self):
        """Test error event logging."""
        self.memory_agent.log_event(
            event_type=EventType.ERROR,
            log_level=LogLevel.ERROR,
            source_agent_pool=self.pool_name,
            source_agent_id="test_agent",
            title="Test Error",
            content="Test error message",
            tags={"test", "error"}
        )
        
        # Query for error events
        error_filter = QueryFilter(
            event_types=[EventType.ERROR],
            source_agent_pools=[self.pool_name]
        )
        
        results = self.memory_agent.query_events(error_filter)
        self.assertEqual(len(results.events), 1)
        self.assertEqual(results.events[0].log_level, LogLevel.ERROR)

if __name__ == '__main__':
    unittest.main()
```

## Performance Considerations

### 1. Batch Operations

Use batch operations for high-volume scenarios:

```python
# Instead of individual logging
for transaction in transactions:
    memory_agent.log_event(...)  # Inefficient for large volumes

# Use batch logging
batch_events = [
    create_transaction_event(...) for transaction in transactions
]
memory_agent.log_events_batch(batch_events)  # More efficient
```

### 2. Asynchronous Logging

For high-frequency scenarios, consider asynchronous logging:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncMemoryLogger:
    def __init__(self, memory_agent):
        self.memory_agent = memory_agent
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pending_events = []
        self.batch_size = 100
    
    async def log_event_async(self, **kwargs):
        """Log event asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.memory_agent.log_event, 
            **kwargs
        )
    
    def add_to_batch(self, event_data):
        """Add event to batch for later processing."""
        self.pending_events.append(event_data)
        
        if len(self.pending_events) >= self.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """Flush pending events in batch."""
        if self.pending_events:
            self.memory_agent.log_events_batch(self.pending_events)
            self.pending_events.clear()
```

## Monitoring and Maintenance

### 1. Storage Monitoring

Monitor storage usage and performance:

```python
def monitor_memory_agent(memory_agent):
    """Monitor memory agent health and performance."""
    
    stats = memory_agent.get_statistics()
    
    # Check storage size
    storage_stats = stats['storage_stats']
    total_events = storage_stats['total_events']
    
    # Alert if storage is growing too fast
    if total_events > 1000000:  # 1M events threshold
        send_alert(f"Memory storage has {total_events:,} events")
    
    # Check error rates
    agent_stats = stats['agent_stats']
    error_rate = agent_stats['errors'] / max(agent_stats['events_stored'], 1)
    
    if error_rate > 0.01:  # 1% error rate threshold
        send_alert(f"High error rate in memory agent: {error_rate:.2%}")
```

### 2. Cleanup and Archival

Implement cleanup policies:

```python
def cleanup_old_events(memory_agent, days_to_keep=90):
    """Clean up events older than specified days."""
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    # This would require extending the storage backend with cleanup methods
    # For now, manual file cleanup can be performed on the storage directory
    storage_dir = Path("memory_storage")
    
    for file_path in storage_dir.glob("events_*.json"):
        # Extract date from filename
        date_str = file_path.stem.replace("events_", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff_date:
                print(f"Archiving old file: {file_path}")
                # Move to archive or delete
                archive_dir = storage_dir / "archive"
                archive_dir.mkdir(exist_ok=True)
                file_path.rename(archive_dir / file_path.name)
        except ValueError:
            continue
```

## Conclusion

The External Memory Agent provides a robust, scalable foundation for unified logging across all agent pools in the FinAgent-Orchestration system. By following this integration guide, you can:

1. **Standardize Logging**: Ensure consistent event logging across all agent pools
2. **Enable Rich Analysis**: Support complex queries and analysis workflows  
3. **Prepare for ML/RL**: Create the data foundation for future backtesting and reinforcement learning
4. **Monitor System Health**: Implement comprehensive monitoring and alerting
5. **Scale Efficiently**: Handle high-volume logging scenarios with batch operations

The modular design allows for gradual migration and easy extension for future requirements. Start with basic integration and gradually add advanced features like real-time hooks and performance monitoring as needed.
