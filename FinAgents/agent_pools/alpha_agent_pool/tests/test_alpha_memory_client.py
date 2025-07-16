import asyncio
import uuid
from datetime import datetime, timedelta
from FinAgents.agent_pools.alpha_agent_pool.alpha_memory_client import AlphaMemoryClient

async def main():
    # Use a unique agent_id for test isolation
    agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
    memory_client = AlphaMemoryClient(agent_id=agent_id)

    print("Storing a transaction event...")
    txn_result = await memory_client.store_event(
        event_type="TRANSACTION",
        summary="Test BUY order for MSFT",
        keywords=["transaction", "BUY", "MSFT", agent_id],
        details={"symbol": "MSFT", "type": "BUY", "amount": 10, "status": "SUCCESS", "timestamp": datetime.now().isoformat()},
        log_level="INFO"
    )
    print("Transaction store result:", txn_result)

    print("Storing an error event...")
    error_result = await memory_client.store_error(
        summary="Test order failed",
        details={"symbol": "AAPL", "type": "SELL", "error": "Insufficient funds", "timestamp": datetime.now().isoformat()},
        keywords=["error", "AAPL", agent_id]
    )
    print("Error store result:", error_result)

    print("Storing a generic agent action...")
    action_result = await memory_client.store_action(
        summary="Test agent performed a rebalance",
        details={"action": "rebalance", "timestamp": datetime.now().isoformat()},
        keywords=["action", "rebalance", agent_id]
    )
    print("Action store result:", action_result)

    print("Retrieving recent events (should include above)...")
    events = await memory_client.retrieve_events(agent_id, limit=5)
    print("Retrieved events:", events)

    print("Filtering for TRANSACTION events...")
    filters = {
        "event_types": ["TRANSACTION"],
        "start_time": (datetime.now() - timedelta(days=1)).isoformat(),
        "agent_id": agent_id
    }
    filtered = await memory_client.filter_events(filters, limit=5)
    print("Filtered events:", filtered)

if __name__ == "__main__":
    asyncio.run(main())
