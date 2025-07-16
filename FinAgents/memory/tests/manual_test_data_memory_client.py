import asyncio
from FinAgents.memory.data_memory_client import DataMemoryClient

async def main():
    client = DataMemoryClient(agent_id="manual_test_data_agent")
    dummy_details = {"field": "value"}
    dummy_keywords = ["test", "memory"]
    dummy_summary = "Test summary"

    print("Testing store_event...")
    result = await client.store_event(
        event_type="DATA_EVENT",
        summary=dummy_summary,
        keywords=dummy_keywords,
        details=dummy_details,
        log_level="INFO"
    )
    print("store_event result:", result)

    print("Testing store_error...")
    result = await client.store_error(
        summary="Error summary",
        details=dummy_details
    )
    print("store_error result:", result)

    print("Testing store_action...")
    result = await client.store_action(
        summary="Action summary",
        details=dummy_details
    )
    print("store_action result:", result)

    print("Testing retrieve_events...")
    result = await client.retrieve_events(search_query="test", limit=5)
    print("retrieve_events result:", result)

    print("Testing retrieve_expanded...")
    result = await client.retrieve_expanded(search_query="test", limit=5)
    print("retrieve_expanded result:", result)

    print("Testing filter_events...")
    result = await client.filter_events(filters={"agent_id": "manual_test_data_agent"}, limit=5)
    print("filter_events result:", result)

if __name__ == "__main__":
    asyncio.run(main()) 