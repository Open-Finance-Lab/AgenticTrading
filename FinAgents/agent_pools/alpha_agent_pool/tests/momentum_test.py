import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def main():
    """
    Test the MomentumAgent's ability to fetch price data from the shared memory and record strategy signals.
    This test will:
    1. Call generate_signal without price_list (should fetch from memory and record to momentum_signal_flow.json)
    2. Call generate_signal with a custom price_list (should also record to momentum_signal_flow.json)
    3. Optionally, print the last entry in the momentum_signal_flow.json for verification.
    """
    url = "http://localhost:5051/sse"  # URL of the MCP SSE service endpoint
    async with sse_client(url, headers={}) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            print("Session initialized.")

            # 1. Test: generate_signal without price_list (should use memory data)
            result_mem = await session.call_tool("generate_signal", {
                "request": {
                    "symbol": "AAPL"
                }
            })
            print("generate_signal (from memory) result:", result_mem)

            # 2. Test: generate_signal with explicit price_list
            result_custom = await session.call_tool("generate_signal", {
                "request": {
                    "symbol": "AAPL",
                    "price_list": [100.0, 101.5, 102.3, 104.0, 106.0]
                }
            })
            print("generate_signal (custom price_list) result:", result_custom)

            # 3. Optionally, print the last entry in momentum_signal_flow.json
            import os, json
            flow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../momentum_signal_flow.json"))
            if os.path.exists(flow_path):
                with open(flow_path, 'r') as f:
                    flow = json.load(f)
                print("Last strategy signal in flow:", flow[-1] if flow else None)
            else:
                print("momentum_signal_flow.json not found.")

if __name__ == "__main__":
    asyncio.run(main())