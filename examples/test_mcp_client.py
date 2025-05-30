import os
import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    # Parameters to start the local server
    server_params = StdioServerParameters(
        command="python",
        args=["examples/test_mcp_adapter.py"],
        env={**os.environ, "PYTHONPATH": "."},  # <--- key point
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Read static resource
            config, _ = await session.read_resource("config://app")
            print("[CLIENT] config://app =", config)

            # Read dynamic resource
            profile, _ = await session.read_resource("user://alice/profile")
            print("[CLIENT] user://alice/profile =", profile)

            # Call sync tool
            result = await session.call_tool("execute_task", {"task_name": "test_task", "arguments": {"foo": 123}})
            print("[CLIENT] execute_task =", result)

            # Call async tool
            async_result = await session.call_tool("async_tool_demo", {"param": "async_param_test"})
            print("[CLIENT] async_tool_demo =", async_result)

            # List prompts
            prompts = await session.list_prompts()
            print("[CLIENT] prompts =", prompts)

            # Call prompt
            prompt_result = await session.get_prompt("review_code", arguments={"code": "print('hello world')"})
            print("[CLIENT] review_code =", prompt_result)

if __name__ == "__main__":
    asyncio.run(main()) 