import asyncio
from starlette.testclient import TestClient
from orchestrator.protocols.a2a_adapter import create_a2a_app

async def main():
    app = create_a2a_app()
    with TestClient(app) as client:
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": "msg-1",
                        "role": "user",
                        "parts": [
                            {
                                "kind": "text",
                                "text": "hi"
                            }
                        ],
                        "skill": "hello_world"
                    }
                }
            }
        )
        print("[A2A] Response status:", response.status_code)
        print("[A2A] Response json:", response.json())

if __name__ == "__main__":
    asyncio.run(main())