import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    os.getenv("OPENAI_API_KEY")
    
    from orchestrator.core.orchestrator import orchestrator

    for chunk in orchestrator.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Fetch AAPL price from 2024-02-01 to 2024-02-02, 1 hour interval, plz"
                }
            ]
        }
    ):
        print(chunk)
        print("\n")

if __name__ == "__main__":
    main()
