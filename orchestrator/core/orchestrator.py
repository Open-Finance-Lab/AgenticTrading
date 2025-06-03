from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

CAPITAL = 100000

orchestrator = create_supervisor(
    agents=[], #TODO: add agent pool A2A protocol interaction here
    model=ChatOpenAI(model="gpt-4o"),
    prompt=(
        f'''
        You are the supervisor of an end-to-end automated trading system.

        You have at your disposal:
            a pool of Data Agents (responsible for ingesting, validating, and transforming raw input data)
            a pool of Alpha Agents (dedicated to predictive signal generation using statistical or model-based techniques)
            a pool of Execution Agents (handle order placement, routing, and market interface)

        Your objective is to manage a portfolio of ${CAPITAL}, please assign work to the agents.
        '''
    )
).compile()