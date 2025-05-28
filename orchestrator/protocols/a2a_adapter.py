from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils import new_agent_text_message
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities


class HelloWorldAgent:
    """A simple Hello World agent implementation."""

    async def invoke(self) -> str:
        return "Hello, world!"


class HelloWorldAgentExecutor(AgentExecutor):
    """Executor that wraps HelloWorldAgent for A2A protocol."""

    def __init__(self):
        self.agent = HelloWorldAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Invoke the agent and get the reply
        result = await self.agent.invoke()
        # Enqueue a streaming text message
        event_queue.enqueue_event(
            new_agent_text_message(result, context=context)
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Cancellation not supported for this agent
        raise NotImplementedError("Cancel not supported for HelloWorldAgent")


def create_a2a_app():
    """
    Construct the A2A Starlette application serving the HelloWorldAgent.
    Returns:
        Starlette application instance
    """
    # Public skill definition
    basic_skill = AgentSkill(
        id='hello_world',
        name='Hello World',
        description='Responds with a hello message',
        tags=['hello'],
        examples=['hi', 'hello'],
    )

    # Build agent card
    public_card = AgentCard(
        name='Hello World Agent',
        description='A minimal A2A HelloWorld agent',
        url='http://localhost:9999/',
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[basic_skill],
        supportsAuthenticatedExtendedCard=False,
    )

    # Configure request handler with executor and in-memory task store
    handler = DefaultRequestHandler(
        agent_executor=HelloWorldAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # Create A2A Starlette application without extended card support
    server = A2AStarletteApplication(
        agent_card=public_card,
        http_handler=handler,
    )
    return server.build()


if __name__ == '__main__':
    import uvicorn
    app = create_a2a_app()
    uvicorn.run(app, host='0.0.0.0', port=9999)
