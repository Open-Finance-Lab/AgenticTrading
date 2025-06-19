from agent_pools.data_agent_pool.registry import BaseAgent
from agent_pools.data_agent_pool.schema.news_schema import NewsAPIConfig

class NewsAPIAgent(BaseAgent):
    def __init__(self, config: NewsAPIConfig):
        super().__init__(config.model_dump())

    def fetch_headlines(self, topic: str) -> dict:
        return {
            "topic": topic,
            "headlines": ["Market surges amid optimism", "Crypto rebounds strongly"]
        }