from typing import Dict, Any
from langchain.tools import BaseTool
from ..graphs.market_workflow import MarketDataWorkflow, MarketDataState

class MarketAnalysisTool(BaseTool):
    """
    Integrated market analysis tool with LangGraph workflow.
    
    Capabilities:
    1. Natural language query processing
    2. Automated data retrieval
    3. Comprehensive market analysis
    4. Insight generation
    
    Implementation:
    - LLM-driven intent analysis
    - Graph-based workflow execution
    - Structured response generation
    """
    
    name = "market_analysis"
    description = "Comprehensive market data analysis and insights generation"
    
    def __init__(self, agent: PolygonAgent):
        """Initialize with market data agent"""
        super().__init__()
        self.workflow = MarketDataWorkflow(agent)
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        """
        Execute market analysis workflow.
        
        Process:
        1. State initialization
        2. Workflow execution
        3. Result validation
        4. Response formatting
        
        Args:
            query: Natural language market analysis request
            
        Returns:
            Structured analysis results and insights
        """
        initial_state = MarketDataState(query=query)
        result = await self.workflow.graph.arun(initial_state)
        return result.dict()