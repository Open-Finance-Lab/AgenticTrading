from typing import Dict, Any, List
from langgraph.graph import Graph, Node
from pydantic import BaseModel

class MarketDataState(BaseModel):
    """
    Market data workflow state management.
    
    Tracks:
    - Query processing status
    - Intermediate results
    - Analysis progress
    - Error conditions
    """
    query: str
    plan: Optional[Dict] = None
    data: Optional[Any] = None
    analysis: Optional[Dict] = None
    errors: List[str] = []

class MarketDataWorkflow:
    """
    LangGraph workflow for market data processing.
    
    Implements:
    1. Intent Analysis Node
    2. Data Retrieval Node
    3. Analysis Generation Node
    4. Response Synthesis Node
    """
    
    def __init__(self, agent: PolygonAgent):
        """Initialize workflow with market data agent"""
        self.agent = agent
        self.graph = self._build_graph()
        
    def _build_graph(self) -> Graph:
        """
        Construct market data processing workflow.
        
        Graph Structure:
        Intent Analysis -> Data Retrieval -> Analysis -> Response
        """
        workflow = Graph()
        
        # Add processing nodes
        workflow.add_node("intent_analysis", self._analyze_intent)
        workflow.add_node("data_retrieval", self._retrieve_data)
        workflow.add_node("generate_analysis", self._generate_analysis)
        workflow.add_node("synthesize_response", self._synthesize_response)
        
        # Define workflow edges
        workflow.add_edge("intent_analysis", "data_retrieval")
        workflow.add_edge("data_retrieval", "generate_analysis")
        workflow.add_edge("generate_analysis", "synthesize_response")
        
        return workflow

    async def _analyze_intent(self, state: MarketDataState) -> MarketDataState:
        """
        Analyze user query intent.
        
        Process:
        1. Natural language understanding
        2. Query classification
        3. Parameter extraction
        4. Strategy formulation
        """
        try:
            result = await self.agent.process_intent(state.query)
            state.plan = result["execution_plan"]
        except Exception as e:
            state.errors.append(f"Intent analysis failed: {str(e)}")
        return state