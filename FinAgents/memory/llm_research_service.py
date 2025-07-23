#!/usr/bin/env python3
"""
LLM Research Service - Memory Analysis and Research Module
=========================================================

This module provides LLM-powered research and analysis capabilities
specifically for memory-based research tasks. It isolates LLM usage
to high-level analytical functions while keeping core operations LLM-free.

Features:
1. Memory pattern analysis
2. Semantic memory search  
3. Knowledge graph insights
4. Research query processing
5. Memory relationship analysis

This service connects to the core non-LLM services for data access
but provides intelligent analysis capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import openai
    from dotenv import load_dotenv
    import os
    
    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    LLM_AVAILABLE = bool(OPENAI_API_KEY)
    
    if LLM_AVAILABLE:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
        
except ImportError:
    LLM_AVAILABLE = False
    openai_client = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMResearchService:
    """
    LLM-powered research service for memory analysis.
    
    This service provides intelligent analysis of memory data
    stored in the FinAgent system, focusing on research applications.
    """
    
    def __init__(self):
        self.llm_available = LLM_AVAILABLE
        self.client = openai_client
        self.core_service_url = "http://localhost:8000"
        
    async def analyze_memory_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in memory data using LLM.
        
        Args:
            memories: List of memory objects from database
            
        Returns:
            Analysis results including patterns, insights, and recommendations
        """
        if not self.llm_available:
            return {
                "status": "error",
                "message": "LLM service not available",
                "analysis": None
            }
        
        try:
            # Prepare memory data for analysis
            memory_text = self._prepare_memory_data(memories)
            
            # Create analysis prompt
            prompt = f"""
            Analyze the following memory data from a financial agent system:
            
            {memory_text}
            
            Please provide:
            1. Key patterns in the memory data
            2. Emerging themes or trends
            3. Potential insights for financial research
            4. Recommendations for memory organization
            5. Identified relationships between memories
            
            Format as JSON with clear sections.
            """
            
            # Call OpenAI API
            response = await self._call_openai_async(prompt)
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis": response,
                "memory_count": len(memories)
            }
            
        except Exception as e:
            logger.error(f"Memory pattern analysis failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "analysis": None
            }
    
    async def semantic_memory_search(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform semantic search and analysis of memory content.
        
        Args:
            query: Search query
            context: Optional context for the search
            
        Returns:
            Semantic search results with LLM-enhanced relevance scoring
        """
        if not self.llm_available:
            return {
                "status": "error", 
                "message": "LLM service not available",
                "results": []
            }
        
        try:
            # First, get raw memory results from core service
            raw_memories = await self._fetch_memories_from_core(query)
            
            # Enhance with LLM analysis
            enhanced_results = await self._enhance_search_results(query, raw_memories, context)
            
            return {
                "status": "success",
                "query": query,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "results": enhanced_results
            }
            
        except Exception as e:
            logger.error(f"Semantic memory search failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "results": []
            }
    
    async def generate_research_insights(self, research_topic: str, memory_scope: str = "all") -> Dict[str, Any]:
        """
        Generate research insights based on memory data analysis.
        
        Args:
            research_topic: The research question or topic
            memory_scope: Scope of memory to analyze ("all", "recent", "specific_type")
            
        Returns:
            Research insights and recommendations
        """
        if not self.llm_available:
            return {
                "status": "error",
                "message": "LLM service not available",
                "insights": None
            }
        
        try:
            # Fetch relevant memories based on scope
            memories = await self._fetch_memories_by_scope(memory_scope, research_topic)
            
            # Generate research prompt
            prompt = f"""
            Research Topic: {research_topic}
            
            Based on the following memory data from a financial agent system, 
            provide comprehensive research insights:
            
            {self._prepare_memory_data(memories)}
            
            Please provide:
            1. Key findings related to the research topic
            2. Data-driven insights and trends
            3. Potential research directions
            4. Methodological recommendations
            5. Areas requiring further investigation
            
            Format as structured research report.
            """
            
            # Try to generate insights with LLM
            try:
                insights = await self._call_openai_async(prompt)
            except Exception as llm_error:
                # If LLM fails, provide basic insights without LLM
                logger.warning(f"LLM failed, providing basic insights: {llm_error}")
                insights = self._generate_basic_insights(research_topic, memories)
            
            return {
                "status": "success",
                "research_topic": research_topic,
                "memory_scope": memory_scope,
                "timestamp": datetime.now().isoformat(),
                "insights": insights,
                "memory_count": len(memories),
                "llm_used": isinstance(insights, str) and "Basic analysis" not in insights
            }
            
        except Exception as e:
            logger.error(f"Research insight generation failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "insights": None
            }
    
    async def analyze_memory_relationships(self, memory_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze relationships between specific memories using LLM.
        
        Args:
            memory_ids: List of memory IDs to analyze
            
        Returns:
            Relationship analysis results
        """
        if not self.llm_available:
            return {
                "status": "error",
                "message": "LLM service not available",
                "relationships": []
            }
        
        try:
            # Fetch specific memories
            memories = await self._fetch_memories_by_ids(memory_ids)
            
            # Analyze relationships
            prompt = f"""
            Analyze the relationships between the following memories:
            
            {self._prepare_memory_data(memories)}
            
            Identify:
            1. Direct relationships and connections
            2. Temporal patterns
            3. Thematic similarities
            4. Causal relationships
            5. Contradictions or conflicts
            
            Provide detailed relationship analysis.
            """
            
            relationship_analysis = await self._call_openai_async(prompt)
            
            return {
                "status": "success",
                "memory_ids": memory_ids,
                "timestamp": datetime.now().isoformat(),
                "relationships": relationship_analysis
            }
            
        except Exception as e:
            logger.error(f"Memory relationship analysis failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "relationships": []
            }
    
    def _prepare_memory_data(self, memories: List[Dict[str, Any]]) -> str:
        """Prepare memory data for LLM analysis."""
        if not memories:
            return "No memory data available."
        
        formatted_data = []
        for i, memory in enumerate(memories[:20]):  # Limit to 20 memories to avoid token limits
            formatted_data.append(f"""
Memory {i+1}:
- ID: {memory.get('id', 'unknown')}
- Content: {memory.get('content', 'No content')}
- Type: {memory.get('memory_type', 'unknown')}
- Agent: {memory.get('agent_id', 'unknown')}
- Timestamp: {memory.get('timestamp', 'unknown')}
- Keywords: {memory.get('keywords', [])}
            """)
        
        return "\n".join(formatted_data)
    
    async def _call_openai_async(self, prompt: str) -> str:
        """Call OpenAI API asynchronously."""
        try:
            # Try different models in order of preference
            models_to_try = ["o4-mini", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4-turbo-preview"]
            
            for model in models_to_try:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a financial research analyst specializing in memory data analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000,
                        temperature=0.1
                    )
                    logger.info(f"Successfully used model: {model}")
                    return response.choices[0].message.content
                except Exception as model_error:
                    logger.warning(f"Model {model} failed: {model_error}")
                    continue
            
            # If all models fail
            raise Exception("No available OpenAI models found")
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    async def _fetch_memories_from_core(self, query: str) -> List[Dict[str, Any]]:
        """Fetch memories from core service."""
        # This would connect to the core service API
        # For now, return mock data
        return [
            {
                "id": "mem_001",
                "content": f"Memory related to: {query}",
                "memory_type": "research",
                "agent_id": "research_agent",
                "timestamp": datetime.now().isoformat(),
                "keywords": [query]
            }
        ]
    
    async def _fetch_memories_by_scope(self, scope: str, topic: str) -> List[Dict[str, Any]]:
        """Fetch memories based on scope and topic."""
        # This would implement scope-based memory fetching
        return await self._fetch_memories_from_core(topic)
    
    async def _fetch_memories_by_ids(self, memory_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch specific memories by IDs."""
        # This would fetch specific memories by ID
        memories = []
        for mem_id in memory_ids:
            memories.append({
                "id": mem_id,
                "content": f"Memory content for {mem_id}",
                "memory_type": "research",
                "timestamp": datetime.now().isoformat()
            })
        return memories
    
    async def _enhance_search_results(self, query: str, raw_memories: List[Dict[str, Any]], context: Optional[str]) -> List[Dict[str, Any]]:
        """Enhance search results with LLM analysis."""
        if not raw_memories:
            return []
        
        # Use LLM to enhance relevance and provide insights
        enhanced = []
        for memory in raw_memories:
            enhanced.append({
                **memory,
                "relevance_score": 0.85,  # Would be calculated by LLM
                "llm_insights": "Enhanced insights would be generated here"
            })
        
        return enhanced
    
    def _generate_basic_insights(self, research_topic: str, memories: List[Dict[str, Any]]) -> str:
        """Generate basic insights without LLM when API is unavailable."""
        if not memories:
            return f"""
Basic Analysis Report for: {research_topic}

SUMMARY:
- No memory data available for analysis
- Research topic: {research_topic}
- Analysis method: Rule-based (LLM unavailable)

RECOMMENDATIONS:
1. Collect more memory data related to {research_topic}
2. Establish data collection protocols
3. Retry analysis when LLM service is available
            """
        
        # Basic statistical analysis
        memory_count = len(memories)
        memory_types = set(memory.get('memory_type', 'unknown') for memory in memories)
        agents = set(memory.get('agent_id', 'unknown') for memory in memories)
        
        # Extract keywords
        all_keywords = []
        for memory in memories:
            keywords = memory.get('keywords', [])
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
        
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return f"""
Basic Analysis Report for: {research_topic}

SUMMARY:
- Total memories analyzed: {memory_count}
- Memory types: {', '.join(memory_types)}
- Contributing agents: {', '.join(agents)}
- Analysis method: Rule-based (LLM unavailable)

KEY PATTERNS:
- Most frequent keywords: {', '.join([k for k, v in top_keywords])}
- Memory distribution across types: {len(memory_types)} distinct types
- Agent contribution: {len(agents)} agents involved

BASIC INSIGHTS:
1. Data Volume: {memory_count} memories related to {research_topic}
2. Diversity: {len(memory_types)} different memory types suggest varied research approaches
3. Collaboration: {len(agents)} agents indicate multi-perspective analysis
4. Focus Areas: Top keywords suggest main themes in {research_topic}

RECOMMENDATIONS:
1. Enable LLM service for deeper semantic analysis
2. Investigate relationships between top keywords
3. Analyze temporal patterns in memory creation
4. Cross-reference findings with external data sources

NOTE: This is a basic rule-based analysis. For comprehensive insights, 
please ensure LLM service is available and properly configured.
        """

# Global LLM research service instance
llm_research_service = LLMResearchService()

async def main():
    """Test the LLM research service."""
    print("🧠 FinAgent LLM Research Service")
    print("=" * 50)
    
    if not llm_research_service.llm_available:
        print("❌ LLM service not available (missing OpenAI API key)")
        return
    
    print("✅ LLM service available")
    
    # Test semantic search
    print("\n🔍 Testing semantic search...")
    result = await llm_research_service.semantic_memory_search("financial patterns")
    print(f"Search result: {result['status']}")
    
    # Test research insights
    print("\n📊 Testing research insights...")
    insights = await llm_research_service.generate_research_insights("market volatility analysis")
    print(f"Insights result: {insights['status']}")

if __name__ == "__main__":
    asyncio.run(main())
