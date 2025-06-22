from typing import Dict, List, Optional, Union, Any, Callable
import requests
import os
from datetime import datetime, timedelta
from registry import BaseAgent
from schema.news_schema import NewsAPIConfig
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool
from dotenv import load_dotenv
import json
import logging
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()


class NewsAPIAgent(BaseAgent):
    """
    News API data agent implementation for financial news.
    
    Features:
    - Financial news from major sources
    - Company-specific news
    - Market news and analysis
    - Sector-specific news
    - Real-time news updates
    - Sentiment analysis support
    - Source filtering
    - Date range filtering
    """

    def __init__(self, config: NewsAPIConfig):
        """Initialize News API data agent."""
        super().__init__(config.model_dump())
        self.config = config
        self.api_key = os.getenv('NEWS_API_KEY') or config.api_key
        self.base_url = "https://newsapi.org/v2"
        self.cache_dir = 'data/cache/newsapi'
        os.makedirs(self.cache_dir, exist_ok=True)
        self._validate_config()
        self._init_tools()
        
        # Initialize thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        if not hasattr(self.config, "llm_enabled"):
            raise ValueError("Missing required config parameter: 'llm_enabled'. Please add 'llm_enabled: true/false' to your newsapi.yaml.")
        
        self.llm_enabled = bool(self.config.llm_enabled)
        print(f"News API Agent - llm_enabled config value: {self.llm_enabled}")
        
        if self.llm_enabled:
            self._init_llm_interface()

    def _init_llm_interface(self):
        """Configure LLM interface for news analysis."""
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1 
        )
        
        self.system_prompt = SystemMessage(content="""
You are a professional News API data agent planner.

Your task is to generate an execution plan as a valid JSON object with a "steps" field (a list of tasks). Each step should specify:
- "tool": the tool to use
- "parameters": the parameters for the tool
- "type": the type of data being requested

Available tools:
- "get_company_news": Get news articles for a specific company/stock
- "get_financial_news": Get general financial and business news
- "get_sector_news": Get news for a specific industry sector
- "get_trending_news": Get trending financial news
- "search_news": Search for news with specific keywords
- "get_sources": Get available news sources
- "get_headlines": Get top headlines from financial sources

**Only output a valid JSON object, and nothing else.**

Example:
{
  "steps": [
    {
      "tool": "get_company_news",
      "parameters": {
        "company": "Apple",
        "symbol": "AAPL",
        "days_back": 7
      },
      "type": "company_news"
    },
    {
      "tool": "get_financial_news",
      "parameters": {
        "category": "business",
        "page_size": 20
      },
      "type": "market_news"
    }
  ]
}
""")

    def _init_tools(self):
        """Register available News API operations."""
        self.tools = [
            Tool(
                name="get_company_news",
                func=self.get_company_news,
                description="Get news articles for a specific company or stock symbol"
            ),
            Tool(
                name="get_financial_news",
                func=self.get_financial_news,
                description="Get general financial and business news"
            ),
            Tool(
                name="get_sector_news",
                func=self.get_sector_news,
                description="Get news for a specific industry sector"
            ),
            Tool(
                name="get_trending_news",
                func=self.get_trending_news,
                description="Get trending financial news stories"
            ),
            Tool(
                name="search_news",
                func=self.search_news,
                description="Search for news with specific keywords"
            ),
            Tool(
                name="get_sources",
                func=self.get_sources,
                description="Get available news sources"
            ),
            Tool(
                name="get_headlines",
                func=self.get_headlines,
                description="Get top headlines from financial sources"
            )
        ]

    def get_company_news(self, 
                        company: str,
                        symbol: str = None,
                        days_back: int = 7,
                        page_size: int = 20,
                        language: str = "en") -> List[Dict[str, Any]]:
        """
        Get news articles for a specific company.
        
        Args:
            company: Company name
            symbol: Stock ticker symbol (optional)
            days_back: Number of days to look back
            page_size: Number of articles to return
            language: Language code (en, es, fr, etc.)
        """
        try:
            # Build search query
            query_terms = [company]
            if symbol:
                query_terms.append(symbol)
            
            query = " OR ".join(query_terms)
            
            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'pageSize': min(page_size, 100),  # API limit
                'language': language,
                'domains': 'bloomberg.com,reuters.com,cnbc.com,marketwatch.com,fool.com,seekingalpha.com,finance.yahoo.com',
                'apiKey': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'content': article.get('content', ''),
                    'url_to_image': article.get('urlToImage', ''),
                    'company': company,
                    'symbol': symbol
                })
            
            return articles
            
        except Exception as e:
            raise RuntimeError(f"Failed to get company news for {company}: {str(e)}")

    def get_financial_news(self,
                          category: str = "business",
                          country: str = "us",
                          page_size: int = 20,
                          page: int = 1) -> List[Dict[str, Any]]:
        """
        Get general financial and business news.
        
        Args:
            category: News category (business, general, health, science, sports, technology)
            country: Country code (us, gb, ca, au, etc.)
            page_size: Number of articles to return
            page: Page number
        """
        try:
            params = {
                'category': category,
                'country': country,
                'pageSize': min(page_size, 100),
                'page': page,
                'apiKey': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/top-headlines", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'content': article.get('content', ''),
                    'url_to_image': article.get('urlToImage', ''),
                    'category': category,
                    'country': country
                })
            
            return articles
            
        except Exception as e:
            raise RuntimeError(f"Failed to get financial news: {str(e)}")

    def get_sector_news(self,
                       sector: str,
                       days_back: int = 7,
                       page_size: int = 20,
                       language: str = "en") -> List[Dict[str, Any]]:
        """
        Get news for a specific industry sector.
        
        Args:
            sector: Industry sector (e.g., "technology", "healthcare", "finance")
            days_back: Number of days to look back
            page_size: Number of articles to return
            language: Language code
        """
        try:
            # Build sector-specific query
            sector_keywords = {
                'technology': 'technology OR tech OR software OR AI OR cybersecurity',
                'healthcare': 'healthcare OR pharma OR biotech OR medical OR drug',
                'finance': 'finance OR banking OR fintech OR insurance',
                'energy': 'energy OR oil OR renewable OR solar OR wind',
                'retail': 'retail OR consumer OR e-commerce OR shopping',
                'automotive': 'automotive OR car OR electric vehicle OR EV',
                'real estate': 'real estate OR property OR housing OR REIT'
            }
            
            query = sector_keywords.get(sector.lower(), sector)
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'pageSize': min(page_size, 100),
                'language': language,
                'domains': 'bloomberg.com,reuters.com,cnbc.com,marketwatch.com,techcrunch.com,venturebeat.com',
                'apiKey': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'content': article.get('content', ''),
                    'url_to_image': article.get('urlToImage', ''),
                    'sector': sector
                })
            
            return articles
            
        except Exception as e:
            raise RuntimeError(f"Failed to get sector news for {sector}: {str(e)}")

    def get_trending_news(self,
                         page_size: int = 20,
                         country: str = "us") -> List[Dict[str, Any]]:
        """Get trending financial news stories."""
        try:
            params = {
                'sources': 'bloomberg,reuters,cnbc,the-wall-street-journal,financial-times',
                'pageSize': min(page_size, 100),
                'apiKey': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/top-headlines", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'content': article.get('content', ''),
                    'url_to_image': article.get('urlToImage', ''),
                    'trending': True
                })
            
            return articles
            
        except Exception as e:
            raise RuntimeError(f"Failed to get trending news: {str(e)}")

    def search_news(self,
                   keywords: str,
                   days_back: int = 30,
                   page_size: int = 20,
                   sort_by: str = "publishedAt",
                   language: str = "en") -> List[Dict[str, Any]]:
        """
        Search for news with specific keywords.
        
        Args:
            keywords: Search keywords
            days_back: Number of days to look back
            page_size: Number of articles to return
            sort_by: Sort order (publishedAt, relevancy, popularity)
            language: Language code
        """
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            params = {
                'q': keywords,
                'from': from_date,
                'sortBy': sort_by,
                'pageSize': min(page_size, 100),
                'language': language,
                'apiKey': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'content': article.get('content', ''),
                    'url_to_image': article.get('urlToImage', ''),
                    'keywords': keywords
                })
            
            return articles
            
        except Exception as e:
            raise RuntimeError(f"Failed to search news for '{keywords}': {str(e)}")

    def get_sources(self,
                   category: str = "business",
                   language: str = "en",
                   country: str = "us") -> List[Dict[str, Any]]:
        """Get available news sources."""
        try:
            params = {
                'category': category,
                'language': language,
                'country': country,
                'apiKey': self.api_key
            }
            
            response = requests.get(f"{self.base_url}/sources", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            sources = []
            for source in data.get('sources', []):
                sources.append({
                    'id': source.get('id', ''),
                    'name': source.get('name', ''),
                    'description': source.get('description', ''),
                    'url': source.get('url', ''),
                    'category': source.get('category', ''),
                    'language': source.get('language', ''),
                    'country': source.get('country', '')
                })
            
            return sources
            
        except Exception as e:
            raise RuntimeError(f"Failed to get news sources: {str(e)}")

    def get_headlines(self,
                     sources: str = None,
                     page_size: int = 20,
                     category: str = "business") -> List[Dict[str, Any]]:
        """
        Get top headlines from financial sources.
        
        Args:
            sources: Comma-separated list of source IDs
            page_size: Number of headlines to return
            category: News category
        """
        try:
            params = {
                'pageSize': min(page_size, 100),
                'apiKey': self.api_key
            }
            
            if sources:
                params['sources'] = sources
            else:
                params['category'] = category
                params['country'] = 'us'
            
            response = requests.get(f"{self.base_url}/top-headlines", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            headlines = []
            for article in data.get('articles', []):
                headlines.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', ''),
                    'url_to_image': article.get('urlToImage', '')
                })
            
            return headlines
            
        except Exception as e:
            raise RuntimeError(f"Failed to get headlines: {str(e)}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not self.api_key:
            raise ValueError("News API key is required. Set NEWS_API_KEY environment variable or provide in config.")

    def _parse_intent(self, llm_output: str) -> dict:
        """Parse LLM output into a validated execution plan."""
        try:
            plan = json.loads(llm_output)
        except Exception:
            try:
                json_str = re.search(r'\{.*\}', llm_output, re.DOTALL).group()
                plan = json.loads(json_str)
            except Exception:
                logging.warning("LLM output is not valid JSON. Using default plan.")
                plan = {
                    "steps": [{
                        "tool": "get_financial_news",
                        "parameters": {"category": "business", "page_size": 10},
                        "type": "financial_news"
                    }]
                }

        # Validation
        if "steps" in plan:
            if not isinstance(plan["steps"], list) or not plan["steps"]:
                raise ValueError("Execution plan 'steps' must be a non-empty list.")
            for step in plan["steps"]:
                for field in ["tool", "parameters"]:
                    if field not in step:
                        raise ValueError(f"Step missing required field: {field}")
        else:
            for field in ["tool", "parameters"]:
                if field not in plan:
                    raise ValueError(f"Execution plan missing required field: {field}")

        return plan

    async def process_intent(self, query: str) -> Dict[str, Any]:
        """Process natural language news requests."""
        if not getattr(self, "llm_enabled", True):
            # Default plan for testing
            plan = {
                "steps": [{
                    "tool": "get_financial_news",
                    "parameters": {"category": "business", "page_size": 10},
                    "type": "financial_news"
                }]
            }
            result = await self._execute_strategy(plan)
            return {
                "execution_plan": plan,
                "result": result,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "query_type": plan["steps"][0].get("type") if "steps" in plan else plan.get("type"),
                    "llm_used": False
                }
            }

        # LLM-driven path
        intent_analysis = await self.llm.agenerate([
            [self.system_prompt, HumanMessage(content=query)]
        ])
        plan = self._parse_intent(intent_analysis.generations[0][0].text)
        print("=== News API Execution Plan ===")
        print(json.dumps(plan, indent=2))
        result = await self._execute_strategy(plan)
        
        return {
            "execution_plan": plan,
            "result": result,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query_type": plan["steps"][0].get("type") if "steps" in plan else plan.get("type"),
                "llm_used": True
            }
        }

    async def _execute_strategy(self, plan: Dict) -> Any:
        """Execute generated news strategy."""
        import inspect
        
        try:
            if "steps" in plan:
                # Multi-step plan
                results = []
                for step in plan["steps"]:
                    tool_name = step.get("tool")
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if not tool:
                        available = [t.name for t in self.tools]
                        raise ValueError(f"Tool not found: {tool_name}. Available tools: {available}")
                    
                    func = tool.func
                    params = step.get("parameters", {})
                    
                    # Execute in thread pool if synchronous
                    if inspect.iscoroutinefunction(func):
                        result = await func(**params)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(self.executor, lambda: func(**params))
                    
                    results.append({
                        "step": tool_name,
                        "result": result
                    })
                return results
            else:
                # Single-step plan
                tool_name = plan.get("tool")
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_name}")
                
                func = tool.func
                params = plan.get("parameters", {})
                
                if inspect.iscoroutinefunction(func):
                    return await func(**params)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(self.executor, lambda: func(**params))
                    
        except Exception as e:
            raise RuntimeError(f"Strategy execution failed: {str(e)}")

    def __del__(self):
        """Clean up thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)