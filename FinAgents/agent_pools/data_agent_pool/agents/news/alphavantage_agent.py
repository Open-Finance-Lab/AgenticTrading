from typing import Dict, List, Optional, Union, Any, Callable
import requests
import os
from datetime import datetime, timedelta
from FinAgents.agent_pools.data_agent_pool.registry import BaseAgent
from FinAgents.agent_pools.data_agent_pool.schema.equity_schema import AlphaVantageNewsConfig
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool
from dotenv import load_dotenv
import json
import logging
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

load_dotenv()


class AlphaVantageNewsAgent(BaseAgent):
    """
    Alpha Vantage News & Sentiment API agent implementation.
    
    Features:
    - Market news with sentiment analysis
    - Company-specific news and sentiment
    - Topic-based news filtering
    - Real-time and historical news
    - Sentiment scoring and analysis
    - Source attribution and relevance scoring
    - Time-series sentiment data
    - Market sentiment trends
    """

    def __init__(self, config: AlphaVantageNewsConfig):
        """Initialize Alpha Vantage News API agent."""
        super().__init__(config.model_dump())
        self.config = config
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY') or config.api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.cache_dir = 'data/cache/alpha_vantage_news'
        os.makedirs(self.cache_dir, exist_ok=True)
        self._validate_config()
        self._init_tools()
        
        # Rate limiting (5 requests per minute for free tier)
        self.last_request_time = 0
        self.min_request_interval = 12  # seconds
        
        # Initialize thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=3)  # Lower due to rate limits
        
        if not hasattr(self.config, "llm_enabled"):
            raise ValueError("Missing required config parameter: 'llm_enabled'. Please add 'llm_enabled: true/false' to your alpha_vantage_news.yaml.")
        
        self.llm_enabled = bool(self.config.llm_enabled)
        print(f"Alpha Vantage News Agent - llm_enabled config value: {self.llm_enabled}")
        
        if self.llm_enabled:
            self._init_llm_interface()

    def _init_llm_interface(self):
        """Configure LLM interface for news and sentiment analysis."""
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1 
        )
        
        self.system_prompt = SystemMessage(content="""
You are a professional Alpha Vantage News & Sentiment API agent planner.

Your task is to generate an execution plan as a valid JSON object with a "steps" field (a list of tasks). Each step should specify:
- "tool": the tool to use
- "parameters": the parameters for the tool
- "type": the type of data being requested

Available tools:
- "get_market_news_sentiment": Get market news with sentiment analysis
- "get_company_news_sentiment": Get company-specific news with sentiment scores
- "get_topic_news": Get news filtered by specific topics
- "get_sentiment_analysis": Get detailed sentiment analysis for a symbol
- "get_news_by_time": Get news within specific time ranges
- "analyze_market_sentiment": Analyze overall market sentiment trends

**Only output a valid JSON object, and nothing else.**

Example:
{
  "steps": [
    {
      "tool": "get_company_news_sentiment",
      "parameters": {
        "symbol": "AAPL",
        "topics": "earnings,financial_markets",
        "limit": 50
      },
      "type": "company_sentiment"
    },
    {
      "tool": "get_market_news_sentiment",
      "parameters": {
        "topics": "technology,earnings",
        "sort": "RELEVANCE",
        "limit": 100
      },
      "type": "market_sentiment"
    }
  ]
}
""")

    def _init_tools(self):
        """Register available Alpha Vantage News API operations."""
        self.tools = [
            Tool(
                name="get_market_news_sentiment",
                func=self.get_market_news_sentiment,
                description="Get market news with sentiment analysis"
            ),
            Tool(
                name="get_company_news_sentiment",
                func=self.get_company_news_sentiment,
                description="Get company-specific news with sentiment scores"
            ),
            Tool(
                name="get_topic_news",
                func=self.get_topic_news,
                description="Get news filtered by specific topics"
            ),
            Tool(
                name="get_sentiment_analysis",
                func=self.get_sentiment_analysis,
                description="Get detailed sentiment analysis for a symbol"
            ),
            Tool(
                name="get_news_by_time",
                func=self.get_news_by_time,
                description="Get news within specific time ranges"
            ),
            Tool(
                name="analyze_market_sentiment",
                func=self.analyze_market_sentiment,
                description="Analyze overall market sentiment trends"
            )
        ]

    def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def get_market_news_sentiment(self,
                                 topics: str = "financial_markets",
                                 sort: str = "LATEST",
                                 limit: int = 50) -> Dict[str, Any]:
        """
        Get market news with sentiment analysis.
        
        Args:
            topics: Comma-separated topics (e.g., "financial_markets,earnings,ipo")
            sort: Sort order (LATEST, EARLIEST, RELEVANCE)
            limit: Number of articles (max 1000)
        """
        try:
            self._rate_limit()
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'topics': topics,
                'sort': sort,
                'limit': min(limit, 1000),
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")
            
            # Process sentiment data
            result = {
                'items': data.get('items', '0'),
                'sentiment_score_definition': data.get('sentiment_score_definition', ''),
                'relevance_score_definition': data.get('relevance_score_definition', ''),
                'articles': []
            }
            
            for article in data.get('feed', []):
                article_data = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'time_published': article.get('time_published', ''),
                    'authors': article.get('authors', []),
                    'summary': article.get('summary', ''),
                    'banner_image': article.get('banner_image', ''),
                    'source': article.get('source', ''),
                    'category_within_source': article.get('category_within_source', ''),
                    'source_domain': article.get('source_domain', ''),
                    'topics': article.get('topics', []),
                    'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                    'overall_sentiment_label': article.get('overall_sentiment_label', ''),
                    'ticker_sentiment': []
                }
                
                # Process ticker-specific sentiment
                for ticker_data in article.get('ticker_sentiment', []):
                    article_data['ticker_sentiment'].append({
                        'ticker': ticker_data.get('ticker', ''),
                        'relevance_score': float(ticker_data.get('relevance_score', 0)),
                        'ticker_sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                        'ticker_sentiment_label': ticker_data.get('ticker_sentiment_label', '')
                    })
                
                result['articles'].append(article_data)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get market news sentiment: {str(e)}")

    def get_company_news_sentiment(self,
                                  symbol: str,
                                  topics: str = None,
                                  time_from: str = None,
                                  time_to: str = None,
                                  sort: str = "LATEST",
                                  limit: int = 50) -> Dict[str, Any]:
        """
        Get company-specific news with sentiment scores.
        
        Args:
            symbol: Stock ticker symbol
            topics: Comma-separated topics (optional)
            time_from: Start time (YYYYMMDDTHHMM format)
            time_to: End time (YYYYMMDDTHHMM format)
            sort: Sort order (LATEST, EARLIEST, RELEVANCE)
            limit: Number of articles
        """
        try:
            self._rate_limit()
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'sort': sort,
                'limit': min(limit, 1000),
                'apikey': self.api_key
            }
            
            if topics:
                params['topics'] = topics
            if time_from:
                params['time_from'] = time_from
            if time_to:
                params['time_to'] = time_to
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")
            
            # Process and filter for the specific symbol
            result = {
                'symbol': symbol,
                'items': data.get('items', '0'),
                'articles': [],
                'sentiment_summary': {
                    'total_articles': 0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 0,
                    'average_sentiment': 0,
                    'average_relevance': 0
                }
            }
            
            sentiment_scores = []
            relevance_scores = []
            
            for article in data.get('feed', []):
                # Find sentiment data for our specific symbol
                symbol_sentiment = None
                for ticker_data in article.get('ticker_sentiment', []):
                    if ticker_data.get('ticker', '').upper() == symbol.upper():
                        symbol_sentiment = ticker_data
                        break
                
                if symbol_sentiment:
                    sentiment_score = float(symbol_sentiment.get('ticker_sentiment_score', 0))
                    relevance_score = float(symbol_sentiment.get('relevance_score', 0))
                    sentiment_label = symbol_sentiment.get('ticker_sentiment_label', '')
                    
                    sentiment_scores.append(sentiment_score)
                    relevance_scores.append(relevance_score)
                    
                    # Count sentiment labels
                    if sentiment_label.lower() == 'bullish':
                        result['sentiment_summary']['bullish_count'] += 1
                    elif sentiment_label.lower() == 'bearish':
                        result['sentiment_summary']['bearish_count'] += 1
                    else:
                        result['sentiment_summary']['neutral_count'] += 1
                    
                    article_data = {
                        'title': article.get('title', ''),
                        'url': article.get('url', ''),
                        'time_published': article.get('time_published', ''),
                        'authors': article.get('authors', []),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', ''),
                        'topics': article.get('topics', []),
                        'symbol_relevance_score': relevance_score,
                        'symbol_sentiment_score': sentiment_score,
                        'symbol_sentiment_label': sentiment_label,
                        'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                        'overall_sentiment_label': article.get('overall_sentiment_label', '')
                    }
                    
                    result['articles'].append(article_data)
            
            # Calculate summary statistics
            result['sentiment_summary']['total_articles'] = len(result['articles'])
            if sentiment_scores:
                result['sentiment_summary']['average_sentiment'] = sum(sentiment_scores) / len(sentiment_scores)
            if relevance_scores:
                result['sentiment_summary']['average_relevance'] = sum(relevance_scores) / len(relevance_scores)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get company news sentiment for {symbol}: {str(e)}")

    def get_topic_news(self,
                      topics: str,
                      sort: str = "LATEST",
                      limit: int = 50,
                      time_from: str = None,
                      time_to: str = None) -> Dict[str, Any]:
        """
        Get news filtered by specific topics.
        
        Args:
            topics: Comma-separated topics
            sort: Sort order (LATEST, EARLIEST, RELEVANCE)
            limit: Number of articles
            time_from: Start time (YYYYMMDDTHHMM format)  
            time_to: End time (YYYYMMDDTHHMM format)
        """
        try:
            self._rate_limit()
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'topics': topics,
                'sort': sort,
                'limit': min(limit, 1000),
                'apikey': self.api_key
            }
            
            if time_from:
                params['time_from'] = time_from
            if time_to:
                params['time_to'] = time_to
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")
            
            # Process topic-filtered news
            result = {
                'topics': topics,
                'items': data.get('items', '0'),
                'articles': []
            }
            
            for article in data.get('feed', []):
                article_data = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'time_published': article.get('time_published', ''),
                    'authors': article.get('authors', []),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'topics': article.get('topics', []),
                    'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                    'overall_sentiment_label': article.get('overall_sentiment_label', ''),
                    'relevant_tickers': []
                }
                
                # Extract ticker information
                for ticker_data in article.get('ticker_sentiment', []):
                    article_data['relevant_tickers'].append({
                        'ticker': ticker_data.get('ticker', ''),
                        'relevance_score': float(ticker_data.get('relevance_score', 0)),
                        'sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                        'sentiment_label': ticker_data.get('ticker_sentiment_label', '')
                    })
                
                result['articles'].append(article_data)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get topic news for '{topics}': {str(e)}")

    def get_sentiment_analysis(self,
                              symbol: str,
                              time_from: str = None,
                              time_to: str = None) -> Dict[str, Any]:
        """
        Get detailed sentiment analysis for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            time_from: Start time (YYYYMMDDTHHMM format)
            time_to: End time (YYYYMMDDTHHMM format)
        """
        try:
            # Get company news with sentiment
            news_data = self.get_company_news_sentiment(
                symbol=symbol,
                time_from=time_from,
                time_to=time_to,
                limit=200
            )
            
            # Perform detailed sentiment analysis
            articles = news_data.get('articles', [])
            
            if not articles:
                return {
                    'symbol': symbol,
                    'analysis': 'No articles found for sentiment analysis',
                    'sentiment_metrics': {}
                }
            
            # Calculate sentiment metrics
            sentiment_scores = [a['symbol_sentiment_score'] for a in articles]
            relevance_scores = [a['symbol_relevance_score'] for a in articles]
            
            # Time-based sentiment analysis
            time_series_sentiment = {}
            for article in articles:
                date = article['time_published'][:8]  # YYYYMMDD
                if date not in time_series_sentiment:
                    time_series_sentiment[date] = {
                        'scores': [],
                        'count': 0,
                        'bullish': 0,
                        'bearish': 0,
                        'neutral': 0
                    }
                
                time_series_sentiment[date]['scores'].append(article['symbol_sentiment_score'])
                time_series_sentiment[date]['count'] += 1
                
                label = article['symbol_sentiment_label'].lower()
                if label == 'bullish':
                    time_series_sentiment[date]['bullish'] += 1
                elif label == 'bearish':
                    time_series_sentiment[date]['bearish'] += 1
                else:
                    time_series_sentiment[date]['neutral'] += 1
            
            # Calculate daily averages
            for date in time_series_sentiment:
                scores = time_series_sentiment[date]['scores']
                time_series_sentiment[date]['average_sentiment'] = sum(scores) / len(scores) if scores else 0
            
            # Source analysis
            source_sentiment = {}
            for article in articles:
                source = article['source']
                if source not in source_sentiment:
                    source_sentiment[source] = {
                        'scores': [],
                        'count': 0,
                        'average_sentiment': 0
                    }
                
                source_sentiment[source]['scores'].append(article['symbol_sentiment_score'])
                source_sentiment[source]['count'] += 1
            
            for source in source_sentiment:
                scores = source_sentiment[source]['scores']
                source_sentiment[source]['average_sentiment'] = sum(scores) / len(scores) if scores else 0
            
            result = {
                'symbol': symbol,
                'analysis_period': {
                    'from': time_from,
                    'to': time_to
                },
                'sentiment_metrics': {
                    'total_articles': len(articles),
                    'average_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
                    'sentiment_std': self._calculate_std(sentiment_scores),
                    'average_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
                    'bullish_percentage': (news_data['sentiment_summary']['bullish_count'] / len(articles)) * 100,
                    'bearish_percentage': (news_data['sentiment_summary']['bearish_count'] / len(articles)) * 100,
                    'neutral_percentage': (news_data['sentiment_summary']['neutral_count'] / len(articles)) * 100,
                    'sentiment_trend': self._calculate_trend(sentiment_scores),
                    'high_relevance_articles': len([a for a in articles if a['symbol_relevance_score'] > 0.5])
                },
                'time_series_sentiment': time_series_sentiment,
                'source_analysis': source_sentiment,
                'summary': news_data['sentiment_summary']
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze sentiment for {symbol}: {str(e)}")

    def get_news_by_time(self,
                        time_from: str,
                        time_to: str,
                        topics: str = None,
                        symbols: str = None,
                        limit: int = 100) -> Dict[str, Any]:
        """
        Get news within specific time ranges.
        
        Args:
            time_from: Start time (YYYYMMDDTHHMM format)
            time_to: End time (YYYYMMDDTHHMM format)
            topics: Comma-separated topics (optional)
            symbols: Comma-separated symbols (optional)
            limit: Number of articles
        """
        try:
            self._rate_limit()
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'time_from': time_from,
                'time_to': time_to,
                'sort': 'LATEST',
                'limit': min(limit, 1000),
                'apikey': self.api_key
            }
            
            if topics:
                params['topics'] = topics
            if symbols:
                params['tickers'] = symbols
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Limit: {data['Note']}")
            
            result = {
                'time_range': {
                    'from': time_from,
                    'to': time_to
                },
                'filters': {
                    'topics': topics,
                    'symbols': symbols
                },
                'items': data.get('items', '0'),
                'articles': []
            }
            
            for article in data.get('feed', []):
                article_data = {
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'time_published': article.get('time_published', ''),
                    'authors': article.get('authors', []),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', ''),
                    'topics': article.get('topics', []),
                    'overall_sentiment_score': float(article.get('overall_sentiment_score', 0)),
                    'overall_sentiment_label': article.get('overall_sentiment_label', ''),
                    'ticker_sentiment': []
                }
                
                for ticker_data in article.get('ticker_sentiment', []):
                    article_data['ticker_sentiment'].append({
                        'ticker': ticker_data.get('ticker', ''),
                        'relevance_score': float(ticker_data.get('relevance_score', 0)),
                        'sentiment_score': float(ticker_data.get('ticker_sentiment_score', 0)),
                        'sentiment_label': ticker_data.get('ticker_sentiment_label', '')
                    })
                
                result['articles'].append(article_data)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get news by time range: {str(e)}")

    def analyze_market_sentiment(self,
                               topics: str = "financial_markets,earnings",
                               limit: int = 200) -> Dict[str, Any]:
        """
        Analyze overall market sentiment trends.
        
        Args:
            topics: Topics to analyze
            limit: Number of articles to analyze
        """
        try:
            # Get market news
            news_data = self.get_market_news_sentiment(
                topics=topics,
                limit=limit,
                sort="LATEST"
            )
            
            articles = news_data.get('articles', [])
            
            if not articles:
                return {
                    'analysis': 'No articles found for market sentiment analysis',
                    'market_metrics': {}
                }
            
            # Analyze overall market sentiment
            overall_scores = [a['overall_sentiment_score'] for a in articles]
            
            # Topic-based analysis
            topic_sentiment = {}
            for article in articles:
                for topic in article.get('topics', []):
                    topic_name = topic.get('topic', '')
                    if topic_name not in topic_sentiment:
                        topic_sentiment[topic_name] = {
                            'scores': [],
                            'relevance_scores': [],
                            'count': 0
                        }
                    
                    topic_sentiment[topic_name]['scores'].append(article['overall_sentiment_score'])
                    topic_sentiment[topic_name]['relevance_scores'].append(float(topic.get('relevance_score', 0)))
                    topic_sentiment[topic_name]['count'] += 1
            
            # Calculate topic averages
            for topic in topic_sentiment:
                scores = topic_sentiment[topic]['scores']
                topic_sentiment[topic]['average_sentiment'] = sum(scores) / len(scores) if scores else 0
                
                rel_scores = topic_sentiment[topic]['relevance_scores']
                topic_sentiment[topic]['average_relevance'] = sum(rel_scores) / len(rel_scores) if rel_scores else 0
            
            # Time-based market sentiment
            time_sentiment = {}
            for article in articles:
                date = article['time_published'][:8]  # YYYYMMDD
                if date not in time_sentiment:
                    time_sentiment[date] = {
                        'scores': [],
                        'count': 0
                    }
                
                time_sentiment[date]['scores'].append(article['overall_sentiment_score'])
                time_sentiment[date]['count'] += 1
            
            for date in time_sentiment:
                scores = time_sentiment[date]['scores']
                time_sentiment[date]['average_sentiment'] = sum(scores) / len(scores) if scores else 0
            
            result = {
                'analysis_timestamp': datetime.now().isoformat(),
                'topics_analyzed': topics,
                'total_articles': len(articles),
                'market_metrics': {
                    'overall_market_sentiment': sum(overall_scores) / len(overall_scores) if overall_scores else 0,
                    'sentiment_volatility': self._calculate_std(overall_scores),
                    'sentiment_trend': self._calculate_trend(overall_scores),
                    'bullish_articles': len([s for s in overall_scores if s > 0.1]),
                    'bearish_articles': len([s for s in overall_scores if s < -0.1]),
                    'neutral_articles': len([s for s in overall_scores if -0.1 <= s <= 0.1]),
                    'sentiment_distribution': self._create_sentiment_distribution(overall_scores)
                },
                'topic_analysis': topic_sentiment,
                'time_series_sentiment': time_sentiment,
                'top_sources': self._analyze_sources(articles)
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to analyze market sentiment: {str(e)}")

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend calculation based on first half vs second half
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(values[mid:]) / (len(values) - mid) if len(values) - mid > 0 else 0
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"

    def _create_sentiment_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Create sentiment distribution buckets."""
        distribution = {
            'very_bearish': 0,      # < -0.35
            'bearish': 0,           # -0.35 to -0.15
            'somewhat_bearish': 0,  # -0.15 to -0.05
            'neutral': 0,           # -0.05 to 0.05
            'somewhat_bullish': 0,  # 0.05 to 0.15
            'bullish': 0,           # 0.15 to 0.35
            'very_bullish': 0       # > 0.35
        }
        
        for score in scores:
            if score < -0.35:
                distribution['very_bearish'] += 1
            elif score < -0.15:
                distribution['bearish'] += 1
            elif score < -0.05:
                distribution['somewhat_bearish'] += 1
            elif score <= 0.05:
                distribution['neutral'] += 1
            elif score <= 0.15:
                distribution['somewhat_bullish'] += 1
            elif score <= 0.35:
                distribution['bullish'] += 1
            else:
                distribution['very_bullish'] += 1
        
        return distribution

    def _analyze_sources(self, articles: List[Dict]) -> Dict[str, Any]:
        """Analyze news sources."""
        source_stats = {}
        
        for article in articles:
            source = article.get('source', 'Unknown')
            if source not in source_stats:
                source_stats[source] = {
                    'count': 0,
                    'sentiment_scores': [],
                    'average_sentiment': 0
                }
            
            source_stats[source]['count'] += 1
            source_stats[source]['sentiment_scores'].append(article['overall_sentiment_score'])
        
        # Calculate averages and sort by count
        for source in source_stats:
            scores = source_stats[source]['sentiment_scores']
            source_stats[source]['average_sentiment'] = sum(scores) / len(scores) if scores else 0
        
        # Return top 10 sources by article count
        sorted_sources = sorted(source_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        return dict(sorted_sources[:10])

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set ALPHA_VANTAGE_API_KEY environment variable or provide in config.")

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
                        "tool": "get_market_news_sentiment",
                        "parameters": {"topics": "financial_markets", "limit": 50},
                        "type": "market_sentiment"
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
        """Process natural language news and sentiment requests."""
        if not getattr(self, "llm_enabled", True):
            # Default plan for testing
            plan = {
                "steps": [{
                    "tool": "get_market_news_sentiment",
                    "parameters": {"topics": "financial_markets", "limit": 50},
                    "type": "market_sentiment"
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
        print("=== Alpha Vantage News Execution Plan ===")
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
        """Execute generated news and sentiment strategy."""
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