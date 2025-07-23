#!/usr/bin/env python3
"""
Advanced Memory Indexing and Retrieval System

This module implements intelligent memory indexing using vector embeddings,
semantic search, and advanced query optimization for the FinAgent Memory system.
"""

import asyncio
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Try to import sentence transformers for advanced embeddings
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

@dataclass
class MemoryIndex:
    """Memory index entry with embeddings and metadata"""
    memory_id: str
    content_hash: str
    text_embedding: np.ndarray
    metadata_features: Dict[str, Any]
    timestamp: datetime
    agent_id: str
    memory_type: str
    keywords: List[str]
    performance_score: Optional[float] = None


class IntelligentMemoryIndexer:
    """
    Advanced memory indexing system with semantic search capabilities
    """
    
    def __init__(self, 
                 index_file: str = "memory_index.pkl",
                 use_transformers: bool = True,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the intelligent memory indexer
        
        Args:
            index_file: Path to store the memory index
            use_transformers: Whether to use transformer models for embeddings
            model_name: Name of the sentence transformer model
        """
        self.index_file = index_file
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.memory_index: Dict[str, MemoryIndex] = {}
        
        # Initialize embedding models
        if self.use_transformers:
            try:
                self.transformer_model = SentenceTransformer(model_name)
                print(f"✅ Loaded transformer model: {model_name}")
            except Exception as e:
                print(f"⚠️ Failed to load transformer model: {e}")
                self.use_transformers = False
        
        if not self.use_transformers:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.tfidf_fitted = False
            print("✅ Using TF-IDF vectorizer for embeddings")
        
        # Load existing index
        self.load_index()
    
    def extract_text_content(self, memory_content: Dict[str, Any]) -> str:
        """Extract searchable text from memory content"""
        text_parts = []
        
        # Extract from signal data
        if 'signal_data' in memory_content:
            signal_data = memory_content['signal_data']
            if 'symbol' in signal_data:
                text_parts.append(f"symbol {signal_data['symbol']}")
            if 'reasoning' in signal_data:
                text_parts.append(signal_data['reasoning'])
        
        # Extract from strategy data  
        if 'strategy_type' in memory_content:
            text_parts.append(f"strategy {memory_content['strategy_type']}")
        
        if 'strategy_data' in memory_content:
            strategy_data = memory_content['strategy_data']
            if 'factors' in strategy_data:
                text_parts.extend([f"factor {factor}" for factor in strategy_data['factors']])
        
        # Extract performance metrics as text
        if 'performance_metrics' in memory_content:
            perf = memory_content['performance_metrics']
            if 'returns' in perf:
                returns = perf['returns']
                performance_level = 'high' if returns > 0.1 else 'medium' if returns > 0 else 'low'
                text_parts.append(f"performance {performance_level}")
        
        # Extract metadata
        if 'metadata' in memory_content:
            metadata = memory_content['metadata']
            for key, value in metadata.items():
                if isinstance(value, str):
                    text_parts.append(f"{key} {value}")
        
        return " ".join(text_parts)
    
    def extract_keywords(self, text_content: str) -> List[str]:
        """Extract keywords from text content"""
        # Simple keyword extraction - can be enhanced with NLP libraries
        words = text_content.lower().split()
        
        # Filter meaningful words
        financial_keywords = {
            'buy', 'sell', 'hold', 'signal', 'strategy', 'momentum', 'reversion',
            'performance', 'returns', 'volatility', 'sharpe', 'drawdown',
            'aapl', 'tsla', 'nvda', 'high', 'medium', 'low', 'bullish', 'bearish'
        }
        
        keywords = [word for word in words if word in financial_keywords or len(word) > 3]
        return list(set(keywords))  # Remove duplicates
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """Create text embedding using available model"""
        if self.use_transformers:
            return self.transformer_model.encode([text])[0]
        else:
            # Use TF-IDF
            if not self.tfidf_fitted:
                # Need to fit on existing texts first
                all_texts = [text]
                for idx in self.memory_index.values():
                    all_texts.append(self.extract_text_content({"content": "dummy"}))
                
                if len(all_texts) > 1:
                    self.tfidf_vectorizer.fit(all_texts)
                    self.tfidf_fitted = True
                else:
                    # Return zero vector if no data to fit
                    return np.zeros(100)
            
            try:
                return self.tfidf_vectorizer.transform([text]).toarray()[0]
            except:
                return np.zeros(self.tfidf_vectorizer.max_features or 100)
    
    def calculate_performance_score(self, memory_content: Dict[str, Any]) -> Optional[float]:
        """Calculate performance score for memory ranking"""
        if 'performance_metrics' in memory_content:
            metrics = memory_content['performance_metrics']
            
            # Weighted performance score
            score = 0.0
            weights = {
                'returns': 0.4,
                'sharpe_ratio': 0.3,
                'win_rate': 0.2,
                'max_drawdown': -0.1  # Negative weight for drawdown
            }
            
            for metric, weight in weights.items():
                if metric in metrics:
                    value = metrics[metric]
                    if metric == 'max_drawdown':
                        # Convert negative drawdown to positive contribution
                        score += weight * (1 + value)  # drawdown is negative
                    else:
                        score += weight * value
            
            return max(0.0, min(1.0, score))  # Normalize to [0, 1]
        
        return None
    
    async def index_memory(self, 
                          memory_id: str,
                          agent_id: str,
                          memory_type: str,
                          content: Dict[str, Any],
                          timestamp: Optional[datetime] = None) -> bool:
        """
        Index a memory with intelligent features
        
        Args:
            memory_id: Unique memory identifier
            agent_id: Agent that created the memory
            memory_type: Type of memory (signal, strategy, performance)
            content: Memory content dictionary
            timestamp: Memory timestamp
            
        Returns:
            bool: Success status
        """
        try:
            # Extract text content
            text_content = self.extract_text_content(content)
            
            # Create embedding
            embedding = self.create_text_embedding(text_content)
            
            # Extract keywords
            keywords = self.extract_keywords(text_content)
            
            # Calculate performance score
            performance_score = self.calculate_performance_score(content)
            
            # Create content hash for deduplication
            content_str = json.dumps(content, sort_keys=True)
            content_hash = str(hash(content_str))
            
            # Create metadata features
            metadata_features = {
                'text_length': len(text_content),
                'keyword_count': len(keywords),
                'has_performance': performance_score is not None,
                'agent_type': agent_id.split('_')[0] if '_' in agent_id else 'unknown'
            }
            
            # Create index entry
            index_entry = MemoryIndex(
                memory_id=memory_id,
                content_hash=content_hash,
                text_embedding=embedding,
                metadata_features=metadata_features,
                timestamp=timestamp or datetime.utcnow(),
                agent_id=agent_id,
                memory_type=memory_type,
                keywords=keywords,
                performance_score=performance_score
            )
            
            # Store in index
            self.memory_index[memory_id] = index_entry
            
            # Save index
            await self.save_index()
            
            print(f"✅ Indexed memory {memory_id[:8]}... with {len(keywords)} keywords")
            return True
            
        except Exception as e:
            print(f"❌ Failed to index memory {memory_id}: {e}")
            return False
    
    def semantic_search(self, 
                       query: str, 
                       limit: int = 10,
                       memory_type: Optional[str] = None,
                       agent_id: Optional[str] = None,
                       min_performance: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Perform semantic search on indexed memories
        
        Args:
            query: Search query text
            limit: Maximum number of results
            memory_type: Filter by memory type
            agent_id: Filter by agent ID
            min_performance: Minimum performance score filter
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        if not self.memory_index:
            return []
        
        # Create query embedding
        query_embedding = self.create_text_embedding(query)
        
        # Calculate similarities
        similarities = []
        
        for memory_id, index_entry in self.memory_index.items():
            # Apply filters
            if memory_type and index_entry.memory_type != memory_type:
                continue
            if agent_id and index_entry.agent_id != agent_id:
                continue
            if min_performance and (index_entry.performance_score is None or index_entry.performance_score < min_performance):
                continue
            
            # Calculate similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                index_entry.text_embedding.reshape(1, -1)
            )[0][0]
            
            # Boost similarity based on performance score
            if index_entry.performance_score:
                similarity *= (1 + index_entry.performance_score * 0.2)
            
            similarities.append((memory_id, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def keyword_search(self, 
                      keywords: List[str], 
                      limit: int = 10) -> List[Tuple[str, float]]:
        """
        Perform keyword-based search
        
        Args:
            keywords: List of keywords to search
            limit: Maximum number of results
            
        Returns:
            List of (memory_id, score) tuples
        """
        keyword_scores = []
        
        for memory_id, index_entry in self.memory_index.items():
            # Calculate keyword overlap score
            overlap = len(set(keywords) & set(index_entry.keywords))
            if overlap > 0:
                score = overlap / len(keywords)  # Normalized overlap
                keyword_scores.append((memory_id, score))
        
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:limit]
    
    def get_related_memories(self, 
                           memory_id: str, 
                           limit: int = 5) -> List[Tuple[str, float]]:
        """
        Find memories related to a given memory
        
        Args:
            memory_id: Reference memory ID
            limit: Maximum number of results
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        if memory_id not in self.memory_index:
            return []
        
        reference_entry = self.memory_index[memory_id]
        similarities = []
        
        for other_id, other_entry in self.memory_index.items():
            if other_id == memory_id:
                continue
            
            # Calculate similarity
            similarity = cosine_similarity(
                reference_entry.text_embedding.reshape(1, -1),
                other_entry.text_embedding.reshape(1, -1)
            )[0][0]
            
            similarities.append((other_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def get_trending_keywords(self, 
                            time_window: timedelta = timedelta(hours=24),
                            limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get trending keywords within a time window
        
        Args:
            time_window: Time window to analyze
            limit: Maximum number of keywords
            
        Returns:
            List of (keyword, frequency) tuples
        """
        cutoff_time = datetime.utcnow() - time_window
        keyword_counts = {}
        
        for index_entry in self.memory_index.values():
            if index_entry.timestamp >= cutoff_time:
                for keyword in index_entry.keywords:
                    keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        trending = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return trending[:limit]
    
    async def save_index(self):
        """Save memory index to disk"""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.memory_index, f)
        except Exception as e:
            print(f"⚠️ Failed to save index: {e}")
    
    def load_index(self):
        """Load memory index from disk"""
        try:
            if os.path.exists(self.index_file):
                with open(self.index_file, 'rb') as f:
                    self.memory_index = pickle.load(f)
                print(f"✅ Loaded {len(self.memory_index)} indexed memories")
        except Exception as e:
            print(f"⚠️ Failed to load index: {e}")
            self.memory_index = {}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        if not self.memory_index:
            return {"total_memories": 0}
        
        stats = {
            "total_memories": len(self.memory_index),
            "memory_types": {},
            "agent_types": {},
            "avg_keywords_per_memory": 0,
            "memories_with_performance": 0
        }
        
        total_keywords = 0
        
        for entry in self.memory_index.values():
            # Count by memory type
            stats["memory_types"][entry.memory_type] = stats["memory_types"].get(entry.memory_type, 0) + 1
            
            # Count by agent type
            agent_type = entry.metadata_features.get("agent_type", "unknown")
            stats["agent_types"][agent_type] = stats["agent_types"].get(agent_type, 0) + 1
            
            # Count keywords
            total_keywords += len(entry.keywords)
            
            # Count performance memories
            if entry.performance_score is not None:
                stats["memories_with_performance"] += 1
        
        stats["avg_keywords_per_memory"] = total_keywords / len(self.memory_index)
        
        return stats


# Test function
async def test_intelligent_indexer():
    """Test the intelligent memory indexer"""
    print("🧪 Testing Intelligent Memory Indexer")
    print("=" * 40)
    
    indexer = IntelligentMemoryIndexer()
    
    # Test indexing
    test_memories = [
        {
            "memory_id": "test_001",
            "agent_id": "alpha_agent_001",
            "memory_type": "signal",
            "content": {
                "signal_data": {
                    "symbol": "AAPL",
                    "reasoning": "Strong earnings beat with revenue growth",
                    "confidence": 0.9
                }
            }
        },
        {
            "memory_id": "test_002", 
            "agent_id": "strategy_agent_001",
            "memory_type": "strategy",
            "content": {
                "strategy_type": "momentum_trading",
                "performance_metrics": {
                    "returns": 0.15,
                    "sharpe_ratio": 1.2
                }
            }
        }
    ]
    
    # Index test memories
    for memory in test_memories:
        await indexer.index_memory(**memory)
    
    # Test semantic search
    print("\n🔍 Testing semantic search:")
    results = indexer.semantic_search("AAPL earnings strong performance")
    for memory_id, score in results:
        print(f"  {memory_id}: {score:.3f}")
    
    # Test keyword search
    print("\n🏷️ Testing keyword search:")
    results = indexer.keyword_search(["momentum", "trading"])
    for memory_id, score in results:
        print(f"  {memory_id}: {score:.3f}")
    
    # Show stats
    print("\n📊 Index Statistics:")
    stats = indexer.get_index_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_intelligent_indexer())
