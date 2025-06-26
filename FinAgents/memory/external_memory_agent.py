"""
External Memory Agent for FinAgent-Orchestration System

This module provides a unified, external memory service for all agent pools in the 
FinAgent-Orchestration system. It serves as a centralized logging and event storage
system that supports efficient retrieval, filtering, and future extensibility for
backtesting and reinforcement learning use cases.

Key Features:
- Unified log/event storage and retrieval APIs for all agent pools
- Efficient, scalable storage with support for tags, time ranges, and sources
- Event-driven architecture with async support
- Industrial-grade logging and monitoring
- Extensible design for future ML/RL workflows
- Thread-safe operations with proper error handling

Author: FinAgent-Orchestration Team
Date: 2024
"""

import asyncio
import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager, contextmanager
import aiofiles
import aiosqlite
from abc import ABC, abstractmethod


# Configure logging for the External Memory Agent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of supported event types for classification."""
    TRANSACTION = "transaction"
    OPTIMIZATION = "optimization"
    MARKET_DATA = "market_data"
    PORTFOLIO_UPDATE = "portfolio_update"
    AGENT_COMMUNICATION = "agent_communication"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SYSTEM = "system"
    USER_ACTION = "user_action"
    CUSTOM = "custom"


class LogLevel(Enum):
    """Log level enumeration for event prioritization."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MemoryEvent:
    """
    Represents a single event/log entry in the memory system.
    
    This is the core data structure for all events stored in the memory agent.
    It provides a standardized format for all agent pools to log their activities.
    """
    event_id: str
    timestamp: datetime
    event_type: EventType
    log_level: LogLevel
    source_agent_pool: str
    source_agent_id: str
    title: str
    content: str
    tags: Set[str]
    metadata: Dict[str, Any]
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'log_level': self.log_level.value,
            'source_agent_pool': self.source_agent_pool,
            'source_agent_id': self.source_agent_id,
            'title': self.title,
            'content': self.content,
            'tags': list(self.tags),
            'metadata': self.metadata,
            'session_id': self.session_id,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEvent':
        """Create a MemoryEvent from a dictionary."""
        return cls(
            event_id=data['event_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=EventType(data['event_type']),
            log_level=LogLevel(data['log_level']),
            source_agent_pool=data['source_agent_pool'],
            source_agent_id=data['source_agent_id'],
            title=data['title'],
            content=data['content'],
            tags=set(data['tags']),
            metadata=data['metadata'],
            session_id=data.get('session_id'),
            correlation_id=data.get('correlation_id')
        )


@dataclass
class QueryFilter:
    """
    Filter parameters for querying the memory system.
    
    Provides flexible filtering capabilities for retrieving specific events
    based on various criteria including time ranges, sources, and content.
    """
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[EventType]] = None
    log_levels: Optional[List[LogLevel]] = None
    source_agent_pools: Optional[List[str]] = None
    source_agent_ids: Optional[List[str]] = None
    tags: Optional[Set[str]] = None
    content_search: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    limit: int = 100
    offset: int = 0


@dataclass
class QueryResult:
    """
    Result container for memory queries.
    
    Provides structured results with metadata about the query execution.
    """
    events: List[MemoryEvent]
    total_count: int
    query_time_ms: float
    has_more: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the query result to a dictionary."""
        return {
            'events': [event.to_dict() for event in self.events],
            'total_count': self.total_count,
            'query_time_ms': self.query_time_ms,
            'has_more': self.has_more
        }


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    
    This interface allows for different storage implementations while maintaining
    a consistent API for the memory agent.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass
    
    @abstractmethod
    async def store_event(self, event: MemoryEvent) -> bool:
        """Store a single event."""
        pass
    
    @abstractmethod
    async def store_events_batch(self, events: List[MemoryEvent]) -> int:
        """Store multiple events in a batch operation."""
        pass
    
    @abstractmethod
    async def query_events(self, query_filter: QueryFilter) -> QueryResult:
        """Query events based on filter criteria."""
        pass
    
    @abstractmethod
    async def delete_events(self, event_ids: List[str]) -> int:
        """Delete events by their IDs."""
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup and close storage connections."""
        pass


class SQLiteStorageBackend(StorageBackend):
    """
    SQLite-based storage backend for the memory agent.
    
    Provides a file-based storage solution with full SQL query capabilities.
    Suitable for development, testing, and moderate production workloads.
    """
    
    def __init__(self, db_path: str = "memory_agent.db"):
        """
        Initialize SQLite storage backend.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
    
    async def initialize(self) -> None:
        """Initialize the SQLite database with required tables."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    log_level TEXT NOT NULL,
                    source_agent_pool TEXT NOT NULL,
                    source_agent_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    session_id TEXT,
                    correlation_id TEXT
                )
            ''')
            
            # Create indices for efficient querying
            await db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_events(timestamp)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON memory_events(event_type)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_source_pool ON memory_events(source_agent_pool)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_log_level ON memory_events(log_level)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_session_id ON memory_events(session_id)')
            
            await db.commit()
        
        logger.info(f"SQLite storage backend initialized at {self.db_path}")
    
    async def store_event(self, event: MemoryEvent) -> bool:
        """Store a single event in the database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    INSERT OR REPLACE INTO memory_events 
                    (event_id, timestamp, event_type, log_level, source_agent_pool,
                     source_agent_id, title, content, tags, metadata, session_id, correlation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.log_level.value,
                    event.source_agent_pool,
                    event.source_agent_id,
                    event.title,
                    event.content,
                    json.dumps(list(event.tags)),
                    json.dumps(event.metadata),
                    event.session_id,
                    event.correlation_id
                ))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store event {event.event_id}: {e}")
            return False
    
    async def store_events_batch(self, events: List[MemoryEvent]) -> int:
        """Store multiple events in a batch operation."""
        if not events:
            return 0
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                event_data = []
                for event in events:
                    event_data.append((
                        event.event_id,
                        event.timestamp.isoformat(),
                        event.event_type.value,
                        event.log_level.value,
                        event.source_agent_pool,
                        event.source_agent_id,
                        event.title,
                        event.content,
                        json.dumps(list(event.tags)),
                        json.dumps(event.metadata),
                        event.session_id,
                        event.correlation_id
                    ))
                
                await db.executemany('''
                    INSERT OR REPLACE INTO memory_events 
                    (event_id, timestamp, event_type, log_level, source_agent_pool,
                     source_agent_id, title, content, tags, metadata, session_id, correlation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', event_data)
                
                await db.commit()
                return len(events)
        except Exception as e:
            logger.error(f"Failed to store batch of {len(events)} events: {e}")
            return 0
    
    async def query_events(self, query_filter: QueryFilter) -> QueryResult:
        """Query events based on filter criteria."""
        start_time = datetime.now()
        
        # Build dynamic query
        where_clauses = []
        params = []
        
        if query_filter.start_time:
            where_clauses.append("timestamp >= ?")
            params.append(query_filter.start_time.isoformat())
        
        if query_filter.end_time:
            where_clauses.append("timestamp <= ?")
            params.append(query_filter.end_time.isoformat())
        
        if query_filter.event_types:
            placeholders = ",".join("?" * len(query_filter.event_types))
            where_clauses.append(f"event_type IN ({placeholders})")
            params.extend([et.value for et in query_filter.event_types])
        
        if query_filter.log_levels:
            placeholders = ",".join("?" * len(query_filter.log_levels))
            where_clauses.append(f"log_level IN ({placeholders})")
            params.extend([ll.value for ll in query_filter.log_levels])
        
        if query_filter.source_agent_pools:
            placeholders = ",".join("?" * len(query_filter.source_agent_pools))
            where_clauses.append(f"source_agent_pool IN ({placeholders})")
            params.extend(query_filter.source_agent_pools)
        
        if query_filter.source_agent_ids:
            placeholders = ",".join("?" * len(query_filter.source_agent_ids))
            where_clauses.append(f"source_agent_id IN ({placeholders})")
            params.extend(query_filter.source_agent_ids)
        
        if query_filter.content_search:
            where_clauses.append("(title LIKE ? OR content LIKE ?)")
            search_term = f"%{query_filter.content_search}%"
            params.extend([search_term, search_term])
        
        if query_filter.session_id:
            where_clauses.append("session_id = ?")
            params.append(query_filter.session_id)
        
        if query_filter.correlation_id:
            where_clauses.append("correlation_id = ?")
            params.append(query_filter.correlation_id)
        
        # Construct WHERE clause
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get total count
                count_query = f"SELECT COUNT(*) FROM memory_events {where_sql}"
                async with db.execute(count_query, params) as cursor:
                    total_count = (await cursor.fetchone())[0]
                
                # Get paginated results
                query = f'''
                    SELECT * FROM memory_events {where_sql}
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                '''
                query_params = params + [query_filter.limit, query_filter.offset]
                
                events = []
                async with db.execute(query, query_params) as cursor:
                    async for row in cursor:
                        events.append(self._row_to_event(row))
                
                query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                has_more = (query_filter.offset + len(events)) < total_count
                
                return QueryResult(
                    events=events,
                    total_count=total_count,
                    query_time_ms=query_time_ms,
                    has_more=has_more
                )
        
        except Exception as e:
            logger.error(f"Failed to query events: {e}")
            return QueryResult(events=[], total_count=0, query_time_ms=0, has_more=False)
    
    def _row_to_event(self, row) -> MemoryEvent:
        """Convert a database row to a MemoryEvent object."""
        return MemoryEvent(
            event_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            event_type=EventType(row[2]),
            log_level=LogLevel(row[3]),
            source_agent_pool=row[4],
            source_agent_id=row[5],
            title=row[6],
            content=row[7],
            tags=set(json.loads(row[8])),
            metadata=json.loads(row[9]),
            session_id=row[10],
            correlation_id=row[11]
        )
    
    async def delete_events(self, event_ids: List[str]) -> int:
        """Delete events by their IDs."""
        if not event_ids:
            return 0
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                placeholders = ",".join("?" * len(event_ids))
                query = f"DELETE FROM memory_events WHERE event_id IN ({placeholders})"
                cursor = await db.execute(query, event_ids)
                await db.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to delete events: {e}")
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                stats = {}
                
                # Total events
                async with db.execute("SELECT COUNT(*) FROM memory_events") as cursor:
                    stats['total_events'] = (await cursor.fetchone())[0]
                
                # Events by type
                async with db.execute("""
                    SELECT event_type, COUNT(*) 
                    FROM memory_events 
                    GROUP BY event_type
                """) as cursor:
                    stats['events_by_type'] = dict(await cursor.fetchall())
                
                # Events by source pool
                async with db.execute("""
                    SELECT source_agent_pool, COUNT(*) 
                    FROM memory_events 
                    GROUP BY source_agent_pool
                """) as cursor:
                    stats['events_by_pool'] = dict(await cursor.fetchall())
                
                # Recent activity (last 24 hours)
                yesterday = (datetime.now() - timedelta(days=1)).isoformat()
                async with db.execute("""
                    SELECT COUNT(*) FROM memory_events 
                    WHERE timestamp >= ?
                """, (yesterday,)) as cursor:
                    stats['events_last_24h'] = (await cursor.fetchone())[0]
                
                return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """Cleanup and close storage connections."""
        # SQLite connections are automatically closed with async context managers
        logger.info("SQLite storage backend cleanup completed")


class ExternalMemoryAgent:
    """
    External Memory Agent for the FinAgent-Orchestration System.
    
    This is the main class that provides a unified interface for all agent pools
    to store, retrieve, and manage their logs and events. It acts as a centralized
    memory service with support for filtering, tagging, and future ML/RL workflows.
    
    The agent is designed to be:
    - Thread-safe and async-compatible
    - Highly performant with batch operations
    - Extensible for future requirements
    - Production-ready with proper error handling and logging
    """
    
    def __init__(self, 
                 storage_backend: Optional[StorageBackend] = None,
                 enable_real_time_hooks: bool = True,
                 max_batch_size: int = 1000):
        """
        Initialize the External Memory Agent.
        
        Args:
            storage_backend: Storage backend implementation (defaults to SQLite)
            enable_real_time_hooks: Enable real-time event processing hooks
            max_batch_size: Maximum number of events to process in a single batch
        """
        self.storage_backend = storage_backend or SQLiteStorageBackend()
        self.enable_real_time_hooks = enable_real_time_hooks
        self.max_batch_size = max_batch_size
        
        # Event processing hooks for real-time analysis
        self._event_hooks: List[Callable[[MemoryEvent], None]] = []
        self._batch_hooks: List[Callable[[List[MemoryEvent]], None]] = []
        
        # Internal state
        self._initialized = False
        self._stats = {
            'events_stored': 0,
            'events_retrieved': 0,
            'batch_operations': 0,
            'errors': 0
        }
        
        logger.info("External Memory Agent initialized")
    
    async def initialize(self) -> None:
        """Initialize the memory agent and its storage backend."""
        if self._initialized:
            logger.warning("Memory agent is already initialized")
            return
        
        try:
            await self.storage_backend.initialize()
            self._initialized = True
            logger.info("External Memory Agent successfully initialized")
        except Exception as e:
            logger.error(f"Failed to initialize External Memory Agent: {e}")
            raise
    
    async def log_event(self,
                       event_type: EventType,
                       log_level: LogLevel,
                       source_agent_pool: str,
                       source_agent_id: str,
                       title: str,
                       content: str,
                       tags: Optional[Set[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       session_id: Optional[str] = None,
                       correlation_id: Optional[str] = None) -> str:
        """
        Log a single event to the memory system.
        
        This is the primary method for agent pools to store their events and logs.
        
        Args:
            event_type: Type of the event
            log_level: Log level of the event
            source_agent_pool: Name of the source agent pool
            source_agent_id: ID of the specific agent
            title: Brief title of the event
            content: Detailed content of the event
            tags: Optional set of tags for categorization
            metadata: Optional metadata dictionary
            session_id: Optional session identifier
            correlation_id: Optional correlation identifier for related events
            
        Returns:
            str: Unique event ID
        """
        if not self._initialized:
            raise RuntimeError("Memory agent not initialized. Call initialize() first.")
        
        event_id = str(uuid.uuid4())
        
        event = MemoryEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            log_level=log_level,
            source_agent_pool=source_agent_pool,
            source_agent_id=source_agent_id,
            title=title,
            content=content,
            tags=tags or set(),
            metadata=metadata or {},
            session_id=session_id,
            correlation_id=correlation_id
        )
        
        try:
            success = await self.storage_backend.store_event(event)
            if success:
                self._stats['events_stored'] += 1
                
                # Execute real-time hooks if enabled
                if self.enable_real_time_hooks:
                    for hook in self._event_hooks:
                        try:
                            hook(event)
                        except Exception as e:
                            logger.error(f"Error in event hook: {e}")
                
                logger.debug(f"Event stored successfully: {event_id}")
                return event_id
            else:
                self._stats['errors'] += 1
                raise RuntimeError(f"Failed to store event {event_id}")
                
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to log event: {e}")
            raise
    
    async def log_events_batch(self, events_data: List[Dict[str, Any]]) -> List[str]:
        """
        Log multiple events in a batch operation for improved performance.
        
        Args:
            events_data: List of event dictionaries with required fields
            
        Returns:
            List[str]: List of event IDs for successfully stored events
        """
        if not self._initialized:
            raise RuntimeError("Memory agent not initialized. Call initialize() first.")
        
        if not events_data:
            return []
        
        # Limit batch size for performance
        if len(events_data) > self.max_batch_size:
            logger.warning(f"Batch size {len(events_data)} exceeds maximum {self.max_batch_size}, truncating")
            events_data = events_data[:self.max_batch_size]
        
        events = []
        event_ids = []
        
        for event_data in events_data:
            event_id = str(uuid.uuid4())
            event_ids.append(event_id)
            
            event = MemoryEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=EventType(event_data['event_type']),
                log_level=LogLevel(event_data['log_level']),
                source_agent_pool=event_data['source_agent_pool'],
                source_agent_id=event_data['source_agent_id'],
                title=event_data['title'],
                content=event_data['content'],
                tags=set(event_data.get('tags', [])),
                metadata=event_data.get('metadata', {}),
                session_id=event_data.get('session_id'),
                correlation_id=event_data.get('correlation_id')
            )
            events.append(event)
        
        try:
            stored_count = await self.storage_backend.store_events_batch(events)
            self._stats['events_stored'] += stored_count
            self._stats['batch_operations'] += 1
            
            # Execute batch hooks if enabled
            if self.enable_real_time_hooks and stored_count > 0:
                for hook in self._batch_hooks:
                    try:
                        hook(events[:stored_count])
                    except Exception as e:
                        logger.error(f"Error in batch hook: {e}")
            
            logger.info(f"Batch operation completed: {stored_count}/{len(events)} events stored")
            return event_ids[:stored_count]
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to store batch events: {e}")
            raise
    
    async def query_events(self, query_filter: QueryFilter) -> QueryResult:
        """
        Query events from the memory system with flexible filtering.
        
        Args:
            query_filter: Filter parameters for the query
            
        Returns:
            QueryResult: Query results with events and metadata
        """
        if not self._initialized:
            raise RuntimeError("Memory agent not initialized. Call initialize() first.")
        
        try:
            result = await self.storage_backend.query_events(query_filter)
            self._stats['events_retrieved'] += len(result.events)
            
            logger.debug(f"Query completed: {len(result.events)} events retrieved in {result.query_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to query events: {e}")
            raise
    
    async def get_events_by_session(self, session_id: str, limit: int = 100) -> List[MemoryEvent]:
        """
        Retrieve all events for a specific session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of events to retrieve
            
        Returns:
            List[MemoryEvent]: Events for the session
        """
        query_filter = QueryFilter(session_id=session_id, limit=limit)
        result = await self.query_events(query_filter)
        return result.events
    
    async def get_events_by_correlation(self, correlation_id: str, limit: int = 100) -> List[MemoryEvent]:
        """
        Retrieve all events with a specific correlation ID.
        
        Args:
            correlation_id: Correlation identifier
            limit: Maximum number of events to retrieve
            
        Returns:
            List[MemoryEvent]: Correlated events
        """
        query_filter = QueryFilter(correlation_id=correlation_id, limit=limit)
        result = await self.query_events(query_filter)
        return result.events
    
    async def get_recent_events(self, 
                               source_agent_pool: Optional[str] = None,
                               hours: int = 24,
                               limit: int = 100) -> List[MemoryEvent]:
        """
        Get recent events from the specified time period.
        
        Args:
            source_agent_pool: Optional filter by agent pool
            hours: Number of hours to look back
            limit: Maximum number of events to retrieve
            
        Returns:
            List[MemoryEvent]: Recent events
        """
        start_time = datetime.now() - timedelta(hours=hours)
        query_filter = QueryFilter(
            start_time=start_time,
            source_agent_pools=[source_agent_pool] if source_agent_pool else None,
            limit=limit
        )
        result = await self.query_events(query_filter)
        return result.events
    
    def add_event_hook(self, hook: Callable[[MemoryEvent], None]) -> None:
        """
        Add a real-time event processing hook.
        
        Args:
            hook: Function to call for each stored event
        """
        self._event_hooks.append(hook)
        logger.info("Event hook added")
    
    def add_batch_hook(self, hook: Callable[[List[MemoryEvent]], None]) -> None:
        """
        Add a batch processing hook.
        
        Args:
            hook: Function to call for each batch of stored events
        """
        self._batch_hooks.append(hook)
        logger.info("Batch hook added")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the memory system.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        storage_stats = await self.storage_backend.get_statistics()
        
        combined_stats = {
            'agent_stats': self._stats.copy(),
            'storage_stats': storage_stats,
            'initialized': self._initialized,
            'hooks_enabled': self.enable_real_time_hooks,
            'event_hooks_count': len(self._event_hooks),
            'batch_hooks_count': len(self._batch_hooks)
        }
        
        return combined_stats
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            await self.storage_backend.cleanup()
            self._initialized = False
            logger.info("External Memory Agent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise


# Convenience functions for common operations
async def create_memory_agent(storage_path: Optional[str] = None) -> ExternalMemoryAgent:
    """
    Create and initialize a new External Memory Agent instance.
    
    Args:
        storage_path: Optional custom path for SQLite database
        
    Returns:
        ExternalMemoryAgent: Initialized memory agent
    """
    if storage_path:
        storage_backend = SQLiteStorageBackend(storage_path)
    else:
        storage_backend = SQLiteStorageBackend()
    
    agent = ExternalMemoryAgent(storage_backend)
    await agent.initialize()
    return agent


def create_transaction_event(agent_pool: str, 
                           agent_id: str,
                           transaction_type: str,
                           symbol: str,
                           quantity: float,
                           price: float,
                           cost: float,
                           session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to create a standardized transaction event dictionary.
    
    Args:
        agent_pool: Source agent pool name
        agent_id: Source agent identifier
        transaction_type: Type of transaction (buy/sell)
        symbol: Trading symbol
        quantity: Transaction quantity
        price: Transaction price
        cost: Transaction cost
        session_id: Optional session identifier
        
    Returns:
        Dict[str, Any]: Event dictionary ready for logging
    """
    return {
        'event_type': EventType.TRANSACTION.value,
        'log_level': LogLevel.INFO.value,
        'source_agent_pool': agent_pool,
        'source_agent_id': agent_id,
        'title': f'{transaction_type.upper()} {symbol}',
        'content': f'Executed {transaction_type} order: {quantity} shares of {symbol} at ${price:.2f}',
        'tags': ['transaction', transaction_type, symbol],
        'metadata': {
            'transaction_type': transaction_type,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'timestamp': datetime.now().isoformat()
        },
        'session_id': session_id
    }


def create_optimization_event(agent_pool: str,
                            agent_id: str,
                            optimization_type: str,
                            result: Dict[str, Any],
                            session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to create a standardized optimization event dictionary.
    
    Args:
        agent_pool: Source agent pool name
        agent_id: Source agent identifier
        optimization_type: Type of optimization performed
        result: Optimization results
        session_id: Optional session identifier
        
    Returns:
        Dict[str, Any]: Event dictionary ready for logging
    """
    return {
        'event_type': EventType.OPTIMIZATION.value,
        'log_level': LogLevel.INFO.value,
        'source_agent_pool': agent_pool,
        'source_agent_id': agent_id,
        'title': f'{optimization_type} Optimization Completed',
        'content': f'Optimization process completed with {len(result)} results',
        'tags': ['optimization', optimization_type],
        'metadata': {
            'optimization_type': optimization_type,
            'result': result,
            'timestamp': datetime.now().isoformat()
        },
        'session_id': session_id
    }
