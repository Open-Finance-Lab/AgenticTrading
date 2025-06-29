"""
External Memory Agent Interface for FinAgent-Orchestration System

This module provides a simplified, dependency-free interface for the External Memory Agent.
It serves as a unified logging and event storage system for all agent pools.

Key Features:
- Unified log/event storage and retrieval APIs
- File-based storage with JSON serialization
- Thread-safe operations
- Industrial-grade logging and monitoring
- Extensible design for future ML/RL workflows

Author: FinAgent-Orchestration Team
Date: 2025
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, asdict
import time

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


class FileStorageBackend:
    """
    File-based storage backend for the memory agent.
    
    Provides a simple, dependency-free storage solution using JSON files
    organized by date for efficient querying and maintenance.
    """
    
    def __init__(self, storage_dir: str = "memory_storage"):
        """
        Initialize file storage backend.
        
        Args:
            storage_dir: Directory path for storing memory files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        
        # Create index file for fast querying
        self.index_file = self.storage_dir / "index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load or create the index file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    loaded_index = json.load(f)
                self.index = {
                    'events_count': loaded_index.get('events_count', 0),
                    'last_updated': loaded_index.get('last_updated', datetime.now().isoformat()),
                    'agent_pools': set(loaded_index.get('agent_pools', [])),
                    'event_types': set(loaded_index.get('event_types', []))
                }
            except (json.JSONDecodeError, FileNotFoundError):
                self.index = {
                    'events_count': 0,
                    'last_updated': datetime.now().isoformat(),
                    'agent_pools': set(),
                    'event_types': set()
                }
        else:
            self.index = {
                'events_count': 0,
                'last_updated': datetime.now().isoformat(),
                'agent_pools': set(),
                'event_types': set()
            }
    
    def _save_index(self) -> None:
        """Save the index file."""
        # Convert sets to lists for JSON serialization
        index_to_save = self.index.copy()
        index_to_save['agent_pools'] = list(self.index['agent_pools'])
        index_to_save['event_types'] = list(self.index['event_types'])
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index_to_save, f, indent=2, ensure_ascii=False)
    
    def _get_file_path(self, date: datetime) -> Path:
        """Get the file path for a specific date."""
        date_str = date.strftime("%Y-%m-%d")
        return self.storage_dir / f"events_{date_str}.json"
    
    def store_event(self, event: MemoryEvent) -> bool:
        """Store a single event in the appropriate daily file."""
        try:
            with self.lock:
                file_path = self._get_file_path(event.timestamp)
                
                # Load existing events for the day
                events = []
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        events = json.load(f)
                
                # Add new event
                events.append(event.to_dict())
                
                # Save updated events
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(events, f, indent=2, ensure_ascii=False)
                
                # Update index
                self.index['events_count'] += 1
                self.index['agent_pools'].add(event.source_agent_pool)
                self.index['event_types'].add(event.event_type.value)
                self.index['last_updated'] = datetime.now().isoformat()
                self._save_index()
                
                return True
        except Exception as e:
            logger.error(f"Failed to store event {event.event_id}: {e}")
            return False
    
    def store_events_batch(self, events: List[MemoryEvent]) -> int:
        """Store multiple events in batch."""
        if not events:
            return 0
        
        stored_count = 0
        with self.lock:
            # Group events by date for efficient storage
            events_by_date = {}
            for event in events:
                date_key = event.timestamp.strftime("%Y-%m-%d")
                if date_key not in events_by_date:
                    events_by_date[date_key] = []
                events_by_date[date_key].append(event)
            
            # Store events for each date
            for date_str, date_events in events_by_date.items():
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    file_path = self._get_file_path(date_obj)
                    
                    # Load existing events
                    existing_events = []
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            existing_events = json.load(f)
                    
                    # Add new events
                    for event in date_events:
                        existing_events.append(event.to_dict())
                        stored_count += 1
                        
                        # Update index
                        self.index['agent_pools'].add(event.source_agent_pool)
                        self.index['event_types'].add(event.event_type.value)
                    
                    # Save updated events
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_events, f, indent=2, ensure_ascii=False)
                
                except Exception as e:
                    logger.error(f"Failed to store events for date {date_str}: {e}")
            
            # Update index
            self.index['events_count'] += stored_count
            self.index['last_updated'] = datetime.now().isoformat()
            self._save_index()
        
        return stored_count
    
    def query_events(self, query_filter: QueryFilter) -> QueryResult:
        """Query events based on filter criteria."""
        start_time = time.time()
        
        # Determine date range for file scanning
        start_date = query_filter.start_time or datetime.now() - timedelta(days=30)
        end_date = query_filter.end_time or datetime.now()
        
        all_events = []
        
        # Scan files within date range
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        while current_date <= end_date:
            file_path = self._get_file_path(current_date)
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        daily_events = json.load(f)
                    
                    for event_data in daily_events:
                        try:
                            event = MemoryEvent.from_dict(event_data)
                            if self._matches_filter(event, query_filter):
                                all_events.append(event)
                        except Exception as e:
                            logger.warning(f"Failed to parse event: {e}")
                
                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {e}")
            
            current_date += timedelta(days=1)
        
        # Sort by timestamp (most recent first)
        all_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply pagination
        total_count = len(all_events)
        start_idx = query_filter.offset
        end_idx = start_idx + query_filter.limit
        paginated_events = all_events[start_idx:end_idx]
        
        query_time_ms = (time.time() - start_time) * 1000
        has_more = end_idx < total_count
        
        return QueryResult(
            events=paginated_events,
            total_count=total_count,
            query_time_ms=query_time_ms,
            has_more=has_more
        )
    
    def _matches_filter(self, event: MemoryEvent, query_filter: QueryFilter) -> bool:
        """Check if an event matches the query filter."""
        # Time range check
        if query_filter.start_time and event.timestamp < query_filter.start_time:
            return False
        if query_filter.end_time and event.timestamp > query_filter.end_time:
            return False
        
        # Event type check
        if query_filter.event_types and event.event_type not in query_filter.event_types:
            return False
        
        # Log level check
        if query_filter.log_levels and event.log_level not in query_filter.log_levels:
            return False
        
        # Source checks
        if query_filter.source_agent_pools and event.source_agent_pool not in query_filter.source_agent_pools:
            return False
        if query_filter.source_agent_ids and event.source_agent_id not in query_filter.source_agent_ids:
            return False
        
        # Tag check
        if query_filter.tags and not query_filter.tags.intersection(event.tags):
            return False
        
        # Content search
        if query_filter.content_search:
            search_text = query_filter.content_search.lower()
            if (search_text not in event.title.lower() and 
                search_text not in event.content.lower()):
                return False
        
        # Session and correlation checks
        if query_filter.session_id and event.session_id != query_filter.session_id:
            return False
        if query_filter.correlation_id and event.correlation_id != query_filter.correlation_id:
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self.lock:
            return {
                'total_events': self.index['events_count'],
                'agent_pools': list(self.index['agent_pools']),
                'event_types': list(self.index['event_types']),
                'last_updated': self.index['last_updated'],
                'storage_directory': str(self.storage_dir)
            }


class ExternalMemoryAgent:
    """
    External Memory Agent for the FinAgent-Orchestration System.
    
    This is the main class that provides a unified interface for all agent pools
    to store, retrieve, and manage their logs and events. It acts as a centralized
    memory service with support for filtering, tagging, and future ML/RL workflows.
    
    The agent is designed to be:
    - Thread-safe and sync-compatible
    - Highly performant with batch operations
    - Extensible for future requirements
    - Production-ready with proper error handling and logging
    """
    
    def __init__(self, 
                 storage_backend: Optional[FileStorageBackend] = None,
                 enable_real_time_hooks: bool = True,
                 max_batch_size: int = 1000):
        """
        Initialize the External Memory Agent.
        
        Args:
            storage_backend: Storage backend implementation (defaults to FileStorage)
            enable_real_time_hooks: Enable real-time event processing hooks
            max_batch_size: Maximum number of events to process in a single batch
        """
        self.storage_backend = storage_backend or FileStorageBackend()
        self.enable_real_time_hooks = enable_real_time_hooks
        self.max_batch_size = max_batch_size
        
        # Event processing hooks for real-time analysis
        self._event_hooks: List[Callable[[MemoryEvent], None]] = []
        self._batch_hooks: List[Callable[[List[MemoryEvent]], None]] = []
        
        # Internal state
        self._initialized = True  # File storage doesn't need async initialization
        self._stats = {
            'events_stored': 0,
            'events_retrieved': 0,
            'batch_operations': 0,
            'errors': 0
        }
        
        logger.info("External Memory Agent initialized with file storage backend")
    
    def log_event(self,
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
            success = self.storage_backend.store_event(event)
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
    
    def log_events_batch(self, events_data: List[Dict[str, Any]]) -> List[str]:
        """
        Log multiple events in a batch operation for improved performance.
        
        Args:
            events_data: List of event dictionaries with required fields
            
        Returns:
            List[str]: List of event IDs for successfully stored events
        """
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
            stored_count = self.storage_backend.store_events_batch(events)
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
    
    def query_events(self, query_filter: QueryFilter) -> QueryResult:
        """
        Query events from the memory system with flexible filtering.
        
        Args:
            query_filter: Filter parameters for the query
            
        Returns:
            QueryResult: Query results with events and metadata
        """
        try:
            result = self.storage_backend.query_events(query_filter)
            self._stats['events_retrieved'] += len(result.events)
            
            logger.debug(f"Query completed: {len(result.events)} events retrieved in {result.query_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Failed to query events: {e}")
            raise
    
    def get_events_by_session(self, session_id: str, limit: int = 100) -> List[MemoryEvent]:
        """
        Retrieve all events for a specific session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of events to retrieve
            
        Returns:
            List[MemoryEvent]: Events for the session
        """
        query_filter = QueryFilter(session_id=session_id, limit=limit)
        result = self.query_events(query_filter)
        return result.events
    
    def get_events_by_correlation(self, correlation_id: str, limit: int = 100) -> List[MemoryEvent]:
        """
        Retrieve all events with a specific correlation ID.
        
        Args:
            correlation_id: Correlation identifier
            limit: Maximum number of events to retrieve
            
        Returns:
            List[MemoryEvent]: Correlated events
        """
        query_filter = QueryFilter(correlation_id=correlation_id, limit=limit)
        result = self.query_events(query_filter)
        return result.events
    
    def get_recent_events(self, 
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
        result = self.query_events(query_filter)
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the memory system.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        storage_stats = self.storage_backend.get_statistics()
        
        combined_stats = {
            'agent_stats': self._stats.copy(),
            'storage_stats': storage_stats,
            'initialized': self._initialized,
            'hooks_enabled': self.enable_real_time_hooks,
            'event_hooks_count': len(self._event_hooks),
            'batch_hooks_count': len(self._batch_hooks)
        }
        
        return combined_stats
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("External Memory Agent cleanup completed")


# Convenience functions for common operations
def create_memory_agent(storage_path: Optional[str] = None) -> ExternalMemoryAgent:
    """
    Create and initialize a new External Memory Agent instance.
    
    Args:
        storage_path: Optional custom path for file storage
        
    Returns:
        ExternalMemoryAgent: Initialized memory agent
    """
    if storage_path:
        storage_backend = FileStorageBackend(storage_path)
    else:
        storage_backend = FileStorageBackend()
    
    agent = ExternalMemoryAgent(storage_backend)
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


def create_market_data_event(agent_pool: str,
                           agent_id: str,
                           symbol: str,
                           data_type: str,
                           data: Dict[str, Any],
                           session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to create a standardized market data event dictionary.
    
    Args:
        agent_pool: Source agent pool name
        agent_id: Source agent identifier
        symbol: Trading symbol
        data_type: Type of market data (price, volume, etc.)
        data: Market data dictionary
        session_id: Optional session identifier
        
    Returns:
        Dict[str, Any]: Event dictionary ready for logging
    """
    return {
        'event_type': EventType.MARKET_DATA.value,
        'log_level': LogLevel.INFO.value,
        'source_agent_pool': agent_pool,
        'source_agent_id': agent_id,
        'title': f'{data_type} update for {symbol}',
        'content': f'Market data update: {data_type} for {symbol}',
        'tags': ['market_data', data_type, symbol],
        'metadata': {
            'symbol': symbol,
            'data_type': data_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        },
        'session_id': session_id
    }


def create_error_event(agent_pool: str,
                      agent_id: str,
                      error_type: str,
                      error_message: str,
                      error_details: Optional[Dict[str, Any]] = None,
                      session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to create a standardized error event dictionary.
    
    Args:
        agent_pool: Source agent pool name
        agent_id: Source agent identifier
        error_type: Type of error
        error_message: Error message
        error_details: Optional additional error details
        session_id: Optional session identifier
        
    Returns:
        Dict[str, Any]: Event dictionary ready for logging
    """
    return {
        'event_type': EventType.ERROR.value,
        'log_level': LogLevel.ERROR.value,
        'source_agent_pool': agent_pool,
        'source_agent_id': agent_id,
        'title': f'{error_type} Error',
        'content': error_message,
        'tags': ['error', error_type],
        'metadata': {
            'error_type': error_type,
            'error_message': error_message,
            'error_details': error_details or {},
            'timestamp': datetime.now().isoformat()
        },
        'session_id': session_id
    }
