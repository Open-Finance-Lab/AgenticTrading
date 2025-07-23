#!/usr/bin/env python3
"""
Real-time Stream Processing System for FinAgent Memory

This module implements real-time event streaming, processing pipelines,
and reactive memory management for the FinAgent ecosystem.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque
import websockets
from asyncio import Queue

# Try to import Redis for stream processing
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Memory event types"""
    SIGNAL_RECEIVED = "signal_received"
    STRATEGY_SHARED = "strategy_shared"
    PERFORMANCE_UPDATED = "performance_updated"
    AGENT_CONNECTED = "agent_connected"
    AGENT_DISCONNECTED = "agent_disconnected"
    MEMORY_INDEXED = "memory_indexed"
    PATTERN_DETECTED = "pattern_detected"
    ALERT_TRIGGERED = "alert_triggered"

@dataclass
class MemoryEvent:
    """Memory system event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    agent_id: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    priority: int = 0  # Higher = more important

class StreamProcessor:
    """
    Real-time stream processor for memory events
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize stream processor
        
        Args:
            redis_url: Redis connection URL for streaming
        """
        self.redis_url = redis_url
        self.redis_client = None
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.event_queue = Queue(maxsize=1000)
        self.processing_stats = {
            "events_processed": 0,
            "events_per_second": 0,
            "last_reset_time": time.time()
        }
        self.is_running = False
        
        # Event buffer for batch processing
        self.event_buffer = deque(maxlen=100)
        
        logger.info("Real-time stream processor initialized")
    
    async def connect(self) -> bool:
        """Connect to Redis streams"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory processing")
            return True
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Connected to Redis for stream processing")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """
        Register event handler
        
        Args:
            event_type: Type of event to handle
            handler: Async function to handle the event
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")
    
    async def publish_event(self, event: MemoryEvent):
        """
        Publish event to stream
        
        Args:
            event: Memory event to publish
        """
        try:
            # Add to local queue
            await self.event_queue.put(event)
            
            # Publish to Redis if available
            if self.redis_client:
                event_data = {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "agent_id": event.agent_id,
                    "payload": json.dumps(event.payload),
                    "metadata": json.dumps(event.metadata),
                    "priority": event.priority
                }
                
                await self.redis_client.xadd(
                    f"memory_events:{event.event_type.value}",
                    event_data
                )
            
            logger.debug(f"Published event {event.event_id}: {event.event_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
    
    async def start_processing(self):
        """Start event processing loop"""
        self.is_running = True
        logger.info("🚀 Starting real-time event processing")
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self._process_event_queue()),
            asyncio.create_task(self._update_stats()),
            asyncio.create_task(self._batch_process_events())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Event processing error: {e}")
        finally:
            self.is_running = False
    
    async def stop_processing(self):
        """Stop event processing"""
        self.is_running = False
        logger.info("🛑 Stopping event processing")
    
    async def _process_event_queue(self):
        """Process events from queue"""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                # Add to buffer for batch processing
                self.event_buffer.append(event)
                
                # Process event handlers
                await self._handle_event(event)
                
                # Update stats
                self.processing_stats["events_processed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: MemoryEvent):
        """Handle individual event"""
        if event.event_type in self.event_handlers:
            for handler in self.event_handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")
    
    async def _batch_process_events(self):
        """Process events in batches for analytics"""
        while self.is_running:
            try:
                await asyncio.sleep(5)  # Process every 5 seconds
                
                if self.event_buffer:
                    events = list(self.event_buffer)
                    self.event_buffer.clear()
                    
                    # Analyze event patterns
                    await self._analyze_event_patterns(events)
                    
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    async def _analyze_event_patterns(self, events: List[MemoryEvent]):
        """Analyze patterns in event stream"""
        if not events:
            return
        
        # Count events by type
        event_counts = {}
        agent_activity = {}
        
        for event in events:
            event_type = event.event_type.value
            agent_id = event.agent_id
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            agent_activity[agent_id] = agent_activity.get(agent_id, 0) + 1
        
        # Detect high activity patterns
        for agent_id, count in agent_activity.items():
            if count > 10:  # High activity threshold
                await self._trigger_alert("high_agent_activity", {
                    "agent_id": agent_id,
                    "event_count": count,
                    "time_window": "5_seconds"
                })
        
        logger.debug(f"Analyzed {len(events)} events: {event_counts}")
    
    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """Trigger system alert"""
        alert_event = MemoryEvent(
            event_id=f"alert_{int(time.time())}",
            event_type=EventType.ALERT_TRIGGERED,
            timestamp=datetime.utcnow(),
            agent_id="system",
            payload={"alert_type": alert_type, "data": data},
            metadata={},
            priority=10
        )
        
        await self.publish_event(alert_event)
    
    async def _update_stats(self):
        """Update processing statistics"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds
                
                current_time = time.time()
                time_diff = current_time - self.processing_stats["last_reset_time"]
                
                if time_diff > 0:
                    events_per_second = self.processing_stats["events_processed"] / time_diff
                    self.processing_stats["events_per_second"] = events_per_second
                    self.processing_stats["last_reset_time"] = current_time
                    self.processing_stats["events_processed"] = 0
                
            except Exception as e:
                logger.error(f"Stats update error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "is_running": self.is_running,
            "queue_size": self.event_queue.qsize(),
            "buffer_size": len(self.event_buffer),
            "events_per_second": self.processing_stats["events_per_second"],
            "total_handlers": sum(len(handlers) for handlers in self.event_handlers.values()),
            "handler_types": list(self.event_handlers.keys())
        }


class WebSocketManager:
    """
    WebSocket manager for real-time client connections
    """
    
    def __init__(self):
        """Initialize WebSocket manager"""
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscriptions: Dict[str, List[EventType]] = {}
        
    async def register_client(self, client_id: str, websocket: websockets.WebSocketServerProtocol):
        """Register WebSocket client"""
        self.connections[client_id] = websocket
        self.subscriptions[client_id] = []
        logger.info(f"Client {client_id} connected")
    
    async def unregister_client(self, client_id: str):
        """Unregister WebSocket client"""
        if client_id in self.connections:
            del self.connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"Client {client_id} disconnected")
    
    async def subscribe_client(self, client_id: str, event_types: List[EventType]):
        """Subscribe client to event types"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id] = event_types
            logger.info(f"Client {client_id} subscribed to {[et.value for et in event_types]}")
    
    async def broadcast_event(self, event: MemoryEvent):
        """Broadcast event to subscribed clients"""
        event_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "agent_id": event.agent_id,
            "payload": event.payload,
            "metadata": event.metadata,
            "priority": event.priority
        }
        
        message = json.dumps(event_data)
        
        # Send to subscribed clients
        for client_id, event_types in self.subscriptions.items():
            if event.event_type in event_types:
                websocket = self.connections.get(client_id)
                if websocket:
                    try:
                        await websocket.send(message)
                    except Exception as e:
                        logger.error(f"Failed to send to client {client_id}: {e}")
                        await self.unregister_client(client_id)


class ReactiveMemoryManager:
    """
    Reactive memory manager with event-driven processing
    """
    
    def __init__(self, stream_processor: StreamProcessor):
        """
        Initialize reactive memory manager
        
        Args:
            stream_processor: Stream processor instance
        """
        self.stream_processor = stream_processor
        self.websocket_manager = WebSocketManager()
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("Reactive memory manager initialized")
    
    def _register_handlers(self):
        """Register event handlers"""
        self.stream_processor.register_handler(
            EventType.SIGNAL_RECEIVED,
            self._handle_signal_received
        )
        
        self.stream_processor.register_handler(
            EventType.STRATEGY_SHARED,
            self._handle_strategy_shared
        )
        
        self.stream_processor.register_handler(
            EventType.PERFORMANCE_UPDATED,
            self._handle_performance_updated
        )
        
        self.stream_processor.register_handler(
            EventType.ALERT_TRIGGERED,
            self._handle_alert_triggered
        )
    
    async def _handle_signal_received(self, event: MemoryEvent):
        """Handle signal received event"""
        payload = event.payload
        
        # Extract signal information
        signal_data = payload.get("signal_data", {})
        symbol = signal_data.get("symbol")
        signal_type = payload.get("signal_type")
        
        logger.info(f"Processing signal: {signal_type} for {symbol} from {event.agent_id}")
        
        # Broadcast to WebSocket clients
        await self.websocket_manager.broadcast_event(event)
        
        # Trigger pattern detection if needed
        if signal_data.get("confidence", 0) > 0.9:
            await self._trigger_pattern_detection(event)
    
    async def _handle_strategy_shared(self, event: MemoryEvent):
        """Handle strategy shared event"""
        payload = event.payload
        strategy_type = payload.get("strategy_type")
        
        logger.info(f"Processing strategy: {strategy_type} from {event.agent_id}")
        
        # Broadcast to clients
        await self.websocket_manager.broadcast_event(event)
    
    async def _handle_performance_updated(self, event: MemoryEvent):
        """Handle performance update event"""
        payload = event.payload
        returns = payload.get("returns", 0)
        
        logger.info(f"Performance update: {returns:.2%} returns from {event.agent_id}")
        
        # Broadcast to clients
        await self.websocket_manager.broadcast_event(event)
    
    async def _handle_alert_triggered(self, event: MemoryEvent):
        """Handle alert triggered event"""
        payload = event.payload
        alert_type = payload.get("alert_type")
        
        logger.warning(f"Alert triggered: {alert_type}")
        
        # Broadcast high-priority alert
        await self.websocket_manager.broadcast_event(event)
    
    async def _trigger_pattern_detection(self, event: MemoryEvent):
        """Trigger pattern detection for high-confidence signals"""
        pattern_event = MemoryEvent(
            event_id=f"pattern_{int(time.time())}",
            event_type=EventType.PATTERN_DETECTED,
            timestamp=datetime.utcnow(),
            agent_id="pattern_detector",
            payload={
                "pattern_type": "high_confidence_signal",
                "trigger_event": event.event_id,
                "confidence": event.payload.get("signal_data", {}).get("confidence")
            },
            metadata={"source_agent": event.agent_id},
            priority=5
        )
        
        await self.stream_processor.publish_event(pattern_event)
    
    async def start(self):
        """Start reactive memory manager"""
        logger.info("🚀 Starting reactive memory manager")
        await self.stream_processor.start_processing()
    
    async def stop(self):
        """Stop reactive memory manager"""
        logger.info("🛑 Stopping reactive memory manager")
        await self.stream_processor.stop_processing()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "stream_processor": self.stream_processor.get_stats(),
            "websocket_connections": len(self.websocket_manager.connections),
            "active_subscriptions": len(self.websocket_manager.subscriptions),
            "timestamp": datetime.utcnow().isoformat()
        }


# Test and demonstration functions
async def test_stream_processing():
    """Test stream processing system"""
    print("🧪 Testing Real-time Stream Processing")
    print("=" * 40)
    
    # Initialize components
    processor = StreamProcessor()
    await processor.connect()
    
    reactive_manager = ReactiveMemoryManager(processor)
    
    # Test event publishing
    test_events = [
        MemoryEvent(
            event_id="test_001",
            event_type=EventType.SIGNAL_RECEIVED,
            timestamp=datetime.utcnow(),
            agent_id="test_alpha_agent",
            payload={
                "signal_type": "buy_signal",
                "signal_data": {
                    "symbol": "AAPL",
                    "confidence": 0.95,
                    "price": 190.50
                }
            },
            metadata={},
            priority=5
        ),
        MemoryEvent(
            event_id="test_002",
            event_type=EventType.STRATEGY_SHARED,
            timestamp=datetime.utcnow(),
            agent_id="test_strategy_agent",
            payload={
                "strategy_type": "momentum_trading",
                "performance_metrics": {
                    "returns": 0.15,
                    "sharpe_ratio": 1.2
                }
            },
            metadata={},
            priority=3
        )
    ]
    
    # Publish test events
    for event in test_events:
        await processor.publish_event(event)
        await asyncio.sleep(0.1)
    
    # Let processor run briefly
    processing_task = asyncio.create_task(processor.start_processing())
    await asyncio.sleep(2)
    await processor.stop_processing()
    
    # Show stats
    stats = reactive_manager.get_system_stats()
    print(f"\n📊 System Statistics:")
    print(f"  Events processed: {stats['stream_processor']['events_per_second']:.1f}/sec")
    print(f"  Queue size: {stats['stream_processor']['queue_size']}")
    print(f"  Buffer size: {stats['stream_processor']['buffer_size']}")
    print(f"  Total handlers: {stats['stream_processor']['total_handlers']}")
    
    print("\n✅ Stream processing test completed!")


if __name__ == "__main__":
    asyncio.run(test_stream_processing())
