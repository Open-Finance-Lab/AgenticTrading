from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, UTC

class PlannerMessageType(Enum):
    """Types of messages that can be exchanged with the DAG Planner"""
    QUERY = "query"                    # User query for DAG planning
    MEMORY_UPDATE = "memory_update"    # Memory agent updates
    DAG_REQUEST = "dag_request"        # Request for DAG execution
    DAG_RESPONSE = "dag_response"      # Response with planned DAG
    STATUS_UPDATE = "status_update"    # Status updates
    ERROR = "error"                    # Error messages

@dataclass
class PlannerMessage:
    """Base message structure for DAG Planner communication"""
    message_type: PlannerMessageType
    timestamp: datetime
    payload: Dict[str, Any]
    source: str
    correlation_id: Optional[str] = None

    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps({
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "source": self.source,
            "correlation_id": self.correlation_id
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'PlannerMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        message_type = PlannerMessageType(data["message_type"])
        
        # Create appropriate message type based on message_type
        if message_type == PlannerMessageType.QUERY:
            return QueryMessage(
                query=data["payload"]["query"],
                context=data["payload"]["context"],
                source=data["source"],
                correlation_id=data.get("correlation_id")
            )
        elif message_type == PlannerMessageType.MEMORY_UPDATE:
            return MemoryUpdateMessage(
                memory_data=data["payload"]["memory_data"],
                source=data["source"],
                correlation_id=data.get("correlation_id")
            )
        elif message_type == PlannerMessageType.DAG_RESPONSE:
            return DAGResponseMessage(
                dag=data["payload"]["dag"],
                metadata=data["payload"]["metadata"],
                correlation_id=data["correlation_id"],
                source=data["source"]
            )
        else:
            # For other message types, return base PlannerMessage
            return cls(
                message_type=message_type,
                timestamp=datetime.fromisoformat(data["timestamp"]),
                payload=data["payload"],
                source=data["source"],
                correlation_id=data.get("correlation_id")
            )

    def __init__(
        self,
        message_type: PlannerMessageType,
        payload: Dict[str, Any],
        source: str,
        correlation_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.message_type = message_type
        self.payload = payload
        self.source = source
        self.correlation_id = correlation_id
        self.timestamp = timestamp or datetime.now(UTC)

@dataclass
class QueryMessage(PlannerMessage):
    """Message containing user query for DAG planning"""
    def __init__(
        self,
        query: str,
        context: Dict[str, Any],
        source: str = "client",
        correlation_id: Optional[str] = None
    ):
        super().__init__(
            message_type=PlannerMessageType.QUERY,
            payload={"query": query, "context": context},
            source=source,
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC)
        )

@dataclass
class MemoryUpdateMessage(PlannerMessage):
    """Message containing memory agent updates"""
    def __init__(
        self,
        memory_data: Dict[str, Any],
        source: str = "memory_agent",
        correlation_id: Optional[str] = None
    ):
        super().__init__(
            message_type=PlannerMessageType.MEMORY_UPDATE,
            payload={"memory_data": memory_data},
            source=source,
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC)
        )

@dataclass
class DAGResponseMessage(PlannerMessage):
    """Message containing the planned DAG"""
    def __init__(
        self,
        dag: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
        correlation_id: str,
        source: str = "planner"
    ):
        super().__init__(
            message_type=PlannerMessageType.DAG_RESPONSE,
            payload={"dag": dag, "metadata": metadata},
            source=source,
            correlation_id=correlation_id,
            timestamp=datetime.now(UTC)
        ) 