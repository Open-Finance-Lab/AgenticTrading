from __future__ import annotations

from typing import Protocol, Optional, Dict, Any, List

from ..domain.models import AlphaTask, Ack, AlphaResult


class OrchestratorPort(Protocol):
    """Port for task intake and result subscription.
    
    This port defines the interface for submitting alpha factor generation
    tasks and subscribing to results. Implementations should ensure
    idempotency and proper error handling.
    """

    def submit(self, task: AlphaTask) -> Ack:
        """Submit a task for processing.
        
        Args:
            task: Immutable alpha task containing all required parameters
            
        Returns:
            Acknowledgment with status and tracking information
            
        Raises:
            ValueError: If task validation fails
            RuntimeError: If system is unavailable
        """
        ...

    def subscribe(self, topic: str) -> None:
        """Subscribe to result notifications.
        
        Args:
            topic: Topic name for result streaming
        """
        ...

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a submitted task.
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            Status dictionary or None if task not found
        """
        ...

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task.
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            True if cancellation was successful
        """
        ...

    def list_active_tasks(self) -> List[str]:
        """List all currently active task IDs.
        
        Returns:
            List of active task identifiers
        """
        ...

