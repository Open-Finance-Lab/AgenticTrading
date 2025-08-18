"""Timeout policy implementation for operation timeouts."""

from __future__ import annotations

import asyncio
import signal
import threading
from typing import Any, Callable, Optional


class TimeoutError(Exception):
    """Exception raised when operation times out."""
    pass


class TimeoutPolicy:
    """Policy for managing operation timeouts."""

    def __init__(self, default_timeout: float = 30.0):
        """Initialize timeout policy.
        
        Args:
            default_timeout: Default timeout in seconds
        """
        self.default_timeout = default_timeout

    async def call_async(self,
                        func: Callable[..., Any],
                        timeout: Optional[float] = None,
                        *args,
                        **kwargs) -> Any:
        """Execute async function with timeout.
        
        Args:
            func: Async function to execute
            timeout: Timeout in seconds (uses default if None)
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If operation times out
        """
        actual_timeout = timeout or self.default_timeout
        
        try:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=actual_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {actual_timeout} seconds")

    def call_sync(self,
                  func: Callable[..., Any],
                  timeout: Optional[float] = None,
                  *args,
                  **kwargs) -> Any:
        """Execute sync function with timeout.
        
        Args:
            func: Function to execute
            timeout: Timeout in seconds (uses default if None)
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If operation times out
        """
        actual_timeout = timeout or self.default_timeout
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=actual_timeout)
        
        if thread.is_alive():
            # Thread is still running, operation timed out
            raise TimeoutError(f"Operation timed out after {actual_timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
