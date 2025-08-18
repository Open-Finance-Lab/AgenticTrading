"""Retry policy implementations for resilient operations."""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Type, Union


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""

    @abstractmethod
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if operation should be retried.
        
        Args:
            attempt: Current attempt number (starting from 1)
            exception: Exception that caused the failure
            
        Returns:
            True if operation should be retried
        """
        pass

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay in seconds before next retry.
        
        Args:
            attempt: Current attempt number (starting from 1)
            
        Returns:
            Delay in seconds
        """
        pass


class ExponentialBackoffRetry(RetryPolicy):
    """Exponential backoff retry policy with jitter."""

    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True,
                 retryable_exceptions: Optional[tuple] = None):
        """Initialize exponential backoff retry policy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            backoff_multiplier: Multiplier for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that should trigger retries
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Check if retry should be attempted."""
        if attempt >= self.max_attempts:
            return False
        
        return isinstance(exception, self.retryable_exceptions)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = min(
            self.base_delay * (self.backoff_multiplier ** (attempt - 1)),
            self.max_delay
        )
        
        if self.jitter:
            # Add up to 25% jitter to prevent thundering herd
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
            
        return delay


async def retry_async(
    operation: Callable[..., Any],
    policy: RetryPolicy,
    *args,
    **kwargs
) -> Any:
    """Execute an async operation with retry policy.
    
    Args:
        operation: Async function to execute
        policy: Retry policy to use
        *args: Arguments to pass to operation
        **kwargs: Keyword arguments to pass to operation
        
    Returns:
        Result of successful operation
        
    Raises:
        The last exception if all retries are exhausted
    """
    attempt = 0
    last_exception = None
    
    while True:
        attempt += 1
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if not policy.should_retry(attempt, e):
                raise e
            
            delay = policy.get_delay(attempt)
            if delay > 0:
                await asyncio.sleep(delay)


def retry_sync(
    operation: Callable[..., Any],
    policy: RetryPolicy,
    *args,
    **kwargs
) -> Any:
    """Execute a sync operation with retry policy.
    
    Args:
        operation: Function to execute
        policy: Retry policy to use
        *args: Arguments to pass to operation
        **kwargs: Keyword arguments to pass to operation
        
    Returns:
        Result of successful operation
        
    Raises:
        The last exception if all retries are exhausted
    """
    attempt = 0
    last_exception = None
    
    while True:
        attempt += 1
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            last_exception = e
            
            if not policy.should_retry(attempt, e):
                raise e
            
            delay = policy.get_delay(attempt)
            if delay > 0:
                time.sleep(delay)
