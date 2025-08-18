"""Backpressure policies for load management."""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional


class BackpressureError(Exception):
    """Exception raised when backpressure limit is exceeded."""
    pass


class BackpressurePolicy(ABC):
    """Abstract base class for backpressure policies."""

    @abstractmethod
    def should_accept(self, current_load: int) -> bool:
        """Determine if new request should be accepted.
        
        Args:
            current_load: Current system load metric
            
        Returns:
            True if request should be accepted
        """
        pass

    @abstractmethod
    def get_rejection_reason(self) -> str:
        """Get reason for request rejection."""
        pass


class QueueBackpressure(BackpressurePolicy):
    """Queue-based backpressure policy."""

    def __init__(self,
                 max_queue_size: int = 100,
                 warning_threshold: float = 0.8):
        """Initialize queue backpressure policy.
        
        Args:
            max_queue_size: Maximum allowed queue size
            warning_threshold: Warning threshold as fraction of max size
        """
        self.max_queue_size = max_queue_size
        self.warning_threshold = warning_threshold
        self._current_size = 0
        self._lock = threading.RLock()

    def should_accept(self, current_load: int) -> bool:
        """Check if request should be accepted based on queue size."""
        with self._lock:
            self._current_size = current_load
            return current_load < self.max_queue_size

    def get_rejection_reason(self) -> str:
        """Get rejection reason."""
        return f"Queue size ({self._current_size}) exceeds limit ({self.max_queue_size})"

    def is_warning_level(self) -> bool:
        """Check if current load is at warning level."""
        with self._lock:
            return self._current_size >= (self.max_queue_size * self.warning_threshold)


class RateLimitBackpressure(BackpressurePolicy):
    """Rate limiting backpressure policy."""

    def __init__(self,
                 max_requests_per_second: float = 10.0,
                 window_size: float = 1.0):
        """Initialize rate limit backpressure policy.
        
        Args:
            max_requests_per_second: Maximum requests per second
            window_size: Time window size in seconds
        """
        self.max_requests_per_second = max_requests_per_second
        self.window_size = window_size
        self._requests: list[float] = []
        self._lock = threading.RLock()

    def should_accept(self, current_load: int) -> bool:
        """Check if request should be accepted based on rate limit."""
        current_time = time.time()
        
        with self._lock:
            # Remove old requests outside the window
            cutoff_time = current_time - self.window_size
            self._requests = [req_time for req_time in self._requests if req_time > cutoff_time]
            
            # Check if we can accept new request
            if len(self._requests) < (self.max_requests_per_second * self.window_size):
                self._requests.append(current_time)
                return True
            
            return False

    def get_rejection_reason(self) -> str:
        """Get rejection reason."""
        return f"Rate limit exceeded ({self.max_requests_per_second} req/s)"


class AdaptiveBackpressure(BackpressurePolicy):
    """Adaptive backpressure policy that adjusts based on system performance."""

    def __init__(self,
                 initial_limit: int = 50,
                 min_limit: int = 10,
                 max_limit: int = 200,
                 adjustment_factor: float = 0.1):
        """Initialize adaptive backpressure policy.
        
        Args:
            initial_limit: Initial capacity limit
            min_limit: Minimum capacity limit
            max_limit: Maximum capacity limit
            adjustment_factor: Factor for adjusting limits
        """
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.adjustment_factor = adjustment_factor
        self._current_limit = initial_limit
        self._recent_success_rate = 1.0
        self._lock = threading.RLock()

    def should_accept(self, current_load: int) -> bool:
        """Check if request should be accepted based on adaptive limit."""
        with self._lock:
            return current_load < self._current_limit

    def update_success_rate(self, success_rate: float) -> None:
        """Update recent success rate and adjust limits.
        
        Args:
            success_rate: Recent success rate (0.0 to 1.0)
        """
        with self._lock:
            self._recent_success_rate = success_rate
            
            if success_rate > 0.9:
                # High success rate, increase capacity
                adjustment = self._current_limit * self.adjustment_factor
                self._current_limit = min(self.max_limit, self._current_limit + adjustment)
            elif success_rate < 0.7:
                # Low success rate, decrease capacity
                adjustment = self._current_limit * self.adjustment_factor
                self._current_limit = max(self.min_limit, self._current_limit - adjustment)

    def get_rejection_reason(self) -> str:
        """Get rejection reason."""
        return f"Adaptive limit exceeded (current: {self._current_limit})"

    @property
    def current_limit(self) -> int:
        """Get current adaptive limit."""
        with self._lock:
            return int(self._current_limit)
