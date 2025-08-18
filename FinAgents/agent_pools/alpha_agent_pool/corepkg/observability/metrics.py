"""Metrics collection for Alpha Agent Pool."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from collections import defaultdict, deque


class Metric(ABC):
    """Base class for metrics."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or {}
    
    @abstractmethod
    def get_value(self) -> float:
        """Get current metric value."""
        pass


class Counter(Metric):
    """Counter metric that only increases."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        super().__init__(name, description, labels)
        self._value = 0.0
        self._lock = threading.RLock()
    
    def increment(self, amount: float = 1.0) -> None:
        """Increment counter by amount."""
        if amount < 0:
            raise ValueError("Counter increment must be >= 0")
        
        with self._lock:
            self._value += amount
    
    def get_value(self) -> float:
        """Get current counter value."""
        with self._lock:
            return self._value


class Gauge(Metric):
    """Gauge metric that can increase or decrease."""
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        super().__init__(name, description, labels)
        self._value = 0.0
        self._lock = threading.RLock()
    
    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value
    
    def increment(self, amount: float = 1.0) -> None:
        """Increment gauge by amount."""
        with self._lock:
            self._value += amount
    
    def decrement(self, amount: float = 1.0) -> None:
        """Decrement gauge by amount."""
        with self._lock:
            self._value -= amount
    
    def get_value(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._value


class Histogram(Metric):
    """Histogram metric for tracking distributions."""
    
    def __init__(self, 
                 name: str, 
                 description: str = "", 
                 labels: Optional[Dict[str, str]] = None,
                 buckets: Optional[List[float]] = None):
        super().__init__(name, description, labels)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._bucket_counts = [0] * len(self.buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.RLock()
    
    def observe(self, value: float) -> None:
        """Observe a value."""
        with self._lock:
            self._sum += value
            self._count += 1
            
            # Update bucket counts
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._bucket_counts[i] += 1
    
    def get_value(self) -> float:
        """Get average value (sum/count)."""
        with self._lock:
            return self._sum / self._count if self._count > 0 else 0.0
    
    def get_percentile(self, percentile: float) -> float:
        """Get approximate percentile value."""
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100")
        
        with self._lock:
            if self._count == 0:
                return 0.0
            
            target_count = (percentile / 100.0) * self._count
            cumulative_count = 0
            
            for i, bucket_count in enumerate(self._bucket_counts):
                cumulative_count += bucket_count
                if cumulative_count >= target_count:
                    return self.buckets[i]
            
            return self.buckets[-1] if self.buckets else 0.0
    
    @property
    def count(self) -> int:
        """Get total observation count."""
        with self._lock:
            return self._count
    
    @property
    def sum(self) -> float:
        """Get sum of all observations."""
        with self._lock:
            return self._sum


class MetricsCollector:
    """Collects and manages metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()
    
    def register_counter(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Counter:
        """Register a new counter metric."""
        metric_key = self._get_metric_key(name, labels)
        
        with self._lock:
            if metric_key in self._metrics:
                metric = self._metrics[metric_key]
                if not isinstance(metric, Counter):
                    raise ValueError(f"Metric {metric_key} is not a Counter")
                return metric
            
            counter = Counter(name, description, labels)
            self._metrics[metric_key] = counter
            return counter
    
    def register_gauge(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Gauge:
        """Register a new gauge metric."""
        metric_key = self._get_metric_key(name, labels)
        
        with self._lock:
            if metric_key in self._metrics:
                metric = self._metrics[metric_key]
                if not isinstance(metric, Gauge):
                    raise ValueError(f"Metric {metric_key} is not a Gauge")
                return metric
            
            gauge = Gauge(name, description, labels)
            self._metrics[metric_key] = gauge
            return gauge
    
    def register_histogram(self, 
                          name: str, 
                          description: str = "", 
                          labels: Optional[Dict[str, str]] = None,
                          buckets: Optional[List[float]] = None) -> Histogram:
        """Register a new histogram metric."""
        metric_key = self._get_metric_key(name, labels)
        
        with self._lock:
            if metric_key in self._metrics:
                metric = self._metrics[metric_key]
                if not isinstance(metric, Histogram):
                    raise ValueError(f"Metric {metric_key} is not a Histogram")
                return metric
            
            histogram = Histogram(name, description, labels, buckets)
            self._metrics[metric_key] = histogram
            return histogram
    
    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """Get a metric by name and labels."""
        metric_key = self._get_metric_key(name, labels)
        
        with self._lock:
            return self._metrics.get(metric_key)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics."""
        with self._lock:
            return self._metrics.copy()
    
    def get_summary(self) -> Dict[str, float]:
        """Get a summary of all metrics values."""
        with self._lock:
            return {name: metric.get_value() for name, metric in self._metrics.items()}
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global metrics collector
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _global_collector


# Convenience functions for common metrics
def increment_counter(name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter metric."""
    counter = _global_collector.register_counter(name, labels=labels)
    counter.increment(amount)


def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Set a gauge metric value."""
    gauge = _global_collector.register_gauge(name, labels=labels)
    gauge.set(value)


def observe_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Observe a value in a histogram metric."""
    histogram = _global_collector.register_histogram(name, labels=labels)
    histogram.observe(value)
