"""Structured logging for Alpha Agent Pool."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            "timestamp": datetime.now().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add correlation ID if available
        correlation_id = getattr(threading.current_thread(), "correlation_id", None)
        if correlation_id:
            log_data["correlation_id"] = correlation_id
        
        return json.dumps(log_data, ensure_ascii=False)


class StructuredLogger:
    """Structured logger with context support and file logging."""
    
    def __init__(self, name: str, level: str = "INFO", log_to_file: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file:
            self._setup_file_handler(name)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _setup_file_handler(self, name: str):
        """Setup file handler for logging."""
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Create log file path
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"{name}_{timestamp}.log"
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)
            
            # Log the file path
            print(f"📝 Logging to file: {log_file}")
            
        except Exception as e:
            print(f"⚠️  Failed to setup file logging: {e}")
    
    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log message with additional context."""
        extra_fields = {}
        
        # Add provided context
        for key, value in kwargs.items():
            if key not in ("exc_info", "stack_info", "stacklevel"):
                extra_fields[key] = value
        
        # Create log record with extra fields
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=kwargs.get("exc_info"),
        )
        record.extra_fields = extra_fields
        
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def log_task_start(self, task_id: str, strategy_id: str, **kwargs) -> None:
        """Log task start event."""
        self.info(
            "Task started",
            task_id=task_id,
            strategy_id=strategy_id,
            event_type="task_start",
            **kwargs
        )
    
    def log_task_complete(self, task_id: str, duration_ms: float, **kwargs) -> None:
        """Log task completion event."""
        self.info(
            "Task completed",
            task_id=task_id,
            duration_ms=duration_ms,
            event_type="task_complete",
            **kwargs
        )
    
    def log_task_error(self, task_id: str, error: str, **kwargs) -> None:
        """Log task error event."""
        self.error(
            "Task failed",
            task_id=task_id,
            error=error,
            event_type="task_error",
            **kwargs
        )
    
    def log_strategy_execution(self, strategy_id: str, duration_ms: float, **kwargs) -> None:
        """Log strategy execution event."""
        self.info(
            "Strategy executed",
            strategy_id=strategy_id,
            duration_ms=duration_ms,
            event_type="strategy_execution",
            **kwargs
        )
    
    def log_feature_fetch(self, feature_name: str, duration_ms: float, **kwargs) -> None:
        """Log feature fetch event."""
        self.info(
            "Feature fetched",
            feature_name=feature_name,
            duration_ms=duration_ms,
            event_type="feature_fetch",
            **kwargs
        )


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}
_logger_lock = threading.RLock()


def get_logger(name: str, level: str = "INFO", log_to_file: bool = True) -> StructuredLogger:
    """Get or create a structured logger.
    
    Args:
        name: Logger name
        level: Log level
        log_to_file: Whether to log to file
        
    Returns:
        StructuredLogger instance
    """
    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = StructuredLogger(name, level, log_to_file)
        return _loggers[name]


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current thread.
    
    Args:
        correlation_id: Unique identifier for correlating logs
    """
    threading.current_thread().correlation_id = correlation_id


def get_correlation_id() -> Optional[str]:
    """Get correlation ID for current thread.
    
    Returns:
        Correlation ID or None if not set
    """
    return getattr(threading.current_thread(), "correlation_id", None)
