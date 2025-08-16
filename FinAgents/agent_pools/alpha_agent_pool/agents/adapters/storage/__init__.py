"""Storage adapters for result publishing and persistence."""

from .outbox_adapter import FileOutboxAdapter

__all__ = ["FileOutboxAdapter"]
