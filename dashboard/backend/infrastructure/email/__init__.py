"""Email infrastructure: transactional-mail provider adapters.

Naming this package ``email`` does not shadow the standard library. Python 3
uses absolute imports, so a bare ``import email`` anywhere in the codebase
still resolves to stdlib; only ``dashboard.backend.infrastructure.email``
reaches this package.
"""
