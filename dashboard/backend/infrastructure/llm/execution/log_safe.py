"""One rule for the identifiers ``ERROR: llm.`` lines print.

Those lines are ``key=value`` pairs that the parent relays to the service log
live and passes through ``_redact_credentials``. A value carrying a space,
``=``, ``:`` or a newline would break that shape -- or forge a second line --
so every reporter prints identifiers through ``log_safe_token`` rather than
keeping its own pattern.
"""

from __future__ import annotations

import re

_LOG_SAFE_TOKEN = re.compile(r"[A-Za-z0-9._-]{1,128}")


def log_safe_token(value: object, fallback: str = "-") -> str:
    """``value`` when it is a log-safe identifier, else ``fallback``.

    Validates, never rewrites: a sanitised user label can collide with a real
    identifier (``"复盘"`` and ``"_"`` reduce to the same token), and a
    collision in an operator log is worse than an honest ``-``. ``fullmatch``
    rather than ``match`` with ``$``, which accepts a trailing newline.
    """
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        value = str(value)
    if isinstance(value, str) and _LOG_SAFE_TOKEN.fullmatch(value):
        return value
    return fallback


__all__ = ["log_safe_token"]
