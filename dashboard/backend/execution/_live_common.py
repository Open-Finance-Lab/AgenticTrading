"""Shared pure helpers for the live-money broker orchestration modules.

``execution.alpaca_live_service`` and ``execution.robinhood_live_service`` are
both live-money paths (every order they emit can reach a real brokerage
account) that evolved in parallel and ended up with byte-identical helpers for
timestamping, audit-log writing, quantity flooring and rejection-record
shaping. This module is the single source of truth for those bodies.

Each service module keeps its own module-level function of the same name
(``_utcnow_iso``, ``_audit_sync``, ``_audit``, ``_floor_quantity``,
``_rejection``) as a thin wrapper delegating here, so existing tests that
monkeypatch those module-level names (e.g. ``AUDIT_DIR``) keep working
unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def audit_sync(audit_dir: Path, logger: logging.Logger, label: str, event: str, payload: Dict[str, Any]) -> None:
    """Append one audit record under ``audit_dir``. Never raises -- an
    unwritable audit dir must not abort a live run mid-flight."""
    try:
        audit_dir.mkdir(parents=True, exist_ok=True)
        record = {"ts": utcnow_iso(), "event": event, **payload}
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = audit_dir / f"audit_{day}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.exception("%s audit write failed for event %s", label, event)


async def audit(audit_dir: Path, logger: logging.Logger, label: str, event: str, payload: Dict[str, Any]) -> None:
    """Async wrapper: the audit write is blocking file I/O, so keep it off the loop."""
    await asyncio.to_thread(audit_sync, audit_dir, logger, label, event, payload)


def floor_quantity(qty: float) -> float:
    """Floor to 4 decimal places.

    Rounding *down* matters: round-half-up could push the resulting notional
    back above the USD cap that produced the quantity in the first place.
    """
    try:
        value = float(qty)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return math.floor(value * 10000.0) / 10000.0


def rejection(order: Dict[str, Any], reason: str, detail: str) -> Dict[str, Any]:
    return {"order": order, "reason": reason, "detail": detail}
