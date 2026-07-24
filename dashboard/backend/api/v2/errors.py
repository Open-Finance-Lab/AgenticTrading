"""Uniform error envelope for /api/v2 (spec §5.4)."""

from __future__ import annotations

import math

from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _sanitize_nonfinite(obj):
    """Recursively replace non-finite floats (inf/-inf/nan) with their string
    form so a payload can always be JSON-serialized."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj

ERROR_CODES = [
    "validation_failed", "step_already_closed", "run_not_found", "agent_not_found",
    "unauthorized", "forbidden_scope", "rate_limited", "universe_violation",
    "insufficient_cash", "invalid_symbol", "invalid_status", "too_many_active_runs",
    "too_many_active_runs_global",
]


class ApiError(Exception):
    """Raised anywhere in v2; rendered as {"error": {...}} by api_error_handler."""

    def __init__(self, code: str, message: str, status: int = 400,
                 details: Optional[Dict[str, Any]] = None, retryable: bool = False,
                 headers: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details
        self.retryable = retryable
        # Response headers to carry on the rendered envelope. The handler
        # builds a fresh JSONResponse, so headers set on the per-request
        # Response object before the raise would otherwise be dropped.
        self.headers = headers or {}

    def to_envelope(self) -> Dict[str, Any]:
        return {"error": {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }}


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    headers = {name: str(value) for name, value in exc.headers.items()}
    if exc.code == "rate_limited" and exc.details and "retry_after" in exc.details:
        headers["Retry-After"] = str(exc.details["retry_after"])
    return JSONResponse(status_code=exc.status, content=exc.to_envelope(), headers=headers)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Render request-body validation errors through the uniform envelope (spec §5.4).

    Scoped to /api/v2 so the legacy v1 surface keeps FastAPI's default {"detail": ...}.

    Validation errors echo the offending input back. When that input is a
    non-finite float (Python's lenient JSON parser accepts the ``Infinity``
    token), ``JSONResponse``'s ``json.dumps(allow_nan=False)`` would blow up
    and turn a clean 422 into a confusing 500 — sanitize both branches.
    """
    errors = _sanitize_nonfinite(jsonable_encoder(exc.errors()))
    if request.url.path.startswith("/api/v2"):
        return JSONResponse(status_code=422, content={"error": {
            "code": "validation_failed",
            "message": "Request validation failed",
            "details": {"errors": errors},
            "retryable": False,
        }})
    return JSONResponse(status_code=422, content={"detail": errors})
