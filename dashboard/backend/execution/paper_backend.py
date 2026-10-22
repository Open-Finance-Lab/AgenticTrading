"""PaperBackend — DESIGNED-FOR STUB (spec §4.2, Phase B).

Paper trading has no execution path in the codebase today: AlpacaPaperTradingClient
is read-only and there is no order-submission or step loop. Building this means real
new code (live order submission, a realtime decision-cadence scheduler, live bar
assembly) and needs its own design pass. This stub exists so the parity seam is real
and the eventual drop-in is mechanical.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dashboard.backend.execution.base import ExecutionBackend

_NOT_BUILT = "PaperBackend is a designed-for stub (Phase B); not built in v1."


class PaperBackend(ExecutionBackend):
    loop = "realtime"

    def build_context(self) -> Dict[str, Any]:
        raise NotImplementedError(_NOT_BUILT)

    def apply_decisions(self, actions: List[Dict[str, Any]],
                        expected_step: Optional[int] = None) -> Dict[str, Any]:
        raise NotImplementedError(_NOT_BUILT)

    def status(self) -> Dict[str, Any]:
        raise NotImplementedError(_NOT_BUILT)

    def result(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError(_NOT_BUILT)
