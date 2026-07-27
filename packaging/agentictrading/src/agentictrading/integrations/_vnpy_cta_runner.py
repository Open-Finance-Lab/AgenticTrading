"""Run a local vn.py CTA adapter through ATL's existing typed AgentRunner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from ..runner import AgentRunner
from ..models import RunResult
from ._vnpy_cta_adapter import VnpyCtaAdapter, VnpyCtaDataError
from ._vnpy_cta_core import (
    ARTIFACT_SCHEMA_VERSION,
    VnpyCtaAuditArtifact,
    build_safe_manifest,
    save_audit_artifact,
)


_UNCLEAN_COUNTERS = {
    "atl_rejections",
    "atl_rejection",
    "error_hold",
    "fatal_data_error",
    "local_rejections",
    "run_error",
    "timed_out_orders",
    "timeout_hold",
    "unsupported_actions",
}


@dataclass(frozen=True)
class VnpyCtaRunSummary:
    """Final ATL result plus local vn.py-specific audit evidence."""

    result: RunResult
    artifact: VnpyCtaAuditArtifact
    artifact_path: Path
    artifact_sha256: str
    clean: bool

    @property
    def run_id(self) -> Optional[str]:
        return self.result.run_id

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict(self.result.metrics)

    @property
    def diagnostics(self) -> Dict[str, int]:
        return dict(self.artifact.summary)

    @property
    def compare_url(self) -> Optional[str]:
        value = self.result.raw.get("compare_url") if self.result.raw else None
        return str(value) if value else None


class VnpyCtaATLRunner:
    """Thin orchestration wrapper; HTTP and step semantics stay in AgentRunner."""

    def __init__(
        self,
        client: Any,
        adapter: VnpyCtaAdapter,
        *,
        artifact_path: Path | str,
    ) -> None:
        self.client = client
        self.adapter = adapter
        self.artifact_path = Path(artifact_path).expanduser()

    @staticmethod
    def _dates(start_date: str, end_date: str) -> None:
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise ValueError("start_date and end_date must use YYYY-MM-DD") from exc
        if end < start:
            raise ValueError("end_date must not be before start_date")

    def _save_partial(self, error: Exception) -> None:
        status = "fatal_data_error" if isinstance(error, VnpyCtaDataError) else "run_error"
        self.adapter.abort(error, status=status)
        save_audit_artifact(self.adapter.finalize_artifact(), self.artifact_path)

    def run_backtest(
        self,
        *,
        agent_version_id: str,
        start_date: str,
        end_date: str,
        initial_cash: Optional[float] = None,
        poll_interval: float = 2.0,
        max_wait_seconds: Optional[float] = 300.0,
    ) -> VnpyCtaRunSummary:
        self._dates(start_date, end_date)
        config = build_safe_manifest(
            {
                "integration": "vnpy_cta",
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "integration_manifest": self.adapter.manifest,
            }
        )
        runner = AgentRunner(self.client, self.adapter)
        try:
            result = runner.run_backtest(
                agent_version_id,
                environment_id="us-equity-hourly-v1",
                start_date=start_date,
                end_date=end_date,
                symbols=["AAPL"],
                initial_cash=initial_cash,
                config=config,
                poll_interval=poll_interval,
                max_wait_seconds=max_wait_seconds,
            )
        except Exception as exc:
            self._save_partial(exc)
            raise

        artifact = self.adapter.finalize_artifact()
        digest = save_audit_artifact(artifact, self.artifact_path)
        clean = not any(
            artifact.summary.get(counter, 0) > 0 for counter in _UNCLEAN_COUNTERS
        )
        return VnpyCtaRunSummary(
            result=result,
            artifact=artifact,
            artifact_path=self.artifact_path,
            artifact_sha256=digest,
            clean=clean,
        )


__all__ = ["VnpyCtaRunSummary", "VnpyCtaATLRunner"]
