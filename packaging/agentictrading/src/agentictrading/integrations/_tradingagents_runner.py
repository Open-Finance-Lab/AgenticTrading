"""ATLClient orchestration for replaying TradingAgents artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Tuple

from ..exceptions import ATLAPIError, ATLConflictError, ATLRunFailedError
from ..models import RunResult
from ..runner import _FAILED_STATES, _STEP_AUTOHELD_CODES, wait_for_poll
from ._tradingagents_core import (
    ArtifactValidationError,
    TradingAgentsDecisionArtifact,
    TradingAgentsGenerationError,
    _canonical_date,
)
from ._tradingagents_replay import (
    TradingAgentsReplayDiagnostics,
    TradingAgentsReplayPlanner,
    TradingAgentsReplayValidationError,
)


class TradingAgentsReplayIncompleteError(RuntimeError):
    """Raised after a run completes without reaching every artifact record."""

    def __init__(
        self,
        *,
        run_id: str,
        analysis_dates: Tuple[str, ...],
        result: RunResult,
        diagnostics: TradingAgentsReplayDiagnostics,
    ) -> None:
        self.run_id = run_id
        self.analysis_dates = analysis_dates
        self.result = result
        self.diagnostics = diagnostics
        joined = ", ".join(analysis_dates)
        super().__init__(
            f"ATL run {run_id} completed before these TradingAgents records "
            f"became executable: {joined}"
        )


@dataclass(frozen=True)
class TradingAgentsATLRunOutcome:
    """Completed ATL result plus local replay and execution diagnostics."""

    result: RunResult
    replay: TradingAgentsReplayDiagnostics
    fills: Tuple[Dict[str, Any], ...]
    rejections: Tuple[Dict[str, Any], ...]

    @property
    def run_id(self) -> Optional[str]:
        return self.result.run_id

    @property
    def fills_count(self) -> int:
        return len(self.fills)

    @property
    def timeout_holds(self) -> int:
        """Auto-held steps as counted by ATL (server-side metric)."""
        try:
            return int(self.result.metrics.get("timeout_holds") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def autoheld_steps(self) -> int:
        """Auto-held steps this client observed, to reconcile with ATL's count.

        A gap between the two means submissions were rejected for a reason the
        client did not classify as an auto-hold, or the run auto-held steps this
        client never reached.
        """
        return self.replay.autoheld_steps

    @property
    def compare_url(self) -> Optional[str]:
        return (
            self.result.raw.get("compare_url")
            or self.result.run.get("compare_url")
        )


class TradingAgentsATLRunner:
    """Drive an offline TradingAgents artifact through ATL's typed Run API."""

    ENVIRONMENT_ID = "us-equity-hourly-v1"

    def __init__(self, client: Any) -> None:
        self.client = client
        self._validated_symbols = set()

    def run_backtest(
        self,
        *,
        artifact: TradingAgentsDecisionArtifact,
        artifact_sha256: str,
        agent_version_id: str,
        start_date: str,
        end_date: str,
        poll_interval: float = 2.0,
        max_wait_seconds: Optional[float] = 300.0,
    ) -> TradingAgentsATLRunOutcome:
        if not agent_version_id:
            raise ArtifactValidationError("agent_version_id is required")
        if poll_interval <= 0:
            raise ArtifactValidationError("poll_interval must be greater than 0")

        start = _canonical_date(start_date, field="start_date")
        end = _canonical_date(end_date, field="end_date")
        if start >= end:
            raise ArtifactValidationError("start_date must be before end_date")
        if any(
            date.fromisoformat(record.analysis_date) >= end
            for record in artifact.decisions
        ):
            raise ArtifactValidationError(
                "every analysis_date must be earlier than end_date"
            )
        valid_count = sum(
            record.status == "valid" for record in artifact.decisions
        )
        error_count = len(artifact.decisions) - valid_count
        if valid_count == 0:
            raise TradingAgentsGenerationError(
                "artifact must contain at least one valid TradingAgents decision"
            )

        self.validate_symbol(str(artifact.manifest["symbol"]).upper())
        planner = TradingAgentsReplayPlanner(artifact, artifact_sha256)
        integration_config = {
            "id": "tradingagents",
            "artifact_schema_version": artifact.schema_version,
            "artifact_sha256": artifact_sha256.lower(),
            "tradingagents_version": artifact.manifest.get(
                "tradingagents_version"
            ),
            "symbol": planner.symbol,
            "analysis_dates": [
                record.analysis_date for record in artifact.decisions
            ],
            "valid_decisions": valid_count,
            "error_decisions": error_count,
            "decision_data_source": "tradingagents_configured_vendors",
            "execution_data_source": "atl_alpaca",
        }

        run = None
        fills = []
        rejections = []
        try:
            run = self.client.create_run(
                agent_version_id,
                environment_id=self.ENVIRONMENT_ID,
                start_date=start_date,
                end_date=end_date,
                symbols=[planner.symbol],
                config={"integration": integration_config},
            )
            if not run.id:
                raise ATLAPIError(
                    "ATL create_run returned no run_id",
                    code="missing_run_id",
                )

            waited = 0.0
            while True:
                step = self.client.get_next_step(run.id)
                status = step.status
                if status == "completed":
                    break
                if status in _FAILED_STATES:
                    raise ATLRunFailedError(
                        step.message or f"run entered state {status!r}",
                        code=status,
                    )
                if status == "awaiting_decision":
                    waited = 0.0
                    decision = planner.decision_for_step(step)
                    try:
                        execution = self.client.submit_decision(
                            run.id, step.id, decision
                        )
                    except ATLConflictError as exc:
                        if exc.code not in _STEP_AUTOHELD_CODES:
                            raise
                        # ATL closed this step without our decision, so nothing
                        # was executed. Roll the proposal back: the records stay
                        # eligible for the next step, and a run that ends first
                        # reports them as unprocessed instead of counting an
                        # order that never reached the exchange.
                        planner.discard(step)
                        continue
                    planner.commit(step)
                    fills.extend(dict(item) for item in execution.fills)
                    rejections.extend(
                        dict(item) for item in execution.rejections
                    )
                    continue
                if status in ("loading", "pending", "executing"):
                    waited = wait_for_poll(
                        self.client,
                        poll_interval,
                        waited,
                        max_wait_seconds,
                        str(status),
                    )
                    continue
                raise ATLAPIError(
                    f"unexpected step status {status!r}",
                    code="unexpected_step_status",
                )

            result = self.client.get_run_result(run.id)
            diagnostics = planner.finalize()
            outcome = TradingAgentsATLRunOutcome(
                result=result,
                replay=diagnostics,
                fills=tuple(fills),
                rejections=tuple(rejections),
            )
            if diagnostics.unprocessed_dates:
                raise TradingAgentsReplayIncompleteError(
                    run_id=run.id,
                    analysis_dates=diagnostics.unprocessed_dates,
                    result=result,
                    diagnostics=diagnostics,
                )
            return outcome
        except ATLAPIError as exc:
            raise exc.with_run_id(run.id if run is not None else None)
        except TradingAgentsReplayValidationError as exc:
            # A mid-run contract break aborts before the run reaches a terminal
            # state; without the id the caller cannot find the run it left open.
            raise exc.with_run_id(run.id if run is not None else None)

    def validate_symbol(self, symbol: str) -> None:
        """Fail before model generation when ATL cannot trade ``symbol``."""
        symbol = str(symbol or "").strip().upper()
        if symbol in self._validated_symbols:
            return
        environments = self.client.list_environments()
        environment = next(
            (
                item
                for item in environments
                if item.get("environment_id") == self.ENVIRONMENT_ID
            ),
            None,
        )
        if environment is None:
            raise TradingAgentsReplayValidationError(
                f"ATL environment {self.ENVIRONMENT_ID!r} is unavailable"
            )
        universe = environment.get("universe") or []
        if symbol not in universe:
            raise TradingAgentsReplayValidationError(
                f"{symbol} is not in the {self.ENVIRONMENT_ID} universe"
            )
        self._validated_symbols.add(symbol)

