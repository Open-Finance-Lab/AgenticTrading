"""T+1 AgentRunner adapter for local vn.py CtaTemplate strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple

from ..models import Decision, ExecutionResult, Observation, RunResult
from ._vnpy_cta_core import (
    CapturedCtaOrder,
    VnpyCtaAuditArtifact,
    VnpyCtaAuditRecord,
    build_audit_artifact,
    build_safe_manifest,
    map_captured_order,
    sanitize_error_message,
    validate_ohlcv_values,
)


class VnpyCtaDataError(RuntimeError):
    """Raised when ATL's observation cannot satisfy the vn.py BarData contract."""


@dataclass(frozen=True)
class _PendingDecision:
    sequence: int
    execution_timestamp: str
    submitted_captured: Tuple[CapturedCtaOrder, ...]


def _parse_aware_timestamp(value: Any) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise VnpyCtaDataError(f"invalid Bar timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VnpyCtaDataError("Bar timestamp must be timezone-aware")
    return parsed


class VnpyCtaAdapter:
    """Translate delayed ATL observations into local CTA calls and Decisions."""

    def __init__(
        self,
        runtime: Any,
        *,
        symbol: str,
        manifest: Mapping[str, Any],
    ) -> None:
        self.runtime = runtime
        self.symbol = str(symbol).upper()
        if self.symbol != "AAPL":
            raise ValueError("vn.py CTA MVP supports AAPL only")
        self._manifest = build_safe_manifest(manifest)
        self._records: list[VnpyCtaAuditRecord] = []
        self._deferred_bar: Optional[Dict[str, Any]] = None
        self._pending: Optional[_PendingDecision] = None
        self._seen_timestamps: set[str] = set()
        self._last_timestamp: Optional[datetime] = None
        self._completed = False
        self.runtime.start()

    def _bar_from(self, observation: Observation) -> Dict[str, Any]:
        bars = observation.market.get("bars") or {}
        raw = bars.get(self.symbol)
        if not isinstance(raw, Mapping):
            raise VnpyCtaDataError(
                f"Observation is missing the current {self.symbol} OHLCV bar"
            )
        timestamp = _parse_aware_timestamp(raw.get("timestamp"))
        timestamp_text = timestamp.isoformat()
        if timestamp_text in self._seen_timestamps:
            raise VnpyCtaDataError(f"duplicate Bar timestamp: {timestamp_text}")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise VnpyCtaDataError("Bar timestamps must be strictly increasing")

        try:
            values = validate_ohlcv_values(raw)
        except ValueError as exc:
            raise VnpyCtaDataError(f"Bar failed OHLCV validation: {exc}") from exc
        return {"symbol": self.symbol, "timestamp": timestamp_text, **values}

    def _position_from(self, observation: Observation) -> int:
        for position in observation.positions:
            if str(position.get("symbol", "")).upper() != self.symbol:
                continue
            value = position.get("quantity", position.get("shares", 0))
            if isinstance(value, bool):
                raise VnpyCtaDataError("portfolio position must be an integer")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise VnpyCtaDataError("portfolio position must be an integer") from exc
            if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                raise VnpyCtaDataError("portfolio position must be a non-negative integer")
            return int(numeric)
        return 0

    def _replace_record(self, sequence: int, **changes: Any) -> None:
        for index, record in enumerate(self._records):
            if record.sequence == sequence:
                self._records[index] = replace(record, **changes)
                return
        raise RuntimeError(f"missing audit record sequence {sequence}")

    def _mark_pending_timeout(self) -> None:
        pending = self._pending
        if pending is None:
            return
        record = next(
            item for item in self._records if item.sequence == pending.sequence
        )
        diagnostics = dict(record.diagnostics)
        if pending.submitted_captured:
            diagnostics["timed_out_orders"] = len(pending.submitted_captured)
        self._replace_record(
            pending.sequence,
            status="timeout_hold",
            execution={"accepted": False, "outcome": "timeout_hold"},
            diagnostics=diagnostics,
        )
        for captured in pending.submitted_captured:
            self.runtime.reject_captured_order(
                captured,
                reason="timeout_hold",
                timestamp=pending.execution_timestamp,
            )
        self._pending = None

    @staticmethod
    def _execution_dict(result: ExecutionResult) -> Dict[str, Any]:
        return {
            "accepted": result.accepted,
            "fills": list(result.fills),
            "warnings": list(result.warnings),
            "rejections": list(result.rejections),
            "portfolio_after": dict(result.portfolio_after),
            "run_status": result.run_status,
        }

    def decide(self, observation: Observation) -> Decision:
        if self._completed:
            raise RuntimeError("vn.py CTA adapter has already completed")
        self._mark_pending_timeout()
        current_bar = self._bar_from(observation)
        current_timestamp = current_bar["timestamp"]
        position = self._position_from(observation)
        self.runtime.sync_position(position)

        sequence = len(self._records)
        captured: Tuple[CapturedCtaOrder, ...] = ()
        submitted_captured: list[CapturedCtaOrder] = []
        orders = []
        warnings: list[str] = []
        diagnostics: Dict[str, int] = {}
        error: Optional[str] = None

        if self._deferred_bar is None:
            status = "warmup_hold"
            signal_timestamp = None
            signal_bar: Dict[str, Any] = {}
        else:
            signal_bar = self._deferred_bar
            signal_timestamp = str(signal_bar["timestamp"])
            try:
                captured = tuple(self.runtime.on_bar(signal_bar))
            except Exception as exc:
                captured = tuple(self.runtime.drain_captured_orders())
                status = "error_hold"
                error = sanitize_error_message(exc)
                for captured_order in captured:
                    self.runtime.reject_captured_order(
                        captured_order,
                        reason="strategy_error",
                        timestamp=current_timestamp,
                    )
            else:
                for captured_order in captured:
                    mapping = map_captured_order(
                        captured_order,
                        symbol=self.symbol,
                        current_position=position,
                    )
                    warnings.extend(mapping.warnings)
                    if mapping.order is not None:
                        orders.append(mapping.order)
                        submitted_captured.append(captured_order)
                        continue
                    key = (
                        "unsupported_actions"
                        if mapping.status == "unsupported_action"
                        else "local_rejections"
                    )
                    diagnostics[key] = diagnostics.get(key, 0) + 1
                    self.runtime.reject_captured_order(
                        captured_order,
                        reason=mapping.reason or mapping.status,
                        timestamp=current_timestamp,
                    )

                if orders and diagnostics:
                    status = "partial_submission"
                elif orders:
                    status = "decision_submitted"
                elif diagnostics.get("local_rejections"):
                    status = "local_rejection"
                elif diagnostics.get("unsupported_actions"):
                    status = "unsupported_action"
                else:
                    status = "strategy_hold"

        record = VnpyCtaAuditRecord(
            sequence=sequence,
            observation_timestamp=current_timestamp,
            signal_timestamp=signal_timestamp,
            status=status,
            bar=dict(current_bar),
            signal_bar=dict(signal_bar),
            captured_orders=captured,
            submitted_orders=tuple(order.to_dict() for order in orders),
            diagnostics=diagnostics,
            warnings=tuple(dict.fromkeys(warnings)),
            error=error,
        )
        self._records.append(record)
        self._deferred_bar = current_bar
        self._seen_timestamps.add(current_timestamp)
        self._last_timestamp = _parse_aware_timestamp(current_timestamp)
        self._pending = _PendingDecision(
            sequence=sequence,
            execution_timestamp=current_timestamp,
            submitted_captured=tuple(submitted_captured),
        )

        return Decision(
            orders=orders,
            rationale=(
                f"vn.py CTA {status}; signal={signal_timestamp or 'warmup'}"
            ),
            trace={
                "integration": "vnpy_cta",
                "status": status,
                "sequence": sequence,
                "signal_timestamp": signal_timestamp,
                "observation_timestamp": current_timestamp,
                "diagnostics": diagnostics,
            },
        )

    def on_execution_result(self, result: ExecutionResult) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("received execution result without a pending decision")
        record = next(
            item for item in self._records if item.sequence == pending.sequence
        )
        diagnostics = dict(record.diagnostics)
        if result.fills:
            diagnostics["fills"] = diagnostics.get("fills", 0) + len(result.fills)
        if result.rejections:
            diagnostics["atl_rejections"] = diagnostics.get(
                "atl_rejections", 0
            ) + len(result.rejections)
        status = record.status
        if result.rejections and not result.fills and pending.submitted_captured:
            status = "atl_rejection"
        elif result.rejections and result.fills:
            status = "partial_submission"

        self.runtime.apply_execution_result(
            result,
            captured_orders=pending.submitted_captured,
            timestamp=pending.execution_timestamp,
        )
        self._replace_record(
            pending.sequence,
            status=status,
            execution=self._execution_dict(result),
            diagnostics=diagnostics,
        )
        self._pending = None

    def on_run_completed(self, result: RunResult) -> None:
        if self._completed:
            return
        self._mark_pending_timeout()
        if result.run_id:
            self._manifest["run_id"] = result.run_id
        if self._deferred_bar is not None:
            self._records.append(
                VnpyCtaAuditRecord(
                    sequence=len(self._records),
                    observation_timestamp=str(self._deferred_bar["timestamp"]),
                    signal_timestamp=str(self._deferred_bar["timestamp"]),
                    status="terminal_bar_skipped",
                    bar=dict(self._deferred_bar),
                    signal_bar=dict(self._deferred_bar),
                )
            )
        self.runtime.stop()
        self._completed = True

    def finalize_artifact(self) -> VnpyCtaAuditArtifact:
        return build_audit_artifact(
            manifest=self._manifest,
            records=tuple(self._records),
        )

    @property
    def manifest(self) -> Dict[str, Any]:
        return build_safe_manifest(self._manifest)

    def abort(self, error: Exception, *, status: str = "run_error") -> None:
        """Stop a partial run and record an API/runtime failure without a HOLD."""
        if self._completed:
            return
        pending = self._pending
        if pending is not None:
            for captured in pending.submitted_captured:
                self.runtime.reject_captured_order(
                    captured,
                    reason="run_error",
                    timestamp=pending.execution_timestamp,
                )
            self._pending = None
        last_bar = dict(self._deferred_bar or {})
        timestamp = str(last_bar.get("timestamp") or "run_not_started")
        self._records.append(
            VnpyCtaAuditRecord(
                sequence=len(self._records),
                observation_timestamp=timestamp,
                status=status,
                bar=last_bar,
                error=sanitize_error_message(error),
            )
        )
        try:
            self.runtime.stop()
        finally:
            self._completed = True


__all__ = ["VnpyCtaDataError", "VnpyCtaAdapter"]
