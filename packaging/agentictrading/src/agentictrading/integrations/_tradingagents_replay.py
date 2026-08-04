"""Pure T+1 replay planning for TradingAgents decision artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..models import Decision, Order, Step
from ._tradingagents_core import (
    TradingAgentsDecisionArtifact,
    TradingAgentsDecisionRecord,
)

MARKET_TIMEZONE = "America/New_York"


class TradingAgentsReplayValidationError(ValueError):
    """Raised when an ATL Step cannot safely execute the replay contract."""

    def __init__(self, message: str, *, run_id: Optional[str] = None) -> None:
        self.message = str(message)
        self.run_id = run_id
        super().__init__(self._format())

    def _format(self) -> str:
        if self.run_id:
            return f"{self.message} {{run {self.run_id}}}"
        return self.message

    def with_run_id(
        self, run_id: Optional[str]
    ) -> "TradingAgentsReplayValidationError":
        """Attach the ATL run id (if unset) and refresh the message.

        Mirrors ``ATLAPIError.with_run_id`` so a mid-run abort still names the
        run an operator has to inspect. Returns ``self`` so callers can
        ``raise exc.with_run_id(run.id)`` without losing the traceback.
        """
        if run_id and not self.run_id:
            self.run_id = run_id
            self.args = (self._format(),)
        return self


def market_timezone() -> ZoneInfo:
    """Return the exchange timezone, or explain how to install the tz database.

    ``zoneinfo`` reads the operating system's IANA database. Windows never
    ships one and slim containers often strip it, so this SDK declares the
    ``tzdata`` wheel as a Windows dependency. Anywhere else a missing database
    must still fail with a fixable instruction instead of a bare
    ``ZoneInfoNotFoundError`` raised from the middle of a replay.
    """
    try:
        return ZoneInfo(MARKET_TIMEZONE)
    except ZoneInfoNotFoundError as exc:
        raise TradingAgentsReplayValidationError(
            "the IANA time-zone database is unavailable, so ATL step "
            f"timestamps cannot be converted to {MARKET_TIMEZONE}; "
            "install it with 'pip install tzdata'"
        ) from exc


@dataclass(frozen=True)
class TradingAgentsReplayDiagnostics:
    """Local audit counters collected while replaying one artifact.

    Every counter records a decision ATL *accepted*. Steps the server finalized
    on its own (deadline exceeded) are counted in ``autoheld_steps`` only, and
    the records they would have consumed stay in ``unprocessed_dates``.
    """

    processed_dates: Tuple[str, ...]
    unprocessed_dates: Tuple[str, ...]
    buy_orders: int
    sell_orders: int
    model_holds: int
    error_holds: int
    passive_holds: int
    constraint_holds: int
    superseded: int
    price_too_high_holds: int = 0
    autoheld_steps: int = 0


@dataclass(frozen=True)
class _PendingPlan:
    """A decision proposed for one step and not yet accepted by ATL."""

    decision: Decision
    counter: str
    consumed_dates: Tuple[str, ...]
    superseded: int


class TradingAgentsReplayPlanner:
    """Convert offline TradingAgents records into idempotent ATL Decisions."""

    def __init__(
        self,
        artifact: TradingAgentsDecisionArtifact,
        artifact_sha256: str,
    ) -> None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(artifact_sha256 or "")):
            raise TradingAgentsReplayValidationError(
                "artifact_sha256 must be a 64-character hexadecimal digest"
            )
        self.artifact = artifact
        self.artifact_sha256 = artifact_sha256.lower()
        self.symbol = str(artifact.manifest["symbol"]).upper()
        # An analysis_date never changes, but decision_for_step runs once per
        # hourly ATL step and would otherwise re-parse every record's date on
        # each one. Parse them all exactly once, here.
        self._dated_records = tuple(
            (record, date.fromisoformat(record.analysis_date))
            for record in artifact.decisions
        )
        self._processed = set()
        self._decision_cache: Dict[str, Decision] = {}
        self._pending: Dict[str, _PendingPlan] = {}
        # Keyed by TradingAgentsReplayDiagnostics field name so finalize() can
        # splat them; a plain dict keeps a bad key failing loudly (KeyError).
        self._counts: Dict[str, int] = {
            "buy_orders": 0,
            "sell_orders": 0,
            "model_holds": 0,
            "error_holds": 0,
            "passive_holds": 0,
            "constraint_holds": 0,
            "price_too_high_holds": 0,
        }
        self._superseded = 0
        self._autoheld_steps = 0

    def decision_for_step(self, step: Step) -> Decision:
        """Propose the decision for one ATL Step without external calls.

        Nothing is consumed here: an artifact record is only marked processed
        once :meth:`commit` confirms ATL accepted the decision built from it.
        Proposing and consuming in one move would let a step the server
        auto-holds report an order that never reached the exchange while
        silently dropping the signal behind it.
        """
        cache_key = self._step_key(step)
        committed = self._decision_cache.get(cache_key)
        if committed is not None:
            return committed
        pending = self._pending.get(cache_key)
        if pending is not None:
            return pending.decision

        trading_date = self._trading_date(step)
        eligible = [
            record
            for record, analysis_date in self._dated_records
            if record.analysis_date not in self._processed
            and analysis_date < trading_date
        ]
        if not eligible:
            plan = _PendingPlan(
                decision=Decision(
                    orders=[],
                    rationale=(
                        f"TradingAgents passive_hold: no record eligible on "
                        f"{trading_date.isoformat()}"
                    ),
                ),
                counter="passive_holds",
                consumed_dates=(),
                superseded=0,
            )
        else:
            decision, counter = self._decision_for_record(step, eligible[-1])
            plan = _PendingPlan(
                decision=decision,
                counter=counter,
                consumed_dates=tuple(
                    record.analysis_date for record in eligible
                ),
                superseded=len(eligible) - 1,
            )
        self._pending[cache_key] = plan
        return plan.decision

    def commit(self, step: Step) -> None:
        """Consume the proposal for ``step`` after ATL accepted the decision."""
        cache_key = self._step_key(step)
        plan = self._pending.pop(cache_key, None)
        if plan is None:
            return
        self._processed.update(plan.consumed_dates)
        self._superseded += plan.superseded
        self._counts[plan.counter] += 1
        self._decision_cache[cache_key] = plan.decision

    def discard(self, step: Step) -> None:
        """Roll back the proposal for a step ATL finalized without our decision.

        The records it would have consumed stay eligible, so the next step
        executes them; if the run ends first they surface through
        ``unprocessed_dates`` instead of vanishing.
        """
        if self._pending.pop(self._step_key(step), None) is None:
            return
        self._autoheld_steps += 1

    def finalize(self) -> TradingAgentsReplayDiagnostics:
        """Return an immutable snapshot, including records never reached."""
        ordered_dates = tuple(
            record.analysis_date for record in self.artifact.decisions
        )
        return TradingAgentsReplayDiagnostics(
            processed_dates=tuple(
                value for value in ordered_dates if value in self._processed
            ),
            unprocessed_dates=tuple(
                value for value in ordered_dates if value not in self._processed
            ),
            superseded=self._superseded,
            autoheld_steps=self._autoheld_steps,
            **self._counts,
        )

    @staticmethod
    def _step_key(step: Step) -> str:
        return str(
            step.id
            or f"{step.run_id}:{step.sequence}:{step.timestamp}"
        )

    @staticmethod
    def _trading_date(step: Step) -> date:
        timestamp = str(step.timestamp or "")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise TradingAgentsReplayValidationError(
                f"Step timestamp is not ISO-8601: {timestamp!r}"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise TradingAgentsReplayValidationError(
                "Step timestamp must include a timezone"
            )
        return parsed.astimezone(market_timezone()).date()

    def _decision_for_record(
        self,
        step: Step,
        record: TradingAgentsDecisionRecord,
    ):
        prefix = (
            f"TradingAgents analysis_date={record.analysis_date} "
            f"artifact={self.artifact_sha256[:12]}"
        )
        if record.status == "error":
            return (
                Decision(
                    orders=[],
                    rationale=(
                        f"{prefix} generation_error={record.error_type}: "
                        f"{record.error_message}"
                    )[:500],
                ),
                "error_holds",
            )

        if record.atl_action == "HOLD":
            return self._hold(prefix, record, "model_hold"), "model_holds"

        constraints = step.constraints or {}
        allowed = constraints.get("allowed_symbols")
        if not isinstance(allowed, (list, tuple, set)) or self.symbol not in allowed:
            raise TradingAgentsReplayValidationError(
                f"{self.symbol} is missing from Step constraints.allowed_symbols"
            )
        weight = self._positive_number(
            constraints.get("max_position_weight"),
            field="max_position_weight",
        )
        if weight > 1:
            raise TradingAgentsReplayValidationError(
                "max_position_weight must be no greater than 1"
            )

        observation = step.observation
        if observation is None:
            raise TradingAgentsReplayValidationError(
                "Step observation is required for replay"
            )
        held = self._held_shares(observation.positions)

        if record.atl_action == "SELL":
            if held <= 0:
                return (
                    self._hold(prefix, record, "sell_without_position"),
                    "constraint_holds",
                )
            return (
                Decision(
                    orders=self._market_order("sell", held),
                    rationale=f"{prefix} rating={record.rating}; close_position",
                ),
                "sell_orders",
            )

        features = observation.features
        symbol_features = features.get(self.symbol) if isinstance(features, dict) else None
        price_value = symbol_features.get("price") if isinstance(symbol_features, dict) else None
        # A missing key and an unusable value (None, NaN, <= 0) are the same
        # outcome here, and _positive_number already folds the first into the
        # second, so one check covers both.
        price = self._positive_number(price_value, field="price", hold_on_error=True)
        if price is None:
            return self._hold(prefix, record, "missing_price"), "constraint_holds"
        equity = self._positive_number(
            observation.portfolio.get("equity"), field="portfolio.equity"
        )
        budget = equity * weight
        target_shares = math.floor(budget / price)
        if target_shares <= 0:
            # One share costs more than the position cap allows, so this BUY is
            # unexecutable by arithmetic, not by market conditions. Name both
            # numbers: otherwise it is indistinguishable from a model HOLD.
            return (
                self._hold(
                    prefix,
                    record,
                    f"price_too_high_for_target price={price:.2f} "
                    f"max_position_budget={budget:.2f}",
                ),
                "price_too_high_holds",
            )
        buy_shares = target_shares - held
        if buy_shares <= 0:
            return (
                self._hold(prefix, record, "already_at_target"),
                "constraint_holds",
            )
        return (
            Decision(
                orders=self._market_order("buy", buy_shares),
                rationale=f"{prefix} rating={record.rating}; target_weight={weight:g}",
            ),
            "buy_orders",
        )

    @staticmethod
    def _hold(
        prefix: str, record: TradingAgentsDecisionRecord, reason: str
    ) -> Decision:
        """Build the empty-order Decision that names why nothing was traded."""
        return Decision(
            orders=[], rationale=f"{prefix} rating={record.rating}; {reason}"
        )

    def _market_order(self, side: str, quantity: int) -> list:
        return [
            Order(
                symbol=self.symbol,
                side=side,
                quantity=quantity,
                quantity_type="shares",
                order_type="market",
            )
        ]

    def _held_shares(self, positions: Sequence[Mapping[str, Any]]) -> int:
        held = 0
        for position in positions:
            if not isinstance(position, Mapping):
                raise TradingAgentsReplayValidationError(
                    "portfolio positions must be objects"
                )
            if str(position.get("symbol", "")).upper() != self.symbol:
                continue
            quantity = position.get("quantity")
            try:
                numeric = float(quantity)
            except (TypeError, ValueError) as exc:
                raise TradingAgentsReplayValidationError(
                    "portfolio position quantity must be an integer"
                ) from exc
            if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                raise TradingAgentsReplayValidationError(
                    "portfolio position quantity must be a non-negative integer"
                )
            held += int(numeric)
        return held

    @staticmethod
    def _positive_number(value: Any, *, field: str, hold_on_error: bool = False):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            if hold_on_error:
                return None
            raise TradingAgentsReplayValidationError(
                f"{field} must be a positive number"
            ) from exc
        if not math.isfinite(numeric) or numeric <= 0:
            if hold_on_error:
                return None
            raise TradingAgentsReplayValidationError(
                f"{field} must be a positive number"
            )
        return numeric

