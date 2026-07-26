"""Pure T+1 replay planning for TradingAgents decision artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

from ..models import Decision, Order, Step
from ._tradingagents_core import (
    TradingAgentsDecisionArtifact,
    TradingAgentsDecisionRecord,
)


class TradingAgentsReplayValidationError(ValueError):
    """Raised when an ATL Step cannot safely execute the replay contract."""


@dataclass(frozen=True)
class TradingAgentsReplayDiagnostics:
    """Local audit counters collected while replaying one artifact."""

    processed_dates: Tuple[str, ...]
    unprocessed_dates: Tuple[str, ...]
    buy_orders: int
    sell_orders: int
    model_holds: int
    error_holds: int
    passive_holds: int
    constraint_holds: int
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
        self._processed = set()
        self._decision_cache: Dict[str, Decision] = {}
        self._buy_orders = 0
        self._sell_orders = 0
        self._model_holds = 0
        self._error_holds = 0
        self._passive_holds = 0
        self._constraint_holds = 0
        self._superseded = 0

    def decision_for_step(self, step: Step) -> Decision:
        """Return the decision for one ATL Step without external calls."""
        cache_key = self._step_key(step)
        cached = self._decision_cache.get(cache_key)
        if cached is not None:
            return cached

        trading_date = self._trading_date(step)
        eligible = [
            record
            for record in self.artifact.decisions
            if record.analysis_date not in self._processed
            and date.fromisoformat(record.analysis_date) < trading_date
        ]
        if not eligible:
            decision = Decision(
                orders=[],
                rationale=(
                    f"TradingAgents passive_hold: no record eligible on "
                    f"{trading_date.isoformat()}"
                ),
            )
            self._passive_holds += 1
            self._decision_cache[cache_key] = decision
            return decision

        chosen = eligible[-1]
        decision, counter = self._decision_for_record(step, chosen)

        for stale in eligible[:-1]:
            self._processed.add(stale.analysis_date)
        self._superseded += len(eligible) - 1
        self._processed.add(chosen.analysis_date)
        setattr(self, counter, getattr(self, counter) + 1)
        self._decision_cache[cache_key] = decision
        return decision

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
            buy_orders=self._buy_orders,
            sell_orders=self._sell_orders,
            model_holds=self._model_holds,
            error_holds=self._error_holds,
            passive_holds=self._passive_holds,
            constraint_holds=self._constraint_holds,
            superseded=self._superseded,
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
        return parsed.astimezone(ZoneInfo("America/New_York")).date()

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
                "_error_holds",
            )

        if record.atl_action == "HOLD":
            return (
                Decision(
                    orders=[],
                    rationale=f"{prefix} rating={record.rating}; model_hold",
                ),
                "_model_holds",
            )

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
                    Decision(
                        orders=[],
                        rationale=(
                            f"{prefix} rating={record.rating}; "
                            "sell_without_position"
                        ),
                    ),
                    "_constraint_holds",
                )
            return (
                Decision(
                    orders=[
                        Order(
                            symbol=self.symbol,
                            side="sell",
                            quantity=held,
                            quantity_type="shares",
                            order_type="market",
                        )
                    ],
                    rationale=f"{prefix} rating={record.rating}; close_position",
                ),
                "_sell_orders",
            )

        features = observation.features
        symbol_features = features.get(self.symbol) if isinstance(features, dict) else None
        price_value = symbol_features.get("price") if isinstance(symbol_features, dict) else None
        if price_value is None:
            return (
                Decision(
                    orders=[],
                    rationale=f"{prefix} rating={record.rating}; missing_price",
                ),
                "_constraint_holds",
            )
        price = self._positive_number(price_value, field="price", hold_on_error=True)
        if price is None:
            return (
                Decision(
                    orders=[],
                    rationale=f"{prefix} rating={record.rating}; missing_price",
                ),
                "_constraint_holds",
            )
        equity = self._positive_number(
            observation.portfolio.get("equity"), field="portfolio.equity"
        )
        target_shares = math.floor(equity * weight / price)
        if target_shares <= 0:
            return (
                Decision(
                    orders=[],
                    rationale=(
                        f"{prefix} rating={record.rating}; "
                        "price_too_high_for_target"
                    ),
                ),
                "_constraint_holds",
            )
        buy_shares = target_shares - held
        if buy_shares <= 0:
            return (
                Decision(
                    orders=[],
                    rationale=f"{prefix} rating={record.rating}; already_at_target",
                ),
                "_constraint_holds",
            )
        return (
            Decision(
                orders=[
                    Order(
                        symbol=self.symbol,
                        side="buy",
                        quantity=buy_shares,
                        quantity_type="shares",
                        order_type="market",
                    )
                ],
                rationale=f"{prefix} rating={record.rating}; target_weight={weight:g}",
            ),
            "_buy_orders",
        )

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

