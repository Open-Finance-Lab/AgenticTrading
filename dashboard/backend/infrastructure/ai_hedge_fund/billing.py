"""ATL Credits metering for the hosted AI Hedge Fund runtime.

Upstream makes its model calls inside its own interpreter, so ATL cannot meter
them one by one the way ``infrastructure/llm/execution/service.py`` does. The
unit here is one decision (one bridge subprocess): reserve a ceiling before it,
settle at the token usage the bridge reports after it. The run's reservations
share the run id of the signed execution handoff, so the existing
``finalize_run`` (child and parent) releases any hold a killed child leaves.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional, Sequence

from dashboard.backend.domain.agents.runtime import AI_HEDGE_FUND_ANALYSTS
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.execution.service import (
    reserve_platform_credits,
)
from dashboard.backend.infrastructure.llm.token_cost import (
    credits_micro_for_usd,
    estimate_cost_usd,
)

PROVIDER_ID = "openrouter"

# Per-call budget behind a decision's reservation ceiling. Generous on purpose:
# a ceiling below the real spend is capped at settlement (the operator eats the
# rest) rather than recorded as an overage, which would restrict the account
# for ATL's own estimate. Every selected analyst plus the portfolio manager is
# counted as one call per ticker, though several analysts are rule-based.
CEILING_INPUT_TOKENS_PER_CALL = 8_000
CEILING_OUTPUT_TOKENS_PER_CALL = 2_000


def decision_calls(analysts: Optional[Sequence[str]], ticker_count: int) -> int:
    """The model calls one decision's ceiling budgets for.

    No analysts configured means upstream runs all of them.
    """
    analyst_count = len(analysts or AI_HEDGE_FUND_ANALYSTS)
    return (analyst_count + 1) * max(int(ticker_count), 1)


def decision_ceiling_micro(
    model: str, analysts: Optional[Sequence[str]], ticker_count: int
) -> int:
    """The most one decision is held for, in Credit micro-units.

    One formula for the launch preflight and the per-decision hold, so the
    route cannot admit a run whose decisions the child then refuses.
    """
    calls = decision_calls(analysts, ticker_count)
    return credits_micro_for_usd(
        estimate_cost_usd(
            model,
            calls * CEILING_INPUT_TOKENS_PER_CALL,
            calls * CEILING_OUTPUT_TOKENS_PER_CALL,
        )
    )


class DecisionUsage(NamedTuple):
    """What the bridge's meter counted over one decision."""

    input_tokens: int
    output_tokens: int
    # Model calls that ended, and those of them that reported no usage.
    calls: int
    unmetered_calls: int


def parse_usage(value: Any) -> Optional[DecisionUsage]:
    """The bridge's usage report, else None (absent, malformed, inconsistent)."""
    if not isinstance(value, Mapping):
        return None
    counts = []
    for key in DecisionUsage._fields:
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        counts.append(count)
    usage = DecisionUsage(*counts)
    if usage.unmetered_calls > usage.calls:
        return None
    return usage


class HostedDecisionBilling:
    """Reserve, settle and release one run's per-decision Credit holds."""

    billing_mode = "platform_credits"
    provider_id = PROVIDER_ID

    def __init__(self, *, credits: Any, user_id: int, run_id: str, model_id: str):
        self.credits = credits
        self.user_id = int(user_id)
        self.run_id = str(run_id)
        self.model_id = model_id
        self._next_call_index = 0

    @classmethod
    def from_handoff(cls, *, credits: Any, handoff: Any) -> "HostedDecisionBilling":
        return cls(
            credits=credits,
            user_id=handoff.user_id,
            run_id=handoff.run_id,
            model_id=handoff.model_id,
        )

    def reserve(self, amount_micro: int) -> str:
        """Hold one decision's ceiling; the run stops when it cannot."""
        call_index = self._next_call_index
        self._next_call_index += 1
        reservation = reserve_platform_credits(
            self.credits,
            user_id=self.user_id,
            run_id=self.run_id,
            call_index=call_index,
            amount_micro=int(amount_micro),
            provider_id=PROVIDER_ID,
        )
        if reservation.status != "open":
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)
        return reservation.reservation_id

    def close(
        self,
        reservation_id: str,
        *,
        ceiling_micro: int,
        usage: Optional[DecisionUsage],
        outcome: str,
    ) -> int:
        """Settle or release one decision's hold; returns the micro charged.

        ``outcome`` is ``completed``, ``killed`` or ``failed``. The charge
        fails closed whenever the spend is unknown, settling at the ceiling:

        - no usage report on a ``completed`` or ``killed`` decision (an
          unreadable output, a timeout, a signal, a crash past the bridge's
          started marker): its model calls may have run;
        - a report with unmetered calls: those calls ran and their provider
          (or a langchain change) dropped the usage, which is a broken meter.

        Released only when nothing was spent: a report of zero calls (every
        analyst rule-based, every ticker pre-filled), or ``failed`` with no
        report, which the adapter says solely when the bridge left no file at
        all, i.e. died before any model call. Otherwise the metered tokens.
        """
        input_tokens = output_tokens = None
        if usage is None and outcome == "failed":
            uncapped = 0
            basis = "no_model_call"
        elif usage is None:
            uncapped = int(ceiling_micro)
            basis = "ceiling"
            if outcome == "completed":
                print(
                    "ERROR: ai_hedge_fund.usage_unreported "
                    "billing at the decision ceiling",
                    flush=True,
                )
        elif usage.unmetered_calls:
            uncapped = int(ceiling_micro)
            basis = "ceiling"
            print(
                "ERROR: ai_hedge_fund.usage_unmetered "
                f"calls={usage.calls} unmetered_calls={usage.unmetered_calls} "
                "billing at the decision ceiling",
                flush=True,
            )
        elif usage.calls == 0:
            uncapped = 0
            basis = "no_model_call"
        else:
            input_tokens, output_tokens = usage.input_tokens, usage.output_tokens
            uncapped = credits_micro_for_usd(
                estimate_cost_usd(self.model_id, input_tokens, output_tokens)
            )
            basis = "usage"
        charged = min(uncapped, int(ceiling_micro))
        if uncapped > charged:
            print(
                "WARNING: ai_hedge_fund.usage_above_ceiling "
                f"uncapped_micro={uncapped} ceiling_micro={ceiling_micro}",
                flush=True,
            )
        try:
            if charged <= 0:
                self.credits.release_llm_credits(reservation_id, reason=basis)
                return 0
            settlement = self.credits.settle_llm_credits(
                reservation_id,
                actual_micro=charged,
                evidence={
                    "source": "ai_hedge_fund",
                    "model_id": self.model_id,
                    "provider_id": PROVIDER_ID,
                    "outcome": outcome,
                    "basis": basis,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "calls": usage.calls if usage is not None else None,
                    "unmetered_calls": (
                        usage.unmetered_calls if usage is not None else None
                    ),
                    "ceiling_micro": int(ceiling_micro),
                    "uncapped_micro": uncapped,
                },
            )
        except Exception as exc:  # noqa: BLE001 - store errors stay internal
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED) from exc
        if settlement.status != "settled":
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)
        return charged
