"""ATL Credits metering for the hosted AI Hedge Fund runtime.

Upstream makes its model calls inside its own interpreter, so ATL cannot meter
them one by one the way ``infrastructure/llm/execution/service.py`` does. The
unit here is one decision (one bridge subprocess): reserve a ceiling before it,
settle at the token usage the bridge reports after it. The run's reservations
share the run id of the signed execution handoff, so the existing
``finalize_run`` (child and parent) releases any hold a killed child leaves.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard.backend.domain.agents.runtime import AI_HEDGE_FUND_ANALYSTS
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
    InsufficientCreditsError,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
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


def decision_ceiling_micro(
    model: str, analysts: Optional[Sequence[str]], ticker_count: int
) -> int:
    """The most one decision is held for, in Credit micro-units.

    One formula for the launch preflight and the per-decision hold, so the
    route cannot admit a run whose first decision the child then refuses.
    No analysts configured means upstream runs all of them.
    """
    analyst_count = len(analysts or AI_HEDGE_FUND_ANALYSTS)
    calls = (analyst_count + 1) * max(int(ticker_count), 1)
    return credits_micro_for_usd(
        estimate_cost_usd(
            model,
            calls * CEILING_INPUT_TOKENS_PER_CALL,
            calls * CEILING_OUTPUT_TOKENS_PER_CALL,
        )
    )


def parse_usage(value: Any) -> Optional[tuple[int, int]]:
    """``(input_tokens, output_tokens)`` from the bridge's report, else None."""
    if not isinstance(value, Mapping):
        return None
    counts = []
    for key in ("input_tokens", "output_tokens"):
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        counts.append(count)
    return counts[0], counts[1]


class HostedDecisionBilling:
    """Reserve, settle and release one run's per-decision Credit holds."""

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
        try:
            reservation = self.credits.reserve_llm_credits(
                user_id=self.user_id,
                run_id=self.run_id,
                call_index=call_index,
                amount_micro=int(amount_micro),
                provider_id=PROVIDER_ID,
            )
        except CreditAccountRestrictedStoreError as exc:
            try:
                balance = self.credits.get_balance(self.user_id)
                reason = balance.restriction_reason
                outstanding_micro = balance.outstanding_credits_micro
            except Exception:  # noqa: BLE001 - keep the safe fallback
                reason = None
                outstanding_micro = 0
            raise LLMExecutionError.account_restricted(
                reason, outstanding_micro
            ) from exc
        except InsufficientCreditsError as exc:
            raise LLMExecutionError(
                ExecutionErrorCategory.INSUFFICIENT_CREDITS
            ) from exc
        except Exception as exc:  # noqa: BLE001 - store errors stay internal
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED) from exc
        if reservation.status != "open":
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)
        return reservation.reservation_id

    def close(
        self,
        reservation_id: str,
        *,
        ceiling_micro: int,
        usage: Optional[tuple[int, int]],
        outcome: str,
    ) -> int:
        """Settle or release one decision's hold; returns the micro charged.

        ``outcome`` is ``completed``, ``killed`` or ``failed``. Reported usage
        is billed whatever the outcome. Without it the charge fails closed: a
        completed decision whose usage went unreported is a broken meter, and a
        killed one (a timeout, a signal, any failure past the bridge's started
        marker) could have called the model, so both settle at the ceiling.
        Only ``failed`` is released, which the adapter reports solely when the
        bridge left no file at all, i.e. died before any model call.
        """
        if usage is not None:
            input_tokens, output_tokens = usage
            uncapped = credits_micro_for_usd(
                estimate_cost_usd(self.model_id, input_tokens, output_tokens)
            )
            basis = "usage"
            if outcome == "completed" and input_tokens + output_tokens == 0:
                # Legitimate only when every selected analyst is rule-based and
                # upstream's portfolio manager pre-filled every ticker as a
                # hold (it skips its model call then). Anything else is a meter
                # that stopped counting (a provider omitting usage, a
                # langchain-core change), so it is released but never silently.
                print(
                    "WARNING: ai_hedge_fund.usage_zero completed decision "
                    "reported no model tokens",
                    flush=True,
                )
        elif outcome == "failed":
            input_tokens = output_tokens = None
            uncapped = 0
            basis = "no_model_call"
        else:
            input_tokens = output_tokens = None
            uncapped = int(ceiling_micro)
            basis = "ceiling"
            if outcome == "completed":
                print(
                    "ERROR: ai_hedge_fund.usage_unreported "
                    "billing at the decision ceiling",
                    flush=True,
                )
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
                    "ceiling_micro": int(ceiling_micro),
                    "uncapped_micro": uncapped,
                },
            )
        except Exception as exc:  # noqa: BLE001 - store errors stay internal
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED) from exc
        if settlement.status != "settled":
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)
        return charged
