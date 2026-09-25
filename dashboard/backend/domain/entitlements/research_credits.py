"""Credit metering for research-agent runs (design N2/PR2 billing).

**Route-0 semantics (interim, teacher-approved direction).** One research run
costs ``RESEARCH_RUN_CREDIT_COST`` credits, debited at submit. Today the
provider key belongs to the agent author, not the platform — the operator is
not yet paying the LLM bill — but the *user-facing* currency is already the
platform credit, and the operator will settle with the author separately.
When the LLM gateway lands (route 1) the same functions keep working: the
debit's meaning shifts from "pre-paid the author" to "paid the operator's
key" without a call-site change.

**Configurable on purpose.** Unlike ``RUN_CREDIT_COST`` (backtests), the price
lives in ``RESEARCH_RUN_CREDIT_COST``: Deep Research spend varies by agent and
provider tier in a way backtest granularity never did, and the operator tunes
the number in the Render dashboard rather than by redeploying.

**Arming.** Same strict opt-in as backtest metering: nothing debits unless
``CREDITS_METERING_ENABLED`` is truthy, so existing deployments behave
exactly as before until the operator arms billing.

**Refunds.** The debit happens before the agent service is called. It is
refunded only when the service never accepted the run (connection failure,
5xx, 422 we passed through after charging) — i.e. when no Deep Research
invocation can have started. Once the service accepts, a later failed run
still spent provider money and is not refunded.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

_TRUTHY = ("1", "true", "yes", "on")

_ANONYMOUS_REFUSAL = (
    "Sign in to start a research run. This deployment meters research runs "
    "against an account's credit balance, and a signed-out session has none."
)


def metering_enabled() -> bool:
    """Same master switch as backtest metering: one flag, one billing story."""
    return (os.getenv("CREDITS_METERING_ENABLED") or "").strip().lower() in _TRUTHY


def research_run_cost() -> int:
    """Credits per research run, from env; falls back to 10 when unset/unparseable."""
    raw = (os.getenv("RESEARCH_RUN_CREDIT_COST") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 10
    return value if value > 0 else 10


class ResearchCreditOutcome(NamedTuple):
    """Mirror of credits.CreditOutcome for the research surface."""

    allowed: bool
    charged: bool
    balance: Optional[int] = None
    detail: str = ""


def authorize_research_run(user_id: Optional[int]) -> ResearchCreditOutcome:
    """Debit credits for one research run. See module docstring for semantics.

    Fails **open** on a store error, matching ``authorize_llm_run``: a store
    outage must not take the research surface down site-wide; the print keeps
    the degradation from reading as "metering off".
    """
    if not metering_enabled():
        return ResearchCreditOutcome(allowed=True, charged=False)
    if not user_id:
        return ResearchCreditOutcome(allowed=False, charged=False, detail=_ANONYMOUS_REFUSAL)
    cost = research_run_cost()
    try:
        from dashboard.backend import users as users_module

        balance = users_module.user_store.try_spend_credits(int(user_id), cost)
    except Exception:  # noqa: BLE001 - metering must not break the surface
        print("research metering: balance lookup failed; allowing this run unmetered")
        return ResearchCreditOutcome(allowed=True, charged=False)
    if balance is None:
        return ResearchCreditOutcome(
            allowed=False,
            charged=False,
            detail=(
                f"This account is out of credits for research runs (cost: {cost} "
                "credits per run). Ask an admin to top up the balance."
            ),
        )
    return ResearchCreditOutcome(allowed=True, charged=True, balance=balance)


def refund_research_run(user_id: Optional[int]) -> None:
    """Give back the debit when the agent service never accepted the run.

    Called from the route's failure path, so it must never raise.
    """
    if not user_id:
        return
    try:
        from dashboard.backend import users as users_module

        users_module.user_store.refund_credits(int(user_id), research_run_cost())
    except Exception:  # noqa: BLE001 - see docstring
        print("research metering: refund failed; the run's credits were not returned")
