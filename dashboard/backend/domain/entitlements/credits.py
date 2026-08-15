"""Credit metering for operator-funded LLM spend.

One credit buys one LLM-driven dashboard backtest.

**What is metered, and why only that.** A credit denominates operator money, so
it is spent exactly where the operator's own API key pays for the model:
``POST /backtest/run`` with ``decision_source='llm'``, whose background
subprocess calls the LLM once per trading hour on the server's key. Nothing
else qualifies. A rule-based run makes no model call at all. The protocol
surfaces (``/api/v1/runs``, ``/api/v2/runs``) hand the agent a market context
and take back a decision the agent's *own* client produced -- the backend never
sees a token, let alone pays for one -- so metering them would bill users for a
resource the operator never bought. Those surfaces are bounded instead by
``max_concurrent_backtests`` (``domain/runs/service.py``), which is a
concurrency control rather than a budget; the two entitlements deliberately do
not overlap.

**Granularity.** Per run, not per LLM call. A 31-day run does cost more than a
one-day run, but ``MAX_BACKTEST_DAYS`` bounds that ratio at ~31x, and the
alternative -- reserving an estimate at accept and settling against the real
``llm_calls`` at finish -- buys precision that nothing here can currently use:
there is no invoice, no per-model price applied to a balance, and no user-facing
statement in which the difference between 3 and 40 calls would ever appear. Per
run is also the unit an admin can reason about ("this account gets 50
backtests") and the unit a user experiences.

**Refunds.** The debit happens at accept, before the thread is scheduled, so an
in-flight run has already paid. It is given back only when the run turns out to
have made no LLM call at all -- a crash before the first decision, a market-data
outage, a subprocess that never started. ``agent_runs.llm_calls`` is the right
witness precisely because it is a *billing* counter: unlike ``llm_decisions``
(the H6 coverage counter, incremented only at the decision path's success exit)
it also ticks on a truncated or unparseable response, which cost real money.
Refund on zero calls means the account paid for exactly the runs that spent
something.

**Arming.** Off unless ``CREDITS_METERING_ENABLED`` is truthy. Metering changes
who can run a backtest, so it follows ``LEADERBOARD_DAILY_AUTO_DEPLOY``'s
strict-opt-in convention rather than defaulting on for anyone who deploys. Until
it is armed nothing here debits, and a balance is a number the console reports.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

# What one LLM-driven dashboard backtest costs. Not configurable: an operator
# who wants runs to cost more changes what a credit is worth to them by
# granting fewer, and a per-run price that varied by deployment would make
# every balance in the console mean something different.
RUN_CREDIT_COST = 1

_TRUTHY = ("1", "true", "yes", "on")

_ANONYMOUS_REFUSAL = (
    "Sign in to run an LLM backtest. This deployment meters LLM runs against "
    "an account's credit balance, and a signed-out session has none."
)


def metering_enabled() -> bool:
    """Whether credit metering is armed for this deployment.

    Read per call rather than captured at import: an operator flipping the
    Render var gets it on the next restart either way, but a module-level
    constant would also freeze the value for every test in the session.
    """
    return (os.getenv("CREDITS_METERING_ENABLED") or "").strip().lower() in _TRUTHY


class CreditOutcome(NamedTuple):
    """The decision, and whether it left a debit to unwind.

    ``charged`` is what the caller must carry to the run's end: it is False for
    every path that did not take a credit (metering off, store outage), so a
    refund can never give back one that was not spent.
    """

    allowed: bool
    charged: bool
    balance: Optional[int] = None
    detail: str = ""


def authorize_llm_run(user_id: Optional[int]) -> CreditOutcome:
    """Debit one credit for an LLM-driven dashboard backtest.

    ``allowed=False`` carries the 402 the caller should answer with. A signed-out
    caller is refused outright when metering is armed: there is no per-session
    balance to spend, and inventing one would make signing out the cheaper
    option -- the same incentive inversion ``resolve_owner_cap_context`` had to
    correct for the concurrency cap, except here the resource is money rather
    than a worker slot.

    Fails **open** on a store error, matching the per-owner run cap: metering is
    a budget on top of controls that still hold (the per-client 10/hour limiter
    on ``/backtest/run``, the single-flight runner, ``MAX_BACKTEST_DAYS``), so a
    Neon outage degrades to today's behaviour instead of taking LLM backtests
    down site-wide. The print is what keeps that from being silent -- an outage
    would otherwise be indistinguishable from metering being switched off.
    """
    if not metering_enabled():
        return CreditOutcome(allowed=True, charged=False)
    if not user_id:
        return CreditOutcome(allowed=False, charged=False, detail=_ANONYMOUS_REFUSAL)
    try:
        from dashboard.backend import users as users_module

        balance = users_module.user_store.try_spend_credits(
            int(user_id), RUN_CREDIT_COST
        )
    except Exception:  # noqa: BLE001 - metering must not break the runner
        # Static marker, no payload: a psycopg error stringifies with its
        # connection DSN embedded, and CodeQL flags anything reaching a log
        # sink from a caught store exception. The failing store reports its own
        # details; this line exists only to distinguish an outage from "off".
        print("credit metering: balance lookup failed; allowing this run unmetered")
        return CreditOutcome(allowed=True, charged=False)
    if balance is None:
        return CreditOutcome(
            allowed=False,
            charged=False,
            detail=(
                "This account is out of LLM backtest credits. Rule-based "
                "backtests are unaffected; ask an admin to top up the balance."
            ),
        )
    return CreditOutcome(allowed=True, charged=True, balance=balance)


def refund_llm_run(user_id: Optional[int], *, llm_calls: int) -> None:
    """Give back the run's credit when it made no LLM call.

    Called from the background thread's ``finally``, so it must never raise:
    the run's own outcome is already recorded and a refund failure must not
    replace it with a traceback. ``llm_calls > 0`` means the model was invoked
    and the money is gone -- a failed run that got half way still spent it.
    """
    if not user_id or llm_calls > 0:
        return
    try:
        from dashboard.backend import users as users_module

        users_module.user_store.refund_credits(int(user_id), RUN_CREDIT_COST)
    except Exception:  # noqa: BLE001 - see docstring
        print("credit metering: refund failed; the run's credit was not returned")
