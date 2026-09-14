"""Portfolio service: bootstrap + allocate / reclaim against agent sleeves.

**Agent sleeves are the source of truth.** ``user_portfolios.cash_available``
is a cache of ``equity - sum(agent.cash_allocation)``, refreshed by
``_reconcile`` on every read and after every write.

That is a deliberate choice over treating the ledger row as authoritative and
moving it in lockstep with each sleeve. The two live in different tables with
no shared transaction, so a lockstep design needs a compensating write on every
failure path -- and, worse, it silently inherits whatever drift already exists.
Drift is not hypothetical here: agents created before #175 (and guest agents
later claimed by a signed-in owner via ``reclaim_on_session_match``) hold a
``cash_allocation`` that never debited any ledger. Under a lockstep design
their owner's portfolio bootstraps at the full equity, so the same money is
counted twice, deleting such an agent credits cash the account never spent, and
the resulting ``cash_available > equity`` blows up a request whose agent row is
already deleted.

Deriving the figure instead makes every one of those states self-correct on the
next read, and makes delete a write that cannot fail.

**Which sleeves count is the caller's scope, not just the account.**
``GET /api/v1/agents`` answers with the owner's agents *union* the unclaimed
guest agents of the calling browser, and the Capital Allocation pie draws one
slice per agent in that answer. Summing the owner's agents alone therefore
published a total the panel beside it contradicted, and handed the difference
back as spendable cash. ``AgentScope`` carries the browser/trading session so
both figures are summed over the same set -- see ``_split_allocations`` for the
one figure that deliberately does *not* follow the scope.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from dashboard.backend.domain.backtesting.constants import (
    DEFAULT_PORTFOLIO_EQUITY,
    MAX_AGENT_CASH_ALLOCATION,
)
from dashboard.backend.domain.portfolios.repository import (
    InsufficientCashError,
    portfolio_store,
)

# Money comparisons are on floats; anything under a hundredth of a cent is
# representation noise, not a real difference.
_EPSILON = 1e-6


class InsufficientSleeveError(ValueError):
    """Raised when reclaim exceeds the agent's cash_allocation sleeve."""


class AgentScope(NamedTuple):
    """The extra agents ``GET /api/v1/agents`` surfaces beyond the owner's own.

    Mirrors the non-account halves of ``api/dependencies._owner_context`` so
    the portfolio sums the same set the My Agents list renders. Empty (the
    default) reproduces the account-only behaviour, which is what a caller with
    no browser identity -- a cron, a test, the SDK -- should get.
    """

    browser_session: Optional[str] = None
    trading_session: Optional[str] = None

    @classmethod
    def from_owner_context(cls, ctx: Optional[Mapping[str, Any]]) -> "AgentScope":
        if not ctx:
            return cls()
        return cls(
            browser_session=ctx.get("browser_session") or None,
            trading_session=ctx.get("trading_session") or None,
        )


EMPTY_SCOPE = AgentScope()


class PortfolioService:
    def __init__(self) -> None:
        # Guards the read-check-write sequence in the transfer methods, which
        # spans two tables and therefore cannot be delegated to the database.
        # Process-local: it serialises the transfers a single worker handles,
        # which is what the app runs today. Under multiple workers a losing
        # racer can still over-allocate -- but _reconcile then reports the true
        # (over-allocated) figure on the next read instead of hiding it, so the
        # failure is visible and self-correcting rather than silent corruption.
        self._locks: Dict[int, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, owner_user_id: int) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(int(owner_user_id), threading.Lock())

    # -- reconciliation ------------------------------------------------------

    def _agents_of(
        self, owner_user_id: int, scope: AgentScope = EMPTY_SCOPE
    ) -> List[Dict[str, Any]]:
        # Imported per-call, not at module scope: the tests swap
        # ``agents.repository.agent_store`` for a temp-DB instance, and a
        # module-level ``from ... import agent_store`` would bind the original.
        from dashboard.backend.domain.agents.repository import agent_store

        return agent_store.list_agents(
            owner_user_id=int(owner_user_id),
            owner_browser_session=scope.browser_session,
            trading_session_id=scope.trading_session,
        )

    def _split_allocations(
        self, owner_user_id: int, scope: AgentScope = EMPTY_SCOPE
    ) -> Tuple[float, float]:
        """``(owned, visible)`` sleeve totals, from a single query.

        ``visible`` is every sleeve the caller's My Agents list renders --
        the figure the Capital Allocation panel must agree with. ``owned`` is
        the subset actually bound to the account.

        They are returned apart because only one of them may be *persisted*.
        ``user_portfolios`` is keyed by user, so caching the visible figure
        would make a row rewrite itself on every read from a browser holding a
        different set of unclaimed agents -- two browsers of one account
        clobbering each other forever, with a log line each time.
        """
        uid = int(owner_user_id)
        owned = 0.0
        visible = 0.0
        for agent in self._agents_of(uid, scope):
            amount = float(agent.get("cash_allocation") or 0)
            visible += amount
            if agent.get("owner_user_id") == uid:
                owned += amount
        return owned, visible

    def _reconcile(
        self, owner_user_id: int, scope: AgentScope = EMPTY_SCOPE
    ) -> Dict[str, Any]:
        """Return the portfolio with cash_available derived from the sleeves."""
        uid = int(owner_user_id)
        row = portfolio_store.get_or_create(uid, equity=DEFAULT_PORTFOLIO_EQUITY)
        equity = float(row["equity"])
        owned, allocated = self._split_allocations(uid, scope)
        # Clamped at 0: a legacy account whose sleeves already exceed equity has
        # no unallocated cash, and a negative figure would read as a debt the
        # product does not model. ``allocated`` below still reports the true
        # (over-equity) total, so that state stays visible rather than hidden.
        owned_expected = max(equity - owned, 0.0)
        if abs(owned_expected - float(row["cash_available"])) > _EPSILON:
            print(
                f"portfolio reconcile: user={uid} "
                f"cash_available {float(row['cash_available']):.2f} -> "
                f"{owned_expected:.2f} "
                f"(sleeves {owned:.2f}, equity {equity:.2f})"
            )
            row = portfolio_store.set_cash_available(uid, owned_expected)
        unclaimed = max(allocated - owned, 0.0)
        return {
            **row,
            "cash_available": max(equity - allocated, 0.0),
            "allocated": allocated,
            # Named rather than folded silently into the total: an unclaimed
            # sleeve on a signed-in account is a claim that has not run yet,
            # not a steady state, and "nothing to claim" must stay tellable
            # apart from "the claim failed and nobody logged it".
            "unclaimed_allocated": unclaimed,
        }

    def get_or_create_portfolio(
        self, owner_user_id: int, scope: AgentScope = EMPTY_SCOPE
    ) -> Dict[str, Any]:
        return self._reconcile(owner_user_id, scope)

    # -- validation ----------------------------------------------------------

    def _check_new_sleeve(
        self,
        *,
        owner_user_id: int,
        old_sleeve: float,
        new_sleeve: float,
        scope: AgentScope = EMPTY_SCOPE,
    ) -> None:
        """Raise if moving a sleeve from ``old_sleeve`` to ``new_sleeve`` is invalid.

        Pure validation -- writes nothing. Callers run it before any other
        write, so a rejected transfer cannot leave half of one behind.
        """
        if new_sleeve < 0:
            raise ValueError("allocation must be >= 0")
        if new_sleeve > float(MAX_AGENT_CASH_ALLOCATION) + _EPSILON:
            raise ValueError(
                f"Agent allocation would exceed max ({MAX_AGENT_CASH_ALLOCATION:,.0f})."
            )
        delta = new_sleeve - old_sleeve
        if delta <= _EPSILON:
            return
        available = float(self._reconcile(owner_user_id, scope)["cash_available"])
        if delta > available + _EPSILON:
            raise InsufficientCashError(
                f"Insufficient unallocated cash (have {available:.2f}, need {delta:.2f})."
            )

    def check_agent_allocation(
        self,
        *,
        owner_user_id: int,
        agent: Dict[str, Any],
        new_amount: float,
        scope: AgentScope = EMPTY_SCOPE,
    ) -> None:
        """Validate an absolute sleeve change without writing anything."""
        self._check_new_sleeve(
            owner_user_id=owner_user_id,
            old_sleeve=float(agent.get("cash_allocation") or 0),
            new_sleeve=float(new_amount),
            scope=scope,
        )

    def ensure_cash_for_new_agent(
        self,
        owner_user_id: int,
        amount: float,
        scope: AgentScope = EMPTY_SCOPE,
    ) -> None:
        """Validate that a not-yet-created agent's sleeve can be funded."""
        self._check_new_sleeve(
            owner_user_id=owner_user_id,
            old_sleeve=0.0,
            new_sleeve=float(amount or 0),
            scope=scope,
        )

    # -- transfers -----------------------------------------------------------

    def _write_sleeve(
        self,
        *,
        owner_user_id: int,
        agent: Dict[str, Any],
        new_sleeve: float,
        scope: AgentScope = EMPTY_SCOPE,
    ) -> Dict[str, Any]:
        from dashboard.backend.domain.agents.repository import agent_store

        updated = agent_store.update_agent(
            agent["agent_id"], cash_allocation=float(new_sleeve)
        )
        if not updated:
            raise RuntimeError(f"agent {agent['agent_id']} disappeared during transfer")
        return {"portfolio": self._reconcile(owner_user_id, scope), "agent": updated}

    def set_agent_allocation(
        self,
        *,
        owner_user_id: int,
        agent: Dict[str, Any],
        new_amount: float,
        scope: AgentScope = EMPTY_SCOPE,
    ) -> Dict[str, Any]:
        """Set an absolute sleeve; cash_available follows from it."""
        with self._lock_for(owner_user_id):
            new_sleeve = float(new_amount)
            self.check_agent_allocation(
                owner_user_id=owner_user_id,
                agent=agent,
                new_amount=new_sleeve,
                scope=scope,
            )
            return self._write_sleeve(
                owner_user_id=owner_user_id,
                agent=agent,
                new_sleeve=new_sleeve,
                scope=scope,
            )

    def allocate_to_agent(
        self,
        *,
        owner_user_id: int,
        agent: Dict[str, Any],
        amount: float,
        scope: AgentScope = EMPTY_SCOPE,
    ) -> Dict[str, Any]:
        """Move ``amount`` from unallocated cash → the agent's sleeve."""
        amount_f = float(amount)
        if amount_f <= 0:
            raise ValueError("allocate amount must be > 0")
        with self._lock_for(owner_user_id):
            sleeve = float(agent.get("cash_allocation") or 0)
            self._check_new_sleeve(
                owner_user_id=owner_user_id,
                old_sleeve=sleeve,
                new_sleeve=sleeve + amount_f,
                scope=scope,
            )
            return self._write_sleeve(
                owner_user_id=owner_user_id,
                agent=agent,
                new_sleeve=sleeve + amount_f,
                scope=scope,
            )

    def reclaim_from_agent(
        self,
        *,
        owner_user_id: int,
        agent: Dict[str, Any],
        amount: float,
        scope: AgentScope = EMPTY_SCOPE,
    ) -> Dict[str, Any]:
        """Move ``amount`` from the agent's sleeve → unallocated cash."""
        amount_f = float(amount)
        if amount_f <= 0:
            raise ValueError("reclaim amount must be > 0")
        with self._lock_for(owner_user_id):
            sleeve = float(agent.get("cash_allocation") or 0)
            if amount_f > sleeve + _EPSILON:
                raise InsufficientSleeveError(
                    f"Insufficient agent allocation "
                    f"(have {sleeve:.2f}, need {amount_f:.2f})."
                )
            return self._write_sleeve(
                owner_user_id=owner_user_id,
                agent=agent,
                new_sleeve=max(sleeve - amount_f, 0.0),
                scope=scope,
            )

    def reclaim_all_on_delete(
        self, *, owner_user_id: Optional[int], scope: AgentScope = EMPTY_SCOPE
    ) -> Optional[Dict[str, Any]]:
        """Refresh the ledger after an agent row was deleted.

        Call *after* the delete: the sleeve is gone from the sum, so reconciling
        returns exactly the post-delete figure. There is no credit to compute,
        nothing to validate, and -- unlike a credit -- nothing that can raise on
        an account whose agent has already been destroyed.
        """
        if not owner_user_id:
            return None
        with self._lock_for(int(owner_user_id)):
            return self._reconcile(int(owner_user_id), scope)


portfolio_service = PortfolioService()

# Re-export errors for routers/tests.
__all__ = [
    "EMPTY_SCOPE",
    "AgentScope",
    "InsufficientCashError",
    "InsufficientSleeveError",
    "PortfolioService",
    "portfolio_service",
]
