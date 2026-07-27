"""``AgentRunner`` - a high-level loop so contributors only write decision logic.

A user implements an object with a ``decide(observation)`` method (and,
optionally, ``on_execution_result`` / ``on_run_completed`` hooks). The runner
handles the full lifecycle::

    create run
      -> poll next step
      -> wait while loading (bounded by max_wait_seconds)
      -> call agent.decide(observation)
      -> submit decision
      -> receive execution result
      -> repeat until completed
      -> retrieve final result

Lifecycle handling
------------------
``get_next_step`` returns one of ``loading`` / ``awaiting_decision`` /
``completed``; the backend raises for ``failed`` (``ATLRunFailedError``) and
``run_not_active`` (``ATLConflictError``). The runner handles each explicitly and
raises ``ATLAPIError`` on any unexpected status rather than spinning forever.

Hook failures
-------------
A decision is submitted to the backend *before* its hooks run. If
``on_execution_result`` or ``on_run_completed`` raises, that exception is
surfaced to the caller; the runner never resubmits an already-accepted decision
because a local hook failed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:  # Protocol is available on 3.8+; guard for older typing backends.
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore

    def runtime_checkable(cls):  # type: ignore
        return cls

from .atl_client import ATLClient
from .exceptions import ATLAPIError, ATLConflictError, ATLRunFailedError, ATLTimeoutError
from .models import Decision, Observation, RunResult


@runtime_checkable
class TradingAgentProtocol(Protocol):
    """Minimal contract an agent must satisfy to be run by ``AgentRunner``."""

    def decide(self, observation: Observation) -> Union[Decision, Dict[str, Any]]:
        ...


# Run states the backend may surface that mean "stop, this run will not proceed".
_FAILED_STATES = {"failed", "cancelled", "canceled"}

# Conflict codes that mean "the step was auto-held server-side; the run is still
# live" — e.g. the agent took longer than the environment's decision window
# (default 60s). These must NOT abort the run.
#
# NOTE: a genuinely-late decision surfaces as ``decision_deadline_exceeded``: the
# backend consults the engine decision log and, when the step was auto-held with
# decision_source == "timeout_hold", raises that documented code (MEDIUM #8).
# ``step_already_finalized`` is kept in the set too — it's what a genuine
# double-submit of an already-finalized step returns, and keeping it makes the
# runner robust against older backends / edge orderings. Do NOT drop either. The
# backend contract is locked by test_late_decision_returns_autoheld_code
# (dashboard backend test_protocol_api).
_STEP_AUTOHELD_CODES = {"decision_deadline_exceeded", "step_already_finalized"}


def wait_for_poll(
    client: Any,
    poll_interval: float,
    waited: float,
    max_wait_seconds: Optional[float],
    status: str,
) -> float:
    """Sleep one poll interval and return the new cumulative idle time.

    Shared by every runner that drives the step loop (see
    ``integrations._tradingagents_runner``) so the timeout arithmetic and its
    error code exist once. Callers must reject ``poll_interval <= 0`` before
    entering the loop, which is what keeps ``sleep_for`` non-zero here.
    """
    if max_wait_seconds is not None and waited >= max_wait_seconds:
        raise ATLTimeoutError(
            f"exceeded max_wait_seconds={max_wait_seconds} while run was '{status}'",
            code="runner_wait_timeout",
        )
    sleep_for = poll_interval
    if max_wait_seconds is not None:
        sleep_for = min(poll_interval, max(0.0, max_wait_seconds - waited))
    client.wait(sleep_for)
    return waited + sleep_for


class AgentRunner:
    """Drive an agent through a full run via an :class:`ATLClient`."""

    def __init__(self, client: ATLClient, agent: Any) -> None:
        if not callable(getattr(agent, "decide", None)):
            raise TypeError("agent must implement a callable decide(observation) method")
        self.client = client
        self.agent = agent

    def run_backtest(
        self,
        agent_version_id: str,
        *,
        environment_id: str,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        initial_cash: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        poll_interval: float = 2.0,
        max_wait_seconds: Optional[float] = 300.0,
        max_steps: Optional[int] = None,
    ) -> RunResult:
        """Create a backtest run and drive it to completion, returning the result.

        Parameters
        ----------
        poll_interval:
            Seconds to sleep between polls while the run is ``loading`` or in a
            transient state.
        max_wait_seconds:
            Maximum *cumulative* time to spend waiting on non-actionable states
            before raising :class:`ATLTimeoutError`. Pass ``None`` to wait
            indefinitely (must be explicitly chosen).
        max_steps:
            Optional cap on the number of decisions submitted (mostly for tests).
            When this cap stops the loop the run is *not* finalized, so this
            returns the metrics gathered so far rather than the (409-raising)
            final result.

        Note
        ----
        The environment enforces a per-step decision deadline (default 60s). If
        ``agent.decide`` plus submission takes longer, the backend auto-holds
        that step and the runner advances to the next one instead of failing —
        so a single slow decision never aborts the whole run.
        """
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")

        run = self.client.create_run(
            agent_version_id,
            environment_id=environment_id,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_cash=initial_cash,
            config=config,
        )

        try:
            completed = False
            steps_done = 0
            waited = 0.0
            while True:
                step = self.client.get_next_step(run.id)
                status = step.status

                if status == "completed":
                    completed = True
                    break

                if status in _FAILED_STATES:
                    raise ATLRunFailedError(
                        step.message or f"run entered state '{status}'",
                        code=status,
                        response=step.raw,
                    )

                if status == "awaiting_decision":
                    waited = 0.0  # progress made; reset the idle timer
                    decision = self.agent.decide(step.observation)
                    try:
                        result = self.client.submit_decision(run.id, step.id, decision)
                    except ATLConflictError as exc:
                        if exc.code not in _STEP_AUTOHELD_CODES:
                            raise
                        # The decision window elapsed (or the step was already
                        # finalized): the step auto-held server-side and the run
                        # is still live. Advance instead of aborting the run.
                        result = None
                    if result is not None:
                        # Decision accepted server-side; a hook raising here
                        # surfaces to the caller and never resubmits.
                        self._fire("on_execution_result", result)

                    steps_done += 1
                    if max_steps is not None and steps_done >= max_steps:
                        break
                    continue

                if status in ("loading", "pending", "executing"):
                    waited = self._wait(poll_interval, waited, max_wait_seconds, status)
                    continue

                # Any other status is unexpected; fail loudly instead of spinning.
                raise ATLAPIError(
                    f"unexpected step status '{status}' from get_next_step",
                    code="unexpected_step_status",
                    response=step.raw,
                )

            if completed:
                final = self.client.get_run_result(run.id)
                self._fire("on_run_completed", final)
                return final

            # Stopped early via max_steps: the run isn't finalized, so /result
            # would 409. Return the metrics gathered so far instead.
            return RunResult(
                run_id=run.id,
                status="running",
                metrics=self.client.get_run_metrics(run.id),
            )
        except ATLAPIError as exc:
            # Attach the run id to any backend error so the caller can locate it.
            raise exc.with_run_id(run.id)

    def _wait(
        self,
        poll_interval: float,
        waited: float,
        max_wait_seconds: Optional[float],
        status: str,
    ) -> float:
        return wait_for_poll(
            self.client, poll_interval, waited, max_wait_seconds, status
        )

    def _fire(self, hook: str, payload: Any) -> None:
        fn = getattr(self.agent, hook, None)
        if callable(fn):
            fn(payload)  # exceptions propagate intentionally


__all__ = ["AgentRunner", "TradingAgentProtocol"]
