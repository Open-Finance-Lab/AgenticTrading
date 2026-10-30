"""Monthly spend ledger and hard stop for the Live Leaderboard's model deploys.

The live board's model calls go straight to the operator's CommonStack key and
bypass the Credits layer, so nothing else bounds them. Checkpoint rows cannot
serve as the record either: a session H6 refuses to store, a run that dies
before its next session close, and every row a forced or config replay deletes
were all billed and leave no row behind. So every billable attempt is priced
into one ``agent_runs`` row per (month, entry), cumulative across successes,
failures, H6 refusals and replays, and the cap is checked against that.

The row is invisible to every other reader by construction: its session
(``leaderboard-live-spend``) is not the board's, its mode
(``leaderboard_spend``) is not ``leaderboard`` or ``backtest``, and its run id
does not start with ``run_``. ``clear_live_session_runs`` deletes only the
board's session, so the ledger survives a ``--clear``: that money was spent.

Spend is an estimate: provider-reported tokens times the
``infrastructure/llm/pricing.py`` snapshot, because CommonStack has no usage
endpoint. A month-to-date figure per entry is ``max(ledger, latest checkpoint
row)``, so the first deploy after this module shipped starts from what the
checkpoints already record instead of from zero.

A leaf: imports the runs database and the price table, never ``live`` or
``service`` (``live`` imports this module).
"""

from __future__ import annotations

import json
import math
import os
from datetime import date
from typing import Any, Dict, Mapping, Optional, Tuple

from dashboard.backend.database import db
from dashboard.backend.infrastructure.llm import token_cost

LIVE_SPEND_SESSION_ID = "leaderboard-live-spend"
LIVE_SPEND_MODE = "leaderboard_spend"
LEDGER_NAME = "live_spend"

BUDGET_ENV = "LEADERBOARD_MONTHLY_BUDGET_USD"
DEFAULT_MONTHLY_BUDGET_USD = 50.0
MAX_MONTHLY_BUDGET_USD = 10_000.0

OUTCOME_OK = "ok"
OUTCOME_FAILED = "failed"
OUTCOME_BUDGET_HALTED = "budget_halted"
OUTCOMES = frozenset({OUTCOME_OK, OUTCOME_FAILED, OUTCOME_BUDGET_HALTED})

KIND_BUDGET = "budget"
KIND_MODEL_COVERAGE = "model_coverage"
KIND_MARKET_DATA_GAP = "market_data_gap"
KIND_TIMEOUT = "timeout"
KIND_NETWORK = "network"
KIND_INTERRUPTED = "interrupted"
KIND_ERROR = "error"
KINDS = frozenset({
    KIND_BUDGET,
    KIND_MODEL_COVERAGE,
    KIND_MARKET_DATA_GAP,
    KIND_TIMEOUT,
    KIND_NETWORK,
    KIND_INTERRUPTED,
    KIND_ERROR,
})

# Matched by class *name* along the exception's MRO, so this leaf never imports
# the modules that define them (live.py, service.py) or any client library.
# Exact names first, in order: TimeoutError is an OSError, so it is tested
# before anything broader.
_KIND_BY_TYPE_NAME: Tuple[Tuple[str, str], ...] = (
    ("LeaderboardFallbackError", KIND_MODEL_COVERAGE),
    ("LiveSessionGapError", KIND_MARKET_DATA_GAP),
    ("TimeoutError", KIND_TIMEOUT),
    ("ConnectionError", KIND_NETWORK),
)
# Then name parts, for client-library classes that subclass neither builtin:
# psycopg_pool.PoolTimeout, httpx.ReadTimeout and anthropic.APITimeoutError
# carry "Timeout"; httpx.ConnectError and anthropic.APIConnectionError carry
# "Connect". "Timeout" goes first because APITimeoutError subclasses
# APIConnectionError. On the live path a provider SDK error never gets here
# (PortfolioManager turns it into a rule-based step, which H6 then reports as
# model_coverage); a store timeout from a checkpoint or ledger write does.
_KIND_BY_TYPE_NAME_PART: Tuple[Tuple[str, str], ...] = (
    ("Timeout", KIND_TIMEOUT),
    ("Connect", KIND_NETWORK),
)

# Junk values already reported, so a refresh that reads the cap once per entry
# prints the warning once rather than once per read.
_warned_values: set = set()


class LiveBudgetExceeded(RuntimeError):
    """The month's live model budget is spent; the entry is not traded further.

    A ``RuntimeError`` so a caller that predates it still treats it as a
    failed deploy, but ``refresh_live_leaderboard`` catches it first and
    reports it as a halt, not a failure.
    """


def monthly_budget_usd() -> float:
    """The live board's monthly model budget in USD, read on every call.

    Unset or blank is the default, silently. Junk, ``nan``/``inf``, negative
    or above ``MAX_MONTHLY_BUDGET_USD`` warns once and falls back to the
    default: this runs on every live deploy and must never raise. ``0`` is a
    real cap that blocks every billable live deploy, not "off": a hard stop
    that a typo could turn into "unlimited" would fail in the wrong direction.
    To lift the cap, set a large number.
    """
    raw = os.getenv(BUDGET_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MONTHLY_BUDGET_USD
    try:
        value: Optional[float] = float(raw.strip())
    except ValueError:
        value = None
    if (
        value is None
        or not math.isfinite(value)
        or value < 0
        or value > MAX_MONTHLY_BUDGET_USD
    ):
        if raw not in _warned_values:
            _warned_values.add(raw)
            print(
                f"WARNING: {BUDGET_ENV}={raw!r} is not a dollar amount between 0 "
                f"and {MAX_MONTHLY_BUDGET_USD:.0f}; using the default "
                f"{DEFAULT_MONTHLY_BUDGET_USD:.2f}",
                flush=True,
            )
        return DEFAULT_MONTHLY_BUDGET_USD
    return 0.0 if value == 0 else value


def spend_run_id(month_start: str, entry_id: str) -> str:
    """The ledger row's id: one per (month, entry), so writes are upserts."""
    return f"live_spend_{month_start[:4]}{month_start[5:7]}_{entry_id}"


def _metadata(run: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    meta = (run or {}).get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            return {}
    return meta if isinstance(meta, dict) else {}


def _usd(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0 else 0.0


def _count(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _spend(run: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    run = run or {}
    return {
        "usd": _usd(run.get("est_cost_usd")),
        "input_tokens": _count(run.get("input_tokens")),
        "output_tokens": _count(run.get("output_tokens")),
        "llm_calls": _count(run.get("llm_calls")),
    }


def month_ledger_rows(month_start: str) -> Dict[str, Dict[str, Any]]:
    """This month's ledger row per entry id."""
    rows: Dict[str, Dict[str, Any]] = {}
    for run in db.get_runs_by_session(LIVE_SPEND_SESSION_ID) or []:
        if run.get("mode") != LIVE_SPEND_MODE or run.get("start_date") != month_start:
            continue
        entry_id = run.get("llm_model")
        if entry_id:
            rows[str(entry_id)] = run
    return rows


def _merged_spend(
    ledger: Mapping[str, Mapping[str, Any]],
    checkpoint_rows: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Month-to-date spend per entry: whichever of ledger and checkpoint is larger.

    In steady state the ledger is the larger (it also holds failed and replayed
    attempts). The checkpoint wins only before an entry's first priced attempt,
    which is what makes the first deploy after rollout start from real spend.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for entry_id in set(ledger) | set(checkpoint_rows):
        ledger_spend = _spend(ledger.get(entry_id))
        checkpoint_spend = _spend(checkpoint_rows.get(entry_id))
        merged[entry_id] = (
            checkpoint_spend
            if checkpoint_spend["usd"] > ledger_spend["usd"]
            else ledger_spend
        )
    return merged


def budget_status(
    month_start: str,
    *,
    checkpoint_rows: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Cap, month-to-date spend and whether the cap is reached. Operator-facing:
    the refresh result and log carry it, the public GET never does."""
    spend = _merged_spend(month_ledger_rows(month_start), checkpoint_rows or {})
    spent = round(sum(item["usd"] for item in spend.values()), 6)
    cap = monthly_budget_usd()
    return {
        "month": month_start[:7],
        "cap_usd": cap,
        "spent_usd": spent,
        "remaining_usd": round(max(cap - spent, 0.0), 6),
        "halted": spent >= cap,
    }


def classify_attempt(exc: Optional[BaseException]) -> Tuple[str, Optional[str]]:
    """``(outcome, kind)`` for how an attempt ended. The kind is a coarse
    class and never the exception's message, which can carry provider text."""
    if exc is None:
        return OUTCOME_OK, None
    if isinstance(exc, LiveBudgetExceeded):
        return OUTCOME_BUDGET_HALTED, KIND_BUDGET
    if not isinstance(exc, Exception):
        return OUTCOME_FAILED, KIND_INTERRUPTED
    names = [cls.__name__ for cls in type(exc).__mro__]
    for name, kind in _KIND_BY_TYPE_NAME:
        if name in names:
            return OUTCOME_FAILED, kind
    for part, kind in _KIND_BY_TYPE_NAME_PART:
        if any(part in name for name in names):
            return OUTCOME_FAILED, kind
    return OUTCOME_FAILED, KIND_ERROR


def clean_last_attempt(value: Any) -> Optional[Dict[str, Any]]:
    """The public ``{on, outcome, kind}`` triple, or None for anything else.

    Whitelisted rather than passed through, so a key a later writer adds (an
    error message, a dollar figure) cannot reach the public payload.
    """
    if not isinstance(value, dict):
        return None
    on = value.get("on")
    outcome = value.get("outcome")
    if not isinstance(on, str) or outcome not in OUTCOMES:
        return None
    try:
        date.fromisoformat(on)
    except ValueError:
        return None
    kind = value.get("kind")
    if outcome == OUTCOME_OK:
        kind = None
    elif kind not in KINDS:
        kind = KIND_ERROR
    return {"on": on, "outcome": outcome, "kind": kind}


def last_attempts(month_start: str) -> Dict[str, Dict[str, Any]]:
    """Each entry's last finished attempt this month, as ``clean_last_attempt``."""
    out: Dict[str, Dict[str, Any]] = {}
    for entry_id, run in month_ledger_rows(month_start).items():
        last = clean_last_attempt(_metadata(run).get("last_attempt"))
        if last is not None:
            out[entry_id] = last
    return out


class LiveAttempt:
    """One billable deploy of one live entry, priced into the month's ledger.

    Built only once a deploy has work to do, so an entry that is already
    current costs nothing and writes nothing. ``check_can_start`` is the
    pre-check (before any bar fetch or model build), ``record_session`` the
    upsert at every session close, ``stop_if_next_session_would_cross`` the
    in-run stop, and ``finish`` the final pricing, whatever ended the attempt.
    """

    def __init__(
        self,
        entry: Mapping[str, Any],
        *,
        month_start: str,
        freeze_end: str,
        checkpoint_rows: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.entry_id = str(entry["id"])
        self.agent_name = str(entry.get("name") or self.entry_id)
        self.default_model_id = entry.get("model_id")
        self.month_start = month_start
        self.freeze_end = freeze_end
        self.cap_usd = monthly_budget_usd()
        self.strategy_impl: Any = None
        self.sessions_closed = 0
        ledger = month_ledger_rows(month_start)
        spend = _merged_spend(ledger, checkpoint_rows)
        self._base = spend.get(self.entry_id) or _spend(None)
        self._others_usd = sum(
            item["usd"] for entry_id, item in spend.items() if entry_id != self.entry_id
        )
        meta = _metadata(ledger.get(self.entry_id))
        self._prior_attempts = _count(meta.get("attempts"))
        self._prior_last = clean_last_attempt(meta.get("last_attempt"))

    def attempt_usd(self) -> float:
        """What this attempt has billed so far, from the strategy's own tokens."""
        impl = self.strategy_impl
        if impl is None:
            return 0.0
        model_id = getattr(impl, "model_id", None) or self.default_model_id
        return token_cost.estimate_cost_usd(
            model_id,
            _count(getattr(impl, "input_tokens", 0)),
            _count(getattr(impl, "output_tokens", 0)),
        )

    def entry_spent_usd(self) -> float:
        return round(self._base["usd"] + self.attempt_usd(), 6)

    def month_spent_usd(self) -> float:
        return round(self._others_usd + self.entry_spent_usd(), 6)

    def _exceeded(self, spent: float, why: str) -> LiveBudgetExceeded:
        return LiveBudgetExceeded(
            f"Live '{self.entry_id}': month {self.month_start[:7]} has spent "
            f"${spent:.2f} of its ${self.cap_usd:.2f} model budget ({BUDGET_ENV}); {why}"
        )

    def check_can_start(self) -> None:
        spent = self.month_spent_usd()
        if spent >= self.cap_usd:
            raise self._exceeded(spent, "not started")

    def record_session(self) -> None:
        """Upsert the ledger with everything billed so far. Called at every
        session close before H6 is consulted, so a session H6 refuses to store
        is priced too, and a process killed later loses at most one session."""
        self.sessions_closed += 1
        self._write(last_attempt=self._prior_last, attempts=self._prior_attempts)

    def stop_if_next_session_would_cross(self) -> None:
        """Raise when this attempt's own average session cost would take the
        month past the cap. Only called while a session is still to come."""
        if self.sessions_closed <= 0:
            return
        spent = self.month_spent_usd()
        per_session = self.attempt_usd() / self.sessions_closed
        if spent + per_session > self.cap_usd:
            raise self._exceeded(
                spent,
                f"stopped before a session averaging ${per_session:.4f} would cross it",
            )

    def finish(self, exc: Optional[BaseException]) -> None:
        """Final upsert, with how the attempt ended as ``last_attempt``."""
        outcome, kind = classify_attempt(exc)
        self._write(
            last_attempt={"on": self.freeze_end, "outcome": outcome, "kind": kind},
            attempts=self._prior_attempts + 1,
        )

    def _write(self, *, last_attempt: Optional[Dict[str, Any]], attempts: int) -> None:
        impl = self.strategy_impl
        metadata: Dict[str, Any] = {
            "ledger": LEDGER_NAME,
            "price_source": token_cost.PRICING_SOURCE_VERSION,
            "attempts": attempts,
        }
        if last_attempt is not None:
            metadata["last_attempt"] = last_attempt
        db.insert_run(
            run_id=spend_run_id(self.month_start, self.entry_id),
            session_id=LIVE_SPEND_SESSION_ID,
            agent_name=self.agent_name,
            mode=LIVE_SPEND_MODE,
            start_date=self.month_start,
            end_date=self.freeze_end,
            initial_equity=0.0,
            llm_model=self.entry_id,
            llm_calls=self._base["llm_calls"] + _count(getattr(impl, "llm_calls", 0)),
            input_tokens=self._base["input_tokens"]
            + _count(getattr(impl, "input_tokens", 0)),
            output_tokens=self._base["output_tokens"]
            + _count(getattr(impl, "output_tokens", 0)),
            est_cost_usd=self.entry_spent_usd(),
            metadata=metadata,
        )
