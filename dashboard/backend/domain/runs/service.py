"""Agent-Environment Protocol orchestration (Run API service layer).

This is a thin adapter over the existing external backtest engine
(``external_backtest_service``). It exposes the generalized Run/Step/Decision/
ExecutionResult protocol while delegating all data loading, validation,
execution and persistence to the engine and ``llm_validator``. It adds:

* stable, immutable step IDs over the engine's single mutable "current step",
* idempotency keys and per-step finalization guards,
* protocol-shaped Observation / ExecutionResult payloads,
* a simple run/step state machine and clear deadline errors.

Moved verbatim (Phase 3B1) from ``dashboard/backend/run_service.py``; the
original module was removed in Phase 4A. Public functions, ``ProtocolRun``,
module-level registries, signatures, and behavior are unchanged; only the module
location moved (and the ``run_store`` import now points at the canonical
repository).
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import dashboard.backend.domain.backtesting.external_run_service as ebs
from dashboard.backend.database import db
from dashboard.backend.domain.backtesting.constants import (
    INITIAL_CAPITAL,
    MAX_BACKTEST_INITIAL_CAPITAL,
    resolve_initial_capital,
)
from dashboard.backend.execution.base import TERMINAL_STATUSES
from dashboard.backend.domain.runs.environment import get_environment
from dashboard.backend.infrastructure.llm.validator import DJIA_30, MAX_ORDER_SHARES
from dashboard.backend.domain.runs.protocol import (
    PROTOCOL_VERSION,
    VALID_SIDES,
    DecisionIn,
    ProtocolError,
    order_to_action,
    resolve_order_quantity,
)
from dashboard.backend.domain.runs.repository import run_store
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation

_runs: Dict[str, "ProtocolRun"] = {}
_registry_lock = threading.Lock()

# Cap concurrent (non-terminal) runs per agent to bound resource use / abuse.
MAX_ACTIVE_RUNS_PER_AGENT = int(os.getenv("MAX_ACTIVE_RUNS_PER_AGENT", "5"))
# Global backstop across ALL agents (Decision 3 of the 2026-07-24 scale spec):
# beyond this, creates 429 with Retry-After instead of degrading everyone.
# Default 100 = the highest load ever exercised by the loadtest harness; raise
# it only after a measured smoke run at the higher value. 0 disables.
MAX_ACTIVE_RUNS_GLOBAL = int(os.getenv("MAX_ACTIVE_RUNS_GLOBAL", "100"))
# How often the background reaper drains abandoned runs and evicts terminal ones.
REAPER_INTERVAL_SECONDS = float(os.getenv("RUN_REAPER_INTERVAL_SECONDS", "60"))
# Startup recovery marks ALL non-terminal rows failed; only correct when a single
# process owns the DB (the current single-instance deployment). A multi-worker or
# overlapping rolling-deploy setup sharing one DB must disable it (set to 0) and
# rely on heartbeat-based stale recovery instead (RUN_HEARTBEAT_STALE_SECONDS):
# the reaper heartbeats every live run each pass, so only runs whose owning
# process died go stale and get failed — a sibling worker's runs are safe.
RUN_RECOVERY_ON_STARTUP = os.getenv("RUN_RECOVERY_ON_STARTUP", "1").lower() not in ("0", "false", "no")
# How long a non-terminal run may go without a heartbeat before any instance's
# reaper declares its owner dead and fails it. Must be comfortably larger than
# REAPER_INTERVAL_SECONDS (heartbeat cadence); 0 disables stale recovery.
RUN_HEARTBEAT_STALE_SECONDS = float(os.getenv("RUN_HEARTBEAT_STALE_SECONDS", "300"))

_reaper_thread: Optional[threading.Thread] = None
_reaper_lock = threading.Lock()
# Serializes create_run so the per-agent active-run cap can't be raced past.
# Shared with the v2 create path (api/v2/runs.py) — both surfaces count active
# runs from the same protocol_runs ledger, so both check-then-insert sequences
# must serialize on the same lock or an agent could race one create per surface
# past the combined cap.
_create_lock = threading.Lock()


def _analytics_owner_user_id(agent_id: Optional[str]) -> Optional[int]:
    """Resolve a persisted Agent owner without inventing API-only subjects."""

    if not agent_id:
        return None
    try:
        from dashboard.backend.domain.agents.repository import agent_store

        agent = agent_store.get_agent(agent_id)
        owner_user_id = agent.get("owner_user_id") if agent else None
        return int(owner_user_id) if owner_user_id is not None else None
    except Exception:
        return None


def _emit_run_transition(
    *,
    event_name: str,
    run_id: str,
    agent_id: Optional[str],
    error_category: Optional[str] = None,
) -> None:
    user_id = _analytics_owner_user_id(agent_id)
    if user_id is None:
        return
    analytics_instrumentation.emit_run_event(
        event_name=event_name,
        user_id=user_id,
        run_id=run_id,
        error_category=error_category,
    )


def resolve_owner_cap_context(agent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Prefetch the per-owner cap's inputs; ``None`` means no cap applies.

    Two owner shapes, one budget each: an **account** (its
    ``max_concurrent_backtests`` entitlement, shared by every agent it owns) or,
    for an unclaimed agent, the **browser session** that created it at the
    default quota. The second exists because the first alone made signing in
    strictly worse than staying anonymous — see the branch comment below.

    Called BEFORE ``_create_lock`` on purpose: every read here is a remote
    Postgres round-trip in a durable deployment (entitlements on
    ``USERS_DATABASE_URL``, ownership on ``CONTENT_DATABASE_URL``), and the
    create lock serializes every v1+v2 run creation on the server — holding it
    across network I/O would queue every unrelated create behind one owner's
    DB latency. Only the local count-then-insert is a race the lock must close
    (see ``check_owner_active_run_cap``); the limit and owned-id list are as
    fresh just before the lock as inside it — an admin editing them mid-create
    was never a race the lock protected against.

    Fails open on store errors: run creation must not gain a hard dependency
    on the users/content databases being reachable (before this cap it had
    none), and the per-agent + global caps are local and still bound the
    request. The print marks the boundary — an outage here otherwise looks
    exactly like "no cap configured".
    """
    owner_user_id = agent.get("owner_user_id")
    creating_id = agent.get("agent_id")
    # Lazy imports: users triggers store construction (SESSION_HASH_SECRET
    # resolution) and agents may sit on CONTENT_DATABASE_URL Postgres; neither
    # belongs in this module's import graph for callers that never create runs.
    try:
        from dashboard.backend import users as users_module
        from dashboard.backend.domain.agents.repository import agent_store

        if owner_user_id:
            limit = int(
                users_module.user_store.get_entitlements(int(owner_user_id))[
                    "max_concurrent_backtests"
                ]
            )
            scope = "account"
            agent_ids = [
                a.get("agent_id")
                for a in agent_store.list_agents(owner_user_id=int(owner_user_id))
                if a.get("agent_id")
            ]
        elif creating_id:
            # Unclaimed agent. Returning None here — "no account, so no cap" —
            # is what made signing in strictly worse than staying anonymous:
            # a logged-in user shared one account budget across every agent
            # they owned, while the same person logged out got the per-agent
            # cap multiplied by however many agents they registered. Same
            # engine, same LLM spend, opposite direction of travel. Bill the
            # browser session the agents were created under at the default
            # budget instead, so the entitlement is a floor everyone starts
            # from rather than a penalty for having an account.
            #
            # Not a bound against a determined caller: a browser session is a
            # self-chosen header, free to rotate, and no per-key budget can fix
            # that. MAX_ACTIVE_RUNS_GLOBAL is what actually bounds the server;
            # this closes the incentive, not the hole.
            limit = int(users_module.DEFAULT_MAX_CONCURRENT_BACKTESTS)
            scope = "session"
            agent_ids = list(agent_store.list_owner_scope_agent_ids(creating_id))
        else:
            return None
    except Exception:  # noqa: BLE001 - the cap must not break creation
        # Static marker, no payload: store exceptions can carry connection
        # details (a psycopg error embeds the DSN), and CodeQL
        # py/clear-text-logging-sensitive-data flags anything flowing here
        # from the agent dict or the raised error. This line exists only to
        # make the outage distinguishable from "no cap configured" — the
        # failing store reports the details itself.
        print(
            "owner run cap: entitlement lookup failed; "
            "skipping the per-account cap for this create"
        )
        return None
    if creating_id and creating_id not in agent_ids:
        agent_ids.append(creating_id)
    return {"limit": limit, "agent_ids": agent_ids, "scope": scope}


def check_owner_active_run_cap(
    context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Per-owner concurrency entitlement; ``None`` means clear to create.

    ``max_concurrent_backtests`` (the admin-console entitlement) binds here:
    every agent owned by one account shares the account's budget of active
    protocol runs, across both the v1 and v2 surfaces. Both call this under
    the shared ``_create_lock`` with a context prefetched OUTSIDE it by
    ``resolve_owner_cap_context``, so the lock only ever waits on the local
    SQLite count below — that count-then-insert is the one race it must
    close. The dashboard's own runner needs no per-account gate (it is
    single-flight for the whole process).

    Unclaimed agents are billed to the browser session that created them at the
    default quota, so the same person cannot enlarge their budget by logging
    out — see ``resolve_owner_cap_context`` for why that is an incentive fix and
    not a bound.

    The legacy ``/api/v1/backtest/*`` surface still carries no owner identity to
    bill against, and its runs never reach ``protocol_runs`` at all, so this
    function cannot see them. It carries its own budgets instead
    (``domain/backtesting/external_run_service``).

    ``credits`` deliberately has no counterpart *here*. It meters the one
    path that spends the operator's own LLM budget — the dashboard's backtest
    runner — while these surfaces hand the agent a market context and take
    back a decision the agent's own LLM client produced, at no cost to the
    operator. Charging them would bill a user for a resource nobody bought.
    See ``domain/entitlements/credits.py``.
    """
    if context is None:
        return None
    active = run_store.count_active_runs_for_agents(context["agent_ids"])
    if active >= context["limit"]:
        return {
            "active_runs_for_account": active,
            "limit": context["limit"],
            "scope": context.get("scope", "account"),
        }
    return None


def owner_cap_message(violation: Dict[str, Any]) -> str:
    """One message for both surfaces, so the SDK sees a single wording."""
    if violation.get("scope") == "session":
        # An anonymous caller has no admin to ask and no account page to look
        # at, so pointing them at one would be a dead end. Signing in is the
        # action that actually changes their limit.
        return (
            "This browser session is at its concurrent-backtest limit "
            f"({violation['active_runs_for_account']} of {violation['limit']} active); "
            "wait for a run to finish, cancel one, or sign in to use an account limit"
        )
    return (
        "This account is at its concurrent-backtest limit "
        f"({violation['active_runs_for_account']} of {violation['limit']} active); "
        "wait for a run to finish, cancel one, or ask an admin to raise the limit"
    )

# Extra per-pass reaper work registered by other surfaces (the composition root
# registers the v2 sweep here). Kept as a registry so this domain module never
# imports api/* (layering: domain must not depend on the API layer).
_extra_reaper_sweeps: List[Callable[[], Any]] = []


def register_reaper_sweep(sweep: Callable[[], Any]) -> None:
    """Register a callable the reaper invokes once per pass (idempotent)."""
    if sweep not in _extra_reaper_sweeps:
        _extra_reaper_sweeps.append(sweep)

# Engine status -> protocol run status
_RUN_STATUS_MAP = {
    "loading": "running",
    "waiting_decision": "running",
    "completed": "completed",
    "failed": "failed",
}


def _new_decision_id() -> str:
    return f"dec_{uuid.uuid4().hex[:12]}"


def _new_step_id() -> str:
    return f"step_{uuid.uuid4().hex[:12]}"


class ProtocolRun:
    """In-memory protocol state for one run, layered on an engine session."""

    def __init__(self, *, record: Dict[str, Any], environment: Dict[str, Any]):
        self.run_id: str = record["run_id"]
        self.backtest_id: Optional[str] = record.get("backtest_id")
        self.session_id: str = record["session_id"]
        self.config: Dict[str, Any] = record.get("config") or {}
        self.environment = environment
        self.result_run_id: Optional[str] = record.get("result_run_id")
        self.status: str = record.get("status") or "running"

        self.lock = threading.Lock()
        self.seq_to_step_id: Dict[int, str] = {}
        self.step_seq: Dict[str, int] = {}
        self.step_meta: Dict[str, Dict[str, Any]] = {}
        # idempotency_key -> stored ExecutionResult
        self.idempotency: Dict[str, Dict[str, Any]] = {}
        # sequence -> {"idempotency_key", "result"}
        self.step_results_by_seq: Dict[int, Dict[str, Any]] = {}

    def session(self):
        if not self.backtest_id:
            return None
        return ebs.get_session(self.backtest_id)

    def ensure_step_id(self, seq: int, timestamp: Any, deadline: Any) -> str:
        # Under the run lock: two concurrent steps/next polls must not mint two
        # step_ids for one sequence — protocol_steps' UNIQUE(run_id, sequence)
        # would turn the old benign last-write-wins race into an IntegrityError.
        with self.lock:
            sid = self.seq_to_step_id.get(seq)
            if sid is None:
                sid = _new_step_id()
                self.seq_to_step_id[seq] = sid
                self.step_seq[sid] = seq
                self.step_meta[sid] = {
                    "sequence": seq,
                    "timestamp": timestamp,
                    "deadline_at": deadline,
                    "status": "awaiting_decision",
                }
            else:
                meta = self.step_meta[sid]
                if (
                    meta.get("timestamp") == timestamp
                    and meta.get("deadline_at") == deadline
                    and meta.get("status") == "awaiting_decision"
                ):
                    # Repeat poll of an unchanged step: steps/next is polled in
                    # a loop, so skip the per-poll synchronous SQLite write.
                    return sid
                meta.update(
                    {"timestamp": timestamp, "deadline_at": deadline, "status": "awaiting_decision"}
                )
            # Mirror to protocol_steps so step-id lookups and idempotent replays
            # survive a process restart (rehydrated by _get_run).
            run_store.save_step(self.run_id, sid, seq, timestamp, deadline)
            return sid

    def constraints(self) -> Dict[str, Any]:
        env_constraints = self.environment.get("constraints", {})
        # Always resolve to a concrete allow-list so enforcement in
        # ``submit_decision`` has something to check against (fall back to the
        # full DJIA-30 universe when neither the run config nor the environment
        # narrows it).
        symbols = (
            self.config.get("symbols")
            or self.environment.get("universe")
            or list(DJIA_30)
        )
        return {
            "allowed_symbols": symbols,
            "allow_short": bool(env_constraints.get("allow_short", False)),
            "max_position_weight": env_constraints.get("max_position_weight", 0.25),
            "max_orders": env_constraints.get("max_orders", 10),
        }


def _rehydrate_steps(run: "ProtocolRun") -> None:
    """Rebuild the in-memory step bookkeeping from protocol_steps after a
    restart (or after the reaper dropped a terminal run from the registry)."""
    for row in run_store.get_steps(run.run_id):
        seq = row["sequence"]
        sid = row["step_id"]
        run.seq_to_step_id[seq] = sid
        run.step_seq[sid] = seq
        run.step_meta[sid] = {
            "sequence": seq,
            "timestamp": row.get("timestamp"),
            "deadline_at": row.get("deadline_at"),
            "status": row.get("status") or "awaiting_decision",
        }
        if row.get("result") is not None and row.get("idempotency_key"):
            run.idempotency[(sid, row["idempotency_key"])] = row["result"]
            run.step_results_by_seq[seq] = {
                "idempotency_key": row["idempotency_key"],
                "result": row["result"],
            }


def _get_run(run_id: str) -> "ProtocolRun":
    with _registry_lock:
        run = _runs.get(run_id)
    if run is None:
        # Allow read-only access to a finalized run after the in-memory
        # session is gone (e.g. metrics/result fetched later).
        record = run_store.get_run(run_id)
        if record is None:
            raise ProtocolError("run_not_found", "Run not found", status_code=404)
        env = get_environment(record.get("environment_id")) or {}
        run = ProtocolRun(record=record, environment=env)
        _rehydrate_steps(run)
        with _registry_lock:
            # Double-check under the lock: a concurrent caller may have built and
            # registered the same run while we were constructing ours. Return the
            # winner so both callers share one ProtocolRun (one lock, one cache).
            existing = _runs.get(run_id)
            if existing is not None:
                return existing
            _runs[run_id] = run
    return run


def _sync_status(run: "ProtocolRun") -> Dict[str, Any]:
    """Reconcile engine status into the protocol run; persist completion."""
    session = run.session()
    if session is None:
        return {"engine_status": None, "run_status": run.status}

    engine_status = session.get_status()  # applies timeout side-effects
    estatus = engine_status.get("status")
    previous_status = run.status
    run.status = _RUN_STATUS_MAP.get(estatus, run.status)
    record = run_store.get_run(run.run_id) or {}

    if estatus == "completed" and not run.result_run_id and session.run_id:
        run.result_run_id = session.run_id
        run_store.update_run(
            run.run_id,
            result_run_id=session.run_id,
            status="completed",
        )
        if previous_status != "completed":
            _emit_run_transition(
                event_name="backtest_completed",
                run_id=run.run_id,
                agent_id=record.get("agent_id"),
            )
    elif estatus == "failed":
        run_store.update_run(run.run_id, status="failed")
        if previous_status != "failed":
            _emit_run_transition(
                event_name="backtest_failed",
                run_id=run.run_id,
                agent_id=record.get("agent_id"),
                error_category="internal_error",
            )
    return {"engine_status": engine_status, "run_status": run.status}


# ----------------------------------------------------------------------
# Run lifecycle: startup recovery + background reaper
# ----------------------------------------------------------------------


def recover_orphaned_runs() -> int:
    """Fail runs left non-terminal by a crash/restart (their in-memory engine
    session is gone and cannot resume). Call once on process startup.

    Marks EVERY non-terminal row failed, so it is only correct when a single
    process owns the DB. Gated behind RUN_RECOVERY_ON_STARTUP for multi-worker
    deployments (see the constant)."""
    if not RUN_RECOVERY_ON_STARTUP:
        return 0
    return run_store.fail_unfinished_runs()


def reap_runs() -> int:
    """Drive abandoned runs forward through any elapsed decision deadlines, then
    free terminal runs entirely: evict the heavy engine session (market-data
    buffers) AND drop the ProtocolRun from ``_runs`` — its step-id map and
    idempotency cache are persisted in protocol_steps, so reads and idempotent
    retries keep working via _get_run's rehydration. Idempotent and safe to
    call periodically. Returns the number of sessions evicted this pass."""
    with _registry_lock:
        runs = list(_runs.values())
    reaped = 0
    live_run_ids: List[str] = []
    for run in runs:
        try:
            session = run.session()
            if session is not None:
                session.drain_expired()
                _sync_status(run)
                if run.status in TERMINAL_STATUSES and run.backtest_id:
                    if ebs.evict_session(run.backtest_id):
                        reaped += 1
                elif run.status not in TERMINAL_STATUSES:
                    # Live in this process: keep its heartbeat fresh so no
                    # sibling instance's stale recovery can fail it. A
                    # non-terminal run WITHOUT a session is deliberately not
                    # heartbeated — it can never progress here, so letting it
                    # go stale is exactly how it gets recovered.
                    live_run_ids.append(run.run_id)
            if run.status in TERMINAL_STATUSES:
                # Terminal state is DB-backed (protocol_runs + protocol_steps);
                # keeping the ProtocolRun would only grow _runs forever. But a
                # request may be mid-flight on this run (holding run.lock, its
                # finalize_step not yet committed) — popping now would strand
                # its writes on an orphaned object and let a concurrent reader
                # rehydrate a stale row. Defer to the next sweep instead.
                if run.lock.acquire(blocking=False):
                    try:
                        with _registry_lock:
                            _runs.pop(run.run_id, None)
                    finally:
                        run.lock.release()
        except Exception as exc:  # a single wedged run must not stall the sweep
            print(f"⚠️ reap_runs: skipping {run.run_id}: {exc}")
    try:
        run_store.heartbeat_runs(live_run_ids)
    except Exception as exc:
        print(f"⚠️ reap_runs: heartbeat pass failed: {exc}")
    # Other surfaces' sweeps (e.g. v2: drain deadlines, heartbeat, archive
    # terminal backends) — registered via register_reaper_sweep so they run
    # BEFORE stale recovery and their live runs never look abandoned.
    for sweep in list(_extra_reaper_sweeps):
        try:
            sweep()
        except Exception as exc:
            print(f"⚠️ reap_runs: registered sweep failed: {exc}")
    if RUN_HEARTBEAT_STALE_SECONDS > 0:
        try:
            stale = run_store.fail_stale_runs(RUN_HEARTBEAT_STALE_SECONDS)
            if stale:
                print(f"🧹 stale-run recovery: failed {stale} abandoned run(s)")
        except Exception as exc:
            print(f"⚠️ reap_runs: stale recovery failed: {exc}")
    return reaped


def start_reaper(interval_seconds: Optional[float] = None) -> None:
    """Start the background reaper daemon (idempotent — a second call no-ops)."""
    global _reaper_thread
    interval = interval_seconds if interval_seconds is not None else REAPER_INTERVAL_SECONDS
    with _reaper_lock:
        if _reaper_thread is not None and _reaper_thread.is_alive():
            return

        def _loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    reap_runs()
                except Exception as exc:
                    print(f"⚠️ run reaper pass failed: {exc}")

        _reaper_thread = threading.Thread(target=_loop, daemon=True, name="run-reaper")
        _reaper_thread.start()


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def create_run(
    *,
    agent: Dict[str, Any],
    agent_version: Dict[str, Any],
    environment_id: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    environment = get_environment(environment_id)
    if environment is None:
        raise ProtocolError("unknown_environment", f"Unknown environment '{environment_id}'", 404)
    if environment.get("type") != "backtest":
        raise ProtocolError(
            "unsupported_environment",
            "Only backtest environments are supported in this version",
            400,
        )

    start_date = config.get("start_date")
    end_date = config.get("end_date")
    if not start_date or not end_date:
        raise ProtocolError("invalid_config", "config.start_date and config.end_date are required", 400)

    symbols = config.get("symbols")
    if symbols:
        invalid = [s for s in symbols if s not in DJIA_30]
        if invalid:
            raise ProtocolError(
                "invalid_symbols",
                f"Symbols not in environment universe: {invalid}",
                400,
                details={"invalid_symbols": invalid},
            )

    # Simulation capital is independent of the agent's portfolio sleeve
    # (cash_allocation). Callers may set config.initial_cash up to
    # MAX_BACKTEST_INITIAL_CAPITAL; omitted → INITIAL_CAPITAL.
    initial_cash = config.get("initial_cash")
    if initial_cash is None:
        resolved_capital = float(INITIAL_CAPITAL)
    else:
        try:
            requested = float(initial_cash)
        except (TypeError, ValueError):
            raise ProtocolError(
                "invalid_config", "config.initial_cash must be a number", 400
            )
        if requested <= 0:
            raise ProtocolError(
                "invalid_config",
                "config.initial_cash must be greater than 0",
                400,
            )
        if requested > float(MAX_BACKTEST_INITIAL_CAPITAL) + 1e-9:
            raise ProtocolError(
                "invalid_config",
                f"config.initial_cash cannot exceed {MAX_BACKTEST_INITIAL_CAPITAL:g}",
                400,
                details={"max_initial_cash": MAX_BACKTEST_INITIAL_CAPITAL},
            )
        resolved_capital = resolve_initial_capital(requested)

    mode = config.get("mode", "safe_trading")
    if mode not in ("safe_trading", "buy_and_hold"):
        raise ProtocolError("invalid_config", f"Unsupported mode '{mode}'", 400)

    # Bound concurrent resource use: refuse a new run once the agent already has
    # MAX_ACTIVE_RUNS_PER_AGENT non-terminal runs (each pins an in-memory engine
    # session holding market data). The reaper frees these as they finish.
    # Serialize the cap check with the actual run creation so two concurrent
    # creates from one agent can't both observe an under-limit count and both
    # proceed past the cap (check-then-act TOCTOU).
    agent_id = agent.get("agent_id")
    # Possibly-remote reads (users/content Postgres) stay OFF the shared lock.
    owner_cap_context = resolve_owner_cap_context(agent)
    with _create_lock:
        if agent_id:
            active = run_store.count_active_runs(agent_id)
            if active >= MAX_ACTIVE_RUNS_PER_AGENT:
                raise ProtocolError(
                    "too_many_active_runs",
                    f"Agent already has {active} active runs "
                    f"(limit {MAX_ACTIVE_RUNS_PER_AGENT}); wait for one to finish",
                    429,
                    details={"active_runs": active, "limit": MAX_ACTIVE_RUNS_PER_AGENT},
                )

        violation = check_owner_active_run_cap(owner_cap_context)
        if violation is not None:
            raise ProtocolError(
                "too_many_active_runs_for_account",
                owner_cap_message(violation),
                429,
                details=violation,
            )

        if MAX_ACTIVE_RUNS_GLOBAL > 0:
            total = run_store.count_active_runs_total()
            if total >= MAX_ACTIVE_RUNS_GLOBAL:
                raise ProtocolError(
                    "too_many_active_runs_global",
                    f"The server is at its active-run capacity "
                    f"({total} of {MAX_ACTIVE_RUNS_GLOBAL}); retry shortly",
                    429,
                    details={"active_runs_total": total,
                             "limit": MAX_ACTIVE_RUNS_GLOBAL},
                    headers={"Retry-After": "30"},
                )

        start_res = ebs.start_backtest(
            session_id=agent["session_id"],
            agent_name=agent.get("name") or "external-agent",
            model_name=agent.get("model_name") or "local-model",
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            # The effective allow-list: observation features must cover every
            # symbol constraints() advertises as tradeable (LOW #11).
            symbols=config.get("symbols") or DJIA_30,
            initial_capital=resolved_capital,
        )
        backtest_id = start_res["backtest_id"]

        record = run_store.create_run(
            agent_id=agent.get("agent_id"),
            agent_version_id=agent_version.get("agent_version_id") if agent_version else None,
            session_id=agent["session_id"],
            environment_id=environment_id,
            environment_type=environment.get("type"),
            config=config,
            backtest_id=backtest_id,
            status="running",
        )

        owner_user_id = agent.get("owner_user_id")
        if owner_user_id is not None:
            analytics_instrumentation.emit_run_event(
                event_name="backtest_requested",
                user_id=int(owner_user_id),
                run_id=record["run_id"],
            )
            analytics_instrumentation.emit_run_event(
                event_name="backtest_started",
                user_id=int(owner_user_id),
                run_id=record["run_id"],
            )

        run = ProtocolRun(record=record, environment=environment)
        with _registry_lock:
            _runs[run.run_id] = run

    return run_view(run.run_id)


def run_view(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    status = _sync_status(run)
    session = run.session()
    record = run_store.get_run(run_id) or {}
    view = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run.run_id,
        "agent_id": record.get("agent_id"),
        "agent_version_id": record.get("agent_version_id"),
        "environment": {
            "environment_id": record.get("environment_id"),
            "type": record.get("environment_type"),
        },
        "config": run.config,
        "status": run.status,
        "result_run_id": run.result_run_id,
        "created_at": record.get("created_at"),
    }
    if session is not None:
        engine = status["engine_status"] or {}
        view["progress"] = {
            "step_index": engine.get("step_index"),
            "total_steps": engine.get("total_steps"),
        }
        if engine.get("metrics"):
            view["metrics"] = engine["metrics"]
        if engine.get("compare_url"):
            view["compare_url"] = engine["compare_url"]
    return view


def run_status(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    status = _sync_status(run)
    engine = status["engine_status"] or {}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run.run_id,
        "status": run.status,
        "step_index": engine.get("step_index"),
        "total_steps": engine.get("total_steps"),
        "result_run_id": run.result_run_id,
    }


def get_next_step(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    session = run.session()
    if session is None:
        # Session freed after completion (reaped) or gone after a restart —
        # answer from persisted state instead of erroring on a finished run.
        _sync_status(run)
        if run.status == "completed" or run.result_run_id:
            return {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": run.run_id,
                "status": "completed",
                "result_run_id": run.result_run_id,
                "message": "Run completed; no further steps.",
            }
        if run.status == "failed":
            raise ProtocolError("run_failed", "Run failed", 500)
        raise ProtocolError("run_not_active", "Run session is no longer active", 409)

    step = session.get_current_step()
    estatus = step.get("status")

    if estatus == "loading":
        return {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run.run_id,
            "status": "loading",
            "message": step.get("message", "Loading market data..."),
        }
    if estatus == "failed":
        raise ProtocolError("run_failed", step.get("error") or "Run failed", 500)
    if estatus == "completed":
        _sync_status(run)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run.run_id,
            "status": "completed",
            "result_run_id": run.result_run_id,
            "message": "Run completed; no further steps.",
        }

    seq = step["step_index"]
    timestamp = step["timestamp"]
    deadline = step.get("decision_deadline_at")
    step_id = run.ensure_step_id(seq, timestamp, deadline)
    return _build_step_view(run, session, seq, step_id, step)


def get_step(run_id: str, step_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    seq = run.step_seq.get(step_id)
    if seq is None:
        raise ProtocolError("unknown_step", "Unknown step_id for this run", 404)

    session = run.session()
    if session is not None:
        # Trigger any pending timeout, then classify.
        session.get_status()
        if session.status == "waiting_decision" and session.step_index == seq:
            step = session.get_current_step()
            if step.get("status") == "waiting_decision":
                return _build_step_view(run, session, seq, step_id, step)

    meta = run.step_meta.get(step_id, {})
    status = _historical_step_status(run, seq)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run.run_id,
        "step_id": step_id,
        "sequence": seq,
        "timestamp": _iso(meta.get("timestamp")),
        "deadline_at": meta.get("deadline_at"),
        "status": status,
    }


def submit_decision(run_id: str, step_id: str, decision: DecisionIn) -> Dict[str, Any]:
    run = _get_run(run_id)
    with run.lock:
        # Idempotent replay, scoped to (step_id, key): the same idempotency_key
        # reused on a *different* step must not replay the earlier step's result.
        idem_key = (step_id, decision.idempotency_key)
        if idem_key in run.idempotency:
            return run.idempotency[idem_key]

        session = run.session()
        if session is None:
            raise ProtocolError("run_not_active", "Run session is no longer active", 409)

        session.get_status()  # apply timeout side effects
        seq = run.step_seq.get(step_id)
        if seq is None:
            raise ProtocolError("unknown_step", "Unknown step_id for this run", 404)

        current_index = session.step_index
        if session.status == "completed" or seq < current_index:
            prior = run.step_results_by_seq.get(seq)
            if prior and prior["idempotency_key"] == decision.idempotency_key:
                return prior["result"]
            # A step auto-held because its decision deadline elapsed has no
            # finalized protocol decision (``step_results_by_seq`` unset) but is
            # logged by the engine with ``decision_source == "timeout_hold"``.
            # ``get_status()`` above applies that auto-hold and advances
            # ``step_index`` before we reach here, so the dedicated
            # ``decision_deadline_exceeded`` raise at ``submit_decisions()`` below
            # is effectively unreachable for the common case — surface the
            # documented code here instead, so an agent/SDK can tell a missed
            # deadline from a genuine double-submit (which keeps its
            # ``step_already_finalized`` below, since ``prior`` is set).
            if prior is None and _step_decision_source(run, seq) == "timeout_hold":
                raise ProtocolError(
                    "decision_deadline_exceeded",
                    "Decision arrived after the step deadline; the step was auto-held",
                    409,
                )
            raise ProtocolError(
                "step_already_finalized",
                "This step already has a finalized decision",
                409,
            )
        if seq > current_index:
            raise ProtocolError("step_not_active", "Step is not the active step", 409)

        # seq == current active step
        timestamp = session.timestamps[current_index]
        prices = session.protocol_current_prices(timestamp)
        portfolio_before = session.protocol_portfolio(timestamp)
        equity_before = portfolio_before["equity"]
        cash_before = float(session.manager.cash)
        positions_before = dict(session.manager.positions)

        # Enforce the constraints the environment advertises (previously
        # returned to the agent but never checked). ``allowed_symbols`` is a
        # concrete allow-list; ``max_orders`` is a decision-level cap;
        # ``max_position_weight`` is enforced per order below.
        constraints = run.constraints()
        allowed_symbols = set(constraints.get("allowed_symbols") or DJIA_30)
        max_orders = _coerce_nonneg_int(constraints.get("max_orders"))
        max_position_weight = _coerce_positive_float(constraints.get("max_position_weight"))

        # A decision that exceeds ``max_orders`` violates the advertised
        # contract; reject the whole decision rather than silently truncating.
        # This raises before any order is finalized, so the step stays open for
        # a corrected resubmission (new idempotency_key).
        if max_orders is not None and len(decision.orders) > max_orders:
            raise ProtocolError(
                "too_many_orders",
                f"Decision has {len(decision.orders)} orders; max_orders is {max_orders}",
                400,
                details={"max_orders": max_orders, "submitted": len(decision.orders)},
            )

        accepted_actions: List[Dict[str, Any]] = []
        # Parallel to accepted_actions: the protocol-shaped order each action
        # came from, so post-engine rejections can echo the order as submitted
        # instead of leaking the internal engine action dict.
        accepted_order_reprs: List[Dict[str, Any]] = []
        rejections: List[Dict[str, Any]] = []
        confidence = decision.confidence if decision.confidence is not None else 0.75
        # Buy shares provisionally accepted earlier in THIS decision, per symbol,
        # so the position cap accounts for intra-decision accumulation (several
        # buys of the same symbol) rather than judging each order in isolation.
        pending_buy_shares: Dict[str, int] = {}

        for order in decision.orders:
            side = order.side.lower()
            order_repr = order.model_dump()
            if side not in VALID_SIDES:
                rejections.append({"order": order_repr, "reason": "invalid_side"})
                continue
            if order.symbol not in allowed_symbols:
                rejections.append({"order": order_repr, "reason": "invalid_symbol"})
                continue
            price = prices.get(order.symbol, 0)
            shares, qerr = resolve_order_quantity(
                order, price=price, equity=equity_before
            )
            if qerr:
                rejections.append({"order": order_repr, "reason": qerr})
                continue
            if shares <= 0:
                rejections.append({"order": order_repr, "reason": "zero_quantity"})
                continue
            # Reject an over-cap order on its own (H2/H3): pre-filtering here
            # keeps a single oversized order from voiding the whole decision —
            # the remaining valid orders still reach the engine and execute.
            # ``existing_shares`` folds in what's already held AND what earlier
            # orders in this same decision provisionally bought.
            held = (positions_before.get(order.symbol, 0) or 0) + pending_buy_shares.get(order.symbol, 0)
            cap_reason = _exceeds_position_cap(
                side,
                order.symbol,
                shares,
                price,
                equity=equity_before,
                existing_shares=held,
                max_position_weight=max_position_weight,
            )
            if cap_reason:
                rejections.append({"order": order_repr, "reason": cap_reason})
                continue
            # A single order above the engine's hard per-order share ceiling
            # would fail its all-or-nothing batch validator and void the WHOLE
            # decision. Reject it per-order here so valid siblings still execute.
            if shares > MAX_ORDER_SHARES:
                rejections.append({"order": order_repr, "reason": "exceeds_max_order_size"})
                continue
            if side == "buy":
                pending_buy_shares[order.symbol] = pending_buy_shares.get(order.symbol, 0) + shares
            accepted_actions.append(
                order_to_action(
                    order, shares=shares, confidence=confidence, rationale=decision.rationale
                )
            )
            accepted_order_reprs.append(order_repr)

        trades_before = session.trade_count()
        engine_result = session.submit_decisions({"actions": accepted_actions})

        if not engine_result.get("accepted"):
            err = engine_result.get("error", "")
            if err == "step_already_closed":
                raise ProtocolError(
                    "decision_deadline_exceeded",
                    "Decision arrived after the step deadline; step auto-held",
                    409,
                    details={"outcome": engine_result.get("outcome")},
                )
            if err == "backtest_already_completed":
                raise ProtocolError("run_completed", "Run already completed", 409)
            # validation_hold / invalid_status: treat all submitted orders as rejected
            for order_repr in accepted_order_reprs:
                rejections.append({"order": order_repr, "reason": err or "validation_failed"})
            accepted_actions = []
            accepted_order_reprs = []
            executed = []
        else:
            executed = engine_result.get("executed", []) or []

        # Reconcile: any accepted action the engine did not execute is rejected.
        executed_keys = {(e.get("symbol"), e.get("action")) for e in executed}
        for action, order_repr in zip(accepted_actions, accepted_order_reprs):
            key = (action["symbol"], action["action"])
            if key not in executed_keys:
                rejections.append(
                    {"order": order_repr, "reason": _infer_rejection(action, cash_before, prices, positions_before)}
                )

        fills = session.fills_since(trades_before)
        exec_ts = session.executed_step_timestamp()
        portfolio_after = session.protocol_portfolio(exec_ts)
        decision_id = _new_decision_id()
        run_completed = session.status == "completed"

        result = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run.run_id,
            "step_id": step_id,
            "decision_id": decision_id,
            "accepted": bool(engine_result.get("accepted")),
            "validation": {
                "passed": len(rejections) == 0,
                "warnings": [],
                "rejections": rejections,
            },
            "fills": fills,
            "portfolio_after": portfolio_after,
            "run_status": "completed" if run_completed else "running",
        }

        # Record finalization + idempotency (scoped to this step_id), and
        # persist so the replay survives a restart.
        run.idempotency[idem_key] = result
        run.step_results_by_seq[seq] = {
            "idempotency_key": decision.idempotency_key,
            "result": result,
        }
        if step_id in run.step_meta:
            run.step_meta[step_id]["status"] = "completed"
        run_store.finalize_step(
            run.run_id, step_id, decision.idempotency_key, result
        )

        if run_completed:
            _sync_status(run)

        return result


# ----------------------------------------------------------------------
# Result / log accessors
# ----------------------------------------------------------------------


def list_steps(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    _sync_status(run)
    decisions = _decisions_raw(run)
    steps = []
    for d in decisions:
        seq = d.get("step_index", 0)
        source = d.get("decision_source")
        status = "timed_out" if source == "timeout_hold" else "completed"
        steps.append({
            "sequence": seq,
            "step_id": run.seq_to_step_id.get(seq),
            "timestamp": d.get("timestamp"),
            "status": status,
            "decision_source": source,
            "actions_executed": d.get("actions_executed", 0),
        })
    if not steps and run.step_meta:
        # No decision log (post-restart, run not finalized): answer from the
        # rehydrated protocol_steps bookkeeping instead of reporting zero
        # steps for a run that really processed some. Only accepted decisions
        # finalize a step, so a completed row's source is external_agent by
        # construction; its executed count comes from the stored result.
        for seq in sorted(run.seq_to_step_id):
            sid = run.seq_to_step_id[seq]
            meta = run.step_meta.get(sid, {})
            prior = run.step_results_by_seq.get(seq)
            completed = meta.get("status") == "completed"
            steps.append({
                "sequence": seq,
                "step_id": sid,
                "timestamp": meta.get("timestamp"),
                "status": "completed" if completed else (meta.get("status") or "awaiting_decision"),
                "decision_source": "external_agent" if completed else None,
                "actions_executed": len((prior or {}).get("result", {}).get("fills") or []),
            })
    return {"protocol_version": PROTOCOL_VERSION, "run_id": run_id, "steps": steps, "count": len(steps)}


def list_decisions(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    _sync_status(run)
    decisions = _decisions_raw(run)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "decisions": decisions,
        "count": len(decisions),
    }


def list_trades(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    _sync_status(run)
    if run.result_run_id:
        trades = db.get_trades(run.result_run_id)
    else:
        session = run.session()
        trades = _normalize_live_trades(session.manager.trades) if session else []
    return {"protocol_version": PROTOCOL_VERSION, "run_id": run_id, "trades": trades, "count": len(trades)}


def get_metrics(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    status = _sync_status(run)
    metrics: Dict[str, Any] = {}
    if run.result_run_id:
        dbrun = db.get_run(run.result_run_id)
        if dbrun:
            metrics = {
                "total_return": dbrun.get("total_return"),
                "sharpe_ratio": dbrun.get("sharpe_ratio"),
                "max_drawdown": dbrun.get("max_drawdown"),
                "num_trades": dbrun.get("num_trades"),
                "final_equity": dbrun.get("final_equity"),
                "llm_calls": dbrun.get("llm_calls"),
                "input_tokens": dbrun.get("input_tokens"),
                "output_tokens": dbrun.get("output_tokens"),
                "est_cost_usd": dbrun.get("est_cost_usd"),
            }
    else:
        engine = status["engine_status"] or {}
        metrics = engine.get("metrics") or {}
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "status": run.status,
        "metrics": metrics,
    }


def get_result(run_id: str) -> Dict[str, Any]:
    run = _get_run(run_id)
    _sync_status(run)
    if not run.result_run_id:
        raise ProtocolError(
            "run_not_completed",
            "Run has not completed; results are not yet available",
            409,
            details={"status": run.status},
        )
    result = ebs.get_run_result(run.result_run_id, run.session_id)
    if result is None:
        raise ProtocolError("result_not_found", "Result not found for this run", 404)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "result_run_id": run.result_run_id,
        "status": run.status,
        **result,
    }


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _build_step_view(run, session, seq, step_id, step) -> Dict[str, Any]:
    snapshot = step.get("market_snapshot", {})
    timestamp = session.timestamps[seq]
    market_data = session.protocol_market_data(timestamp)
    portfolio = session.protocol_portfolio(timestamp, market_data=market_data)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run.run_id,
        "step_id": step_id,
        "sequence": seq,
        "timestamp": step.get("timestamp"),
        "deadline_at": step.get("decision_deadline_at"),
        "status": "awaiting_decision",
        "observation": {
            "market": {
                "bars": session.protocol_bars(timestamp, market_data=market_data),
                "features": snapshot.get("top_signals", {}),
                "events": [],
            },
            "portfolio": portfolio,
        },
        "constraints": run.constraints(),
    }


def _step_decision_source(run, seq) -> Optional[str]:
    """The engine's recorded ``decision_source`` for step ``seq`` (e.g.
    ``'timeout_hold'`` for a deadline auto-hold), or ``None`` if the step has no
    logged decision yet. Single source of truth for both historical-status
    reporting and the late-decision code in ``submit_decision``.
    """
    for d in _decisions_raw(run):
        if d.get("step_index") == seq:
            return d.get("decision_source")
    return None


def _historical_step_status(run, seq) -> str:
    if run.step_results_by_seq.get(seq):
        return "completed"
    # Fall back to the persisted/engine decision log.
    source = _step_decision_source(run, seq)
    if source == "timeout_hold":
        return "timed_out"
    if source is not None:
        return "completed"
    # No decision log at all (post-restart, run not finalized): answer from the
    # persisted step row rather than fabricating "completed" — a step that
    # auto-held or was mid-await when the process died never completed.
    sid = run.seq_to_step_id.get(seq)
    meta = run.step_meta.get(sid) if sid else None
    if meta and meta.get("status") != "completed":
        return meta.get("status") or "awaiting_decision"
    return "completed"


def _decisions_raw(run) -> List[Dict[str, Any]]:
    if run.result_run_id:
        return db.get_decisions(run.result_run_id)
    session = run.session()
    if session is None:
        return []
    return session.get_decisions()


def _normalize_live_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in trades:
        ts = t.get("timestamp")
        if hasattr(ts, "isoformat"):
            ts = ts.isoformat()
        qty = int(t.get("shares") or 0)
        price = float(t.get("price") or 0)
        out.append({
            "timestamp": ts,
            "symbol": t.get("symbol"),
            "quantity": qty,
            "side": str(t.get("side", "")).upper(),
            "price": price,
            "value": float(t.get("cost") or t.get("proceeds") or qty * price),
            "reason": t.get("reason"),
        })
    return out


def _coerce_positive_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite positive float, or None if it isn't one.

    Constraint values come from a data-driven registry (and one day external
    config), so tolerate strings/None/garbage instead of crashing on a bad type.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def _coerce_nonneg_int(value: Any) -> Optional[int]:
    """Return ``value`` as a non-negative int, or None if it can't be one."""
    try:
        i = int(value)
    except (TypeError, ValueError):
        return None
    return i if i >= 0 else None


def _exceeds_position_cap(
    side: str,
    symbol: str,
    shares: int,
    price: float,
    *,
    equity: float,
    existing_shares: int,
    max_position_weight: Optional[float],
) -> Optional[str]:
    """Return a rejection reason if a BUY would push the position past the cap.

    ``max_position_weight`` is a fraction of total equity that any single
    position may occupy (already coerced to a finite positive float or None).
    Sells reduce exposure and are never capped here. The resulting position is
    valued at the current price and includes ``existing_shares`` (held plus
    already-accepted-this-decision).
    """
    if side != "buy":
        return None
    if not max_position_weight:
        return None
    if equity <= 0 or price <= 0:
        return None
    resulting_notional = (existing_shares + shares) * price
    if resulting_notional > max_position_weight * equity:
        return "exceeds_max_position_weight"
    return None


def _infer_rejection(action, cash_before, prices, positions_before) -> str:
    side = action["action"]
    symbol = action["symbol"]
    if action.get("confidence", 1.0) < 0.3:
        return "below_min_confidence"
    if side == "sell":
        if positions_before.get(symbol, 0) <= 0:
            return "no_position"
        return "not_executed"
    if side == "buy":
        price = prices.get(symbol, 0)
        if price <= 0:
            return "missing_price"
        if action["position_size"] * price > cash_before:
            return "insufficient_cash"
    return "not_executed"


def _iso(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
