"""v2 runs: create · status · context · decisions · result · decisions-log · cancel.

One canonical run_id (minted here) drives the whole lifecycle (spec §4.3). Runs live
in process memory (single-worker assumption, spec §12).
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from dashboard.backend.domain.backtesting import external_run_service as ext
from dashboard.backend.api.v2.errors import ApiError
from dashboard.backend.api.v2.models import (
    DecisionRequest, RunManifest, SCHEMA_VERSION, UNIVERSE_KEY, validate_actions,
)
from dashboard.backend.api.v2.auth_scopes import require_scope
from dashboard.backend.database import db
# The v1 service's create lock is shared on purpose: both surfaces count
# active runs from the same protocol_runs ledger, so both check-then-insert
# sequences must serialize on ONE lock or an agent could race one create per
# surface past the combined cap. Known trade-off: run creation across BOTH
# surfaces serializes on this lock, including its quick SQLite writes —
# acceptable because creates are rare and bounded (cap per agent), and
# correctness of the cap beats create throughput. Long work (market-data
# load) stays off the lock (background thread).
from dashboard.backend.domain.runs.service import (
    MAX_ACTIVE_RUNS_GLOBAL,
    MAX_ACTIVE_RUNS_PER_AGENT,
    _create_lock as _shared_create_lock,
    check_owner_active_run_cap,
    owner_cap_message,
    resolve_owner_cap_context,
    _analytics_owner_user_id,
)
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation
# Late-bound module reference (run_repo.run_store) so tests that swap the
# run_store singleton cover this module too.
from dashboard.backend.domain.runs import repository as run_repo
from dashboard.backend.execution.backtest_backend import (
    ArchivedBacktestBackend,
    BacktestBackend,
)
from dashboard.backend.execution.base import TERMINAL_STATUSES as _TERMINAL_STATUSES
from dashboard.backend.api.v2.rate_limit import enforce

router = APIRouter(prefix="/v2/runs", tags=["v2-runs"])

# run_id -> {"backend": ExecutionBackend, "session_id": str, "agent_id": str|None}
_runs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

# MAX_ACTIVE_RUNS_PER_AGENT (imported from the v1 run service — one knob for
# both surfaces): each active run pins an in-memory engine session holding
# market data. The cap is enforced across BOTH surfaces: every v2 run writes a
# protocol_runs row through its lifecycle, and both create paths count
# run_store.count_active_runs() under the shared v1 create lock.
#
# Documented consequence of the shared ledger: an agent holding both surfaces'
# credentials can see its v2 runs listed by /api/v1/runs (status-level record
# only; v1 step/decision queries have nothing to serve for them) and read its
# own v1 runs' terminal rows through v2 rehydration. This is intentional —
# run ids are canonical and both surfaces are the same agent identity; v2 is
# the canonical surface, v1 the compat view.


def _mint_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:8]}"


def register_run(run_id: str, backend: Any, session_id: str,
                 agent_id: Optional[str] = None) -> None:
    """Register a backend under a run_id (used by create + tests)."""
    with _lock:
        _runs[run_id] = {"backend": backend, "session_id": session_id,
                         "agent_id": agent_id}


def _terminal_status(backend: Any) -> str:
    """Resolve a no-longer-active backend's terminal state.

    Prefers the raw session attribute (a plain read, no locks). status() is
    only consulted when there is no session — and only ever on an INACTIVE
    backend, where it cannot cascade into deadline handling or _finalize().
    """
    status = getattr(getattr(backend, "session", None), "status", None)
    if status is None:
        try:
            status = (backend.status() or {}).get("status")
        except Exception:
            status = None
    return status if status in ("completed", "closed") else "failed"


def _archive_run(run_id: str, entry: Dict[str, Any], backend: Any) -> None:
    """Fold a finished backend into its protocol_runs row and swap in a
    DB-backed tombstone (frees the engine session's market-data buffers)."""
    status = _terminal_status(backend)
    session = getattr(backend, "session", None)
    result_run_id = getattr(session, "run_id", None) or run_id

    def _progress(attr: str) -> Optional[int]:
        # Prefer the live session's count, else the backend's own (a tombstone
        # carries them directly). `is None`, not `or`, so a real 0 is kept.
        val = getattr(session, attr, None) if session is not None else None
        return val if val is not None else getattr(backend, attr, None)

    step_index = _progress("step_index")
    total_steps = _progress("total_steps")
    try:
        run_repo.run_store.update_run(
            run_id,
            status=status,
            result_run_id=result_run_id if status == "completed" else None,
            # Persist step counts so a post-restart from_record reports real
            # progress for a failed/closed run instead of a misleading 0/0.
            step_index=step_index,
            total_steps=total_steps,
        )
    except Exception as exc:
        # Do NOT swap in the tombstone: with the backend gone nothing would
        # ever retry this write and the row would freeze in an active status
        # (mis-holding a cap slot until stale recovery mislabels it failed).
        # Keep the backend live so the next sweep retries.
        print(f"⚠️ v2 archive: row update failed for {run_id}, retrying next sweep: {exc}")
        return
    owner_user_id = _analytics_owner_user_id(entry.get("agent_id"))
    if owner_user_id is not None:
        event_name = {
            "completed": "backtest_completed",
            "closed": "backtest_cancelled",
        }.get(status, "backtest_failed")
        analytics_instrumentation.emit_run_event(
            event_name=event_name,
            user_id=owner_user_id,
            run_id=run_id,
            error_category=("internal_error" if event_name == "backtest_failed" else None),
        )
    archived = ArchivedBacktestBackend(
        run_id=run_id,
        session_id=entry["session_id"],
        status=status,
        error=getattr(session, "error", None),
        step_index=step_index,
        total_steps=total_steps,
        result_run_id=result_run_id,
    )
    with _lock:
        live = _runs.get(run_id)
        # An in-flight request may still hold the old backend object; it stays
        # alive via that reference and its session is thread-safe, so swapping
        # under a racing read is benign.
        if live is not None and live["backend"] is backend:
            live["backend"] = archived


def _reconcile_terminal_backends(agent_id: Optional[str] = None) -> None:
    """Fold any backend that finished since the last sweep into its row.

    Runs under the shared create lock, so it must stay passive on LIVE runs:
    only the is_active() peek is consulted; status()/advance() are never
    called on an active backend here."""
    with _lock:
        items = [
            (rid, e) for rid, e in _runs.items()
            if agent_id is None or e.get("agent_id") == agent_id
        ]
    for run_id, entry in items:
        backend = entry["backend"]
        if isinstance(backend, ArchivedBacktestBackend):
            continue
        try:
            if backend.is_active():
                continue
        except Exception:
            continue  # unknown state keeps its row (and cap slot) — fail closed
        _archive_run(run_id, entry, backend)


def _active_run_count(agent_id: str) -> int:
    """Active runs across BOTH surfaces — the protocol_runs ledger is shared
    with /api/v1. Reconcile first so a v2 run that finished since the last
    reaper sweep does not hold a phantom cap slot."""
    _reconcile_terminal_backends(agent_id)
    return run_repo.run_store.count_active_runs(agent_id)


def _rehydrate_terminal_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Rebuild a registry entry from a terminal protocol_runs row.

    After a restart the in-memory registry is empty but the run's terminal
    state (startup recovery marks orphans failed), result linkage, decisions
    and idempotency acks are all DB-backed — owners keep read/replay access
    instead of a 404. Non-terminal rows are NOT rehydrated: with no live
    backend here they belong to another worker or are awaiting recovery."""
    try:
        record = run_repo.run_store.get_run(run_id)
    except Exception:
        return None
    if not record or record.get("status") not in _TERMINAL_STATUSES:
        return None
    entry = {
        "backend": ArchivedBacktestBackend.from_record(record),
        "session_id": record["session_id"],
        "agent_id": record.get("agent_id"),
    }
    with _lock:
        existing = _runs.get(run_id)
        if existing is not None:
            return existing
        _runs[run_id] = entry
    return entry


def _require_run(run_id: str, session_id: str) -> Any:
    with _lock:
        entry = _runs.get(run_id)
    if entry is None:
        entry = _rehydrate_terminal_run(run_id)
    # A run owned by another session answers exactly like a missing one — a
    # message-text difference would let any key holder enumerate run ids.
    if not entry or entry["session_id"] != session_id:
        raise ApiError("run_not_found", f"Run {run_id} not found", status=404)
    return entry["backend"]


def reap_v2_runs() -> int:
    """Per-pass v2 sweep, registered with the v1 reaper (composition root).

    Drives abandoned runs through elapsed decision deadlines (advance()),
    heartbeats rows whose backend is live in this process, and archives
    terminal backends (row update + tombstone swap). Returns the number of
    backends archived this pass."""
    with _lock:
        items = list(_runs.items())
    live_ids = []
    prunable = []  # tombstones archived in a PRIOR pass — safe to evict now
    reaped = 0
    for run_id, entry in items:
        backend = entry["backend"]
        if isinstance(backend, ArchivedBacktestBackend):
            prunable.append((run_id, backend))
            continue
        try:
            backend.advance()  # applies any pending deadline auto-hold
        except Exception as exc:  # a wedged run must not stall the sweep
            print(f"⚠️ v2 reap: drain failed for {run_id}: {exc}")
        try:
            active = backend.is_active()
        except Exception:
            active = True  # unknown state: keep it registered, try next pass
        if active:
            live_ids.append(run_id)
        else:
            _archive_run(run_id, entry, backend)
            reaped += 1
    # Evict prior-pass tombstones so _runs is bounded by ACTIVE runs, not by
    # total historical runs. The terminal row is DB-backed, so a later read
    # rebuilds an equivalent tombstone via _rehydrate_terminal_run. Runs
    # archived THIS pass stay one interval, so an immediately-following read is
    # still served in-process (and single-pass tests see the tombstone).
    if prunable:
        with _lock:
            for run_id, backend in prunable:
                cur = _runs.get(run_id)
                if cur is not None and cur["backend"] is backend:
                    _runs.pop(run_id, None)
    if live_ids:
        try:
            run_repo.run_store.heartbeat_runs(live_ids)
        except Exception as exc:
            print(f"⚠️ v2 reap: heartbeat pass failed: {exc}")
    return reaped


# -- pure helpers (unit-testable without HTTP) -----------------------------

def _context_for(run_id: str, session_id: str) -> Dict[str, Any]:
    backend = _require_run(run_id, session_id)
    return backend.build_context()


# Rejections that CONSUMED the step (the engine auto-held and advanced). These
# must replay under their idempotency key: a same-key retry that missed the
# cache would pass the step re-check and execute the stale actions against the
# NEXT step's prices. Non-consuming errors (invalid_status) change nothing
# server-side, so the same key stays retryable once the run is ready.
_STEP_CONSUMING_REJECTIONS = {"step_already_closed", "validation_failed"}


def _replay_rejection(marker: Dict[str, Any]) -> None:
    raise ApiError(
        marker["code"], marker["message"], status=marker["status"],
        details=marker.get("details"), retryable=marker.get("retryable", False),
    )


def _submit_for(run_id: str, session_id: str, idem_key: str,
                raw_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    backend = _require_run(run_id, session_id)
    step = backend.current_step_index()
    existing = db.get_idempotency(run_id, step, idem_key)
    if existing is not None:
        if isinstance(existing, dict) and "__rejection__" in existing:
            _replay_rejection(existing["__rejection__"])
        return existing
    # Partial execution (spec §5.3): drop schema-invalid actions with reasons,
    # execute the rest. If all are invalid, the step auto-holds (validation_hold).
    valid, rejected = validate_actions(raw_actions)
    try:
        ack = backend.apply_decisions(valid)
    except ApiError as exc:
        if exc.code in _STEP_CONSUMING_REJECTIONS:
            db.put_idempotency(run_id, step, idem_key, {"__rejection__": {
                "code": exc.code, "message": exc.message, "status": exc.status,
                "retryable": exc.retryable, "details": exc.details,
            }})
        raise
    if rejected:
        ack["rejected"] = list(ack.get("rejected") or []) + rejected
        if not valid:
            ack["decision_source"] = "validation_hold"
    db.put_idempotency(run_id, step, idem_key, ack)
    return ack


def _result_for(run_id: str, session_id: str) -> Dict[str, Any]:
    backend = _require_run(run_id, session_id)
    res = backend.result()
    if res is None:
        raise ApiError("invalid_status", "Run not finished", status=409, retryable=True)
    return res


# -- request body ----------------------------------------------------------

class CreateRunBody(BaseModel):
    mode: str = Field(default="backtest", pattern="^backtest$")
    universe: str = Field(default=UNIVERSE_KEY, pattern="^djia_30$")
    start_date: str
    end_date: str
    agent_name: str = Field(default="external-agent", min_length=1, max_length=100)
    model_name: str = Field(default="local-model", min_length=1, max_length=100)
    strategy_mode: str = Field(default="safe_trading", pattern="^(safe_trading|buy_and_hold)$")


# -- endpoints -------------------------------------------------------------

@router.post("")
def create_run(body: CreateRunBody, response: Response,
               agent: dict = Depends(require_scope("runs:write"))):
    """Mint the canonical run_id, write the manifest, start the backtest load.

    ``def`` (threadpool), like every v1 protocol handler: creation touches the
    DB and the engine session synchronously (B0/H4 convention).
    """
    enforce(agent["agent_id"], response)
    # Possibly-remote reads (users/content Postgres) stay OFF the shared lock.
    owner_cap_context = resolve_owner_cap_context(agent)
    with _shared_create_lock:
        active = _active_run_count(agent["agent_id"])
        if active >= MAX_ACTIVE_RUNS_PER_AGENT:
            raise ApiError(
                "too_many_active_runs",
                f"Agent already has {active} active runs "
                f"(limit {MAX_ACTIVE_RUNS_PER_AGENT}); wait for one to finish "
                "or cancel one",
                status=429, retryable=True,
                details={"active_runs": active, "limit": MAX_ACTIVE_RUNS_PER_AGENT},
            )
        violation = check_owner_active_run_cap(owner_cap_context)
        if violation is not None:
            raise ApiError(
                "too_many_active_runs_for_account",
                owner_cap_message(violation),
                status=429, retryable=True,
                details=violation,
            )
        if MAX_ACTIVE_RUNS_GLOBAL > 0:
            total = run_repo.run_store.count_active_runs_total()
            if total >= MAX_ACTIVE_RUNS_GLOBAL:
                raise ApiError(
                    "too_many_active_runs_global",
                    f"The server is at its active-run capacity "
                    f"({total} of {MAX_ACTIVE_RUNS_GLOBAL}); retry shortly",
                    status=429, retryable=True,
                    details={"active_runs_total": total,
                             "limit": MAX_ACTIVE_RUNS_GLOBAL},
                    # Explicit: ApiError's auto Retry-After injection only
                    # fires for code "rate_limited".
                    headers={"Retry-After": "30"},
                )
        run_id = _mint_run_id()
        backend = BacktestBackend(
            run_id=run_id, session_id=agent["session_id"], agent_name=body.agent_name,
            model_name=body.model_name, start_date=body.start_date, end_date=body.end_date,
            mode=body.strategy_mode,
        )
        manifest = RunManifest(
            agent_name=body.agent_name, model_name=body.model_name, mode="backtest",
            universe=UNIVERSE_KEY, start_date=body.start_date, end_date=body.end_date,
            decision_timeout_seconds=ext.DECISION_TIMEOUT_SECONDS,
            schema_version=SCHEMA_VERSION, news_sentiment_source=backend.news_sentiment_source,
        )
        db.insert_run_manifest(run_id, manifest.model_dump())
        # The run's protocol_runs row: the ledger shared with /api/v1 that the
        # cap counts, startup recovery fails, and post-restart reads rehydrate.
        # Inserted under the create lock so a concurrent create sees it.
        run_repo.run_store.create_run(
            run_id=run_id,
            agent_id=agent["agent_id"],
            agent_version_id=None,
            session_id=agent["session_id"],
            environment_id=None,
            environment_type="backtest",
            config={
                "start_date": body.start_date, "end_date": body.end_date,
                "mode": body.strategy_mode, "universe": body.universe,
            },
            backtest_id=None,
            status="loading",
        )
        owner_user_id = agent.get("owner_user_id")
        if owner_user_id is not None:
            analytics_instrumentation.emit_run_event(
                event_name="backtest_requested",
                user_id=int(owner_user_id),
                run_id=run_id,
            )
        try:
            backend.start_background_load()
            register_run(run_id, backend, agent["session_id"], agent["agent_id"])
            if owner_user_id is not None:
                analytics_instrumentation.emit_run_event(
                    event_name="backtest_started",
                    user_id=int(owner_user_id),
                    run_id=run_id,
                )
        except Exception:
            # Never leak an active-looking row for a run that never started.
            try:
                run_repo.run_store.update_run(run_id, status="failed")
                if owner_user_id is not None:
                    analytics_instrumentation.emit_run_event(
                        event_name="backtest_failed",
                        user_id=int(owner_user_id),
                        run_id=run_id,
                        error_category="internal_error",
                    )
            except Exception:
                pass
            raise
    return {
        "run_id": run_id, "mode": "backtest", "status": "loading",
        "loop": backend.loop, "decision_timeout_seconds": ext.DECISION_TIMEOUT_SECONDS,
    }


@router.get("/{run_id}")
def run_status(run_id: str, agent: dict = Depends(require_scope("runs:read"))):
    backend = _require_run(run_id, agent["session_id"])
    return backend.status()


@router.get("/{run_id}/context")
def get_context(run_id: str, response: Response,
                agent: dict = Depends(require_scope("context:read"))):
    """get_context — typed context envelope for the current step."""
    enforce(agent["agent_id"], response)
    return _context_for(run_id, agent["session_id"])


@router.post("/{run_id}/decisions")
def submit_decision(run_id: str, body: DecisionRequest, response: Response,
                    agent: dict = Depends(require_scope("decisions:write"))):
    """submit_decision — idempotent per (run_id, idempotency_key); a replay
    returns the original ack even after the run has advanced to a later step.

    ``def`` (threadpool): the final step's submit runs ``_finalize()`` — two
    baseline backtests — which must never block the event loop (B0/H4).
    """
    enforce(agent["agent_id"], response)
    return _submit_for(run_id, agent["session_id"], body.idempotency_key, body.actions)


@router.get("/{run_id}/result")
def get_result(run_id: str, agent: dict = Depends(require_scope("runs:read"))):
    """get_result — metrics, equity, trades, decisions, manifest."""
    return _result_for(run_id, agent["session_id"])


@router.get("/{run_id}/decisions")
def decisions_log(run_id: str, agent: dict = Depends(require_scope("runs:read"))):
    backend = _require_run(run_id, agent["session_id"])
    return {"run_id": run_id, "decisions": backend.decisions()}


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str, agent: dict = Depends(require_scope("runs:write"))):
    """Cancel a live run; on an already-terminal run this is a no-op that
    reports the run's TRUE state. Persisting a hardcoded "closed" here used
    to clobber a completed run's ledger row (and hide its metrics) when a
    cleanup cancel raced the final decision."""
    backend = _require_run(run_id, agent["session_id"])
    try:
        active = backend.is_active()
    except Exception:
        active = False
    if active:
        backend.cancel()  # backend-level guard never clobbers terminal state
    status = _terminal_status(backend)
    if active:
        # active is read from backend.is_active(), which an ArchivedBacktestBackend
        # (tombstone) always reports False — so active already implies a live
        # backend. Persist only a transition this request actually caused: if the
        # run finalized between the peek and cancel(), _terminal_status already
        # reads "completed" and that is what lands — never a downgrade.
        try:
            run_repo.run_store.update_run(run_id, status=status)
            owner_user_id = agent.get("owner_user_id")
            if owner_user_id is not None:
                analytics_instrumentation.emit_run_event(
                    event_name=(
                        "backtest_completed"
                        if status == "completed"
                        else "backtest_cancelled"
                        if status == "closed"
                        else "backtest_failed"
                    ),
                    user_id=int(owner_user_id),
                    run_id=run_id,
                    error_category=(
                        "internal_error" if status not in {"completed", "closed"} else None
                    ),
                )
        except Exception:
            pass  # best-effort; the sweep reconciles
    return {"run_id": run_id, "status": status}
