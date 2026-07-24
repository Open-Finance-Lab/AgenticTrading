"""BacktestBackend — wraps ExternalBacktestSession behind the ExecutionBackend seam.

Schema parity: build_context/apply_decisions/result emit the typed v2 shapes.
Lifecycle parity: loop == "lockstep" (decision_deadline_at + auto-hold apply).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from dashboard.backend.api.v2.errors import ApiError
from dashboard.backend.api.v2.models import SCHEMA_VERSION
from dashboard.backend.database import db
from dashboard.backend.execution.base import ExecutionBackend, TERMINAL_STATUSES
from dashboard.backend.infrastructure.llm.validator import DJIA_30
from dashboard.backend.domain.backtesting import external_run_service as ext
# Late-bound module reference (run_repo.run_store) so tests that swap the
# run_store singleton cover this module too.
from dashboard.backend.domain.runs import repository as run_repo

logger = logging.getLogger(__name__)


def load_news_sentiment(universe: List[str], timestamp: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Populate the news_sentiment slot from the news_sentiment adapter, fail-closed.

    dashboard/backend/integrations/news_sentiment.py exposes
    get_news_sentiment(universe, timestamp) -> {"news_sentiment": {...}, "news_overview": str|None}.
    A backtest must not die because news is unavailable, so both failure modes
    below degrade to an empty slot — but they LOG, because fail-closed and
    fail-silent are different things: without a log, a producer contract break
    is indistinguishable from a quiet news day.

    Exception level rather than warning is deliberate: the adapter handles its
    own expected misses quietly (404, network, bad key) AND reports its own
    wire-shape drift (see get_news_sentiment), so anything still escaping it is
    genuinely unexpected. Don't reach back across the boundary to interpret what
    escaped — an except here that explains the adapter's internals goes stale the
    moment they move.
    """
    try:
        from dashboard.backend.integrations.news_sentiment import get_news_sentiment  # type: ignore
    except Exception:
        logger.exception(
            "load_news_sentiment: news_sentiment adapter is unimportable; "
            "sentiment slot left empty for this step")
        return {}, None
    try:
        data = get_news_sentiment(universe, timestamp) or {}
        return data.get("news_sentiment", {}) or {}, data.get("news_overview")
    except Exception as exc:
        logger.exception(
            "load_news_sentiment: adapter raised %r; sentiment slot left empty "
            "for this step", exc)
        return {}, None


def _context_hash(envelope: Dict[str, Any]) -> str:
    payload = json.dumps(envelope, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BacktestBackend(ExecutionBackend):
    loop = "lockstep"

    def __init__(self, *, run_id: str, session_id: str, agent_name: str,
                 model_name: str, start_date: str, end_date: str,
                 mode: str = "safe_trading"):
        self.run_id = run_id
        self.session = ext.ExternalBacktestSession(
            backtest_id=run_id, session_id=session_id, agent_name=agent_name,
            model_name=model_name, start_date=start_date, end_date=end_date,
            mode=mode, run_id=run_id,
        )
        self.news_sentiment_source: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------

    def load_blocking(self) -> None:
        self.session.load_market_data()

    def start_background_load(self) -> None:
        # Fast path (runs under the shared create lock — peek only, never
        # get_dataset): a resident dataset skips the loader thread entirely.
        dataset = ext.market_data_store.peek(
            DJIA_30, self.session.start_date, self.session.end_date)
        if dataset is not None:
            self.session.adopt_dataset(dataset)
            # Mirror the background loader's post-load row transition, with the
            # same don't-resurrect-a-terminal-run guard.
            with self.session._step_lock:
                status_now = self.session.status
            if status_now not in TERMINAL_STATUSES:
                try:
                    run_repo.run_store.update_run(self.run_id, status="running")
                except Exception:
                    pass  # best-effort; both statuses count as active anyway
            return

        def _load() -> None:
            try:
                self.session.load_market_data()
            except (Exception, SystemExit) as exc:  # mirror v1 start_backtest behavior
                # Missing creds now raise MarketDataUnavailableError (plain
                # Exception, B0 deep fix); the SystemExit catch stays as
                # defense-in-depth — a daemon thread swallows an uncaught
                # SystemExit silently and the run would sit in "loading" forever.
                # Take the session lock: HTTP readers inspect status/error under
                # the same lock (get_current_step/get_status), so the writer must
                # hold it too to avoid a data race on these fields. A cancel()
                # that won during the load already wrote a terminal "closed";
                # don't clobber it (mirror cancel()'s own terminal guard).
                with self.session._step_lock:
                    if self.session.status not in TERMINAL_STATUSES:
                        self.session.status = "failed"
                        self.session.error = str(exc)
                    final_status = self.session.status
                # Mirror the run's true terminal state into its protocol_runs row
                # so the unified active-run cap frees this slot without waiting
                # for the reaper's next reconcile pass.
                try:
                    run_repo.run_store.update_run(self.run_id, status=final_status)
                except Exception:
                    pass  # row bookkeeping is best-effort; the sweep reconciles
                return
            # Loaded. Advance the row to 'running' only if the run wasn't
            # cancelled/finished while loading — load_market_data leaves a
            # terminal status untouched under the lock, so reviving it to
            # 'running' here would resurrect a run the agent already cancelled.
            with self.session._step_lock:
                status_now = self.session.status
            if status_now not in TERMINAL_STATUSES:
                try:
                    run_repo.run_store.update_run(self.run_id, status="running")
                except Exception:
                    pass  # best-effort; both statuses count as active anyway

        self.session.status = "loading"
        threading.Thread(target=_load, daemon=True).start()

    def current_step_index(self) -> int:
        return self.session.step_index

    # -- context -----------------------------------------------------------

    def build_context(self) -> Dict[str, Any]:
        step = self.session.get_current_step()
        status = step.get("status")
        base: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": "backtest",
            "loop": self.loop,
            "universe": list(DJIA_30),
            "status": status,
            "step_index": self.session.step_index,
            "total_steps": self.session.total_steps,
            "news_sentiment": {},
            "news_overview": None,
        }
        if status != "waiting_decision":
            return base

        snap = step["market_snapshot"]
        sentiment, overview = load_news_sentiment(list(DJIA_30), step.get("timestamp"))
        envelope = {
            **base,
            "step_index": step["step_index"],
            "total_steps": step["total_steps"],
            "timestamp": step["timestamp"],
            "decision_deadline_at": step["decision_deadline_at"],
            "decision_timeout_seconds": step["decision_timeout_seconds"],
            "portfolio": snap["portfolio"],
            "current_holdings": snap["current_holdings"],
            "recent_trades": snap["recent_trades"],
            "top_signals": snap["top_signals"],
            "news_sentiment": sentiment,
            "news_overview": overview,
            "decision_format": step["decision_format"],
        }
        # Record the hash of exactly what we served, keyed by step, for the decision log.
        self.session.context_ref_by_step[step["step_index"]] = _context_hash(envelope)
        return envelope

    # -- decisions ---------------------------------------------------------

    @staticmethod
    def _raise_rejection(result: Dict[str, Any]) -> None:
        """Map the engine's non-accepted submit results to typed v2 errors.

        Flattening them into the ack shape would report a refused submission
        as an accepted-looking decision attributed to "external_agent"."""
        err = str(result.get("error") or "invalid_status")
        if err == "step_already_closed":
            raise ApiError(
                "step_already_closed",
                "The step closed before the decision arrived (step auto-held)",
                status=409, retryable=True,
                details={"outcome": result.get("outcome"),
                         "next_step": result.get("next_step")},
            )
        if err == "backtest_already_completed":
            raise ApiError("invalid_status", "Run already completed", status=409)
        if err.startswith("invalid_status:"):
            state = err.split(":", 1)[1]
            raise ApiError(
                "invalid_status",
                f"Run is not awaiting a decision (status: {state})", status=409,
                retryable=state == "loading",
            )
        # Engine-level payload rejection (parse failure etc.) — the step was
        # auto-held with decision_source="validation_hold" by the engine.
        raise ApiError(
            "validation_failed", err, status=422,
            details={"outcome": result.get("outcome")},
        )

    def apply_decisions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Actions arrive pre-validated from the v2 boundary (validate_actions);
        # this backend executes them and reports execution results. Schema-level
        # rejections are merged in by the boundary.
        result = self.session.submit_decisions({"actions": actions})
        if not result.get("accepted", True):
            self._raise_rejection(result)

        executed = [
            {"action": e.get("action"), "symbol": e.get("symbol"),
             "shares": e.get("shares"), "price": None}
            for e in (result.get("executed") or [])
        ]

        ack = {
            "accepted": bool(result.get("accepted", True)),
            "executed": executed,
            "rejected": [],
            "decision_source": result.get("decision_source") or "external_agent",
            "next_step": result.get("next_step", self.session.step_index),
            "status": result.get("status", self.session.status),
            "run_id": self.run_id,
            "metrics": result.get("metrics"),
        }
        if ack["status"] == "completed":
            # The final submit finalized the run — record the terminal state on
            # its protocol_runs row so cap counting and restart recovery see it.
            try:
                run_repo.run_store.update_run(
                    self.run_id,
                    status="completed",
                    result_run_id=self.session.run_id or self.run_id,
                )
            except Exception:
                pass  # best-effort; the reaper sweep reconciles
        return ack

    def decisions(self) -> List[Dict[str, Any]]:
        return self.session.get_decisions()

    def advance(self) -> None:
        # Lockstep engine advances inside submit; this only applies a pending
        # timeout. drain_expired() does exactly that (same auto-hold loop the v1
        # reaper uses) without get_current_step()'s discarded market-snapshot
        # rebuild — this runs every reaper pass, per live run, under _step_lock.
        self.session.drain_expired()

    def cancel(self) -> None:
        # Cancel only closes a run that is still running — never clobber a
        # terminal state. A finalizing submit racing a cancel would otherwise
        # flip completed → closed, and every later read (row reconcile,
        # archive, metrics) derives from this field.
        #
        # Fast path: a lock-free status read (same pattern as is_active). If the
        # run is already terminal — including "completed", which _finalize
        # publishes BEFORE its slow baseline block — return without taking the
        # lock, so a cancel never blocks for the finalize/baseline duration
        # (seconds-to-minutes held under _step_lock).
        if self.session.status in TERMINAL_STATUSES:
            return
        with self.session._step_lock:
            if self.session.status not in TERMINAL_STATUSES:
                self.session.status = "closed"

    # -- status / result ---------------------------------------------------

    def status(self) -> Dict[str, Any]:
        s = self.session.get_status()
        s["mode"] = "backtest"
        s["loop"] = self.loop
        return s

    def is_active(self) -> bool:
        # Deliberately an unlocked attribute read: the cap check runs under the
        # global create lock and must never take _step_lock (get_status can
        # cascade into _maybe_apply_timeout/_finalize). Worst case the cap
        # briefly counts a just-finished run — fine for a resource cap.
        return self.session.status not in TERMINAL_STATUSES

    def result(self) -> Optional[Dict[str, Any]]:
        if not self.session.run_id:
            return None
        base = ext.get_run_result(self.session.run_id, self.session.session_id)
        if base is None:
            return None
        base["manifest"] = db.get_run_manifest(self.run_id)
        return base


class ArchivedBacktestBackend(ExecutionBackend):
    """DB-backed tombstone for a terminal run whose engine session was freed.

    The reaper swaps this in for a finished BacktestBackend (releasing the
    market-data buffers — ~99% of a session's memory), and the v2 API
    rehydrates one from a terminal protocol_runs row after a restart. Reads
    (status/result/decisions and DB-cached idempotent replays) keep working;
    new decisions get the same invalid_status rejection the live path gives
    for a terminal run.
    """

    loop = "lockstep"

    _TERMINAL = TERMINAL_STATUSES

    def __init__(self, *, run_id: str, session_id: str, status: str,
                 error: Optional[str] = None, step_index: int = 0,
                 total_steps: int = 0, result_run_id: Optional[str] = None):
        self.run_id = run_id
        self.session_id = session_id
        self.terminal_status = status if status in self._TERMINAL else "failed"
        self.error = error
        self.step_index = int(step_index or 0)
        self.total_steps = int(total_steps or 0)
        self.result_run_id = result_run_id or run_id

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "ArchivedBacktestBackend":
        """Rebuild from a terminal protocol_runs row (post-restart reads).

        Step counts are persisted on the row at archive time (step_index /
        total_steps), so a failed/closed run reports its real progress instead
        of 0/0. Legacy rows that predate those columns fall back to the
        completed run's decision log (one row per executed step); a legacy
        failed run has no such log and stays 0/0, which is genuinely
        unrecoverable."""
        status = record.get("status") or "failed"
        result_run_id = record.get("result_run_id") or record["run_id"]
        step_index = record.get("step_index")
        total_steps = record.get("total_steps")
        if (step_index is None or total_steps is None) and status == "completed":
            try:
                count = len(db.get_decisions(result_run_id))
                step_index = count if step_index is None else step_index
                total_steps = count if total_steps is None else total_steps
            except Exception:
                pass
        step_index = step_index or 0
        total_steps = total_steps or 0
        return cls(
            run_id=record["run_id"],
            session_id=record["session_id"],
            status=status,
            result_run_id=record.get("result_run_id"),
            step_index=step_index,
            total_steps=total_steps,
        )

    # -- lifecycle: everything is over -------------------------------------

    def is_active(self) -> bool:
        return False

    def current_step_index(self) -> int:
        return self.step_index

    def advance(self) -> None:
        return None

    def cancel(self) -> None:
        return None  # already terminal; do not clobber completed → closed

    # -- reads --------------------------------------------------------------

    def build_context(self) -> Dict[str, Any]:
        # Mirrors the live backend's non-waiting envelope.
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": "backtest",
            "loop": self.loop,
            "universe": list(DJIA_30),
            "status": self.terminal_status,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "news_sentiment": {},
            "news_overview": None,
        }

    def apply_decisions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Same errors the live path raises for a terminal session (via
        # _raise_rejection), so archival is invisible to error handling.
        if self.terminal_status == "completed":
            raise ApiError("invalid_status", "Run already completed", status=409)
        raise ApiError(
            "invalid_status",
            f"Run is not awaiting a decision (status: {self.terminal_status})",
            status=409,
        )

    def status(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "backtest_id": self.run_id,
            "status": self.terminal_status,
            "step_index": self.step_index,
            "total_steps": self.total_steps,
            "run_id": self.result_run_id,
            "mode": "backtest",
            "loop": self.loop,
        }
        manifest = db.get_run_manifest(self.run_id) or {}
        base["agent_name"] = manifest.get("agent_name")
        base["model_name"] = manifest.get("model_name")
        if self.terminal_status == "completed":
            row = db.get_run(self.result_run_id)
            if row:
                base["metrics"] = ext.build_final_metrics(row)
                # The live get_status() carries baseline_run_ids + compare_url
                # for a completed run; rebuild them from the persisted baseline
                # columns so archival doesn't silently drop those fields.
                baseline_run_ids: Dict[str, str] = {}
                if row.get("baseline_buyhold_run_id"):
                    baseline_run_ids["buy_and_hold"] = row["baseline_buyhold_run_id"]
                if row.get("baseline_djia_run_id"):
                    baseline_run_ids["djia"] = row["baseline_djia_run_id"]
                base["baseline_run_ids"] = baseline_run_ids
                base["compare_url"] = ext.build_compare_url(
                    self.result_run_id, baseline_run_ids
                )
        if self.error:
            base["error"] = self.error
        return base

    def decisions(self) -> List[Dict[str, Any]]:
        try:
            return db.get_decisions(self.result_run_id)
        except Exception:
            return []

    def result(self) -> Optional[Dict[str, Any]]:
        base = ext.get_run_result(self.result_run_id, self.session_id)
        if base is None:
            return None
        base["manifest"] = db.get_run_manifest(self.run_id)
        return base
