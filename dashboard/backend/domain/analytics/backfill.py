"""Deterministic, authoritative Analytics history reconstruction.

Only source records that prove an authenticated owner and a safe lifecycle
outcome become Analytics events.  Browser/session/network history is never
invented.  The source adapter deliberately reads each database independently;
cross-database ownership is resolved in Python.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Protocol
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dashboard.backend.infrastructure.llm.execution.models import LLMRunEvidence

from .models import (
    ALLOWED_SERVER_EVENT_NAMES,
    EVENT_GROUP_BY_NAME,
    AnalyticsEventRecord,
)
from .repository import analytics_store


_BACKFILL_EVENT_NAMESPACE = UUID("bfa8a3a5-9691-5fe4-8871-6174934f73a8")
_TERMINAL_RUN_EVENTS = {
    "completed": ("backtest_completed", "succeeded", None),
    "failed": ("backtest_failed", "failed", None),
    "cancelled": ("backtest_cancelled", "cancelled", None),
    "canceled": ("backtest_cancelled", "cancelled", None),
    "closed": ("backtest_cancelled", "cancelled", None),
}


def _as_utc(value: object, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid timestamp") from exc
    else:
        raise ValueError(f"{field_name} must be a valid timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        # Existing SQLite CURRENT_TIMESTAMP rows are timezone-naive UTC.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class BackfillCandidate(BaseModel):
    """One safe event reconstructed from an authoritative source record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_name: str = Field(min_length=1, max_length=64)
    user_id: int = Field(gt=0)
    occurred_at: datetime
    source_event_id: str = Field(min_length=1, max_length=200)
    source_record_type: str = Field(min_length=1, max_length=64)
    source_record_id: str = Field(min_length=1, max_length=200)
    correlation_id: str | None = Field(default=None, min_length=1, max_length=200)
    provider_id: str | None = Field(default=None, min_length=1, max_length=128)
    model_id: str | None = Field(default=None, min_length=1, max_length=256)
    billing_mode: str | None = Field(default=None, min_length=1, max_length=32)
    outcome: str | None = Field(default=None, min_length=1, max_length=32)
    error_category: str | None = Field(default=None, min_length=1, max_length=64)
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_name")
    @classmethod
    def require_server_event(cls, value: str) -> str:
        if value not in ALLOWED_SERVER_EVENT_NAMES:
            raise ValueError("backfill accepts only server-authoritative events")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value, "occurred_at")


class BackfillCollection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: list[BackfillCandidate] = Field(default_factory=list)
    skipped_unmapped_owner: int = Field(default=0, ge=0)
    skipped_invalid: int = Field(default=0, ge=0)


class BackfillReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    considered: int = Field(default=0, ge=0)
    would_insert: int = Field(default=0, ge=0)
    inserted: int = Field(default=0, ge=0)
    duplicates: int = Field(default=0, ge=0)
    skipped_outside_window: int = Field(default=0, ge=0)
    skipped_unmapped_owner: int = Field(default=0, ge=0)
    skipped_invalid: int = Field(default=0, ge=0)
    write_failures: int = Field(default=0, ge=0)
    repair_failures: int = Field(default=0, ge=0)
    affected_users: int = Field(default=0, ge=0)
    repaired_snapshots: int = Field(default=0, ge=0)
    rebuilt_rollup_days: int = Field(default=0, ge=0)
    source_event_ids: list[str] = Field(default_factory=list)


class BackfillSource(Protocol):
    def collect(self, *, start: datetime, end: datetime) -> BackfillCollection: ...


def _query_all(store: Any, sql: str) -> list[dict[str, Any]]:
    """Read safe source columns from either store twin without joining stores."""

    if hasattr(store, "database_url"):
        with store._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return [dict(row) for row in cur.fetchall()]
    conn = store._get_connection()
    try:
        return [dict(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def _all_users(user_store: Any) -> list[dict[str, Any]]:
    users: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = user_store.list_users_admin(limit=500, offset=offset)
        users.extend(page)
        if len(page) < 500:
            return users
        offset += len(page)


def _credit_candidates(
    credits_store: Any,
) -> tuple[list[BackfillCandidate], int]:
    candidates: list[BackfillCandidate] = []
    invalid = 0
    try:
        reservations = _query_all(
            credits_store,
            """
            SELECT reservation_id, user_id, run_id, call_index,
                   reserved_grant_micro, reserved_purchased_micro,
                   status, created_at, updated_at
            FROM credit_llm_reservations
            ORDER BY created_at, reservation_id
            """,
        )
        usage_entries = _query_all(
            credits_store,
            """
            SELECT id, user_id, reservation_id, run_id, call_index,
                   bucket, amount_micro, created_at
            FROM credit_llm_usage_entries
            ORDER BY created_at, id
            """,
        )
    except Exception:
        return [], 1

    debited_by_reservation: dict[str, dict[str, int]] = defaultdict(
        lambda: {"grant": 0, "purchased": 0}
    )
    for row in usage_entries:
        try:
            bucket = str(row["bucket"])
            amount = -int(row["amount_micro"])
            if bucket not in {"grant", "purchased"} or amount <= 0:
                raise ValueError("invalid Credits usage row")
            reservation_id = str(row["reservation_id"])
            debited_by_reservation[reservation_id][bucket] += amount
            candidates.append(
                BackfillCandidate(
                    event_name="credits_settled",
                    user_id=int(row["user_id"]),
                    occurred_at=_as_utc(row["created_at"], "created_at"),
                    source_event_id=(
                        f"resource:credits_settled:{reservation_id}:{bucket}"
                    ),
                    source_record_type="credit_reservation",
                    source_record_id=reservation_id,
                    correlation_id=str(row["run_id"]),
                    billing_mode="platform_credits",
                    properties={"amount_micro": amount, "bucket": bucket},
                )
            )
        except Exception:
            invalid += 1

    for row in reservations:
        try:
            reservation_id = str(row["reservation_id"])
            user_id = int(row["user_id"])
            run_id = str(row["run_id"])
            reserved = {
                "grant": int(row.get("reserved_grant_micro") or 0),
                "purchased": int(row.get("reserved_purchased_micro") or 0),
            }
            for bucket, amount in reserved.items():
                if amount <= 0:
                    continue
                candidates.append(
                    BackfillCandidate(
                        event_name="credits_reserved",
                        user_id=user_id,
                        occurred_at=_as_utc(row["created_at"], "created_at"),
                        source_event_id=(
                            f"resource:credits_reserved:{reservation_id}:{bucket}"
                        ),
                        source_record_type="credit_reservation",
                        source_record_id=reservation_id,
                        correlation_id=run_id,
                        billing_mode="platform_credits",
                        properties={"amount_micro": amount, "bucket": bucket},
                    )
                )
            if str(row.get("status")) not in {"released", "settled"}:
                continue
            for bucket, reserved_amount in reserved.items():
                released = max(
                    0,
                    reserved_amount
                    - debited_by_reservation[reservation_id][bucket],
                )
                if released <= 0:
                    continue
                candidates.append(
                    BackfillCandidate(
                        event_name="credits_refunded",
                        user_id=user_id,
                        occurred_at=_as_utc(row["updated_at"], "updated_at"),
                        source_event_id=(
                            f"resource:credits_refunded:{reservation_id}:{bucket}"
                        ),
                        source_record_type="credit_reservation",
                        source_record_id=reservation_id,
                        correlation_id=run_id,
                        billing_mode="platform_credits",
                        properties={"amount_micro": released, "bucket": bucket},
                    )
                )
        except Exception:
            invalid += 1
    return candidates, invalid


def _usage_candidate(
    run: dict[str, Any],
    *,
    user_id: int,
) -> BackfillCandidate | None:
    metadata = run.get("metadata")
    raw = metadata.get("llm_execution") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return None
    safe_projection = {
        name: raw[name]
        for name in LLMRunEvidence.model_fields
        if name in raw
    }
    evidence = LLMRunEvidence.model_validate(safe_projection)
    # Aggregate evidence cannot be split into truthful per-call token counts.
    # A one-call run maps exactly to the live call_index=0 identity; multi-call
    # aggregates are skipped instead of fabricating call-level rows.
    if evidence.call_count != 1:
        raise ValueError("multi-call evidence lacks per-call usage")
    run_id = str(run["run_id"])
    cost_usd = 0.0
    if evidence.billing_mode.value == "platform_credits":
        cost_usd = (
            evidence.provider_cost_usd
            if evidence.provider_cost_usd is not None
            else evidence.estimated_cost_usd or 0.0
        )
    return BackfillCandidate(
        event_name="model_usage_recorded",
        user_id=user_id,
        occurred_at=_as_utc(
            run.get("updated_at") or run.get("created_at"),
            "usage occurred_at",
        ),
        source_event_id=f"resource:model_usage_recorded:{run_id}:0",
        source_record_type="run",
        source_record_id=run_id,
        correlation_id=run_id,
        provider_id=evidence.provider_id,
        model_id=evidence.model_id,
        billing_mode=evidence.billing_mode.value,
        outcome="succeeded",
        properties={
            "input_tokens": evidence.input_tokens,
            "output_tokens": evidence.output_tokens,
            "cost_micro_usd": max(0, round(cost_usd * 1_000_000)),
        },
    )


class AuthoritativeBackfillSource:
    """Compose existing source stores without cross-database joins."""

    def __init__(
        self,
        *,
        user_store: Any | None = None,
        agent_store: Any | None = None,
        protocol_run_store: Any | None = None,
        run_history_store: Any | None = None,
        credits_store: Any | None = None,
    ):
        if user_store is None:
            from dashboard.backend.users import user_store as default_user_store

            user_store = default_user_store
        if agent_store is None:
            from dashboard.backend.domain.agents.repository import (
                agent_store as default_agent_store,
            )

            agent_store = default_agent_store
        if protocol_run_store is None:
            from dashboard.backend.domain.runs.repository import (
                run_store as default_protocol_run_store,
            )

            protocol_run_store = default_protocol_run_store
        if run_history_store is None:
            from dashboard.backend.database import db as default_run_history_store

            run_history_store = default_run_history_store
        if credits_store is None:
            from dashboard.backend.domain.credits.repository import (
                credits_store as default_credits_store,
            )

            credits_store = default_credits_store
        self.user_store = user_store
        self.agent_store = agent_store
        self.protocol_run_store = protocol_run_store
        self.run_history_store = run_history_store
        self.credits_store = credits_store

    def collect(self, *, start: datetime, end: datetime) -> BackfillCollection:
        del start, end  # The service applies the authoritative inclusive window.
        candidates: list[BackfillCandidate] = []
        invalid = 0
        unmapped = 0

        users = _all_users(self.user_store)
        user_ids = {int(user["id"]) for user in users}
        for user in users:
            try:
                user_id = int(user["id"])
                candidates.append(
                    BackfillCandidate(
                        event_name="account_signed_up",
                        user_id=user_id,
                        occurred_at=_as_utc(user["created_at"], "created_at"),
                        source_event_id=f"account:account_signed_up:{user_id}",
                        source_record_type="user",
                        source_record_id=str(user_id),
                    )
                )
            except Exception:
                invalid += 1

        try:
            agents = _query_all(
                self.agent_store,
                """
                SELECT agent_id, session_id, owner_user_id, created_at
                FROM external_agents
                ORDER BY created_at, agent_id
                """,
            )
        except Exception:
            agents = []
            invalid += 1

        owned_agents: list[dict[str, Any]] = []
        for agent in agents:
            owner = agent.get("owner_user_id")
            if owner is None or int(owner) not in user_ids:
                unmapped += 1
                continue
            owned_agents.append(agent)
            try:
                candidates.append(
                    BackfillCandidate(
                        event_name="agent_created",
                        user_id=int(owner),
                        occurred_at=_as_utc(agent["created_at"], "created_at"),
                        source_event_id=(
                            f"agent:agent_created:{str(agent['agent_id'])}"
                        ),
                        source_record_type="agent",
                        source_record_id=str(agent["agent_id"]),
                    )
                )
            except Exception:
                invalid += 1

        result_run_ids: set[str] = set()
        for agent in owned_agents:
            owner = int(agent["owner_user_id"])
            try:
                protocol_runs = self.protocol_run_store.list_runs(
                    str(agent["agent_id"])
                )
            except Exception:
                invalid += 1
                continue
            for run in protocol_runs:
                if run.get("result_run_id"):
                    result_run_ids.add(str(run["result_run_id"]))
                terminal = _TERMINAL_RUN_EVENTS.get(str(run.get("status")).lower())
                if terminal is None:
                    continue
                try:
                    event_name, outcome, error_category = terminal
                    run_id = str(run["run_id"])
                    candidates.append(
                        BackfillCandidate(
                            event_name=event_name,
                            user_id=owner,
                            occurred_at=_as_utc(
                                run.get("updated_at") or run.get("created_at"),
                                "run occurred_at",
                            ),
                            source_event_id=f"run:{event_name}:{run_id}",
                            source_record_type="run",
                            source_record_id=run_id,
                            correlation_id=run_id,
                            outcome=outcome,
                            error_category=error_category,
                        )
                    )
                except Exception:
                    invalid += 1

        sessions = [
            str(agent["session_id"])
            for agent in owned_agents
            if agent.get("session_id")
        ]
        try:
            runs_by_session = self.run_history_store.get_runs_by_sessions(sessions)
        except Exception:
            runs_by_session = {}
            invalid += 1
        for agent in owned_agents:
            owner = int(agent["owner_user_id"])
            for run in runs_by_session.get(str(agent.get("session_id")), []):
                run_id = str(run.get("run_id") or "")
                if not run_id or run_id in result_run_ids:
                    continue
                if str(run.get("mode") or "").lower() in {
                    "baseline",
                    "paper_baseline",
                }:
                    continue
                try:
                    occurred_at = _as_utc(
                        run.get("updated_at") or run.get("created_at"),
                        "run occurred_at",
                    )
                    candidates.append(
                        BackfillCandidate(
                            event_name="backtest_completed",
                            user_id=owner,
                            occurred_at=occurred_at,
                            source_event_id=f"run:backtest_completed:{run_id}",
                            source_record_type="run",
                            source_record_id=run_id,
                            correlation_id=run_id,
                            outcome="succeeded",
                        )
                    )
                    usage = _usage_candidate(run, user_id=owner)
                    if usage is not None:
                        candidates.append(usage)
                except Exception:
                    invalid += 1

        credit_rows, credit_invalid = _credit_candidates(self.credits_store)
        candidates.extend(credit_rows)
        invalid += credit_invalid
        return BackfillCollection(
            candidates=candidates,
            skipped_unmapped_owner=unmapped,
            skipped_invalid=invalid,
        )


def _existing_source_event_ids(store: Any, source_ids: Iterable[str]) -> set[str]:
    values = sorted(set(source_ids))
    if not values or not hasattr(store, "_get_connection"):
        return set()
    existing: set[str] = set()
    for offset in range(0, len(values), 500):
        chunk = values[offset : offset + 500]
        if hasattr(store, "database_url"):
            with store._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT source_event_id FROM analytics_events "
                        "WHERE source_event_id = ANY(%s)",
                        (chunk,),
                    )
                    existing.update(
                        str(row["source_event_id"]) for row in cur.fetchall()
                    )
        else:
            placeholders = ",".join("?" for _ in chunk)
            with store._get_connection() as conn:
                rows = conn.execute(
                    "SELECT source_event_id FROM analytics_events "
                    f"WHERE source_event_id IN ({placeholders})",
                    chunk,
                ).fetchall()
            existing.update(str(row["source_event_id"]) for row in rows)
    return existing


def _event(candidate: BackfillCandidate, *, received_at: datetime) -> AnalyticsEventRecord:
    return AnalyticsEventRecord(
        event_id=str(uuid5(_BACKFILL_EVENT_NAMESPACE, candidate.source_event_id)),
        schema_version=1,
        event_name=candidate.event_name,
        event_group=EVENT_GROUP_BY_NAME[candidate.event_name],
        user_id=candidate.user_id,
        session_id=None,
        occurred_at=candidate.occurred_at,
        received_at=received_at,
        event_source="backfill",
        source_event_id=candidate.source_event_id,
        source_record_type=candidate.source_record_type,
        source_record_id=candidate.source_record_id,
        correlation_id=candidate.correlation_id,
        page_view=None,
        provider_id=candidate.provider_id,
        model_id=candidate.model_id,
        billing_mode=candidate.billing_mode,
        outcome=candidate.outcome,
        error_category=candidate.error_category,
        country_code=None,
        device_category=None,
        browser_family=None,
        network_hash=None,
        properties=candidate.properties,
    )


def backfill_analytics(
    *,
    now: datetime | None = None,
    days: int = 180,
    before: datetime | None = None,
    dry_run: bool = False,
    source: BackfillSource | None = None,
    store: Any = analytics_store,
    recalculate_snapshot: Any | None = None,
    rebuild_rollup: Any | None = None,
) -> BackfillReport:
    """Reconstruct safe events inside ``[now - days, before]``.

    Bulk repairs run once per affected subject/day after all inserts complete.
    Dry runs enumerate and validate sources but perform no writes at all.
    """

    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 180:
        raise ValueError("days must be an integer from 1 through 180")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")
    current = _as_utc(now or datetime.now(timezone.utc), "now")
    end = _as_utc(before or current, "before")
    start = current - timedelta(days=days)
    if end > current:
        raise ValueError("before cannot be in the future")
    if end < start:
        raise ValueError("before cannot be earlier than the backfill window")

    collection = (source or AuthoritativeBackfillSource()).collect(
        start=start,
        end=end,
    )
    candidates = sorted(
        collection.candidates,
        key=lambda item: (item.occurred_at, item.source_event_id),
    )
    in_window = [item for item in candidates if start <= item.occurred_at <= end]
    skipped_outside = len(candidates) - len(in_window)
    existing = _existing_source_event_ids(
        store,
        (item.source_event_id for item in in_window),
    )
    pending = [item for item in in_window if item.source_event_id not in existing]
    prepared: list[tuple[BackfillCandidate, AnalyticsEventRecord]] = []
    validation_failures = 0
    for candidate in pending:
        try:
            prepared.append((candidate, _event(candidate, received_at=current)))
        except Exception:
            validation_failures += 1
    if dry_run:
        return BackfillReport(
            considered=len(candidates),
            would_insert=len(prepared),
            inserted=0,
            duplicates=len(in_window) - len(pending),
            skipped_outside_window=skipped_outside,
            skipped_unmapped_owner=collection.skipped_unmapped_owner,
            skipped_invalid=collection.skipped_invalid + validation_failures,
            affected_users=len({item.user_id for item, _event_record in prepared}),
            rebuilt_rollup_days=len(
                {
                    item.occurred_at.date()
                    for item, _event_record in prepared
                    if item.occurred_at.date() < current.date()
                }
            ),
            source_event_ids=[
                item.source_event_id for item, _event_record in prepared
            ],
        )

    inserted_ids: list[str] = []
    affected_users: set[int] = set()
    affected_days: set[date] = set()
    duplicates = len(in_window) - len(pending)
    write_failures = 0
    for candidate, event_record in prepared:
        try:
            result = store.append_event(event_record)
        except Exception:
            write_failures += 1
            continue
        if not result.created:
            duplicates += 1
            continue
        inserted_ids.append(candidate.source_event_id)
        affected_users.add(candidate.user_id)
        if candidate.occurred_at.date() < current.date():
            affected_days.add(candidate.occurred_at.date())

    if recalculate_snapshot is None:
        from .states import AnalyticsStateStore, recalculate_user_snapshot

        state_store = AnalyticsStateStore(store)

        def recalculate_snapshot(user_id: int, **kwargs: Any) -> Any:
            return recalculate_user_snapshot(user_id, store=state_store, **kwargs)

    if rebuild_rollup is None:
        from .rollups import AnalyticsRollupStore, rollup_day

        rollup_store = AnalyticsRollupStore(store)

        def rebuild_rollup(day: date, **kwargs: Any) -> Any:
            return rollup_day(day, store=rollup_store, **kwargs)

    repair_failures = 0
    repaired_snapshots = 0
    rebuilt_rollup_days = 0
    for user_id in sorted(affected_users):
        try:
            recalculate_snapshot(user_id, now=current)
            repaired_snapshots += 1
        except Exception:
            repair_failures += 1
    for day in sorted(affected_days):
        try:
            rebuild_rollup(day, now=datetime.combine(
                day + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ))
            rebuilt_rollup_days += 1
        except Exception:
            repair_failures += 1

    return BackfillReport(
        considered=len(candidates),
        would_insert=len(prepared),
        inserted=len(inserted_ids),
        duplicates=duplicates,
        skipped_outside_window=skipped_outside,
        skipped_unmapped_owner=collection.skipped_unmapped_owner,
        skipped_invalid=collection.skipped_invalid + validation_failures,
        write_failures=write_failures,
        repair_failures=repair_failures,
        affected_users=len(affected_users),
        repaired_snapshots=repaired_snapshots,
        rebuilt_rollup_days=rebuilt_rollup_days,
        source_event_ids=inserted_ids,
    )


__all__ = [
    "AuthoritativeBackfillSource",
    "BackfillCandidate",
    "BackfillCollection",
    "BackfillReport",
    "BackfillSource",
    "backfill_analytics",
]
