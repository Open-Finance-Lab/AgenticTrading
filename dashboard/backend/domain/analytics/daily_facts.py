"""One set-based pass per UTC day over the whole user population.

Every query here is parameterised by a date, never by a user id. That is the
property the read-budget test enforces, and it is the difference between this
module and the snapshot sweep it replaces: cost grows with days, not with
users times minutes (design SS6.9, SS6.12).

Scheduling lives in ``daily_job.py`` (its own worker thread, D23). This module
is the tick: claim yesterday, compute it, release or complete it, recompute any
recent day whose evidence changed, and give the retention coordinator its turn.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from dashboard.backend.domain.user_groups import coerce_user_group

from .lifecycle import (
    OperationalSignals,
    calculate_lifecycle,
    calculate_operational_state,
    commercial_tier,
)
from .lifecycle_reads import build_lifecycle_inputs
from .value_repository import (
    ActivityUpdate,
    LifecycleTransitionRow,
    RecentFactTotals,
    UserDailyFact,
    _timestamp,
)


DAILY_FACTS_JOB = "analytics_daily_facts"
RECOMPUTE_LOOKBACK_DAYS = 6
MAX_RECOMPUTES_PER_TICK = 2
TRAILING_WINDOW_DAYS = 30


class DailyFactsReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_date: date | None = None
    claimed: bool = False
    users_written: int = Field(default=0, ge=0)
    transitions_written: int = Field(default=0, ge=0)
    partial: bool = False
    failed_steps: tuple[str, ...] = ()
    # Earlier days this tick recomputed because late events landed in them.
    # Reported rather than silent: a day that keeps reappearing here is a
    # clock-skew or a replay problem, and the only place it is visible.
    recomputed_dates: tuple[date, ...] = ()


class _DayOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    users_written: int = 0
    transitions_written: int = 0
    failed_steps: tuple[str, ...] = ()


def _utc_now(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("now must include a timezone")
    return current.astimezone(timezone.utc)


def _default_value_store():
    from .repository import analytics_store
    from .value_repository import build_value_analytics_store

    return build_value_analytics_store(analytics_store)


def _default_run_history_store():
    from dashboard.backend.database import db

    return db


def _default_rollup() -> Callable[..., Any]:
    from .rollups import rollup_day

    return rollup_day


def _default_retention():
    from .retention import analytics_retention_coordinator

    return analytics_retention_coordinator


def _end_of_day(day: date) -> datetime:
    return datetime.combine(day, time(23, 59, 59, 999999), tzinfo=timezone.utc)


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def _count(totals: Any, field: str) -> int:
    return int(getattr(totals, field)) if totals is not None else 0


def _compute_day(
    day: date,
    *,
    now: datetime,
    store: Any,
    run_history_store: Any,
    rollup: Callable[..., Any],
) -> _DayOutcome:
    """Steps 1-6 of design SS6.9 for one day, each wrapped individually.

    Step order differs from the design's numbering in one place: the eligible
    population is read first, because the operational step (5) answers for
    exactly those ids. Query count is unchanged.
    """
    from .rollups import AnalyticsRollupStore

    analytics = store.analytics_base
    end_of_day = _end_of_day(day)
    failed: list[str] = []

    def step(name: str, action: Callable[[], Any], default: Any) -> Any:
        try:
            return action()
        except Exception as exc:
            failed.append(name)
            print(
                f"WARNING: analytics.daily_facts.{name}_failed "
                f"category={type(exc).__name__[:80]}"
            )
            return default

    # 0. Who has a day D at all: every non-admin, non-excluded account created
    #    on or before the end of D (9a: an account created after D has no D).
    def subjects_step() -> list[dict[str, Any]]:
        eligible = []
        for subject in analytics.list_daily_subjects():
            created_at = _timestamp(subject["created_at"])
            if created_at <= end_of_day:
                eligible.append({**subject, "created_at": created_at})
        return eligible

    subjects = step("subjects", subjects_step, None)
    if subjects is None:
        # Nothing can be written without the population; every later step
        # would only mark a day partial that has no rows to be partial.
        return _DayOutcome(failed_steps=tuple(failed))
    eligible_ids = [int(subject["id"]) for subject in subjects]

    # 1. Anonymous rollups for D (moved here from run_analytics_maintenance).
    step(
        "rollup",
        lambda: rollup(day, store=AnalyticsRollupStore(analytics), now=_midnight(now.date())),
        None,
    )

    # 2. One one-day event scan, then the idempotent activity correction.
    def events_step() -> dict[int, Any]:
        totals = store.aggregate_events_for_day(day)
        store.record_activity_batch(
            [
                ActivityUpdate(
                    user_id=item.user_id,
                    activated_at=item.first_success_at,
                    last_activity_at=item.last_activity_at,
                )
                for item in totals.values()
            ],
            now=now,
        )
        return totals

    events = step("events", events_step, {})

    # 3. Operator-funded cost from the run-history store.
    operator_cost = step(
        "runs", lambda: run_history_store.aggregate_operator_cost_for_day(day), {}
    )

    # 4. Own spend and lifetime net purchases from the credits store, then the
    #    same activity correction from the ledger's own timestamps.
    def ledger_step() -> dict[int, dict[str, Any]]:
        totals = store.credits_base.aggregate_ledger_for_day(day)
        store.record_activity_batch(
            [
                ActivityUpdate(
                    user_id=user_id,
                    last_activity_at=_timestamp(row["last_activity_at"]),
                )
                for user_id, row in totals.items()
                if row.get("last_activity_at")
            ],
            now=now,
        )
        return totals

    ledger = step("ledger", ledger_step, {})

    # 5. Operational signals for the eligible population, as of the end of D,
    #    read population-wide so no statement carries an IN list.
    signals = step(
        "operational",
        lambda: store.list_operational_signals(
            eligible_ids, now=end_of_day, population_wide=True
        ),
        {},
    )

    quality = "partial" if failed else "complete"

    # 6. The fact rows, then the transitions.
    def facts_step() -> tuple[int, list[tuple[UserDailyFact, int]]]:
        activity = store.list_activity(None)
        window = store.sum_recent_facts(
            None,
            start=day - timedelta(days=TRAILING_WINDOW_DAYS - 1),
            end=day - timedelta(days=1),
        )
        rows_with_days: list[tuple[UserDailyFact, int]] = []
        for subject in subjects:
            user_id = int(subject["id"])
            day_totals = events.get(user_id)
            active = day_totals is not None and day_totals.active
            prior = window.get(user_id, RecentFactTotals())
            # 9b: the window is D-29..D-1 in the table; D's own contribution
            # comes from the events aggregate already in hand, so the stored
            # segment equals what the read path computes tomorrow.
            combined = RecentFactTotals(
                active_days=prior.active_days + (1 if active else 0),
                successful_backtests=prior.successful_backtests
                + _count(day_totals, "runs_completed"),
                runs_requested=prior.runs_requested + _count(day_totals, "runs_requested"),
                runs_completed=prior.runs_completed + _count(day_totals, "runs_completed"),
                runs_failed=prior.runs_failed + _count(day_totals, "runs_failed"),
                runs_cancelled=prior.runs_cancelled + _count(day_totals, "runs_cancelled"),
                operator_cost_micro=prior.operator_cost_micro,
                own_spend_micro=prior.own_spend_micro,
                days_present=prior.days_present + 1,
                last_active_date=day if active else prior.last_active_date,
            )
            # 9a: evidence is clamped to the end of D inside build_lifecycle_inputs;
            # the day's own last activity is threaded in so a user whose only
            # recent activity is on D is not read as dormant.
            inputs = build_lifecycle_inputs(
                user_id,
                created_at=subject["created_at"],
                activity=activity.get(user_id),
                totals=combined,
                as_of=end_of_day,
                day_activity_at=day_totals.last_activity_at if day_totals is not None else None,
            )
            lifecycle = calculate_lifecycle(inputs, end_of_day)
            operational = calculate_operational_state(
                signals.get(user_id) or OperationalSignals(user_id=user_id), end_of_day
            )
            ledger_row = ledger.get(user_id, {})
            rows_with_days.append(
                (
                    UserDailyFact(
                        snapshot_date=day,
                        user_id=user_id,
                        lifecycle_segment=lifecycle.segment,
                        lifecycle_reason_code=lifecycle.reason_code,
                        operational_state=operational.state,
                        operational_reason_code=(
                            None if operational.state == "healthy" else operational.reason_code
                        ),
                        tier=commercial_tier(
                            int(ledger_row.get("lifetime_net_purchased_micro", 0))
                        ),
                        user_group=coerce_user_group(subject.get("user_group")),
                        active=bool(active),
                        runs_requested=_count(day_totals, "runs_requested"),
                        runs_completed=_count(day_totals, "runs_completed"),
                        runs_failed=_count(day_totals, "runs_failed"),
                        runs_cancelled=_count(day_totals, "runs_cancelled"),
                        operator_cost_micro=int(operator_cost.get(user_id, 0)),
                        own_spend_micro=int(ledger_row.get("own_spend_micro", 0)),
                        data_quality=quality,
                        calculated_at=now,
                    ),
                    lifecycle.inactive_days,
                )
            )
        written = store.upsert_daily_facts([row for row, _days in rows_with_days])
        return written, rows_with_days

    written, rows_with_days = step("facts", facts_step, (0, []))
    if "facts" in failed:
        return _DayOutcome(failed_steps=tuple(failed))

    def transitions_step() -> int:
        previous = {
            row.user_id: row for row in store.list_facts_for_date(day - timedelta(days=1))
        }
        changed: list[LifecycleTransitionRow] = []
        for row, inactive_days in rows_with_days:
            before = previous.get(row.user_id)
            if before is None or before.lifecycle_segment == row.lifecycle_segment:
                continue
            changed.append(
                LifecycleTransitionRow(
                    user_id=row.user_id,
                    snapshot_date=day,
                    from_segment=before.lifecycle_segment,
                    to_segment=row.lifecycle_segment,
                    inactive_days=inactive_days,
                    data_quality=quality,
                    created_at=now,
                )
            )
        return store.append_lifecycle_transitions(changed)

    transitions = step("transitions", transitions_step, 0)
    return _DayOutcome(
        users_written=written,
        transitions_written=transitions,
        failed_steps=tuple(failed),
    )


def run_daily_facts(
    *,
    now: datetime | None = None,
    value_store: Any = None,
    run_history_store: Any = None,
    rollup: Callable[..., Any] | None = None,
    retention: Any = None,
) -> DailyFactsReport:
    """One worker tick: yesterday if due, late arrivals, retention.

    The due check is the claim itself (Task 8): a tick on a day that is
    already complete costs exactly one store call, the refused
    compare-and-set. A claimed day whose steps all succeed is completed; a
    claimed day with any failed step is written ``partial`` (the UI labels
    such periods "Incomplete data") and **released**, so the next tick
    retries the whole day and corrects the partial rows once the source is
    back. Every retry is the same fixed set of set-based queries, so a source
    that stays down costs one bounded batch per tick, never more.

    The late-arrival sweep (design SS6.9 step 7) runs only on a claimed tick,
    for the six days before D; it does not touch the claim. The retention
    coordinator runs on every tick -- its own clock makes that free when it
    is not due, and calling it here rather than from the reaper is what keeps
    the two from double-writing (design SS6.9 step 8).
    """
    current = _utc_now(now)
    store = value_store if value_store is not None else _default_value_store()
    runs = run_history_store if run_history_store is not None else _default_run_history_store()
    rollup_fn = rollup if rollup is not None else _default_rollup()
    coordinator = retention if retention is not None else _default_retention()
    target = current.date() - timedelta(days=1)

    claimed = store.claim_projection_day(DAILY_FACTS_JOB, day=target, now=current)
    report = DailyFactsReport(snapshot_date=target, claimed=claimed)
    if claimed:
        outcome = _compute_day(
            target, now=current, store=store, run_history_store=runs, rollup=rollup_fn
        )
        if outcome.failed_steps:
            store.release_projection_day(DAILY_FACTS_JOB, now=current)
        else:
            store.complete_projection_day(DAILY_FACTS_JOB, day=target, now=current)

        recomputed: list[date] = []
        try:
            stale_days = store.list_days_needing_recompute(
                since=target - timedelta(days=RECOMPUTE_LOOKBACK_DAYS),
                until=target - timedelta(days=1),
            )
        except Exception as exc:
            stale_days = []
            print(
                "WARNING: analytics.daily_facts.recompute_scan_failed "
                f"category={type(exc).__name__[:80]}"
            )
        for stale_day in stale_days[:MAX_RECOMPUTES_PER_TICK]:
            _compute_day(
                stale_day, now=current, store=store, run_history_store=runs, rollup=rollup_fn
            )
            recomputed.append(stale_day)

        report = DailyFactsReport(
            snapshot_date=target,
            claimed=True,
            users_written=outcome.users_written,
            transitions_written=outcome.transitions_written,
            partial=bool(outcome.failed_steps),
            failed_steps=outcome.failed_steps,
            recomputed_dates=tuple(recomputed),
        )

    try:
        coordinator.run_if_due()
    except Exception as exc:
        print(
            "WARNING: analytics.daily_facts.retention_failed "
            f"category={type(exc).__name__[:80]}"
        )
    return report


__all__ = ["DAILY_FACTS_JOB", "DailyFactsReport", "run_daily_facts"]
