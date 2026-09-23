"""Build lifecycle inputs from stored facts instead of an event scan.

``calculate_lifecycle`` always took a struct of timestamps and two counts.
What changed is where the struct comes from: two indexed reads of
``user_activity`` and ``user_daily_facts`` rather than a 180-day scan of
``analytics_events`` per user (design SS6.6).

Nothing in ``api/`` calls this module yet. PR B wires it into the one-user
profile route; the daily job (``daily_facts.py``) is its only caller in PR A.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

from .lifecycle import LifecycleInputs
from .repository_common import positive_user_id
from .value_repository import RecentFactTotals, UserActivity


def build_lifecycle_inputs(
    user_id: int,
    *,
    created_at: datetime,
    activity: UserActivity | None,
    totals: RecentFactTotals | None,
    as_of: datetime,
    day_activity_at: datetime | None = None,
) -> LifecycleInputs:
    """Assemble one user's lifecycle inputs from stored rows, as of ``as_of``.

    A user with no ``user_activity`` row has never done anything meaningful;
    ``calculate_lifecycle`` anchors on ``created_at`` in that case, so a
    brand-new account reads as New rather than as instantly Dormant.

    ``active_days_30d`` is capped at 30 because the model bounds it there and
    a migrated or double-counted window must not raise a validation error on
    a read path.

    **Every timestamp is clamped to ``as_of``, and that is not defensive
    programming -- it is the only thing that lets this function serve a past
    day at all.** ``calculate_lifecycle`` raises
    ``ValueError("lifecycle evidence cannot occur after as_of")`` when its
    anchor is newer than ``as_of`` (lifecycle.py:242-243), and
    ``user_activity`` is a single row per user overwritten in place: it says
    what is true *now*, never what was true at the end of some earlier day.
    The live read path passes ``as_of=now`` and nothing clamps; the daily job
    passes the end of the day it is computing, and without this every user
    who acted after midnight would abort the whole population's fact write.

    ``activated_at`` newer than ``as_of`` becomes None rather than being
    clamped to ``as_of``: "activated after this day" means "not activated as
    of this day", and pretending they activated at midnight would put them in
    the wrong retention cohort.

    ``last_meaningful_activity_at`` is only replaced when the stored value is
    newer than ``as_of``, because when it is not, it is authoritative and more
    precise than any fallback. The replacement walks newest-first:

    1. ``day_activity_at`` -- the user's own last activity **inside the day
       being computed**, from the daily job's one-day event aggregate. This
       has to come first and it is the case a date-window fallback alone
       silently gets wrong: a user whose only recent activity is on ``D``
       itself has no row in the ``D-29..D-1`` window at all, so skipping
       straight to ``totals`` would read them as inactive for thirty days and
       classify an active user as Dormant.
    2. ``totals.last_active_date`` -- the fact table *does* keep history,
       which is the whole reason it exists. Midnight-start of that date is the
       conservative choice; ``calculate_lifecycle`` counts ``inactive_days`` in
       whole UTC dates, so the time of day is never read.
    3. None -- no activity anywhere in the window, so ``created_at`` anchors
       the calculation, which is what a dormant classification needs.

    A user whose ``created_at`` is after ``as_of`` is **not** this function's
    problem to clamp -- ``calculate_lifecycle`` would raise on
    ``account_age_days < 0``, and rightly so, because an account that did not
    exist on day D has no day D. The daily job excludes those users from the
    eligible set instead; the live read path can never hit it.
    """
    subject_id = positive_user_id(user_id)
    counts = totals or RecentFactTotals()
    boundary = as_of

    activated_at = activity.activated_at if activity is not None else None
    if activated_at is not None and activated_at > boundary:
        activated_at = None

    last_activity = (
        activity.last_meaningful_activity_at if activity is not None else None
    )
    if last_activity is not None and last_activity > boundary:
        if day_activity_at is not None and day_activity_at <= boundary:
            last_activity = day_activity_at
        elif counts.last_active_date is not None:
            last_activity = datetime.combine(
                counts.last_active_date, time.min, tzinfo=timezone.utc
            )
        else:
            last_activity = None

    return LifecycleInputs(
        user_id=subject_id,
        created_at=created_at,
        first_successful_backtest_at=activated_at,
        last_meaningful_activity_at=last_activity,
        active_days_30d=min(30, counts.active_days),
        successful_backtests_30d=counts.successful_backtests,
    )


__all__ = ["build_lifecycle_inputs"]
