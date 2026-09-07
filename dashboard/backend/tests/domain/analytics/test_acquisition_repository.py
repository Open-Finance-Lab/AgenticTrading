"""Persistence contracts for acquisition attribution and correction audit."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from dashboard.backend.domain.analytics.acquisition import AcquisitionAttribution
from dashboard.backend.domain.analytics.repository import AnalyticsStore
from dashboard.backend.users import UserStore


NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def acquisition_store(tmp_path):
    db_path = tmp_path / "acquisition.db"
    users = UserStore(db_path=db_path)
    admin = users.create_user(
        "acquisition-admin@example.test", "Acquisition Admin", "SecurePass1!"
    )
    users.apply_admin_patch(admin["id"], role="admin")
    targets = [
        users.create_user(
            f"acquisition-{index}@example.test", f"Acquisition {index}", "SecurePass1!"
        )
        for index in range(5)
    ]
    return (
        AnalyticsStore(db_path=db_path),
        int(admin["id"]),
        [int(user["id"]) for user in targets],
    )


def assert_attribution_round_trip_contract(store, admin_id: int, user_id: int):
    first = store.record_initial_attribution(
        AcquisitionAttribution(
            user_id=user_id,
            source="community",
            cohort="launch-2026",
            method="invite",
            attributed_at=NOW,
        )
    )
    replay = store.record_initial_attribution(
        AcquisitionAttribution(
            user_id=user_id,
            source="student",
            cohort="must-not-overwrite",
            method="manual",
            attributed_at=NOW,
        )
    )

    assert replay == first
    corrected = store.update_user_attribution(
        user_id,
        source="competition",
        cohort="competition-2026",
        actor_user_id=admin_id,
        now=NOW,
    )
    assert corrected.source == "competition"
    assert corrected.cohort == "competition-2026"
    assert corrected.original_source == "community"
    assert corrected.original_cohort == "launch-2026"
    assert corrected.last_corrected_by_admin_id == admin_id

    audit = store.list_attribution_audit(user_id)
    assert audit == [
        {
            "actor_user_id": admin_id,
            "subject_user_id": user_id,
            "changed_at": NOW.isoformat(),
            "previous_source": "community",
            "new_source": "competition",
            "previous_cohort": "launch-2026",
            "new_cohort": "competition-2026",
        }
    ]
    assert "reason" not in audit[0]


def test_sqlite_attribution_round_trip_is_idempotent_and_audited(acquisition_store):
    store, admin_id, user_ids = acquisition_store
    assert_attribution_round_trip_contract(store, admin_id, user_ids[0])


def test_sqlite_supports_all_sources_and_batched_reads(acquisition_store):
    store, _admin_id, user_ids = acquisition_store
    sources = ["student", "community", "friend", "competition", "unknown"]
    for user_id, source in zip(user_ids, sources, strict=True):
        store.record_initial_attribution(
            AcquisitionAttribution(user_id=user_id, source=source)
        )

    rows = store.list_user_attributions([*user_ids, user_ids[0]])

    assert [rows[user_id].source for user_id in user_ids] == sources


def test_attribution_schema_is_idempotent_and_has_no_raw_tracking_fields(
    acquisition_store,
):
    store, _admin_id, _user_ids = acquisition_store
    store._init_schema()
    store._init_schema()
    with store._get_connection() as conn:
        tables = {
            "user_acquisition_attributions": {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(user_acquisition_attributions)"
                ).fetchall()
            },
            "admin_analytics_attribution_audit": {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(admin_analytics_attribution_audit)"
                ).fetchall()
            },
        }

    all_columns = set().union(*tables.values())
    assert {"source", "cohort", "original_source", "original_cohort"} <= all_columns
    assert {"invite_token", "referral_url", "reason", "provider_payload"}.isdisjoint(
        all_columns
    )
