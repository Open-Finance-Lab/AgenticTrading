"""Legacy quota coverage and the unified /backtest/run billing seam.

Folded in from issue #351: ``user_entitlements.credits`` was stored, capped,
admin-editable and returned to clients. Unified model execution no longer
charges that quota per run: BYOK bypasses ATL Credits, while Platform Credits
are reserved and settled from actual provider usage inside the worker.
"""

import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dashboard.backend.api.routers.backtests as bt
import dashboard.backend.users as users_module
from dashboard.backend.app import app
from dashboard.backend.domain.entitlements import credits
from dashboard.backend.domain.model_providers.execution_catalog import (
    ExecutionModelRoute,
)
from dashboard.backend.tests._frontend_source import APP_HTML, APP_JS, fn_body

# Captured before the autouse fixture below replaces it with a no-op: one test
# exercises an early worker failure while every endpoint test must never launch
# a subprocess.
_REAL_RUN_BACKGROUND = bt.run_backtest_background


class _ExecutionPreflightService:
    def preflight_execution_model(self, provider_id, catalog_model_id):
        assert provider_id == "openrouter"
        return ExecutionModelRoute(
            catalog_id=catalog_model_id,
            label=catalog_model_id,
            provider_model_id=catalog_model_id,
        )

    def preflight_user_default_credential(self, user_id, provider_id):
        assert int(user_id) > 0
        assert provider_id == "openrouter"

    def preflight_platform_credential(self, provider_id):
        assert provider_id == "openrouter"


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield users_module.UserStore(db_path=Path(tmpdir) / "users.db")


def _user(store, email="a@example.com"):
    return store.create_user(email, "A", "securepass1")


# ===========================================================================
# Store ledger
# ===========================================================================

def test_spend_seeds_the_defaults_then_debits(store):
    user = _user(store)
    assert store.try_spend_credits(user["id"]) == users_module.DEFAULT_CREDITS - 1
    ent = store.get_entitlements(user["id"])
    assert ent["credits"] == users_module.DEFAULT_CREDITS - 1
    # A metered spend is not an admin edit. Stamping the provenance columns
    # would make the console report every backtest as an entitlement change.
    assert ent["updated_at"] is None
    assert ent["updated_by_admin_id"] is None


def test_spend_refuses_at_zero_without_disturbing_the_row(store):
    user = _user(store)
    store.set_entitlements(user["id"], credits=1, updated_by_admin_id=user["id"])
    stamped = store.get_entitlements(user["id"])["updated_at"]

    assert store.try_spend_credits(user["id"]) == 0
    assert store.try_spend_credits(user["id"]) is None
    ent = store.get_entitlements(user["id"])
    assert ent["credits"] == 0
    # A refusal writes nothing at all — not even a touched timestamp.
    assert ent["updated_at"] == stamped


def test_a_refused_spend_never_seeds_todays_default(store, monkeypatch):
    """The seed is skipped when the default could not cover the spend.

    Seeding unconditionally would materialise a 0 row for every account that
    tried while the default was 0 — and because the default only ever fills in
    for a *missing* row, raising ``DEFAULT_CREDITS`` afterwards would never
    reach them again. The bug would look exactly like "the env var did
    nothing".
    """
    monkeypatch.setattr(users_module, "DEFAULT_CREDITS", 0)
    user = _user(store)
    assert store.try_spend_credits(user["id"]) is None

    monkeypatch.setattr(users_module, "DEFAULT_CREDITS", 5)
    assert store.try_spend_credits(user["id"]) == 4


def test_spend_for_an_unknown_account_creates_no_row(store):
    assert store.try_spend_credits(999_999) is None
    # SQLite does not enforce the FK on user_entitlements, so the seed's
    # ``SELECT ... FROM users WHERE id`` is the only thing keeping a ghost row
    # out of the table — and the only reason both twins behave alike here.
    conn = sqlite3.connect(str(store.db_path))
    try:
        rows = conn.execute(
            "SELECT COUNT(*) FROM user_entitlements WHERE user_id = 999999"
        ).fetchone()[0]
    finally:
        conn.close()
    assert rows == 0


def test_refund_returns_the_credit(store):
    user = _user(store)
    spent = store.try_spend_credits(user["id"])
    assert store.refund_credits(user["id"]) == spent + 1


def test_refund_clamps_at_the_admin_cap(store):
    user = _user(store)
    store.set_entitlements(
        user["id"],
        credits=users_module.MAX_CREDITS_CAP,
        updated_by_admin_id=user["id"],
    )
    # A double refund must not mint a balance past the ceiling an admin PATCH
    # is itself held to.
    assert store.refund_credits(user["id"]) == users_module.MAX_CREDITS_CAP


def test_refund_without_a_row_is_a_noop(store):
    user = _user(store)
    assert store.refund_credits(user["id"]) is None


@pytest.mark.parametrize("amount", [0, -1])
def test_nonpositive_amounts_are_rejected(store, amount):
    user = _user(store)
    with pytest.raises(ValueError, match="invalid_credit_amount"):
        store.try_spend_credits(user["id"], amount)
    with pytest.raises(ValueError, match="invalid_credit_amount"):
        store.refund_credits(user["id"], amount)


# ===========================================================================
# Policy
# ===========================================================================

def test_metering_is_off_unless_explicitly_armed(monkeypatch):
    assert credits.metering_enabled() is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("CREDITS_METERING_ENABLED", value)
        assert credits.metering_enabled() is True, value
    for value in ("0", "false", "", "maybe"):
        monkeypatch.setenv("CREDITS_METERING_ENABLED", value)
        assert credits.metering_enabled() is False, value


def test_disarmed_never_consults_the_store(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("the store must not be read when metering is off")

    monkeypatch.setattr(
        users_module.user_store, "try_spend_credits", _boom, raising=False
    )
    outcome = credits.authorize_llm_run(1)
    assert outcome.allowed is True
    assert outcome.charged is False


def test_armed_refuses_a_signed_out_caller(monkeypatch):
    monkeypatch.setenv("CREDITS_METERING_ENABLED", "1")
    outcome = credits.authorize_llm_run(None)
    assert outcome.allowed is False
    assert outcome.charged is False
    assert "sign in" in outcome.detail.lower()


def test_store_outage_fails_open_and_leaks_nothing(monkeypatch, capsys):
    monkeypatch.setenv("CREDITS_METERING_ENABLED", "1")

    def _boom(*args, **kwargs):
        # Shaped like the real thing: psycopg stringifies with its DSN.
        raise RuntimeError("connection to server at 'ep-secret-42.neon.tech' failed")

    monkeypatch.setattr(
        users_module.user_store, "try_spend_credits", _boom, raising=False
    )
    outcome = credits.authorize_llm_run(1)
    # Fails open, matching the per-owner run cap: metering sits on top of
    # controls that still hold, so a Neon outage degrades to today's behaviour
    # rather than taking LLM backtests down site-wide.
    assert outcome.allowed is True
    assert outcome.charged is False

    printed = capsys.readouterr().out
    # ...but not silently: an outage must not read as "metering is off".
    assert "credit metering" in printed
    # The marker stays static. Interpolating the exception would put the
    # connection string in the log an operator pastes into a ticket.
    assert "neon.tech" not in printed


def test_refund_only_fires_when_no_llm_call_was_made(monkeypatch):
    seen = []
    monkeypatch.setattr(
        users_module.user_store,
        "refund_credits",
        lambda user_id, amount: seen.append((user_id, amount)),
        raising=False,
    )
    # The model ran: the money is gone however the run ended.
    credits.refund_llm_run(7, llm_calls=3)
    assert seen == []

    credits.refund_llm_run(7, llm_calls=0)
    assert seen == [(7, credits.RUN_CREDIT_COST)]

    # Nothing was charged in the first place.
    credits.refund_llm_run(None, llm_calls=0)
    assert len(seen) == 1


def test_a_failing_refund_never_escapes(monkeypatch, capsys):
    def _boom(*args, **kwargs):
        raise RuntimeError("store down")

    monkeypatch.setattr(
        users_module.user_store, "refund_credits", _boom, raising=False
    )
    # Called from the background thread's finally: raising here would replace
    # the run's own recorded outcome with a traceback.
    credits.refund_llm_run(7, llm_calls=0)
    assert "refund failed" in capsys.readouterr().out


# ===========================================================================
# The /backtest/run seam
# ===========================================================================

@pytest.fixture(autouse=True)
def _reset_backtest_guards(monkeypatch):
    bt._backtest_rate_limiter.reset()
    bt.backtest_status.update({
        "running": False,
        "error": None,
        "runs_count": 0,
        "started_at": None,
        "progress_file": None,
        "live_run_id": None,
    })
    monkeypatch.setattr(bt, "run_backtest_background", lambda *a, **k: None)
    yield
    bt._backtest_rate_limiter.reset()


@pytest.fixture
def metered(monkeypatch):
    """TestClient over a fresh legacy quota store and safe execution preflight.

    A private store because these tests sign up accounts and edit balances;
    doing that in the shared conftest store would leak rows into every later
    test in the session.
    """
    monkeypatch.setenv("CREDITS_METERING_ENABLED", "1")
    monkeypatch.setenv("ENABLE_IFIND_ASHARE", "true")
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "test-token-not-a-secret")
    monkeypatch.setattr(bt, "ensure_llm_client_available", object, raising=False)
    monkeypatch.setattr(
        bt,
        "get_model_provider_service",
        lambda: _ExecutionPreflightService(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        store = users_module.UserStore(db_path=Path(tmpdir) / "users.db")
        monkeypatch.setattr(users_module, "user_store", store)
        yield TestClient(app), store


def _signup(client, email="user@example.com"):
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "display_name": "U", "password": "SecurePass1!"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["user"]


def _llm_run(client, **overrides):
    body = {
        "start_date": "2026-05-01",
        "end_date": "2026-05-02",
        "data_source": "ifind_ashare",
        "universe": "a_share_demo_6",
        "timeframe": "60m",
        "decision_source": "llm",
        "billing_mode": "byok",
        "provider_id": "openrouter",
        "model": "deepseek/deepseek-v4-pro",
    }
    body.update(overrides)
    return client.post(
        "/backtest/run", json=body, headers={"X-Session-Id": str(uuid.uuid4())}
    )


def test_signed_out_llm_run_is_refused_before_execution_preflight(metered):
    client, _ = metered
    resp = _llm_run(client)
    assert resp.status_code == 401
    assert "sign in" in resp.text.lower()


def test_signed_in_byok_run_does_not_debit_legacy_quota(metered):
    client, store = metered
    user = _signup(client)
    before = store.get_entitlements(user["id"])["credits"]
    resp = _llm_run(client)
    assert resp.status_code == 200, resp.text
    assert resp.json()["billing_mode"] == "byok"
    assert store.get_entitlements(user["id"])["credits"] == before


def test_byok_run_ignores_the_retired_per_run_quota(metered):
    client, store = metered
    user = _signup(client)
    store.set_entitlements(user["id"], credits=0, updated_by_admin_id=user["id"])
    resp = _llm_run(client)
    assert resp.status_code == 200, resp.text
    assert resp.json()["billing_mode"] == "byok"
    assert store.get_entitlements(user["id"])["credits"] == 0


def test_a_rule_based_run_is_never_charged(metered):
    client, store = metered
    user = _signup(client)
    store.set_entitlements(user["id"], credits=0, updated_by_admin_id=user["id"])
    # No model call, no operator spend — an empty balance must not block it.
    resp = client.post(
        "/backtest/run",
        json={
            "start_date": "2026-05-01",
            "end_date": "2026-05-02",
            "decision_source": "rule_based",
        },
        headers={"X-Session-Id": str(uuid.uuid4())},
    )
    assert resp.status_code == 200, resp.text
    assert store.get_entitlements(user["id"])["credits"] == 0


def test_a_run_refused_for_capacity_is_not_charged(metered):
    client, store = metered
    user = _signup(client)
    before = store.get_entitlements(user["id"])["credits"]

    bt._reset_slots_for_tests()
    try:
        # Fill the server-wide ceiling with other people's runs.
        #
        # This test used to set ``backtest_status["running"] = True``, back when
        # one global flag WAS the refusal. Under the slot ledger that flag
        # refuses nothing, so left as it was this test would have gone on
        # passing while every capacity-refused caller was silently charged --
        # the assertion still holds when the request is *accepted*, because an
        # accepted run is legitimately debited.
        for index in range(bt.MAX_ACTIVE_DASHBOARD_BACKTESTS):
            assert bt._try_acquire_backtest_slot(
                live_run_id=f"other-run-{index}",
                session_id=f"other-session-{index}",
                user_id=None,
            ) is None

        resp = _llm_run(client)
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        # The debit sits after the concurrency check on purpose: a caller turned
        # away because someone else's backtest is running never got a run.
        assert store.get_entitlements(user["id"])["credits"] == before
    finally:
        bt._reset_slots_for_tests()


def test_byok_does_not_depend_on_the_legacy_metering_switch(metered, monkeypatch):
    client, store = metered
    monkeypatch.delenv("CREDITS_METERING_ENABLED", raising=False)
    user = _signup(client)
    store.set_entitlements(user["id"], credits=0, updated_by_admin_id=user["id"])
    resp = _llm_run(client)
    assert resp.status_code == 200, resp.text
    assert store.get_entitlements(user["id"])["credits"] == 0


def test_a_thread_that_never_starts_needs_no_flat_credit_refund(metered, monkeypatch):
    client, store = metered
    user = _signup(client)
    before = store.get_entitlements(user["id"])["credits"]

    class _DeadThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(bt, "_BackgroundThread", _DeadThread)
    with pytest.raises(RuntimeError):
        _llm_run(client)

    # Unified billing reserves per model call inside the worker, so a thread
    # that never starts has no flat run charge to refund.
    assert store.get_entitlements(user["id"])["credits"] == before
    # And the slot is released, or it burns one of the owner's concurrent slots
    # and one of the server's for the life of the process. Asserted on the
    # ledger, not just the legacy mirror: the mirror tracks whichever slot
    # changed last, so it reads False while the slot is still held.
    assert bt.count_active_dashboard_backtests() == 0
    assert bt.backtest_status["running"] is False


def test_a_run_that_never_reaches_the_model_does_not_mutate_legacy_quota(metered):
    _client, store = metered
    user = _user(store, "worker@example.com")
    store.set_entitlements(user["id"], credits=5, updated_by_admin_id=user["id"])

    # An unknown data source fails before any decision or model reservation.
    _REAL_RUN_BACKGROUND(
        start_date="2026-05-01",
        end_date="2026-05-02",
        session_id="s",
        data_source="not-a-real-source",
        live_run_id="never-persisted",
    )
    assert store.get_entitlements(user["id"])["credits"] == 5


def test_admin_stats_reports_whether_the_column_binds(metered):
    client, store = metered
    user = _signup(client)
    store.apply_admin_patch(user["id"], role="admin")
    data = client.get("/api/admin/stats").json()
    assert data["credits_metering_enabled"] is True
    assert data["default_credits"] == users_module.DEFAULT_CREDITS


# ===========================================================================
# Console label (static-source guard — /app has no JS test toolchain)
# ===========================================================================

def test_console_credits_note_is_server_driven():
    # A hardcoded "(not enforced yet)" would keep saying that after an operator
    # armed metering — the exact staleness #351 was about.
    assert 'id="adminCreditsNote"' in APP_HTML
    assert "not enforced yet" not in APP_HTML

    body = fn_body("function setAdminCreditsNote(stats)")
    assert "credits_metering_enabled" in body
    # Three states: a failed stats call must not render as "metering off".
    assert "status unavailable" in body
    assert "metering off" in body
    assert "loadAdminStats" in APP_JS
