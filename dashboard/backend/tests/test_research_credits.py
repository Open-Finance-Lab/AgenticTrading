"""Research-run credit metering (route 0 billing, design N2/PR2).

Pure-function level: the authorize/refund helpers against a temp sqlite
store, mirroring test_credit_metering.py's seams. The route wiring is
exercised end-to-end in test_admin_absorption-style browser flow instead —
these tests own the billing decision table.
"""

import tempfile
from pathlib import Path

import pytest

import dashboard.backend.users as users_module
from dashboard.backend.domain.entitlements import research_credits


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield users_module.UserStore(db_path=Path(tmpdir) / "users.db")


@pytest.fixture
def seeded_user(store, monkeypatch):
    """A signed-up account with a known balance on the temp sqlite store."""
    monkeypatch.setattr(users_module, "user_store", store)
    user = store.create_user(
        "research.credits@example.test", "Research Credits", "SecurePass1!"
    )
    store.set_entitlements(user["id"], credits=20)
    return user


def test_metering_off_allows_without_charging(seeded_user, monkeypatch):
    monkeypatch.delenv("CREDITS_METERING_ENABLED", raising=False)
    outcome = research_credits.authorize_research_run(seeded_user["id"])
    assert outcome.allowed and not outcome.charged


def test_metering_armed_debits_the_configured_cost(seeded_user, monkeypatch):
    monkeypatch.setenv("CREDITS_METERING_ENABLED", "1")
    monkeypatch.setenv("RESEARCH_RUN_CREDIT_COST", "5")
    outcome = research_credits.authorize_research_run(seeded_user["id"])
    assert outcome.allowed and outcome.charged
    assert outcome.balance == 15


def test_empty_balance_refuses_with_402_shape(seeded_user, monkeypatch):
    monkeypatch.setenv("CREDITS_METERING_ENABLED", "1")
    monkeypatch.setenv("RESEARCH_RUN_CREDIT_COST", "25")
    outcome = research_credits.authorize_research_run(seeded_user["id"])
    assert not outcome.allowed and not outcome.charged
    assert "out of credits" in outcome.detail


def test_anonymous_refused_when_armed(seeded_user, monkeypatch):
    monkeypatch.setenv("CREDITS_METERING_ENABLED", "1")
    outcome = research_credits.authorize_research_run(None)
    assert not outcome.allowed
    assert "Sign in" in outcome.detail


def test_bad_cost_value_falls_back_to_10(monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_CREDIT_COST", "not-a-number")
    assert research_credits.research_run_cost() == 10
    monkeypatch.setenv("RESEARCH_RUN_CREDIT_COST", "0")
    assert research_credits.research_run_cost() == 10
    monkeypatch.setenv("RESEARCH_RUN_CREDIT_COST", "7")
    assert research_credits.research_run_cost() == 7


def test_refund_returns_the_configured_cost(seeded_user, monkeypatch):
    monkeypatch.setenv("CREDITS_METERING_ENABLED", "1")
    monkeypatch.setenv("RESEARCH_RUN_CREDIT_COST", "5")
    research_credits.authorize_research_run(seeded_user["id"])
    research_credits.refund_research_run(seeded_user["id"])
    import dashboard.backend.users as users_module

    entitlements = users_module.user_store.get_entitlements(seeded_user["id"])
    assert entitlements["credits"] == 20


def test_store_refund_is_unconditional_so_the_route_must_gate_on_charged(seeded_user, monkeypatch):
    """refund_credits adds without checking history — that is fine for the
    backtest call sites (all post-debit) and is why the research route's
    refund is gated on `outcome.charged` rather than trusted to placement."""
    monkeypatch.setenv("CREDITS_METERING_ENABLED", "1")
    research_credits.refund_research_run(seeded_user["id"])  # nothing was debited
    import dashboard.backend.users as users_module

    entitlements = users_module.user_store.get_entitlements(seeded_user["id"])
    assert entitlements["credits"] == 30  # unconditional: the ROUTE owns the gate
