"""Tests for the SQLite-backed strategy store (H1).

The key property is persistence across processes/restarts: a strategy written by
one store instance must be readable by a fresh instance pointed at the same DB
file — the whole point of moving off the ephemeral JSON file that Render wiped on
every deploy.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import app
from dashboard.backend.domain.strategies.repository import StrategyStore, secrets


def _store(tmp_path):
    return StrategyStore(db_path=tmp_path / "strategies_test.db")


def test_create_and_get_roundtrip(tmp_path):
    store = _store(tmp_path)
    rec = store.create(prompt="  buy low sell high  ", description=" my idea ", source="web", owner="u1")
    assert rec["code"]
    assert rec["prompt"] == "buy low sell high"  # trimmed
    assert rec["description"] == "my idea"
    assert rec["source"] == "web"
    assert rec["owner"] == "u1"
    assert rec["last_run_id"] is None and rec["last_run_at"] is None

    fetched = store.get(rec["code"])
    assert fetched == rec


def test_default_store_lands_on_persistent_db_path():
    """The heart of H1: the default store writes to DATABASE_PATH (the persistent
    disk on Render), not the ephemeral repo storage dir the old JSON file used."""
    from dashboard.backend.database import DB_PATH
    from dashboard.backend.domain.strategies.repository import StrategyStore, strategy_store

    assert StrategyStore().db_path == DB_PATH
    assert strategy_store.db_path == DB_PATH


def test_persists_across_store_instances(tmp_path):
    """A fresh store on the same DB file sees strategies written earlier — this is
    what the old ephemeral JSON file failed to guarantee on Render."""
    db = tmp_path / "strategies_test.db"
    rec = StrategyStore(db_path=db).create(prompt="momentum on breakouts")
    reopened = StrategyStore(db_path=db).get(rec["code"])
    assert reopened is not None
    assert reopened["prompt"] == "momentum on breakouts"


def test_empty_prompt_rejected(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create(prompt="   ")


def test_get_missing_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.get("nope") is None
    assert store.get("") is None


def test_set_last_run(tmp_path):
    store = _store(tmp_path)
    rec = store.create(prompt="mean reversion")
    updated = store.set_last_run(rec["code"], "ext_123")
    assert updated is not None
    assert updated["last_run_id"] == "ext_123"
    assert updated["last_run_at"] is not None
    # missing code → None (no row updated)
    assert store.set_last_run("does-not-exist", "ext_x") is None


def test_codes_are_unique(tmp_path):
    store = _store(tmp_path)
    codes = {store.create(prompt=f"strategy {i}")["code"] for i in range(25)}
    assert len(codes) == 25


# ----------------------------------------------------------------------
# Share-code collision retry.
#
# test_codes_are_unique above can't reach this: random 8-hex codes never
# collide by chance, so the retry loop has been dead code to the suite until
# now. These force the collision instead of mocking the insert. The Postgres
# twin mirrors them exactly (test_strategy_store_postgres.py) -- its loop is
# structurally different (ON CONFLICT DO NOTHING + rowcount rather than
# catching IntegrityError, since a UniqueViolation would abort the whole
# transaction) and must behave identically.
# ----------------------------------------------------------------------


def test_create_retries_past_a_code_collision(tmp_path, monkeypatch):
    store = _store(tmp_path)
    first = store.create(prompt="first strategy")

    codes = iter([first["code"], "fresh456"])
    # NB: secrets IS the stdlib secrets module, so this
    # patches token_hex process-wide rather than just this module's view of
    # it. Harmless -- monkeypatch restores it and nothing else in this test
    # calls token_hex -- but don't read it as module-scoped.
    monkeypatch.setattr(
        secrets, "token_hex", lambda nbytes: next(codes)
    )

    second = store.create(prompt="second strategy")
    assert second["code"] == "fresh456"
    assert store.get(first["code"])["prompt"] == "first strategy"
    assert store.get("fresh456")["prompt"] == "second strategy"


def test_create_widens_code_space_after_20_collisions(tmp_path, monkeypatch):
    store = _store(tmp_path)
    first = store.create(prompt="first strategy")

    calls = {"n": 0}

    def fake_token_hex(nbytes):
        calls["n"] += 1
        if calls["n"] <= 20:
            return first["code"]  # every narrow attempt collides
        return "w" * 16  # the widened attempt succeeds

    monkeypatch.setattr(secrets, "token_hex", fake_token_hex)

    second = store.create(prompt="second strategy")
    assert second["code"] == "w" * 16
    # 20 narrow attempts (token_hex(_CODE_LENGTH // 2) -> 8 chars) + 1 widened
    # (token_hex(_CODE_LENGTH) -> 16 chars). See repository.py:112-124.
    assert calls["n"] == 21


def test_create_raises_when_even_widened_code_collides(tmp_path, monkeypatch):
    store = _store(tmp_path)
    first = store.create(prompt="first strategy")

    monkeypatch.setattr(
        secrets, "token_hex", lambda nbytes: first["code"]
    )

    with pytest.raises(RuntimeError):
        store.create(prompt="second strategy")


# ----------------------------------------------------------------------
# Router integration (uses the module singleton -> conftest temp DB)
# ----------------------------------------------------------------------


def test_strategies_router_create_and_fetch():
    client = TestClient(app)
    prompt = f"router strategy {uuid.uuid4()}"
    created = client.post("/api/strategies", json={"prompt": prompt, "source": "web"})
    assert created.status_code == 200, created.text
    body = created.json()
    code = body["code"]
    assert body["prompt"] == prompt
    assert "share_url" in body and code in body["share_url"]

    fetched = client.get(f"/api/strategies/{code}")
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["prompt"] == prompt

    missing = client.get("/api/strategies/does-not-exist")
    assert missing.status_code == 404

    empty = client.post("/api/strategies", json={"prompt": "   "})
    assert empty.status_code in (400, 422)  # blank prompt rejected
