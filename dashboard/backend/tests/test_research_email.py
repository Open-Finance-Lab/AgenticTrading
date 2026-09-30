"""Research run finalization and the "report ready" email outbox.

Completion is claimed once (route poll and sweeper can both see it), and the
email is drained by the sweeper with bounded retries instead of being sent
inline on whichever poll happened to notice completion -- where a failed send
was lost for good and a slow Brevo POST blocked the status request.

The send_email fakes are ``async def`` on purpose: a sync fake returning True
is exactly the double that once hid an un-awaited coroutine.
"""

from __future__ import annotations

import sqlite3
import uuid
from types import SimpleNamespace

import pytest

from dashboard.backend.api.routers import research
from dashboard.backend.domain.agents import research_store
from dashboard.backend.infrastructure.email import sender


def _new_run(*, email_me: bool = True, status: str = "running") -> str:
    run_id = f"rr_{uuid.uuid4().hex[:10]}"
    research_store.create_run(
        run_id=run_id,
        user_id=7,
        template_id="tpl",
        service_run_id="svc",
        reservation_id="res_1",
        estimate_micro=1000,
        status=status,
        settings={},
        email_me=email_me,
    )
    return run_id


def _row(run_id: str) -> dict:
    with research_store._connect() as conn:
        return dict(conn.execute(
            "SELECT * FROM research_runs WHERE run_id = ?", (run_id,)
        ).fetchone())


def _sql(statement: str, *params) -> None:
    with research_store._connect() as conn:
        conn.execute(statement, params)


@pytest.fixture
def outbox(monkeypatch):
    """Isolate the outbox to runs this test creates; record every send."""
    sent: list[tuple] = []
    result = {"value": True}

    async def fake_send_email(to, subject, text_body):
        sent.append((to, subject, text_body))
        if isinstance(result["value"], Exception):
            raise result["value"]
        return result["value"]

    monkeypatch.setattr(sender, "send_email", fake_send_email)
    monkeypatch.setattr(
        research.users_module.user_store, "get_user_by_id",
        lambda user_id: {"id": user_id, "email": "user@example.com"},
    )
    monkeypatch.setattr(
        research.marketplace_mod, "get_marketplace_template",
        lambda template_id: {"id": template_id, "name": "Due Diligence"},
    )
    _sql("UPDATE research_runs SET emailed = 1")  # other tests' leftovers
    return sent, result


def _complete(run_id: str) -> None:
    assert research_store.claim_terminal(run_id, "completed")


def test_claim_terminal_admits_exactly_one_finalizer():
    run_id = _new_run()

    assert research_store.claim_terminal(run_id, "completed") is True
    assert research_store.claim_terminal(run_id, "completed") is False
    assert research_store.claim_terminal(run_id, "failed", error="x") is False
    assert _row(run_id)["status"] == "completed"


def test_finalize_settles_and_completes_without_sending_email(monkeypatch, outbox):
    sent, _ = outbox
    settled: list[str] = []
    monkeypatch.setattr(research.research_store, "store_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(
        research.credits_service, "settle_llm_credits",
        lambda reservation_id, **k: settled.append(reservation_id),
    )
    run_id = _new_run()
    run = research_store.get_run(run_id, 7)

    research._finalize_completed_run(run, {"artifacts": {}, "report_markdown": "# ok"})

    assert settled == ["res_1"]
    assert _row(run_id)["status"] == "completed"
    assert sent == []  # the outbox sends, not the poll


def test_settle_failure_leaves_the_run_retryable(monkeypatch):
    monkeypatch.setattr(research.research_store, "store_artifacts", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(research.credits_service, "settle_llm_credits", boom)
    run_id = _new_run()

    with pytest.raises(RuntimeError):
        research._finalize_completed_run(research_store.get_run(run_id, 7), {})

    assert _row(run_id)["status"] == "running"
    assert run_id in {r["run_id"] for r in research_store.list_nonterminal_runs()}


def test_expired_running_run_is_failed_and_releases_reservation(monkeypatch):
    run_id = _new_run()
    _sql(
        "UPDATE research_runs SET created_at = datetime('now', '-31 minutes')"
        " WHERE run_id = ?",
        run_id,
    )
    released: list[tuple[str, str]] = []
    monkeypatch.setattr(
        research.credits_service,
        "release_llm_credits",
        lambda reservation_id, *, reason: released.append((reservation_id, reason)),
    )
    monkeypatch.setattr(research, "_service_base", lambda template: "https://service")
    monkeypatch.setattr(
        research.httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"status": "running"},
        ),
    )

    result = research._poll_run_once(
        research_store.get_run(run_id, 7),
        {"id": "tpl"},
    )

    assert result["status"] == "failed"
    assert "maximum runtime" in result["error"]
    assert released == [("res_1", "research run exceeded maximum runtime")]
    assert _row(run_id)["status"] == "failed"


def test_expired_completed_upstream_without_result_is_failed(monkeypatch):
    run_id = _new_run()
    _sql(
        "UPDATE research_runs SET created_at = datetime('now', '-31 minutes')"
        " WHERE run_id = ?",
        run_id,
    )
    released: list[str] = []
    monkeypatch.setattr(
        research.credits_service,
        "release_llm_credits",
        lambda reservation_id, *, reason: released.append(reservation_id),
    )
    monkeypatch.setattr(research, "_service_base", lambda template: "https://service")

    class Response:
        def __init__(self, payload=None, error=None):
            self.payload = payload
            self.error = error

        def raise_for_status(self):
            if self.error:
                raise self.error

        def json(self):
            return self.payload

    upstream_error = research.httpx.HTTPStatusError(
        "result not ready",
        request=research.httpx.Request("GET", "https://service/runs/svc/result"),
        response=research.httpx.Response(409),
    )
    responses = iter([
        Response({"status": "completed"}),
        Response(error=upstream_error),
    ])
    monkeypatch.setattr(research.httpx, "get", lambda *args, **kwargs: next(responses))

    result = research._poll_run_once(
        research_store.get_run(run_id, 7),
        {"id": "tpl"},
    )

    assert result["status"] == "failed"
    assert released == ["res_1"]
    assert _row(run_id)["status"] == "failed"


def test_expiry_loser_returns_the_database_terminal_state(monkeypatch):
    run_id = _new_run()
    run = research_store.get_run(run_id, 7)
    assert run is not None
    monkeypatch.setattr(
        research.credits_service,
        "release_llm_credits",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(research.research_store, "claim_terminal", lambda *args, **kwargs: False)
    fresh = {**run, "status": "completed", "completed_at": "2026-09-30T07:00:00+00:00"}
    monkeypatch.setattr(research.research_store, "get_run", lambda *args: fresh)

    result = research._expire_run(run)

    assert result is fresh
    assert result["status"] == "completed"


def test_partial_completed_row_is_repaired_by_the_sweeper_queue():
    run_id = _new_run()
    _sql(
        "UPDATE research_runs SET status = 'completed', completed_at = NULL"
        " WHERE run_id = ?",
        run_id,
    )

    queued = {row["run_id"] for row in research_store.list_nonterminal_runs()}

    assert run_id in queued


def test_route_poll_skips_a_run_the_sweeper_is_polling(monkeypatch):
    run_id = _new_run()
    polled: list[str] = []
    monkeypatch.setattr(research, "_template_or_404", lambda template_id: {"id": template_id})
    monkeypatch.setattr(research, "_poll_run_once", lambda run, template: polled.append(run["run_id"]))

    with research._single_flight(run_id) as owned:
        assert owned
        body = research.get_run_status(run_id, {"id": 7, "email": "user@example.com"})

    assert polled == []
    assert body["status"] == "running"


def test_outbox_sends_once_and_marks_emailed(outbox):
    sent, _ = outbox
    run_id = _new_run()
    _complete(run_id)

    research._deliver_pending_emails()
    research._deliver_pending_emails()

    assert [to for to, _, _ in sent] == ["user@example.com"]
    assert "Due Diligence" in sent[0][1]
    assert _row(run_id)["emailed"] == 1


def test_failed_send_is_retried_after_backoff(outbox):
    sent, result = outbox
    result["value"] = False
    run_id = _new_run()
    _complete(run_id)

    research._deliver_pending_emails()
    research._deliver_pending_emails()  # still inside the backoff window

    assert len(sent) == 1
    row = _row(run_id)
    assert (row["emailed"], row["email_attempts"]) == (0, 1)

    result["value"] = True
    _sql("UPDATE research_runs SET email_next_attempt_at = datetime('now', '-1 second')"
         " WHERE run_id = ?", run_id)
    research._deliver_pending_emails()

    assert len(sent) == 2
    assert _row(run_id)["emailed"] == 1


def test_outbox_gives_up_loudly_after_max_attempts(outbox, capsys):
    sent, result = outbox
    result["value"] = False
    run_id = _new_run()
    _complete(run_id)

    for _ in range(research.EMAIL_MAX_ATTEMPTS + 2):
        research._deliver_pending_emails()
        _sql("UPDATE research_runs SET email_next_attempt_at = NULL WHERE run_id = ?", run_id)

    assert len(sent) == research.EMAIL_MAX_ATTEMPTS
    assert f"run {run_id} abandoned" in capsys.readouterr().out


def test_raising_sender_is_contained_and_retried(outbox, capsys):
    sent, result = outbox
    result["value"] = RuntimeError("event loop trouble")
    run_id = _new_run()
    _complete(run_id)

    research._deliver_pending_emails()

    assert len(sent) == 1
    assert _row(run_id)["emailed"] == 0
    assert "research report email raised" in capsys.readouterr().out


def test_unconfigured_email_is_logged_not_silent(outbox, monkeypatch, capsys):
    monkeypatch.undo()  # the real sender, with no Brevo credentials
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.delenv("ACCOUNT_EMAIL_FROM", raising=False)
    monkeypatch.setattr(
        research.users_module.user_store, "get_user_by_id",
        lambda user_id: {"id": user_id, "email": "user@example.com"},
    )
    monkeypatch.setattr(
        research.marketplace_mod, "get_marketplace_template",
        lambda template_id: {"id": template_id, "name": "Due Diligence"},
    )
    run_id = _new_run()
    _complete(run_id)

    research._deliver_pending_emails()

    assert "BREVO_API_KEY" in capsys.readouterr().out
    assert _row(run_id)["emailed"] == 0


def test_outbox_ignores_stale_and_opted_out_runs(outbox):
    sent, _ = outbox
    stale = _new_run()
    _complete(stale)
    _sql("UPDATE research_runs SET completed_at = datetime('now', '-2 days')"
         " WHERE run_id = ?", stale)
    opted_out = _new_run(email_me=False)
    _complete(opted_out)
    unfinished = _new_run()

    research._deliver_pending_emails()

    assert sent == []
    assert {_row(r)["email_attempts"] for r in (stale, opted_out, unfinished)} == {0}


def test_outbox_columns_migrate_onto_an_existing_table(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.execute(
            "CREATE TABLE research_runs (run_id TEXT PRIMARY KEY, user_id INTEGER,"
            " template_id TEXT, service_run_id TEXT, reservation_id TEXT,"
            " status TEXT, settings_json TEXT, email_me INTEGER DEFAULT 0,"
            " emailed INTEGER DEFAULT 0, error TEXT, created_at TIMESTAMP,"
            " completed_at TIMESTAMP)"
        )
    monkeypatch.setattr(research_store, "DB_PATH", legacy)

    research_store._init_schema()

    with sqlite3.connect(legacy) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(research_runs)")}
    assert {"email_attempts", "email_next_attempt_at"} <= columns
