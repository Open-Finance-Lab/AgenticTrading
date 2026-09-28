"""The research "email me" notification is actually sent.

``send_email`` is a coroutine and ``_finalize_completed_run`` is sync. Called
bare, it returned an un-run coroutine -- truthy -- so the run was marked
emailed while nothing was sent. The fakes here are ``async def`` on purpose: a
sync fake returning True is exactly the double that hid the bug.
"""

from __future__ import annotations

import pytest

from dashboard.backend.api.routers import research
from dashboard.backend.infrastructure.email import sender


@pytest.fixture
def finalize(monkeypatch):
    marked: list[str] = []
    monkeypatch.setattr(research.research_store, "store_artifacts", lambda *a, **k: None)
    monkeypatch.setattr(research.research_store, "update_run_status", lambda *a, **k: None)
    monkeypatch.setattr(research.research_store, "mark_emailed", marked.append)
    monkeypatch.setattr(research.credits_service, "settle_llm_credits", lambda *a, **k: None)
    monkeypatch.setattr(sender, "email_configured", lambda: True)

    def run(send_result: bool) -> tuple[list[tuple], list[str]]:
        sent: list[tuple] = []

        async def fake_send_email(to, subject, text_body):
            sent.append((to, subject, text_body))
            return send_result

        monkeypatch.setattr(sender, "send_email", fake_send_email)
        research._finalize_completed_run(
            {
                "run_id": "rr_1",
                "template_id": "tpl",
                "reservation_id": "res_1",
                "estimate_micro": 1000,
                "email_me": True,
                "emailed": False,
            },
            {"name": "Due Diligence"},
            {"artifacts": {}, "report_markdown": "# ok"},
            "user@example.com",
        )
        return sent, marked

    return run


def test_completed_run_sends_the_email_and_marks_it(finalize):
    sent, marked = finalize(True)

    assert [to for to, _, _ in sent] == ["user@example.com"]
    assert "Due Diligence" in sent[0][1]
    assert marked == ["rr_1"]


def test_failed_send_is_not_marked_emailed(finalize):
    sent, marked = finalize(False)

    assert len(sent) == 1
    assert marked == []
