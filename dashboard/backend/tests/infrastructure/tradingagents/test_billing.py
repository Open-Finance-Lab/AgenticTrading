"""ATL Credits metering of the hosted TradingAgents runtime.

Against a real SQLite Credits store, so the holds, debits and releases asserted
here are the ledger's own. The lifecycle is AI Hedge Fund's
(``tests/infrastructure/ai_hedge_fund/test_billing.py``); these cases pin what
is TradingAgents-specific: the ceiling, the gateway lane the hold is recorded
against, and the call counter agreeing with what was billed.
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
from datetime import date, datetime, timezone

import pytest

from dashboard.backend.domain.agents.runtime import AgentRuntimeContext
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.infrastructure.ai_hedge_fund.billing import (
    DecisionUsage,
    HostedDecisionBilling,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.token_cost import (
    credits_micro_for_usd,
    estimate_cost_usd,
)
from dashboard.backend.infrastructure.tradingagents.adapter import (
    TradingAgentsConfigurationError,
    TradingAgentsRuntime,
    TradingAgentsRuntimeError,
)
from dashboard.backend.infrastructure.tradingagents.billing import (
    SOURCE,
    billing_from_handoff,
    decision_calls,
    decision_ceiling_micro,
)

FUNDED_USER = 1
EMPTY_USER = 2
RUN_ID = "agent_20260501_000000_tabill01"
MODEL = "anthropic/claude-haiku-4-5"
CEILING = decision_ceiling_micro(MODEL, ["market"])


def _credits(tmp_path) -> CreditsService:
    path = tmp_path / "credits.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO users (id, email, display_name, password_hash, role, created_at)
            VALUES (?, ?, ?, 'unused', 'user', '2026-08-13T00:00:00+00:00')
            """,
            [
                (FUNDED_USER, "funded@example.com", "Funded"),
                (EMPTY_USER, "empty@example.com", "Empty"),
            ],
        )
    service = CreditsService(store=CreditsStore(path))
    assert service.grant_default_signup_credits(FUNDED_USER)
    return service


def _runtime(credits, runner, *, user_id=FUNDED_USER):
    return TradingAgentsRuntime(
        {"analysts": ["market"]},
        runner=runner,
        environment={"COMMONSTACK_API_KEY": "test-key"},
        billing=HostedDecisionBilling(
            credits=credits,
            user_id=user_id,
            run_id=RUN_ID,
            model_id=MODEL,
            provider_id="commonstack",
            source=SOURCE,
        ),
    )


def _context(day=13):
    return AgentRuntimeContext(
        timestamp=datetime(2026, 3, day, 15, tzinfo=timezone.utc),
        backtest_start_date="2026-03-01",
        symbols=["AAPL"],
        cash=1_000.0,
        total_equity=1_000.0,
        positions={},
        entry_prices={},
        current_prices={"AAPL": 50.0},
        latest_market_date_before_decision=date(2026, 3, day - 1),
    )


def _output(usage):
    return {
        "schema_version": "tradingagents-atl-v1",
        "manifest": {"symbol": "AAPL", "tradingagents_version": "0.3.1"},
        "usage": usage,
        "decision": {
            "analysis_date": "2026-03-12",
            "rating": "Hold",
            "atl_action": "HOLD",
            "status": "valid",
            "attempts": 1,
            "raw_sha256": "a" * 64,
            "raw_excerpt": "FINAL TRANSACTION PROPOSAL: HOLD",
        },
    }


class _Runner:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self, payload, *, timeout_seconds):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _rows(tmp_path):
    """Every hold of the run: (status, provider, reserved, settled, evidence)."""
    with sqlite3.connect(tmp_path / "credits.db") as conn:
        rows = conn.execute(
            """
            SELECT status, provider_id, reserved_micro, settled_micro, evidence_json
            FROM credit_llm_reservations WHERE run_id = ? ORDER BY call_index
            """,
            (RUN_ID,),
        ).fetchall()
    return [
        (row[0], row[1], row[2], row[3], json.loads(row[4]) if row[4] else None)
        for row in rows
    ]


def _usage(input_tokens, output_tokens, *, calls=11, unmetered_calls=0):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calls": calls,
        "unmetered_calls": unmetered_calls,
    }


def test_the_ceiling_budgets_both_attempts_of_every_node():
    # (8 fixed nodes + 4 per analyst) x 2 attempts.
    assert decision_calls(["market"]) == 24
    assert decision_calls(["market", "news"]) == 32
    assert CEILING == credits_micro_for_usd(
        estimate_cost_usd(MODEL, 24 * 16_000, 24 * 2_000)
    )
    # Priced, not free: a $0 ceiling would hold nothing and settle nothing.
    assert decision_ceiling_micro("nvidia/nemotron-3-nano-30b-a3b", ["market"]) > 0


def test_a_decision_settles_at_its_metered_usage_on_the_gateways_lane(tmp_path):
    credits = _credits(tmp_path)
    runtime = _runtime(credits, _Runner(_output(_usage(60_000, 8_000))))

    runtime.decide(_context())

    [(status, provider, reserved, settled, evidence)] = _rows(tmp_path)
    assert status == "settled"
    assert provider == "commonstack"
    assert reserved == CEILING
    assert settled == credits_micro_for_usd(estimate_cost_usd(MODEL, 60_000, 8_000))
    assert evidence["source"] == "tradingagents"
    assert evidence["basis"] == "usage"
    assert runtime.calls == 1


def test_a_bridge_that_stopped_before_any_model_call_is_released(tmp_path):
    credits = _credits(tmp_path)
    runtime = _runtime(
        credits,
        _Runner(TradingAgentsRuntimeError("could not start: exec format error")),
    )

    with pytest.raises(TradingAgentsRuntimeError):
        runtime.decide(_context())

    [(status, _provider, _reserved, settled, _evidence)] = _rows(tmp_path)
    assert status == "released"
    assert not settled
    assert runtime.calls == 0


def test_a_killed_decision_bills_the_ceiling_and_counts_the_call(tmp_path):
    credits = _credits(tmp_path)
    runtime = _runtime(
        credits,
        _Runner(TradingAgentsRuntimeError("timed out after 600 seconds", killed=True)),
    )

    with pytest.raises(TradingAgentsRuntimeError):
        runtime.decide(_context())

    [(status, _provider, _reserved, settled, evidence)] = _rows(tmp_path)
    assert status == "settled"
    assert settled == CEILING
    assert evidence["basis"] == "ceiling"
    assert runtime.calls == 1


def test_a_fatal_failure_after_spend_settles_what_was_spent(tmp_path):
    """The account ran dry mid-debate: the run aborts, but the calls before it
    are billed and counted -- one number, in the ledger and on the run row."""
    credits = _credits(tmp_path)
    runtime = _runtime(
        credits,
        _Runner(
            TradingAgentsConfigurationError(
                "insufficient_quota",
                usage=DecisionUsage(
                    input_tokens=20_000, output_tokens=2_000, calls=4, unmetered_calls=0
                ),
            )
        ),
    )

    with pytest.raises(TradingAgentsConfigurationError):
        runtime.decide(_context())

    [(status, _provider, _reserved, settled, _evidence)] = _rows(tmp_path)
    assert status == "settled"
    assert settled == credits_micro_for_usd(estimate_cost_usd(MODEL, 20_000, 2_000))
    assert runtime.calls == 1


def test_an_empty_balance_stops_the_run_before_upstream_is_called(tmp_path):
    credits = _credits(tmp_path)
    runner = _Runner(_output(_usage(1, 1)))
    runtime = _runtime(credits, runner, user_id=EMPTY_USER)

    with pytest.raises(LLMExecutionError) as excinfo:
        runtime.decide(_context())

    assert excinfo.value.category is ExecutionErrorCategory.INSUFFICIENT_CREDITS
    assert runner.calls == 0
    assert runtime.calls == 0


def test_a_cli_run_without_a_handoff_bills_nobody():
    runner = _Runner(_output(None))
    runtime = TradingAgentsRuntime(
        {"analysts": ["market"]},
        runner=runner,
        environment={"COMMONSTACK_API_KEY": "test-key"},
    )

    runtime.decide(_context())

    assert runner.calls == 1 and runtime.billing is None


def test_the_meter_records_against_the_handoffs_gateway():
    class _Handoff:
        user_id = FUNDED_USER
        run_id = RUN_ID
        model_id = "nvidia/nemotron-3-nano-30b-a3b"
        provider_id = "openrouter"

    billing = billing_from_handoff(credits=object(), handoff=_Handoff())

    assert billing.provider_id == "openrouter"
    assert billing.source == "tradingagents"
    assert billing.model_id == "nvidia/nemotron-3-nano-30b-a3b"


def test_worker_meters_a_tradingagents_handoff(monkeypatch):
    """The child turns the route's handoff into per-decision holds, as it does
    for AI Hedge Fund, and drives no model client of its own."""
    from dashboard.backend.infrastructure.llm.execution.handoff import (
        create_execution_handoff,
    )
    from dashboard.scripts import backtest_hourly_agent as cli

    token = create_execution_handoff(
        user_id=FUNDED_USER,
        run_id=RUN_ID,
        billing_mode="platform_credits",
        provider_id="commonstack",
        model_id=MODEL,
    )
    monkeypatch.setattr(
        sys, "stdin", io.TextIOWrapper(io.BytesIO(token.encode("ascii")))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_hourly_agent",
            "--execution-handoff-stdin",
            "--run-id", RUN_ID,
            "--model", MODEL,
            "--runtime-type", "tradingagents",
            "--decision-source", "llm",
            "--start", "2026-05-01",
            "--end", "2026-05-04",
        ],
    )
    captured = {}

    class _Stop(Exception):
        pass

    def _engine(*_args, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(cli, "HourlyBacktester", _engine)

    with pytest.raises(_Stop):
        cli.main()

    assert captured["execution_client"] is None
    billing = captured["hosted_billing"]
    assert isinstance(billing, HostedDecisionBilling)
    assert (billing.user_id, billing.run_id, billing.model_id) == (
        FUNDED_USER,
        RUN_ID,
        MODEL,
    )
    assert billing.provider_id == "commonstack"
    assert billing.source == "tradingagents"
