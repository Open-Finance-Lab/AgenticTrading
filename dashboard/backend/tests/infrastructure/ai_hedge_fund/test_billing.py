"""ATL Credits metering of the hosted AI Hedge Fund runtime.

Against a real SQLite Credits store, so the holds, debits and releases asserted
here are the ledger's own, not a fake's idea of them.
"""

from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import sys
import types
from datetime import date, datetime, timezone

import pytest

from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_ANALYSTS,
    AgentRuntimeContext,
)
from dashboard.backend.domain.credits.repository import CreditsStore
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.infrastructure.ai_hedge_fund import bridge
from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PROVIDER,
    AiHedgeFundOutputError,
    AiHedgeFundRuntime,
    AiHedgeFundRuntimeError,
    AiHedgeFundSubprocessRunner,
)
from dashboard.backend.infrastructure.ai_hedge_fund.billing import (
    DecisionUsage,
    HostedDecisionBilling,
    decision_ceiling_micro,
    parse_usage,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.token_cost import (
    credits_micro_for_usd,
    estimate_cost_usd,
)

FUNDED_USER = 1
EMPTY_USER = 2
RUN_ID = "agent_20260501_000000_ahfbill1"
SYMBOLS = ["AAPL", "MSFT", "IBM", "JPM", "DIS"]
CEILING = decision_ceiling_micro(DEFAULT_MODEL_NAME, ["ben_graham"], len(SYMBOLS))


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
    return AiHedgeFundRuntime(
        {"analysts": ["ben_graham"]},
        runner=runner,
        environment={},
        billing=HostedDecisionBilling(
            credits=credits,
            user_id=user_id,
            run_id=RUN_ID,
            model_id=DEFAULT_MODEL_NAME,
        ),
    )


def _context(day=1):
    return AgentRuntimeContext(
        timestamp=datetime(2026, 5, day, 14, tzinfo=timezone.utc),
        backtest_start_date="2026-05-01",
        symbols=list(SYMBOLS),
        cash=1_000.0,
        total_equity=1_000.0,
        positions={},
        entry_prices={},
        current_prices={symbol: 10.0 for symbol in SYMBOLS},
        latest_market_date_before_decision=date(2026, 4, day + 20),
    )


class _Runner:
    """Answers each decision with the next scripted output or exception."""

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
    """Every hold of the run: (call_index, status, reserved, settled, evidence)."""
    with sqlite3.connect(tmp_path / "credits.db") as conn:
        rows = conn.execute(
            """
            SELECT call_index, status, reserved_micro, settled_micro, evidence_json
            FROM credit_llm_reservations WHERE run_id = ? ORDER BY call_index
            """,
            (RUN_ID,),
        ).fetchall()
    return [
        (
            row[0],
            row[1],
            row[2],
            row[3],
            json.loads(row[4]) if row[4] else None,
        )
        for row in rows
    ]


def _usage(input_tokens, output_tokens, *, calls=1, unmetered_calls=0):
    """The bridge's usage report, as it lands in its output file."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "calls": calls,
        "unmetered_calls": unmetered_calls,
    }


def _usage_micro(input_tokens, output_tokens):
    return credits_micro_for_usd(
        estimate_cost_usd(DEFAULT_MODEL_NAME, input_tokens, output_tokens)
    )


def test_ceiling_counts_every_analyst_when_none_are_configured():
    assert decision_ceiling_micro(DEFAULT_MODEL_NAME, None, 5) == decision_ceiling_micro(
        DEFAULT_MODEL_NAME, list(AI_HEDGE_FUND_ANALYSTS), 5
    )
    # Priced, not free: a $0 ceiling would hold nothing and settle nothing.
    assert CEILING > 0


def test_each_decision_settles_at_its_reported_usage(tmp_path):
    credits = _credits(tmp_path)
    runner = _Runner(
        {"decisions": {}, "usage": _usage(12_000, 3_000, calls=6)},
        {"decisions": {}, "usage": _usage(4_000, 1_000, calls=2)},
    )
    runtime = _runtime(credits, runner)

    runtime.decide(_context(1))
    runtime.decide(_context(2))

    rows = _rows(tmp_path)
    assert [(index, status) for index, status, *_ in rows] == [
        (0, "settled"),
        (1, "settled"),
    ]
    assert [row[2] for row in rows] == [CEILING, CEILING]
    assert [row[3] for row in rows] == [
        _usage_micro(12_000, 3_000),
        _usage_micro(4_000, 1_000),
    ]
    assert rows[0][4]["basis"] == "usage"
    assert rows[0][4]["source"] == "ai_hedge_fund"
    spent, calls = credits.sum_run_llm_spend(FUNDED_USER, RUN_ID)
    assert (spent, calls) == (rows[0][3] + rows[1][3], 2)


def test_unreported_usage_on_a_completed_decision_bills_the_ceiling_loudly(
    tmp_path, capsys
):
    credits = _credits(tmp_path)
    runtime = _runtime(credits, _Runner({"decisions": {}, "usage": None}))

    runtime.decide(_context())

    [(_, status, _, settled, evidence)] = _rows(tmp_path)
    assert (status, settled, evidence["basis"]) == ("settled", CEILING, "ceiling")
    assert "ERROR: ai_hedge_fund.usage_unreported" in capsys.readouterr().out


def test_calls_that_reported_no_usage_bill_the_ceiling_loudly(tmp_path, capsys):
    """A meter that counts calls but no tokens is broken, never free.

    Zero tokens alone cannot say whether the model was called; the call count
    can, so a provider (or a langchain change) that drops usage is billed the
    ceiling and named in an ERROR the parent relays to the service log.
    """
    credits = _credits(tmp_path)
    runtime = _runtime(
        credits,
        _Runner({"decisions": {}, "usage": _usage(0, 0, calls=3, unmetered_calls=3)}),
    )

    runtime.decide(_context())

    [(_, status, _, settled, evidence)] = _rows(tmp_path)
    assert (status, settled, evidence["basis"]) == ("settled", CEILING, "ceiling")
    assert (evidence["calls"], evidence["unmetered_calls"]) == (3, 3)
    assert "ERROR: ai_hedge_fund.usage_unmetered calls=3 unmetered_calls=3" in (
        capsys.readouterr().out
    )


def test_a_decision_that_made_no_model_call_is_released_quietly(tmp_path, capsys):
    # Every analyst rule-based and every ticker pre-filled: nothing was spent.
    credits = _credits(tmp_path)
    runtime = _runtime(credits, _Runner({"decisions": {}, "usage": _usage(0, 0, calls=0)}))

    runtime.decide(_context())

    [(_, status, _, settled, _)] = _rows(tmp_path)
    assert (status, settled) == ("released", 0)
    assert "ai_hedge_fund." not in capsys.readouterr().out


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"input_tokens": 1, "output_tokens": 1},  # no call counts: an old bridge
        _usage(1, 1, calls=1, unmetered_calls=2),  # more unmetered than calls
        _usage(1, -1),
        _usage(True, 1),
    ],
)
def test_a_malformed_usage_report_is_no_report(value):
    assert parse_usage(value) is None


@pytest.mark.parametrize(
    "interrupt",
    [
        # A cancel or the parent's run timeout: SIGTERM, which the child's
        # handler turns into SystemExit inside subprocess.run.
        SystemExit(143),
        KeyboardInterrupt(),
        # Anything else past the runner's own errors, e.g. decoding output.
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ],
)
def test_an_interrupted_decision_bills_the_ceiling_instead_of_leaving_the_hold(
    tmp_path, interrupt
):
    """Left open, the hold would reach `finalize_run` and be released free."""
    credits = _credits(tmp_path)
    runtime = _runtime(credits, _Runner(interrupt))

    with pytest.raises(type(interrupt)):
        runtime.decide(_context())

    [(_, status, _, settled, evidence)] = _rows(tmp_path)
    assert (status, settled) == ("settled", CEILING)
    assert (evidence["outcome"], evidence["basis"]) == ("killed", "ceiling")


def test_usage_above_the_ceiling_is_capped_rather_than_recorded_as_overage(
    tmp_path, capsys
):
    credits = _credits(tmp_path)
    runtime = _runtime(
        credits,
        _Runner(
            {
                "decisions": {},
                "usage": _usage(50_000_000, 0),
            }
        ),
    )

    runtime.decide(_context())

    [(_, status, _, settled, evidence)] = _rows(tmp_path)
    assert (status, settled) == ("settled", CEILING)
    assert evidence["uncapped_micro"] > CEILING
    # No overage, so the account is not restricted over ATL's own estimate.
    assert credits.get_balance(FUNDED_USER).account_status == "active"
    assert "WARNING: ai_hedge_fund.usage_above_ceiling" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error, status, settled",
    [
        # Killed mid-run (timeout, signal, crash past the started marker): its
        # model calls may have run, so it fails closed at the ceiling.
        (AiHedgeFundRuntimeError("timed out", killed=True), "settled", CEILING),
        # Failed after reporting what it had spent.
        (
            AiHedgeFundRuntimeError(
                "exit 1", usage=DecisionUsage(10_000, 2_000, 4, 0)
            ),
            "settled",
            _usage_micro(10_000, 2_000),
        ),
        # Failed after calls whose usage went unreported: still the ceiling.
        (
            AiHedgeFundRuntimeError("exit 1", usage=DecisionUsage(0, 0, 2, 2)),
            "settled",
            CEILING,
        ),
        # Failed before its usage scope opened: no model call was made.
        (AiHedgeFundRuntimeError("exit 1"), "released", 0),
    ],
)
def test_a_failed_decision_is_billed_by_what_its_failure_shows(
    tmp_path, error, status, settled
):
    credits = _credits(tmp_path)
    runtime = _runtime(credits, _Runner(error))

    with pytest.raises(AiHedgeFundRuntimeError):
        runtime.decide(_context())

    [(_, row_status, _, row_settled, _)] = _rows(tmp_path)
    assert (row_status, row_settled) == (status, settled)


def test_an_empty_balance_stops_the_run_before_upstream_is_called(tmp_path):
    credits = _credits(tmp_path)
    runner = _Runner({"decisions": {}})
    runtime = _runtime(credits, runner, user_id=EMPTY_USER)

    with pytest.raises(LLMExecutionError) as excinfo:
        runtime.decide(_context())

    # Not an AgentRuntimeError, so the engine does not hold the step and carry
    # on: it reaches the child's `llm.run_failed` line like a pipeline run's.
    assert excinfo.value.category is ExecutionErrorCategory.INSUFFICIENT_CREDITS
    assert runner.calls == 0
    assert _rows(tmp_path) == []


def _runner_exiting(monkeypatch, tmp_path, returncode, written):
    """A runner whose bridge leaves ``written`` (text, or None for no file)."""
    python = tmp_path / "python"
    python.write_text("")
    python.chmod(0o755)

    def fake_run(command, **kwargs):
        if written is not None:
            output = command[command.index("--output") + 1]
            with open(output, "w", encoding="utf-8") as handle:
                handle.write(written)
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return AiHedgeFundSubprocessRunner({"AI_HEDGE_FUND_PYTHON": str(python)})


@pytest.mark.parametrize(
    "returncode, written, usage, killed",
    [
        # The bridge's handler reported what it spent before exiting.
        (
            1,
            json.dumps({"started": True, "usage": _usage(7, 3)}),
            DecisionUsage(7, 3, 1, 0),
            False,
        ),
        # An OOM SIGKILL after the marker: no report, spend unknown.
        (-9, json.dumps({"started": True}), None, True),
        # Killed mid-write: a truncated file is still proof it started.
        (-9, '{"started": tr', None, True),
        # No file: died before upstream could call the model.
        (1, None, None, False),
    ],
)
def test_a_failed_exit_fails_closed_once_the_bridge_has_started(
    monkeypatch, tmp_path, returncode, written, usage, killed
):
    runner = _runner_exiting(monkeypatch, tmp_path, returncode, written)

    with pytest.raises(AiHedgeFundRuntimeError) as excinfo:
        runner.run({"tickers": ["AAPL"]}, timeout_seconds=5)

    assert (excinfo.value.usage, excinfo.value.killed) == (usage, killed)


@pytest.mark.parametrize("written", ["not json", "[]"])
def test_an_unreadable_clean_exit_bills_the_ceiling(monkeypatch, tmp_path, written):
    # A clean exit ran upstream to the end; refusing its output must not make
    # the decision free.
    runner = _runner_exiting(monkeypatch, tmp_path, 0, written)

    with pytest.raises(AiHedgeFundOutputError) as excinfo:
        runner.run({"tickers": ["AAPL"]}, timeout_seconds=5)

    assert excinfo.value.killed is True


def _bridge_args(monkeypatch, tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            {
                "tickers": ["AAPL"],
                "start_date": "2026-02-03",
                "end_date": "2026-05-04",
                "portfolio": {"cash": 1000},
                "model_name": DEFAULT_MODEL_NAME,
                "model_provider": DEFAULT_MODEL_PROVIDER,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["bridge.py", "--input", str(input_path), "--output", str(output_path)],
    )
    monkeypatch.setattr(bridge, "_disable_dotenv_loading", lambda: None)
    return output_path


def _fake_upstream(monkeypatch, run_hedge_fund):
    """Install a fake upstream and fake langchain-core callback hooks.

    The fakes stand in for the two langchain-core names the meter uses: a
    ``BaseCallbackHandler`` base, and ``register_configure_hook``, which here
    records the meter's context var. The returned function reads that var, so
    a fake model call reaches the active meter the way a real call's callback
    manager does. The real pair, worker threads included, was exercised
    against upstream's pinned langchain stack while writing this.
    """
    fake_main = types.ModuleType("src.main")
    fake_main.run_hedge_fund = run_hedge_fund
    fake_src = types.ModuleType("src")
    fake_src.main = fake_main
    monkeypatch.setitem(sys.modules, "src", fake_src)
    monkeypatch.setitem(sys.modules, "src.main", fake_main)

    hooks = []
    callbacks = types.ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = type("BaseCallbackHandler", (), {})
    context = types.ModuleType("langchain_core.tracers.context")
    context.register_configure_hook = (
        lambda var, inheritable: hooks.append((var, inheritable))
    )
    tracers = types.ModuleType("langchain_core.tracers")
    tracers.context = context
    core = types.ModuleType("langchain_core")
    core.callbacks = callbacks
    core.tracers = tracers
    for name, module in (
        ("langchain_core", core),
        ("langchain_core.callbacks", callbacks),
        ("langchain_core.tracers", tracers),
        ("langchain_core.tracers.context", context),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    def active_meter():
        var, inheritable = hooks[-1]
        # Inheritable, or the analysts' worker-thread calls go unseen.
        assert inheritable is True
        return var.get()

    return active_meter


def _model_call(meter, usage_metadata):
    """One finished chat-model call as langchain hands it to ``on_llm_end``."""
    message = types.SimpleNamespace(usage_metadata=usage_metadata)
    meter.on_llm_end(
        types.SimpleNamespace(generations=[[types.SimpleNamespace(message=message)]])
    )


def test_bridge_reports_summed_usage_with_the_decisions(monkeypatch, tmp_path):
    output_path = _bridge_args(monkeypatch, tmp_path)

    def run_hedge_fund(**_kwargs):
        meter = active_meter()
        _model_call(meter, {"input_tokens": 900, "output_tokens": 90})
        _model_call(meter, {"input_tokens": 100, "output_tokens": 10})
        return {"decisions": {}, "analyst_signals": {}}

    active_meter = _fake_upstream(monkeypatch, run_hedge_fund)

    bridge.main()

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["usage"] == _usage(1000, 100, calls=2)
    assert output["decisions"] == {}


@pytest.mark.parametrize(
    "usage_metadata",
    [None, {}, {"input_tokens": 5}, {"input_tokens": 5, "output_tokens": "1"}],
)
def test_bridge_counts_a_call_without_usage_as_unmetered(
    monkeypatch, tmp_path, usage_metadata
):
    output_path = _bridge_args(monkeypatch, tmp_path)

    def run_hedge_fund(**_kwargs):
        meter = active_meter()
        _model_call(meter, {"input_tokens": 900, "output_tokens": 90})
        _model_call(meter, usage_metadata)
        return {"decisions": {}}

    active_meter = _fake_upstream(monkeypatch, run_hedge_fund)

    bridge.main()

    assert json.loads(output_path.read_text(encoding="utf-8"))["usage"] == _usage(
        900, 90, calls=2, unmetered_calls=1
    )


def test_bridge_reports_zero_calls_when_upstream_made_none(monkeypatch, tmp_path):
    output_path = _bridge_args(monkeypatch, tmp_path)
    _fake_upstream(monkeypatch, lambda **_kwargs: {"decisions": {}})

    bridge.main()

    assert json.loads(output_path.read_text(encoding="utf-8"))["usage"] == _usage(
        0, 0, calls=0
    )


def test_bridge_reports_usage_spent_before_a_failure(monkeypatch, tmp_path):
    output_path = _bridge_args(monkeypatch, tmp_path)

    def run_hedge_fund(**_kwargs):
        _model_call(active_meter(), {"input_tokens": 50, "output_tokens": 5})
        raise RuntimeError("upstream exploded")

    active_meter = _fake_upstream(monkeypatch, run_hedge_fund)

    with pytest.raises(RuntimeError):
        bridge.main()

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "started": True,
        "usage": _usage(50, 5),
    }


def test_bridge_marks_itself_started_before_upstream_can_call_the_model(
    monkeypatch, tmp_path
):
    output_path = _bridge_args(monkeypatch, tmp_path)
    seen = {}

    def run_hedge_fund(**_kwargs):
        seen["at_call"] = json.loads(output_path.read_text(encoding="utf-8"))
        return {"decisions": {}}

    _fake_upstream(monkeypatch, run_hedge_fund)

    bridge.main()

    assert seen["at_call"] == {"started": True}


def test_bridge_without_the_callback_hook_api_reports_none(monkeypatch, tmp_path):
    output_path = _bridge_args(monkeypatch, tmp_path)
    _fake_upstream(monkeypatch, lambda **_kwargs: {"decisions": {}})
    # An older langchain-core: the import fails, so ATL bills the ceiling.
    monkeypatch.setitem(sys.modules, "langchain_core.tracers.context", None)

    bridge.main()

    assert json.loads(output_path.read_text(encoding="utf-8"))["usage"] is None


def test_worker_meters_a_hosted_handoff_instead_of_driving_a_model_client(
    monkeypatch,
):
    """The child turns the route's handoff into per-decision holds.

    No model client: the engine would read its `sampling_record` and stamp a
    pinned-sampling policy on a run whose model it never called.
    """
    from dashboard.backend.infrastructure.llm.execution.handoff import (
        create_execution_handoff,
    )
    from dashboard.scripts import backtest_hourly_agent as cli

    token = create_execution_handoff(
        user_id=FUNDED_USER,
        run_id=RUN_ID,
        billing_mode="platform_credits",
        provider_id="openrouter",
        model_id=DEFAULT_MODEL_NAME,
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
            "--model", DEFAULT_MODEL_NAME,
            "--runtime-type", "ai_hedge_fund",
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
        DEFAULT_MODEL_NAME,
    )
