"""The Live Leaderboard's monthly spend ledger and $50/month hard stop.

Every billable live attempt is priced into one ``agent_runs`` row per
(month, entry) in the ``leaderboard-live-spend`` session, whatever ended it:
success, a crash, an H6 refusal, a forced replay or the budget itself.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx
import psycopg_pool
import pytest
from fastapi.testclient import TestClient

from dashboard.backend.api.v2.leaderboard import build_leaderboard
from dashboard.backend.app import app
from dashboard.backend.database import db
from dashboard.backend.domain.leaderboard import live_budget
import dashboard.backend.domain.leaderboard.live as live
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401
    SEED,
    _ET,
    _ScriptedAgent,
    _deepseek,
    _insert_checkpoint,
    _insert_live_run,
    _live_aug27,
    _month_rows,
    _one_bar_frame,
    _pt,
    # Autouse: these cases are dated August 2026, before any entry's
    # live_from month, so without it the roster would be DeepSeek alone.
    _roster_without_join_dates,
    _use_agent,
)

_AUG = "2026-08-01"
_FREEZE = "2026-08-27"
_DEEPSEEK_MODEL = "deepseek/deepseek-v4-pro"  # $0.435 per 1M input tokens
_PROVIDER_REQUEST = httpx.Request("POST", "https://api.commonstack.ai/v1/messages")


def _clear_live_rows():
    for session in (live.LIVE_SESSION_ID, live_budget.LIVE_SPEND_SESSION_ID):
        for run in db.get_runs_by_session(session) or []:
            db.delete_run(run["run_id"])


@pytest.fixture(autouse=True)
def _isolated_ledger(monkeypatch, tmp_path):
    """The test DB is session-scoped: start and end every test with no board
    rows and no ledger rows, the cap at its default, and no warning memo."""
    monkeypatch.setattr(live, "_LIVE_REFRESH_STATE_PATH", tmp_path / "live_state.json")
    monkeypatch.delenv(live_budget.BUDGET_ENV, raising=False)
    monkeypatch.setattr(live_budget, "_warned_values", set())
    _clear_live_rows()
    yield
    _clear_live_rows()


def _impl(input_tokens=0, output_tokens=0, llm_calls=0, model_id=_DEEPSEEK_MODEL):
    return SimpleNamespace(
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        llm_calls=llm_calls,
    )


def _entry(entry_id="deepseek_v4_pro"):
    return {"id": entry_id, "name": f"{entry_id} name", "model_id": _DEEPSEEK_MODEL}


def _attempt(entry_id="deepseek_v4_pro", checkpoint_rows=None, freeze_end=_FREEZE, month=_AUG):
    return live_budget.LiveAttempt(
        _entry(entry_id),
        month_start=month,
        freeze_end=freeze_end,
        checkpoint_rows=checkpoint_rows or {},
    )


def _insert_ledger(entry_id, usd, *, month=_AUG, last_attempt=None):
    metadata = {"ledger": live_budget.LEDGER_NAME}
    if last_attempt is not None:
        metadata["last_attempt"] = last_attempt
    db.insert_run(
        run_id=live_budget.spend_run_id(month, entry_id),
        session_id=live_budget.LIVE_SPEND_SESSION_ID,
        agent_name=entry_id,
        mode=live_budget.LIVE_SPEND_MODE,
        start_date=month,
        end_date=month,
        initial_equity=0.0,
        llm_model=entry_id,
        est_cost_usd=usd,
        metadata=metadata,
    )


def _ledger(entry_id="deepseek_v4_pro", month=_AUG):
    return db.get_run(live_budget.spend_run_id(month, entry_id))


# --- the cap ---------------------------------------------------------------


def test_budget_defaults_to_fifty_dollars_when_unset(capsys):
    assert live_budget.monthly_budget_usd() == 50.0
    assert capsys.readouterr().out == ""


def test_a_blank_budget_is_the_default_without_a_warning(monkeypatch, capsys):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "   ")
    assert live_budget.monthly_budget_usd() == 50.0
    assert capsys.readouterr().out == ""


def test_the_budget_is_read_on_every_call_not_at_import(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "12.5")
    assert live_budget.monthly_budget_usd() == 12.5
    monkeypatch.setenv(live_budget.BUDGET_ENV, " 7 ")
    assert live_budget.monthly_budget_usd() == 7.0


@pytest.mark.parametrize("raw", ["abc", "nan", "inf", "-inf", "-1", "10000.01", "1e9"])
def test_a_junk_budget_warns_once_and_falls_back(monkeypatch, capsys, raw):
    monkeypatch.setenv(live_budget.BUDGET_ENV, raw)
    assert live_budget.monthly_budget_usd() == 50.0
    assert live_budget.monthly_budget_usd() == 50.0
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert lines == [
        f"WARNING: LEADERBOARD_MONTHLY_BUDGET_USD={raw!r} is not a dollar amount "
        "between 0 and 10000; using the default 50.00"
    ]


@pytest.mark.parametrize("raw", ["0", "0.0", "-0"])
def test_zero_is_a_cap_that_blocks_everything_not_off(monkeypatch, raw):
    monkeypatch.setenv(live_budget.BUDGET_ENV, raw)
    assert live_budget.monthly_budget_usd() == 0.0
    with pytest.raises(live_budget.LiveBudgetExceeded, match=r"\$0.00 of its \$0.00"):
        _attempt().check_can_start()


def test_the_conftest_strips_the_budget_for_the_suite():
    conftest = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in conftest.splitlines() if not line.lstrip().startswith("#")
    )
    assert 'os.environ.pop("LEADERBOARD_MONTHLY_BUDGET_USD", None)' in code


# --- outcome classes ---------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (None, ("ok", None)),
        (live_budget.LiveBudgetExceeded("spent"), ("budget_halted", "budget")),
        (lb_service.LeaderboardFallbackError("h6"), ("failed", "model_coverage")),
        (live.LiveSessionGapError("gap"), ("failed", "market_data_gap")),
        (TimeoutError("slow"), ("failed", "timeout")),
        (ConnectionResetError("reset"), ("failed", "network")),
        # Client-library classes that subclass neither builtin, matched on the
        # name parts "Timeout" and then "Connect". APITimeoutError subclasses
        # APIConnectionError, so it also pins that order.
        (anthropic.APITimeoutError(request=_PROVIDER_REQUEST), ("failed", "timeout")),
        (anthropic.APIConnectionError(request=_PROVIDER_REQUEST), ("failed", "network")),
        (httpx.ReadTimeout("read timed out"), ("failed", "timeout")),
        (httpx.ConnectError("connection refused"), ("failed", "network")),
        (psycopg_pool.PoolTimeout("no connection in 30s"), ("failed", "timeout")),
        (RuntimeError("worker killed"), ("failed", "error")),
        (ValueError("bad config"), ("failed", "error")),
        (KeyboardInterrupt(), ("failed", "interrupted")),
    ],
)
def test_an_attempt_is_classified_without_its_message(exc, expected):
    assert live_budget.classify_attempt(exc) == expected


def test_last_attempt_is_whitelisted_to_three_fields():
    clean = live_budget.clean_last_attempt
    assert clean({
        "on": "2026-08-27", "outcome": "failed", "kind": "network",
        "error": "sk-live-SECRET", "spent_usd": 49.0,
    }) == {"on": "2026-08-27", "outcome": "failed", "kind": "network"}
    assert clean({"on": "2026-08-27", "outcome": "ok", "kind": "network"})["kind"] is None
    assert clean({"on": "2026-08-27", "outcome": "failed", "kind": "boom"})["kind"] == "error"
    assert clean({"on": "2026-08-27", "outcome": "exploded"}) is None
    assert clean({"on": "Aug 27", "outcome": "ok"}) is None
    assert clean(["2026-08-27", "ok"]) is None
    assert clean(None) is None


# --- the ledger row ------------------------------------------------------------


def test_spend_run_id_is_one_row_per_month_and_entry():
    assert live_budget.spend_run_id(_AUG, "deepseek_v4_pro") == "live_spend_202608_deepseek_v4_pro"


def test_attempts_accumulate_into_one_row_and_carry_no_error_text():
    first = _attempt()
    first.strategy_impl = _impl(input_tokens=1_000_000, llm_calls=7)
    first.record_session()
    row = _ledger()
    assert row["session_id"] == "leaderboard-live-spend"
    assert row["mode"] == "leaderboard_spend"
    assert row["start_date"] == _AUG and row["end_date"] == _FREEZE
    assert row["est_cost_usd"] == pytest.approx(0.435)
    assert row["input_tokens"] == 1_000_000 and row["llm_calls"] == 7
    # A session close is not the end of the attempt: no outcome yet.
    assert "last_attempt" not in row["metadata"]
    first.finish(None)
    assert _ledger()["metadata"]["last_attempt"] == {
        "on": _FREEZE, "outcome": "ok", "kind": None,
    }

    second = _attempt()
    second.strategy_impl = _impl(input_tokens=1_000_000, llm_calls=3)
    second.finish(RuntimeError("provider said sk-live-SECRET"))
    row = _ledger()
    assert row["est_cost_usd"] == pytest.approx(0.87)
    assert row["input_tokens"] == 2_000_000 and row["llm_calls"] == 10
    assert row["metadata"]["attempts"] == 2
    assert row["metadata"]["price_source"] == lb_service.token_cost.PRICING_SOURCE_VERSION
    assert row["metadata"]["last_attempt"] == {
        "on": _FREEZE, "outcome": "failed", "kind": "error",
    }
    assert "SECRET" not in json.dumps(row, default=str)


def test_a_session_close_keeps_the_previous_outcome_until_the_attempt_ends():
    _insert_ledger("deepseek_v4_pro", 1.0, last_attempt={
        "on": "2026-08-26", "outcome": "ok", "kind": None,
    })
    attempt = _attempt()
    attempt.strategy_impl = _impl(input_tokens=1_000_000)
    attempt.record_session()
    assert _ledger()["metadata"]["last_attempt"]["on"] == "2026-08-26"
    assert _ledger()["est_cost_usd"] == pytest.approx(1.435)


def test_the_first_attempt_seeds_from_a_larger_checkpoint():
    checkpoint = {
        "deepseek_v4_pro": {
            "est_cost_usd": 3.0, "input_tokens": 5, "output_tokens": 6, "llm_calls": 7,
        }
    }
    assert live_budget.budget_status(_AUG, checkpoint_rows=checkpoint)["spent_usd"] == 3.0
    attempt = _attempt(checkpoint_rows=checkpoint)
    attempt.strategy_impl = _impl(input_tokens=1_000_000, llm_calls=7)
    attempt.finish(None)
    row = _ledger()
    assert row["est_cost_usd"] == pytest.approx(3.435)
    assert row["input_tokens"] == 1_000_005
    assert row["output_tokens"] == 6
    assert row["llm_calls"] == 14


def test_a_larger_ledger_wins_over_the_checkpoint():
    _insert_ledger("deepseek_v4_pro", 9.0)
    checkpoint = {"deepseek_v4_pro": {"est_cost_usd": 3.0}}
    assert live_budget.budget_status(_AUG, checkpoint_rows=checkpoint)["spent_usd"] == 9.0


def test_month_spend_sums_every_entry_of_that_month_only():
    _insert_ledger("deepseek_v4_pro", 1.0)
    _insert_ledger("dropped_model", 2.0)
    _insert_ledger("deepseek_v4_pro", 60.0, month="2026-07-01")
    checkpoint = {"qwen3_7_plus": {"est_cost_usd": 0.5}, "spy_index": {"est_cost_usd": 0}}
    august = live_budget.budget_status(_AUG, checkpoint_rows=checkpoint)
    assert august == {
        "month": "2026-08",
        "cap_usd": 50.0,
        "spent_usd": 3.5,
        "remaining_usd": 46.5,
        "halted": False,
    }
    july = live_budget.budget_status("2026-07-01")
    assert july["spent_usd"] == 60.0 and july["halted"] is True
    attempt = _attempt(checkpoint_rows=checkpoint)
    assert attempt.month_spent_usd() == pytest.approx(3.5)


@pytest.mark.parametrize("figure", [None, "n/a", float("nan"), float("inf"), -3.0])
def test_an_unreadable_spend_figure_counts_as_nothing(figure):
    """Rows written before the ledger can carry NULL or junk costs; they must
    neither crash the pre-check nor turn the month's total into nan."""
    status = live_budget.budget_status(
        _AUG, checkpoint_rows={"deepseek_v4_pro": {"est_cost_usd": figure}}
    )
    assert status["spent_usd"] == 0.0 and status["halted"] is False


def test_the_pre_check_and_the_in_run_stop(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "1.0")
    _insert_ledger("qwen3_7_plus", 0.6)
    attempt = _attempt()
    attempt.check_can_start()  # 0.60 < 1.00
    attempt.strategy_impl = _impl(input_tokens=400_000)  # $0.174 per session
    attempt.record_session()
    attempt.stop_if_next_session_would_cross()  # 0.774 + 0.174 = 0.948 <= 1.00
    attempt.strategy_impl.input_tokens = 800_000
    attempt.record_session()
    with pytest.raises(live_budget.LiveBudgetExceeded, match="would cross it"):
        attempt.stop_if_next_session_would_cross()  # 0.948 + 0.174 > 1.00
    assert _ledger()["est_cost_usd"] == pytest.approx(0.348)

    _insert_ledger("qwen3_7_plus", 1.0)
    with pytest.raises(live_budget.LiveBudgetExceeded, match="not started"):
        _attempt().check_can_start()


def test_last_attempts_reads_this_month_only():
    _insert_ledger("deepseek_v4_pro", 1.0, last_attempt={
        "on": "2026-08-27", "outcome": "budget_halted", "kind": "budget",
    })
    _insert_ledger("qwen3_7_plus", 1.0, month="2026-07-01", last_attempt={
        "on": "2026-07-31", "outcome": "failed", "kind": "network",
    })
    _insert_ledger("claude_haiku_4_5", 1.0)  # no attempt finished yet
    assert live_budget.last_attempts(_AUG) == {
        "deepseek_v4_pro": {"on": "2026-08-27", "outcome": "budget_halted", "kind": "budget"},
    }


# --- nobody else reads the ledger ------------------------------------------------


def test_ledger_rows_are_invisible_to_every_other_reader_and_survive_clear():
    today = datetime.now(timezone.utc).date()
    operator_cost_before = db.aggregate_operator_cost_for_day(today)
    attempt = _attempt()
    attempt.strategy_impl = _impl(input_tokens=1_000_000)
    attempt.finish(None)
    run_id = live_budget.spend_run_id(_AUG, "deepseek_v4_pro")

    # The admin daily job's operator-cost roll-up skips rows with no owner.
    assert db.aggregate_operator_cost_for_day(today) == operator_cost_before

    assert live.latest_live_month_runs(_AUG, _FREEZE) == {}
    assert live._entry_month_runs("deepseek_v4_pro", _AUG, _FREEZE) == []
    for session in (live.LIVE_SESSION_ID, live_budget.LIVE_SPEND_SESSION_ID):
        assert lb_service._find_cached_run("deepseek_v4_pro", _AUG, _FREEZE, session) is None
    assert run_id not in {row["run_id"] for row in build_leaderboard(db.get_all_runs())}

    # The public run listing defaults to mode=backtest. Asked for the ledger's
    # mode by name it does list the row, through the RunMetadata allow-list:
    # no dollar figure, no token counts and no metadata (so no last_attempt).
    client = TestClient(app)
    headers = {"X-Session-Id": "ledger-test"}
    listed = client.get("/runs", headers=headers)
    assert listed.status_code == 200, listed.text
    assert run_id not in {row["run_id"] for row in listed.json()}
    by_mode = client.get("/runs?mode=leaderboard_spend", headers=headers).json()
    record = next(row for row in by_mode if row["run_id"] == run_id)
    assert not {"est_cost_usd", "input_tokens", "output_tokens", "metadata"} & set(record)
    # The per-run routes: one answers only a caller who names the ledger's
    # session, with an empty curve and no dollar figure; the other rejects that
    # session id outright, since it is not a UUID.
    spend_session = {"X-Session-Id": live_budget.LIVE_SPEND_SESSION_ID}
    single = client.get(f"/api/backtest/{run_id}", headers=spend_session)
    assert single.status_code == 200, single.text
    assert single.json()["data"] == []
    assert "0.435" not in single.text and "input_tokens" not in single.text
    assert client.get(f"/runs/{run_id}", headers=spend_session).status_code == 400

    # The money was spent: clearing the board's rows must not forget it.
    live.clear_live_session_runs()
    assert db.get_run(run_id)["est_cost_usd"] == pytest.approx(0.435)


# --- tokens a crash leaves behind ---------------------------------------------


def _token_counting_manager(*, raise_in_decision_on=None, raise_in_execute_on=None):
    """A PortfolioManager stand-in that bills 100 input / 50 output tokens per
    model call and can die inside the call or just after it."""

    class Manager:
        def __init__(self, **kwargs):
            self.equity_history = []
            self.trades = []
            self.llm_calls = self.llm_decisions = 0
            self.input_tokens = self.output_tokens = 0

        def restore_state(self, snapshot):
            pass

        def snapshot_state(self):
            return {"cash": 1.0, "positions": {}}

        def get_portfolio_state(self, *args):
            return {}

        def make_trading_decision_with_llm(self, state, client, **kwargs):
            if self.llm_calls + 1 == raise_in_decision_on:
                raise RuntimeError("provider down")
            self.llm_calls += 1
            self.llm_decisions += 1
            self.input_tokens += 100
            self.output_tokens += 50
            return {"actions": []}

        def execute_actions(self, *args):
            if self.llm_calls == raise_in_execute_on:
                raise RuntimeError("broker rejected")

        def update_equity(self, market_data, price_cache, ts):
            self.equity_history.append({"timestamp": ts, "equity": 100.0})

        def get_equity_curve(self):
            return self.equity_history

    return Manager


@pytest.mark.parametrize(
    ("manager_kwargs", "tokens"),
    [
        # Dies inside the 4th call, before it is billed: three calls priced.
        ({"raise_in_decision_on": 4}, (300, 150)),
        # Dies after the 2nd call returned and was billed, before its fills.
        ({"raise_in_execute_on": 2}, (200, 100)),
    ],
)
def test_llm_agent_publishes_tokens_billed_before_a_crash(monkeypatch, manager_kwargs, tokens):
    """The manager is local to the loop, so tokens it counted were lost with it
    when a run died before its next session close. The Live spend ledger
    prices a dead attempt from the strategy's own counters."""
    import pandas as pd

    from dashboard.backend.domain.leaderboard.strategies import llm_agent

    monkeypatch.setattr(
        llm_agent, "PortfolioManager", _token_counting_manager(**manager_kwargs)
    )
    stamps = [pd.Timestamp(f"2026-08-03T{h}:00:00Z") for h in range(13, 20)]
    strategy = llm_agent.LLMAgentStrategy({"strategy": "llm_agent", "model_id": "m"})
    with pytest.raises(RuntimeError):
        strategy._run_decision_loop(
            client=object(), timestamps=stamps, symbols=["AAPL"], data={},
            price_cache={}, initial_capital=100.0, model_id="m",
            on_session_close=lambda day, curve: None,
        )
    assert (strategy.input_tokens, strategy.output_tokens) == tokens
    # The H6 counters still move only at a session close, which never came.
    assert strategy.llm_calls == 0 and strategy.decision_steps == 0


# --- enforcement in the live deploy ---------------------------------------------


def _no_billable_work(*_args, **_kwargs):
    raise AssertionError("a budget halt reached the model or the market data")


class _DiesBeforeTheClose(_ScriptedAgent):
    """Bills a session, then dies before that session's close hook fires."""

    def __init__(self, die_on, error, **kwargs):
        super().__init__(**kwargs)
        self.die_on = die_on
        self.error = error

    def run(self, bars, start, end, capital, starting_snapshot=None, on_session_close=None):
        def hook(day, curve):
            if day.isoformat() == self.die_on:
                raise self.error
            on_session_close(day, curve)

        return super().run(
            bars, start, end, capital,
            starting_snapshot=starting_snapshot, on_session_close=hook,
        )


def test_a_zero_budget_blocks_before_any_bar_fetch_or_model_build(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0")
    monkeypatch.setattr(lb_service, "get_strategy", _no_billable_work)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _no_billable_work)
    freeze_cfg = _live_aug27()
    with pytest.raises(live_budget.LiveBudgetExceeded, match="not started"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert _month_rows("deepseek_v4_pro") == []
    ledger = _ledger()
    assert ledger["est_cost_usd"] == 0
    assert ledger["metadata"]["last_attempt"] == {
        "on": _FREEZE, "outcome": "budget_halted", "kind": "budget",
    }


def test_an_entry_already_current_is_not_halted_and_writes_nothing(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0")
    _insert_checkpoint(_FREEZE, "08-27")
    freeze_cfg = _live_aug27()
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["cached"] is True
    assert _ledger() is None


def test_a_snapshot_that_already_covers_a_non_session_freeze_is_not_halted(monkeypatch):
    """The second way to be current: the newest checkpoint is the last session
    before a freeze_end that is not itself a session. live_clock never freezes
    on such a day, but a deploy handed one must not build the model, fetch
    bars, write a ledger row or halt either."""
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0")
    monkeypatch.setattr(lb_service, "get_strategy", _no_billable_work)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _no_billable_work)
    _insert_checkpoint("2026-08-28", "08-28")
    freeze_cfg = {**_live_aug27(), "end_date": "2026-08-29"}  # a Saturday
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["cached"] is True and row["segment"] is None
    assert row["run_id"] == "lb_deepseek_v4_pro_20260801_20260828"
    assert _ledger() is None


def test_a_catch_up_stops_before_the_session_that_would_cross_the_cap(monkeypatch):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0.8")
    _insert_checkpoint("2026-08-24", "08-24")
    agent = _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    freeze_cfg = _live_aug27()
    with pytest.raises(live_budget.LiveBudgetExceeded, match="would cross it"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    # $0.435 spent; a second $0.435 session would reach $0.87 > $0.80.
    assert agent.traded == ["2026-08-25"]
    # The session that was paid for is stored, not thrown away.
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-08-25"]
    ledger = _ledger()
    assert ledger["est_cost_usd"] == pytest.approx(0.435)
    assert ledger["metadata"]["last_attempt"]["outcome"] == "budget_halted"


def test_the_last_session_of_a_segment_is_never_stopped_by_its_projection(monkeypatch):
    """Nothing is left to stop after it, so raising would only relabel a
    finished night as a halt. The next night's pre-check decides."""
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0.5")
    _insert_checkpoint("2026-08-26", "08-26")
    _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    freeze_cfg = _live_aug27()
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["window"]["end_date"] == _FREEZE
    assert _ledger()["metadata"]["last_attempt"]["outcome"] == "ok"


def test_a_run_that_dies_mid_session_is_priced_in_full(monkeypatch):
    _insert_checkpoint("2026-08-24", "08-24")
    _use_agent(monkeypatch, _DiesBeforeTheClose(
        "2026-08-26",
        ConnectionResetError("reset by peer, key sk-live-SECRET"),
        step_tokens=(1_000_000, 0),
    ))
    freeze_cfg = _live_aug27()
    with pytest.raises(ConnectionResetError):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert [r["end_date"] for r in _month_rows("deepseek_v4_pro")] == ["2026-08-25"]
    ledger = _ledger()
    # 08-25 was priced at its close; 08-26's tokens only by the final pricing.
    assert ledger["est_cost_usd"] == pytest.approx(0.87)
    assert ledger["metadata"]["last_attempt"] == {
        "on": _FREEZE, "outcome": "failed", "kind": "network",
    }
    assert "SECRET" not in json.dumps(ledger, default=str)


def test_a_ledger_that_cannot_be_written_never_hides_the_deploy_error(monkeypatch, capsys):
    _insert_checkpoint("2026-08-24", "08-24")
    real_insert = db.insert_run

    def ledger_down_at_the_end(**row):
        if row["run_id"].startswith("live_spend_") and "last_attempt" in row["metadata"]:
            raise OSError("ledger disk full")
        return real_insert(**row)

    monkeypatch.setattr(db, "insert_run", ledger_down_at_the_end)
    _use_agent(monkeypatch, _ScriptedAgent(
        crash_on={"2026-08-26"}, step_tokens=(1_000_000, 0),
    ))
    freeze_cfg = _live_aug27()
    with pytest.raises(RuntimeError, match="worker killed"):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert (
        "ERROR: live.spend_ledger_write_failed entry=deepseek_v4_pro error=OSError"
        in capsys.readouterr().out
    )
    # The session-close upsert for 08-25 landed before the final one failed.
    assert _ledger()["est_cost_usd"] == pytest.approx(0.435)


def test_a_segment_h6_refuses_to_store_is_still_billed(monkeypatch):
    _use_agent(monkeypatch, _ScriptedAgent(
        misses={"2026-08-03": 7}, step_tokens=(1_000_000, 0),
    ))
    freeze_cfg = _live_aug27()
    with pytest.raises(lb_service.LeaderboardFallbackError):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert _month_rows("deepseek_v4_pro") == []
    ledger = _ledger()
    assert ledger["est_cost_usd"] == pytest.approx(0.435)
    assert ledger["metadata"]["last_attempt"]["kind"] == "model_coverage"


def test_a_forced_replay_adds_to_the_month_instead_of_resetting_it(monkeypatch):
    freeze_cfg = _live_aug27()
    _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    row = live.deploy_live_model_increment(
        _deepseek(freeze_cfg), freeze_cfg, force_refresh=True
    )
    month = 19 * 0.435  # 19 sessions, Aug 3 to Aug 27
    # The checkpoint chain starts over with the replay; the money does not.
    assert db.get_run(row["run_id"])["est_cost_usd"] == pytest.approx(month)
    assert _ledger()["est_cost_usd"] == pytest.approx(2 * month)
    assert _ledger()["metadata"]["attempts"] == 2


def test_the_first_deploy_after_rollout_counts_what_checkpoints_already_spent(monkeypatch):
    # A row written before the ledger existed records $49.50 of the $50 cap.
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260824", "deepseek_v4_pro", "2026-08-24",
        [_pt("2026-08-24T19:00:00+00:00", SEED)],
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": "08-24"}},
        est_cost_usd=49.5,
    )
    agent = _use_agent(monkeypatch, _ScriptedAgent(step_tokens=(1_000_000, 0)))
    freeze_cfg = _live_aug27()
    with pytest.raises(live_budget.LiveBudgetExceeded):
        live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert agent.traded == ["2026-08-25"]
    assert _ledger()["est_cost_usd"] == pytest.approx(49.935)


def test_a_new_month_starts_with_nothing_spent(monkeypatch):
    _insert_ledger("deepseek_v4_pro", 60.0, month="2026-07-01")
    _use_agent(monkeypatch, _ScriptedAgent())
    freeze_cfg = _live_aug27()
    row = live.deploy_live_model_increment(_deepseek(freeze_cfg), freeze_cfg)
    assert row["cached"] is False
    assert _ledger(month="2026-07-01")["est_cost_usd"] == 60.0
    assert _ledger()["est_cost_usd"] < 1


def test_a_spent_month_halts_the_roster_with_one_log_line(monkeypatch, capsys):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "get_strategy", _no_billable_work)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _no_billable_work)
    as_of = datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    roster = [entry["id"] for entry in live.live_llm_entries(_live_aug27())]

    result = live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)
    assert result["models_deployed"] is False
    assert result["model_failures"] == []
    assert result["budget_halted"] == roster
    assert result["budget"] == {
        "month": "2026-08",
        "cap_usd": 0.0,
        "spent_usd": 0.0,
        "remaining_usd": 0.0,
        "halted": True,
    }
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if "live.budget_exhausted" in line] == [
        "ERROR: live.budget_exhausted month=2026-08 cap_usd=0.00 spent_usd=0.00 "
        f"skipped={','.join(roster)}"
    ]
    assert "live.model_deploy_failed" not in out
    # Saved, and not mistaken for a finished night: the next cron tries again.
    assert live._LIVE_REFRESH_STATE_PATH.exists()
    assert live.refresh_live_leaderboard(deploy_models=True, as_of=as_of)["skipped"] is False


def test_halts_and_failures_are_reported_apart(monkeypatch, capsys):
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0.8")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    freeze_cfg = _live_aug27()
    deepseek = _deepseek(freeze_cfg)
    twin = {**deepseek, "id": "deepseek_twin", "name": "DeepSeek twin"}
    monkeypatch.setattr(live, "live_llm_entries", lambda config=None: [deepseek, twin])
    dies = _DiesBeforeTheClose(
        "2026-08-03", RuntimeError("provider reset"), step_tokens=(2_000_000, 0),
    )
    monkeypatch.setattr(
        lb_service,
        "get_strategy",
        lambda entry: dies if entry["id"] == "deepseek_v4_pro" else _no_billable_work(),
    )

    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    # DeepSeek billed $0.87 and died; that spend leaves the twin nothing.
    assert [f["entry_id"] for f in result["model_failures"]] == ["deepseek_v4_pro"]
    assert result["budget_halted"] == ["deepseek_twin"]
    assert result["models_deployed"] is False
    assert result["budget"]["spent_usd"] == pytest.approx(0.87)
    out = capsys.readouterr().out
    assert "ERROR: live.model_deploy_failed entry=deepseek_v4_pro" in out
    assert (
        "ERROR: live.budget_exhausted month=2026-08 cap_usd=0.80 spent_usd=0.87 "
        "skipped=deepseek_twin"
    ) in out


def test_a_halt_after_a_success_reports_each_entry_once(monkeypatch):
    """Since #668 the success path follows the try statement, so it must run
    only in the statement's else clause: a halt that fell through to it would
    append the entry before it to model_results a second time."""
    monkeypatch.setenv(live_budget.BUDGET_ENV, "0.4")
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    freeze_cfg = _live_aug27()
    deepseek = _deepseek(freeze_cfg)
    twin = {**deepseek, "id": "deepseek_twin", "name": "DeepSeek twin"}
    monkeypatch.setattr(live, "live_llm_entries", lambda config=None: [deepseek, twin])
    # One session left to trade, so DeepSeek's own projection never stops it.
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260826", "deepseek_v4_pro", "2026-08-26",
        [_pt("2026-08-26T19:00:00+00:00", SEED)],
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": "08-26"}},
    )
    monkeypatch.setattr(
        lb_service,
        "get_strategy",
        lambda entry: _ScriptedAgent(step_tokens=(1_000_000, 0))
        if entry["id"] == "deepseek_v4_pro" else _no_billable_work(),
    )

    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    # DeepSeek's one session ($0.435) spends the $0.40 cap; the twin's
    # pre-check halts it before it is built.
    assert [row["entry_id"] for row in result["model_results"]] == ["deepseek_v4_pro"]
    assert result["budget_halted"] == ["deepseek_twin"]
    assert result["model_failures"] == []


def test_a_halt_between_two_shared_failures_does_not_reset_the_stop(monkeypatch, capsys):
    """A halt is kept out of RosterFailures, even one that closed sessions
    first: it stops before the segment's last checkpoint and its end-of-run
    checks, so it has not shown that the shared path works. Two failures with
    the same error either side of it still stop the roster. A wrong stop leaves
    the rest unbilled until the next night; a wrong reset bills another entry
    against the same fault."""
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    entries = [{"id": f"entry_{n}", "name": f"Entry {n}"} for n in range(4)]
    monkeypatch.setattr(live, "live_llm_entries", lambda config=None: entries)
    called = []

    def deploy(entry, freeze_cfg, **kwargs):
        called.append(entry["id"])
        if entry["id"] == "entry_1":
            raise live_budget.LiveBudgetExceeded("stopped between sessions")
        raise OSError("disk full")

    monkeypatch.setattr(live, "deploy_live_model_increment", deploy)

    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    assert result["budget_halted"] == ["entry_1"]
    assert called == ["entry_0", "entry_1", "entry_2"]
    assert [f["entry_id"] for f in result["model_failures"]] == [
        "entry_0",
        "entry_2",
        "entry_3",
    ]
    assert (
        "ERROR: live.model_not_attempted entry=entry_3 after=entry_0,entry_2 "
        "error=OSError"
    ) in capsys.readouterr().out


def test_a_budget_read_that_fails_still_prunes_and_saves(monkeypatch, capsys):
    """The budget read runs after every model has traded and been billed. If it
    raised out of the refresh, the prune and the state save would be skipped,
    which is what #668 stopped the roster loop from doing."""
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: _ScriptedAgent())

    def unreadable(*args, **kwargs):
        raise OSError("pool timeout")

    monkeypatch.setattr(live_budget, "budget_status", unreadable)

    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    assert result["budget"] is None
    assert result["models_deployed"] is True
    assert "pruned_runs" in result
    assert live._live_refresh_state()["budget"] is None
    assert "ERROR: live.budget_status_failed error=OSError" in capsys.readouterr().out


def test_a_clean_night_reports_the_budget_and_no_halt(monkeypatch, capsys):
    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", lambda **kwargs: {})
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    monkeypatch.setattr(lb_service, "get_strategy", lambda entry: _ScriptedAgent())
    roster = [entry["id"] for entry in live.live_llm_entries(_live_aug27())]
    result = live.refresh_live_leaderboard(
        deploy_models=True, as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET)
    )
    assert result["models_deployed"] is True
    assert result["budget_halted"] == []
    assert result["budget"]["halted"] is False
    assert result["budget"]["cap_usd"] == 50.0
    assert set(live_budget.month_ledger_rows(_AUG)) == set(roster)
    assert "live.budget_exhausted" not in capsys.readouterr().out


# --- what the public GET shows -----------------------------------------------------


def _get_aug27():
    return live.get_live_leaderboard(as_of=datetime(2026, 8, 27, 17, 39, tzinfo=_ET))


@pytest.fixture
def _read_only(monkeypatch):
    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("the live GET path reached the compute path")

    monkeypatch.setattr(lb_service, "ensure_leaderboard_runs", _must_not_run)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", _must_not_run)


def test_get_live_gains_only_a_halt_flag_and_last_attempts(monkeypatch, _read_only):
    """The ledger's whole public footprint is ``live_status.budget_halted`` and
    each model's ``last_attempt``. The per-entry ``est_cost_usd`` the GET
    already published from the checkpoint row predates the ledger and is left
    as it was; the cap and the month-to-date spend never reach the payload."""
    monkeypatch.setenv(live_budget.BUDGET_ENV, "48.7654")
    _insert_live_run(
        "lb_deepseek_v4_pro_20260801_20260827", "deepseek_v4_pro", _FREEZE,
        [_pt(f"{_FREEZE}T19:00:00+00:00", SEED)],
        metadata={live.LIVE_SNAPSHOT_KEY: {"cash": SEED, "positions": {}, "day": "08-27"}},
        est_cost_usd=12.3456,
    )
    before = _get_aug27()
    _insert_ledger("deepseek_v4_pro", 48.7654, last_attempt={
        "on": _FREEZE, "outcome": "budget_halted", "kind": "budget",
        "error": "sk-live-SECRET", "spent_usd": 48.7654,
    })
    after = _get_aug27()

    # live_status gains one boolean, and the ledger moves nothing else in it.
    assert before["live_status"]["budget_halted"] is False
    assert after["live_status"]["budget_halted"] is True
    assert set(after["live_status"]) == set(before["live_status"])
    assert {
        key for key, value in after["live_status"].items()
        if value != before["live_status"][key]
    } == {"budget_halted"}

    # Each entry gains last_attempt, exactly {on, outcome, kind}; nothing else moves.
    before_by_id = {entry["entry_id"]: entry for entry in before["entries"]}
    assert {entry["entry_id"] for entry in after["entries"]} == set(before_by_id)
    for entry in after["entries"]:
        old = before_by_id[entry["entry_id"]]
        assert set(entry) == set(old)
        assert {key for key in entry if entry[key] != old[key]} <= {"last_attempt"}
    deepseek = next(e for e in after["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert deepseek["last_attempt"] == {
        "on": _FREEZE, "outcome": "budget_halted", "kind": "budget",
    }
    assert all(e["last_attempt"] is None for e in after["entries"] if not e["is_model"])

    # The pre-existing per-entry figure is the checkpoint row's, untouched...
    assert deepseek["est_cost_usd"] == pytest.approx(12.3456)
    # ...and neither the cap nor the month-to-date spend reaches the payload.
    text = json.dumps(after)
    assert "48.7654" not in text
    assert "SECRET" not in text
    for key in ("cap_usd", "spent_usd", "remaining_usd"):
        assert key not in text


def test_get_live_reports_a_failed_night_without_halting_the_board(_read_only):
    _insert_ledger("deepseek_v4_pro", 0.5, last_attempt={
        "on": _FREEZE, "outcome": "failed", "kind": "market_data_gap",
    })
    payload = _get_aug27()
    assert payload["live_status"]["budget_halted"] is False
    deepseek = next(e for e in payload["entries"] if e["entry_id"] == "deepseek_v4_pro")
    assert deepseek["last_attempt"]["outcome"] == "failed"
    assert deepseek["last_attempt"]["kind"] == "market_data_gap"


def test_get_live_ignores_last_months_ledger_and_an_empty_one(_read_only):
    payload = _get_aug27()
    assert payload["live_status"]["budget_halted"] is False
    assert all(entry["last_attempt"] is None for entry in payload["entries"])
    _insert_ledger("deepseek_v4_pro", 60.0, month="2026-07-01", last_attempt={
        "on": "2026-07-31", "outcome": "budget_halted", "kind": "budget",
    })
    assert _get_aug27()["live_status"]["budget_halted"] is False
