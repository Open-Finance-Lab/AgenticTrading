"""Order sizing and per-step accounting for the hosted TradingAgents runtime.

Every case here runs at capital the product actually allows. That is the point
of the file: ``INITIAL_CAPITAL`` is $1,000 and ``MAX_BACKTEST_INITIAL_CAPITAL``
is $3,000, so with ``TRADINGAGENTS_MAX_POSITION_WEIGHT`` at its 0.2 default the
position budget is $200-$600 -- below one share of a good half of the Dow. The
existing adapter suite builds its contexts at ``total_equity=10_000``, which is
3.3x the product maximum, and at ``price=50.0``; at those numbers the whole
class of failure below is invisible.
"""

from datetime import date, datetime, timezone

import pytest

from dashboard.backend.domain.agents.runtime import AgentRuntimeContext
from dashboard.backend.domain.backtesting.constants import (
    INITIAL_CAPITAL,
    MAX_BACKTEST_INITIAL_CAPITAL,
)
from dashboard.backend.infrastructure.llm.validator import MAX_ORDER_SHARES
from dashboard.backend.infrastructure.tradingagents.adapter import (
    DEFAULT_MAX_POSITION_WEIGHT,
    TradingAgentsOutputError,
    TradingAgentsRuntime,
)


def _context(
    *,
    total_equity=float(INITIAL_CAPITAL),
    price=50.0,
    positions=None,
    cutoff=date(2026, 3, 12),
    timestamp=None,
):
    return AgentRuntimeContext(
        timestamp=timestamp or datetime(2026, 3, 13, 15, tzinfo=timezone.utc),
        backtest_start_date="2026-03-01",
        symbols=["AAPL"],
        cash=total_equity,
        total_equity=total_equity,
        positions=dict(positions or {}),
        entry_prices={},
        current_prices={"AAPL": price},
        latest_market_date_before_decision=cutoff,
    )


def _output(*, rating="Buy", action="BUY"):
    return {
        "schema_version": "tradingagents-atl-v1",
        "manifest": {
            "symbol": "AAPL",
            "tradingagents_version": "0.3.1",
            "selected_analysts": ["market"],
            "safe_config_sha256": "0" * 64,
        },
        "decision": {
            "analysis_date": "2026-03-12",
            "rating": rating,
            "atl_action": action,
            "status": "valid",
            "attempts": 1,
            "raw_sha256": "a" * 64,
            "raw_excerpt": "FINAL TRANSACTION PROPOSAL",
        },
    }


class _Runner:
    """Returns a queued output per call, or raises a queued exception."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def run(self, payload, *, timeout_seconds, environment=None):
        self.calls += 1
        item = self.outputs.pop(0) if self.outputs else _output()
        if isinstance(item, BaseException):
            raise item
        return item


def _runtime(runner, *, weight=None):
    environment = {
        "COMMONSTACK_API_KEY": "test-key",
        "TRADINGAGENTS_PYTHON": "/usr/bin/python3",
    }
    if weight is not None:
        environment["TRADINGAGENTS_MAX_POSITION_WEIGHT"] = str(weight)
    return TradingAgentsRuntime(
        {"analysts": ["market"]}, runner=runner, environment=environment
    )


# ---------------------------------------------------------------------------
# 1. The capital bound this whole file exists for.
# ---------------------------------------------------------------------------

def test_the_products_capital_bounds_are_what_this_file_assumes():
    # If these move, the arithmetic in the cases below stops describing the
    # product and the file needs rereading rather than silently passing.
    assert INITIAL_CAPITAL == 1000
    assert MAX_BACKTEST_INITIAL_CAPITAL == 3000
    assert DEFAULT_MAX_POSITION_WEIGHT == 0.2


@pytest.mark.parametrize(
    "equity,price",
    [
        # The default sleeve against real late-2026 prices. $200 of budget.
        (1000.0, 255.0),   # AAPL
        (1000.0, 420.0),   # MSFT
        (1000.0, 690.0),   # META
        # Even the product's maximum sleeve ($600 of budget) cannot reach the
        # pricier Dow names.
        (3000.0, 690.0),
        (3000.0, 1100.0),
    ],
)
def test_a_budget_below_one_share_emits_no_order_and_says_why(equity, price):
    runner = _Runner([_output()])
    runtime = _runtime(runner)

    result = runtime.decide(_context(total_equity=equity, price=price))

    assert result["actions"] == []
    trace = runtime.decision_audit_rows[0]["actions_submitted"][0]
    # Not attributed to the model: the rating was a Buy.
    assert trace["upstream_action"] == "buy"
    assert trace["mapped_atl_action"] == "BUY"
    assert trace["filter_reason"] == "price_above_position_budget"
    # And the arithmetic is on the row, so a reader can see *why* without
    # having to reconstruct the budget.
    sizing = trace["sizing"]
    assert sizing["position_budget"] == pytest.approx(equity * 0.2)
    assert sizing["price"] == pytest.approx(price)
    assert sizing["max_position_weight"] == 0.2


def test_the_reason_is_aggregated_so_a_flat_run_is_explainable():
    # Three Buy days at a price the budget cannot reach.
    runner = _Runner([_output(), _output(), _output()])
    runtime = _runtime(runner)
    for day in (13, 16, 17):
        runtime.decide(
            _context(
                total_equity=1000.0,
                price=255.0,
                timestamp=datetime(2026, 3, day, 15, tzinfo=timezone.utc),
            )
        )

    assert runtime.sizing_filter_counts == {"price_above_position_budget": 3}
    # The model drove all three; it is the capital that produced no trades.
    assert runtime.decisions == 3
    assert runtime.calls == 3


def test_a_model_hold_is_a_different_reason_from_a_capital_refusal():
    runner = _Runner([_output(rating="Hold", action="HOLD")])
    runtime = _runtime(runner)
    runtime.decide(_context(total_equity=1000.0, price=255.0))
    # The two must never collapse into one bucket: one is the strategy
    # working, the other is the strategy never getting a chance to.
    assert runtime.sizing_filter_counts == {"model_hold": 1}


def test_an_affordable_price_still_buys_at_the_product_default():
    runner = _Runner([_output()])
    runtime = _runtime(runner)
    # $1,000 equity, 20% budget = $200; a $40 share is 5 of them.
    result = runtime.decide(_context(total_equity=1000.0, price=40.0))
    assert [a["action"] for a in result["actions"]] == ["buy"]
    assert result["actions"][0]["shares"] == 5


# ---------------------------------------------------------------------------
# 2. MAX_ORDER_SHARES: the per-order cap this adapter computes its way past.
# ---------------------------------------------------------------------------

def test_a_sub_dollar_ticker_is_clamped_to_the_per_order_cap():
    runner = _Runner([_output()])
    # Weight 1.0 is a legal operator setting; $3,000 of budget at $0.01 is
    # 300,000 shares, 30x the cap.
    runtime = _runtime(runner, weight=1.0)

    result = runtime.decide(_context(total_equity=3000.0, price=0.01))

    assert result["actions"][0]["shares"] == MAX_ORDER_SHARES
    # The cap is on the row, so the clamp is visible rather than inferred.
    trace = runtime.decision_audit_rows[0]["actions_submitted"][0]
    assert trace["sizing"]["max_order_shares"] == MAX_ORDER_SHARES


def test_an_oversized_order_is_not_raised_as_a_retryable_failure():
    # Before the clamp this raised TradingAgentsOutputError, which the engine
    # treats as transient -- and the computation is deterministic, so the run
    # spent its whole failure budget, and real money, rediscovering the same
    # arithmetic once per trading day.
    runner = _Runner([_output(), _output(), _output()])
    runtime = _runtime(runner, weight=1.0)
    for day in (13, 16, 17):
        result = runtime.decide(
            _context(
                total_equity=3000.0,
                price=0.01,
                timestamp=datetime(2026, 3, day, 15, tzinfo=timezone.utc),
            )
        )
        assert result["actions"], "the order must be emitted, not refused"
    assert runtime.decisions == 3


# ---------------------------------------------------------------------------
# 3. Per-step accounting: a failed call must not rewrite another day's row.
# ---------------------------------------------------------------------------

def test_execution_count_lands_on_the_call_it_belongs_to():
    runner = _Runner([_output(), _output()])
    runtime = _runtime(runner)

    runtime.decide(_context(timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc)))
    runtime.record_latest_execution(1)
    runtime.decide(_context(timestamp=datetime(2026, 3, 16, 15, tzinfo=timezone.utc)))
    runtime.record_latest_execution(1)

    rows = runtime.decision_audit_rows
    assert [row["step_index"] for row in rows] == [0, 1]
    assert [row["actions_executed"] for row in rows] == [1, 1]


def test_a_validation_failure_does_not_overwrite_the_previous_days_count():
    # Day 1 executes a trade. Day 2's response fails validation: it bills a
    # call and appends no row, so the engine -- which calls
    # record_latest_execution whenever the call counter moved -- used to apply
    # day 2's zero to day 1's row.
    bad = _output()
    bad["decision"]["rating"] = "Strong Buy"  # not in the five-tier map
    runner = _Runner([_output(), bad])
    runtime = _runtime(runner)

    runtime.decide(_context(timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc)))
    runtime.record_latest_execution(1)

    with pytest.raises(TradingAgentsOutputError):
        runtime.decide(
            _context(timestamp=datetime(2026, 3, 16, 15, tzinfo=timezone.utc))
        )
    runtime.record_latest_execution(0)

    rows = runtime.decision_audit_rows
    assert len(rows) == 1
    assert rows[0]["step_index"] == 0
    assert rows[0]["actions_executed"] == 1, (
        "day 1 executed a trade; day 2's failure must not zero it"
    )


def test_a_billed_call_that_decided_nothing_is_not_counted_as_a_decision():
    bad = _output()
    bad["decision"]["rating"] = "Strong Buy"
    runner = _Runner([bad])
    runtime = _runtime(runner)

    with pytest.raises(TradingAgentsOutputError):
        runtime.decide(_context())

    # Paid for, so `calls` ticks...
    assert runtime.calls == 1
    # ...but nothing was decided, so coverage must not claim it.
    assert runtime.decisions == 0


def test_execution_count_is_written_once_per_call():
    runner = _Runner([_output()])
    runtime = _runtime(runner)
    runtime.decide(_context())
    runtime.record_latest_execution(2)
    # A second write for the same step would be a bug upstream of here;
    # silently accepting it is how the cross-day overwrite stayed hidden.
    runtime.record_latest_execution(0)
    assert runtime.decision_audit_rows[0]["actions_executed"] == 2


# ---------------------------------------------------------------------------
# 4. A step that produced no decision at all must be recorded.
# ---------------------------------------------------------------------------

def test_the_first_day_without_a_prior_market_date_is_recorded_as_skipped():
    runner = _Runner([])
    runtime = _runtime(runner)

    result = runtime.decide(_context(cutoff=None))

    assert result["actions"] == []
    assert runner.calls == 0
    assert runtime.calls == 0
    assert runtime.decisions == 0
    # The decisive part: a one-day backtest is exactly this step, and used to
    # complete as a clean run that had never asked the model anything.
    assert runtime.skipped_steps == [
        {
            "timestamp": "2026-03-13T15:00:00+00:00",
            "reason": "no_market_date_before_decision",
        }
    ]


def test_a_capped_order_tops_up_instead_of_reporting_the_target_as_reached():
    """The cap belongs on the order, not on the target position.

    Clamping the *target* meant the position itself could never exceed the
    per-order ceiling, and once it reached it the next day reported
    `already_at_target_weight` -- false, since the position was a few percent
    of the 20% target it was still trying to reach. A fourth cause reported as
    one of the other three defeats the point of `sizing_filter_counts`.
    """
    runner = _Runner([_output(), _output()])
    runtime = _runtime(runner, weight=1.0)
    context = dict(total_equity=3000.0, price=0.01)

    # Day 1: the target is 300,000 shares; the order is capped at 10,000.
    first = runtime.decide(
        _context(
            **context,
            timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc),
        )
    )
    assert first["actions"][0]["shares"] == MAX_ORDER_SHARES
    sizing = runtime.decision_audit_rows[0]["actions_submitted"][0]["sizing"]
    assert sizing["target_shares"] == 300_000
    assert sizing["order_capped"] is True

    # Day 2, holding the cap: still far from the target, so it tops up again
    # rather than claiming the target weight is held.
    second = runtime.decide(
        _context(
            **context,
            positions={"AAPL": MAX_ORDER_SHARES},
            timestamp=datetime(2026, 3, 16, 15, tzinfo=timezone.utc),
        )
    )
    assert second["actions"], "a position at 3% of target must keep buying"
    assert second["actions"][0]["shares"] == MAX_ORDER_SHARES
    trace = runtime.decision_audit_rows[1]["actions_submitted"][0]
    assert trace["filter_reason"] is None
    assert "already_at_target_weight" not in runtime.sizing_filter_counts


def test_a_position_genuinely_at_target_still_reports_that():
    runner = _Runner([_output()])
    runtime = _runtime(runner)
    # $1,000 equity, 20% budget = $200; a $40 share gives a 5-share target.
    result = runtime.decide(
        _context(total_equity=1000.0, price=40.0, positions={"AAPL": 5})
    )
    assert result["actions"] == []
    assert runtime.sizing_filter_counts == {"already_at_target_weight": 1}


def test_an_order_within_the_cap_is_not_marked_capped():
    runner = _Runner([_output()])
    runtime = _runtime(runner)
    runtime.decide(_context(total_equity=1000.0, price=40.0))
    sizing = runtime.decision_audit_rows[0]["actions_submitted"][0]["sizing"]
    assert sizing["order_capped"] is False
    assert sizing["target_shares"] == 5


# ---------------------------------------------------------------------------
# 5. A decision killed mid-flight was paid for. The run must say so.
# ---------------------------------------------------------------------------

def test_a_timed_out_decision_counts_as_a_billed_call():
    """`calls` is the billing counter, and a timeout is the designed-in case.

    One decision is two upstream passes (580-830s measured) and the shipped
    budget is 600s, so a kill after ~10 of 11 model calls is routine rather
    than exceptional. With the increment only on the success path, a five-day
    run whose every day timed out persisted `llm_calls = 0` and
    `llm_model = "rule-based"` -- reporting zero platform spend for four
    near-complete billed debates.
    """
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsRuntimeError,
    )

    runner = _Runner(
        [TradingAgentsRuntimeError("TradingAgents runtime timed out after 600 seconds")]
    )
    runtime = _runtime(runner)

    with pytest.raises(TradingAgentsRuntimeError):
        runtime.decide(_context())

    assert runtime.calls == 1, "the tokens were spent and discarded"
    # Nothing was decided, so coverage must not claim it.
    assert runtime.decisions == 0
    assert runtime.decision_audit_rows == []


def test_a_decision_that_never_reached_the_child_is_not_billed():
    """A configuration fault spends nothing: no interpreter, a payload the
    bridge refused before `generate()`, or a credential the gateway rejected.
    """
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsConfigurationError,
    )

    runner = _Runner(
        [TradingAgentsConfigurationError("TradingAgents runtime is not installed")]
    )
    runtime = _runtime(runner)

    with pytest.raises(TradingAgentsConfigurationError):
        runtime.decide(_context())

    assert runtime.calls == 0
    assert runtime.decisions == 0


def test_a_timeout_does_not_rewrite_the_previous_days_execution_count():
    """The counter now moves on a mid-flight failure, so the audit pointer has
    to be cleared before the call -- otherwise this step's execution count
    lands on the previous trading day's row."""
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsRuntimeError,
    )

    runner = _Runner(
        [_output(), TradingAgentsRuntimeError("timed out after 600 seconds")]
    )
    runtime = _runtime(runner)

    runtime.decide(_context(timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc)))
    runtime.record_latest_execution(1)

    with pytest.raises(TradingAgentsRuntimeError):
        runtime.decide(
            _context(timestamp=datetime(2026, 3, 16, 15, tzinfo=timezone.utc))
        )
    runtime.record_latest_execution(0)

    rows = runtime.decision_audit_rows
    assert len(rows) == 1
    assert rows[0]["actions_executed"] == 1


def test_a_timeout_cannot_write_to_a_row_whose_count_was_never_recorded():
    """The pre-call clear, exercised on the path where it is load-bearing.

    After a normal day `record_latest_execution` clears the pointer itself, so
    the clear before the call looks redundant. It is not: if a day's count is
    never recorded -- the engine skips it when the dispatch raised before
    execution -- the pointer stays on that row, and the next day's timeout
    (which now moves the call counter) would write its own count there.
    """
    from dashboard.backend.infrastructure.tradingagents.adapter import (
        TradingAgentsRuntimeError,
    )

    runner = _Runner(
        [_output(), TradingAgentsRuntimeError("timed out after 600 seconds")]
    )
    runtime = _runtime(runner)

    # Day 1 succeeds, and its execution count is deliberately NOT recorded.
    runtime.decide(_context(timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc)))

    with pytest.raises(TradingAgentsRuntimeError):
        runtime.decide(
            _context(timestamp=datetime(2026, 3, 16, 15, tzinfo=timezone.utc))
        )
    # The engine calls this because the call counter moved.
    runtime.record_latest_execution(7)

    rows = runtime.decision_audit_rows
    assert len(rows) == 1
    assert rows[0]["step_index"] == 0
    # Day 1's row keeps its own (unset) count; day 2's number must not land
    # on it.
    assert rows[0]["actions_executed"] == 0
