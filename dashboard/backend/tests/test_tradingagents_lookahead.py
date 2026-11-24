"""Look-ahead safety of the hosted TradingAgents runtime.

A historical backtest is only meaningful if every input the analysts read was
available on the decision date. Three of upstream's four analysts read data
that is bounded by the wall clock instead, and one of them pre-fetches it into
the prompt before the model is even invoked -- so the only control available at
ATL's boundary is refusing the analyst.

The tests here are deliberately *behavioural*: each drives the real guard and
asserts that a contaminated run never reaches the paid upstream call. A test
that only asserted the evidence table's contents would pass just as happily
with the guard deleted.

The evidence behind the classification itself lives in
``domain/agents/runtime.py`` and was read off the pinned upstream revision; see
``test_tradingagents_upstream_contract.py`` for the probe that re-checks it
against an installed upstream.
"""

from datetime import date, datetime, timezone

import pytest

from dashboard.backend.domain.agents.runtime import (
    TRADINGAGENTS_ANALYSTS,
    TRADINGAGENTS_AS_OF_SAFE_ANALYSTS,
    TRADINGAGENTS_LOOKAHEAD_EVIDENCE,
    AgentRuntimeConfigurationError,
    AgentRuntimeContext,
    describe_tradingagents_lookahead,
    tradingagents_lookahead_unsafe_analysts,
)
from dashboard.backend.infrastructure.tradingagents.adapter import (
    TradingAgentsConfigurationError,
    TradingAgentsRuntime,
)

# The run being analyzed, and the wall-clock "today" it would be analyzed on.
# The gap is the whole point: every assertion below is about evidence from
# October reaching a decision dated March.
ANALYSIS_DATE = date(2026, 3, 12)


class _RecordingRunner:
    """A runner that records whether it was ever reached.

    The contaminated-roster cases assert on ``payloads`` being empty rather
    than on an exception alone: the exception is only worth anything if it is
    raised *before* the upstream subprocess spends money and returns a
    decision built on future data.
    """

    def __init__(self, output=None):
        self.output = output or _future_dated_output()
        self.payloads = []

    def run(self, payload, *, timeout_seconds, environment=None):
        self.payloads.append(payload)
        return self.output


def _future_dated_output():
    """A plausible bridge result whose evidence postdates the analysis date.

    This is what upstream actually returns today for a March analysis when the
    social analyst runs: a confident rating derived from StockTwits and Reddit
    content fetched in October, with nothing in the payload marking it stale.
    """
    return {
        "schema_version": "tradingagents-atl-v1",
        "manifest": {
            "symbol": "AAPL",
            "tradingagents_version": "0.3.1",
            "selected_analysts": ["market", "social"],
            "safe_config_sha256": "0" * 64,
        },
        "decision": {
            "analysis_date": ANALYSIS_DATE.isoformat(),
            "rating": "Buy",
            "atl_action": "BUY",
            "status": "valid",
            "attempts": 1,
            "raw_sha256": "a" * 64,
            "raw_excerpt": (
                "StockTwits 2026-10-26: ripping after the October print. "
                "Reddit r/wallstreetbets 2026-10-25: calls printing. "
                "FINAL TRANSACTION PROPOSAL: **BUY**"
            ),
        },
    }


def _context(*, symbols=("AAPL",), cutoff=ANALYSIS_DATE):
    return AgentRuntimeContext(
        timestamp=datetime(2026, 3, 13, 15, tzinfo=timezone.utc),
        backtest_start_date="2026-03-01",
        symbols=list(symbols),
        cash=1_000.0,
        total_equity=1_000.0,
        positions={},
        entry_prices={},
        current_prices={"AAPL": 50.0},
        latest_market_date_before_decision=cutoff,
    )


def _runtime(analysts, runner):
    return TradingAgentsRuntime(
        {"analysts": list(analysts)},
        runner=runner,
        environment={
            "COMMONSTACK_API_KEY": "test-key",
            "TRADINGAGENTS_PYTHON": "/usr/bin/python3",
        },
    )


# ---------------------------------------------------------------------------
# 1. The classification itself.
# ---------------------------------------------------------------------------

def test_only_market_is_as_of_safe():
    # Pins the audit's conclusion. If upstream gains an as-of path for one of
    # these, this is the test to update -- deliberately, by removing that
    # evidence entry, not by loosening the guard.
    assert TRADINGAGENTS_AS_OF_SAFE_ANALYSTS == ("market",)
    assert set(TRADINGAGENTS_LOOKAHEAD_EVIDENCE) == {
        "social",
        "news",
        "fundamentals",
    }


def test_safe_and_unsafe_partition_the_roster():
    # No analyst may be absent from both sets: an analyst nobody classified
    # would default to "allowed" and silently reopen the hole.
    assert set(TRADINGAGENTS_AS_OF_SAFE_ANALYSTS) | set(
        TRADINGAGENTS_LOOKAHEAD_EVIDENCE
    ) == set(TRADINGAGENTS_ANALYSTS)


def test_every_evidence_entry_names_a_tool_and_a_reason():
    for analyst, entries in TRADINGAGENTS_LOOKAHEAD_EVIDENCE.items():
        assert entries, analyst
        for tool, reason in entries:
            assert tool.strip(), analyst
            # The reason is what a reviewer reads to decide whether the
            # classification is still true; an empty one makes the table a
            # list of opinions.
            assert len(reason.strip()) > 40, (analyst, tool)


@pytest.mark.parametrize(
    "roster,expected",
    [
        (["market"], []),
        (["social"], ["social"]),
        (["news"], ["news"]),
        (["fundamentals"], ["fundamentals"]),
        # Order follows the canonical roster, not the caller's ordering.
        (["fundamentals", "social", "market"], ["social", "fundamentals"]),
        (
            ["market", "social", "news", "fundamentals"],
            ["social", "news", "fundamentals"],
        ),
        ([], []),
        (None, []),
    ],
)
def test_unsafe_analysts_are_reported_in_canonical_order(roster, expected):
    assert tradingagents_lookahead_unsafe_analysts(roster) == expected


def test_description_names_each_offending_tool():
    message = describe_tradingagents_lookahead(["market", "social", "fundamentals"])
    # The user has to be able to act on this, so every refused analyst and the
    # tool that caused it must appear.
    assert "social" in message
    assert "fundamentals" in message
    assert "fetch_stocktwits_messages" in message
    assert "fetch_reddit_posts" in message
    assert "get_fundamentals" in message
    # And the roster that would work.
    assert "market" in message
    # A clean roster produces no message at all, so callers can branch on it.
    assert describe_tradingagents_lookahead(["market"]) == ""


# ---------------------------------------------------------------------------
# 2. The guard: October evidence must never become a March decision.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unsafe", sorted(TRADINGAGENTS_LOOKAHEAD_EVIDENCE))
def test_historical_decision_refuses_each_contaminated_analyst(unsafe):
    runner = _RecordingRunner()
    runtime = _runtime(["market", unsafe], runner)

    with pytest.raises(TradingAgentsConfigurationError) as excinfo:
        runtime.decide(_context())

    assert unsafe in str(excinfo.value)
    # The decisive assertion: upstream was never invoked, so the October
    # evidence in _future_dated_output() could not have been paid for, let
    # alone turned into an order.
    assert runner.payloads == []


def test_future_dated_evidence_cannot_be_returned_as_a_valid_decision():
    # The scenario the requirement names: a March decision whose data source
    # answers with October content. The run must refuse rather than produce a
    # decision that downstream treats as ordinary.
    runner = _RecordingRunner(_future_dated_output())
    runtime = _runtime(["market", "social"], runner)

    with pytest.raises(TradingAgentsConfigurationError):
        runtime.decide(_context())

    assert runner.payloads == []
    # No audit row either: a refused configuration must not look like a call
    # that happened.
    assert runtime.decision_audit_rows == []
    assert runtime.calls == 0


def test_safe_roster_still_runs_and_reaches_upstream():
    # The guard must not be a blanket refusal -- the point is that `market`
    # keeps working, otherwise the runtime has no product left.
    runner = _RecordingRunner()
    runtime = _runtime(["market"], runner)

    result = runtime.decide(_context())

    assert len(runner.payloads) == 1
    assert runner.payloads[0]["selected_analysts"] == ["market"]
    assert "actions" in result


def test_refusal_is_a_configuration_error_so_the_engine_aborts():
    # Classed as configuration rather than transient on purpose: the roster is
    # identical on every trading day, so retrying it once per day would turn
    # one clear error into a long silent all-holds run that still bills.
    runner = _RecordingRunner()
    runtime = _runtime(["social"], runner)
    with pytest.raises(AgentRuntimeConfigurationError):
        runtime.decide(_context())
