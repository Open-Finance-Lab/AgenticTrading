"""The child publishes its pre-loop phases, so the card is never dark.

Before this, `_publish_live_progress` was the only writer of the progress
file and it ran only inside the bar loop. Everything before bar 1 -- imports,
schema DDL, the bar fetch, aggregation, indicators, then bar 1's full pipeline
-- was invisible to the card and unmeasured. Each phase write here carries the
phase in progress and the finished ones with timestamps, so the payload that
drives the card is also the measurement.
"""
import ast
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import pytz

from dashboard.backend.domain.backtesting import engine as engine_module
from dashboard.backend.domain.backtesting.currency import CurrencyContext
from dashboard.backend.domain.backtesting.engine import (
    PROGRESS_PHASES,
    HourlyBacktester,
)


def _bare(tmp_path, launched_at=None):
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_phases"
    # Needed by the cases that call `_publish_live_progress`, not by
    # `publish_phase`. The live writer's payload calls `_serialize_trades`
    # (engine.py:626 -> the def at :412), which resolves the currency context
    # at `:414` -- its second statement, and above the loop over trades
    # rather than inside it, so an empty trades list does not spare it. On a
    # __new__'d instance `_require_currency_context` (`:822`) raises
    # MarketDataUnavailableError. Verified by running it. Without this line
    # four cases below ERROR rather than run, one of them
    # (`test_a_terminal_phase_keeps_the_loops_last_numbers`) the only
    # guard on the `saving`-blanks-the-card regression this whole task exists
    # for -- so the headline fix would ship unpinned while the file looked
    # covered. `tests/backtesting/test_ifind_ashare_engine.py:258` sets the
    # same line for the same reason; this is the established pattern, not a
    # workaround.
    backtester.currency_context = CurrencyContext.identity("USD", "US/Eastern")
    backtester._init_progress_phases(launched_at)
    return backtester


def _payload(tmp_path):
    return json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))


def _manager():
    return SimpleNamespace(
        get_equity_curve=lambda: [],
        trades=[],
        rejected_orders=[],
        order_events=[],
    )


def _manager_with_curve():
    """A manager whose payload is worth losing -- the point of the terminal
    phase test below is that `saving` must not throw this away."""
    return SimpleNamespace(
        get_equity_curve=lambda: [
            {"timestamp": "2026-04-01T14:00:00", "equity": 1000.0, "cash": 0.0,
             "positions_value": 1000.0},
            {"timestamp": "2026-04-01T15:00:00", "equity": 1100.0, "cash": 0.0,
             "positions_value": 1100.0},
        ],
        trades=[],
        rejected_orders=[],
        order_events=[],
    )


def test_first_phase_closes_the_launch_gap(tmp_path):
    launched_at = time.time() - 5
    backtester = _bare(tmp_path, launched_at=launched_at)

    backtester.publish_phase("loading_bars")

    payload = _payload(tmp_path)
    assert payload["run_id"] == "agent_phases"
    assert payload["phase"] == "loading_bars"
    assert payload["step"] == 0
    assert payload["total_steps"] == 0
    assert payload["equity_curve"] == []
    assert [p["name"] for p in payload["phases"]] == ["starting"]
    assert payload["phases"][0]["started_at"] == launched_at
    assert payload["phases"][0]["ended_at"] >= launched_at + 5
    assert payload["phase_started_at"] == payload["phases"][0]["ended_at"]


def test_phases_accumulate_in_order_and_carry_the_bar_count(tmp_path):
    backtester = _bare(tmp_path)

    backtester.publish_phase("loading_bars")
    backtester.publish_phase("indicators")
    backtester.publish_phase("first_decision", total_steps=49)

    payload = _payload(tmp_path)
    assert payload["phase"] == "first_decision"
    assert payload["total_steps"] == 49
    assert [p["name"] for p in payload["phases"]] == ["loading_bars", "indicators"]
    for finished in payload["phases"]:
        assert finished["ended_at"] >= finished["started_at"]


def test_republishing_the_same_phase_does_not_close_it(tmp_path):
    backtester = _bare(tmp_path)
    backtester.publish_phase("loading_bars")
    started = _payload(tmp_path)["phase_started_at"]

    backtester.publish_phase("loading_bars")

    payload = _payload(tmp_path)
    assert payload["phases"] == []
    assert payload["phase_started_at"] == started


def test_live_progress_flips_to_running_and_keeps_the_history(tmp_path):
    backtester = _bare(tmp_path)
    backtester.publish_phase("first_decision", total_steps=49)

    backtester._publish_live_progress(1, 49, _manager())

    payload = _payload(tmp_path)
    assert payload["step"] == 1
    assert payload["total_steps"] == 49
    assert payload["phase"] == "running"
    assert [p["name"] for p in payload["phases"]] == ["first_decision"]

    backtester._publish_live_progress(2, 49, _manager())
    assert [p["name"] for p in _payload(tmp_path)["phases"]] == ["first_decision"]


def test_saving_closes_running(tmp_path):
    backtester = _bare(tmp_path)
    backtester._publish_live_progress(49, 49, _manager())

    backtester.publish_phase("saving")

    payload = _payload(tmp_path)
    assert payload["phase"] == "saving"
    assert payload["phases"][-1]["name"] == "running"
    assert payload["total_steps"] == 49


def test_unknown_phase_is_refused(tmp_path):
    backtester = _bare(tmp_path)
    with pytest.raises(ValueError):
        backtester.publish_phase("warming_up")
    assert PROGRESS_PHASES == (
        "starting",
        "loading_bars",
        "indicators",
        "first_decision",
        "running",
        "saving",
    )


@pytest.mark.parametrize("launched_at", [0, 0.0], ids=["zero-int", "zero-float"])
def test_launch_time_zero_is_a_time_not_an_absence(tmp_path, launched_at):
    """`--launched-at 0` parses to 0.0 through argparse's type=float, and 0.0
    is falsy. A truthiness gate dropped the entire `starting` phase for it --
    the one window this plan can only measure, never narrate -- with no error
    and no log line. The gate is `is not None`, and deliberately nothing more:
    an absurd launch time now yields an absurdly long `starting` row in
    phases[], which is visible and therefore fixable. Range-checking it would
    restore the silent drop under a different name."""
    backtester = _bare(tmp_path, launched_at=launched_at)

    backtester.publish_phase("loading_bars")

    payload = _payload(tmp_path)
    assert [p["name"] for p in payload["phases"]] == ["starting"]
    assert payload["phases"][0]["started_at"] == 0.0


def test_legacy_new_construction_still_publishes_live_progress(tmp_path):
    """Callers that build the engine with __new__ (tests, legacy tools) never
    ran _init_progress_phases; the live writer must not care."""
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_legacy"
    # Set here too rather than routed through `_bare`: this case's whole point
    # is that nothing called `_init_progress_phases`, so it cannot use the
    # helper. See the comment there for why `_publish_live_progress` raises
    # without it.
    backtester.currency_context = CurrencyContext.identity("USD", "US/Eastern")

    backtester._publish_live_progress(3, 10, _manager())

    payload = _payload(tmp_path)
    assert payload["step"] == 3
    assert payload["phase"] == "running"
    assert payload["phases"] == []


def test_no_progress_file_means_no_write_and_no_error(tmp_path):
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = None
    backtester.live_run_id = None
    backtester._init_progress_phases(None)

    backtester.publish_phase("loading_bars")

    assert not (tmp_path / "progress.json").exists()


def test_a_missing_progress_file_attribute_is_not_an_error(tmp_path):
    """Absent is not None, and only this case covers absent.

    The case above sets `progress_file = None`, which a bare
    `if not self.progress_file` handles fine. Step 6 makes `publish_phase` the
    first statement of `load_data`, and `load_data` is documented as usable on
    an instance built with `__new__` (engine.py:647-648) -- where the attribute
    was never assigned at all, because `__init__` assigns it and the class does
    not carry it. That caller exists today:
    test_market_data_errors.py::test_engine_load_data_empty_raises_not_exits
    sets data_loader, start_date, end_date, data_source and symbols, and
    nothing else. Without the getattr in `publish_phase` it fails with an
    AttributeError that names neither this task nor the
    MarketDataUnavailableError it is actually asserting.
    """
    backtester = object.__new__(HourlyBacktester)

    backtester.publish_phase("loading_bars")

    # The phase clock still ran: it is state, not a side effect of writing.
    assert backtester._progress_phase == "loading_bars"
    assert list(tmp_path.iterdir()) == []


def test_write_failure_is_reported_not_raised(tmp_path, capsys):
    backtester = _bare(tmp_path)
    backtester.progress_file = str(tmp_path / "missing" / "progress.json")

    backtester.publish_phase("loading_bars")

    assert "Could not write live progress" in capsys.readouterr().out


def test_a_terminal_phase_keeps_the_loops_last_numbers(tmp_path):
    """`saving` is published AFTER the loop, and a payload of zeros is a lie
    there.

    The pre-loop skeleton (`step: 0`, `equity_curve: []`) is honest before bar
    1 and destructive after the last one: the Backtest panel computes its bar
    straight off this file (`stepPct`, app.js:8600 -- the 1s poller, one of
    two such sites; Task 4 gives both one helper) and snaps 99% -> 0%, and
    the My Agents fold *replaces* its stored entry (app.js:8609), so the
    sparkline, the equity label and `49/49` all blank at the finish line of
    every run -- for the whole baseline/persistence tail, with nothing red
    anywhere. Asserting `phase == "saving"` cannot see any of that; asserting
    what the payload still carries can.
    """
    backtester = _bare(tmp_path)
    backtester._publish_live_progress(49, 49, _manager_with_curve())

    backtester.publish_phase("saving")

    payload = _payload(tmp_path)
    assert payload["phase"] == "saving"
    assert payload["step"] == 49
    assert payload["total_steps"] == 49
    assert [point["equity"] for point in payload["equity_curve"]] == [1000.0, 1100.0]
    # Keys the skeleton does not have at all: their presence is the proof the
    # last payload was carried rather than rebuilt.
    assert "trades" in payload
    assert "order_events_count" in payload


def test_a_pre_loop_phase_still_publishes_the_skeleton(tmp_path):
    """The other half of the rule above: with nothing published yet there is
    nothing to carry, and inventing a step here would give the card an ETA
    anchor before the loop exists."""
    backtester = _bare(tmp_path)

    backtester.publish_phase("loading_bars")

    payload = _payload(tmp_path)
    assert payload["step"] == 0
    assert payload["equity_curve"] == []
    assert "trades" not in payload


def test_a_file_less_engine_still_closes_first_decision():
    """The phase clock is state, not a side effect of writing.

    `_set_progress_phase` runs *above* `_publish_live_progress`'s
    `if not self.progress_file: return` (engine.py:590). Below that return an
    engine with no progress file -- every CLI run, the external-run session,
    the algo service -- never leaves `first_decision`, and the next transition
    closes a `first_decision` spanning the entire bar loop: the one number this
    phase exists to produce, "how long until the first model decision?",
    reported as hours instead of seconds, on the path where the stdout line is
    the only reader there is.
    """
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = None
    backtester.live_run_id = None
    backtester._init_progress_phases(None)
    backtester.publish_phase("first_decision", total_steps=2)

    backtester._publish_live_progress(1, 2, _manager())
    backtester._publish_live_progress(2, 2, _manager())
    backtester.publish_phase("saving")

    assert [p["name"] for p in backtester._progress_phases] == [
        "first_decision",
        "running",
    ]


def test_the_starting_record_splits_what_the_child_could_not_write(tmp_path):
    """One `starting` phase, four numbers -- and MISSING is not zero.

    An undifferentiated `starting` cannot say whether the start is dominated by
    process spawn, by pandas and the SDK imports, or by store construction and
    DDL, so it cannot say whether Task 5 helped. The phase *name* stays one
    name; the *record* carries the boundaries. A mark nobody passed is absent
    rather than 0.0, because an unmeasured phase and a free one are different
    facts. `schema_init_seconds` is the deliberate exception: present and 0.0
    in a worker, which is the evidence the flag fired.
    """
    launched_at = time.time() - 6
    backtester = object.__new__(HourlyBacktester)
    backtester.progress_file = str(tmp_path / "progress.json")
    backtester.live_run_id = "agent_split"
    backtester._init_progress_phases(
        launched_at,
        startup_clock={
            "child_entered_at": launched_at + 0.4,
            "imports_done_at": launched_at + 3.1,
            "schema_init_seconds": 0.0,
        },
    )

    backtester.publish_phase("loading_bars")
    payload = _payload(tmp_path)
    starting = payload["phases"][0]
    assert starting["name"] == "starting"
    assert starting["started_at"] == launched_at
    assert starting["child_entered_at"] == launched_at + 0.4
    assert starting["imports_done_at"] == launched_at + 3.1
    assert starting["schema_init_seconds"] == 0.0
    # Only `starting` carries them: it alone describes an interval the child
    # could not write to. Anywhere else they would be three constants repeated.
    # `duration_seconds` is the one key every record carries -- it is the
    # steady-clock measurement, not a startup-clock stamp.
    backtester.publish_phase("indicators")
    later = _payload(tmp_path)["phases"][1]
    assert later["name"] == "loading_bars"
    assert set(later) == {"name", "started_at", "ended_at", "duration_seconds"}

    # An in-process engine passes nothing, so nothing is claimed.
    bare = _bare(tmp_path, launched_at=time.time() - 2)
    bare.publish_phase("loading_bars")
    assert set(_payload(tmp_path)["phases"][0]) == {
        "name",
        "started_at",
        "ended_at",
        "duration_seconds",
    }


def test_a_wall_clock_step_cannot_distort_an_in_child_phase_duration(
    tmp_path, monkeypatch
):
    """REGRESSION (#509). Every phase duration used to be a difference of two
    `time.time()` reads, which is not monotonic: an NTP correction between the
    two distorts the interval and a backward step makes it NEGATIVE. These
    numbers are the measurement the next latency decision is made on, so a
    silently wrong one is worse than a missing one.

    Simulated by stepping the wall clock backwards a full minute mid-run --
    exactly what an NTP correction on a long-running instance does.
    """
    bare = _bare(tmp_path, launched_at=None)
    bare.publish_phase("loading_bars")
    fake_now = time.time()
    monkeypatch.setattr(engine_module, "wall_clock", lambda: fake_now - 60)
    bare.publish_phase("indicators")

    finished = _payload(tmp_path)["phases"][0]
    assert finished["name"] == "loading_bars"
    assert finished["duration_seconds"] >= 0.0
    assert finished["duration_seconds"] < 5.0
    # The wall-clock instants keep the stepped value on purpose: they exist to
    # be correlated against log lines, which took the same step.
    assert finished["ended_at"] - finished["started_at"] < -50


def test_the_starting_phase_is_still_measured_on_the_wall_clock(tmp_path):
    """`starting` cannot use the steady clock and this is not a shortcut.

    It is `child_entered_at - <the parent's --launched-at>`: a genuinely
    cross-process interval, and `monotonic()` has a per-process epoch, so the
    parent's reading and the child's are not comparable at all. The wall clock
    is the only shared reference, which is why `_init_progress_phases` leaves
    the steady mark None for exactly this phase.
    """
    launched_at = time.time() - 6
    bare = _bare(tmp_path, launched_at=launched_at)
    assert bare._progress_phase == "starting"
    assert bare._progress_phase_started_steady is None
    bare.publish_phase("loading_bars")

    starting = _payload(tmp_path)["phases"][0]
    assert starting["name"] == "starting"
    assert starting["duration_seconds"] == pytest.approx(
        starting["ended_at"] - starting["started_at"]
    )
    assert starting["duration_seconds"] == pytest.approx(6, abs=1)
    # And the phase opened after it does get a steady mark.
    assert bare._progress_phase_started_steady is not None


def test_every_phase_is_published_where_the_work_happens():
    """The four marking sites of Step 6, pinned -- and nothing else here can.

    Every unit case above builds the engine with `object.__new__` and calls
    `publish_phase` by hand, so not one of them can see whether the *engine*
    ever calls it. Delete all four marks in Step 6 and every one of them stays
    green while the card goes dark again, which is the whole defect this task
    exists to fix.

    Position, not just presence. `load_data`'s mark has to be its first
    statement or the phase opens *after* the fetch it is naming, and a run
    that spends ninety seconds inside `fetch_bars` is exactly the run the card
    is dark for. That is the half the drive test below cannot check: it sees
    that `loading_bars` was published, not that it was published before the
    fetch.

    AST, not `"publish_phase" in source`: a substring test passes on a call
    sitting in a comment, in a docstring, or in a different method entirely.

    This guard and `test_a_fake_loader_drives_every_phase_in_order` below are
    a pair, and the spec's Testing section asks for both halves: this one
    reads the source and can pin *where* a mark sits but never that it runs;
    that one runs the real methods over a fake tape and can pin the order, the
    bar count and the carry but never the position. Neither replaces the
    other. Do not delete this one on the grounds that the drive test covers
    the marks.
    """
    tree = ast.parse(Path(engine_module.__file__).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HourlyBacktester"
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }

    def published(method):
        """(phase name, top-level statement index) per self.publish_phase call."""
        found = []
        for index, statement in enumerate(method.body):
            for node in ast.walk(statement):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "publish_phase"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    found.append((node.args[0].value, index))
        return found

    # "First statement" means first after the docstring, which is itself an
    # AST statement. Both methods have one, and Step 6 puts the mark below it.
    def first_body_index(method):
        return 1 if ast.get_docstring(method) is not None else 0

    for name, phase in (
        ("load_data", "loading_bars"),
        ("calculate_indicators", "indicators"),
    ):
        marks = published(methods[name])
        assert marks == [(phase, first_body_index(methods[name]))], (
            f"{name} must publish {phase!r} as its first statement after the "
            f"docstring, and nowhere else; found {marks}"
        )

    # `run_agent_backtest` marks two points inside its body rather than at the
    # top, so order is what can be pinned: `first_decision` only once the bar
    # count exists (there is nothing to report before it), and `saving` before
    # the agent's own row is written, which is the write the phase names.
    run = methods["run_agent_backtest"]
    marks = published(run)
    assert [phase for phase, _ in marks] == ["first_decision", "saving"], marks
    (_, first_decision_at), (_, saving_at) = marks
    total_steps_at = min(
        index
        for index, statement in enumerate(run.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "total_steps"
            for target in statement.targets
        )
    )
    insert_run_at = min(
        index
        for index, statement in enumerate(run.body)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert_run"
    )
    assert total_steps_at < first_decision_at < saving_at < insert_run_at, (
        total_steps_at,
        first_decision_at,
        saving_at,
        insert_run_at,
    )


def test_a_fake_loader_drives_every_phase_in_order(tmp_path, monkeypatch):
    """The spec's drive test: the real methods, over a fake tape, in order.

    Spec, Testing/Track A: "a fake loader drives `load_data` ->
    `calculate_indicators` -> the first bar, and the progress file is asserted
    at each phase (`phase`, `phases`, `total_steps` known before the first
    decision)". This is that case; the AST guard above is its other half, for
    the one thing this cannot see (see that docstring).

    Four claims live only here:

    1. The marks are *reached*. The guard proves they are in the source at the
       right statement index; a mark inside a method nothing calls satisfies
       it just as well.
    2. The names arrive in the documented order at runtime, with `running`
       interleaved by `_publish_live_progress` rather than by `publish_phase`
       -- so the order is a property of two writers agreeing, which no
       single-writer unit case above exercises.
    3. The bar count published with `first_decision` is the real one. Source
       order only says `total_steps = ...` precedes the call; it says nothing
       about the value reaching the file, and `0/N` on the card is that value.
       Pinned against the number of decisions the loop actually asks for.
    4. `saving` carries the loop's last payload in a *real* run.
       `test_a_terminal_phase_keeps_the_loops_last_numbers` pins the same rule
       against a hand-built manager, which cannot catch a carry broken by
       something the real loop does to the payload between the last step and
       the terminal mark.

    The fakes follow `tests/backtesting/test_engine_move.py:42-116`, the
    established loader/db pair for driving the real engine with no network and
    no real DB, and the 70-bar tape is the one its own
    `test_run_agent_backtest_smoke` (`:230`) already runs end to end -- 70
    bars in, 70 decision bars out, one decision each. They are copied rather
    than imported: that module imports
    `dashboard.scripts.backtest_hourly_agent` at module scope, and a case that
    goes red when an unrelated file renames a fixture points at the wrong
    place.
    """

    class _Loader:
        def __init__(self, bars):
            self.bars = bars

        def fetch_bars(self, symbols, start_date, end_date):
            return {symbol: self.bars[symbol] for symbol in symbols}

    class _DB:
        def insert_run(self, **kwargs):
            pass

        def insert_equity_points(self, run_id, points):
            pass

        def insert_trades(self, run_id, trades):
            pass

        def insert_decisions(self, run_id, decisions):
            pass

    eastern = pytz.timezone("US/Eastern")
    stamps = []
    day = datetime(2026, 3, 2)  # a Monday
    while len(stamps) < 70:
        if day.weekday() < 5:
            stamps.extend(
                eastern.localize(datetime(day.year, day.month, day.day, hour, 0))
                for hour in range(10, 16)
            )
        day += timedelta(days=1)
    stamps = stamps[:70]
    prices = [100.0 + ((index % 7) - 3) * 0.5 + index * 0.1 for index in range(70)]
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": prices,
            "volume": [1000] * 70,
        },
        index=pd.DatetimeIndex(stamps),
    )
    monkeypatch.setattr(
        engine_module,
        "create_market_data_provider",
        lambda data_source="alpaca", universe=None: _Loader(
            {"AAPL": frame, "MSFT": frame.copy()}
        ),
    )
    monkeypatch.setattr(engine_module, "db", _DB())

    decisions_asked = []
    at_first_decision = []
    real_decision = engine_module.PortfolioManager.make_trading_decision

    def snapshotting_decision(self, state):
        # The loop asks for the decision (`engine.py:1499`) before it publishes
        # the step (`:1621`), so this hook is the only point from which the
        # `first_decision` payload -- the one the card renders as `0/N` -- can
        # be read while it is still what the file says. It delegates rather
        # than scripting a reply: the subject is the phase machinery, and a
        # scripted decision would quietly change which branches of the loop run.
        decisions_asked.append(state["timestamp"])
        if not at_first_decision:
            at_first_decision.append(_payload(tmp_path))
        return real_decision(self, state)

    monkeypatch.setattr(
        engine_module.PortfolioManager,
        "make_trading_decision",
        snapshotting_decision,
    )

    launched_at = time.time() - 4
    backtester = HourlyBacktester(
        "2026-03-01",
        "2026-04-01",
        "phase-drive",
        use_llm=False,
        symbols=["AAPL", "MSFT"],
        live_run_id="agent_drive",
        progress_file=str(tmp_path / "progress.json"),
        launched_at=launched_at,
    )
    # Construction alone writes nothing: `_init_progress_phases` only opens
    # `starting` in memory, so a run that dies before `load_data` leaves no
    # half-built payload on disk for the poller to read.
    assert not (tmp_path / "progress.json").exists()

    backtester.load_data()
    after_load = _payload(tmp_path)
    assert after_load["run_id"] == "agent_drive"
    assert after_load["phase"] == "loading_bars"
    assert [p["name"] for p in after_load["phases"]] == ["starting"]
    assert after_load["phases"][0]["started_at"] == launched_at
    assert after_load["step"] == 0
    assert after_load["total_steps"] == 0
    assert after_load["equity_curve"] == []

    backtester.calculate_indicators()
    after_indicators = _payload(tmp_path)
    assert after_indicators["phase"] == "indicators"
    assert [p["name"] for p in after_indicators["phases"]] == [
        "starting",
        "loading_bars",
    ]
    # MUTATION TEST: delete `record_phase_metric("fetch_seconds", ...)` from
    # load_data and this fails. That line is the only PRODUCTION write of the
    # fetch/aggregate split -- the measurement the design gates "is caching
    # the aggregated output worth a second change?" on -- and every other
    # case covering the metric calls `record_phase_metric` by hand on an
    # `object.__new__` instance, so not one of them can see the engine drop it.
    loading_bars = after_indicators["phases"][1]
    assert "fetch_seconds" in loading_bars, (
        "engine.load_data must record the fetch/aggregate split"
    )
    assert 0.0 <= loading_bars["fetch_seconds"] <= (
        loading_bars["ended_at"] - loading_bars["started_at"]
    )

    run_id, equity_curve = backtester.run_agent_backtest()
    assert run_id and equity_curve

    assert at_first_decision, "the loop never asked for a decision"
    opening = at_first_decision[0]
    assert opening["phase"] == "first_decision"
    assert opening["step"] == 0
    assert [p["name"] for p in opening["phases"]] == [
        "starting",
        "loading_bars",
        "indicators",
    ]
    # Claim 3: the N the card shows before step 1 is the count of steps the
    # loop goes on to take, not a placeholder that happens to be non-zero.
    assert opening["total_steps"] == len(decisions_asked) > 0

    final = _payload(tmp_path)
    assert final["phase"] == "saving"
    assert [p["name"] for p in final["phases"]] == [
        "starting",
        "loading_bars",
        "indicators",
        "first_decision",
        "running",
    ]
    # Claim 4. `step`/`total_steps` survive the terminal mark, and the two keys
    # the pre-loop skeleton does not have at all are the proof the last live
    # payload was carried rather than rebuilt.
    assert final["step"] == final["total_steps"] == opening["total_steps"]
    assert final["equity_curve"]
    assert "trades" in final
    assert "order_events_count" in final


def test_record_phase_metric_lands_on_the_phase_that_was_open():
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=100.0)
    engine.record_phase_metric("fetch_seconds", 12.5)
    engine.publish_phase("loading_bars")
    engine.record_phase_metric("fetch_seconds", 3.25)
    engine.publish_phase("indicators")
    by_name = {entry["name"]: entry for entry in engine._progress_phases}
    assert by_name["starting"]["fetch_seconds"] == 12.5
    assert by_name["loading_bars"]["fetch_seconds"] == 3.25


def test_phase_extras_do_not_leak_into_the_next_phase():
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=0.0)
    engine.publish_phase("loading_bars")
    engine.record_phase_metric("fetch_seconds", 1.0)
    engine.publish_phase("indicators")
    engine.publish_phase("first_decision")
    by_name = {entry["name"]: entry for entry in engine._progress_phases}
    assert "fetch_seconds" in by_name["loading_bars"]
    assert "fetch_seconds" not in by_name["indicators"]


def test_record_phase_metric_tolerates_an_uninitialised_instance():
    """Every accessor in this block tolerates an instance built with __new__
    that never ran _init_progress_phases."""
    engine = object.__new__(HourlyBacktester)
    engine.record_phase_metric("fetch_seconds", 2.0)
    engine.publish_phase("loading_bars")  # must not raise


def test_a_metric_recorded_with_no_phase_open_lands_nowhere():
    """MUTATION TEST: drop the `_progress_phase is None` gate in
    record_phase_metric and this must fail. With no launch time nothing is
    open before the first publish_phase; a number recorded then has no owner,
    and generalising the extras merge would otherwise hand it to the first
    phase that closes -- `loading_bars`, which did not incur it."""
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=None)
    engine.record_phase_metric("fetch_seconds", 9.0)
    engine.publish_phase("loading_bars")
    engine.publish_phase("indicators")
    by_name = {entry["name"]: entry for entry in engine._progress_phases}
    assert "starting" not in by_name
    assert "fetch_seconds" not in by_name["loading_bars"]


def test_a_startup_clock_without_a_launch_time_does_not_leak_onto_loading_bars():
    """MUTATION TEST: seed `_progress_phase_extra` from `startup_clock`
    unconditionally in _init_progress_phases and this must fail.
    backtest_hourly_agent.py always passes startup_clock but launched_at only
    from --launched-at, which a bare CLI run omits. Today those three keys are
    simply dropped (`starting` never closes); once extras belong to whichever
    phase closes, they would surface on `loading_bars` as if it had a spawn
    time and a schema DDL cost."""
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(
        launched_at=None,
        startup_clock={
            "child_entered_at": 1.0,
            "imports_done_at": 3.0,
            "schema_init_seconds": 0.0,
        },
    )
    engine.publish_phase("loading_bars")
    engine.publish_phase("indicators")
    by_name = {entry["name"]: entry for entry in engine._progress_phases}
    assert set(by_name) == {"loading_bars"}
    assert not {"child_entered_at", "imports_done_at", "schema_init_seconds"} & set(
        by_name["loading_bars"]
    )


def test_the_starting_breakdown_line_is_unchanged(capsys):
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(
        launched_at=0.0,
        startup_clock={
            "child_entered_at": 1.0,
            "imports_done_at": 3.0,
            "schema_init_seconds": 0.0,
        },
    )
    engine.publish_phase("loading_bars")
    out = capsys.readouterr().out
    assert "spawn+interpreter 1.00s" in out
    assert "imports+stores 2.00s" in out
    assert "schema DDL 0.00s" in out


def test_closing_loading_bars_prints_the_fetch_split(capsys):
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=0.0)
    engine.publish_phase("loading_bars")
    engine.record_phase_metric("fetch_seconds", 0.0)
    engine.publish_phase("indicators")
    out = capsys.readouterr().out
    assert "fetch 0.00s" in out
    assert "aggregate+verify" in out


def test_no_split_line_without_the_metric(capsys):
    engine = object.__new__(HourlyBacktester)
    engine._init_progress_phases(launched_at=0.0)
    engine.publish_phase("loading_bars")
    engine.publish_phase("indicators")
    assert "aggregate+verify" not in capsys.readouterr().out
