"""Live board: the 24h trade memory survives the nightly process boundary.

Two nights through ``deploy_live_model_increment`` with the real
``LLMAgentStrategy`` decision loop and the real ``PortfolioManager``; only the
bar fetch and the model are replaced. Each night builds a fresh strategy and a
fresh manager, as the nightly job does, so whatever night 2 knows about night 1
arrived through the checkpoint row in ``agent_runs``.
"""

import json
from datetime import date, datetime

import pandas as pd

from dashboard.backend.database import db
from dashboard.backend.domain.backtesting import portfolio_manager as pm_mod
import dashboard.backend.domain.leaderboard.live as live
from dashboard.backend.domain.leaderboard.live_trade_stats import TRADE_STATS_KEY
import dashboard.backend.domain.leaderboard.service as lb_service
from dashboard.backend.domain.leaderboard.strategies import llm_agent
from dashboard.backend.tests.test_live_leaderboard import (  # noqa: F401 - fixture
    _ET,
    _deepseek,
    _insert_checkpoint,
    _isolated_live_session,
    _month_rows,
    _one_bar_frame,
)

# Open stamps of the six decision bars, 10:00-15:00 ET in August (UTC-4).
_DECISION_HOURS_UTC = (14, 15, 16, 17, 18, 19)


class _ReplyByClock:
    """A model that answers each decision by the prompt's timestamp.

    ``_capture`` makes the prompt text the snapshot's timestamp, so the
    script can say what to do at which bar; every other bar holds.
    """

    def __init__(self, actions_at):
        actions_by_stamp = actions_at

        class _Messages:
            @staticmethod
            def create(**kwargs):
                stamp = kwargs["messages"][0]["content"]
                actions = actions_by_stamp.get(
                    stamp, [{"symbol": "AAPL", "action": "hold", "confidence": 0.9}]
                )
                block = type("Block", (), {"text": json.dumps({"actions": actions})})()
                usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 5})()
                return type("Reply", (), {"content": [block], "usage": usage})()

        self.messages = _Messages()


class _OneSymbolAgent(llm_agent.LLMAgentStrategy):
    """The real decision loop over one flat-priced symbol and a scripted model.

    Only ``run`` is replaced, to skip the bar download and indicator build;
    the loop, the manager, the snapshot and the session-close hook are real.
    """

    def __init__(self, entry, actions_at):
        super().__init__(entry)
        self._client = _ReplyByClock(actions_at)

    def required_symbols(self):
        return ["AAPL"]

    def run(self, bars, start_date, end_date, initial_capital,
            starting_snapshot=None, on_session_close=None):
        days = live._trading_days_inclusive(
            date.fromisoformat(start_date), date.fromisoformat(end_date)
        )
        stamps = [
            pd.Timestamp(f"{day.isoformat()}T{hour}:00:00Z")
            for day in days
            for hour in _DECISION_HOURS_UTC
        ]
        frame = pd.DataFrame(
            {"close": [100.0] * len(stamps)}, index=pd.DatetimeIndex(stamps)
        )
        self.used_llm = True
        self.planned_decision_steps = len(stamps)
        manager = self._run_decision_loop(
            client=self._client,
            timestamps=stamps,
            symbols=["AAPL"],
            data={"AAPL": frame},
            price_cache={},
            initial_capital=initial_capital,
            model_id=self.model_id,
            starting_snapshot=starting_snapshot,
            on_session_close=on_session_close,
        )
        self._record_progress(manager, decision_steps=len(stamps))
        return manager.get_equity_curve()


def _capture(monkeypatch):
    """Record every prompt snapshot by its decision timestamp."""
    prompts = {}

    def capture(snapshot, **_kwargs):
        prompts[snapshot["timestamp"]] = snapshot
        return snapshot["timestamp"]

    monkeypatch.setattr(pm_mod, "create_prompt", capture)
    return prompts


def _night(monkeypatch, day, actions_at):
    """One nightly increment as of 17:39 ET on Aug ``day``, in a fresh strategy."""
    freeze_cfg = live.live_freeze_config(datetime(2026, 8, day, 17, 39, tzinfo=_ET))
    entry = _deepseek(freeze_cfg)
    agent = _OneSymbolAgent(entry, actions_at)
    monkeypatch.setattr(lb_service, "get_strategy", lambda _entry: agent)
    monkeypatch.setattr(lb_service, "fetch_hourly_bars", lambda *args: _one_bar_frame())
    row = live.deploy_live_model_increment(entry, freeze_cfg)
    return agent, row


def _buy(shares):
    return [{"symbol": "AAPL", "action": "buy", "confidence": 0.9, "position_size": shares}]


def _shown(snapshot):
    return [(t["side"], t["shares"], t["timestamp"]) for t in snapshot["recent_trades"]]


def _two_nights(monkeypatch):
    """Night 1 (Aug 26) buys 10 at 10:00 and 10 at 14:00 ET; night 2 (Aug 27)
    sells all 20 at 11:00 ET. Each resumes from a checkpoint and trades one
    session. Returns the prompts, both nights, and night 1's stored snapshot
    (read before night 2 supersedes its row)."""
    # A checkpoint from before the memory existed: night 1 resumes from it.
    _insert_checkpoint("2026-08-25", "08-25")
    prompts = _capture(monkeypatch)
    night1 = _night(monkeypatch, 26, {
        "2026-08-26T14:00:00+00:00": _buy(10),
        "2026-08-26T18:00:00+00:00": _buy(10),
    })
    stored1 = live._snapshot_from_run(db.get_run(night1[1]["run_id"]))
    night2 = _night(monkeypatch, 27, {
        "2026-08-27T15:00:00+00:00": [
            {"symbol": "AAPL", "action": "sell", "confidence": 0.9},
        ],
    })
    assert night1[1]["segment"] == {"start_date": "2026-08-26", "end_date": "2026-08-26"}
    assert night2[1]["segment"] == {"start_date": "2026-08-27", "end_date": "2026-08-27"}
    return prompts, night1, night2, stored1


def test_night_two_prompts_see_night_ones_trades_for_24_hours(monkeypatch):
    prompts, _night1, _night2, _stored1 = _two_nights(monkeypatch)
    # 10:00 ET: night 1's 14:00 ET buy is 20h old and shown; its 10:00 ET buy
    # is exactly 24h old and is not (the window is strict).
    assert _shown(prompts["2026-08-27T14:00:00+00:00"]) == [
        ("BUY", 10.0, "2026-08-26T18:00:00+00:00"),
    ]
    # After night 2's own sell, the restored buy still comes first.
    assert _shown(prompts["2026-08-27T16:00:00+00:00"]) == [
        ("BUY", 10.0, "2026-08-26T18:00:00+00:00"),
        ("SELL", 20.0, "2026-08-27T15:00:00+00:00"),
    ]
    # 14:00 ET: night 1's last buy is now 24h old as well.
    assert _shown(prompts["2026-08-27T18:00:00+00:00"]) == [
        ("SELL", 20.0, "2026-08-27T15:00:00+00:00"),
    ]


def test_restored_trades_are_never_counted_as_new_trades(monkeypatch):
    _prompts, (night1, _row1), (night2, row2), _stored1 = _two_nights(monkeypatch)
    assert night1.num_trades() == 2
    # Night 2 traded once; the two buys it remembers are not its trades.
    assert night2.num_trades() == 1
    assert [t["side"] for t in night2.trades] == ["SELL"]
    stored = db.get_run(row2["run_id"])
    assert stored["num_trades"] == 3
    stats = stored["metadata"][TRADE_STATS_KEY]
    assert (stats["buys"], stats["sells"]) == (2, 1)
    assert [r["run_id"] for r in _month_rows("deepseek_v4_pro")] == [row2["run_id"]]


def test_each_checkpoint_round_trips_the_memory_through_agent_runs(monkeypatch):
    _prompts, _night1, (_night2, row2), stored1 = _two_nights(monkeypatch)
    assert stored1["recent_trades"] == [
        {
            "symbol": "AAPL",
            "side": "BUY",
            "shares": 10.0,
            "price": 100.0,
            "timestamp": "2026-08-26T14:00:00+00:00",
        },
        {
            "symbol": "AAPL",
            "side": "BUY",
            "shares": 10.0,
            "price": 100.0,
            "timestamp": "2026-08-26T18:00:00+00:00",
        },
    ]
    # Night 2's row carries the tail anchored on its own sell: the 10:00 ET
    # buy from night 1 is more than 24h older than that and is dropped.
    stored2 = live._snapshot_from_run(db.get_run(row2["run_id"]))
    assert [(t["side"], t["timestamp"]) for t in stored2["recent_trades"]] == [
        ("BUY", "2026-08-26T18:00:00+00:00"),
        ("SELL", "2026-08-27T15:00:00+00:00"),
    ]


def test_the_checkpoint_row_records_where_the_memory_started(monkeypatch):
    _prompts, (night1, _row1), (_night2, row2), stored1 = _two_nights(monkeypatch)
    # Night 1 resumed from a checkpoint written before the marker existed, so
    # the memory starts at its first decision bar, 10:00 ET on Aug 26.
    first_bar = "2026-08-26T14:00:00+00:00"
    snap1 = night1.last_portfolio_snapshot
    assert (snap1["trade_memory"], snap1["trade_memory_since"]) == (1, first_bar)
    assert (stored1["trade_memory"], stored1["trade_memory_since"]) == (1, first_bar)
    # Night 1's row is superseded and deleted. The one row left still says the
    # month joined a night without memory (Aug 26) to nights with it.
    assert [r["run_id"] for r in _month_rows("deepseek_v4_pro")] == [row2["run_id"]]
    stored2 = db.get_run(row2["run_id"])["metadata"][live.LIVE_SNAPSHOT_KEY]
    assert (stored2["trade_memory"], stored2["trade_memory_since"]) == (1, first_bar)
