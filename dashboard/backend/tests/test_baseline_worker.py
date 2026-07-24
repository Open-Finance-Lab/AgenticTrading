"""Baseline worker: dedup by config, atomic publish, overflow drop (T2)."""

import threading

import pytest

import dashboard.backend.database as db_module
from dashboard.backend.domain.backtesting import baseline_worker as bw


class _FakeBacktester:
    instances = 0

    def __init__(self, start, end, session_id, use_llm=False, mode="safe_trading"):
        type(self).instances += 1
        self.all_data = None

    def run_buyhold_baseline(self):
        return f"buyhold_{type(self).instances}", []

    def run_djia_baseline(self):
        return f"djia_{type(self).instances}", []


class _RecordingDb:
    def __init__(self):
        self.baseline_writes = []

    def update_run_baselines(self, run_id, *, djia_run_id=None, buyhold_run_id=None):
        self.baseline_writes.append((run_id, djia_run_id, buyhold_run_id))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    bw._reset_for_tests()
    _FakeBacktester.instances = 0
    fake_db = _RecordingDb()
    monkeypatch.setattr(db_module, "db", fake_db)
    monkeypatch.setattr(bw, "HourlyBacktester", _FakeBacktester)
    yield fake_db
    bw._reset_for_tests()


def _job(run_id, start="2026-04-15", end="2026-04-16", mode="safe_trading",
         publish=None):
    return bw.BaselineJob(
        run_id=run_id, session_id=f"sess_{run_id}", start_date=start,
        end_date=end, mode=mode, all_data={"AAPL": object()},
        publish=publish or (lambda ids: None),
    )


def test_same_config_jobs_run_baselines_once(_isolate):
    published = {}
    for i in range(3):
        assert bw.submit(_job(f"r{i}", publish=lambda ids, i=i: published.__setitem__(i, ids)))
    assert bw.wait_idle(10)
    assert _FakeBacktester.instances == 1                      # dedup: 1, not 3
    assert len(_isolate.baseline_writes) == 3                  # every run row linked
    assert published[0] == published[1] == published[2]
    assert published[0] == {"buy_and_hold": "buyhold_1", "djia": "djia_1"}
    assert published[0] is not published[1]                    # fresh dict per job


def test_distinct_configs_run_their_own_baselines():
    assert bw.submit(_job("a", start="2026-04-13", end="2026-04-14"))
    assert bw.submit(_job("b", start="2026-04-15", end="2026-04-16"))
    assert bw.wait_idle(10)
    assert _FakeBacktester.instances == 2


def test_job_failure_is_swallowed_and_worker_continues(capsys, monkeypatch):
    class _Boom(_FakeBacktester):
        def run_buyhold_baseline(self):
            raise RuntimeError("baseline blew up")

    monkeypatch.setattr(bw, "HourlyBacktester", _Boom)
    done = threading.Event()
    bw.submit(_job("bad"))
    assert bw.wait_idle(10)
    monkeypatch.setattr(bw, "HourlyBacktester", _FakeBacktester)
    bw.submit(_job("good", start="2026-05-01", end="2026-05-02",
                   publish=lambda ids: done.set()))
    assert bw.wait_idle(10) and done.is_set()   # worker survived the failure
    assert "Baseline generation failed" in capsys.readouterr().out


def test_full_queue_drops_job_with_print(capsys, monkeypatch):
    bw._reset_for_tests(maxsize=1)
    entered, release = threading.Event(), threading.Event()

    class _Blocking(_FakeBacktester):
        def run_buyhold_baseline(self):
            entered.set()
            release.wait(10)
            return "buyhold_x", []

    monkeypatch.setattr(bw, "HourlyBacktester", _Blocking)
    assert bw.submit(_job("one", start="2026-06-01", end="2026-06-02"))
    assert entered.wait(10)                     # worker busy on job one
    assert bw.submit(_job("two", start="2026-06-03", end="2026-06-04"))  # fills slot
    assert bw.submit(_job("three", start="2026-06-05", end="2026-06-06")) is False
    assert "Baseline queue full" in capsys.readouterr().out
    release.set()
    assert bw.wait_idle(10)
