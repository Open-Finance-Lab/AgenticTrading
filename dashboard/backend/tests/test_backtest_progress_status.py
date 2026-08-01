"""Progress-file freshness, so the UI can tell 'working' from 'stuck'.

The status payload already carried step/total_steps; what it could not answer
was whether those numbers were current. A run whose subprocess wedges keeps
reporting its last step forever, which reads identically to steady progress.
"""

import json
import os
import time

from dashboard.backend.api.routers import backtests


def _seed(tmp_path, monkeypatch, name="backtest_progress_test.json"):
    progress_file = tmp_path / name
    progress_file.write_text(json.dumps({"step": 7, "total_steps": 240}), encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    return progress_file


def test_progress_carries_the_file_mtime(tmp_path, monkeypatch):
    progress_file = _seed(tmp_path, monkeypatch)

    payload = backtests._read_backtest_progress()

    assert payload["step"] == 7
    assert payload["total_steps"] == 240
    assert payload["progress_updated_at"] == progress_file.stat().st_mtime
    assert payload["progress_updated_at"] <= time.time() + 1


def test_progress_carries_a_server_computed_age(tmp_path, monkeypatch):
    """The age, not just the timestamp, is what the browser reads.

    Differencing the mtime against the client clock makes any machine more than
    the staleness threshold out of step indistinguishable from a wedged run: a
    fast clock pins a permanent "No progress for 47m" onto a healthy backtest, a
    slow one suppresses the warning forever. Both ends of this subtraction are
    read in one process, so it carries no skew.
    """
    _seed(tmp_path, monkeypatch)

    payload = backtests._read_backtest_progress()

    assert 0 <= payload["progress_age_seconds"] < 30


def test_age_reports_the_real_gap_for_a_wedged_run(tmp_path, monkeypatch):
    """The case the field exists for: a subprocess that stopped writing keeps
    reporting its last step forever, which reads identically to progress."""
    progress_file = _seed(tmp_path, monkeypatch, "stale.json")
    five_minutes_ago = time.time() - 300
    os.utime(progress_file, (five_minutes_ago, five_minutes_ago))

    payload = backtests._read_backtest_progress()

    assert 295 < payload["progress_age_seconds"] < 310


def test_age_is_never_negative(tmp_path, monkeypatch):
    """A clock stepping backwards between the write and this read would
    otherwise report "-3s", which reads as a bug rather than as freshness."""
    progress_file = _seed(tmp_path, monkeypatch, "future.json")
    later = time.time() + 600
    os.utime(progress_file, (later, later))

    assert backtests._read_backtest_progress()["progress_age_seconds"] == 0.0


def test_missing_progress_file_still_returns_none(tmp_path, monkeypatch):
    """Unchanged behaviour: the status payload omits `progress` entirely rather
    than shipping a half-populated object."""
    monkeypatch.setitem(
        backtests.backtest_status, "progress_file", str(tmp_path / "nope.json")
    )
    assert backtests._read_backtest_progress() is None


def test_malformed_progress_file_still_returns_none(tmp_path, monkeypatch):
    progress_file = tmp_path / "broken.json"
    progress_file.write_text("{not json", encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    assert backtests._read_backtest_progress() is None


def test_non_dict_progress_file_still_returns_none(tmp_path, monkeypatch):
    progress_file = tmp_path / "list.json"
    progress_file.write_text("[1, 2, 3]", encoding="utf-8")
    monkeypatch.setitem(backtests.backtest_status, "progress_file", str(progress_file))
    assert backtests._read_backtest_progress() is None
