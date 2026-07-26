"""Tests for baseline run pairing with external backtests."""

from dashboard.backend.baseline_resolver import resolve_baselines_for_run


def test_resolve_baselines_uses_persisted_ids():
    ext = {
        "run_id": "ext_20260415_120000",
        "created_at": "2026-04-15T12:00:00",
        "start_date": "2026-04-15",
        "end_date": "2026-04-30",
        "baseline_djia_run_id": "djia_index_aaa",
        "baseline_buyhold_run_id": "buyhold_aaa",
    }
    djia, buyhold = resolve_baselines_for_run(ext, [ext])
    assert djia == "djia_index_aaa"
    assert buyhold == "buyhold_aaa"


def test_resolve_baselines_matches_finalize_window_not_latest():
    session_runs = [
        {
            "run_id": "ext_long",
            "created_at": "2026-04-15T10:00:00",
            "start_date": "2026-04-15",
            "end_date": "2026-04-30",
        },
        {
            "run_id": "djia_index_long",
            "agent_name": "DJIA",
            "created_at": "2026-04-15T10:00:05",
            "start_date": "2026-04-15",
            "end_date": "2026-04-30",
        },
        {
            "run_id": "buyhold_long",
            "agent_name": "buy-and-hold",
            "created_at": "2026-04-15T10:00:06",
            "start_date": "2026-04-15",
            "end_date": "2026-04-30",
        },
        {
            "run_id": "ext_short",
            "created_at": "2026-04-16T10:00:00",
            "start_date": "2026-04-15",
            "end_date": "2026-04-22",
        },
        {
            "run_id": "djia_index_short",
            "agent_name": "DJIA",
            "created_at": "2026-04-16T10:00:05",
            "start_date": "2026-04-15",
            "end_date": "2026-04-22",
        },
        {
            "run_id": "buyhold_short",
            "agent_name": "buy-and-hold",
            "created_at": "2026-04-16T10:00:06",
            "start_date": "2026-04-15",
            "end_date": "2026-04-22",
        },
    ]

    ext_long = session_runs[0]
    djia, buyhold = resolve_baselines_for_run(ext_long, session_runs)
    assert djia == "djia_index_long"
    assert buyhold == "buyhold_long"

    ext_short = session_runs[3]
    djia, buyhold = resolve_baselines_for_run(ext_short, session_runs)
    assert djia == "djia_index_short"
    assert buyhold == "buyhold_short"
