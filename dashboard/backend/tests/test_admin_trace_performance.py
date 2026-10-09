import pytest
from fastapi import HTTPException
from dashboard.backend.api.routers import admin_traces as api
from dashboard.backend.database import db


def test_exact_run_and_bounded_points(monkeypatch):
    monkeypatch.setattr(api.trace_store, 'get_trace', lambda _: {'run_id': 'exact', 'status': 'failed'})
    calls = []
    def run(run_id):
        calls.append(run_id)
        return {'initial_equity': 100}
    monkeypatch.setattr(db, 'get_run', run)
    monkeypatch.setattr(db, 'get_equity_curve', lambda _: [{'timestamp': str(i), 'equity': 100 + i} for i in range(2001)])
    result = api.get_trace_performance('trace')
    assert calls == ['exact']
    assert len(result['points']) == 1000
    assert result['points'][0]['equity'] == 100
    assert result['points'][-1]['equity'] == 2100
    assert result['partial'] and result['sampled']
    assert result['metrics']['return_pct'] == 2000


def test_missing_result_is_not_a_flat_curve(monkeypatch):
    monkeypatch.setattr(api.trace_store, 'get_trace', lambda _: {'run_id': 'gone', 'status': 'completed'})
    monkeypatch.setattr(db, 'get_run', lambda _: None)
    result = api.get_trace_performance('trace')
    assert result['source'] == 'unavailable'
    assert result['points'] == []
    assert result['metrics']['return_pct'] is None


def test_missing_trace_is_404(monkeypatch):
    monkeypatch.setattr(api.trace_store, 'get_trace', lambda _: None)
    with pytest.raises(HTTPException) as exc:
        api.get_trace_performance('missing')
    assert exc.value.status_code == 404


def test_invalid_equity_is_excluded_and_drawdown_uses_full_series(monkeypatch):
    monkeypatch.setattr(api.trace_store, 'get_trace', lambda _: {'run_id': 'r', 'status': 'completed'})
    # A real metadata row is nonempty.
    monkeypatch.setattr(db, 'get_run', lambda _: {'run_id': 'r'})
    monkeypatch.setattr(db, 'get_equity_curve', lambda _: [{'timestamp': str(i), 'equity': v} for i, v in enumerate([100, None, float('nan'), 80, 110])])
    result = api.get_trace_performance('t')
    assert len(result['points']) == 3
    assert result['metrics']['max_drawdown_pct'] == 20


def test_performance_route_keeps_admin_dependency():
    from dashboard.backend.api.auth import require_admin
    route = next(r for r in api.router.routes if r.path.endswith('/performance'))
    assert require_admin in [dependency.call for dependency in route.dependant.dependencies]
