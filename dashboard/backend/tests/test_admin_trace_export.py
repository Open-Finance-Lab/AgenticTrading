import json
import pytest
from fastapi.testclient import TestClient
from dashboard.backend.app import app
from dashboard.backend.api.auth import require_admin
from dashboard.backend.api.routers import admin_traces as api
from dashboard.backend.domain.traces.repository import TraceStore
from dashboard.backend.domain.traces.export import build_export, redact


def test_export_reads_all_pages_but_excludes_new_events(tmp_path, monkeypatch):
    store = TraceStore(tmp_path / 'traces.db')
    trace = store.create_trace(trace_id='t', run_id='r')
    for i in range(205):
        store.append_event(trace_id='t', event_type='decision_recorded', actor_type='agent', payload={'reason':'hold'})
    original = store.event_high_watermark
    def boundary(trace_id):
        cutoff = original(trace_id)
        store.append_event(trace_id='t', event_type='run_completed', actor_type='system', payload={})
        return cutoff
    monkeypatch.setattr(store, 'event_high_watermark', boundary)
    with build_export(store, trace, 'json') as file:
        result = json.load(file)
    assert len(result['events']) == 205
    assert result['through_sequence'] == 205
    assert result['incomplete_run'] is True
    assert [e['sequence_no'] for e in result['events']] == list(range(1, 206))


def test_redaction_and_markdown_escape(tmp_path):
    assert redact({'api_key':'abc', 'user_id':2, 'message':'contact a@example.com Bearer abc123'}) == {'api_key':'[REDACTED]', 'user_id':'[REDACTED]', 'message':'contact [REDACTED] [REDACTED]'}
    store = TraceStore(tmp_path / 'traces.db')
    trace = store.create_trace(trace_id='t', run_id='r')
    store.append_event(trace_id='t', event_type='run_failed', actor_type='system', payload={'message':'<script>bad()</script>```'})
    with build_export(store, trace, 'markdown') as file:
        text = file.read().decode()
    assert '<script>' not in text
    assert '&lt;script&gt;' in text
    assert 'Run is incomplete' in text


@pytest.mark.parametrize('suffix', ['export?format=json', 'export?format=markdown', 'backtest'])
def test_new_endpoints_reject_non_admin(suffix):
    response = TestClient(app).get('/api/admin/traces/missing/' + suffix)
    assert response.status_code in (401, 403)


def test_admin_can_view_other_owners_exact_backtest(monkeypatch):
    from dashboard.backend.database import db
    monkeypatch.setattr(api.trace_store, 'get_trace', lambda _: {'run_id':'other-owner-run'})
    def get_run(run_id):
        assert run_id == 'other-owner-run'
        return {'run_id':run_id, 'owner_user_id':99, 'session_id':'private', 'initial_equity':1000, 'metadata':{'api_key':'private'}}
    monkeypatch.setattr(db, 'get_run', get_run)
    app.dependency_overrides[require_admin] = lambda: {'id':1,'is_admin':True}
    try:
        response = TestClient(app).get('/api/admin/traces/t/backtest')
        assert response.status_code == 200
        assert response.json()['run']['run_id'] == 'other-owner-run'
        assert 'private' not in response.text
    finally:
        app.dependency_overrides.pop(require_admin, None)


@pytest.mark.parametrize('format,extension', [('json', 'json'), ('markdown', 'md')])
def test_export_http_download_and_validation(tmp_path, monkeypatch, format, extension):
    store = TraceStore(tmp_path / 'http-traces.db')
    store.create_trace(trace_id='download', run_id='r')
    store.append_event(trace_id='download', event_type='run_started', actor_type='system', payload={})
    monkeypatch.setattr(api, 'trace_store', store)
    app.dependency_overrides[require_admin] = lambda: {'id': 1, 'is_admin': True}
    try:
        client = TestClient(app)
        response = client.get('/api/admin/traces/download/export', params={'format': format})
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['content-disposition'] == f'attachment; filename="agent-trace.{extension}"'
        if format == 'json':
            assert len(response.json()['events']) == 1
        else:
            assert 'Event 1: run_started' in response.text
        assert client.get('/api/admin/traces/missing/export').status_code == 404
        assert client.get('/api/admin/traces/download/export?format=invalid').status_code == 422
    finally:
        app.dependency_overrides.pop(require_admin, None)
