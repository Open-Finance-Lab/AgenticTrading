

import dashboard.backend.api.v2.runs as runs_mod
from dashboard.backend.tests._v2_fakes import FakeBackend


def _register_fake(run_id="run_unit_1", session_id="sess_unit_1"):
    backend = FakeBackend(run_id=run_id, session_id=session_id)
    runs_mod.register_run(run_id, backend, session_id)
    return backend


def test_lifecycle_create_context_decide_result():
    _register_fake()  # registers the run; the backend handle is not needed here
    # context
    ctx = runs_mod._context_for("run_unit_1", "sess_unit_1")
    assert ctx["status"] == "waiting_decision"
    # decide twice → completes (FakeBackend has 2 steps)
    runs_mod._submit_for("run_unit_1", "sess_unit_1", "key-a", [
        {"action": "buy", "symbol": "AAPL", "confidence": 0.8,
         "reasoning": "momentum strong", "position_size": 5}])
    ack = runs_mod._submit_for("run_unit_1", "sess_unit_1", "key-b", [])
    assert ack["status"] == "completed"
    # result
    res = runs_mod._result_for("run_unit_1", "sess_unit_1")
    assert res["manifest"]["universe"] == "djia_30"


def test_wrong_session_cannot_read_run():
    _register_fake(run_id="run_unit_2", session_id="owner")
    import pytest
    from dashboard.backend.api.v2.errors import ApiError
    with pytest.raises(ApiError) as exc:
        runs_mod._context_for("run_unit_2", "intruder")
    assert exc.value.status == 404
    assert exc.value.code == "run_not_found"
