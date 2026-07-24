"""T3: 60 s deadline default + the timeout_holds integrity counter."""

import dashboard.backend.domain.backtesting.external_run_service as ebs


def test_default_decision_timeout_is_60():
    # conftest strips EXTERNAL_AGENT_DECISION_TIMEOUT_SECONDS at import, so
    # the module constant IS the default.
    assert ebs.DECISION_TIMEOUT_SECONDS == 60
