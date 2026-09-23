"""D24 hygiene on the admin analytics router: loud 503s, no dead users-list stack."""

from __future__ import annotations

import inspect

import pytest
from fastapi import HTTPException

from dashboard.backend.api.routers import admin_analytics
from dashboard.backend.domain.analytics import query_service


def test_an_unrecognised_exception_is_logged_before_it_becomes_a_503(capsys):
    """A bad SQL statement and an exhausted pool used to be indistinguishable
    in prod logs: the router had no print and raised `from None`."""
    with pytest.raises(HTTPException) as info:
        admin_analytics._raise_service_error(RuntimeError("secret-canary"))

    printed = capsys.readouterr().out
    assert info.value.status_code == 503
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "ERROR: admin_analytics.unhandled category=RuntimeError" in printed
    assert "secret-canary" not in printed
    # The display-safe detail is unchanged.
    assert info.value.detail == "Analytics is temporarily unavailable."


@pytest.mark.parametrize(
    "exc,status",
    [(LookupError("missing"), 404), (ValueError("bad"), 422)],
)
def test_recognised_exceptions_keep_their_quiet_mapping(capsys, exc, status):
    with pytest.raises(HTTPException) as info:
        admin_analytics._raise_service_error(exc)

    assert info.value.status_code == status
    assert info.value.__cause__ is None
    assert capsys.readouterr().out == ""


def test_the_dead_users_list_stack_is_gone():
    assert not hasattr(admin_analytics, "_user_filters")
    assert not hasattr(admin_analytics, "_USER_SORTS")
    assert not hasattr(query_service, "AnalyticsUserFilters")
    assert not hasattr(query_service, "PaginatedUsers")
    assert not hasattr(query_service.AnalyticsQueryService, "list_users")
    assert "AnalyticsUserFilters" not in query_service.__all__
    assert "PaginatedUsers" not in query_service.__all__
    # The live users route is untouched: it still answers from the value stack.
    assert "_value_user_filters" in inspect.getsource(admin_analytics)
