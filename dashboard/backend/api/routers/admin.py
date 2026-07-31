"""Admin/debug routes (Phase 3D4A).

Moved verbatim from ``dashboard/backend/app.py``. The external path
``/admin/runs/{run_id}`` and its behavior are unchanged; registered directly on
the app.

``DELETE /admin/clear`` used to live here and is deliberately gone -- do not
re-add it. It called ``db.clear_all()`` behind no authentication whatsoever:
the session middleware only checks that ``X-Session-Id`` parses as a UUID,
which any caller can mint, and unlike its sibling below it verified ownership
of nothing. That was survivable while run history lived in the ephemeral
SQLite file a redeploy restored -- the wipe undid itself. Once
``AGENT_RUNS_DATABASE_URL`` makes that history durable, a single anonymous
``curl`` irreversibly destroys the only copy, with no redeploy to recover
from. Nothing ever called it (no frontend, SDK, Discord bot or doc), so it was
removed rather than gated -- there is no admin/owner tier in this codebase to
gate it with (``users.role`` exists but is read for authorization nowhere).
``db.clear_all()`` itself stays: it is still used by the test suite and by
``dashboard/scripts/backtest_hourly_agent.py --clear``, both of which run
against a database the operator chose on purpose.
"""

from fastapi import APIRouter, HTTPException, Request

from dashboard.backend.database import db

router = APIRouter()


@router.delete("/admin/runs/{run_id}")
async def admin_delete_run(run_id: str, request: Request):
    """⚠️ Delete a specific run (must be owned by session)."""
    session_id = request.state.session_id
    
    # Verify ownership before deleting
    run = db.get_run_with_session(run_id, session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found or not yours")
    
    db.delete_run(run_id)
    return {"status": "deleted", "run_id": run_id}
