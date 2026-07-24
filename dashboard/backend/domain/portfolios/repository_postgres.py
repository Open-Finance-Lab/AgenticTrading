"""Postgres-backed PortfolioStore.

Selected when ``CONTENT_DATABASE_URL`` is set (see repository._build_portfolio_store).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import psycopg

from dashboard.backend.db_url import require_postgres_url
from dashboard.backend.domain.backtesting.constants import DEFAULT_PORTFOLIO_EQUITY
from dashboard.backend.domain.portfolios.repository import (
    _public_portfolio,
    _utcnow_iso,
)


class PostgresPortfolioStore:
    """One portfolio row per signed-in user, backed by Postgres."""

    def __init__(self, database_url: str):
        self.database_url = require_postgres_url(database_url)
        self._init_schema()

    def _get_connection(self):
        # Pooled checkout: same context-manager transaction semantics as
        # psycopg.connect (commit on clean exit), returned to the pool on close.
        from dashboard.backend.db_pool import get_pool
        return get_pool(self.database_url).connection()

    def _init_schema(self) -> None:
        # Runs once per process, from __init__ -- not per query. Re-running it
        # on every read would issue DDL on the request path and thrash the
        # shared pool with schema churn.
        #
        # ADDING A COLUMN LATER (#175 allocate/reclaim)? It must go in an
        # `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` below, *not* only in the
        # CREATE. CREATE TABLE IF NOT EXISTS silently no-ops once the table
        # exists, so an existing deployment would never gain the column and
        # every query naming it would raise UndefinedColumn -- 500ing this
        # whole surface while /health stays green. Nothing catches it first:
        # SQLite is the default tier in tests, and CI's Postgres container is
        # empty on every run, so the @pg_only tier only exercises the CREATE
        # path. See agents/repository_postgres.py for the worked example.
        #
        # owner_user_id is a plain INTEGER with no FK to users(id): same
        # rationale as external_agents (split USERS vs CONTENT databases).
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_portfolios (
                        owner_user_id INTEGER PRIMARY KEY,
                        equity DOUBLE PRECISION NOT NULL,
                        cash_available DOUBLE PRECISION NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )

    def get(self, owner_user_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM user_portfolios WHERE owner_user_id = %s",
                    (int(owner_user_id),),
                )
                row = cur.fetchone()
        return _public_portfolio(row) if row else None

    def create(
        self,
        owner_user_id: int,
        *,
        equity: float = DEFAULT_PORTFOLIO_EQUITY,
    ) -> Dict[str, Any]:
        now = _utcnow_iso()
        equity_f = float(equity)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_portfolios (
                        owner_user_id, equity, cash_available, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (int(owner_user_id), equity_f, equity_f, now, now),
                )
        created = self.get(owner_user_id)
        if created is None:  # pragma: no cover - the INSERT above just committed
            raise RuntimeError(
                f"portfolio for user {owner_user_id} vanished immediately after INSERT"
            )
        return created

    def get_or_create(
        self,
        owner_user_id: int,
        *,
        equity: float = DEFAULT_PORTFOLIO_EQUITY,
    ) -> Dict[str, Any]:
        existing = self.get(owner_user_id)
        if existing is not None:
            return existing
        try:
            return self.create(owner_user_id, equity=equity)
        except psycopg.errors.UniqueViolation:
            raced = self.get(owner_user_id)
            if raced is None:
                raise
            return raced

    def set_cash_available(
        self, owner_user_id: int, cash_available: float
    ) -> Dict[str, Any]:
        """Overwrite cash_available with an already-validated figure.

        Blind write, mirroring the SQLite twin: ``service._reconcile`` derives
        the value from the agent sleeves (a different table), so a
        ``SELECT ... FOR UPDATE`` here would lock a row whose contents this
        statement does not depend on. Bounds are the service's job.
        """
        self.get_or_create(owner_user_id)
        uid = int(owner_user_id)
        value = float(cash_available)
        now = _utcnow_iso()
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_portfolios
                    SET cash_available = %s, updated_at = %s
                    WHERE owner_user_id = %s
                    """,
                    (value, now, uid),
                )
        updated = self.get(uid)
        if updated is None:  # pragma: no cover - the UPDATE above just committed
            raise RuntimeError(f"portfolio for user {uid} vanished during update")
        return updated
