from fastapi import APIRouter

from dashboard.backend.api.routers.agent_versions import router as agent_versions_router
from dashboard.backend.api.routers.agents import router as agents_router
from dashboard.backend.api.routers.algo import router as algo_router
from dashboard.backend.api.routers.analytics import router as analytics_router
from dashboard.backend.api.routers.admin_analytics import (
    router as admin_analytics_router,
)
from dashboard.backend.api.routers.admin_users import router as admin_users_router
from dashboard.backend.api.auth import router as auth_router
from dashboard.backend.api.routers.discord import router as discord_router
from dashboard.backend.api.routers.credits import router as credits_router
from dashboard.backend.api.routers.admin_credits import router as admin_credits_router
from dashboard.backend.api.routers.admin_model_providers import router as admin_model_providers_router
from dashboard.backend.api.routers.model_credentials import router as model_credentials_router
from dashboard.backend.api.routers.environments import router as environments_router
from dashboard.backend.api.routers.external_backtest import router as external_backtest_router
from dashboard.backend.api.health import router as health_router
from dashboard.backend.api.routers.leaderboard import router as leaderboard_router
from dashboard.backend.api.routers.news import router as news_router
from dashboard.backend.api.routers.portfolio import router as portfolio_router
from dashboard.backend.api.routers.runs import router as runs_router
from dashboard.backend.api.routers.strategies import router as strategies_router
from dashboard.backend.api.routers.robinhood_live import router as robinhood_router
from dashboard.backend.api.v2.router import v2_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(admin_users_router)
api_router.include_router(algo_router)
api_router.include_router(agents_router)
api_router.include_router(analytics_router)
api_router.include_router(admin_analytics_router)
api_router.include_router(discord_router)
api_router.include_router(credits_router)
api_router.include_router(admin_credits_router)
api_router.include_router(admin_model_providers_router)
api_router.include_router(model_credentials_router)
api_router.include_router(agent_versions_router)
api_router.include_router(external_backtest_router)
api_router.include_router(runs_router)
api_router.include_router(environments_router)
api_router.include_router(leaderboard_router)
api_router.include_router(strategies_router)
api_router.include_router(portfolio_router)
api_router.include_router(news_router)
api_router.include_router(robinhood_router)
api_router.include_router(v2_router)
