"""Agent lifecycle service.

Canonical home (Phase 3A2) for agent and agent-version *workflows* that were
previously implemented directly inside the FastAPI routers
(``dashboard/backend/api/agents.py`` and ``dashboard/backend/api/agent_versions.py``).

The service coordinates the agent and agent-version repositories plus run
statistics. It is framework-agnostic: it must NOT import FastAPI routers, app.py,
scripts, or frontend code. Routers stay responsible for HTTP concerns (request
parsing, status codes) and translate the small domain exceptions defined here into
``HTTPException`` with the exact same messages as before.

Behavior (SQL, schemas, ownership rules, API-key handling, version semantics) is
preserved exactly; only the call site moved.
"""

from typing import Any, Dict, List, Optional, Tuple

from dashboard.backend.domain.agents.repository import agent_store, _UNSET
from dashboard.backend.domain.agents import auth_cache
from dashboard.backend.domain.agents.version_repository import (
    VALID_EXECUTION_MODES,
    VALID_VERIFICATION_LEVELS,
    agent_version_store,
)
from dashboard.backend.database import db


# Baseline comparison series written alongside every backtest. They are plotted
# for comparison but are not standalone agent runs, so they are excluded from
# agent run listings.
BASELINE_AGENT_NAMES = {"DJIA", "buy-and-hold"}

# Card sparkline budget — enough to show path shape without shipping full curves.
_SPARKLINE_MAX_POINTS = 36


def sample_equity_sparkline(
    curve: Optional[List[Dict[str, Any]]],
    *,
    max_points: int = _SPARKLINE_MAX_POINTS,
) -> List[float]:
    """Downsample an equity timeseries to a short list of equity floats.

    Always includes the first and last points when downsampling so the card
    sparkline ends on the same value as Ending Value / final equity.
    """
    equities: List[float] = []
    for point in curve or []:
        try:
            equities.append(float(point["equity"]))
        except (TypeError, ValueError, KeyError):
            continue
    if not equities:
        return []
    if len(equities) <= max_points:
        return equities
    last = len(equities) - 1
    return [
        equities[round(i * last / (max_points - 1))]
        for i in range(max_points)
    ]


class AgentServiceError(Exception):
    """Base class for agent-service domain errors."""


class AgentNotFoundError(AgentServiceError):
    """Raised when an agent does not exist."""


class AgentAccessDeniedError(AgentServiceError):
    """Raised when the caller does not own / cannot access an agent."""


class NoExternalRunsError(AgentServiceError):
    """Raised when importing a session that has no external backtest runs."""


class MarketplaceTemplateNotFoundError(AgentServiceError):
    """Raised when a marketplace template id is unknown."""


class InvalidVersionFieldError(AgentServiceError):
    """Raised when an agent-version field is outside its allowed set."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AgentService:
    """Coordinate agent + agent-version repositories and run statistics."""

    def __init__(self, *, agents=agent_store, versions=agent_version_store, database=db):
        self.agents = agents
        self.versions = versions
        self.db = database

    # ------------------------------------------------------------------
    # Run statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_external(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [r for r in runs if str(r.get("run_id", "")).startswith("ext_")]

    @staticmethod
    def _filter_agent_session(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The agent's own backtest runs for a session, excluding baselines.

        Built-in (platform-hosted) agents are backtested through the website
        workflow (``/backtest/run``), whose runs are not ``ext_``-prefixed, so
        their cards must surface session runs rather than external-only ones.
        Each backtest also writes DJIA / buy-and-hold baseline rows in the same
        session; those are comparison series for the plot, not standalone runs,
        so they are filtered out of the agent's run list.
        """
        return [r for r in runs if r.get("agent_name") not in BASELINE_AGENT_NAMES]

    def _external_runs(self, session_id: str) -> List[Dict[str, Any]]:
        return self._filter_external(self.db.get_runs_by_session(session_id) or [])

    def _session_runs(self, session_id: str) -> List[Dict[str, Any]]:
        return self._filter_agent_session(self.db.get_runs_by_session(session_id) or [])

    def agent_with_stats(
        self,
        agent: Dict[str, Any],
        *,
        session_runs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Enrich an agent record with run statistics.

        ``session_runs`` optionally supplies the agent session's raw
        (unfiltered) runs so batched listings can prefetch them in one query
        instead of one query per agent.
        """
        if session_runs is None:
            session_runs = self.db.get_runs_by_session(agent["session_id"]) or []
        if (agent.get("agent_type") or "external") == "builtin":
            ext_runs = self._filter_agent_session(session_runs)
        else:
            ext_runs = self._filter_external(session_runs)
        latest = None
        if ext_runs:
            latest = sorted(ext_runs, key=lambda r: r.get("created_at") or "", reverse=True)[0]
        result = dict(agent)
        result["run_count"] = len(ext_runs)
        result["latest_run"] = latest
        result["runs"] = sorted(
            ext_runs,
            key=lambda r: r.get("created_at") or "",
            reverse=True,
        )
        result["total_llm_calls"] = sum(int(r.get("llm_calls") or 0) for r in ext_runs)
        result["total_input_tokens"] = sum(int(r.get("input_tokens") or 0) for r in ext_runs)
        result["total_output_tokens"] = sum(int(r.get("output_tokens") or 0) for r in ext_runs)
        result["total_est_cost_usd"] = round(
            sum(float(r.get("est_cost_usd") or 0) for r in ext_runs), 6
        )
        return result

    def attach_equity_sparklines(
        self, agents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Attach ``equity_sparkline`` from each agent's latest run curve.

        Batches curve loads so agent-card listings stay O(unique latest runs)
        instead of one query per agent.
        """
        run_ids: List[str] = []
        seen = set()
        for agent in agents:
            latest = agent.get("latest_run") or {}
            run_id = latest.get("run_id")
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            run_ids.append(run_id)
        if not run_ids:
            return agents
        curves = self.db.get_equity_curves(run_ids)
        for agent in agents:
            latest = agent.get("latest_run")
            if not latest:
                continue
            spark = sample_equity_sparkline(curves.get(latest.get("run_id")) or [])
            if not spark:
                continue
            agent["equity_sparkline"] = spark
            latest = dict(latest)
            latest["equity_sparkline"] = spark
            agent["latest_run"] = latest
        return agents

    def list_external_runs(self, session_id: str) -> List[Dict[str, Any]]:
        ext_runs = self._external_runs(session_id)
        ext_runs.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return ext_runs

    # ------------------------------------------------------------------
    # Access / ownership
    # ------------------------------------------------------------------

    def require_access(
        self,
        agent_id: str,
        *,
        user_id: Optional[int] = None,
        browser_session: Optional[str] = None,
        trading_session: Optional[str] = None,
        reclaim_on_session_match: bool = False,
    ) -> Dict[str, Any]:
        agent = self.agents.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError()
        if self.agents.owns_agent(
            agent,
            owner_user_id=user_id,
            owner_browser_session=browser_session,
        ):
            return agent
        if (
            reclaim_on_session_match
            and trading_session
            and browser_session
            and agent.get("session_id") == trading_session
        ):
            self.agents.reclaim_agent(
                agent_id,
                owner_user_id=user_id,
                owner_browser_session=browser_session,
            )
            reclaimed = self.agents.get_agent(agent_id)
            return reclaimed or agent
        raise AgentAccessDeniedError()

    # ------------------------------------------------------------------
    # Agent CRUD / workflows
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self.agents.get_agent(agent_id)

    def update_agent(
        self,
        agent_id: str,
        *,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
        pipeline: Any = _UNSET,
        cash_allocation: Any = _UNSET,
    ) -> Dict[str, Any]:
        agent = self.agents.update_agent(
            agent_id,
            name=name,
            model_name=model_name,
            description=description,
            pipeline=pipeline,
            cash_allocation=cash_allocation,
        )
        if not agent:
            raise AgentNotFoundError()
        return self.attach_equity_sparklines([self.agent_with_stats(agent)])[0]

    def create_agent(
        self,
        *,
        name: str,
        model_name: str,
        owner_user_id: Optional[int],
        owner_browser_session: Optional[str],
        agent_type: str = "external",
        description: Optional[str] = None,
        cash_allocation: Optional[float] = None,
    ) -> Dict[str, Any]:
        from dashboard.backend.domain.backtesting.constants import (
            DEFAULT_AGENT_CASH_ALLOCATION,
        )

        if cash_allocation is None:
            cash_allocation = float(DEFAULT_AGENT_CASH_ALLOCATION)
        return self.agents.create_agent(
            name=name,
            model_name=model_name,
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            agent_type=agent_type,
            description=description,
            cash_allocation=cash_allocation,
        )

    def clone_marketplace_template(
        self,
        *,
        template_id: str,
        owner_user_id: Optional[int],
        owner_browser_session: Optional[str],
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Copy a marketplace template into the caller's My Agents list."""
        from dashboard.backend.domain.agents import marketplace as marketplace_mod

        template = marketplace_mod.get_marketplace_template(template_id)
        if not template:
            raise MarketplaceTemplateNotFoundError()

        resolved_name = (name or template.get("name") or "Marketplace Agent").strip()
        agent = self.create_agent(
            name=resolved_name,
            model_name=str(template.get("model_name") or "local-model").strip() or "local-model",
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            agent_type="builtin",
            description=template.get("description"),
        )
        pipeline = template.get("pipeline")
        if isinstance(pipeline, list) and pipeline:
            agent = self.agents.update_agent(agent["agent_id"], pipeline=pipeline) or agent
        return self.attach_equity_sparklines([self.agent_with_stats(agent)])[0]

    def list_builtin_agents_with_stats(self) -> List[Dict[str, Any]]:
        """List all built-in agents (platform-wide) enriched with run stats.

        Serves the public, unauthenticated ``/agents/builtin`` listing, so the
        per-agent run stats are prefetched with a single batched query rather
        than one query per agent.
        """
        agents = self.agents.list_builtin_agents()
        runs_by_session = self.db.get_runs_by_sessions([a["session_id"] for a in agents])
        enriched = [
            self.agent_with_stats(a, session_runs=runs_by_session.get(a["session_id"], []))
            for a in agents
        ]
        return self.attach_equity_sparklines(enriched)

    def list_agents_with_stats(
        self,
        *,
        owner_user_id: Optional[int],
        owner_browser_session: Optional[str],
        trading_session_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """List owned agents enriched with run stats (batched, no N+1).

        Same prefetch pattern as ``list_builtin_agents_with_stats``: one
        ``get_runs_by_sessions`` query for the whole listing, then per-agent
        enrichment from the prefetched map.
        """
        agents = self.agents.list_agents(
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            trading_session_id=trading_session_id,
        )
        # Same batching as list_builtin_agents_with_stats — one runs query for
        # the whole list instead of get_runs_by_session per agent (N+1).
        runs_by_session = self.db.get_runs_by_sessions(
            [a["session_id"] for a in agents]
        )
        enriched = [
            self.agent_with_stats(
                a, session_runs=runs_by_session.get(a["session_id"], [])
            )
            for a in agents
        ]
        return self.attach_equity_sparklines(enriched)

    def claim_account_agents(
        self,
        *,
        browser_session: str,
        user_id: int,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        claimed = self.agents.claim_browser_agents_to_user(browser_session, user_id)
        agents = self.agents.list_agents(owner_user_id=user_id)
        enriched = [self.agent_with_stats(a) for a in agents]
        return claimed, self.attach_equity_sparklines(enriched)

    def import_session(
        self,
        *,
        session_id: str,
        user_id: Optional[int],
        name: Optional[str],
        model_name: Optional[str],
    ) -> Tuple[Dict[str, Any], bool]:
        """Register the current trading session as an agent.

        Returns ``(agent, imported)`` where ``imported`` is True when the agent
        did not already exist for this session.
        """
        ext_runs = self._external_runs(session_id)
        if not ext_runs:
            raise NoExternalRunsError()
        latest = sorted(ext_runs, key=lambda r: r.get("created_at") or "", reverse=True)[0]
        resolved_name = (name or latest.get("agent_name") or "external-agent").strip()
        resolved_model = (model_name or latest.get("llm_model") or "local-model").strip()

        existing = self.agents.get_agent_by_session(session_id)
        agent = self.agents.register_or_get_agent(
            session_id=session_id,
            name=resolved_name,
            model_name=resolved_model,
            owner_user_id=user_id,
            owner_browser_session=session_id,
        )
        return agent, existing is None

    def resolve_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        return self.agents.resolve_api_key(api_key)

    def delete_agent(self, agent_id: str) -> bool:
        result = self.agents.delete_agent(agent_id)
        auth_cache.invalidate_agent(agent_id)
        return result

    def rotate_api_key(self, agent_id: str) -> Optional[str]:
        new_key = self.agents.rotate_api_key(agent_id)
        auth_cache.invalidate_agent(agent_id)
        return new_key

    def activate_agent(
        self,
        agent_id: str,
        *,
        user_id: Optional[int] = None,
        browser_session: Optional[str] = None,
    ) -> None:
        self.agents.claim_agent(
            agent_id,
            owner_user_id=user_id,
            owner_browser_session=browser_session,
        )

    # ------------------------------------------------------------------
    # Agent versions
    # ------------------------------------------------------------------

    def create_version(
        self,
        *,
        agent_id: str,
        version: str,
        execution_mode: str,
        architecture: Optional[str],
        model_backbones: List[str],
        decision_frequency: str,
        code_commit: Optional[str],
        prompt_hash: Optional[str],
        config_hash: Optional[str],
        prompt: Optional[str],
        config: Optional[Dict[str, Any]],
        verification_level: str,
    ) -> Dict[str, Any]:
        if execution_mode not in VALID_EXECUTION_MODES:
            raise InvalidVersionFieldError(f"Invalid execution_mode: {execution_mode}")
        if verification_level not in VALID_VERIFICATION_LEVELS:
            raise InvalidVersionFieldError(f"Invalid verification_level: {verification_level}")
        return self.versions.create_version(
            agent_id=agent_id,
            version=version,
            execution_mode=execution_mode,
            architecture=architecture,
            model_backbones=model_backbones,
            decision_frequency=decision_frequency,
            code_commit=code_commit,
            prompt_hash=prompt_hash,
            config_hash=config_hash,
            prompt=prompt,
            config=config,
            verification_level=verification_level,
        )

    def list_versions(self, agent_id: str) -> List[Dict[str, Any]]:
        return self.versions.list_versions(agent_id)

    def get_version(self, agent_version_id: str) -> Optional[Dict[str, Any]]:
        return self.versions.get_version(agent_version_id)


agent_service = AgentService()
