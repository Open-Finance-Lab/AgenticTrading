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
from dashboard.backend.domain.agents.credential_store import agent_credential_store
from dashboard.backend.domain.agents.defaults import (
    STARTER_AGENTS,
    default_starter_pipeline,
)
from dashboard.backend.domain.agents.taxonomy import coerce_category, normalize_category
from dashboard.backend.domain.agents.runtime import (
    DEFAULT_RUNTIME_TYPE,
    PIPELINE_RUNTIME_TYPE,
    normalize_runtime_config,
    normalize_runtime_type,
)
from dashboard.backend.domain.agents.version_repository import (
    VALID_EXECUTION_MODES,
    VALID_VERIFICATION_LEVELS,
    agent_version_store,
)
from dashboard.backend.database import db
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation


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
        # repository.create_agent() returns the plaintext api_key on every agent
        # it mints (any type, not just external), attached directly on the dict
        # this method receives -- so the shallow copy above would otherwise carry
        # it into every enriched listing. A route that wants to surface the key
        # once (create, import-session) pops it off the *raw* agent dict itself,
        # which this copy does not disturb -- it never removes it from the
        # source object, only omits it from the enriched copy returned here.
        result.pop("api_key", None)
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
        agent, _owned = self.resolve_access(
            agent_id,
            user_id=user_id,
            browser_session=browser_session,
            trading_session=trading_session,
            reclaim_on_session_match=reclaim_on_session_match,
        )
        return agent

    def resolve_access(
        self,
        agent_id: str,
        *,
        user_id: Optional[int] = None,
        browser_session: Optional[str] = None,
        trading_session: Optional[str] = None,
        reclaim_on_session_match: bool = False,
    ) -> Tuple[Dict[str, Any], bool]:
        """``(agent, owned)`` -- ``owned`` is False for a session-match grant.

        A matching ``session_id`` is enough to *reach* an unclaimed agent (the
        legacy dashboard reclaim, pinned by
        ``test_patch_agent_legacy_session_owner``), but it is not an ownership
        credential. Callers that write ownership -- activate, which claims the
        agent for the signed-in user -- must not treat it as one: binding an
        account on a bare session match is irreversible now that
        ``owns_agent`` refuses browser-only access to bound rows, so it would
        lock the real owner out for good rather than losing a tug-of-war they
        can win back.
        """
        agent = self.agents.get_agent(agent_id)
        if not agent:
            raise AgentNotFoundError()
        if self.agents.owns_agent(
            agent,
            owner_user_id=user_id,
            owner_browser_session=browser_session,
        ):
            return agent, True
        if (
            reclaim_on_session_match
            and trading_session
            and browser_session
            and agent.get("session_id") == trading_session
        ):
            # Only reclaim still-unclaimed guest agents. Matching session_id
            # must never let a later account on the same browser steal an
            # already-bound agent (activate/restore used to COALESCE overwrite).
            if agent.get("owner_user_id") is not None:
                raise AgentAccessDeniedError()
            # Refresh the browser stamp only. owner_user_id stays NULL: this
            # grant rests on a session id, not a credential, and an account
            # binding made here cannot be undone by the real owner.
            self.agents.reclaim_agent(
                agent_id,
                owner_browser_session=browser_session,
            )
            reclaimed = self.agents.get_agent(agent_id)
            return (reclaimed or agent), False
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
        runtime_type: Any = _UNSET,
        runtime_config: Any = _UNSET,
        cash_allocation: Any = _UNSET,
        backtest_allocation: Any = _UNSET,
        live_trading_enabled: Any = _UNSET,
        category: Any = _UNSET,
    ) -> Dict[str, Any]:
        current = self.agents.get_agent(agent_id)
        if not current:
            raise AgentNotFoundError()
        # Enforced here as well as in the route body so the column's invariant --
        # a whitelisted slug or NULL -- holds by construction for every caller,
        # not by the discipline of whoever writes the next one. The value is read
        # back out on the unauthenticated /api/v1/agents/builtin listing.
        if category is not _UNSET:
            category = coerce_category(category)
        resolved_runtime_type = (
            normalize_runtime_type(runtime_type)
            if runtime_type is not _UNSET
            else current.get("runtime_type") or DEFAULT_RUNTIME_TYPE
        )
        if runtime_config is not _UNSET:
            resolved_runtime_config = normalize_runtime_config(
                resolved_runtime_type, runtime_config
            )
        elif runtime_type is not _UNSET:
            # A runtime switch starts from that runtime's defaults. Carrying
            # implementation-specific config across the boundary can leave an
            # agent persisted in a combination the dispatcher cannot run.
            resolved_runtime_config = normalize_runtime_config(
                resolved_runtime_type, {}
            )
        else:
            resolved_runtime_config = _UNSET
        previous_runtime_type = current.get("runtime_type") or DEFAULT_RUNTIME_TYPE
        agent = self.agents.update_agent(
            agent_id,
            name=name,
            model_name=model_name,
            description=description,
            pipeline=pipeline,
            runtime_type=(
                resolved_runtime_type if runtime_type is not _UNSET else _UNSET
            ),
            runtime_config=resolved_runtime_config,
            cash_allocation=cash_allocation,
            backtest_allocation=backtest_allocation,
            live_trading_enabled=live_trading_enabled,
            category=category,
        )
        if not agent:
            raise AgentNotFoundError()
        if resolved_runtime_type != previous_runtime_type:
            # Runtime-owned credentials follow runtime_config: a switch retires
            # them rather than leaving them addressed to a runtime this agent no
            # longer has. Skipping this is how a key becomes unreachable -- the
            # credential routes gate on the *current* runtime, so the value
            # would survive with no way for its owner to see or remove it.
            agent_credential_store.delete_all(agent_id)
        owner_user_id = agent.get("owner_user_id")
        if owner_user_id is not None:
            analytics_instrumentation.emit_agent_event(
                event_name="agent_updated",
                user_id=int(owner_user_id),
                agent_id=agent_id,
                version=agent.get("last_used_at"),
            )
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
        runtime_type: str = DEFAULT_RUNTIME_TYPE,
        runtime_config: Optional[Dict[str, Any]] = None,
        cash_allocation: Optional[float] = None,
        backtest_allocation: Optional[float] = None,
        seed_default_pipeline: bool = True,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register an agent.

        Built-in agents are seeded with the default starter instruction so a new
        user can run a backtest straight from My Agents without opening
        Configure. Callers that immediately write their own pipeline (the
        marketplace clone) pass ``seed_default_pipeline=False`` to skip the
        write they are about to overwrite. External agents are driven by the
        protocol API and never use the editor pipeline, so they are not seeded.
        """
        from dashboard.backend.domain.backtesting.constants import (
            DEFAULT_AGENT_CASH_ALLOCATION,
        )

        runtime_type = normalize_runtime_type(runtime_type)
        runtime_config = normalize_runtime_config(runtime_type, runtime_config or {})
        category = coerce_category(category)  # see update_agent for why
        if cash_allocation is None:
            cash_allocation = float(DEFAULT_AGENT_CASH_ALLOCATION)
        agent = self.agents.create_agent(
            name=name,
            model_name=model_name,
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            agent_type=agent_type,
            description=description,
            runtime_type=runtime_type,
            runtime_config=runtime_config,
            cash_allocation=cash_allocation,
            backtest_allocation=backtest_allocation,
            category=category,
        )
        if (
            seed_default_pipeline
            and agent_type == "builtin"
            and runtime_type == PIPELINE_RUNTIME_TYPE
        ):
            # Merge rather than reassign: create_agent's dict carries the
            # one-time plaintext ``api_key`` that the route pops and shows once,
            # and update_agent re-reads the row (which only stores the hash).
            pipeline = default_starter_pipeline()
            self.agents.update_agent(agent["agent_id"], pipeline=pipeline)
            agent["pipeline"] = pipeline
        if owner_user_id is not None:
            analytics_instrumentation.emit_agent_event(
                event_name="agent_created",
                user_id=int(owner_user_id),
                agent_id=agent["agent_id"],
            )
        return agent

    def provision_starter_agents(
        self,
        *,
        owner_user_id: int,
        owner_browser_session: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Create the Prompted Models starter cards for a brand-new account.

        Called from signup so the first My Agents paint cannot be skipped by a
        stale browser ``localStorage`` guard (user ids recycle on local SQLite
        and on Render's ephemeral disk). Idempotent on ``model_name``: an
        account that already has DeepSeek still receives GPT-5.5 and Claude.
        Fail-open per card: signup must still succeed if one write fails.
        """
        from dashboard.backend.domain.backtesting.constants import (
            DEFAULT_AGENT_CASH_ALLOCATION,
        )
        from dashboard.backend.domain.portfolios.service import portfolio_service

        owned = list(self.agents.list_agents(owner_user_id=int(owner_user_id)))
        by_model = {str(agent.get("model_name") or ""): agent for agent in owned}
        created: List[Dict[str, Any]] = []
        for spec in STARTER_AGENTS:
            model_name = spec["model_name"]
            existing_agent = by_model.get(model_name)
            if existing_agent:
                if existing_agent.get("name") != spec["name"]:
                    try:
                        updated = self.agents.update_agent(
                            existing_agent["agent_id"], name=spec["name"]
                        )
                        if updated:
                            by_model[model_name] = updated
                    except Exception as exc:
                        print(
                            f"[agents] ERROR starter rename failed for "
                            f"user={owner_user_id} model={model_name}: {exc}"
                        )
                continue
            try:
                portfolio_service.ensure_cash_for_new_agent(
                    int(owner_user_id), float(DEFAULT_AGENT_CASH_ALLOCATION)
                )
                agent = self.create_agent(
                    name=spec["name"],
                    model_name=model_name,
                    owner_user_id=int(owner_user_id),
                    owner_browser_session=owner_browser_session,
                    agent_type="builtin",
                    description=spec["description"],
                    cash_allocation=float(DEFAULT_AGENT_CASH_ALLOCATION),
                )
                created.append(agent)
                by_model[model_name] = agent
            except Exception as exc:
                print(
                    f"[agents] ERROR starter provision failed for "
                    f"user={owner_user_id} model={model_name}: {exc}"
                )
        try:
            portfolio_service.get_or_create_portfolio(int(owner_user_id))
        except Exception as exc:
            print(
                f"[agents] ERROR starter portfolio reconcile failed "
                f"for user={owner_user_id}: {exc}"
            )
        return created

    def provision_starter_agent(
        self,
        *,
        owner_user_id: int,
        owner_browser_session: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """First starter only — prefer ``provision_starter_agents``."""
        created = self.provision_starter_agents(
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
        )
        return created[0] if created else None

    def _create_builtin_copy(
        self,
        *,
        name: str,
        model_name: str,
        owner_user_id: Optional[int],
        owner_browser_session: Optional[str],
        description: Optional[str],
        runtime_type: str,
        runtime_config: Dict[str, Any],
        category: Optional[str],
        pipeline: Optional[List[Dict[str, Any]]],
        backtest_allocation: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a built-in agent and, if the source carried its own pipeline,
        write it in -- the create-then-copy-the-pipeline-then-enrich tail shared
        by ``clone_marketplace_template`` (source: a catalog template) and
        ``duplicate_agent`` (source: a stored agent row). Callers resolve name,
        model, runtime, and category from their own source *before* calling this
        -- those resolutions differ genuinely (template vs. row, three-tier vs.
        two-tier model fallback, different name defaults) and stay put; only the
        mechanical creation sequence below is identical between them.

        ``backtest_allocation`` is simulated capital with no ledger coupling
        (validated only by ``ge=MIN_BACKTEST_INITIAL_CAPITAL,
        le=MAX_BACKTEST_INITIAL_CAPITAL``), so it is safe to copy from a
        source. ``cash_allocation`` is a real ledger debit and must NOT be
        copied here -- ``create_agent`` below is deliberately not passed one,
        so it falls back to ``DEFAULT_AGENT_CASH_ALLOCATION`` like any other
        fresh agent.
        """
        has_own_pipeline = isinstance(pipeline, list) and bool(pipeline)
        agent = self.create_agent(
            name=name,
            model_name=model_name,
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            agent_type="builtin",
            description=description,
            runtime_type=runtime_type,
            runtime_config=runtime_config,
            backtest_allocation=backtest_allocation,
            # A source carrying its own pipeline overwrites the seed below, so
            # skip that write. A source WITHOUT one still needs the starter
            # instruction, or the copy lands with an empty Configure screen.
            seed_default_pipeline=(
                runtime_type == PIPELINE_RUNTIME_TYPE and not has_own_pipeline
            ),
            category=category,
        )
        if has_own_pipeline:
            agent = self.agents.update_agent(agent["agent_id"], pipeline=pipeline) or agent
        return self.attach_equity_sparklines([self.agent_with_stats(agent)])[0]

    def clone_marketplace_template(
        self,
        *,
        template_id: str,
        owner_user_id: Optional[int],
        owner_browser_session: Optional[str],
        name: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Copy a marketplace template into the caller's My Agents list."""
        from dashboard.backend.domain.agents import marketplace as marketplace_mod

        template = marketplace_mod.get_marketplace_template(template_id)
        if not template:
            raise MarketplaceTemplateNotFoundError()

        resolved_name = (name or template.get("name") or "Marketplace Agent").strip()
        runtime_type = normalize_runtime_type(template.get("runtime_type"))
        runtime_config = normalize_runtime_config(
            runtime_type, template.get("runtime_config") or {}
        )
        # No whitelist, on purpose: create_agent doesn't validate model_name
        # either, so rejecting here would be inconsistent, and a Literal would
        # publish an openapi enum -- the deploy gate #313 had to discharge.
        # Blank/omitted means "the template's own model".
        resolved_model = (model_name or "").strip() or str(
            template.get("model_name") or "local-model"
        ).strip() or "local-model"
        return self._create_builtin_copy(
            name=resolved_name,
            model_name=resolved_model,
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            description=template.get("description"),
            runtime_type=runtime_type,
            runtime_config=runtime_config,
            # Lenient on purpose: the live catalog still carries legacy values
            # ("Foundation", "Advanced", "Hosted") until they're recategorized,
            # and those must stamp None rather than reject the clone.
            category=normalize_category(template.get("category")),
            pipeline=template.get("pipeline"),
            # A catalog template has no backtest allocation of its own.
            backtest_allocation=None,
        )

    def duplicate_agent(
        self,
        *,
        agent_id: str,
        model_name: str,
        name: Optional[str] = None,
        owner_user_id: Optional[int],
        owner_browser_session: Optional[str],
    ) -> Dict[str, Any]:
        """Copy an existing built-in agent onto a different model.

        Built-in only: ``create_agent`` mints a one-time plaintext API key for
        external agents, and duplicating one would open a credential-issuing
        path this hook has no reason to have.

        The generated name collides freely (two copies onto DeepSeek both read
        ``X copy``). Names are not unique anywhere else in this product, and
        de-duplicating would mean a lookup for a cosmetic gain.
        """
        source = self.agents.get_agent(agent_id)
        if not source:
            raise AgentNotFoundError()
        if (source.get("agent_type") or "builtin") != "builtin":
            raise AgentServiceError("Only built-in agents can be duplicated")

        resolved_name = (name or f"{source.get('name') or 'Agent'} copy").strip()
        runtime_type = normalize_runtime_type(source.get("runtime_type"))
        runtime_config = normalize_runtime_config(
            runtime_type, source.get("runtime_config") or {}
        )
        return self._create_builtin_copy(
            name=resolved_name,
            model_name=(model_name or "").strip()
            or str(source.get("model_name") or "local-model"),
            owner_user_id=owner_user_id,
            owner_browser_session=owner_browser_session,
            description=source.get("description"),
            runtime_type=runtime_type,
            runtime_config=runtime_config,
            # Lenient, matching clone_marketplace_template: a stored legacy
            # category must stamp None rather than reject the duplicate.
            category=normalize_category(source.get("category")),
            pipeline=source.get("pipeline"),
            # Copied (unlike cash_allocation -- see _create_builtin_copy's
            # docstring) so the two agents' equity curves start from the same
            # simulated capital and stay comparable. NULL on the source passes
            # through as None rather than substituting a value.
            backtest_allocation=source.get("backtest_allocation"),
        )

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
        # session_id here is the caller-supplied owner context, so re-importing
        # a session that already belongs to another account must not silently
        # rename it, re-own it, or hand its record back to the caller.
        if (
            existing
            and existing.get("owner_user_id") is not None
            and existing.get("owner_user_id") != user_id
        ):
            raise AgentAccessDeniedError()
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
        current = self.agents.get_agent(agent_id)
        result = self.agents.delete_agent(agent_id)
        auth_cache.invalidate_agent(agent_id)
        if result and current and current.get("owner_user_id") is not None:
            analytics_instrumentation.emit_agent_event(
                event_name="agent_deleted",
                user_id=int(current["owner_user_id"]),
                agent_id=agent_id,
            )
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
