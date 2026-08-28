"""Registered external agents — persistent sessions and API keys.

Canonical location (Phase 3A3). Moved verbatim from
``dashboard/backend/api/agents.py``, which is now a thin compatibility
re-export shim. Endpoint paths, prefixes, tags, schemas, status codes,
exception messages, ownership/auth behavior, and ``AgentService`` calls are
unchanged; only the module location moved.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
)

from dashboard.backend.domain.backtesting.constants import (
    DEFAULT_AGENT_CASH_ALLOCATION,
    MAX_AGENT_CASH_ALLOCATION,
    MAX_BACKTEST_INITIAL_CAPITAL,
    MIN_BACKTEST_INITIAL_CAPITAL,
)
from dashboard.backend.domain.agents.repository import _UNSET
from dashboard.backend.domain.agents.taxonomy import AgentCategory, coerce_category
from dashboard.backend.domain.agents.credential_store import (
    FINANCIAL_DATASETS_CREDENTIAL,
    agent_credential_store,
)
from dashboard.backend.domain.agents.service import (
    AgentAccessDeniedError,
    AgentNotFoundError,
    AgentServiceError,
    MarketplaceTemplateNotFoundError,
    NoExternalRunsError,
    agent_service,
)
from dashboard.backend.domain.agents import marketplace as marketplace_catalog
from dashboard.backend.domain.agents.runtime import (
    AI_HEDGE_FUND_RUNTIME_TYPE,
    normalize_runtime_config,
)
from dashboard.backend.api.dependencies import (
    _owner_context,
    _require_agent_access,
    _require_owner_context,
    _resolve_agent_access,
)
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter, client_key
from dashboard.backend.domain.portfolios.service import portfolio_service

router = APIRouter(prefix="/v1/agents", tags=["agents"])


# The ``Literal`` (not a bare ``str`` plus a hand-rolled check) is what puts the
# allowed slugs into ``openapi.json``, which is how the frontend PR probes that a
# deploy carrying this vocabulary is actually live. ``BeforeValidator`` folds
# case/whitespace and maps "" to None ahead of that check.
CategoryField = Annotated[Optional[AgentCategory], BeforeValidator(coerce_category)]

_CATEGORY_DESCRIPTION = (
    'Shelf slug, or null/"" for unshelved. Case and surrounding whitespace are '
    "folded; unknown values are rejected with 422."
)


class CreateAgentBody(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    model_name: str = Field(default="local-model", max_length=100)
    agent_type: str = Field(default="external", max_length=20)
    description: Optional[str] = Field(default=None, max_length=280)
    runtime_type: Literal["pipeline", "ai_hedge_fund"] = "pipeline"
    runtime_config: Dict[str, Any] = Field(default_factory=dict)
    cash_allocation: Optional[float] = Field(
        default=DEFAULT_AGENT_CASH_ALLOCATION,
        ge=0,
        le=MAX_AGENT_CASH_ALLOCATION,
    )
    backtest_allocation: Optional[float] = Field(
        default=None,
        ge=MIN_BACKTEST_INITIAL_CAPITAL,
        le=MAX_BACKTEST_INITIAL_CAPITAL,
    )
    category: CategoryField = Field(default=None, description=_CATEGORY_DESCRIPTION)


class PipelineStep(BaseModel):
    """One sub-agent in the editor pipeline. Extra keys are ignored."""

    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = Field(default=None, max_length=100)
    presetKey: Optional[str] = Field(default=None, max_length=50)
    label: str = Field(default="Sub-agent", max_length=100)
    prompt: str = Field(default="", max_length=8000)
    outputFormat: str = Field(default="", max_length=8000)


class UpdateAgentBody(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    model_name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=280)
    pipeline: Optional[List[PipelineStep]] = Field(default=None, max_length=50)
    runtime_type: Optional[Literal["pipeline", "ai_hedge_fund"]] = None
    runtime_config: Optional[Dict[str, Any]] = None
    cash_allocation: Optional[float] = Field(
        default=None,
        ge=0,
        le=MAX_AGENT_CASH_ALLOCATION,
    )
    backtest_allocation: Optional[float] = Field(
        default=None,
        ge=MIN_BACKTEST_INITIAL_CAPITAL,
        le=MAX_BACKTEST_INITIAL_CAPITAL,
    )
    live_trading_enabled: Optional[bool] = None
    category: CategoryField = Field(default=None, description=_CATEGORY_DESCRIPTION)

    @field_validator("name", "model_name")
    @classmethod
    def _reject_blank(cls, value: Optional[str]) -> Optional[str]:
        # ``min_length=1`` counts the raw length, so a whitespace-only value
        # ("   ") passes it and then strips to "" in the route below — an empty
        # write into a NOT NULL column. Reject it as a 422 instead of storing "".
        # (Create is deliberately lenient here: ``CreateAgentBody.model_name``
        # coerces empties to "local-model" rather than 422-ing.)
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value


MAX_CREDENTIAL_LENGTH = 4096


class FinancialDatasetsCredentialBody(BaseModel):
    """Carries the submitted key.

    Deliberately *no* ``field_validator``: ``SecretStr`` masks the value in
    reprs and logs, but it does not reach Pydantic's error envelope, and
    FastAPI serializes ``input`` into every 422 body. A validator that rejects
    a key therefore echoes that key back in plaintext. The blank/length checks
    live in the route handler instead, where they raise ``HTTPException`` and
    the response carries only the message.
    """

    api_key: SecretStr


@router.post("")
def create_agent(
    body: CreateAgentBody,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Register an agent. Returns session_id and api_key (shown once).

    ``agent_type`` is ``external`` for user-connected trading clients or
    ``builtin`` for platform-hosted agents (which are also discoverable from
    Discord). Any other value is normalized to ``external``.
    """
    ctx = _require_owner_context(request, authorization)
    agent_type = "builtin" if body.agent_type.strip().lower() == "builtin" else "external"
    cash = float(
        body.cash_allocation
        if body.cash_allocation is not None
        else DEFAULT_AGENT_CASH_ALLOCATION
    )

    # Signed-in users fund the sleeve from their account portfolio (#175).
    # Check before creating: the new sleeve *is* the debit, so there is nothing
    # to roll back if the insert fails, and nothing to compensate.
    if ctx["user_id"] and cash > 0:
        try:
            portfolio_service.ensure_cash_for_new_agent(ctx["user_id"], cash)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        runtime_config = normalize_runtime_config(
            body.runtime_type, body.runtime_config
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    agent = agent_service.create_agent(
        name=body.name.strip(),
        model_name=body.model_name.strip() or "local-model",
        owner_user_id=ctx["user_id"],
        owner_browser_session=ctx["browser_session"],
        agent_type=agent_type,
        description=(body.description.strip() if body.description else None),
        runtime_type=body.runtime_type,
        runtime_config=runtime_config,
        cash_allocation=cash,
        backtest_allocation=body.backtest_allocation,
        category=body.category,
    )
    if ctx["user_id"]:
        portfolio_service.get_or_create_portfolio(ctx["user_id"])
    return {
        "agent": agent_service.agent_with_stats(agent),
        "session_id": agent["session_id"],
        "api_key": agent.pop("api_key"),
        "client_hint": (
            f"python3 external_agent_client.py --api {request.base_url.scheme}://{request.base_url.netloc} "
            f"--api-key <api_key> --start 2026-04-15 --end 2026-04-16"
        ),
    }


@router.get("")
def list_agents(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """List agents owned by the logged-in user and/or this browser session."""
    ctx = _owner_context(request, authorization)
    if not ctx["user_id"] and not ctx["browser_session"]:
        return {"agents": []}

    agents = agent_service.list_agents_with_stats(
        owner_user_id=ctx["user_id"],
        owner_browser_session=ctx["browser_session"],
        trading_session_id=ctx.get("trading_session"),
    )
    return {"agents": agents}


@router.get("/marketplace")
def list_marketplace_agents():
    """List open agent templates available in the Agent Supermarket.

    Public and unauthenticated. Templates are config-driven and do not expose
    API keys or owner information.
    """
    return {"templates": marketplace_catalog.list_marketplace_templates()}


class CloneMarketplaceBody(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    # Deliberately unvalidated beyond length -- see clone_marketplace_template.
    model_name: Optional[str] = Field(default=None, max_length=100)


class DuplicateAgentBody(BaseModel):
    """``model_name`` is required here (unlike clone): the whole point of the
    action is running the same strategy somewhere else."""

    model_name: str = Field(min_length=1, max_length=100)
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)

    @field_validator("model_name")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        # Same trap as UpdateAgentBody._reject_blank: ``min_length=1`` counts
        # the raw length, so a whitespace-only value ("   ") passes it. Unlike
        # CreateAgentBody.model_name (optional, coerces blank to
        # "local-model" on purpose), this field is required -- the service's
        # ``.strip() or source.get("model_name")`` fallback would otherwise
        # silently reuse the source's own model, defeating the one thing this
        # action does. Reject it as a 422 instead.
        if not value.strip():
            raise ValueError("must not be blank")
        return value


@router.post("/marketplace/{template_id}/clone")
def clone_marketplace_agent(
    template_id: str,
    body: CloneMarketplaceBody,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Copy a marketplace template into the caller's My Agents list."""
    ctx = _require_owner_context(request, authorization)
    cash = float(DEFAULT_AGENT_CASH_ALLOCATION)
    if ctx["user_id"] and cash > 0:
        try:
            portfolio_service.ensure_cash_for_new_agent(ctx["user_id"], cash)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        agent = agent_service.clone_marketplace_template(
            template_id=template_id,
            owner_user_id=ctx["user_id"],
            owner_browser_session=ctx["browser_session"],
            name=body.name.strip() if body.name else None,
            model_name=body.model_name,
        )
    except MarketplaceTemplateNotFoundError:
        raise HTTPException(status_code=404, detail="Marketplace template not found")
    if ctx["user_id"]:
        portfolio_service.get_or_create_portfolio(ctx["user_id"])
    return {"agent": agent}


@router.get("/builtin")
def list_builtin_agents():
    """List all platform-hosted (built-in) agents.

    Public and unauthenticated: built-in agents are globally discoverable so
    integrations like the Discord ``/agent`` command can offer them to anyone.
    Only non-sensitive, presentational fields are returned (no API keys/owners).
    """
    agents = agent_service.list_builtin_agents_with_stats()
    public = []
    for agent in agents:
        latest = agent.get("latest_run") or {}
        public.append(
            {
                "agent_id": agent["agent_id"],
                "name": agent["name"],
                "model_name": agent.get("model_name") or "local-model",
                "description": agent.get("description"),
                "category": agent.get("category"),
                "run_count": agent.get("run_count", 0),
                "latest_return": latest.get("total_return"),
                "latest_sharpe": latest.get("sharpe_ratio"),
            }
        )
    return {"agents": public}


@router.post("/claim-account")
def claim_account_agents(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Link this browser's agents to the logged-in user (call after login)."""
    ctx = _owner_context(request, authorization)
    if not ctx["user_id"]:
        raise HTTPException(status_code=401, detail="Log in to claim agents")
    if not ctx["browser_session"]:
        raise HTTPException(status_code=400, detail="Missing browser session")
    claimed, agents = agent_service.claim_account_agents(
        browser_session=ctx["browser_session"],
        user_id=ctx["user_id"],
    )
    return {"claimed": claimed, "agents": agents}


class ImportSessionBody(BaseModel):
    name: Optional[str] = Field(default=None, max_length=100)
    model_name: Optional[str] = Field(default=None, max_length=100)


@router.post("/import-session")
def import_session_agent(
    body: ImportSessionBody,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Register the current trading session as an agent (for CLI runs without prior signup)."""
    ctx = _require_owner_context(request, authorization)
    session_id = ctx["browser_session"]
    try:
        agent, imported = agent_service.import_session(
            session_id=session_id,
            user_id=ctx["user_id"],
            name=body.name,
            model_name=body.model_name,
        )
    except NoExternalRunsError:
        raise HTTPException(status_code=404, detail="No external backtest runs for this session")
    except AgentAccessDeniedError:
        raise HTTPException(status_code=403, detail="Agent belongs to another account")

    result = {"agent": agent_service.agent_with_stats(agent), "imported": imported}
    if imported and agent.get("api_key"):
        result["api_key"] = agent["api_key"]
    return result


@router.get("/resolve")
def resolve_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Resolve a registered agent API key to its trading session (for CLI clients)."""
    agent = agent_service.resolve_api_key(x_api_key or "")
    if not agent:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {
        "agent_id": agent["agent_id"],
        "name": agent["name"],
        "session_id": agent["session_id"],
        "model_name": agent["model_name"],
    }


@router.get("/{agent_id}/runs")
def list_agent_runs(
    agent_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """List external backtest runs for an agent."""
    ctx = _require_owner_context(request, authorization)
    agent = _require_agent_access(agent_id, ctx)
    runs = agent_service.list_external_runs(agent["session_id"])
    return {"runs": runs}


@router.get("/{agent_id}")
def get_agent(
    agent_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    ctx = _require_owner_context(request, authorization)
    agent = _require_agent_access(agent_id, ctx)
    return {"agent": agent_service.agent_with_stats(agent)}


@router.patch("/{agent_id}")
def update_agent(
    agent_id: str,
    body: UpdateAgentBody,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Update agent display name, description, and/or sub-agent pipeline."""
    ctx = _require_owner_context(request, authorization)
    _require_agent_access(agent_id, ctx, reclaim_on_session_match=True)
    fields_set = body.model_fields_set
    pipeline_provided = "pipeline" in fields_set
    cash_allocation_provided = "cash_allocation" in fields_set
    backtest_allocation_provided = "backtest_allocation" in fields_set
    live_trading_provided = "live_trading_enabled" in fields_set
    runtime_type_provided = "runtime_type" in fields_set
    runtime_config_provided = "runtime_config" in fields_set
    category_provided = "category" in fields_set
    if (
        body.name is None
        and body.model_name is None
        and body.description is None
        and not pipeline_provided
        and not cash_allocation_provided
        and not backtest_allocation_provided
        and not live_trading_provided
        and not runtime_type_provided
        and not runtime_config_provided
        and not category_provided
    ):
        raise HTTPException(status_code=400, detail="No fields to update")

    # ``category_provided`` (not ``body.category is not None``) is what separates
    # "clear the shelf" from "leave it alone" -- ``{"category": null}`` and
    # ``{"category": ""}`` both arrive here as None with the key in ``fields_set``.
    category_arg = body.category if category_provided else _UNSET

    if pipeline_provided:
        pipeline_arg = (
            [step.model_dump() for step in body.pipeline]
            if body.pipeline is not None
            else None
        )
    else:
        pipeline_arg = _UNSET

    cash_allocation_arg = body.cash_allocation if cash_allocation_provided else _UNSET
    backtest_allocation_arg = (
        body.backtest_allocation if backtest_allocation_provided else _UNSET
    )
    live_trading_arg = body.live_trading_enabled if live_trading_provided else _UNSET
    runtime_type_arg = body.runtime_type if runtime_type_provided else _UNSET
    runtime_config_arg = body.runtime_config if runtime_config_provided else _UNSET

    # When a signed-in owner changes cash_allocation, the portfolio ledger has
    # to follow (#175). Validate it *first* -- writing nothing -- so a rejected
    # allocation cannot leave the other fields half-applied, and hand the sleeve
    # write to the service afterwards so the ledger and the sleeve move once.
    ledger_new_amount = None
    if cash_allocation_provided and ctx.get("user_id"):
        existing = agent_service.get_agent(agent_id)
        if existing and existing.get("owner_user_id") == ctx["user_id"]:
            ledger_new_amount = float(
                body.cash_allocation if body.cash_allocation is not None else 0
            )
            try:
                portfolio_service.check_agent_allocation(
                    owner_user_id=ctx["user_id"],
                    agent=existing,
                    new_amount=ledger_new_amount,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            # The service owns this write; don't also pass it to update_agent.
            cash_allocation_arg = _UNSET

    try:
        agent = agent_service.update_agent(
            agent_id,
            name=body.name.strip() if body.name is not None else None,
            model_name=body.model_name.strip() if body.model_name is not None else None,
            description=body.description,
            pipeline=pipeline_arg,
            runtime_type=runtime_type_arg,
            runtime_config=runtime_config_arg,
            cash_allocation=cash_allocation_arg,
            backtest_allocation=backtest_allocation_arg,
            live_trading_enabled=live_trading_arg,
            category=category_arg,
        )
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if ledger_new_amount is not None:
        agent = portfolio_service.set_agent_allocation(
            owner_user_id=ctx["user_id"],
            agent=agent,
            new_amount=ledger_new_amount,
        )["agent"]
    return {"agent": agent}


# Storing a hosted credential is gated on the hosted runtime; reading its
# status and deleting it deliberately are NOT. Gating all three strands any key
# whose agent later switches to ``pipeline``: the owner could no longer see it,
# replace it, or delete it, and the row outlived the agent itself.
def _require_ai_hedge_fund_agent(agent: Dict[str, Any]) -> None:
    if (agent.get("runtime_type") or "pipeline") != AI_HEDGE_FUND_RUNTIME_TYPE:
        raise HTTPException(
            status_code=422,
            detail="Financial Datasets credentials apply only to AI Hedge Fund agents",
        )


_credential_rate_limiter = FixedWindowRateLimiter(max_events=20, window_seconds=3600)


@router.get("/{agent_id}/credentials/financial-datasets")
def get_financial_datasets_credential_status(
    agent_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Return configured status without ever returning credential plaintext."""
    ctx = _owner_context(request, authorization)
    # Called for its authorization side effect; the row itself is not needed
    # here, since status is deliberately not gated on the current runtime.
    _require_agent_access(agent_id, ctx, api_key=x_api_key)
    status = agent_credential_store.get_public(
        agent_id, FINANCIAL_DATASETS_CREDENTIAL
    )
    return status or {
        "credential": FINANCIAL_DATASETS_CREDENTIAL,
        "configured": False,
        "updated_at": None,
    }


@router.put("/{agent_id}/credentials/financial-datasets")
def set_financial_datasets_credential(
    agent_id: str,
    body: FinancialDatasetsCredentialBody,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Encrypt and store the user's Financial Datasets API key."""
    if not _credential_rate_limiter.allow(client_key(request)):
        raise HTTPException(
            status_code=429,
            detail="Too many credential updates recently; please try again later.",
        )
    ctx = _owner_context(request, authorization)
    agent = _require_agent_access(agent_id, ctx, api_key=x_api_key)
    _require_ai_hedge_fund_agent(agent)
    # Validated here rather than in the model so a rejection never echoes the
    # submitted key back through Pydantic's ``input`` field. See the body class.
    raw_key = body.api_key.get_secret_value().strip()
    if not raw_key:
        raise HTTPException(status_code=422, detail="api_key must not be blank")
    if len(raw_key) > MAX_CREDENTIAL_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"api_key must be at most {MAX_CREDENTIAL_LENGTH} characters",
        )
    try:
        return agent_credential_store.upsert(
            agent_id,
            FINANCIAL_DATASETS_CREDENTIAL,
            raw_key,
        )
    except RuntimeError as exc:
        # The encryption helper names a missing/malformed setting but never its
        # value, so this remains actionable without exposing the submitted key.
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/{agent_id}/credentials/financial-datasets")
def delete_financial_datasets_credential(
    agent_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    ctx = _owner_context(request, authorization)
    _require_agent_access(agent_id, ctx, api_key=x_api_key)
    return {
        "credential": FINANCIAL_DATASETS_CREDENTIAL,
        "configured": False,
        "deleted": agent_credential_store.delete(
            agent_id, FINANCIAL_DATASETS_CREDENTIAL
        ),
    }


@router.delete("/{agent_id}")
def delete_agent(
    agent_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    ctx = _owner_context(request, authorization)
    agent = _require_agent_access(agent_id, ctx, api_key=x_api_key)
    sleeve = float(agent.get("cash_allocation") or 0)
    owner_user_id = agent.get("owner_user_id")
    # Unconditional on purpose. Gating this on the *current* runtime_type let a
    # credential stored while the agent was hosted survive the agent forever
    # once it was switched to pipeline -- decryptable, ownerless, unreachable.
    # The delete is one cheap statement and is a no-op for pipeline agents.
    agent_credential_store.delete_all(agent_id)
    agent_service.delete_agent(agent_id)
    # Refresh the account ledger now the sleeve is gone from the sum (#175).
    # Reconciling cannot fail here -- unlike crediting the sleeve back, which
    # overshoots equity on any agent that never debited the ledger in the first
    # place, and would 500 a request whose agent row is already deleted.
    if owner_user_id:
        portfolio_service.reclaim_all_on_delete(owner_user_id=int(owner_user_id))
    return {"status": "deleted", "agent_id": agent_id, "reclaimed": sleeve}


@router.post("/{agent_id}/rotate-api-key")
def rotate_agent_api_key(
    agent_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Generate a new API key for an agent. The previous key stops working immediately."""
    ctx = _owner_context(request, authorization)
    _require_agent_access(agent_id, ctx, api_key=x_api_key)
    api_key = agent_service.rotate_api_key(agent_id)
    if not api_key:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = agent_service.get_agent(agent_id)
    return {
        "agent": agent_service.agent_with_stats(agent),
        "api_key": api_key,
    }


@router.post("/{agent_id}/duplicate")
def duplicate_agent(
    agent_id: str,
    body: DuplicateAgentBody,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Copy a built-in agent onto a different model ("Run on another model").

    Does not start a backtest: the caller lands on the new agent with Run primed.
    """
    ctx = _require_owner_context(request, authorization)
    _require_agent_access(agent_id, ctx, reclaim_on_session_match=True)
    cash = float(DEFAULT_AGENT_CASH_ALLOCATION)
    if ctx["user_id"] and cash > 0:
        try:
            portfolio_service.ensure_cash_for_new_agent(ctx["user_id"], cash)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        agent = agent_service.duplicate_agent(
            agent_id=agent_id,
            model_name=body.model_name,
            name=body.name.strip() if body.name else None,
            owner_user_id=ctx["user_id"],
            owner_browser_session=ctx["browser_session"],
        )
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail="Agent not found")
    except AgentServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if ctx["user_id"]:
        portfolio_service.get_or_create_portfolio(ctx["user_id"])
    return {"agent": agent}


@router.post("/{agent_id}/activate")
def activate_agent(
    agent_id: str,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    """Return session info for switching the dashboard to this agent."""
    ctx = _require_owner_context(request, authorization)
    agent, owned = _resolve_agent_access(
        agent_id, ctx, reclaim_on_session_match=True
    )
    # Claiming binds owner_user_id, and owns_agent refuses browser-only access
    # to a bound row -- so a claim made off a bare session_id match would lock
    # the agent's real owner out with no way back. Only bind when the caller
    # proved ownership with a real credential.
    agent_service.activate_agent(
        agent_id,
        user_id=ctx.get("user_id") if owned else None,
        browser_session=ctx.get("browser_session"),
    )
    return {
        "agent_id": agent["agent_id"],
        "name": agent["name"],
        "session_id": agent["session_id"],
        "model_name": agent["model_name"],
    }
