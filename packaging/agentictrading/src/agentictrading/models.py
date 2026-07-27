"""Typed request/response models for the Agent-Environment Protocol (v1).

These are lightweight ``dataclasses`` (Pydantic is not a dependency of this
package). Each parsed model keeps the original payload on ``.raw`` for forward
compatibility and debugging, but the documented workflow only needs the typed
fields (``step.status``, ``step.observation``, ``result.metrics``, ...).

The shapes below were taken directly from the backend implementation
(``dashboard/backend/run_service.py``, ``api/runs.py``, ``api/agent_versions.py``,
``api/environments.py``, ``external_backtest_service.get_run_result``).

Endpoint -> response shape (fields the SDK relies on)
----------------------------------------------------
POST /api/v1/agents/{agent_id}/versions -> {"agent_version": {agent_version_id,
    agent_id, version, execution_mode, architecture, model_backbones[],
    decision_frequency, code_commit, prompt_hash, config_hash,
    verification_level, created_at}}

POST /api/v1/runs, GET /api/v1/runs/{id} -> {protocol_version, run_id, agent_id,
    agent_version_id, environment:{environment_id, type}, config:{...}, status,
    result_run_id, created_at, progress?:{step_index, total_steps},
    metrics?:{...}, compare_url?}

GET /api/v1/runs/{id}/status -> {protocol_version, run_id, status, step_index,
    total_steps, result_run_id}

GET /api/v1/runs/{id}/steps/next ->
    loading:    {protocol_version, run_id, status:"loading", message}
    completed:  {protocol_version, run_id, status:"completed", result_run_id, message}
    awaiting:   {protocol_version, run_id, step_id, sequence, timestamp,
                 deadline_at, status:"awaiting_decision",
                 observation:{market:{bars, features, events}, portfolio:{cash,
                 equity, positions[]}}, constraints:{allowed_symbols, allow_short,
                 max_position_weight, max_orders}}

GET /api/v1/runs/{id}/steps/{step_id} -> awaiting view (as above) OR historical:
    {protocol_version, run_id, step_id, sequence, timestamp, deadline_at,
     status: "completed"|"timed_out"|"awaiting_decision"}

POST /api/v1/runs/{id}/steps/{step_id}/decision -> {protocol_version, run_id,
    step_id, decision_id, accepted, validation:{passed, warnings[],
    rejections:[{order, reason}]}, fills:[{symbol, side, requested_quantity,
    filled_quantity, fill_price}], portfolio_after:{cash, equity, positions[]},
    run_status:"running"|"completed"}

GET /api/v1/runs/{id}/result -> {protocol_version, run_id, result_run_id, status,
    run:{...}, equity_curve:[...], trades:[...], decisions:[...], metrics:{
    total_return, sharpe_ratio, max_drawdown, num_trades, final_equity,
    llm_calls, input_tokens, output_tokens, est_cost_usd, timeout_holds}}

GET /api/v1/runs/{id}/trades     -> {protocol_version, run_id, trades:[...], count}
GET /api/v1/runs/{id}/decisions  -> {protocol_version, run_id, decisions:[...], count}
GET /api/v1/runs/{id}/metrics    -> {protocol_version, run_id, status, metrics:{...}}
GET /api/v1/environments         -> {"environments": [{environment_id, type, ...}]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ----------------------------------------------------------------------
# Request models (also support to_dict for submission)
# ----------------------------------------------------------------------


@dataclass
class Order:
    """A single order in a decision (``orders-v1`` schema)."""

    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    quantity_type: str = "shares"  # "shares" | "notional" | "weight"
    order_type: str = "market"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity_type": self.quantity_type,
            "quantity": self.quantity,
            "order_type": self.order_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        return cls(
            symbol=data["symbol"],
            side=data.get("side", "buy"),
            quantity=data.get("quantity", 0),
            quantity_type=data.get("quantity_type", "shares"),
            order_type=data.get("order_type", "market"),
        )


@dataclass
class Decision:
    """An agent's response to a step. An empty ``orders`` list is an explicit HOLD."""

    orders: List[Union[Order, Dict[str, Any]]] = field(default_factory=list)
    confidence: Optional[float] = None
    rationale: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None

    def _orders_as_dicts(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for order in self.orders:
            out.append(order.to_dict() if isinstance(order, Order) else dict(order))
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the decision (without run/step/idempotency context)."""
        body: Dict[str, Any] = {"orders": self._orders_as_dicts()}
        if self.confidence is not None:
            body["confidence"] = self.confidence
        if self.rationale is not None:
            body["rationale"] = self.rationale
        if self.trace is not None:
            body["trace"] = self.trace
        return body

    def to_body(self, *, run_id: str, step_id: str, idempotency_key: str) -> Dict[str, Any]:
        """Full request body including routing + idempotency context."""
        body = self.to_dict()
        body["run_id"] = run_id
        body["step_id"] = step_id
        body["idempotency_key"] = idempotency_key
        return body

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        return cls(
            orders=[Order.from_dict(o) if isinstance(o, dict) else o for o in (data.get("orders") or [])],
            confidence=data.get("confidence"),
            rationale=data.get("rationale"),
            trace=data.get("trace"),
        )


# ----------------------------------------------------------------------
# Response models
# ----------------------------------------------------------------------


@dataclass
class AgentVersion:
    agent_version_id: Optional[str] = None
    agent_id: Optional[str] = None
    version: Optional[str] = None
    execution_mode: Optional[str] = None
    architecture: Optional[str] = None
    model_backbones: List[str] = field(default_factory=list)
    decision_frequency: Optional[str] = None
    verification_level: Optional[str] = None
    created_at: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> Optional[str]:
        return self.agent_version_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentVersion":
        return cls(
            agent_version_id=data.get("agent_version_id"),
            agent_id=data.get("agent_id"),
            version=data.get("version"),
            execution_mode=data.get("execution_mode"),
            architecture=data.get("architecture"),
            model_backbones=data.get("model_backbones") or [],
            decision_frequency=data.get("decision_frequency"),
            verification_level=data.get("verification_level"),
            created_at=data.get("created_at"),
            raw=data,
        )


@dataclass
class Run:
    run_id: Optional[str] = None
    status: Optional[str] = None
    agent_id: Optional[str] = None
    agent_version_id: Optional[str] = None
    environment_id: Optional[str] = None
    environment_type: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    result_run_id: Optional[str] = None
    created_at: Optional[str] = None
    step_index: Optional[int] = None
    total_steps: Optional[int] = None
    metrics: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> Optional[str]:
        return self.run_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Run":
        env = data.get("environment") or {}
        progress = data.get("progress") or {}
        return cls(
            run_id=data.get("run_id"),
            status=data.get("status"),
            agent_id=data.get("agent_id"),
            agent_version_id=data.get("agent_version_id"),
            environment_id=env.get("environment_id") or data.get("environment_id"),
            environment_type=env.get("type") or data.get("environment_type"),
            config=data.get("config") or {},
            result_run_id=data.get("result_run_id"),
            created_at=data.get("created_at"),
            step_index=progress.get("step_index"),
            total_steps=progress.get("total_steps"),
            metrics=data.get("metrics"),
            raw=data,
        )


@dataclass
class RunStatus:
    run_id: Optional[str] = None
    status: Optional[str] = None
    step_index: Optional[int] = None
    total_steps: Optional[int] = None
    result_run_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunStatus":
        return cls(
            run_id=data.get("run_id"),
            status=data.get("status"),
            step_index=data.get("step_index"),
            total_steps=data.get("total_steps"),
            result_run_id=data.get("result_run_id"),
            raw=data,
        )


@dataclass
class Observation:
    """Market + portfolio context for a step."""

    market: Dict[str, Any] = field(default_factory=dict)
    portfolio: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> Dict[str, Any]:
        return self.market.get("features") or {}

    @property
    def positions(self) -> List[Dict[str, Any]]:
        return self.portfolio.get("positions") or []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            market=data.get("market") or {},
            portfolio=data.get("portfolio") or {},
            raw=data,
        )


@dataclass
class Step:
    status: Optional[str] = None
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    sequence: Optional[int] = None
    timestamp: Optional[str] = None
    deadline_at: Optional[str] = None
    observation: Optional[Observation] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    result_run_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> Optional[str]:
        return self.step_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        obs = data.get("observation")
        return cls(
            status=data.get("status"),
            run_id=data.get("run_id"),
            step_id=data.get("step_id"),
            sequence=data.get("sequence"),
            timestamp=data.get("timestamp"),
            deadline_at=data.get("deadline_at"),
            observation=Observation.from_dict(obs) if obs else None,
            constraints=data.get("constraints") or {},
            message=data.get("message"),
            result_run_id=data.get("result_run_id"),
            raw=data,
        )


@dataclass
class ExecutionResult:
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    decision_id: Optional[str] = None
    accepted: bool = False
    validation: Dict[str, Any] = field(default_factory=dict)
    fills: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_after: Dict[str, Any] = field(default_factory=dict)
    run_status: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def rejections(self) -> List[Dict[str, Any]]:
        return self.validation.get("rejections") or []

    @property
    def warnings(self) -> List[Any]:
        return self.validation.get("warnings") or []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        return cls(
            run_id=data.get("run_id"),
            step_id=data.get("step_id"),
            decision_id=data.get("decision_id"),
            accepted=bool(data.get("accepted")),
            validation=data.get("validation") or {},
            fills=data.get("fills") or [],
            portfolio_after=data.get("portfolio_after") or {},
            run_status=data.get("run_status"),
            raw=data,
        )


@dataclass
class RunResult:
    run_id: Optional[str] = None
    result_run_id: Optional[str] = None
    status: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    run: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunResult":
        return cls(
            run_id=data.get("run_id"),
            result_run_id=data.get("result_run_id"),
            status=data.get("status"),
            metrics=data.get("metrics") or {},
            equity_curve=data.get("equity_curve") or [],
            trades=data.get("trades") or [],
            decisions=data.get("decisions") or [],
            run=data.get("run") or {},
            raw=data,
        )


__all__ = [
    "Order",
    "Decision",
    "AgentVersion",
    "Run",
    "RunStatus",
    "Observation",
    "Step",
    "ExecutionResult",
    "RunResult",
]
