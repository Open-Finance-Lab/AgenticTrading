"""Lazy vn.py runtime bridge used by the local CTA adapter."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
import re
from dataclasses import dataclass
from datetime import datetime
from types import ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ..models import ExecutionResult
from ._vnpy_cta_core import CapturedCtaOrder, sanitize_error_message, validate_ohlcv_values


class VnpyCtaDependencyError(RuntimeError):
    """Raised when the optional vn.py CTA runtime is unavailable."""


class VnpyCtaCompatibilityError(RuntimeError):
    """Raised when installed vn.py packages are outside the supported range."""


@dataclass(frozen=True)
class VnpyBindings:
    """All optional vn.py symbols needed by the integration."""

    vnpy_version: str
    cta_version: str
    CtaTemplate: type
    EngineType: Any
    BarData: type
    OrderData: type
    TradeData: type
    Direction: Any
    Offset: Any
    Exchange: Any
    Interval: Any
    Status: Any
    OrderType: Any


def _module(import_module: Callable[[str], ModuleType], name: str) -> ModuleType:
    try:
        return import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise VnpyCtaDependencyError(
            "vn.py CTA integration requires optional dependencies; install with "
            "pip install 'agentictrading[vnpy]'"
        ) from exc


def load_vnpy_bindings(
    *,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
    version_resolver: Callable[[str], str] = importlib.metadata.version,
) -> VnpyBindings:
    """Import vn.py symbols only when a caller explicitly creates the runtime."""
    constants = _module(import_module, "vnpy.trader.constant")
    objects = _module(import_module, "vnpy.trader.object")
    cta = _module(import_module, "vnpy_ctastrategy")
    cta_base = _module(import_module, "vnpy_ctastrategy.base")
    try:
        vnpy_version = version_resolver("vnpy")
        cta_version = version_resolver("vnpy_ctastrategy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise VnpyCtaDependencyError(
            "vn.py CTA package metadata is missing; install with "
            "pip install 'agentictrading[vnpy]'"
        ) from exc

    return VnpyBindings(
        vnpy_version=vnpy_version,
        cta_version=cta_version,
        CtaTemplate=cta.CtaTemplate,
        EngineType=cta_base.EngineType,
        BarData=objects.BarData,
        OrderData=objects.OrderData,
        TradeData=objects.TradeData,
        Direction=constants.Direction,
        Offset=constants.Offset,
        Exchange=constants.Exchange,
        Interval=constants.Interval,
        Status=constants.Status,
        OrderType=constants.OrderType,
    )


def _major_minor(version: str) -> Tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    if not match:
        raise VnpyCtaCompatibilityError(f"invalid package version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _validate_versions(bindings: VnpyBindings) -> None:
    vnpy = _major_minor(bindings.vnpy_version)
    cta = _major_minor(bindings.cta_version)
    if vnpy[0] != 4 or vnpy < (4, 4):
        raise VnpyCtaCompatibilityError(
            f"unsupported vnpy version {bindings.vnpy_version!r}; expected >=4.4,<5"
        )
    if cta[0] != 1 or cta < (1, 3):
        raise VnpyCtaCompatibilityError(
            "unsupported vnpy_ctastrategy version "
            f"{bindings.cta_version!r}; expected >=1.3,<2"
        )


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).strip().lower()


def _aware_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid timezone-aware timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"invalid timezone-aware timestamp: {value!r}")
    return parsed


class AtlCtaEngine:
    """Minimal engine surface that captures CtaTemplate order calls."""

    def __init__(self, bindings: VnpyBindings, *, symbol: str) -> None:
        self.bindings = bindings
        self.symbol = symbol.upper()
        self.current_timestamp: Optional[str] = None
        self.events: list[Dict[str, Any]] = []
        self._captured: list[CapturedCtaOrder] = []
        self._next_order = 1

    def send_order(
        self,
        strategy: Any,
        direction: Any,
        offset: Any,
        price: float,
        volume: float,
        stop: bool,
        lock: bool,
        net: bool,
    ) -> list[str]:
        order_id = f"atl-cta-{self._next_order}"
        self._next_order += 1
        self._captured.append(
            CapturedCtaOrder(
                order_id=order_id,
                timestamp=self.current_timestamp or "",
                symbol=self.symbol,
                direction=_enum_name(direction),
                offset=_enum_name(offset),
                price=price,
                volume=volume,
                stop=bool(stop),
                lock=bool(lock),
                net=bool(net),
            )
        )
        return [order_id]

    def drain_captured_orders(self) -> Tuple[CapturedCtaOrder, ...]:
        captured = tuple(self._captured)
        self._captured.clear()
        return captured

    def cancel_order(self, strategy: Any, vt_orderid: str) -> None:
        self.events.append(
            {"type": "cancel_not_supported", "order_id": str(vt_orderid)}
        )

    def cancel_all(self, strategy: Any) -> None:
        self.events.append({"type": "cancel_all_not_supported"})

    def get_engine_type(self) -> Any:
        return self.bindings.EngineType.BACKTESTING

    def get_pricetick(self, strategy: Any) -> float:
        return 0.01

    def get_size(self, strategy: Any) -> int:
        return 1

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Any,
        callback: Callable[..., Any],
        use_database: bool = False,
    ) -> list[Any]:
        self.events.append(
            {
                "type": "history_preload_unavailable",
                "vt_symbol": str(vt_symbol),
                "days": int(days),
                "use_database": bool(use_database),
            }
        )
        return []

    def load_tick(self, *args: Any, **kwargs: Any) -> list[Any]:
        self.events.append({"type": "tick_history_not_supported"})
        return []

    def write_log(self, msg: str, strategy: Any = None) -> None:
        self.events.append(
            {"type": "strategy_log", "message": sanitize_error_message(msg)}
        )

    def put_strategy_event(self, strategy: Any) -> None:
        return None

    def send_notification(self, msg: str, strategy: Any) -> None:
        self.events.append(
            {
                "type": "notification_not_sent",
                "message": sanitize_error_message(msg),
            }
        )

    def sync_strategy_data(self, strategy: Any) -> None:
        self.events.append({"type": "strategy_sync_local_only"})


class VnpyCtaRuntime:
    """Own a local CtaTemplate instance and translate ATL lifecycle callbacks."""

    def __init__(
        self,
        strategy_class: type,
        *,
        symbol: str,
        settings: Mapping[str, Any],
        strategy_name: Optional[str] = None,
        bindings: Optional[VnpyBindings] = None,
    ) -> None:
        self.bindings = bindings or load_vnpy_bindings()
        _validate_versions(self.bindings)
        if not isinstance(strategy_class, type) or not issubclass(
            strategy_class, self.bindings.CtaTemplate
        ):
            raise VnpyCtaCompatibilityError(
                "strategy class must inherit vnpy_ctastrategy.CtaTemplate"
            )

        self.symbol = str(symbol).upper()
        if self.symbol != "AAPL":
            raise VnpyCtaCompatibilityError("vn.py CTA MVP supports AAPL only")
        self.engine = AtlCtaEngine(self.bindings, symbol=self.symbol)
        self.strategy = strategy_class(
            self.engine,
            strategy_name or strategy_class.__name__,
            f"{self.symbol}.SMART",
            dict(settings),
        )
        self.started = False
        self._next_trade = 1

    @property
    def events(self) -> list[Dict[str, Any]]:
        return self.engine.events

    def start(self) -> None:
        if self.started:
            return
        self.strategy.on_init()
        self.strategy.inited = True
        self.strategy.trading = True
        self.strategy.on_start()
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return
        try:
            self.strategy.on_stop()
        finally:
            self.strategy.trading = False
            self.started = False

    def sync_position(self, quantity: Any) -> None:
        if isinstance(quantity, bool):
            raise ValueError("position must be a non-negative integer")
        try:
            numeric = float(quantity)
        except (TypeError, ValueError) as exc:
            raise ValueError("position must be a non-negative integer") from exc
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise ValueError("position must be a non-negative integer")
        self.strategy.pos = int(numeric)

    def _bar_data(self, payload: Mapping[str, Any]) -> Any:
        timestamp = _aware_datetime(payload.get("timestamp"))
        values = validate_ohlcv_values(payload)
        return self.bindings.BarData(
            gateway_name="ATL",
            symbol=self.symbol,
            exchange=self.bindings.Exchange.SMART,
            datetime=timestamp,
            interval=self.bindings.Interval.HOUR,
            open_price=values["open"],
            high_price=values["high"],
            low_price=values["low"],
            close_price=values["close"],
            volume=values["volume"],
        )

    def on_bar(self, payload: Mapping[str, Any]) -> Tuple[CapturedCtaOrder, ...]:
        if not self.started:
            raise RuntimeError("vn.py CTA runtime must be started before on_bar")
        bar = self._bar_data(payload)
        self.engine.current_timestamp = bar.datetime.isoformat()
        self.strategy.on_bar(bar)
        return self.drain_captured_orders()

    def drain_captured_orders(self) -> Tuple[CapturedCtaOrder, ...]:
        """Orders captured by the engine since the last drain.

        Exposed at the runtime's lifecycle level (rather than requiring
        callers to reach into ``self.engine``) so a caller recovering from a
        ``strategy.on_bar()`` exception can collect whatever orders the
        strategy sent before it raised, without depending on the concrete
        engine implementation.
        """
        return self.engine.drain_captured_orders()

    def _enum_member(self, enum: Any, name: str) -> Any:
        return getattr(enum, str(name).upper())

    def _order_data(
        self,
        captured: CapturedCtaOrder,
        *,
        status: Any,
        traded: float,
        timestamp: str,
    ) -> Any:
        return self.bindings.OrderData(
            gateway_name="ATL",
            symbol=self.symbol,
            exchange=self.bindings.Exchange.SMART,
            orderid=captured.order_id,
            type=self.bindings.OrderType.MARKET,
            direction=self._enum_member(self.bindings.Direction, captured.direction),
            offset=self._enum_member(self.bindings.Offset, captured.offset),
            price=float(captured.price),
            volume=float(captured.volume),
            traded=float(traded),
            status=status,
            datetime=_aware_datetime(timestamp),
            reference="ATL vn.py CTA adapter",
        )

    def reject_captured_order(
        self,
        captured: CapturedCtaOrder,
        *,
        reason: str,
        timestamp: str,
    ) -> None:
        order = self._order_data(
            captured,
            status=self.bindings.Status.REJECTED,
            traded=0,
            timestamp=timestamp,
        )
        self.strategy.on_order(order)
        self.events.append(
            {
                "type": "order_rejected",
                "order_id": captured.order_id,
                "reason": sanitize_error_message(reason),
            }
        )

    @staticmethod
    def _captured_side(captured: CapturedCtaOrder) -> Optional[str]:
        pair = (captured.direction.lower(), captured.offset.lower())
        if pair == ("long", "open"):
            return "buy"
        if pair == ("short", "close"):
            return "sell"
        return None

    @staticmethod
    def _closest_quantity_match(
        candidate_indices: list[int],
        items: Sequence[Mapping[str, Any]],
        volume: float,
        quantity_of: Callable[[Mapping[str, Any]], Any],
    ) -> Optional[int]:
        """Among same-symbol/same-side candidates, prefer the one whose
        submitted quantity is closest to ``volume``. ATL's ``orders-v1`` wire
        schema carries no client-supplied order id to correlate fills or
        rejections back to a specific captured order, so (symbol, side) alone
        is ambiguous whenever a strategy emits more than one same-side order
        in a single ``on_bar()`` call; matching on quantity too is a
        best-effort tie-breaker, not a guarantee.
        """
        best_index: Optional[int] = None
        best_distance: Optional[float] = None
        for index in candidate_indices:
            try:
                requested = float(quantity_of(items[index]))
            except (TypeError, ValueError):
                continue
            distance = abs(requested - float(volume))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None:
            return best_index
        return candidate_indices[0] if candidate_indices else None

    def apply_execution_result(
        self,
        result: ExecutionResult,
        *,
        captured_orders: Sequence[CapturedCtaOrder],
        timestamp: str,
    ) -> None:
        unused_fills = list(result.fills)
        rejections = list(result.rejections)
        for captured in captured_orders:
            side = self._captured_side(captured)
            fill_candidates = [
                index
                for index, fill in enumerate(unused_fills)
                if str(fill.get("symbol", "")).upper() == self.symbol
                and str(fill.get("side", "")).lower() == side
            ]
            fill_index = self._closest_quantity_match(
                fill_candidates,
                unused_fills,
                captured.volume,
                lambda fill: fill.get("requested_quantity"),
            )
            if fill_index is not None:
                fill = unused_fills.pop(fill_index)
                filled = float(fill.get("filled_quantity") or 0)
                status = (
                    self.bindings.Status.ALLTRADED
                    if filled >= float(captured.volume)
                    else self.bindings.Status.PARTTRADED
                )
                self.strategy.on_order(
                    self._order_data(
                        captured,
                        status=status,
                        traded=filled,
                        timestamp=timestamp,
                    )
                )
                if filled > 0:
                    if side == "buy":
                        self.strategy.pos += filled
                    elif side == "sell":
                        self.strategy.pos -= filled
                    trade = self.bindings.TradeData(
                        gateway_name="ATL",
                        symbol=self.symbol,
                        exchange=self.bindings.Exchange.SMART,
                        orderid=captured.order_id,
                        tradeid=f"atl-trade-{self._next_trade}",
                        direction=self._enum_member(
                            self.bindings.Direction, captured.direction
                        ),
                        offset=self._enum_member(self.bindings.Offset, captured.offset),
                        price=float(fill.get("fill_price") or 0),
                        volume=filled,
                        datetime=_aware_datetime(timestamp),
                    )
                    self._next_trade += 1
                    self.strategy.on_trade(trade)
                continue

            rejection_candidates = [
                index
                for index, item in enumerate(rejections)
                if str((item.get("order") or {}).get("symbol", "")).upper()
                == self.symbol
                and str((item.get("order") or {}).get("side", "")).lower() == side
            ]
            rejection_index = self._closest_quantity_match(
                rejection_candidates,
                rejections,
                captured.volume,
                lambda item: (item.get("order") or {}).get("quantity"),
            )
            if rejection_index is not None:
                rejection = rejections.pop(rejection_index)
                self.reject_captured_order(
                    captured,
                    reason=str(rejection.get("reason") or "atl_rejection"),
                    timestamp=timestamp,
                )


__all__ = [
    "VnpyCtaDependencyError",
    "VnpyCtaCompatibilityError",
    "VnpyBindings",
    "load_vnpy_bindings",
    "AtlCtaEngine",
    "VnpyCtaRuntime",
]
