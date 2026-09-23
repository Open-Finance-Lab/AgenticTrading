"""Call-counting wrappers for the read-budget tests.

Not a test module (no ``test_`` prefix): imported by
test_operational_signals.py and test_read_budget.py. Wraps any store so every
public method call is recorded as ``(name, args, kwargs)`` and delegates to the
real object, so the SQL still runs against real SQLite -- the budget is
measured on real behaviour, not on a stub that agrees with everything.
"""

from __future__ import annotations

import inspect
from typing import Any


class CountingSpy:
    def __init__(self, target: Any, name: str) -> None:
        self._target = target
        self.name = name
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, attr: str) -> Any:
        value = getattr(self._target, attr)
        if attr.startswith("_") or not callable(value):
            # Private helpers (``_get_connection``) and plain attributes
            # (``analytics_base``, ``credits_base``) pass through unchanged.
            return value

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((attr, args, kwargs))
            return value(*args, **kwargs)

        return wrapped

    def reset(self) -> None:
        self.calls.clear()

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def calls_with_scalar_user_id(self) -> list[str]:
        """Calls that bound an ``int`` to a parameter named ``user_id``.

        A batched call passes a sequence (or None) and does not match. Binding
        through the real signature rather than eyeballing ``args[0]`` means a
        positional ``user_id`` is caught as surely as a keyword one.
        """
        offenders: list[str] = []
        for name, args, kwargs in self.calls:
            try:
                signature = inspect.signature(getattr(self._target, name))
                bound = signature.bind_partial(*args, **kwargs)
            except (TypeError, ValueError):
                continue
            if isinstance(bound.arguments.get("user_id"), int):
                offenders.append(f"{self.name}.{name}")
        return offenders


class SpyBundle:
    """Every spy the daily job touches, addressable by store name."""

    def __init__(self, **spies: CountingSpy) -> None:
        self.spies = spies

    def __getattr__(self, name: str) -> CountingSpy:
        try:
            return self.spies[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def reset(self) -> None:
        for spy in self.spies.values():
            spy.reset()

    @property
    def total_calls(self) -> int:
        return sum(spy.total_calls for spy in self.spies.values())

    @property
    def calls_with_scalar_user_id(self) -> list[str]:
        return [
            offender
            for spy in self.spies.values()
            for offender in spy.calls_with_scalar_user_id
        ]

    def calls_by_store(self) -> dict[str, int]:
        return {name: spy.total_calls for name, spy in self.spies.items()}
