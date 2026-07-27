"""Minimal streaming vn.py CTA strategy for the ATL integration example."""

from __future__ import annotations

from collections import deque

from vnpy.trader.object import BarData
from vnpy_ctastrategy import CtaTemplate


class DoubleMaStrategy(CtaTemplate):
    """Trade long-only crossovers after enough hourly bars have arrived."""

    author = "Agentic Trading Lab"

    fast_window = 5
    slow_window = 20
    fixed_size = 1

    fast_ma = 0.0
    slow_ma = 0.0

    parameters = ["fast_window", "slow_window", "fixed_size"]
    variables = ["fast_ma", "slow_ma"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        if self.fast_window < 1 or self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window >= 1")
        if self.fixed_size < 1 or int(self.fixed_size) != self.fixed_size:
            raise ValueError("fixed_size must be a positive integer")
        self._closes = deque(maxlen=int(self.slow_window))
        self._previous_fast = None
        self._previous_slow = None

    def on_init(self):
        self.write_log("Double MA strategy initialized; waiting for streaming bars")

    def on_start(self):
        self.write_log("Double MA strategy started")

    def on_stop(self):
        self.write_log("Double MA strategy stopped")

    def on_bar(self, bar: BarData):
        close = float(bar.close_price)
        self._closes.append(close)
        if len(self._closes) < self.slow_window:
            return

        closes = list(self._closes)
        self.fast_ma = sum(closes[-self.fast_window :]) / self.fast_window
        self.slow_ma = sum(closes) / self.slow_window

        if self._previous_fast is not None and self._previous_slow is not None:
            crossed_up = (
                self._previous_fast <= self._previous_slow
                and self.fast_ma > self.slow_ma
            )
            crossed_down = (
                self._previous_fast >= self._previous_slow
                and self.fast_ma < self.slow_ma
            )
            if crossed_up and self.pos == 0:
                self.buy(close, int(self.fixed_size))
            elif crossed_down and self.pos > 0:
                self.sell(close, int(self.pos))

        self._previous_fast = self.fast_ma
        self._previous_slow = self.slow_ma
        self.put_event()


__all__ = ["DoubleMaStrategy"]
