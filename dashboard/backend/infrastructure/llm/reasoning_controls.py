"""Reasoning-effort spellings shared by every client that reads one.

One set, imported by the legacy OpenRouter client
(``providers/openrouter.py``) and the execution adapter
(``execution/adapters/openai.py``), so the two cannot disagree about whether
an effort string turns thinking off. The results panel keeps a JavaScript
copy (``formatBacktestSampling`` in ``frontend/app.js``), held equal to this
one by ``tests/test_backtest_sampling_row.py``.
"""

from __future__ import annotations

REASONING_OFF_VALUES = frozenset({"none", "off", "false", "0", "disabled"})

__all__ = ["REASONING_OFF_VALUES"]
