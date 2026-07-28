"""Optional client-side integrations for external agent frameworks.

The public surface is defined once, by :mod:`.tradingagents`. Re-exporting it
by name here meant two hand-maintained lists, which had already drifted apart.
"""

from . import vnpy_cta
from .tradingagents import *  # noqa: F401,F403
from .tradingagents import __all__ as _tradingagents_all

__all__ = [*_tradingagents_all, "vnpy_cta"]
