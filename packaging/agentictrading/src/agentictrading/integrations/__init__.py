"""Optional client-side integrations for external agent frameworks.

The public surface is defined once, by :mod:`.tradingagents`. Re-exporting it
by name here meant two hand-maintained lists, which had already drifted apart.
"""

from .tradingagents import *  # noqa: F401,F403
from .tradingagents import __all__
