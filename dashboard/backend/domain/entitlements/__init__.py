"""Account entitlement policy.

The *storage* for entitlements lives in ``users.py`` (the ``user_entitlements``
table and its two store twins); this package holds the policy that decides when
one is spent. Kept apart from ``domain/runs`` because that module governs the
Agent-Environment Protocol surfaces, and the metered path -- the dashboard's own
backtest runner -- is explicitly not one of them.
"""
