"""Canonical user-source groups used by admin account and analytics views."""

from typing import Literal, Mapping, cast

UserGroup = Literal[
    "internal",
    "invited",
    "organic",
    "competition",
    "partner",
    "unknown",
]

USER_GROUPS: tuple[UserGroup, ...] = (
    "internal",
    "invited",
    "organic",
    "competition",
    "partner",
    "unknown",
)

USER_GROUP_LABELS: Mapping[UserGroup, str] = {
    "internal": "Internal",
    "invited": "Invited",
    "organic": "Organic",
    "competition": "Competition",
    "partner": "Partner",
    "unknown": "Unknown",
}

# The group an account is born into, and the value every unrecognised stored
# string reads as. Both user stores hard-code this same literal in their
# ``users`` DDL and in the normalising UPDATE beside it -- SQL that an f-string
# would hide from the source-text twin-parity guard
# (tests/test_store_twin_parity.py), so the duplication is deliberate and that
# guard asserts the copies equal this constant.
DEFAULT_USER_GROUP: UserGroup = "unknown"


def parse_user_group(value: object) -> UserGroup:
    """Validate and normalize a user-supplied group value.

    This is the strict write path: only strings naming one of the canonical
    groups are accepted. Callers can map the stable ``invalid_user_group``
    error to their API validation response.
    """

    if not isinstance(value, str):
        raise ValueError("invalid_user_group")
    normalized = value.strip().lower()
    if normalized not in USER_GROUPS:
        raise ValueError("invalid_user_group")
    return cast(UserGroup, normalized)


def coerce_user_group(value: object) -> UserGroup:
    """Read a stored value defensively, defaulting malformed data to the default.

    This is the read-path belt. The braces are the normalising UPDATE each store
    runs in ``_init_schema``: coercing here makes a malformed stored value
    invisible on screen, which is exactly why nothing would ever repair the
    column if the read path were the only fallback.
    """

    try:
        return parse_user_group(value)
    except ValueError:
        return DEFAULT_USER_GROUP


def user_group_label(value: UserGroup) -> str:
    """Return the canonical English display label for a group."""

    return USER_GROUP_LABELS[value]
