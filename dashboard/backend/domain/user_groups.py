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
    """Read a stored value defensively, defaulting malformed data to unknown."""

    try:
        return parse_user_group(value)
    except ValueError:
        return "unknown"


def user_group_label(value: UserGroup) -> str:
    """Return the canonical English display label for a group."""

    return USER_GROUP_LABELS[value]
