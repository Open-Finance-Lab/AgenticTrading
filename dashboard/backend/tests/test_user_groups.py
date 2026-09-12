"""Contract tests for the canonical admin user-group taxonomy."""

import pytest

from dashboard.backend.domain.user_groups import (
    USER_GROUPS,
    USER_GROUP_LABELS,
    coerce_user_group,
    parse_user_group,
    user_group_label,
)


def test_groups_have_fixed_product_order_and_labels():
    assert USER_GROUPS == (
        "internal",
        "invited",
        "organic",
        "competition",
        "partner",
        "unknown",
    )
    assert [USER_GROUP_LABELS[group] for group in USER_GROUPS] == [
        "Internal",
        "Invited",
        "Organic",
        "Competition",
        "Partner",
        "Unknown",
    ]


@pytest.mark.parametrize("value", ["internal", " ORGANIC ", "competition"])
def test_parse_user_group_normalizes_supported_values(value):
    assert parse_user_group(value) in USER_GROUPS


@pytest.mark.parametrize("value", [None, "", "not-a-group", 3, True])
def test_parse_user_group_rejects_invalid_writes(value):
    with pytest.raises(ValueError, match="invalid_user_group"):
        parse_user_group(value)


@pytest.mark.parametrize("value", [None, "", "old-acquisition-value", object()])
def test_coerce_user_group_defaults_bad_stored_values_to_unknown(value):
    assert coerce_user_group(value) == "unknown"


def test_user_group_label_uses_canonical_display_label():
    assert user_group_label("organic") == "Organic"
