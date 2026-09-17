"""The target-shape fixtures: today's payloads plus the §9 fields the /admin page renders.

PR C ships before the contract re-cut (design §13, re-ordered 2026-09-16). The
committed fixtures under fixtures/admin_analytics/ stay today's shapes and are
validated against the committed models in test_admin_analytics_frontend.py; the
copies under target/ add the fields PR D puts on the models. This module pins
that the two sets differ by exactly those fields and nothing else, so neither
can drift from the other. PR D deletes target/ and this module together — the
second test below fails on purpose the moment a committed fixture carries a §9
field, which is that PR's signal to do so.
"""

import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "admin_analytics"
TARGET = FIXTURES / "target"

# fixture name -> [(path to the object that gains keys, keys added by design §9)]
TARGET_FIELDS = {
    "overview.json": [((), {"billing_lane_mix"})],
    "overview_partial_error.json": [((), {"billing_lane_mix"})],
    "operational.json": [((), {"top_operational_reasons"})],
    "commercial.json": [((), {"purchased_by_day", "consumed_by_day"})],
    "users.json": [
        (("items", 0), {"user_group", "role", "group_badge", "last_meaningful_activity_at"}),
        (("items", 1), {"user_group", "role", "group_badge", "last_meaningful_activity_at"}),
    ],
    "user_detail.json": [((), {"user_group", "role", "group_badge", "last_meaningful_activity_at"})],
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _at(payload, json_path):
    node = payload
    for step in json_path:
        node = node[step]
    return node


def test_target_fixtures_exist_for_every_recut_payload():
    assert sorted(path.name for path in TARGET.glob("*.json")) == sorted(TARGET_FIELDS)


def test_target_fixtures_add_exactly_the_section_9_fields():
    for name, additions in TARGET_FIELDS.items():
        committed = _load(FIXTURES / name)
        target = _load(TARGET / name)
        for json_path, keys in additions:
            target_node = _at(target, json_path)
            committed_node = _at(committed, json_path)
            assert keys <= target_node.keys(), (name, json_path, keys - target_node.keys())
            assert keys.isdisjoint(committed_node.keys()), (
                name, json_path, "a committed fixture carries a §9 field: PR D has landed, delete target/ and this module",
            )
            for key in keys:
                target_node.pop(key)
        assert target == committed, f"{name}: target differs from committed beyond the §9 fields"


def test_target_group_badges_are_the_server_precedence_not_a_client_rule():
    """D11: the fixture records what resolve_group_badge returns; nothing client-side derives it."""
    users = _load(TARGET / "users.json")
    badges = {
        item["user_id"]: (item["role"], item["user_group"], item["commercial_tier"], item["group_badge"])
        for item in users["items"]
    }
    assert badges[101] == ("user", "invited", "invested", "invited")
    assert badges[102] == ("user", "unknown", "unpaid", "free")
