"""``log_safe_token``: the one rule ``ERROR: llm.`` reporters print identifiers through."""

import pytest

from dashboard.backend.infrastructure.llm.execution.log_safe import log_safe_token


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("agent_20260928_024706_cbda3555", "agent_20260928_024706_cbda3555"),
        ("step-1.v2", "step-1.v2"),
        (7, "7"),
        ("x" * 128, "x" * 128),
    ],
)
def test_safe_identifiers_pass_through(value, expected):
    assert log_safe_token(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        # ``re.match`` with ``^...$`` accepts this: ``$`` matches before a
        # trailing newline, which would let a value forge a second log line.
        "run_1\n",
        "run_1\nERROR: llm.forged x=1",
        "a b",
        "k=v",
        "a:b",
        "复盘",
        "",
        "x" * 129,
        None,
        True,
        1.5,
    ],
)
def test_unsafe_values_fall_back(value):
    assert log_safe_token(value) == "-"
    assert log_safe_token(value, fallback="?") == "?"
