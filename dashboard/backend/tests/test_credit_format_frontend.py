from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
FORMAT_JS = FRONTEND / "js" / "credit-format.js"
APP_HTML = (FRONTEND / "app.html").read_text(encoding="utf-8")


requires_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="node is required for exact Credits formatter tests",
)


def _run_formatter(expressions: list[str]) -> list[str]:
    source = FORMAT_JS.read_text(encoding="utf-8")
    script = "\n".join(
        [
            "const window = globalThis;",
            source,
            f"console.log(JSON.stringify([{', '.join(expressions)}]));",
        ]
    )
    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


@requires_node
def test_credit_formatter_keeps_exact_fixed_six_decimal_values():
    assert _run_formatter(
        [
            'CreditFormat.formatCredits("4.79")',
            'CreditFormat.formatCredits("-0.000137")',
            'CreditFormat.formatCredits("1234")',
            "CreditFormat.formatCreditsMicro(4790000)",
            "CreditFormat.formatCreditsMicro(-137)",
            'CreditFormat.formatCreditsMicro("9007199254740993")',
        ]
    ) == [
        "4.790000",
        "-0.000137",
        "1,234.000000",
        "4.790000",
        "-0.000137",
        "9,007,199,254.740993",
    ]


@requires_node
def test_credit_formatter_rejects_missing_float_and_overprecision_values():
    assert _run_formatter(
        [
            "CreditFormat.formatCredits(null)",
            'CreditFormat.formatCredits("1.0000001")',
            "CreditFormat.formatCreditsMicro(0.5)",
            "CreditFormat.formatCreditsMicro(Number.NaN)",
        ]
    ) == ["—", "—", "—", "—"]


def test_credit_formatter_loads_before_every_consumer():
    formatter_at = APP_HTML.index('src="js/credit-format.js?v=1"')
    for asset in (
        'src="js/credits.js?v=8"',
        'src="js/admin-credits.js?v=6"',
        'src="js/admin-analytics.js?v=2"',
    ):
        assert formatter_at < APP_HTML.index(asset)


def test_static_credit_package_amounts_use_six_decimals():
    for amount in ("0.500000", "1.000000", "2.000000", "5.000000"):
        assert f">{amount} Credits</span>" in APP_HTML
    assert "$1 = 1.000000 Credits" in APP_HTML
