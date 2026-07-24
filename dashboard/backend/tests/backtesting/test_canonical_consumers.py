"""Import-boundary tests for Phase 2C4.

Verifies that the backend consumers now use the canonical backtesting/trading
modules and that the only remaining backend-to-scripts dependency is
``external_backtest_service`` importing ``HourlyBacktester``.

The "does not load the legacy script" checks run in a subprocess with an isolated
DATABASE_PATH so that prior in-process test imports cannot hide a dependency.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

FLAT = "backtest_hourly_agent"
CANON_SCRIPT = "dashboard.scripts.backtest_hourly_agent"


def _import_isolated(module: str) -> dict:
    """Import ``module`` in a clean subprocess; report loaded script modules."""
    code = (
        "import importlib, sys, json\n"
        f"importlib.import_module({module!r})\n"
        "print(json.dumps({\n"
        f"  'flat': {FLAT!r} in sys.modules,\n"
        f"  'canon': {CANON_SCRIPT!r} in sys.modules,\n"
        "}))\n"
    )
    with tempfile.TemporaryDirectory(prefix="atl_consumers_") as tmp:
        env = {**os.environ, "DATABASE_PATH": os.path.join(tmp, "t.db")}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
    assert proc.returncode == 0, f"import failed for {module}:\n{proc.stderr}"
    import json
    last = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")][-1]
    return json.loads(last)


# ---------------------------------------------------------------------------
# Isolated import boundary
# ---------------------------------------------------------------------------

def test_leaderboard_baselines_does_not_load_legacy_script():
    res = _import_isolated("dashboard.backend.domain.leaderboard.baselines")
    assert res["flat"] is False
    assert res["canon"] is False


def test_llm_agent_does_not_load_legacy_script():
    res = _import_isolated("dashboard.backend.domain.leaderboard.strategies.llm_agent")
    assert res["flat"] is False
    assert res["canon"] is False


def test_external_backtest_service_loads_no_script():
    res = _import_isolated("dashboard.backend.domain.backtesting.external_run_service")
    # Phase 2C5: HourlyBacktester moved to the canonical engine module, so the
    # service no longer loads ANY dashboard.scripts module (zero backend-to-scripts
    # dependencies). The flat module name must never be created either.
    assert res["flat"] is False
    assert res["canon"] is False


# ---------------------------------------------------------------------------
# Canonical identities (in-process)
# ---------------------------------------------------------------------------

def test_external_backtest_service_uses_canonical_symbols():
    import dashboard.backend.domain.backtesting.external_run_service as ebs
    from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
    from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader
    from dashboard.backend.domain.backtesting.features import TechnicalIndicators
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    assert ebs.PortfolioManager is PortfolioManager
    assert ebs.AlpacaDataLoader is AlpacaDataLoader
    assert ebs.TechnicalIndicators is TechnicalIndicators
    # T2: HourlyBacktester consumption moved from the service to the deduping
    # baseline_worker; the worker binds the canonical engine class, and the
    # service no longer references it at all.
    from dashboard.backend.domain.backtesting import baseline_worker as bw

    assert bw.HourlyBacktester is HourlyBacktester
    assert bw.HourlyBacktester.__module__ == (
        "dashboard.backend.domain.backtesting.engine"
    )
    assert not hasattr(ebs, "HourlyBacktester")
    assert not hasattr(ebs, "bha")


def test_leaderboard_baselines_uses_canonical_symbols():
    import dashboard.backend.domain.leaderboard.baselines as lb
    from dashboard.backend.infrastructure.market_data.alpaca_bars import AlpacaDataLoader
    from dashboard.backend.domain.backtesting.metrics import (
        calculate_max_drawdown,
        calculate_sharpe,
    )

    assert lb.AlpacaDataLoader is AlpacaDataLoader
    assert lb.calculate_sharpe is calculate_sharpe
    assert lb.calculate_max_drawdown is calculate_max_drawdown
    assert lb.INITIAL_CAPITAL == 1000
    assert not hasattr(lb, "bha")


def test_llm_agent_uses_canonical_symbols():
    import dashboard.backend.domain.leaderboard.strategies.llm_agent as la
    from dashboard.backend.domain.backtesting.features import TechnicalIndicators
    from dashboard.backend.domain.backtesting.portfolio_manager import PortfolioManager
    from dashboard.backend.infrastructure.llm.backtest_harness import (
        HAS_ANTHROPIC,
        default_model_name,
        make_llm_client,
    )

    assert la.TechnicalIndicators is TechnicalIndicators
    assert la.PortfolioManager is PortfolioManager
    # llm_agent now builds its client via the canonical make_llm_client()
    # (gateway-aware) rather than referencing Anthropic directly, and resolves
    # its default model id via default_model_name() so it matches that gateway.
    assert la.make_llm_client is make_llm_client
    assert la.HAS_ANTHROPIC == HAS_ANTHROPIC
    assert la.default_model_name is default_model_name
    assert not hasattr(la, "bha")
    assert not hasattr(la, "_load_bha")


# ---------------------------------------------------------------------------
# Constant move + legacy re-export
# ---------------------------------------------------------------------------

def test_initial_capital_canonical_and_reexported():
    from dashboard.backend.domain.backtesting.constants import INITIAL_CAPITAL
    from dashboard.scripts import backtest_hourly_agent as bha

    assert INITIAL_CAPITAL == 1000
    assert bha.INITIAL_CAPITAL == 1000
    assert bha.INITIAL_CAPITAL is INITIAL_CAPITAL


def test_no_flat_module_identity_in_process():
    # Nothing in the suite should create the flat module name.
    assert FLAT not in sys.modules
