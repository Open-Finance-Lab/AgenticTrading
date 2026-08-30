"""Deep fix for the sys.exit()-in-library-code class (B0/H4 follow-up).

AlpacaDataLoader, BaselineGenerator and the engine used to sys.exit(1) on
missing credentials / missing SDK / empty data. SystemExit is a BaseException:
it sails past `except Exception`, silently kills daemon threads, and wedged
the ASGI loop (the original B0 hang). The B0/H4 fixes added
`except (Exception, SystemExit)` guards at every known call site — this is the
class fix: the libraries raise MarketDataUnavailableError (a plain Exception)
and only the CLI entrypoints translate it to an exit code.
"""

import pytest

import dashboard.backend.baseline_generator as bg_mod
import dashboard.backend.infrastructure.market_data.alpaca_bars as bars_mod


def _clear_creds(monkeypatch, tmp_path, mod):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    # Point the file fallback at an empty directory so a developer's local
    # credentials/alpaca.json can't satisfy the lookup.
    monkeypatch.setattr(mod, "CREDENTIALS_DIR", tmp_path)


def test_market_data_error_is_a_plain_exception():
    """The whole point of the class fix: `except Exception` at server
    boundaries must catch it (SystemExit never was)."""
    assert issubclass(bars_mod.MarketDataUnavailableError, Exception)
    assert not issubclass(bars_mod.MarketDataUnavailableError, SystemExit)


def test_alpaca_loader_missing_credentials_raises_not_exits(monkeypatch, tmp_path):
    _clear_creds(monkeypatch, tmp_path, bars_mod)
    with pytest.raises(bars_mod.MarketDataUnavailableError):
        bars_mod.AlpacaDataLoader()


def test_baseline_fetch_missing_credentials_raises_not_exits(monkeypatch, tmp_path):
    _clear_creds(monkeypatch, tmp_path, bg_mod)
    generator = bg_mod.BaselineGenerator()
    with pytest.raises(bars_mod.MarketDataUnavailableError):
        generator._ensure_credentials()


def test_engine_load_data_empty_raises_not_exits():
    from dashboard.backend.domain.backtesting.engine import HourlyBacktester

    backtester = HourlyBacktester.__new__(HourlyBacktester)  # skip creds init
    backtester.data_loader = type(
        "EmptyLoader", (), {"fetch_bars": lambda self, *a, **k: {}}
    )()
    backtester.start_date = "2026-01-01"
    backtester.end_date = "2026-01-02"
    backtester.data_source = "alpaca"
    backtester.symbols = ["AAPL"]
    with pytest.raises(bars_mod.MarketDataUnavailableError):
        backtester.load_data()


def test_cli_entrypoints_translate_the_error_to_an_exit_code():
    """The CLI boundary is where process exit belongs: both backtest scripts'
    __main__ blocks must catch MarketDataUnavailableError and sys.exit(1)
    (source-level guard, same style as tests/integrations/)."""
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    for name in ("backtest_hourly_agent.py", "backtest_custom_algo.py"):
        source = (scripts_dir / name).read_text(encoding="utf-8")
        assert "MarketDataUnavailableError" in source, (
            f"{name} must translate MarketDataUnavailableError to an exit code"
        )


def test_backtest_cli_closes_pools_at_exit(monkeypatch):
    """Short-lived backtest CLIs must return pooled workers before exit."""
    from pathlib import Path

    from dashboard.backend import db_pool
    from dashboard.scripts import backtest_hourly_agent

    calls = []
    monkeypatch.setattr(db_pool, "close_all_pools", lambda: calls.append(True))

    backtest_hourly_agent._close_pools()

    assert calls == [True]

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    custom_source = (scripts_dir / "backtest_custom_algo.py").read_text(
        encoding="utf-8"
    )
    assert "def _close_pools" in custom_source
    assert "_close_pools()" in custom_source
