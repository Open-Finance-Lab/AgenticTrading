"""SQLite persistence for reporting and native backtest amounts."""

import sqlite3

from dashboard.backend.database import BacktestDatabase


def _insert_run(db: BacktestDatabase, run_id: str) -> None:
    db.insert_run(
        run_id=run_id,
        session_id="currency-audit",
        agent_name="Agent",
        mode="backtest",
        start_date="2026-04-01",
        end_date="2026-04-02",
        initial_equity=1_000,
    )


def test_currency_audit_fields_round_trip_and_usd_rows_remain_nullable(tmp_path):
    db = BacktestDatabase(tmp_path / "currency-audit.db")
    _insert_run(db, "ifind")
    _insert_run(db, "alpaca")

    db.insert_equity_points(
        "ifind",
        [
            {
                "timestamp": "2026-04-01T10:30:00+08:00",
                "equity": 1_000,
                "cash": 800,
                "positions_value": 200,
                "native_equity": 7_000,
                "native_cash": 5_600,
                "native_positions_value": 1_400,
                "fx_rate": 7.0,
            }
        ],
    )
    db.insert_equity_points(
        "alpaca",
        [
            {
                "timestamp": "2026-04-01T10:30:00-04:00",
                "equity": 1_000,
                "cash": 1_000,
                "positions_value": 0,
            }
        ],
    )
    db.insert_trades(
        "ifind",
        [
            {
                "timestamp": "2026-04-01T10:30:00+08:00",
                "symbol": "600519.SH",
                "side": "BUY",
                "quantity": 1,
                "price": 200,
                "value": 200,
                "reference_price": 199.9,
                "gross_value": 200,
                "slippage_amount": 0.1,
                "commission": 0.1,
                "stamp_duty": 0,
                "transfer_fee": 0.01,
                "total_fees": 0.11,
                "net_cash_impact": -200.11,
                "native_price": 1_400,
                "native_value": 1_400,
                "native_reference_price": 1_399.3,
                "native_gross_value": 1_400,
                "native_slippage_amount": 0.7,
                "native_commission": 0.7,
                "native_stamp_duty": 0,
                "native_transfer_fee": 0.07,
                "native_total_fees": 0.77,
                "native_net_cash_impact": -1_400.77,
                "fx_rate": 7.0,
                "market_rule_date": "2026-04-01",
                "market_rule_suspended": False,
                "market_rule_closing_limit_state": "upper",
                "market_rule_official_close": 1_399.0,
                "market_rule_closing_gate_effective": True,
            }
        ],
    )

    ifind_equity = db.get_equity_curve("ifind")[0]
    usd_equity = db.get_equity_curve("alpaca")[0]
    trade = db.get_trades("ifind")[0]
    assert ifind_equity["native_equity"] == 7_000
    assert ifind_equity["fx_rate"] == 7.0
    assert "native_equity" not in usd_equity
    assert "fx_rate" not in usd_equity
    assert trade["price"] == 200
    assert trade["native_price"] == 1_400
    assert trade["native_value"] == 1_400
    assert trade["commission"] == 0.1
    assert trade["total_fees"] == 0.11
    assert trade["net_cash_impact"] == -200.11
    assert trade["native_commission"] == 0.7
    assert trade["native_total_fees"] == 0.77
    assert trade["native_net_cash_impact"] == -1_400.77
    assert trade["fx_rate"] == 7.0
    assert trade["market_rule_date"] == "2026-04-01"
    assert trade["market_rule_suspended"] == 0
    assert trade["market_rule_closing_limit_state"] == "upper"
    assert trade["market_rule_official_close"] == 1_399.0
    assert trade["market_rule_closing_gate_effective"] == 1


def test_existing_schema_is_idempotently_migrated_with_nullable_audit_fields(tmp_path):
    path = tmp_path / "legacy-currency.db"
    first = BacktestDatabase(path)
    del first

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE equity_timeseries RENAME TO old_equity")
    conn.execute(
        """
        CREATE TABLE equity_timeseries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            equity REAL NOT NULL,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            daily_return REAL,
            UNIQUE(run_id, timestamp)
        )
        """
    )
    conn.execute("DROP TABLE old_equity")
    conn.commit()
    conn.close()

    BacktestDatabase(path)
    BacktestDatabase(path)

    conn = sqlite3.connect(path)
    equity_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(equity_timeseries)")
    }
    trade_columns = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
    conn.close()
    assert {
        "native_equity",
        "native_cash",
        "native_positions_value",
        "fx_rate",
    } <= equity_columns
    assert {
        "reference_price",
        "gross_value",
        "slippage_amount",
        "commission",
        "stamp_duty",
        "transfer_fee",
        "total_fees",
        "net_cash_impact",
        "native_price",
        "native_value",
        "native_reference_price",
        "native_gross_value",
        "native_slippage_amount",
        "native_commission",
        "native_stamp_duty",
        "native_transfer_fee",
        "native_total_fees",
        "native_net_cash_impact",
        "fx_rate",
        "market_rule_date",
        "market_rule_suspended",
        "market_rule_closing_limit_state",
        "market_rule_official_close",
        "market_rule_closing_gate_effective",
    } <= trade_columns
