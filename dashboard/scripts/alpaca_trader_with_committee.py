#!/usr/bin/env python3
"""
Alpaca Trading Bot - With Trading Committee Discussion

Before executing any trade, this bot convenes a Trading Committee consisting of:
1. Sentiment Agent - Analyzes market sentiment
2. Technical Agent - Analyzes price action  
3. Risk Agent - Assesses portfolio risk

The committee discusses the proposed trade and votes on execution.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

# Direct-execution bootstrap: make the repo root importable so canonical
# `dashboard.backend.*` imports resolve (no-op when run as part of the package).
from _bootstrap import ensure_repo_root

ensure_repo_root()
from dashboard.backend.paths import DATA_DIR, DASHBOARD_DIR
from dashboard.backend.infrastructure.llm.validator import DJIA_30

LOG_FILE = DATA_DIR / "trading_log.json"
SENTIMENT_FILE = DASHBOARD_DIR / "sentiment_scores.json"
# Note: alpaca_cli.py is loaded dynamically, will be updated below
COMMITTEE_SCRIPT = Path(__file__).parent / "trading_committee.py"

# Trading config
# ⭐ CUSTOMIZE THIS LIST to change which stocks are scanned for trading.
# Defaults to the canonical Dow-30 (validator.DJIA_30) — replace with any
# ticker list to customize.
SYMBOLS = list(DJIA_30)

MAX_SHARES_PER_SYMBOL = 5
MIN_CASH_RESERVE = 5000


class AlpacaBotWithCommittee:
    def __init__(self):
        # Note: alpaca_cli.py would be in scripts/ directory
        self.cli_path = Path(__file__).parent / "alpaca_cli.py"
        self.sentiment_scores = self.load_sentiment_scores()
    
    def load_sentiment_scores(self):
        """Load sentiment scores from sentiment_scores.json."""
        try:
            if SENTIMENT_FILE.exists():
                data = json.loads(SENTIMENT_FILE.read_text())
                return data.get("stocks", {})
        except Exception as e:
            print(f"⚠️ Warning: Could not load sentiment scores: {e}", file=__import__('sys').stderr)
        return {}
    
    def run_cmd(self, *args):
        """Run alpaca CLI command."""
        try:
            result = subprocess.run(
                ["python3", str(self.cli_path)] + list(args),
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except Exception:
            return None
    
    def parse_account(self, output):
        """Parse account output."""
        if not output:
            return None
        
        account = {}
        for line in output.split('\n'):
            line = line.strip()
            
            if "Buying Power:" in line:
                try:
                    value = line.split("$")[1].replace(",", "").strip()
                    account["buying_power"] = float(value)
                except Exception:
                    pass
            elif "Cash:" in line and "Portfolio" not in line:
                try:
                    value = line.split("$")[1].replace(",", "").strip()
                    account["cash"] = float(value)
                except Exception:
                    pass
            elif "Portfolio Value:" in line:
                try:
                    value = line.split("$")[1].replace(",", "").strip()
                    account["portfolio_value"] = float(value)
                except Exception:
                    pass
        
        return account if account else None
    
    def parse_quote(self, output, symbol):
        """Parse quote output."""
        if not output or "No data" in output:
            return None
        
        quote = {}
        for line in output.split('\n'):
            if symbol in line and "$" in line:
                try:
                    parts = line.split("$")
                    bid = float(parts[1].strip().split()[0])
                    ask = float(parts[2].strip().split()[0])
                    quote["bid"] = bid
                    quote["ask"] = ask
                    quote["mid"] = (bid + ask) / 2
                    quote["spread"] = ask - bid
                    return quote
                except Exception:
                    pass
        
        return None
    
    def get_account(self):
        """Fetch account."""
        output = self.run_cmd("account")
        return self.parse_account(output)
    
    def get_quote(self, symbol):
        """Get current quote."""
        output = self.run_cmd("quote", symbol)
        return self.parse_quote(output, symbol)
    
    def place_order(self, symbol, qty, side="buy"):
        """Place an actual order."""
        try:
            output = self.run_cmd("order", side, symbol, str(qty), "--force")
            return output
        except Exception as e:
            return f"Error: {e}"
    
    def analyze_spread(self, symbol, quote):
        """Analyze bid-ask spread for trading signal."""
        if not quote:
            return None
        
        spread_pct = (quote["spread"] / quote["mid"]) * 100
        
        if spread_pct < 0.5:
            liquidity = "EXCELLENT"
            signal = "READY_TO_TRADE"
        elif spread_pct < 1.0:
            liquidity = "GOOD"
            signal = "READY_TO_TRADE"
        else:
            liquidity = "POOR"
            signal = "WAIT"
        
        return {
            "symbol": symbol,
            "bid": quote["bid"],
            "ask": quote["ask"],
            "mid": quote["mid"],
            "spread": quote["spread"],
            "spread_pct": round(spread_pct, 3),
            "liquidity": liquidity,
            "signal": signal
        }
    
    def run_trading_committee(self, symbol, current_price, portfolio_cash):
        """
        Run the Trading Committee discussion before executing a trade.
        Returns the committee's recommendation.
        """
        try:
            result = subprocess.run(
                ["python3", str(COMMITTEE_SCRIPT), symbol, str(current_price), str(portfolio_cash)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse JSON output from committee
            output_lines = result.stdout.split('\n')
            json_start = None
            
            # Find where JSON starts
            for i, line in enumerate(output_lines):
                if line.strip().startswith('{'):
                    json_start = i
                    break
            
            if json_start is not None:
                json_str = '\n'.join(output_lines[json_start:])
                recommendation = json.loads(json_str)
                
                # Print committee discussion (from stdout before JSON)
                if json_start > 0:
                    discussion = '\n'.join(output_lines[:json_start])
                    print(discussion)
                
                return recommendation
            else:
                print(f"⚠️ Warning: Could not parse committee output for {symbol}", file=__import__('sys').stderr)
                return None
        
        except Exception as e:
            print(f"⚠️ Warning: Committee discussion failed for {symbol}: {e}", file=__import__('sys').stderr)
            return None


def log_action(action, details):
    """Log action."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    if LOG_FILE.exists():
        try:
            logs = json.loads(LOG_FILE.read_text())
        except Exception:
            logs = []
    
    logs.append({
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details
    })
    
    LOG_FILE.write_text(json.dumps(logs, indent=2))


def main():
    """Main trading logic with committee approval."""
    print("🎯 Alpaca Smart Trading Bot (With Committee)\n")
    
    bot = AlpacaBotWithCommittee()
    
    # Check account
    print("💼 Checking account...")
    account = bot.get_account()
    if not account:
        print("❌ Could not fetch account")
        return
    
    cash = account.get("cash", 0)
    portfolio = account.get("portfolio_value", 0)
    power = account.get("buying_power", 0)
    
    print(f"   Cash: ${cash:,.2f}")
    print(f"   Portfolio: ${portfolio:,.2f}")
    print(f"   Buying Power: ${power:,.2f}\n")
    
    log_action("account_check", {
        "cash": cash,
        "portfolio": portfolio,
        "buying_power": power
    })
    
    # Fetch quotes
    print(f"📊 Analyzing {len(SYMBOLS)} symbols...\n")
    analyses = []
    
    for symbol in SYMBOLS:
        print(f"   Fetching {symbol}...")
        quote = bot.get_quote(symbol)
        
        if quote:
            analysis = bot.analyze_spread(symbol, quote)
            if analysis:
                analyses.append(analysis)
                print(f"     {symbol}: Bid ${analysis['bid']:.2f} | Ask ${analysis['ask']:.2f} | Spread {analysis['spread_pct']:.3f}% | {analysis['liquidity']}")
        else:
            print(f"     {symbol}: ⚠️ No data")
    
    print(f"\n📈 Market Analysis Summary:\n")
    
    excellent = [a for a in analyses if a["liquidity"] == "EXCELLENT"]
    good = [a for a in analyses if a["liquidity"] == "GOOD"]
    poor = [a for a in analyses if a["liquidity"] == "POOR"]
    
    print(f"   Excellent Liquidity: {len(excellent)}")
    print(f"   Good Liquidity: {len(good)}")
    print(f"   Poor Liquidity: {len(poor)}")
    
    log_action("market_analysis", {
        "total_symbols": len(analyses),
        "excellent_liquidity": len(excellent),
        "analyses": analyses
    })
    
    # Show best opportunities
    if excellent:
        print(f"\n✅ Excellent Opportunities (sort by spread):")
        for analysis in sorted(excellent, key=lambda x: x["spread_pct"])[:3]:
            print(f"   {analysis['symbol']}: ${analysis['mid']:.2f} (spread {analysis['spread_pct']:.3f}%)")
    
    # Find best trade and run committee
    if cash > MIN_CASH_RESERVE and excellent:
        print(f"\n" + "="*70)
        print(f"🏛️ TRADING COMMITTEE CONVENING")
        print(f"="*70)
        
        # Pick best by spread quality
        best_trade = sorted(excellent, key=lambda x: x["spread_pct"])[0]
        symbol = best_trade["symbol"]
        
        print(f"Proposed Trade: BUY {symbol} @ ${best_trade['mid']:.2f}")
        print(f"Liquidity: {best_trade['spread_pct']:.3f}% spread\n")
        
        # ⭐ RUN TRADING COMMITTEE ⭐
        committee_rec = bot.run_trading_committee(symbol, best_trade['mid'], cash)
        
        if committee_rec:
            consensus = committee_rec.get("consensus", "DO_NOT_TRADE")
            conviction = committee_rec.get("conviction", "None")
            qty = committee_rec.get("position_size", 0)
            
            print(f"\n{'='*70}")
            print(f"📋 COMMITTEE DECISION: {consensus}")
            print(f"   Conviction: {conviction}")
            print(f"   Recommended Position: {qty} shares")
            print(f"{'='*70}\n")
            
            # Execute if committee approved
            if qty > 0:
                print(f"✅ TRADE APPROVED BY COMMITTEE")
                print(f"   Symbol: {symbol}")
                print(f"   Entry: ${best_trade['mid']:.2f}")
                print(f"   Shares: {qty}")
                print(f"   Estimated Cost: ${best_trade['mid'] * qty:,.2f}\n")
                
                print(f"🚀 Executing order...")
                order_result = bot.place_order(symbol, qty, "buy")
                
                if order_result and "error" not in order_result.lower():
                    print(f"✅ ORDER EXECUTED")
                    
                    log_action("trade_executed_by_committee", {
                        "symbol": symbol,
                        "quantity": qty,
                        "entry_price": best_trade["mid"],
                        "spread": best_trade["spread_pct"],
                        "committee_consensus": consensus,
                        "committee_conviction": conviction,
                        "status": "success"
                    })
                else:
                    print(f"❌ ORDER FAILED: {order_result}")
                    
                    log_action("trade_failed", {
                        "symbol": symbol,
                        "quantity": qty,
                        "error": order_result
                    })
            else:
                print(f"❌ TRADE REJECTED BY COMMITTEE")
                print(f"   Reason: {consensus}")
                
                log_action("trade_rejected_by_committee", {
                    "symbol": symbol,
                    "consensus": consensus,
                    "conviction": conviction
                })
        else:
            print(f"⚠️ Committee discussion failed. Skipping trade.")
            log_action("trade_skipped", {
                "reason": "committee_discussion_failed"
            })
    else:
        reason = "insufficient cash" if cash <= MIN_CASH_RESERVE else "no excellent liquidity"
        print(f"\n⏸️ No trades executed ({reason})")
    
    print("\n✅ Market analysis complete!")


if __name__ == "__main__":
    main()
