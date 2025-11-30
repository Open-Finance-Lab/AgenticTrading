    def run_simple_backtest_paper_interface(self, predictions, start_time="2023-01-01", 
                                            end_time="2023-12-31", look_back_period=20, 
                                            investment_horizon=5, topk=50, 
                                            risk_thresholds=None, transaction_costs=None,
                                            data_cleaning_rules=None, plot_results=True,
                                            output_dir=None, total_capital=100000.0,
                                            market_data=None):
        """
        Simple backtest function following paper interface design.
        Now implements full value-based accounting and rebalancing.
        """
        print("🚀 Running simple backtest with value-based accounting")
        print(f"   Period: {start_time} to {end_time}")
        print(f"   Capital: ${total_capital:,.2f}")
        
        try:
            # Default parameters
            if risk_thresholds is None:
                risk_thresholds = {'max_position_size': 0.1}
            if transaction_costs is None:
                transaction_costs = {'open_cost': 0.0005, 'close_cost': 0.0015, 'slippage': 0.0}
            
            cost_rate = transaction_costs.get('open_cost', 0.0005) + transaction_costs.get('close_cost', 0.0005)
            
            # 1. Prepare Market Data
            price_lookup = {}
            universe = set()
            
            if market_data is not None:
                md = market_data.copy()
                if isinstance(md.index, pd.MultiIndex): md = md.reset_index()
                col_map = {c: c.lower() for c in md.columns}
                md = md.rename(columns=col_map)
                if 'instrument' in md.columns: md = md.rename(columns={'instrument': 'symbol'})
                if 'datetime' in md.columns: md = md.rename(columns={'datetime': 'date'})
                
                if 'close' in md.columns and 'symbol' in md.columns and 'date' in md.columns:
                    md['date'] = pd.to_datetime(md['date'])
                    # Drop duplicates
                    md = md.drop_duplicates(subset=['date', 'symbol'])
                    # Create lookup (date, symbol) -> price
                    price_pivot = md.pivot(index='date', columns='symbol', values='close')
                    universe = set(md['symbol'].unique())
                    
                    # Forward fill prices
                    price_pivot = price_pivot.fillna(method='ffill')
                    
                    # Convert to dict for O(1) access
                    price_lookup = price_pivot.to_dict(orient='index') # {date: {sym: price}}
            
            if not price_lookup:
                print("⚠️  No market data prices found. Cannot run value-based backtest.")
                return {'status': 'error', 'message': 'No price data'}

            # 2. Align Dates
            valid_dates = sorted(list(price_lookup.keys()))
            dates = [d for d in valid_dates if pd.to_datetime(start_time) <= d <= pd.to_datetime(end_time)]
            
            if not dates:
                return {'status': 'error', 'message': 'No dates in range'}

            # 3. Simulation Loop
            cash = total_capital
            holdings = {sym: 0.0 for sym in universe} # shares
            portfolio_history = []
            
            last_rebalance_idx = -999
            
            # Debug predictions index
            preds_lookup = {}
            if hasattr(predictions, 'index'):
                if isinstance(predictions.index, pd.MultiIndex):
                    preds_df = predictions.reset_index()
                    if preds_df.shape[1] == 3:
                         preds_df.columns = ['date', 'symbol', 'score']
                    else:
                         preds_df.columns = ['date', 'symbol', 'score'] 
                else:
                    preds_df = predictions.reset_index()
                    
                preds_df['date'] = pd.to_datetime(preds_df['date'])
                preds_df = preds_df.drop_duplicates(subset=['date', 'symbol'])
                preds_lookup = preds_df.pivot(index='date', columns='symbol', values='score').to_dict(orient='index')
            
            for i, date in enumerate(dates):
                # Current Prices
                daily_prices = price_lookup.get(date, {})
                if not daily_prices: continue
                
                # Calculate Equity (Mark to Market)
                stock_value = sum(holdings.get(sym, 0.0) * daily_prices.get(sym, 0.0) for sym in universe)
                total_equity = cash + stock_value
                
                # Rebalance Logic
                daily_cost = 0.0
                
                if i - last_rebalance_idx >= investment_horizon:
                    last_rebalance_idx = i
                    
                    # 1. Get Signal Scores for this date
                    scores = preds_lookup.get(date, {})
                    
                    # 2. Construct Target Portfolio (Top K Equal Weight)
                    valid_scores = {s: sc for s, sc in scores.items() if s in universe and not np.isnan(sc)}
                    
                    target_weights = {sym: 0.0 for sym in universe}
                    
                    if valid_scores:
                        sorted_assets = sorted(valid_scores.items(), key=lambda x: x[1], reverse=True)
                        selected = [s for s, sc in sorted_assets[:topk] if sc > 0]
                        
                        if selected:
                            weight = 1.0 / len(selected)
                            weight = min(weight, risk_thresholds.get('max_position_size', 1.0))
                            
                            for sym in selected:
                                target_weights[sym] = weight
                    
                    # 3. Execute Trades (Reallocate)
                    trades = [] # (sym, diff_val, price)
                    
                    for sym in universe:
                        price = daily_prices.get(sym, 0.0)
                        if price <= 0: continue 
                        
                        target_val = total_equity * target_weights[sym]
                        current_val = holdings[sym] * price
                        diff_val = target_val - current_val
                        
                        if abs(diff_val) > (total_equity * 0.001): # Trade if > 0.1% change
                            trades.append((sym, diff_val, price))
                            
                    # Process Sells First (to raise cash)
                    for sym, diff_val, price in [t for t in trades if t[1] < 0]:
                        sell_val = abs(diff_val)
                        cost = sell_val * cost_rate
                        shares_to_sell = sell_val / price
                        
                        if shares_to_sell > holdings[sym]: 
                            shares_to_sell = holdings[sym]
                            sell_val = shares_to_sell * price
                            cost = sell_val * cost_rate
                        
                        holdings[sym] -= shares_to_sell
                        cash += (sell_val - cost)
                        daily_cost += cost
                        
                    # Process Buys
                    for sym, diff_val, price in [t for t in trades if t[1] > 0]:
                        buy_val = diff_val
                        cost = buy_val * cost_rate
                        cost_to_buy = buy_val + cost
                        
                        if cash >= cost_to_buy:
                            shares_to_buy = buy_val / price
                            holdings[sym] += shares_to_buy
                            cash -= cost_to_buy
                            daily_cost += cost
                        else:
                            available = cash
                            if available > 1.0: # min cash
                                trade_val = available / (1 + cost_rate)
                                shares = trade_val / price
                                holdings[sym] += shares
                                cash = 0.0
                                daily_cost += (available - trade_val)

                # Record History
                final_stock_val = sum(holdings.get(sym, 0.0) * daily_prices.get(sym, 0.0) for sym in universe)
                
                portfolio_history.append({
                    'date': date,
                    'equity': cash + final_stock_val,
                    'cash': cash,
                    'cost': daily_cost,
                    'holdings_count': sum(1 for h in holdings.values() if h > 0.001)
                })

            # 4. Analysis
            history_df = pd.DataFrame(portfolio_history).set_index('date')
            if history_df.empty:
                 return {'status': 'error', 'message': 'No history generated'}
                 
            returns_series = history_df['equity'].pct_change().fillna(0.0)
            
            # Stats
            total_return = (history_df['equity'].iloc[-1] / total_capital) - 1
            
            # Snapshot
            print("\n   📊 Portfolio Holdings Snapshot (Sample):")
            if len(history_df) > 0:
                sample_indices = np.linspace(0, len(history_df)-1, 5, dtype=int)
                for i in sample_indices:
                    d = history_df.index[i]
                    row = history_df.loc[d]
                    print(f"      {str(d).split()[0]}: Equity=${row['equity']:,.0f} (Cash=${row['cash']:,.0f}), Positions={int(row['holdings_count'])}")

            results = {
                'status': 'success',
                'performance_metrics': {
                    'total_return': total_return,
                    'sharpe_ratio': (returns_series.mean() / returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0,
                    'max_drawdown': (history_df['equity'] / history_df['equity'].cummax() - 1).min(),
                    'volatility': returns_series.std() * np.sqrt(252)
                },
                'returns_series': returns_series,
                'positions': [] 
            }
            
            # Step 5: Visualizations
            if plot_results and VISUALIZER_AVAILABLE:
                try:
                    visualizer = BacktestVisualizer()
                    plots = {}
                    if output_dir: os.makedirs(output_dir, exist_ok=True)
                    
                    plots['pnl_curve'] = visualizer.plot_pnl_curve(
                        returns_series, save_path=f"{output_dir}/pnl_curve.png" if output_dir else None
                    )
                    plots['drawdown'] = visualizer.plot_drawdown(
                        returns_series, save_path=f"{output_dir}/drawdown.png" if output_dir else None
                    )
                    results['visualizations'] = {'plots_generated': True, 'output_dir': output_dir}
                    print(f"✅ Generated visualization plots")
                except Exception as e:
                    print(f"⚠️ Visualization failed: {e}")
            
            return results
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
