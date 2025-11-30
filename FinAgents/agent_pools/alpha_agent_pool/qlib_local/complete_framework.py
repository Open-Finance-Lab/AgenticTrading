"""
Complete Backtesting Framework with Standard Input/Output Interfaces
Integrates all components with exact input/output format specifications
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_interfaces import (
    DatasetInput, FactorInput, ModelInput, StrategyInput, OutputFormat,
    DatasetInterface, FactorInterface, ModelInterface, StrategyInterface,
    EXAMPLE_DATASET_INPUT, EXAMPLE_FACTOR_INPUT, EXAMPLE_MODEL_INPUT,
    EXAMPLE_STRATEGY_INPUT, EXAMPLE_OUTPUT_FORMAT
)
from output_processor import OutputProcessor, PerformanceMetrics

# Import the separated classes
from synthetic_dataset_provider import SyntheticDatasetProvider
from standard_factor_calculator import StandardFactorCalculator
from standard_model_trainer import StandardModelTrainer
from standard_strategy_executor import StandardStrategyExecutor

class BacktestingFramework:
    """
    Main backtesting framework that orchestrates the entire process
    with standardized input/output interfaces
    """
    
    def __init__(self):
        self.dataset_provider = None
        self.factors = []
        self.models = []
        self.strategy = None
        self.output_processor = None
        
    def run_complete_backtest(self,
                            dataset_input: DatasetInput,
                            factor_inputs: List[FactorInput],
                            model_input: Optional[ModelInput],
                            strategy_input: StrategyInput,
                            output_format: OutputFormat,
                            split_method: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run complete backtesting process with standardized inputs/outputs
        
        Args:
            dataset_input: Dataset configuration
            factor_inputs: List of factor configurations
            model_input: Model configuration (optional)
            strategy_input: Strategy configuration
            output_format: Output format specification
            
        Returns:
            Complete results dictionary with all outputs
        """
        
        print(" Starting Complete Backtesting Process...")
        
        # Initialize output processor
        self.output_processor = OutputProcessor(output_format)
        
        # Step 1: Load Dataset
        print(" Loading hourly dataset...")
        dataset_provider = SyntheticDatasetProvider()
        data = dataset_provider.load_data(dataset_input)
        data_quality = dataset_provider.validate_data(data)
        print(f" Dataset loaded: {len(data)} rows, {data_quality['symbols']} symbols")

        # Step 1.5: Load ETF benchmark data if specified in output_format
        benchmark_data = {}
        etf_date_range = None
        
        if hasattr(output_format, "etf_symbols") and output_format.etf_symbols:
            # Use hourly ETF data for better alignment with strategy
            print(" Loading hourly ETF benchmark data...")
            etf_data_temp = {}
            for symbol in output_format.etf_symbols:
                file_path = f"/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup/{symbol}_hourly.csv"
                if os.path.exists(file_path):
                    df_etf = pd.read_csv(file_path)
                    df_etf.columns = [col.lower() for col in df_etf.columns]
                    
                    # Handle datetime column name variations
                    datetime_col = None
                    for col in df_etf.columns:
                        if 'datetime' in col.lower() or 'date' in col.lower():
                            datetime_col = col
                            break
                    
                    if datetime_col is None:
                        print(f"   Warning: No datetime column found in {symbol}, skipping")
                        continue
                    
                    # Handle timezone for hourly data
                    df_etf['datetime'] = pd.to_datetime(df_etf[datetime_col], utc=True).dt.tz_convert(None)
                    df_etf.set_index('datetime', inplace=True)
                    
                    # Filter out unrealistic future dates
                    max_reasonable_date = pd.Timestamp('2024-12-31')
                    df_etf = df_etf[df_etf.index <= max_reasonable_date]
                    
                    etf_data_temp[symbol] = df_etf
                    print(f" Loaded hourly ETF {symbol}: {len(df_etf)} periods (before filtering)")
                else:
                    print(f"ETF hourly benchmark file not found for {symbol}: {file_path}")
                    # Try daily as fallback
                    daily_path = f"/Users/lijifeng/Documents/AI_agent/FinAgent-Orchestration/FinAgents/agent_pools/alpha_agent_pool/qlib/qlib_data/etf_backup/{symbol}_daily.csv"
                    if os.path.exists(daily_path):
                        df_etf = pd.read_csv(daily_path, parse_dates=['Date'])
                    df_etf.columns = [col.lower() for col in df_etf.columns]
                    df_etf['date'] = pd.to_datetime(df_etf['date'], utc=True).dt.tz_convert(None)
                    df_etf.set_index('date', inplace=True)
                    etf_data_temp[symbol] = df_etf
                    print(f" Loaded daily ETF {symbol} as fallback: {len(df_etf)} periods")
            
            if etf_data_temp:
                # Find the common date range across all ETFs
                start_dates = [df.index.min() for df in etf_data_temp.values()]
                end_dates = [df.index.max() for df in etf_data_temp.values()]
                etf_start_date = max(start_dates)  # Latest start date
                etf_end_date = min(end_dates)      # Earliest end date
                
                print(f" ETF data available from {etf_start_date.date()} to {etf_end_date.date()}")
                
                # Update dataset input to match ETF date range
                original_start = pd.to_datetime(dataset_input.start_date)
                original_end = pd.to_datetime(dataset_input.end_date)
                
                aligned_start = max(original_start, etf_start_date)
                aligned_end = min(original_end, etf_end_date)
                
                print(f" Aligning strategy data to {aligned_start.date()} - {aligned_end.date()}")
                
                # Re-filter the main dataset with aligned dates
                data = data[(data['date'] >= aligned_start) & (data['date'] <= aligned_end)]
                print(f" Strategy dataset trimmed to {len(data)} rows for date alignment")
                
                print(f" After date alignment, processing ETF returns...")
                
                # Process ETF data with aligned date range
                for symbol, df_etf in etf_data_temp.items():
                    df_etf_filtered = df_etf[(df_etf.index >= aligned_start) & (df_etf.index <= aligned_end)]
                    close_prices = df_etf_filtered['close']
                    hourly_returns = close_prices.pct_change().dropna()
                    benchmark_data[symbol] = hourly_returns
                    print(f" Loaded {symbol} benchmark: {len(hourly_returns)} hours of returns data")
                
                etf_date_range = (aligned_start, aligned_end)
        
        # Store benchmark data for later use (will be filtered to test period when needed)
        self.main_data = data  # Store for later benchmark creation
        
        # Step 2: Calculate Factors
        print(" Calculating factors...")
        factor_calculator = StandardFactorCalculator()
        factor_values = {}
        
        for factor_input in factor_inputs:
            factor_data = factor_calculator.calculate(data, factor_input)
            factor_validation = factor_calculator.validate_factor(factor_data)
            factor_values[factor_input.factor_name] = factor_data
            print(f" Factor '{factor_input.factor_name}' calculated: {factor_validation['valid_ratio']:.2%} valid values")
        
        factor_df = pd.DataFrame(factor_values) if factor_values else None
        
        # Step 3: Train Model (if specified)
        # Add proper train/test split for backtesting to avoid data leakage
        model_predictions = None
        test_factor_df = factor_df  # Default to using all factors if no model
        
        if model_input is not None:
            print(" Training model...")
            model_trainer = StandardModelTrainer()
            
            if factor_df is not None:
                # Use pre-calculated factors as features with additional feature engineering
                features = self._engineer_features(factor_df)
                
                # Determine prediction horizon based on rebalancing frequency
                if hasattr(strategy_input, 'rebalance_frequency'):
                    if strategy_input.rebalance_frequency == "hourly":
                        prediction_horizon = 1  # Predict 1 hour ahead
                    elif strategy_input.rebalance_frequency == "daily":
                        prediction_horizon = 24  # Predict 24 hours ahead (1 day for hourly data)
                    else:
                        prediction_horizon = 1  # Default to 1 period ahead
                else:
                    prediction_horizon = 1  # Default
                
                # Determine target type based on strategy configuration
                if hasattr(model_input, 'target_type'):
                    target_type = model_input.target_type
                else:
                    # Default to market-neutral for factor models (removes market beta)
                    target_type = "market_neutral"
                
                # Create real future returns as target labels for regression
                targets = self._create_real_targets(data, 
                                                  prediction_horizon=prediction_horizon, 
                                                  target_type=target_type)
                
                # Split data based on original dataset timeline, not synthetic targets
                # This ensures proper chronological order and avoids data leakage
                if split_method is None:
                    split_method = {"type": "ratio", "train_ratio": 0.8}
                
                # Get original data timeline for proper splitting
                original_dates = sorted(data['date'].unique())
                
                # Track test period start date for visualization
                test_start_date = None
                
                if split_method["type"] == "ratio":
                    train_ratio = split_method.get("train_ratio", 0.8)
                    
                    # Split based on original timeline to ensure chronological order
                    split_idx = int(len(original_dates) * train_ratio)
                    train_dates = original_dates[:split_idx]
                    test_dates = original_dates[split_idx:]
                    
                    # Store test start date for visualization
                    test_start_date = test_dates[0] if test_dates else None
                    
                    if train_dates and test_dates:
                        print(f" Train period: {train_dates[0]} to {train_dates[-1]}")
                        print(f" Test period: {test_dates[0]} to {test_dates[-1]}")
                    else:
                        print("Warning: Insufficient data for train/test split")
                        if not train_dates:
                            print("   No training dates available")
                        if not test_dates:
                            print("   No testing dates available")
                    
                elif split_method["type"] == "date":
                    # Split by specific date
                    split_date = pd.to_datetime(split_method.get("split_date"))
                    test_start_date = split_date
                    
                    train_dates = [d for d in original_dates if d < split_date]
                    test_dates = [d for d in original_dates if d >= split_date]
                    
                    if train_dates and test_dates:
                        print(f" Train period: {train_dates[0]} to {train_dates[-1]}")
                        print(f" Test period: {test_dates[0]} to {test_dates[-1]}")
                    else:
                        print("Warning: Insufficient data for date-based train/test split")
                        if not train_dates:
                            print(f"   No training dates before {split_date}")
                        if not test_dates:
                            print(f"   No testing dates after {split_date}")
                
                # Align features and targets based on available data after target generation
                common_index = features.index.intersection(targets.index)
                features_aligned = features.loc[common_index]
                targets_aligned = targets.loc[common_index]
                
                # Filter aligned data based on train/test dates
                if split_method["type"] == "ratio" or split_method["type"] == "date":
                    if train_dates and test_dates:
                        # Create masks based on the properly split dates
                        train_mask = features_aligned.index.get_level_values('date').isin(train_dates)
                        test_mask = features_aligned.index.get_level_values('date').isin(test_dates)
                        
                        # Split features and targets
                        X_train = features_aligned[train_mask]
                        y_train = targets_aligned[train_mask]
                        X_test = features_aligned[test_mask]
                        y_test = targets_aligned[test_mask]
                    else:
                        # Insufficient data for split - use all data for both train and test
                        print("Using all available data for both training and testing due to insufficient data")
                        X_train = features_aligned
                        y_train = targets_aligned
                        X_test = features_aligned
                        y_test = targets_aligned
                    
                else:
                    # Fallback: use all data (not recommended for real backtesting)
                    print("Warning: No proper train/test split, using all data")
                    X_train, y_train = features_aligned, targets_aligned
                    X_test, y_test = features_aligned, targets_aligned
                    test_start_date = None
                
                # Validate data consistency
                print(f" Data validation:")
                print(f"   Features shape: {features_aligned.shape}")
                print(f"   Targets shape: {targets_aligned.shape}")
                print(f"   Train samples: {len(X_train)} features, {len(y_train)} targets")
                print(f"   Test samples: {len(X_test)} features, {len(y_test)} targets")
                
                # Ensure indices match
                if not X_train.index.equals(y_train.index):
                    print("Warning: Training feature and target indices don't match!")
                if not X_test.index.equals(y_test.index):
                    print("Warning: Testing feature and target indices don't match!")
                
                print(f" Data split: {len(X_train)} training samples, {len(X_test)} test samples")
                
                # Determine retrain/walk-forward step
                # priority: explicit retrain_step_periods > retrain_frequency string > defaults
                walk_step = None
                if getattr(model_input, 'retrain_step_periods', None):
                    walk_step = int(model_input.retrain_step_periods)
                elif getattr(model_input, 'retrain_frequency', None):
                    rf = model_input.retrain_frequency
                    if rf == 'weekly' and strategy_input.rebalance_frequency == 'hourly':
                        # Estimate periods per trading day from available original_dates
                        try:
                            dates_series = pd.Series(original_dates)
                            days = dates_series.dt.date
                            counts = days.value_counts()
                            # Use median count per day as robust estimator; default to 7 if unable to compute
                            periods_per_day = int(counts.median()) if len(counts) > 0 else 7
                        except Exception:
                            periods_per_day = 7

                        # Weekly step = 5 trading days * periods per day
                        walk_step = 5 * periods_per_day
                    elif rf == 'weekly' and strategy_input.rebalance_frequency == 'daily':
                        walk_step = 7
                    elif rf == 'daily' and strategy_input.rebalance_frequency == 'hourly':
                        walk_step = 24
                    else:
                        # Default conservative step: 1 rebalance period
                        walk_step = 1
                else:
                    # Default behavior: no retrain (single train), to preserve backward compatibility
                    walk_step = None

                # If walk_step is None -> single training as before
                if walk_step is None:
                    # Train model only on training set
                    model = model_trainer.train(X_train, y_train, model_input)
                    
                    # Generate predictions only on test set (out-of-sample)
                    model_predictions = model_trainer.predict(X_test, model)
                    
                    # Validate model performance on test set
                    model_validation = model_trainer.validate_model(model, (X_test, y_test))
                    print(f" Model trained: {model_validation.get('accuracy', 0):.2%} accuracy (out-of-sample)")
                    
                    # Use only test set data for downstream strategy generation
                    test_factor_df = X_test
                else:
                    # Perform walk-forward retraining with step = walk_step (in rebalance periods)
                    print(f" Performing walk-forward retrain with step {walk_step} periods")
                    # Build a sorted list of unique dates in test set (chronological)
                    test_dates_sorted = sorted(test_dates)

                    # Container for aggregated predictions across test period
                    predictions_list = []

                    # For each walk window starting point within test_dates_sorted
                    i = 0
                    while i < len(test_dates_sorted):
                        # Determine window of test dates to predict in this step
                        window_start = test_dates_sorted[i]
                        window_end_idx = min(i + walk_step, len(test_dates_sorted))
                        window_dates = test_dates_sorted[i:window_end_idx]

                        # Determine training dates for this retrain: take the most recent training_period dates before window_start
                        # Combine train_dates (pre-split) and any dates before window_start
                        all_past_dates = [d for d in original_dates if d < window_start]
                        # Use the tail training_period dates
                        train_window_dates = all_past_dates[-model_input.training_period:]

                        if not train_window_dates:
                            # If no historical dates available, fall back to full training set
                            train_mask = features_aligned.index.get_level_values('date').isin(train_dates)
                        else:
                            train_mask = features_aligned.index.get_level_values('date').isin(train_window_dates)

                        predict_mask = features_aligned.index.get_level_values('date').isin(window_dates)

                        X_train_w = features_aligned[train_mask]
                        y_train_w = targets_aligned[train_mask]
                        X_pred_w = features_aligned[predict_mask]

                        # Skip if no data
                        if len(X_train_w) == 0 or len(X_pred_w) == 0:
                            i = window_end_idx
                            continue

                        # Train and predict for this window
                        model_w = model_trainer.train(X_train_w, y_train_w, model_input)
                        preds_w = model_trainer.predict(X_pred_w, model_w)
                        predictions_list.append(preds_w)

                        print(f" Retrained on {len(X_train_w)} samples; predicted {len(preds_w)} samples for window {window_start} to {window_dates[-1]}")

                        # Advance by walk_step
                        i = window_end_idx

                    # Concatenate all window predictions into a single Series aligned to feature index
                    if predictions_list:
                        model_predictions = pd.concat(predictions_list).sort_index()
                    else:
                        model_predictions = pd.Series(dtype=float)

                    # Use concatenated predictions as signal source
                    test_factor_df = X_test

                # --- Auto-threshold logic: if user sets signal_threshold to 'auto' or <= 0,
                # compute suggested thresholds from model predictions and apply them
                try:
                    if hasattr(strategy_input, 'signal_threshold') and (
                        str(strategy_input.signal_threshold).lower() == 'auto' or float(strategy_input.signal_threshold) <= 0
                    ):
                        preds = model_predictions.copy()
                        # Align and compute distribution stats
                        pred_std = preds.std()
                        pred_mean_abs = preds.abs().mean()

                        # Suggested values: min_signal_strength = mean_abs, signal_threshold = 0.5 * std
                        suggested_min = float(pred_mean_abs)
                        suggested_thresh = float(0.5 * pred_std)

                        # Apply suggested thresholds to strategy_input (temporary override)
                        print(f" Auto threshold enabled: setting min_signal_strength={suggested_min:.6g}, signal_threshold={suggested_thresh:.6g}")
                        # Store originals to restore later if needed
                        strategy_input._orig_signal_threshold = getattr(strategy_input, 'signal_threshold', None)
                        strategy_input._orig_min_signal_strength = getattr(strategy_input, 'min_signal_strength', None)

                        strategy_input.signal_threshold = suggested_thresh
                        strategy_input.min_signal_strength = suggested_min
                except Exception as e:
                    print(f"Warning: Failed to compute auto thresholds: {e}")
        
        # Step 4: Generate Strategy Signals
        # Use only test set data for signal generation to ensure proper backtesting
        print(" Generating strategy signals...")
        strategy_executor = StandardStrategyExecutor()
        
        # Use model predictions (from test set) or test factor values for signals
        signal_data = model_predictions if model_predictions is not None else test_factor_df.iloc[:, 0] if test_factor_df is not None else None
        
        if signal_data is not None:
            signals = strategy_executor.generate_signals(signal_data, strategy_input)
            portfolio_weights = strategy_executor.construct_portfolio(signals, strategy_input)
            print(f" Strategy signals generated: {len(signals)} signals")
        else:
            # Fallback to simple buy-and-hold
            portfolio_weights = self._create_buyhold_portfolio(data, strategy_input)
            signals = pd.DataFrame()
            print("No signals data, using buy-and-hold strategy")
        
        # Step 5: Calculate Strategy Returns
        print(" Calculating strategy returns...")
        strategy_returns = self._calculate_strategy_returns(data, portfolio_weights, strategy_input)
        
        # Create buy-and-hold benchmark only for single-stock strategies
        # For multi-stock portfolios, rely on ETF benchmarks only
        if hasattr(self, 'main_data') and len(self.main_data) > 0:
            symbols_in_data = self.main_data['symbol'].unique()
            
            # Only create single-stock buy-and-hold if we have exactly one symbol
            # and ETF comparison is disabled
            if len(symbols_in_data) == 1 and not output_format.include_etf_comparison:
                main_symbol = symbols_in_data[0]
                
                # Filter to test period only (same dates as strategy returns)
                test_dates = strategy_returns.index
                test_data = self.main_data[self.main_data['date'].isin(test_dates)]
                
                if len(test_data) > 0:
                    symbol_data = test_data[test_data['symbol'] == main_symbol].copy()
                    symbol_data = symbol_data.set_index('date')['close'].sort_index()
                    buy_hold_returns = symbol_data.pct_change().dropna()
                    
                    # Align with strategy returns dates
                    buy_hold_returns = buy_hold_returns.reindex(strategy_returns.index).fillna(0)
                    benchmark_data[f'{main_symbol}_BuyHold'] = buy_hold_returns
                    print(f" Added {main_symbol} buy-and-hold benchmark for test period: {len(buy_hold_returns)} days")
                    
                    # Calculate cumulative return for comparison
                    bh_cumulative = (1 + buy_hold_returns).cumprod().iloc[-1] - 1
                    strategy_cumulative = (1 + strategy_returns).cumprod().iloc[-1] - 1
                    print(f" Performance comparison - Strategy: {strategy_cumulative:.2%}, Buy-Hold: {bh_cumulative:.2%}")
                else:
                    print("No test period data found for buy-and-hold benchmark")
            elif len(symbols_in_data) > 1:
                # Create equal-weighted rebalancing portfolio benchmark for multi-stock strategies
                print(f" Multi-stock portfolio detected ({len(symbols_in_data)} stocks)")
                print(" Creating equal-weighted hourly rebalancing benchmark")
                
                # Filter to test period and calculate equal-weighted portfolio returns
                test_dates = strategy_returns.index
                test_data = self.main_data[self.main_data['date'].isin(test_dates)]
                
                if len(test_data) > 0:
                    # Pivot data to get returns for each stock
                    close_prices = test_data.pivot(index='date', columns='symbol', values='close')
                    # Calculate hourly returns for each stock
                    hourly_returns = close_prices.pct_change().fillna(0)
                    
                    # Equal-weighted rebalancing strategy: rebalance to equal weights every hour
                    equal_weight = 1.0 / len(symbols_in_data)
                    portfolio_returns = hourly_returns * equal_weight  # Each stock gets equal weight
                    portfolio_returns = portfolio_returns.sum(axis=1)  # Sum across stocks
                    
                    # Align with strategy returns dates
                    portfolio_benchmark = portfolio_returns.reindex(strategy_returns.index).fillna(0)
                    benchmark_data['EqualWeighted_Rebalancing'] = portfolio_benchmark
                    
                    # Calculate performance comparison
                    portfolio_cumulative = (1 + portfolio_benchmark).cumprod().iloc[-1] - 1
                    strategy_cumulative = (1 + strategy_returns).cumprod().iloc[-1] - 1
                    print(f" Added equal-weighted rebalancing benchmark: {len(portfolio_benchmark)} periods")
                    print(f" Performance comparison - Strategy: {strategy_cumulative:.2%}, Equal-Weight Rebalancing: {portfolio_cumulative:.2%}")
                else:
                    print("No test period data found for portfolio benchmark")
            else:
                print(" ETF comparison enabled - skipping single-stock buy-and-hold benchmark")
        
        # Step 6: Generate Comprehensive Output
        print(" Generating reports and charts...")
        results = self.output_processor.process_backtest_results(
            strategy_returns=strategy_returns,
            strategy_positions=portfolio_weights,
            factor_values=factor_df,
            model_predictions=model_predictions,
            benchmark_data=benchmark_data if benchmark_data else None,
            test_start_date=test_start_date if 'test_start_date' in locals() else None
        )
        
        # Add additional metadata
        results['input_summary'] = {
            'dataset': dataset_input,
            'factors': factor_inputs,
            'model': model_input,
            'strategy': strategy_input,
            'output_format': output_format
        }
        
        results['data_quality'] = data_quality
        
        print(" Backtesting completed successfully!")
        print(f" Results saved to: {output_format.output_directory}")
        
        return results
    
    def _engineer_features(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """Apply simplified feature engineering to improve model predictive power"""
        
        if factor_df is None or len(factor_df) == 0:
            return factor_df
        
        print(" Applying streamlined feature engineering...")
        
        # Start with original factors
        engineered_features = factor_df.fillna(0).copy()
        
        # Convert to pivot format for vectorized operations
        feature_pivot = {}
        for col in engineered_features.columns:
            feature_pivot[col] = engineered_features[col].unstack(level='symbol')
        
        # Add simple derived features using vectorized operations
        for col_name, col_data in feature_pivot.items():
            if 'momentum' in col_name:
                # Add momentum acceleration (diff)
                acceleration = col_data.diff()
                accel_col_name = f"{col_name}_accel"
                
                # Add rolling mean (5-period trend)
                rolling_mean = col_data.rolling(window=5, min_periods=1).mean()
                mean_col_name = f"{col_name}_5h_mean"
                
                # Stack back to MultiIndex format
                accel_stacked = acceleration.stack()
                accel_stacked.index.names = ['date', 'symbol']
                engineered_features[accel_col_name] = accel_stacked
                
                mean_stacked = rolling_mean.stack()
                mean_stacked.index.names = ['date', 'symbol']
                engineered_features[mean_col_name] = mean_stacked
        
        # Cross-sectional ranking (vectorized)
        for col in factor_df.columns:
            if 'momentum' in col:
                # Calculate cross-sectional ranks
                col_data = engineered_features[col].unstack(level='symbol')
                ranks = col_data.rank(axis=1, pct=True)  # Rank across symbols at each date
                
                rank_col_name = f"{col}_rank"
                rank_stacked = ranks.stack()
                rank_stacked.index.names = ['date', 'symbol']
                engineered_features[rank_col_name] = rank_stacked
        
        # Fill any remaining NaN values
        engineered_features = engineered_features.fillna(0)
        
        print(f" Feature engineering completed: {len(factor_df.columns)} → {len(engineered_features.columns)} features")
        
        return engineered_features
    
    def _create_real_targets(self, data: pd.DataFrame, prediction_horizon: int = 1, target_type: str = "absolute_return") -> pd.Series:
        """Create real target labels for regression model using actual market data
        
        Args:
            data: Historical OHLCV data with columns ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            prediction_horizon: Number of periods ahead to predict (1 = next period, 24 = next day for hourly data)
            target_type: Type of target to create
                - "absolute_return": Individual stock future returns
                - "excess_return": Returns in excess of market average
                - "market_neutral": Returns in excess of equal-weighted market portfolio
            
        Returns:
            pd.Series: Target labels indexed by (date, symbol) for regression model
        """
        
        # Defensive copy and ensure proper datetime ordering
        df = data.copy()
        if 'date' not in df.columns:
            raise ValueError("Input data must contain a 'date' column")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Convert to pivot format for easier calculation (rows = date, cols = symbol)
        close_prices = df.pivot(index='date', columns='symbol', values='close')

        # Ensure we have enough forward history for requested horizon
        h = int(prediction_horizon)
        if h < 1:
            raise ValueError("prediction_horizon must be >= 1")

        if len(close_prices) <= h:
            print(f"Warning: Not enough time periods ({len(close_prices)}) for prediction_horizon={h}. Returning empty target series.")
            self.last_future_1h = pd.Series(dtype=float)
            return pd.Series(dtype=float)

        # Calculate h-step ahead simple returns: (close_{t+h} / close_t) - 1
        future_h = close_prices.shift(-h) / close_prices - 1.0

        # Also compute 1-hour ahead returns for diagnostics and optional use
        future_1h = close_prices.shift(-1) / close_prices - 1.0

        # Select the base target according to target_type (uses future_h)
        if target_type == "absolute_return":
            targets = future_h
        elif target_type in ("excess_return", "market_neutral"):
            # Subtract cross-sectional (row) mean to remove market component
            row_mean = future_h.mean(axis=1)
            targets = future_h.subtract(row_mean, axis=0)
        else:
            # Unknown target_type -> fallback to absolute future_h
            targets = future_h

        # Clean invalid values
        targets = targets.replace([np.inf, -np.inf], np.nan)

        # Winsorize per-symbol using robust quantiles (1%-99%) when enough samples
        for symbol in list(targets.columns):
            s = targets[symbol].dropna()
            if s.size >= 20:
                lo = s.quantile(0.01)
                hi = s.quantile(0.99)
                targets[symbol] = targets[symbol].clip(lower=lo, upper=hi)
            # if too few samples, leave as-is (will be dropped later)

        # Stack into series with MultiIndex (date, symbol) to match features format
        targets_series = targets.stack()
        targets_series.index.names = ['date', 'symbol']
        targets_series = targets_series.dropna()

        # Also store the 1-hour ahead returns as an attribute for diagnostics
        future_1h_series = future_1h.stack()
        future_1h_series.index.names = ['date', 'symbol']
        self.last_future_1h = future_1h_series.dropna()

        # Print summary statistics for both targets (h-step) and 1-hour horizon
        print(f" Target type: {target_type}")
        print(f"   h={h} target - Mean: {targets_series.mean():.6f}, Std: {targets_series.std():.6f}, Samples: {len(targets_series)}")
        print(f"   1-hour target - Mean: {self.last_future_1h.mean():.6f}, Std: {self.last_future_1h.std():.6f}, Samples: {len(self.last_future_1h)}")

        return targets_series
    
    def _create_buyhold_portfolio(self, data: pd.DataFrame, strategy_input: StrategyInput) -> pd.DataFrame:
        """Create simple buy-and-hold portfolio weights"""
        
        symbols = data['symbol'].unique()
        dates = sorted(data['date'].unique())
        
        # Equal weight portfolio
        weight = 1.0 / len(symbols)
        
        portfolio_data = []
        for date in dates:
            for symbol in symbols:
                portfolio_data.append({
                    'date': date,
                    'symbol': symbol,
                    'weight': weight
                })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        return portfolio_df.pivot(index='date', columns='symbol', values='weight').fillna(0)
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, 
                                  portfolio_weights: pd.DataFrame,
                                  strategy_input: StrategyInput) -> pd.Series:
        """Calculate strategy returns from portfolio weights (only for periods with positions)"""
        
        if len(portfolio_weights) == 0:
            return pd.Series(dtype=float)
        
        # Get returns data
        close_prices = data.pivot(index='date', columns='symbol', values='close')
        hourly_returns = close_prices.pct_change().fillna(0)
        
        # Only calculate returns for periods where we have portfolio weights
        portfolio_start = portfolio_weights.index.min()
        portfolio_end = portfolio_weights.index.max()
        
        print(f" Calculating strategy returns for period: {portfolio_start} to {portfolio_end}")
        
        # Filter returns to portfolio period only
        period_returns = hourly_returns[
            (hourly_returns.index >= portfolio_start) & 
            (hourly_returns.index <= portfolio_end)
        ]
        
        # Align weights with returns (only for the portfolio period)
        aligned_weights = portfolio_weights.reindex(period_returns.index).ffill().fillna(0)
        aligned_returns = period_returns.reindex(aligned_weights.index).fillna(0)
        
        print(f" Data alignment: {len(aligned_weights)} weight periods, {len(aligned_returns)} return periods")
        
        # CRITICAL: Use lagged weights for proper t->t+1 signal timing
        # Signal at time t should generate return at t+1
        lagged_weights = aligned_weights.shift(1).fillna(0)
        strategy_returns = (lagged_weights * aligned_returns).sum(axis=1)
        
        # Add timing assertion
        assert len(lagged_weights) == len(aligned_returns), "Weight and return time dimension mismatch"
        
        print(f"💡 Signal timing check: t-period signal -> t+1 period return, using lagged weights")
        
        # Apply transaction costs based on weight changes
        weight_changes = aligned_weights.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * strategy_input.transaction_cost
        
        # Net returns after costs
        net_returns = strategy_returns - transaction_costs
        
        print(f" Strategy returns calculated: {len(net_returns)} periods")
        print(f" Return statistics: Mean={net_returns.mean():.6f}, Std={net_returns.std():.6f}")
        
        return net_returns.fillna(0)


# SyntheticDatasetProvider class has been moved to synthetic_dataset_provider.py


# StandardFactorCalculator class has been moved to standard_factor_calculator.py



# Example usage and testing
if __name__ == "__main__":
    
    # Initialize framework
    framework = BacktestingFramework()
    
    # Define exact inputs with dates that overlap with ETF data
    dataset_config = DatasetInput(
        source_type="synthetic",
        start_date="2024-01-01",  # Updated to have overlap with ETF data
        end_date="2025-01-01",    # Updated to have overlap with ETF data
        required_fields=["open", "high", "low", "close", "volume"],
        universe="top_100",
        adjust_price=True,
        fill_method="ffill",
        min_periods=252
    )
    
    factor_configs = [
        FactorInput(
            factor_name="momentum_10d",
            factor_type="alpha",
            calculation_method="expression",
            expression="close/Ref(close,10)-1",
            lookback_period=10
        ),
        FactorInput(
            factor_name="volatility_20d",
            factor_type="risk",
            calculation_method="expression",
            expression="std(returns,20)",
            lookback_period=20
        )
    ]
    
    model_config = EXAMPLE_MODEL_INPUT
    
    # Enhanced strategy config with continuous position sizing
    strategy_config = StrategyInput(
        strategy_name="continuous_long_only",
        strategy_type="long_only",
        position_method="equal_weight",
        max_position_size=1.0,  # Allow full position for single stock
        min_position_size=0.01,
        num_positions=1,  # Single stock strategy
        long_ratio=1.0,
        leverage=1.0,
        rebalance_frequency="daily",
        rebalance_threshold=0.05,
        stop_loss=None,
        take_profit=None,
        max_drawdown_limit=None,
        signal_threshold=0.0,
        min_holding_days=1,  # Allow daily rebalancing
        min_signal_strength=0.001,
        position_sizing_method="dynamic",  # Use dynamic continuous sizing
        max_consecutive_losses=5,
        profit_taking_threshold=0.15,
        stop_loss_threshold=-0.08,
        use_continuous_positions=True,  # Enable continuous position sizing
        max_position_weight=1.0,  # Maximum 100% position for single stock
        min_position_weight=0.0,  # Minimum 0% position
        signal_scaling_factor=2.0,  # Scale factor for signal strength
        position_decay_rate=0.98,  # Slower position decay (2% daily)
        signal_smoothing_window=3,  # 3-day signal smoothing
        transaction_cost=0.001,
        slippage=0.0005,
        benchmark_symbols=["SPY", "QQQ"]
    )
    
    # Create output config with ETF comparison
    output_config = OutputFormat(
        generate_summary_report=True,
        generate_detailed_report=True,
        generate_performance_chart=True,
        include_etf_comparison=True,  # Enable ETF comparison
        etf_symbols=["SPY"],  # Use only SPY for simpler testing
        save_to_html=True,
        save_to_excel=True,
        output_directory="./backtest_results"
    )
    
    # Run complete backtest with train/test split
    split_config = {"type": "ratio", "train_ratio": 0.8}
    
    results = framework.run_complete_backtest(
        dataset_input=dataset_config,
        factor_inputs=factor_configs,
        model_input=model_config,
        strategy_input=strategy_config,
        output_format=output_config,
        split_method=split_config
    )
    
    print("\n" + "="*60)
    print("- BACKTESTING FRAMEWORK RESULTS SUMMARY")
    print("="*60)
    
    # Print key results
    strategy_metrics = results['strategy_metrics']
    print(f" Strategy Performance:")
    print(f"   Annual Return: {strategy_metrics.annual_return:.2%}")
    print(f"   Sharpe Ratio: {strategy_metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {strategy_metrics.max_drawdown:.2%}")
    print(f"   Win Rate: {strategy_metrics.win_rate:.2%}")
    
    print(f"\n Output Files Generated:")
    for file_type, file_path in results.get('chart_paths', {}).items():
        print(f"   {file_type}: {file_path}")
    
    if 'summary_report_path' in results:
        print(f"   Summary Report: {results['summary_report_path']}")
    
    print(f"\n Complete results available in: {output_config.output_directory}")
