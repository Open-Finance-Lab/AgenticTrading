"""
Debug Model Predictions and Signals
Check why all signals are 0
"""

import pandas as pd
import numpy as np
from complete_framework import BacktestingFramework, SyntheticDatasetProvider, StandardFactorCalculator, StandardModelTrainer
from data_interfaces import *
from sklearn.linear_model import Ridge

def debug_model_predictions():
    """Debug the model prediction process"""
    
    print(" Debugging Model Predictions and Signal Generation...")
    print("=" * 60)
    
    # Use the same configuration as the demo
    dataset_config = DatasetInput(
        source_type="csv_hourly",
        file_path="./qlib_data/stock_backup/",
        start_date="2022-09-01",
        end_date="2024-12-31",
        required_fields=["open", "high", "low", "close", "volume"],
        universe="custom_list",
        custom_symbols=["AAPL", "MSFT", "GOOGL", "JPM", "TSLA", "NVDA", "META"],
        adjust_price=True,
        fill_method="ffill",
        min_periods=24
    )
    
    factor_configs = [
        FactorInput(factor_name="momentum_1h", factor_type="alpha", calculation_method="expression", lookback_period=1),
        FactorInput(factor_name="momentum_4h", factor_type="alpha", calculation_method="expression", lookback_period=4),
        FactorInput(factor_name="momentum_24h", factor_type="alpha", calculation_method="expression", lookback_period=24),
        FactorInput(factor_name="volatility_2h", factor_type="risk", calculation_method="expression", lookback_period=2),
        FactorInput(factor_name="rsi_6h", factor_type="technical", calculation_method="expression", lookback_period=6),
        FactorInput(factor_name="bollinger_position", factor_type="technical", calculation_method="expression", lookback_period=20),
        FactorInput(factor_name="volume_momentum", factor_type="volume", calculation_method="expression", lookback_period=5),
        FactorInput(factor_name="price_acceleration", factor_type="alpha", calculation_method="expression", lookback_period=1)
    ]
    
    model_config = ModelInput(
        model_name="advanced_lgbm_trading_model",
        model_type="tree",
        implementation="lightgbm",
        model_class="LGBMRegressor",
        target_type="market_neutral",
        hyperparameters={
            "n_estimators": 300,
                "max_depth": 10,
                "learning_rate": 0.01,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "min_child_samples": 20,
                "min_split_gain": 0.0,
                "num_leaves": 63,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "early_stopping_rounds": 200,
                "n_estimators": 1000,
                "eval_metric": "rmse"
        },
        training_method="rolling",
        training_period=504,
        validation_period=126,
        rebalance_frequency="hourly"
    )
    
    strategy_config = StrategyInput(
        strategy_name="debug_strategy",
        strategy_type="long_only",
        position_method="factor_weight",
        num_positions=7,
        rebalance_frequency="hourly",
        signal_threshold=0.0005,
        use_continuous_positions=True,
        max_position_weight=0.4,
        min_position_weight=0.05,
        signal_scaling_factor=2.0,
        position_sizing_method="dynamic",
        position_decay_rate=0.1,
        signal_smoothing_window=3,
        min_holding_hours=1.0,
        min_holding_days=1,
        min_signal_strength=0.0003,
        max_leverage=1.5,
        target_leverage=1.2,
        long_short_balance=0.5,
        max_consecutive_losses=3,
        profit_taking_threshold=0.05,
        stop_loss_threshold=-0.03,
        transaction_cost=0.0005,
        slippage=0.0002
    )
    
    # Initialize framework
    framework = BacktestingFramework()
    
    # 1. Load data
    print(" Step 1: Loading data...")
    data_loader = SyntheticDatasetProvider()
    data = data_loader.load_data(dataset_config)
    print(f"   Data shape: {data.shape}")
    # data uses a plain column 'date'
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    
    # 2. Calculate factors
    print("\n Step 2: Calculating factors...")
    factor_calculator = StandardFactorCalculator()
    # Calculate factors one-by-one (same approach as framework)
    factor_values = {}
    for factor_input in factor_configs:
        s = factor_calculator.calculate(data, factor_input)
        factor_values[factor_input.factor_name] = s
        val = factor_calculator.validate_factor(s)
        print(f"   {factor_input.factor_name}: {val['valid_ratio']:.2%} valid values")
    # Combine into DataFrame with MultiIndex (date, symbol)
    if factor_values:
        factor_df = pd.concat(factor_values, axis=1)
        # concat creates a column MultiIndex (factor_name, ). Flatten
        factor_df.columns = [c for c in factor_df.columns.get_level_values(0)]
    else:
        factor_df = pd.DataFrame()
    print(f"   Factor shape: {factor_df.shape}")
    print(f"   Factor columns: {list(factor_df.columns)}")
    
    # Check factor statistics
    print("\n Factor Statistics:")
    for col in factor_df.columns:
        factor_data = factor_df[col].dropna()
        print(f"   {col}: Mean={factor_data.mean():.6f}, Std={factor_data.std():.6f}, Range=[{factor_data.min():.6f}, {factor_data.max():.6f}]")
    
    # 3. Feature engineering
    print("\n Step 3: Feature engineering...")
    engineered_features = framework._engineer_features(factor_df)
    print(f"   Engineered features shape: {engineered_features.shape}")
    print(f"   Feature columns: {list(engineered_features.columns)}")
    
    # 4. Create targets
    print("\n Step 4: Creating targets...")
    # Create targets: prediction_horizon default 1 (next period), pass target_type explicitly
    targets = framework._create_real_targets(data, prediction_horizon=1, target_type=model_config.target_type)
    print(f"   Target shape: {targets.shape}")
    target_stats = targets.dropna()
    print(f"   Target statistics: Mean={target_stats.mean():.6f}, Std={target_stats.std():.6f}")
    
    # 5. Data alignment
    print("\n Step 5: Data alignment...")
    # Align features and targets by common MultiIndex (date, symbol)
    common_index = engineered_features.index.intersection(targets.index)
    aligned_features = engineered_features.loc[common_index]
    aligned_targets = targets.loc[common_index]
    print(f"   Aligned features shape: {aligned_features.shape}")
    print(f"   Aligned targets shape: {aligned_targets.shape}")
    
    # 6. Train/test split
    print("\n✂️ Step 6: Train/test split...")
    split_date = pd.Timestamp('2024-04-24')
    original_dates = aligned_features.index.get_level_values('date').unique().sort_values()
    train_dates = original_dates[original_dates < split_date]
    test_dates = original_dates[original_dates >= split_date]
    
    print(f"   Train period: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates)")
    print(f"   Test period: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} dates)")
    
    # Create train/test masks
    train_mask = aligned_features.index.get_level_values('date').isin(train_dates)
    test_mask = aligned_features.index.get_level_values('date').isin(test_dates)
    
    X_train = aligned_features[train_mask]
    y_train = aligned_targets[train_mask]
    X_test = aligned_features[test_mask]
    y_test = aligned_targets[test_mask]
    
    print(f"   Train samples: {len(X_train)} features, {len(y_train)} targets")
    print(f"   Test samples: {len(X_test)} features, {len(y_test)} targets")
    
    # 7. Train model
    print("\n Step 7: Training model...")
    model_trainer = StandardModelTrainer()
    trained_model = model_trainer.train(X_train, y_train, model_config)
    print(f"   Model trained: {type(trained_model)}")
    # Diagnostic: print model summary if available
    try:
        print('   Model keys:', getattr(trained_model, 'keys', lambda: trained_model.__dict__.keys())())
    except Exception:
        pass
    
    # 8. Make predictions
    print("\n🔮 Step 8: Making predictions...")
    # Predict using trainer wrapper if available
    try:
        predictions = model_trainer.predict(X_test, trained_model)
    except Exception:
        predictions = trained_model.predict(X_test)
    print(f"   Predictions shape: {predictions.shape}")
    
    # Create prediction series with proper index
    test_index = X_test.index
    predictions_series = pd.Series(predictions, index=test_index, name='prediction')
    
    print(f"   Prediction statistics:")
    print(f"     Mean: {predictions.mean():.6f}")
    print(f"     Std: {predictions.std():.6f}")
    print(f"     Min: {predictions.min():.6f}")
    print(f"     Max: {predictions.max():.6f}")
    print(f"     Range: {predictions.max() - predictions.min():.6f}")
    
    # Check prediction distribution by symbol
    print(f"\n Prediction by Symbol:")
    for symbol in predictions_series.index.get_level_values('symbol').unique():
        symbol_preds = predictions_series[predictions_series.index.get_level_values('symbol') == symbol]
        print(f"   {symbol}: Mean={symbol_preds.mean():.6f}, Std={symbol_preds.std():.6f}, Count={len(symbol_preds)}")
    
    # 9. Check signal thresholds
    print("\n🚨 Step 9: Checking signal thresholds...")
    print(f"   signal_threshold: {strategy_config.signal_threshold}")
    print(f"   min_signal_strength: {strategy_config.min_signal_strength}")
    print(f"   min_position_weight: {strategy_config.min_position_weight}")
    
    # Check how many predictions exceed thresholds
    abs_predictions = np.abs(predictions)
    above_signal_threshold = np.sum(abs_predictions > strategy_config.signal_threshold)
    above_min_strength = np.sum(abs_predictions > strategy_config.min_signal_strength)
    
    print(f"   Predictions above signal_threshold ({strategy_config.signal_threshold}): {above_signal_threshold}/{len(predictions)} ({100*above_signal_threshold/len(predictions):.2f}%)")
    print(f"   Predictions above min_signal_strength ({strategy_config.min_signal_strength}): {above_min_strength}/{len(predictions)} ({100*above_min_strength/len(predictions):.2f}%)")
    
    # 10. Test signal generation with a single time point
    print("\n Step 10: Testing signal generation...")
    
    # Get a sample of predictions for one time point
    test_date = test_dates[0]
    sample_predictions = predictions_series[predictions_series.index.get_level_values('date') == test_date]
    
    print(f"   Sample date: {test_date}")
    print(f"   Sample predictions:")
    for symbol, pred in sample_predictions.items():
        print(f"     {symbol[1]}: {pred:.6f}")
    
    # Check if any would generate signals
    significant_predictions = sample_predictions[np.abs(sample_predictions) > strategy_config.min_signal_strength]
    print(f"   Significant predictions (>{strategy_config.min_signal_strength}): {len(significant_predictions)}")
    
    if len(significant_predictions) > 0:
        print("    Some predictions are significant enough to generate signals")
        for symbol, pred in significant_predictions.items():
            print(f"     {symbol[1]}: {pred:.6f}")
    else:
        print("    No predictions are significant enough to generate signals")
        print(f"   💡 Consider lowering min_signal_strength (current: {strategy_config.min_signal_strength})")
        
        # Suggest better thresholds
        pred_std = predictions.std()
        pred_mean_abs = np.abs(predictions).mean()
        
        print(f"\n💡 Suggested threshold adjustments:")
        print(f"   Current prediction std: {pred_std:.6f}")
        print(f"   Current prediction mean(abs): {pred_mean_abs:.6f}")
        print(f"   Suggested min_signal_strength: {pred_mean_abs:.6f} (mean absolute)")
        print(f"   Suggested signal_threshold: {pred_std/2:.6f} (0.5 * std)")

        # Temporarily override strategy thresholds with suggested conservative values for quick testing
        suggested_min_strength = max(pred_mean_abs, 1e-8)
        suggested_threshold = max(pred_std / 2, 1e-9)
        print("\nApplying temporary threshold override for quick test:")
        strategy_config.min_signal_strength = suggested_min_strength
        strategy_config.signal_threshold = suggested_threshold
        print(f"   Overridden min_signal_strength: {strategy_config.min_signal_strength:.8f}")
        print(f"   Overridden signal_threshold: {strategy_config.signal_threshold:.8f}")

        # --- Baseline: train a Ridge regression on same features to compare predictions ---
        try:
            ridge_alpha = 0.1
            print("\nTraining Ridge baseline (alpha=0.1) for comparison...")
            ridge = Ridge(alpha=ridge_alpha, random_state=42)
            # Recreate a simple train/test split consistent with earlier split
            total_len = len(aligned_features)
            split_idx = int(total_len * 0.7)
            X_all = aligned_features.values
            y_all = aligned_targets.values
            X_train_b, X_test_b = X_all[:split_idx], X_all[split_idx:]
            y_train_b, y_test_b = y_all[:split_idx], y_all[split_idx:]
            ridge.fit(X_train_b, y_train_b)
            ridge_pred = ridge.predict(X_test_b)
            print(f"Ridge baseline prediction stats: mean={np.mean(ridge_pred):.8f}, std={np.std(ridge_pred):.8f}")
        except Exception as e:
            print(f"Failed to run Ridge baseline: {e}")

if __name__ == "__main__":
    debug_model_predictions()
