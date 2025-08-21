"""
Standard Strategy Executor for Backtesting Framework
Handles signal generation and portfolio construction
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from data_interfaces import StrategyInput, StrategyInterface
from collections import deque


class StandardStrategyExecutor(StrategyInterface):
    """Standard strategy executor with position tracking for single stock strategies"""
    
    def __init__(self):
        # Track current positions for single stock strategies - now supports continuous weights
        self.current_positions = {}  # {symbol: {'weight': float, 'entry_date': date, 'entry_price': float, 'entry_prediction': float, 'signal_history': []}}
        self.position_history = []   # Track all position changes
        self.consecutive_losses = {}  # Track consecutive losses per symbol
        self.price_data = {}  # Cache price data for stop loss/take profit calculations
        self.signal_buffer = {}  # Buffer for signal smoothing {symbol: deque}
    
    def _calculate_continuous_position_weight(self, prediction: float, strategy_input: StrategyInput, 
                                            current_weight: float = 0.0, symbol: str = None) -> float:
        """Calculate continuous position weight based on signal strength and strategy parameters"""
        
        if not strategy_input.use_continuous_positions:
            # Fall back to binary positions (0 or max weight)
            if abs(prediction) > strategy_input.signal_threshold:
                return strategy_input.max_position_weight if prediction > 0 else -strategy_input.max_position_weight
            else:
                return 0.0
        
        # Apply signal smoothing if enabled (only if window > 1)
        if strategy_input.signal_smoothing_window > 1 and symbol:
            if symbol not in self.signal_buffer:
                self.signal_buffer[symbol] = deque(maxlen=strategy_input.signal_smoothing_window)
            
            self.signal_buffer[symbol].append(prediction)
            smoothed_prediction = sum(self.signal_buffer[symbol]) / len(self.signal_buffer[symbol])
        else:
            smoothed_prediction = prediction
        
        # Calculate base weight from signal strength
        if strategy_input.position_sizing_method == "signal_scaled":
            # Scale position size directly by signal strength
            raw_weight = smoothed_prediction * strategy_input.signal_scaling_factor
            
        elif strategy_input.position_sizing_method == "dynamic":
            # Use tanh function for smooth scaling with saturation
            signal_strength = abs(smoothed_prediction)
            
            # Apply tanh scaling for smooth transition
            if signal_strength > strategy_input.min_signal_strength:
                # Fix normalization: use signal_threshold as denominator to avoid +0.01 domination
                # This ensures small signals don't get amplified inappropriately
                threshold = max(strategy_input.signal_threshold, 0.001)  # Minimum threshold to avoid division by zero
                normalized_signal = signal_strength / threshold
                scaled_strength = np.tanh(normalized_signal * strategy_input.signal_scaling_factor)
                raw_weight = scaled_strength * strategy_input.max_position_weight
                
                # Apply sign of prediction
                raw_weight = raw_weight if smoothed_prediction > 0 else -raw_weight
            else:
                raw_weight = 0.0
                
        elif strategy_input.position_sizing_method == "volatility_adjusted":
            # Adjust position size based on signal volatility (simplified version)
            # This would typically use historical volatility of the asset
            signal_strength = abs(smoothed_prediction)
            volatility_factor = min(1.0, 1.0 / (signal_strength + 0.01))  # Lower volatility = higher position
            raw_weight = smoothed_prediction * strategy_input.signal_scaling_factor * volatility_factor
            
        else:  # "fixed" method
            # Fixed position size based on signal direction
            if abs(smoothed_prediction) > strategy_input.signal_threshold:
                raw_weight = strategy_input.max_position_weight if smoothed_prediction > 0 else -strategy_input.max_position_weight
            else:
                raw_weight = 0.0
        
        # 1. Explicit enforcement of maximum position weights first
        # Clip to maximum bounds before applying decay logic
        max_bound = strategy_input.max_position_weight
        min_bound = 0.0 if strategy_input.strategy_type == "long_only" else -max_bound
        raw_weight = np.clip(raw_weight, min_bound, max_bound)
        
        # Apply position decay with consistent units (compare prediction strength vs normalized current weight)
        # Fix: normalize current_weight back to "prediction equivalent" for fair comparison
        if current_weight != 0 and abs(smoothed_prediction) > 0:
            # Convert current_weight back to equivalent prediction strength for comparison
            current_prediction_equivalent = abs(current_weight) / strategy_input.max_position_weight
            if abs(smoothed_prediction) < current_prediction_equivalent:
                decay_factor = strategy_input.position_decay_rate
                raw_weight = raw_weight * decay_factor + current_weight * (1 - decay_factor)
        
        # Apply constraints for different strategy types again after decay
        if strategy_input.strategy_type == "long_only":
            raw_weight = max(0, raw_weight)  # No short positions
        
        # 2. Final clipping to ensure bounds are respected
        final_weight = np.clip(raw_weight, min_bound, max_bound)

        # 3. Apply minimum position threshold with proper sign handling
        if abs(final_weight) > 0 and abs(final_weight) < strategy_input.min_position_weight:
            # If position is non-zero but below minimum, set to minimum (preserving sign)
            min_pos = strategy_input.min_position_weight
            if strategy_input.strategy_type == "long_only":
                # For long_only, minimum positive weight or zero
                final_weight = min_pos if final_weight > 0 else 0.0
            else:
                # For long_short, apply minimum absolute value with correct sign
                final_weight = min_pos * np.sign(final_weight)
        
        return final_weight
    
    def _smooth_position_transition(self, current_weight: float, target_weight: float, 
                                  time_held: float, min_holding_time: float, 
                                  rebalance_frequency: str) -> float:
        """Smooth position transitions to avoid frequent small changes"""
        
        # If we haven't held the position for minimum time, maintain current weight
        if time_held < min_holding_time and current_weight != 0:
            return current_weight
            
        # Calculate transition speed based on rebalancing frequency and weight difference
        weight_diff = abs(target_weight - current_weight)
        
        if rebalance_frequency == 'hourly':
            # For hourly rebalancing, use faster transitions
            if weight_diff < 0.1:  # Less than 10% change
                transition_speed = 0.5  # 50% transition per hour
            else:
                transition_speed = 0.8  # 80% transition per hour for larger changes
        else:
            # For daily/weekly/monthly rebalancing, use slower transitions
            if weight_diff < 0.1:  # Less than 10% change
                transition_speed = 0.3  # 30% transition per period
            else:
                transition_speed = 0.6  # 60% transition per period for larger changes
            
        # Calculate smoothed weight
        smoothed_weight = current_weight + (target_weight - current_weight) * transition_speed
        
        return smoothed_weight

    def _enforce_min_max_and_normalize(self, weights: Dict[str, float], strategy_input: StrategyInput) -> Dict[str, float]:
        """Enforce per-position min/max and normalize to respect target leverage.

        This ensures that any non-zero position is at least `min_position_weight` and
        no position exceeds `max_position_weight`. If the total absolute weight
        exceeds the target leverage, weights are proportionally scaled down while
        preserving the floor where possible.
        """
        min_pos = float(getattr(strategy_input, 'min_position_weight', 0.0))
        max_pos = float(getattr(strategy_input, 'max_position_weight', 1.0))
        target_leverage = float(getattr(strategy_input, 'target_leverage', 1.0))

        # Apply floor to any non-zero weights
        adjusted = {s: (np.sign(w) * max(min_pos, abs(w)) if abs(w) > 0 else 0.0) for s, w in weights.items()}

        # Cap per-position at max_pos
        for s in adjusted:
            if abs(adjusted[s]) > max_pos:
                adjusted[s] = np.sign(adjusted[s]) * max_pos

        # Iteratively normalize to target_leverage while trying to preserve floor
        for _ in range(5):
            total_abs = sum(abs(v) for v in adjusted.values())
            if total_abs <= target_leverage or total_abs == 0:
                break

            # scale down proportionally
            scale = target_leverage / total_abs
            adjusted = {s: np.sign(v) * max(min_pos, abs(v) * scale) if abs(v) > 0 else 0.0 for s, v in adjusted.items()}

            # re-cap after scaling
            for s in adjusted:
                if abs(adjusted[s]) > max_pos:
                    adjusted[s] = np.sign(adjusted[s]) * max_pos

        # Final cleanup: ensure values respect floor and cap
        for s in adjusted:
            if 0 < abs(adjusted[s]) < min_pos:
                adjusted[s] = np.sign(adjusted[s]) * min_pos
            if abs(adjusted[s]) > max_pos:
                adjusted[s] = np.sign(adjusted[s]) * max_pos

        return adjusted
    
    def generate_signals(self, predictions: pd.Series, strategy_input: StrategyInput) -> pd.DataFrame:
        """Generate hourly portfolio weights for all stocks at each rebalancing time"""
        
        signals_data = []
        
        # Define parameters
        signal_threshold = getattr(strategy_input, 'signal_threshold', 0.0)
        min_holding_days = getattr(strategy_input, 'min_holding_days', 3)
        rebalance_frequency = getattr(strategy_input, 'rebalance_frequency', 'daily')
        min_signal_strength = getattr(strategy_input, 'min_signal_strength', 0.01)
        
        # Sort dates to process chronologically
        sorted_dates = sorted(predictions.index.get_level_values('date').unique())
        all_symbols = sorted(predictions.index.get_level_values('symbol').unique())
        
        print(f" Generating hourly portfolio weights for {len(all_symbols)} stocks across {len(sorted_dates)} time periods")
        
        # Group by date and generate portfolio weights for ALL stocks at each time
        for date in sorted_dates:
            date_predictions = predictions.loc[date]
            
            # Generate portfolio weights for ALL stocks at this time point
            portfolio_weights = {}
            total_abs_signal = 0
            
            # Calculate individual weights for all stocks
            for symbol in all_symbols:
                if symbol in date_predictions.index:
                    prediction = date_predictions[symbol]
                else:
                    prediction = 0.0  # Default to 0 if no prediction available
                    
                # Get current position for this symbol
                current_pos = self.current_positions.get(symbol, {'weight': 0.0, 'entry_date': None})
                current_weight = current_pos['weight']
                
                # Calculate target weight based on prediction
                target_weight = self._calculate_continuous_position_weight(
                    prediction, strategy_input, current_weight, symbol
                )
                    
                portfolio_weights[symbol] = {
                    'prediction': prediction,
                    'current_weight': current_weight,
                    'target_weight': target_weight
                }
                
                total_abs_signal += abs(prediction)
            
            # Normalize weights and generate portfolio for all symbols
            if strategy_input.position_method == "factor_weight":
                # Filter symbols with sufficient signal strength first
                symbols_with_signal = [s for s in all_symbols 
                                     if abs(portfolio_weights[s]['prediction']) > min_signal_strength]
                
                if len(symbols_with_signal) > 0:
                    # Calculate total signal for normalization (only from significant signals)
                    total_signal_strength = sum(abs(portfolio_weights[s]['prediction']) 
                                              for s in symbols_with_signal)
                    
                    # Calculate raw weights first
                    raw_weights = {}
                    for symbol in all_symbols:
                        symbol_data = portfolio_weights[symbol]
                        prediction = symbol_data['prediction']
                        
                        # Only trade if signal is significant
                        if symbol in symbols_with_signal and total_signal_strength > 0:
                            # Calculate normalized weight based on relative prediction strength
                            raw_weight = (abs(prediction) / total_signal_strength) * np.sign(prediction)
                            # Apply strategy constraints
                            if strategy_input.strategy_type == "long_only":
                                raw_weight = max(0, raw_weight)  # No short positions
                            raw_weights[symbol] = raw_weight
                        else:
                            raw_weights[symbol] = 0.0
                    
                    # Apply position size limits and renormalize if needed
                    max_weight_limit = strategy_input.max_position_weight
                    total_raw_weight = sum(abs(w) for w in raw_weights.values())
                    
                    # Check if any weight exceeds limit
                    needs_rebalancing = any(abs(w) > max_weight_limit for w in raw_weights.values())
                    
                    if needs_rebalancing and total_raw_weight > 0:
                        # Cap weights at maximum and renormalize
                        capped_weights = {}
                        for symbol, raw_weight in raw_weights.items():
                            if abs(raw_weight) > max_weight_limit:
                                capped_weights[symbol] = max_weight_limit * np.sign(raw_weight)
                            else:
                                capped_weights[symbol] = raw_weight
                        
                        # Renormalize to maintain target leverage
                        total_capped = sum(abs(w) for w in capped_weights.values())
                        target_leverage = min(strategy_input.target_leverage, 1.2)  # Respect our 1.2 limit
                        
                        if total_capped > 0:
                            # Make sure normalization doesn't violate individual weight limits
                            normalization_factor = min(target_leverage / total_capped, 1.0)
                            scaled = {symbol: w * normalization_factor for symbol, w in capped_weights.items()}
                            # Enforce min/max and normalize with helper
                            final_weights = self._enforce_min_max_and_normalize(scaled, strategy_input)
                            
                            # Double-check that no weight exceeds the limit after normalization
                            max_final_weight = max(abs(w) for w in final_weights.values()) if final_weights.values() else 0
                            if max_final_weight > max_weight_limit:
                                # If still exceeding, apply conservative scaling
                                conservative_factor = max_weight_limit / max_final_weight
                                final_weights = {symbol: w * conservative_factor for symbol, w in final_weights.items()}
                        else:
                            final_weights = capped_weights
                    else:
                        # Even for raw weights, ensure they don't exceed limits
                        # Even for raw weights, ensure min/max and normalize
                        final_weights = self._enforce_min_max_and_normalize(raw_weights, strategy_input)
                    
                    # Now assign final weights to positions
                    for symbol in all_symbols:
                        weight = final_weights.get(symbol, 0.0)  # Default to 0 if not in final_weights

                        # Retrieve symbol-specific bookkeeping data
                        symbol_data = portfolio_weights.get(symbol, {'current_weight': 0.0, 'prediction': 0.0})
                        prediction = symbol_data.get('prediction', 0.0)
                        weight_before = symbol_data.get('current_weight', 0.0)

                        # Update position tracking
                        self.current_positions[symbol] = {
                            'weight': weight,
                            'entry_date': date,
                            'entry_price': prediction
                        }

                        # CRITICAL ASSERTIONS for constraint checking
                        assert weight >= -1e-6, f"Long-only constraint violation: {symbol} weight={weight:.6f} < 0"
                        max_weight_limit = strategy_input.max_position_weight + 0.001  # Small tolerance for floating point
                        assert abs(weight) <= max_weight_limit, f"Single position weight limit exceeded: {symbol} weight={abs(weight):.3f} > {strategy_input.max_position_weight:.1%}"

                        # Record signal for all stocks at this time point
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': weight,
                            'prediction': prediction,
                            'weight_before': weight_before,
                            'weight_after': weight,
                            'weight_change': abs(weight - weight_before),
                            'action': 'REBALANCE'
                        })
                    
                    # Portfolio-level constraint check after all weights assigned
                    current_total_leverage = sum(abs(self.current_positions[s]['weight']) for s in all_symbols)
                    assert current_total_leverage <= 1.21, f"Portfolio leverage exceeded: {current_total_leverage:.3f} > 1.2"
                else:
                    # No significant signals, record zero weights for all stocks
                    for symbol in all_symbols:
                        symbol_data = portfolio_weights[symbol]
                        weight = 0.0
                        
                        # Update position tracking
                        self.current_positions[symbol] = {
                            'weight': weight,
                            'entry_date': date,
                            'entry_price': symbol_data['prediction']
                        }
                        
                        # Record signal for all stocks
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': weight,
                            'prediction': symbol_data['prediction'],
                            'weight_before': symbol_data['current_weight'],
                            'weight_after': weight,
                            'weight_change': abs(weight - symbol_data['current_weight']),
                            'action': 'NO_SIGNAL'
                        })
            
            elif strategy_input.position_method == "equal_weight":
                # Equal weight across stocks with signals
                symbols_with_signal = [s for s in all_symbols if abs(portfolio_weights[s]['prediction']) > min_signal_strength]
                
                if len(symbols_with_signal) > 0:
                    equal_weight = 1.0 / len(symbols_with_signal)

                    # Build raw equal weights dict first
                    raw_equal = {}
                    for symbol in all_symbols:
                        symbol_data = portfolio_weights[symbol]
                        prediction = symbol_data['prediction']
                        if symbol in symbols_with_signal:
                            w = equal_weight if prediction > 0 else 0.0
                            if strategy_input.strategy_type == "long_short":
                                w = equal_weight * np.sign(prediction)
                        else:
                            w = 0.0
                        raw_equal[symbol] = w

                    # Enforce min/max and normalize
                    final_equal_weights = self._enforce_min_max_and_normalize(raw_equal, strategy_input)

                    for symbol in all_symbols:
                        symbol_data = portfolio_weights[symbol]
                        prediction = symbol_data['prediction']
                        weight_before = symbol_data['current_weight']
                        weight = final_equal_weights.get(symbol, 0.0)

                        # Update position tracking
                        self.current_positions[symbol] = {
                            'weight': weight,
                            'entry_date': date,
                            'entry_price': prediction
                        }

                        # Record signal for all stocks
                        signals_data.append({
                            'date': date,
                            'symbol': symbol,
                            'signal': weight,
                            'prediction': prediction,
                            'weight_before': weight_before,
                            'weight_after': weight,
                            'weight_change': abs(weight - weight_before),
                            'action': 'REBALANCE'
                        })
                
            else:
                # Use continuous position weights directly
                for symbol in all_symbols:
                    symbol_data = portfolio_weights[symbol]
                    target_weight = symbol_data['target_weight']
                    current_weight = symbol_data['current_weight']
                    prediction = symbol_data['prediction']
                    
                    # Update position tracking
                    self.current_positions[symbol] = {
                        'weight': target_weight,
                        'entry_date': date,
                        'entry_price': prediction
                    }
                    
                    # Record signal for all stocks
                    signals_data.append({
                        'date': date,
                        'symbol': symbol,
                        'signal': target_weight,
                        'prediction': prediction,
                        'weight_before': current_weight,
                        'weight_after': target_weight,
                        'weight_change': abs(target_weight - current_weight),
                        'action': 'REBALANCE'
                    })
        
        return pd.DataFrame(signals_data)
    
    def _can_rebalance(self, current_date, entry_date, rebalance_frequency):
        """Check if rebalancing is allowed based on frequency settings"""
        if entry_date is None:
            return True
        
        # Calculate time difference based on rebalancing frequency
        if rebalance_frequency == 'hourly':
            # For hourly rebalancing, check hours difference
            time_diff_hours = (current_date - entry_date).total_seconds() / 3600
            return time_diff_hours >= 1.0  # Allow rebalancing every hour
        else:
            # For daily/weekly/monthly, use days difference
            days_diff = (current_date - entry_date).days
        
        if rebalance_frequency == 'daily':
            return True  # Daily rebalancing always allowed
        elif rebalance_frequency == 'weekly':
            return days_diff >= 7
        elif rebalance_frequency == 'monthly':
            return days_diff >= 30
        else:
            return True  # Default to daily
    
    def _check_stop_loss_take_profit(self, symbol, current_price, entry_price, strategy_input):
        """Check if stop loss or take profit conditions are met"""
        if entry_price is None or current_price is None:
            return False, "NONE"
        
        # Calculate current P&L percentage
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Check take profit (if enabled)
        profit_threshold = getattr(strategy_input, 'profit_taking_threshold', None)
        if profit_threshold is not None and pnl_pct >= profit_threshold:
            return True, "TAKE_PROFIT"
        
        # Check stop loss (if enabled)
        stop_loss_threshold = getattr(strategy_input, 'stop_loss_threshold', None)
        if stop_loss_threshold is not None and pnl_pct <= stop_loss_threshold:
            return True, "STOP_LOSS"
        
        return False, "NONE"
    
    def _should_pause_trading(self, symbol, strategy_input):
        """Check if trading should be paused due to consecutive losses"""
        max_losses = getattr(strategy_input, 'max_consecutive_losses', None)
        if max_losses is None:
            return False
        
        consecutive_losses = self.consecutive_losses.get(symbol, 0)
        return consecutive_losses >= max_losses
    
    def _update_consecutive_losses(self, symbol, trade_result):
        """Update consecutive loss counter based on trade result"""
        if trade_result == "LOSS":
            self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
        elif trade_result == "PROFIT":
            self.consecutive_losses[symbol] = 0  # Reset on profit
        # No change for neutral trades
    
    def _get_current_price(self, symbol, date, data_source=None):
        """Get current price for stop loss/take profit calculations"""
        # This is a placeholder - in real implementation, you would fetch actual price data
        # For now, we'll use prediction as a proxy for price movement
        if symbol in self.price_data and date in self.price_data[symbol]:
            return self.price_data[symbol][date]
        return None
    
    def _record_trade_result(self, symbol, entry_price, exit_price, exit_reason):
        """Record the result of a completed trade"""
        if entry_price is None or exit_price is None:
            return "NEUTRAL"
        
        pnl_pct = (exit_price - entry_price) / entry_price
        
        if pnl_pct > 0.001:  # Profit threshold (0.1%)
            result = "PROFIT"
        elif pnl_pct < -0.001:  # Loss threshold (-0.1%)
            result = "LOSS"
        else:
            result = "NEUTRAL"
        
        # Update consecutive losses
        self._update_consecutive_losses(symbol, result)
        
        return result
    
    def get_position_summary(self):
        """Get summary of current positions and trading history"""
        return {
            'current_positions': self.current_positions.copy(),
            'position_history': self.position_history.copy(),
            'consecutive_losses': self.consecutive_losses.copy(),
            'total_trades': len(self.position_history),
            'symbols_traded': len(set([trade['symbol'] for trade in self.position_history])),
            'winning_trades': len([t for t in self.position_history if t.get('action') in ['TAKE_PROFIT']]),
            'losing_trades': len([t for t in self.position_history if t.get('action') in ['STOP_LOSS']])
        }
    
    def construct_portfolio(self, signals: pd.DataFrame, strategy_input: StrategyInput) -> pd.DataFrame:
        """Construct portfolio weights from continuous signals"""
        
        if len(signals) == 0:
            return pd.DataFrame()
        
        portfolio_data = []
        
        for date in signals['date'].unique():
            date_signals = signals[signals['date'] == date]
            
            if strategy_input.use_continuous_positions:
                # Direct use of continuous weights from signals
                for _, row in date_signals.iterrows():
                    portfolio_data.append({
                        'date': date,
                        'symbol': row['symbol'],
                        'weight': row['signal']  # Signal now contains the continuous weight
                    })
            
            elif strategy_input.position_method == "equal_weight":
                # Traditional equal weight allocation for binary signals
                total_positions = len(date_signals)
                weight_per_position = 1.0 / total_positions if total_positions > 0 else 0
                
                for _, row in date_signals.iterrows():
                    # For binary signals: 1 = long, -1 = short, 0 = no position
                    signal_weight = 1 if row['signal'] > 0 else (-1 if row['signal'] < 0 else 0)
                    portfolio_data.append({
                        'date': date,
                        'symbol': row['symbol'],
                        'weight': weight_per_position * signal_weight
                    })
            
            elif strategy_input.position_method == "factor_weight":
                # Weight by prediction strength (for binary signals)
                total_abs_prediction = date_signals['prediction'].abs().sum()
                
                if total_abs_prediction > 0:
                    for _, row in date_signals.iterrows():
                        signal_direction = 1 if row['signal'] > 0 else (-1 if row['signal'] < 0 else 0)
                        weight = (abs(row['prediction']) / total_abs_prediction) * signal_direction
                        portfolio_data.append({
                            'date': date,
                            'symbol': row['symbol'],
                            'weight': weight
                        })
        
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            portfolio_weights = portfolio_df.pivot(index='date', columns='symbol', values='weight').fillna(0)
            
            # Normalize weights if using continuous positions to ensure they don't exceed limits
            if strategy_input.use_continuous_positions:
                # Apply maximum position limits
                portfolio_weights = portfolio_weights.clip(
                    lower=-strategy_input.max_position_weight if strategy_input.strategy_type != "long_only" else 0,
                    upper=strategy_input.max_position_weight
                )
                
                # Normalize to ensure total leverage doesn't exceed limits
                for date in portfolio_weights.index:
                    date_weights = portfolio_weights.loc[date]
                    total_leverage = date_weights.abs().sum()
                    
                    if total_leverage > strategy_input.leverage:
                        # Scale down proportionally
                        scaling_factor = strategy_input.leverage / total_leverage
                        portfolio_weights.loc[date] = date_weights * scaling_factor
            
            return portfolio_weights
        else:
            return pd.DataFrame()
