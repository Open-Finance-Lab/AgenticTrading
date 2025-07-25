"""
Data Mining Agent for Empirical Alpha Strategy Discovery

This agent uses machine learning and statistical methods to discover alpha patterns
in financial data through data mining techniques. It analyzes historical patterns,
correlations, and anomalies to generate empirical trading signals.

Academic Framework:
Based on empirical finance methodologies and data mining approaches in quantitative
finance, including feature engineering, pattern recognition, and statistical learning.

Author: FinAgent Research Team  
Created: 2025-07-25
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class DataMiningSignal:
    """Data structure for data mining-derived trading signals."""
    symbol: str
    signal_strength: float
    pattern_type: str
    confidence: float
    features_used: List[str]
    timestamp: datetime
    expected_return: Optional[float] = None
    risk_score: Optional[float] = None
    pattern_metadata: Optional[Dict[str, Any]] = None

@dataclass  
class MarketPattern:
    """Data structure for discovered market patterns."""
    pattern_id: str
    pattern_type: str
    features: Dict[str, float]
    frequency: float
    success_rate: float
    avg_return: float
    risk_metrics: Dict[str, float]

class DataMiningAgent:
    """
    Data Mining Agent for discovering empirical alpha patterns in financial data.
    
    This agent employs various machine learning techniques including:
    - Feature engineering from price and volume data
    - Pattern clustering and classification
    - Statistical anomaly detection
    - Predictive modeling for return forecasting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Data Mining Agent with configuration parameters.
        
        Args:
            config: Configuration dictionary containing:
                - lookback_window: Historical data window (default: 50)
                - n_clusters: Number of pattern clusters (default: 5)
                - min_confidence: Minimum confidence threshold (default: 0.6)
                - feature_selection: List of features to use (default: comprehensive)
        """
        self.config = config or {}
        self.lookback_window = self.config.get('lookback_window', 50)
        self.n_clusters = self.config.get('n_clusters', 5)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.pattern_clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.return_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
        
        # Pattern storage
        self.discovered_patterns: List[MarketPattern] = []
        self.feature_importance: Dict[str, float] = {}
        
        logger.info(f"DataMiningAgent initialized with lookback_window={self.lookback_window}, "
                   f"n_clusters={self.n_clusters}")
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive features from price and volume data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        try:
            # Price-based features
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['volatility'] = features['returns'].rolling(20).std()
            features['price_momentum'] = data['close'] / data['close'].shift(10) - 1
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(data['close'])
            features['macd'] = self._calculate_macd(data['close'])
            features['bollinger_position'] = self._calculate_bollinger_position(data['close'])
            
            # Volume features
            if 'volume' in data.columns:
                features['volume_sma'] = data['volume'].rolling(20).mean()
                features['volume_ratio'] = data['volume'] / features['volume_sma']
                features['price_volume_trend'] = (data['close'].pct_change() * 
                                                data['volume']).rolling(10).sum()
            
            # Statistical features
            features['skewness'] = features['returns'].rolling(20).skew()
            features['kurtosis'] = features['returns'].rolling(20).kurt()
            features['sharpe_ratio'] = (features['returns'].rolling(20).mean() / 
                                      features['returns'].rolling(20).std())
            
            # Regime features
            features['trend_strength'] = self._calculate_trend_strength(data['close'])
            features['market_regime'] = self._identify_market_regime(features['returns'])
            
            # Cross-sectional features (if multiple securities)
            features['relative_strength'] = data['close'] / data['close'].rolling(252).mean()
            
            # Drop NaN values
            features = features.dropna()
            
            logger.debug(f"Engineered {len(features.columns)} features from market data")
            return features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        return (prices - lower_band) / (upper_band - lower_band)
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        def trend_slope(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        
        return prices.rolling(window).apply(trend_slope, raw=False)
    
    def _identify_market_regime(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Identify market regime (0: low vol, 1: high vol, 2: trending)."""
        volatility = returns.rolling(window).std()
        vol_threshold = volatility.quantile(0.7)
        
        trend = returns.rolling(window).mean().abs()
        trend_threshold = trend.quantile(0.7)
        
        regime = pd.Series(0, index=returns.index)  # Default: low vol
        regime[volatility > vol_threshold] = 1      # High vol
        regime[trend > trend_threshold] = 2         # Trending
        
        return regime
    
    def discover_patterns(self, features: pd.DataFrame) -> List[MarketPattern]:
        """
        Discover market patterns using clustering and statistical analysis.
        
        Args:
            features: Engineered features DataFrame
            
        Returns:
            List of discovered MarketPattern objects
        """
        if len(features) < self.lookback_window:
            logger.warning(f"Insufficient data for pattern discovery: {len(features)} rows")
            return []
        
        try:
            # Prepare data for clustering
            feature_matrix = features.fillna(0).values
            scaled_features = self.scaler.fit_transform(feature_matrix)
            
            # Apply PCA for dimensionality reduction
            pca_features = self.pca.fit_transform(scaled_features)
            
            # Perform clustering
            cluster_labels = self.pattern_clusterer.fit_predict(pca_features)
            
            patterns = []
            for cluster_id in range(self.n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if cluster_mask.sum() < 5:  # Skip small clusters
                    continue
                
                cluster_data = features[cluster_mask]
                cluster_returns = cluster_data['returns'] if 'returns' in cluster_data.columns else pd.Series()
                
                if len(cluster_returns) == 0:
                    continue
                
                # Calculate pattern metrics
                pattern = MarketPattern(
                    pattern_id=f"cluster_{cluster_id}",
                    pattern_type=f"empirical_pattern_{cluster_id}",
                    features=cluster_data.mean().to_dict(),
                    frequency=len(cluster_data) / len(features),
                    success_rate=len(cluster_returns[cluster_returns > 0]) / len(cluster_returns),
                    avg_return=cluster_returns.mean(),
                    risk_metrics={
                        'volatility': cluster_returns.std(),
                        'max_drawdown': (cluster_returns.cumsum() - cluster_returns.cumsum().cummax()).min(),
                        'sharpe_ratio': cluster_returns.mean() / cluster_returns.std() if cluster_returns.std() > 0 else 0
                    }
                )
                patterns.append(pattern)
            
            self.discovered_patterns = patterns
            logger.info(f"Discovered {len(patterns)} market patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern discovery: {e}")
            return []
    
    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[DataMiningSignal]:
        """
        Generate trading signals based on discovered patterns and ML predictions.
        
        Args:
            data: Market data DataFrame
            symbol: Stock symbol
            
        Returns:
            List of DataMiningSignal objects
        """
        try:
            # Engineer features
            features = self.engineer_features(data)
            if len(features) < self.lookback_window:
                return []
            
            # Discover or update patterns
            if not self.discovered_patterns:
                self.discover_patterns(features)
            
            signals = []
            
            # Get latest feature values
            latest_features = features.iloc[-1].fillna(0)
            
            # Pattern-based signals
            for pattern in self.discovered_patterns:
                if pattern.success_rate > 0.5 and pattern.avg_return > 0:
                    # Calculate similarity to pattern
                    pattern_features = pd.Series(pattern.features)
                    common_features = set(latest_features.index) & set(pattern_features.index)
                    
                    if len(common_features) > 0:
                        similarity = self._calculate_pattern_similarity(
                            latest_features[list(common_features)], 
                            pattern_features[list(common_features)]
                        )
                        
                        if similarity > self.min_confidence:
                            signal_strength = similarity * pattern.avg_return * 10  # Scale signal
                            
                            signal = DataMiningSignal(
                                symbol=symbol,
                                signal_strength=signal_strength,
                                pattern_type=pattern.pattern_type,
                                confidence=similarity,
                                features_used=list(common_features),
                                timestamp=datetime.now(),
                                expected_return=pattern.avg_return,
                                risk_score=pattern.risk_metrics.get('volatility', 0),
                                pattern_metadata={
                                    'pattern_id': pattern.pattern_id,
                                    'success_rate': pattern.success_rate,
                                    'frequency': pattern.frequency
                                }
                            )
                            signals.append(signal)
            
            # ML-based return prediction
            if len(features) > 50:  # Enough data for ML
                try:
                    X = features.iloc[:-1].fillna(0).values
                    y = features['returns'].shift(-1).iloc[:-1].fillna(0).values
                    
                    # Train predictor
                    self.return_predictor.fit(X, y)
                    
                    # Predict next return
                    predicted_return = self.return_predictor.predict([latest_features.fillna(0).values])[0]
                    prediction_confidence = min(abs(predicted_return) * 20, 1.0)  # Scaled confidence
                    
                    if prediction_confidence > self.min_confidence:
                        ml_signal = DataMiningSignal(
                            symbol=symbol,
                            signal_strength=predicted_return * 10,  # Scale signal
                            pattern_type="ml_prediction",
                            confidence=prediction_confidence,
                            features_used=list(features.columns),
                            timestamp=datetime.now(),
                            expected_return=predicted_return,
                            risk_score=features['volatility'].iloc[-1] if 'volatility' in features.columns else 0,
                            pattern_metadata={'model_type': 'random_forest'}
                        )
                        signals.append(ml_signal)
                        
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")
            
            # Sort by confidence
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Generated {len(signals)} data mining signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def _calculate_pattern_similarity(self, features1: pd.Series, features2: pd.Series) -> float:
        """Calculate similarity between feature vectors."""
        try:
            # Normalize features
            norm1 = features1 / (features1.abs().sum() + 1e-8)
            norm2 = features2 / (features2.abs().sum() + 1e-8)
            
            # Calculate cosine similarity
            dot_product = (norm1 * norm2).sum()
            return max(0, dot_product)  # Ensure non-negative
            
        except Exception:
            return 0.0
    
    def get_strategy_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data mining strategy."""
        return {
            'strategy_type': 'data_mining',
            'lookback_window': self.lookback_window,
            'n_clusters': self.n_clusters,
            'min_confidence': self.min_confidence,
            'patterns_discovered': len(self.discovered_patterns),
            'description': 'Empirical pattern discovery using machine learning and statistical analysis'
        }

# Factory function for compatibility
def create_data_mining_agent(config: Optional[Dict[str, Any]] = None) -> DataMiningAgent:
    """Create and return a DataMiningAgent instance."""
    return DataMiningAgent(config)

# For backward compatibility and testing
if __name__ == "__main__":
    # Simple test
    agent = DataMiningAgent()
    
    # Generate test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    test_data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 0,
        'low': 0,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    test_data['high'] = test_data[['open', 'close']].max(axis=1) + np.random.rand(100) * 2
    test_data['low'] = test_data[['open', 'close']].min(axis=1) - np.random.rand(100) * 2
    
    # Test signal generation
    signals = agent.generate_signals(test_data, 'TEST')
    print(f"Generated {len(signals)} test signals")
    for signal in signals[:3]:  # Show first 3
        print(f"Signal: {signal.pattern_type}, Strength: {signal.signal_strength:.3f}, Confidence: {signal.confidence:.3f}")
