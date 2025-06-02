import numpy as np
import pandas as pd
from typing import Dict, Any, List
from ...registry import AlphaAgent
from ...schema.agent_config import AlphaAgentConfig, AlphaAgentType

class TechnicalAlphaAgent(AlphaAgent):
    """Technical analysis based alpha generation agent"""
    
    def __init__(self, config: AlphaAgentConfig):
        super().__init__(config)
        self._validate_technical_config()
        
    def _validate_technical_config(self):
        """Validate technical analysis specific configuration"""
        required_params = ["lookback_period", "indicators"]
        for param in required_params:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
                
    async def generate_alpha(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alpha signals using technical indicators"""
        try:
            # Convert input data to DataFrame
            df = pd.DataFrame(data["ohlcv"])
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            
            # Calculate technical indicators
            signals = {}
            for indicator in self.config.parameters["indicators"]:
                if indicator == "rsi":
                    signals["rsi"] = self._calculate_rsi(df)
                elif indicator == "macd":
                    signals["macd"] = self._calculate_macd(df)
                elif indicator == "bollinger":
                    signals["bollinger"] = self._calculate_bollinger_bands(df)
                    
            # Generate trading signals
            alpha_signals = self._generate_trading_signals(signals, df)
            
            return {
                "signals": alpha_signals,
                "indicators": signals,
                "metadata": {
                    "timestamp": df["timestamp"].iloc[-1],
                    "indicators_used": self.config.parameters["indicators"]
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Error generating technical alpha: {str(e)}")
            
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate generated signals"""
        try:
            if "signals" not in signal:
                return False
                
            # Validate signal structure
            required_fields = ["position", "strength", "confidence"]
            for field in required_fields:
                if field not in signal["signals"]:
                    return False
                    
            # Validate signal values
            if not (-1 <= signal["signals"]["position"] <= 1):
                return False
            if not (0 <= signal["signals"]["strength"] <= 1):
                return False
            if not (0 <= signal["signals"]["confidence"] <= 1):
                return False
                
            return True
            
        except Exception:
            return False
            
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {
            "macd": macd,
            "signal": signal,
            "histogram": macd - signal
        }
        
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return {
            "middle": sma,
            "upper": upper_band,
            "lower": lower_band
        }
        
    def _generate_trading_signals(self, indicators: Dict[str, Any], df: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals from technical indicators"""
        # Combine signals from different indicators
        position = 0.0
        strength = 0.0
        confidence = 0.0
        
        # RSI signals
        if "rsi" in indicators:
            rsi = indicators["rsi"].iloc[-1]
            if rsi < 30:  # Oversold
                position += 1
                strength += (30 - rsi) / 30
                confidence += 0.3
            elif rsi > 70:  # Overbought
                position -= 1
                strength += (rsi - 70) / 30
                confidence += 0.3
                
        # MACD signals
        if "macd" in indicators:
            macd = indicators["macd"]
            if macd["histogram"].iloc[-1] > 0 and macd["histogram"].iloc[-2] <= 0:
                position += 0.5
                strength += abs(macd["histogram"].iloc[-1])
                confidence += 0.2
            elif macd["histogram"].iloc[-1] < 0 and macd["histogram"].iloc[-2] >= 0:
                position -= 0.5
                strength += abs(macd["histogram"].iloc[-1])
                confidence += 0.2
                
        # Bollinger Bands signals
        if "bollinger" in indicators:
            bb = indicators["bollinger"]
            close = df["close"].iloc[-1]
            if close < bb["lower"].iloc[-1]:
                position += 0.5
                strength += (bb["lower"].iloc[-1] - close) / bb["lower"].iloc[-1]
                confidence += 0.2
            elif close > bb["upper"].iloc[-1]:
                position -= 0.5
                strength += (close - bb["upper"].iloc[-1]) / bb["upper"].iloc[-1]
                confidence += 0.2
                
        # Normalize signals
        position = max(min(position, 1), -1)
        strength = min(strength, 1)
        confidence = min(confidence, 1)
        
        return {
            "position": position,
            "strength": strength,
            "confidence": confidence
        } 