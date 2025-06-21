import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from ...registry import AlphaAgent
from ...schema.agent_config import AlphaAgentConfig, AlphaAgentType

class MLAlphaAgent(AlphaAgent):
    """Machine learning based alpha generation agent"""
    
    def __init__(self, config: AlphaAgentConfig):
        super().__init__(config)
        self._validate_ml_config()
        self._model = None
        self._scaler = None
        self._feature_columns = None
        self._load_model()
        
    def _validate_ml_config(self):
        """Validate ML specific configuration"""
        required_params = ["model_path", "feature_columns", "prediction_threshold"]
        for param in required_params:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
                
    def _load_model(self):
        """Load the trained ML model and scaler"""
        try:
            model_path = self.config.parameters["model_path"]
            if not os.path.exists(model_path):
                raise ValueError(f"Model path does not exist: {model_path}")
                
            # Load model and scaler
            self._model = joblib.load(os.path.join(model_path, "model.joblib"))
            self._scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
            self._feature_columns = self.config.parameters["feature_columns"]
            
        except Exception as e:
            raise RuntimeError(f"Error loading ML model: {str(e)}")
            
    async def generate_alpha(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alpha signals using ML model"""
        try:
            # Prepare features
            features = self._prepare_features(data)
            if features is None:
                return self._generate_neutral_signal()
                
            # Scale features
            scaled_features = self._scaler.transform(features)
            
            # Generate predictions
            predictions = self._model.predict_proba(scaled_features)
            feature_importance = self._get_feature_importance()
            
            # Generate trading signals
            alpha_signals = self._generate_trading_signals(predictions, feature_importance)
            
            return {
                "signals": alpha_signals,
                "predictions": predictions.tolist(),
                "feature_importance": feature_importance,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_type": type(self._model).__name__,
                    "features_used": self._feature_columns
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Error generating ML alpha: {str(e)}")
            
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate generated signals"""
        try:
            if "signals" not in signal:
                return False
                
            # Validate signal structure
            required_fields = ["position", "strength", "confidence", "prediction_probability"]
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
            if not (0 <= signal["signals"]["prediction_probability"] <= 1):
                return False
                
            return True
            
        except Exception:
            return False
            
    def _prepare_features(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare features for ML model"""
        try:
            # Extract features from input data
            features = {}
            for feature in self._feature_columns:
                if feature in data:
                    features[feature] = data[feature]
                else:
                    # Handle missing features
                    return None
                    
            # Convert to DataFrame
            df = pd.DataFrame([features])
            return df
            
        except Exception:
            return None
            
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model"""
        if not hasattr(self._model, "feature_importances_"):
            return {}
            
        importance = self._model.feature_importances_
        return dict(zip(self._feature_columns, importance))
        
    def _generate_trading_signals(
        self,
        predictions: np.ndarray,
        feature_importance: Dict[str, float]
    ) -> Dict[str, float]:
        """Generate trading signals from ML predictions"""
        # Get prediction probabilities
        prob_up = predictions[0][1]  # Probability of upward movement
        prob_down = predictions[0][0]  # Probability of downward movement
        
        # Calculate position based on prediction probabilities
        threshold = self.config.parameters["prediction_threshold"]
        if prob_up > threshold:
            position = (prob_up - threshold) / (1 - threshold)
        elif prob_down > threshold:
            position = -(prob_down - threshold) / (1 - threshold)
        else:
            position = 0.0
            
        # Calculate confidence based on feature importance
        confidence = np.mean(list(feature_importance.values())) if feature_importance else 0.5
        
        # Calculate signal strength
        strength = abs(position)
        
        return {
            "position": position,
            "strength": strength,
            "confidence": confidence,
            "prediction_probability": max(prob_up, prob_down)
        }
        
    def _generate_neutral_signal(self) -> Dict[str, float]:
        """Generate neutral signal when features are missing"""
        return {
            "position": 0.0,
            "strength": 0.0,
            "confidence": 0.0,
            "prediction_probability": 0.5
        } 