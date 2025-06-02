import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta
from ...registry import AlphaAgent
from ...schema.agent_config import AlphaAgentConfig, AlphaAgentType

class EventDrivenAlphaAgent(AlphaAgent):
    """Event-driven alpha generation agent"""
    
    def __init__(self, config: AlphaAgentConfig):
        super().__init__(config)
        self._validate_event_config()
        self._event_patterns = self._initialize_event_patterns()
        
    def _validate_event_config(self):
        """Validate event-driven specific configuration"""
        required_params = ["event_types", "impact_threshold", "decay_factor"]
        for param in required_params:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
                
    def _initialize_event_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize event impact patterns"""
        return {
            "earnings": {
                "impact_window": timedelta(days=2),
                "decay_function": lambda x: np.exp(-x * self.config.parameters["decay_factor"]),
                "threshold": self.config.parameters["impact_threshold"]
            },
            "news": {
                "impact_window": timedelta(hours=4),
                "decay_function": lambda x: np.exp(-x * self.config.parameters["decay_factor"] * 2),
                "threshold": self.config.parameters["impact_threshold"] * 0.8
            },
            "analyst_rating": {
                "impact_window": timedelta(days=1),
                "decay_function": lambda x: np.exp(-x * self.config.parameters["decay_factor"] * 1.5),
                "threshold": self.config.parameters["impact_threshold"] * 0.6
            },
            "insider_trading": {
                "impact_window": timedelta(days=3),
                "decay_function": lambda x: np.exp(-x * self.config.parameters["decay_factor"] * 0.8),
                "threshold": self.config.parameters["impact_threshold"] * 0.7
            }
        }
        
    async def generate_alpha(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alpha signals based on events"""
        try:
            # Process events
            events = data.get("events", [])
            if not events:
                return self._generate_neutral_signal()
                
            # Calculate event impacts
            event_impacts = self._calculate_event_impacts(events)
            
            # Generate trading signals
            alpha_signals = self._generate_trading_signals(event_impacts)
            
            return {
                "signals": alpha_signals,
                "event_impacts": event_impacts,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "events_processed": len(events),
                    "event_types": list(set(e["type"] for e in events))
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Error generating event-driven alpha: {str(e)}")
            
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate generated signals"""
        try:
            if "signals" not in signal:
                return False
                
            # Validate signal structure
            required_fields = ["position", "strength", "confidence", "event_impact"]
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
            if not (0 <= signal["signals"]["event_impact"] <= 1):
                return False
                
            return True
            
        except Exception:
            return False
            
    def _calculate_event_impacts(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate impact of each event type"""
        impacts = {}
        current_time = datetime.now()
        
        for event in events:
            event_type = event["type"]
            if event_type not in self._event_patterns:
                continue
                
            # Get event pattern
            pattern = self._event_patterns[event_type]
            
            # Calculate time decay
            event_time = datetime.fromisoformat(event["timestamp"])
            time_diff = (current_time - event_time).total_seconds() / 3600  # hours
            if time_diff > pattern["impact_window"].total_seconds() / 3600:
                continue
                
            # Calculate impact with decay
            base_impact = event.get("impact", 0.5)
            decayed_impact = base_impact * pattern["decay_function"](time_diff)
            
            # Apply threshold
            if decayed_impact < pattern["threshold"]:
                continue
                
            # Aggregate impacts
            if event_type not in impacts:
                impacts[event_type] = 0
            impacts[event_type] += decayed_impact
            
        # Normalize impacts
        for event_type in impacts:
            impacts[event_type] = min(impacts[event_type], 1.0)
            
        return impacts
        
    def _generate_trading_signals(self, event_impacts: Dict[str, float]) -> Dict[str, float]:
        """Generate trading signals from event impacts"""
        if not event_impacts:
            return self._generate_neutral_signal()
            
        # Calculate weighted position
        position = 0.0
        total_impact = 0.0
        confidence = 0.0
        
        # Event type weights
        weights = {
            "earnings": 1.0,
            "news": 0.8,
            "analyst_rating": 0.6,
            "insider_trading": 0.7
        }
        
        for event_type, impact in event_impacts.items():
            weight = weights.get(event_type, 0.5)
            position += impact * weight * (1 if impact > 0 else -1)
            total_impact += impact * weight
            confidence += impact * weight
            
        # Normalize signals
        if total_impact > 0:
            position = position / total_impact
            confidence = confidence / len(event_impacts)
            
        return {
            "position": max(min(position, 1), -1),
            "strength": min(total_impact, 1),
            "confidence": min(confidence, 1),
            "event_impact": min(total_impact, 1)
        }
        
    def _generate_neutral_signal(self) -> Dict[str, float]:
        """Generate neutral signal when no events are present"""
        return {
            "position": 0.0,
            "strength": 0.0,
            "confidence": 0.0,
            "event_impact": 0.0
        } 