from __future__ import annotations

from typing import Protocol, Dict, Any, List, Optional
from datetime import datetime


class FeatureSpec:
    """Feature specification for data retrieval."""
    
    def __init__(self,
                 feature_name: str,
                 symbols: List[str],
                 start_date: datetime,
                 end_date: datetime,
                 frequency: str = "1D",
                 transformations: Optional[List[str]] = None):
        self.feature_name = feature_name
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.transformations = transformations or []


class FeatureTable:
    """Container for feature data with metadata."""
    
    def __init__(self,
                 feature_name: str,
                 data: Dict[str, Any],
                 metadata: Optional[Dict[str, Any]] = None):
        self.feature_name = feature_name
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.now()


class FeaturePort(Protocol):
    """Port for feature/data retrieval or computation.
    
    This port defines the interface for fetching market data and computing
    derived features for alpha factor generation strategies.
    """

    async def fetch(self, spec: FeatureSpec) -> FeatureTable:
        """Fetch feature data according to specification.
        
        Args:
            spec: Feature specification containing symbols, dates, and transformations
            
        Returns:
            Feature table containing requested data with metadata
            
        Raises:
            ValueError: If specification is invalid
            RuntimeError: If data source is unavailable
            TimeoutError: If fetch operation times out
        """
        ...

    async def compute(self, node_ctx: Dict[str, Any]) -> FeatureTable:
        """Compute derived features from existing data.
        
        Args:
            node_ctx: Computation context containing:
                - feature_name: Name of feature to compute
                - inputs: Input data for computation
                - parameters: Computation parameters
                
        Returns:
            Computed feature table with derivation metadata
            
        Raises:
            ValueError: If computation context is invalid
            RuntimeError: If computation fails
        """
        ...

    async def list_available_features(self) -> List[str]:
        """List all available feature names.
        
        Returns:
            List of feature names that can be fetched or computed
        """
        ...

    async def get_feature_schema(self, feature_name: str) -> Dict[str, Any]:
        """Get schema information for a specific feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Schema dictionary containing field types and descriptions
        """
        ...

    async def validate_spec(self, spec: FeatureSpec) -> bool:
        """Validate feature specification.
        
        Args:
            spec: Feature specification to validate
            
        Returns:
            True if specification is valid and can be fulfilled
        """
        ...

