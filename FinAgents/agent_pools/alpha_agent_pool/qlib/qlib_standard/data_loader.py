"""
Qlib Standard Data Loader Implementation

This module implements the Qlib standard DataLoader interface for loading
and processing financial time series data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from qlib.data.dataset.loader import DataLoader
from qlib.utils import get_or_create_path
import warnings


class QlibCSVDataLoader(DataLoader):
    """
    Standard Qlib DataLoader implementation for CSV data sources.
    
    This loader follows the Qlib DataLoader interface and loads data from
    CSV files with proper multi-index formatting (datetime, instrument).
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        freq: str = "1H",
        **kwargs
    ):
        """
        Initialize the Qlib standard CSV data loader.
        
        Args:
            data_path: Path to the CSV data file
            feature_columns: List of feature column names to load
            target_columns: List of target column names to load  
            freq: Data frequency (e.g., '1H' for hourly, 'D' for daily)
            **kwargs: Additional arguments
        """
        self.data_path = Path(data_path)
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.freq = freq
        self._data_cache = None
        
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
    
    def load(
        self,
        instruments: Optional[Union[str, Dict, List]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data following Qlib DataLoader interface.
        
        Args:
            instruments: Instrument universe specification (None for all instruments)
            start_time: Start time for data loading (e.g., '2020-01-01')
            end_time: End time for data loading (e.g., '2023-12-31')
            
        Returns:
            pd.DataFrame: Multi-index DataFrame with (datetime, instrument) index
                         and feature/label columns with proper Qlib formatting
        """
        # Load and cache data on first call
        if self._data_cache is None:
            self._load_and_prepare_data()
        
        # Start with cached data
        data = self._data_cache.copy()
        
        # Apply time filtering
        if start_time is not None:
            start_time = pd.Timestamp(start_time)
            data = data[data.index.get_level_values('datetime') >= start_time]
            
        if end_time is not None:
            end_time = pd.Timestamp(end_time)
            data = data[data.index.get_level_values('datetime') <= end_time]
        
        # Apply instrument filtering
        if instruments is not None:
            if isinstance(instruments, str):
                # Single instrument
                data = data[data.index.get_level_values('instrument') == instruments]
            elif isinstance(instruments, list):
                # List of instruments
                data = data[data.index.get_level_values('instrument').isin(instruments)]
            elif isinstance(instruments, dict):
                # Instrument universe config (simplified implementation)
                if 'instruments' in instruments:
                    inst_list = instruments['instruments']
                    if isinstance(inst_list, list):
                        data = data[data.index.get_level_values('instrument').isin(inst_list)]
        
        if data.empty:
            warnings.warn("No data available for the specified filters")
            return self._create_empty_dataframe()
        
        return data
    
    def _load_and_prepare_data(self) -> None:
        """
        Load CSV data and prepare it in Qlib standard format.
        
        Converts data to multi-index format with (datetime, instrument) index
        and organizes columns into feature and label groups.
        """
        try:
            # Load CSV data
            raw_data = pd.read_csv(self.data_path)
            
            # Ensure required columns exist
            required_cols = ['datetime', 'symbol']
            missing_cols = [col for col in required_cols if col not in raw_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert datetime column
            raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
            
            # Set multi-index
            raw_data = raw_data.set_index(['datetime', 'symbol'])
            raw_data.index.names = ['datetime', 'instrument']
            
            # Sort index for better performance
            raw_data = raw_data.sort_index()
            
            # Organize columns into feature and label groups
            self._data_cache = self._organize_columns(raw_data)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.data_path}: {str(e)}")
    
    def _organize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Organize columns into Qlib standard format with feature and label groups.
        
        Args:
            data: Raw DataFrame with multi-index
            
        Returns:
            pd.DataFrame: Formatted DataFrame with proper column hierarchy
        """
        # Determine feature and label columns
        all_columns = data.columns.tolist()
        
        # Use provided feature columns or infer from data
        if self.feature_columns:
            feature_cols = [col for col in self.feature_columns if col in all_columns]
        else:
            # Exclude common label column patterns
            exclude_patterns = ['label', 'target', 'return', 'y']
            feature_cols = [
                col for col in all_columns 
                if not any(pattern in col.lower() for pattern in exclude_patterns)
            ]
        
        # Use provided target columns or infer from data
        if self.target_columns:
            label_cols = [col for col in self.target_columns if col in all_columns]
        else:
            # Include common label column patterns
            include_patterns = ['label', 'target', 'return', 'y']
            label_cols = [
                col for col in all_columns 
                if any(pattern in col.lower() for pattern in include_patterns)
            ]
        
        # Create multi-level column index
        feature_columns = [(col, 'feature') for col in feature_cols]
        label_columns = [(col, 'label') for col in label_cols]
        
        # Combine all columns
        all_column_tuples = feature_columns + label_columns
        
        if not all_column_tuples:
            # If no columns specified, treat all as features
            all_column_tuples = [(col, 'feature') for col in all_columns]
        
        # Create new DataFrame with multi-level columns
        column_data = {}
        for col, group in all_column_tuples:
            if col in data.columns:
                column_data[(group, col)] = data[col]
        
        result = pd.DataFrame(column_data, index=data.index)
        result.columns = pd.MultiIndex.from_tuples(
            result.columns, names=['field_group', 'field_name']
        )
        
        return result
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """
        Create an empty DataFrame with proper structure for cases with no data.
        
        Returns:
            pd.DataFrame: Empty DataFrame with correct index and column structure
        """
        # Create empty multi-index
        empty_index = pd.MultiIndex.from_tuples(
            [], names=['datetime', 'instrument']
        )
        
        # Create empty multi-level columns
        empty_columns = pd.MultiIndex.from_tuples(
            [], names=['field_group', 'field_name']
        )
        
        return pd.DataFrame(index=empty_index, columns=empty_columns)


class QlibSyntheticDataLoader(DataLoader):
    """
    Synthetic data loader for testing and development purposes.
    
    Generates synthetic financial data that follows Qlib format conventions.
    """
    
    def __init__(
        self,
        instruments: List[str],
        start_time: str = "2020-01-01",
        end_time: str = "2023-12-31", 
        freq: str = "1H",
        feature_dims: int = 5,
        add_labels: bool = True,
        **kwargs
    ):
        """
        Initialize synthetic data loader.
        
        Args:
            instruments: List of instrument symbols to generate data for
            start_time: Start time for synthetic data generation
            end_time: End time for synthetic data generation
            freq: Data frequency
            feature_dims: Number of synthetic features to generate
            add_labels: Whether to add synthetic labels
            **kwargs: Additional arguments
        """
        self.instruments = instruments
        self.start_time = start_time
        self.end_time = end_time
        self.freq = freq
        self.feature_dims = feature_dims
        self.add_labels = add_labels
        self._data_cache = None
    
    def load(
        self,
        instruments: Optional[Union[str, Dict, List]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load synthetic data following Qlib DataLoader interface.
        
        Args:
            instruments: Instrument filter (optional, uses constructor instruments if None)
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            
        Returns:
            pd.DataFrame: Synthetic data in Qlib format
        """
        # Generate data if not cached
        if self._data_cache is None:
            self._generate_synthetic_data()
        
        # Apply filters similar to CSV loader
        data = self._data_cache.copy()
        
        # Time filtering
        if start_time:
            data = data[data.index.get_level_values('datetime') >= pd.Timestamp(start_time)]
        if end_time:
            data = data[data.index.get_level_values('datetime') <= pd.Timestamp(end_time)]
            
        # Instrument filtering  
        if instruments:
            if isinstance(instruments, str):
                data = data[data.index.get_level_values('instrument') == instruments]
            elif isinstance(instruments, list):
                data = data[data.index.get_level_values('instrument').isin(instruments)]
        
        return data
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic financial data in Qlib format."""
        # Create date range
        date_range = pd.date_range(
            start=self.start_time, 
            end=self.end_time, 
            freq=self.freq
        )
        
        # Create multi-index for all combinations
        index_tuples = [
            (dt, inst) for dt in date_range for inst in self.instruments
        ]
        multi_index = pd.MultiIndex.from_tuples(
            index_tuples, names=['datetime', 'instrument']
        )
        
        # Generate random features
        np.random.seed(42)  # For reproducibility
        n_samples = len(multi_index)
        
        data_dict = {}
        
        # Generate features
        for i in range(self.feature_dims):
            feature_name = f'feature_{i+1}'
            # Generate auto-correlated time series
            feature_data = np.random.randn(n_samples) * 0.1
            data_dict[('feature', feature_name)] = feature_data
        
        # Generate synthetic labels if requested
        if self.add_labels:
            # Create momentum-based synthetic label
            returns = np.random.randn(n_samples) * 0.02  # 2% volatility
            data_dict[('label', 'label_1h')] = returns
            
            # Create market-neutral targets
            neutral_returns = returns - np.mean(returns)
            data_dict[('label', 'label_market_neutral')] = neutral_returns
        
        # Create DataFrame
        self._data_cache = pd.DataFrame(data_dict, index=multi_index)
        self._data_cache.columns = pd.MultiIndex.from_tuples(
            self._data_cache.columns, names=['field_group', 'field_name']
        )
