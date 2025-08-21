"""
Qlib Standard Data Handler Implementation

This module implements data handling using Qlib's DataHandler framework
for comprehensive data preprocessing and feature engineering pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings

# Qlib data handler imports
from qlib.data.dataset.handler import DataHandler, DataHandlerLP
from qlib.data.dataset.processor import (
    Processor, DropnaLabel, DropCol, FilterCol, 
    MinMaxNorm, ZScoreNorm, RobustZScoreNorm,
    CSZScoreNorm, CSRankNorm, Fillna, ProcessInf
)
from qlib.data.dataset.loader import DataLoader


class QlibDataHandler(DataHandlerLP):
    """
    Standard Qlib DataHandler implementation with flexible preprocessing.
    
    This class extends Qlib's DataHandlerLP to provide comprehensive
    data preprocessing pipelines for machine learning workflows.
    """
    
    def __init__(
        self,
        data_loader: Union[DataLoader, Dict, str],
        feature_config: Optional[Dict[str, Any]] = None,
        label_config: Optional[Dict[str, Any]] = None,
        processor_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the Qlib data handler.
        
        Args:
            data_loader: Data loader instance or configuration
            feature_config: Feature processing configuration
            label_config: Label processing configuration  
            processor_config: Data processor configuration
            **kwargs: Additional arguments for parent class
        """
        # Set up processor configurations
        self.feature_config = feature_config or {}
        self.label_config = label_config or {}
        self.processor_config = processor_config or {}
        
        # Setup default processors
        infer_processors, learn_processors, shared_processors = self._setup_processors()
        
        # Initialize parent class
        super().__init__(
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            shared_processors=shared_processors,
            **kwargs
        )
    
    def _setup_processors(self) -> Tuple[List, List, List]:
        """
        Setup data processing pipelines for different phases.
        
        Returns:
            Tuple[List, List, List]: (infer_processors, learn_processors, shared_processors)
        """
        # Shared processors (applied to all data)
        shared_processors = []
        
        # Add basic data cleaning processors
        if self.processor_config.get('handle_inf', True):
            shared_processors.append({
                'class': 'ProcessInf',
                'kwargs': {}
            })
        
        if self.processor_config.get('fill_na', True):
            shared_processors.append({
                'class': 'Fillna',
                'kwargs': {
                    'fields_group': 'feature',
                    'fill_value': self.processor_config.get('fill_value', 0)
                }
            })
        
        # Feature normalization
        normalization_method = self.processor_config.get('normalization', 'zscore')
        if normalization_method and normalization_method != 'none':
            norm_config = self._get_normalization_config(normalization_method)
            shared_processors.append(norm_config)
        
        # Inference processors (for prediction/inference)
        infer_processors = []
        
        # Feature selection for inference
        if self.feature_config.get('selected_features'):
            infer_processors.append({
                'class': 'FilterCol',
                'kwargs': {
                    'fields_group': 'feature',
                    'col_list': self.feature_config['selected_features']
                }
            })
        
        # Learning processors (for training only)
        learn_processors = []
        
        # Drop samples with missing labels
        if self.label_config.get('drop_na_labels', True):
            learn_processors.append({
                'class': 'DropnaLabel',
                'kwargs': {
                    'fields_group': 'label'
                }
            })
        
        # Drop specific columns if specified
        if self.processor_config.get('drop_columns'):
            learn_processors.append({
                'class': 'DropCol',
                'kwargs': {
                    'col_list': self.processor_config['drop_columns']
                }
            })
        
        return infer_processors, learn_processors, shared_processors
    
    def _get_normalization_config(self, method: str) -> Dict[str, Any]:
        """
        Get normalization processor configuration.
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'robust', 'cs_zscore', 'cs_rank')
            
        Returns:
            Dict[str, Any]: Processor configuration
        """
        # Get fit time range from config
        fit_start = self.processor_config.get('fit_start_time')
        fit_end = self.processor_config.get('fit_end_time')
        
        if method == 'zscore':
            if fit_start and fit_end:
                return {
                    'class': 'ZScoreNorm',
                    'kwargs': {
                        'fit_start_time': fit_start,
                        'fit_end_time': fit_end,
                        'fields_group': 'feature'
                    }
                }
            else:
                # Use cross-sectional z-score if no fit time specified
                return {
                    'class': 'CSZScoreNorm',
                    'kwargs': {
                        'fields_group': 'feature'
                    }
                }
        
        elif method == 'minmax':
            if not (fit_start and fit_end):
                raise ValueError("MinMax normalization requires fit_start_time and fit_end_time")
            return {
                'class': 'MinMaxNorm',
                'kwargs': {
                    'fit_start_time': fit_start,
                    'fit_end_time': fit_end,
                    'fields_group': 'feature'
                }
            }
        
        elif method == 'robust':
            if not (fit_start and fit_end):
                raise ValueError("Robust normalization requires fit_start_time and fit_end_time")
            return {
                'class': 'RobustZScoreNorm',
                'kwargs': {
                    'fit_start_time': fit_start,
                    'fit_end_time': fit_end,
                    'fields_group': 'feature',
                    'clip_outlier': self.processor_config.get('clip_outlier', True)
                }
            }
        
        elif method == 'cs_zscore':
            return {
                'class': 'CSZScoreNorm',
                'kwargs': {
                    'fields_group': 'feature',
                    'method': 'zscore'
                }
            }
        
        elif method == 'cs_rank':
            return {
                'class': 'CSRankNorm',
                'kwargs': {
                    'fields_group': 'feature'
                }
            }
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def setup_data(
        self,
        enable_cache: bool = True,
        **kwargs
    ) -> None:
        """
        Setup data with enhanced configuration options.
        
        Args:
            enable_cache: Whether to enable data caching
            **kwargs: Additional setup arguments
        """
        # Setup handler arguments
        handler_kwargs = {
            'enable_cache': enable_cache
        }
        
        # Add any additional arguments
        handler_kwargs.update(kwargs)
        
        # Call parent setup
        super().setup_data(handler_kwargs=handler_kwargs)
    
    def fetch(
        self,
        selector: Union[str, slice, List[str]] = slice(None),
        level: str = 'datetime',
        col_set: Union[str, List[str]] = 'feature',
        data_key: str = 'infer',
        squeeze: bool = False
    ) -> pd.DataFrame:
        """
        Enhanced fetch method with additional options.
        
        Args:
            selector: Time selector for data fetching
            level: Index level for selection
            col_set: Column set to fetch ('feature', 'label', or list)
            data_key: Data key ('infer', 'learn', or raw data key)
            squeeze: Whether to squeeze single-column results
            
        Returns:
            pd.DataFrame: Fetched data
        """
        try:
            # Call parent fetch method
            data = super().fetch(
                selector=selector,
                level=level,
                col_set=col_set,
                data_key=data_key
            )
            
            # Apply post-processing if configured
            data = self._post_process_data(data, col_set, data_key)
            
            # Squeeze if requested and appropriate
            if squeeze and isinstance(data, pd.DataFrame) and data.shape[1] == 1:
                data = data.iloc[:, 0]
            
            return data
            
        except Exception as e:
            warnings.warn(f"Error fetching data: {str(e)}")
            # Return empty DataFrame with appropriate structure
            return self._create_empty_dataframe(col_set)
    
    def _post_process_data(
        self,
        data: pd.DataFrame,
        col_set: Union[str, List[str]],
        data_key: str
    ) -> pd.DataFrame:
        """
        Apply post-processing to fetched data.
        
        Args:
            data: Raw fetched data
            col_set: Column set that was fetched
            data_key: Data key that was used
            
        Returns:
            pd.DataFrame: Post-processed data
        """
        if data.empty:
            return data
        
        # Apply any additional data quality checks
        if self.processor_config.get('final_quality_check', True):
            data = self._apply_quality_checks(data)
        
        # Apply feature selection if specified for this col_set
        if col_set == 'feature' and self.feature_config.get('final_feature_selection'):
            selected_features = self.feature_config['final_feature_selection']
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-level columns
                available_features = [
                    col for col in data.columns 
                    if col[0] == 'feature' and col[1] in selected_features
                ]
                if available_features:
                    data = data[available_features]
            else:
                # Simple columns
                available_features = [col for col in selected_features if col in data.columns]
                if available_features:
                    data = data[available_features]
        
        return data
    
    def _apply_quality_checks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final data quality checks.
        
        Args:
            data: Input data
            
        Returns:
            pd.DataFrame: Quality-checked data
        """
        # Check for and handle any remaining infinite values
        if np.isinf(data.select_dtypes(include=[np.number])).any().any():
            warnings.warn("Infinite values detected in data, replacing with NaN")
            data = data.replace([np.inf, -np.inf], np.nan)
        
        # Check for extremely high variance features (potential data errors)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].std() > 1000:  # Configurable threshold
                warnings.warn(f"High variance detected in column {col}")
        
        # Check for constant features
        constant_cols = []
        for col in numeric_cols:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols and self.processor_config.get('drop_constant_features', True):
            warnings.warn(f"Dropping constant features: {constant_cols}")
            data = data.drop(columns=constant_cols)
        
        return data
    
    def _create_empty_dataframe(self, col_set: Union[str, List[str]]) -> pd.DataFrame:
        """
        Create empty DataFrame with appropriate structure.
        
        Args:
            col_set: Column set specification
            
        Returns:
            pd.DataFrame: Empty DataFrame with correct structure
        """
        # Create empty multi-index
        empty_index = pd.MultiIndex.from_tuples(
            [], names=['datetime', 'instrument']
        )
        
        if isinstance(col_set, str):
            # Single column group
            empty_columns = pd.MultiIndex.from_tuples(
                [], names=['field_group', 'field_name']
            )
        else:
            # Multiple column groups or specific columns
            empty_columns = pd.Index([])
        
        return pd.DataFrame(index=empty_index, columns=empty_columns)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the processed data.
        
        Returns:
            Dict[str, Any]: Data summary information
        """
        summary = {
            'total_samples': 0,
            'instruments': 0,
            'time_range': None,
            'feature_count': 0,
            'label_count': 0,
            'missing_data_ratio': 0.0
        }
        
        try:
            # Get basic data info
            if hasattr(self, '_data') and self._data is not None:
                data = self._data
                summary['total_samples'] = len(data)
                
                if isinstance(data.index, pd.MultiIndex):
                    summary['instruments'] = data.index.get_level_values('instrument').nunique()
                    dates = data.index.get_level_values('datetime')
                    summary['time_range'] = (dates.min(), dates.max())
                
                # Count features and labels
                if isinstance(data.columns, pd.MultiIndex):
                    feature_cols = [col for col in data.columns if col[0] == 'feature']
                    label_cols = [col for col in data.columns if col[0] == 'label']
                    summary['feature_count'] = len(feature_cols)
                    summary['label_count'] = len(label_cols)
                else:
                    summary['feature_count'] = len(data.columns)
                
                # Calculate missing data ratio
                total_cells = data.size
                missing_cells = data.isna().sum().sum()
                summary['missing_data_ratio'] = missing_cells / total_cells if total_cells > 0 else 0.0
        
        except Exception as e:
            warnings.warn(f"Error generating data summary: {str(e)}")
        
        return summary
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """
        Validate data integrity and quality.
        
        Returns:
            Dict[str, Any]: Validation results and warnings
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            if not hasattr(self, '_data') or self._data is None:
                validation_results['errors'].append("No data loaded")
                validation_results['is_valid'] = False
                return validation_results
            
            data = self._data
            
            # Check for empty data
            if data.empty:
                validation_results['errors'].append("Data is empty")
                validation_results['is_valid'] = False
                return validation_results
            
            # Check index structure
            if not isinstance(data.index, pd.MultiIndex):
                validation_results['warnings'].append("Data does not have MultiIndex (datetime, instrument)")
            elif data.index.names != ['datetime', 'instrument']:
                validation_results['warnings'].append(f"Unexpected index names: {data.index.names}")
            
            # Check for missing data
            missing_ratio = data.isna().sum().sum() / data.size
            if missing_ratio > 0.1:
                validation_results['warnings'].append(f"High missing data ratio: {missing_ratio:.2%}")
            
            # Check for infinite values
            numeric_data = data.select_dtypes(include=[np.number])
            if np.isinf(numeric_data).any().any():
                validation_results['warnings'].append("Infinite values detected in data")
            
            # Check for duplicate index values
            if data.index.duplicated().any():
                validation_results['errors'].append("Duplicate index values detected")
                validation_results['is_valid'] = False
            
            # Check feature/label balance
            if isinstance(data.columns, pd.MultiIndex):
                feature_count = sum(1 for col in data.columns if col[0] == 'feature')
                label_count = sum(1 for col in data.columns if col[0] == 'label')
                
                if feature_count == 0:
                    validation_results['errors'].append("No feature columns found")
                    validation_results['is_valid'] = False
                
                if label_count == 0:
                    validation_results['warnings'].append("No label columns found")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {str(e)}")
            validation_results['is_valid'] = False
        
        return validation_results
