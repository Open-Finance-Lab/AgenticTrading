"""
Model Risk Management Agent

Author: Jifeng Li
License: openMDW
Description: Comprehensive model risk management including model validation,
             performance monitoring, model inventory management, and governance.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of financial models"""
    PRICING = "pricing"
    RISK = "risk"
    TRADING = "trading"
    CREDIT = "credit"
    MARKET_RISK = "market_risk"
    OPERATIONAL_RISK = "operational_risk"
    STRESS_TESTING = "stress_testing"
    REGULATORY = "regulatory"
    VALUATION = "valuation"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"


class ModelStatus(Enum):
    """Model lifecycle status"""
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    APPROVED = "approved"
    PRODUCTION = "production"
    MONITORING = "monitoring"
    REVIEW = "review"
    RETIRED = "retired"
    SUSPENDED = "suspended"


class ValidationResult(Enum):
    """Model validation results"""
    PASS = "pass"
    CONDITIONAL_PASS = "conditional_pass"
    FAIL = "fail"
    PENDING = "pending"


@dataclass
class ModelMetadata:
    """Model metadata and governance information"""
    model_id: str
    name: str
    model_type: ModelType
    version: str
    developer: str
    business_owner: str
    description: str
    purpose: str
    status: ModelStatus
    created_date: datetime
    last_updated: datetime
    approval_date: Optional[datetime] = None
    retirement_date: Optional[datetime] = None
    next_review_date: Optional[datetime] = None
    regulatory_classification: str = "non_regulatory"
    criticality_level: str = "medium"  # low, medium, high, critical
    documentation_links: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking metrics"""
    model_id: str
    measurement_date: datetime
    accuracy_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    data_quality_metrics: Dict[str, float]
    usage_statistics: Dict[str, int]
    benchmark_comparisons: Dict[str, float]
    alerts: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Model validation report"""
    model_id: str
    validation_id: str
    validator: str
    validation_date: datetime
    result: ValidationResult
    methodology: str
    test_results: Dict[str, Any]
    findings: List[str]
    recommendations: List[str]
    limitations: List[str]
    approval_conditions: List[str] = field(default_factory=list)
    next_validation_date: Optional[datetime] = None


@dataclass
class ModelChange:
    """Model change tracking"""
    change_id: str
    model_id: str
    change_type: str  # code, data, parameters, documentation
    change_description: str
    changed_by: str
    change_date: datetime
    impact_assessment: str
    approval_required: bool
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None


class ModelRiskManager:
    """
    Comprehensive model risk management system
    
    Provides:
    - Model inventory management
    - Model validation framework
    - Performance monitoring
    - Change management
    - Governance and reporting
    - Model lifecycle management
    """
    
    def __init__(self):
        """Initialize the model risk manager"""
        self.model_inventory: Dict[str, ModelMetadata] = {}
        self.performance_history: Dict[str, List[ModelPerformanceMetrics]] = {}
        self.validation_reports: Dict[str, List[ValidationReport]] = {}
        self.change_log: List[ModelChange] = []
        
        # Risk thresholds
        self.performance_thresholds = {
            'accuracy_decline': 0.05,  # 5% accuracy decline threshold
            'stability_breach': 0.10,  # 10% stability metric breach
            'data_quality_minimum': 0.90,  # 90% data quality minimum
            'usage_spike': 2.0,  # 2x usage increase threshold
            'benchmark_underperformance': 0.02  # 2% underperformance vs benchmark
        }
        
        # Validation schedules by criticality
        self.validation_schedules = {
            'critical': 180,    # days - every 6 months
            'high': 365,       # days - annually
            'medium': 730,     # days - every 2 years
            'low': 1095        # days - every 3 years
        }
    
    async def register_model(
        self,
        model_metadata: ModelMetadata
    ) -> str:
        """
        Register a new model in the inventory
        
        Args:
            model_metadata: Model metadata information
            
        Returns:
            str: Model ID
        """
        try:
            # Generate model ID if not provided
            if not model_metadata.model_id:
                model_metadata.model_id = self._generate_model_id(model_metadata)
            
            # Set initial dates
            model_metadata.created_date = datetime.now()
            model_metadata.last_updated = datetime.now()
            
            # Calculate next review date
            if model_metadata.criticality_level in self.validation_schedules:
                days_to_review = self.validation_schedules[model_metadata.criticality_level]
                model_metadata.next_review_date = datetime.now() + timedelta(days=days_to_review)
            
            # Store in inventory
            self.model_inventory[model_metadata.model_id] = model_metadata
            
            # Initialize performance tracking
            self.performance_history[model_metadata.model_id] = []
            self.validation_reports[model_metadata.model_id] = []
            
            logger.info(f"Registered model: {model_metadata.model_id} - {model_metadata.name}")
            return model_metadata.model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    async def validate_model(
        self,
        model_id: str,
        validator: str,
        validation_config: Dict[str, Any]
    ) -> ValidationReport:
        """
        Perform comprehensive model validation
        
        Args:
            model_id: Model identifier
            validator: Name of validator
            validation_config: Validation configuration and tests
            
        Returns:
            ValidationReport: Validation results
        """
        try:
            if model_id not in self.model_inventory:
                raise ValueError(f"Model {model_id} not found in inventory")
            
            model = self.model_inventory[model_id]
            validation_id = f"VAL_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Perform validation tests
            test_results = {}
            findings = []
            recommendations = []
            limitations = []
            
            # Data quality validation
            if 'data_quality' in validation_config:
                data_quality_result = await self._validate_data_quality(
                    model_id, validation_config['data_quality']
                )
                test_results['data_quality'] = data_quality_result
                
                if data_quality_result['overall_score'] < 0.9:
                    findings.append("Data quality below acceptable threshold")
                    recommendations.append("Improve data cleansing and validation processes")
            
            # Model accuracy validation
            if 'accuracy_tests' in validation_config:
                accuracy_result = await self._validate_model_accuracy(
                    model_id, validation_config['accuracy_tests']
                )
                test_results['accuracy'] = accuracy_result
                
                if accuracy_result['accuracy_score'] < validation_config['accuracy_tests'].get('min_accuracy', 0.85):
                    findings.append("Model accuracy below minimum threshold")
                    recommendations.append("Retrain model or adjust parameters")
            
            # Stability validation
            if 'stability_tests' in validation_config:
                stability_result = await self._validate_model_stability(
                    model_id, validation_config['stability_tests']
                )
                test_results['stability'] = stability_result
                
                if stability_result['stability_score'] < 0.8:
                    findings.append("Model stability concerns identified")
                    recommendations.append("Investigate parameter sensitivity and model robustness")
            
            # Bias and fairness validation
            if 'bias_tests' in validation_config:
                bias_result = await self._validate_model_bias(
                    model_id, validation_config['bias_tests']
                )
                test_results['bias'] = bias_result
                
                if bias_result['bias_detected']:
                    findings.append("Model bias detected in protected characteristics")
                    recommendations.append("Implement bias mitigation techniques")
            
            # Performance benchmarking
            if 'benchmark_tests' in validation_config:
                benchmark_result = await self._validate_model_benchmarks(
                    model_id, validation_config['benchmark_tests']
                )
                test_results['benchmarks'] = benchmark_result
                
                if benchmark_result['underperforming_benchmarks']:
                    findings.append("Model underperforming against benchmarks")
                    recommendations.append("Investigate model improvements or benchmark relevance")
            
            # Determine overall validation result
            overall_result = self._determine_validation_result(test_results, findings)
            
            # Create validation report
            report = ValidationReport(
                model_id=model_id,
                validation_id=validation_id,
                validator=validator,
                validation_date=datetime.now(),
                result=overall_result,
                methodology=validation_config.get('methodology', 'Standard validation'),
                test_results=test_results,
                findings=findings,
                recommendations=recommendations,
                limitations=limitations
            )
            
            # Set next validation date based on result and criticality
            if overall_result == ValidationResult.PASS:
                days_to_next = self.validation_schedules[model.criticality_level]
                report.next_validation_date = datetime.now() + timedelta(days=days_to_next)
            elif overall_result == ValidationResult.CONDITIONAL_PASS:
                report.next_validation_date = datetime.now() + timedelta(days=90)  # 3 months
            else:  # FAIL
                report.next_validation_date = datetime.now() + timedelta(days=30)  # 1 month
            
            # Store validation report
            self.validation_reports[model_id].append(report)
            
            # Update model status
            if overall_result == ValidationResult.PASS:
                model.status = ModelStatus.APPROVED
                model.approval_date = datetime.now()
            elif overall_result == ValidationResult.FAIL:
                model.status = ModelStatus.SUSPENDED
            
            model.next_review_date = report.next_validation_date
            model.last_updated = datetime.now()
            
            logger.info(f"Validation completed for model {model_id}: {overall_result.value}")
            return report
            
        except Exception as e:
            logger.error(f"Error validating model {model_id}: {str(e)}")
            raise
    
    async def monitor_model_performance(
        self,
        model_id: str,
        performance_data: Dict[str, Any]
    ) -> ModelPerformanceMetrics:
        """
        Monitor model performance and detect issues
        
        Args:
            model_id: Model identifier
            performance_data: Performance data and metrics
            
        Returns:
            ModelPerformanceMetrics: Performance metrics with alerts
        """
        try:
            if model_id not in self.model_inventory:
                raise ValueError(f"Model {model_id} not found in inventory")
            
            # Extract metrics from performance data
            accuracy_metrics = performance_data.get('accuracy', {})
            stability_metrics = performance_data.get('stability', {})
            performance_metrics = performance_data.get('performance', {})
            data_quality_metrics = performance_data.get('data_quality', {})
            usage_statistics = performance_data.get('usage', {})
            benchmark_comparisons = performance_data.get('benchmarks', {})
            
            # Generate alerts based on thresholds
            alerts = await self._generate_performance_alerts(
                model_id, accuracy_metrics, stability_metrics, 
                performance_metrics, data_quality_metrics, usage_statistics
            )
            
            # Create performance metrics record
            metrics = ModelPerformanceMetrics(
                model_id=model_id,
                measurement_date=datetime.now(),
                accuracy_metrics=accuracy_metrics,
                stability_metrics=stability_metrics,
                performance_metrics=performance_metrics,
                data_quality_metrics=data_quality_metrics,
                usage_statistics=usage_statistics,
                benchmark_comparisons=benchmark_comparisons,
                alerts=alerts
            )
            
            # Store performance metrics
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            self.performance_history[model_id].append(metrics)
            
            # Update model status if critical alerts
            critical_alerts = [alert for alert in alerts if 'CRITICAL' in alert]
            if critical_alerts:
                model = self.model_inventory[model_id]
                if model.status == ModelStatus.PRODUCTION:
                    model.status = ModelStatus.REVIEW
                    model.last_updated = datetime.now()
            
            logger.info(f"Performance monitoring completed for model {model_id}: {len(alerts)} alerts")
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring model performance {model_id}: {str(e)}")
            raise
    
    async def track_model_change(
        self,
        model_change: ModelChange
    ) -> str:
        """
        Track and manage model changes
        
        Args:
            model_change: Model change details
            
        Returns:
            str: Change ID
        """
        try:
            # Generate change ID if not provided
            if not model_change.change_id:
                model_change.change_id = f"CHG_{model_change.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Set change date
            model_change.change_date = datetime.now()
            
            # Determine if approval is required
            high_impact_changes = ['code', 'parameters', 'methodology']
            if model_change.change_type in high_impact_changes:
                model_change.approval_required = True
            
            # Store change
            self.change_log.append(model_change)
            
            # Update model metadata
            if model_change.model_id in self.model_inventory:
                model = self.model_inventory[model_change.model_id]
                model.last_updated = datetime.now()
                
                # Change status if significant change
                if model_change.approval_required and model.status == ModelStatus.PRODUCTION:
                    model.status = ModelStatus.VALIDATION
            
            logger.info(f"Tracked model change: {model_change.change_id}")
            return model_change.change_id
            
        except Exception as e:
            logger.error(f"Error tracking model change: {str(e)}")
            raise
    
    async def generate_model_inventory_report(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model inventory report
        
        Args:
            filters: Optional filters for the report
            
        Returns:
            Dict containing inventory report
        """
        try:
            # Apply filters if provided
            models = self.model_inventory.values()
            if filters:
                models = [model for model in models if self._matches_filters(model, filters)]
            
            # Calculate summary statistics
            total_models = len(models)
            models_by_type = {}
            models_by_status = {}
            models_by_criticality = {}
            
            for model in models:
                # By type
                model_type = model.model_type.value
                models_by_type[model_type] = models_by_type.get(model_type, 0) + 1
                
                # By status
                status = model.status.value
                models_by_status[status] = models_by_status.get(status, 0) + 1
                
                # By criticality
                criticality = model.criticality_level
                models_by_criticality[criticality] = models_by_criticality.get(criticality, 0) + 1
            
            # Models requiring attention
            models_due_review = []
            models_with_alerts = []
            suspended_models = []
            
            for model in models:
                # Due for review
                if model.next_review_date and model.next_review_date <= datetime.now():
                    models_due_review.append(model.model_id)
                
                # With recent alerts
                if model.model_id in self.performance_history:
                    recent_metrics = self.performance_history[model.model_id][-1:] if self.performance_history[model.model_id] else []
                    for metrics in recent_metrics:
                        if metrics.alerts:
                            models_with_alerts.append(model.model_id)
                            break
                
                # Suspended models
                if model.status == ModelStatus.SUSPENDED:
                    suspended_models.append(model.model_id)
            
            return {
                'report_date': datetime.now(),
                'summary': {
                    'total_models': total_models,
                    'models_by_type': models_by_type,
                    'models_by_status': models_by_status,
                    'models_by_criticality': models_by_criticality
                },
                'attention_required': {
                    'models_due_review': models_due_review,
                    'models_with_alerts': list(set(models_with_alerts)),
                    'suspended_models': suspended_models
                },
                'governance_metrics': {
                    'total_validations': sum(len(reports) for reports in self.validation_reports.values()),
                    'total_changes': len(self.change_log),
                    'approval_pending_changes': len([c for c in self.change_log if c.approval_required and not c.approved_by])
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating inventory report: {str(e)}")
            raise
    
    def _generate_model_id(self, metadata: ModelMetadata) -> str:
        """Generate unique model ID"""
        content = f"{metadata.name}_{metadata.model_type.value}_{metadata.developer}_{datetime.now().isoformat()}"
        hash_value = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"MODEL_{hash_value.upper()}"
    
    async def _validate_data_quality(
        self,
        model_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data quality for model"""
        # Simulate data quality validation
        # In practice, this would perform actual data quality checks
        return {
            'completeness_score': 0.95,
            'accuracy_score': 0.92,
            'consistency_score': 0.88,
            'timeliness_score': 0.90,
            'overall_score': 0.91,
            'issues_found': ['Minor data inconsistencies in field X']
        }
    
    async def _validate_model_accuracy(
        self,
        model_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model accuracy"""
        # Simulate accuracy validation
        return {
            'accuracy_score': 0.87,
            'precision': 0.85,
            'recall': 0.89,
            'f1_score': 0.87,
            'auc_roc': 0.91,
            'confusion_matrix': [[100, 15], [10, 120]]
        }
    
    async def _validate_model_stability(
        self,
        model_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model stability"""
        # Simulate stability validation
        return {
            'stability_score': 0.85,
            'parameter_sensitivity': 0.15,
            'prediction_variance': 0.08,
            'temporal_stability': 0.90
        }
    
    async def _validate_model_bias(
        self,
        model_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model for bias"""
        # Simulate bias validation
        return {
            'bias_detected': False,
            'demographic_parity': 0.95,
            'equalized_odds': 0.93,
            'protected_characteristics_tested': ['gender', 'age', 'ethnicity']
        }
    
    async def _validate_model_benchmarks(
        self,
        model_id: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate model against benchmarks"""
        # Simulate benchmark validation
        return {
            'benchmark_scores': {
                'industry_standard': 0.89,
                'previous_model': 0.85,
                'simple_baseline': 0.75
            },
            'underperforming_benchmarks': [],
            'relative_performance': 1.05
        }
    
    def _determine_validation_result(
        self,
        test_results: Dict[str, Any],
        findings: List[str]
    ) -> ValidationResult:
        """Determine overall validation result"""
        critical_findings = [f for f in findings if 'critical' in f.lower() or 'fail' in f.lower()]
        moderate_findings = [f for f in findings if 'below' in f.lower() or 'concern' in f.lower()]
        
        if critical_findings:
            return ValidationResult.FAIL
        elif moderate_findings:
            return ValidationResult.CONDITIONAL_PASS
        else:
            return ValidationResult.PASS
    
    async def _generate_performance_alerts(
        self,
        model_id: str,
        accuracy_metrics: Dict[str, float],
        stability_metrics: Dict[str, float],
        performance_metrics: Dict[str, float],
        data_quality_metrics: Dict[str, float],
        usage_statistics: Dict[str, int]
    ) -> List[str]:
        """Generate performance alerts based on thresholds"""
        alerts = []
        
        # Check accuracy decline
        if 'accuracy_score' in accuracy_metrics:
            if model_id in self.performance_history and self.performance_history[model_id]:
                previous_accuracy = self.performance_history[model_id][-1].accuracy_metrics.get('accuracy_score', 0)
                current_accuracy = accuracy_metrics['accuracy_score']
                if previous_accuracy - current_accuracy > self.performance_thresholds['accuracy_decline']:
                    alerts.append(f"CRITICAL: Accuracy declined by {(previous_accuracy - current_accuracy):.2%}")
        
        # Check stability metrics
        if 'stability_score' in stability_metrics:
            if stability_metrics['stability_score'] < (1 - self.performance_thresholds['stability_breach']):
                alerts.append(f"WARNING: Model stability below threshold: {stability_metrics['stability_score']:.2%}")
        
        # Check data quality
        if 'overall_score' in data_quality_metrics:
            if data_quality_metrics['overall_score'] < self.performance_thresholds['data_quality_minimum']:
                alerts.append(f"WARNING: Data quality below minimum: {data_quality_metrics['overall_score']:.2%}")
        
        # Check usage patterns
        if 'total_predictions' in usage_statistics:
            if model_id in self.performance_history and self.performance_history[model_id]:
                previous_usage = self.performance_history[model_id][-1].usage_statistics.get('total_predictions', 0)
                current_usage = usage_statistics['total_predictions']
                if current_usage > previous_usage * self.performance_thresholds['usage_spike']:
                    alerts.append(f"INFO: Usage spike detected: {current_usage} vs {previous_usage}")
        
        return alerts
    
    def _matches_filters(
        self,
        model: ModelMetadata,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if model matches filter criteria"""
        for filter_key, filter_value in filters.items():
            if filter_key == 'model_type' and model.model_type.value != filter_value:
                return False
            elif filter_key == 'status' and model.status.value != filter_value:
                return False
            elif filter_key == 'criticality' and model.criticality_level != filter_value:
                return False
            elif filter_key == 'developer' and model.developer != filter_value:
                return False
        return True
