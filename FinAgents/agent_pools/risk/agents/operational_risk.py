"""
Operational Risk Analysis Agent

Author: Jifeng Li
License: openMDW
Description: Comprehensive operational risk analysis including fraud detection,
             system failures, human error, and compliance risk assessment.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OperationalRiskEvent:
    """Data class for operational risk events"""
    event_id: str
    event_type: str  # fraud, system_failure, human_error, compliance, etc.
    severity: str    # low, medium, high, critical
    impact_amount: float
    business_line: str
    event_date: datetime
    resolution_date: Optional[datetime] = None
    description: str = ""
    root_cause: str = ""
    mitigation_actions: List[str] = None

    def __post_init__(self):
        if self.mitigation_actions is None:
            self.mitigation_actions = []


@dataclass
class OpRiskMetrics:
    """Operational risk metrics and KPIs"""
    total_events: int
    total_losses: float
    avg_loss_per_event: float
    frequency_by_type: Dict[str, int]
    losses_by_business_line: Dict[str, float]
    severity_distribution: Dict[str, int]
    resolution_time_avg: float  # in days
    trend_analysis: Dict[str, float]


class OperationalRiskAnalyzer:
    """
    Comprehensive operational risk analysis engine
    
    Provides:
    - Event tracking and analysis
    - Loss distribution modeling
    - Frequency analysis
    - Scenario analysis
    - Key risk indicators (KRIs)
    - Regulatory capital calculation
    """
    
    def __init__(self):
        """Initialize the operational risk analyzer"""
        self.events_history: List[OperationalRiskEvent] = []
        self.kri_thresholds = {
            'system_downtime_hours': 24,
            'failed_transactions_pct': 0.05,
            'staff_turnover_rate': 0.15,
            'compliance_violations': 5,
            'fraud_incidents_monthly': 10
        }
        
    async def record_operational_event(
        self,
        event_data: Dict[str, Any]
    ) -> OperationalRiskEvent:
        """
        Record a new operational risk event
        
        Args:
            event_data: Dictionary containing event details
            
        Returns:
            OperationalRiskEvent: The recorded event
        """
        try:
            event = OperationalRiskEvent(
                event_id=event_data.get('event_id', f"OP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                event_type=event_data['event_type'],
                severity=event_data.get('severity', 'medium'),
                impact_amount=float(event_data.get('impact_amount', 0)),
                business_line=event_data['business_line'],
                event_date=event_data.get('event_date', datetime.now()),
                resolution_date=event_data.get('resolution_date'),
                description=event_data.get('description', ''),
                root_cause=event_data.get('root_cause', ''),
                mitigation_actions=event_data.get('mitigation_actions', [])
            )
            
            self.events_history.append(event)
            
            logger.info(f"Recorded operational risk event: {event.event_id}")
            return event
            
        except Exception as e:
            logger.error(f"Error recording operational event: {str(e)}")
            raise
    
    async def calculate_operational_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> OpRiskMetrics:
        """
        Calculate comprehensive operational risk metrics
        
        Args:
            start_date: Start date for analysis period
            end_date: End date for analysis period
            
        Returns:
            OpRiskMetrics: Calculated operational risk metrics
        """
        try:
            # Filter events by date range
            filtered_events = self._filter_events_by_date(start_date, end_date)
            
            if not filtered_events:
                return OpRiskMetrics(
                    total_events=0,
                    total_losses=0.0,
                    avg_loss_per_event=0.0,
                    frequency_by_type={},
                    losses_by_business_line={},
                    severity_distribution={},
                    resolution_time_avg=0.0,
                    trend_analysis={}
                )
            
            # Calculate basic metrics
            total_events = len(filtered_events)
            total_losses = sum(event.impact_amount for event in filtered_events)
            avg_loss_per_event = total_losses / total_events if total_events > 0 else 0
            
            # Frequency by type
            frequency_by_type = {}
            for event in filtered_events:
                frequency_by_type[event.event_type] = frequency_by_type.get(event.event_type, 0) + 1
            
            # Losses by business line
            losses_by_business_line = {}
            for event in filtered_events:
                line = event.business_line
                losses_by_business_line[line] = losses_by_business_line.get(line, 0) + event.impact_amount
            
            # Severity distribution
            severity_distribution = {}
            for event in filtered_events:
                severity_distribution[event.severity] = severity_distribution.get(event.severity, 0) + 1
            
            # Average resolution time
            resolved_events = [e for e in filtered_events if e.resolution_date]
            if resolved_events:
                resolution_times = [(e.resolution_date - e.event_date).days for e in resolved_events]
                resolution_time_avg = sum(resolution_times) / len(resolution_times)
            else:
                resolution_time_avg = 0.0
            
            # Trend analysis
            trend_analysis = await self._calculate_trends(filtered_events)
            
            return OpRiskMetrics(
                total_events=total_events,
                total_losses=total_losses,
                avg_loss_per_event=avg_loss_per_event,
                frequency_by_type=frequency_by_type,
                losses_by_business_line=losses_by_business_line,
                severity_distribution=severity_distribution,
                resolution_time_avg=resolution_time_avg,
                trend_analysis=trend_analysis
            )
            
        except Exception as e:
            logger.error(f"Error calculating operational metrics: {str(e)}")
            raise
    
    async def assess_fraud_risk(
        self,
        transaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess fraud risk for transactions or activities
        
        Args:
            transaction_data: Transaction or activity data
            
        Returns:
            Dict containing fraud risk assessment
        """
        try:
            risk_score = 0.0
            risk_factors = []
            
            # Transaction amount analysis
            amount = transaction_data.get('amount', 0)
            if amount > 100000:  # Large transaction threshold
                risk_score += 0.3
                risk_factors.append("Large transaction amount")
            
            # Frequency analysis
            user_id = transaction_data.get('user_id')
            recent_transactions = transaction_data.get('recent_transaction_count', 0)
            if recent_transactions > 10:  # High frequency threshold
                risk_score += 0.2
                risk_factors.append("High transaction frequency")
            
            # Geographic analysis
            location = transaction_data.get('location', '')
            if location in ['high_risk_country_1', 'high_risk_country_2']:  # Example high-risk locations
                risk_score += 0.4
                risk_factors.append("High-risk geographic location")
            
            # Time-based analysis
            transaction_time = transaction_data.get('timestamp', datetime.now())
            if transaction_time.hour < 6 or transaction_time.hour > 22:  # Off-hours
                risk_score += 0.1
                risk_factors.append("Off-hours transaction")
            
            # Historical pattern analysis
            user_pattern = transaction_data.get('deviates_from_pattern', False)
            if user_pattern:
                risk_score += 0.3
                risk_factors.append("Deviates from user pattern")
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "HIGH"
            elif risk_score >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'risk_score': min(risk_score, 1.0),  # Cap at 1.0
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendation': self._get_fraud_recommendation(risk_level),
                'requires_review': risk_score >= 0.4
            }
            
        except Exception as e:
            logger.error(f"Error assessing fraud risk: {str(e)}")
            raise
    
    async def monitor_key_risk_indicators(
        self,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Monitor key risk indicators and alert on threshold breaches
        
        Args:
            current_metrics: Current KRI values
            
        Returns:
            Dict containing KRI status and alerts
        """
        try:
            alerts = []
            kri_status = {}
            
            for kri, current_value in current_metrics.items():
                threshold = self.kri_thresholds.get(kri)
                
                if threshold is None:
                    kri_status[kri] = {
                        'value': current_value,
                        'status': 'NO_THRESHOLD',
                        'threshold': None
                    }
                    continue
                
                # Check threshold breach
                breached = current_value > threshold
                
                kri_status[kri] = {
                    'value': current_value,
                    'threshold': threshold,
                    'status': 'BREACHED' if breached else 'NORMAL',
                    'breach_percentage': ((current_value - threshold) / threshold * 100) if breached else 0
                }
                
                if breached:
                    alerts.append({
                        'kri': kri,
                        'current_value': current_value,
                        'threshold': threshold,
                        'severity': 'HIGH' if current_value > threshold * 1.5 else 'MEDIUM',
                        'timestamp': datetime.now()
                    })
            
            return {
                'kri_status': kri_status,
                'alerts': alerts,
                'total_breaches': len(alerts),
                'overall_status': 'ALERT' if alerts else 'NORMAL'
            }
            
        except Exception as e:
            logger.error(f"Error monitoring KRIs: {str(e)}")
            raise
    
    async def calculate_operational_var(
        self,
        confidence_level: float = 0.99,
        time_horizon_days: int = 365
    ) -> Dict[str, float]:
        """
        Calculate Operational Value at Risk using loss distribution approach
        
        Args:
            confidence_level: Confidence level for VaR calculation
            time_horizon_days: Time horizon in days
            
        Returns:
            Dict containing OpVaR calculations
        """
        try:
            if not self.events_history:
                return {
                    'operational_var': 0.0,
                    'expected_loss': 0.0,
                    'unexpected_loss': 0.0,
                    'loss_distribution_params': {}
                }
            
            # Extract loss amounts
            losses = [event.impact_amount for event in self.events_history if event.impact_amount > 0]
            
            if not losses:
                return {
                    'operational_var': 0.0,
                    'expected_loss': 0.0,
                    'unexpected_loss': 0.0,
                    'loss_distribution_params': {}
                }
            
            losses_array = np.array(losses)
            
            # Calculate expected loss (mean)
            expected_loss = np.mean(losses_array)
            
            # Calculate percentile for VaR
            var_percentile = np.percentile(losses_array, confidence_level * 100)
            
            # Calculate unexpected loss (standard deviation)
            unexpected_loss = np.std(losses_array)
            
            # Fit loss distribution (log-normal assumption)
            log_losses = np.log(losses_array + 1)  # Add 1 to handle zero losses
            mu = np.mean(log_losses)
            sigma = np.std(log_losses)
            
            # Scale for time horizon (square root of time scaling)
            time_scaling_factor = np.sqrt(time_horizon_days / 365)
            scaled_var = var_percentile * time_scaling_factor
            
            return {
                'operational_var': scaled_var,
                'expected_loss': expected_loss,
                'unexpected_loss': unexpected_loss,
                'loss_distribution_params': {
                    'mu': mu,
                    'sigma': sigma,
                    'distribution_type': 'lognormal'
                },
                'confidence_level': confidence_level,
                'time_horizon_days': time_horizon_days
            }
            
        except Exception as e:
            logger.error(f"Error calculating operational VaR: {str(e)}")
            raise
    
    async def scenario_analysis(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform operational risk scenario analysis
        
        Args:
            scenarios: List of scenario definitions
            
        Returns:
            Dict containing scenario analysis results
        """
        try:
            scenario_results = []
            
            for scenario in scenarios:
                scenario_name = scenario.get('name', 'Unnamed Scenario')
                event_types = scenario.get('event_types', [])
                severity_multiplier = scenario.get('severity_multiplier', 1.0)
                frequency_multiplier = scenario.get('frequency_multiplier', 1.0)
                
                # Calculate baseline metrics for specified event types
                relevant_events = [
                    event for event in self.events_history
                    if event.event_type in event_types
                ]
                
                if relevant_events:
                    baseline_losses = sum(event.impact_amount for event in relevant_events)
                    baseline_frequency = len(relevant_events)
                    
                    # Apply scenario stress factors
                    stressed_losses = baseline_losses * severity_multiplier
                    stressed_frequency = baseline_frequency * frequency_multiplier
                    
                    scenario_results.append({
                        'scenario_name': scenario_name,
                        'baseline_losses': baseline_losses,
                        'stressed_losses': stressed_losses,
                        'baseline_frequency': baseline_frequency,
                        'stressed_frequency': stressed_frequency,
                        'loss_increase': stressed_losses - baseline_losses,
                        'loss_increase_pct': ((stressed_losses - baseline_losses) / baseline_losses * 100) if baseline_losses > 0 else 0
                    })
                else:
                    scenario_results.append({
                        'scenario_name': scenario_name,
                        'baseline_losses': 0,
                        'stressed_losses': 0,
                        'baseline_frequency': 0,
                        'stressed_frequency': 0,
                        'loss_increase': 0,
                        'loss_increase_pct': 0
                    })
            
            # Calculate aggregate impact
            total_additional_losses = sum(result['loss_increase'] for result in scenario_results)
            
            return {
                'scenario_results': scenario_results,
                'total_additional_losses': total_additional_losses,
                'most_severe_scenario': max(scenario_results, key=lambda x: x['loss_increase']) if scenario_results else None,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in scenario analysis: {str(e)}")
            raise
    
    def _filter_events_by_date(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[OperationalRiskEvent]:
        """Filter events by date range"""
        if not start_date and not end_date:
            return self.events_history
        
        filtered = []
        for event in self.events_history:
            if start_date and event.event_date < start_date:
                continue
            if end_date and event.event_date > end_date:
                continue
            filtered.append(event)
        
        return filtered
    
    async def _calculate_trends(
        self,
        events: List[OperationalRiskEvent]
    ) -> Dict[str, float]:
        """Calculate trend analysis for events"""
        try:
            if len(events) < 2:
                return {}
            
            # Sort events by date
            sorted_events = sorted(events, key=lambda x: x.event_date)
            
            # Calculate monthly trends
            monthly_counts = {}
            monthly_losses = {}
            
            for event in sorted_events:
                month_key = event.event_date.strftime('%Y-%m')
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                monthly_losses[month_key] = monthly_losses.get(month_key, 0) + event.impact_amount
            
            # Calculate trend (simple linear trend)
            months = sorted(monthly_counts.keys())
            if len(months) >= 2:
                count_values = [monthly_counts[month] for month in months]
                loss_values = [monthly_losses[month] for month in months]
                
                # Simple trend calculation (change from first to last period)
                count_trend = (count_values[-1] - count_values[0]) / len(count_values) if len(count_values) > 1 else 0
                loss_trend = (loss_values[-1] - loss_values[0]) / len(loss_values) if len(loss_values) > 1 else 0
                
                return {
                    'frequency_trend': count_trend,
                    'loss_trend': loss_trend,
                    'periods_analyzed': len(months)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating trends: {str(e)}")
            return {}
    
    def _get_fraud_recommendation(self, risk_level: str) -> str:
        """Get fraud risk recommendation based on risk level"""
        recommendations = {
            'LOW': 'Proceed with standard processing',
            'MEDIUM': 'Apply enhanced monitoring and secondary verification',
            'HIGH': 'Hold transaction for manual review and investigation'
        }
        return recommendations.get(risk_level, 'Unknown risk level')
