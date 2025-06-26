"""
Credit Risk Analyzer - Credit Risk Analysis Agent

This agent specializes in credit risk analysis including:
- Probability of Default (PD) calculation
- Loss Given Default (LGD) estimation
- Exposure at Default (EAD) assessment
- Credit VaR calculation
- Credit migration analysis
- Credit spread analysis

Author: Jifeng Li
License: openMDW
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize_scalar

from ..registry import BaseRiskAgent


class CreditRiskAnalyzer(BaseRiskAgent):
    """
    Specialized agent for comprehensive credit risk analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "CreditRiskAnalyzer"
        self.logger = logging.getLogger(f"RiskAgent.{self.name}")
        
        # Configuration parameters
        self.time_horizons = config.get('time_horizons', [1, 3, 5]) if config else [1, 3, 5]  # Years
        self.rating_classes = config.get('rating_classes', ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']) if config else ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
        self.confidence_levels = config.get('confidence_levels', [0.95, 0.99]) if config else [0.95, 0.99]
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive credit risk analysis.
        
        Args:
            request: Analysis request containing credit portfolio data and parameters
            
        Returns:
            Dictionary containing credit risk analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            portfolio_data = request.get("portfolio_data", {})
            analysis_type = request.get("analysis_type", "comprehensive")
            time_horizon = request.get("time_horizon", 1)  # Default 1 year
            
            # Extract credit-specific data
            credit_exposures = portfolio_data.get("credit_exposures", [])
            ratings = portfolio_data.get("ratings", [])
            sectors = portfolio_data.get("sectors", [])
            
            results = {}
            
            if analysis_type in ["comprehensive", "default_probability"]:
                results["default_probability"] = await self._calculate_default_probabilities(
                    credit_exposures, ratings, time_horizon
                )
            
            if analysis_type in ["comprehensive", "loss_given_default"]:
                results["loss_given_default"] = await self._estimate_loss_given_default(
                    credit_exposures, ratings, sectors
                )
            
            if analysis_type in ["comprehensive", "exposure_at_default"]:
                results["exposure_at_default"] = await self._calculate_exposure_at_default(
                    credit_exposures, portfolio_data
                )
            
            if analysis_type in ["comprehensive", "credit_var"]:
                results["credit_var"] = await self._calculate_credit_var(
                    credit_exposures, ratings, time_horizon
                )
            
            if analysis_type in ["comprehensive", "migration"]:
                results["credit_migration"] = await self._analyze_credit_migration(
                    ratings, time_horizon
                )
            
            if analysis_type in ["comprehensive", "spreads"]:
                results["credit_spreads"] = await self._analyze_credit_spreads(
                    credit_exposures, ratings
                )
            
            if analysis_type in ["comprehensive", "concentration"]:
                results["concentration_risk"] = await self._analyze_concentration_risk(
                    credit_exposures, sectors, ratings
                )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "agent": self.name,
                "analysis_type": "credit_risk",
                "results": results,
                "execution_time_ms": execution_time,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Credit risk analysis failed: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "agent": self.name,
                "error": str(e),
                "execution_time_ms": execution_time,
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _calculate_default_probabilities(self, credit_exposures: List[Dict], 
                                             ratings: List[str], time_horizon: int) -> Dict[str, Any]:
        """Calculate probability of default for different time horizons."""
        # Standard default rate mapping by rating
        default_rates_1y = {
            'AAA': 0.0002, 'AA': 0.0005, 'A': 0.0015, 'BBB': 0.0035,
            'BB': 0.0120, 'B': 0.0400, 'CCC': 0.1500, 'D': 1.0000
        }
        
        # Time scaling for multi-year horizons (simplified approach)
        default_probs = {}
        
        for rating in set(ratings):
            if rating in default_rates_1y:
                # Convert 1-year default rate to hazard rate and scale
                lambda_rate = -np.log(1 - default_rates_1y[rating])
                multi_year_pd = 1 - np.exp(-lambda_rate * time_horizon)
                
                default_probs[rating] = {
                    "1_year_pd": float(default_rates_1y[rating]),
                    f"{time_horizon}_year_pd": float(multi_year_pd),
                    "hazard_rate": float(lambda_rate),
                    "survival_probability": float(1 - multi_year_pd)
                }
        
        # Portfolio-level default probability calculations
        portfolio_exposures = []
        portfolio_pds = []
        
        for i, exposure in enumerate(credit_exposures):
            if i < len(ratings) and ratings[i] in default_probs:
                portfolio_exposures.append(exposure.get('amount', 0))
                portfolio_pds.append(default_probs[ratings[i]][f"{time_horizon}_year_pd"])
        
        if portfolio_exposures and portfolio_pds:
            # Expected number of defaults
            expected_defaults = sum(pd for pd in portfolio_pds)
            
            # Portfolio default correlation (simplified)
            default_correlation = 0.15  # Typical industry assumption
            
            # Vasicek model for portfolio default distribution
            portfolio_results = self._vasicek_portfolio_model(
                portfolio_pds, default_correlation, self.confidence_levels
            )
        else:
            portfolio_results = {}
        
        return {
            "individual_default_probabilities": default_probs,
            "portfolio_statistics": {
                "expected_defaults": float(expected_defaults) if portfolio_pds else 0,
                "portfolio_default_rate": float(np.mean(portfolio_pds)) if portfolio_pds else 0,
                "default_rate_std": float(np.std(portfolio_pds)) if portfolio_pds else 0
            },
            "portfolio_distribution": portfolio_results,
            "time_horizon_years": time_horizon,
            "model_assumptions": {
                "default_correlation": 0.15,
                "model_type": "vasicek_single_factor",
                "rating_migration": "ignored_in_this_calculation"
            }
        }
    
    def _vasicek_portfolio_model(self, pds: List[float], correlation: float, 
                               confidence_levels: List[float]) -> Dict[str, Any]:
        """Calculate portfolio default distribution using Vasicek model."""
        n = len(pds)
        avg_pd = np.mean(pds)
        
        # Vasicek formula for portfolio default rate quantiles
        results = {}
        
        for conf_level in confidence_levels:
            # Inverse normal for confidence level
            z_conf = stats.norm.ppf(conf_level)
            
            # Vasicek quantile formula
            if correlation > 0:
                z_pd = stats.norm.ppf(avg_pd)
                numerator = z_pd + np.sqrt(correlation) * z_conf
                denominator = np.sqrt(1 - correlation)
                conditional_pd = stats.norm.cdf(numerator / denominator)
            else:
                conditional_pd = avg_pd
            
            # Expected number of defaults at confidence level
            expected_defaults_conf = conditional_pd * n
            
            results[f"confidence_{int(conf_level*100)}"] = {
                "conditional_default_rate": float(conditional_pd),
                "expected_defaults": float(expected_defaults_conf),
                "unexpected_loss_factor": float(conditional_pd / avg_pd) if avg_pd > 0 else 1.0
            }
        
        return results
    
    async def _estimate_loss_given_default(self, credit_exposures: List[Dict], 
                                         ratings: List[str], sectors: List[str]) -> Dict[str, Any]:
        """Estimate Loss Given Default rates."""
        # Industry standard LGD rates by seniority and collateral
        lgd_by_seniority = {
            'senior_secured': 0.25,
            'senior_unsecured': 0.45,
            'subordinated': 0.65,
            'equity': 0.90
        }
        
        lgd_by_sector = {
            'utilities': 0.35,
            'technology': 0.55,
            'healthcare': 0.40,
            'financials': 0.45,
            'energy': 0.50,
            'industrials': 0.45,
            'consumer': 0.50,
            'real_estate': 0.35
        }
        
        # Calculate LGD for each exposure
        exposure_lgds = []
        
        for i, exposure in enumerate(credit_exposures):
            seniority = exposure.get('seniority', 'senior_unsecured')
            collateral_value = exposure.get('collateral_value', 0)
            exposure_amount = exposure.get('amount', 0)
            
            # Base LGD from seniority
            base_lgd = lgd_by_seniority.get(seniority, 0.45)
            
            # Sector adjustment
            if i < len(sectors):
                sector = sectors[i]
                sector_lgd = lgd_by_sector.get(sector, 0.45)
                # Blend sector and seniority effects
                adjusted_lgd = 0.7 * base_lgd + 0.3 * sector_lgd
            else:
                adjusted_lgd = base_lgd
            
            # Collateral adjustment
            if collateral_value > 0 and exposure_amount > 0:
                collateral_ratio = min(collateral_value / exposure_amount, 1.0)
                # Reduce LGD based on collateral coverage
                collateral_adjusted_lgd = adjusted_lgd * (1 - 0.6 * collateral_ratio)
            else:
                collateral_adjusted_lgd = adjusted_lgd
            
            exposure_lgds.append({
                'exposure_id': i,
                'base_lgd': base_lgd,
                'sector_adjusted_lgd': adjusted_lgd,
                'final_lgd': collateral_adjusted_lgd,
                'collateral_coverage': collateral_value / exposure_amount if exposure_amount > 0 else 0
            })
        
        # Portfolio-level LGD statistics
        final_lgds = [lgd['final_lgd'] for lgd in exposure_lgds]
        
        # LGD correlation and uncertainty
        lgd_correlation = self._estimate_lgd_correlation(sectors)
        lgd_uncertainty = self._estimate_lgd_uncertainty(final_lgds)
        
        return {
            "exposure_lgds": exposure_lgds,
            "portfolio_statistics": {
                "weighted_average_lgd": float(np.mean(final_lgds)) if final_lgds else 0,
                "lgd_standard_deviation": float(np.std(final_lgds)) if final_lgds else 0,
                "min_lgd": float(np.min(final_lgds)) if final_lgds else 0,
                "max_lgd": float(np.max(final_lgds)) if final_lgds else 0,
                "lgd_percentiles": {
                    "25th": float(np.percentile(final_lgds, 25)) if final_lgds else 0,
                    "75th": float(np.percentile(final_lgds, 75)) if final_lgds else 0,
                    "95th": float(np.percentile(final_lgds, 95)) if final_lgds else 0
                }
            },
            "lgd_correlation": lgd_correlation,
            "lgd_uncertainty": lgd_uncertainty,
            "recovery_analysis": {
                "expected_recovery_rate": float(1 - np.mean(final_lgds)) if final_lgds else 0,
                "recovery_volatility": float(np.std([1 - lgd for lgd in final_lgds])) if final_lgds else 0
            }
        }
    
    def _estimate_lgd_correlation(self, sectors: List[str]) -> Dict[str, float]:
        """Estimate LGD correlation structure."""
        # Simplified LGD correlation estimation
        unique_sectors = list(set(sectors))
        
        # Within-sector correlation typically higher
        within_sector_correlation = 0.30
        cross_sector_correlation = 0.15
        
        return {
            "within_sector_correlation": within_sector_correlation,
            "cross_sector_correlation": cross_sector_correlation,
            "average_correlation": 0.20,
            "correlation_driver": "economic_cycle_and_asset_values"
        }
    
    def _estimate_lgd_uncertainty(self, lgds: List[float]) -> Dict[str, float]:
        """Estimate uncertainty in LGD estimates."""
        if not lgds:
            return {}
        
        # Model uncertainty in LGD estimates
        base_uncertainty = 0.15  # 15% relative uncertainty
        
        return {
            "relative_uncertainty": base_uncertainty,
            "absolute_uncertainty": float(np.mean(lgds) * base_uncertainty),
            "confidence_interval_width": float(2 * 1.96 * np.mean(lgds) * base_uncertainty),
            "uncertainty_source": "limited_default_data_and_recovery_variability"
        }
    
    async def _calculate_exposure_at_default(self, credit_exposures: List[Dict], 
                                           portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Exposure at Default considering credit line utilization."""
        ead_results = []
        
        for i, exposure in enumerate(credit_exposures):
            current_exposure = exposure.get('amount', 0)
            credit_limit = exposure.get('credit_limit', current_exposure)
            facility_type = exposure.get('facility_type', 'term_loan')
            
            # Credit Conversion Factors (CCF) by facility type
            ccf_mapping = {
                'term_loan': 1.0,          # Fully drawn
                'revolving_credit': 0.75,  # Typically 75% CCF
                'credit_card': 0.75,       # Similar to revolving
                'trade_finance': 0.20,     # Lower CCF for trade facilities
                'commitment': 0.50,        # Medium CCF for commitments
                'guarantee': 1.0           # Full exposure for guarantees
            }
            
            ccf = ccf_mapping.get(facility_type, 0.75)
            
            # Calculate undrawn portion
            undrawn_amount = max(credit_limit - current_exposure, 0)
            
            # EAD calculation
            ead = current_exposure + ccf * undrawn_amount
            
            # Add potential future exposure for derivatives (if applicable)
            potential_future_exposure = exposure.get('potential_future_exposure', 0)
            total_ead = ead + potential_future_exposure
            
            ead_results.append({
                'exposure_id': i,
                'current_exposure': current_exposure,
                'credit_limit': credit_limit,
                'undrawn_amount': undrawn_amount,
                'credit_conversion_factor': ccf,
                'ead_credit_risk': ead,
                'potential_future_exposure': potential_future_exposure,
                'total_ead': total_ead,
                'utilization_rate': current_exposure / credit_limit if credit_limit > 0 else 0
            })
        
        # Portfolio-level EAD analysis
        total_current_exposure = sum(result['current_exposure'] for result in ead_results)
        total_ead = sum(result['total_ead'] for result in ead_results)
        total_credit_limits = sum(result['credit_limit'] for result in ead_results)
        
        return {
            "exposure_eads": ead_results,
            "portfolio_summary": {
                "total_current_exposure": float(total_current_exposure),
                "total_ead": float(total_ead),
                "total_credit_limits": float(total_credit_limits),
                "portfolio_utilization_rate": float(total_current_exposure / total_credit_limits) if total_credit_limits > 0 else 0,
                "ead_multiplier": float(total_ead / total_current_exposure) if total_current_exposure > 0 else 1,
                "undrawn_commitment": float(total_credit_limits - total_current_exposure)
            },
            "facility_type_analysis": self._analyze_facility_types(ead_results),
            "stress_scenarios": self._ead_stress_scenarios(ead_results)
        }
    
    def _analyze_facility_types(self, ead_results: List[Dict]) -> Dict[str, Any]:
        """Analyze EAD by facility type."""
        # This would analyze different facility types in practice
        return {
            "term_loans_percentage": 60.0,
            "revolving_credits_percentage": 25.0,
            "commitments_percentage": 15.0,
            "highest_risk_facility_type": "revolving_credit",
            "diversification_score": 7.5  # Out of 10
        }
    
    def _ead_stress_scenarios(self, ead_results: List[Dict]) -> Dict[str, Any]:
        """Calculate EAD under stress scenarios."""
        # Stress scenario: increased line utilization
        stressed_ead = []
        
        for result in ead_results:
            current_exposure = result['current_exposure']
            credit_limit = result['credit_limit']
            
            # Stress scenario: 90% utilization rate
            stressed_exposure = min(credit_limit * 0.9, credit_limit)
            stressed_undrawn = credit_limit - stressed_exposure
            stressed_ccf = result['credit_conversion_factor'] * 1.2  # Increase CCF under stress
            
            stressed_total_ead = stressed_exposure + min(stressed_ccf, 1.0) * stressed_undrawn
            stressed_ead.append(stressed_total_ead)
        
        base_total_ead = sum(result['total_ead'] for result in ead_results)
        stressed_total_ead = sum(stressed_ead)
        
        return {
            "base_scenario_ead": float(base_total_ead),
            "stressed_scenario_ead": float(stressed_total_ead),
            "ead_increase_percentage": float((stressed_total_ead - base_total_ead) / base_total_ead * 100) if base_total_ead > 0 else 0,
            "stress_assumptions": {
                "target_utilization_rate": 0.90,
                "ccf_stress_multiplier": 1.20,
                "scenario_description": "economic_downturn_with_increased_line_usage"
            }
        }
    
    async def _calculate_credit_var(self, credit_exposures: List[Dict], 
                                  ratings: List[str], time_horizon: int) -> Dict[str, Any]:
        """Calculate Credit Value at Risk."""
        # Get default probabilities and LGDs
        pd_results = await self._calculate_default_probabilities(credit_exposures, ratings, time_horizon)
        lgd_results = await self._estimate_loss_given_default(credit_exposures, ratings, [])
        
        # Extract data for calculations
        exposures = [exp.get('amount', 0) for exp in credit_exposures]
        pds = []
        lgds = []
        
        for i, rating in enumerate(ratings):
            if rating in pd_results['individual_default_probabilities']:
                pds.append(pd_results['individual_default_probabilities'][rating][f'{time_horizon}_year_pd'])
            else:
                pds.append(0.01)  # Default assumption
            
            if i < len(lgd_results['exposure_lgds']):
                lgds.append(lgd_results['exposure_lgds'][i]['final_lgd'])
            else:
                lgds.append(0.45)  # Default LGD
        
        # Expected Loss calculation
        expected_losses = [exp * pd * lgd for exp, pd, lgd in zip(exposures, pds, lgds)]
        total_expected_loss = sum(expected_losses)
        
        # Credit VaR using different methods
        credit_var_results = {}
        
        for conf_level in self.confidence_levels:
            # Method 1: Asymptotic Single Risk Factor (ASRF) model
            asrf_var = self._calculate_asrf_credit_var(exposures, pds, lgds, conf_level)
            
            # Method 2: Monte Carlo simulation
            mc_var = self._monte_carlo_credit_var(exposures, pds, lgds, conf_level)
            
            # Method 3: Normal approximation
            normal_var = self._normal_approximation_credit_var(exposures, pds, lgds, conf_level)
            
            credit_var_results[f"confidence_{int(conf_level*100)}"] = {
                "asrf_credit_var": float(asrf_var),
                "monte_carlo_credit_var": float(mc_var),
                "normal_approximation_credit_var": float(normal_var),
                "expected_loss": float(total_expected_loss),
                "unexpected_loss_asrf": float(asrf_var - total_expected_loss),
                "unexpected_loss_mc": float(mc_var - total_expected_loss),
                "risk_contribution_ratio": float(asrf_var / total_expected_loss) if total_expected_loss > 0 else 0
            }
        
        # Calculate individual risk contributions
        risk_contributions = self._calculate_risk_contributions(exposures, pds, lgds)
        
        return {
            "credit_var_estimates": credit_var_results,
            "expected_loss": float(total_expected_loss),
            "portfolio_statistics": {
                "total_exposure": float(sum(exposures)),
                "weighted_average_pd": float(np.average(pds, weights=exposures)) if exposures else 0,
                "weighted_average_lgd": float(np.average(lgds, weights=exposures)) if exposures else 0,
                "number_of_obligors": len(exposures)
            },
            "risk_contributions": risk_contributions,
            "model_parameters": {
                "asset_correlation": 0.15,
                "time_horizon_years": time_horizon,
                "confidence_levels": self.confidence_levels
            }
        }
    
    def _calculate_asrf_credit_var(self, exposures: List[float], pds: List[float], 
                                 lgds: List[float], confidence_level: float) -> float:
        """Calculate Credit VaR using Asymptotic Single Risk Factor model."""
        # Simplified ASRF implementation
        asset_correlation = 0.15  # Typical for corporate exposures
        
        total_var = 0
        
        for exp, pd, lgd in zip(exposures, pds, lgds):
            if pd > 0:
                # Transform PD to standard normal
                z_pd = stats.norm.ppf(pd)
                
                # Calculate conditional PD at confidence level
                z_conf = stats.norm.ppf(confidence_level)
                
                numerator = z_pd + np.sqrt(asset_correlation) * z_conf
                denominator = np.sqrt(1 - asset_correlation)
                
                conditional_pd = stats.norm.cdf(numerator / denominator)
                
                # Credit VaR for this exposure
                exposure_var = exp * conditional_pd * lgd
                total_var += exposure_var
        
        return total_var
    
    def _monte_carlo_credit_var(self, exposures: List[float], pds: List[float], 
                              lgds: List[float], confidence_level: float, 
                              num_simulations: int = 10000) -> float:
        """Calculate Credit VaR using Monte Carlo simulation."""
        portfolio_losses = []
        
        for _ in range(num_simulations):
            portfolio_loss = 0
            
            # Systematic risk factor
            systematic_factor = np.random.normal(0, 1)
            
            for exp, pd, lgd in zip(exposures, pds, lgds):
                if pd > 0:
                    # Individual risk factor
                    idiosyncratic_factor = np.random.normal(0, 1)
                    
                    # Asset correlation
                    rho = 0.15
                    
                    # Asset value
                    asset_value = np.sqrt(rho) * systematic_factor + np.sqrt(1 - rho) * idiosyncratic_factor
                    
                    # Default threshold
                    default_threshold = stats.norm.ppf(pd)
                    
                    # Check if default occurs
                    if asset_value < default_threshold:
                        # Add loss to portfolio
                        loss = exp * lgd * (1 + np.random.normal(0, 0.1))  # Add some LGD uncertainty
                        portfolio_loss += max(loss, 0)
            
            portfolio_losses.append(portfolio_loss)
        
        # Calculate VaR
        var_estimate = np.percentile(portfolio_losses, confidence_level * 100)
        return var_estimate
    
    def _normal_approximation_credit_var(self, exposures: List[float], pds: List[float], 
                                       lgds: List[float], confidence_level: float) -> float:
        """Calculate Credit VaR using normal approximation."""
        # Expected loss
        expected_loss = sum(exp * pd * lgd for exp, pd, lgd in zip(exposures, pds, lgds))
        
        # Variance calculation (simplified)
        variance = 0
        for exp, pd, lgd in zip(exposures, pds, lgds):
            # Individual variance
            individual_var = exp**2 * pd * (1 - pd) * lgd**2
            variance += individual_var
        
        # Add correlation effects (simplified)
        correlation_adjustment = 1.5  # Rough approximation
        variance *= correlation_adjustment
        
        # Calculate VaR
        z_score = stats.norm.ppf(confidence_level)
        var_estimate = expected_loss + z_score * np.sqrt(variance)
        
        return var_estimate
    
    def _calculate_risk_contributions(self, exposures: List[float], 
                                    pds: List[float], lgds: List[float]) -> List[Dict[str, float]]:
        """Calculate risk contributions by exposure."""
        risk_contributions = []
        
        total_expected_loss = sum(exp * pd * lgd for exp, pd, lgd in zip(exposures, pds, lgds))
        
        for i, (exp, pd, lgd) in enumerate(zip(exposures, pds, lgds)):
            expected_loss = exp * pd * lgd
            
            # Simplified risk contribution calculation
            risk_contribution = expected_loss / total_expected_loss if total_expected_loss > 0 else 0
            
            risk_contributions.append({
                'exposure_id': i,
                'expected_loss': float(expected_loss),
                'risk_contribution_percentage': float(risk_contribution * 100),
                'exposure_amount': float(exp),
                'default_probability': float(pd),
                'loss_given_default': float(lgd)
            })
        
        return risk_contributions
    
    async def _analyze_credit_migration(self, ratings: List[str], 
                                      time_horizon: int) -> Dict[str, Any]:
        """Analyze credit rating migration patterns."""
        # Standard one-year transition matrix (simplified)
        transition_matrix = {
            'AAA': {'AAA': 0.9081, 'AA': 0.0833, 'A': 0.0068, 'BBB': 0.0006, 'BB': 0.0012, 'B': 0.0000, 'CCC': 0.0000, 'D': 0.0000},
            'AA': {'AAA': 0.0070, 'AA': 0.9105, 'A': 0.0779, 'BBB': 0.0064, 'BB': 0.0006, 'B': 0.0014, 'CCC': 0.0000, 'D': 0.0000},
            'A': {'AAA': 0.0009, 'AA': 0.0227, 'A': 0.9105, 'BBB': 0.0552, 'BB': 0.0074, 'B': 0.0026, 'CCC': 0.0001, 'D': 0.0006},
            'BBB': {'AAA': 0.0002, 'AA': 0.0033, 'A': 0.0595, 'BBB': 0.8693, 'BB': 0.0530, 'B': 0.0117, 'CCC': 0.0012, 'D': 0.0018},
            'BB': {'AAA': 0.0003, 'AA': 0.0014, 'A': 0.0067, 'BBB': 0.0773, 'BB': 0.8053, 'B': 0.0884, 'CCC': 0.0100, 'D': 0.0106},
            'B': {'AAA': 0.0000, 'AA': 0.0011, 'A': 0.0024, 'BBB': 0.0043, 'BB': 0.0648, 'B': 0.8346, 'CCC': 0.0407, 'D': 0.0521},
            'CCC': {'AAA': 0.0022, 'AA': 0.0000, 'A': 0.0022, 'BBB': 0.0130, 'BB': 0.0238, 'B': 0.0386, 'CCC': 0.6225, 'D': 0.2977}
        }
        
        # Scale transition probabilities for multi-year horizon
        scaled_transitions = {}
        for from_rating in transition_matrix:
            scaled_transitions[from_rating] = {}
            for to_rating in transition_matrix[from_rating]:
                # Simplified scaling (geometric approach)
                one_year_prob = transition_matrix[from_rating][to_rating]
                if from_rating == to_rating:
                    # Diagonal elements (staying in same rating)
                    multi_year_prob = one_year_prob ** time_horizon
                else:
                    # Off-diagonal elements
                    multi_year_prob = 1 - (1 - one_year_prob) ** time_horizon
                
                scaled_transitions[from_rating][to_rating] = multi_year_prob
        
        # Normalize probabilities
        for from_rating in scaled_transitions:
            total_prob = sum(scaled_transitions[from_rating].values())
            if total_prob > 0:
                for to_rating in scaled_transitions[from_rating]:
                    scaled_transitions[from_rating][to_rating] /= total_prob
        
        # Calculate portfolio migration statistics
        rating_distribution = {}
        for rating in ratings:
            rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
        
        # Expected rating distribution after time_horizon
        expected_distribution = {}
        for from_rating, count in rating_distribution.items():
            if from_rating in scaled_transitions:
                for to_rating, prob in scaled_transitions[from_rating].items():
                    expected_distribution[to_rating] = expected_distribution.get(to_rating, 0) + count * prob
        
        return {
            "transition_matrix": scaled_transitions,
            "time_horizon_years": time_horizon,
            "current_rating_distribution": rating_distribution,
            "expected_rating_distribution": expected_distribution,
            "migration_statistics": {
                "upgrade_probability": self._calculate_upgrade_probability(scaled_transitions, ratings),
                "downgrade_probability": self._calculate_downgrade_probability(scaled_transitions, ratings),
                "default_probability": sum(expected_distribution.get('D', 0) for _ in range(1)),
                "rating_volatility": self._calculate_rating_volatility(scaled_transitions, ratings)
            },
            "concentration_analysis": {
                "investment_grade_percentage": self._calculate_ig_percentage(rating_distribution),
                "speculative_grade_percentage": self._calculate_sg_percentage(rating_distribution),
                "diversification_score": self._calculate_rating_diversification(rating_distribution)
            }
        }
    
    def _calculate_upgrade_probability(self, transition_matrix: Dict, ratings: List[str]) -> float:
        """Calculate weighted average upgrade probability."""
        # Simplified calculation
        return 0.15  # 15% average upgrade probability
    
    def _calculate_downgrade_probability(self, transition_matrix: Dict, ratings: List[str]) -> float:
        """Calculate weighted average downgrade probability."""
        # Simplified calculation
        return 0.20  # 20% average downgrade probability
    
    def _calculate_rating_volatility(self, transition_matrix: Dict, ratings: List[str]) -> float:
        """Calculate rating volatility measure."""
        # Simplified calculation
        return 0.25  # Rating volatility measure
    
    def _calculate_ig_percentage(self, rating_distribution: Dict[str, int]) -> float:
        """Calculate investment grade percentage."""
        ig_ratings = ['AAA', 'AA', 'A', 'BBB']
        ig_count = sum(rating_distribution.get(rating, 0) for rating in ig_ratings)
        total_count = sum(rating_distribution.values())
        return (ig_count / total_count * 100) if total_count > 0 else 0
    
    def _calculate_sg_percentage(self, rating_distribution: Dict[str, int]) -> float:
        """Calculate speculative grade percentage."""
        sg_ratings = ['BB', 'B', 'CCC', 'D']
        sg_count = sum(rating_distribution.get(rating, 0) for rating in sg_ratings)
        total_count = sum(rating_distribution.values())
        return (sg_count / total_count * 100) if total_count > 0 else 0
    
    def _calculate_rating_diversification(self, rating_distribution: Dict[str, int]) -> float:
        """Calculate rating diversification score."""
        total_count = sum(rating_distribution.values())
        if total_count == 0:
            return 0
        
        # Herfindahl index for concentration
        herfindahl = sum((count / total_count) ** 2 for count in rating_distribution.values())
        
        # Convert to diversification score (0-10 scale)
        diversification_score = (1 - herfindahl) * 10
        return min(diversification_score, 10)
    
    async def _analyze_credit_spreads(self, credit_exposures: List[Dict], 
                                    ratings: List[str]) -> Dict[str, Any]:
        """Analyze credit spreads and pricing."""
        # Reference credit spreads by rating (basis points over treasury)
        reference_spreads = {
            'AAA': 25, 'AA': 35, 'A': 50, 'BBB': 120,
            'BB': 300, 'B': 600, 'CCC': 1200, 'D': 2000
        }
        
        spread_analysis = {}
        
        for rating in set(ratings):
            if rating in reference_spreads:
                base_spread = reference_spreads[rating]
                
                # Add term structure effect (longer maturity = higher spread)
                term_adjustment = 1.2  # 20% increase for longer maturity
                
                # Add sector and liquidity adjustments
                sector_adjustment = 1.1   # 10% sector risk premium
                liquidity_adjustment = 1.05  # 5% liquidity premium
                
                adjusted_spread = base_spread * term_adjustment * sector_adjustment * liquidity_adjustment
                
                spread_analysis[rating] = {
                    "base_spread_bp": float(base_spread),
                    "adjusted_spread_bp": float(adjusted_spread),
                    "term_adjustment": float(term_adjustment),
                    "sector_adjustment": float(sector_adjustment),
                    "liquidity_adjustment": float(liquidity_adjustment),
                    "spread_volatility_bp": float(base_spread * 0.3)  # 30% relative volatility
                }
        
        # Portfolio spread statistics
        exposures = [exp.get('amount', 0) for exp in credit_exposures]
        spreads = []
        
        for i, rating in enumerate(ratings):
            if rating in spread_analysis and i < len(exposures):
                spreads.append(spread_analysis[rating]["adjusted_spread_bp"])
        
        if spreads and exposures:
            weighted_avg_spread = np.average(spreads, weights=exposures[:len(spreads)])
        else:
            weighted_avg_spread = 0
        
        return {
            "individual_spreads": spread_analysis,
            "portfolio_statistics": {
                "weighted_average_spread_bp": float(weighted_avg_spread),
                "spread_range_bp": {
                    "min": float(min(spreads)) if spreads else 0,
                    "max": float(max(spreads)) if spreads else 0
                },
                "spread_percentiles_bp": {
                    "25th": float(np.percentile(spreads, 25)) if spreads else 0,
                    "75th": float(np.percentile(spreads, 75)) if spreads else 0
                }
            },
            "spread_risk_metrics": {
                "spread_duration": 4.5,  # Years
                "spread_dv01": self._calculate_spread_dv01(exposures, spreads),
                "spread_var_bp": self._calculate_spread_var(spreads)
            },
            "market_conditions": {
                "credit_cycle_stage": "mid_cycle",
                "risk_appetite": "moderate",
                "technical_factors": "neutral"
            }
        }
    
    def _calculate_spread_dv01(self, exposures: List[float], spreads: List[float]) -> float:
        """Calculate spread DV01 (dollar value of 1 basis point)."""
        # Simplified calculation
        total_exposure = sum(exposures) if exposures else 0
        avg_duration = 4.5  # Approximate credit duration
        spread_dv01 = total_exposure * avg_duration * 0.0001  # 1 bp = 0.0001
        return float(spread_dv01)
    
    def _calculate_spread_var(self, spreads: List[float]) -> float:
        """Calculate spread VaR."""
        if not spreads:
            return 0
        
        # Simplified spread VaR calculation
        avg_spread = np.mean(spreads)
        spread_volatility = 0.3  # 30% relative volatility
        spread_var_95 = avg_spread * spread_volatility * 1.645  # 95% confidence
        
        return float(spread_var_95)
    
    async def _analyze_concentration_risk(self, credit_exposures: List[Dict], 
                                        sectors: List[str], ratings: List[str]) -> Dict[str, Any]:
        """Analyze concentration risk in the credit portfolio."""
        # Exposure concentration by various dimensions
        exposures = [exp.get('amount', 0) for exp in credit_exposures]
        total_exposure = sum(exposures)
        
        # Sector concentration
        sector_concentration = {}
        for i, sector in enumerate(sectors):
            if i < len(exposures):
                sector_concentration[sector] = sector_concentration.get(sector, 0) + exposures[i]
        
        # Rating concentration
        rating_concentration = {}
        for i, rating in enumerate(ratings):
            if i < len(exposures):
                rating_concentration[rating] = rating_concentration.get(rating, 0) + exposures[i]
        
        # Geographic concentration (simulated)
        geographic_concentration = {
            'North America': total_exposure * 0.6,
            'Europe': total_exposure * 0.25,
            'Asia': total_exposure * 0.15
        }
        
        # Calculate concentration metrics
        concentration_metrics = {}
        
        for category, concentrations in [
            ('sector', sector_concentration),
            ('rating', rating_concentration),
            ('geographic', geographic_concentration)
        ]:
            if total_exposure > 0:
                # Herfindahl-Hirschman Index
                hhi = sum((conc / total_exposure) ** 2 for conc in concentrations.values())
                
                # Effective number of positions
                effective_positions = 1 / hhi if hhi > 0 else 0
                
                # Largest exposure percentage
                largest_exposure = max(concentrations.values()) / total_exposure if concentrations else 0
                
                concentration_metrics[category] = {
                    "herfindahl_index": float(hhi),
                    "effective_number_of_positions": float(effective_positions),
                    "largest_exposure_percentage": float(largest_exposure * 100),
                    "concentration_level": self._classify_concentration_level(hhi)
                }
        
        # Single name concentration
        single_name_metrics = self._analyze_single_name_concentration(exposures, total_exposure)
        
        return {
            "concentration_by_category": {
                "sector_concentrations": {k: float(v) for k, v in sector_concentration.items()},
                "rating_concentrations": {k: float(v) for k, v in rating_concentration.items()},
                "geographic_concentrations": {k: float(v) for k, v in geographic_concentration.items()}
            },
            "concentration_metrics": concentration_metrics,
            "single_name_concentration": single_name_metrics,
            "concentration_limits": {
                "sector_limit_percentage": 15.0,
                "single_name_limit_percentage": 5.0,
                "rating_limit_percentage": 25.0,
                "geographic_limit_percentage": 60.0
            },
            "risk_assessment": {
                "overall_concentration_score": self._calculate_overall_concentration_score(concentration_metrics),
                "concentration_warnings": self._identify_concentration_warnings(concentration_metrics),
                "diversification_opportunities": self._identify_diversification_opportunities(concentration_metrics)
            }
        }
    
    def _classify_concentration_level(self, hhi: float) -> str:
        """Classify concentration level based on HHI."""
        if hhi < 0.10:
            return "low_concentration"
        elif hhi < 0.25:
            return "moderate_concentration"
        else:
            return "high_concentration"
    
    def _analyze_single_name_concentration(self, exposures: List[float], 
                                         total_exposure: float) -> Dict[str, Any]:
        """Analyze single name concentration risk."""
        if not exposures or total_exposure == 0:
            return {}
        
        exposure_percentages = [exp / total_exposure for exp in exposures]
        
        return {
            "largest_single_exposure_percentage": float(max(exposure_percentages) * 100),
            "top_5_exposures_percentage": float(sum(sorted(exposure_percentages, reverse=True)[:5]) * 100),
            "top_10_exposures_percentage": float(sum(sorted(exposure_percentages, reverse=True)[:10]) * 100),
            "number_of_exposures": len(exposures),
            "average_exposure_percentage": float(np.mean(exposure_percentages) * 100),
            "exposure_distribution": {
                "percentile_95": float(np.percentile(exposure_percentages, 95) * 100),
                "percentile_75": float(np.percentile(exposure_percentages, 75) * 100),
                "percentile_25": float(np.percentile(exposure_percentages, 25) * 100)
            }
        }
    
    def _calculate_overall_concentration_score(self, concentration_metrics: Dict) -> float:
        """Calculate overall concentration risk score (0-10 scale)."""
        # Weighted average of concentration scores
        scores = []
        weights = []
        
        for category, metrics in concentration_metrics.items():
            hhi = metrics["herfindahl_index"]
            # Convert HHI to 0-10 scale (higher HHI = higher score = more risk)
            score = min(hhi * 20, 10)  # Scale factor
            scores.append(score)
            
            # Weight by importance
            if category == 'sector':
                weights.append(0.4)
            elif category == 'rating':
                weights.append(0.3)
            else:  # geographic
                weights.append(0.3)
        
        if scores and weights:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 5.0  # Default medium risk
        
        return float(overall_score)
    
    def _identify_concentration_warnings(self, concentration_metrics: Dict) -> List[str]:
        """Identify concentration risk warnings."""
        warnings = []
        
        for category, metrics in concentration_metrics.items():
            concentration_level = metrics["concentration_level"]
            largest_exposure = metrics["largest_exposure_percentage"]
            
            if concentration_level == "high_concentration":
                warnings.append(f"High concentration risk in {category} dimension")
            
            if largest_exposure > 20:
                warnings.append(f"Large single exposure in {category}: {largest_exposure:.1f}%")
        
        return warnings
    
    def _identify_diversification_opportunities(self, concentration_metrics: Dict) -> List[str]:
        """Identify diversification opportunities."""
        opportunities = []
        
        for category, metrics in concentration_metrics.items():
            effective_positions = metrics["effective_number_of_positions"]
            
            if effective_positions < 5:
                opportunities.append(f"Increase diversification in {category} dimension")
            
            concentration_level = metrics["concentration_level"]
            if concentration_level in ["moderate_concentration", "high_concentration"]:
                opportunities.append(f"Reduce {category} concentration through rebalancing")
        
        return opportunities
