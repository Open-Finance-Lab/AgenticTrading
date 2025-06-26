"""
Liquidity Risk Analyzer - Liquidity Risk Analysis Agent

This agent specializes in liquidity risk analysis including:
- Market liquidity assessment
- Funding liquidity analysis
- Liquidity ratios calculation
- Stress testing scenarios
- Bid-ask spread analysis

Author: Jifeng Li
License: openMDW
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from scipy import stats

from ..registry import BaseRiskAgent


class LiquidityRiskAnalyzer(BaseRiskAgent):
    """
    Specialized agent for comprehensive liquidity risk analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.name = "LiquidityRiskAnalyzer"
        self.logger = logging.getLogger(f"RiskAgent.{self.name}")
        
        # Configuration parameters
        self.stress_scenarios = config.get('stress_scenarios', ['mild', 'moderate', 'severe']) if config else ['mild', 'moderate', 'severe']
        self.liquidity_horizons = config.get('liquidity_horizons', [1, 7, 30, 90]) if config else [1, 7, 30, 90]  # Days
    
    async def analyze(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive liquidity risk analysis.
        
        Args:
            request: Analysis request containing portfolio and funding data
            
        Returns:
            Dictionary containing liquidity risk analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            portfolio_data = request.get("portfolio_data", {})
            funding_data = request.get("funding_data", {})
            analysis_type = request.get("analysis_type", "comprehensive")
            
            results = {}
            
            if analysis_type in ["comprehensive", "market_liquidity"]:
                results["market_liquidity"] = await self._analyze_market_liquidity(portfolio_data)
            
            if analysis_type in ["comprehensive", "funding_liquidity"]:
                results["funding_liquidity"] = await self._analyze_funding_liquidity(funding_data)
            
            if analysis_type in ["comprehensive", "liquidity_ratios"]:
                results["liquidity_ratios"] = await self._calculate_liquidity_ratios(
                    portfolio_data, funding_data
                )
            
            if analysis_type in ["comprehensive", "stress_testing"]:
                results["stress_testing"] = await self._perform_liquidity_stress_testing(
                    portfolio_data, funding_data
                )
            
            if analysis_type in ["comprehensive", "cash_flow"]:
                results["cash_flow_analysis"] = await self._analyze_cash_flows(
                    portfolio_data, funding_data
                )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "agent": self.name,
                "analysis_type": "liquidity_risk",
                "results": results,
                "execution_time_ms": execution_time,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Liquidity risk analysis failed: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "agent": self.name,
                "error": str(e),
                "execution_time_ms": execution_time,
                "status": "error",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _analyze_market_liquidity(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market liquidity of portfolio assets."""
        assets = portfolio_data.get("securities", [])
        weights = portfolio_data.get("weights", [])
        
        # Asset liquidity classification
        liquidity_classes = {
            'cash': {'liquidity_score': 10, 'liquidation_time_days': 0, 'bid_ask_spread': 0.0001},
            'government_bonds': {'liquidity_score': 9, 'liquidation_time_days': 1, 'bid_ask_spread': 0.0005},
            'corporate_bonds_ig': {'liquidity_score': 7, 'liquidation_time_days': 2, 'bid_ask_spread': 0.002},
            'corporate_bonds_hy': {'liquidity_score': 5, 'liquidation_time_days': 5, 'bid_ask_spread': 0.005},
            'large_cap_stocks': {'liquidity_score': 8, 'liquidation_time_days': 1, 'bid_ask_spread': 0.001},
            'small_cap_stocks': {'liquidity_score': 6, 'liquidation_time_days': 3, 'bid_ask_spread': 0.003},
            'emerging_markets': {'liquidity_score': 4, 'liquidation_time_days': 7, 'bid_ask_spread': 0.008},
            'alternatives': {'liquidity_score': 2, 'liquidation_time_days': 30, 'bid_ask_spread': 0.02},
            'private_equity': {'liquidity_score': 1, 'liquidation_time_days': 365, 'bid_ask_spread': 0.05}
        }
        
        # Simulate asset classifications
        asset_liquidity = {}
        for i, asset in enumerate(assets):
            # Assign random liquidity class for demonstration
            asset_type = np.random.choice(list(liquidity_classes.keys()))
            asset_weight = weights[i] if i < len(weights) else 0
            
            liquidity_info = liquidity_classes[asset_type].copy()
            liquidity_info['asset_weight'] = asset_weight
            liquidity_info['asset_type'] = asset_type
            
            # Add market impact
            liquidity_info['market_impact'] = self._calculate_market_impact(
                asset_weight, liquidity_info['liquidity_score']
            )
            
            asset_liquidity[asset] = liquidity_info
        
        # Portfolio-level liquidity metrics
        portfolio_liquidity = self._calculate_portfolio_liquidity_metrics(asset_liquidity)
        
        # Liquidity concentration analysis
        concentration_analysis = self._analyze_liquidity_concentration(asset_liquidity)
        
        # Time to liquidation analysis
        liquidation_analysis = await self._analyze_liquidation_timeframes(asset_liquidity)
        
        return {
            "asset_liquidity": asset_liquidity,
            "portfolio_liquidity_metrics": portfolio_liquidity,
            "concentration_analysis": concentration_analysis,
            "liquidation_analysis": liquidation_analysis,
            "liquidity_risk_assessment": {
                "overall_liquidity_score": portfolio_liquidity.get("weighted_liquidity_score", 5),
                "liquidity_risk_level": self._classify_liquidity_risk_level(
                    portfolio_liquidity.get("weighted_liquidity_score", 5)
                ),
                "key_risks": self._identify_key_liquidity_risks(asset_liquidity),
                "recommendations": self._generate_liquidity_recommendations(asset_liquidity)
            }
        }
    
    def _calculate_market_impact(self, asset_weight: float, liquidity_score: int) -> Dict[str, float]:
        """Calculate market impact for asset liquidation."""
        # Market impact increases with position size and decreases with liquidity
        base_impact = 0.001  # 10 bps base impact
        
        # Size effect
        size_multiplier = 1 + (asset_weight * 10)  # Linear increase with weight
        
        # Liquidity effect
        liquidity_discount = liquidity_score / 10  # Higher score = lower impact
        
        market_impact = base_impact * size_multiplier / liquidity_discount
        
        return {
            "temporary_impact": float(market_impact * 0.6),  # 60% temporary
            "permanent_impact": float(market_impact * 0.4),  # 40% permanent
            "total_impact": float(market_impact)
        }
    
    def _calculate_portfolio_liquidity_metrics(self, asset_liquidity: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate portfolio-level liquidity metrics."""
        if not asset_liquidity:
            return {}
        
        total_weight = sum(info['asset_weight'] for info in asset_liquidity.values())
        
        if total_weight == 0:
            return {}
        
        # Weighted average liquidity score
        weighted_liquidity_score = sum(
            info['liquidity_score'] * info['asset_weight'] 
            for info in asset_liquidity.values()
        ) / total_weight
        
        # Weighted average liquidation time
        weighted_liquidation_time = sum(
            info['liquidation_time_days'] * info['asset_weight'] 
            for info in asset_liquidity.values()
        ) / total_weight
        
        # Weighted average bid-ask spread
        weighted_bid_ask_spread = sum(
            info['bid_ask_spread'] * info['asset_weight'] 
            for info in asset_liquidity.values()
        ) / total_weight
        
        # Portfolio turnover capacity (daily)
        daily_turnover_capacity = sum(
            info['asset_weight'] / max(info['liquidation_time_days'], 1)
            for info in asset_liquidity.values()
        )
        
        return {
            "weighted_liquidity_score": float(weighted_liquidity_score),
            "weighted_liquidation_time_days": float(weighted_liquidation_time),
            "weighted_bid_ask_spread": float(weighted_bid_ask_spread),
            "daily_turnover_capacity": float(daily_turnover_capacity),
            "illiquid_assets_percentage": float(self._calculate_illiquid_percentage(asset_liquidity))
        }
    
    def _calculate_illiquid_percentage(self, asset_liquidity: Dict[str, Dict]) -> float:
        """Calculate percentage of illiquid assets (liquidation time > 5 days)."""
        total_weight = sum(info['asset_weight'] for info in asset_liquidity.values())
        
        if total_weight == 0:
            return 0
        
        illiquid_weight = sum(
            info['asset_weight'] for info in asset_liquidity.values()
            if info['liquidation_time_days'] > 5
        )
        
        return (illiquid_weight / total_weight) * 100
    
    def _analyze_liquidity_concentration(self, asset_liquidity: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze liquidity concentration risk."""
        # Group by liquidity buckets
        liquidity_buckets = {
            'highly_liquid': [],      # Score 8-10
            'moderately_liquid': [],  # Score 5-7
            'illiquid': []           # Score 1-4
        }
        
        for asset, info in asset_liquidity.items():
            score = info['liquidity_score']
            weight = info['asset_weight']
            
            if score >= 8:
                liquidity_buckets['highly_liquid'].append(weight)
            elif score >= 5:
                liquidity_buckets['moderately_liquid'].append(weight)
            else:
                liquidity_buckets['illiquid'].append(weight)
        
        # Calculate percentages
        total_weight = sum(sum(bucket) for bucket in liquidity_buckets.values())
        
        bucket_percentages = {}
        for bucket_name, weights in liquidity_buckets.items():
            bucket_percentages[bucket_name] = (sum(weights) / total_weight * 100) if total_weight > 0 else 0
        
        # Concentration metrics
        concentration_hhi = sum((pct / 100) ** 2 for pct in bucket_percentages.values())
        
        return {
            "liquidity_bucket_percentages": bucket_percentages,
            "concentration_hhi": float(concentration_hhi),
            "concentration_level": "high" if concentration_hhi > 0.5 else "moderate" if concentration_hhi > 0.33 else "low",
            "diversification_score": float((1 - concentration_hhi) * 10)
        }
    
    async def _analyze_liquidation_timeframes(self, asset_liquidity: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze portfolio liquidation timeframes under different scenarios."""
        scenarios = {
            'normal_market': {'stress_multiplier': 1.0, 'market_impact_multiplier': 1.0},
            'stressed_market': {'stress_multiplier': 3.0, 'market_impact_multiplier': 2.0},
            'crisis_market': {'stress_multiplier': 10.0, 'market_impact_multiplier': 5.0}
        }
        
        liquidation_analysis = {}
        
        for scenario_name, scenario_params in scenarios.items():
            stress_mult = scenario_params['stress_multiplier']
            impact_mult = scenario_params['market_impact_multiplier']
            
            # Calculate liquidation capacity for different time horizons
            liquidation_capacity = {}
            
            for horizon in self.liquidity_horizons:
                capacity = 0
                total_impact = 0
                
                for asset, info in asset_liquidity.items():
                    liquidation_time = info['liquidation_time_days'] * stress_mult
                    
                    if liquidation_time <= horizon:
                        # Can liquidate within horizon
                        capacity += info['asset_weight']
                        
                        # Add market impact
                        base_impact = info['market_impact']['total_impact']
                        stressed_impact = base_impact * impact_mult
                        total_impact += stressed_impact * info['asset_weight']
                
                liquidation_capacity[f"{horizon}_days"] = {
                    "liquidation_capacity_percentage": float(capacity * 100),
                    "average_market_impact": float(total_impact / capacity) if capacity > 0 else 0
                }
            
            liquidation_analysis[scenario_name] = liquidation_capacity
        
        return liquidation_analysis
    
    def _classify_liquidity_risk_level(self, liquidity_score: float) -> str:
        """Classify overall liquidity risk level."""
        if liquidity_score >= 8:
            return "low_risk"
        elif liquidity_score >= 6:
            return "moderate_risk"
        elif liquidity_score >= 4:
            return "high_risk"
        else:
            return "very_high_risk"
    
    def _identify_key_liquidity_risks(self, asset_liquidity: Dict[str, Dict]) -> List[str]:
        """Identify key liquidity risks in the portfolio."""
        risks = []
        
        # Check for concentration in illiquid assets
        illiquid_percentage = self._calculate_illiquid_percentage(asset_liquidity)
        if illiquid_percentage > 30:
            risks.append(f"High concentration in illiquid assets: {illiquid_percentage:.1f}%")
        
        # Check for large positions in individual illiquid assets
        for asset, info in asset_liquidity.items():
            if info['liquidation_time_days'] > 10 and info['asset_weight'] > 0.1:
                risks.append(f"Large position in illiquid asset {asset}: {info['asset_weight']*100:.1f}%")
        
        # Check for high market impact
        high_impact_assets = [
            asset for asset, info in asset_liquidity.items()
            if info['market_impact']['total_impact'] > 0.01  # > 100 bps
        ]
        
        if high_impact_assets:
            risks.append(f"High market impact assets: {len(high_impact_assets)} positions")
        
        return risks
    
    def _generate_liquidity_recommendations(self, asset_liquidity: Dict[str, Dict]) -> List[str]:
        """Generate liquidity management recommendations."""
        recommendations = []
        
        # Analyze current liquidity profile
        illiquid_percentage = self._calculate_illiquid_percentage(asset_liquidity)
        
        if illiquid_percentage > 25:
            recommendations.append("Consider reducing allocation to illiquid assets")
        
        # Check for liquidity concentration
        liquidity_scores = [info['liquidity_score'] for info in asset_liquidity.values()]
        if len(set(liquidity_scores)) < 3:
            recommendations.append("Diversify across different liquidity buckets")
        
        # Check for cash buffer
        cash_allocation = sum(
            info['asset_weight'] for info in asset_liquidity.values()
            if info.get('asset_type') == 'cash'
        )
        
        if cash_allocation < 0.05:
            recommendations.append("Consider maintaining a minimum 5% cash buffer")
        
        # Large position recommendations
        large_positions = [
            (asset, info) for asset, info in asset_liquidity.items()
            if info['asset_weight'] > 0.15
        ]
        
        if large_positions:
            recommendations.append("Consider reducing size of large concentrated positions")
        
        return recommendations
    
    async def _analyze_funding_liquidity(self, funding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze funding liquidity and financing risks."""
        funding_sources = funding_data.get("funding_sources", [])
        funding_costs = funding_data.get("funding_costs", [])
        maturities = funding_data.get("maturities", [])
        
        # Funding diversity analysis
        funding_diversity = self._analyze_funding_diversity(funding_sources)
        
        # Maturity profile analysis
        maturity_analysis = self._analyze_maturity_profile(maturities, funding_sources)
        
        # Funding cost analysis
        cost_analysis = self._analyze_funding_costs(funding_costs, funding_sources)
        
        # Rollover risk assessment
        rollover_risk = self._assess_rollover_risk(maturities, funding_sources)
        
        # Contingency funding analysis
        contingency_funding = self._analyze_contingency_funding(funding_data)
        
        return {
            "funding_diversity": funding_diversity,
            "maturity_profile": maturity_analysis,
            "funding_cost_analysis": cost_analysis,
            "rollover_risk": rollover_risk,
            "contingency_funding": contingency_funding,
            "funding_risk_assessment": {
                "overall_funding_risk": self._calculate_overall_funding_risk(
                    funding_diversity, maturity_analysis, rollover_risk
                ),
                "key_vulnerabilities": self._identify_funding_vulnerabilities(
                    funding_sources, maturities
                ),
                "funding_recommendations": self._generate_funding_recommendations(
                    funding_diversity, maturity_analysis
                )
            }
        }
    
    def _analyze_funding_diversity(self, funding_sources: List[Dict]) -> Dict[str, Any]:
        """Analyze diversity of funding sources."""
        if not funding_sources:
            return {"error": "No funding sources provided"}
        
        # Categorize funding sources
        source_categories = {}
        total_funding = 0
        
        for source in funding_sources:
            category = source.get('category', 'other')
            amount = source.get('amount', 0)
            
            source_categories[category] = source_categories.get(category, 0) + amount
            total_funding += amount
        
        # Calculate percentages
        source_percentages = {}
        if total_funding > 0:
            for category, amount in source_categories.items():
                source_percentages[category] = (amount / total_funding) * 100
        
        # Calculate concentration metrics
        concentration_hhi = sum((pct / 100) ** 2 for pct in source_percentages.values())
        effective_sources = 1 / concentration_hhi if concentration_hhi > 0 else 0
        
        return {
            "funding_source_percentages": source_percentages,
            "concentration_hhi": float(concentration_hhi),
            "effective_number_of_sources": float(effective_sources),
            "diversity_score": float((1 - concentration_hhi) * 10),
            "largest_source_percentage": float(max(source_percentages.values())) if source_percentages else 0
        }
    
    def _analyze_maturity_profile(self, maturities: List[int], 
                                funding_sources: List[Dict]) -> Dict[str, Any]:
        """Analyze funding maturity profile."""
        if not maturities or not funding_sources:
            return {"error": "Insufficient maturity data"}
        
        # Maturity buckets (days)
        maturity_buckets = {
            'overnight': (0, 1),
            '1_week': (2, 7),
            '1_month': (8, 30),
            '3_months': (31, 90),
            '6_months': (91, 180),
            '1_year': (181, 365),
            'over_1_year': (366, float('inf'))
        }
        
        bucket_amounts = {bucket: 0 for bucket in maturity_buckets}
        total_funding = sum(source.get('amount', 0) for source in funding_sources)
        
        for i, maturity in enumerate(maturities):
            if i < len(funding_sources):
                amount = funding_sources[i].get('amount', 0)
                
                for bucket_name, (min_days, max_days) in maturity_buckets.items():
                    if min_days <= maturity <= max_days:
                        bucket_amounts[bucket_name] += amount
                        break
        
        # Calculate percentages
        bucket_percentages = {}
        if total_funding > 0:
            for bucket, amount in bucket_amounts.items():
                bucket_percentages[bucket] = (amount / total_funding) * 100
        
        # Calculate weighted average maturity
        if maturities and funding_sources:
            amounts = [source.get('amount', 0) for source in funding_sources]
            weighted_avg_maturity = np.average(maturities, weights=amounts) if sum(amounts) > 0 else 0
        else:
            weighted_avg_maturity = 0
        
        return {
            "maturity_bucket_percentages": bucket_percentages,
            "weighted_average_maturity_days": float(weighted_avg_maturity),
            "short_term_funding_percentage": float(
                bucket_percentages.get('overnight', 0) + 
                bucket_percentages.get('1_week', 0) + 
                bucket_percentages.get('1_month', 0)
            ),
            "maturity_concentration": self._calculate_maturity_concentration(bucket_percentages)
        }
    
    def _calculate_maturity_concentration(self, bucket_percentages: Dict[str, float]) -> Dict[str, float]:
        """Calculate maturity concentration metrics."""
        # HHI for maturity concentration
        hhi = sum((pct / 100) ** 2 for pct in bucket_percentages.values())
        
        return {
            "concentration_hhi": float(hhi),
            "concentration_level": "high" if hhi > 0.5 else "moderate" if hhi > 0.33 else "low"
        }
    
    def _analyze_funding_costs(self, funding_costs: List[float], 
                             funding_sources: List[Dict]) -> Dict[str, Any]:
        """Analyze funding cost structure."""
        if not funding_costs or not funding_sources:
            return {"error": "Insufficient funding cost data"}
        
        amounts = [source.get('amount', 0) for source in funding_sources]
        
        # Weighted average cost
        if amounts and funding_costs:
            weighted_avg_cost = np.average(funding_costs, weights=amounts) if sum(amounts) > 0 else 0
        else:
            weighted_avg_cost = 0
        
        # Cost statistics
        cost_stats = {
            "weighted_average_cost": float(weighted_avg_cost),
            "min_cost": float(min(funding_costs)) if funding_costs else 0,
            "max_cost": float(max(funding_costs)) if funding_costs else 0,
            "cost_volatility": float(np.std(funding_costs)) if funding_costs else 0,
            "cost_spread": float(max(funding_costs) - min(funding_costs)) if funding_costs else 0
        }
        
        # Cost sensitivity analysis
        cost_sensitivity = self._analyze_cost_sensitivity(funding_costs, amounts)
        
        return {
            "cost_statistics": cost_stats,
            "cost_sensitivity": cost_sensitivity,
            "expensive_funding_percentage": self._calculate_expensive_funding_percentage(
                funding_costs, amounts, weighted_avg_cost
            )
        }
    
    def _analyze_cost_sensitivity(self, funding_costs: List[float], 
                                amounts: List[float]) -> Dict[str, float]:
        """Analyze sensitivity to funding cost changes."""
        if not funding_costs or not amounts:
            return {}
        
        total_funding = sum(amounts)
        weighted_avg_cost = np.average(funding_costs, weights=amounts) if total_funding > 0 else 0
        
        # Sensitivity to 100bp increase
        shocked_costs = [cost + 0.01 for cost in funding_costs]  # +100bp
        shocked_avg_cost = np.average(shocked_costs, weights=amounts) if total_funding > 0 else 0
        
        cost_sensitivity_100bp = shocked_avg_cost - weighted_avg_cost
        
        return {
            "cost_sensitivity_100bp": float(cost_sensitivity_100bp),
            "relative_sensitivity": float(cost_sensitivity_100bp / weighted_avg_cost) if weighted_avg_cost > 0 else 0
        }
    
    def _calculate_expensive_funding_percentage(self, funding_costs: List[float], 
                                              amounts: List[float], 
                                              avg_cost: float) -> float:
        """Calculate percentage of expensive funding (above average + 2 std dev)."""
        if not funding_costs or avg_cost == 0:
            return 0
        
        cost_std = np.std(funding_costs)
        expensive_threshold = avg_cost + 2 * cost_std
        
        expensive_amount = sum(
            amounts[i] for i, cost in enumerate(funding_costs)
            if i < len(amounts) and cost > expensive_threshold
        )
        
        total_amount = sum(amounts)
        return (expensive_amount / total_amount * 100) if total_amount > 0 else 0
    
    def _assess_rollover_risk(self, maturities: List[int], 
                            funding_sources: List[Dict]) -> Dict[str, Any]:
        """Assess rollover risk for maturing funding."""
        if not maturities or not funding_sources:
            return {"error": "Insufficient data for rollover risk assessment"}
        
        rollover_risk_periods = [30, 90, 180, 365]  # Days
        rollover_analysis = {}
        
        total_funding = sum(source.get('amount', 0) for source in funding_sources)
        
        for period in rollover_risk_periods:
            maturing_amount = sum(
                funding_sources[i].get('amount', 0)
                for i, maturity in enumerate(maturities)
                if i < len(funding_sources) and maturity <= period
            )
            
            rollover_percentage = (maturing_amount / total_funding * 100) if total_funding > 0 else 0
            
            # Risk classification
            if rollover_percentage > 50:
                risk_level = "high"
            elif rollover_percentage > 25:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            rollover_analysis[f"{period}_days"] = {
                "maturing_amount": float(maturing_amount),
                "rollover_percentage": float(rollover_percentage),
                "risk_level": risk_level
            }
        
        # Peak rollover period
        peak_period = max(rollover_analysis.items(), 
                         key=lambda x: x[1]["rollover_percentage"])
        
        return {
            "rollover_by_period": rollover_analysis,
            "peak_rollover_period": peak_period[0],
            "peak_rollover_percentage": peak_period[1]["rollover_percentage"],
            "overall_rollover_risk": self._classify_overall_rollover_risk(rollover_analysis)
        }
    
    def _classify_overall_rollover_risk(self, rollover_analysis: Dict) -> str:
        """Classify overall rollover risk level."""
        # Look at near-term rollover risk (30 and 90 days)
        near_term_risk = rollover_analysis.get("30_days", {}).get("rollover_percentage", 0)
        medium_term_risk = rollover_analysis.get("90_days", {}).get("rollover_percentage", 0)
        
        if near_term_risk > 30 or medium_term_risk > 50:
            return "high"
        elif near_term_risk > 15 or medium_term_risk > 30:
            return "moderate"
        else:
            return "low"
    
    def _analyze_contingency_funding(self, funding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contingency funding capacity."""
        # Simulate contingency funding sources
        contingency_sources = {
            'committed_credit_lines': 100000000,  # $100M
            'repo_capacity': 50000000,            # $50M
            'asset_sales_capacity': 200000000,    # $200M
            'central_bank_facilities': 75000000   # $75M
        }
        
        total_contingency = sum(contingency_sources.values())
        current_usage = funding_data.get('contingency_usage', 0)
        available_contingency = total_contingency - current_usage
        
        # Stress testing contingency adequacy
        stress_scenarios = {
            'mild_stress': 0.25,     # 25% of contingency needed
            'moderate_stress': 0.50, # 50% of contingency needed
            'severe_stress': 0.75    # 75% of contingency needed
        }
        
        scenario_analysis = {}
        for scenario, usage_rate in stress_scenarios.items():
            required_funding = total_contingency * usage_rate
            adequacy = "adequate" if available_contingency >= required_funding else "inadequate"
            
            scenario_analysis[scenario] = {
                "required_funding": float(required_funding),
                "available_funding": float(available_contingency),
                "adequacy": adequacy,
                "coverage_ratio": float(available_contingency / required_funding) if required_funding > 0 else float('inf')
            }
        
        return {
            "contingency_sources": contingency_sources,
            "total_contingency_capacity": float(total_contingency),
            "available_contingency": float(available_contingency),
            "current_usage_percentage": float(current_usage / total_contingency * 100) if total_contingency > 0 else 0,
            "stress_scenario_analysis": scenario_analysis,
            "contingency_risk_assessment": {
                "adequacy_level": self._assess_contingency_adequacy(scenario_analysis),
                "key_dependencies": ["committed_credit_lines", "asset_sales_capacity"],
                "potential_constraints": self._identify_contingency_constraints()
            }
        }
    
    def _assess_contingency_adequacy(self, scenario_analysis: Dict) -> str:
        """Assess overall contingency funding adequacy."""
        # Check coverage in severe stress scenario
        severe_coverage = scenario_analysis.get("severe_stress", {}).get("coverage_ratio", 0)
        
        if severe_coverage >= 1.5:
            return "strong"
        elif severe_coverage >= 1.0:
            return "adequate"
        elif severe_coverage >= 0.75:
            return "weak"
        else:
            return "inadequate"
    
    def _identify_contingency_constraints(self) -> List[str]:
        """Identify potential constraints on contingency funding."""
        return [
            "Market conditions may limit asset sales capacity",
            "Credit line covenants may restrict access",
            "Regulatory changes may affect central bank facilities",
            "Collateral requirements may increase under stress"
        ]
    
    def _calculate_overall_funding_risk(self, funding_diversity: Dict, 
                                      maturity_analysis: Dict, 
                                      rollover_risk: Dict) -> str:
        """Calculate overall funding risk level."""
        risk_factors = 0
        
        # Diversity risk
        if funding_diversity.get("diversity_score", 5) < 5:
            risk_factors += 1
        
        # Maturity risk
        if maturity_analysis.get("short_term_funding_percentage", 0) > 40:
            risk_factors += 1
        
        # Rollover risk
        if rollover_risk.get("overall_rollover_risk") == "high":
            risk_factors += 2
        elif rollover_risk.get("overall_rollover_risk") == "moderate":
            risk_factors += 1
        
        # Classify overall risk
        if risk_factors >= 3:
            return "high"
        elif risk_factors >= 2:
            return "moderate"
        else:
            return "low"
    
    def _identify_funding_vulnerabilities(self, funding_sources: List[Dict], 
                                        maturities: List[int]) -> List[str]:
        """Identify key funding vulnerabilities."""
        vulnerabilities = []
        
        # Check for large single sources
        if funding_sources:
            amounts = [source.get('amount', 0) for source in funding_sources]
            total_funding = sum(amounts)
            
            if total_funding > 0:
                max_source_pct = max(amounts) / total_funding * 100
                if max_source_pct > 25:
                    vulnerabilities.append(f"Large single funding source: {max_source_pct:.1f}%")
        
        # Check for short-term concentration
        if maturities:
            short_term_count = sum(1 for m in maturities if m <= 30)
            if short_term_count / len(maturities) > 0.4:
                vulnerabilities.append("High concentration in short-term funding")
        
        # Add other vulnerabilities
        vulnerabilities.extend([
            "Potential covenant violations under stress",
            "Limited geographic diversity of funding sources",
            "Concentration in specific funding markets"
        ])
        
        return vulnerabilities
    
    def _generate_funding_recommendations(self, funding_diversity: Dict, 
                                        maturity_analysis: Dict) -> List[str]:
        """Generate funding management recommendations."""
        recommendations = []
        
        # Diversity recommendations
        if funding_diversity.get("diversity_score", 5) < 6:
            recommendations.append("Diversify funding sources across different markets and counterparties")
        
        # Maturity recommendations
        if maturity_analysis.get("short_term_funding_percentage", 0) > 35:
            recommendations.append("Extend average maturity profile to reduce rollover risk")
        
        # General recommendations
        recommendations.extend([
            "Maintain adequate contingency funding capacity",
            "Monitor funding market conditions and early warning indicators",
            "Establish and test crisis funding plans",
            "Consider liability management strategies to optimize funding profile"
        ])
        
        return recommendations
    
    async def _calculate_liquidity_ratios(self, portfolio_data: Dict[str, Any], 
                                        funding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate standard liquidity ratios."""
        # Simulate balance sheet data
        cash_and_equivalents = 50000000  # $50M
        liquid_assets = 150000000        # $150M
        total_assets = 1000000000        # $1B
        short_term_liabilities = 200000000  # $200M
        demand_deposits = 300000000      # $300M
        total_liabilities = 800000000    # $800M
        
        # Standard liquidity ratios
        ratios = {
            "liquidity_coverage_ratio": {
                "value": float((cash_and_equivalents + liquid_assets * 0.85) / (demand_deposits * 0.25)),
                "minimum_requirement": 1.0,
                "current_status": "adequate"
            },
            "net_stable_funding_ratio": {
                "value": float(600000000 / 550000000),  # Stable funding / Required stable funding
                "minimum_requirement": 1.0,
                "current_status": "adequate"
            },
            "cash_ratio": {
                "value": float(cash_and_equivalents / short_term_liabilities),
                "benchmark": 0.20,
                "current_status": "adequate" if cash_and_equivalents / short_term_liabilities >= 0.20 else "below_benchmark"
            },
            "quick_ratio": {
                "value": float((cash_and_equivalents + liquid_assets) / short_term_liabilities),
                "benchmark": 1.0,
                "current_status": "adequate" if (cash_and_equivalents + liquid_assets) / short_term_liabilities >= 1.0 else "below_benchmark"
            },
            "loan_to_deposit_ratio": {
                "value": float(500000000 / demand_deposits),  # Loans / Deposits
                "benchmark": 0.80,
                "current_status": "adequate"
            }
        }
        
        # Calculate trend analysis (simulated)
        trend_analysis = self._calculate_ratio_trends(ratios)
        
        # Peer comparison (simulated)
        peer_comparison = self._generate_peer_comparison(ratios)
        
        return {
            "liquidity_ratios": ratios,
            "trend_analysis": trend_analysis,
            "peer_comparison": peer_comparison,
            "ratio_assessment": {
                "overall_liquidity_position": self._assess_overall_liquidity_position(ratios),
                "key_strengths": self._identify_liquidity_strengths(ratios),
                "areas_for_improvement": self._identify_liquidity_improvement_areas(ratios)
            }
        }
    
    def _calculate_ratio_trends(self, ratios: Dict[str, Dict]) -> Dict[str, str]:
        """Calculate trends in liquidity ratios (simulated)."""
        # Simulate trend data
        trends = {}
        for ratio_name in ratios.keys():
            # Random trend for demonstration
            trend_direction = np.random.choice(["improving", "stable", "deteriorating"], 
                                             p=[0.3, 0.4, 0.3])
            trends[ratio_name] = trend_direction
        
        return trends
    
    def _generate_peer_comparison(self, ratios: Dict[str, Dict]) -> Dict[str, str]:
        """Generate peer comparison for liquidity ratios."""
        # Simulate peer comparison
        comparisons = {}
        for ratio_name in ratios.keys():
            # Random comparison for demonstration
            comparison = np.random.choice(["above_peer_median", "at_peer_median", "below_peer_median"], 
                                        p=[0.4, 0.3, 0.3])
            comparisons[ratio_name] = comparison
        
        return comparisons
    
    def _assess_overall_liquidity_position(self, ratios: Dict[str, Dict]) -> str:
        """Assess overall liquidity position."""
        adequate_ratios = sum(
            1 for ratio_data in ratios.values()
            if ratio_data.get("current_status") == "adequate"
        )
        
        total_ratios = len(ratios)
        adequacy_rate = adequate_ratios / total_ratios
        
        if adequacy_rate >= 0.8:
            return "strong"
        elif adequacy_rate >= 0.6:
            return "adequate"
        else:
            return "weak"
    
    def _identify_liquidity_strengths(self, ratios: Dict[str, Dict]) -> List[str]:
        """Identify liquidity position strengths."""
        strengths = []
        
        for ratio_name, ratio_data in ratios.items():
            if ratio_data.get("current_status") == "adequate":
                strengths.append(f"Strong {ratio_name.replace('_', ' ')}")
        
        return strengths
    
    def _identify_liquidity_improvement_areas(self, ratios: Dict[str, Dict]) -> List[str]:
        """Identify areas for liquidity improvement."""
        improvements = []
        
        for ratio_name, ratio_data in ratios.items():
            if ratio_data.get("current_status") == "below_benchmark":
                improvements.append(f"Improve {ratio_name.replace('_', ' ')}")
        
        return improvements
    
    async def _perform_liquidity_stress_testing(self, portfolio_data: Dict[str, Any], 
                                               funding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive liquidity stress testing."""
        stress_scenarios = {
            "idiosyncratic_stress": {
                "description": "Institution-specific stress affecting funding access",
                "funding_loss_percentage": 25,
                "asset_liquidity_reduction": 30,
                "market_impact_multiplier": 2.0
            },
            "market_wide_stress": {
                "description": "Broad market stress affecting all institutions",
                "funding_loss_percentage": 15,
                "asset_liquidity_reduction": 50,
                "market_impact_multiplier": 3.0
            },
            "combined_stress": {
                "description": "Combination of idiosyncratic and market stress",
                "funding_loss_percentage": 40,
                "asset_liquidity_reduction": 60,
                "market_impact_multiplier": 5.0
            }
        }
        
        stress_results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Calculate stressed liquidity position
            stressed_result = await self._calculate_stressed_liquidity_position(
                portfolio_data, funding_data, scenario_params
            )
            
            stress_results[scenario_name] = stressed_result
        
        # Survival analysis
        survival_analysis = self._perform_survival_analysis(stress_results)
        
        # Stress test summary
        stress_summary = self._generate_stress_test_summary(stress_results)
        
        return {
            "stress_scenarios": stress_scenarios,
            "stress_results": stress_results,
            "survival_analysis": survival_analysis,
            "stress_test_summary": stress_summary,
            "recommendations": self._generate_stress_test_recommendations(stress_results)
        }
    
    async def _calculate_stressed_liquidity_position(self, portfolio_data: Dict[str, Any], 
                                                   funding_data: Dict[str, Any], 
                                                   scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate liquidity position under stress scenario."""
        # Base liquidity position
        base_cash = 50000000  # $50M
        base_liquid_assets = 150000000  # $150M
        base_funding = 800000000  # $800M
        
        # Apply stress factors
        funding_loss_pct = scenario_params["funding_loss_percentage"] / 100
        liquidity_reduction_pct = scenario_params["asset_liquidity_reduction"] / 100
        market_impact_mult = scenario_params["market_impact_multiplier"]
        
        # Stressed values
        stressed_funding_loss = base_funding * funding_loss_pct
        stressed_liquid_assets = base_liquid_assets * (1 - liquidity_reduction_pct)
        stressed_market_impact = base_liquid_assets * 0.02 * market_impact_mult  # 2% base impact
        
        # Net liquidity position
        available_liquidity = base_cash + stressed_liquid_assets - stressed_market_impact
        funding_gap = stressed_funding_loss
        net_liquidity_position = available_liquidity - funding_gap
        
        # Survival period calculation (simplified)
        daily_funding_need = 5000000  # $5M daily outflow under stress
        survival_days = max(net_liquidity_position / daily_funding_need, 0) if daily_funding_need > 0 else float('inf')
        
        return {
            "stressed_funding_loss": float(stressed_funding_loss),
            "available_liquidity": float(available_liquidity),
            "net_liquidity_position": float(net_liquidity_position),
            "survival_days": float(survival_days),
            "liquidity_adequacy": "adequate" if net_liquidity_position > 0 else "inadequate",
            "stress_impact": {
                "funding_impact": float(stressed_funding_loss),
                "asset_liquidity_impact": float(base_liquid_assets - stressed_liquid_assets),
                "market_impact": float(stressed_market_impact)
            }
        }
    
    def _perform_survival_analysis(self, stress_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Perform survival analysis across stress scenarios."""
        survival_days = []
        
        for scenario_name, result in stress_results.items():
            days = result.get("survival_days", 0)
            if days != float('inf'):
                survival_days.append(days)
        
        if survival_days:
            min_survival = min(survival_days)
            avg_survival = np.mean(survival_days)
            max_survival = max(survival_days)
        else:
            min_survival = avg_survival = max_survival = 0
        
        # Survival assessment
        if min_survival >= 30:
            survival_assessment = "strong"
        elif min_survival >= 14:
            survival_assessment = "adequate"
        elif min_survival >= 7:
            survival_assessment = "weak"
        else:
            survival_assessment = "critical"
        
        return {
            "minimum_survival_days": float(min_survival),
            "average_survival_days": float(avg_survival),
            "maximum_survival_days": float(max_survival),
            "survival_assessment": survival_assessment,
            "critical_scenarios": [
                scenario for scenario, result in stress_results.items()
                if result.get("survival_days", float('inf')) < 14
            ]
        }
    
    def _generate_stress_test_summary(self, stress_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary of stress test results."""
        # Count scenarios by adequacy
        adequate_scenarios = sum(
            1 for result in stress_results.values()
            if result.get("liquidity_adequacy") == "adequate"
        )
        
        total_scenarios = len(stress_results)
        adequacy_rate = adequate_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        # Overall stress test rating
        if adequacy_rate >= 0.8:
            overall_rating = "pass"
        elif adequacy_rate >= 0.5:
            overall_rating = "conditional_pass"
        else:
            overall_rating = "fail"
        
        return {
            "total_scenarios_tested": total_scenarios,
            "scenarios_passed": adequate_scenarios,
            "pass_rate": float(adequacy_rate * 100),
            "overall_rating": overall_rating,
            "key_vulnerabilities": self._identify_stress_test_vulnerabilities(stress_results),
            "stress_test_date": datetime.utcnow().isoformat()
        }
    
    def _identify_stress_test_vulnerabilities(self, stress_results: Dict[str, Dict]) -> List[str]:
        """Identify key vulnerabilities from stress testing."""
        vulnerabilities = []
        
        for scenario_name, result in stress_results.items():
            if result.get("liquidity_adequacy") == "inadequate":
                vulnerabilities.append(f"Fails {scenario_name.replace('_', ' ')} scenario")
            
            survival_days = result.get("survival_days", float('inf'))
            if survival_days < 30 and survival_days != float('inf'):
                vulnerabilities.append(f"Short survival period in {scenario_name}: {survival_days:.0f} days")
        
        return vulnerabilities
    
    def _generate_stress_test_recommendations(self, stress_results: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []
        
        # Check overall performance
        failed_scenarios = [
            scenario for scenario, result in stress_results.items()
            if result.get("liquidity_adequacy") == "inadequate"
        ]
        
        if failed_scenarios:
            recommendations.append("Increase liquidity buffer to pass all stress scenarios")
            recommendations.append("Diversify funding sources to reduce concentration risk")
        
        # Check survival periods
        short_survival_scenarios = [
            scenario for scenario, result in stress_results.items()
            if result.get("survival_days", float('inf')) < 30 and result.get("survival_days") != float('inf')
        ]
        
        if short_survival_scenarios:
            recommendations.append("Extend liquidity survival period to at least 30 days")
        
        # General recommendations
        recommendations.extend([
            "Develop contingency funding plans for each stress scenario",
            "Establish early warning indicators for liquidity stress",
            "Regular stress testing and scenario updating",
            "Consider additional high-quality liquid assets"
        ])
        
        return recommendations
    
    async def _analyze_cash_flows(self, portfolio_data: Dict[str, Any], 
                                funding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cash flow patterns and projections."""
        # Simulate cash flow data
        time_horizons = [1, 7, 30, 90, 180, 365]  # Days
        
        cash_flow_projections = {}
        
        for horizon in time_horizons:
            # Simulate cash inflows and outflows
            inflows = {
                "interest_income": horizon * 50000,  # $50k per day
                "principal_repayments": horizon * 20000,  # $20k per day
                "asset_sales": horizon * 10000 if horizon <= 30 else 0,  # Only short-term
                "new_funding": horizon * 100000 if horizon >= 7 else 0  # Weekly funding
            }
            
            outflows = {
                "interest_payments": horizon * 40000,  # $40k per day
                "operational_expenses": horizon * 30000,  # $30k per day
                "funding_maturities": horizon * 80000 if horizon >= 30 else 0,  # Monthly maturities
                "margin_calls": horizon * 5000,  # $5k per day
                "other_outflows": horizon * 15000  # $15k per day
            }
            
            total_inflows = sum(inflows.values())
            total_outflows = sum(outflows.values())
            net_cash_flow = total_inflows - total_outflows
            
            cash_flow_projections[f"{horizon}_days"] = {
                "inflows": inflows,
                "outflows": outflows,
                "total_inflows": float(total_inflows),
                "total_outflows": float(total_outflows),
                "net_cash_flow": float(net_cash_flow),
                "cumulative_net_flow": float(net_cash_flow)  # Simplified
            }
        
        # Cash flow analysis
        cash_flow_analysis = self._analyze_cash_flow_patterns(cash_flow_projections)
        
        # Stress testing cash flows
        stressed_cash_flows = self._stress_test_cash_flows(cash_flow_projections)
        
        return {
            "cash_flow_projections": cash_flow_projections,
            "cash_flow_analysis": cash_flow_analysis,
            "stressed_cash_flows": stressed_cash_flows,
            "cash_flow_recommendations": self._generate_cash_flow_recommendations(cash_flow_analysis)
        }
    
    def _analyze_cash_flow_patterns(self, cash_flow_projections: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze cash flow patterns and characteristics."""
        # Extract net cash flows
        net_flows = [data["net_cash_flow"] for data in cash_flow_projections.values()]
        
        # Calculate statistics
        positive_flows = [flow for flow in net_flows if flow > 0]
        negative_flows = [flow for flow in net_flows if flow < 0]
        
        analysis = {
            "cash_flow_statistics": {
                "average_net_flow": float(np.mean(net_flows)),
                "net_flow_volatility": float(np.std(net_flows)),
                "positive_flow_periods": len(positive_flows),
                "negative_flow_periods": len(negative_flows),
                "worst_net_flow": float(min(net_flows)),
                "best_net_flow": float(max(net_flows))
            },
            "cash_flow_stability": {
                "stability_score": self._calculate_cash_flow_stability(net_flows),
                "predictability": "high" if np.std(net_flows) < np.abs(np.mean(net_flows)) else "low"
            },
            "funding_gap_analysis": {
                "periods_with_negative_flows": len(negative_flows),
                "maximum_funding_need": float(abs(min(net_flows))) if negative_flows else 0,
                "cumulative_funding_gap": float(sum(negative_flows)) if negative_flows else 0
            }
        }
        
        return analysis
    
    def _calculate_cash_flow_stability(self, net_flows: List[float]) -> float:
        """Calculate cash flow stability score (0-10)."""
        if not net_flows:
            return 5.0
        
        # Stability based on consistency and positive trend
        mean_flow = np.mean(net_flows)
        std_flow = np.std(net_flows)
        
        # Coefficient of variation (lower is more stable)
        cv = std_flow / abs(mean_flow) if mean_flow != 0 else float('inf')
        
        # Convert to 0-10 scale (lower CV = higher stability)
        stability_score = max(0, 10 - cv * 2)
        
        return float(min(stability_score, 10))
    
    def _stress_test_cash_flows(self, cash_flow_projections: Dict[str, Dict]) -> Dict[str, Dict]:
        """Stress test cash flow projections."""
        stress_scenarios = {
            "revenue_stress": {"inflow_reduction": 0.2, "outflow_increase": 0.1},
            "funding_stress": {"inflow_reduction": 0.4, "outflow_increase": 0.2},
            "market_stress": {"inflow_reduction": 0.3, "outflow_increase": 0.3}
        }
        
        stressed_projections = {}
        
        for scenario_name, stress_params in stress_scenarios.items():
            inflow_reduction = stress_params["inflow_reduction"]
            outflow_increase = stress_params["outflow_increase"]
            
            scenario_projections = {}
            
            for period, data in cash_flow_projections.items():
                stressed_inflows = data["total_inflows"] * (1 - inflow_reduction)
                stressed_outflows = data["total_outflows"] * (1 + outflow_increase)
                stressed_net_flow = stressed_inflows - stressed_outflows
                
                scenario_projections[period] = {
                    "stressed_inflows": float(stressed_inflows),
                    "stressed_outflows": float(stressed_outflows),
                    "stressed_net_flow": float(stressed_net_flow),
                    "stress_impact": float(stressed_net_flow - data["net_cash_flow"])
                }
            
            stressed_projections[scenario_name] = scenario_projections
        
        return stressed_projections
    
    def _generate_cash_flow_recommendations(self, cash_flow_analysis: Dict[str, Any]) -> List[str]:
        """Generate cash flow management recommendations."""
        recommendations = []
        
        # Check cash flow stability
        stability_score = cash_flow_analysis.get("cash_flow_stability", {}).get("stability_score", 5)
        if stability_score < 6:
            recommendations.append("Improve cash flow predictability through better forecasting")
        
        # Check for negative flows
        negative_periods = cash_flow_analysis.get("funding_gap_analysis", {}).get("periods_with_negative_flows", 0)
        if negative_periods > 2:
            recommendations.append("Address recurring negative cash flow periods")
        
        # General recommendations
        recommendations.extend([
            "Maintain cash flow forecasting capabilities across multiple time horizons",
            "Establish automatic cash flow monitoring and alerting systems",
            "Diversify revenue sources to reduce cash flow volatility",
            "Optimize timing of major cash outflows where possible"
        ])
        
        return recommendations
