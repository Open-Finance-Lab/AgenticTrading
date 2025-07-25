"""
Alpha Agent Pool Observation Lens

This module provides comprehensive observation and monitoring capabilities for alpha agent
activities through the memory system. It acts as a transparent lens to observe, analyze,
and report on all alpha agent operations in real-time.

Architecture:
- Real-time monitoring of alpha factor discovery processes
- Strategy development and configuration tracking
- Performance analysis and backtesting observation
- Cross-agent learning coordination monitoring
- Memory bridge activity analysis

The observation lens connects to the memory agent to provide institutional-grade
monitoring capabilities for alpha strategy research operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import httpx
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentActivity:
    """Data structure for tracking individual agent activities"""
    agent_id: str
    activity_type: str
    timestamp: str
    status: str
    details: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None


@dataclass
class SystemSnapshot:
    """Complete system state snapshot at a point in time"""
    timestamp: str
    active_agents: List[str]
    total_activities: int
    success_rate: float
    memory_bridge_status: str
    recent_activities: List[AgentActivity]


class AlphaAgentObservationLens:
    """
    Comprehensive observation lens for monitoring alpha agent pool activities.
    
    This class provides real-time monitoring, analysis, and reporting capabilities
    for all alpha agent operations through the memory system interface.
    
    Key Features:
    - Real-time activity monitoring
    - Performance metrics tracking
    - System health observation
    - Historical analysis capabilities
    - Automated reporting generation
    """
    
    def __init__(self, memory_endpoints: Dict[str, str] = None):
        """
        Initialize the observation lens with memory system connections.
        
        Args:
            memory_endpoints: Dictionary of memory service endpoints
        """
        self.memory_endpoints = memory_endpoints or {
            "a2a_memory": "http://127.0.0.1:8002",
            "mcp_memory": "http://127.0.0.1:8001",
            "legacy_memory": "http://127.0.0.1:8000"
        }
        
        self.observed_activities: List[AgentActivity] = []
        self.system_snapshots: List[SystemSnapshot] = []
        self.monitoring_active = False
        
        logger.info("🔍 Alpha Agent Observation Lens initialized")
    
    async def start_monitoring(self, monitoring_interval: int = 30):
        """
        Start continuous monitoring of alpha agent activities.
        
        Args:
            monitoring_interval: Monitoring frequency in seconds
        """
        self.monitoring_active = True
        logger.info(f"🚀 Starting continuous monitoring (interval: {monitoring_interval}s)")
        
        try:
            while self.monitoring_active:
                await self._capture_system_snapshot()
                await asyncio.sleep(monitoring_interval)
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.monitoring_active = False
    
    def stop_monitoring(self):
        """Stop the continuous monitoring process."""
        self.monitoring_active = False
        logger.info("🛑 Monitoring stopped")
    
    async def _capture_system_snapshot(self):
        """Capture a complete system state snapshot."""
        try:
            snapshot_time = datetime.now(timezone.utc).isoformat()
            
            # Check memory system connectivity
            memory_status = await self._check_memory_connectivity()
            
            # Retrieve recent activities from memory
            recent_activities = await self._retrieve_recent_activities()
            
            # Calculate system metrics
            success_rate = self._calculate_success_rate(recent_activities)
            active_agents = self._identify_active_agents(recent_activities)
            
            # Create system snapshot
            snapshot = SystemSnapshot(
                timestamp=snapshot_time,
                active_agents=active_agents,
                total_activities=len(recent_activities),
                success_rate=success_rate,
                memory_bridge_status=memory_status,
                recent_activities=recent_activities[-10:]  # Last 10 activities
            )
            
            self.system_snapshots.append(snapshot)
            
            # Log snapshot summary
            logger.info(f"📸 System snapshot captured: {len(active_agents)} active agents, "
                       f"{len(recent_activities)} activities, {success_rate:.1%} success rate")
            
        except Exception as e:
            logger.error(f"❌ Failed to capture system snapshot: {e}")
    
    async def _check_memory_connectivity(self) -> str:
        """Check connectivity status of all memory services."""
        connectivity_status = {}
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for service, endpoint in self.memory_endpoints.items():
                try:
                    response = await client.get(endpoint)
                    if response.status_code in [200, 405, 404]:  # 405/404 are acceptable for service discovery
                        connectivity_status[service] = "connected"
                    else:
                        connectivity_status[service] = f"error_{response.status_code}"
                except Exception as e:
                    connectivity_status[service] = f"disconnected_{str(e)[:20]}"
        
        # Determine overall status
        connected_services = sum(1 for status in connectivity_status.values() if status == "connected")
        total_services = len(connectivity_status)
        
        if connected_services == total_services:
            return "fully_connected"
        elif connected_services > 0:
            return f"partially_connected_{connected_services}/{total_services}"
        else:
            return "disconnected"
    
    async def _retrieve_recent_activities(self, hours_back: int = 1) -> List[AgentActivity]:
        """Retrieve recent agent activities from memory system."""
        activities = []
        
        try:
            # Mock implementation - in production, this would query the actual memory system
            # For now, we'll simulate recent activities based on common patterns
            
            current_time = datetime.now(timezone.utc)
            
            # Simulate typical alpha agent activities
            simulated_activities = [
                AgentActivity(
                    agent_id="alpha_factor_discovery_engine",
                    activity_type="factor_discovery",
                    timestamp=(current_time - timedelta(minutes=45)).isoformat(),
                    status="completed",
                    details={"factors_discovered": 12, "significance_threshold": 0.05},
                    performance_metrics={"avg_ir": 1.2, "discovery_time": 45.3}
                ),
                AgentActivity(
                    agent_id="strategy_configuration_engine",
                    activity_type="strategy_development",
                    timestamp=(current_time - timedelta(minutes=30)).isoformat(),
                    status="completed",
                    details={"strategy_id": "momentum_v3", "risk_level": "moderate"},
                    performance_metrics={"config_time": 30.1, "validation_score": 0.85}
                ),
                AgentActivity(
                    agent_id="comprehensive_backtest_engine",
                    activity_type="backtesting",
                    timestamp=(current_time - timedelta(minutes=15)).isoformat(),
                    status="completed",
                    details={"backtest_period": "2018-2023", "strategy_count": 1},
                    performance_metrics={"sharpe_ratio": 1.45, "total_return": 0.234}
                )
            ]
            
            activities.extend(simulated_activities)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to retrieve activities from memory: {e}")
        
        return activities
    
    def _calculate_success_rate(self, activities: List[AgentActivity]) -> float:
        """Calculate success rate from recent activities."""
        if not activities:
            return 0.0
        
        successful = sum(1 for activity in activities if activity.status == "completed")
        return successful / len(activities)
    
    def _identify_active_agents(self, activities: List[AgentActivity]) -> List[str]:
        """Identify currently active agents from recent activities."""
        recent_threshold = datetime.now(timezone.utc) - timedelta(hours=1)
        
        active_agents = set()
        for activity in activities:
            try:
                activity_time = datetime.fromisoformat(activity.timestamp.replace('Z', '+00:00'))
                if activity_time >= recent_threshold:
                    active_agents.add(activity.agent_id)
            except ValueError:
                continue  # Skip invalid timestamps
        
        return list(active_agents)
    
    async def get_real_time_status(self) -> Dict[str, Any]:
        """Get current real-time status of the alpha agent pool."""
        try:
            current_snapshot = await self._capture_system_snapshot()
            
            if self.system_snapshots:
                latest_snapshot = self.system_snapshots[-1]
                
                return {
                    "observation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "system_status": {
                        "memory_connectivity": latest_snapshot.memory_bridge_status,
                        "active_agents": latest_snapshot.active_agents,
                        "total_activities": latest_snapshot.total_activities,
                        "success_rate": f"{latest_snapshot.success_rate:.1%}"
                    },
                    "recent_activity_summary": {
                        "last_10_activities": [
                            {
                                "agent": activity.agent_id,
                                "type": activity.activity_type,
                                "status": activity.status,
                                "timestamp": activity.timestamp
                            }
                            for activity in latest_snapshot.recent_activities
                        ]
                    },
                    "performance_insights": await self._generate_performance_insights(),
                    "observation_lens_status": "active" if self.monitoring_active else "standby"
                }
            else:
                return {
                    "observation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "system_status": "initializing",
                    "message": "No snapshots captured yet"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get real-time status: {e}")
            return {
                "observation_timestamp": datetime.now(timezone.utc).isoformat(),
                "system_status": "error",
                "error": str(e)
            }
    
    async def _generate_performance_insights(self) -> Dict[str, Any]:
        """Generate performance insights from observed activities."""
        if not self.observed_activities:
            return {"status": "no_data", "insights": []}
        
        insights = []
        
        # Analyze success trends
        recent_activities = self.observed_activities[-20:]  # Last 20 activities
        success_rate = self._calculate_success_rate(recent_activities)
        
        if success_rate >= 0.9:
            insights.append("🎯 Excellent performance: >90% success rate maintained")
        elif success_rate >= 0.7:
            insights.append("✅ Good performance: 70-90% success rate")
        else:
            insights.append("⚠️ Performance concern: <70% success rate detected")
        
        # Analyze agent activity patterns
        agent_activity_count = {}
        for activity in recent_activities:
            agent_activity_count[activity.agent_id] = agent_activity_count.get(activity.agent_id, 0) + 1
        
        if agent_activity_count:
            most_active_agent = max(agent_activity_count.items(), key=lambda x: x[1])
            insights.append(f"🏆 Most active agent: {most_active_agent[0]} ({most_active_agent[1]} activities)")
        
        # Analyze performance metrics trends
        performance_metrics = [
            activity.performance_metrics 
            for activity in recent_activities 
            if activity.performance_metrics
        ]
        
        if performance_metrics:
            insights.append(f"📊 Performance data available for {len(performance_metrics)} activities")
        
        return {
            "status": "available",
            "insights_count": len(insights),
            "insights": insights,
            "analysis_period": "last_20_activities"
        }
    
    async def generate_observation_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """Generate comprehensive observation report."""
        report_timestamp = datetime.now(timezone.utc).isoformat()
        
        report = {
            "observation_report": {
                "generation_timestamp": report_timestamp,
                "monitoring_period": {
                    "snapshots_captured": len(self.system_snapshots),
                    "activities_observed": len(self.observed_activities),
                    "monitoring_active": self.monitoring_active
                },
                "system_overview": await self._generate_system_overview(),
                "agent_performance_analysis": await self._generate_agent_performance_analysis(),
                "memory_system_analysis": await self._generate_memory_system_analysis(),
                "recommendations": await self._generate_recommendations()
            }
        }
        
        if save_to_file:
            report_filename = f"observation_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            report_path = Path(__file__).parent / "reports" / report_filename
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📄 Observation report saved to: {report_path}")
        
        return report
    
    async def _generate_system_overview(self) -> Dict[str, Any]:
        """Generate system overview for the report."""
        if not self.system_snapshots:
            return {"status": "no_data"}
        
        latest_snapshot = self.system_snapshots[-1]
        
        return {
            "current_status": {
                "memory_connectivity": latest_snapshot.memory_bridge_status,
                "active_agents_count": len(latest_snapshot.active_agents),
                "recent_activities_count": latest_snapshot.total_activities,
                "system_success_rate": f"{latest_snapshot.success_rate:.1%}"
            },
            "monitoring_summary": {
                "total_snapshots": len(self.system_snapshots),
                "monitoring_duration": "continuous" if self.monitoring_active else "completed",
                "data_quality": "high" if len(self.system_snapshots) > 5 else "limited"
            }
        }
    
    async def _generate_agent_performance_analysis(self) -> Dict[str, Any]:
        """Generate agent performance analysis."""
        if not self.observed_activities:
            return {"status": "no_agent_data"}
        
        agent_stats = {}
        for activity in self.observed_activities:
            agent_id = activity.agent_id
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "total_activities": 0,
                    "successful_activities": 0,
                    "activity_types": set(),
                    "performance_metrics": []
                }
            
            stats = agent_stats[agent_id]
            stats["total_activities"] += 1
            if activity.status == "completed":
                stats["successful_activities"] += 1
            stats["activity_types"].add(activity.activity_type)
            if activity.performance_metrics:
                stats["performance_metrics"].append(activity.performance_metrics)
        
        # Convert sets to lists for JSON serialization
        for agent_id, stats in agent_stats.items():
            stats["activity_types"] = list(stats["activity_types"])
            stats["success_rate"] = stats["successful_activities"] / stats["total_activities"]
        
        return {
            "agents_analyzed": len(agent_stats),
            "agent_statistics": agent_stats,
            "top_performers": sorted(
                agent_stats.items(), 
                key=lambda x: x[1]["success_rate"], 
                reverse=True
            )[:3]
        }
    
    async def _generate_memory_system_analysis(self) -> Dict[str, Any]:
        """Generate memory system connectivity analysis."""
        connectivity_history = []
        
        for snapshot in self.system_snapshots:
            connectivity_history.append({
                "timestamp": snapshot.timestamp,
                "status": snapshot.memory_bridge_status
            })
        
        return {
            "connectivity_samples": len(connectivity_history),
            "connectivity_history": connectivity_history[-10:],  # Last 10 samples
            "stability_assessment": self._assess_connectivity_stability(connectivity_history)
        }
    
    def _assess_connectivity_stability(self, connectivity_history: List[Dict]) -> str:
        """Assess memory system connectivity stability."""
        if not connectivity_history:
            return "no_data"
        
        stable_connections = sum(
            1 for entry in connectivity_history 
            if entry["status"] in ["fully_connected", "connected"]
        )
        
        stability_ratio = stable_connections / len(connectivity_history)
        
        if stability_ratio >= 0.95:
            return "highly_stable"
        elif stability_ratio >= 0.8:
            return "stable"
        elif stability_ratio >= 0.6:
            return "moderate"
        else:
            return "unstable"
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate operational recommendations based on observations."""
        recommendations = []
        
        if not self.system_snapshots:
            recommendations.append("📊 Increase monitoring frequency to gather more system insights")
            return recommendations
        
        latest_snapshot = self.system_snapshots[-1]
        
        # Memory connectivity recommendations
        if "disconnected" in latest_snapshot.memory_bridge_status:
            recommendations.append("🔧 Investigate memory system connectivity issues")
        elif "partially_connected" in latest_snapshot.memory_bridge_status:
            recommendations.append("⚠️ Monitor memory system partial connectivity")
        
        # Performance recommendations
        if latest_snapshot.success_rate < 0.8:
            recommendations.append("🎯 Investigate causes of reduced success rate")
        
        # Activity level recommendations
        if latest_snapshot.total_activities < 5:
            recommendations.append("📈 Consider increasing agent activity levels")
        elif len(latest_snapshot.active_agents) < 2:
            recommendations.append("🤝 Consider activating additional agents for redundancy")
        
        if not recommendations:
            recommendations.append("✅ System operating within normal parameters")
        
        return recommendations


# Convenience functions for quick observation
async def quick_system_check() -> Dict[str, Any]:
    """Perform a quick system health check."""
    lens = AlphaAgentObservationLens()
    return await lens.get_real_time_status()


async def generate_instant_report() -> Dict[str, Any]:
    """Generate an instant observation report."""
    lens = AlphaAgentObservationLens()
    # Capture a few quick snapshots
    for _ in range(3):
        await lens._capture_system_snapshot()
        await asyncio.sleep(1)
    
    return await lens.generate_observation_report(save_to_file=False)


if __name__ == "__main__":
    # Command-line interface for the observation lens
    import sys
    
    async def main():
        if len(sys.argv) > 1 and sys.argv[1] == "monitor":
            # Start continuous monitoring
            lens = AlphaAgentObservationLens()
            await lens.start_monitoring(monitoring_interval=30)
        elif len(sys.argv) > 1 and sys.argv[1] == "report":
            # Generate instant report
            report = await generate_instant_report()
            print(json.dumps(report, indent=2))
        else:
            # Quick system check
            status = await quick_system_check()
            print(json.dumps(status, indent=2))
    
    asyncio.run(main())
