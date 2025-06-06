import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class PublicSafetySystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)

        # Security policies from config
        self.security_policies = config.get("security_policies", {})
        self.default_policy_params = {
            "patrol_density": 0.5, # Scale of 0-1
            "surveillance_coverage": 0.3, # Scale of 0-1
            "community_policing": False,
            "emergency_response_time": 15, # minutes
            "neighborhood_watch": False,
            "smart_policing": False
        }
        
        # Crime statistics (can be updated by CommunityBehaviorSystem or external data)
        self.crime_rate = config.get("base_crime_rate", 0.02) # Overall initial crime rate (e.g. incidents per 1000 people per epoch)
        self.crime_hotspots = {} # region_id -> crime_level_modifier
        self.incident_reports_epoch = []

        # Agent perceived safety
        self.agent_perceived_safety = {} # agent_id -> safety_score (0-1)

        # Historical data
        self.crime_rate_history = []
        self.patrol_density_history = []
        self.surveillance_coverage_history = []
        self.response_time_history = []
        self.reported_incidents_history = []
        self.avg_agent_perception_history = []

        self.logger.info("PublicSafetySystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent safety profiles and regional crime data."""
        num_agents = len(all_agent_data)
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            residence_type = agent_data.get("basic_info", {}).get("residence_type", "urban").lower()
            # Initial perceived safety can be influenced by residence type
            if "urban" in residence_type:
                initial_safety = random.uniform(0.4, 0.7)
            elif "suburban" in residence_type:
                initial_safety = random.uniform(0.6, 0.8)
            else: # rural
                initial_safety = random.uniform(0.7, 0.9)
            self.agent_perceived_safety[agent_id] = initial_safety
            self.system_state[f"residence_type_{agent_id}"] = residence_type
            self.system_state[f"age_{agent_id}"] = agent_data.get("basic_info", {}).get("age", 30)
            self.system_state[f"agent_perceived_safety_{agent_id}"] = initial_safety

        # Initialize some crime hotspots (example)
        # In a real model, this would come from data or be dynamically generated
        # For simplicity, let's assume 3 regions for 100 agents
        num_regions = max(1, num_agents // 30)
        for i in range(num_regions):
            self.crime_hotspots[f"region_{i}"] = random.uniform(1.0, 1.5) # Modifier > 1 means higher crime
        if not self.crime_hotspots: # Ensure at least one region if num_agents is small
            self.crime_hotspots["region_0"] = 1.2

        self.crime_rate_history.append(self.crime_rate)
        if self.agent_perceived_safety:
            self.avg_agent_perception_history.append(np.mean(list(self.agent_perceived_safety.values())))
        else:
            self.avg_agent_perception_history.append(0.5) # Default if no agents

        self.logger.info(f"Initialized safety profiles for {num_agents} agents.")

    def get_current_policy_params(self, epoch: int) -> Dict[str, Any]:
        return self.security_policies.get(str(epoch), self.default_policy_params)

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide safety information to agents."""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        policy_params = self.get_current_policy_params(current_epoch)

        agent_residence = self.system_state.get(f"residence_type_{agent_id}", "urban")
        # Simplified: assign agent to a region for hotspot info
        # More complex: agent location based on their daily activities
        num_regions = len(self.crime_hotspots)
        agent_region_id = f"region_{int(agent_id) % num_regions}" if num_regions > 0 else "region_0" # agent_id assumed numeric str
        regional_crime_modifier = self.crime_hotspots.get(agent_region_id, 1.0)
        
        current_regional_crime_rate = self.crime_rate * regional_crime_modifier
        perceived_safety_score = self.agent_perceived_safety.get(agent_id, 0.5)

        # Safety measures active in the current policy
        active_measures = []
        if policy_params.get("community_policing"): active_measures.append("community_policing_initiatives")
        if policy_params.get("neighborhood_watch"): active_measures.append("neighborhood_watch_programs")
        if policy_params.get("smart_policing"): active_measures.append("smart_policing_technologies")
        if policy_params.get("surveillance_coverage", 0) > 0.5: active_measures.append("increased_cctv_surveillance")
        
        policy_announcement = ""
        if current_epoch == 2:
            policy_announcement = "Public safety reforms initiated: increased patrols and community policing efforts."
        elif current_epoch == 4:
            policy_announcement = "Full implementation of smart policing and enhanced surveillance for improved public safety."

        return {
            "crime_statistics": {
                "overall_crime_rate_level": "high" if self.crime_rate > 0.03 else "medium" if self.crime_rate > 0.01 else "low",
                "your_area_crime_risk": "high" if current_regional_crime_rate > 0.03 else "medium" if current_regional_crime_rate > 0.01 else "low",
                "recent_incident_types": ["theft", "vandalism"] if self.crime_rate > 0.015 else ["minor_disturbances"], # Simplified
                "policy_announcement": policy_announcement
            },
            "patrol_coverage": {
                "patrol_density_level": "high" if policy_params["patrol_density"] > 0.7 else "medium",
                "police_visibility_in_area": "visible" if policy_params["patrol_density"] * regional_crime_modifier > 0.6 else "occasional", # Simplified
                "community_policing_active": policy_params["community_policing"]
            },
            "safety_measures": {
                "surveillance_coverage_pct": policy_params["surveillance_coverage"] * 100,
                "emergency_response_time_min": policy_params["emergency_response_time"],
                "active_initiatives": active_measures
            },
            "incident_reports": { # This could be a feed of recent (anonymized) incidents
                "recent_incidents_in_area": len([i for i in self.incident_reports_epoch if i["region"] == agent_region_id]),
                "types_of_recent_incidents": list(set(i["type"] for i in self.incident_reports_epoch if i["region"] == agent_region_id))[:2]
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Update safety perceptions, crime rates based on policies and agent behaviors."""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        policy_params = self.get_current_policy_params(current_epoch)
        self.incident_reports_epoch = [] # Reset for the new epoch

        # Update system_state with current policy for other systems
        self.system_state["current_patrol_density"] = policy_params["patrol_density"]
        self.system_state["community_policing_active"] = policy_params["community_policing"]
        self.system_state["neighborhood_watch_active"] = policy_params["neighborhood_watch"]
        self.system_state["current_emergency_response_time"] = policy_params["emergency_response_time"]
        self.system_state["smart_policing_active"] = policy_params["smart_policing"]

        # Update crime rate based on policy (simplified)
        # Base crime rate can be influenced by CommunityBehaviorSystem through shared state or other means
        # For now, PublicSafetySystem manages its view of crime_rate based on its direct policies.
        crime_reduction_factor = 0
        crime_reduction_factor += policy_params["patrol_density"] * 0.2 # More patrols, less crime
        crime_reduction_factor += policy_params["surveillance_coverage"] * 0.15 # More surveillance
        if policy_params["community_policing"]: crime_reduction_factor += 0.1
        if policy_params["smart_policing"]: crime_reduction_factor += 0.15
        
        # Max reduction capped, e.g., at 50% of base rate through these direct measures
        self.crime_rate = self.system_state.get("base_crime_rate",0.02) * max(0.3, 1 - crime_reduction_factor)
        self.system_state["current_crime_rate"] = self.crime_rate # Share updated crime rate

        # Simulate some incidents for the epoch based on the new crime rate
        num_agents = len(self.agent_perceived_safety)
        expected_incidents = int(self.crime_rate * num_agents * 10) # Scaled for visibility
        
        # Distribute incidents across hotspots
        for _ in range(expected_incidents):
            region = random.choice(list(self.crime_hotspots.keys()))
            if random.random() < self.crime_hotspots[region] - 0.5: # Higher modifier = higher chance for this region
                 incident_type = random.choice(["theft", "vandalism", "assault", "disturbance"])
                 self.incident_reports_epoch.append({"region": region, "type": incident_type, "epoch": current_epoch})

        # Update agent perceived safety based on decisions and current crime levels
        for agent_id, decisions in agent_decisions.items():
            safety_decision = decisions.get("PublicSafetySystem", {})
            precautions = safety_decision.get("safety_precautions", "normal_behavior")
            
            current_perception = self.agent_perceived_safety.get(agent_id, 0.5)
            perception_change = 0
            
            # Impact of crime rate
            num_regions = len(self.crime_hotspots)
            agent_region_id = f"region_{int(agent_id) % num_regions}" if num_regions > 0 else "region_0"
            regional_crime_modifier = self.crime_hotspots.get(agent_region_id, 1.0)
            effective_regional_crime = self.crime_rate * regional_crime_modifier
            
            if effective_regional_crime > 0.025: perception_change -= 0.1 # Higher crime, lower perception
            elif effective_regional_crime < 0.01: perception_change += 0.05 # Lower crime, higher perception

            # Impact of personal precautions
            if precautions == "high_vigilance": perception_change += 0.05
            elif precautions == "moderate_caution": perception_change += 0.02
            
            # Impact of policy visibility
            if policy_params["patrol_density"] > 0.7: perception_change += 0.05
            if policy_params["community_policing"]: perception_change += 0.03

            self.agent_perceived_safety[agent_id] = max(0.1, min(0.9, current_perception + perception_change))
            self.system_state[f"agent_perceived_safety_{agent_id}"] = self.agent_perceived_safety[agent_id]

        # Record history
        self.crime_rate_history.append(self.crime_rate)
        self.patrol_density_history.append(policy_params["patrol_density"])
        self.surveillance_coverage_history.append(policy_params["surveillance_coverage"])
        self.response_time_history.append(policy_params["emergency_response_time"])
        self.reported_incidents_history.append(len(self.incident_reports_epoch))
        if self.agent_perceived_safety:
            self.avg_agent_perception_history.append(np.mean(list(self.agent_perceived_safety.values())))
        else:
            self.avg_agent_perception_history.append(0.5) # Default if no agents

        self.logger.info(f"Epoch {current_epoch}: Crime Rate={self.crime_rate:.4f}, Incidents={len(self.incident_reports_epoch)}, Avg Perception={np.mean(list(self.agent_perceived_safety.values())):.2f}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate public safety policy effectiveness."""
        total_epochs = len(self.crime_rate_history)
        if total_epochs == 0:
            self.logger.info("No data to evaluate.")
            return {"message": "No data to evaluate."}

        initial_crime_rate = self.crime_rate_history[0]
        final_crime_rate = self.crime_rate_history[-1]
        crime_reduction_pct = (initial_crime_rate - final_crime_rate) / initial_crime_rate * 100 if initial_crime_rate > 0 else 0

        avg_incidents_per_epoch = np.mean(self.reported_incidents_history) if self.reported_incidents_history else 0
        initial_incidents = self.reported_incidents_history[0] if self.reported_incidents_history else 0
        final_incidents = self.reported_incidents_history[-1] if self.reported_incidents_history else 0
        incidents_change_pct = (initial_incidents - final_incidents) / initial_incidents * 100 if initial_incidents > 0 else 0
        
        agent_safety_values = list(self.agent_perceived_safety.values())
        avg_perception = np.mean(agent_safety_values) if agent_safety_values else 0
        std_dev_perception = np.std(agent_safety_values) if agent_safety_values else 0
        
        initial_avg_perception = self.avg_agent_perception_history[0] if self.avg_agent_perception_history else 0
        final_avg_perception = self.avg_agent_perception_history[-1] if self.avg_agent_perception_history else 0
        perception_trend = "improving" if final_avg_perception > initial_avg_perception else "declining" if final_avg_perception < initial_avg_perception else "stable"

        # Effectiveness of specific measures (conceptual)
        final_policy = self.get_current_policy_params(total_epochs -1)
        policy_effectiveness_score = 0
        if final_policy.get("patrol_density", 0) > 0.7: policy_effectiveness_score += 0.2
        if final_policy.get("surveillance_coverage", 0) > 0.6: policy_effectiveness_score += 0.2
        if final_policy.get("community_policing", False): policy_effectiveness_score += 0.3
        if final_policy.get("smart_policing", False): policy_effectiveness_score += 0.3

        # Share helper flags for CommunityBehaviorSystem evaluation
        self.system_state["community_policing_active_final_epoch"] = final_policy.get("community_policing", False)
        self.system_state["patrol_density_increased_during_sim"] = self.patrol_density_history[-1] > self.patrol_density_history[0] if len(self.patrol_density_history) > 1 else False

        # Determine effectiveness and recommendation
        effectiveness = "effective" if crime_reduction_pct > 20 and avg_perception > 0.65 else "partially_effective" if crime_reduction_pct > 10 else "ineffective"
        recommendation = "Consider enhancing community engagement and predictive policing." if crime_reduction_pct < 15 else "Maintain current strategies and monitor evolving trends."

        result = {
            "crime_reduction_metrics": {
                "initial_crime_rate": float(initial_crime_rate),
                "final_crime_rate": float(final_crime_rate),
                "overall_crime_reduction_pct": float(crime_reduction_pct),
                "average_incidents_per_epoch": float(avg_incidents_per_epoch),
                "change_in_reported_incidents_pct": float(incidents_change_pct)
            },
            "citizen_perception": {
                "average_perceived_safety_score": float(avg_perception),
                "std_dev_perceived_safety_score": float(std_dev_perception),
                "safety_perception_trend": perception_trend
            },
            "policy_effectiveness": {
                "final_patrol_density": float(final_policy["patrol_density"]),
                "final_surveillance_coverage": float(final_policy["surveillance_coverage"]),
                "final_emergency_response_time": float(final_policy["emergency_response_time"]),
                "community_policing_impact": "positive" if final_policy["community_policing"] and crime_reduction_pct > 10 else "neutral",
                "overall_policy_score_conceptual": float(policy_effectiveness_score)
            },
            "trends": {
                "crime_rate_over_time": [float(x) for x in self.crime_rate_history],
                "reported_incidents_over_time": [int(x) for x in self.reported_incidents_history],
                "patrol_density_over_time": [float(x) for x in self.patrol_density_history],
                "surveillance_coverage_over_time": [float(x) for x in self.surveillance_coverage_history]
            },
            "summary_assessment": {
                "effectiveness": effectiveness,
                "recommendation": recommendation
            }
        }
        self.logger.info(f"result: {result}")
        return result

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage."""
        current_policy = self.get_current_policy_params(self.current_time.get_current_epoch() if self.current_time else 0)
        return {
            "current_crime_rate": self.crime_rate,
            "reported_incidents_current_epoch": len(self.incident_reports_epoch),
            "average_perceived_safety": np.mean(list(self.agent_perceived_safety.values())) if self.agent_perceived_safety else 0,
            "patrol_density": current_policy["patrol_density"],
            "surveillance_coverage": current_policy["surveillance_coverage"],
            "community_policing_active": current_policy["community_policing"],
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 