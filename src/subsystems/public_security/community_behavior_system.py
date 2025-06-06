import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class CommunityBehaviorSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)

        # Behavior model parameters from config
        self.behavior_config = config.get("behavior_model_config", {})
        self.base_crime_rate_config = config.get("base_crime_rate", 0.02) # This is a config parameter, not live state
        # self.crime_reduction_per_patrol_config = config.get("crime_reduction_per_patrol", 0.3) # Unused
        # self.community_effect_config = config.get("community_effect", 0.2) # Unused
        
        # Community state
        self.community_cohesion_score = 0.5 # Overall cohesion (0-1)
        self.public_activity_level = 0.5  # Level of public engagement (0-1)
        self.overall_safety_perception = 0.5 # Aggregated from agents (0-1)
        
        # Agent behavioral states (not directly decided, but emergent or influenced)
        self.agent_social_activity_level = {} # agent_id -> score (0-1)
        self.agent_trust_in_police = {}       # agent_id -> score (0-1)
        self.agent_community_engagement = {}  # agent_id -> score (0-1)
        self.agent_perceived_safety = {}      # agent_id -> score (0-1)

        # Historical data
        self.cohesion_history = []
        self.activity_level_history = []
        self.safety_perception_history = []
        self.trust_in_police_history = [] # Average trust

        self.logger.info("CommunityBehaviorSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent community interaction profiles."""
        # Set the base crime rate for PublicSafetySystem if it needs it (example of inter-system state init)
        self.system_state["base_crime_rate"] = self.base_crime_rate_config
        
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            social_style = agent_data.get("social_attributes", {}).get("social_style", "moderate").lower()
            # residence_type = agent_data.get("basic_info", {}).get("residence_type", "urban").lower()

            # Store necessary agent attributes from their profile
            self.system_state[f"social_style_{agent_id}"] = social_style
            
            # Initial social activity and trust
            if "outgoing" in social_style:
                self.agent_social_activity_level[agent_id] = random.uniform(0.6, 0.9)
                self.agent_trust_in_police[agent_id] = random.uniform(0.5, 0.8)
                self.agent_community_engagement[agent_id] = random.uniform(0.5, 0.8)
            elif "introverted" in social_style:
                self.agent_social_activity_level[agent_id] = random.uniform(0.2, 0.5)
                self.agent_trust_in_police[agent_id] = random.uniform(0.3, 0.6)
                self.agent_community_engagement[agent_id] = random.uniform(0.2, 0.5)
            else: # moderate
                self.agent_social_activity_level[agent_id] = random.uniform(0.4, 0.7)
                self.agent_trust_in_police[agent_id] = random.uniform(0.4, 0.7)
                self.agent_community_engagement[agent_id] = random.uniform(0.3, 0.6)
        
        self._update_aggregate_community_state()
        self.logger.info(f"Initialized community behavior profiles for {len(all_agent_data)} agents.")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide community context and safety perception information."""
        # This system primarily observes and models emergent community behaviors
        # rather than providing direct decision inputs for agents in this basic setup.
        # It can provide context about the community environment.
        
        # Information from PublicSafetySystem (via shared state or direct query if available)
        # patrol_density = self.system_state.get("current_patrol_density", 0.5)
        # community_policing_active = self.system_state.get("community_policing_active", False)
        # current_crime_rate = self.system_state.get("current_crime_rate", self.base_crime_rate_config)

        agent_social_activity = self.agent_social_activity_level.get(agent_id, 0.5)
        agent_trust = self.agent_trust_in_police.get(agent_id, 0.5)
        agent_engagement = self.agent_community_engagement.get(agent_id, 0.5)

        return {
            "community_cohesion": {
                "overall_cohesion_score": self.community_cohesion_score,
                "neighborhood_watch_active": self.system_state.get("neighborhood_watch_active", False), # From policy
                "social_event_frequency": "monthly" if self.public_activity_level > 0.6 else "quarterly" # Simplified
            },
            "public_activities": {
                "general_activity_level": self.public_activity_level,
                "park_usage_level": "high" if self.public_activity_level > 0.7 and self.overall_safety_perception > 0.6 else "medium",
                "evening_stroll_comfort": "comfortable" if self.overall_safety_perception > 0.65 else "cautious"
            },
            "safety_perception": {
                "community_wide_safety_perception": self.overall_safety_perception,
                "trust_in_police_level": np.mean(list(self.agent_trust_in_police.values())) if self.agent_trust_in_police else 0.5,
                "your_trust_in_police": agent_trust,
                "your_community_engagement_level": agent_engagement
            }
        }

    def _update_aggregate_community_state(self):
        """Helper to update overall community scores based on agent states."""
        if self.agent_social_activity_level:
            self.public_activity_level = np.mean(list(self.agent_social_activity_level.values()))
        if self.agent_trust_in_police:
            avg_trust = np.mean(list(self.agent_trust_in_police.values()))
            self.trust_in_police_history.append(avg_trust) # Store average for history
        if self.agent_community_engagement:
            avg_engagement = np.mean(list(self.agent_community_engagement.values()))
            # Cohesion is a mix of engagement and trust (simplified formula)
            self.community_cohesion_score = (avg_engagement * 0.6) + (avg_trust * 0.4) if self.agent_trust_in_police else avg_engagement

        # Overall safety perception can be an aggregate of PublicSafetySystem's agent_perceived_safety
        # This requires PublicSafetySystem to write its per-agent perceptions to shared system_state
        # or for this system to have a way to query it.
        # For now, let's assume PublicSafetySystem updates a shared key or we use an internal proxy.
        all_agent_safety_perceptions = [v for k,v in self.system_state.items() if k.startswith('agent_perceived_safety_')]
        if all_agent_safety_perceptions:
             self.overall_safety_perception = np.mean(all_agent_safety_perceptions)
        elif self.agent_perceived_safety: # Fallback to internal state
             self.overall_safety_perception = np.mean(list(self.agent_perceived_safety.values()))


    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Update community behaviors based on agent actions and safety environment."""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Safety context from PublicSafetySystem (assuming it has updated system_state or we can access it)
        current_crime_rate = self.system_state.get("current_crime_rate", self.base_crime_rate_config)
        patrol_density = self.system_state.get("current_patrol_density", 0.5)
        community_policing_active = self.system_state.get("community_policing_active", False)
        neighborhood_watch_active = self.system_state.get("neighborhood_watch_active", False)
        effective_response_time = self.system_state.get("current_emergency_response_time", 15)
        # smart_policing_active = self.system_state.get("smart_policing_active", False) # Available if needed
        
        # Update agent behavioral attributes based on perceived safety and policies
        for agent_id in self.agent_social_activity_level.keys(): # Iterate existing agents
            # Trust in police can be affected by response times, community policing efforts
            trust_change = 0
            if community_policing_active: trust_change += 0.05
            # Assume effective_response_time is available from PublicSafetySystem or policy
            # effective_response_time = self.system_state.get("current_emergency_response_time", 15) # Already fetched
            if effective_response_time < 10: trust_change += 0.05
            elif effective_response_time > 20: trust_change -= 0.05
            # Direct negative experiences (not modeled yet) would decrease trust
            current_trust = self.agent_trust_in_police.get(agent_id, 0.5)
            self.agent_trust_in_police[agent_id] = max(0.1, min(0.9, current_trust + trust_change))

            # Community engagement can be boosted by neighborhood watch, lower crime
            engagement_change = 0
            if neighborhood_watch_active: engagement_change += 0.1
            if current_crime_rate < 0.015: engagement_change += 0.05 # Safer, more engagement
            current_engagement = self.agent_community_engagement.get(agent_id, 0.5)
            self.agent_community_engagement[agent_id] = max(0.1, min(0.9, current_engagement + engagement_change))

            # Social activity level can be influenced by perceived safety (from PublicSafetySystem)
            agent_safety_perception = self.system_state.get(f"agent_perceived_safety_{agent_id}", self.overall_safety_perception)
            activity_change = 0
            if agent_safety_perception > 0.7: activity_change += 0.1
            elif agent_safety_perception < 0.4: activity_change -= 0.1
            current_activity = self.agent_social_activity_level.get(agent_id, 0.5)
            self.agent_social_activity_level[agent_id] = max(0.1, min(0.9, current_activity + activity_change))

        self._update_aggregate_community_state()

        # Record history
        self.cohesion_history.append(self.community_cohesion_score)
        self.activity_level_history.append(self.public_activity_level)
        self.safety_perception_history.append(self.overall_safety_perception)
        # trust_in_police_history is updated in _update_aggregate_community_state

        self.logger.info(f"Epoch {current_epoch}: Cohesion={self.community_cohesion_score:.2f}, Activity Lvl={self.public_activity_level:.2f}, Safety Perception={self.overall_safety_perception:.2f}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate changes in community behavior and cohesion."""
        total_epochs = len(self.cohesion_history)
        if total_epochs == 0:
            self.logger.info("No data to evaluate.")
            return {"message": "No data to evaluate."}

        initial_cohesion = float(self.cohesion_history[0])
        final_cohesion = float(self.cohesion_history[-1])
        cohesion_change = final_cohesion - initial_cohesion

        initial_activity = float(self.activity_level_history[0])
        final_activity = float(self.activity_level_history[-1])
        activity_change = final_activity - initial_activity
        
        initial_trust = float(self.trust_in_police_history[0]) if self.trust_in_police_history else 0.5
        final_trust = float(self.trust_in_police_history[-1]) if self.trust_in_police_history else 0.5
        trust_change_overall = final_trust - initial_trust # Renamed from trust_change to avoid conflict

        # Calculate standard deviations for agent-level metrics
        agent_activity_values = list(self.agent_social_activity_level.values()) if self.agent_social_activity_level else [0]
        std_dev_activity = float(np.std(agent_activity_values))

        agent_trust_values = list(self.agent_trust_in_police.values()) if self.agent_trust_in_police else [0]
        std_dev_trust = float(np.std(agent_trust_values))

        agent_engagement_values = list(self.agent_community_engagement.values()) if self.agent_community_engagement else [0]
        std_dev_engagement = float(np.std(agent_engagement_values))
        avg_engagement = float(np.mean(agent_engagement_values))
        
        # Correlate cohesion with safety policies (conceptual)
        community_policing_effectiveness = "positive" if cohesion_change > 0.05 and self.system_state.get("community_policing_active_final_epoch", False) else "neutral" # Threshold to 0.05 for more sensitivity
        
        # Calculate correlation safely
        correlation = 0.0
        if len(self.safety_perception_history) > 1 and len(self.activity_level_history) > 1:
            try:
                correlation = float(np.corrcoef(self.safety_perception_history, self.activity_level_history)[0,1])
            except:
                correlation = 0.0
                self.logger.warning("Failed to calculate correlation between safety perception and activity level")

        result = {
            "community_cohesion_metrics": {
                "initial_cohesion_score": initial_cohesion,
                "final_cohesion_score": final_cohesion,
                "change_in_cohesion": cohesion_change,
                "cohesion_level": "strong" if final_cohesion > 0.7 else "moderate" if final_cohesion > 0.4 else "weak",
                "avg_community_engagement": avg_engagement, # Added
                "std_dev_community_engagement": std_dev_engagement # Added
            },
            "public_activity_metrics": {
                "initial_activity_level": initial_activity,
                "final_activity_level": final_activity,
                "change_in_activity": activity_change,
                "std_dev_social_activity": std_dev_activity, # Added
                "activity_status": "vibrant" if final_activity > 0.7 else "subdued" if final_activity < 0.4 else "normal"
            },
            "trust_and_engagement": {
                "initial_avg_trust_in_police": initial_trust,
                "final_avg_trust_in_police": final_trust,
                "change_in_trust": trust_change_overall, # Use renamed variable
                "std_dev_trust_in_police": std_dev_trust, # Added
                "trust_level": "high" if final_trust > 0.7 else "medium" if final_trust > 0.4 else "low"
            },
            "policy_influence_on_community": {
                "impact_of_community_policing_on_cohesion": community_policing_effectiveness,
                "correlation_safety_activity": correlation,
                "influence_of_patrols_on_trust": "positive" if trust_change_overall > 0.05 and self.system_state.get("patrol_density_increased_during_sim", False) else "neutral"
            },
            "trends": {
                "cohesion_over_time": [float(x) for x in self.cohesion_history],
                "public_activity_over_time": [float(x) for x in self.activity_level_history],
                "avg_trust_in_police_over_time": [float(x) for x in self.trust_in_police_history]
            }
        }

        # Log all metrics
        self.logger.info("Community Behavior Evaluation Results:")
        self.logger.info(f"result: {result}")
        return result

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage."""
        return {
            "community_cohesion_score": self.community_cohesion_score,
            "public_activity_level": self.public_activity_level,
            "overall_safety_perception_by_community": self.overall_safety_perception,
            "average_trust_in_police": np.mean(list(self.agent_trust_in_police.values())) if self.agent_trust_in_police else 0,
            "average_community_engagement": np.mean(list(self.agent_community_engagement.values())) if self.agent_community_engagement else 0,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 