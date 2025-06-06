import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class DigitalEntertainmentSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)

        # Config parameters
        self.gaming_satisfaction_base = config.get("gaming_satisfaction_base", 0.8)
        self.alternative_satisfaction_config = config.get("alternative_satisfaction", {
            "sports": 0.7, "reading": 0.6, "socializing": 0.75,
            "studying": 0.3, "other_digital": 0.65, "none": 0.1
        })
        self.platform_migration_rate = config.get("platform_migration_rate", 0.2)

        # Agent states
        self.agent_overall_satisfaction = defaultdict(lambda: random.uniform(0.5, 0.9)) # agent_id -> satisfaction score
        self.agent_platform_preference = defaultdict(lambda: "mainstream_gaming") # agent_id -> platform type
        self.agent_current_gaming_hours = {} # Store gaming hours from YouthBehaviorSystem for satisfaction calculation
        self.agent_chosen_alternative = {}   # Store chosen alternative for satisfaction

        # System-level metrics
        self.avg_youth_satisfaction_epoch = 0
        self.platform_usage_distribution = defaultdict(int) # platform_type -> count

        # Historical data
        self.avg_satisfaction_history = []
        self.platform_distribution_history = defaultdict(list)
        self.gaming_trends_history = [] # Placeholder for broader market trends

        self.logger.info("DigitalEntertainmentSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        self.platform_usage_distribution["mainstream_gaming"] = 0
        self.platform_usage_distribution["alternative_platforms"] = 0
        self.platform_usage_distribution["non_gaming_digital"] = 0

        agent_ids_from_data = []
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            agent_ids_from_data.append(agent_id)
            self.system_state[f"age_{agent_id}"] = agent_data.get("basic_info", {}).get("age", 25)
            self.platform_usage_distribution["mainstream_gaming"] += 1 # Assume everyone starts on mainstream
        
        # Calculate initial satisfaction for all agents before first metric update
        # This relies on YouthBehaviorSystem having already run its init and populated system_state with initial gaming_hours
        youth_age_threshold = self.config.get("youth_age_threshold_for_satisfaction_metric", 18) # Define youth age for this system's metrics

        current_epoch_initial_satisfaction_sum_youth = 0
        current_epoch_youth_count = 0
        current_epoch_initial_satisfaction_sum_all = 0
        current_epoch_total_agents = 0

        for agent_id in agent_ids_from_data:
            initial_gaming_hours = self.system_state.get(f"gaming_hours_{agent_id}", 0) # Get from YBS init state
            # For initial satisfaction calculation, assume no alternative chosen yet and no policy impact
            self._calculate_satisfaction(agent_id, initial_gaming_hours, "none", False)
            
            agent_satisfaction = self.agent_overall_satisfaction.get(agent_id, 0.0)
            current_epoch_initial_satisfaction_sum_all += agent_satisfaction
            current_epoch_total_agents += 1
            
            age = self.system_state.get(f"age_{agent_id}")
            if age is not None and age < youth_age_threshold:
                current_epoch_initial_satisfaction_sum_youth += agent_satisfaction
                current_epoch_youth_count += 1
        
        if current_epoch_youth_count > 0:
            self.avg_youth_satisfaction_epoch = current_epoch_initial_satisfaction_sum_youth / current_epoch_youth_count
        elif current_epoch_total_agents > 0: # No youth, calculate average for all agents
            self.avg_youth_satisfaction_epoch = current_epoch_initial_satisfaction_sum_all / current_epoch_total_agents
        else: # No agents at all
            self.avg_youth_satisfaction_epoch = self.gaming_satisfaction_base # Default if no agents

        self._update_epoch_metrics(epoch=-1) # Initial state metrics, now avg_youth_satisfaction_epoch is calculated
        self.logger.info(f"Initialized digital entertainment profiles for {len(all_agent_data)} agents. Initial avg youth satisfaction: {self.avg_youth_satisfaction_epoch:.2f}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        # This system often works in tandem with YouthBehaviorSystem's outputs.
        # Information provided to agent might be more about market trends than direct decision inputs here.
        current_satisfaction = self.agent_overall_satisfaction.get(agent_id, 0.7)
        current_platform = self.agent_platform_preference.get(agent_id, "mainstream_gaming")

        # Gaming trends (conceptual)
        gaming_market_trend = random.choice(["esports_growing", "casual_mobile_dominant", "vr_emerging"])
        new_game_releases = random.randint(5, 20) # Placeholder for new titles

        return {
            "gaming_market_trends": {
                "current_dominant_trend": gaming_market_trend,
                "new_game_releases_this_month": new_game_releases,
                "popularity_of_social_gaming": "high", # Placeholder
                "availability_of_unrestricted_platforms": "moderate" # Placeholder
            },
            "platform_information": {
                "your_current_platform_type": current_platform,
                "mainstream_gaming_platform_features": ["strict_age_controls", "popular_titles", "social_features"],
                "alternative_gaming_platform_features": ["looser_restrictions", "niche_games", "international_servers"]
            },
            "your_entertainment_status": {
                "estimated_current_satisfaction_level": current_satisfaction,
                "impact_of_gaming_limits_on_fun": "moderate_decrease" if self.system_state.get(f"gaming_hours_reduction_impact_{agent_id}", False) else "none"
            }
        }

    def _calculate_satisfaction(self, agent_id: str, gaming_hours: float, alternative_activity: str, policy_applies: bool):
        gaming_satisfaction = self.gaming_satisfaction_base * (min(gaming_hours, 3) / 3) # Cap at 3 hours for max satisfaction from gaming
        
        alternative_sat_score = self.alternative_satisfaction_config.get(alternative_activity, 0.1)
        
        # If policy applies and gaming is restricted, alternative satisfaction contributes more.
        # If no restriction or alternative is "none", gaming satisfaction dominates.
        if policy_applies and gaming_hours < self.system_state.get(f"intended_gaming_hours_{agent_id}", gaming_hours + 1) and alternative_activity != "none":
            # Weighted average: e.g., 40% gaming, 60% alternative if alternative is chosen due to restriction
            total_satisfaction = (gaming_satisfaction * 0.4) + (alternative_sat_score * 0.6)
        else:
            total_satisfaction = gaming_satisfaction # Primarily from gaming if no restriction or no alternative chosen
            if alternative_activity != "none": # If they chose an alternative without restriction, it adds a bit
                total_satisfaction = (total_satisfaction + alternative_sat_score) / 1.8 # average but gaming weighted higher

        # Introduce some randomness and clamp
        final_satisfaction = max(0.1, min(1.0, total_satisfaction + random.uniform(-0.05, 0.05)))
        self.agent_overall_satisfaction[agent_id] = final_satisfaction

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        # Get current gaming policy details from YouthBehaviorSystem (or this system could also load it)
        # For this example, assume YouthBehaviorSystem has run and its state is available in self.system_state
        current_gaming_policy = self.system_state.get("current_gaming_policy_details", { # Expected to be set by YBS or a manager
            "restriction_active": False, "applies_to_age": 18
        })

        self.platform_usage_distribution.clear()
        self.platform_usage_distribution["mainstream_gaming"] = 0
        self.platform_usage_distribution["alternative_platforms"] = 0
        self.platform_usage_distribution["non_gaming_digital"] = 0

        temp_total_satisfaction_youth = 0
        youth_count = 0

        for agent_id, decisions in agent_decisions.items():
            age = self.system_state.get(f"age_{agent_id}", 25)
            current_gaming_hours = self.system_state.get(f"gaming_hours_{agent_id}", 0) # From YBS
            chosen_alternative = self.system_state.get(f"chosen_alternative_{agent_id}", "none") # From YBS
            compliance_strategy = self.system_state.get(f"compliance_strategy_{agent_id}", "full_compliance") # From YBS state potentially

            policy_applies_to_agent = age < current_gaming_policy.get("applies_to_age", 18) and current_gaming_policy.get("restriction_active", False)
            self.system_state[f"gaming_hours_reduction_impact_{agent_id}"] = policy_applies_to_agent and current_gaming_hours < self.system_state.get(f"intended_gaming_hours_{agent_id}", current_gaming_hours+0.1)

            self._calculate_satisfaction(agent_id, current_gaming_hours, chosen_alternative, policy_applies_to_agent)

            if age < current_gaming_policy.get("applies_to_age", 18):
                temp_total_satisfaction_youth += self.agent_overall_satisfaction[agent_id]
                youth_count += 1
            
            # Platform switching logic (simplified)
            current_platform = self.agent_platform_preference.get(agent_id, "mainstream_gaming")
            if policy_applies_to_agent and compliance_strategy == "switch_platforms":
                if random.random() < self.platform_migration_rate:
                    current_platform = "alternative_platforms"
            elif chosen_alternative == "other_digital":
                 current_platform = "non_gaming_digital"
            # Could also revert if policy becomes lenient but this is simpler
            self.agent_platform_preference[agent_id] = current_platform
            self.platform_usage_distribution[current_platform] += 1
        
        if youth_count > 0:
            self.avg_youth_satisfaction_epoch = temp_total_satisfaction_youth / youth_count
        else:
            self.avg_youth_satisfaction_epoch = np.mean(list(self.agent_overall_satisfaction.values())) if self.agent_overall_satisfaction else 0.7

        self._update_epoch_metrics(current_epoch)

    def _update_epoch_metrics(self, epoch: int):
        self.avg_satisfaction_history.append(self.avg_youth_satisfaction_epoch)
        
        total_agents_on_platforms = sum(self.platform_usage_distribution.values())
        for platform, count in self.platform_usage_distribution.items():
            self.platform_distribution_history[platform].append(count / total_agents_on_platforms if total_agents_on_platforms > 0 else 0)
        # Ensure all tracked platforms have a value for this epoch
        all_platform_keys = set(self.platform_distribution_history.keys())
        current_platform_keys = set(self.platform_usage_distribution.keys())
        for key_to_pad in all_platform_keys - current_platform_keys:
             # This case should ideally not happen if initialized correctly
             if key_to_pad not in self.platform_distribution_history: self.platform_distribution_history[key_to_pad] = [0] * (len(self.avg_satisfaction_history)-1)
             self.platform_distribution_history[key_to_pad].append(0)

        # Conceptual gaming trend (e.g. overall market size or engagement, not modeled deeply here)
        self.gaming_trends_history.append(random.uniform(0.9, 1.1) * (self.gaming_trends_history[-1] if self.gaming_trends_history else 100))

        log_epoch = epoch if epoch != -1 else "Initial"
        self.logger.info(f"Epoch {log_epoch}: Avg Youth Satisfaction={self.avg_youth_satisfaction_epoch:.2f}")
        self.logger.debug(f"Platform distribution epoch {log_epoch}: {dict(self.platform_usage_distribution)}")

    def evaluate(self) -> Dict[str, Any]:
        total_epochs = len(self.avg_satisfaction_history)
        if total_epochs <= 1: return {"message": "Not enough data for evaluation."}

        initial_satisfaction = float(self.avg_satisfaction_history[0])
        final_satisfaction = float(self.avg_satisfaction_history[-1])
        change_in_satisfaction = float(final_satisfaction - initial_satisfaction)

        platform_shift_summary = {}
        for platform, trend in self.platform_distribution_history.items():
            if len(trend) >= total_epochs:
                platform_shift_summary[platform] = {
                    "initial": float(trend[0]), 
                    "final": float(trend[total_epochs-1]), 
                    "change": float(trend[total_epochs-1] - trend[0])
                }
            elif trend: # Should not happen if padded correctly
                platform_shift_summary[platform] = {
                    "initial": float(trend[0]), 
                    "final": float(trend[-1]), 
                    "change": float(trend[-1] - trend[0])
                }
        
        # Check if policy was active (simplified check based on one of the systems)
        # This should ideally get the actual policy from YouthBehaviorSystem history or config
        policy_was_active = final_satisfaction < initial_satisfaction - 0.05 # Heuristic for policy impact

        evaluation_results = {
            "youth_wellbeing_metrics": {
                "initial_avg_satisfaction": float(initial_satisfaction),
                "final_avg_satisfaction": float(final_satisfaction),
                "change_in_satisfaction_level": float(change_in_satisfaction),
                "overall_wellbeing_impact": str("negative" if change_in_satisfaction < -0.1 else "neutral" if change_in_satisfaction < 0.05 else "positive")
            },
            "market_adaptation": {
                "platform_usage_shift_details": platform_shift_summary,
                "evidence_of_platform_migration": str("significant" if platform_shift_summary.get("alternative_platforms", {}).get("change",0) > 0.1 else "minor"),
                "growth_in_non_gaming_digital_engagement": str("yes" if platform_shift_summary.get("non_gaming_digital", {}).get("change",0) > 0.05 else "no")
            },
            "unintended_consequences": {
                "potential_shift_to_less_regulated_platforms": str("high" if platform_shift_summary.get("alternative_platforms", {}).get("final",0) > 0.15 else "low"),
                "satisfaction_recovery_via_alternatives": str("partial" if policy_was_active and final_satisfaction > initial_satisfaction - 0.15 and final_satisfaction < initial_satisfaction else "limited" if policy_was_active else "n/a")
            },
            "trends_data": {
                "avg_satisfaction_over_time": [float(x) for x in self.avg_satisfaction_history],
                "platform_distribution_over_time": {k: [float(x) for x in v] for k,v in self.platform_distribution_history.items()},
                "conceptual_gaming_market_trend": [float(x) for x in self.gaming_trends_history]
            }
        }
        self.logger.info(f'evaluation_results = {evaluation_results}')
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        current_epoch_num = self.current_time.get_current_epoch() if self.current_time else 0
        return {
            "average_youth_digital_entertainment_satisfaction": self.avg_youth_satisfaction_epoch,
            "mainstream_gaming_platform_usage_share": self.platform_distribution_history["mainstream_gaming"][-1] if self.platform_distribution_history["mainstream_gaming"] else 0,
            "alternative_gaming_platform_usage_share": self.platform_distribution_history["alternative_platforms"][-1] if self.platform_distribution_history["alternative_platforms"] else 0,
            "non_gaming_digital_usage_share": self.platform_distribution_history["non_gaming_digital"][-1] if self.platform_distribution_history["non_gaming_digital"] else 0,
            "current_epoch": current_epoch_num
        } 