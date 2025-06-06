import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class YouthBehaviorSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)

        self.gaming_policies_config = config.get("gaming_policies", {})
        self.default_gaming_policy = {
            "restriction_active": False,
            "max_hours_weekday": 24,
            "max_hours_weekend": 24,
            "allowed_time_slots": "any",
            "age_verification": False,
            "applies_to_age": 18, # Age under which restrictions apply
            "real_name_registration": False
        }

        # Agent states specific to youth behavior
        self.agent_gaming_hours_per_day = defaultdict(lambda: random.uniform(0.5, 3)) # agent_id -> hours
        self.agent_compliance_level = defaultdict(lambda: random.uniform(0.7, 1.0)) # agent_id -> 0-1 compliance score
        self.agent_chosen_alternative = defaultdict(lambda: "none") # agent_id -> chosen alternative activity
        self.agent_age = {} # agent_id -> age, for quick access

        # System-level metrics
        self.avg_gaming_hours_youth_epoch = 0
        self.policy_compliance_rate_epoch = 0
        self.alternative_activity_adoption_rate = defaultdict(int) # activity_type -> count

        # Historical data
        self.avg_gaming_hours_history = []
        self.compliance_rate_history = []
        self.alternative_activity_trends = defaultdict(list)

        self.logger.info("YouthBehaviorSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        num_youth_agents = 0
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            age = agent_data.get("basic_info", {}).get("age", 25) # Default to adult if not specified
            self.agent_age[agent_id] = age
            self.system_state[f"age_{agent_id}"] = age
            self.system_state[f"education_{agent_id}"] = agent_data.get("basic_info", {}).get("education_level", "N/A")
            self.system_state[f"social_style_{agent_id}"] = agent_data.get("social_attributes", {}).get("social_style", "moderate")

            if age < self.default_gaming_policy["applies_to_age"]:
                num_youth_agents +=1
                # Initial gaming hours for youth, can be influenced by social style
                social_style = self.system_state.get(f"social_style_{agent_id}", "moderate")
                if "introverted" in social_style:
                    self.agent_gaming_hours_per_day[agent_id] = random.uniform(1.5, 4.0)
                elif "outgoing" in social_style:
                    self.agent_gaming_hours_per_day[agent_id] = random.uniform(0.5, 2.0)
                else:
                    self.agent_gaming_hours_per_day[agent_id] = random.uniform(1.0, 3.0)
            else:
                self.agent_gaming_hours_per_day[agent_id] = random.uniform(0, 1.5) # Adults game less or not at all in this context
                self.agent_compliance_level[agent_id] = 1.0 # Not subject to policy
            
            self.system_state[f"gaming_hours_{agent_id}"] = self.agent_gaming_hours_per_day[agent_id]

        self._update_epoch_metrics(epoch=-1) # Initial state metrics
        self.logger.info(f"Initialized youth behavior for {len(all_agent_data)} agents. Monitored youth (age < {self.default_gaming_policy['applies_to_age']}): {num_youth_agents}")
        if num_youth_agents == 0:
            self.logger.warning("No youth agents (age < {self.default_gaming_policy['applies_to_age']}) were identified in the sample. Youth-specific gaming restriction metrics will reflect this absence.")

    def get_current_gaming_policy(self, epoch: int) -> Dict[str, Any]:
        return self.gaming_policies_config.get(str(epoch), self.default_gaming_policy)

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        policy = self.get_current_gaming_policy(current_epoch)
        agent_age = self.agent_age.get(agent_id, 25)

        is_policy_applicable = agent_age < policy.get("applies_to_age", 18) and policy["restriction_active"]
        
        policy_details_for_agent = "No specific gaming restrictions currently apply to you." 
        if is_policy_applicable:
            policy_details_for_agent = (
                f"Gaming limit: Weekdays max {policy['max_hours_weekday']}h, Weekends max {policy['max_hours_weekend']}h. "
                f"Allowed time: {policy['allowed_time_slots']}. Age verification: {policy['age_verification']}. "
                f"Real name registration: {policy.get('real_name_registration', False)}."
            )

        # Simplified peer behavior and parental attitudes (could be from another system or agent interactions)
        peer_gaming_avg = np.mean([gh for aid, gh in self.agent_gaming_hours_per_day.items() if self.agent_age.get(aid, 25) < policy.get("applies_to_age", 18)])
        parental_attitude_proxy = "strict" if policy["restriction_active"] else "lenient"

        return {
            "gaming_restrictions": {
                "policy_active_for_you": is_policy_applicable,
                "details": policy_details_for_agent,
                "max_weekday_hours_limit": policy["max_hours_weekday"] if is_policy_applicable else 24,
                "max_weekend_hours_limit": policy["max_hours_weekend"] if is_policy_applicable else 24,
                "allowed_time_slots_info": policy["allowed_time_slots"] if is_policy_applicable else "any",
                "age_verification_enforced": policy["age_verification"] if is_policy_applicable else False
            },
            "alternative_activities_info": {
                "suggestions": ["sports_clubs", "local_library_events", "youth_centers", "online_courses"],
                "community_programs_available": True # Placeholder
            },
            "social_context": {
                "peer_average_gaming_hours": round(peer_gaming_avg,1),
                "general_parental_attitude_on_gaming": parental_attitude_proxy,
                "school_policy_on_gaming_discussion": "discouraged_during_school_hours" # Placeholder
            },
            "your_current_behavior": {
                "reported_daily_gaming_hours": self.agent_gaming_hours_per_day.get(agent_id, 0),
                "current_compliance_strategy_estimate": self.agent_compliance_level.get(agent_id, 1.0),
                "last_chosen_alternative_activity": self.agent_chosen_alternative.get(agent_id, "none")
            }
        }

    def _apply_policy_to_agent(self, agent_id: str, policy: Dict[str, Any], gaming_time_intention: float, compliance_strategy: str):
        agent_age = self.agent_age.get(agent_id, 25)
        original_gaming_hours = gaming_time_intention
        adjusted_gaming_hours = original_gaming_hours
        compliance_score_effect = 0 # 0 means full compliance pressure, 1 means no pressure

        if agent_age < policy.get("applies_to_age", 18) and policy["restriction_active"]:
            # Simplified: Assume weekend for now for max_hours, more complex logic for weekday/weekend/timeslot needed
            # This part needs a proper calendar/time system to be accurate.
            # For now, let's use a generic daily limit based on stricter of weekday/weekend for simplicity.
            daily_limit = min(policy["max_hours_weekday"], policy["max_hours_weekend"]) 
            if policy["max_hours_weekday"] == 0 and policy["max_hours_weekend"] > 0:
                 daily_limit = policy["max_hours_weekend"] # E.g. weekend only gaming
            if policy["max_hours_weekday"] > 0 and policy["max_hours_weekend"] == 0:
                 daily_limit = policy["max_hours_weekday"] # E.g. weekday only gaming

            # Compliance strategy effect
            if compliance_strategy == "full_compliance":
                compliance_score_effect = 0.0
                adjusted_gaming_hours = min(original_gaming_hours, daily_limit)
            elif compliance_strategy == "find_loopholes":
                compliance_score_effect = 0.3 # Partially effective at bypassing
                adjusted_gaming_hours = min(original_gaming_hours, daily_limit + (original_gaming_hours - daily_limit) * 0.3) 
            elif compliance_strategy == "use_parent_account":
                compliance_score_effect = 0.6 # More effective bypass if age verification is main hurdle
                adjusted_gaming_hours = min(original_gaming_hours, daily_limit + (original_gaming_hours - daily_limit) * 0.6) if policy["age_verification"] else original_gaming_hours
            elif compliance_strategy == "switch_platforms":
                compliance_score_effect = 0.4 # Switches to less regulated platforms
                # Actual hours might not change much if they find an alternative
                # This should be linked to DigitalEntertainmentSystem
            
            # Clamp to not be negative or excessively high due to loopholes
            adjusted_gaming_hours = max(0, min(adjusted_gaming_hours, original_gaming_hours))
            self.agent_gaming_hours_per_day[agent_id] = adjusted_gaming_hours
            
            # Update compliance level based on how much they had to adjust
            if original_gaming_hours > 0:
                self.agent_compliance_level[agent_id] = (adjusted_gaming_hours / original_gaming_hours) * (1-compliance_score_effect) + compliance_score_effect
            else:
                self.agent_compliance_level[agent_id] = 1.0
        else:
            self.agent_gaming_hours_per_day[agent_id] = original_gaming_hours # No policy or not applicable
            self.agent_compliance_level[agent_id] = 1.0
        
        self.system_state[f"gaming_hours_{agent_id}"] = self.agent_gaming_hours_per_day[agent_id]

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        policy = self.get_current_gaming_policy(current_epoch)
        
        self.alternative_activity_adoption_rate.clear()

        for agent_id, decisions in agent_decisions.items():
            decision = decisions.get("YouthBehaviorSystem", {})
            gaming_time_intention = float(decision.get("gaming_time_intention", self.agent_gaming_hours_per_day.get(agent_id,1)))
            compliance_strategy = decision.get("compliance_strategy", "full_compliance")
            alternative_activity = decision.get("alternative_activity", "none")

            self._apply_policy_to_agent(agent_id, policy, gaming_time_intention, compliance_strategy)
            
            if self.agent_age.get(agent_id, 25) < policy.get("applies_to_age", 18) and policy["restriction_active"]:
                if self.agent_gaming_hours_per_day[agent_id] < gaming_time_intention and alternative_activity != "none":
                    self.agent_chosen_alternative[agent_id] = alternative_activity
                    self.alternative_activity_adoption_rate[alternative_activity] += 1
                else:
                    self.agent_chosen_alternative[agent_id] = "none"
            else:
                self.agent_chosen_alternative[agent_id] = "none"
            self.system_state[f"chosen_alternative_{agent_id}"] = self.agent_chosen_alternative[agent_id]

        self._update_epoch_metrics(current_epoch)

    def _update_epoch_metrics(self, epoch: int):
        youth_gaming_hours_total = 0
        youth_subject_to_policy_count = 0
        total_compliance_score = 0
        
        policy_for_epoch = self.get_current_gaming_policy(epoch if epoch !=-1 else 0)
        age_limit = policy_for_epoch.get("applies_to_age",18)

        for agent_id, age in self.agent_age.items():
            if age < age_limit:
                youth_gaming_hours_total += self.agent_gaming_hours_per_day.get(agent_id,0)
                youth_subject_to_policy_count +=1
                if policy_for_epoch["restriction_active"]:
                     total_compliance_score += self.agent_compliance_level.get(agent_id,1.0)
        
        self.avg_gaming_hours_youth_epoch = youth_gaming_hours_total / youth_subject_to_policy_count if youth_subject_to_policy_count > 0 else 0
        self.avg_gaming_hours_history.append(self.avg_gaming_hours_youth_epoch)

        if policy_for_epoch["restriction_active"] and youth_subject_to_policy_count > 0:
            self.policy_compliance_rate_epoch = total_compliance_score / youth_subject_to_policy_count
        elif not policy_for_epoch["restriction_active"]:
             self.policy_compliance_rate_epoch = 1.0 # Full compliance if no policy active
        else: # policy active but no youth
            self.policy_compliance_rate_epoch = 1.0 
        self.compliance_rate_history.append(self.policy_compliance_rate_epoch)

        for act_type, count in self.alternative_activity_adoption_rate.items():
            self.alternative_activity_trends[act_type].append(count)
        # Ensure all tracked alternatives have a value for this epoch
        all_alt_keys = set(self.alternative_activity_trends.keys())
        current_alt_keys = set(self.alternative_activity_adoption_rate.keys())
        for key_to_pad in all_alt_keys - current_alt_keys:
            self.alternative_activity_trends[key_to_pad].append(0)

        log_epoch = epoch if epoch != -1 else "Initial"
        self.logger.info(f"Epoch {log_epoch}: Avg Youth Gaming Hours={self.avg_gaming_hours_youth_epoch:.2f}, Compliance Rate={self.policy_compliance_rate_epoch:.2f}")
        self.logger.debug(f"Alternative activities adopted in epoch {log_epoch}: {dict(self.alternative_activity_adoption_rate)}")

    def evaluate(self) -> Dict[str, Any]:
        total_epochs = len(self.avg_gaming_hours_history)
        if total_epochs <=1: return {"message": "Not enough data for evaluation (need >1 epoch)."}

        initial_gaming_hours = float(self.avg_gaming_hours_history[0])
        final_gaming_hours = float(self.avg_gaming_hours_history[-1])
        change_in_gaming_hours = float(final_gaming_hours - initial_gaming_hours)
        
        initial_compliance = float(self.compliance_rate_history[0])
        final_compliance = float(self.compliance_rate_history[-1])
        
        # Alternative activity evaluation
        total_alternative_adoption = float(sum(sum(v_list) for k,v_list in self.alternative_activity_trends.items()))
        most_popular_alternative = "none"
        highest_adoption_count = 0
        for act_type, trend_list in self.alternative_activity_trends.items():
            current_type_total_adoption = sum(trend_list)
            if current_type_total_adoption > highest_adoption_count:
                highest_adoption_count = current_type_total_adoption
                most_popular_alternative = act_type
        
        # Check if any policy was active
        policy_was_active = any(self.get_current_gaming_policy(ep)["restriction_active"] for ep in range(total_epochs-1))

        evaluation_results = {
            "gaming_behavior_change": {
                "initial_avg_youth_gaming_hours": float(initial_gaming_hours),
                "final_avg_youth_gaming_hours": float(final_gaming_hours),
                "change_in_avg_gaming_hours": float(change_in_gaming_hours),
                "reduction_in_gaming_hours_pct": float(-change_in_gaming_hours / initial_gaming_hours * 100) if initial_gaming_hours > 0 and change_in_gaming_hours < 0 else 0.0
            },
            "policy_effectiveness": {
                "initial_compliance_rate": float(initial_compliance),
                "final_policy_compliance_rate": float(final_compliance),
                "effectiveness_of_restrictions": str("effective" if policy_was_active and final_gaming_hours < initial_gaming_hours * 0.8 else "partially_effective" if policy_was_active and final_gaming_hours < initial_gaming_hours else "ineffective_or_not_applicable"),
                "overall_policy_impact_assessment": str("significant_reduction" if policy_was_active and change_in_gaming_hours < -1 else "moderate_reduction" if policy_was_active and change_in_gaming_hours < -0.5 else "limited_effect")
            },
            "alternative_activities": {
                "total_alternative_activities_recorded": float(total_alternative_adoption),
                "most_popular_alternative_activity": str(most_popular_alternative),
                "shift_to_alternatives": str("observed" if total_alternative_adoption > (sum(1 for age in self.agent_age.values() if age < self.default_gaming_policy["applies_to_age"]) * 0.1 * (total_epochs-1) ) else "minor")
            },
            "trends_data": {
                "avg_gaming_hours_over_time": [float(x) for x in self.avg_gaming_hours_history],
                "compliance_rate_over_time": [float(x) for x in self.compliance_rate_history],
                "alternative_activity_adoption_trends": {k: [float(x) for x in v] for k,v in self.alternative_activity_trends.items()}
            }
        }
        self.logger.info(f"evaluation_results={evaluation_results}")
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        current_epoch_num = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.get_current_gaming_policy(current_epoch_num)
        num_youth_affected = sum(1 for age in self.agent_age.values() if age < current_policy.get("applies_to_age",18) and current_policy["restriction_active"])
        return {
            "average_youth_gaming_hours": self.avg_gaming_hours_youth_epoch,
            "policy_compliance_rate": self.policy_compliance_rate_epoch,
            "number_of_youth_affected_by_current_policy": num_youth_affected,
            "most_adopted_alternative_current_epoch": max(self.alternative_activity_adoption_rate, key=self.alternative_activity_adoption_rate.get) if self.alternative_activity_adoption_rate else "none",
            "current_epoch": current_epoch_num
        } 