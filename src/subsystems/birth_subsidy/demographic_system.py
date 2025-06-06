import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class DemographicSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Birth subsidy policy parameters
        self.birth_policies = config.get("birth_policies", {})
        
        # Agent fertility tracking
        self.agent_children = defaultdict(int)  # agent_id -> number of children
        self.agent_intention = defaultdict(float)  # agent_id -> current fertility intention
        self.agent_ideal_children = defaultdict(int)  # agent_id -> ideal number of children
        
        # Historical data
        self.birth_history = []
        self.intention_history = []
        self.subsidy_payments_history = []
        
        self.logger.info("DemographicSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent fertility profiles with reasonable starting values"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            age = agent_data.get("basic_info", {}).get("age", 30)
            gender = agent_data.get("basic_info", {}).get("gender", "").lower()
            marital_status = agent_data.get("basic_info", {}).get("marital_status", "").lower()
            
            # Skip males for simplicity in fertility tracking
            if gender == 'male' or gender == '男':
                continue
                
            # Initialize with some existing children based on age and marital status
            if "married" in marital_status or "relationship" in marital_status:
                if 20 <= age < 30:
                    num_children = random.choices([0, 1], weights=[0.7, 0.3])[0]
                elif 30 <= age < 40:
                    num_children = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
                elif 40 <= age < 50:
                    num_children = random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
                else:
                    num_children = 0
            else:
                num_children = 0
            
            self.agent_children[agent_id] = num_children
            
            # Initialize fertility age window
            if 20 <= age <= 45:
                self.system_state[f"fertility_eligible_{agent_id}"] = True
                
                # Initialize with reasonable fertility intention based on age and existing children
                base_intention = max(0.0, 0.5 - (0.1 * num_children))
                age_factor = 1.0 - abs(age - 32) / 15  # Peak at age 32
                self.agent_intention[agent_id] = base_intention * age_factor
                
                # Initialize ideal number of children
                if num_children > 0:
                    self.agent_ideal_children[agent_id] = max(num_children, random.choices([1, 2, 3], weights=[0.4, 0.5, 0.1])[0])
                else:
                    self.agent_ideal_children[agent_id] = random.choices([0, 1, 2, 3], weights=[0.2, 0.3, 0.4, 0.1])[0]
            else:
                self.system_state[f"fertility_eligible_{agent_id}"] = False
                self.agent_intention[agent_id] = 0.0
                self.agent_ideal_children[agent_id] = num_children
        
        # Share initial intentions with FamilyPlanningSystem
        for agent_id, intention in self.agent_intention.items():
            self.system_state[f"fertility_intention_{agent_id}"] = intention
        
        self.logger.info(f"Initialized fertility profiles for {len(self.agent_children)} agents")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide fertility and subsidy information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.birth_policies.get(str(current_epoch), 
                                                self.birth_policies.get("0", {}))
        
        # Calculate agent's current fertility status
        current_children = self.agent_children.get(agent_id, 0)
        
        # Calculate applicable subsidies
        applicable_subsidy = 0
        if current_children == 0:
            applicable_subsidy = current_policy.get("first_child_subsidy", 0)
        elif current_children == 1:
            applicable_subsidy = current_policy.get("second_child_subsidy", 0)
        elif current_children == 2:
            applicable_subsidy = current_policy.get("third_child_subsidy", 0)
        
        # Check if policy changed recently
        policy_announcement = ""
        if current_epoch == 2:
            policy_announcement = "Government announces new birth subsidies to encourage higher birth rates"
        elif current_epoch == 4:
            policy_announcement = "Enhanced child subsidies now available, including higher payments and free kindergarten"
        
        # Get social norm from FamilyPlanningSystem if available
        social_norm = self.system_state.get("social_norm_fertility", 1.5)
        
        return {
            "birth_policies": {
                "first_child_subsidy": current_policy.get("first_child_subsidy", 0),
                "second_child_subsidy": current_policy.get("second_child_subsidy", 0),
                "third_child_subsidy": current_policy.get("third_child_subsidy", 0),
                "monthly_childcare_allowance": current_policy.get("monthly_childcare_allowance", 0),
                "maternity_leave_days": current_policy.get("maternity_leave_days", 98),
                "paternity_leave_days": current_policy.get("paternity_leave_days", 0),
                "free_kindergarten": current_policy.get("free_kindergarten", False)
            },
            "subsidy_amounts": {
                "your_current_children": current_children,
                "applicable_subsidy": applicable_subsidy,
                "monthly_allowance": current_policy.get("monthly_childcare_allowance", 0) if current_children > 0 else 0
            },
            "demographic_trends": {
                "policy_announcement": policy_announcement,
                "societal_fertility_norm": social_norm
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent fertility decisions"""
        monthly_births = 0
        monthly_intentions_sum = 0.0
        monthly_intentions_count = 0
        
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.birth_policies.get(str(current_epoch), 
                                                self.birth_policies.get("0", {}))
        
        # Track subsidy payments this month
        monthly_subsidy_payments = 0
        
        # Get social norm from FamilyPlanningSystem
        social_norm = self.system_state.get("social_norm_fertility", 1.5)
        work_life_balance = self.system_state.get("work_life_balance", 0.5)
        
        for agent_id, decisions in agent_decisions.items():
            if "DemographicSystem" not in decisions:
                continue
                
            # Skip if agent is not fertility eligible
            if not self.system_state.get(f"fertility_eligible_{agent_id}", False):
                continue
                
            decision = decisions["DemographicSystem"]
            fertility_intention = decision.get("fertility_intention", 0)
            fertility_intention = float(fertility_intention) if fertility_intention is not None else 0.0
            ideal_children = decision.get("ideal_number_of_children", 0)
            ideal_children = int(ideal_children) if ideal_children is not None else 0
            
            # Update agent's stored intentions
            self.agent_intention[agent_id] = fertility_intention
            self.agent_ideal_children[agent_id] = ideal_children
            
            # Share intention with FamilyPlanningSystem
            self.system_state[f"fertility_intention_{agent_id}"] = fertility_intention
            
            # Track intentions for statistics
            monthly_intentions_sum += fertility_intention
            monthly_intentions_count += 1
            
            # Process birth probability
            current_children = self.agent_children.get(agent_id, 0)
            
            # Skip if already at or above ideal number of children
            if current_children >= ideal_children:
                continue
                
            # Calculate birth probability based on intention and policy subsidies
            base_probability = fertility_intention * 0.2  # Base monthly conception probability
            
            # Policy effect: subsidies increase probability
            if current_children == 0:
                subsidy_factor = current_policy.get("first_child_subsidy", 0) / 10000  # Normalize
            elif current_children == 1:
                subsidy_factor = current_policy.get("second_child_subsidy", 0) / 10000
            else:
                subsidy_factor = current_policy.get("third_child_subsidy", 0) / 10000
                
            # Cap the subsidy effect
            subsidy_factor = min(subsidy_factor, 0.5)
            
            # Social norm effect (higher norm increases probability)
            norm_factor = 0.1 * (social_norm - 1.5) if social_norm > 1.5 else 0
            
            # Work-life balance effect
            wlb_factor = 0.1 * (work_life_balance - 0.5) if work_life_balance > 0.5 else 0
            
            final_probability = base_probability * (1 + subsidy_factor + norm_factor + wlb_factor)
            
            # Determine if birth occurs
            if random.random() < final_probability:
                self.agent_children[agent_id] += 1
                monthly_births += 1
                
                # Calculate subsidy payment
                if current_children == 0:
                    payment = current_policy.get("first_child_subsidy", 0)
                elif current_children == 1:
                    payment = current_policy.get("second_child_subsidy", 0)
                else:
                    payment = current_policy.get("third_child_subsidy", 0)
                    
                monthly_subsidy_payments += payment
                self.logger.debug(f"Agent {agent_id} had child #{current_children+1}, received {payment} subsidy")
        
        # Calculate average intention
        monthly_avg_intention = monthly_intentions_sum / monthly_intentions_count if monthly_intentions_count > 0 else 0
            
        # Record historical data
        self.birth_history.append(monthly_births)
        self.intention_history.append(monthly_avg_intention)
        self.subsidy_payments_history.append(monthly_subsidy_payments)
        
        self.logger.info(f"Epoch {current_epoch}: Births={monthly_births}, "
                        f"Avg Intention={monthly_avg_intention:.2f}, "
                        f"Subsidy Payments={monthly_subsidy_payments}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate birth subsidy policy effectiveness with simplified metrics"""
        # Ensure we have data to evaluate
        if len(self.birth_history) < 6:
            self.logger.warning("Not enough data for proper evaluation")
            return {"error": "Insufficient data for evaluation"}
        
        # Calculate birth rate by policy phase
        pre_policy_births = sum(self.birth_history[:2])
        initial_policy_births = sum(self.birth_history[2:4])
        enhanced_policy_births = sum(self.birth_history[4:])
        
        # Calculate birth rate changes
        pre_policy_rate = pre_policy_births / 2 if pre_policy_births > 0 else 0.1  # Avoid division by zero
        initial_policy_rate = initial_policy_births / 2
        enhanced_policy_rate = enhanced_policy_births / 2
        
        # Calculate percentage changes
        initial_policy_change = ((initial_policy_rate - pre_policy_rate) / pre_policy_rate) * 100 if pre_policy_rate > 0 else 0
        enhanced_policy_change = ((enhanced_policy_rate - pre_policy_rate) / pre_policy_rate) * 100 if pre_policy_rate > 0 else 0
        
        # Calculate intention changes
        initial_intention = np.mean(self.intention_history[:2]) if len(self.intention_history) >= 2 else 0
        final_intention = np.mean(self.intention_history[-2:]) if len(self.intention_history) >= 2 else 0
        intention_change = final_intention - initial_intention
        
        # Calculate subsidy efficiency
        total_subsidies = sum(self.subsidy_payments_history)
        births_after_policy = initial_policy_births + enhanced_policy_births
        subsidy_per_birth = total_subsidies / births_after_policy if births_after_policy > 0 else 0
        
        # Analyze distribution by number of children
        children_distribution = {}
        for agent_id, num_children in self.agent_children.items():
            children_distribution[num_children] = children_distribution.get(num_children, 0) + 1
        
        # Simplified evaluation results
        evaluation_results = {
            "birth_metrics": {
                "pre_policy_monthly_births": float(pre_policy_rate),
                "initial_policy_monthly_births": float(initial_policy_rate),
                "enhanced_policy_monthly_births": float(enhanced_policy_rate),
                "initial_policy_change_percent": float(initial_policy_change),
                "enhanced_policy_change_percent": float(enhanced_policy_change)
            },
            "intention_metrics": {
                "initial_fertility_intention": float(initial_intention),
                "final_fertility_intention": float(final_intention),
                "intention_change": float(intention_change)
            },
            "subsidy_metrics": {
                "total_subsidy_expenditure": float(total_subsidies),
                "subsidy_per_birth": float(subsidy_per_birth)
            },
            "demographic_impact": {
                "children_distribution": children_distribution,
                "average_children_per_agent": float(sum(num * count for num, count in children_distribution.items()) / 
                                            sum(children_distribution.values()) if children_distribution else 0)
            },
            "time_series": {
                "birth_history": [float(x) for x in self.birth_history],
                "intention_history": [float(x) for x in self.intention_history],
                "subsidy_payments": [float(x) for x in self.subsidy_payments_history]
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "total_children": sum(self.agent_children.values()),
            "total_agents_with_children": sum(1 for count in self.agent_children.values() if count > 0),
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 