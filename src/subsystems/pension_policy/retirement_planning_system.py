import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class RetirementPlanningSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)

        # Retirement planning parameters from config
        self.retirement_config = config.get("retirement_planning_config", {})
        self.default_target_replacement_rate = self.retirement_config.get("target_replacement_rate", 0.7)
        self.default_risk_aversion = self.retirement_config.get("risk_aversion_factor", 0.5)
        self.life_expectancy_table = self.retirement_config.get("life_expectancy_table", {
            60: 22, 65: 18, 70: 15 # Age: years remaining (simplified)
        })

        # Agent retirement plans
        self.agent_target_retirement_age = {} # agent_id -> desired retirement age
        self.agent_savings_rate = {}          # agent_id -> additional savings rate for retirement
        self.agent_risk_profile = {}          # agent_id -> investment risk (low, medium, high)
        self.agent_expected_retirement_income = {}

        # Historical data
        self.avg_target_retirement_age_history = []
        self.avg_savings_rate_history = []
        self.retirement_confidence_history = [] # Avg. confidence score (0-1)

        self.logger.info("RetirementPlanningSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent retirement plans based on demographics and financial status."""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            age = agent_data.get("basic_info", {}).get("age", 30)
            income = agent_data.get("economic_attributes", {}).get("annual_income", 50000)
            # financial_literacy = agent_data.get("skills", {}).get("financial_literacy", 0.5)

            # Store initial state for decision making
            self.system_state[f"age_{agent_id}"] = age
            self.system_state[f"income_{agent_id}"] = income
            # self.system_state[f"financial_literacy_{agent_id}"] = financial_literacy

            # Simplified initial target retirement age (can be influenced by policy later)
            self.agent_target_retirement_age[agent_id] = 65 + random.choice([-2, 0, 2, 3]) 
            
            # Initial savings rate (additional to pension)
            self.agent_savings_rate[agent_id] = random.uniform(0.03, 0.12) # * financial_literacy
            
            # Initial risk profile
            risk_rand = random.random()
            if risk_rand < 0.3:
                 self.agent_risk_profile[agent_id] = "low"
            elif risk_rand < 0.7:
                 self.agent_risk_profile[agent_id] = "medium"
            else:
                 self.agent_risk_profile[agent_id] = "high"
            
            self.agent_expected_retirement_income[agent_id] = 0 # Will be calculated with pension info

        self.logger.info(f"Initialized retirement plans for {len(all_agent_data)} agents.")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide retirement planning information and tools to agents."""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        # Policy info from PensionSystem might be relevant here via shared state or direct query if allowed
        # For now, assume some defaults or that PensionSystem has updated system_state

        age = self.system_state.get(f"age_{agent_id}", 30)
        income = self.system_state.get(f"income_{agent_id}", 50000)
        current_pension_balance = self.system_state.get(f"pension_balance_{agent_id}", 0) # Assume PensionSystem might populate this
        projected_pension_benefit = self.system_state.get(f"projected_pension_{agent_id}", income * 0.4) # Placeholder

        target_age = self.agent_target_retirement_age.get(agent_id, 65)
        savings_rate = self.agent_savings_rate.get(agent_id, 0.05)
        risk_profile = self.agent_risk_profile.get(agent_id, "medium")

        # Estimate years to retirement
        years_to_retirement = max(0, target_age - age)

        # Estimate accumulated private savings (simplified)
        # This needs a proper accumulation logic in step(), this is just for info
        estimated_private_savings = income * savings_rate * years_to_retirement * (1 + (0.03 if risk_profile == 'low' else 0.05 if risk_profile == 'medium' else 0.07) / 2) ** years_to_retirement
        
        # Estimate total retirement income
        # Life expectancy at retirement for annuitization (simplified)
        le_at_retirement = self.life_expectancy_table.get(target_age, self.life_expectancy_table.get(65,18))
        annual_private_payout = estimated_private_savings / le_at_retirement if le_at_retirement > 0 else 0
        total_expected_annual_income = projected_pension_benefit + annual_private_payout
        self.agent_expected_retirement_income[agent_id] = total_expected_annual_income

        # Retirement readiness assessment (simplified)
        target_annual_income = income * self.default_target_replacement_rate
        readiness_score = min(1.0, total_expected_annual_income / target_annual_income) if target_annual_income > 0 else 0.5
        
        # Social norms for retirement (could be influenced by other agents' decisions)
        avg_target_age_overall = np.mean(list(self.agent_target_retirement_age.values())) if self.agent_target_retirement_age else 65

        return {
            "retirement_plan_status": {
                "your_target_retirement_age": target_age,
                "your_current_savings_rate": savings_rate,
                "your_investment_risk_profile": risk_profile,
                "estimated_years_to_retirement": years_to_retirement,
                "projected_annual_retirement_income": total_expected_annual_income,
            },
            "financial_outlook": {
                "target_replacement_rate": self.default_target_replacement_rate,
                "current_income_for_target": income,
                "target_annual_retirement_income_goal": target_annual_income,
                "retirement_readiness_score": readiness_score, # 0-1, 1 means on track
                "income_gap_at_retirement": max(0, target_annual_income - total_expected_annual_income)
            },
            "planning_tools": {
                # These are prompts for agent decisions
                "retirement_age_options": [max(age + 5, 60), 62, 65, 67, 70],
                "savings_rate_options": [0.03, 0.05, 0.08, 0.10, 0.12, 0.15],
                "risk_profile_options": ["low", "medium", "high"]
            },
            "social_context": {
                "average_planned_retirement_age": avg_target_age_overall,
                "common_retirement_challenges": ["healthcare_costs", "inflation", "longevity_risk"],
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent decisions on retirement planning."""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        total_agents = len(agent_decisions)
        sum_target_age = 0
        sum_savings_rate = 0
        sum_confidence = 0 # Based on readiness score from previous info step

        for agent_id, decisions in agent_decisions.items():
            # Update agent state based on their decisions from this system
            plan_decision = decisions.get("RetirementPlanningSystem", {})
            
            if "target_retirement_age" in plan_decision:
                raw_target_age = plan_decision.get("target_retirement_age") # Use .get for safety
                if raw_target_age is not None:
                    try:
                        self.agent_target_retirement_age[agent_id] = int(raw_target_age)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Agent {agent_id} provided invalid target_retirement_age: '{raw_target_age}'. Using current or default.")
                        # Optionally keep existing: self.agent_target_retirement_age.get(agent_id, default_age_if_needed)
                else:
                    self.logger.warning(f"Agent {agent_id} provided None for target_retirement_age. Existing value maintained or default will be used if not set.")

            if "additional_savings_rate" in plan_decision:
                raw_savings_rate = plan_decision.get("additional_savings_rate") # Use .get for safety
                if raw_savings_rate is not None:
                    try:
                        self.agent_savings_rate[agent_id] = float(raw_savings_rate)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Agent {agent_id} provided invalid additional_savings_rate: '{raw_savings_rate}'. Using current or default 0.0.")
                        self.agent_savings_rate[agent_id] = self.agent_savings_rate.get(agent_id, 0.0)
                else:
                    self.logger.warning(f"Agent {agent_id} provided None for additional_savings_rate. Using current or default 0.0.")
                    self.agent_savings_rate[agent_id] = self.agent_savings_rate.get(agent_id, 0.0)
            
            if "investment_risk_profile" in plan_decision:
                self.agent_risk_profile[agent_id] = plan_decision["investment_risk_profile"]

            # Accumulate private savings (very simplified, actual accumulation should be in a financial system or here with more detail)
            # For now, this system primarily influences decisions; actual financial changes are managed by PensionSystem or a future PrivateSavingsSystem.
            
            # Update sums for averages
            sum_target_age += self.agent_target_retirement_age.get(agent_id, 65)
            sum_savings_rate += self.agent_savings_rate.get(agent_id, 0.05)
            
            # Re-calculate readiness based on latest info for confidence (could use info from get_system_information's perspective)
            # This is a bit circular if used for history without care. Let's use a proxy.
            income = self.system_state.get(f"income_{agent_id}", 50000)
            expected_income = self.agent_expected_retirement_income.get(agent_id, income * 0.5) # From previous get_info
            target_income = income * self.default_target_replacement_rate
            confidence = min(1.0, expected_income / target_income) if target_income > 0 else 0.5
            sum_confidence += confidence

        # Record historical data
        avg_target_age = sum_target_age / total_agents if total_agents > 0 else 65
        avg_savings_rate = sum_savings_rate / total_agents if total_agents > 0 else 0.05
        avg_confidence = sum_confidence / total_agents if total_agents > 0 else 0.5
        
        self.avg_target_retirement_age_history.append(avg_target_age)
        self.avg_savings_rate_history.append(avg_savings_rate)
        self.retirement_confidence_history.append(avg_confidence)

        self.logger.info(f"Epoch {current_epoch}: Avg Target Age={avg_target_age:.1f}, Avg Savings Rate={avg_savings_rate:.2%}, Avg Confidence={avg_confidence:.2f}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the effectiveness of retirement planning support and citizen preparedness."""
        total_epochs = len(self.avg_target_retirement_age_history)
        if total_epochs == 0:
            return {"message": "No data to evaluate."}

        initial_avg_age = float(self.avg_target_retirement_age_history[0])
        final_avg_age = float(self.avg_target_retirement_age_history[-1])
        age_change = float(final_avg_age - initial_avg_age)

        initial_avg_savings = float(self.avg_savings_rate_history[0])
        final_avg_savings = float(self.avg_savings_rate_history[-1])
        savings_change = float(final_avg_savings - initial_avg_savings)
        
        initial_confidence = float(self.retirement_confidence_history[0])
        final_confidence = float(self.retirement_confidence_history[-1])
        confidence_change = float(final_confidence - initial_confidence)

        # Agent distribution by retirement readiness
        readiness_distribution = {"low": 0.0, "medium": 0.0, "high": 0.0}
        num_agents_evaluated = 0
        for agent_id in self.agent_target_retirement_age.keys(): # Iterate over agents with plans
            income = float(self.system_state.get(f"income_{agent_id}", 50000))
            expected_income = float(self.agent_expected_retirement_income.get(agent_id, income * 0.5))
            target_income = float(income * self.default_target_replacement_rate)
            readiness = float(min(1.0, expected_income / target_income) if target_income > 0 else 0.5)
            num_agents_evaluated += 1
            if readiness < 0.5:
                readiness_distribution["low"] += 1
            elif readiness < 0.8:
                readiness_distribution["medium"] += 1
            else:
                readiness_distribution["high"] += 1
        
        if num_agents_evaluated > 0:
            for key in readiness_distribution:
                readiness_distribution[key] = float(readiness_distribution[key] / num_agents_evaluated)

        # Convert history lists to lists of floats
        age_trend = [float(x) for x in self.avg_target_retirement_age_history]
        savings_trend = [float(x) for x in self.avg_savings_rate_history]
        confidence_trend = [float(x) for x in self.retirement_confidence_history]

        
        evaluation_results ={
            "planning_behavior": {
                "initial_avg_target_retirement_age": float(initial_avg_age),
                "final_avg_target_retirement_age": float(final_avg_age),
                "change_in_target_age": float(age_change),
                "initial_avg_savings_rate": float(initial_avg_savings),
                "final_avg_savings_rate": float(final_avg_savings),
                "change_in_savings_rate": float(savings_change),
            },
            "retirement_preparedness": {
                "initial_avg_confidence": float(initial_confidence),
                "final_avg_confidence": float(final_confidence),
                "change_in_confidence": float(confidence_change),
                "readiness_distribution_pct": readiness_distribution,
                "overall_preparedness_level": str("good" if final_confidence > 0.7 else "fair" if final_confidence > 0.5 else "poor")
            },
            "policy_effectiveness": {
                "shift_towards_earlier_planning": str("yes" if savings_change > 0.01 and confidence_change > 0.05 else "no"),
                "impact_on_retirement_age_expectations": str("delayed" if age_change > 0.5 else "advanced" if age_change < -0.5 else "neutral"),
            },
            "trends": {
                "avg_target_age_trend": age_trend,
                "avg_savings_rate_trend": savings_trend,
                "avg_confidence_trend": confidence_trend,
            },
            "summary": str(f"Retirement planning behavior shows {'an increase' if savings_change > 0 else 'a decrease' if savings_change < 0 else 'no change'} in savings, and {'improved' if confidence_change > 0 else 'declined' if confidence_change < 0 else 'stable'} confidence.")
        }
        self.logger.info(f"evaluation_results={evaluation_results}")
        return evaluation_results
        

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage."""
        total_agents = len(self.agent_target_retirement_age)
        avg_target_age = np.mean(list(self.agent_target_retirement_age.values())) if total_agents > 0 else 0
        avg_savings_rate = np.mean(list(self.agent_savings_rate.values())) if total_agents > 0 else 0
        
        # Simplified confidence for persistence
        sum_confidence = 0
        for agent_id in self.agent_target_retirement_age.keys():
            income = self.system_state.get(f"income_{agent_id}", 50000)
            expected_income = self.agent_expected_retirement_income.get(agent_id, income * 0.5)
            target_income = income * self.default_target_replacement_rate
            confidence = min(1.0, expected_income / target_income) if target_income > 0 else 0.5
            sum_confidence += confidence
        avg_confidence = sum_confidence / total_agents if total_agents > 0 else 0
        
        return {
            "average_target_retirement_age": avg_target_age,
            "average_additional_savings_rate": avg_savings_rate,
            "average_retirement_confidence": avg_confidence,
            "number_of_agents_planning": total_agents,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 