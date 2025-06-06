import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class EntrepreneurshipSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)

        self.tax_policies_config = config.get("tax_policies", {})
        self.default_tax_policy = {
            "sme_tax_rate": 0.25,
            "startup_tax_exemption_years": 0,
            "r_and_d_tax_credit": 0.1,
            "employment_subsidy_per_person": 0,
            "simplified_registration": False,
            "free_business_consulting": False
        }

        self.agent_is_entrepreneur = {} # agent_id -> bool
        self.agent_business_info = {}   # agent_id -> {type, age, employees, revenue, profit}
        self.new_businesses_epoch = 0
        self.failed_businesses_epoch = 0

        # Historical data
        self.num_entrepreneurs_history = []
        self.sme_tax_rate_history = []
        self.new_business_creations_history = []
        self.business_failure_rate_history = [] # Conceptual for now

        self.logger.info("EntrepreneurshipSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            # Initialize based on pre-existing conditions or random small chance
            self.agent_is_entrepreneur[agent_id] = agent_data.get("employment_info", {}).get("is_entrepreneur", random.random() < 0.05)
            if self.agent_is_entrepreneur[agent_id]:
                self.agent_business_info[agent_id] = {
                    "type": agent_data.get("employment_info", {}).get("business_type", "retail"),
                    "age_years": agent_data.get("employment_info", {}).get("business_age_years", random.randint(1,10)),
                    "employees": agent_data.get("employment_info", {}).get("business_employees", random.randint(0,5)),
                    "annual_revenue": agent_data.get("economic_attributes",{}).get("business_annual_revenue", random.uniform(20000, 200000)),
                    "annual_profit": agent_data.get("economic_attributes",{}).get("business_annual_profit", random.uniform(5000, 50000))
                }
            else:
                self.agent_business_info[agent_id] = {}
            
            # Store relevant agent attributes in system_state
            self.system_state[f"age_{agent_id}"] = agent_data.get("basic_info", {}).get("age", 30)
            self.system_state[f"education_{agent_id}"] = agent_data.get("basic_info", {}).get("education_level", "high_school")
            self.system_state[f"risk_tolerance_{agent_id}"] = agent_data.get("psychological_attributes", {}).get("risk_tolerance", 0.5)
            self.system_state[f"income_{agent_id}"] = agent_data.get("economic_attributes", {}).get("annual_income", 0)
            self.system_state[f"is_entrepreneur_{agent_id}"] = self.agent_is_entrepreneur[agent_id]

        self._update_history_and_logs(epoch=-1) # Initial state log
        self.logger.info(f"Initialized entrepreneurship status for {len(all_agent_data)} agents.")

    def get_current_tax_policy(self, epoch: int) -> Dict[str, Any]:
        return self.tax_policies_config.get(str(epoch), self.default_tax_policy)

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        policy = self.get_current_tax_policy(current_epoch)
        is_entrepreneur = self.agent_is_entrepreneur.get(agent_id, False)
        business_info = self.agent_business_info.get(agent_id, {})

        # Simulate market conditions (could be another system or external data)
        market_outlook = "optimistic" if policy["sme_tax_rate"] < 0.15 else "neutral"
        startup_costs_level = "low" if policy.get("simplified_registration") else "moderate"
        
        policy_summary = f"Current SME tax rate: {policy['sme_tax_rate'] * 100}%. "
        if policy.get("startup_tax_exemption_years",0) > 0:
            policy_summary += f"Tax exemption for {policy['startup_tax_exemption_years']} years for new startups. "
        if policy.get("free_business_consulting", False):
            policy_summary += "Free business consulting available."

        return {
            "tax_policies": {
                "sme_tax_rate": policy["sme_tax_rate"],
                "startup_tax_exemption_years": policy.get("startup_tax_exemption_years",0),
                "r_and_d_tax_credit": policy["r_and_d_tax_credit"],
                "employment_subsidy_per_new_hire": policy["employment_subsidy_per_person"],
                "simplified_registration_process": policy.get("simplified_registration", False),
                "access_to_free_consulting": policy.get("free_business_consulting", False),
                "policy_summary": policy_summary
            },
            "market_conditions": {
                "overall_economic_outlook": market_outlook,
                "startup_costs_level": startup_costs_level,
                "available_funding_options": ["venture_capital", "bank_loans", "angel_investors"] if market_outlook == "optimistic" else ["bank_loans"],
                "talent_pool_availability": "good" # Placeholder
            },
            "your_business_status": { # Only relevant if agent is already an entrepreneur
                "is_entrepreneur": is_entrepreneur,
                "business_type": business_info.get("type", "N/A"),
                "business_age_years": business_info.get("age_years", 0),
                "number_of_employees": business_info.get("employees", 0),
                "estimated_annual_revenue": business_info.get("annual_revenue", 0),
                "estimated_annual_profit": business_info.get("annual_profit", 0)
            },
            "business_opportunities": { # Placeholder ideas
                "emerging_sectors": ["green_technology", "ai_services", "personalized_health"],
                "government_support_programs": ["sme_grants", "innovation_funds"] if policy["sme_tax_rate"] < 0.2 else []
            }
        }

    def _simulate_business_performance(self, agent_id: str, policy: Dict[str, Any]):
        biz_info = self.agent_business_info.get(agent_id)
        if not biz_info or not self.agent_is_entrepreneur.get(agent_id):
            return

        # Business growth/decline simulation (simplified)
        biz_info["age_years"] += 1
        base_growth_rate = random.uniform(-0.05, 0.15) # Base market fluctuation
        policy_effect = (self.default_tax_policy["sme_tax_rate"] - policy["sme_tax_rate"]) * 0.5 # Tax cut bonus
        if biz_info["age_years"] <= policy.get("startup_tax_exemption_years",0):
            policy_effect += 0.1 # Exemption bonus
        
        growth_rate = base_growth_rate + policy_effect
        biz_info["annual_revenue"] *= (1 + growth_rate)
        
        # Profit calculation (simplified)
        costs = biz_info["annual_revenue"] * random.uniform(0.6, 0.9) # Operational costs
        profit_before_tax = biz_info["annual_revenue"] - costs
        
        tax_rate = policy["sme_tax_rate"]
        if biz_info["age_years"] <= policy.get("startup_tax_exemption_years",0):
            tax_rate = 0 # Tax exemption
        tax_paid = profit_before_tax * tax_rate
        biz_info["annual_profit"] = profit_before_tax - tax_paid

        # R&D Tax credit impact (simplified: increases profit)
        biz_info["annual_profit"] += biz_info["annual_revenue"] * policy["r_and_d_tax_credit"] * random.uniform(0.05, 0.15) 

        # Employment subsidy (simplified: increases profit)
        biz_info["annual_profit"] += biz_info["employees"] * policy["employment_subsidy_per_person"] 

        # Chance of failure (simplified)
        if biz_info["annual_profit"] < 0 and random.random() < 0.1: # Higher chance if loss-making
            self.agent_is_entrepreneur[agent_id] = False
            self.system_state[f"is_entrepreneur_{agent_id}"] = False
            # self.agent_business_info[agent_id] = {} # Reset info
            self.failed_businesses_epoch += 1
            self.logger.info(f"Business for agent {agent_id} failed.")

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        policy = self.get_current_tax_policy(current_epoch)
        self.new_businesses_epoch = 0
        self.failed_businesses_epoch = 0

        for agent_id, decisions in agent_decisions.items():
            decision = decisions.get("EntrepreneurshipSystem", {})
            intention = decision.get("entrepreneurship_intention", 0)
            business_type = decision.get("business_type", "none")
            hiring_intention = decision.get("hiring_intention", "0") # string from config

            # Process existing entrepreneurs
            if self.agent_is_entrepreneur.get(agent_id):
                self._simulate_business_performance(agent_id, policy)
                # Update employee count based on hiring intention (if applicable)
                current_employees = self.agent_business_info[agent_id].get("employees", 0)
                if hiring_intention == "1-5": new_hires = random.randint(1,5)
                elif hiring_intention == "6-20": new_hires = random.randint(6,20)
                elif hiring_intention == "21-50": new_hires = random.randint(21,50)
                else: new_hires = 0
                self.agent_business_info[agent_id]["employees"] = current_employees + new_hires
                self.system_state[f"num_employees_biz_{agent_id}"] = self.agent_business_info[agent_id]["employees"]

            # Process new entrepreneurs
            elif not self.agent_is_entrepreneur.get(agent_id) and intention > 0.7 and business_type != "none":
                # Simplified: high intention and selected type means they start
                # Factors: agent risk tolerance, education, income (not explicitly used here for decision but available in system_state)
                risk_tolerance = self.system_state.get(f"risk_tolerance_{agent_id}", 0.3)
                if random.random() < intention * risk_tolerance: # Higher intention & risk tolerance, higher chance
                    self.agent_is_entrepreneur[agent_id] = True
                    self.system_state[f"is_entrepreneur_{agent_id}"] = True
                    initial_employees = 0
                    if hiring_intention == "1-5": initial_employees = random.randint(1,5)
                    elif hiring_intention == "6-20": initial_employees = random.randint(1,5) # Start small
                    
                    self.agent_business_info[agent_id] = {
                        "type": business_type,
                        "age_years": 0,
                        "employees": initial_employees,
                        "annual_revenue": random.uniform(10000, 50000), # Initial small revenue
                        "annual_profit": random.uniform(-5000, 5000)    # Can start with loss
                    }
                    self.new_businesses_epoch += 1
                    self.logger.info(f"Agent {agent_id} started a new business: {business_type}")
                    self.system_state[f"num_employees_biz_{agent_id}"] = initial_employees
        
        self._update_history_and_logs(current_epoch)

    def _update_history_and_logs(self, epoch: int):
        num_entrepreneurs = sum(1 for is_e in self.agent_is_entrepreneur.values() if is_e)
        self.num_entrepreneurs_history.append(num_entrepreneurs)
        
        policy_to_log = self.get_current_tax_policy(epoch if epoch !=-1 else 0)
        self.sme_tax_rate_history.append(policy_to_log["sme_tax_rate"])
        
        if epoch != -1: # Don't log for initial state
            self.new_business_creations_history.append(self.new_businesses_epoch)
            total_businesses = num_entrepreneurs + self.failed_businesses_epoch # Approx total active at start of epoch
            failure_rate = self.failed_businesses_epoch / total_businesses if total_businesses > 0 else 0
            self.business_failure_rate_history.append(failure_rate)

        log_epoch = epoch if epoch != -1 else "Initial"
        self.logger.info(f"Epoch {log_epoch}: Entrepreneurs={num_entrepreneurs}, New Businesses={self.new_businesses_epoch if epoch != -1 else 'N/A'}, Failed Businesses={self.failed_businesses_epoch if epoch != -1 else 'N/A'}")

    def evaluate(self) -> Dict[str, Any]:
        total_epochs = len(self.num_entrepreneurs_history)
        if total_epochs <= 1: # Need at least initial and one step
            return {"message": "Not enough data to evaluate (need more than 1 epoch)."}

        initial_entrepreneurs = self.num_entrepreneurs_history[0]
        final_entrepreneurs = self.num_entrepreneurs_history[-1]
        change_in_entrepreneurs = final_entrepreneurs - initial_entrepreneurs
        
        avg_new_businesses = np.mean(self.new_business_creations_history) if self.new_business_creations_history else 0
        avg_failure_rate = np.mean(self.business_failure_rate_history) if self.business_failure_rate_history else 0

        initial_tax_rate = self.sme_tax_rate_history[0]
        final_tax_rate = self.sme_tax_rate_history[-1]

        # Impact assessment (conceptual)
        tax_reduction_impact = "positive" if final_entrepreneurs > initial_entrepreneurs and final_tax_rate < initial_tax_rate else "neutral"
        if final_entrepreneurs < initial_entrepreneurs and final_tax_rate < initial_tax_rate:
            tax_reduction_impact = "ineffective_or_counteracted"

        # Calculate total jobs created by new SMEs (approximation)
        total_jobs_from_new_smes = 0
        for agent_id, biz_info in self.agent_business_info.items():
            if self.agent_is_entrepreneur.get(agent_id) and biz_info.get("age_years",0) < total_epochs-1 : # Businesses started during sim
                 total_jobs_from_new_smes += biz_info.get("employees",0)

        return {
            "entrepreneurship_trends": {
                "initial_number_of_entrepreneurs": initial_entrepreneurs,
                "final_number_of_entrepreneurs": final_entrepreneurs,
                "change_in_number_of_entrepreneurs": change_in_entrepreneurs,
                "average_new_businesses_per_epoch": avg_new_businesses,
                "average_business_failure_rate": avg_failure_rate
            },
            "policy_impact_on_smes": {
                "initial_sme_tax_rate": initial_tax_rate,
                "final_sme_tax_rate": final_tax_rate,
                "impact_of_tax_reduction_on_entrepreneurship": tax_reduction_impact,
                "effectiveness_of_startup_exemptions": "likely_positive" if avg_new_businesses > (len(self.agent_is_entrepreneur) * 0.01) else "unclear", # If new businesses > 1% of agents
                "estimated_jobs_created_by_new_smes_in_sim": total_jobs_from_new_smes
            },
            "economic_contribution": { # Highly simplified
                "total_sme_employees": sum(b.get("employees",0) for b in self.agent_business_info.values() if b),
                "total_sme_revenue_approx": sum(b.get("annual_revenue",0) for b in self.agent_business_info.values() if b),
                "total_sme_profit_approx": sum(b.get("annual_profit",0) for b in self.agent_business_info.values() if b)
            },
            "trends_data": {
                "entrepreneurs_over_time": self.num_entrepreneurs_history,
                "new_businesses_over_time": self.new_business_creations_history,
                "sme_tax_rate_over_time": self.sme_tax_rate_history
            }
        }

    def get_state_for_persistence(self) -> Dict[str, Any]:
        current_policy = self.get_current_tax_policy(self.current_time.get_current_epoch() if self.current_time else 0)
        num_active_smes = sum(1 for is_e in self.agent_is_entrepreneur.values() if is_e)
        return {
            "total_active_smes": num_active_smes,
            "new_businesses_current_epoch": self.new_businesses_epoch,
            "failed_businesses_current_epoch": self.failed_businesses_epoch,
            "current_sme_tax_rate": current_policy["sme_tax_rate"],
            "startup_tax_exemption_active": current_policy.get("startup_tax_exemption_years", 0) > 0,
            "average_business_age_years": np.mean([b.get("age_years",0) for b in self.agent_business_info.values() if b and self.agent_is_entrepreneur.get(self._get_agent_id_from_biz(b))]) if num_active_smes > 0 else 0,
            "average_employees_per_sme": np.mean([b.get("employees",0) for b in self.agent_business_info.values() if b and self.agent_is_entrepreneur.get(self._get_agent_id_from_biz(b))]) if num_active_smes > 0 else 0, 
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        }

    def _get_agent_id_from_biz(self, biz_info_val) -> str: # Helper for persistence
        for agent_id, biz_info in self.agent_business_info.items():
            if biz_info is biz_info_val:
                return agent_id
        return "" 