import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class PensionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)

        # Pension policy parameters
        self.pension_policies = config.get("pension_policies", {})
        self.default_policy = {
            "contribution_rate_employee": 0.08,
            "contribution_rate_employer": 0.12,
            "retirement_age": 65,
            "min_contribution_years": 15,
            "pension_calculation_formula": "average_salary * years_worked * 0.015",
            "fund_annual_return_rate": 0.05
        }

        # Agent pension accounts
        self.agent_pension_balances = defaultdict(float)  # agent_id -> current balance
        self.agent_contribution_years = defaultdict(int)  # agent_id -> total years contributed
        self.agent_retirement_status = defaultdict(bool)  # agent_id -> True if retired
        self.agent_pension_payouts = defaultdict(float)   # agent_id -> monthly pension amount

        # System statistics
        self.total_contributions_epoch = 0
        self.total_payouts_epoch = 0
        self.total_pension_fund_value = 0  # Could be sum of all agent_pension_balances

        # Historical data
        self.fund_value_history = []
        self.contribution_history = []
        self.payout_history = []
        self.num_retirees_history = []

        self.logger.info("PensionSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent pension profiles based on age and work history."""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            age = agent_data.get("basic_info", {}).get("age", 30)
            employment_status = agent_data.get("employment_info", {}).get("status", "unemployed")
            years_in_workforce = agent_data.get("employment_info", {}).get("years_in_workforce", 0)
            
            self.system_state[f"age_{agent_id}"] = age
            self.system_state[f"employment_status_{agent_id}"] = employment_status
            self.system_state[f"income_{agent_id}"] = agent_data.get("economic_attributes", {}).get("annual_income", 0)


            # Initialize contribution years and balance (simplified)
            if "employed" in employment_status.lower():
                # Assume contributions started when they entered workforce, up to a max relevant period
                contributed_years = min(years_in_workforce, age - 18) # Assuming work starts at 18
                self.agent_contribution_years[agent_id] = contributed_years
                
                # Estimate initial balance (very simplified)
                avg_salary_estimation = agent_data.get("economic_attributes", {}).get("annual_income", 50000) * 0.8 # Past salary lower
                initial_balance = avg_salary_estimation * (self.default_policy["contribution_rate_employee"] + self.default_policy["contribution_rate_employer"]) * contributed_years
                initial_balance *= (1 + self.default_policy["fund_annual_return_rate"] / 2) ** contributed_years # Simplified compound interest
                self.agent_pension_balances[agent_id] = initial_balance
            
            self.agent_retirement_status[agent_id] = age >= self.default_policy["retirement_age"]
            if self.agent_retirement_status[agent_id]:
                self._calculate_pension_payout(agent_id, self.default_policy)


        self.total_pension_fund_value = sum(self.agent_pension_balances.values())
        self.logger.info(f"Initialized pension profiles for {len(all_agent_data)} agents. Total fund: {self.total_pension_fund_value:.2f}")

    def get_current_policy(self, epoch: int) -> Dict[str, Any]:
        """Helper to get policy for the current epoch, falling back to default."""
        return self.pension_policies.get(str(epoch), self.default_policy)

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide pension information to agents."""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        policy = self.get_current_policy(current_epoch)

        age = self.system_state.get(f"age_{agent_id}", 30)
        balance = self.agent_pension_balances.get(agent_id, 0)
        contrib_years = self.agent_contribution_years.get(agent_id, 0)
        is_retired = self.agent_retirement_status.get(agent_id, False)
        payout = self.agent_pension_payouts.get(agent_id, 0) if is_retired else 0
        
        projected_pension = 0
        if not is_retired:
            # Simplified projection
            avg_salary = self.system_state.get(f"income_{agent_id}", 50000)
            years_to_retirement = max(0, policy.get("retirement_age", self.default_policy["retirement_age"]) - age)
            projected_contrib_years = contrib_years + years_to_retirement
            # Formula: average_salary * years_worked * 0.015
            projected_pension = avg_salary * projected_contrib_years * 0.015 # Assuming formula value is 0.015


        return {
            "pension_account": {
                "current_balance": balance,
                "years_contributed": contrib_years,
                "estimated_monthly_pension": payout / 12 if is_retired else projected_pension / 12,
                "is_retired": is_retired
            },
            "pension_policy": {
                "retirement_age": policy.get("retirement_age", self.default_policy["retirement_age"]),
                "contribution_rate_employee": policy.get("contribution_rate_employee", self.default_policy["contribution_rate_employee"]),
                "contribution_rate_employer": policy.get("contribution_rate_employer", self.default_policy["contribution_rate_employer"]),
                "min_contribution_years": policy.get("min_contribution_years", self.default_policy["min_contribution_years"]),
                "pension_eligible": contrib_years >= policy.get("min_contribution_years", self.default_policy["min_contribution_years"])
            },
            "fund_status": {
                "annual_return_rate": policy.get("fund_annual_return_rate", self.default_policy["fund_annual_return_rate"]),
                "total_fund_value": self.total_pension_fund_value,
                "sustainability_outlook": "stable" # This would be a more complex calculation
            }
        }

    def _calculate_pension_payout(self, agent_id: str, policy: Dict[str, Any]):
        """Calculate pension payout for a retiring agent."""
        contrib_years = self.agent_contribution_years.get(agent_id, 0)
        avg_salary = self.system_state.get(f"income_{agent_id}", 50000) # Use current income as proxy for career average

        if contrib_years < policy["min_contribution_years"]:
            self.agent_pension_payouts[agent_id] = 0
            return

        # Using the formula string (simplified execution)
        # "average_salary * years_worked * 0.015"
        # For simplicity, let's assume the factor is 0.015
        payout_factor = 0.015 
        try:
            # A more robust way would be to parse and evaluate the formula carefully
            # For now, a simplified direct calculation based on expected structure
            if "average_salary" in policy["pension_calculation_formula"] and \
               "years_worked" in policy["pension_calculation_formula"]:
                # Try to extract a numeric factor from the formula string
                formula_parts = policy["pension_calculation_formula"].split("*")
                for part in formula_parts:
                    try:
                        payout_factor = float(part.strip())
                        break 
                    except ValueError:
                        continue
        except Exception as e:
            self.logger.warning(f"Could not parse pension formula '{policy['pension_calculation_formula']}'. Using default factor {payout_factor}. Error: {e}")
        
        annual_pension = avg_salary * contrib_years * payout_factor
        self.agent_pension_payouts[agent_id] = annual_pension
        self.logger.info(f"Agent {agent_id} retired. Calculated annual pension: {annual_pension:.2f}")


    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process contributions, fund growth, retirements, and payouts."""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        policy = self.get_current_policy(current_epoch)

        self.total_contributions_epoch = 0
        self.total_payouts_epoch = 0
        num_new_retirees = 0
        
        # Grow existing pension fund balances
        for agent_id in list(self.agent_pension_balances.keys()): # Iterate over copy for modification
            if not self.agent_retirement_status.get(agent_id, False): # Fund grows only if not retired
                 self.agent_pension_balances[agent_id] *= (1 + policy["fund_annual_return_rate"]) # Annual return applied monthly/epocally

        # Process agents
        for agent_id, decisions in agent_decisions.items():
            age = self.system_state.get(f"age_{agent_id}", 30)
            employment_status = self.system_state.get(f"employment_status_{agent_id}", "unemployed")
            income = self.system_state.get(f"income_{agent_id}", 0)

            # Handle retirement decisions (can come from RetirementPlanningSystem)
            retirement_decision = decisions.get("RetirementPlanningSystem", {}).get("retire_now", "defer")

            if not self.agent_retirement_status.get(agent_id, False): # Agent is not yet retired
                # Check for automatic retirement
                if age >= policy["retirement_age"] or retirement_decision == "retire":
                    self.agent_retirement_status[agent_id] = True
                    self._calculate_pension_payout(agent_id, policy)
                    num_new_retirees += 1
                    # No contributions in retirement epoch
                else: # Still working
                    if "employed" in employment_status.lower() and income > 0:
                        employee_contrib = income * policy["contribution_rate_employee"]
                        employer_contrib = income * policy["contribution_rate_employer"]
                        total_contrib = employee_contrib + employer_contrib
                        
                        self.agent_pension_balances[agent_id] += total_contrib
                        self.agent_contribution_years[agent_id] = self.agent_contribution_years.get(agent_id,0) + 1 # Assuming epoch is a year
                        self.total_contributions_epoch += total_contrib
            
            # Process payouts for retired agents
            if self.agent_retirement_status.get(agent_id, False):
                monthly_payout = self.agent_pension_payouts.get(agent_id, 0) / 12 # Assuming epoch is a year, payout is monthly
                # In a more complex model, payout might deplete balance or come from a central fund
                # For simplicity, let's assume payouts don't directly reduce individual recorded balances here,
                # but are drawn from the overall 'total_pension_fund_value' concept.
                self.total_payouts_epoch += monthly_payout * 12 # Annual payout for the epoch

        # Update total fund value
        # Sum of individual balances reflects a defined contribution system view
        self.total_pension_fund_value = sum(self.agent_pension_balances.values()) 
        # Alternatively, for a PAYG or mixed system, fund value is more complex:
        # self.total_pension_fund_value += self.total_contributions_epoch - self.total_payouts_epoch
        # self.total_pension_fund_value *= (1 + policy["fund_annual_return_rate"]) # If there's a central fund growing

        # Record history
        self.fund_value_history.append(self.total_pension_fund_value)
        self.contribution_history.append(self.total_contributions_epoch)
        self.payout_history.append(self.total_payouts_epoch)
        current_retirees = sum(1 for r in self.agent_retirement_status.values() if r)
        self.num_retirees_history.append(current_retirees)

        self.logger.info(f"Epoch {current_epoch}: Contributions={self.total_contributions_epoch:.2f}, Payouts={self.total_payouts_epoch:.2f}, Fund Value={self.total_pension_fund_value:.2f}, Retirees={current_retirees}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate pension system health and policy impact."""
        total_epochs = len(self.fund_value_history)
        if total_epochs == 0:
            return {"message": "No data to evaluate."}

        avg_contributions = float(np.mean(self.contribution_history)) if self.contribution_history else 0.0
        avg_payouts = float(np.mean(self.payout_history)) if self.payout_history else 0.0
        
        # Fund sustainability: (contributions - payouts) / payouts. Positive is good.
        sustainability_ratio = float((avg_contributions - avg_payouts) / avg_payouts) if avg_payouts > 0 else 1.0 if avg_contributions > 0 else 0.0
        
        # Coverage: percentage of agents with pension
        eligible_agents = sum(1 for years in self.agent_contribution_years.values() if years > 0)
        total_agents = len(self.agent_pension_balances)
        coverage_rate = float(eligible_agents / total_agents) if total_agents > 0 else 0.0
        
        # Replacement rate: average pension relative to average income (simplified)
        avg_pension = 0.0
        num_pensioners = 0
        for agent_id, payout in self.agent_pension_payouts.items():
            if self.agent_retirement_status.get(agent_id, False) and payout > 0:
                avg_pension += float(payout)
                num_pensioners += 1
        avg_pension = float(avg_pension / num_pensioners) if num_pensioners > 0 else 0.0
        
        avg_income_all = float(np.mean([self.system_state.get(f"income_{aid}", 0.0) for aid in self.agent_pension_balances.keys()]))
        replacement_rate = float(avg_pension / avg_income_all) if avg_income_all > 0 else 0.0
        
        # Trends
        fund_growth_rate = float((self.fund_value_history[-1] - self.fund_value_history[0]) / self.fund_value_history[0]) if total_epochs > 1 and self.fund_value_history[0] > 0 else 0.0
        retiree_growth_rate = float((self.num_retirees_history[-1] - self.num_retirees_history[0]) / self.num_retirees_history[0]) if total_epochs > 1 and self.num_retirees_history[0] > 0 else 0.0

        # Convert numpy arrays to lists for JSON serialization
        fund_value_trend = [float(x) for x in self.fund_value_history]
        retiree_pop_trend = [int(x) for x in self.num_retirees_history]
        
        
        evaluation_results = {
            "system_health": {
                "total_fund_value": float(self.total_pension_fund_value),
                "average_contributions_per_epoch": float(avg_contributions),
                "average_payouts_per_epoch": float(avg_payouts),
                "sustainability_ratio": float(sustainability_ratio),
                "funding_status": "healthy" if sustainability_ratio > 0.1 else "strained" if sustainability_ratio > -0.1 else "critical"
            },
            "coverage_adequacy": {
                "pension_coverage_rate": float(coverage_rate),
                "average_replacement_rate": float(replacement_rate),
                "num_retirees": int(self.num_retirees_history[-1]) if self.num_retirees_history else 0,
                "adequacy_level": "good" if replacement_rate > 0.6 else "moderate" if replacement_rate > 0.4 else "low"
            },
            "policy_impact": {
                "impact_on_retirement_age": "neutral",
                "impact_on_fund_growth": "positive" if fund_growth_rate > 0.05 else "neutral",
            },
            "trends": {
                "fund_value_trend": fund_value_trend,
                "retiree_population_trend": retiree_pop_trend,
                "fund_growth_rate_simulation": float(fund_growth_rate * 100),
                "retiree_growth_rate_simulation": float(retiree_growth_rate * 100),
            },
            "overall_assessment": "stable"
        }
        self.logger.info(f"evaluation_results={evaluation_results}")
        return evaluation_results
    
    
    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage."""
        current_retirees = sum(1 for r in self.agent_retirement_status.values() if r)
        return {
            "total_pension_fund_value": self.total_pension_fund_value,
            "total_contributions_epoch": self.total_contributions_epoch,
            "total_payouts_epoch": self.total_payouts_epoch,
            "number_of_contributors": sum(1 for employed in self.system_state if employed.startswith("employment_status_") and "employed" in self.system_state[employed].lower()),
            "number_of_retirees": current_retirees,
            "average_pension_balance": np.mean(list(self.agent_pension_balances.values())) if self.agent_pension_balances else 0,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 