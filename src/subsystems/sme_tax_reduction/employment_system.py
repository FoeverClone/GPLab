import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class EmploymentSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)

        self.base_unemployment_rate = config.get("base_unemployment_rate", 0.05)
        self.job_creation_multiplier = config.get("job_creation_multiplier", 2.5)
        self.wage_growth_from_competition = config.get("wage_growth_from_competition", 0.02)

        # Agent employment states
        self.agent_employment_status = {} # agent_id -> "employed", "unemployed", "student", "retired"
        self.agent_occupation = {}        # agent_id -> "developer", "teacher", "retail_worker", etc.
        self.agent_income = {}            # agent_id -> annual_income

        # System-level stats
        self.current_unemployment_rate = self.base_unemployment_rate
        self.total_workforce = 0
        self.total_employed = 0
        self.available_jobs = 0 # Conceptual: jobs created by SMEs and other sectors

        # Historical data
        self.unemployment_rate_history = []
        self.avg_wage_history = []
        self.job_creation_sme_history = [] # Jobs from EntrepreneurshipSystem

        self.logger.info("EmploymentSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        num_potential_workforce = 0
        num_employed_initial = 0
        total_initial_income = 0

        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            age = agent_data.get("basic_info", {}).get("age", 30)
            emp_info = agent_data.get("employment_info", {})
            eco_attrs = agent_data.get("economic_attributes", {})

            status = emp_info.get("status", "unemployed")
            if age < 18 or age > 65: # Simplified: not in typical workforce age
                status = "student" if age < 18 else "retired"
            
            self.agent_employment_status[agent_id] = status
            self.agent_occupation[agent_id] = emp_info.get("occupation", "N/A")
            initial_income = eco_attrs.get("annual_income", 0)
            self.agent_income[agent_id] = initial_income

            self.system_state[f"employment_status_{agent_id}"] = status
            self.system_state[f"occupation_{agent_id}"] = self.agent_occupation[agent_id]
            self.system_state[f"income_{agent_id}"] = initial_income

            if 18 <= age <= 65:
                num_potential_workforce += 1
                if status == "employed":
                    num_employed_initial += 1
                    total_initial_income += initial_income
        
        self.total_workforce = num_potential_workforce
        self.total_employed = num_employed_initial
        if self.total_workforce > 0:
            self.current_unemployment_rate = (self.total_workforce - self.total_employed) / self.total_workforce
        else:
            self.current_unemployment_rate = 0
        
        # Estimate initial available jobs (conceptual)
        self.available_jobs = self.total_employed * 1.05 # Assume 5% vacancy initially

        self.unemployment_rate_history.append(self.current_unemployment_rate)
        avg_initial_wage = total_initial_income / num_employed_initial if num_employed_initial > 0 else 0
        self.avg_wage_history.append(avg_initial_wage)
        self.job_creation_sme_history.append(0) # No SME job creation at init

        self.logger.info(f"Initialized employment status for {len(all_agent_data)} agents. Initial Unemployment: {self.current_unemployment_rate:.2%}")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        status = self.agent_employment_status.get(agent_id, "N/A")
        occupation = self.agent_occupation.get(agent_id, "N/A")
        income = self.agent_income.get(agent_id, 0)
        
        # Job market info (could be more dynamic)
        job_market_outlook = "growing" if self.current_unemployment_rate < self.base_unemployment_rate - 0.01 else "stable"
        if self.current_unemployment_rate > self.base_unemployment_rate + 0.01: job_market_outlook = "shrinking"

        avg_market_wage = np.mean(self.avg_wage_history) if self.avg_wage_history else 50000 # Fallback

        return {
            "job_market_overview": {
                "current_unemployment_rate": self.current_unemployment_rate,
                "job_market_outlook": job_market_outlook, # growing, stable, shrinking
                "average_market_wage_level": avg_market_wage, # Placeholder for avg salary
                "most_in_demand_skills": ["digital_literacy", "data_analysis", "project_management"] # Placeholder
            },
            "your_employment_details": {
                "current_employment_status": status,
                "current_occupation": occupation,
                "current_annual_income": income
            },
            "employment_opportunities": {
                # These would ideally be dynamic from job postings/vacancies
                "available_job_postings_count": int(max(0, self.available_jobs - self.total_employed)),
                "sectors_with_high_demand": ["technology", "healthcare", "logistics"] # Placeholder
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # 1. Account for jobs created by new/expanding SMEs from EntrepreneurshipSystem
        # This requires EntrepreneurshipSystem to update shared state or have a getter method
        new_sme_jobs_created_this_epoch = 0
        sme_job_changes = 0 # Net change from SMEs
        for agent_id_es in self.system_state: # Iterate through all system_state keys
            if agent_id_es.startswith("is_entrepreneur_") and self.system_state[agent_id_es]:
                # This agent is an entrepreneur
                biz_owner_id = agent_id_es.split("is_entrepreneur_")[1]
                num_employees_key = f"num_employees_biz_{biz_owner_id}"
                prev_employees_key = f"prev_num_employees_biz_{biz_owner_id}"
                
                current_employees = self.system_state.get(num_employees_key, 0)
                prev_employees = self.system_state.get(prev_employees_key, 0)

                if self.system_state.get(f"new_business_epoch_{biz_owner_id}") == current_epoch:
                    sme_job_changes += current_employees # All employees are new jobs from new biz
                else:
                    sme_job_changes += (current_employees - prev_employees) # Net change for existing biz
                
                self.system_state[prev_employees_key] = current_employees # Update for next epoch
        
        new_sme_jobs_created_this_epoch = max(0, sme_job_changes) # Only count net positive as creation for this metric
        self.job_creation_sme_history.append(new_sme_jobs_created_this_epoch)
        self.available_jobs += sme_job_changes # Net change affects available jobs

        # 2. Simulate job seeking and hiring (simplified)
        # Agents who are unemployed might find jobs if available_jobs > total_employed
        # Agents might lose jobs if available_jobs < total_employed (layoffs - not explicitly modeled here, but unemployment rate reflects it)
        
        num_newly_employed = 0
        num_newly_unemployed = 0 # Due to conceptual job losses / churn

        potential_job_seekers = [aid for aid, status in self.agent_employment_status.items() if status == "unemployed" and 18 <= self.system_state.get(f"age_{aid}", 30) <= 65]
        random.shuffle(potential_job_seekers)

        job_vacancies = max(0, int(self.available_jobs - self.total_employed))

        for seeker_id in potential_job_seekers:
            if job_vacancies > 0:
                if random.random() < 0.3: # Chance of finding a job if one is vacant (simplified)
                    self.agent_employment_status[seeker_id] = "employed"
                    # Assign a generic occupation and income for simplicity
                    self.agent_occupation[seeker_id] = random.choice(["service_worker", "admin_staff", "junior_analyst"])
                    new_income = np.mean(self.avg_wage_history) * random.uniform(0.8, 1.2) if self.avg_wage_history else 40000
                    self.agent_income[seeker_id] = new_income
                    
                    self.system_state[f"employment_status_{seeker_id}"] = "employed"
                    self.system_state[f"occupation_{seeker_id}"] = self.agent_occupation[seeker_id]
                    self.system_state[f"income_{seeker_id}"] = new_income
                    
                    num_newly_employed += 1
                    job_vacancies -= 1
            else:
                break # No more vacancies
        
        # Simulate some job churn/loss (very simplified)
        currently_employed_list = [aid for aid, status in self.agent_employment_status.items() if status == "employed"]
        if self.current_unemployment_rate > self.base_unemployment_rate + 0.02: # If economy is bad
            num_to_lose_jobs = int(len(currently_employed_list) * 0.02) # 2% might lose jobs
            lost_job_ids = random.sample(currently_employed_list, min(num_to_lose_jobs, len(currently_employed_list)))
            for lost_id in lost_job_ids:
                self.agent_employment_status[lost_id] = "unemployed"
                # self.agent_income[lost_id] = 0 # Income drops
                self.system_state[f"employment_status_{lost_id}"] = "unemployed"
                # self.system_state[f"income_{lost_id}"] = 0
                num_newly_unemployed +=1

        # 3. Update system-level stats
        self.total_employed = sum(1 for status in self.agent_employment_status.values() if status == "employed")
        if self.total_workforce > 0:
            self.current_unemployment_rate = (self.total_workforce - self.total_employed) / self.total_workforce
        else:
            self.current_unemployment_rate = 0
        
        current_total_income = sum(self.agent_income[aid] for aid, status in self.agent_employment_status.items() if status == "employed")
        current_avg_wage = current_total_income / self.total_employed if self.total_employed > 0 else 0
        
        # Wage growth due to competition for talent (if unemployment is low)
        if self.current_unemployment_rate < self.base_unemployment_rate - 0.01:
            current_avg_wage *= (1 + self.wage_growth_from_competition)
            # Distribute this wage growth somewhat (simplified)
            for aid in self.agent_income:
                if self.agent_employment_status.get(aid) == "employed":
                    self.agent_income[aid] *= (1 + self.wage_growth_from_competition * random.uniform(0.5, 1.5))
                    self.system_state[f"income_{aid}"] = self.agent_income[aid]

        self.unemployment_rate_history.append(self.current_unemployment_rate)
        self.avg_wage_history.append(current_avg_wage)

        self.logger.info(f"Epoch {current_epoch}: Unemployment={self.current_unemployment_rate:.2%}, Employed={self.total_employed}, SME Jobs Created={new_sme_jobs_created_this_epoch}, Avg Wage={current_avg_wage:.2f}")

    def evaluate(self) -> Dict[str, Any]:
        total_epochs = len(self.unemployment_rate_history)
        if total_epochs <= 1:
            return {"message": "Not enough data to evaluate (need more than 1 epoch)."}

        initial_unemployment_rate = self.unemployment_rate_history[0]
        final_unemployment_rate = self.unemployment_rate_history[-1]
        change_in_unemployment = final_unemployment_rate - initial_unemployment_rate

        initial_avg_wage = self.avg_wage_history[0]
        final_avg_wage = self.avg_wage_history[-1]
        change_in_avg_wage = final_avg_wage - initial_avg_wage
        
        total_sme_jobs_sim = sum(self.job_creation_sme_history)

        return {
            "employment_market_summary": {
                "initial_unemployment_rate": initial_unemployment_rate,
                "final_unemployment_rate": final_unemployment_rate,
                "change_in_unemployment_rate_points": change_in_unemployment * 100, # As points
                "overall_employment_trend": "improving" if change_in_unemployment < -0.005 else "worsening" if change_in_unemployment > 0.005 else "stable"
            },
            "wage_dynamics": {
                "initial_average_wage": initial_avg_wage,
                "final_average_wage": final_avg_wage,
                "change_in_average_wage": change_in_avg_wage,
                "wage_growth_pct": (change_in_avg_wage / initial_avg_wage * 100) if initial_avg_wage > 0 else 0
            },
            "impact_of_sme_policy": {
                "total_jobs_estimated_from_smes_during_sim": total_sme_jobs_sim,
                "contribution_of_smes_to_employment": "significant" if total_sme_jobs_sim > (self.total_workforce * 0.01 * (total_epochs-1)) else "moderate" # If SME jobs are >1% of workforce over sim duration
            },
            "trends_data": {
                "unemployment_rate_over_time": self.unemployment_rate_history,
                "average_wage_over_time": self.avg_wage_history,
                "sme_job_creation_over_time": self.job_creation_sme_history
            }
        }

    def get_state_for_persistence(self) -> Dict[str, Any]:
        return {
            "current_unemployment_rate": self.current_unemployment_rate,
            "total_employed_count": self.total_employed,
            "total_workforce_count": self.total_workforce,
            "average_wage_level": self.avg_wage_history[-1] if self.avg_wage_history else 0,
            "jobs_created_by_smes_current_epoch": self.job_creation_sme_history[-1] if self.job_creation_sme_history else 0,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 