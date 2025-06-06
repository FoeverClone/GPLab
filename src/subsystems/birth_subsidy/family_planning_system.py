import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class FamilyPlanningSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Fertility model parameters
        self.peak_fertility_age = config.get("peak_fertility_age", 28)
        self.fertility_age_range = config.get("fertility_age_range", [20, 45])
        self.base_fertility_rate = config.get("base_fertility_rate", 0.05)
        
        # Social factors tracking
        self.social_norm_fertility = 1.5  # Initial societal norm for number of children
        self.work_life_balance_score = 0.5  # Score from 0-1 indicating work-life balance quality
        
        # Peer influence network
        self.peer_networks = defaultdict(list)  # agent_id -> list of peer agent_ids
        
        # Historical tracking
        self.social_norm_history = []
        self.work_life_balance_history = []
        
        self.logger.info("FamilyPlanningSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize peer networks and social factors"""
        all_agent_ids = [str(agent_data.get("id")) for agent_data in all_agent_data]
        
        # Build simplified peer networks based on age proximity
        age_groups = defaultdict(list)
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            age = agent_data.get("basic_info", {}).get("age", 30)
            age_group = age // 10  # Group by decade
            age_groups[age_group].append(agent_id)
            
            # Store age for reference
            self.system_state[f"age_{agent_id}"] = age
            
            # Store gender for reference
            gender = agent_data.get("basic_info", {}).get("gender", "").lower()
            self.system_state[f"gender_{agent_id}"] = gender
        
        # Assign peers from same age group and adjacent age groups
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            age = agent_data.get("basic_info", {}).get("age", 30)
            age_group = age // 10
            
            # Add peers from same age group
            potential_peers = list(age_groups[age_group])
            if agent_id in potential_peers:
                potential_peers.remove(agent_id)
            
            # Add some peers from adjacent age groups
            if age_group > 1 and age_group-1 in age_groups:
                potential_peers.extend(random.sample(age_groups[age_group-1], 
                                                  min(3, len(age_groups[age_group-1]))))
            if age_group < 9 and age_group+1 in age_groups:
                potential_peers.extend(random.sample(age_groups[age_group+1], 
                                                  min(3, len(age_groups[age_group+1]))))
            
            # Select actual peers
            if potential_peers:
                self.peer_networks[agent_id] = random.sample(potential_peers, 
                                                     min(10, len(potential_peers)))
        
        self.logger.info(f"Initialized peer networks for {len(all_agent_ids)} agents")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide social factors information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Get agent's peers' fertility intentions from DemographicSystem
        peer_fertility = []
        for peer_id in self.peer_networks.get(agent_id, []):
            peer_intention = self.system_state.get(f"fertility_intention_{peer_id}", None)
            if peer_intention is not None:
                peer_fertility.append(peer_intention)
        
        avg_peer_fertility = np.mean(peer_fertility) if peer_fertility else 0.3
        
        # Calculate work-life balance based on policy epoch
        work_life_balance = self.work_life_balance_score
        
        # Apply age-based factors
        age = self.system_state.get(f"age_{agent_id}", 30)
        gender = self.system_state.get(f"gender_{agent_id}", "unknown")
        
        # Age-based fertility factors
        age_fertility_factor = 0.0
        if self.fertility_age_range[0] <= age <= self.fertility_age_range[1]:
            # Peak fertility at peak_fertility_age, declining on either side
            age_distance = abs(age - self.peak_fertility_age)
            age_range = (self.fertility_age_range[1] - self.fertility_age_range[0]) / 2
            age_fertility_factor = 1.0 - (age_distance / age_range)
            age_fertility_factor = max(0.0, min(1.0, age_fertility_factor))
        
        return {
            "social_norms": {
                "societal_fertility_norm": self.social_norm_fertility,
                "peer_group_fertility": avg_peer_fertility,
                "age_appropriate_fertility": age_fertility_factor
            },
            "peer_fertility": {
                "peers_with_children": len([p for p in peer_fertility if p > 0.5]),
                "peers_planning_children": len([p for p in peer_fertility if 0.2 < p <= 0.5]),
                "social_influence_factor": 0.7 if avg_peer_fertility > 0.4 else 0.3
            },
            "work_life_balance": {
                "parental_leave_quality": work_life_balance,
                "career_impact": "moderate" if gender == "female" and work_life_balance < 0.7 else "low",
                "childcare_availability": "improved" if current_epoch >= 4 else "limited"
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process social factors related to family planning"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Calculate average fertility intention from demographic decisions
        fertility_intentions = []
        for agent_id in self.system_state:
            if agent_id.startswith("fertility_intention_"):
                intention = self.system_state[agent_id]
                if intention is not None:
                    fertility_intentions.append(intention)
        
        avg_fertility_intention = np.mean(fertility_intentions) if fertility_intentions else 0.3
        
        # Update social norms based on policy effects and current fertility intentions
        base_norm_change = 0.0
        
        # Policy introduction effect
        if current_epoch == 2:
            base_norm_change = 0.1
        elif current_epoch == 3:
            base_norm_change = 0.2
        elif current_epoch >= 4:
            base_norm_change = 0.3
        
        # Fertility intention feedback effect (higher intentions reinforce norm change)
        intention_effect = 0.0
        if avg_fertility_intention > 0.3:
            intention_effect = 0.05
        
        # Calculate new social norm
        target_norm = 1.5 + base_norm_change + intention_effect
        
        # Apply gradual change (don't jump immediately)
        self.social_norm_fertility = self.social_norm_fertility * 0.7 + target_norm * 0.3
        
        # Update work-life balance based on policy
        target_balance = 0.5  # Base value
        if current_epoch >= 2:
            target_balance += 0.1  # Initial policy effect
        if current_epoch >= 4:
            target_balance += 0.1  # Enhanced policy effect
            
        # Apply gradual change
        self.work_life_balance_score = self.work_life_balance_score * 0.7 + target_balance * 0.3
        
        # Share updated social factors with DemographicSystem
        self.system_state["social_norm_fertility"] = self.social_norm_fertility
        self.system_state["work_life_balance"] = self.work_life_balance_score
        
        # Record historical data
        self.social_norm_history.append(self.social_norm_fertility)
        self.work_life_balance_history.append(self.work_life_balance_score)
        
        self.logger.info(f"Epoch {current_epoch}: Social norm fertility={self.social_norm_fertility:.1f}, "
                        f"Work-life balance={self.work_life_balance_score:.1f}, "
                        f"Avg fertility intention={avg_fertility_intention:.2f}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate social aspects of birth subsidy policy with simplified metrics"""
        # Ensure we have data to evaluate
        if len(self.social_norm_history) < 6:
            self.logger.warning("Not enough data for proper evaluation")
            return {"error": "Insufficient data for evaluation"}
        
        # Calculate changes in social norms
        initial_norm = self.social_norm_history[0] if self.social_norm_history else 1.5
        final_norm = self.social_norm_history[-1] if self.social_norm_history else 1.5
        norm_change = final_norm - initial_norm
        
        # Calculate work-life balance improvement
        initial_balance = self.work_life_balance_history[0] if self.work_life_balance_history else 0.5
        final_balance = self.work_life_balance_history[-1] if self.work_life_balance_history else 0.5
        balance_change = final_balance - initial_balance
        
        # Count agents with high fertility intentions by gender and age groups
        intention_by_gender = {"female": 0, "male": 0, "other": 0}
        intention_by_age_group = {"20s": 0, "30s": 0, "40s": 0, "other": 0}
        total_by_gender = {"female": 0, "male": 0, "other": 0}
        total_by_age_group = {"20s": 0, "30s": 0, "40s": 0, "other": 0}
        
        for agent_id in self.system_state:
            if agent_id.startswith("fertility_intention_"):
                agent_id_clean = agent_id.replace("fertility_intention_", "")
                intention = self.system_state[agent_id]
                if intention is None:
                    continue
                
                # Get age and gender
                age = self.system_state.get(f"age_{agent_id_clean}", 0)
                gender = self.system_state.get(f"gender_{agent_id_clean}", "other")
                
                # Count by gender
                if gender in total_by_gender:
                    total_by_gender[gender] += 1
                    if intention > 0.5:
                        intention_by_gender[gender] += 1
                else:
                    total_by_gender["other"] += 1
                    if intention > 0.5:
                        intention_by_gender["other"] += 1
                
                # Count by age group
                if 20 <= age < 30:
                    total_by_age_group["20s"] += 1
                    if intention > 0.5:
                        intention_by_age_group["20s"] += 1
                elif 30 <= age < 40:
                    total_by_age_group["30s"] += 1
                    if intention > 0.5:
                        intention_by_age_group["30s"] += 1
                elif 40 <= age < 50:
                    total_by_age_group["40s"] += 1
                    if intention > 0.5:
                        intention_by_age_group["40s"] += 1
                else:
                    total_by_age_group["other"] += 1
                    if intention > 0.5:
                        intention_by_age_group["other"] += 1
        
        # Calculate percentages
        pct_by_gender = {
            gender: float((intention_by_gender[gender] / total_by_gender[gender] * 100) if total_by_gender[gender] > 0 else 0)
            for gender in intention_by_gender
        }
        
        pct_by_age_group = {
            age_group: float((intention_by_age_group[age_group] / total_by_age_group[age_group] * 100) if total_by_age_group[age_group] > 0 else 0)
            for age_group in intention_by_age_group
        }
        
        # Simplified evaluation results
        evaluation_results = {
            "social_norm_metrics": {
                "initial_fertility_norm": float(initial_norm),
                "final_fertility_norm": float(final_norm),
                "norm_change": float(norm_change)
            },
            "work_life_balance_metrics": {
                "initial_work_life_balance": float(initial_balance),
                "final_work_life_balance": float(final_balance),
                "balance_improvement": float(balance_change)
            },
            "demographic_breakdown": {
                "high_intention_by_gender_pct": pct_by_gender,
                "high_intention_by_age_group_pct": pct_by_age_group,
                "most_responsive_group": max(pct_by_age_group.items(), key=lambda x: x[1])[0] if pct_by_age_group else "none"
            },
            "time_series": {
                "social_norm_history": [float(x) for x in self.social_norm_history],
                "work_life_balance_history": [float(x) for x in self.work_life_balance_history]
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")
        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "social_norm_fertility": self.social_norm_fertility,
            "work_life_balance_score": self.work_life_balance_score,
            "avg_peer_network_size": sum(len(peers) for peers in self.peer_networks.values()) / len(self.peer_networks) if self.peer_networks else 0,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 