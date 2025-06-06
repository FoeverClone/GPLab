import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class CitizenAdoptionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Technology adoption parameters
        self.innovation_diffusion_rate = config.get("innovation_diffusion_rate", 0.15)
        self.age_adoption_factors = config.get("age_adoption_factors", {
            "18-30": 0.9,
            "31-45": 0.7,
            "46-60": 0.5,
            "61+": 0.3
        })
        
        self.education_adoption_boost = config.get("education_adoption_boost", {
            "high_school": 0.1,
            "college": 0.2,
            "graduate": 0.3
        })
        
        # Adoption tracking
        self.agent_adoption_levels = {}  # agent_id -> adoption level (0-1)
        self.agent_satisfaction = {}     # agent_id -> satisfaction score (0-1)
        self.agent_barriers = defaultdict(list)  # agent_id -> list of adoption barriers
        
        # Digital divide metrics
        self.age_group_adoption = defaultdict(list)  # age group -> list of adoption levels
        self.education_group_adoption = defaultdict(list)  # education level -> list of adoption levels
        self.rural_urban_adoption = {"urban": [], "suburban": [], "rural": []}  # area type -> list of adoption levels
        
        # Historical data
        self.overall_adoption_history = []
        self.satisfaction_history = []
        self.digital_divide_score_history = []
        
        self.logger.info("CitizenAdoptionSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent adoption profiles"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Extract demographic data
            age = agent_data.get("basic_info", {}).get("age", 30)
            education = agent_data.get("basic_info", {}).get("education_level", "").lower()
            residence = agent_data.get("basic_info", {}).get("residence_type", "").lower()
            
            # Store for reference
            self.system_state[f"age_{agent_id}"] = age
            self.system_state[f"education_{agent_id}"] = education
            self.system_state[f"residence_{agent_id}"] = residence
            
            # Calculate initial adoption probability based on demographics
            age_factor = 0
            if age < 30:
                age_factor = self.age_adoption_factors.get("18-30", 0.9)
            elif age < 45:
                age_factor = self.age_adoption_factors.get("31-45", 0.7)
            elif age < 60:
                age_factor = self.age_adoption_factors.get("46-60", 0.5)
            else:
                age_factor = self.age_adoption_factors.get("61+", 0.3)
                
            # Education boost
            education_boost = 0
            if "college" in education or "university" in education:
                education_boost = self.education_adoption_boost.get("college", 0.2)
            elif "graduate" in education or "postgraduate" in education:
                education_boost = self.education_adoption_boost.get("graduate", 0.3)
            elif "high school" in education or "secondary" in education:
                education_boost = self.education_adoption_boost.get("high_school", 0.1)
                
            # Residence factor (digital divide)
            residence_factor = 0
            if "urban" in residence:
                residence_factor = 0.1
            elif "suburban" in residence:
                residence_factor = 0
            else:  # rural
                residence_factor = -0.2
                
            # Calculate initial adoption level with some randomness
            base_adoption = age_factor + education_boost + residence_factor
            random_factor = random.uniform(-0.1, 0.1)
            initial_adoption = max(0.05, min(0.95, base_adoption + random_factor))
            
            self.agent_adoption_levels[agent_id] = initial_adoption
            
            # Initialize satisfaction with some correlation to adoption
            initial_satisfaction = initial_adoption * 0.8 + random.uniform(0, 0.2)
            self.agent_satisfaction[agent_id] = initial_satisfaction
            
            # Identify adoption barriers
            barriers = []
            if age > 60:
                barriers.append("age_related_digital_literacy")
            if education_boost < 0.2:
                barriers.append("technical_skill_gap")
            if residence_factor < 0:
                barriers.append("connectivity_issues")
            if random.random() < 0.3:
                barriers.append("trust_concerns")
                
            self.agent_barriers[agent_id] = barriers
            
            # Categorize for digital divide analysis
            age_group = "18-30" if age < 30 else "31-45" if age < 45 else "46-60" if age < 60 else "61+"
            self.age_group_adoption[age_group].append(initial_adoption)
            
            education_group = "basic" if education_boost == 0 else "high_school" if education_boost == 0.1 else "college" if education_boost == 0.2 else "graduate"
            self.education_group_adoption[education_group].append(initial_adoption)
            
            area_type = "urban" if "urban" in residence else "suburban" if "suburban" in residence else "rural"
            self.rural_urban_adoption[area_type].append(initial_adoption)
        
        self.logger.info(f"Initialized adoption profiles for {len(all_agent_data)} citizens")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide adoption statistics and digital divide information"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Calculate agent's position relative to peers
        adoption_level = self.agent_adoption_levels.get(agent_id, 0.5)
        
        # Calculate average adoption by demographic groups
        age_groups_avg = {group: np.mean(levels) for group, levels in self.age_group_adoption.items()}
        education_groups_avg = {group: np.mean(levels) for group, levels in self.education_group_adoption.items()}
        area_type_avg = {area: np.mean(levels) for area, levels in self.rural_urban_adoption.items()}
        
        # Calculate overall adoption rate
        overall_adoption = np.mean(list(self.agent_adoption_levels.values())) if self.agent_adoption_levels else 0
        
        # Calculate agent's demographic groups
        age = self.system_state.get(f"age_{agent_id}", 30)
        education = self.system_state.get(f"education_{agent_id}", "").lower()
        residence = self.system_state.get(f"residence_{agent_id}", "").lower()
        
        age_group = "18-30" if age < 30 else "31-45" if age < 45 else "46-60" if age < 60 else "61+"
        education_group = "basic"
        if "college" in education or "university" in education:
            education_group = "college"
        elif "graduate" in education or "postgraduate" in education:
            education_group = "graduate"
        elif "high school" in education or "secondary" in education:
            education_group = "high_school"
            
        area_type = "urban" if "urban" in residence else "suburban" if "suburban" in residence else "rural"
        
        # Calculate digital divide score
        max_group_adoption = max(age_groups_avg.values())
        min_group_adoption = min(age_groups_avg.values())
        age_divide_score = (max_group_adoption - min_group_adoption) / max_group_adoption if max_group_adoption > 0 else 0
        
        max_edu_adoption = max(education_groups_avg.values())
        min_edu_adoption = min(education_groups_avg.values())
        education_divide_score = (max_edu_adoption - min_edu_adoption) / max_edu_adoption if max_edu_adoption > 0 else 0
        
        max_area_adoption = max(area_type_avg.values())
        min_area_adoption = min(area_type_avg.values())
        geographic_divide_score = (max_area_adoption - min_area_adoption) / max_area_adoption if max_area_adoption > 0 else 0
        
        overall_divide_score = (age_divide_score + education_divide_score + geographic_divide_score) / 3
        
        # Specific barriers for this agent
        agent_barriers = self.agent_barriers.get(agent_id, [])
        
        return {
            "adoption_rates": {
                "overall_adoption_rate": overall_adoption,
                "your_adoption_level": adoption_level,
                "your_demographic_adoption": {
                    "age_group": f"{age_group}: {age_groups_avg.get(age_group, 0):.2f}",
                    "education_level": f"{education_group}: {education_groups_avg.get(education_group, 0):.2f}",
                    "area_type": f"{area_type}: {area_type_avg.get(area_type, 0):.2f}"
                },
                "adoption_trend": "increasing" if len(self.overall_adoption_history) > 1 and self.overall_adoption_history[-1] > self.overall_adoption_history[0] else "stable"
            },
            "user_satisfaction": {
                "average_satisfaction": np.mean(list(self.agent_satisfaction.values())) if self.agent_satisfaction else 0.5,
                "your_satisfaction": self.agent_satisfaction.get(agent_id, 0.5),
                "satisfaction_by_age_group": {group: self._calculate_group_satisfaction(group, "age") for group in self.age_group_adoption},
                "satisfaction_trend": "improving" if len(self.satisfaction_history) > 1 and self.satisfaction_history[-1] > self.satisfaction_history[0] else "stable"
            },
            "digital_divide": {
                "overall_divide_score": overall_divide_score,
                "age_divide_score": age_divide_score,
                "education_divide_score": education_divide_score,
                "geographic_divide_score": geographic_divide_score,
                "your_adoption_barriers": agent_barriers,
                "divide_trend": "decreasing" if len(self.digital_divide_score_history) > 1 and self.digital_divide_score_history[-1] < self.digital_divide_score_history[0] else "persistent"
            }
        }
    
    def _calculate_group_satisfaction(self, group: str, group_type: str) -> float:
        """Calculate average satisfaction for a demographic group"""
        group_agents = []
        
        if group_type == "age":
            for agent_id in self.agent_adoption_levels:
                age = self.system_state.get(f"age_{agent_id}", 30)
                agent_group = "18-30" if age < 30 else "31-45" if age < 45 else "46-60" if age < 60 else "61+"
                if agent_group == group:
                    group_agents.append(agent_id)
        
        elif group_type == "education":
            for agent_id in self.agent_adoption_levels:
                education = self.system_state.get(f"education_{agent_id}", "").lower()
                if group == "basic" and not any(term in education for term in ["high school", "secondary", "college", "university", "graduate", "postgraduate"]):
                    group_agents.append(agent_id)
                elif group == "high_school" and any(term in education for term in ["high school", "secondary"]):
                    group_agents.append(agent_id)
                elif group == "college" and any(term in education for term in ["college", "university"]):
                    group_agents.append(agent_id)
                elif group == "graduate" and any(term in education for term in ["graduate", "postgraduate"]):
                    group_agents.append(agent_id)
        
        elif group_type == "area":
            for agent_id in self.agent_adoption_levels:
                residence = self.system_state.get(f"residence_{agent_id}", "").lower()
                area_type = "urban" if "urban" in residence else "suburban" if "suburban" in residence else "rural"
                if area_type == group:
                    group_agents.append(agent_id)
        
        # Calculate average satisfaction
        group_satisfaction = [self.agent_satisfaction.get(agent_id, 0.5) for agent_id in group_agents]
        return np.mean(group_satisfaction) if group_satisfaction else 0.5

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process citizen adoption dynamics"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Reset demographic tracking for this epoch
        self.age_group_adoption = defaultdict(list)
        self.education_group_adoption = defaultdict(list)
        self.rural_urban_adoption = {"urban": [], "suburban": [], "rural": []}
        
        # Extract e-government decisions and feedback
        for agent_id, decisions in agent_decisions.items():
            # Get agent demographics
            age = self.system_state.get(f"age_{agent_id}", 30)
            education = self.system_state.get(f"education_{agent_id}", "").lower()
            residence = self.system_state.get(f"residence_{agent_id}", "").lower()
            
            # Process EGovernmentSystem decisions if available
            digital_adoption = 0
            feedback_type = "none"
            
            if "EGovernmentSystem" in decisions:
                decision = decisions["EGovernmentSystem"]
                channel_choice = decision.get("service_channel_choice", "offline_only")
                digital_adoption = float(decision.get("digital_adoption_level", 0))
                feedback_type = decision.get("feedback_type", "none")
            
            # If no explicit decision, use previous adoption level with innovation diffusion
            if digital_adoption == 0:
                current_adoption = self.agent_adoption_levels.get(agent_id, 0.5)
                # Apply innovation diffusion (Rogers' theory)
                peers_influence = self._calculate_peer_influence(agent_id)
                digital_adoption = current_adoption + (peers_influence - current_adoption) * self.innovation_diffusion_rate
            
            # Update adoption level
            self.agent_adoption_levels[agent_id] = digital_adoption
            
            # Update satisfaction based on feedback
            current_satisfaction = self.agent_satisfaction.get(agent_id, 0.5)
            if feedback_type == "positive":
                satisfaction_adjustment = random.uniform(0.05, 0.15)
            elif feedback_type == "negative":
                satisfaction_adjustment = random.uniform(-0.15, -0.05)
            elif feedback_type == "constructive":
                satisfaction_adjustment = random.uniform(-0.05, 0.05)
            else:
                satisfaction_adjustment = 0
                
            new_satisfaction = max(0.1, min(0.9, current_satisfaction + satisfaction_adjustment))
            self.agent_satisfaction[agent_id] = new_satisfaction
            
            # Update demographic tracking
            age_group = "18-30" if age < 30 else "31-45" if age < 45 else "46-60" if age < 60 else "61+"
            self.age_group_adoption[age_group].append(digital_adoption)
            
            education_group = "basic"
            if "college" in education or "university" in education:
                education_group = "college"
            elif "graduate" in education or "postgraduate" in education:
                education_group = "graduate"
            elif "high school" in education or "secondary" in education:
                education_group = "high_school"
            self.education_group_adoption[education_group].append(digital_adoption)
            
            area_type = "urban" if "urban" in residence else "suburban" if "suburban" in residence else "rural"
            self.rural_urban_adoption[area_type].append(digital_adoption)
        
        # Calculate overall metrics
        overall_adoption = np.mean(list(self.agent_adoption_levels.values())) if self.agent_adoption_levels else 0
        avg_satisfaction = np.mean(list(self.agent_satisfaction.values())) if self.agent_satisfaction else 0
        
        # Calculate digital divide score
        age_groups_avg = {group: np.mean(levels) for group, levels in self.age_group_adoption.items() if levels}
        education_groups_avg = {group: np.mean(levels) for group, levels in self.education_group_adoption.items() if levels}
        area_type_avg = {area: np.mean(levels) for area, levels in self.rural_urban_adoption.items() if levels}
        
        max_group_adoption = max(age_groups_avg.values()) if age_groups_avg else 0
        min_group_adoption = min(age_groups_avg.values()) if age_groups_avg else 0
        age_divide_score = (max_group_adoption - min_group_adoption) / max_group_adoption if max_group_adoption > 0 else 0
        
        max_edu_adoption = max(education_groups_avg.values()) if education_groups_avg else 0
        min_edu_adoption = min(education_groups_avg.values()) if education_groups_avg else 0
        education_divide_score = (max_edu_adoption - min_edu_adoption) / max_edu_adoption if max_edu_adoption > 0 else 0
        
        max_area_adoption = max(area_type_avg.values()) if area_type_avg else 0
        min_area_adoption = min(area_type_avg.values()) if area_type_avg else 0
        geographic_divide_score = (max_area_adoption - min_area_adoption) / max_area_adoption if max_area_adoption > 0 else 0
        
        overall_divide_score = (age_divide_score + education_divide_score + geographic_divide_score) / 3
        
        # Record historical data
        self.overall_adoption_history.append(overall_adoption)
        self.satisfaction_history.append(avg_satisfaction)
        self.digital_divide_score_history.append(overall_divide_score)
        
        self.logger.info(f"Epoch {current_epoch}: Overall adoption={overall_adoption:.2f}, "
                        f"Satisfaction={avg_satisfaction:.2f}, "
                        f"Digital divide={overall_divide_score:.2f}")
    
    def _calculate_peer_influence(self, agent_id: str) -> float:
        """Calculate peer influence on adoption using a simplified network model"""
        # In a real implementation, we would use social network data
        # Here we use demographic similarity as a proxy for social connections
        
        agent_age = self.system_state.get(f"age_{agent_id}", 30)
        agent_education = self.system_state.get(f"education_{agent_id}", "").lower()
        agent_residence = self.system_state.get(f"residence_{agent_id}", "").lower()
        
        peer_adoptions = []
        
        # Find peers with similar demographics
        for peer_id, adoption in self.agent_adoption_levels.items():
            if peer_id == agent_id:
                continue
                
            peer_age = self.system_state.get(f"age_{peer_id}", 30)
            peer_education = self.system_state.get(f"education_{peer_id}", "").lower()
            peer_residence = self.system_state.get(f"residence_{peer_id}", "").lower()
            
            # Calculate demographic similarity
            age_similarity = 1.0 - min(abs(agent_age - peer_age) / 50, 1.0)
            education_similarity = 1.0 if agent_education == peer_education else 0.5
            residence_similarity = 1.0 if agent_residence == peer_residence else 0.5
            
            # Overall similarity
            similarity = (age_similarity + education_similarity + residence_similarity) / 3
            
            # If similarity is high enough, consider as peer
            if similarity > 0.6:
                # Weight peer adoption by similarity
                peer_adoptions.append(adoption * similarity)
        
        # Return average peer adoption or 0.5 if no peers
        return np.mean(peer_adoptions) if peer_adoptions else 0.5

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate citizen adoption patterns and digital divide"""
        # Calculate adoption growth
        initial_adoption = self.overall_adoption_history[0] if self.overall_adoption_history else 0
        final_adoption = self.overall_adoption_history[-1] if self.overall_adoption_history else 0
        adoption_growth = final_adoption - initial_adoption
        
        # Calculate satisfaction change
        initial_satisfaction = self.satisfaction_history[0] if self.satisfaction_history else 0
        final_satisfaction = self.satisfaction_history[-1] if self.satisfaction_history else 0
        satisfaction_change = final_satisfaction - initial_satisfaction
        
        # Calculate digital divide improvement
        initial_divide = self.digital_divide_score_history[0] if self.digital_divide_score_history else 1
        final_divide = self.digital_divide_score_history[-1] if self.digital_divide_score_history else 1
        divide_improvement = initial_divide - final_divide  # Lower score is better
        
        # Analyze adoption by demographic groups
        age_groups_avg = {group: np.mean(levels) for group, levels in self.age_group_adoption.items() if levels}
        education_groups_avg = {group: np.mean(levels) for group, levels in self.education_group_adoption.items() if levels}
        area_type_avg = {area: np.mean(levels) for area, levels in self.rural_urban_adoption.items() if levels}
        
        # Find most common barriers
        all_barriers = []
        for barriers in self.agent_barriers.values():
            all_barriers.extend(barriers)
            
        barrier_counts = defaultdict(int)
        for barrier in all_barriers:
            barrier_counts[barrier] += 1
            
        top_barriers = sorted([(barrier, count) for barrier, count in barrier_counts.items()], 
                             key=lambda x: x[1], reverse=True)
        
        # Calculate adoption phases based on Rogers' diffusion theory
        adoptions = list(self.agent_adoption_levels.values())
        innovators = sum(1 for a in adoptions if a > 0.8) / len(adoptions) if adoptions else 0
        early_adopters = sum(1 for a in adoptions if 0.6 < a <= 0.8) / len(adoptions) if adoptions else 0
        early_majority = sum(1 for a in adoptions if 0.4 < a <= 0.6) / len(adoptions) if adoptions else 0
        late_majority = sum(1 for a in adoptions if 0.2 < a <= 0.4) / len(adoptions) if adoptions else 0
        laggards = sum(1 for a in adoptions if a <= 0.2) / len(adoptions) if adoptions else 0
        
        evaluation_results = {
            "adoption_metrics": {
                "initial_adoption_rate": initial_adoption,
                "final_adoption_rate": final_adoption,
                "adoption_growth": adoption_growth,
                "diffusion_stages": {
                    "innovators": innovators,
                    "early_adopters": early_adopters,
                    "early_majority": early_majority,
                    "late_majority": late_majority,
                    "laggards": laggards
                }
            },
            "satisfaction_metrics": {
                "initial_satisfaction": initial_satisfaction,
                "final_satisfaction": final_satisfaction,
                "satisfaction_improvement": satisfaction_change
            },
            "digital_divide_metrics": {
                "initial_divide_score": initial_divide,
                "final_divide_score": final_divide,
                "divide_improvement": divide_improvement,
                "demographic_gaps": {
                    "age_gap": max(age_groups_avg.values()) - min(age_groups_avg.values()) if age_groups_avg else 0,
                    "education_gap": max(education_groups_avg.values()) - min(education_groups_avg.values()) if education_groups_avg else 0,
                    "urban_rural_gap": max(area_type_avg.values()) - min(area_type_avg.values()) if area_type_avg else 0
                }
            },
            "barrier_analysis": {
                "top_barriers": [{"barrier": b[0], "count": b[1]} for b in top_barriers[:3]],
                "barriers_per_agent": len(all_barriers) / len(self.agent_barriers) if self.agent_barriers else 0
            },
            "demographic_adoption": {
                "by_age_group": age_groups_avg,
                "by_education": education_groups_avg,
                "by_area_type": area_type_avg,
                "most_adoptive_group": max(age_groups_avg.items(), key=lambda x: x[1])[0] if age_groups_avg else "none",
                "least_adoptive_group": min(age_groups_avg.items(), key=lambda x: x[1])[0] if age_groups_avg else "none"
            },
            "policy_effectiveness": {
                "achieved_high_adoption": final_adoption > 0.7,
                "reduced_digital_divide": divide_improvement > 0.1,
                "improved_satisfaction": satisfaction_change > 0.1,
                "overall_success": final_adoption > 0.7 and divide_improvement > 0.1 and satisfaction_change > 0.1
            },
            "time_series": {
                "adoption_history": self.overall_adoption_history,
                "satisfaction_history": self.satisfaction_history,
                "digital_divide_history": self.digital_divide_score_history
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")

        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        overall_adoption = np.mean(list(self.agent_adoption_levels.values())) if self.agent_adoption_levels else 0
        avg_satisfaction = np.mean(list(self.agent_satisfaction.values())) if self.agent_satisfaction else 0
        
        # Calculate digital divide score
        age_groups_avg = {group: np.mean(levels) for group, levels in self.age_group_adoption.items() if levels}
        education_groups_avg = {group: np.mean(levels) for group, levels in self.education_group_adoption.items() if levels}
        area_type_avg = {area: np.mean(levels) for area, levels in self.rural_urban_adoption.items() if levels}
        
        max_group_adoption = max(age_groups_avg.values()) if age_groups_avg else 0
        min_group_adoption = min(age_groups_avg.values()) if age_groups_avg else 0
        age_divide_score = (max_group_adoption - min_group_adoption) / max_group_adoption if max_group_adoption > 0 else 0
        
        max_edu_adoption = max(education_groups_avg.values()) if education_groups_avg else 0
        min_edu_adoption = min(education_groups_avg.values()) if education_groups_avg else 0
        education_divide_score = (max_edu_adoption - min_edu_adoption) / max_edu_adoption if max_edu_adoption > 0 else 0
        
        max_area_adoption = max(area_type_avg.values()) if area_type_avg else 0
        min_area_adoption = min(area_type_avg.values()) if area_type_avg else 0
        geographic_divide_score = (max_area_adoption - min_area_adoption) / max_area_adoption if max_area_adoption > 0 else 0
        
        overall_divide_score = (age_divide_score + education_divide_score + geographic_divide_score) / 3
        
        return {
            "overall_adoption_rate": overall_adoption,
            "average_satisfaction": avg_satisfaction,
            "digital_divide_score": overall_divide_score,
            "age_divide_score": age_divide_score,
            "education_divide_score": education_divide_score,
            "geographic_divide_score": geographic_divide_score,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 