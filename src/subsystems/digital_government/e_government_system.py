import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class EGovernmentSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Digital government policy parameters
        self.digital_policies = config.get("digital_policies", {})
        
        # Service tracking
        self.available_services = []
        self.platform_quality = 0.5
        self.mobile_app_available = False
        self.ai_assistant_available = False
        
        # Usage statistics
        self.service_usage = defaultdict(int)  # service_name -> usage count
        self.channel_usage = defaultdict(int)  # channel -> usage count
        self.agent_digital_usage = {}  # agent_id -> digital service usage count
        self.agent_feedback = {}       # agent_id -> feedback type count
        
        # Historical data
        self.digital_adoption_rate_history = []
        self.service_satisfaction_history = []
        self.platform_quality_history = []
        
        self.logger.info("EGovernmentSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent digital profiles"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Calculate digital literacy based on age and education
            age = agent_data.get("basic_info", {}).get("age", 30)
            education = agent_data.get("basic_info", {}).get("education_level", "").lower()
            
            # Base digital literacy score
            if "graduate" in education or "postgraduate" in education:
                literacy_base = 0.8
            elif "college" in education or "university" in education:
                literacy_base = 0.7
            elif "high school" in education or "secondary" in education:
                literacy_base = 0.5
            else:
                literacy_base = 0.3
            
            # Age adjustment
            if age < 30:
                age_factor = 0.2
            elif age < 50:
                age_factor = 0.0
            elif age < 65:
                age_factor = -0.2
            else:
                age_factor = -0.4
                
            digital_literacy = max(0.1, min(0.9, literacy_base + age_factor))
            self.system_state[f"digital_literacy_{agent_id}"] = digital_literacy
            
            # Initialize usage counter
            self.agent_digital_usage[agent_id] = 0
        
        self.logger.info(f"Initialized digital profiles for {len(all_agent_data)} agents")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide digital government service information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.digital_policies.get(str(current_epoch), 
                                                self.digital_policies.get("0", {}))
        
        # Update available services based on policy
        self.available_services = current_policy.get("services_online", [])
        self.platform_quality = current_policy.get("platform_quality", 0.5)
        self.mobile_app_available = current_policy.get("mobile_app", False)
        self.ai_assistant_available = current_policy.get("ai_assistant", False)
        
        # Agent's usage history
        digital_usage = self.agent_digital_usage.get(agent_id, 0)
        
        # Calculate service availability metrics
        service_coverage = len(self.available_services) / 10  # Assuming 10 is max services
        
        # Determine service quality based on platform quality and agent's digital literacy
        agent_digital_literacy = self.system_state.get(f"digital_literacy_{agent_id}", 0.5)
        
        # Quality experienced is influenced by platform quality and user's literacy
        experienced_quality = (self.platform_quality * 0.7) + (agent_digital_literacy * 0.3)
        
        # User support information
        user_support = current_policy.get("user_support", "limited")
        support_quality = {"limited": 0.3, "enhanced": 0.6, "comprehensive": 0.8, "24/7": 0.9}.get(user_support, 0.3)
        
        # Platform features based on policy
        features = ["basic_online_forms"]
        if self.mobile_app_available:
            features.append("mobile_app")
        if self.ai_assistant_available:
            features.append("ai_assistant")
        if current_policy.get("biometric_authentication", False):
            features.append("biometric_authentication")
            
        # Calculate wait time reduction compared to offline
        wait_time_reduction = 0
        if "all_services" in self.available_services:
            wait_time_reduction = 0.9
        elif len(self.available_services) > 3:
            wait_time_reduction = 0.7
        elif len(self.available_services) > 1:
            wait_time_reduction = 0.5
        else:
            wait_time_reduction = 0.3
            
        # Check if policy has changed
        policy_changed = False
        policy_announcement = ""
        if current_epoch > 0:
            prev_policy = self.digital_policies.get(str(current_epoch - 1), {})
            if prev_policy != current_policy:
                policy_changed = True
                if current_epoch == 2:
                    policy_announcement = "Major digital government initiative launched with new online services"
                elif current_epoch == 4:
                    policy_announcement = "Full digital transformation with all services now available online"
        
        return {
            "available_services": {
                "services_list": self.available_services,
                "coverage_percentage": service_coverage * 100,
                "mobile_app_available": self.mobile_app_available,
                "policy_announcement": policy_announcement
            },
            "platform_features": {
                "features_list": features,
                "platform_quality_score": self.platform_quality,
                "wait_time_reduction": wait_time_reduction
            },
            "support_resources": {
                "support_type": user_support,
                "ai_assistance": self.ai_assistant_available,
                "digital_literacy_tools": self.platform_quality > 0.7,
                "helpline_quality": support_quality
            },
            "service_quality": {
                "reliability_score": experienced_quality,
                "your_experience_rating": experienced_quality * 5,  # On a 5-point scale
                "estimated_time_savings": f"{int(wait_time_reduction * 100)}% faster than offline",
                "your_usage_count": digital_usage
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent digital government interactions"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.digital_policies.get(str(current_epoch), 
                                                self.digital_policies.get("0", {}))
        
        # Update system state based on policy
        self.available_services = current_policy.get("services_online", [])
        self.platform_quality = current_policy.get("platform_quality", 0.5)
        
        # Reset usage statistics for this epoch
        self.service_usage = defaultdict(int)
        self.channel_usage = defaultdict(int)
        satisfaction_scores = []
        digital_users = 0
        total_agents = len(agent_decisions)
        
        for agent_id, decisions in agent_decisions.items():
            if "EGovernmentSystem" not in decisions:
                continue
                
            decision = decisions["EGovernmentSystem"]
            channel_choice = decision.get("service_channel_choice", "offline_only")
            
            raw_digital_adoption = decision.get("digital_adoption_level")
            digital_adoption = 0.0  # Default value
            if raw_digital_adoption is not None:
                try:
                    digital_adoption = float(raw_digital_adoption)
                except (ValueError, TypeError):
                    self.logger.warning(f"Agent {agent_id} provided invalid digital_adoption_level: '{raw_digital_adoption}'. Using default 0.0.")
            else:
                # This handles both missing key (if .get() didn't have a default) or key present with value None.
                # Given the original had a default in .get(), this 'else' implies the key was present with None,
                # or we want to be explicit about None values.
                self.logger.info(f"Agent {agent_id} provided no or None digital_adoption_level. Using default 0.0.")

            feedback = decision.get("feedback_type", "none")
            
            # Record channel choice
            self.channel_usage[channel_choice] += 1
            
            # Process usage based on channel choice
            if channel_choice in ["online_only", "mixed_preference"]:
                # Agent uses digital services
                service_count = len(self.available_services)
                
                # Calculate how many services the agent uses based on adoption level
                if channel_choice == "online_only":
                    used_services = max(1, int(service_count * 0.8))
                else:  # mixed_preference
                    used_services = max(1, int(service_count * 0.4))
                
                # Update usage statistics
                self.agent_digital_usage[agent_id] = self.agent_digital_usage.get(agent_id, 0) + used_services
                digital_users += 1
                
                # Record which services were used
                for i in range(min(used_services, len(self.available_services))):
                    service = self.available_services[i] if i < len(self.available_services) else "basic_info_query"
                    self.service_usage[service] += 1
            
            # Process feedback
            if feedback != "none":
                self.agent_feedback[agent_id] = feedback
                
                # Calculate satisfaction based on feedback
                satisfaction = 0
                if feedback == "positive":
                    satisfaction = random.uniform(0.7, 1.0)
                elif feedback == "negative":
                    satisfaction = random.uniform(0.1, 0.4)
                elif feedback == "constructive":
                    satisfaction = random.uniform(0.4, 0.7)
                    
                satisfaction_scores.append(satisfaction)
        
        # Calculate digital adoption rate
        digital_adoption_rate = digital_users / total_agents if total_agents > 0 else 0
        
        # Calculate average satisfaction
        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0.5
        
        # Record historical data
        self.digital_adoption_rate_history.append(digital_adoption_rate)
        self.service_satisfaction_history.append(avg_satisfaction)
        self.platform_quality_history.append(self.platform_quality)
        
        self.logger.info(f"Epoch {current_epoch}: Digital adoption={digital_adoption_rate:.2f}, "
                        f"Satisfaction={avg_satisfaction:.2f}, "
                        f"Services={len(self.available_services)}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate digital government effectiveness"""
        # Calculate adoption rate changes
        initial_adoption = self.digital_adoption_rate_history[0] if self.digital_adoption_rate_history else 0
        final_adoption = self.digital_adoption_rate_history[-1] if self.digital_adoption_rate_history else 0
        adoption_change = final_adoption - initial_adoption
        
        # Calculate satisfaction trends
        initial_satisfaction = self.service_satisfaction_history[0] if self.service_satisfaction_history else 0
        final_satisfaction = self.service_satisfaction_history[-1] if self.service_satisfaction_history else 0
        satisfaction_change = final_satisfaction - initial_satisfaction
        
        # Calculate platform quality improvement
        initial_quality = self.platform_quality_history[0] if self.platform_quality_history else 0.5
        final_quality = self.platform_quality_history[-1] if self.platform_quality_history else 0.5
        quality_improvement = final_quality - initial_quality
        
        # Analyze channel preferences
        channel_distribution = dict(self.channel_usage)
        total_users = sum(self.channel_usage.values())
        
        channel_percentages = {}
        for channel, count in self.channel_usage.items():
            channel_percentages[channel] = (count / total_users * 100) if total_users > 0 else 0
            
        # Analyze service popularity
        service_popularity = sorted([(service, count) for service, count in self.service_usage.items()], 
                                   key=lambda x: x[1], reverse=True)
        most_popular_services = [item[0] for item in service_popularity[:3]] if service_popularity else []
        
        # Analyze feedback
        feedback_counts = {"positive": 0, "negative": 0, "constructive": 0, "none": 0}
        for feedback in self.agent_feedback.values():
            feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
            
        # Calculate overall success metrics
        high_adoption = final_adoption > 0.7
        improved_satisfaction = satisfaction_change > 0.2
        quality_sufficient = final_quality > 0.8
        
        evaluation_results = {
            "adoption_metrics": {
                "initial_adoption_rate": initial_adoption,
                "final_adoption_rate": final_adoption,
                "adoption_growth": adoption_change,
                "adoption_by_channel": channel_percentages
            },
            "satisfaction_metrics": {
                "initial_satisfaction": initial_satisfaction,
                "final_satisfaction": final_satisfaction,
                "satisfaction_improvement": satisfaction_change,
                "feedback_distribution": feedback_counts
            },
            "service_metrics": {
                "total_services_available": len(self.available_services),
                "most_popular_services": most_popular_services,
                "platform_quality_improvement": quality_improvement,
                "service_usage_distribution": dict(self.service_usage)
            },
            "efficiency_gains": {
                "estimated_paper_reduction": final_adoption * 0.9 if "all_services" in self.available_services else final_adoption * 0.5,
                "estimated_time_saved_hours": sum(self.agent_digital_usage.values()) * 0.5,  # Assume 0.5 hours saved per digital interaction
                "administrative_cost_reduction": final_adoption * 0.7 if final_quality > 0.8 else final_adoption * 0.4
            },
            "policy_effectiveness": {
                "achieved_high_adoption": high_adoption,
                "improved_user_satisfaction": improved_satisfaction,
                "delivered_quality_platform": quality_sufficient,
                "overall_success": high_adoption and improved_satisfaction and quality_sufficient
            },
            "time_series": {
                "adoption_rate_history": self.digital_adoption_rate_history,
                "satisfaction_history": self.service_satisfaction_history,
                "platform_quality_history": self.platform_quality_history
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")

        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        total_users = sum(self.channel_usage.values())
        digital_users = self.channel_usage.get("online_only", 0) + self.channel_usage.get("mixed_preference", 0)
        
        return {
            "digital_adoption_rate": digital_users / total_users if total_users > 0 else 0,
            "available_services": len(self.available_services),
            "platform_quality": self.platform_quality,
            "mobile_app_available": self.mobile_app_available,
            "ai_assistant_available": self.ai_assistant_available,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 