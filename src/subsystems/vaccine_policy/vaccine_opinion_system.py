"""
Vaccine Opinion System - Manages vaccine campaigns and public sentiment
"""
import random
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger
from src.utils.data_loader import get_nested_value


class VaccineOpinionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Optional[Any] = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Load opinion influence parameters
        self.opinion_influence = config.get("opinion_influence", {})
        
        # Initialize system state
        self.system_state = {
            "vaccine_campaigns": config.get("vaccine_campaigns", {}),
            "media_coverage": config.get("media_coverage", {}),
            "public_sentiment": {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
            "agent_opinions": {},  # agent_id -> opinion data
            "vaccination_willingness_by_epoch": defaultdict(list),  # Track willingness over time
            "trust_levels_by_epoch": defaultdict(lambda: {"high": 0, "moderate": 0, "low": 0}),
            "opinion_count_by_epoch": defaultdict(int),
            "sentiment_distribution_by_epoch": defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0}),
            "timing_preferences_by_epoch": defaultdict(lambda: {"immediate": 0, "wait_1_week": 0, "wait_2_weeks": 0, "wait_longer": 0}),
            "age_group_willingness": defaultdict(lambda: defaultdict(list))  # Track by age group
        }
        
        self.logger.info(f"VaccineOpinionSystem initialized with {len(self.system_state['vaccine_campaigns'])} campaign phases")
    
    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize with agent data"""
        self.logger.info(f"Initializing opinion system with {len(all_agent_data)} agents")
        
        # Initialize agent opinions
        for agent_data in all_agent_data:
            agent_id = str(get_nested_value(agent_data, "id"))
            age = get_nested_value(agent_data, "basic_info.age", 30)
            education = get_nested_value(agent_data, "basic_info.education_level", "unknown")
            
            # Initial willingness based on demographics
            base_willingness = 0.5
            if age > 60:
                base_willingness += 0.1  # Elderly more willing
            if "master" in education.lower() or "phd" in education.lower():
                base_willingness += 0.1  # Higher education more willing
            
            self.system_state["agent_opinions"][agent_id] = {
                "vaccination_willingness": base_willingness,
                "trust_in_vaccine": "moderate",
                "has_expressed_opinion": False,
                "opinion_history": [],
                "preferred_timing": "wait_1_week",
                "age": age,
                "education": education,
                "neighbor_influence_received": 0
            }
        
        # Post initial sentiment to blackboard
        self._post_to_blackboard("public_vaccine_sentiment", self.system_state["public_sentiment"])
    
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent decisions for this epoch"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        self.logger.info(f"Processing opinions for epoch {current_epoch}")
        
        epoch_willingness = []
        epoch_trust = {"high": 0, "moderate": 0, "low": 0}
        opinion_count = 0
        sentiment_count = {"positive": 0, "neutral": 0, "negative": 0}
        timing_preferences = {"immediate": 0, "wait_1_week": 0, "wait_2_weeks": 0, "wait_longer": 0}
        age_group_willingness = {"0-18": [], "19-40": [], "41-60": [], "61+": []}
        
        # Get neighbor vaccination status from blackboard
        neighbor_vaccination_rates = self._get_from_blackboard("agent_neighbor_vaccination_rates", {})
        
        # Process each agent's decision
        for agent_id, all_decisions in agent_decisions.items():
            decisions = all_decisions.get(self.name, {})
            
            if agent_id in self.system_state["agent_opinions"]:
                agent_opinion = self.system_state["agent_opinions"][agent_id]
                
                # Apply neighbor influence
                neighbor_rate = neighbor_vaccination_rates.get(agent_id, 0)
                neighbor_influence = neighbor_rate * self.opinion_influence.get("neighbor_influence_weight", 0.2)
                
                # Update vaccination willingness
                if "vaccination_willingness" in decisions:
                    try:
                        willingness = float(decisions["vaccination_willingness"])
                        # Apply trust level influence
                        trust_factor = self.opinion_influence.get("trust_to_willingness_factor", 0.3)
                        if agent_opinion["trust_in_vaccine"] == "high":
                            willingness = min(1.0, willingness + trust_factor)
                        elif agent_opinion["trust_in_vaccine"] == "low":
                            willingness = max(0.0, willingness - trust_factor)
                        
                        # Apply neighbor influence
                        willingness = min(1.0, willingness + neighbor_influence)
                        
                        agent_opinion["vaccination_willingness"] = willingness
                        agent_opinion["neighbor_influence_received"] = neighbor_influence
                        epoch_willingness.append(willingness)
                        
                        # Track by age group
                        age = agent_opinion["age"]
                        if age < 19:
                            age_group_willingness["0-18"].append(willingness)
                        elif age < 41:
                            age_group_willingness["19-40"].append(willingness)
                        elif age < 61:
                            age_group_willingness["41-60"].append(willingness)
                        else:
                            age_group_willingness["61+"].append(willingness)
                            
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid willingness value for agent {agent_id}")
                
                # Update trust level
                if "trust_in_vaccine" in decisions:
                    trust_level = decisions["trust_in_vaccine"]
                    if trust_level in ["high", "moderate", "low"]:
                        agent_opinion["trust_in_vaccine"] = trust_level
                        epoch_trust[trust_level] += 1
                
                # Update timing preference
                if "preferred_timing" in decisions:
                    timing = decisions["preferred_timing"]
                    if timing in timing_preferences:
                        agent_opinion["preferred_timing"] = timing
                        timing_preferences[timing] += 1
                
                # Process expressed opinions
                if decisions.get("express_vaccine_opinion") == "yes":
                    opinion_count += 1
                    agent_opinion["has_expressed_opinion"] = True
                    
                    if "opinion_content" in decisions:
                        opinion_entry = {
                            "epoch": current_epoch,
                            "content": decisions["opinion_content"],
                            "trust_level": agent_opinion["trust_in_vaccine"],
                            "willingness": agent_opinion["vaccination_willingness"]
                        }
                        agent_opinion["opinion_history"].append(opinion_entry)
                        
                        # Sentiment analysis based on trust and willingness
                        if agent_opinion["trust_in_vaccine"] == "high" and agent_opinion["vaccination_willingness"] > 0.7:
                            sentiment_count["positive"] += 1
                        elif agent_opinion["trust_in_vaccine"] == "low" or agent_opinion["vaccination_willingness"] < 0.3:
                            sentiment_count["negative"] += 1
                        else:
                            sentiment_count["neutral"] += 1
        
        # Update epoch statistics
        if epoch_willingness:
            avg_willingness = np.mean(epoch_willingness)
            self.system_state["vaccination_willingness_by_epoch"][current_epoch] = {
                "average": avg_willingness,
                "std": np.std(epoch_willingness),
                "min": min(epoch_willingness),
                "max": max(epoch_willingness)
            }
            
            # Store age group statistics
            for age_group, willingness_list in age_group_willingness.items():
                if willingness_list:
                    self.system_state["age_group_willingness"][current_epoch][age_group] = {
                        "average": np.mean(willingness_list),
                        "count": len(willingness_list)
                    }
        
        self.system_state["trust_levels_by_epoch"][current_epoch] = epoch_trust
        self.system_state["opinion_count_by_epoch"][current_epoch] = opinion_count
        self.system_state["sentiment_distribution_by_epoch"][current_epoch] = sentiment_count
        self.system_state["timing_preferences_by_epoch"][current_epoch] = timing_preferences
        
        # Update overall public sentiment
        if opinion_count > 0:
            total_opinions = sum(sentiment_count.values())
            self.system_state["public_sentiment"] = {
                "positive": sentiment_count["positive"] / total_opinions,
                "neutral": sentiment_count["neutral"] / total_opinions,
                "negative": sentiment_count["negative"] / total_opinions
            }
        
        # Post updated data to blackboard
        self._post_to_blackboard("public_vaccine_sentiment", self.system_state["public_sentiment"])
        self._post_to_blackboard("average_vaccine_willingness", avg_willingness if epoch_willingness else 0.5)
        self._post_to_blackboard("agent_timing_preferences", {
            agent_id: opinion["preferred_timing"] 
            for agent_id, opinion in self.system_state["agent_opinions"].items()
        })
        
        self.logger.info(f"Epoch {current_epoch}: {opinion_count} opinions expressed, "
                        f"avg willingness: {avg_willingness if epoch_willingness else 'N/A'}")
    
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Get environment information for a specific agent"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Get current campaign message
        campaign_message = self.system_state["vaccine_campaigns"].get(
            str(current_epoch), 
            "No specific campaign message this period."
        )
        
        # Get media coverage for current epoch
        media_coverage = self.system_state["media_coverage"].get(
            str(current_epoch),
            {"positive": 0.5, "neutral": 0.3, "negative": 0.2}
        )
        
        # Get agent's own opinion data
        agent_opinion = self.system_state["agent_opinions"].get(agent_id, {})
        
        # Get infection status from blackboard to influence decisions
        current_infection_rate = self._get_from_blackboard("current_infection_rate", 0.0)
        
        return {
            "vaccine_campaigns": campaign_message,
            "public_sentiment": self.system_state["public_sentiment"],
            "media_coverage": media_coverage,
            "your_current_willingness": agent_opinion.get("vaccination_willingness", 0.5),
            "your_trust_level": agent_opinion.get("trust_in_vaccine", "moderate"),
            "your_preferred_timing": agent_opinion.get("preferred_timing", "wait_1_week"),
            "community_infection_rate": current_infection_rate,
            "neighbor_influence": agent_opinion.get("neighbor_influence_received", 0)
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the opinion system performance"""
        self.logger.info("Evaluating Vaccine Opinion System...")
        
        # Calculate overall statistics
        total_agents = len(self.system_state["agent_opinions"])
        agents_expressed_opinion = sum(
            1 for opinion in self.system_state["agent_opinions"].values() 
            if opinion["has_expressed_opinion"]
        )
        
        # Calculate final vaccination willingness distribution
        final_willingness = [
            opinion["vaccination_willingness"] 
            for opinion in self.system_state["agent_opinions"].values()
        ]
        
        # Calculate trust level distribution
        final_trust_dist = {"high": 0, "moderate": 0, "low": 0}
        final_timing_dist = {"immediate": 0, "wait_1_week": 0, "wait_2_weeks": 0, "wait_longer": 0}
        
        for opinion in self.system_state["agent_opinions"].values():
            final_trust_dist[opinion["trust_in_vaccine"]] += 1
            final_timing_dist[opinion["preferred_timing"]] += 1
        
        evaluation_results = {
            "total_agents": total_agents,
            "agents_expressed_opinion": agents_expressed_opinion,
            "opinion_expression_rate": agents_expressed_opinion / total_agents if total_agents > 0 else 0,
            "final_willingness_stats": {
                "mean": np.mean(final_willingness) if final_willingness else 0,
                "std": np.std(final_willingness) if final_willingness else 0,
                "min": min(final_willingness) if final_willingness else 0,
                "max": max(final_willingness) if final_willingness else 0
            },
            "final_trust_distribution": final_trust_dist,
            "final_timing_distribution": final_timing_dist,
            "willingness_by_epoch": dict(self.system_state["vaccination_willingness_by_epoch"]),
            "trust_evolution": dict(self.system_state["trust_levels_by_epoch"]),
            "opinion_count_evolution": dict(self.system_state["opinion_count_by_epoch"]),
            "sentiment_evolution": dict(self.system_state["sentiment_distribution_by_epoch"]),
            "timing_preferences_evolution": dict(self.system_state["timing_preferences_by_epoch"]),
            "age_group_willingness_evolution": dict(self.system_state["age_group_willingness"])
        }
        
        # Log key metrics
        self.logger.info(f"=== Vaccine Opinion System Evaluation Results ===")
        self.logger.info(f"Total agents: {total_agents}")
        self.logger.info(f"Agents who expressed opinion: {agents_expressed_opinion} ({evaluation_results['opinion_expression_rate']:.2%})")
        self.logger.info(f"Final average vaccination willingness: {evaluation_results['final_willingness_stats']['mean']:.3f}")
        self.logger.info(f"Final trust distribution: High={final_trust_dist['high']}, "
                        f"Moderate={final_trust_dist['moderate']}, Low={final_trust_dist['low']}")
        self.logger.info(f"Final timing preferences: Immediate={final_timing_dist['immediate']}, "
                        f"Wait 1 week={final_timing_dist['wait_1_week']}, "
                        f"Wait 2 weeks={final_timing_dist['wait_2_weeks']}, "
                        f"Wait longer={final_timing_dist['wait_longer']}")
        
        # Log time series data
        self.logger.info("\n--- Vaccination Willingness Evolution ---")
        for epoch, stats in self.system_state["vaccination_willingness_by_epoch"].items():
            self.logger.info(f"Epoch {epoch}: avg={stats['average']:.3f}, std={stats['std']:.3f}")
        
        self.logger.info("\n--- Trust Level Evolution ---")
        for epoch, trust_dist in self.system_state["trust_levels_by_epoch"].items():
            total = sum(trust_dist.values())
            if total > 0:
                self.logger.info(f"Epoch {epoch}: High={trust_dist['high']}/{total} ({trust_dist['high']/total:.2%}), "
                               f"Moderate={trust_dist['moderate']}/{total} ({trust_dist['moderate']/total:.2%}), "
                               f"Low={trust_dist['low']}/{total} ({trust_dist['low']/total:.2%})")
        
        self.logger.info("\n--- Age Group Willingness Evolution ---")
        for epoch, age_data in self.system_state["age_group_willingness"].items():
            self.logger.info(f"Epoch {epoch}:")
            for age_group, stats in age_data.items():
                self.logger.info(f"  {age_group}: avg={stats['average']:.3f}, n={stats['count']}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results 