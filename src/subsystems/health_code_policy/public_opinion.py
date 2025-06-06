from typing import Dict, Any, List
import random
from collections import defaultdict, Counter

from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger


class PublicOpinionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Any = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Policy announcements by epoch
        self.policy_announcements = config.get("policy_announcements", {})
        
        # Opinion tracking
        self.opinions_by_epoch = defaultdict(list)
        self.sentiment_distribution = defaultdict(lambda: Counter())
        self.opinion_topics = defaultdict(lambda: Counter())
        
        # Agent attributes for opinion formation
        self.agent_attributes = {}
        
        # Public sentiment metrics
        self.public_sentiment_score = 0.5  # 0-1, 0.5 is neutral
        self.sentiment_volatility = 0.0
        
    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize with agent data"""
        self.logger.info(f"Initializing {self.name} with {len(all_agent_data)} agents")
        
        # Store relevant agent attributes
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            self.agent_attributes[agent_id] = {
                "age": agent_data.get("basic_info", {}).get("age", 30),
                "education_level": agent_data.get("basic_info", {}).get("education_level", "high_school"),
                "residence_type": agent_data.get("basic_info", {}).get("residence_type", "urban")
            }
        
        # Initialize system state
        self.system_state = {
            "total_agents": len(all_agent_data),
            "current_policy": "Normal period",
            "public_sentiment_score": self.public_sentiment_score
        }
        
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent decisions about expressing opinions"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        self.logger.info(f"Processing opinions for epoch {epoch}")
        
        # Get current policy announcement
        current_policy = self.policy_announcements.get(str(epoch), "No announcement")
        self.system_state["current_policy"] = current_policy
        
        # Process each agent's opinion decision
        epoch_opinions = []
        epoch_sentiments = Counter()
        
        for agent_id, decisions in agent_decisions.items():
            opinion_decision = decisions.get(self.name, {})
            
            if opinion_decision.get("express_opinion") == "yes":
                opinion_content = opinion_decision.get("opinion_content", "")
                sentiment = opinion_decision.get("opinion_sentiment", "neutral")
                
                opinion_entry = {
                    "agent_id": agent_id,
                    "content": opinion_content,
                    "sentiment": sentiment,
                    "agent_attrs": self.agent_attributes.get(agent_id, {})
                }
                
                epoch_opinions.append(opinion_entry)
                epoch_sentiments[sentiment] += 1
                
                # Extract topics from opinion content
                self._extract_topics(opinion_content, epoch)
        
        # Store opinions
        self.opinions_by_epoch[epoch] = epoch_opinions
        self.sentiment_distribution[epoch] = epoch_sentiments
        
        # Calculate public sentiment metrics
        self._update_public_sentiment(epoch_sentiments)
        
        # Post to blackboard
        self._post_to_blackboard("public_sentiment_score", self.public_sentiment_score)
        self._post_to_blackboard("current_policy_announcement", current_policy)
        
        # Log statistics
        total_opinions = len(epoch_opinions)
        self.logger.info(f"Epoch {epoch}: {total_opinions} opinions expressed")
        self.logger.info(f"Sentiment distribution: {dict(epoch_sentiments)}")
        self.logger.info(f"Public sentiment score: {self.public_sentiment_score:.3f}")
        
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Return environment information for a specific agent"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Get recent opinions from similar agents (same residence type)
        agent_attrs = self.agent_attributes.get(agent_id, {})
        residence_type = agent_attrs.get("residence_type", "urban")
        
        recent_opinions = []
        if epoch > 0 and (epoch - 1) in self.opinions_by_epoch:
            for opinion in self.opinions_by_epoch[epoch - 1][-10:]:  # Last 10 opinions
                if opinion["agent_attrs"].get("residence_type") == residence_type:
                    recent_opinions.append({
                        "content": opinion["content"][:100] + "...",  # Truncate
                        "sentiment": opinion["sentiment"]
                    })
        
        # Get infection data from blackboard to influence sentiment
        active_infections = self._get_from_blackboard("active_infections", 0)
        total_agents = len(self.agent_attributes)
        infection_rate = active_infections / max(total_agents, 1) if total_agents > 0 else 0
        
        # Adjust public sentiment based on infection rate
        sentiment_description = self._get_sentiment_description(infection_rate, epoch)
        
        return {
            "policy_announcements": self.policy_announcements.get(str(epoch), "No announcement"),
            "public_sentiment": {
                "score": self.public_sentiment_score,
                "trend": "improving" if self.sentiment_volatility > 0 else "worsening" if self.sentiment_volatility < 0 else "stable",
                "recent_opinions": recent_opinions[:3],  # Show top 3
                "infection_concern": infection_rate,
                "situation_description": sentiment_description
            },
            "media_reports": self._generate_media_report(epoch, infection_rate)
        }
    
    def _get_sentiment_description(self, infection_rate: float, epoch: int) -> str:
        """Generate sentiment description based on infection rate and policy epoch"""
        if infection_rate < 0.01:
            return "Public optimistic about health situation"
        elif infection_rate < 0.05:
            return "Public cautiously optimistic but concerned"
        elif infection_rate < 0.15:
            return "Growing public anxiety about infection spread"
        else:
            return "High public concern and anxiety about health crisis"
    
    def _extract_topics(self, opinion_content: str, epoch: int):
        """Extract topics from opinion content"""
        # Simple keyword extraction
        keywords = ["freedom", "safety", "privacy", "economy", "health", "control", "quarantine", "work", "family"]
        
        content_lower = opinion_content.lower()
        for keyword in keywords:
            if keyword in content_lower:
                self.opinion_topics[epoch][keyword] += 1
    
    def _update_public_sentiment(self, sentiments: Counter):
        """Update public sentiment score based on current opinions and infection situation"""
        total = sum(sentiments.values())
        if total == 0:
            return
        
        # Get infection data to influence sentiment
        active_infections = self._get_from_blackboard("active_infections", 0)
        total_agents = len(self.agent_attributes)
        infection_rate = active_infections / max(total_agents, 1) if total_agents > 0 else 0
        
        # Calculate weighted sentiment from opinions
        sentiment_weights = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
        weighted_sum = sum(sentiments[s] * sentiment_weights[s] for s in sentiment_weights)
        opinion_sentiment = weighted_sum / total
        
        # Adjust sentiment based on infection rate (more infections -> more negative)
        infection_penalty = min(infection_rate * 5, 0.5)  # Max 50% penalty
        adjusted_sentiment = opinion_sentiment * (1 - infection_penalty)
        
        # Calculate volatility
        self.sentiment_volatility = adjusted_sentiment - self.public_sentiment_score
        
        # Update with smoothing
        self.public_sentiment_score = 0.7 * self.public_sentiment_score + 0.3 * adjusted_sentiment
        
    def _generate_media_report(self, epoch: int, infection_rate: float = 0) -> str:
        """Generate a media report based on current situation"""
        if infection_rate > 0.15:
            return "Media reports growing health crisis and calls for stronger measures"
        elif infection_rate > 0.05:
            return "Media coverage focuses on infection spread and policy effectiveness"
        elif self.public_sentiment_score > 0.7:
            return "Media reports cautious optimism about health code policy"
        elif self.public_sentiment_score < 0.3:
            return "Media highlights public concerns about health restrictions"
        else:
            return "Media coverage shows balanced view of health code measures"
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the public opinion system"""
        self.logger.info("=== Public Opinion System Evaluation ===")
        
        # Calculate overall metrics
        total_opinions = sum(len(opinions) for opinions in self.opinions_by_epoch.values())
        
        # Sentiment trends over time
        sentiment_trends = {
            "positive": [],
            "neutral": [],
            "negative": []
        }
        
        for epoch in sorted(self.opinions_by_epoch.keys()):
            sentiments = self.sentiment_distribution[epoch]
            total_epoch = sum(sentiments.values())
            if total_epoch > 0:
                for sentiment in sentiment_trends:
                    sentiment_trends[sentiment].append(
                        sentiments[sentiment] / total_epoch
                    )
        
        # Topic frequency analysis
        all_topics = Counter()
        for topics in self.opinion_topics.values():
            all_topics.update(topics)
        
        # Opinion participation by demographics
        participation_by_education = defaultdict(int)
        sentiment_by_age = defaultdict(lambda: Counter())
        
        for opinions in self.opinions_by_epoch.values():
            for opinion in opinions:
                attrs = opinion["agent_attrs"]
                participation_by_education[attrs.get("education_level", "unknown")] += 1
                
                age = attrs.get("age", 30)
                age_group = "young" if age < 30 else "middle" if age < 60 else "senior"
                sentiment_by_age[age_group][opinion["sentiment"]] += 1
        
        evaluation_results = {
            "total_opinions_expressed": total_opinions,
            "average_opinions_per_epoch": total_opinions / max(len(self.opinions_by_epoch), 1),
            "final_public_sentiment": self.public_sentiment_score,
            "sentiment_trends": sentiment_trends,
            "top_topics": dict(all_topics.most_common(5)),
            "participation_by_education": dict(participation_by_education),
            "sentiment_by_age_group": {k: dict(v) for k, v in sentiment_by_age.items()},
            "policy_response_analysis": self._analyze_policy_response()
        }
        
        # Log key metrics
        self.logger.info(f"Total opinions expressed: {total_opinions}")
        self.logger.info(f"Final public sentiment score: {self.public_sentiment_score:.3f}")
        self.logger.info(f"Top topics: {dict(all_topics.most_common(3))}")
        
        # Log time series data
        self.logger.info("Sentiment trends over epochs:")
        for epoch in sorted(self.opinions_by_epoch.keys()):
            sentiments = self.sentiment_distribution[epoch]
            self.logger.info(f"  Epoch {epoch}: {dict(sentiments)}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def _analyze_policy_response(self) -> Dict[str, Any]:
        """Analyze how public opinion responded to policy changes"""
        response_analysis = {}
        
        # Identify major policy change epochs (2 and 4 based on config)
        policy_change_epochs = [2, 4]
        
        for change_epoch in policy_change_epochs:
            if change_epoch in self.sentiment_distribution and (change_epoch - 1) in self.sentiment_distribution:
                before = self.sentiment_distribution[change_epoch - 1]
                after = self.sentiment_distribution[change_epoch]
                
                # Calculate sentiment shift
                total_before = sum(before.values())
                total_after = sum(after.values())
                
                if total_before > 0 and total_after > 0:
                    negative_shift = (after["negative"] / total_after) - (before["negative"] / total_before)
                    positive_shift = (after["positive"] / total_after) - (before["positive"] / total_before)
                    
                    response_analysis[f"epoch_{change_epoch}_response"] = {
                        "negative_shift": negative_shift,
                        "positive_shift": positive_shift,
                        "overall_impact": "negative" if negative_shift > 0.1 else "positive" if positive_shift > 0.1 else "neutral"
                    }
        
        return response_analysis 