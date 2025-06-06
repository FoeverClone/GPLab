"""
Vaccine Health System - Manages infection status, vaccination, and social network transmission
"""
import random
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
import numpy as np

from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger
from src.utils.data_loader import get_nested_value


class VaccineHealthSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Optional[Any] = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Load configuration parameters
        self.vaccine_params = config.get("vaccine_params", {})
        self.infection_params = config.get("infection_params", {})
        self.social_network_config = config.get("social_network", {})
        
        # Initialize system state
        self.system_state = {
            "agent_health": {},  # agent_id -> health data
            "social_network": {},  # agent_id -> list of connected agent_ids
            "infection_count_by_epoch": defaultdict(int),
            "vaccination_count_by_epoch": defaultdict(int),
            "cumulative_vaccinations": 0,
            "recovery_count_by_epoch": defaultdict(int),
            "vaccination_rate_by_epoch": defaultdict(float),
            "infection_rate_by_epoch": defaultdict(float),
            "vaccine_effectiveness_data": defaultdict(dict),
            "age_group_infections": defaultdict(lambda: defaultdict(int)),
            "breakthrough_infections_by_epoch": defaultdict(int),
            "delayed_vaccination_count": defaultdict(int)
        }
        
        self.logger.info(f"VaccineHealthSystem initialized with vaccine effectiveness: {self.vaccine_params.get('vaccine_effectiveness', 0.85)}")
    
    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent health states and social network"""
        self.logger.info(f"Initializing health system with {len(all_agent_data)} agents")
        
        initial_infection_rate = self.vaccine_params.get("initial_infection_rate", 0.05)
        agent_ids = []
        
        # Initialize health states
        for agent_data in all_agent_data:
            agent_id = str(get_nested_value(agent_data, "id"))
            agent_ids.append(agent_id)
            
            # Get age for risk calculation
            age = get_nested_value(agent_data, "basic_info.age", 30)
            age_group = self._get_age_group(age)
            
            # Check for chronic conditions
            chronic_conditions = get_nested_value(agent_data, "health_attributes.chronic_conditions", [])
            has_chronic = len(chronic_conditions) > 0 if isinstance(chronic_conditions, list) else False
            
            # Adjust initial infection probability by age and chronic conditions
            adjusted_infection_rate = initial_infection_rate
            if age > 60:
                adjusted_infection_rate *= 1.5
            if has_chronic:
                adjusted_infection_rate *= self.infection_params.get("chronic_condition_severity_multiplier", 1.5)
            
            # Determine initial infection status
            is_infected = random.random() < adjusted_infection_rate
            
            self.system_state["agent_health"][agent_id] = {
                "infection_status": "infected" if is_infected else "susceptible",
                "vaccination_status": "unvaccinated",
                "days_infected": 0 if not is_infected else 1,
                "days_since_vaccination": 0,
                "days_since_recovery": 0,
                "age": age,
                "age_group": age_group,
                "has_chronic_conditions": has_chronic,
                "infection_history": [],
                "vaccination_date": None,
                "vaccine_effectiveness": 0.0,
                "current_vaccine_effectiveness": 0.0,
                "vaccination_scheduled": False,
                "vaccination_delay": 0
            }
        
        # Build social network
        self._build_social_network(agent_ids)
        
        # Post initial statistics to blackboard
        infected_count = sum(1 for health in self.system_state["agent_health"].values() 
                           if health["infection_status"] == "infected")
        self._post_to_blackboard("initial_infection_count", infected_count)
        self._post_to_blackboard("vaccine_availability", 0.0)  # No vaccines initially
        
        self.logger.info(f"Initial infection count: {infected_count} ({infected_count/len(agent_ids)*100:.1f}%)")
    
    def _get_age_group(self, age: int) -> str:
        """Categorize age into groups"""
        if age < 19:
            return "0-18"
        elif age < 41:
            return "19-40"
        elif age < 61:
            return "41-60"
        else:
            return "61+"
    
    def _build_social_network(self, agent_ids: List[str]):
        """Build social network connections between agents"""
        avg_connections = self.social_network_config.get("avg_connections", 10)
        connection_types = self.social_network_config.get("connection_types", ["family", "work", "friend", "neighbor"])
        
        for agent_id in agent_ids:
            # Random number of connections around the average
            num_connections = max(1, int(np.random.normal(avg_connections, avg_connections/3)))
            num_connections = min(num_connections, len(agent_ids) - 1)
            
            # Select random connections (excluding self)
            possible_connections = [aid for aid in agent_ids if aid != agent_id]
            connections = random.sample(possible_connections, num_connections)
            
            # Assign connection types with some bias (more family connections for elderly)
            agent_age = self.system_state["agent_health"][agent_id]["age"]
            self.system_state["social_network"][agent_id] = []
            
            for conn_id in connections:
                if agent_age > 60 and random.random() < 0.4:
                    conn_type = "family"
                else:
                    conn_type = random.choice(connection_types)
                    
                self.system_state["social_network"][agent_id].append({
                    "agent_id": conn_id,
                    "type": conn_type
                })
    
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent decisions and update health states"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        self.logger.info(f"Processing health updates for epoch {current_epoch}")
        
        # Get vaccine availability for current epoch
        vaccine_availability = self.vaccine_params["vaccine_availability_by_epoch"].get(
            str(current_epoch), 0.0
        )
        self._post_to_blackboard("vaccine_availability", vaccine_availability)
        
        # Get timing preferences from opinion system
        timing_preferences = self._get_from_blackboard("agent_timing_preferences", {})
        
        # Update vaccine effectiveness for already vaccinated agents
        self._update_vaccine_effectiveness()
        
        # Process vaccinations with timing considerations
        vaccination_count = self._process_vaccinations(agent_decisions, vaccine_availability, timing_preferences)
        
        # Then process infections
        new_infections, breakthrough_infections = self._process_infections()
        
        # Process recoveries
        recoveries = self._process_recoveries()
        
        # Update immunity duration for recovered agents
        self._update_recovered_immunity()
        
        # Calculate and post neighbor vaccination rates
        self._calculate_neighbor_vaccination_rates()
        
        # Update statistics
        total_agents = len(self.system_state["agent_health"])
        infected_count = sum(1 for health in self.system_state["agent_health"].values() 
                           if health["infection_status"] == "infected")
        vaccinated_count = sum(1 for health in self.system_state["agent_health"].values() 
                             if health["vaccination_status"] == "vaccinated")
        
        # Track age group infections
        for agent_id, health in self.system_state["agent_health"].items():
            if health["infection_status"] == "infected":
                self.system_state["age_group_infections"][current_epoch][health["age_group"]] += 1
        
        self.system_state["infection_count_by_epoch"][current_epoch] = infected_count
        self.system_state["vaccination_count_by_epoch"][current_epoch] = vaccination_count
        self.system_state["recovery_count_by_epoch"][current_epoch] = recoveries
        self.system_state["vaccination_rate_by_epoch"][current_epoch] = vaccinated_count / total_agents
        self.system_state["infection_rate_by_epoch"][current_epoch] = infected_count / total_agents
        self.system_state["cumulative_vaccinations"] = vaccinated_count
        self.system_state["breakthrough_infections_by_epoch"][current_epoch] = breakthrough_infections
        
        # Post key metrics to blackboard
        self._post_to_blackboard("current_infection_rate", infected_count / total_agents)
        self._post_to_blackboard("current_vaccination_rate", vaccinated_count / total_agents)
        
        self.logger.info(f"Epoch {current_epoch}: {vaccination_count} new vaccinations, "
                        f"{new_infections} new infections ({breakthrough_infections} breakthrough), "
                        f"{recoveries} recoveries")
    
    def _update_vaccine_effectiveness(self):
        """Update vaccine effectiveness based on time since vaccination"""
        delay_days = self.vaccine_params.get("vaccine_effectiveness_delay", 14)
        full_effectiveness = self.vaccine_params.get("vaccine_effectiveness", 0.85)
        partial_effectiveness = self.vaccine_params.get("vaccine_partial_effectiveness", 0.3)
        
        for agent_id, health_data in self.system_state["agent_health"].items():
            if health_data["vaccination_status"] == "vaccinated":
                health_data["days_since_vaccination"] += 1
                
                # Calculate current effectiveness
                if health_data["days_since_vaccination"] < delay_days:
                    # Linear increase from partial to full effectiveness
                    progress = health_data["days_since_vaccination"] / delay_days
                    health_data["current_vaccine_effectiveness"] = (
                        partial_effectiveness + (full_effectiveness - partial_effectiveness) * progress
                    )
                else:
                    health_data["current_vaccine_effectiveness"] = full_effectiveness
    
    def _calculate_neighbor_vaccination_rates(self):
        """Calculate vaccination rate among each agent's neighbors"""
        neighbor_rates = {}
        
        for agent_id, connections in self.system_state["social_network"].items():
            vaccinated_neighbors = 0
            for connection in connections:
                neighbor_id = connection["agent_id"]
                if self.system_state["agent_health"][neighbor_id]["vaccination_status"] == "vaccinated":
                    vaccinated_neighbors += 1
            
            neighbor_rates[agent_id] = vaccinated_neighbors / len(connections) if connections else 0
        
        self._post_to_blackboard("agent_neighbor_vaccination_rates", neighbor_rates)
    
    def _process_vaccinations(self, agent_decisions: Dict[str, Dict[str, Any]], 
                            availability: float, timing_preferences: Dict[str, str]) -> int:
        """Process vaccination decisions with timing considerations"""
        vaccination_count = 0
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Categorize agents by their vaccination timing preference
        immediate_agents = []
        delayed_agents = []
        
        for agent_id, all_decisions in agent_decisions.items():
            decisions = all_decisions.get(self.name, {})
            health_data = self.system_state["agent_health"].get(agent_id)
            
            if health_data and health_data["vaccination_status"] == "unvaccinated":
                # Check if agent has scheduled vaccination from previous epoch
                if health_data["vaccination_scheduled"]:
                    health_data["vaccination_delay"] -= 1
                    if health_data["vaccination_delay"] <= 0:
                        immediate_agents.append(agent_id)
                        health_data["vaccination_scheduled"] = False
                
                elif decisions.get("get_vaccinated") == "yes":
                    timing_pref = timing_preferences.get(agent_id, "immediate")
                    
                    if timing_pref == "immediate":
                        immediate_agents.append(agent_id)
                    elif timing_pref == "wait_1_week":
                        health_data["vaccination_scheduled"] = True
                        health_data["vaccination_delay"] = 1
                        self.system_state["delayed_vaccination_count"][current_epoch] += 1
                    elif timing_pref == "wait_2_weeks":
                        health_data["vaccination_scheduled"] = True
                        health_data["vaccination_delay"] = 2
                        self.system_state["delayed_vaccination_count"][current_epoch] += 1
        
        # Apply vaccine availability constraint
        max_vaccines = int(len(self.system_state["agent_health"]) * availability)
        
        # Prioritize elderly and those with chronic conditions
        priority_agents = []
        regular_agents = []
        
        for agent_id in immediate_agents:
            health_data = self.system_state["agent_health"][agent_id]
            if health_data["age"] > 60 or health_data["has_chronic_conditions"]:
                priority_agents.append(agent_id)
            else:
                regular_agents.append(agent_id)
        
        # Vaccinate priority group first
        agents_to_vaccinate = priority_agents[:max_vaccines]
        remaining_vaccines = max_vaccines - len(agents_to_vaccinate)
        
        if remaining_vaccines > 0:
            agents_to_vaccinate.extend(regular_agents[:remaining_vaccines])
        
        # Vaccinate selected agents
        for agent_id in agents_to_vaccinate:
            health_data = self.system_state["agent_health"][agent_id]
            health_data["vaccination_status"] = "vaccinated"
            health_data["vaccination_date"] = current_epoch
            health_data["days_since_vaccination"] = 0
            health_data["vaccine_effectiveness"] = self.vaccine_params.get("vaccine_effectiveness", 0.85)
            health_data["current_vaccine_effectiveness"] = self.vaccine_params.get("vaccine_partial_effectiveness", 0.3)
            vaccination_count += 1
        
        return vaccination_count
    
    def _process_infections(self) -> tuple[int, int]:
        """Process disease transmission through social network"""
        new_infections = 0
        breakthrough_infections = 0
        base_transmission_rate = self.infection_params.get("base_transmission_rate", 0.06)
        transmission_by_type = self.social_network_config.get("transmission_probability_by_type", {})
        
        # Collect currently infected agents
        infected_agents = [
            agent_id for agent_id, health in self.system_state["agent_health"].items()
            if health["infection_status"] == "infected"
        ]
        
        # Process transmission for each susceptible agent
        for agent_id, health_data in self.system_state["agent_health"].items():
            if health_data["infection_status"] == "susceptible":
                # Check exposure through social network
                connections = self.system_state["social_network"].get(agent_id, [])
                
                for connection in connections:
                    connected_id = connection["agent_id"]
                    connection_type = connection["type"]
                    
                    if connected_id in infected_agents:
                        # Use connection-type specific transmission rate
                        transmission_prob = transmission_by_type.get(connection_type, base_transmission_rate)
                        
                        # Adjust for age-based susceptibility
                        age_factor = self.infection_params["age_severity_factors"].get(
                            health_data["age_group"], 1.0
                        )
                        transmission_prob *= (0.5 + 0.5 * age_factor)  # Age affects susceptibility
                        
                        # Reduce if agent is vaccinated
                        if health_data["vaccination_status"] == "vaccinated":
                            effectiveness = health_data["current_vaccine_effectiveness"]
                            transmission_prob *= (1 - effectiveness)
                        
                        # Transmission occurs
                        if random.random() < transmission_prob:
                            health_data["infection_status"] = "infected"
                            health_data["days_infected"] = 1
                            health_data["infection_history"].append({
                                "epoch": self.current_time.get_current_epoch() if self.current_time else 0,
                                "source": connected_id,
                                "vaccinated": health_data["vaccination_status"] == "vaccinated"
                            })
                            new_infections += 1
                            
                            if health_data["vaccination_status"] == "vaccinated":
                                breakthrough_infections += 1
                            
                            break  # Only get infected once per epoch
        
        return new_infections, breakthrough_infections
    
    def _process_recoveries(self) -> int:
        """Process recovery from infection"""
        recoveries = 0
        recovery_range = self.infection_params.get("recovery_days", [7, 21])
        age_recovery_modifiers = self.infection_params.get("age_recovery_modifiers", {})
        
        for agent_id, health_data in self.system_state["agent_health"].items():
            if health_data["infection_status"] == "infected":
                health_data["days_infected"] += 1
                
                # Calculate recovery time based on age and chronic conditions
                base_recovery_days = random.randint(recovery_range[0], recovery_range[1])
                age_modifier = age_recovery_modifiers.get(health_data["age_group"], 1.0)
                
                if health_data["has_chronic_conditions"]:
                    age_modifier *= 1.3
                
                required_recovery_days = int(base_recovery_days * age_modifier)
                
                if health_data["days_infected"] >= required_recovery_days:
                    health_data["infection_status"] = "recovered"
                    health_data["days_infected"] = 0
                    health_data["days_since_recovery"] = 0
                    recoveries += 1
        
        return recoveries
    
    def _update_recovered_immunity(self):
        """Update immunity status for recovered agents"""
        immunity_duration = self.infection_params.get("recovered_immunity_duration", 90)
        
        for agent_id, health_data in self.system_state["agent_health"].items():
            if health_data["infection_status"] == "recovered":
                health_data["days_since_recovery"] += 1
                
                # Check if immunity has waned
                if health_data["days_since_recovery"] > immunity_duration:
                    health_data["infection_status"] = "susceptible"
                    health_data["days_since_recovery"] = 0
    
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Get health information for a specific agent"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        health_data = self.system_state["agent_health"].get(agent_id, {})
        
        # Get neighbor health status
        connections = self.system_state["social_network"].get(agent_id, [])
        neighbor_infection_count = 0
        neighbor_vaccination_count = 0
        family_infected = False
        
        for connection in connections:
            neighbor_id = connection["agent_id"]
            neighbor_health = self.system_state["agent_health"].get(neighbor_id, {})
            if neighbor_health.get("infection_status") == "infected":
                neighbor_infection_count += 1
                if connection["type"] == "family":
                    family_infected = True
            if neighbor_health.get("vaccination_status") == "vaccinated":
                neighbor_vaccination_count += 1
        
        # Get vaccine availability from blackboard
        vaccine_availability = self._get_from_blackboard("vaccine_availability", 0.0)
        
        # Get vaccination willingness from opinion system
        avg_willingness = self._get_from_blackboard("average_vaccine_willingness", 0.5)
        
        return {
            "infection_status": health_data.get("infection_status", "unknown"),
            "vaccination_status": health_data.get("vaccination_status", "unvaccinated"),
            "days_infected": health_data.get("days_infected", 0),
            "vaccine_availability": vaccine_availability,
            "neighbor_infection_rate": neighbor_infection_count / len(connections) if connections else 0,
            "neighbor_vaccination_rate": neighbor_vaccination_count / len(connections) if connections else 0,
            "family_member_infected": family_infected,
            "community_vaccine_sentiment": avg_willingness,
            "age_risk_group": health_data.get("age_group", "unknown"),
            "has_chronic_conditions": health_data.get("has_chronic_conditions", False),
            "vaccination_scheduled": health_data.get("vaccination_scheduled", False),
            "current_vaccine_effectiveness": health_data.get("current_vaccine_effectiveness", 0.0)
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the health system performance"""
        self.logger.info("Evaluating Vaccine Health System...")
        
        # Calculate final statistics
        total_agents = len(self.system_state["agent_health"])
        final_stats = {
            "susceptible": 0,
            "infected": 0,
            "recovered": 0,
            "vaccinated": 0,
            "vaccinated_infected": 0,
            "unvaccinated_infected": 0,
            "vaccinated_never_infected": 0
        }
        
        age_group_stats = {
            "0-18": {"total": 0, "infected": 0, "vaccinated": 0},
            "19-40": {"total": 0, "infected": 0, "vaccinated": 0},
            "41-60": {"total": 0, "infected": 0, "vaccinated": 0},
            "61+": {"total": 0, "infected": 0, "vaccinated": 0}
        }
        
        for health_data in self.system_state["agent_health"].values():
            final_stats[health_data["infection_status"]] = final_stats.get(health_data["infection_status"], 0) + 1
            age_group = health_data["age_group"]
            age_group_stats[age_group]["total"] += 1
            
            if health_data["vaccination_status"] == "vaccinated":
                final_stats["vaccinated"] += 1
                age_group_stats[age_group]["vaccinated"] += 1
                
                # Check infection history
                ever_infected = any(h["vaccinated"] for h in health_data["infection_history"])
                if ever_infected or health_data["infection_status"] in ["infected", "recovered"]:
                    final_stats["vaccinated_infected"] += 1
                else:
                    final_stats["vaccinated_never_infected"] += 1
            else:
                if health_data["infection_status"] in ["infected", "recovered"]:
                    final_stats["unvaccinated_infected"] += 1
            
            if health_data["infection_status"] in ["infected", "recovered"]:
                age_group_stats[age_group]["infected"] += 1
        
        # Calculate vaccine effectiveness in practice
        vaccinated_total = final_stats["vaccinated"]
        unvaccinated_total = total_agents - vaccinated_total
        
        vaccine_effectiveness_observed = 0
        if vaccinated_total > 0 and unvaccinated_total > 0:
            vaccinated_infection_rate = final_stats["vaccinated_infected"] / vaccinated_total
            unvaccinated_infection_rate = final_stats["unvaccinated_infected"] / unvaccinated_total
            if unvaccinated_infection_rate > 0:
                vaccine_effectiveness_observed = 1 - (vaccinated_infection_rate / unvaccinated_infection_rate)
        
        # Calculate breakthrough infection rate
        breakthrough_rate = 0
        if vaccinated_total > 0:
            total_breakthroughs = sum(self.system_state["breakthrough_infections_by_epoch"].values())
            breakthrough_rate = total_breakthroughs / vaccinated_total
        
        evaluation_results = {
            "total_agents": total_agents,
            "final_health_distribution": final_stats,
            "final_vaccination_rate": final_stats["vaccinated"] / total_agents,
            "final_infection_rate": final_stats["infected"] / total_agents,
            "cumulative_infection_rate": (final_stats["infected"] + final_stats["recovered"]) / total_agents,
            "vaccine_effectiveness_observed": vaccine_effectiveness_observed,
            "breakthrough_infection_rate": breakthrough_rate,
            "age_group_statistics": age_group_stats,
            "vaccination_timeline": dict(self.system_state["vaccination_count_by_epoch"]),
            "infection_timeline": dict(self.system_state["infection_count_by_epoch"]),
            "vaccination_rate_timeline": dict(self.system_state["vaccination_rate_by_epoch"]),
            "infection_rate_timeline": dict(self.system_state["infection_rate_by_epoch"]),
            "breakthrough_infections_timeline": dict(self.system_state["breakthrough_infections_by_epoch"]),
            "delayed_vaccinations_timeline": dict(self.system_state["delayed_vaccination_count"]),
            "age_group_infections_timeline": dict(self.system_state["age_group_infections"])
        }
        
        # Log key metrics
        self.logger.info(f"=== Vaccine Health System Evaluation Results ===")
        self.logger.info(f"Total agents: {total_agents}")
        self.logger.info(f"Final vaccination rate: {evaluation_results['final_vaccination_rate']:.2%}")
        self.logger.info(f"Final active infection rate: {evaluation_results['final_infection_rate']:.2%}")
        self.logger.info(f"Cumulative infection rate: {evaluation_results['cumulative_infection_rate']:.2%}")
        self.logger.info(f"Observed vaccine effectiveness: {evaluation_results['vaccine_effectiveness_observed']:.2%}")
        self.logger.info(f"Breakthrough infection rate: {evaluation_results['breakthrough_infection_rate']:.2%}")
        
        # Log time series data
        self.logger.info("\n--- Vaccination Progress ---")
        for epoch, count in self.system_state["vaccination_count_by_epoch"].items():
            rate = self.system_state["vaccination_rate_by_epoch"][epoch]
            delayed = self.system_state["delayed_vaccination_count"].get(epoch, 0)
            self.logger.info(f"Epoch {epoch}: {count} new vaccinations (cumulative rate: {rate:.2%}), "
                           f"{delayed} delayed")
        
        self.logger.info("\n--- Infection Timeline ---")
        for epoch, count in self.system_state["infection_count_by_epoch"].items():
            rate = self.system_state["infection_rate_by_epoch"][epoch]
            breakthroughs = self.system_state["breakthrough_infections_by_epoch"].get(epoch, 0)
            self.logger.info(f"Epoch {epoch}: {count} active infections (rate: {rate:.2%}), "
                           f"{breakthroughs} breakthroughs")
        
        self.logger.info("\n--- Age Group Statistics ---")
        for age_group, stats in age_group_stats.items():
            if stats["total"] > 0:
                infection_rate = stats["infected"] / stats["total"]
                vaccination_rate = stats["vaccinated"] / stats["total"]
                self.logger.info(f"{age_group}: {stats['total']} agents, "
                               f"infection rate: {infection_rate:.2%}, "
                               f"vaccination rate: {vaccination_rate:.2%}")
        
        self.logger.info("\n--- Final Health Status Distribution ---")
        for status, count in final_stats.items():
            if count > 0:
                self.logger.info(f"{status}: {count} ({count/total_agents:.2%})")
        
        self.evaluation_results = evaluation_results
        return evaluation_results 