from typing import Dict, Any, List
import random
from collections import defaultdict, Counter

from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger


class DiseaseTransmissionSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], blackboard: Any = None):
        super().__init__(name, config, blackboard)
        self.logger = get_logger(name)
        
        # Transmission parameters
        self.transmission_params = config.get("transmission_params", {})
        self.base_infection_rate = self.transmission_params.get("base_infection_rate", 0.02)
        
        # Health code impact parameters
        self.health_code_impact = config.get("health_code_impact", {})
        
        # Agent mobility tracking
        self.agent_mobility = defaultdict(dict)  # agent_id -> {go_out, destination, transport}
        self.daily_contacts = defaultdict(list)  # agent_id -> [contact_ids]
        
        # Location crowding metrics
        self.location_crowding = defaultdict(int)
        self.transport_usage = defaultdict(int)
        
        # Infection tracking
        self.new_infections = defaultdict(int)  # epoch -> count
        self.cumulative_infections = 0
        
        # Agent attributes
        self.agent_attributes = {}
        
    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize with agent data"""
        self.logger.info(f"Initializing {self.name} with {len(all_agent_data)} agents")
        
        # Store relevant agent attributes
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            self.agent_attributes[agent_id] = {
                "age": agent_data.get("basic_info", {}).get("age", 30),
                "employment_status": agent_data.get("economic_attributes", {}).get("employment_status", "employed")
            }
        
        # Initialize system state
        self.system_state = {
            "total_agents": len(all_agent_data),
            "current_infection_rate": self.base_infection_rate,
            "restriction_level": 0
        }
        
    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent mobility decisions and calculate transmission"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        self.logger.info(f"Processing disease transmission for epoch {epoch}")
        
        # Get current health code policy impact
        current_impact = self.health_code_impact.get(str(epoch), {})
        restriction_level = current_impact.get("restriction_level", 0)
        transmission_reduction = current_impact.get("transmission_reduction", 0)
        
        self.system_state["restriction_level"] = restriction_level
        
        # Reset daily tracking
        self.location_crowding.clear()
        self.transport_usage.clear()
        self.daily_contacts.clear()
        
        # Process mobility decisions
        agents_going_out = []
        
        for agent_id, decisions in agent_decisions.items():
            mobility_decision = decisions.get(self.name, {})
            
            go_out = mobility_decision.get("go_out_decision", "no")
            destination = mobility_decision.get("destination_type", "none")
            transport = mobility_decision.get("transport_mode", "none")
            
            # Handle case where destination might be a list (fix TypeError)
            if isinstance(destination, list):
                destination = destination[0] if destination else "none"
            if isinstance(transport, list):
                transport = transport[0] if transport else "none"
                
            self.agent_mobility[agent_id] = {
                "go_out": go_out == "yes",
                "destination": destination,
                "transport": transport
            }
            
            if go_out == "yes":
                agents_going_out.append(agent_id)
                self.location_crowding[destination] += 1
                self.transport_usage[transport] += 1
        
        # Calculate contacts and transmission
        self._calculate_contacts(agents_going_out)
        new_infection_count = self._simulate_transmission(epoch, transmission_reduction)
        
        # Update metrics
        self.new_infections[epoch] = new_infection_count
        self.cumulative_infections += new_infection_count
        
        # Calculate effective R0
        active_infections = self._get_from_blackboard("active_infections", 10)
        effective_r0 = new_infection_count / max(active_infections, 1)
        
        # Post to blackboard
        self._post_to_blackboard("new_infections", new_infection_count)
        self._post_to_blackboard("mobility_rate", len(agents_going_out) / len(agent_decisions))
        self._post_to_blackboard("effective_r0", effective_r0)
        
        # Log statistics
        self.logger.info(f"Epoch {epoch}: {len(agents_going_out)}/{len(agent_decisions)} agents went out")
        self.logger.info(f"New infections: {new_infection_count}, Cumulative: {self.cumulative_infections}")
        self.logger.info(f"Location crowding: {dict(self.location_crowding)}")
        self.logger.info(f"Transport usage: {dict(self.transport_usage)}")
        
    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Return transmission risk information for a specific agent"""
        epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Get current policy restrictions
        current_impact = self.health_code_impact.get(str(epoch), {})
        restriction_level = current_impact.get("restriction_level", 0)
        
        # Calculate current infection rate in population
        active_infections = self._get_from_blackboard("active_infections", 0)
        total_agents = len(self.agent_attributes)
        current_infection_rate = active_infections / max(total_agents, 1)
        
        # Calculate location-specific risks
        location_risks = {}
        for location, multiplier_key in [
            ("work", "work_multiplier"),
            ("shopping", "shopping_multiplier"),
            ("social", "social_gathering_multiplier")
        ]:
            base_risk = self.transmission_params.get(multiplier_key, 1.0) * self.base_infection_rate
            adjusted_risk = base_risk * (1 - current_impact.get("transmission_reduction", 0))
            # Scale by actual infection rate in population
            final_risk = adjusted_risk * (1 + current_infection_rate * 10)  # Risk increases with more infections
            location_risks[location] = min(final_risk, 0.8)  # Cap at 80%
        
        # Public transport risk
        transport_risk = self.base_infection_rate * self.transmission_params.get("public_transport_multiplier", 3.0)
        transport_risk = transport_risk * (1 - current_impact.get("transmission_reduction", 0))
        transport_risk = transport_risk * (1 + current_infection_rate * 10)
        
        # Generate risk description
        if current_infection_rate < 0.01:
            risk_description = "Very low community transmission"
        elif current_infection_rate < 0.05:
            risk_description = "Low community transmission"
        elif current_infection_rate < 0.15:
            risk_description = "Moderate community transmission"
        else:
            risk_description = "High community transmission"
        
        return {
            "infection_rate": current_infection_rate,
            "risk_description": risk_description,
            "health_code_restrictions": {
                "level": restriction_level,
                "description": self._get_restriction_description(restriction_level)
            },
            "public_transport_risk": {
                "risk_level": min(transport_risk, 0.8),
                "crowding": self.transport_usage.get("public_transport", 0)
            },
            "location_risks": location_risks,
            "recent_outbreak_areas": self._get_outbreak_areas(),
            "daily_life_impact": self._get_daily_life_impact(restriction_level)
        }
    
    def _calculate_contacts(self, agents_going_out: List[str]):
        """Calculate contacts between agents based on their destinations"""
        # Group agents by destination
        destination_groups = defaultdict(list)
        transport_groups = defaultdict(list)
        
        for agent_id in agents_going_out:
            mobility = self.agent_mobility[agent_id]
            if mobility["destination"] != "none":
                destination_groups[mobility["destination"]].append(agent_id)
            if mobility["transport"] != "none":
                transport_groups[mobility["transport"]].append(agent_id)
        
        # Generate contacts within same locations
        for location, agents in destination_groups.items():
            if len(agents) > 1:
                # Each agent contacts a random subset of others at the location
                for agent in agents:
                    # Higher contact rates for certain locations
                    if location == "social":
                        num_contacts = min(random.randint(3, 8), len(agents) - 1)
                    elif location == "shopping":
                        num_contacts = min(random.randint(2, 6), len(agents) - 1)
                    elif location == "work":
                        num_contacts = min(random.randint(2, 5), len(agents) - 1)
                    else:
                        num_contacts = min(random.randint(1, 3), len(agents) - 1)
                    
                    if num_contacts > 0:
                        contacts = random.sample([a for a in agents if a != agent], num_contacts)
                        self.daily_contacts[agent].extend(contacts)
        
        # Additional contacts in public transport - higher contact rates
        public_transport_agents = transport_groups.get("public_transport", [])
        if len(public_transport_agents) > 1:
            for agent in public_transport_agents:
                num_contacts = min(random.randint(4, 12), len(public_transport_agents) - 1)
                if num_contacts > 0:
                    contacts = random.sample([a for a in public_transport_agents if a != agent], num_contacts)
                    self.daily_contacts[agent].extend(contacts)
        
        # Random encounters for agents going out (community transmission)
        if len(agents_going_out) > 2:
            for agent in agents_going_out:
                # Small chance of random encounters outside of main destinations
                if random.random() < 0.3:  # 30% chance
                    potential_encounters = [a for a in agents_going_out if a != agent]
                    if potential_encounters:
                        num_random_contacts = min(random.randint(1, 2), len(potential_encounters))
                        random_contacts = random.sample(potential_encounters, num_random_contacts)
                        self.daily_contacts[agent].extend(random_contacts)
    
    def _simulate_transmission(self, epoch: int, transmission_reduction: float) -> int:
        """Simulate disease transmission based on contacts"""
        # Get infected agents from blackboard
        infected_agents = self._get_from_blackboard("infected_agents", set())
        
        new_infections = set()
        
        # If no infected agents, introduce some random infections based on external sources
        if len(infected_agents) == 0 and epoch <= 2:
            # Simulate external introduction of infections (imported cases, etc.)
            num_external_infections = random.randint(1, 3)  # 1-3 external infections
            all_agents = list(self.agent_mobility.keys())
            if all_agents:
                external_infected = random.sample(all_agents, min(num_external_infections, len(all_agents)))
                new_infections.update(external_infected)
                self.logger.info(f"Introduced {len(external_infected)} external infections")
        
        # Simulate contact-based transmission
        for agent_id, contacts in self.daily_contacts.items():
            if agent_id in infected_agents:
                continue  # Already infected
            
            # Check each contact for potential transmission
            for contact_id in contacts:
                if contact_id in infected_agents:
                    # Calculate transmission probability
                    mobility = self.agent_mobility[agent_id]
                    location = mobility.get("destination", "none")
                    
                    # Get location-specific multiplier
                    multiplier = 1.0
                    if location == "work":
                        multiplier = self.transmission_params.get("work_multiplier", 1.5)
                    elif location == "shopping":
                        multiplier = self.transmission_params.get("shopping_multiplier", 1.8)
                    elif location == "social":
                        multiplier = self.transmission_params.get("social_gathering_multiplier", 2.5)
                    
                    # Apply transport multiplier if using public transport
                    if mobility.get("transport") == "public_transport":
                        multiplier *= 1.5
                    
                    # Calculate final transmission probability
                    transmission_prob = self.base_infection_rate * multiplier * (1 - transmission_reduction)
                    
                    if random.random() < transmission_prob:
                        new_infections.add(agent_id)
                        break  # One infection per day is enough
        
        # Additional random environmental transmission for agents who went out
        if epoch <= 1:  # Higher environmental risk in early epochs
            agents_going_out = [aid for aid, mobility in self.agent_mobility.items() if mobility.get("go_out")]
            for agent_id in agents_going_out:
                if agent_id not in infected_agents and agent_id not in new_infections:
                    # Small chance of environmental transmission
                    env_transmission_prob = self.base_infection_rate * 0.3 * (1 - transmission_reduction)
                    if random.random() < env_transmission_prob:
                        new_infections.add(agent_id)
        
        # Post new infections to blackboard
        self._post_to_blackboard("new_infection_list", list(new_infections))
        
        return len(new_infections)
    
    def _get_restriction_description(self, level: float) -> str:
        """Get human-readable description of restriction level"""
        if level < 0.2:
            return "Minimal restrictions - normal movement allowed"
        elif level < 0.5:
            return "Moderate restrictions - some venues require health codes"
        elif level < 0.8:
            return "High restrictions - most venues require green health codes"
        else:
            return "Strict restrictions - only essential movement allowed"
    
    def _get_outbreak_areas(self) -> List[str]:
        """Identify areas with high infection rates"""
        outbreak_areas = []
        
        for location, count in self.location_crowding.items():
            if count > 20:  # Threshold for crowding
                outbreak_areas.append(location)
        
        return outbreak_areas[:3]  # Return top 3
    
    def _get_daily_life_impact(self, restriction_level: float) -> str:
        """Get description of impact on daily life"""
        if restriction_level < 0.2:
            return "Normal daily activities can continue with basic precautions"
        elif restriction_level < 0.5:
            return "Some restrictions on gatherings, work and essential activities continue"
        elif restriction_level < 0.8:
            return "Significant restrictions, only essential activities recommended"
        else:
            return "Strict lockdown, only emergency activities allowed"
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the disease transmission system"""
        self.logger.info("=== Disease Transmission System Evaluation ===")
        
        # Calculate mobility trends
        mobility_by_epoch = []
        destination_distribution = defaultdict(lambda: defaultdict(int))
        transport_distribution = defaultdict(lambda: defaultdict(int))
        
        # Get all epochs that were processed
        processed_epochs = list(range(self.current_time.get_current_epoch() + 1)) if self.current_time else []
        
        for epoch in processed_epochs:
            epoch_mobility = 0
            epoch_agents = 0
            
            # Count mobility for this epoch by checking agent_mobility data
            for agent_id, mobility in self.agent_mobility.items():
                epoch_agents += 1
                if mobility.get("go_out"):
                    epoch_mobility += 1
                    destination_distribution[epoch][mobility.get("destination", "none")] += 1
                    transport_distribution[epoch][mobility.get("transport", "none")] += 1
            
            if epoch_agents > 0:
                mobility_rate = epoch_mobility / epoch_agents
                mobility_by_epoch.append(mobility_rate)
                self.logger.info(f"Epoch {epoch}: {epoch_mobility}/{epoch_agents} agents went out (rate: {mobility_rate:.3f})")
        
        # Calculate infection metrics
        infection_timeline = []
        for epoch in processed_epochs:
            infection_count = self.new_infections.get(epoch, 0)
            infection_timeline.append(infection_count)
        
        # Calculate R0 estimates
        r0_estimates = []
        for i in range(1, len(infection_timeline)):
            if infection_timeline[i-1] > 0:
                r0_estimates.append(infection_timeline[i] / infection_timeline[i-1])
        
        # Impact of restrictions
        restriction_impact = self._analyze_restriction_impact()
        
        evaluation_results = {
            "total_infections": self.cumulative_infections,
            "infection_timeline": infection_timeline,
            "peak_infections": max(infection_timeline) if infection_timeline else 0,
            "peak_epoch": infection_timeline.index(max(infection_timeline)) if infection_timeline and max(infection_timeline) > 0 else -1,
            "average_mobility_rate": sum(mobility_by_epoch) / len(mobility_by_epoch) if mobility_by_epoch else 0,
            "mobility_by_epoch": mobility_by_epoch,
            "destination_preferences": {
                epoch: dict(dests) for epoch, dests in destination_distribution.items()
            },
            "transport_preferences": {
                epoch: dict(trans) for epoch, trans in transport_distribution.items()
            },
            "r0_estimates": r0_estimates,
            "average_r0": sum(r0_estimates) / len(r0_estimates) if r0_estimates else 0,
            "restriction_effectiveness": restriction_impact
        }
        
        # Log key metrics
        self.logger.info(f"Total infections from transmission system: {self.cumulative_infections}")
        self.logger.info(f"Peak infections: {evaluation_results['peak_infections']} at epoch {evaluation_results['peak_epoch']}")
        self.logger.info(f"Average mobility rate: {evaluation_results['average_mobility_rate']:.3f}")
        self.logger.info(f"Average R0: {evaluation_results['average_r0']:.3f}")
        
        # Log time series data
        self.logger.info("Infection timeline:")
        for epoch, count in enumerate(infection_timeline):
            self.logger.info(f"  Epoch {epoch}: {count} new infections")
        
        self.logger.info("Mobility rates by epoch:")
        for epoch, rate in enumerate(mobility_by_epoch):
            self.logger.info(f"  Epoch {epoch}: {rate:.3f}")
        
        self.evaluation_results = evaluation_results
        return evaluation_results
    
    def _analyze_restriction_impact(self) -> Dict[str, Any]:
        """Analyze the impact of health code restrictions on transmission"""
        impact_analysis = {}
        
        # Compare infection rates before and after major policy changes
        policy_epochs = [2, 3, 4, 5]  # Based on config
        
        for epoch in policy_epochs:
            if epoch in self.new_infections and (epoch - 1) in self.new_infections:
                before = self.new_infections[epoch - 1]
                after = self.new_infections[epoch]
                
                reduction = (before - after) / max(before, 1)
                
                impact_analysis[f"epoch_{epoch}_impact"] = {
                    "infections_before": before,
                    "infections_after": after,
                    "reduction_rate": reduction,
                    "effectiveness": "high" if reduction > 0.3 else "moderate" if reduction > 0.1 else "low"
                }
        
        return impact_analysis 