import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class TransportChoiceSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Carbon policy parameters
        self.carbon_policies = config.get("carbon_policies", {})
        
        # Agent transportation tracking
        self.agent_transport_choices = {}  # agent_id -> primary transport mode
        self.agent_credit_balances = {}    # agent_id -> current carbon credit balance
        self.agent_daily_trips = {}        # agent_id -> average daily trips
        
        # System statistics
        self.mode_share = defaultdict(int)  # transport mode -> count
        self.emissions_saved = 0
        self.credit_market_volume = 0
        
        # Historical data
        self.mode_share_history = []
        self.emissions_history = []
        self.credit_price_history = []
        
        self.logger.info("TransportChoiceSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize agent transport profiles"""
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Initial transport mode based on residence type and income
            residence_type = agent_data.get("basic_info", {}).get("residence_type", "").lower()
            income_level = agent_data.get("economic_attributes", {}).get("income_level", "").lower()
            
            # Set initial transportation mode
            if "urban" in residence_type:
                if "high" in income_level:
                    mode = random.choices(["private_car", "public_transit"], weights=[0.7, 0.3])[0]
                else:
                    mode = random.choices(["private_car", "public_transit", "bicycle"], weights=[0.3, 0.5, 0.2])[0]
            elif "suburban" in residence_type:
                mode = random.choices(["private_car", "public_transit"], weights=[0.8, 0.2])[0]
            else:  # rural
                mode = "private_car"  # Limited options in rural areas
            
            self.agent_transport_choices[agent_id] = mode
            self.mode_share[mode] += 1
            
            # Set initial carbon credit balance
            self.agent_credit_balances[agent_id] = 0
            
            # Set initial trip frequency based on residence
            if "urban" in residence_type:
                trips = random.choices([2, 3, 4], weights=[0.2, 0.5, 0.3])[0]
            elif "suburban" in residence_type:
                trips = random.choices([2, 3, 4, 5], weights=[0.1, 0.3, 0.4, 0.2])[0]
            else:  # rural
                trips = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
                
            self.agent_daily_trips[agent_id] = trips
        
        self.logger.info(f"Initialized transport profiles for {len(all_agent_data)} agents")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide transport and carbon credit information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.carbon_policies.get(str(current_epoch), 
                                                self.carbon_policies.get("0", {}))
        
        # Get agent's current transport choice
        current_mode = self.agent_transport_choices.get(agent_id, "private_car")
        current_credit_balance = self.agent_credit_balances.get(agent_id, 0)
        daily_trips = self.agent_daily_trips.get(agent_id, 3)
        
        # Calculate monthly carbon emissions for current choice
        monthly_emissions = self._calculate_monthly_emissions(current_mode, daily_trips)
        
        # Calculate carbon credits required for current usage
        monthly_credits_required = 0
        if current_policy.get("system_active", False):
            credits_per_km = current_policy.get(f"{current_mode}_credits_per_km", 0)
            # Assume 20 working days per month and 10km per trip
            monthly_credits_required = credits_per_km * daily_trips * 20 * 10
        
        # Check if credit system is active
        credit_system_active = current_policy.get("system_active", False)
        
        # Calculate carbon prices and credit information
        credit_price = current_policy.get("credit_price", 0)
        credit_allocation = current_policy.get("monthly_credit_allocation", 0)
        
        # Calculate potential transportation costs with credits
        transport_costs = {}
        if credit_system_active:
            for mode in ["private_car", "public_transit", "bicycle", "walk"]:
                base_cost = self._calculate_transport_base_cost(mode)
                credits_per_km = current_policy.get(f"{mode}_credits_per_km", 0)
                monthly_credits = credits_per_km * daily_trips * 20 * 10
                
                # Calculate net cost including carbon credits
                if monthly_credits > 0:  # Requires credits
                    if monthly_credits > credit_allocation:
                        # Need to buy additional credits
                        additional_credits = monthly_credits - credit_allocation
                        credit_cost = additional_credits * credit_price
                        net_cost = base_cost + credit_cost
                    else:
                        net_cost = base_cost
                else:  # Earns credits (negative credit consumption)
                    earned_credits = abs(monthly_credits)
                    credit_earnings = earned_credits * credit_price
                    net_cost = base_cost - credit_earnings
                
                transport_costs[mode] = {
                    "base_cost": base_cost,
                    "carbon_credit_cost": monthly_credits * credit_price if monthly_credits > 0 else 0,
                    "carbon_credit_earnings": abs(monthly_credits) * credit_price if monthly_credits < 0 else 0,
                    "net_cost": net_cost
                }
        else:
            # If credit system is not active, only show base costs
            for mode in ["private_car", "public_transit", "bicycle", "walk"]:
                base_cost = self._calculate_transport_base_cost(mode)
                transport_costs[mode] = {
                    "base_cost": base_cost,
                    "carbon_credit_cost": 0,
                    "carbon_credit_earnings": 0,
                    "net_cost": base_cost
                }
        
        return {
            "transport_options": {
                "available_modes": ["private_car", "public_transit", "bicycle", "walk"],
                "your_current_choice": current_mode,
                "daily_trips": daily_trips,
                "monthly_emissions_kg": monthly_emissions
            },
            "carbon_prices": {
                "credit_system_active": credit_system_active,
                "credit_price": credit_price,
                "monthly_allocation": credit_allocation,
                "your_current_balance": current_credit_balance,
                "credits_required_monthly": monthly_credits_required
            },
            "credit_balance": {
                "current_balance": current_credit_balance,
                "sufficient_for_current_usage": current_credit_balance + credit_allocation >= monthly_credits_required,
                "projected_next_month": current_credit_balance + credit_allocation - monthly_credits_required
            },
            "public_transit_quality": {
                "frequency": "good" if "urban" in self.system_state.get(f"residence_{agent_id}", "") else "limited",
                "coverage": "extensive" if "urban" in self.system_state.get(f"residence_{agent_id}", "") else "partial",
                "comparative_time_cost": 1.5  # Ratio compared to car travel time
            },
            "transport_costs": transport_costs
        }
    
    def _calculate_monthly_emissions(self, mode: str, daily_trips: int) -> float:
        """Calculate monthly CO2 emissions in kg"""
        # Emissions factors in g CO2/km
        emissions_factors = {
            "private_car": 120,
            "public_transit": 30,
            "bicycle": 0,
            "walk": 0,
            "mixed": 60  # Average of car and transit
        }
        
        # Assume 20 working days per month and 10km per trip
        monthly_km = daily_trips * 20 * 10
        
        return (emissions_factors.get(mode, 0) * monthly_km) / 1000  # Convert to kg
    
    def _calculate_transport_base_cost(self, mode: str) -> float:
        """Calculate base cost of transport mode (without carbon credits)"""
        # Monthly costs in monetary units
        base_costs = {
            "private_car": 2000,  # Includes fuel, maintenance, parking
            "public_transit": 500,  # Monthly pass
            "bicycle": 100,  # Maintenance and depreciation
            "walk": 0,
            "mixed": 1200  # Average of multiple modes
        }
        
        return base_costs.get(mode, 0)

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process agent transport decisions"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        current_policy = self.carbon_policies.get(str(current_epoch), 
                                                self.carbon_policies.get("0", {}))
        
        # Reset mode share for this epoch
        self.mode_share = defaultdict(int)
        total_emissions = 0
        total_credit_transactions = 0
        
        # Check if carbon credit system is active
        credit_system_active = current_policy.get("system_active", False)
        credit_allocation = current_policy.get("monthly_credit_allocation", 0)
        
        for agent_id, decisions in agent_decisions.items():
            # Store residence type for future reference
            if agent_id in self.system_state:
                residence_type = decisions.get("basic_info", {}).get("residence_type", "")
                if residence_type:
                    self.system_state[f"residence_{agent_id}"] = residence_type
            
            # Add monthly credit allocation if system is active
            if credit_system_active:
                self.agent_credit_balances[agent_id] = self.agent_credit_balances.get(agent_id, 0) + credit_allocation
            
            if "TransportChoiceSystem" not in decisions:
                # Use previous mode if no new decision
                current_mode = self.agent_transport_choices.get(agent_id, "private_car")
                self.mode_share[current_mode] += 1
                
                # Calculate emissions for default behavior
                daily_trips = self.agent_daily_trips.get(agent_id, 3)
                total_emissions += self._calculate_monthly_emissions(current_mode, daily_trips)
                
                continue
                
            decision = decisions["TransportChoiceSystem"]
            chosen_mode = decision.get("primary_transport_mode", "private_car")
            credit_strategy = decision.get("carbon_credit_strategy", "ignore_system")
            trip_frequency = decision.get("trip_frequency", "medium")
            
            # Update transport choice
            self.agent_transport_choices[agent_id] = chosen_mode
            self.mode_share[chosen_mode] += 1
            
            # Convert trip frequency to numeric value
            if trip_frequency == "low":
                daily_trips = random.uniform(1, 2)
            elif trip_frequency == "medium":
                daily_trips = random.uniform(3, 4)
            else:  # high
                daily_trips = random.uniform(5, 7)
                
            self.agent_daily_trips[agent_id] = daily_trips
            
            # Calculate emissions
            monthly_emissions = self._calculate_monthly_emissions(chosen_mode, daily_trips)
            total_emissions += monthly_emissions
            
            # Process credit consumption if system is active
            if credit_system_active:
                credits_per_km = current_policy.get(f"{chosen_mode}_credits_per_km", 0)
                # Assume 20 working days per month and 10km per trip
                monthly_credit_change = credits_per_km * daily_trips * 20 * 10
                
                # Update agent's credit balance
                current_balance = self.agent_credit_balances.get(agent_id, 0)
                
                # Process credit trading if enabled
                if current_policy.get("credit_trading_allowed", False) and credit_strategy == "buy_credits":
                    if monthly_credit_change > 0 and current_balance < monthly_credit_change:
                        # Agent needs to buy credits
                        credits_to_buy = monthly_credit_change - current_balance
                        # Record transaction volume
                        total_credit_transactions += credits_to_buy
                
                # Update balance
                self.agent_credit_balances[agent_id] = current_balance - monthly_credit_change
        
        # Calculate emissions saved compared to baseline (all car)
        baseline_emissions = len(agent_decisions) * self._calculate_monthly_emissions("private_car", 3)
        emissions_saved = baseline_emissions - total_emissions
        self.emissions_saved = emissions_saved
        
        # Record historical data
        self.mode_share_history.append(dict(self.mode_share))
        self.emissions_history.append(emissions_saved)
        self.credit_market_volume = total_credit_transactions
        
        # Store credit price for historical tracking
        self.credit_price_history.append(current_policy.get("credit_price", 0))
        
        self.logger.info(f"Epoch {current_epoch}: Mode share={dict(self.mode_share)}, "
                        f"Emissions saved={emissions_saved:.1f}kg, "
                        f"Credit transactions={total_credit_transactions}")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate carbon credit transport policy effectiveness"""
        # Calculate mode shift
        initial_mode_share = self.mode_share_history[0] if self.mode_share_history else {}
        final_mode_share = self.mode_share_history[-1] if self.mode_share_history else {}
        
        # Calculate car usage reduction
        initial_car_share = initial_mode_share.get("private_car", 0) / sum(initial_mode_share.values()) if sum(initial_mode_share.values()) > 0 else 0
        final_car_share = final_mode_share.get("private_car", 0) / sum(final_mode_share.values()) if sum(final_mode_share.values()) > 0 else 0
        car_share_reduction = (initial_car_share - final_car_share) * 100
        
        # Calculate sustainable mode increase
        sustainable_modes = ["public_transit", "bicycle", "walk"]
        initial_sustainable = sum(initial_mode_share.get(mode, 0) for mode in sustainable_modes)
        final_sustainable = sum(final_mode_share.get(mode, 0) for mode in sustainable_modes)
        initial_sustainable_share = initial_sustainable / sum(initial_mode_share.values()) if sum(initial_mode_share.values()) > 0 else 0
        final_sustainable_share = final_sustainable / sum(final_mode_share.values()) if sum(final_mode_share.values()) > 0 else 0
        sustainable_share_increase = (final_sustainable_share - initial_sustainable_share) * 100
        
        # Calculate emissions reduction
        total_emissions_saved = sum(self.emissions_history)
        avg_monthly_emissions_saved = total_emissions_saved / len(self.emissions_history) if self.emissions_history else 0
        
        # Analyze policy phases
        pre_policy_emissions = np.mean(self.emissions_history[:2]) if len(self.emissions_history) >= 2 else 0
        early_policy_emissions = np.mean(self.emissions_history[2:4]) if len(self.emissions_history) >= 4 else 0
        strict_policy_emissions = np.mean(self.emissions_history[4:]) if len(self.emissions_history) >= 6 else 0
        
        evaluation_results = {
            "mode_shift_metrics": {
                "initial_mode_share": initial_mode_share,
                "final_mode_share": final_mode_share,
                "car_usage_reduction_pct": car_share_reduction,
                "sustainable_mode_increase_pct": sustainable_share_increase
            },
            "emissions_metrics": {
                "total_emissions_saved_kg": total_emissions_saved,
                "average_monthly_savings_kg": avg_monthly_emissions_saved,
                "emissions_trend": "increasing" if strict_policy_emissions > early_policy_emissions else "stable" if abs(strict_policy_emissions - early_policy_emissions) < 100 else "decreasing"
            },
            "credit_market_metrics": {
                "final_credit_price": self.credit_price_history[-1] if self.credit_price_history else 0,
                "price_trend": "increasing" if len(self.credit_price_history) > 1 and self.credit_price_history[-1] > self.credit_price_history[0] else "stable",
                "market_volume": self.credit_market_volume
            },
            "policy_effectiveness": {
                "reduced_car_usage": car_share_reduction > 10,
                "increased_sustainable_transport": sustainable_share_increase > 15,
                "significant_emissions_reduction": avg_monthly_emissions_saved > 1000,
                "overall_success": car_share_reduction > 10 and sustainable_share_increase > 15 and avg_monthly_emissions_saved > 1000
            },
            "policy_phase_comparison": {
                "pre_policy_emissions_saved": pre_policy_emissions,
                "initial_policy_emissions_saved": early_policy_emissions,
                "strict_policy_emissions_saved": strict_policy_emissions,
                "policy_strengthening_effect": (strict_policy_emissions - early_policy_emissions) / early_policy_emissions * 100 if early_policy_emissions > 0 else 0
            },
            "time_series": {
                "mode_share_history": self.mode_share_history,
                "emissions_saved_history": self.emissions_history,
                "credit_price_history": self.credit_price_history
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")

        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "mode_share": dict(self.mode_share),
            "emissions_saved": self.emissions_saved,
            "credit_market_volume": self.credit_market_volume,
            "car_share_pct": self.mode_share.get("private_car", 0) / sum(self.mode_share.values()) * 100 if sum(self.mode_share.values()) > 0 else 0,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 