import random
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from src.subsystems.base import SocialSystemBase
from src.utils.logger import get_logger

class CarbonTradingSystem(SocialSystemBase):
    def __init__(self, name: str, config: Dict[str, Any], **kwargs):
        super().__init__(name, config)
        self.logger = get_logger(name)
        
        # Carbon market parameters
        self.transport_emissions = config.get("transport_emissions", {
            "private_car": 120,  # g CO2/km
            "public_transit": 30,
            "bicycle": 0,
            "walk": 0
        })
        
        self.initial_credit_supply = config.get("initial_credit_supply", 10000)
        self.price_elasticity = config.get("price_elasticity", 0.3)
        
        # Market state
        self.current_price = 1.0
        self.current_supply = self.initial_credit_supply
        self.current_demand = 0
        self.credit_trading_enabled = False
        
        # Trading history
        self.buyers = defaultdict(int)  # agent_id -> credits bought
        self.sellers = defaultdict(int)  # agent_id -> credits sold
        self.monthly_volume = 0
        
        # Historical data
        self.price_history = []
        self.volume_history = []
        self.supply_history = []
        self.demand_history = []
        self.emissions_history = []
        
        self.logger.info("CarbonTradingSystem initialized")

    def init(self, all_agent_data: List[Dict[str, Any]]):
        """Initialize emission profiles"""
        # Store agent types for emissions calculations
        for agent_data in all_agent_data:
            agent_id = str(agent_data.get("id"))
            
            # Store residence type and income for reference
            residence_type = agent_data.get("basic_info", {}).get("residence_type", "").lower()
            income_level = agent_data.get("economic_attributes", {}).get("income_level", "").lower()
            
            self.system_state[f"residence_{agent_id}"] = residence_type
            self.system_state[f"income_{agent_id}"] = income_level
        
        self.logger.info(f"Initialized emission profiles for {len(all_agent_data)} agents")

    def get_system_information(self, agent_id: str) -> Dict[str, Any]:
        """Provide carbon market information to agents"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Determine if trading is enabled for current epoch
        policy_key = str(current_epoch)
        if policy_key in self.system_state.get("carbon_policies", {}):
            self.credit_trading_enabled = self.system_state["carbon_policies"][policy_key].get("credit_trading_allowed", False)
        
        # Get agent's trading history
        bought_credits = self.buyers.get(agent_id, 0)
        sold_credits = self.sellers.get(agent_id, 0)
        
        # Calculate price trend
        price_trend = "stable"
        if len(self.price_history) > 1:
            price_change = (self.current_price - self.price_history[-2]) / self.price_history[-2] if self.price_history[-2] > 0 else 0
            if price_change > 0.05:
                price_trend = "rising"
            elif price_change < -0.05:
                price_trend = "falling"
        
        # Calculate supply-demand balance
        market_balance = "balanced"
        if self.current_demand > self.current_supply * 1.1:
            market_balance = "undersupplied"
        elif self.current_supply > self.current_demand * 1.1:
            market_balance = "oversupplied"
        
        # Calculate total emission reductions
        total_emission_reduction = sum(self.emissions_history)
        
        return {
            "market_prices": {
                "current_credit_price": self.current_price,
                "price_trend": price_trend,
                "trading_enabled": self.credit_trading_enabled,
                "your_trading_history": {
                    "credits_bought": bought_credits,
                    "credits_sold": sold_credits,
                    "net_position": sold_credits - bought_credits
                }
            },
            "trading_volume": {
                "monthly_volume": self.monthly_volume,
                "volume_trend": "increasing" if len(self.volume_history) > 1 and self.volume_history[-1] > self.volume_history[-2] else "decreasing" if len(self.volume_history) > 1 and self.volume_history[-1] < self.volume_history[-2] else "stable",
                "market_liquidity": "high" if self.monthly_volume > 1000 else "medium" if self.monthly_volume > 500 else "low"
            },
            "emission_statistics": {
                "total_emission_reduction_kg": total_emission_reduction,
                "market_balance": market_balance,
                "transport_modes_emissions": {
                    mode: f"{emissions} g/km" for mode, emissions in self.transport_emissions.items()
                }
            }
        }

    def step(self, agent_decisions: Dict[str, Dict[str, Any]]):
        """Process carbon market dynamics"""
        current_epoch = self.current_time.get_current_epoch() if self.current_time else 0
        
        # Reset monthly statistics
        self.monthly_volume = 0
        self.buyers.clear()
        self.sellers.clear()
        total_emissions_saved = 0
        
        # Check carbon policy from TransportChoiceSystem
        carbon_policies = {}
        if "carbon_policies" in self.system_state:
            carbon_policies = self.system_state["carbon_policies"]
        elif "carbon_policy" in self.system_state:
            carbon_policies = {str(current_epoch): self.system_state["carbon_policy"]}
        
        current_policy = carbon_policies.get(str(current_epoch), {})
        self.credit_trading_enabled = current_policy.get("credit_trading_allowed", False)
        
        # Update market supply based on policy allocation
        if current_policy.get("system_active", False):
            monthly_allocation = current_policy.get("monthly_credit_allocation", 0)
            population = len(agent_decisions)
            
            # Add new credits to supply
            new_credits = monthly_allocation * population
            self.current_supply += new_credits
        
        # Process trading decisions and calculate emissions
        self.current_demand = 0
        total_car_km = 0
        total_sustainable_km = 0
        
        for agent_id, decisions in agent_decisions.items():
            # Extract transport decisions from TransportChoiceSystem
            if "TransportChoiceSystem" in decisions:
                transport_decision = decisions["TransportChoiceSystem"]
                transport_mode = transport_decision.get("primary_transport_mode", "private_car")
                
                # Calculate kilometers traveled
                # Assume 20 working days and average trip distance
                trip_frequency = transport_decision.get("trip_frequency", "medium")
                if trip_frequency == "low":
                    daily_trips = 1.5
                elif trip_frequency == "medium":
                    daily_trips = 3.5
                else:  # high
                    daily_trips = 6.0
                
                monthly_km = daily_trips * 20 * 10  # 10km per trip
                
                # Track car vs sustainable kilometers
                if transport_mode == "private_car":
                    total_car_km += monthly_km
                elif transport_mode in ["public_transit", "bicycle", "walk"]:
                    total_sustainable_km += monthly_km
                    
                # Calculate emissions for this agent
                mode_emission = self.transport_emissions.get(transport_mode, 0)
                monthly_emissions = (mode_emission * monthly_km) / 1000  # kg
                
                # Calculate baseline emissions (if everyone drove cars)
                baseline_emissions = (self.transport_emissions["private_car"] * monthly_km) / 1000
                emissions_saved = baseline_emissions - monthly_emissions
                total_emissions_saved += emissions_saved
                
                # Process credit trading strategy
                credit_strategy = transport_decision.get("carbon_credit_strategy", "ignore_system")
                
                if self.credit_trading_enabled and credit_strategy in ["buy_credits", "conserve_credits"]:
                    # Estimate credit needs
                    credit_balance = self.system_state.get(f"credit_balance_{agent_id}", 0)
                    credits_per_km = current_policy.get(f"{transport_mode}_credits_per_km", 0)
                    monthly_credit_change = credits_per_km * monthly_km
                    
                    if monthly_credit_change > 0 and credit_balance < monthly_credit_change and credit_strategy == "buy_credits":
                        # Need to buy credits
                        credits_to_buy = monthly_credit_change - credit_balance
                        self.buyers[agent_id] += credits_to_buy
                        self.monthly_volume += credits_to_buy
                        self.current_demand += credits_to_buy
                    
                    elif monthly_credit_change < 0 and credit_strategy == "conserve_credits":
                        # Can sell excess credits
                        credits_to_sell = abs(monthly_credit_change)
                        self.sellers[agent_id] += credits_to_sell
                        self.monthly_volume += credits_to_sell
                        self.current_supply += credits_to_sell
        
        # Update market price based on supply and demand
        if self.credit_trading_enabled and self.current_supply > 0:
            supply_demand_ratio = self.current_demand / self.current_supply
            price_adjustment = (supply_demand_ratio - 1) * self.price_elasticity
            self.current_price = max(0.5, min(5.0, self.current_price * (1 + price_adjustment)))
        
        # Record historical data
        self.price_history.append(self.current_price)
        self.volume_history.append(self.monthly_volume)
        self.supply_history.append(self.current_supply)
        self.demand_history.append(self.current_demand)
        self.emissions_history.append(total_emissions_saved)
        
        # Calculate car share
        total_km = total_car_km + total_sustainable_km
        car_share = total_car_km / total_km if total_km > 0 else 0
        
        self.logger.info(f"Epoch {current_epoch}: Credit price={self.current_price:.2f}, "
                        f"Trading volume={self.monthly_volume}, "
                        f"Car share={car_share:.1%}, "
                        f"Emissions saved={total_emissions_saved:.1f}kg")

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate carbon trading system effectiveness"""
        # Calculate price trends
        initial_price = self.price_history[0] if self.price_history else 1.0
        final_price = self.price_history[-1] if self.price_history else 1.0
        price_change_pct = ((final_price - initial_price) / initial_price) * 100 if initial_price > 0 else 0
        
        # Calculate trading volume growth
        initial_volume = np.mean(self.volume_history[:2]) if len(self.volume_history) >= 2 else 0
        final_volume = np.mean(self.volume_history[-2:]) if len(self.volume_history) >= 2 else 0
        volume_growth_pct = ((final_volume - initial_volume) / initial_volume) * 100 if initial_volume > 0 else 0
        
        # Calculate emissions reduction by phase
        pre_policy_emissions = np.sum(self.emissions_history[:2]) if len(self.emissions_history) >= 2 else 0
        policy_phase1_emissions = np.sum(self.emissions_history[2:4]) if len(self.emissions_history) >= 4 else 0
        policy_phase2_emissions = np.sum(self.emissions_history[4:]) if len(self.emissions_history) >= 6 else 0
        
        # Calculate market efficiency metrics
        market_participants = len(self.buyers) + len(self.sellers)
        participation_rate = market_participants / len(self.system_state) if self.system_state else 0
        price_volatility = np.std(self.price_history) / np.mean(self.price_history) if self.price_history else 0
        
        # Calculate emissions per credit
        total_credits_traded = sum(self.volume_history)
        total_emissions_saved = sum(self.emissions_history)
        emissions_per_credit = total_emissions_saved / total_credits_traded if total_credits_traded > 0 else 0
        
        evaluation_results = {
            "market_metrics": {
                "initial_price": initial_price,
                "final_price": final_price,
                "price_change_pct": price_change_pct,
                "price_volatility": price_volatility,
                "market_volume_growth_pct": volume_growth_pct
            },
            "trading_activity": {
                "total_volume": sum(self.volume_history),
                "unique_buyers": len(self.buyers),
                "unique_sellers": len(self.sellers),
                "market_participation_rate": participation_rate
            },
            "emissions_impact": {
                "total_emissions_saved_kg": total_emissions_saved,
                "pre_policy_emissions": pre_policy_emissions,
                "policy_phase1_emissions": policy_phase1_emissions,
                "policy_phase2_emissions": policy_phase2_emissions,
                "emissions_reduction_trend": "accelerating" if policy_phase2_emissions > policy_phase1_emissions > pre_policy_emissions else "stable" if abs(policy_phase2_emissions - policy_phase1_emissions) < 100 else "diminishing"
            },
            "market_efficiency": {
                "emissions_per_credit_kg": emissions_per_credit,
                "price_signal_effectiveness": "high" if price_change_pct > 20 and total_emissions_saved > 10000 else "medium" if price_change_pct > 10 and total_emissions_saved > 5000 else "low",
                "supply_demand_balance": "balanced" if abs(self.current_demand - self.current_supply) / self.current_supply < 0.2 else "imbalanced"
            },
            "policy_effectiveness": {
                "created_functional_market": market_participants > 10 and total_credits_traded > 1000,
                "reduced_emissions": total_emissions_saved > 10000,
                "price_discovery": price_volatility < 0.3,
                "overall_success": market_participants > 10 and total_emissions_saved > 10000 and price_volatility < 0.3
            },
            "time_series": {
                "price_history": self.price_history,
                "volume_history": self.volume_history,
                "emissions_history": self.emissions_history,
                "supply_history": self.supply_history,
                "demand_history": self.demand_history
            }
        }
        
        self.logger.info(f"evaluation_results={evaluation_results}")

        
        return evaluation_results

    def get_state_for_persistence(self) -> Dict[str, Any]:
        """Return current state for database storage"""
        return {
            "current_price": self.current_price,
            "monthly_volume": self.monthly_volume,
            "current_supply": self.current_supply,
            "current_demand": self.current_demand,
            "market_balance": self.current_demand / self.current_supply if self.current_supply > 0 else 0,
            "trading_enabled": self.credit_trading_enabled,
            "current_epoch": self.current_time.get_current_epoch() if self.current_time else 0
        } 