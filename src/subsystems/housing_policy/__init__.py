"""
Housing Policy Simulation Subsystems

This package contains subsystems for simulating housing purchase restriction policies.
"""

from src.utils.logger import get_logger

logger = get_logger("housing_policy")

def integrate_subsystems(subsystems, system_integrations):
    """
    Integrate subsystems based on configuration.
    
    Args:
        subsystems (dict): Dictionary of instantiated subsystems
        system_integrations (list): List of integration configurations
    """
    if not system_integrations:
        logger.info("No system integrations defined")
        return
    
    for integration in system_integrations:
        source_name = integration.get("source_system")
        target_name = integration.get("target_system")
        method_name = integration.get("integration_method")
        description = integration.get("description", "")
        
        if not all([source_name, target_name, method_name]):
            logger.warning(f"Incomplete integration definition: {integration}")
            continue
        
        if source_name not in subsystems or target_name not in subsystems:
            logger.warning(f"Cannot integrate: missing system {source_name} or {target_name}")
            continue
        
        source_system = subsystems[source_name]
        target_system = subsystems[target_name]
        
        try:
            if hasattr(target_system, method_name) and callable(getattr(target_system, method_name)):
                getattr(target_system, method_name)(source_system)
                logger.info(f"Integrated {source_name} with {target_name} using {method_name}: {description}")
            else:
                logger.warning(f"Integration method {method_name} not found in {target_name}")
        except Exception as e:
            logger.error(f"Error integrating {source_name} with {target_name}: {str(e)}") 