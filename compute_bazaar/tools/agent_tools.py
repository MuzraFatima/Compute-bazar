"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Agent Safe Toolset (Week 7)
=============================================================
 Provides safe tools for autonomous AI agents:
   - check_resources(): Inspects market inventory & pricing
   - check_budget(): Inspects agent budget and surplus
   - estimate_cost(): Forecasts compute resources and cost via ML predictor
   - execute_task(): Safely runs tasks and logs performance metrics
=============================================================
"""

from __future__ import annotations

import time
import random
from typing import Any, Dict, List, Optional, Callable
from loguru import logger

from compute_bazaar.prediction.predictor import ResourcePredictor, ResourcePrediction


def check_resources(marketplace=None) -> Dict[str, Any]:
    """
    Safe Tool: Check current resource availability, capacity, and dynamic pricing.

    Args:
        marketplace: Optional DynamicMarketplace instance.
                     If None, returns standard default inventory snapshot.
    """
    if marketplace is not None:
        return marketplace.get_market_summary()

    # Default baseline market inventory
    return {
        "available_resources": [
            {"name": "GPU-Slot-1", "type": "gpu", "available_capacity": 1.0, "current_price": 30.0},
            {"name": "GPU-Slot-2", "type": "gpu", "available_capacity": 1.0, "current_price": 30.0},
            {"name": "API-Budget-100", "type": "api_budget", "available_capacity": 100.0, "current_price": 20.0},
            {"name": "RAM-Block-32GB", "type": "ram", "available_capacity": 32.0, "current_price": 15.0},
            {"name": "CPU-Cores-8", "type": "cpu", "available_capacity": 8.0, "current_price": 10.0},
        ],
        "status": "operational",
    }


def check_budget(agent) -> Dict[str, Any]:
    """
    Safe Tool: Inspect an agent's current financial status and memory history.

    Args:
        agent: BaseAgent instance
    """
    memory_summary = agent.memory.get_summary() if hasattr(agent, "memory") else {}
    return {
        "agent_name": getattr(agent, "name", "unknown"),
        "current_budget": getattr(agent, "budget", 0.0),
        "total_spent": memory_summary.get("total_spent", 0.0),
        "total_surplus": memory_summary.get("total_surplus", 0.0),
        "total_wins": memory_summary.get("total_wins", 0),
        "total_losses": memory_summary.get("total_losses", 0),
        "is_starved": memory_summary.get("is_starved", False),
    }


def estimate_cost(
    task_description: str,
    predictor: Optional[ResourcePredictor] = None,
) -> ResourcePrediction:
    """
    Safe Tool: Predict compute requirements, execution time, and cost for a task.

    Args:
        task_description: Natural language task request
        predictor: Optional ResourcePredictor instance
    """
    p = predictor or ResourcePredictor()
    return p.predict(task_description)


def execute_task(
    agent,
    task_description: str,
    allocation_result: Optional[Dict[str, Any]] = None,
    predictor: Optional[ResourcePredictor] = None,
) -> Dict[str, Any]:
    """
    Safe Tool: Safely execute a task using awarded compute resources.

    Simulates runtime execution, records performance metrics into
    ResourcePredictor and AgentMemory, and updates agent budget.

    Args:
        agent: BaseAgent executing the task
        task_description: Task prompt
        allocation_result: Optional dict containing awarded resources and payment
        predictor: Optional ResourcePredictor instance to record performance
    """
    pred = (predictor or ResourcePredictor()).predict(task_description)
    
    # Check if agent won resources in auction or has budget
    won = True
    payment = pred.predicted_cost
    allocated = pred.recommended_bundle

    if allocation_result:
        won = allocation_result.get("won", True)
        payment = allocation_result.get("payment", pred.predicted_cost)
        allocated = allocation_result.get("resources", pred.recommended_bundle)

    if not won:
        logger.warning(f"[{agent.name}] Cannot execute task -- auction allocation lost.")
        return {
            "status": "failed",
            "reason": "Auction allocation lost",
            "task_description": task_description,
            "executed": False,
        }

    # Simulate task execution time
    sim_time_sec = round(pred.predicted_execution_time_sec * random.uniform(0.9, 1.1), 2)
    actual_cost = round(payment, 2)

    # Record execution into predictor for ML feedback loop
    if predictor:
        predictor.record_execution(
            task_description=task_description,
            allocated_resources=allocated,
            execution_time_sec=sim_time_sec,
            actual_cost=actual_cost,
            tokens_used=pred.predicted_api_tokens,
        )

    logger.info(f"[{agent.name}] Task executed successfully | duration={sim_time_sec}s | cost=${actual_cost:.2f}")

    return {
        "status": "completed",
        "agent_name": agent.name,
        "task_description": task_description,
        "resources_used": allocated,
        "execution_time_sec": sim_time_sec,
        "actual_cost": actual_cost,
        "executed": True,
    }


class ToolRegistry:
    """
    Registry enabling AI agents to discover, inspect, and invoke safe tools.
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {
            "check_resources": check_resources,
            "check_budget": check_budget,
            "estimate_cost": estimate_cost,
            "execute_task": execute_task,
        }

    def register_tool(self, name: str, func: Callable):
        """Register a custom tool."""
        self._tools[name] = func

    def get_tool(self, name: str) -> Optional[Callable]:
        """Fetch tool function by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List names of all registered tools."""
        return list(self._tools.keys())

    def invoke(self, name: str, **kwargs) -> Any:
        """Invoke a tool by name with arguments."""
        func = self.get_tool(name)
        if not func:
            raise ValueError(f"Unknown tool '{name}'. Available tools: {self.list_tools()}")
        return func(**kwargs)
