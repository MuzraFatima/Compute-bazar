"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Autonomous AI Agents Module
=============================================================
 Exports the agent classes and decision data structures.
=============================================================
"""

from compute_bazaar.agents.base import (
    BaseAgent,
    TaskDecision,
    AgentCapability,
    to_engine_bid,
)
from compute_bazaar.agents.memory import AgentMemory, BidOutcome
from compute_bazaar.agents.coding import CodingAgent
from compute_bazaar.agents.research import ResearchAgent
from compute_bazaar.agents.data import DataAgent
from compute_bazaar.agents.analysis import AnalysisAgent

__all__ = [
    "BaseAgent",
    "TaskDecision",
    "AgentCapability",
    "to_engine_bid",
    "AgentMemory",
    "BidOutcome",
    "CodingAgent",
    "ResearchAgent",
    "DataAgent",
    "AnalysisAgent",
]


