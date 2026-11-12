"""
=============================================================
 COMPUTE BAZAAR — Combinatorial Auction for AI Compute Resources
=============================================================
 Open-source framework and Python SDK for multi-agent
 combinatorial compute resource allocation via WDP and VCG pricing.
=============================================================
"""

__version__ = "0.1.0"

from compute_bazaar.agents import (
    AgentCapability,
    AgentMemory,
    AnalysisAgent,
    BaseAgent,
    BidOutcome,
    CodingAgent,
    DataAgent,
    ResearchAgent,
    TaskDecision,
    to_engine_bid,
)
from compute_bazaar.autobid.strategies import (
    AdaptiveQLearner,
    BiddingStrategy,
    BudgetAwareShadingStrategy,
    ContextualBanditStrategy,
    TruthfulStrategy,
    UrgentPriorityStrategy,
)
from compute_bazaar.client.sdk import (
    ComputeBazaarClient,
    ComputeBazaarError,
)
from compute_bazaar.fairness import (
    FairnessTracker,
    calculate_starvation_score,
    jains_fairness_index,
)
from compute_bazaar.generators.workload import (
    TASK_TEMPLATES,
    generate_synthetic_bid,
    generate_workload_batch,
)
from compute_bazaar.llm import GroqLLMClient, LLMConfig, LLMError, LLMResponse, LLMUsage
from compute_bazaar.marketplace import (
    DynamicMarketplace,
    MarketResource,
    MarketResourceType,
    ResourceProvider,
)
from compute_bazaar.models.bundle import (
    AgentTier,
    AllocationResult,
    AuctionRoundOutcome,
    CombinatorialBid,
    PriorityLevel,
    ResourceItem,
    ResourceType,
)
from compute_bazaar.prediction import (
    ResourcePrediction,
    ResourcePredictor,
)
from compute_bazaar.tools import (
    ToolRegistry,
    check_budget,
    check_resources,
    estimate_cost,
    execute_task,
)

__all__ = [
    "__version__",
    # Models
    "ResourceItem",
    "ResourceType",
    "CombinatorialBid",
    "AllocationResult",
    "AuctionRoundOutcome",
    "PriorityLevel",
    "AgentTier",
    # Generators
    "generate_synthetic_bid",
    "generate_workload_batch",
    "TASK_TEMPLATES",
    # ML Autobid Strategies
    "BiddingStrategy",
    "TruthfulStrategy",
    "BudgetAwareShadingStrategy",
    "UrgentPriorityStrategy",
    "AdaptiveQLearner",
    "ContextualBanditStrategy",
    # Client SDK
    "ComputeBazaarClient",
    "ComputeBazaarError",
    # LLM Module (2.0)
    "GroqLLMClient",
    "LLMConfig",
    "LLMResponse",
    "LLMUsage",
    "LLMError",
    # Autonomous AI Agents (2.0)
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
    # Dynamic Marketplace (2.0 Week 5)
    "ResourceProvider",
    "MarketResource",
    "MarketResourceType",
    "DynamicMarketplace",
    # Fairness & Memory (2.0 Week 6)
    "jains_fairness_index",
    "calculate_starvation_score",
    "FairnessTracker",
    # Prediction & Tools (2.0 Week 7)
    "ResourcePrediction",
    "ResourcePredictor",
    "check_resources",
    "check_budget",
    "estimate_cost",
    "execute_task",
    "ToolRegistry",
]
