"""
=============================================================
 COMPUTE BAZAAR — Combinatorial Auction for AI Compute Resources
=============================================================
 Open-source framework and Python SDK for multi-agent
 combinatorial compute resource allocation via WDP and VCG pricing.
=============================================================
"""

__version__ = "0.1.0"

from compute_bazaar.models.bundle import (
    ResourceItem,
    ResourceType,
    CombinatorialBid,
    AllocationResult,
    AuctionRoundOutcome,
    PriorityLevel,
    AgentTier,
)
from compute_bazaar.generators.workload import (
    generate_synthetic_bid,
    generate_workload_batch,
    TASK_TEMPLATES,
)
from compute_bazaar.autobid.strategies import (
    BiddingStrategy,
    TruthfulStrategy,
    BudgetAwareShadingStrategy,
    UrgentPriorityStrategy,
    AdaptiveQLearner,
    ContextualBanditStrategy,
)
from compute_bazaar.client.sdk import (
    ComputeBazaarClient,
    ComputeBazaarError,
)
from compute_bazaar.llm import GroqLLMClient, LLMConfig, LLMResponse, LLMUsage, LLMError
from compute_bazaar.agents import (
    BaseAgent,
    TaskDecision,
    AgentCapability,
    to_engine_bid,
    AgentMemory,
    BidOutcome,
    CodingAgent,
    ResearchAgent,
    DataAgent,
    AnalysisAgent,
)
from compute_bazaar.marketplace import (
    ResourceProvider,
    MarketResource,
    MarketResourceType,
    DynamicMarketplace,
)
from compute_bazaar.fairness import (
    jains_fairness_index,
    calculate_starvation_score,
    FairnessTracker,
)
from compute_bazaar.prediction import (
    ResourcePrediction,
    ResourcePredictor,
)
from compute_bazaar.tools import (
    check_resources,
    check_budget,
    estimate_cost,
    execute_task,
    ToolRegistry,
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


