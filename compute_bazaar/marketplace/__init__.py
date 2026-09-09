"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Dynamic Marketplace Module (Week 5)
=============================================================
 Extends Compute Bazaar with resource providers, multi-resource
 supply tracking (GPU, CPU, RAM, LLM/API), dynamic availability,
 and scarcity-based dynamic pricing.
=============================================================
"""

from compute_bazaar.marketplace.provider import (
    ResourceProvider,
    MarketResource,
    MarketResourceType,
)
from compute_bazaar.marketplace.manager import DynamicMarketplace

__all__ = [
    "ResourceProvider",
    "MarketResource",
    "MarketResourceType",
    "DynamicMarketplace",
]
