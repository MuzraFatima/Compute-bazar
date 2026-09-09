"""
=============================================================
 COMPUTE BAZAAR 2.0 -- ResourceProvider & MarketResource (Week 5)
=============================================================
 Defines compute resource providers and market resources:
   - ResourceProvider: cloud/infrastructure suppliers
   - MarketResource: individual compute/API resource units
     (GPU, CPU, RAM, LLM/API, Storage, TPU)
   - Dynamic capacity, minimum floor pricing, and state tracking
=============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MarketResourceType(str, Enum):
    """Supported resource categories in Compute Bazaar 2.0."""
    GPU = "gpu"
    CPU = "cpu"
    RAM = "ram"
    API_BUDGET = "api_budget"
    LLM_API = "llm_api"
    STORAGE = "storage"
    TPU = "tpu"


@dataclass
class MarketResource:
    """
    Represents an individual compute resource provided to the market.

    Attributes:
        name: Unique resource identifier (e.g. 'GPU-Slot-1', 'Groq-Llama-API')
        resource_type: Category ('gpu', 'cpu', 'ram', 'llm_api', etc.)
        provider_id: ID of the supplying ResourceProvider
        total_capacity: Maximum resource capacity (e.g., slots, cores, GB)
        available_capacity: Currently unallocated capacity
        min_price: Minimum floor/reserve price set by provider
        current_price: Dynamic price calculated based on market scarcity
        unit: Unit of measurement ('slot', 'cores', 'gb', 'tokens', 'dollars')
        is_available: Operational availability flag (False if offline/maintenance)
    """
    name: str
    resource_type: str = MarketResourceType.GPU.value
    provider_id: str = "default_provider"
    total_capacity: float = 1.0
    available_capacity: float = 1.0
    min_price: float = 10.0
    current_price: float = 10.0
    unit: str = "slot"
    is_available: bool = True

    def __post_init__(self):
        if self.current_price < self.min_price:
            self.current_price = self.min_price

    def allocate(self, amount: float = 1.0) -> bool:
        """
        Attempt to allocate capacity from this resource.
        Returns True if successful, False if insufficient available capacity.
        """
        if not self.is_available or self.available_capacity < amount:
            return False
        self.available_capacity -= amount
        return True

    def release(self, amount: float = 1.0):
        """Release allocated capacity back to available pool."""
        self.available_capacity = min(self.total_capacity, self.available_capacity + amount)

    def update_dynamic_price(self, scarcity_factor: float, alpha: float = 0.5):
        """
        Update dynamic price based on scarcity factor:
          current_price = min_price * (1 + alpha * scarcity_factor)
        """
        factor = max(0.0, float(scarcity_factor))
        self.current_price = round(self.min_price * (1.0 + alpha * factor), 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "resource_type": self.resource_type,
            "provider_id": self.provider_id,
            "total_capacity": self.total_capacity,
            "available_capacity": self.available_capacity,
            "min_price": self.min_price,
            "current_price": self.current_price,
            "unit": self.unit,
            "is_available": self.is_available,
        }


@dataclass
class ResourceProvider:
    """
    Represents an infrastructure provider listing resources in Compute Bazaar.

    Examples:
      - Groq Infrastructure (LLM API tokens & throughput)
      - Lambda Labs (H100 GPU clusters)
      - AWS US-East (EC2 CPU & RAM blocks)
      - Local Enterprise Server Farm
    """
    provider_id: str
    name: str
    provider_type: str = "cloud"  # "cloud", "api_vendor", "on_prem", "decentralized"
    reliability_score: float = 1.0  # 0.0 to 1.0
    is_active: bool = True
    resources: Dict[str, MarketResource] = field(default_factory=dict)

    def add_resource(self, resource: MarketResource):
        """Register a resource under this provider."""
        resource.provider_id = self.provider_id
        self.resources[resource.name] = resource

    def update_resource_capacity(self, resource_name: str, total_capacity: float):
        """Update total capacity for a resource, adjusting available capacity proportionally."""
        if resource_name in self.resources:
            res = self.resources[resource_name]
            diff = total_capacity - res.total_capacity
            res.total_capacity = total_capacity
            res.available_capacity = max(0.0, res.available_capacity + diff)

    def set_resource_availability(self, resource_name: str, is_available: bool):
        """Toggle dynamic operational availability of a resource (e.g. maintenance)."""
        if resource_name in self.resources:
            self.resources[resource_name].is_available = is_available

    def get_total_capacity(self) -> float:
        """Return aggregate total capacity across all provider resources."""
        return sum(r.total_capacity for r in self.resources.values())

    def get_available_resources(self) -> List[MarketResource]:
        """Return list of active, available resources."""
        if not self.is_active:
            return []
        return [r for r in self.resources.values() if r.is_available]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "provider_type": self.provider_type,
            "reliability_score": self.reliability_score,
            "is_active": self.is_active,
            "resource_count": len(self.resources),
            "resources": [r.to_dict() for r in self.resources.values()],
        }
