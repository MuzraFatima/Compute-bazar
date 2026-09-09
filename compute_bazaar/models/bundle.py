"""
=============================================================
 COMPUTE BAZAAR — Core Data Models & Schemas
=============================================================
 Data classes and structures representing auction resources,
 combinatorial bundles, bids, and allocation decisions.
=============================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Set


class ResourceType(str, Enum):
    GPU = "gpu"
    API_BUDGET = "api_budget"
    RAM = "ram"
    CPU = "cpu"
    STORAGE = "storage"
    TPU = "tpu"


class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class AgentTier(str, Enum):
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"


@dataclass
class ResourceItem:
    """Represents a compute resource unit."""
    name: str
    resource_type: str = ResourceType.GPU.value
    capacity: float = 1.0
    unit: str = "slot"
    cost_per_unit: float = 0.0
    is_available: bool = True

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, ResourceItem):
            return self.name == other.name
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "resource_type": self.resource_type,
            "capacity": self.capacity,
            "unit": self.unit,
            "cost_per_unit": self.cost_per_unit,
            "is_available": self.is_available,
        }


@dataclass
class CombinatorialBid:
    """Represents a combinatorial bid submitted by an agent for a bundle."""
    agent_name: str
    resource_bundle: List[str]
    bid_amount: float
    bid_id: Optional[str] = None
    task_description: str = ""
    priority: str = PriorityLevel.NORMAL.value
    valuation: Optional[float] = None
    submitted_at: Optional[str] = None
    is_winner: bool = False
    payment_amount: Optional[float] = None

    def __post_init__(self):
        if self.valuation is None:
            self.valuation = self.bid_amount

    @property
    def bundle_size(self) -> int:
        return len(self.resource_bundle)

    @property
    def value_per_resource(self) -> float:
        if not self.resource_bundle:
            return 0.0
        return self.bid_amount / len(self.resource_bundle)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid_id": self.bid_id,
            "agent_name": self.agent_name,
            "resource_bundle": self.resource_bundle,
            "bid_amount": self.bid_amount,
            "task_description": self.task_description,
            "priority": self.priority,
            "valuation": self.valuation,
            "is_winner": self.is_winner,
            "payment_amount": self.payment_amount,
        }


@dataclass
class AllocationResult:
    """Represents an allocation awarded to a winning agent."""
    round_id: int
    agent_name: str
    bid_id: str
    resources_allocated: List[str]
    bid_amount: float
    payment_amount: float
    discount_applied: float = 0.0
    surplus: float = 0.0

    def __post_init__(self):
        self.surplus = self.bid_amount - self.payment_amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "agent_name": self.agent_name,
            "bid_id": self.bid_id,
            "resources_allocated": self.resources_allocated,
            "bid_amount": self.bid_amount,
            "payment_amount": self.payment_amount,
            "discount_applied": self.discount_applied,
            "surplus": self.surplus,
        }


@dataclass
class AuctionRoundOutcome:
    """Outcome and diagnostics for an entire combinatorial auction round."""
    round_number: int
    solver_method: str
    total_value_extracted: float
    winning_bids: List[CombinatorialBid] = field(default_factory=list)
    all_bids: List[CombinatorialBid] = field(default_factory=list)
    unallocated_resources: List[str] = field(default_factory=list)
    allocations: List[AllocationResult] = field(default_factory=list)
    solve_duration_ms: float = 0.0
    greedy_benchmark_value: float = 0.0
    efficiency_gain_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "solver_method": self.solver_method,
            "total_value_extracted": self.total_value_extracted,
            "winners_count": len(self.winning_bids),
            "bids_count": len(self.all_bids),
            "unallocated_resources": self.unallocated_resources,
            "solve_duration_ms": self.solve_duration_ms,
            "greedy_benchmark_value": self.greedy_benchmark_value,
            "efficiency_gain_pct": self.efficiency_gain_pct,
            "allocations": [a.to_dict() for a in self.allocations],
        }
