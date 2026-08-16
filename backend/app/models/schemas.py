"""
=============================================================
 COMPUTE BAZAAR — Pydantic Schemas (Weeks 3-8)
=============================================================
 Defines request/response models for validation in endpoints.
=============================================================
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# ── Agent Schemas ────────────────────────────────────────────
class AgentRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, description="Unique identifier name of the agent")
    description: Optional[str] = Field(None, description="What this agent does")
    tier: str = Field("bronze", description="Subscription tier: gold, silver, or bronze")

class AgentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    tier: str
    budget_total: float
    budget_remaining: float
    api_key: str
    created_at: str
    last_active_at: Optional[str]

    class Config:
        from_attributes = True

# ── Bid Schemas ──────────────────────────────────────────────
class BidSubmit(BaseModel):
    resource_bundle: List[str] = Field(..., min_items=1, description="List of resource names required")
    bid_amount: float = Field(..., gt=0.0, description="Valuation amount for the resource bundle")
    task_description: str = Field("", description="Short description of the task")
    priority: str = Field("normal", description="Priority level: critical, high, normal, low")

class BidResponse(BaseModel):
    id: int
    auction_id: int
    agent_name: str
    resource_bundle: List[str]
    bid_amount: float
    task_description: str
    priority: str
    submitted_at: str
    is_winner: bool
    payment_amount: Optional[float]

# ── Allocation Schemas ───────────────────────────────────────
class AllocationResponse(BaseModel):
    id: int
    agent_name: str
    resources_allocated: List[str]
    bid_amount: float
    payment_amount: float
    value_to_system: float
    allocated_at: str

# ── Resource Schemas ─────────────────────────────────────────
class ResourceResponse(BaseModel):
    id: int
    name: str
    resource_type: str
    capacity: float
    unit: str
    cost_per_unit: float
    is_available: bool

    class Config:
        from_attributes = True

# ── Auction Schemas ──────────────────────────────────────────
class AuctionRoundResponse(BaseModel):
    id: int
    round_number: int
    status: str
    started_at: str
    closed_at: Optional[str]
    total_value_extracted: float
    total_agents_participated: int
    solver_method: str
    bids: List[BidResponse] = []
    allocations: List[AllocationResponse] = []

    class Config:
        from_attributes = True

# ── Analytics & Comparison ──────────────────────────────────
class ComparisonAnalytics(BaseModel):
    total_rounds: int
    bazaar_total_value: float
    round_robin_total_value: float
    cumulative_gain_pct: float
    savings_total: float
