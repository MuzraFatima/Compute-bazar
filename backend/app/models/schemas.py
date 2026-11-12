"""
=============================================================
 COMPUTE BAZAAR — Pydantic Schemas (Weeks 3-8)
=============================================================
 Defines request/response models for validation in endpoints.
=============================================================
"""

from pydantic import BaseModel, ConfigDict, Field


# ── Agent Schemas ────────────────────────────────────────────
class AgentRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, description="Unique identifier name of the agent")
    description: str | None = Field(None, description="What this agent does")
    tier: str = Field("bronze", description="Subscription tier: gold, silver, or bronze")


class AgentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    tier: str
    budget_total: float
    budget_remaining: float
    api_key: str
    created_at: str
    last_active_at: str | None


# ── Bid Schemas ──────────────────────────────────────────────
class BidSubmit(BaseModel):
    resource_bundle: list[str] = Field(..., min_length=1, description="List of resource names required")
    bid_amount: float = Field(..., gt=0.0, description="Valuation amount for the resource bundle")
    task_description: str = Field("", description="Short description of the task")
    priority: str = Field("normal", description="Priority level: critical, high, normal, low")


class BidResponse(BaseModel):
    id: int
    auction_id: int
    agent_name: str
    resource_bundle: list[str]
    bid_amount: float
    task_description: str
    priority: str
    submitted_at: str
    is_winner: bool
    payment_amount: float | None


# ── Allocation Schemas ───────────────────────────────────────
class AllocationResponse(BaseModel):
    id: int
    agent_name: str
    resources_allocated: list[str]
    bid_amount: float
    payment_amount: float
    value_to_system: float
    allocated_at: str


# ── Resource Schemas ─────────────────────────────────────────
class ResourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    resource_type: str
    capacity: float
    unit: str
    cost_per_unit: float
    is_available: bool


# ── Auction Schemas ──────────────────────────────────────────
class AuctionRoundResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    round_number: int
    status: str
    started_at: str
    closed_at: str | None
    total_value_extracted: float
    total_agents_participated: int
    solver_method: str
    bids: list[BidResponse] = []
    allocations: list[AllocationResponse] = []


# ── Analytics & Comparison ──────────────────────────────────
class ComparisonAnalytics(BaseModel):
    total_rounds: int
    bazaar_total_value: float
    round_robin_total_value: float
    cumulative_gain_pct: float
    savings_total: float
