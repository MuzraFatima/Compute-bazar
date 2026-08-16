"""
=============================================================
 COMPUTE BAZAAR — API Endpoints (Weeks 3-8)
=============================================================
 Implements REST API routes:
   - Agent registration and profile retrieval
   - Combinatorial bid submissions
   - Available resources and reserve pricing
   - Real-time auction status
   - Simulation state management (start, stop, step)
   - Performance comparison analytics (vs. Round-Robin)
=============================================================
"""

import json
import secrets
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.db.models import Agent, Resource, Auction, Bid, Allocation
from backend.app.models.schemas import (
    AgentRegister, AgentResponse, BidSubmit, BidResponse,
    ResourceResponse, AuctionRoundResponse, ComparisonAnalytics, AllocationResponse
)
from backend.app.api.auth import get_db, get_current_agent, create_access_token
from backend.app.auction.simulation import start_simulation_loop, stop_simulation_loop, is_simulation_running

router = APIRouter()

# ── Agent Endpoints ──────────────────────────────────────────

@router.post("/agents/register", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
def register_agent(payload: AgentRegister, db: Session = Depends(get_db)):
    """Register a new AI agent and allocate initial budget."""
    existing = db.query(Agent).filter(Agent.name == payload.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent with name '{payload.name}' already registered"
        )
    
    # Generate unique API key
    api_key = f"cb_{secrets.token_hex(16)}"
    
    agent = Agent(
        name=payload.name,
        description=payload.description,
        api_key=api_key,
        budget_total=1000.0,
        budget_remaining=1000.0,
        tier=payload.tier.lower() if payload.tier.lower() in ("gold", "silver", "bronze") else "bronze"
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


@router.post("/auth/token")
def login_for_access_token(api_key: str, db: Session = Depends(get_db)):
    """Exchange X-API-Key for a JWT access token."""
    agent = db.query(Agent).filter(Agent.api_key == api_key, Agent.is_active == True).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    token = create_access_token(data={"sub": agent.name})
    return {"access_token": token, "token_type": "bearer"}


@router.get("/agents/me", response_model=AgentResponse)
def get_agent_profile(current_agent: Agent = Depends(get_current_agent)):
    """Get the profile, budget, and active status of the authenticated agent."""
    return current_agent


# ── Bid Endpoints ────────────────────────────────────────────

@router.post("/bids/submit", response_model=BidResponse, status_code=status.HTTP_201_CREATED)
def submit_bid(payload: BidSubmit, current_agent: Agent = Depends(get_current_agent), db: Session = Depends(get_db)):
    """
    Submit a combinatorial bid.
    Checks resource existence and agent's remaining budget.
    """
    # 1. Fetch current open auction round
    active_auction = db.query(Auction).filter(Auction.status == "open").order_by(Auction.round_number.desc()).first()
    if not active_auction:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active auction round is open for bidding at this time."
        )

    # 2. Check if agent can afford the bid
    if payload.bid_amount > current_agent.budget_remaining:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient remaining budget. Bid: ${payload.bid_amount:.2f}, Remaining: ${current_agent.budget_remaining:.2f}"
        )

    # 3. Verify that all resources in the bundle exist and are available
    requested_resources = payload.resource_bundle
    resources_in_db = db.query(Resource).filter(
        Resource.name.in_(requested_resources),
        Resource.is_available == True
    ).all()
    
    if len(resources_in_db) != len(requested_resources):
        found_names = {r.name for r in resources_in_db}
        missing = [r for r in requested_resources if r not in found_names]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Some resources are offline or do not exist: {missing}"
        )

    # 4. Save the bid to the database
    bid = Bid(
        auction_id=active_auction.id,
        agent_id=current_agent.id,
        resource_bundle=json.dumps(payload.resource_bundle),
        bid_amount=payload.bid_amount,
        task_description=payload.task_description,
        priority=payload.priority.lower()
    )
    db.add(bid)
    
    # Update agent activity timestamp
    current_agent.last_active_at = datetime.utcnow().isoformat()
    db.commit()
    db.refresh(bid)

    # Build response model
    return BidResponse(
        id=bid.id,
        auction_id=bid.auction_id,
        agent_name=current_agent.name,
        resource_bundle=payload.resource_bundle,
        bid_amount=bid.bid_amount,
        task_description=bid.task_description,
        priority=bid.priority,
        submitted_at=bid.submitted_at,
        is_winner=bid.is_winner,
        payment_amount=bid.payment_amount
    )


# ── Resource Endpoints ───────────────────────────────────────

@router.get("/resources", response_model=List[ResourceResponse])
def get_resources(db: Session = Depends(get_db)):
    """List all available compute resources and their specifications."""
    return db.query(Resource).all()


# ── Auction Endpoints ────────────────────────────────────────

@router.get("/auctions/current", response_model=AuctionRoundResponse)
def get_current_auction(db: Session = Depends(get_db)):
    """Retrieve detailed state of the current active or most recent auction round."""
    # Find latest round
    auction = db.query(Auction).order_by(Auction.round_number.desc()).first()
    if not auction:
        # Create initial round if DB is empty
        auction = Auction(round_number=1, status="open")
        db.add(auction)
        db.commit()
        db.refresh(auction)

    # Map bids
    bids = []
    for b in auction.bids:
        bids.append(BidResponse(
            id=b.id,
            auction_id=b.auction_id,
            agent_name=b.agent.name,
            resource_bundle=json.loads(b.resource_bundle),
            bid_amount=b.bid_amount,
            task_description=b.task_description or "",
            priority=b.priority,
            submitted_at=b.submitted_at,
            is_winner=b.is_winner,
            payment_amount=b.payment_amount
        ))

    # Map allocations
    allocations = []
    for a in auction.allocations:
        allocations.append(AllocationResponse(
            id=a.id,
            agent_name=a.agent.name,
            resources_allocated=json.loads(a.resources_allocated),
            bid_amount=a.bid_amount,
            payment_amount=a.payment_amount,
            value_to_system=a.value_to_system,
            allocated_at=a.allocated_at
        ))

    return AuctionRoundResponse(
        id=auction.id,
        round_number=auction.round_number,
        status=auction.status,
        started_at=auction.started_at,
        closed_at=auction.closed_at,
        total_value_extracted=auction.total_value_extracted,
        total_agents_participated=auction.total_agents_participated,
        solver_method=auction.solver_method,
        bids=bids,
        allocations=allocations
    )


# ── Simulation Controls ──────────────────────────────────────

@router.post("/simulation/start")
def start_simulation(db: Session = Depends(get_db)):
    """Start background simulation spawning agents with various bidding strategies."""
    if is_simulation_running():
        return {"status": "already_running"}
    start_simulation_loop()
    return {"status": "started"}


@router.post("/simulation/stop")
def stop_simulation():
    """Stop the background simulation loop."""
    stop_simulation_loop()
    return {"status": "stopped"}


@router.get("/simulation/status")
def get_simulation_status(db: Session = Depends(get_db)):
    """Get status of current simulation and agent details."""
    active_agents = db.query(Agent).filter(Agent.is_active == True).all()
    return {
        "is_running": is_simulation_running(),
        "agents": [
            {
                "name": a.name,
                "tier": a.tier,
                "budget_total": a.budget_total,
                "budget_remaining": a.budget_remaining,
                "last_active": a.last_active_at
            }
            for a in active_agents
        ]
    }


# ── Analytics & Comparison ──────────────────────────────────

@router.get("/analytics/comparison", response_model=ComparisonAnalytics)
def get_comparison_analytics(db: Session = Depends(get_db)):
    """
    Computes cumulative efficiency gains:
      - Compute Bazaar value (sum of all closed auctions total value extracted)
      - Round-Robin value (simulated counterfactual for each round)
    """
    closed_auctions = db.query(Auction).filter(Auction.status == "closed").all()
    
    total_rounds = len(closed_auctions)
    bazaar_val = sum(a.total_value_extracted for a in closed_auctions)
    
    # Simulating standard Round-Robin counterfactual:
    # First bid in each round wins its resources until resources are exhausted
    rr_val = 0.0
    for a in closed_auctions:
        allocated = set()
        for bid in sorted(a.bids, key=lambda b: b.submitted_at):  # arrival order
            bundle = set(json.loads(bid.resource_bundle))
            if bundle.isdisjoint(allocated):
                allocated.update(bundle)
                rr_val += bid.bid_amount

    savings = sum(
        sum(b.bid_amount - (b.payment_amount or 0.0) for b in a.bids if b.is_winner)
        for a in closed_auctions
    )

    gain = ((bazaar_val - rr_val) / rr_val * 100) if rr_val > 0 else 0.0

    return ComparisonAnalytics(
        total_rounds=total_rounds,
        bazaar_total_value=bazaar_val,
        round_robin_total_value=rr_val,
        cumulative_gain_pct=gain,
        savings_total=savings
    )
