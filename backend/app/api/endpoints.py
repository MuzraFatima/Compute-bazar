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
from datetime import datetime, timezone
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
from backend.app.auction.agent_simulation import (
    start_autonomous_simulation, stop_autonomous_simulation,
    is_autonomous_simulation_running, get_llm_telemetry,
)

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
    current_agent.last_active_at = datetime.now(timezone.utc).isoformat()
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
async def start_simulation(mode: str = "classic", db: Session = Depends(get_db)):
    """
    Start background simulation.
    - mode=classic   → rule-based bot agents (fast, no LLM)
    - mode=autonomous → Groq-powered AI agents (real LLM reasoning)
    """
    if mode == "autonomous":
        if is_autonomous_simulation_running():
            return {"status": "already_running", "mode": "autonomous"}
        # Stop classic if running
        if is_simulation_running():
            stop_simulation_loop()
        start_autonomous_simulation()
        return {"status": "started", "mode": "autonomous"}
    else:
        if is_simulation_running():
            return {"status": "already_running", "mode": "classic"}
        # Stop autonomous if running
        if is_autonomous_simulation_running():
            stop_autonomous_simulation()
        start_simulation_loop()
        return {"status": "started", "mode": "classic"}


@router.post("/simulation/stop")
async def stop_simulation():
    """Stop whichever simulation is currently running."""
    stop_simulation_loop()
    stop_autonomous_simulation()
    return {"status": "stopped"}


@router.get("/simulation/status")
def get_simulation_status(db: Session = Depends(get_db)):
    """Get status of current simulation, mode, and agent details."""
    active_agents = db.query(Agent).filter(Agent.is_active == True).all()
    autonomous_running = is_autonomous_simulation_running()
    classic_running = is_simulation_running()
    return {
        "is_running": autonomous_running or classic_running,
        "mode": "autonomous" if autonomous_running else ("classic" if classic_running else "stopped"),
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


# ── Dashboard 2.0 Real Database Metrics Endpoint ────────────

from compute_bazaar.fairness import jains_fairness_index
from compute_bazaar.llm import GroqLLMClient
from backend.app.db.models import Base

@router.get("/dashboard/metrics")
def get_dashboard_metrics(db: Session = Depends(get_db)):
    """
    Returns real, live system metrics computed directly from database tables
    and active backend components for the Compute Bazaar 2.0 Dashboard.
    """
    # Ensure database tables exist on active connection dialect
    try:
        Base.metadata.create_all(bind=db.get_bind())
    except Exception:
        pass

    # 1. Agents Data & Performance
    agents = db.query(Agent).all()
    agent_metrics = []
    agent_surpluses = []
    
    for a in agents:
        bids = db.query(Bid).filter(Bid.agent_id == a.id).all()
        total_bids = len(bids)
        winning_bids = [b for b in bids if b.is_winner]
        losing_bids = [b for b in bids if not b.is_winner]
        wins = len(winning_bids)
        losses = len(losing_bids)
        win_rate = round((wins / total_bids * 100.0), 1) if total_bids > 0 else 0.0
        
        spent = sum(b.payment_amount or 0.0 for b in winning_bids)
        surplus = sum((b.bid_amount - (b.payment_amount or 0.0)) for b in winning_bids)
        agent_surpluses.append(surplus)
        
        agent_metrics.append({
            "id": a.id,
            "name": a.name,
            "tier": a.tier,
            "is_active": a.is_active,
            "budget_total": a.budget_total,
            "budget_remaining": a.budget_remaining,
            "total_spent": round(spent, 2),
            "total_surplus": round(surplus, 2),
            "total_bids": total_bids,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
        })
        
    jains_index = jains_fairness_index(agent_surpluses) if agent_surpluses else 1.0

    # 2. Resources & Utilization
    resources = db.query(Resource).all()
    resource_metrics = []
    category_utilization = {"gpu": 0.0, "cpu": 0.0, "ram": 0.0, "api_budget": 0.0}
    
    recent_allocations = db.query(Allocation).order_by(Allocation.id.desc()).limit(20).all()
    assigned_counts: dict[str, float] = {}
    for alloc in recent_allocations:
        try:
            allocated_list = json.loads(alloc.resources_allocated)
            for res_name in allocated_list:
                assigned_counts[res_name] = assigned_counts.get(res_name, 0.0) + 1.0
        except Exception:
            pass

    for r in resources:
        used = min(r.capacity, assigned_counts.get(r.name, 0.0))
        util = round((used / r.capacity * 100.0), 1) if r.capacity > 0 else 0.0
        resource_metrics.append({
            "id": r.id,
            "name": r.name,
            "resource_type": r.resource_type,
            "capacity": r.capacity,
            "used": used,
            "unit": r.unit,
            "cost_per_unit": r.cost_per_unit,
            "is_available": r.is_available,
            "utilization_pct": util,
        })
        if r.resource_type in category_utilization:
            category_utilization[r.resource_type] = max(category_utilization[r.resource_type], util)

    # 3. Financials & Welfare
    total_bids_count = db.query(Bid).count()
    winning_bids_count = db.query(Bid).filter(Bid.is_winner == True).count()
    losing_bids_count = db.query(Bid).filter(Bid.is_winner == False).count()
    
    all_winning_bids = db.query(Bid).filter(Bid.is_winner == True).all()
    social_welfare = sum(b.bid_amount for b in all_winning_bids)
    total_spending = sum(b.payment_amount or 0.0 for b in all_winning_bids)
    estimated_savings = max(0.0, social_welfare - total_spending)

    # 4. Auction History
    all_auctions = db.query(Auction).order_by(Auction.round_number.desc()).limit(15).all()
    auction_history = []
    for a in all_auctions:
        bid_count = len(a.bids)
        winners_count = sum(1 for b in a.bids if b.is_winner)
        auction_history.append({
            "id": a.id,
            "round_number": a.round_number,
            "status": a.status,
            "started_at": a.started_at,
            "closed_at": a.closed_at,
            "total_value_extracted": round(a.total_value_extracted, 2),
            "solver_method": a.solver_method,
            "bids_count": bid_count,
            "winners_count": winners_count,
        })

    # 5. LLM / Groq Telemetry — use real accumulated stats from autonomous simulation
    try:
        llm_telemetry = get_llm_telemetry()
    except Exception:
        llm_telemetry = {"calls": 0, "tokens": 0, "avg_latency_ms": 0.0, "is_mock": True, "model": "openai/gpt-oss-120b"}

    return {
        "summary": {
            "active_agents_count": len([a for a in agents if a.is_active]),
            "total_resources_count": len(resources),
            "total_bids_count": total_bids_count,
            "winning_bids_count": winning_bids_count,
            "losing_bids_count": losing_bids_count,
            "social_welfare": round(social_welfare, 2),
            "total_spending": round(total_spending, 2),
            "estimated_savings": round(estimated_savings, 2),
            "jains_fairness_index": round(jains_index, 4),
            "total_auction_rounds": len(all_auctions),
        },
        "agents": agent_metrics,
        "resources": resource_metrics,
        "utilization_by_category": category_utilization,
        "auction_history": auction_history,
        "llm_telemetry": llm_telemetry,
    }

