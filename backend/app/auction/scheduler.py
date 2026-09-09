"""
=============================================================
 COMPUTE BAZAAR — Auction Loop Scheduler (Weeks 3-8)
=============================================================
 Implements the background periodic loop that:
   - Closes open auctions every 10 seconds
   - Runs Winner Determination (WDP) LP solver
   - Computes VCG pricing
   - Deducts agent budgets and saves allocations to DB
   - Refills agent budgets (simulation continuous loop)
   - Broadcasts updates to WebSocket clients
   - Starts the next open round
=============================================================
"""

import json
import asyncio
from datetime import datetime, timezone
from loguru import logger
from sqlalchemy.orm import Session

from backend.app.db.models import SessionLocal, Auction, Bid, Allocation, Agent, Resource
from backend.app.auction.engine import AuctionEngine, Resource as EngineResource, Bid as EngineBid
from backend.app.auction.pricing import apply_tier_pricing
from backend.app.api.websocket import manager

# Interval between auction rounds (10 seconds for smooth simulation demonstration)
ROUND_INTERVAL_SECONDS = 10

# Global shutdown event
_stop_event = asyncio.Event()
_scheduler_task = None


async def run_auction_round(db: Session):
    """
    Executes one complete auction round:
      1. Collects all submitted bids for the current open round.
      2. If no bids, closes the round and opens a new one.
      3. If bids exist:
         - Converts DB models to Engine models.
         - Runs LP solver and determines optimal winners.
         - Computes VCG payments.
         - Enforces Gold/Silver/Bronze discounts.
         - Deducts agent budgets.
         - Saves allocations, updates bid states.
      4. Saves results, closes current round, creates a new one.
      5. Broadcasts live outcomes to all connected browsers.
    """
    # 1. Fetch current open auction round
    current_round = db.query(Auction).filter(Auction.status == "open").order_by(Auction.round_number.desc()).first()
    if not current_round:
        # Create one if missing
        current_round = Auction(round_number=1, status="open")
        db.add(current_round)
        db.commit()
        db.refresh(current_round)

    logger.info(f"Closing Round #{current_round.round_number} for solving...")

    # Close current round
    current_round.status = "closed"
    current_round.closed_at = datetime.now(timezone.utc).isoformat()
    db.commit()

    # 2. Get resources in system to pass to solver
    db_resources = db.query(Resource).filter(Resource.is_available == True).all()
    engine_resources = [
        EngineResource(name=r.name, resource_type=r.resource_type, capacity=r.capacity, unit=r.unit)
        for r in db_resources
    ]
    res_map = {r.name: r for r in engine_resources}

    # 3. Get all bids for this round
    db_bids = db.query(Bid).filter(Bid.auction_id == current_round.id).all()
    
    if not db_bids:
        logger.info(f"No bids submitted for Round #{current_round.round_number}. Closing cleanly.")
        current_round.total_value_extracted = 0.0
        current_round.total_agents_participated = 0
        db.commit()
        
        # Broadcast round close with no winners
        await manager.broadcast_json({
            "type": "round_closed",
            "round_number": current_round.round_number,
            "winners": [],
            "total_value": 0.0,
            "allocations": [],
            "unallocated": [r.name for r in db_resources]
        })
        
        # Start next round
        next_round_num = current_round.round_number + 1
        new_round = Auction(round_number=next_round_num, status="open")
        db.add(new_round)
        db.commit()
        
        await manager.broadcast_json({
            "type": "new_round_opened",
            "round_number": next_round_num
        })
        return

    # 4. Map DB Bids to Engine Bids for solving
    engine_bids = []
    agent_id_map = {}  # agent name -> ID
    agent_tier_map = {}  # agent_id -> tier
    
    for b in db_bids:
        agent = b.agent
        agent_id_map[agent.name] = agent.id
        agent_tier_map[agent.name] = agent.tier
        
        engine_bids.append(EngineBid(
            bid_id=str(b.id),
            agent_id=agent.name,
            resource_bundle=json.loads(b.resource_bundle),
            bid_amount=b.bid_amount,
            task_description=b.task_description or "",
            priority=b.priority
        ))

    # 5. Initialize engine and run auction
    engine = AuctionEngine(engine_resources)
    for eb in engine_bids:
        engine.submit_bid(eb)
        
    result = engine.run_auction(round_id=current_round.round_number)
    
    # 6. Apply tier discounts and payments
    priced_payments = apply_tier_pricing(result, agent_tier_map)
    priced_map = {p.bid_id: p for p in priced_payments}

    # 7. Apply allocations in DB
    winners_list = []
    total_val = 0.0
    for eb in result.winning_bids:
        # Match back to DB bid
        db_bid = db.query(Bid).filter(Bid.id == int(eb.bid_id)).first()
        if not db_bid:
            continue
            
        payment_info = priced_map.get(eb.bid_id)
        final_payment = payment_info.final_payment if payment_info else eb.bid_amount
        
        db_bid.is_winner = True
        db_bid.payment_amount = final_payment
        total_val += db_bid.bid_amount

        # Deduct budget from Agent
        agent = db_bid.agent
        agent.budget_remaining = max(0.0, agent.budget_remaining - final_payment)

        # Record Allocation
        allocation = Allocation(
            auction_id=current_round.id,
            agent_id=agent.id,
            bid_id=db_bid.id,
            resources_allocated=db_bid.resource_bundle,
            bid_amount=db_bid.bid_amount,
            payment_amount=final_payment,
            value_to_system=db_bid.bid_amount
        )
        db.add(allocation)
        
        winners_list.append({
            "agent": agent.name,
            "bid": db_bid.id,
            "bundle": json.loads(db_bid.resource_bundle),
            "amount": db_bid.bid_amount,
            "payment": final_payment,
            "savings": db_bid.bid_amount - final_payment
        })

    # Save round stats
    current_round.total_value_extracted = total_val
    current_round.total_agents_participated = len(set(b.agent_id for b in db_bids))
    db.commit()

    # Update ML simulation agent win histories
    try:
        from backend.app.auction.simulation import update_simulation_win_rates
        update_simulation_win_rates(db, current_round.id)
    except Exception as e:
        logger.error(f"Error updating win rates: {e}")

    # Refill budgets of active agents who are running low to keep the simulation interactive
    active_agents = db.query(Agent).filter(Agent.is_active == True).all()
    for a in active_agents:
        if a.budget_remaining < 150.0:
            a.budget_remaining = min(a.budget_total, a.budget_remaining + 300.0)
    db.commit()

    # 8. Broadcast round close outcomes to WebSocket
    await manager.broadcast_json({
        "type": "round_closed",
        "round_number": current_round.round_number,
        "winners": winners_list,
        "total_value": total_val,
        "unallocated": result.unallocated_resources,
        "greedy_value": result.greedy_value,
        "efficiency_gain_pct": result.efficiency_vs_greedy
    })

    # 9. Open and start the next round
    next_round_num = current_round.round_number + 1
    new_round = Auction(round_number=next_round_num, status="open")
    db.add(new_round)
    db.commit()
    
    await manager.broadcast_json({
        "type": "new_round_opened",
        "round_number": next_round_num
    })


async def _scheduler_loop():
    """Periodic loop runner."""
    logger.info("Starting background Auction Scheduler Loop...")
    db = SessionLocal()
    try:
        while not _stop_event.is_set():
            # Wait for round duration or cancel event
            try:
                await asyncio.sleep(ROUND_INTERVAL_SECONDS)
                await run_auction_round(db)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background auction round execution: {e}")
                await asyncio.sleep(2)  # recover sleep
    finally:
        db.close()
        logger.info("Auction Scheduler Loop stopped.")


def start_scheduler():
    """Start the periodic auction background task."""
    global _scheduler_task, _stop_event
    _stop_event.clear()
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(_scheduler_loop())
        logger.info("Background Auction Scheduler started.")


def stop_scheduler():
    """Stop the background scheduler task."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        _stop_event.set()
        _scheduler_task.cancel()
        logger.info("Signaled scheduler loop shutdown.")
