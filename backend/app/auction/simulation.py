"""
=============================================================
 COMPUTE BAZAAR — Automated Agent Simulator (Weeks 3-8)
=============================================================
 Simulates agents bidding in the auction rounds.
 Behaviors:
   - Agent-Alpha (Gold, ML-Adaptive): adjusts bids based on win rate
   - Agent-Beta (Gold, Truthful): always bids its true valuation
   - Agent-Gamma (Silver, Strategic Shading): underbids by 30% to maximize surplus
   - Agent-Delta (Silver, Critical Impulsive): high-value, critical priority bids
   - Agent-Epsilon (Bronze, Bargain Hunter): low bids, high volume
=============================================================
"""

import json
import random
import asyncio
from loguru import logger
from sqlalchemy.orm import Session

from backend.app.db.models import SessionLocal, Agent, Resource, Bid, Auction, init_db
from backend.app.api.websocket import manager

# Simulated Agent configurations
AGENT_BEHAVIORS = {
    "Agent-Alpha":   {"tier": "gold",   "strategy": "ml_adaptive", "desc": "Reinforcement learning bidder"},
    "Agent-Beta":    {"tier": "gold",   "strategy": "truthful",    "desc": "True value bidder"},
    "Agent-Gamma":   {"tier": "silver", "strategy": "shading",     "desc": "Strategic under-bidder"},
    "Agent-Delta":   {"tier": "silver", "strategy": "critical",    "desc": "Urgent critical task bidder"},
    "Agent-Epsilon": {"tier": "bronze", "strategy": "bargain",     "desc": "Low cost task seeker"},
}

_sim_loop_task = None
_sim_stop_event = asyncio.Event()


def ensure_simulated_agents_exist(db: Session):
    """Check and create the simulated agents if they do not exist."""
    init_db()
    for name, config in AGENT_BEHAVIORS.items():
        agent = db.query(Agent).filter(Agent.name == name).first()
        if not agent:
            agent = Agent(
                name=name,
                description=config["desc"],
                api_key=f"sim_key_{name.lower().replace('-', '_')}",
                budget_total=1000.0,
                budget_remaining=1000.0,
                tier=config["tier"]
            )
            db.add(agent)
    db.commit()


class WinRateTracker:
    """Tracks simple win-rate history for ML adaptive agents."""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.history = []  # List of booleans (True = won, False = lost)

    def record_round(self, won: bool):
        self.history.append(won)
        if len(self.history) > 10:
            self.history.pop(0)

    def get_win_rate(self) -> float:
        if not self.history:
            return 0.5
        return sum(1 for w in self.history if w) / len(self.history)


# Track win rates globally for ML agent in simulation memory
win_trackers = {name: WinRateTracker(name) for name in AGENT_BEHAVIORS}


def calculate_bid_amount(agent_name: str, valuation: float, strategy: str) -> float:
    """Compute bidding amount based on strategy profile."""
    if strategy == "truthful":
        # Always bids true valuation
        return valuation
        
    elif strategy == "shading":
        # Bids 30% lower to attempt high profit margin
        return valuation * 0.70
        
    elif strategy == "critical":
        # Urgent high value task, bids aggressively (95%)
        return valuation * 0.95
        
    elif strategy == "bargain":
        # Wants cheap slots, bids only 40% of value
        return valuation * 0.40
        
    elif strategy == "ml_adaptive":
        # ML bidding strategy: adjusts bid multiplier dynamically
        win_rate = win_trackers[agent_name].get_win_rate()
        if win_rate < 0.3:
            # Losing too much, bid higher (aggressively up to 90%)
            multiplier = 0.90
        elif win_rate > 0.7:
            # Winning too easily, shave bid down to 55% to save budget
            multiplier = 0.55
        else:
            # Stable rate, bid standard 75%
            multiplier = 0.75
        return valuation * multiplier
        
    return valuation * 0.80


async def generate_simulated_bids(db: Session, auction_id: int):
    """
    Simulate bids from active simulated agents for the current round.
    Saves bids directly to database.
    """
    db_resources = db.query(Resource).filter(Resource.is_available == True).all()
    if not db_resources:
        return
        
    res_names = [r.name for r in db_resources]
    
    # Let each agent submit a bid with some probability
    for agent_name, config in AGENT_BEHAVIORS.items():
        if random.random() > 0.85:
            continue  # Skip bidding in this round occasionally to mimic network delays

        agent = db.query(Agent).filter(Agent.name == agent_name).first()
        if not agent or agent.budget_remaining < 50.0:
            continue  # Out of budget
            
        # Select a random resource bundle (1 to 3 items)
        bundle_size = random.randint(1, min(3, len(res_names)))
        bundle = random.sample(res_names, bundle_size)
        
        # Calculate valuation base ($40 per resource item on average)
        valuation = sum(random.uniform(30.0, 100.0) for _ in bundle)
        bid_val = calculate_bid_amount(agent_name, valuation, config["strategy"])
        
        # Clamp bid to remaining budget
        bid_val = min(bid_val, agent.budget_remaining)
        if bid_val <= 1.0:
            continue

        # Priority definition
        priority = "normal"
        if config["strategy"] == "critical":
            priority = "critical"
        elif config["strategy"] == "bargain":
            priority = "low"
        elif random.random() > 0.8:
            priority = "high"

        tasks = [
            "Process financial ledger transaction",
            "Generate AI image thumbnail",
            "Fine-tune neural network weights",
            "Customer sentiment classifier task",
            "Compile LLM benchmark analytics",
            "Clean dataset outlier entries",
            "Cache inference logits batch"
        ]
        task_desc = random.choice(tasks)

        # Create and save bid
        bid = Bid(
            auction_id=auction_id,
            agent_id=agent.id,
            resource_bundle=json.dumps(bundle),
            bid_amount=round(bid_val, 2),
            task_description=task_desc,
            priority=priority
        )
        db.add(bid)
        db.commit()

        # Update win tracker (this will get updated once the round closes)
        # We query and update win rates on round completion
        
        # Broadcast the newly submitted bid to WebSockets for live feed
        await manager.broadcast_json({
            "type": "bid_submitted",
            "agent": agent_name,
            "bundle": bundle,
            "amount": round(bid_val, 2),
            "priority": priority,
            "task": task_desc,
            "submitted_at": bid.submitted_at
        })


async def _simulation_loop():
    """Background simulator runner."""
    logger.info("Simulation Background Loop started.")
    db = SessionLocal()
    try:
        ensure_simulated_agents_exist(db)
        
        while not _sim_stop_event.is_set():
            try:
                # Find current open auction round
                active_auction = db.query(Auction).filter(Auction.status == "open").order_by(Auction.round_number.desc()).first()
                if active_auction:
                    # Submit some simulated bids
                    await generate_simulated_bids(db, active_auction.id)
                    
                # Submit bids every 2.5 seconds to keep the live feed scrolling
                await asyncio.sleep(2.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(2)
    finally:
        db.close()
        logger.info("Simulation Background Loop stopped.")


def start_simulation_loop():
    """Start spawning simulated agents in the background."""
    global _sim_loop_task, _sim_stop_event
    _sim_stop_event.clear()
    if _sim_loop_task is None or _sim_loop_task.done():
        _sim_loop_task = asyncio.create_task(_simulation_loop())
        logger.info("Bidding simulator task spawned.")


def stop_simulation_loop():
    """Signal simulation task to stop."""
    global _sim_loop_task
    if _sim_loop_task and not _sim_loop_task.done():
        _sim_stop_event.set()
        _sim_loop_task.cancel()
        logger.info("Signaled bidding simulator shutdown.")


def is_simulation_running() -> bool:
    """Return whether background simulation is currently active."""
    return _sim_loop_task is not None and not _sim_loop_task.done()


def update_simulation_win_rates(db: Session, closed_round_id: int):
    """
    Called after an auction closes. Update win histories
    so ML agents adapt their strategies.
    """
    bids = db.query(Bid).filter(Bid.auction_id == closed_round_id).all()
    for b in bids:
        agent_name = b.agent.name
        if agent_name in win_trackers:
            win_trackers[agent_name].record_round(b.is_winner)
