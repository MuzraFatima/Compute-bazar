"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Agent Memory Module (Week 6)
=============================================================
 Implements persistent cognitive memory for autonomous AI agents:
   - Tracks previous bids, win/loss outcomes, payments, and surplus
   - Tracks historical resource demand and clearing prices
   - Detects starvation (consecutive losses) and high win rates
   - Provides analytics to adapt future bidding strategies
=============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from compute_bazaar.models.bundle import CombinatorialBid


@dataclass
class BidOutcome:
    """Record of a completed auction round outcome for an agent's bid."""
    bid_id: str
    round_id: int
    won: bool
    bid_amount: float
    payment_amount: float = 0.0
    surplus: float = 0.0
    resources_allocated: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid_id": self.bid_id,
            "round_id": self.round_id,
            "won": self.won,
            "bid_amount": self.bid_amount,
            "payment_amount": self.payment_amount,
            "surplus": self.surplus,
            "resources_allocated": self.resources_allocated,
            "timestamp": self.timestamp,
        }


class AgentMemory:
    """
    Cognitive memory bank for an autonomous AI agent.

    Stores:
      - History of created/submitted bids
      - Round-by-round win/loss outcomes and payments
      - Historical clearing prices per resource type
      - Win rate, consecutive loss counters, and cumulative surplus
    """

    def __init__(self, agent_name: str = "agent"):
        self.agent_name = agent_name
        self.bid_history: List[CombinatorialBid] = []
        self.outcome_history: List[BidOutcome] = []
        self.resource_clearing_prices: Dict[str, List[float]] = {}
        
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.total_spent: float = 0.0
        self.total_surplus: float = 0.0

    def record_bid(self, bid: CombinatorialBid):
        """Record a newly generated bid in memory."""
        self.bid_history.append(bid)

    def record_outcome(
        self,
        bid_id: str,
        round_id: int,
        won: bool,
        bid_amount: float,
        payment_amount: float = 0.0,
        resources_allocated: Optional[List[str]] = None,
    ):
        """
        Record the auction round outcome for a bid.

        Args:
            bid_id: Identifier of the submitted bid
            round_id: Auction round number
            won: True if bid won allocation
            bid_amount: Declared bid amount
            payment_amount: Actual payment charged (e.g. VCG price)
            resources_allocated: List of awarded resource names
        """
        allocated = resources_allocated or []
        surplus = max(0.0, bid_amount - payment_amount) if won else 0.0

        outcome = BidOutcome(
            bid_id=bid_id,
            round_id=round_id,
            won=won,
            bid_amount=bid_amount,
            payment_amount=payment_amount if won else 0.0,
            surplus=surplus,
            resources_allocated=allocated,
        )
        self.outcome_history.append(outcome)

        # Update counters
        if won:
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.total_spent += payment_amount
            self.total_surplus += surplus

            # Record per-resource clearing prices
            if allocated and payment_amount > 0:
                per_res_price = payment_amount / len(allocated)
                for res in allocated:
                    if res not in self.resource_clearing_prices:
                        self.resource_clearing_prices[res] = []
                    self.resource_clearing_prices[res].append(per_res_price)
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def get_win_rate(self, window: int = 10) -> float:
        """
        Calculate win rate over the last `window` rounds.
        Returns float between 0.0 (no wins) and 1.0 (all wins).
        """
        if not self.outcome_history:
            return 0.0

        recent = self.outcome_history[-window:]
        wins = sum(1 for o in recent if o.won)
        return wins / len(recent)

    def is_starved(self, threshold: int = 3) -> bool:
        """
        Returns True if the agent is experiencing starvation
        (e.g., >= `threshold` consecutive losses).
        """
        return self.consecutive_losses >= threshold

    def is_dominating(self, threshold: int = 5) -> bool:
        """
        Returns True if the agent has won >= `threshold` consecutive rounds.
        """
        return self.consecutive_wins >= threshold

    def get_average_clearing_price(self, resource_name: str, default: float = 10.0) -> float:
        """
        Return the average clearing price observed for a resource.
        Falls back to default if no history exists.
        """
        prices = self.resource_clearing_prices.get(resource_name, [])
        if not prices:
            return default
        return sum(prices) / len(prices)

    def get_summary(self) -> Dict[str, Any]:
        """Return memory analytics summary."""
        return {
            "agent_name": self.agent_name,
            "total_bids_created": len(self.bid_history),
            "total_rounds_participated": len(self.outcome_history),
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate_recent": round(self.get_win_rate(10), 2),
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "is_starved": self.is_starved(),
            "total_spent": round(self.total_spent, 2),
            "total_surplus": round(self.total_surplus, 2),
        }
