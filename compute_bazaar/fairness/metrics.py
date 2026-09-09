"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Fairness Metrics & Anti-Starvation (Week 6)
=============================================================
 Provides fairness analytics and protection mechanisms:
   - Jain's Fairness Index computation for agent surplus and win counts
   - Starvation detection and risk score calculation
   - Priority-aware allocation boost suggestions
   - Resource domination detection (preventing single agent monopoly)
=============================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger


def jains_fairness_index(values: List[float]) -> float:
    """
    Compute Jain's Fairness Index for a list of values (e.g. agent surpluses or wins).

    Formula:
      J(x) = (sum(x_i))^2 / (n * sum(x_i^2))

    Range: [1/n, 1.0]
      - 1.0 = perfectly fair (equal distribution across all agents)
      - 1/n = maximally unfair (one agent receives everything)

    Args:
        values: List of non-negative float/int values.

    Returns:
        Fairness index float in [0.0, 1.0].
    """
    if not values:
        return 1.0

    # Ensure non-negative
    non_neg = [max(0.0, float(v)) for v in values]
    n = len(non_neg)

    if n == 0:
        return 1.0

    sum_x = sum(non_neg)
    sum_x_sq = sum(v ** 2 for v in non_neg)

    if sum_x_sq == 0.0 or sum_x == 0.0:
        return 1.0  # Equal distribution of zero surplus is considered fair

    index = (sum_x ** 2) / (n * sum_x_sq)
    return round(max(0.0, min(1.0, index)), 4)


def calculate_starvation_score(consecutive_losses: int, win_rate: float) -> float:
    """
    Calculate a normalized starvation risk score in [0.0, 1.0].

    High score indicates an agent is being starved out of the auction market.
    """
    loss_factor = min(1.0, consecutive_losses / 5.0)
    win_factor = max(0.0, 1.0 - win_rate)
    score = 0.6 * loss_factor + 0.4 * win_factor
    return round(max(0.0, min(1.0, score)), 4)


class FairnessTracker:
    """
    Tracks market fairness, agent win rates, surplus equity, and starvation state.

    Used by the auction orchestrator or dashboard to monitor market equity
    and suggest priority boosts for starved participants.
    """

    def __init__(self):
        # agent_name -> statistics dict
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        self.round_history: List[Dict[str, Any]] = []

    def register_agent(self, agent_name: str):
        """Register an agent for fairness tracking."""
        if agent_name not in self.agent_stats:
            self.agent_stats[agent_name] = {
                "total_bids": 0,
                "total_wins": 0,
                "consecutive_losses": 0,
                "consecutive_wins": 0,
                "total_surplus": 0.0,
                "total_spent": 0.0,
                "history": [],  # List of bools: True for win, False for loss
            }

    def record_round_outcome(
        self,
        round_id: int,
        outcomes: Dict[str, Dict[str, Any]],
    ):
        """
        Record auction round outcomes for all participating agents.

        Args:
            round_id: Auction round number
            outcomes: Map of agent_name -> {"won": bool, "surplus": float, "payment": float}
        """
        for agent_name, res in outcomes.items():
            self.register_agent(agent_name)
            stats = self.agent_stats[agent_name]

            won = bool(res.get("won", False))
            surplus = float(res.get("surplus", 0.0))
            payment = float(res.get("payment", 0.0))

            stats["total_bids"] += 1
            stats["history"].append(won)

            if won:
                stats["total_wins"] += 1
                stats["consecutive_wins"] += 1
                stats["consecutive_losses"] = 0
                stats["total_surplus"] += surplus
                stats["total_spent"] += payment
            else:
                stats["consecutive_losses"] += 1
                stats["consecutive_wins"] = 0

        self.round_history.append({"round_id": round_id, "outcomes": outcomes})

    def compute_jains_fairness_surplus(self) -> float:
        """Compute Jain's Fairness Index for agent surplus distribution."""
        if not self.agent_stats:
            return 1.0
        surpluses = [s["total_surplus"] for s in self.agent_stats.values()]
        return jains_fairness_index(surpluses)

    def compute_jains_fairness_wins(self) -> float:
        """Compute Jain's Fairness Index for agent win counts."""
        if not self.agent_stats:
            return 1.0
        wins = [float(s["total_wins"]) for s in self.agent_stats.values()]
        return jains_fairness_index(wins)

    def detect_starved_agents(self, threshold: int = 3) -> List[str]:
        """
        Return names of agents currently experiencing starvation
        (consecutive_losses >= threshold).
        """
        return [
            name for name, stats in self.agent_stats.items()
            if stats["consecutive_losses"] >= threshold
        ]

    def detect_dominating_agents(self, domination_threshold: float = 0.65) -> List[str]:
        """
        Return names of agents that are dominating resource allocations
        (win_rate >= domination_threshold over at least 5 rounds).
        """
        dominating = []
        for name, stats in self.agent_stats.items():
            history = stats["history"]
            if len(history) >= 5:
                recent_win_rate = sum(1 for w in history[-10:] if w) / len(history[-10:])
                if recent_win_rate >= domination_threshold:
                    dominating.append(name)
        return dominating

    def get_priority_boost_recommendation(self, agent_name: str) -> str:
        """
        Suggest a priority boost for an agent based on its starvation status.

        Returns:
            "critical" if heavily starved (>= 4 consecutive losses)
            "high" if moderately starved (>= 2 consecutive losses)
            "normal" otherwise
        """
        stats = self.agent_stats.get(agent_name)
        if not stats:
            return "normal"

        losses = stats["consecutive_losses"]
        if losses >= 4:
            return "critical"
        elif losses >= 2:
            return "high"
        return "normal"

    def get_fairness_report(self) -> Dict[str, Any]:
        """Return comprehensive market fairness diagnostics."""
        starved = self.detect_starved_agents()
        dominating = self.detect_dominating_agents()

        return {
            "total_registered_agents": len(self.agent_stats),
            "jains_fairness_surplus": self.compute_jains_fairness_surplus(),
            "jains_fairness_wins": self.compute_jains_fairness_wins(),
            "starved_agents": starved,
            "dominating_agents": dominating,
            "agent_summaries": {
                name: {
                    "total_wins": s["total_wins"],
                    "total_bids": s["total_bids"],
                    "consecutive_losses": s["consecutive_losses"],
                    "total_surplus": round(s["total_surplus"], 2),
                    "starvation_score": calculate_starvation_score(
                        s["consecutive_losses"],
                        (s["total_wins"] / s["total_bids"]) if s["total_bids"] > 0 else 0.0,
                    ),
                }
                for name, s in self.agent_stats.items()
            },
        }
