"""
=============================================================
 COMPUTE BAZAAR — Machine Learning & Auto-Bidding Strategies
=============================================================
 Autonomous agent bidding strategies:
   - Truthful (Dominant strategy in VCG auctions)
   - Shading (Surplus margin maximization)
   - AdaptiveQLearner (RL Q-Learning agent with reward-based updates)
   - UrgentPriority (Latency/deadline sensitive critical bidding)
   - ContextualBandit (Exploration-exploitation bandit agent)
=============================================================
"""

import abc
import math
import random
from typing import Dict, List, Optional, Tuple, Any
from compute_bazaar.models.bundle import CombinatorialBid, AllocationResult, PriorityLevel


class BiddingStrategy(abc.ABC):
    """Abstract Base Class for agent bidding strategies."""

    def __init__(self, agent_name: str, tier: str = "bronze"):
        self.agent_name = agent_name
        self.tier = tier

    @abc.abstractmethod
    def compute_bid_amount(self, valuation: float, bundle: List[str], current_budget: float) -> float:
        """Compute the bid amount to submit given true valuation and bundle."""
        pass

    def feedback(self, bid: CombinatorialBid, won: bool, payment: float, surplus: float):
        """Receive outcome feedback from the completed auction round to learn/adapt."""
        pass


class TruthfulStrategy(BiddingStrategy):
    """
    Truthful Bidding Strategy:
    Under Vickrey-Clarke-Groves (VCG) mechanism, bidding your exact
    true valuation is a dominant strategy (DSIC).
    """

    def compute_bid_amount(self, valuation: float, bundle: List[str], current_budget: float) -> float:
        return min(valuation, current_budget)


class BudgetAwareShadingStrategy(BiddingStrategy):
    """
    Shading Strategy:
    Bids a fraction (e.g. 70%) of true valuation to secure high profit margins (surplus),
    while pacing expenditure against total remaining budget.
    """

    def __init__(self, agent_name: str, shade_fraction: float = 0.75, tier: str = "silver"):
        super().__init__(agent_name, tier)
        self.shade_fraction = shade_fraction

    def compute_bid_amount(self, valuation: float, bundle: List[str], current_budget: float) -> float:
        bid = valuation * self.shade_fraction
        return max(0.0, min(bid, current_budget))


class UrgentPriorityStrategy(BiddingStrategy):
    """
    Urgent Critical Strategy:
    For mission-critical, deadline-sensitive workloads. Bids up to 98% of valuation
    with Critical priority tagging to maximize allocation probability.
    """

    def compute_bid_amount(self, valuation: float, bundle: List[str], current_budget: float) -> float:
        bid = valuation * 0.98
        return max(0.0, min(bid, current_budget))


class AdaptiveQLearner(BiddingStrategy):
    """
    Reinforcement Learning (Q-Learning) Auto-Bidder:
    Maintains a discrete action space of bidding multipliers (e.g., 0.40 to 1.00).
    State is discretely approximated by current win rate or competition density.
    Reward = (Valuation - Payment) if won else 0 (with small penalty if repeatedly losing).
    """

    def __init__(
        self,
        agent_name: str,
        actions: Optional[List[float]] = None,
        learning_rate: float = 0.15,
        discount_factor: float = 0.90,
        epsilon: float = 0.20,
        tier: str = "gold",
    ):
        super().__init__(agent_name, tier)
        self.actions = actions or [0.40, 0.55, 0.70, 0.85, 0.95, 1.00]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: state (win_rate_bucket) -> action_idx -> expected_q_value
        self.q_table: Dict[int, List[float]] = {}
        self.history: List[bool] = []
        self.last_state: int = 0
        self.last_action_idx: int = 0

    def _get_state(self) -> int:
        """Discretize recent 10 rounds win-rate into 4 discrete buckets [0, 1, 2, 3]."""
        if not self.history:
            return 1
        recent = self.history[-10:]
        win_rate = sum(1 for w in recent if w) / len(recent)
        if win_rate < 0.25:
            return 0  # Cold / losing
        elif win_rate < 0.50:
            return 1  # Moderate
        elif win_rate < 0.75:
            return 2  # Good
        return 3      # High win rate

    def compute_bid_amount(self, valuation: float, bundle: List[str], current_budget: float) -> float:
        state = self._get_state()
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.actions)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.actions) - 1)
        else:
            action_idx = int(max(range(len(self.actions)), key=lambda i: self.q_table[state][i]))

        self.last_state = state
        self.last_action_idx = action_idx

        multiplier = self.actions[action_idx]
        bid = valuation * multiplier
        return max(0.0, min(round(bid, 2), current_budget))

    def feedback(self, bid: CombinatorialBid, won: bool, payment: float, surplus: float):
        """Update Q-table with observed reward."""
        self.history.append(won)
        reward = surplus if won else -1.0  # Penalize losing slightly to encourage competitive bidding
        
        next_state = self._get_state()
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * len(self.actions)

        # Q-learning Bellman update: Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
        current_q = self.q_table[self.last_state][self.last_action_idx]
        max_future_q = max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[self.last_state][self.last_action_idx] = new_q


class ContextualBanditStrategy(BiddingStrategy):
    """
    Multi-Armed Bandit Strategy:
    Uses Upper Confidence Bound (UCB1) per bundle size to learn optimal bid multipliers.
    """

    def __init__(self, agent_name: str, tier: str = "gold"):
        super().__init__(agent_name, tier)
        self.multipliers = [0.50, 0.65, 0.80, 0.95, 1.00]
        # Counts and rewards per bundle size category (1-item, 2-item, 3+-item)
        self.action_counts: Dict[str, List[int]] = {
            "small": [0] * len(self.multipliers),
            "medium": [0] * len(self.multipliers),
            "large": [0] * len(self.multipliers),
        }
        self.action_rewards: Dict[str, List[float]] = {
            "small": [0.0] * len(self.multipliers),
            "medium": [0.0] * len(self.multipliers),
            "large": [0.0] * len(self.multipliers),
        }
        self.last_category: str = "small"
        self.last_action_idx: int = 0

    def _category(self, bundle: List[str]) -> str:
        if len(bundle) <= 1:
            return "small"
        elif len(bundle) == 2:
            return "medium"
        return "large"

    def compute_bid_amount(self, valuation: float, bundle: List[str], current_budget: float) -> float:
        cat = self._category(bundle)
        total_counts = sum(self.action_counts[cat])
        
        # UCB selection
        selected_idx = 0
        if 0 in self.action_counts[cat]:
            selected_idx = self.action_counts[cat].index(0)
        else:
            best_score = -float("inf")
            for i in range(len(self.multipliers)):
                avg_reward = self.action_rewards[cat][i] / self.action_counts[cat][i]
                confidence = math.sqrt(2 * math.log(total_counts) / self.action_counts[cat][i])
                score = avg_reward + confidence
                if score > best_score:
                    best_score = score
                    selected_idx = i

        self.last_category = cat
        self.last_action_idx = selected_idx
        multiplier = self.multipliers[selected_idx]
        bid = valuation * multiplier
        return max(0.0, min(round(bid, 2), current_budget))

    def feedback(self, bid: CombinatorialBid, won: bool, payment: float, surplus: float):
        cat = self.last_category
        idx = self.last_action_idx
        reward = max(0.0, surplus) if won else 0.0
        self.action_counts[cat][idx] += 1
        self.action_rewards[cat][idx] += reward
