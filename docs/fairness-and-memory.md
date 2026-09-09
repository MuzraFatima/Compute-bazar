# Compute Bazaar 2.0 -- Agent Memory & Fairness Architecture Guide

> Week 6 deliverable. Documents the cognitive agent memory layer, Jain's fairness index, starvation detection, anti-monopoly domination protection, and adaptive bidding strategies.

---

## 1. Overview

Compute Bazaar 2.0 Week 6 adds cognitive **Agent Memory** and a comprehensive **Fairness Framework** (`compute_bazaar.fairness`). Autonomous AI agents track auction outcomes across rounds to adapt future bidding strategies dynamically, while market orchestrators monitor equity using Jain's Fairness Index and prevent starvation.

```
compute_bazaar/
  agents/
    memory.py        # AgentMemory & BidOutcome classes
    base.py          # BaseAgent with adaptive bidding & outcome recording
  fairness/
    __init__.py      # Package exports (jains_fairness_index, calculate_starvation_score, FairnessTracker)
    metrics.py       # Math implementations & market fairness tracking
```

---

## 2. Cognitive Agent Memory (`AgentMemory`)

Each `BaseAgent` instance maintains an `AgentMemory` bank storing:

| Field | Type | Description |
|:--|:--|:--|
| `bid_history` | `List[CombinatorialBid]` | Complete log of created and submitted bids |
| `outcome_history` | `List[BidOutcome]` | Round-by-round log of win/loss results, VCG payments, and surplus |
| `consecutive_losses` | `int` | Counter tracking active consecutive loss streak |
| `consecutive_wins` | `int` | Counter tracking active consecutive win streak |
| `resource_clearing_prices` | `Dict[str, List[float]]` | Historical winning prices observed per resource |

### Key Memory Analytics
- `get_win_rate(window=10) -> float`: Calculates recent win probability over the last $N$ rounds.
- `is_starved(threshold=3) -> bool`: Returns `True` if agent has suffered $\ge 3$ consecutive losses.
- `get_average_clearing_price(resource_name) -> float`: Returns average clearing price paid for a resource.

---

## 3. Fairness Metrics & Starvation Prevention

### 3.1 Jain's Fairness Index

Computes allocation and surplus equity across all active market participants:

$$J(x) = \frac{\left(\sum_{i=1}^n x_i\right)^2}{n \sum_{i=1}^n x_i^2}$$

- **Range**: $[1/n, 1.0]$
  - `1.0`: Perfectly equal distribution of surplus/wins.
  - `1/n`: Highly concentrated / monopolized distribution.

### 3.2 Starvation Risk Score

Normalized starvation risk formula ($S_r \in [0.0, 1.0]$):

$$S_r = 0.6 \cdot \min\left(1.0, \frac{\text{consecutive\_losses}}{5}\right) + 0.4 \cdot (1.0 - \text{win\_rate})$$

### 3.3 FairnessTracker & Domination Protection

The `FairnessTracker` monitors:
- **Starved Agents**: Identified when consecutive losses exceed threshold ($\ge 3$).
- **Dominating Agents**: Identified when win rate exceeds $65\%$ over recent rounds.
- **Priority Escalation Recommendations**: Recommends `"high"` or `"critical"` priority boosts for starved agents.

---

## 4. Adaptive Bidding Mechanics

Agents adapt their bidding strategy based on their cognitive memory:

1. **Starvation Adaptation**:
   When `memory.is_starved()` is `True`, the agent:
   - Escalates task priority from `normal`/`low` to `high` or `critical`.
   - Boosts its bid shading multiplier to $96\%$ of maximum willingness-to-pay ($WTP$) to increase win probability.

2. **High Win-Rate Margin Maximization**:
   When `memory.get_win_rate(10) > 0.70`, the agent:
   - Shades its bid multiplier down to $72\%$ of $WTP$ to maximize profit surplus margin while maintaining competitiveness.

3. **Historical Clearing Price Alignment**:
   - Ensures bids do not fall below known average market clearing prices ($\text{bid} \ge 1.05 \times P_{\text{clearing}}$), bounded strictly by $WTP$ and remaining budget.

---

## 5. Code Example

```python
from compute_bazaar.agents import CodingAgent
from compute_bazaar.fairness import FairnessTracker, jains_fairness_index
from compute_bazaar.llm import GroqLLMClient, LLMConfig

# 1. Initialize Agent and Tracker
llm = GroqLLMClient(config=LLMConfig(mock_mode=True))
agent = CodingAgent("CodeBot", budget=500.0, llm_client=llm)
tracker = FairnessTracker()

# 2. Agent creates adaptive bid
bid = agent.create_bid("Train neural network model")

# 3. Record Round Outcome into Agent Memory and Tracker
agent.record_auction_outcome(
    bid_id=bid.bid_id,
    round_id=1,
    won=True,
    bid_amount=bid.bid_amount,
    payment_amount=75.0,
    resources_allocated=bid.resource_bundle,
)

tracker.record_round_outcome(1, {
    "CodeBot": {"won": True, "surplus": bid.bid_amount - 75.0, "payment": 75.0}
})

# 4. Check Fairness Analytics
print(f"Jain's Surplus Index: {tracker.compute_jains_fairness_surplus():.4f}")
print(f"Agent Memory Win Rate: {agent.memory.get_win_rate():.0%}")
```

---

## 6. Architectural Safeguards

- **100% Backward Compatible**: Existing ILP WDP solver, VCG pricing, REST endpoints, and SDK remain untouched.
- **Budget Enforcement**: All adaptive bids remain strictly bounded by `min(WTP, estimated_utility, remaining_budget)`.
- **API Key Security**: Tests execute strictly in `MOCK` mode using `LLMConfig(mock_mode=True)`. API keys are never exposed.
