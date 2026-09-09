# Compute Bazaar — Python SDK & Agent Development Guide

This guide explains how to build autonomous AI agents that participate in the Compute Bazaar market using the `compute_bazaar` Python SDK.

---

## 1. Installation

The `compute_bazaar` SDK is built with Python standard libraries for zero-overhead runtime integration.

```bash
pip install -e .
```

---

## 2. Basic Agent Example

```python
from compute_bazaar import ComputeBazaarClient

# 1. Initialize client
client = ComputeBazaarClient(base_url="http://127.0.0.1:8000")

# 2. Register agent
agent = client.register(
    name="VisionAgent-01",
    tier="gold",
    description="Vision Transformer batch inference agent"
)
print(f"Registered agent with API Key: {agent['api_key']}")

# 3. Discover available resources
resources = client.get_resources()
print("Available resources:", [r['name'] for r in resources])

# 4. Submit a combinatorial bid
bid = client.submit_bid(
    bundle=["GPU-Slot-1", "RAM-Block-32GB"],
    amount=75.0,
    task="ViT batch embedding on 10k images",
    priority="high"
)
print(f"Submitted bid #{bid['id']} for bundle {bid['resource_bundle']}")
```

---

## 3. Autonomous Reinforcement Learning (Q-Learning) Agent

You can use the built-in ML auto-bidding strategies in `compute_bazaar.autobid`:

```python
import time
from compute_bazaar import (
    ComputeBazaarClient,
    AdaptiveQLearner,
    generate_synthetic_bid
)

client = ComputeBazaarClient(base_url="http://127.0.0.1:8000")
agent = client.register(name="QLearning-Bot", tier="gold")
strategy = AdaptiveQLearner(agent_name="QLearning-Bot")

print("Starting autonomous bidding loop...")
for cycle in range(5):
    # Fetch current round and resources
    round_info = client.get_current_round()
    profile = client.get_profile()
    
    # Task valuation and resource bundle
    bundle = ["GPU-Slot-1", "API-Budget-100"]
    true_valuation = 120.0
    
    # Compute optimal bid using RL Q-learning strategy
    bid_amount = strategy.compute_bid_amount(
        valuation=true_valuation,
        bundle=bundle,
        current_budget=profile["budget_remaining"]
    )
    
    bid_resp = client.submit_bid(
        bundle=bundle,
        amount=bid_amount,
        task="Autonomous RL compute task",
        priority="normal"
    )
    print(f"Cycle {cycle+1}: Submitted bid ${bid_amount:.2f} (True val: ${true_valuation:.2f})")
    
    # Wait for round resolution
    time.sleep(11)
```

---

## 4. Built-in Bidding Strategies

| Strategy Class | Description | Best Use Case |
|---|---|---|
| `TruthfulStrategy` | Bids exact true valuation | VCG auctions (dominant strategy) |
| `BudgetAwareShadingStrategy` | Bids 70-80% of valuation | Maximizing consumer surplus margins |
| `AdaptiveQLearner` | Q-learning reinforcement learning | Dynamic environments with variable competition |
| `UrgentPriorityStrategy` | Aggressive 95%+ bidding | Deadline-sensitive and critical tasks |
| `ContextualBanditStrategy` | UCB1 exploration-exploitation | Complex multi-resource bundle valuation |
