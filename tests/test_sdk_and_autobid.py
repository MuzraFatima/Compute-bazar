"""
=============================================================
 COMPUTE BAZAAR — Test Suite: Python SDK & ML Auto-Bidding
=============================================================
 Validates:
   - Data models (ResourceItem, CombinatorialBid, AllocationResult)
   - Workload and synthetic bid generation
   - ML auto-bidding strategies (QLearner, Bandit, Truthful, Shading)
   - Python Client SDK methods
=============================================================
"""

import pytest
from compute_bazaar.models.bundle import (
    ResourceItem,
    ResourceType,
    CombinatorialBid,
    AllocationResult,
    AuctionRoundOutcome,
    PriorityLevel,
    AgentTier,
)
from compute_bazaar.generators.workload import (
    generate_synthetic_bid,
    generate_workload_batch,
    TASK_TEMPLATES,
)
from compute_bazaar.autobid.strategies import (
    TruthfulStrategy,
    BudgetAwareShadingStrategy,
    UrgentPriorityStrategy,
    AdaptiveQLearner,
    ContextualBanditStrategy,
)
from compute_bazaar.client.sdk import ComputeBazaarClient, ComputeBazaarError


def test_resource_item_model():
    res = ResourceItem(name="GPU-A100", resource_type=ResourceType.GPU.value, capacity=1.0, cost_per_unit=3.0)
    assert res.name == "GPU-A100"
    assert res.is_available is True
    d = res.to_dict()
    assert d["cost_per_unit"] == 3.0
    assert hash(res) == hash("GPU-A100")


def test_combinatorial_bid_model():
    bid = CombinatorialBid(
        agent_name="Agent-1",
        resource_bundle=["GPU-1", "RAM-1"],
        bid_amount=100.0,
        task_description="Fine-tuning",
    )
    assert bid.bundle_size == 2
    assert bid.value_per_resource == 50.0
    assert bid.valuation == 100.0
    assert bid.to_dict()["agent_name"] == "Agent-1"


def test_allocation_result_model():
    alloc = AllocationResult(
        round_id=1,
        agent_name="Agent-1",
        bid_id="b1",
        resources_allocated=["GPU-1"],
        bid_amount=80.0,
        payment_amount=60.0,
    )
    assert alloc.surplus == 20.0
    assert alloc.to_dict()["surplus"] == 20.0


def test_synthetic_bid_generation():
    bid = generate_synthetic_bid("Agent-Test", available_resource_names=["GPU-Slot-1", "API-Budget-100"])
    assert bid.agent_name == "Agent-Test"
    assert len(bid.resource_bundle) > 0
    assert bid.bid_amount > 0.0


def test_workload_batch_generation():
    batch = generate_workload_batch(num_agents=5)
    assert len(batch) == 5
    assert all(isinstance(b, CombinatorialBid) for b in batch)


def test_truthful_strategy():
    strat = TruthfulStrategy("Agent-Truth")
    bid = strat.compute_bid_amount(valuation=100.0, bundle=["GPU-1"], current_budget=1000.0)
    assert bid == 100.0
    # Capped at remaining budget
    bid_capped = strat.compute_bid_amount(valuation=100.0, bundle=["GPU-1"], current_budget=50.0)
    assert bid_capped == 50.0


def test_shading_strategy():
    strat = BudgetAwareShadingStrategy("Agent-Shade", shade_fraction=0.80)
    bid = strat.compute_bid_amount(valuation=100.0, bundle=["GPU-1"], current_budget=500.0)
    assert bid == 80.0


def test_urgent_priority_strategy():
    strat = UrgentPriorityStrategy("Agent-Urgent")
    bid = strat.compute_bid_amount(valuation=100.0, bundle=["GPU-1"], current_budget=500.0)
    assert bid == 98.0


def test_adaptive_q_learner_feedback():
    strat = AdaptiveQLearner("Agent-QLearn", epsilon=0.0)  # Pure exploitation
    # Initial bid
    bid1 = strat.compute_bid_amount(valuation=100.0, bundle=["GPU-1"], current_budget=500.0)
    assert bid1 > 0.0

    # Provide positive reward feedback
    dummy_bid = CombinatorialBid(agent_name="Agent-QLearn", resource_bundle=["GPU-1"], bid_amount=bid1)
    strat.feedback(bid=dummy_bid, won=True, payment=50.0, surplus=50.0)
    
    # Check that learning updated internal table
    assert len(strat.history) == 1
    assert strat.history[0] is True


def test_contextual_bandit_strategy():
    bandit = ContextualBanditStrategy("Agent-Bandit")
    # Small bundle
    bid_small = bandit.compute_bid_amount(valuation=100.0, bundle=["GPU-1"], current_budget=500.0)
    assert 40.0 <= bid_small <= 100.0
    
    # Feed reward
    dummy_bid = CombinatorialBid(agent_name="Agent-Bandit", resource_bundle=["GPU-1"], bid_amount=bid_small)
    bandit.feedback(bid=dummy_bid, won=True, payment=40.0, surplus=60.0)
    assert bandit.action_counts["small"][bandit.last_action_idx] == 1


def test_client_sdk_initialization():
    client = ComputeBazaarClient(base_url="http://127.0.0.1:8000", api_key="test_key_123")
    assert client.base_url == "http://127.0.0.1:8000"
    assert client.api_key == "test_key_123"
