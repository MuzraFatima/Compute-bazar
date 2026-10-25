"""
=============================================================
 COMPUTE BAZAAR — Week 1 Unit Tests
=============================================================
 Tests every part of the simple auction engine.
 Run with:  python -m pytest tests/ -v
=============================================================
"""

import sys
import os

# Add the project root to Python path so we can import examples.week1_demo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from examples.week1_demo import Resource, Bid, SimpleAuctionEngine
except ImportError:
    from week1_demo import Resource, Bid, SimpleAuctionEngine


# ─────────────────────────────────────────────────────────────
# HELPER: Build a standard engine with 3 resources
# ─────────────────────────────────────────────────────────────

def make_engine():
    resources = [
        Resource("GPU-1",    "gpu",        1.0),
        Resource("GPU-2",    "gpu",        1.0),
        Resource("API-100",  "api_budget", 100.0),
    ]
    return SimpleAuctionEngine(resources)


# ─────────────────────────────────────────────────────────────
# TEST GROUP 1: Resource Validation
# ─────────────────────────────────────────────────────────────

def test_resource_creation():
    """Resources should store name, type, and capacity correctly."""
    r = Resource("GPU-1", "gpu", 1.0)
    assert r.name == "GPU-1"
    assert r.resource_type == "gpu"
    assert r.capacity == 1.0


def test_invalid_bid_unknown_resource():
    """Bidding on a resource that doesn't exist should return False."""
    engine = make_engine()
    bad_bid = Bid("Agent-X", "some task", ["NONEXISTENT-RESOURCE"], 50.0)
    result = engine.add_bid(bad_bid)
    assert result == False, "Should reject bid for unknown resource"


def test_invalid_bid_zero_amount():
    """Bidding $0 or negative should be rejected."""
    engine = make_engine()
    bad_bid = Bid("Agent-X", "some task", ["GPU-1"], 0.0)
    result = engine.add_bid(bad_bid)
    assert result == False, "Should reject zero-amount bid"


def test_invalid_bid_negative_amount():
    """Negative bids must be rejected."""
    engine = make_engine()
    bad_bid = Bid("Agent-X", "some task", ["GPU-1"], -10.0)
    result = engine.add_bid(bad_bid)
    assert result == False, "Should reject negative bid"


def test_valid_bid_accepted():
    """A valid bid should return True when added."""
    engine = make_engine()
    bid = Bid("Agent-A", "valid task", ["GPU-1"], 30.0)
    result = engine.add_bid(bid)
    assert result == True
    assert len(engine.bids) == 1


# ─────────────────────────────────────────────────────────────
# TEST GROUP 2: Winner Determination
# ─────────────────────────────────────────────────────────────

def test_single_bid_wins():
    """With only one bid, it should always win."""
    engine = make_engine()
    engine.add_bid(Bid("Agent-A", "task", ["GPU-1"], 40.0))
    result = engine.run_auction()
    assert "Agent-A" in result["winners"]
    assert result["total_value"] == 40.0


def test_highest_value_per_resource_wins():
    """
    Two agents want the same resource.
    The one with higher value-per-resource should win.
    """
    engine = make_engine()
    engine.add_bid(Bid("Low-Bidder",  "task", ["GPU-1"], 10.0))
    engine.add_bid(Bid("High-Bidder", "task", ["GPU-1"], 80.0))
    result = engine.run_auction()
    assert "High-Bidder" in result["winners"]
    assert "Low-Bidder" not in result["winners"]


def test_no_resource_double_allocated():
    """
    The same resource must NEVER be given to two different agents.
    This is the core correctness constraint of any auction.
    """
    engine = make_engine()
    engine.add_bid(Bid("Agent-A", "task", ["GPU-1"], 50.0))
    engine.add_bid(Bid("Agent-B", "task", ["GPU-1"], 60.0))
    engine.add_bid(Bid("Agent-C", "task", ["GPU-1"], 70.0))
    result = engine.run_auction()

    # Only one agent can win GPU-1
    assert len(result["winners"]) == 1


def test_bundle_bid_wins_over_individual():
    """
    Agent A bids $80 for {GPU-1 + API-100} together.
    Agents B and C bid $30 each for GPU-1 and API-100 separately.
    B+C combined = $60 < A's $80, so A should win.
    """
    engine = make_engine()
    engine.add_bid(Bid("Bundle-Agent", "needs both",    ["GPU-1", "API-100"], 80.0))
    engine.add_bid(Bid("Agent-B",      "needs gpu",     ["GPU-1"],            30.0))
    engine.add_bid(Bid("Agent-C",      "needs api",     ["API-100"],          30.0))
    result = engine.run_auction()
    # Greedy selects by value-per-resource:
    # Bundle-Agent: 80/2 = 40 per resource (highest) → wins first
    assert "Bundle-Agent" in result["winners"]


def test_two_non_conflicting_bids_both_win():
    """
    Agent A wants GPU-1, Agent B wants GPU-2.
    No conflict → both should win.
    """
    engine = make_engine()
    engine.add_bid(Bid("Agent-A", "task on gpu1", ["GPU-1"], 30.0))
    engine.add_bid(Bid("Agent-B", "task on gpu2", ["GPU-2"], 25.0))
    result = engine.run_auction()
    assert "Agent-A" in result["winners"]
    assert "Agent-B" in result["winners"]
    assert result["total_value"] == 55.0


def test_critical_task_beats_low_priority():
    """
    The key Compute Bazaar promise:
    A critical task with high bid beats a low-priority task,
    even if the low-priority task arrived first.
    """
    engine = make_engine()
    # Low-priority arrives "first" (added first)
    engine.add_bid(Bid("Low-Priority",  "resize image",          ["GPU-1"], bid_amount=5.0,  priority="low"))
    engine.add_bid(Bid("Critical-Task", "process payment NOW!",  ["GPU-1"], bid_amount=95.0, priority="critical"))
    result = engine.run_auction()
    assert "Critical-Task" in result["winners"], (
        "FAILURE: Round-robin bug! Critical task should beat low-priority!"
    )
    assert "Low-Priority" not in result["winners"]


# ─────────────────────────────────────────────────────────────
# TEST GROUP 3: Auction Efficiency vs. Round-Robin
# ─────────────────────────────────────────────────────────────

def test_auction_beats_round_robin():
    """
    The auction should extract MORE value than round-robin scheduling.
    If this fails, our sorting/greedy logic is broken.
    """
    engine = make_engine()
    # Deliberately put low-value bid first (round-robin would pick it)
    engine.add_bid(Bid("Early-Low-Value",  "low value task", ["GPU-1"], 10.0))
    engine.add_bid(Bid("Later-High-Value", "high value task", ["GPU-1"], 90.0))
    result = engine.run_auction()

    assert result["total_value"] >= result["round_robin_value"], (
        "Auction total value should be >= round-robin value"
    )


def test_total_value_is_sum_of_winners():
    """total_value in result should equal sum of all winning bid amounts."""
    engine = make_engine()
    engine.add_bid(Bid("Agent-A", "task a", ["GPU-1"],   40.0))
    engine.add_bid(Bid("Agent-B", "task b", ["GPU-2"],   35.0))
    engine.add_bid(Bid("Agent-C", "task c", ["API-100"], 50.0))
    result = engine.run_auction()
    # All 3 are non-conflicting → all should win
    assert result["total_value"] == 125.0


# ─────────────────────────────────────────────────────────────
# TEST GROUP 4: Payment Calculation
# ─────────────────────────────────────────────────────────────

def test_payment_less_than_or_equal_to_bid():
    """Winners should pay ≤ their bid amount (they never overpay)."""
    engine = make_engine()
    bid = Bid("Agent-A", "task", ["GPU-1"], 60.0)
    payment = engine._calculate_payment(bid, [bid])
    assert payment <= 60.0, "Payment must not exceed bid amount"
    assert payment > 0, "Payment must be positive"


# ─────────────────────────────────────────────────────────────
# TEST GROUP 5: State Management
# ─────────────────────────────────────────────────────────────

def test_reset_clears_bids():
    """After reset(), the bids list should be empty for the next round."""
    engine = make_engine()
    engine.add_bid(Bid("Agent-A", "task", ["GPU-1"], 30.0))
    assert len(engine.bids) == 1
    engine.reset()
    assert len(engine.bids) == 0


def test_round_number_increments():
    """Each call to run_auction() should increment the round number."""
    engine = make_engine()
    engine.add_bid(Bid("Agent-A", "task", ["GPU-1"], 10.0))
    engine.run_auction()
    engine.reset()
    engine.add_bid(Bid("Agent-B", "task", ["GPU-2"], 10.0))
    engine.run_auction()
    assert engine.round_number == 2


def test_empty_auction_runs_cleanly():
    """An auction with no bids should run without crashing."""
    engine = make_engine()
    result = engine.run_auction()
    assert result["winners"] == []
    assert result["total_value"] == 0.0
