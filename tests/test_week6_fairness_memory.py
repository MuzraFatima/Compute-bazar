"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Week 6 Tests: Fairness & Agent Memory
=============================================================
 Tests the cognitive agent memory, fairness metrics, starvation
 detection, anti-monopoly domination protection, and adaptive bidding.

 All tests run in MOCK mode (no Groq API key required).

 Validates:
   - AgentMemory bid & outcome tracking, win rate, and clearing prices
   - Jain's fairness index computation for surplus and wins
   - Starvation detection and risk score calculation
   - Priority escalation and adaptive bidding under starvation / high win rate
   - Multi-round simulation with memory feedback and fairness tracking
   - Preserves 100% compatibility with existing AuctionEngine and VCG pricing

 Run with:  python -m pytest tests/test_week6_fairness_memory.py -v
=============================================================
"""

import pytest
from typing import List, Dict

from compute_bazaar.agents import (
    BaseAgent,
    CodingAgent,
    ResearchAgent,
    DataAgent,
    AnalysisAgent,
    AgentMemory,
    BidOutcome,
)
from compute_bazaar.fairness import (
    jains_fairness_index,
    calculate_starvation_score,
    FairnessTracker,
)
from compute_bazaar.llm import GroqLLMClient, LLMConfig
from compute_bazaar.models.bundle import CombinatorialBid
from backend.app.auction.engine import (
    AuctionEngine,
    Resource as EngineResource,
    SolverMethod,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    """Mock Groq client for agent testing."""
    return GroqLLMClient(config=LLMConfig(mock_mode=True))


@pytest.fixture
def memory():
    """Fresh AgentMemory instance."""
    return AgentMemory("TestAgent")


@pytest.fixture
def fairness_tracker():
    """Fresh FairnessTracker instance."""
    return FairnessTracker()


@pytest.fixture
def auction_engine():
    """Standard AuctionEngine instance loaded with 5 compute resources."""
    resources = [
        EngineResource("GPU-Slot-1",     "gpu",        1.0),
        EngineResource("GPU-Slot-2",     "gpu",        1.0),
        EngineResource("API-Budget-100", "api_budget", 100.0),
        EngineResource("RAM-Block-32GB", "ram",        32.0),
        EngineResource("CPU-Cores-8",    "cpu",        8.0),
    ]
    return AuctionEngine(resources, method=SolverMethod.LP)


# ─────────────────────────────────────────────────────────────
# GROUP 1: AgentMemory Cognitive Operations
# ─────────────────────────────────────────────────────────────

def test_agent_memory_record_outcome(memory):
    """AgentMemory should record round outcomes and update win/loss metrics."""
    assert memory.total_wins == 0
    assert memory.total_losses == 0

    # Win round 1
    memory.record_outcome("bid-1", round_id=1, won=True, bid_amount=100.0, payment_amount=80.0, resources_allocated=["GPU-Slot-1"])
    assert memory.total_wins == 1
    assert memory.consecutive_wins == 1
    assert memory.consecutive_losses == 0
    assert memory.total_spent == 80.0
    assert memory.total_surplus == 20.0
    assert memory.get_average_clearing_price("GPU-Slot-1") == 80.0

    # Loss round 2
    memory.record_outcome("bid-2", round_id=2, won=False, bid_amount=90.0)
    assert memory.total_losses == 1
    assert memory.consecutive_losses == 1
    assert memory.consecutive_wins == 0


def test_agent_memory_starvation_and_win_rate(memory):
    """AgentMemory should detect starvation after consecutive losses."""
    assert memory.is_starved(threshold=3) is False

    # Simulate 3 consecutive losses
    for i in range(3):
        memory.record_outcome(f"bid-loss-{i}", round_id=i+1, won=False, bid_amount=50.0)

    assert memory.is_starved(threshold=3) is True
    assert memory.get_win_rate(10) == 0.0


def test_agent_memory_clearing_price_history(memory):
    """Memory should maintain running average clearing prices per resource."""
    memory.record_outcome("b1", round_id=1, won=True, bid_amount=100.0, payment_amount=60.0, resources_allocated=["GPU-Slot-1"])
    memory.record_outcome("b2", round_id=2, won=True, bid_amount=100.0, payment_amount=80.0, resources_allocated=["GPU-Slot-1"])

    avg_price = memory.get_average_clearing_price("GPU-Slot-1")
    assert avg_price == pytest.approx(70.0)  # (60 + 80) / 2


# ─────────────────────────────────────────────────────────────
# GROUP 2: Fairness Metrics & FairnessTracker
# ─────────────────────────────────────────────────────────────

def test_jains_fairness_index_equal_distribution():
    """Jain's index for equal values should be 1.0 (perfectly fair)."""
    assert jains_fairness_index([10.0, 10.0, 10.0, 10.0]) == pytest.approx(1.0)


def test_jains_fairness_index_skewed_distribution():
    """Jain's index for highly skewed values should be low."""
    skewed = [100.0, 0.0, 0.0, 0.0]
    # (100)^2 / (4 * 10000) = 10000 / 40000 = 0.25 (1/n)
    assert jains_fairness_index(skewed) == pytest.approx(0.25)


def test_starvation_risk_score():
    """calculate_starvation_score should scale with losses and low win rates."""
    healthy = calculate_starvation_score(consecutive_losses=0, win_rate=0.80)
    starved = calculate_starvation_score(consecutive_losses=4, win_rate=0.0)

    assert healthy < starved
    assert starved > 0.70


def test_fairness_tracker_recording(fairness_tracker):
    """FairnessTracker should record outcomes and compute market equity."""
    fairness_tracker.record_round_outcome(1, {
        "Agent-A": {"won": True, "surplus": 30.0, "payment": 70.0},
        "Agent-B": {"won": True, "surplus": 25.0, "payment": 50.0},
        "Agent-C": {"won": False, "surplus": 0.0, "payment": 0.0},
    })

    assert len(fairness_tracker.agent_stats) == 3
    assert fairness_tracker.compute_jains_fairness_surplus() < 1.0
    assert fairness_tracker.compute_jains_fairness_wins() < 1.0


def test_fairness_tracker_detect_starvation_and_boost(fairness_tracker):
    """FairnessTracker should identify starved agents and suggest priority boosts."""
    # Simulate Agent-C losing 3 rounds in a row
    for r in range(1, 4):
        fairness_tracker.record_round_outcome(r, {
            "Agent-A": {"won": True, "surplus": 20.0, "payment": 80.0},
            "Agent-C": {"won": False, "surplus": 0.0, "payment": 0.0},
        })

    starved = fairness_tracker.detect_starved_agents(threshold=3)
    assert "Agent-C" in starved
    assert "Agent-A" not in starved

    rec = fairness_tracker.get_priority_boost_recommendation("Agent-C")
    assert rec in ("high", "critical")


# ─────────────────────────────────────────────────────────────
# GROUP 3: Adaptive Bidding via Memory
# ─────────────────────────────────────────────────────────────

def test_adaptive_bidding_under_starvation(mock_llm_client):
    """Starved agent should automatically boost shading multiplier and priority."""
    agent = CodingAgent("StarvedBot", budget=500.0, llm_client=mock_llm_client)

    # Normal state bid
    task = "Refactor module"
    bid_normal = agent.create_bid(task)
    amount_normal = bid_normal.bid_amount

    # Force 3 consecutive losses into memory
    for i in range(3):
        agent.memory.record_outcome(f"b-{i}", round_id=i+1, won=False, bid_amount=50.0)

    assert agent.memory.is_starved(threshold=3) is True

    # Adaptive state bid under starvation
    bid_adaptive = agent.create_bid(task)
    amount_adaptive = bid_adaptive.bid_amount

    assert amount_adaptive > amount_normal
    assert bid_adaptive.priority in ("high", "critical")


def test_adaptive_bidding_under_high_win_rate(mock_llm_client):
    """Agent with >70% win rate should shade bids to maximize surplus margin."""
    agent = CodingAgent("DominantBot", budget=500.0, llm_client=mock_llm_client)

    # Force 8 wins out of 10 in memory
    for i in range(10):
        won = i < 8
        agent.memory.record_outcome(f"b-{i}", round_id=i+1, won=won, bid_amount=100.0, payment_amount=70.0 if won else 0.0)

    assert agent.memory.get_win_rate(10) == 0.80

    decision = agent.analyze_task("Routine task")
    bid_amount = agent.compute_intelligent_bid_amount(decision)

    # Multiplier shaded down to 0.72
    assert bid_amount == pytest.approx(decision.max_willingness_to_pay * 0.72, abs=0.05)


def test_clearing_price_alignment(mock_llm_client):
    """Agent should adjust bid amount above average historical clearing price."""
    agent = CodingAgent("SmartBot", budget=500.0, llm_client=mock_llm_client)

    # Record historical high clearing price for GPU-Slot-1
    for i in range(3):
        agent.memory.record_outcome(f"b-{i}", round_id=i+1, won=True, bid_amount=120.0, payment_amount=90.0, resources_allocated=["GPU-Slot-1"])

    decision = agent.analyze_task("GPU task")
    decision.required_resources = ["GPU-Slot-1"]
    decision.max_willingness_to_pay = 150.0
    decision.estimated_utility = 150.0

    bid_amount = agent.compute_intelligent_bid_amount(decision)
    # Expected: max(raw_bid, avg_clearing * 1.05) = max(~117, 90 * 1.05 = 94.5)
    assert bid_amount >= 94.5


# ─────────────────────────────────────────────────────────────
# GROUP 4: Multi-Round Simulation & End-to-End Integration
# ─────────────────────────────────────────────────────────────

def test_multi_round_fairness_and_memory_simulation(
    mock_llm_client, auction_engine
):
    """
    Multi-round Simulation (5 Rounds):
    4 AI Agents (Coding, Research, Data, Analysis) compete across rounds.
    Outcomes are recorded into agent memory and FairnessTracker.
    Verifies Jain's fairness index tracking, starvation handling,
    and adaptive bid adjustments without breaking auction or VCG.
    """
    agents = [
        CodingAgent("CodeBot", budget=500.0, llm_client=mock_llm_client),
        ResearchAgent("ResearchBot", budget=400.0, llm_client=mock_llm_client),
        DataAgent("DataBot", budget=450.0, llm_client=mock_llm_client),
        AnalysisAgent("AnalysisBot", budget=600.0, llm_client=mock_llm_client),
    ]

    tracker = FairnessTracker()
    tasks = [
        "Develop web service backend",
        "Perform market intelligence report",
        "Clean large dataset and extract vectors",
        "Run Monte Carlo statistical risk analysis",
    ]

    for round_id in range(1, 6):
        auction_engine.reset()
        round_outcomes = {}

        # 1. Agents generate adaptive bids
        for i, agent in enumerate(agents):
            bid = agent.create_bid(tasks[i])
            agent.submit_to_engine(auction_engine, bid)

        # 2. Execute auction via existing ILP WDP solver & VCG pricing
        result = auction_engine.run_auction(round_id=round_id)

        # 3. Record outcomes into agent memory and FairnessTracker
        winner_ids = {w.bid_id for w in result.winning_bids}
        for agent in agents:
            # Find agent's bid in this round
            for b in agent.submitted_bids:
                if any(eb.bid_id == b.bid_id for eb in auction_engine.get_bids()):
                    won = b.bid_id in winner_ids
                    payment = result.payments.get(b.bid_id, 0.0) if won else 0.0
                    resources = next((w.resource_bundle for w in result.winning_bids if w.bid_id == b.bid_id), [])

                    agent.record_auction_outcome(
                        bid_id=b.bid_id,
                        round_id=round_id,
                        won=won,
                        bid_amount=b.bid_amount,
                        payment_amount=payment,
                        resources_allocated=resources,
                    )

                    round_outcomes[agent.name] = {
                        "won": won,
                        "surplus": max(0.0, b.bid_amount - payment) if won else 0.0,
                        "payment": payment,
                    }

        tracker.record_round_outcome(round_id, round_outcomes)

    # Verify simulation metrics
    report = tracker.get_fairness_report()
    assert report["total_registered_agents"] == 4
    assert 0.0 <= report["jains_fairness_surplus"] <= 1.0
    assert 0.0 <= report["jains_fairness_wins"] <= 1.0

    # Ensure all agents have memory history
    for agent in agents:
        summary = agent.memory.get_summary()
        assert summary["total_rounds_participated"] == 5
        assert summary["total_wins"] + summary["total_losses"] == 5
