"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Week 4 Tests: Intelligent Bidding
=============================================================
 Tests the connection of Groq-powered AI agents to the EXISTING
 Compute Bazaar auction engine.

 All tests run in MOCK mode (no API key required).

 Validates:
   - Intelligent bid amount calculation based on Groq decision metrics
     (resource requirements, budget, priority, utility, WTP)
   - Shading factor overrides and priority weighting
   - Single-resource vs Combinatorial bundle bid generation
   - Converter function to_engine_bid()
   - End-to-end integration: AI agents generate intelligent bids,
     submit them to EXISTING AuctionEngine, run ILP WDP solver,
     and calculate VCG pricing.
   - Multi-agent competition scenario (Coding, Research, Data, Analysis)

 Run with:  python -m pytest tests/test_week4_bidding.py -v
=============================================================
"""

import pytest
from typing import List

from compute_bazaar.agents import (
    BaseAgent,
    TaskDecision,
    CodingAgent,
    ResearchAgent,
    DataAgent,
    AnalysisAgent,
    to_engine_bid,
)
from compute_bazaar.models.bundle import CombinatorialBid, PriorityLevel
from compute_bazaar.llm import GroqLLMClient, LLMConfig
from backend.app.auction.engine import (
    AuctionEngine,
    Resource as EngineResource,
    Bid as EngineBid,
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


@pytest.fixture
def coding_agent(mock_llm_client):
    return CodingAgent("DevBot", budget=500.0, llm_client=mock_llm_client)


@pytest.fixture
def research_agent(mock_llm_client):
    return ResearchAgent("ScholarBot", budget=400.0, llm_client=mock_llm_client)


@pytest.fixture
def data_agent(mock_llm_client):
    return DataAgent("ETLBot", budget=450.0, llm_client=mock_llm_client)


@pytest.fixture
def analysis_agent(mock_llm_client):
    return AnalysisAgent("QuantBot", budget=600.0, llm_client=mock_llm_client)


# ─────────────────────────────────────────────────────────────
# GROUP 1: Intelligent Bid Calculation
# ─────────────────────────────────────────────────────────────

def test_priority_based_shading(coding_agent):
    """Critical priority tasks should bid higher fraction of WTP than normal/low priority."""
    d_critical = TaskDecision(
        agent_name="DevBot", agent_type="coding", task_description="Emergency fix",
        understood_task="Critical fix", required_resources=["GPU-Slot-1"],
        complexity=9, estimated_utility=100.0, max_willingness_to_pay=100.0,
        priority="critical", reasoning=["Critical priority"], confidence=0.9,
    )
    d_normal = TaskDecision(
        agent_name="DevBot", agent_type="coding", task_description="Routine update",
        understood_task="Routine task", required_resources=["GPU-Slot-1"],
        complexity=4, estimated_utility=100.0, max_willingness_to_pay=100.0,
        priority="normal", reasoning=["Normal priority"], confidence=0.8,
    )
    d_low = TaskDecision(
        agent_name="DevBot", agent_type="coding", task_description="Background task",
        understood_task="Low task", required_resources=["GPU-Slot-1"],
        complexity=2, estimated_utility=100.0, max_willingness_to_pay=100.0,
        priority="low", reasoning=["Low priority"], confidence=0.7,
    )

    bid_critical = coding_agent.compute_intelligent_bid_amount(d_critical)
    bid_normal = coding_agent.compute_intelligent_bid_amount(d_normal)
    bid_low = coding_agent.compute_intelligent_bid_amount(d_low)

    assert bid_critical > bid_normal > bid_low
    assert bid_critical == pytest.approx(98.0)
    assert bid_normal == pytest.approx(78.0)
    assert bid_low == pytest.approx(55.0)


def test_bid_amount_budget_and_utility_capping(mock_llm_client):
    """Bid amount must never exceed budget or estimated utility."""
    broke_agent = CodingAgent("BrokeBot", budget=40.0, llm_client=mock_llm_client)
    decision = TaskDecision(
        agent_name="BrokeBot", agent_type="coding", task_description="Huge job",
        understood_task="Huge job", required_resources=["GPU-Slot-1"],
        complexity=8, estimated_utility=300.0, max_willingness_to_pay=250.0,
        priority="critical", reasoning=["High WTP but low budget"], confidence=0.9,
    )

    bid_amount = broke_agent.compute_intelligent_bid_amount(decision)
    assert bid_amount <= 40.0  # Capped by budget


def test_custom_shading_factor_override(coding_agent):
    """Passing explicit shading_factor should override default priority multiplier."""
    decision = TaskDecision(
        agent_name="DevBot", agent_type="coding", task_description="Task",
        understood_task="Task", required_resources=["GPU-Slot-1"],
        complexity=5, estimated_utility=200.0, max_willingness_to_pay=100.0,
        priority="normal", reasoning=["Shading override"], confidence=0.8,
    )

    bid_amount = coding_agent.compute_intelligent_bid_amount(decision, shading_factor=0.50)
    assert bid_amount == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────
# GROUP 2: Single-Resource & Bundle Bid Generation
# ─────────────────────────────────────────────────────────────

def test_create_combinatorial_bundle_bid(coding_agent):
    """create_bid(bid_type='bundle') should return a single CombinatorialBid with full bundle."""
    task = "Build and train neural model"
    bid = coding_agent.create_bid(task, bid_type="bundle")

    assert isinstance(bid, CombinatorialBid)
    assert bid.agent_name == "DevBot"
    assert len(bid.resource_bundle) >= 1
    assert bid.bid_amount > 0.0
    assert bid.task_description == task
    assert len(coding_agent.submitted_bids) == 1


def test_create_single_resource_bids(coding_agent):
    """create_bid(bid_type='single') should return a list of per-resource bids."""
    decision = TaskDecision(
        agent_name="DevBot", agent_type="coding", task_description="Multi-resource task",
        understood_task="Multi task", required_resources=["GPU-Slot-1", "CPU-Cores-8"],
        complexity=6, estimated_utility=100.0, max_willingness_to_pay=80.0,
        priority="normal", reasoning=["Two resources needed"], confidence=0.8,
    )

    bids = coding_agent.create_bid("Multi-resource task", decision=decision, bid_type="single")

    assert isinstance(bids, list)
    assert len(bids) == 2
    assert bids[0].resource_bundle == ["GPU-Slot-1"]
    assert bids[1].resource_bundle == ["CPU-Cores-8"]
    # Total sum of per-resource bids equals full intelligent bid amount
    total_split = sum(b.bid_amount for b in bids)
    expected_total = coding_agent.compute_intelligent_bid_amount(decision)
    assert total_split == pytest.approx(expected_total, abs=0.02)


# ─────────────────────────────────────────────────────────────
# GROUP 3: Conversion to EXISTING Auction Engine
# ─────────────────────────────────────────────────────────────

def test_to_engine_bid_converter():
    """to_engine_bid should convert CombinatorialBid to backend EngineBid."""
    cb = CombinatorialBid(
        bid_id="cb-101",
        agent_name="TestAgent",
        resource_bundle=["GPU-Slot-1", "RAM-Block-32GB"],
        bid_amount=125.50,
        task_description="Test conversion",
        priority="high",
    )

    eb = to_engine_bid(cb)

    assert isinstance(eb, EngineBid)
    assert eb.bid_id == "cb-101"
    assert eb.agent_id == "TestAgent"
    assert eb.resource_bundle == ["GPU-Slot-1", "RAM-Block-32GB"]
    assert eb.bid_amount == 125.50
    assert eb.task_description == "Test conversion"
    assert eb.priority == "high"


def test_submit_to_engine_single_and_multiple(coding_agent, auction_engine):
    """submit_to_engine should connect agent bids directly to AuctionEngine."""
    bid1 = coding_agent.create_bid("Task 1", bid_type="bundle")
    results1 = coding_agent.submit_to_engine(auction_engine, bid1)

    assert results1 == [True]
    assert len(auction_engine.get_bids()) == 1

    bids2 = coding_agent.create_bid("Task 2", bid_type="single")
    results2 = coding_agent.submit_to_engine(auction_engine, bids2)

    assert all(results2)
    assert len(auction_engine.get_bids()) == 1 + len(bids2)


# ─────────────────────────────────────────────────────────────
# GROUP 4: End-to-End Auction Integration (Multi-Agent)
# ─────────────────────────────────────────────────────────────

def test_end_to_end_intelligent_auction(
    coding_agent, research_agent, data_agent, analysis_agent, auction_engine
):
    """
    End-to-End Test:
    4 AI Agents analyze distinct domain tasks via Groq (MOCK mode),
    generate intelligent bids, submit to EXISTING AuctionEngine,
    run ILP WDP solver, and verify VCG pricing.
    """
    # 1. Agents generate bids for their tasks
    bid_code = coding_agent.create_bid("Develop LLM fine-tuning pipeline")
    bid_research = research_agent.create_bid("Gather multi-agent market research")
    bid_data = data_agent.create_bid("Clean 50GB dataset and build embeddings")
    bid_analysis = analysis_agent.create_bid("Run Monte Carlo risk modeling")

    # 2. Connect generated bids to EXISTING AuctionEngine
    coding_agent.submit_to_engine(auction_engine, bid_code)
    research_agent.submit_to_engine(auction_engine, bid_research)
    data_agent.submit_to_engine(auction_engine, bid_data)
    analysis_agent.submit_to_engine(auction_engine, bid_analysis)

    assert len(auction_engine.get_bids()) == 4

    # 3. Run Auction via EXISTING ILP WDP Solver & VCG Pricing
    result = auction_engine.run_auction(round_id=1)

    # 4. Verify auction outcomes
    assert result.round_id == 1
    assert len(result.winning_bids) > 0
    assert result.total_value > 0.0

    # Verify core correctness: No resource double-allocated
    allocated: List[str] = []
    for w in result.winning_bids:
        for r in w.resource_bundle:
            assert r not in allocated, f"Resource {r} was double-allocated!"
            allocated.append(r)

    # Verify VCG payments: Winners pay <= their bid amount
    for w in result.winning_bids:
        payment = result.payments[w.bid_id]
        assert payment <= w.bid_amount, f"Winner {w.agent_id} overpaid! Payment ${payment} > Bid ${w.bid_amount}"
        assert payment >= 0.0, f"Winner {w.agent_id} had negative payment!"


def test_mixed_single_and_bundle_bids_auction(
    coding_agent, research_agent, auction_engine
):
    """
    Auction should seamlessly process a mix of bundle bids and single-resource bids.
    """
    bundle_bid = coding_agent.create_bid("Heavy AI job", bid_type="bundle")
    single_bids = research_agent.create_bid("Distributed search", bid_type="single")

    coding_agent.submit_to_engine(auction_engine, bundle_bid)
    research_agent.submit_to_engine(auction_engine, single_bids)

    result = auction_engine.run_auction()

    assert result.total_value >= 0.0
    assert len(result.winning_bids) <= len(auction_engine.get_bids())
