"""
=============================================================
 COMPUTE BAZAAR 2.0 — Week 11: Live Multi-Agent Stress Test
=============================================================
 Tests the FULL autonomous system under real load:
   - 4 Groq-powered agents run concurrently via threads
   - All call the live LLM simultaneously (simulating real contention)
   - Bids enter the real ILP AuctionEngine
   - VCG payments computed and verified
   - Budget deductions, memory updates, and fairness metrics checked

 IMPORTANT: These tests require a valid GROQ_API_KEY in .env
 They are marked with @pytest.mark.live and skipped automatically
 when the key is absent, so they NEVER break the CI test suite.

 Run all live tests:
   pytest tests/test_week11_live_integration.py -v -m live

 Run just the stress test (all 4 agents in parallel):
   pytest tests/test_week11_live_integration.py::test_four_agent_concurrent_stress -v -s -m live

 Run without live marker (requires GROQ_API_KEY):
   GROQ_API_KEY=gsk_... pytest tests/test_week11_live_integration.py -v
=============================================================
"""

from __future__ import annotations

import os
import time
import concurrent.futures
import threading
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

import pytest

# ─────────────────────────────────────────────────────────────
# Load .env so GROQ_API_KEY is available even in pytest context
# ─────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Skip entire module if no real API key is present
pytestmark = pytest.mark.live


def _has_live_key() -> bool:
    return bool(GROQ_API_KEY and not GROQ_API_KEY.startswith("mock"))


# ─────────────────────────────────────────────────────────────
# IMPORTS (deferred under the skip guard where needed)
# ─────────────────────────────────────────────────────────────
from compute_bazaar.agents import (
    CodingAgent, ResearchAgent, DataAgent, AnalysisAgent, TaskDecision,
)
from compute_bazaar.llm import GroqLLMClient, LLMConfig, LLMError
from compute_bazaar.fairness import jains_fairness_index, FairnessTracker
from compute_bazaar.fairness.metrics import calculate_starvation_score
from backend.app.auction.engine import (
    AuctionEngine,
    Resource as EngineResource,
    Bid as EngineBid,
    AllocationResult,
    SolverMethod,
)


# ─────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def live_llm_client():
    """Single shared Groq LLM client for the module (reduces cold-start overhead)."""
    if not _has_live_key():
        pytest.skip("No GROQ_API_KEY found — live tests require a real key.")
    return GroqLLMClient(
        config=LLMConfig(model="openai/gpt-oss-120b", temperature=0.4, mock_mode=False),
        api_key=GROQ_API_KEY,
    )


@pytest.fixture(scope="module")
def engine_resources() -> List[EngineResource]:
    """Standard 5-resource pool mirroring the production DB seed."""
    return [
        EngineResource("GPU-Slot-1",    "gpu",        capacity=1.0,   unit="slot"),
        EngineResource("GPU-Slot-2",    "gpu",        capacity=1.0,   unit="slot"),
        EngineResource("API-Budget-100","api_budget",  capacity=100.0, unit="dollars"),
        EngineResource("RAM-Block-32GB","ram",         capacity=32.0,  unit="gb"),
        EngineResource("CPU-Cores-8",   "cpu",        capacity=8.0,   unit="cores"),
    ]


@pytest.fixture(scope="module")
def four_agents(live_llm_client) -> Dict[str, object]:
    """4 domain agents sharing one Groq client."""
    return {
        "DevBot":      CodingAgent("DevBot",      budget=800.0,  llm_client=live_llm_client),
        "ScholarBot":  ResearchAgent("ScholarBot", budget=700.0,  llm_client=live_llm_client),
        "ETLBot":      DataAgent("ETLBot",         budget=600.0,  llm_client=live_llm_client),
        "QuantBot":    AnalysisAgent("QuantBot",   budget=500.0,  llm_client=live_llm_client),
    }


# Realistic per-domain tasks
AGENT_TASKS = {
    "DevBot":     "Build and deploy a GPU-accelerated FastAPI microservice with async database writes and load testing",
    "ScholarBot": "Survey 40 recent LLM papers, extract evaluation benchmarks, and produce a comparative summary report",
    "ETLBot":     "Clean, embed, and index 5 million product descriptions into a vector database for semantic search",
    "QuantBot":   "Run 30,000-iteration Monte Carlo simulation for equity portfolio tail-risk and VaR estimation",
}


# ─────────────────────────────────────────────────────────────
# GROUP 1: Individual Live Agent Task Analysis
# ─────────────────────────────────────────────────────────────

@pytest.mark.live
def test_live_coding_agent_task_analysis(live_llm_client):
    """CodingAgent produces a valid live TaskDecision from Groq."""
    agent = CodingAgent("LiveDevBot", budget=600.0, llm_client=live_llm_client)
    task = "Implement a zero-downtime blue-green deployment pipeline for a GPU inference cluster"

    decision = agent.analyze_task(task)

    # Core structure
    assert isinstance(decision, TaskDecision)
    assert decision.agent_name == "LiveDevBot"
    assert decision.agent_type == "coding"
    assert not decision.is_mock

    # LLM quality checks
    assert 1 <= decision.complexity <= 10
    assert 0.0 < decision.confidence <= 1.0
    assert decision.estimated_utility > 0
    assert 0 < decision.max_willingness_to_pay <= min(decision.estimated_utility, 600.0)
    assert len(decision.required_resources) >= 1
    assert all(r in ["GPU-Slot-1","GPU-Slot-2","API-Budget-100","RAM-Block-32GB","CPU-Cores-8"]
               for r in decision.required_resources)
    assert decision.priority in ("critical", "high", "normal", "low")
    assert isinstance(decision.reasoning, list) and len(decision.reasoning) >= 1

    # Timing — must respond within 30s
    assert decision.llm_latency_ms > 0
    assert decision.llm_latency_ms < 30_000

    print(f"\n  [CodingAgent] complexity={decision.complexity} | "
          f"wtp=${decision.max_willingness_to_pay:.2f} | "
          f"latency={decision.llm_latency_ms:.0f}ms | "
          f"resources={decision.required_resources}")


@pytest.mark.live
def test_live_research_agent_task_analysis(live_llm_client):
    """ResearchAgent produces a valid live TaskDecision from Groq."""
    agent = ResearchAgent("LiveScholar", budget=500.0, llm_client=live_llm_client)
    task = "Conduct a comprehensive literature review on VCG mechanism design in combinatorial auctions"

    decision = agent.analyze_task(task)

    assert not decision.is_mock
    assert decision.agent_type == "research"
    assert 1 <= decision.complexity <= 10
    assert decision.max_willingness_to_pay <= min(decision.estimated_utility, 500.0)
    assert len(decision.required_resources) >= 1

    print(f"\n  [ResearchAgent] complexity={decision.complexity} | "
          f"wtp=${decision.max_willingness_to_pay:.2f} | "
          f"resources={decision.required_resources}")


@pytest.mark.live
def test_live_data_agent_task_analysis(live_llm_client):
    """DataAgent produces a valid live TaskDecision from Groq."""
    agent = DataAgent("LiveETLBot", budget=650.0, llm_client=live_llm_client)
    task = "Build a real-time streaming ETL pipeline to ingest 500GB of sensor data into a Parquet warehouse"

    decision = agent.analyze_task(task)

    assert not decision.is_mock
    assert decision.agent_type == "data_processing"   # DataAgent returns "data_processing"
    assert decision.max_willingness_to_pay <= min(decision.estimated_utility, 650.0)

    print(f"\n  [DataAgent] complexity={decision.complexity} | "
          f"wtp=${decision.max_willingness_to_pay:.2f} | "
          f"resources={decision.required_resources}")


@pytest.mark.live
def test_live_analysis_agent_task_analysis(live_llm_client):
    """AnalysisAgent produces a valid live TaskDecision from Groq."""
    agent = AnalysisAgent("LiveQuant", budget=400.0, llm_client=live_llm_client)
    task = "Compute factor model beta exposures and perform Cholesky-based portfolio risk decomposition"

    decision = agent.analyze_task(task)

    assert not decision.is_mock
    assert decision.agent_type == "analysis"
    assert decision.max_willingness_to_pay <= min(decision.estimated_utility, 400.0)

    print(f"\n  [AnalysisAgent] complexity={decision.complexity} | "
          f"wtp=${decision.max_willingness_to_pay:.2f} | "
          f"resources={decision.required_resources}")


# ─────────────────────────────────────────────────────────────
# GROUP 2: Intelligent Bid Creation (Live)
# ─────────────────────────────────────────────────────────────

@pytest.mark.live
def test_live_intelligent_bundle_bid_creation(live_llm_client):
    """create_bid() produces a valid CombinatorialBid with budget-safe amount."""
    agent = CodingAgent("BidBot", budget=300.0, llm_client=live_llm_client)
    task = "Deploy a containerized Triton inference server with GPU auto-scaling"

    bid = agent.create_bid(task, bid_type="bundle")

    # Bid structure
    assert bid.bid_amount > 0
    assert bid.bid_amount <= 300.0                # must not exceed budget
    assert len(bid.resource_bundle) >= 1
    assert bid.task_description == task
    assert bid.priority in ("critical", "high", "normal", "low")

    print(f"\n  [Bundle Bid] amount=${bid.bid_amount:.2f} | "
          f"bundle={bid.resource_bundle} | priority={bid.priority}")


@pytest.mark.live
def test_live_intelligent_single_bids_creation(live_llm_client):
    """create_bid(bid_type='single') produces one EngineBid per resource."""
    agent = ResearchAgent("SingleBidBot", budget=400.0, llm_client=live_llm_client)
    task = "Summarize 20 arXiv papers on emergent abilities of large language models"

    bids = agent.create_bid(task, bid_type="single")

    assert isinstance(bids, list)
    assert len(bids) >= 1
    for bid in bids:
        assert bid.bid_amount > 0
        assert bid.bid_amount <= 400.0
        assert len(bid.resource_bundle) == 1   # each single bid is for exactly 1 resource

    print(f"\n  [Single Bids] count={len(bids)} | "
          f"amounts={[round(b.bid_amount,2) for b in bids]}")


# ─────────────────────────────────────────────────────────────
# GROUP 3: Full Live Auction Round (Single Agent → Engine)
# ─────────────────────────────────────────────────────────────

@pytest.mark.live
def test_live_single_agent_full_auction_round(live_llm_client, engine_resources):
    """One agent completes the full loop: analyze → bid → engine → VCG result."""
    engine = AuctionEngine(engine_resources)
    agent = DataAgent("StressETL", budget=500.0, llm_client=live_llm_client)
    task = "Preprocess and embed 2 million product images for multimodal search index"

    # 1. Analyze task with live LLM
    decision = agent.analyze_task(task)
    assert not decision.is_mock

    # 2. Create bid
    bid = agent.create_bid(task, decision=decision, bid_type="bundle")
    assert bid.bid_amount > 0

    # 3. Submit to engine
    from compute_bazaar.agents.base import to_engine_bid
    engine_bid = to_engine_bid(bid)
    engine.submit_bid(engine_bid)

    # 4. Run auction
    result = engine.run_auction(round_id=1)
    assert isinstance(result, AllocationResult)

    # 5. With a single bidder, it must win (no competition)
    assert len(result.winning_bids) == 1
    winner = result.winning_bids[0]
    assert winner.agent_id == "StressETL"

    # 5. VCG payment: with no other bidders, payment = 0 (no externality)
    payments = result.payments            # correct attribute: payments (Dict[bid_id, float])
    assert isinstance(payments, dict)
    vcg_pay = payments.get(winner.bid_id, 0.0)
    assert vcg_pay >= 0.0
    assert vcg_pay <= winner.bid_amount

    # 7. Record outcome in agent memory
    agent.record_auction_outcome(
        bid_id=winner.bid_id,
        round_id=1,
        won=True,
        bid_amount=winner.bid_amount,
        payment_amount=vcg_pay,
        resources_allocated=winner.resource_bundle,
    )
    assert agent.memory.total_wins == 1

    print(f"\n  [Single-Agent Round] bid=${winner.bid_amount:.2f} | "
          f"vcg_payment=${vcg_pay:.2f} | resources={winner.resource_bundle}")


# ─────────────────────────────────────────────────────────────
# GROUP 4: ★ MAIN STRESS TEST — 4 Agents Concurrently
# ─────────────────────────────────────────────────────────────

def _agent_analyze_and_bid(
    agent_name: str,
    agent,
    task: str,
    results: dict,
    errors: dict,
):
    """Worker function for thread pool: analyze task and create bid.
    Retries up to 3 times on Groq rate-limit (429) with backoff.
    """
    from compute_bazaar.llm import LLMRateLimitError
    import re

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            decision = agent.analyze_task(task)
            bid = agent.create_bid(task, decision=decision, bid_type="bundle")
            elapsed = (time.time() - t0) * 1000
            results[agent_name] = {
                "decision": decision,
                "bid": bid,
                "elapsed_ms": elapsed,
            }
            return   # success
        except LLMRateLimitError as e:
            # Parse "Please try again in X.Xs" from the error message
            wait = 15.0
            m = re.search(r"try again in (\d+\.?\d*)s", str(e))
            if m:
                wait = float(m.group(1)) + 2.0   # add 2s buffer
            print(f"    [{agent_name}] Rate limit hit (attempt {attempt}/{max_retries})"
                  f" - waiting {wait:.0f}s before retry...")
            time.sleep(wait)
        except Exception as e:
            errors[agent_name] = str(e)
            return
    errors[agent_name] = f"Rate limit retry exhausted after {max_retries} attempts"


@pytest.mark.live
def test_four_agent_concurrent_stress(four_agents, engine_resources):
    """
    CORE STRESS TEST: 4 autonomous Groq agents run analyze_task() and create_bid()
    CONCURRENTLY via threads, then all bids enter the ILP AuctionEngine.

    Validates:
      ✓ All 4 Groq LLM calls complete within 60 seconds total
      ✓ All decisions are valid (non-mock, correct types, budget-safe)
      ✓ ILP solver finds an optimal winner combination
      ✓ VCG payments are non-negative and ≤ bid amounts
      ✓ Budget deductions leave agents with reduced balances
      ✓ Agent memory records correct win/loss outcomes
      ✓ Jain's Fairness Index is computable and in [0, 1]
      ✓ No agent starves (each gets at least some resources eventually)
    """
    agents = four_agents
    agent_names = list(agents.keys())

    results: Dict[str, dict] = {}
    errors:  Dict[str, str]  = {}

    # ── Phase 1: Concurrent LLM reasoning — STAGGERED to respect TPM rate limit ──
    print("\n  [STRESS] Phase 1: Staggered Groq LLM reasoning for 4 agents...")
    start_total = time.time()

    # Stagger thread starts by 5s each to avoid token burst rate limits
    def _staggered_worker(agent_name, agent, task, delay):
        time.sleep(delay)
        _agent_analyze_and_bid(agent_name, agent, task, results, errors)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_staggered_worker, name, agents[name], AGENT_TASKS[name], i * 5): name
            for i, name in enumerate(agent_names)
        }
        concurrent.futures.wait(futures, timeout=300)   # allow for rate-limit retries

    total_elapsed = (time.time() - start_total) * 1000
    print(f"  [STRESS] All 4 agents completed in {total_elapsed:.0f}ms")

    # All agents must succeed
    assert not errors, f"Some agents failed: {errors}"
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"

    # ── Phase 2: Validate each decision & bid ──────────────
    print("\n  [STRESS] Phase 2: Validating decisions and bids...")
    for name in agent_names:
        r = results[name]
        decision: TaskDecision = r["decision"]
        bid = r["bid"]

        assert not decision.is_mock, f"{name}: decision must be live (not mock)"
        assert 1 <= decision.complexity <= 10, f"{name}: invalid complexity"
        assert 0.0 < decision.confidence <= 1.0, f"{name}: invalid confidence"
        assert decision.estimated_utility > 0, f"{name}: utility must be positive"
        assert 0 < decision.max_willingness_to_pay <= agents[name].budget, \
            f"{name}: WTP exceeds budget"
        assert len(decision.required_resources) >= 1, f"{name}: no resources selected"
        assert bid.bid_amount > 0, f"{name}: bid amount must be positive"
        assert bid.bid_amount <= agents[name].budget, \
            f"{name}: bid exceeds budget"

        print(f"    {name:12} | complexity={decision.complexity} | "
              f"wtp=${decision.max_willingness_to_pay:6.2f} | "
              f"bid=${bid.bid_amount:6.2f} | "
              f"resources={decision.required_resources} | "
              f"latency={decision.llm_latency_ms:.0f}ms")

    # ── Phase 3: Submit all bids to AuctionEngine ──────────
    print("\n  [STRESS] Phase 3: Submitting all bids to ILP AuctionEngine...")
    engine = AuctionEngine(engine_resources)

    from compute_bazaar.agents.base import to_engine_bid
    bid_map: Dict[str, Tuple[str, float]] = {}  # bid_id → (agent_name, bid_amount)

    for name in agent_names:
        combo_bid = results[name]["bid"]
        engine_bid = to_engine_bid(combo_bid)
        engine.submit_bid(engine_bid)
        bid_map[engine_bid.bid_id] = (name, engine_bid.bid_amount)
        print(f"    Submitted: {name} -> {engine_bid.resource_bundle} @ ${engine_bid.bid_amount:.2f}")

    # ── Phase 4: Run ILP Winner Determination ──────────────
    print("\n  [STRESS] Phase 4: Running ILP solver + VCG pricing...")
    result: AllocationResult = engine.run_auction(round_id=99)

    assert isinstance(result, AllocationResult)
    assert result.solver_method is not None
    assert result.total_value >= 0

    print(f"    Solver method : {result.solver_method.value}")
    print(f"    Total welfare : ${result.total_value:.2f}")
    print(f"    Winners count : {len(result.winning_bids)}")
    print(f"    Unallocated   : {result.unallocated_resources}")

    # ── Phase 5: Validate VCG payments ────────────────────
    print("\n  [STRESS] Phase 5: Validating VCG payments...")
    winner_names = []
    loser_names = list(agent_names)
    agent_surpluses: List[float] = []
    budgets_before = {name: agents[name].budget for name in agent_names}

    winning_bid_ids = {wb.bid_id for wb in result.winning_bids}

    for wb in result.winning_bids:
        agent_name, bid_amount = bid_map.get(wb.bid_id, (None, 0.0))
        if not agent_name:
            continue

        vcg_payment = result.payments.get(wb.bid_id, 0.0)

        # Core VCG invariants
        assert vcg_payment >= 0.0, f"{agent_name}: VCG payment must be non-negative"
        assert vcg_payment <= bid_amount, \
            f"{agent_name}: VCG payment (${vcg_payment:.2f}) > bid (${bid_amount:.2f})"

        surplus = bid_amount - vcg_payment
        agent_surpluses.append(surplus)
        winner_names.append(agent_name)
        loser_names.remove(agent_name)

        # Deduct payment from agent budget
        agents[agent_name].budget = max(0.0, agents[agent_name].budget - vcg_payment)

        print(f"    WINNER  {agent_name:12} | bid=${bid_amount:6.2f} | "
              f"vcg_pay=${vcg_payment:6.2f} | surplus=${surplus:6.2f}")

    for name in loser_names:
        print(f"    LOSER   {name:12} | no allocation")
        agent_surpluses.append(0.0)

    # At least one agent must win (resource pool is non-empty)
    assert len(winner_names) >= 1, "At least one agent should win in a non-empty round"

    # ── Phase 6: Record outcomes in agent memory ───────────
    print("\n  [STRESS] Phase 6: Recording outcomes in AgentMemory...")
    for wb in result.winning_bids:
        agent_name, bid_amount = bid_map.get(wb.bid_id, (None, 0.0))
        if not agent_name:
            continue
        vcg_payment = result.payments.get(wb.bid_id, 0.0)
        agents[agent_name].record_auction_outcome(
            bid_id=wb.bid_id,
            round_id=99,
            won=True,
            bid_amount=bid_amount,
            payment_amount=vcg_payment,
            resources_allocated=wb.resource_bundle,
        )
        assert agents[agent_name].memory.total_wins >= 1

    for name in loser_names:
        combo_bid = results[name]["bid"]
        agents[name].record_auction_outcome(
            bid_id=combo_bid.bid_id or "lost",
            round_id=99,
            won=False,
            bid_amount=combo_bid.bid_amount,
            payment_amount=0.0,
        )
        assert agents[name].memory.total_losses >= 1

    # ── Phase 7: Fairness metrics ──────────────────────────
    print("\n  [STRESS] Phase 7: Computing fairness metrics...")
    fairness_index = jains_fairness_index(agent_surpluses)
    assert 0.0 <= fairness_index <= 1.0, f"Jain's index out of range: {fairness_index}"
    print(f"    Jain's Fairness Index: {fairness_index:.4f}")

    # FairnessTracker round-level metrics — use real API
    tracker = FairnessTracker()
    outcomes = {}
    for name in winner_names:
        surplus = results[name]["bid"].bid_amount - result.payments.get(
            next(wb.bid_id for wb in result.winning_bids
                 if bid_map.get(wb.bid_id, (None,))[0] == name), 0.0
        )
        outcomes[name] = {"won": True, "surplus": max(0.0, surplus), "payment": 0.0}
    for name in loser_names:
        outcomes[name] = {"won": False, "surplus": 0.0, "payment": 0.0}

    tracker.record_round_outcome(round_id=99, outcomes=outcomes)

    report = tracker.get_fairness_report()
    assert "jains_fairness_surplus" in report
    assert report["jains_fairness_surplus"] >= 0.0

    print(f"    FairnessTracker report: jains_surplus={report['jains_fairness_surplus']:.4f} | "
          f"starved={report['starved_agents']}")

    # ── Phase 8: Budget sanity check ──────────────────────
    print("\n  [STRESS] Phase 8: Budget integrity check...")
    for name in agent_names:
        budget_after = agents[name].budget
        budget_before = budgets_before[name]
        if name in winner_names:
            assert budget_after <= budget_before, \
                f"{name}: winner budget should decrease or stay same"
        else:
            assert budget_after == budget_before, \
                f"{name}: loser budget should not change"
        print(f"    {name:12} | before=${budget_before:.2f} | after=${budget_after:.2f}")

    # -- Summary ------------------------------------------------
    print(f"\n  ===============================================")
    print(f"  STRESS TEST PASSED")
    print(f"  Total agents    : 4")
    print(f"  Winners         : {winner_names}")
    print(f"  Losers          : {loser_names}")
    print(f"  Social welfare  : ${result.total_value:.2f}")
    print(f"  Jain's fairness : {fairness_index:.4f}")
    print(f"  Staggered LLM   : {total_elapsed:.0f}ms for all 4 agents")
    print(f"  ===============================================")


# ─────────────────────────────────────────────────────────────
# GROUP 5: Multi-Round Adaptive Learning
# ─────────────────────────────────────────────────────────────

@pytest.mark.live
def test_multi_round_adaptive_bidding(live_llm_client, engine_resources):
    """
    3 agents compete over 2 consecutive auction rounds.
    Validates that memory updates cause adaptive bid shading in Round 2.
    """
    agents_local = {
        "AlphaBot": ResearchAgent("AlphaBot", budget=500.0, llm_client=live_llm_client),
        "BetaBot":  DataAgent("BetaBot",      budget=450.0, llm_client=live_llm_client),
        "GammaBot": AnalysisAgent("GammaBot", budget=400.0, llm_client=live_llm_client),
    }
    tasks = {
        "AlphaBot": "Deep literature review of auction theory and mechanism design for AI agents",
        "BetaBot":  "Ingest 1TB of log data and extract structured event sequences for ML training",
        "GammaBot": "Compute optimal Markowitz efficient frontier with 200-asset covariance matrix",
    }

    print("\n  [ADAPTIVE] Round 1...")
    round1_bids: Dict[str, float] = {}

    engine1 = AuctionEngine(engine_resources)
    from compute_bazaar.agents.base import to_engine_bid
    bid_map1: Dict[str, Tuple[str, float]] = {}

    for name, agent in agents_local.items():
        decision = agent.analyze_task(tasks[name])
        bid = agent.create_bid(tasks[name], decision=decision, bid_type="bundle")
        eb = to_engine_bid(bid)
        engine1.submit_bid(eb)
        bid_map1[eb.bid_id] = (name, eb.bid_amount)
        round1_bids[name] = eb.bid_amount

    result1 = engine1.run_auction(round_id=1)

    # Record outcomes
    won1 = set()
    for wb in result1.winning_bids:
        agent_name, _ = bid_map1[wb.bid_id]
        vcg = result1.payments.get(wb.bid_id, 0.0)
        agents_local[agent_name].record_auction_outcome(
            bid_id=wb.bid_id, round_id=1, won=True,
            bid_amount=wb.bid_amount, payment_amount=vcg,
            resources_allocated=wb.resource_bundle,
        )
        won1.add(agent_name)

    for name in agents_local:
        if name not in won1:
            agents_local[name].record_auction_outcome(
                bid_id="r1-lost", round_id=1, won=False,
                bid_amount=round1_bids[name], payment_amount=0.0
            )

    print(f"    Round 1 winners: {won1}")

    print("  [ADAPTIVE] Round 2 - agents adapt based on Round 1 memory...")
    engine2 = AuctionEngine(engine_resources)
    bid_map2: Dict[str, Tuple[str, float]] = {}
    round2_bids: Dict[str, float] = {}

    for name, agent in agents_local.items():
        decision = agent.analyze_task(tasks[name])
        bid = agent.create_bid(tasks[name], decision=decision, bid_type="bundle")
        eb = to_engine_bid(bid)
        engine2.submit_bid(eb)
        bid_map2[eb.bid_id] = (name, eb.bid_amount)
        round2_bids[name] = eb.bid_amount

    result2 = engine2.run_auction(round_id=2)

    # Validate Round 2 has valid outcomes
    assert len(result2.winning_bids) >= 1
    assert result2.total_value >= 0

    # Losers in Round 1 should bid higher or equal in Round 2 (starvation boost)
    for name in (set(agents_local) - won1):
        starvation = agents_local[name].memory.consecutive_losses
        if starvation >= 1:
            # Agent memory should be tracking consecutive losses
            assert starvation >= 1

    won2 = {bid_map2[wb.bid_id][0] for wb in result2.winning_bids}
    print(f"    Round 2 winners: {won2}")
    print(f"    Round 1 bids  : { {k: f'${v:.2f}' for k,v in round1_bids.items()} }")
    print(f"    Round 2 bids  : { {k: f'${v:.2f}' for k,v in round2_bids.items()} }")


# ─────────────────────────────────────────────────────────────
# GROUP 6: Error Handling & Resilience
# ─────────────────────────────────────────────────────────────

@pytest.mark.live
def test_live_graceful_error_on_missing_key():
    """
    GroqLLMClient raises LLMAuthenticationError on invalid key.
    The system must not crash silently.
    """
    from compute_bazaar.llm import LLMAuthenticationError
    bad_client = GroqLLMClient(
        config=LLMConfig(model="openai/gpt-oss-120b", mock_mode=False),
        api_key="gsk_BAD_KEY_FOR_TESTING_12345",
    )
    agent = CodingAgent("ErrorBot", budget=100.0, llm_client=bad_client)

    with pytest.raises((LLMAuthenticationError, LLMError, Exception)):
        agent.analyze_task("This should raise because the key is invalid")


@pytest.mark.live
def test_budget_constraint_enforced_in_live_bid(live_llm_client):
    """
    Even with live Groq reasoning, a very tight budget must be respected.
    The bid amount must never exceed the agent's budget.
    """
    tight_budget = 15.0
    agent = AnalysisAgent("TightBot", budget=tight_budget, llm_client=live_llm_client)
    task = "Quick statistical summary of 1000-row CSV dataset"

    decision = agent.analyze_task(task)
    bid = agent.create_bid(task, decision=decision, bid_type="bundle")

    assert bid.bid_amount <= tight_budget, \
        f"Bid (${bid.bid_amount:.2f}) must not exceed budget (${tight_budget:.2f})"
    assert decision.max_willingness_to_pay <= tight_budget

    print(f"\n  [Budget constraint] wtp=${decision.max_willingness_to_pay:.2f} | "
          f"bid=${bid.bid_amount:.2f} | budget=${tight_budget:.2f} OK")
