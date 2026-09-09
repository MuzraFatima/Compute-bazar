"""
=============================================================
 COMPUTE BAZAAR 2.0 -- End-to-End Live System Demo
=============================================================
 Demonstrates the complete autonomous AI compute workflow:
   1. Multi-Agent Initialization (Groq-powered Research, Data, Analysis agents)
   2. Live Groq LLM Task Analysis (complexity, utility, WTP, resource bundle)
   3. Adaptive Bidding & Bundle Formulation (Week 4 & Week 6 mechanics)
   4. Dynamic Marketplace & Scarcity-based Pricing (Week 5)
   5. Combinatorial Auction Solving & VCG Pricing (WDP via ILP)
   6. Safe Task Execution & ML Feedback Loop (Week 7)
   7. Budget Updates & Fairness Analytics (Jain's Fairness Index & Starvation)
=============================================================
"""

import sys
import os
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

# Ensure compute-bazaar project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure utf-8 encoding for Windows terminal if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables (GROQ_API_KEY)
load_dotenv()

from compute_bazaar.llm import GroqLLMClient, LLMConfig, LLMError, LLMAuthenticationError
from compute_bazaar.agents import ResearchAgent, DataAgent, AnalysisAgent, TaskDecision
from compute_bazaar.marketplace import DynamicMarketplace
from compute_bazaar.fairness import FairnessTracker
from backend.app.auction.engine import AuctionEngine, Resource as EngineResource, SolverMethod


def print_banner(text: str, width: int = 75, char: str = "="):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def print_section(title: str, icon: str = "[STEP]"):
    print(f"\n{'=' * 75}")
    print(f"{icon}  {title}")
    print(f"{'=' * 75}")


def run_end_to_end_demo():
    print_banner("COMPUTE BAZAAR 2.0 -- END-TO-END MULTI-AGENT AUCTION DEMO", char="=")

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Initialize Live Groq LLM Client & Agents
    # ─────────────────────────────────────────────────────────────
    print_section("STEP 1: Initializing Groq LLM Client & Autonomous AI Agents", icon="[1/8]")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not found in environment or .env file.")
        print("  Please add GROQ_API_KEY to your .env file to run live LLM reasoning.")
        sys.exit(1)

    try:
        llm_config = LLMConfig(
            model="openai/gpt-oss-120b",
            temperature=0.4,
            max_tokens=1024,
            mock_mode=False,
        )
        llm_client = GroqLLMClient(config=llm_config, api_key=api_key)
        print(f"  [OK] Groq LLM Client initialized in LIVE MODE (Model: {llm_config.model})")
    except LLMAuthenticationError as e:
        print(f"[Authentication Error] Invalid Groq API key: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] Failed to initialize Groq client: {e}")
        sys.exit(1)

    # Instantiate 3 specialized agents with realistic starting budgets
    research_agent = ResearchAgent(
        name="ArxivBot-Alpha",
        budget=350.0,
        llm_client=llm_client,
    )
    data_agent = DataAgent(
        name="ETLBot-Beta",
        budget=450.0,
        llm_client=llm_client,
    )
    analysis_agent = AnalysisAgent(
        name="QuantBot-Gamma",
        budget=400.0,
        llm_client=llm_client,
    )

    agents = [research_agent, data_agent, analysis_agent]

    print("\n  Registered AI Agents:")
    for a in agents:
        print(f"   * {a.name:<16} | Type: {a.agent_type:<15} | Budget: ${a.budget:.2f} | Domain: {a.capability.value}")
        print(f"     Preferred Resources: {', '.join(a.preferred_resources)}")

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Initialize Marketplace & Available Compute Supply
    # ─────────────────────────────────────────────────────────────
    print_section("STEP 2: Initializing Dynamic Marketplace & Compute Resources", icon="[2/8]")

    marketplace = DynamicMarketplace(name="Compute Bazaar Global Cluster")
    available_market_res = marketplace.list_available_resources()

    print(f"  Marketplace: '{marketplace.name}'")
    print(f"  Active Providers: {len(marketplace.list_providers())} | Operational Resources: {len(available_market_res)}")
    print("\n  Available Market Supply Inventory:")
    print(f"   {'Resource Name':<18} {'Category':<14} {'Capacity':<12} {'Base Floor Price':<18}")
    print(f"   {'-'*18} {'-'*14} {'-'*12} {'-'*18}")
    for r in available_market_res:
        print(f"   {r.name:<18} {r.resource_type:<14} {r.total_capacity:<12.1f} ${r.min_price:<17.2f}")

    fairness_tracker = FairnessTracker()
    for a in agents:
        fairness_tracker.register_agent(a.name)

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Assign Real-World Tasks & Run Live LLM Reasoning
    # ─────────────────────────────────────────────────────────────
    print_section("STEP 3: Live LLM Task Analysis (Groq API Reasoning)", icon="[3/8]")

    tasks = {
        research_agent: (
            "Synthesize recent academic papers on LLM reasoning architectures, "
            "perform search API queries across 50 papers, and generate an executive summary."
        ),
        data_agent: (
            "Clean and process a 20GB multi-modal dataset: tokenize text, extract features, "
            "and compute GPU-accelerated vector embeddings for neural search."
        ),
        analysis_agent: (
            "Execute Monte Carlo risk simulation with 50,000 iterations for portfolio stress "
            "testing and forecast dynamic price elasticity models."
        ),
    }

    decisions: Dict[Any, TaskDecision] = {}

    for agent, task in tasks.items():
        print(f"\n  [Agent: {agent.name}] Analyzing Task via Groq LLM...")
        print(f"   Task: \"{task}\"")
        try:
            t0 = time.perf_counter()
            decision = agent.analyze_task(task)
            elapsed = (time.perf_counter() - t0) * 1000
            decisions[agent] = decision

            print(f"   [DONE] Analysis Complete in {elapsed:.0f}ms (Groq Latency: {decision.llm_latency_ms:.0f}ms | Tokens: {decision.llm_tokens_used})")
            print(f"     - Understood Goal: {decision.understood_task[:70]}...")
            print(f"     - Required Resources: {decision.required_resources}")
            print(f"     - Complexity (1-10): {decision.complexity} | Confidence: {decision.confidence:.2f} | Priority: {decision.priority.upper()}")
            print(f"     - Est. Utility: ${decision.estimated_utility:.2f} | Max WTP: ${decision.max_willingness_to_pay:.2f}")
            if decision.reasoning:
                print(f"     - LLM Reasoning: {decision.reasoning[0]}")
        except LLMError as e:
            print(f"   [FAILED] LLM Analysis Failed for {agent.name}: {e}")
            return

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Formulate Intelligent & Adaptive Combinatorial Bids
    # ─────────────────────────────────────────────────────────────
    print_section("STEP 4: Formulate Intelligent Combinatorial Bids", icon="[4/8]")

    bids = {}
    for agent in agents:
        decision = decisions[agent]
        task = tasks[agent]
        bid = agent.create_bid(task_description=task, decision=decision, bid_type="bundle")
        bids[agent] = bid

        print(f"  * [{agent.name}] Bid Formulated:")
        print(f"     Bid ID: {bid.bid_id}")
        print(f"     Bundle Requested: {bid.resource_bundle}")
        print(f"     Valuation (Utility): ${bid.valuation:.2f}  ==>  Intelligent Bid: ${bid.bid_amount:.2f}")
        print(f"     Priority: {bid.priority.upper()} | Budget Limit: ${agent.budget:.2f}")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Marketplace Demand Tracking & Dynamic Pricing
    # ─────────────────────────────────────────────────────────────
    print_section("STEP 5: Marketplace Scarcity & Dynamic Price Adjustment", icon="[5/8]")

    requested_bundles = [bid.resource_bundle for bid in bids.values()]
    marketplace.record_demand(requested_bundles)
    marketplace.update_dynamic_pricing()

    print("  Resource Scarcity & Dynamic Price Shift in Round 1:")
    print(f"   {'Resource Name':<18} {'Demand':<10} {'Available':<12} {'Scarcity Ratio':<16} {'Updated Price':<14}")
    print(f"   {'-'*18} {'-'*10} {'-'*12} {'-'*16} {'-'*14}")
    for r in available_market_res:
        scarcity = marketplace.calculate_scarcity(r.name)
        demand_count = marketplace.demand_history.get(r.name, 0)
        print(f"   {r.name:<18} {demand_count:<10.0f} {r.available_capacity:<12.1f} {scarcity:<16.2f} ${r.current_price:<13.2f}")

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Submit to Auction Engine & Run Winner Determination (WDP + VCG)
    # ─────────────────────────────────────────────────────────────
    print_section("STEP 6: Auction Solving (ILP Winner Determination & VCG Pricing)", icon="[6/8]")

    engine_resources = [
        EngineResource(r.name, r.resource_type, r.total_capacity)
        for r in available_market_res
    ]
    engine = AuctionEngine(engine_resources, method=SolverMethod.LP)

    for agent, bid in bids.items():
        agent.submit_to_engine(engine, bid)

    allocation = engine.run_auction(round_id=1)

    print(f"  Auction Round {allocation.round_id} Results:")
    print(f"   * Solver Method: {allocation.solver_method.value.upper()} (ILP Optimal WDP)")
    print(f"   * Solve Time: {allocation.solve_time_ms:.2f} ms")
    print(f"   * Total Social Welfare Extracted: ${allocation.total_value:.2f}")
    print(f"   * Benchmark Comparison: LP=${allocation.lp_value:.2f} vs Greedy=${allocation.greedy_value:.2f}")
    print(f"   * Unallocated Compute Slots: {allocation.unallocated_resources or 'None (100% utilized)'}")

    winning_agent_names = {w.agent_id for w in allocation.winning_bids}
    winning_bids_by_agent = {w.agent_id: w for w in allocation.winning_bids}

    print(f"\n  [WINNERS] Winning Allocations ({len(allocation.winning_bids)} Winners):")
    for w in allocation.winning_bids:
        payment = allocation.payments.get(w.bid_id, 0.0)
        discount = w.bid_amount - payment
        print(f"   * Winner: {w.agent_id}")
        print(f"     - Allocated Bundle: {w.resource_bundle}")
        print(f"     - Bid Amount: ${w.bid_amount:.2f} | VCG Payment: ${payment:.2f} (Savings: ${discount:.2f})")

    losing_agents = [a for a in agents if a.name not in winning_agent_names]
    print(f"\n  [LOSERS] Displaced / Losing Agents ({len(losing_agents)} Lost due to resource contention):")
    for a in losing_agents:
        print(f"   * {a.name} (Bid ${bids[a].bid_amount:.2f} for {bids[a].resource_bundle})")

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Safe Task Execution & ML Feedback Loop
    # ─────────────────────────────────────────────────────────────
    print_section("STEP 7: Safe Task Execution & Telemetry Feedback", icon="[7/8]")

    round_outcomes_fairness = {}

    for agent in agents:
        bid = bids[agent]
        task = tasks[agent]
        is_winner = agent.name in winning_agent_names

        if is_winner:
            w = winning_bids_by_agent[agent.name]
            payment = allocation.payments.get(w.bid_id, 0.0)
            surplus = max(0.0, bid.valuation - payment)

            # Record in agent memory and deduct payment from budget
            agent.record_auction_outcome(
                bid_id=bid.bid_id,
                round_id=1,
                won=True,
                bid_amount=bid.bid_amount,
                payment_amount=payment,
                resources_allocated=w.resource_bundle,
            )

            # Safely execute task
            exec_res = agent.execute_awarded_task(
                task_description=task,
                allocation_result={
                    "won": True,
                    "payment": payment,
                    "resources": w.resource_bundle,
                },
            )

            print(f"\n  [SUCCESS] {agent.name}:")
            print(f"     Task Executed: \"{exec_res.get('task_description')[:50]}...\"")
            print(f"     Resources Used: {exec_res.get('resources_used')}")
            print(f"     Simulated Execution Time: {exec_res.get('execution_time_sec')}s | Cost Paid: ${exec_res.get('actual_cost'):.2f}")
            print(f"     Remaining Budget: ${agent.budget:.2f}")

            round_outcomes_fairness[agent.name] = {
                "won": True,
                "surplus": surplus,
                "payment": payment,
            }
        else:
            # Record loss outcome in agent memory
            agent.record_auction_outcome(
                bid_id=bid.bid_id,
                round_id=1,
                won=False,
                bid_amount=bid.bid_amount,
                payment_amount=0.0,
            )

            # Attempt safe execution with lost allocation status
            exec_res = agent.execute_awarded_task(
                task_description=task,
                allocation_result={"won": False},
            )

            print(f"\n  [SKIPPED] {agent.name}:")
            print(f"     Status: {exec_res.get('status').upper()} ({exec_res.get('reason')})")
            print(f"     Budget Untouched: ${agent.budget:.2f}")

            round_outcomes_fairness[agent.name] = {
                "won": False,
                "surplus": 0.0,
                "payment": 0.0,
            }

    # ─────────────────────────────────────────────────────────────
    # STEP 8: Final Summary & System Health Metrics
    # ─────────────────────────────────────────────────────────────
    print_section("STEP 8: Final Metrics & System Equity Summary", icon="[8/8]")

    fairness_tracker.record_round_outcome(round_id=1, outcomes=round_outcomes_fairness)
    fairness_report = fairness_tracker.get_fairness_report()

    print("  Agent Lifecycle Summary Table:")
    print(f"   {'Agent Name':<16} {'Complexity':<12} {'Confidence':<12} {'Bid':<10} {'Status':<10} {'Payment':<10} {'Rem. Budget':<14}")
    print(f"   {'-'*16} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")
    for a in agents:
        d = decisions[a]
        b = bids[a]
        won = a.name in winning_agent_names
        status_str = "WON" if won else "LOST"
        payment_val = allocation.payments.get(winning_bids_by_agent[a.name].bid_id, 0.0) if won else 0.0
        print(f"   {a.name:<16} {d.complexity:<12} {d.confidence:<12.2f} ${b.bid_amount:<9.2f} {status_str:<10} ${payment_val:<9.2f} ${a.budget:<13.2f}")

    print("\n  Marketplace & Fairness Telemetry:")
    print(f"   * Jain's Fairness Index (Surplus Distribution): {fairness_report['jains_fairness_surplus']:.4f}")
    print(f"   * Jain's Fairness Index (Win Rate Distribution): {fairness_report['jains_fairness_wins']:.4f}")
    print(f"   * Starved Agents Detected: {fairness_report['starved_agents'] or 'None'}")
    print(f"   * Dominating Agents Detected: {fairness_report['dominating_agents'] or 'None'}")
    print(f"   * Market Efficiency: Social Welfare = ${allocation.total_value:.2f} across {len(allocation.winning_bids)} allocations")

    print_banner("END-TO-END DEMO COMPLETED SUCCESSFULLY", char="=")


if __name__ == "__main__":
    run_end_to_end_demo()
