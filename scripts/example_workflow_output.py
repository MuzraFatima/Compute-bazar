"""
Three-bidder VCG example to capture real output for the README.
Usage: python scripts/example_workflow_output.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compute_bazaar.agents import CodingAgent
from compute_bazaar.llm import GroqLLMClient, LLMConfig
from compute_bazaar.tools import estimate_cost, execute_task
from backend.app.auction.engine import AuctionEngine, Resource, Bid, SolverMethod

# 1. Initialize Infrastructure & Engine
resources = [
    Resource("GPU-Slot-1", "gpu",        1.0,  "slot"),
    Resource("RAM-Block-32GB", "ram",    32.0, "gb"),
    Resource("CPU-Cores-8",   "cpu",     8.0,  "cores"),
]
engine = AuctionEngine(resources, method=SolverMethod.LP)

# 2. Initialize Autonomous Agent with Groq LLM Reasoning (mock mode)
llm = GroqLLMClient(config=LLMConfig(mock_mode=True))
agent = CodingAgent("AutoDevBot", budget=500.0, llm_client=llm)

# 3. Agent forecasts task needs
task_prompt = "Fine-tune 7B parameter code completion model"
prediction = estimate_cost(task_prompt)
print(f"ML Forecast: {prediction.predicted_execution_time_sec}s runtime | ${prediction.predicted_cost:.2f} cost")

# 4. Three competing bids on overlapping resources — ensures VCG payment > $0
# AutoDevBot bids $145 for GPU+CPU; competitors bid $60 (GPU-only) and $70 (CPU+RAM).
# ILP picks AutoDevBot ($145 > $60+$70... wait, 130<145) so AutoDevBot wins.
# VCG payment = value without AutoDevBot (best alternative = $60+$70=$130) - value of others with AutoDevBot (=$0)
#             = $130 - $0 = $130 → AutoDevBot pays $130, saving $15 (surplus).
bid_a = Bid(bid_id="bid-A", agent_id="AutoDevBot",  resource_bundle=["GPU-Slot-1", "CPU-Cores-8"], bid_amount=145.0)
bid_b = Bid(bid_id="bid-B", agent_id="ResearchBot", resource_bundle=["GPU-Slot-1"],               bid_amount=60.0)
bid_c = Bid(bid_id="bid-C", agent_id="AnalysisBot", resource_bundle=["CPU-Cores-8", "RAM-Block-32GB"], bid_amount=70.0)

engine.submit_bid(bid_a)
engine.submit_bid(bid_b)
engine.submit_bid(bid_c)

# 5. Engine runs auction round (ILP solve + VCG externality pricing)
result = engine.run_auction(round_id=1)

# 6. Winner execution feedback loop
print(f"\nAuction Round #1 — Winners: {len(result.winning_bids)}")
for winner in result.winning_bids:
    vcg_payment = result.payments[winner.bid_id]
    surplus = winner.bid_amount - vcg_payment
    print(f"  Winner: {winner.agent_id} | Bundle: {winner.resource_bundle}")
    print(f"  Bid: ${winner.bid_amount:.2f} | VCG Payment: ${vcg_payment:.2f} | Surplus: ${surplus:.2f}")
    if winner.agent_id == "AutoDevBot":
        execute_task(agent, task_prompt, {"won": True, "payment": vcg_payment, "resources": winner.resource_bundle})
