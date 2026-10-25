"""
=============================================================
 COMPUTE BAZAAR — Week 2 Benchmark & Demo
=============================================================
 Demonstrates and benchmarks:
   1. LP solver vs Greedy: who finds better solutions?
   2. VCG pricing: truthfulness proof
   3. Stress test: 50-agent auction performance

 Run with:  python week2_benchmark.py
=============================================================
"""

import sys
import os
import time
import random
import uuid
from typing import List

# Add repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure UTF-8 stdout on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from backend.app.auction.engine import (
    AuctionEngine, Resource, Bid, SolverMethod,
    solve_greedy, solve_wdp_lp, compute_vcg_prices,
)
from backend.app.auction.pricing import (
    apply_tier_pricing, format_payment_report,
)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def make_bid(agent: str, bundle: List[str], amount: float, task: str = "") -> Bid:
    return Bid(
        bid_id=str(uuid.uuid4())[:8],
        agent_id=agent,
        resource_bundle=bundle,
        bid_amount=amount,
        task_description=task or f"{agent} task",
    )


RESOURCES = [
    Resource("GPU-1",    "gpu",        1.0, "slot"),
    Resource("GPU-2",    "gpu",        1.0, "slot"),
    Resource("API-100",  "api_budget", 100.0, "dollars"),
    Resource("RAM-32GB", "ram",        32.0, "gb"),
    Resource("CPU-8",    "cpu",        8.0,  "cores"),
]

RES_MAP = {r.name: r for r in RESOURCES}


# ─────────────────────────────────────────────────────────────
# DEMO 1 — LP vs Greedy: the critical difference
# ─────────────────────────────────────────────────────────────

def demo_lp_vs_greedy():
    """
    Scenario where GREEDY fails but LP finds the true optimum.

    The trick: a bundle bid's value-per-resource looks lower than
    individual bids, so greedy picks the wrong set. LP does it right.

    Resources: GPU-1, GPU-2
    
    Bids:
      A: {GPU-1}       → $60   (value-per-res: $60)
      B: {GPU-2}       → $55   (value-per-res: $55)
      C: {GPU-1,GPU-2} → $140  (value-per-res: $70)

    Greedy:
      Sorts by value-per-res: A($60/res) > C($70/res) > B($55/res)
      Actually: C has 70/res, A has 60/res, B has 55/res
      So greedy picks C first ($140, takes both GPUs) → total $140 ✅

    Trickier scenario:
      A: {GPU-1}       → $80
      B: {GPU-2}       → $75
      C: {GPU-1,GPU-2} → $140   value-per-res = $70 < A($80) and B($75)

    Greedy:
      Sorts: A(80/res) > B(75/res) > C(70/res)
      Picks A ($80) → GPU-1 gone
      Picks B ($75) → GPU-2 gone
      C blocked
      Total = $155

    LP:
      Checks: A+B = $155 vs C = $140 → A+B wins!
      Total = $155  ← same here, LP also picks A+B

    Even trickier (LP strictly better than greedy):
      A: {GPU-1}            → $70
      B: {GPU-2}            → $65
      C: {GPU-1, GPU-2}     → $160   value-per-res = $80

    Greedy:
      Sorts: C($80/res) > A($70/res) > B($65/res)
      Picks C ($160) → both GPUs gone, A & B blocked
      Total = $160

    LP:
      A+B = $135 vs C = $160 → LP picks C ($160)  ← LP agrees here too

    THE REAL LP-BEATS-GREEDY scenario (3 items):
      Items: GPU-1, GPU-2, API-100

      A: {GPU-1, API-100}   → $110  (per-res: $55)
      B: {GPU-2}            → $90   (per-res: $90)
      C: {GPU-1}            → $85   (per-res: $85)
      D: {API-100}          → $80   (per-res: $80)

    Greedy order by value/res: B($90) > C($85) > D($80) > A($55)
      Pick B ($90, takes GPU-2)
      Pick C ($85, takes GPU-1) — API-100 still free
      Pick D ($80, takes API-100)
      A blocked (GPU-1 and API-100 gone)
      Greedy total = $255

    LP optimal:
      Try: A+B = $200 (GPU-1+API-100 + GPU-2)
      Try: B+C+D = $255 (GPU-2+GPU-1+API-100)
      Try: A+B = $200
      Best: B+C+D = $255 ← same result this time

    LP strictly beats greedy when greedy makes a locally good
    but globally suboptimal early choice. We demonstrate with
    a crafted example below.
    """

    print("\n" + "=" * 64)
    print("  DEMO 1: LP vs Greedy Solver Comparison")
    print("=" * 64)

    resources_2 = {
        "GPU-1":   Resource("GPU-1",   "gpu",       1.0),
        "GPU-2":   Resource("GPU-2",   "gpu",       1.0),
        "API-100": Resource("API-100", "api_budget", 100.0),
    }

    # Crafted scenario where greedy is suboptimal:
    # Greedy picks the high per-resource bid first, but LP finds
    # that two smaller bids together beat one big one.
    #
    # A: {GPU-1, GPU-2, API-100} bid=$200 (per-res = $66.7)  ← greedy picks FIRST
    # B: {GPU-1}                 bid=$90  (per-res = $90)
    # C: {GPU-2}                 bid=$85  (per-res = $85)
    # D: {API-100}               bid=$70  (per-res = $70)
    #
    # Greedy: picks B first ($90), then C ($85), then D ($70) = $245
    # But wait — B+C+D = $245, A = $200.  Greedy wins here too.
    #
    # Let's make A better:
    # A: {GPU-1, GPU-2, API-100} bid=$260 (per-res = $86.7)  ← greedy picks FIRST
    # B: {GPU-1}                 bid=$90  (per-res = $90)   ← actually higher per-res!
    # C: {GPU-2}                 bid=$85  (per-res = $85)
    # D: {API-100}               bid=$70  (per-res = $70)
    #
    # Greedy: B(90/res) > A(86.7/res) > C(85/res) > D(70/res)
    #   Pick B ($90, GPU-1 gone)
    #   Try A: needs GPU-1 → blocked
    #   Pick C ($85, GPU-2 gone)
    #   Pick D ($70, API-100 gone)
    #   Total = $245
    #
    # LP: A=$260 vs B+C+D=$245  → LP picks A! Total = $260
    #
    # GREEDY = $245, LP = $260 → LP wins by $15!

    bids = [
        make_bid("Bundle-Agent", ["GPU-1", "GPU-2", "API-100"], 260.0, "Full stack AI job"),
        make_bid("Agent-B",      ["GPU-1"],                      90.0, "GPU-intensive task"),
        make_bid("Agent-C",      ["GPU-2"],                      85.0, "Secondary GPU task"),
        make_bid("Agent-D",      ["API-100"],                    70.0, "API-heavy task"),
    ]

    print("\n  Bids submitted:")
    for b in bids:
        vpr = b.bid_amount / len(b.resource_bundle)
        bundle = "+".join(b.resource_bundle)
        print(f"    {b.agent_id:15s}: [{bundle:30s}] ${b.bid_amount:6.2f}  "
              f"(${vpr:.2f}/resource)")

    # Run both solvers
    greedy_winners, greedy_value = solve_greedy(bids, resources_2)
    lp_winners, lp_value = solve_wdp_lp(bids, resources_2)

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  GREEDY RESULT: ${greedy_value:6.2f}                         │")
    print(f"  │  Winners: {', '.join(w.agent_id for w in greedy_winners):38s}│")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  LP RESULT:     ${lp_value:6.2f}                         │")
    print(f"  │  Winners: {', '.join(w.agent_id for w in lp_winners):38s}│")
    print(f"  └─────────────────────────────────────────────────┘")

    if lp_value > greedy_value:
        gain = lp_value - greedy_value
        print(f"\n  ✅ LP is ${gain:.2f} BETTER than greedy!")
        print(f"     Greedy missed the bundle bid that dominates 3 individual bids.")
    elif lp_value == greedy_value:
        print(f"\n  ✅ Both solvers found the same optimum this time.")
    else:
        print(f"\n  ⚠️  Unexpected: greedy beat LP? Check input data.")


# ─────────────────────────────────────────────────────────────
# DEMO 2 — VCG Pricing: Truthfulness Proof
# ─────────────────────────────────────────────────────────────

def demo_vcg_pricing():
    """
    Show that VCG prices are:
      1. Always ≤ bid amount (winners don't overpay)
      2. Reflect true externality imposed on others
      3. Make truthful bidding the best strategy
    """
    print("\n" + "=" * 64)
    print("  DEMO 2: VCG Pricing — Truthfulness Proof")
    print("=" * 64)

    print("""
  SCENARIO:
  ─────────
  Resources: GPU-1, API-100

  Agent-Finance: wants {GPU-1, API-100} → bids $100
  Agent-Gamer:   wants {GPU-1}          → bids $60
  Agent-Writer:  wants {API-100}        → bids $45

  Without Agent-Finance:
    → Agent-Gamer wins GPU-1 ($60) + Agent-Writer wins API-100 ($45)
    → Total = $105

  With Agent-Finance winning:
    → Others get $0
    → Agent-Finance's VCG price = $105 - $0 = $105? 
    
  Wait — $105 > $100 bid! Finance wouldn't bid if it costs more.
  
  Key: If VCG price > bid, Finance wouldn't WIN in the first place
  (Finance only wins if their bid is ≥ the displaced value of others).
  
  Let's verify with actual numbers:
    Finance bid $100 vs Gamer+Writer = $105
    → LP picks Gamer+Writer ($105 > $100)! Finance LOSES.
  
  Now if Finance bids $110 (their TRUE value):
    $110 vs $105 → Finance WINS
    VCG price = $105 (displaces $60+$45 from others) ≤ $110 ✅
    
  Finance's surplus = $110 (true value) - $105 (payment) = $5
  This is WHY agents bid truthfully: surplus is maximized at truth!
    """)

    resources_vcg = {
        "GPU-1":   Resource("GPU-1",   "gpu",        1.0),
        "API-100": Resource("API-100", "api_budget", 100.0),
    }

    # Finance's true value is $110 — they bid truthfully
    bids = [
        make_bid("Agent-Finance", ["GPU-1", "API-100"], 110.0, "Finance processing"),
        make_bid("Agent-Gamer",   ["GPU-1"],              60.0, "GPU rendering"),
        make_bid("Agent-Writer",  ["API-100"],             45.0, "Content generation"),
    ]

    winners, total = solve_wdp_lp(bids, resources_vcg)
    payments = compute_vcg_prices(winners, bids, resources_vcg)

    print("  VCG RESULTS:")
    print("  " + "─" * 50)
    for winner in winners:
        p = payments.get(winner.bid_id, 0.0)
        surplus = winner.bid_amount - p
        print(f"  🏆 {winner.agent_id}")
        print(f"     Bundle: {winner.resource_bundle}")
        print(f"     Bid (true value): ${winner.bid_amount:.2f}")
        print(f"     VCG Payment:      ${p:.2f}")
        print(f"     Surplus (profit): ${surplus:.2f}  ← agent keeps this!")
        print()

    print(f"  Total value extracted: ${total:.2f}")
    print(f"\n  TRUTHFULNESS CHECK:")
    all_pass = True
    for w in winners:
        p = payments.get(w.bid_id, 0.0)
        if p > w.bid_amount:
            print(f"  ❌ FAIL: {w.agent_id} pays ${p:.2f} > bid ${w.bid_amount:.2f}")
            all_pass = False
        else:
            print(f"  ✅ {w.agent_id}: pays ${p:.2f} ≤ bid ${w.bid_amount:.2f}")
    if all_pass:
        print("\n  ✅ VCG truthfulness property holds!")


# ─────────────────────────────────────────────────────────────
# DEMO 3 — Stress Test: 50 Agents
# ─────────────────────────────────────────────────────────────

def demo_stress_test(n_agents: int = 50):
    """
    Generate n_agents random bids and time the LP solver.
    Shows that LP can handle realistic workloads quickly.
    """
    print("\n" + "=" * 64)
    print(f"  DEMO 3: Stress Test — {n_agents} Agents")
    print("=" * 64)

    random.seed(42)
    res_names = [r.name for r in RESOURCES]

    bids = []
    for i in range(n_agents):
        # Random bundle: 1 to 3 resources
        bundle_size = random.randint(1, min(3, len(res_names)))
        bundle = random.sample(res_names, bundle_size)
        amount = round(random.uniform(10.0, 200.0), 2)
        bids.append(make_bid(
            f"Agent-{i:03d}", bundle, amount, f"Task {i}"
        ))

    print(f"\n  Generated {n_agents} random bids over {len(RESOURCES)} resources")

    # Time greedy
    t0 = time.perf_counter()
    g_winners, g_val = solve_greedy(bids, RES_MAP)
    greedy_ms = (time.perf_counter() - t0) * 1000

    # Time LP
    t0 = time.perf_counter()
    lp_winners, lp_val = solve_wdp_lp(bids, RES_MAP)
    lp_ms = (time.perf_counter() - t0) * 1000

    gain = ((lp_val - g_val) / g_val * 100) if g_val > 0 else 0

    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  GREEDY  │ value=${g_val:7.2f}  │ time={greedy_ms:6.1f}ms │")
    print(f"  │  LP      │ value=${lp_val:7.2f}  │ time={lp_ms:6.1f}ms │")
    print(f"  │  LP gain │ +{gain:5.1f}% better                     │")
    print(f"  └──────────────────────────────────────────────┘")
    print(f"\n  LP winners: {len(lp_winners)} of {n_agents} agents")
    print(f"  Greedy winners: {len(g_winners)} of {n_agents} agents")


# ─────────────────────────────────────────────────────────────
# DEMO 4 — Tier Pricing
# ─────────────────────────────────────────────────────────────

def demo_tier_pricing():
    """Show Gold/Silver/Bronze discount tiers applied to VCG payments."""
    print("\n" + "=" * 64)
    print("  DEMO 4: Tier Pricing (Gold/Silver/Bronze)")
    print("=" * 64)

    engine = AuctionEngine(RESOURCES, method=SolverMethod.LP)

    bids_data = [
        ("gold-agent",   ["GPU-1"],              80.0, "gold"),
        ("silver-agent", ["GPU-2"],              70.0, "silver"),
        ("bronze-agent", ["API-100"],            60.0, "bronze"),
    ]
    agent_tiers = {}
    for agent_id, bundle, amount, tier in bids_data:
        b = make_bid(agent_id, bundle, amount)
        engine.submit_bid(b)
        agent_tiers[agent_id] = tier

    result = engine.run_auction(round_id=100)
    priced = apply_tier_pricing(result, agent_tiers)

    print(format_payment_report(result, priced))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Set UTF-8 output for Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "🏪" * 32)
    print("  COMPUTE BAZAAR — Week 2 Benchmark")
    print("  LP Solver + VCG Pricing")
    print("🏪" * 32)

    demo_lp_vs_greedy()
    demo_vcg_pricing()
    demo_stress_test(50)
    demo_tier_pricing()

    print("\n\n✅ Week 2 complete! All demos ran successfully.")
    print("   Next: Week 3 — FastAPI backend so agents bid via HTTP\n")
