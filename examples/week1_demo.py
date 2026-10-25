"""
=============================================================
 COMPUTE BAZAAR — Week 1 Demo: Your First Auction!
=============================================================

 WHAT THIS FILE DOES:
 --------------------
 This is a simple Python script that demonstrates the core
 idea of Compute Bazaar: agents bid for compute resources,
 and the BEST (highest-value) combination of bids wins —
 not just whoever asked first.

 Think of it like this:
   - Resources = compute slots (GPU time, API budget, RAM)
   - Agents = AI programs that need resources to do their job
   - Bids = how much each agent values a set of resources
   - Auction = who gets what, fairly & efficiently

 HOW TO RUN:
 -----------
   python week1_demo.py

=============================================================
"""

import sys
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from itertools import combinations

# Add repo root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure UTF-8 stdout on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


# ============================================================
# STEP 1: Define what a "Resource" is
# ============================================================
# A resource is anything agents want to use:
#   - A GPU slot (for running heavy AI computations)
#   - An API budget ($100 to spend on GPT-4 calls)
#   - A RAM chunk (memory for running models)

@dataclass
class Resource:
    name: str        # e.g., "GPU-Slot-1"
    resource_type: str  # e.g., "gpu", "api_budget", "ram"
    capacity: float  # e.g., 1.0 = full slot, 100.0 = $100 budget

    def __repr__(self):
        return f"[{self.name}: {self.resource_type}={self.capacity}]"


# ============================================================
# STEP 2: Define what a "Bid" is
# ============================================================
# A bid is an agent saying: "I want THESE specific resources,
# and I value them at $X."
#
# This is "combinatorial" — an agent can bid on a BUNDLE
# of resources together, not just one at a time.
#
# Example: Agent B might say "I want GPU-1 AND API-budget,
#          together they're worth $80 to me."
#          But GPU-1 alone is only worth $20.

@dataclass
class Bid:
    agent_name: str          # Who is bidding
    task_description: str    # What they'll do with resources
    resource_bundle: List[str]  # Which resources they want (by name)
    bid_amount: float        # How much they value this bundle ($)
    priority: str = "normal" # "critical", "high", "normal", "low"

    def __repr__(self):
        bundle = " + ".join(self.resource_bundle)
        return (f"  🏷️  {self.agent_name}: wants [{bundle}] "
                f"→ bids ${self.bid_amount:.2f} ({self.priority})")


# ============================================================
# STEP 3: The Auction Engine (Simple Greedy Version)
# ============================================================
# The "Winner Determination Problem" (WDP) asks:
#   Given a set of bids, which COMBINATION of bids
#   should we accept to MAXIMIZE total value,
#   WITHOUT giving the same resource to two agents?
#
# In Week 2, we'll use proper linear programming to solve this.
# For now, we use a "greedy" approach: sort by value-per-resource,
# pick the best non-conflicting bids one at a time.

class SimpleAuctionEngine:
    """
    A simple combinatorial auction engine.
    
    In a combinatorial auction:
    - Multiple resources are auctioned simultaneously
    - Agents bid on BUNDLES of resources (not just one at a time)
    - The engine finds the best set of winning bids
    - Each resource can only go to ONE winner
    """

    def __init__(self, resources: List[Resource]):
        self.resources = {r.name: r for r in resources}
        self.bids: List[Bid] = []
        self.round_number = 0

    def add_bid(self, bid: Bid) -> bool:
        """Accept a bid from an agent. Returns True if bid is valid."""
        # Check that all requested resources actually exist
        for res_name in bid.resource_bundle:
            if res_name not in self.resources:
                print(f"  ❌ Invalid bid: Resource '{res_name}' doesn't exist!")
                return False
        
        # Check bid amount is positive
        if bid.bid_amount <= 0:
            print(f"  ❌ Invalid bid: Amount must be > 0, got {bid.bid_amount}")
            return False
        
        self.bids.append(bid)
        return True

    def run_auction(self) -> Dict:
        """
        Run the auction and determine winners.
        
        Algorithm (Greedy Approach):
        1. Sort all bids by "value per resource" (highest first)
        2. Go through bids in order
        3. Accept a bid if ALL its resources are still available
        4. Mark those resources as "taken"
        5. Repeat until no more compatible bids remain
        
        NOTE: This greedy approach is NOT optimal (we'll fix that in Week 2).
              It's a fast, easy-to-understand approximation.
        """
        self.round_number += 1
        print(f"\n{'='*60}")
        print(f"  🏪 COMPUTE BAZAAR — Auction Round #{self.round_number}")
        print(f"{'='*60}")

        # Show available resources
        print(f"\n📦 Available Resources:")
        for name, res in self.resources.items():
            print(f"   • {res.name} ({res.resource_type})")

        # Show all bids received
        print(f"\n📨 Bids Received ({len(self.bids)} total):")
        for bid in self.bids:
            print(bid)

        # Sort bids by value-per-resource (a fairness heuristic)
        sorted_bids = sorted(
            self.bids,
            key=lambda b: b.bid_amount / len(b.resource_bundle),
            reverse=True
        )

        # Track which resources have been allocated
        allocated_resources = set()
        winners = []
        losers = []
        total_value = 0.0

        # Greedy winner determination
        for bid in sorted_bids:
            bundle_set = set(bid.resource_bundle)
            
            # Check if ALL resources in this bundle are still free
            if bundle_set.isdisjoint(allocated_resources):
                # WINNER! Allocate these resources
                winners.append(bid)
                allocated_resources.update(bundle_set)
                total_value += bid.bid_amount
            else:
                # LOSER — at least one resource already taken
                conflict = bundle_set & allocated_resources
                losers.append((bid, conflict))

        # Display results
        print(f"\n🏆 AUCTION RESULTS:")
        print(f"{'─'*60}")
        
        print(f"\n✅ WINNERS ({len(winners)} agents):")
        for bid in winners:
            resources_won = " + ".join(bid.resource_bundle)
            payment = self._calculate_payment(bid, winners)
            print(f"   🥇 {bid.agent_name}")
            print(f"      Task: {bid.task_description}")
            print(f"      Got: {resources_won}")
            print(f"      Bid: ${bid.bid_amount:.2f}  |  Pays: ${payment:.2f}")
            print()

        print(f"❌ LOSERS ({len(losers)} agents):")
        for bid, conflict in losers:
            print(f"   • {bid.agent_name} → lost because {conflict} already taken")

        unallocated = set(self.resources.keys()) - allocated_resources
        if unallocated:
            print(f"\n⚠️  Unallocated Resources: {unallocated}")

        print(f"\n💰 Total Value Extracted: ${total_value:.2f}")

        # Compare with round-robin (just for illustration)
        rr_value = self._simulate_round_robin()
        improvement = ((total_value - rr_value) / rr_value * 100) if rr_value > 0 else 0
        
        print(f"\n📊 vs. Round-Robin Scheduling:")
        print(f"   Compute Bazaar: ${total_value:.2f}")
        print(f"   Round Robin:    ${rr_value:.2f}")
        if improvement > 0:
            print(f"   ✨ Auction is {improvement:.1f}% MORE efficient!")
        else:
            print(f"   (Tied or similar in this small example)")

        print(f"\n{'='*60}\n")

        return {
            "round": self.round_number,
            "winners": [w.agent_name for w in winners],
            "total_value": total_value,
            "round_robin_value": rr_value,
            "efficiency_gain_pct": improvement,
        }

    def _calculate_payment(self, winning_bid: Bid, all_winners: List[Bid]) -> float:
        """
        Calculate how much the winner actually PAYS.
        
        In Week 2, we'll implement proper VCG pricing (game-theory optimal).
        For now, we use a simplified "discount" rule:
          - Pay 90% of your bid (you save 10% as incentive to participate)
        
        VCG Rule (preview): You pay the VALUE you DISPLACED from others,
        not your own bid amount. This makes truthful bidding the best strategy.
        """
        return winning_bid.bid_amount * 0.90

    def _simulate_round_robin(self) -> float:
        """
        Simulate what WOULD happen with round-robin scheduling.
        
        Round-robin: just give resources to agents in the ORDER they arrived,
        ignoring bid values entirely.
        
        Returns the total value that would have been extracted.
        """
        allocated = set()
        total = 0.0
        
        # Round-robin = first-come, first-served (original order)
        for bid in self.bids:
            bundle_set = set(bid.resource_bundle)
            if bundle_set.isdisjoint(allocated):
                allocated.update(bundle_set)
                total += bid.bid_amount
        
        return total

    def reset(self):
        """Clear bids for the next auction round."""
        self.bids = []


# ============================================================
# STEP 4: Run the Demo!
# ============================================================

def run_demo():
    """
    Demo Scenario:
    
    A company runs 5 AI agents simultaneously.
    They have 3 compute resources available.
    Each agent wants a bundle of resources for their task.
    
    The auction engine decides who gets what — fairly and efficiently.
    """

    print("\n" + "🏪" * 30)
    print("  COMPUTE BAZAAR — Week 1 Demo")
    print("  Combinatorial Auction for AI Compute Resources")
    print("🏪" * 30)

    # Define available compute resources
    resources = [
        Resource("GPU-Slot-1",    "gpu",        capacity=1.0),
        Resource("GPU-Slot-2",    "gpu",        capacity=1.0),
        Resource("API-Budget-100","api_budget",  capacity=100.0),  # $100 of API calls
    ]

    # Create the auction engine
    engine = SimpleAuctionEngine(resources)

    # ── Scenario 1: Basic Auction ───────────────────────────
    print("\n\n📚 SCENARIO 1: Basic Priority Auction")
    print("   5 agents, 3 resources, who wins?")
    print("   (Note: Agent B has the most critical task!)")

    engine.add_bid(Bid(
        agent_name="Agent-Alpha",
        task_description="Generate marketing copy for newsletter",
        resource_bundle=["GPU-Slot-1"],
        bid_amount=15.0,
        priority="low"
    ))

    engine.add_bid(Bid(
        agent_name="Agent-Beta",
        task_description="🚨 Process URGENT financial transaction",
        resource_bundle=["GPU-Slot-1", "API-Budget-100"],  # Needs both!
        bid_amount=85.0,
        priority="critical"
    ))

    engine.add_bid(Bid(
        agent_name="Agent-Gamma",
        task_description="Resize product images for website",
        resource_bundle=["GPU-Slot-2"],
        bid_amount=8.0,
        priority="low"
    ))

    engine.add_bid(Bid(
        agent_name="Agent-Delta",
        task_description="Customer support chat response",
        resource_bundle=["API-Budget-100"],
        bid_amount=35.0,
        priority="normal"
    ))

    engine.add_bid(Bid(
        agent_name="Agent-Epsilon",
        task_description="Real-time fraud detection alert",
        resource_bundle=["GPU-Slot-2", "API-Budget-100"],
        bid_amount=70.0,
        priority="high"
    ))

    result1 = engine.run_auction()
    engine.reset()

    # ── Scenario 2: Tight Competition ──────────────────────
    print("\n📚 SCENARIO 2: Tight Competition")
    print("   What if many agents want the same resource?")

    engine.add_bid(Bid(
        agent_name="Agent-Writer",
        task_description="Write a blog post using GPT-4",
        resource_bundle=["API-Budget-100"],
        bid_amount=20.0,
        priority="normal"
    ))

    engine.add_bid(Bid(
        agent_name="Agent-Coder",
        task_description="Debug production code (urgent!)",
        resource_bundle=["API-Budget-100"],
        bid_amount=55.0,
        priority="high"
    ))

    engine.add_bid(Bid(
        agent_name="Agent-Analyst",
        task_description="Monthly report generation",
        resource_bundle=["API-Budget-100"],
        bid_amount=12.0,
        priority="low"
    ))

    engine.add_bid(Bid(
        agent_name="Agent-Trainer",
        task_description="Fine-tune small model",
        resource_bundle=["GPU-Slot-1", "GPU-Slot-2"],
        bid_amount=45.0,
        priority="normal"
    ))

    result2 = engine.run_auction()
    engine.reset()

    # ── Final Summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 WEEK 1 TAKEAWAYS")
    print("=" * 60)
    print("""
  What we just built:
  ✅ Resources — compute slots agents want to use
  ✅ Bids — agents declare how much they value a bundle
  ✅ Combinatorial Auction — finds the best combination of winners
  ✅ vs. Round Robin — proves auction is more efficient

  What's coming in Week 2:
  🔜 Replace greedy solver with proper Linear Programming (WDP)
  🔜 Implement VCG Pricing (game-theory truthful mechanism)
  🔜 Handle much larger sets of bids (100s of agents)
  🔜 Add unit tests to verify correctness

  Key insight from today:
  💡 In Scenario 1, the CRITICAL financial task (Agent-Beta)
     competed with a low-priority image resize (Agent-Alpha)
     for GPU-Slot-1. The AUCTION correctly gave it to Beta.
     Round-robin would have given it to Alpha (arrived first)!
""")


if __name__ == "__main__":
    run_demo()
