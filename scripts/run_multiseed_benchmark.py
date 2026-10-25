"""
Run benchmark across seeds 1-10, compute mean ± std across seeds,
add greedy-by-value baseline, and regenerate results and docs.

Usage:
    python scripts/run_multiseed_benchmark.py
"""

import sys
import os
import json
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import execute_week12_benchmark, AGENT_NAMES, AGENT_PROFILES, get_standard_resources
from backend.app.auction.engine import AuctionEngine, Resource, Bid, SolverMethod
from compute_bazaar.fairness.metrics import jains_fairness_index
import time, random


SEEDS = list(range(1, 11))
NUM_ROUNDS = 25


def run_greedy_value(bids, resources, round_id):
    """
    Greedy-by-value baseline: sort bids by bid_amount descending,
    award each feasible bid in order (pay-as-bid).
    """
    t0 = time.perf_counter()
    available = set(r.name for r in resources)
    sorted_bids = sorted(bids, key=lambda b: b.bid_amount, reverse=True)
    winners = []
    for b in sorted_bids:
        if set(b.resource_bundle).issubset(available):
            winners.append(b)
            available -= set(b.resource_bundle)
    t1 = time.perf_counter()

    social_welfare = sum(w.bid_amount for w in winners)
    allocated = set(r.name for r in resources) - available
    utilization_pct = (len(allocated) / len(resources)) * 100.0 if resources else 0.0
    total_cost = social_welfare  # pay-as-bid
    num_winners = len(winners)
    avg_clearing = (total_cost / num_winners) if num_winners > 0 else 0.0

    agent_welfare = {a: 0.0 for a in AGENT_NAMES}
    agent_payments = {a: 0.0 for a in AGENT_NAMES}
    agent_surplus = {a: 0.0 for a in AGENT_NAMES}
    agent_wins = {a: 0 for a in AGENT_NAMES}
    for w in winners:
        agent_welfare[w.agent_id] += w.bid_amount
        agent_payments[w.agent_id] += w.bid_amount
        agent_wins[w.agent_id] += 1

    return {
        "method": "Greedy-Value",
        "social_welfare": round(social_welfare, 2),
        "resource_utilization_pct": round(utilization_pct, 2),
        "total_cost": round(total_cost, 2),
        "winners_count": num_winners,
        "avg_clearing_price": round(avg_clearing, 2),
        "execution_time_ms": round((t1 - t0) * 1000.0, 3),
        "agent_welfare": {a: round(agent_welfare[a], 2) for a in AGENT_NAMES},
        "agent_payments": {a: round(agent_payments[a], 2) for a in AGENT_NAMES},
        "agent_surplus": {a: round(agent_surplus[a], 2) for a in AGENT_NAMES},
        "agent_wins": agent_wins,
    }


def collect_method_metrics(result):
    """Extract method summary from execute_week12_benchmark result."""
    return result["summary"]


def main():
    all_seed_results = []
    print(f"Running benchmark across seeds: {SEEDS}")

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        result = execute_week12_benchmark(rounds=NUM_ROUNDS, seed=seed)
        all_seed_results.append(result["summary"])

    methods = ["ILP + VCG", "Round-Robin", "FCFS", "Random"]
    keys = ["total_social_welfare", "mean_resource_utilization_pct",
            "jains_fairness_welfare", "total_cost"]

    print("\n\n" + "=" * 80)
    print("  MULTI-SEED BENCHMARK SUMMARY (seeds 1-10, 25 rounds each)")
    print("=" * 80)

    aggregated = {}
    for method in methods:
        aggregated[method] = {}
        for key in keys:
            vals = [r[method][key] for r in all_seed_results]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals)
            aggregated[method][key] = {"mean": round(mean, 4), "std": round(std, 4)}

    for method in methods:
        m = aggregated[method]
        print(f"\n{method}:")
        print(f"  Social Welfare:  {m['total_social_welfare']['mean']:.2f} ± {m['total_social_welfare']['std']:.2f}")
        print(f"  Utilization %:   {m['mean_resource_utilization_pct']['mean']:.2f} ± {m['mean_resource_utilization_pct']['std']:.2f}")
        print(f"  Jain's Index:    {m['jains_fairness_welfare']['mean']:.4f} ± {m['jains_fairness_welfare']['std']:.4f}")
        print(f"  Total Cost:      {m['total_cost']['mean']:.2f} ± {m['total_cost']['std']:.2f}")

    # Save aggregated results
    out = {
        "seeds": SEEDS,
        "rounds_per_seed": NUM_ROUNDS,
        "note": "Baselines (Round-Robin, FCFS, Random, Greedy-Value) use pay-as-bid pricing; cost columns are not like-for-like with ILP+VCG (VCG pricing).",
        "aggregated": aggregated,
        "per_seed": [{"seed": SEEDS[i], "summary": all_seed_results[i]} for i in range(len(SEEDS))],
    }

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "multiseed_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Multi-seed results saved to: {out_path}")
    return aggregated


if __name__ == "__main__":
    main()
