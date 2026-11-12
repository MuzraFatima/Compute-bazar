"""
Large randomized incentive check: agent-level VCG vs the old per-bid VCG.

For N random multi-unit instances (half OR, half XOR bidding) we search a
family of misreports for every agent and record the best utility gain over
truthful reporting. Agent-level VCG should never show a positive gain; the
per-bid variant the engine used before Phase 1 is included to measure how
often it could be gamed.

    python scripts/verify_mechanism.py --instances 10000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from backend.app.auction.engine import solve_exhaustive  # noqa: E402
from compute_bazaar.mechanism.verification import (  # noqa: E402
    Instance,
    _misreports,
    agent_utility,
    best_deviation_gain,
    random_instance,
)


def per_bid_vcg(inst: Instance, reported):
    """The pre-Phase-1 rule: Clarke pivot that removes one bid at a time."""
    winners, value = solve_exhaustive(reported, inst.resources, inst.xor_per_agent)
    pay = defaultdict(float)
    for w in winners:
        rest = [b for b in reported if b.bid_id != w.bid_id]
        _, v_wo = solve_exhaustive(rest, inst.resources, inst.xor_per_agent)
        pay[w.agent_id] += max(0.0, min(v_wo - (value - w.bid_amount), w.bid_amount))
    return winners, dict(pay)


def per_bid_gain(inst: Instance, agent: str) -> float:
    others = [b for b in inst.true_bids if b.agent_id != agent]
    own = [b for b in inst.true_bids if b.agent_id == agent]
    w, p = per_bid_vcg(inst, inst.true_bids)
    truthful = agent_utility(inst, agent, w, p)
    best = float("-inf")
    for report in _misreports(own):
        w2, p2 = per_bid_vcg(inst, others + report)
        best = max(best, agent_utility(inst, agent, w2, p2) - truthful)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/mechanism_verification.json")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    rng = random.Random(args.seed)
    stats = {
        "agent_vcg": {"checks": 0, "violations": 0, "max_gain": 0.0},
        "per_bid_vcg": {"checks": 0, "violations": 0, "max_gain": 0.0},
    }
    t0 = time.perf_counter()
    for k in range(args.instances):
        inst = random_instance(rng, n_agents=4, n_resources=3, max_bids_per_agent=2, xor_per_agent=bool(k % 2))
        for agent in inst.agents:
            for name, fn in (("agent_vcg", best_deviation_gain), ("per_bid_vcg", per_bid_gain)):
                gain = fn(inst, agent)
                s = stats[name]
                s["checks"] += 1
                s["max_gain"] = max(s["max_gain"], gain)
                if gain > 1e-4:
                    s["violations"] += 1
        if (k + 1) % 1000 == 0:
            print(f"{k + 1}/{args.instances} instances, {time.perf_counter() - t0:.0f}s", flush=True)

    for s in stats.values():
        s["violation_rate"] = s["violations"] / max(1, s["checks"])
        s["max_gain"] = round(s["max_gain"], 6)
    out = {
        "instances": args.instances,
        "seed": args.seed,
        "agents_per_instance": 4,
        "resources_per_instance": 3,
        "misreport_family": "independent scaling of each bid by {0 (drop), .5, .8, .95, 1.05, 1.25, 2}",
        "results": stats,
        "runtime_sec": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["results"], indent=2))


if __name__ == "__main__":
    main()
