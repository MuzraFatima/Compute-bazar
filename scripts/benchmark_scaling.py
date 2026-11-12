"""
Solver scaling: winner determination and full VCG pricing vs number of bids.

Random multi-unit instances: 20 resources with 2–8 units each; every agent
submits two OR bids over 1–4 resources, valued per unit with noise. That
makes many near-ties, a hard case for branch-and-bound. Configurations:

  exact-1   exact CBC, counterfactuals solved one after another
  exact-8   exact CBC, counterfactuals in 8 threads (CBC runs as a subprocess)
  gap1%-8   1% relative MIP gap, 8 threads: ε-approximate VCG

For the gap configuration we also report the largest payment deviation from
exact VCG, relative to the optimal welfare V*.

    python scripts/benchmark_scaling.py --sizes 50,100,200,400
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from backend.app.auction import engine as eng  # noqa: E402

CONFIGS = {"exact-1": ("cbc", 0.0, 1), "exact-8": ("cbc", 0.0, 8), "gap1%-8": ("cbc", 0.01, 8)}


def instance(n_bids: int, seed: int, n_resources: int = 20):
    rng = random.Random(seed)
    resources = {f"R{k}": eng.Resource(f"R{k}", "gpu", units=rng.randint(2, 8)) for k in range(n_resources)}
    names = list(resources)
    bids = []
    for i in range(n_bids):
        bundle = rng.sample(names, rng.randint(1, 4))
        qty = {r: rng.randint(1, max(1, resources[r].units // 2)) for r in bundle}
        value = sum(qty.values()) * rng.uniform(5, 15)
        bids.append(eng.Bid(f"b{i}", f"a{i // 2}", bundle, round(value, 2), quantities=qty))
    return bids, resources


def configure(solver: str, gap: float) -> None:
    os.environ["COMPUTE_BAZAAR_SOLVER"] = solver
    os.environ["COMPUTE_BAZAAR_MIP_GAP"] = str(gap)
    eng._SOLVER_CACHE.clear()


def run(bids, resources, workers: int):
    t0 = time.perf_counter()
    winners, value = eng.solve_wdp_lp(bids, resources)
    t1 = time.perf_counter()
    pay = eng.compute_vcg_agent_payments(winners, bids, resources, optimal_value=value, workers=workers)
    t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t0) * 1000, value, pay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="50,100,200,400")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--out", default="results/scaling_benchmark.json")
    args = ap.parse_args()
    logger.remove()

    rows = []
    for n in map(int, args.sizes.split(",")):
        exact_pay: dict[int, tuple[float, dict]] = {}
        for cfg in args.configs.split(","):
            solver, gap, workers = CONFIGS[cfg]
            configure(solver, gap)
            samples, deviations = [], []
            for seed in range(args.repeats):
                wdp, total, value, pay = run(*instance(n, seed), workers)
                samples.append((wdp, total))
                if gap == 0.0:
                    exact_pay[seed] = (value, pay)
                elif seed in exact_pay:
                    v_exact, p_exact = exact_pay[seed]
                    agents = set(p_exact) | set(pay)
                    dev = max((abs(pay.get(a, 0.0) - p_exact.get(a, 0.0)) for a in agents), default=0.0)
                    deviations.append(dev / v_exact if v_exact else 0.0)
            row = {
                "config": cfg,
                "bids": n,
                "wdp_ms": round(statistics.median(s[0] for s in samples), 1),
                "vcg_total_ms": round(statistics.median(s[1] for s in samples), 1),
            }
            if deviations:
                row["max_payment_dev_over_vstar"] = round(max(deviations), 5)
            rows.append(row)
            print(row, flush=True)
    configure("cbc", 0.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"instances": "20 resources, 2-8 units, 2 OR bids per agent", "rows": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
