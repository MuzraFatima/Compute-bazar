"""
Trace-driven comparison of scheduling policies.

    python scripts/run_trace_experiments.py                 # synthetic PAI-shaped, 5 seeds
    python scripts/run_trace_experiments.py --pai data/pai_task_table.csv --seeds 3

Writes results/trace_experiments.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from compute_bazaar.sim import POLICIES, ClusterSimulator, load_pai_trace, synthetic_pai_like  # noqa: E402

METRICS = ("welfare", "avg_jct", "ftf_mean", "ftf_p95", "ftf_jain", "utilization", "dropped", "decision_ms_mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1200)
    ap.add_argument("--users", type=int, default=120)
    ap.add_argument("--gpus", type=int, default=64)
    ap.add_argument("--pai", help="path to Alibaba pai_task_table.csv (optional)")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--out", default="results/trace_experiments.json")
    args = ap.parse_args()
    logger.remove()

    names = args.policies.split(",")
    runs: dict[str, list[dict]] = {n: [] for n in names}
    t0 = time.perf_counter()
    for seed in range(args.seeds):
        if args.pai:
            jobs = load_pai_trace(args.pai, limit=args.jobs, seed=seed)
            source = "alibaba-pai-2020"
        else:
            jobs = synthetic_pai_like(n_jobs=args.jobs, n_users=args.users, seed=seed)
            source = "synthetic-pai-like"
        sim = ClusterSimulator(jobs, capacity={"gpu": args.gpus, "cpu": args.gpus * 8, "ram": args.gpus * 32})
        for n in names:
            runs[n].append(sim.run(POLICIES[n], n).as_dict())
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    summary = {}
    for n, rs in runs.items():
        summary[n] = {
            m: {
                "mean": round(statistics.mean(r[m] for r in rs), 4),
                "std": round(statistics.stdev(r[m] for r in rs), 4) if len(rs) > 1 else 0.0,
            }
            for m in METRICS
        }
    out = {
        "source": source,
        "seeds": args.seeds,
        "jobs_per_seed": args.jobs,
        "users": args.users,
        "cluster": {"gpu": args.gpus, "cpu": args.gpus * 8, "ram_gb": args.gpus * 32},
        "epoch_minutes": 10,
        "summary": summary,
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(
        "\n| Policy | Welfare | Avg JCT (epochs) | FTF rho mean | FTF rho p95 | Jain (1/rho) | GPU util | Dropped | Decision ms |"
    )
    print("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for n in names:
        s = summary[n]
        print(
            f"| {n} | {s['welfare']['mean']:,.0f} ± {s['welfare']['std']:,.0f} | {s['avg_jct']['mean']:.1f} "
            f"| {s['ftf_mean']['mean']:.2f} | {s['ftf_p95']['mean']:.1f} | {s['ftf_jain']['mean']:.3f} "
            f"| {s['utilization']['mean']:.1%} | {s['dropped']['mean']:.0f} | {s['decision_ms_mean']['mean']:.1f} |"
        )


if __name__ == "__main__":
    main()
