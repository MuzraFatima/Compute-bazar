"""
Mechanisms for bidders who don't know their own value: empirical study.

Each round, 12 agents with heterogeneous calibration (see
compute_bazaar/uncertainty/agents.py) each submit one task with an *estimated*
value and need. Mechanisms differ in what they feed the auction and how they
settle (compute_bazaar/uncertainty/mechanisms.py). All mechanisms see the same
task stream per seed.

Also checks Proposition 1 (docs/theory.md) on every round of a values-only
variant: W*(v) − W(x̂, v) ≤ Σ_{i ∈ x̂ ∪ x*} |v̂_i − v_i|.

    python scripts/run_uncertainty_experiments.py --seeds 3 --rounds 200
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402

from backend.app.auction.engine import Bid, Resource, solve_exhaustive  # noqa: E402
from compute_bazaar.uncertainty import MECHANISMS, make_population  # noqa: E402

CAPACITY = {"gpu": 8, "cpu": 16, "ram": 16}


def run_mechanism(name: str, seed: int, rounds: int, n_agents: int) -> dict:
    agents = make_population(n_agents, seed=seed)
    rng = random.Random(10_000 + seed)
    mech = MECHANISMS[name]()
    welfare = revenue = 0.0
    wins = fails = ir_viol = 0
    utils: list[float] = []
    for k in range(rounds):
        tasks = [a.draw_task(rng, k, CAPACITY) for a in agents]
        out = mech.run_round(tasks, CAPACITY)
        welfare += out.realized_welfare
        revenue += sum(out.payments.values())
        wins += len(out.winners)
        fails += len(out.failed)
        ir_viol += sum(1 for u in out.utilities.values() if u < -1e-6)
        utils.extend(out.utilities.values())
    return {
        "welfare": welfare,
        "revenue": revenue,
        "overrun_rate": fails / max(1, wins),
        "ex_post_ir_violation_rate": ir_viol / max(1, wins),
        "mean_winner_utility": statistics.mean(utils) if utils else 0.0,
        "wins": wins,
    }


def check_prop1(seed: int, rounds: int, n_agents: int) -> dict:
    """Values-only setting (true needs): verify the miscalibration welfare bound."""
    agents = make_population(n_agents, seed=seed)
    rng = random.Random(20_000 + seed)
    res = {r: Resource(r, r, units=u) for r, u in CAPACITY.items()}
    worst_ratio, total_loss, total_bound = 0.0, 0.0, 0.0
    for k in range(rounds):
        tasks = [a.draw_task(rng, k, CAPACITY) for a in agents]
        true_bids = [Bid(t.task_id, t.agent, list(t.true_need), t.true_value, quantities=t.true_need) for t in tasks]
        est_bids = [Bid(t.task_id, t.agent, list(t.true_need), t.est_value, quantities=t.true_need) for t in tasks]
        x_star, w_star = solve_exhaustive(true_bids, res)
        x_hat, _ = solve_exhaustive(est_bids, res)
        tv = {t.task_id: t.true_value for t in tasks}
        err = {t.task_id: abs(t.est_value - t.true_value) for t in tasks}
        loss = w_star - sum(tv[b.bid_id] for b in x_hat)
        bound = sum(err[i] for i in {b.bid_id for b in x_hat} | {b.bid_id for b in x_star})
        assert loss <= bound + 1e-6, (loss, bound)
        total_loss += loss
        total_bound += bound
        if bound > 0:
            worst_ratio = max(worst_ratio, loss / bound)
    return {
        "rounds_checked": rounds,
        "violations": 0,
        "mean_loss_over_bound": total_loss / max(total_bound, 1e-9),
        "max_loss_over_bound": worst_ratio,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--agents", type=int, default=12)
    ap.add_argument("--out", default="results/uncertainty_experiments.json")
    args = ap.parse_args()
    logger.remove()

    t0 = time.perf_counter()
    runs: dict[str, list[dict]] = {n: [] for n in MECHANISMS}
    for seed in range(args.seeds):
        for name in MECHANISMS:
            runs[name].append(run_mechanism(name, seed, args.rounds, args.agents))
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    oracle = [r["welfare"] for r in runs["oracle"]]
    summary = {}
    for name, rs in runs.items():
        pct = [r["welfare"] / o for r, o in zip(rs, oracle, strict=True)]
        summary[name] = {
            "welfare": round(statistics.mean(r["welfare"] for r in rs), 1),
            "welfare_pct_of_oracle": round(statistics.mean(pct), 4),
            "welfare_pct_std": round(statistics.stdev(pct), 4) if len(pct) > 1 else 0.0,
            "overrun_rate": round(statistics.mean(r["overrun_rate"] for r in rs), 4),
            "ex_post_ir_violation_rate": round(statistics.mean(r["ex_post_ir_violation_rate"] for r in rs), 4),
            "mean_winner_utility": round(statistics.mean(r["mean_winner_utility"] for r in rs), 2),
            "revenue": round(statistics.mean(r["revenue"] for r in rs), 1),
        }
    prop1 = [check_prop1(s, args.rounds, args.agents) for s in range(args.seeds)]
    out = {
        "setup": {"seeds": args.seeds, "rounds": args.rounds, "agents": args.agents, "capacity": CAPACITY},
        "summary": summary,
        "proposition1_check": prop1,
        "runs": runs,
        "runtime_sec": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        "| Mechanism | Welfare (% of oracle) | Overrun rate | Ex-post IR violations | Mean winner utility | Revenue |"
    )
    print("|:--|--:|--:|--:|--:|--:|")
    for n, s in summary.items():
        print(
            f"| {n} | {s['welfare_pct_of_oracle']:.1%} ± {s['welfare_pct_std']:.1%} | {s['overrun_rate']:.1%} "
            f"| {s['ex_post_ir_violation_rate']:.1%} | {s['mean_winner_utility']:.1f} | {s['revenue']:,.0f} |"
        )
    print("Prop 1:", prop1)


if __name__ == "__main__":
    main()
