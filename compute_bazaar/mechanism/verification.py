"""
Empirical incentive checks for the combinatorial VCG mechanism.

An ``Instance`` holds each agent's *true* bids (bundle + true value). We run
the mechanism on reported bids and measure an agent's quasi-linear utility
against its true values:

    u_a = Σ_{won bids i} true_value_i − p_a

``best_deviation_gain`` searches a family of misreports for one agent
(scaling each bid up/down, dropping bids) and returns the largest utility
gain over truthful reporting. For an exact VCG mechanism this must be ≤ 0
up to float noise; positive values flag a broken incentive property.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, replace

from backend.app.auction.engine import (
    Bid,
    Resource,
    Solver,
    compute_vcg_agent_payments,
    solve_exhaustive,
)

SCALES = (0.0, 0.5, 0.8, 0.95, 1.05, 1.25, 2.0)


@dataclass
class Instance:
    resources: dict[str, Resource]
    true_bids: list[Bid]
    xor_per_agent: bool = False

    @property
    def agents(self) -> list[str]:
        return sorted({b.agent_id for b in self.true_bids})


def random_instance(
    rng: random.Random,
    n_agents: int = 4,
    n_resources: int = 3,
    max_bids_per_agent: int = 2,
    max_units: int = 2,
    xor_per_agent: bool = False,
) -> Instance:
    resources = {f"R{j}": Resource(f"R{j}", "gpu", units=rng.randint(1, max_units)) for j in range(n_resources)}
    names = list(resources)
    bids = []
    for a in range(n_agents):
        for k in range(rng.randint(1, max_bids_per_agent)):
            bundle = rng.sample(names, rng.randint(1, min(3, n_resources)))
            qty = {r: rng.randint(1, resources[r].units) for r in bundle}
            bids.append(
                Bid(
                    bid_id=f"a{a}b{k}",
                    agent_id=f"A{a}",
                    resource_bundle=bundle,
                    bid_amount=round(rng.uniform(1, 100), 2),
                    quantities=qty,
                )
            )
    return Instance(resources, bids, xor_per_agent)


def run_vcg(inst: Instance, reported: list[Bid], solver: Solver = solve_exhaustive):
    winners, value = solver(reported, inst.resources, inst.xor_per_agent)
    payments = compute_vcg_agent_payments(
        winners, reported, inst.resources, inst.xor_per_agent, solver, optimal_value=value
    )
    return winners, payments, value


def agent_utility(inst: Instance, agent: str, winners: list[Bid], payments: dict[str, float]) -> float:
    true_value = {b.bid_id: b.bid_amount for b in inst.true_bids}
    won = sum(true_value[b.bid_id] for b in winners if b.agent_id == agent)
    return won - payments.get(agent, 0.0)


def _misreports(own: list[Bid]):
    """Scale each of the agent's bids independently; scale 0 means 'drop it'."""
    for scales in itertools.product(SCALES, repeat=len(own)):
        if all(s == 1.0 for s in scales):
            continue
        out = [replace(b, bid_amount=round(b.bid_amount * s, 4)) for b, s in zip(own, scales, strict=True) if s > 0]
        yield out


def best_deviation_gain(inst: Instance, agent: str, solver: Solver = solve_exhaustive) -> float:
    others = [b for b in inst.true_bids if b.agent_id != agent]
    own = [b for b in inst.true_bids if b.agent_id == agent]
    winners, payments, _ = run_vcg(inst, inst.true_bids, solver)
    truthful = agent_utility(inst, agent, winners, payments)
    best = float("-inf")
    for report in _misreports(own):
        w, p, _ = run_vcg(inst, others + report, solver)
        best = max(best, agent_utility(inst, agent, w, p) - truthful)
    return best
