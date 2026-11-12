"""
Core-selecting payments (minimum-revenue core, VCG-nearest).

VCG revenue can collapse when losing bidders collude or one bidder splits
into shills: in the LLG example (local bidders want A and B, a global bidder
wants AB), locals who jointly overbid can win both items while paying almost
nothing. Core-selecting auctions (Day & Milgrom 2008) charge the minimum
revenue such that no coalition of bidders would rather deal with the seller
at the reported bids:

    Σ_{i ∈ W \\ C} p_i ≥ V(C) − Σ_{i ∈ W ∩ C} b_i     for every coalition C,
    VCG_i ≤ p_i ≤ b_i.

We find the minimum-revenue point with constraint generation (Day &
Raghavan 2007): solve the LP over the constraints found so far, look for the
most-violated coalition with one WDP in which winners' bids are reduced by
their current surplus, add it, repeat. Among minimum-revenue core points we
then take the one closest to VCG in L1 (a second LP), which keeps payments
as close to truthful VCG as the core allows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import pulp

from backend.app.auction.engine import (
    Bid,
    Resource,
    Solver,
    compute_vcg_agent_payments,
    solve_exhaustive,
)


def core_payments(
    winners: Sequence[Bid],
    bids: Sequence[Bid],
    resources: dict[str, Resource],
    solver: Solver = solve_exhaustive,
    max_iter: int = 50,
) -> dict[str, float]:
    """Agent-level minimum-revenue core payments, VCG-nearest. Returns agent → payment."""
    if not winners:
        return {}
    vcg = compute_vcg_agent_payments(winners, bids, resources, solver=solver)
    won = defaultdict(float)
    for b in winners:
        won[b.agent_id] += b.bid_amount
    W = sorted(won)
    cuts: list[tuple[set[str], float]] = []  # (winners outside C, required revenue from them)
    p = dict(vcg)

    for _ in range(max_iter):
        # Separation: winners keep only their surplus (b - p) in the coalition's valuation.
        adjusted = [
            replace(
                b,
                bid_amount=max(1e-9, b.bid_amount - (won[b.agent_id] - p[b.agent_id]) * b.bid_amount / won[b.agent_id]),
            )
            if b.agent_id in won
            else b
            for b in bids
        ]
        y, z = solver(adjusted, resources)
        if z <= sum(p.values()) + 1e-6:
            break
        coalition = {b.agent_id for b in y}
        in_c = [a for a in W if a in coalition]
        v_c = sum(next(bb.bid_amount for bb in bids if bb.bid_id == b.bid_id) for b in y)
        rhs = v_c - sum(won[a] for a in in_c)
        cuts.append(({a for a in W if a not in coalition}, rhs))
        p = _solve_lp(W, vcg, won, cuts, nearest=False)

    total = sum(p.values())
    return _solve_lp(W, vcg, won, cuts, nearest=True, revenue=total) if cuts else p


def _solve_lp(W, vcg, won, cuts, nearest: bool, revenue: float | None = None) -> dict[str, float]:
    prob = pulp.LpProblem("core", pulp.LpMinimize)
    p = {a: pulp.LpVariable(f"p_{k}", lowBound=vcg[a], upBound=won[a]) for k, a in enumerate(W)}
    for k, (outside, rhs) in enumerate(cuts):
        if outside:
            prob += pulp.lpSum(p[a] for a in outside) >= rhs, f"core_{k}"
    if nearest:
        # Same (minimum) revenue, then closest to VCG; with p ≥ VCG the L1 distance is Σ(p − vcg),
        # so break ties by spreading the excess evenly (minimise the largest excess).
        prob += pulp.lpSum(p.values()) <= revenue + 1e-6, "min_revenue"
        t = pulp.LpVariable("max_excess", lowBound=0)
        for a in W:
            prob += p[a] - vcg[a] <= t
        prob += t
    else:
        prob += pulp.lpSum(p.values())
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return {a: round(float(pulp.value(p[a]) or 0.0), 6) for a in W}
