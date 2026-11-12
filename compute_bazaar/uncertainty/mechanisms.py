"""
Mechanisms for estimated valuations and needs.

Every mechanism runs the same pipeline per round:

  1. choose the value and demand fed to the auction for each task
     (raw estimate, or a calibrated / robust version),
  2. solve the multi-unit WDP exactly,
  3. execute: a winner whose *true* need exceeds what it was allocated on
     any resource overruns and fails (realized value 0),
  4. settle payments.

Settlement rules
  point     standard VCG on reported values: p_a = V*₋ₐ − (V* − v̂_a)
  contingent execution-contingent VCG (Mezzetti 2004; Porter et al. 2008):
            p_a = V*₋ₐ − Σ_{j ≠ a, j won} v_j^realized
            Agent a's utility becomes realized welfare − V*₋ₐ, a term it
            can't influence, so reporting its honest expectation is optimal
            in expectation (docs/theory.md, Proposition 2).

Calibration (``CalibratedVCG``) learns, per agent and from that agent's own
past executions only, a log-bias for value and a coverage-correcting inflation
for needs. It uses only past rounds, so an agent's current report can't move
its own correction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from backend.app.auction.engine import Bid, Resource, compute_vcg_agent_payments, solve_exhaustive, solve_wdp_lp
from compute_bazaar.uncertainty.agents import Task


@dataclass
class RoundOutcome:
    winners: list[Task]
    failed: list[Task]
    payments: dict[str, float]
    realized_welfare: float
    utilities: dict[str, float]


@dataclass
class Mechanism:
    name: str = "base"
    settlement: str = "point"  # "point" | "contingent"
    demand: str = "estimate"  # "estimate" | "upper90" | "truth"
    value: str = "estimate"  # "estimate" | "truth"
    history: dict = field(default_factory=lambda: defaultdict(list))

    # --- hooks for subclasses -------------------------------------------
    def reported_value(self, t: Task) -> float:
        return t.true_value if self.value == "truth" else t.est_value

    def reported_demand(self, t: Task) -> dict[str, int]:
        if self.demand == "truth":
            return dict(t.true_need)
        if self.demand == "upper90":
            return dict(t.need_upper90)
        return dict(t.est_need)

    def bid_menu(self, t: Task) -> list[tuple[dict[str, int], float]]:
        """XOR alternatives (demand, value) for one task. Default: a single bid."""
        return [(self.reported_demand(t), self.reported_value(t))]

    def observe(self, t: Task, succeeded: bool) -> None:
        """Called after execution for each winner; base class learns nothing."""

    # --- round ----------------------------------------------------------
    def run_round(self, tasks: list[Task], capacity: dict[str, int]) -> RoundOutcome:
        resources = {r: Resource(r, r, units=u) for r, u in capacity.items()}
        by_id = {t.task_id: t for t in tasks}
        bids, alloc, owner = [], {}, {}
        for t in tasks:
            menu = self.bid_menu(t)
            for k, (demand, val) in enumerate(menu):
                dem = {r: min(q, capacity[r]) for r, q in demand.items() if q > 0}
                if val <= 0 or not dem:
                    continue
                bid_id = f"{t.task_id}#{k}"
                group = t.task_id if len(menu) > 1 else None
                bids.append(Bid(bid_id, t.agent, list(dem), round(val, 6), quantities=dem, xor_group=group))
                alloc[bid_id] = dem
                owner[bid_id] = t.task_id

        # Brute force is exact and fast for small rounds; menus need the ILP.
        solver = solve_exhaustive if len(bids) <= 18 else solve_wdp_lp
        winners_bids, v_star = solver(bids, resources)
        agent_pay = compute_vcg_agent_payments(winners_bids, bids, resources, solver=solver, optimal_value=v_star)

        winners = [by_id[owner[b.bid_id]] for b in winners_bids]
        alloc = {owner[b.bid_id]: alloc[b.bid_id] for b in winners_bids}
        failed, realized = [], {}
        for t in winners:
            ok = all(alloc[t.task_id].get(r, 0) >= q for r, q in t.true_need.items())
            realized[t.agent] = t.true_value if ok else 0.0
            if not ok:
                failed.append(t)
            self.observe(t, ok)
        welfare = sum(realized.values())

        payments = {}
        if self.settlement == "point":
            payments = dict(agent_pay)
        else:
            reported = {b.agent_id: b.bid_amount for b in winners_bids}
            for t in winners:
                # V*₋ₐ = p_a(point) + (V* − v̂_a); then subtract others' realized values.
                v_minus_a = agent_pay[t.agent] + (v_star - reported[t.agent])
                payments[t.agent] = v_minus_a - (welfare - realized[t.agent])
        utilities = {t.agent: realized[t.agent] - payments.get(t.agent, 0.0) for t in winners}
        return RoundOutcome(winners, failed, payments, welfare, utilities)


class PointVCG(Mechanism):
    def __init__(self, name="vcg_point", demand="estimate"):
        super().__init__(name=name, settlement="point", demand=demand)


class ExecutionContingentVCG(Mechanism):
    def __init__(self, name="vcg_contingent", demand="estimate"):
        super().__init__(name=name, settlement="contingent", demand=demand)


class CalibratedVCG(Mechanism):
    """
    Contingent settlement + per-agent calibration + risk-adjusted XOR menus.

    Value: the shrunk mean of log(v̂ / v_realized) over the agent's past
    successful wins (prior strength ``n0``) is removed from new reports.

    Needs: rather than committing to one reservation size, each task bids a
    menu of XOR alternatives that reserve ``est × f`` units for f in
    ``factors``. Alternative f is valued at

        v_calibrated × P̂_a(true need ≤ ceil(est × f) on every resource)

    where P̂_a is the agent's empirical success rate at that factor (Beta
    prior). The exact WDP then trades overrun risk against capacity under
    current contention: in a quiet round tasks get a generous reservation,
    in a busy one the market accepts more risk on low-value tasks. Under
    contingent settlement, expected realized welfare is the right objective.
    """

    factors = (1.0, 1.25, 1.5, 2.0, 3.0)
    prior_success = (0.45, 0.65, 0.8, 0.92, 0.98)

    def __init__(self, name="vcg_calibrated", n0: float = 3.0, prior_strength: float = 4.0, menus: bool = True):
        super().__init__(name=name, settlement="contingent", demand="estimate")
        self.n0 = n0
        self.prior_strength = prior_strength
        self.menus = menus
        self.value_logs: dict[str, list[float]] = defaultdict(list)
        self.need_ratios: dict[str, list[dict[str, float]]] = defaultdict(list)

    def _bias(self, agent: str) -> float:
        xs = self.value_logs[agent]
        return sum(xs) / (len(xs) + self.n0) if xs else 0.0

    @staticmethod
    def _reserve(est: dict[str, int], f: float) -> dict[str, int]:
        return {r: max(q, math.ceil(q * f - 1e-9)) for r, q in est.items()}

    def success_prob(self, agent: str, k: int) -> float:
        f = self.factors[k]
        hist = self.need_ratios[agent]
        hits = sum(1 for ratios in hist if all(x <= f + 1e-9 for x in ratios.values()))
        a = self.prior_strength
        return (hits + a * self.prior_success[k]) / (len(hist) + a)

    def reported_value(self, t: Task) -> float:
        return t.est_value * math.exp(-self._bias(t.agent))

    def bid_menu(self, t: Task) -> list[tuple[dict[str, int], float]]:
        v = self.reported_value(t)
        if not self.menus:
            k = 2  # fixed 1.5x reservation
            return [(self._reserve(t.est_need, self.factors[k]), v)]
        menu, seen = [], set()
        for k, f in enumerate(self.factors):
            dem = self._reserve(t.est_need, f)
            key = tuple(sorted(dem.items()))
            if key in seen:
                continue
            seen.add(key)
            menu.append((dem, v * self.success_prob(t.agent, k)))
        return menu

    def observe(self, t: Task, succeeded: bool) -> None:
        # True needs are observed after running (a job that overran reports its peak).
        self.need_ratios[t.agent].append({r: q / max(1, t.est_need.get(r, 1)) for r, q in t.true_need.items()})
        if succeeded:
            self.value_logs[t.agent].append(math.log(t.est_value / t.true_value))


def oracle() -> Mechanism:
    return Mechanism(name="oracle", settlement="point", demand="truth", value="truth")


MECHANISMS = {
    "vcg_point": lambda: PointVCG(),
    "vcg_point_upper90": lambda: PointVCG(name="vcg_point_upper90", demand="upper90"),
    "vcg_contingent": lambda: ExecutionContingentVCG(),
    "vcg_contingent_upper90": lambda: ExecutionContingentVCG(name="vcg_contingent_upper90", demand="upper90"),
    "vcg_calibrated_fixed": lambda: CalibratedVCG(name="vcg_calibrated_fixed", menus=False),
    "vcg_calibrated": lambda: CalibratedVCG(),
    "oracle": oracle,
}
