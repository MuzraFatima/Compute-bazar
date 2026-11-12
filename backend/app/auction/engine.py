"""
=============================================================
 COMPUTE BAZAAR — Winner Determination and VCG Pricing
=============================================================

 Model
 ─────
 Resources r ∈ R each have an integer number of identical ``units`` u_r
 (default 1, i.e. a single indivisible slot). A bid i asks for a bundle with
 quantities q_ir (default 1 per listed resource) and declares value v_i.

 Winner determination (WDP), solved exactly as a binary ILP:

     max   Σ_i v_i x_i
     s.t.  Σ_i q_ir x_i ≤ u_r              ∀ r ∈ R        (multi-unit capacity)
           Σ_{i ∈ G} x_i ≤ 1               ∀ XOR group G  (bidding language)
           x_i ∈ {0, 1}

 Bidding language
 ────────────────
 By default bids are OR-combined: an agent may win several of its bids and
 its value is their sum. Bids that share ``Bid.xor_group`` (per agent) are
 mutually exclusive, and ``xor_per_agent=True`` puts all of an agent's bids in
 one XOR group, the standard language for expressing substitutes.

 VCG (Clarke pivot) pricing — per agent, not per bid
 ───────────────────────────────────────────────────
 The bidder in VCG is the agent. For agent a with winning bids W_a:

     p_a = V*(B \\ B_a) − [ V*(B) − Σ_{i ∈ W_a} v_i ]

 where B_a are ALL of a's bids. Removing only one bid (the previous
 behaviour) lets an agent's other bids fill the gap and under-charges it,
 which breaks truthfulness whenever an agent submits more than one bid.
 p_a is split across a's winning bids in proportion to their declared value
 so per-bid accounting still works.

 Under quasi-linear utility, exact WDP and no post-processing, truthful
 reporting is a dominant strategy (DSIC) and payments satisfy
 0 ≤ p_a ≤ Σ_{i ∈ W_a} v_i (individual rationality). Tier discounts and
 reserve floors in ``pricing.py`` run *after* VCG and are documented there as
 non-DSIC policy layers.
=============================================================
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import pulp
from loguru import logger

# Tolerance used when comparing objective values coming back from the solver.
EPS = 1e-6


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────


class SolverMethod(str, Enum):
    GREEDY = "greedy"  # fast approximation, kept as a baseline
    LP = "lp"  # exact binary ILP via CBC
    EXHAUSTIVE = "exhaustive"  # brute force; exact oracle for tiny instances / tests


@dataclass
class Resource:
    """A compute resource agents can bid on.

    ``capacity``/``unit`` describe the physical size of one unit (e.g. 32 GB);
    ``units`` is how many identical copies the market can hand out per round.
    """

    name: str
    resource_type: str  # "gpu", "api_budget", "ram", "cpu"
    capacity: float = 1.0
    unit: str = "slot"
    units: int = 1

    def __post_init__(self):
        if self.units < 0:
            raise ValueError(f"Resource {self.name} cannot have negative units")

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Resource({self.name}, {self.resource_type}, units={self.units})"


@dataclass
class Bid:
    """
    A combinatorial bid: an agent wants this BUNDLE of resources and values it
    at ``bid_amount``. ``quantities`` maps resource → units wanted (default 1
    for each resource in ``resource_bundle``).
    """

    bid_id: str
    agent_id: str
    resource_bundle: list[str]
    bid_amount: float
    task_description: str = ""
    priority: str = "normal"
    quantities: dict[str, int] | None = None
    xor_group: str | None = None

    def __post_init__(self):
        if self.bid_amount <= 0:
            raise ValueError(f"Bid amount must be positive, got {self.bid_amount}")
        if not self.resource_bundle:
            raise ValueError("Resource bundle cannot be empty")
        if self.quantities is not None:
            extra = set(self.quantities) - set(self.resource_bundle)
            if extra:
                raise ValueError(f"Quantities given for resources outside the bundle: {sorted(extra)}")
            if any(q < 1 for q in self.quantities.values()):
                raise ValueError("Quantities must be positive integers")

    def demand(self) -> dict[str, int]:
        """Units requested per resource."""
        q = self.quantities or {}
        return {r: int(q.get(r, 1)) for r in self.resource_bundle}

    def __repr__(self):
        bundle = "+".join(self.resource_bundle)
        return f"Bid({self.agent_id} bids ${self.bid_amount} for [{bundle}])"


@dataclass
class AllocationResult:
    """The outcome of one auction round."""

    round_id: int
    winning_bids: list[Bid]
    payments: dict[str, float]  # bid_id → payment amount
    total_value: float
    solver_method: SolverMethod
    solve_time_ms: float
    unallocated_resources: list[str]

    # Comparison metrics
    greedy_value: float = 0.0
    lp_value: float = 0.0

    # Mechanism bookkeeping
    agent_payments: dict[str, float] = field(default_factory=dict)  # agent_id → VCG payment
    units_used: dict[str, int] = field(default_factory=dict)

    @property
    def efficiency_vs_greedy(self) -> float:
        """How much better LP is than greedy (%)."""
        if self.greedy_value == 0:
            return 0.0
        return (self.lp_value - self.greedy_value) / self.greedy_value * 100


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────


def _valid_bids(bids: Sequence[Bid], resources: dict[str, Resource]) -> list[Bid]:
    """Bids whose resources all exist and whose demand fits total supply."""
    out = []
    for b in bids:
        dem = b.demand()
        if all(r in resources and dem[r] <= resources[r].units for r in dem):
            out.append(b)
    return out


def _xor_key(bid: Bid, xor_per_agent: bool) -> tuple[str, str] | None:
    if xor_per_agent:
        return (bid.agent_id, "*")
    if bid.xor_group is not None:
        return (bid.agent_id, bid.xor_group)
    return None


def _xor_groups(bids: Sequence[Bid], xor_per_agent: bool) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, b in enumerate(bids):
        key = _xor_key(b, xor_per_agent)
        if key is not None:
            groups[key].append(idx)
    return {k: v for k, v in groups.items() if len(v) > 1}


def is_feasible(winners: Sequence[Bid], resources: dict[str, Resource], xor_per_agent: bool = False) -> bool:
    """Check capacity and XOR constraints for a set of winning bids."""
    used: dict[str, int] = defaultdict(int)
    seen_groups: set[tuple[str, str]] = set()
    for b in winners:
        for r, q in b.demand().items():
            if r not in resources:
                return False
            used[r] += q
        key = _xor_key(b, xor_per_agent)
        if key is not None:
            if key in seen_groups:
                return False
            seen_groups.add(key)
    return all(used[r] <= resources[r].units for r in used)


def units_used(winners: Sequence[Bid]) -> dict[str, int]:
    used: dict[str, int] = defaultdict(int)
    for b in winners:
        for r, q in b.demand().items():
            used[r] += q
    return dict(used)


# ─────────────────────────────────────────────────────────────
# SOLVERS
# ─────────────────────────────────────────────────────────────

Solver = Callable[..., tuple[list[Bid], float]]


def solve_greedy(
    bids: Sequence[Bid], resources: dict[str, Resource], xor_per_agent: bool = False
) -> tuple[list[Bid], float]:
    """
    Greedy WDP: sort by value per requested unit, accept while feasible.
    Fast O(n log n) but not always optimal. Kept as a baseline.
    """
    ordered = sorted(
        _valid_bids(bids, resources),
        key=lambda b: b.bid_amount / sum(b.demand().values()),
        reverse=True,
    )
    remaining = {name: r.units for name, r in resources.items()}
    taken_groups: set[tuple[str, str]] = set()
    winners: list[Bid] = []
    total = 0.0
    for bid in ordered:
        key = _xor_key(bid, xor_per_agent)
        if key is not None and key in taken_groups:
            continue
        dem = bid.demand()
        if all(remaining[r] >= q for r, q in dem.items()):
            for r, q in dem.items():
                remaining[r] -= q
            if key is not None:
                taken_groups.add(key)
            winners.append(bid)
            total += bid.bid_amount
    return winners, total


def solve_wdp_lp(
    bids: Sequence[Bid], resources: dict[str, Resource], xor_per_agent: bool = False
) -> tuple[list[Bid], float]:
    """
    Solve the Winner Determination Problem exactly as a binary ILP (PuLP/CBC).

    Returns (winning_bids, total_value). Ties between optimal allocations are
    broken by the solver; welfare is unique, which is all VCG needs.
    """
    valid = _valid_bids(bids, resources)
    if not valid:
        return [], 0.0

    prob = pulp.LpProblem("WinnerDetermination", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(len(valid))]
    prob += pulp.lpSum(b.bid_amount * x[i] for i, b in enumerate(valid)), "TotalValue"

    by_resource: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, b in enumerate(valid):
        for r, q in b.demand().items():
            by_resource[r].append((i, q))
    for k, (r, terms) in enumerate(sorted(by_resource.items())):
        prob += pulp.lpSum(q * x[i] for i, q in terms) <= resources[r].units, f"cap_{k}"

    for k, idxs in enumerate(_xor_groups(valid, xor_per_agent).values()):
        prob += pulp.lpSum(x[i] for i in idxs) <= 1, f"xor_{k}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Feasible"):
        logger.error(f"LP solver returned status: {status}")
        return [], 0.0

    winners = [b for i, b in enumerate(valid) if (pulp.value(x[i]) or 0.0) > 0.5]
    return winners, sum(b.bid_amount for b in winners)


def solve_exhaustive(
    bids: Sequence[Bid], resources: dict[str, Resource], xor_per_agent: bool = False
) -> tuple[list[Bid], float]:
    """
    Exact WDP by enumerating subsets with pruning. Exponential, so only for
    small instances (≲ 18 bids): used as an independent oracle in tests and
    for fast truthfulness sweeps.
    """
    valid = _valid_bids(bids, resources)
    n = len(valid)
    if n > 22:
        raise ValueError(f"solve_exhaustive is for tiny instances; got {n} bids")
    demands = [b.demand() for b in valid]
    keys = [_xor_key(b, xor_per_agent) for b in valid]
    # suffix sums for a simple optimistic bound
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + valid[i].bid_amount

    best_value = 0.0
    best_set: list[int] = []
    remaining = {name: r.units for name, r in resources.items()}
    groups: set[tuple[str, str]] = set()
    chosen: list[int] = []

    def rec(i: int, value: float) -> None:
        nonlocal best_value, best_set
        if value > best_value + EPS:
            best_value, best_set = value, list(chosen)
        if i == n or value + suffix[i] <= best_value + EPS:
            return
        dem, key = demands[i], keys[i]
        if (key is None or key not in groups) and all(remaining[r] >= q for r, q in dem.items()):
            for r, q in dem.items():
                remaining[r] -= q
            if key is not None:
                groups.add(key)
            chosen.append(i)
            rec(i + 1, value + valid[i].bid_amount)
            chosen.pop()
            if key is not None:
                groups.discard(key)
            for r, q in dem.items():
                remaining[r] += q
        rec(i + 1, value)

    rec(0, 0.0)
    winners = [valid[i] for i in best_set]
    return winners, sum(b.bid_amount for b in winners)


SOLVERS: dict[SolverMethod, Solver] = {
    SolverMethod.GREEDY: solve_greedy,
    SolverMethod.LP: solve_wdp_lp,
    SolverMethod.EXHAUSTIVE: solve_exhaustive,
}


# ─────────────────────────────────────────────────────────────
# VCG PRICING ENGINE
# ─────────────────────────────────────────────────────────────


def compute_vcg_agent_payments(
    winning_bids: Sequence[Bid],
    all_bids: Sequence[Bid],
    resources: dict[str, Resource],
    xor_per_agent: bool = False,
    solver: Solver = solve_wdp_lp,
    optimal_value: float | None = None,
) -> dict[str, float]:
    """
    Clarke-pivot payment per winning agent:
        p_a = V*(B without any of a's bids) − (V*(B) − value a won)
    """
    if not winning_bids:
        return {}
    if optimal_value is None:
        _, optimal_value = solver(all_bids, resources, xor_per_agent)

    won_value: dict[str, float] = defaultdict(float)
    for b in winning_bids:
        won_value[b.agent_id] += b.bid_amount

    payments: dict[str, float] = {}
    for agent, value in won_value.items():
        others = [b for b in all_bids if b.agent_id != agent]
        _, v_without = solver(others, resources, xor_per_agent)
        p = v_without - (optimal_value - value)
        # Exact solvers give 0 ≤ p ≤ value; the clamp only absorbs float noise.
        payments[agent] = round(max(0.0, min(p, value)), 6)
        logger.debug(f"VCG | {agent} | won=${value:.2f} | V_without=${v_without:.2f} | payment=${p:.2f}")
    return payments


def split_agent_payments(winning_bids: Sequence[Bid], agent_payments: dict[str, float]) -> dict[str, float]:
    """Spread each agent's payment over its winning bids, proportional to value."""
    won_value: dict[str, float] = defaultdict(float)
    for b in winning_bids:
        won_value[b.agent_id] += b.bid_amount
    out = {}
    for b in winning_bids:
        total = won_value[b.agent_id]
        share = b.bid_amount / total if total > 0 else 0.0
        out[b.bid_id] = round(agent_payments.get(b.agent_id, 0.0) * share, 4)
    return out


def compute_vcg_prices(
    winning_bids: Sequence[Bid],
    all_bids: Sequence[Bid],
    resources: dict[str, Resource],
    xor_per_agent: bool = False,
    solver: Solver = solve_wdp_lp,
    optimal_value: float | None = None,
) -> dict[str, float]:
    """
    VCG payments keyed by bid_id (agent-level Clarke pivot, split per bid).

    Property: payment ≤ bid for every winning bid, payments are non-negative,
    and truthful reporting is a dominant strategy for each agent.
    """
    agent_payments = compute_vcg_agent_payments(winning_bids, all_bids, resources, xor_per_agent, solver, optimal_value)
    return split_agent_payments(winning_bids, agent_payments)


def _solve_optimal(bids: list[Bid], resources: dict[str, Resource]) -> tuple[float, list[Bid]]:
    """Helper kept for older callers: (optimal_value, winning_bids)."""
    winners, value = solve_wdp_lp(bids, resources)
    return value, winners


# ─────────────────────────────────────────────────────────────
# MAIN AUCTION ENGINE (Orchestrator)
# ─────────────────────────────────────────────────────────────


class AuctionEngine:
    """
    The central auction engine for Compute Bazaar.

    Usage:
        engine = AuctionEngine(resources, method=SolverMethod.LP)
        engine.submit_bid(bid)
        result = engine.run_auction(round_id=1)
    """

    def __init__(
        self,
        resources: list[Resource],
        method: SolverMethod = SolverMethod.LP,
        xor_per_agent: bool = False,
        compare_greedy: bool = True,
    ):
        self.resources: dict[str, Resource] = {r.name: r for r in resources}
        self.method = method
        self.xor_per_agent = xor_per_agent
        self.compare_greedy = compare_greedy
        self._bids: list[Bid] = []
        self._bid_ids: set[str] = set()
        self._round_counter = 0

    # ── Bid Management ──────────────────────────────────────

    def submit_bid(self, bid: Bid) -> bool:
        """
        Accept a bid. Returns True if valid, False if rejected.
        Rejects unknown resources, demand above total supply and duplicate ids.
        """
        unknown = [r for r in bid.resource_bundle if r not in self.resources]
        if unknown:
            logger.warning(f"Bid {bid.bid_id} rejected — unknown resources: {unknown}")
            return False

        too_big = [r for r, q in bid.demand().items() if q > self.resources[r].units]
        if too_big:
            logger.warning(f"Bid {bid.bid_id} rejected — asks for more units than exist: {too_big}")
            return False

        if bid.bid_id in self._bid_ids:
            logger.warning(f"Bid {bid.bid_id} rejected — duplicate bid_id")
            return False

        self._bids.append(bid)
        self._bid_ids.add(bid.bid_id)
        logger.info(f"Bid accepted: {bid}")
        return True

    def get_bids(self) -> list[Bid]:
        return list(self._bids)

    def reset(self):
        """Clear bids for the next auction round."""
        self._bids.clear()
        self._bid_ids.clear()

    # ── Auction Execution ───────────────────────────────────

    def run_auction(self, round_id: int | None = None) -> AllocationResult:
        """
        Run one complete auction round: solve the WDP, then compute VCG
        payments with the same solver (so payments are consistent with the
        allocation). Greedy mode prices with the exact ILP's counterfactuals
        only if it is also the allocation rule — i.e. greedy + "VCG" is a
        heuristic and not truthful.
        """
        self._round_counter += 1
        rid = round_id if round_id is not None else self._round_counter
        bids = list(self._bids)

        logger.info(f"[Round {rid}] Starting auction | {len(bids)} bids | method={self.method}")
        t_start = time.perf_counter()

        solver = SOLVERS[self.method]
        winners, value = solver(bids, self.resources, self.xor_per_agent)
        if self.method == SolverMethod.GREEDY:
            greedy_value = value
        elif self.compare_greedy:
            _, greedy_value = solve_greedy(bids, self.resources, self.xor_per_agent)
        else:
            greedy_value = 0.0

        agent_payments = compute_vcg_agent_payments(
            winners, bids, self.resources, self.xor_per_agent, solver, optimal_value=value
        )
        payments = split_agent_payments(winners, agent_payments)

        solve_ms = (time.perf_counter() - t_start) * 1000

        used = units_used(winners)
        unallocated = [r for r in self.resources if used.get(r, 0) == 0]

        result = AllocationResult(
            round_id=rid,
            winning_bids=winners,
            payments=payments,
            total_value=sum(b.bid_amount for b in winners),
            solver_method=self.method,
            solve_time_ms=solve_ms,
            unallocated_resources=unallocated,
            greedy_value=greedy_value,
            lp_value=value,
            agent_payments=agent_payments,
            units_used=used,
        )

        self._log_result(result)
        return result

    def _log_result(self, result: AllocationResult):
        logger.info(
            f"[Round {result.round_id}] Auction complete | "
            f"winners={len(result.winning_bids)} | "
            f"value=${result.total_value:.2f} | "
            f"time={result.solve_time_ms:.1f}ms"
        )


def enumerate_allocations(bids: Sequence[Bid], resources: dict[str, Resource], xor_per_agent: bool = False):
    """Yield every feasible allocation (tiny instances only; for debugging/tests)."""
    valid = _valid_bids(bids, resources)
    for k in range(len(valid) + 1):
        for combo in itertools.combinations(valid, k):
            if is_feasible(combo, resources, xor_per_agent):
                yield list(combo)
