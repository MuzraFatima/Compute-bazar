"""
=============================================================
 COMPUTE BAZAAR — Week 2: LP-Based Winner Determination
=============================================================

 THE UPGRADE FROM WEEK 1:
 ─────────────────────────
 Week 1 used a "greedy" algorithm:
   Sort by value-per-resource → pick top bids one at a time.
   Fast, but NOT always optimal.

 Week 2 uses LINEAR PROGRAMMING (LP):
   Model the problem as a math optimization.
   A solver (CBC via PuLP) finds the EXACT optimal answer.
   Always optimal, handles 1000s of bids.

 THE MATH (plain English):
 ──────────────────────────
 We have:
   B = set of all bids  (each bid i has a value v_i and a bundle S_i)
   R = set of resources

 We want to find binary variables x_i ∈ {0, 1}:
   x_i = 1 means bid i WINS
   x_i = 0 means bid i LOSES

 Maximize:   Σ  v_i * x_i          ← total value extracted
              i

 Subject to: Σ  x_i  ≤ 1  for each resource r ∈ R
              i: r ∈ S_i
              ↑ Each resource allocated to at most ONE winner

             x_i ∈ {0, 1} for all i  ← binary: win or lose

 This is an Integer Linear Program (ILP), solved by CBC solver.

 VCG PRICING:
 ─────────────
 After finding winners, compute each winner's payment:

   payment_i = [Optimal value WITHOUT bid i in the auction]
             - [Optimal value of OTHER winners when i wins]

 This equals the "externality" bid i imposes on others.
 Result: truthful bidding is always the best strategy!
=============================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum

import pulp
from loguru import logger


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

class SolverMethod(str, Enum):
    GREEDY = "greedy"   # Week 1 approach — fast approximation
    LP     = "lp"       # Week 2 approach — exact optimal solution


@dataclass
class Resource:
    """A compute resource agents can bid on."""
    name: str
    resource_type: str   # "gpu", "api_budget", "ram", "cpu"
    capacity: float = 1.0
    unit: str = "slot"

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Resource({self.name}, {self.resource_type})"


@dataclass
class Bid:
    """
    A combinatorial bid: agent wants this BUNDLE of resources,
    and values the bundle at bid_amount dollars.
    """
    bid_id: str           # Unique identifier
    agent_id: str         # Who is bidding
    resource_bundle: List[str]  # Resource names in this bundle
    bid_amount: float     # Agent's declared value for this bundle
    task_description: str = ""
    priority: str = "normal"

    def __post_init__(self):
        if self.bid_amount <= 0:
            raise ValueError(f"Bid amount must be positive, got {self.bid_amount}")
        if not self.resource_bundle:
            raise ValueError("Resource bundle cannot be empty")

    def __repr__(self):
        bundle = "+".join(self.resource_bundle)
        return f"Bid({self.agent_id} bids ${self.bid_amount} for [{bundle}])"


@dataclass
class AllocationResult:
    """The outcome of one auction round."""
    round_id: int
    winning_bids: List[Bid]
    payments: Dict[str, float]        # bid_id → payment amount
    total_value: float
    solver_method: SolverMethod
    solve_time_ms: float
    unallocated_resources: List[str]

    # Comparison metrics
    greedy_value: float = 0.0
    lp_value: float = 0.0

    @property
    def efficiency_vs_greedy(self) -> float:
        """How much better LP is than greedy (%)."""
        if self.greedy_value == 0:
            return 0.0
        return (self.lp_value - self.greedy_value) / self.greedy_value * 100


# ─────────────────────────────────────────────────────────────
# GREEDY SOLVER (Week 1 reference — kept for benchmarking)
# ─────────────────────────────────────────────────────────────

def solve_greedy(bids: List[Bid], resources: Dict[str, Resource]) -> Tuple[List[Bid], float]:
    """
    Greedy WDP: sort by value-per-resource, pick greedily.
    Returns (winning_bids, total_value).
    Fast O(n log n) but not always optimal.
    """
    sorted_bids = sorted(
        bids,
        key=lambda b: b.bid_amount / len(b.resource_bundle),
        reverse=True
    )

    allocated: Set[str] = set()
    winners: List[Bid] = []
    total = 0.0

    for bid in sorted_bids:
        bundle = set(bid.resource_bundle)
        # Validate all resources exist
        if not bundle.issubset(resources.keys()):
            continue
        if bundle.isdisjoint(allocated):
            winners.append(bid)
            allocated.update(bundle)
            total += bid.bid_amount

    return winners, total


# ─────────────────────────────────────────────────────────────
# LP SOLVER — Exact Winner Determination Problem
# ─────────────────────────────────────────────────────────────

def solve_wdp_lp(
    bids: List[Bid],
    resources: Dict[str, Resource],
) -> Tuple[List[Bid], float]:
    """
    Solve the Winner Determination Problem using Integer Linear Programming.

    This finds the EXACT optimal allocation — the combination of bids
    that maximizes total value without any resource conflicts.

    Uses PuLP with the CBC (Coin-or Branch and Cut) solver.
    CBC is free, open-source, and handles problems with 10,000+ variables.

    Returns:
        (winning_bids, total_value)
    """
    if not bids:
        return [], 0.0

    # Filter bids: only keep bids whose resources all exist
    valid_bids = [
        b for b in bids
        if all(r in resources for r in b.resource_bundle)
    ]

    if not valid_bids:
        logger.warning("No valid bids (all resources missing)")
        return [], 0.0

    # ── Step 1: Create the LP problem ──────────────────────────
    prob = pulp.LpProblem("WinnerDetermination", pulp.LpMaximize)

    # ── Step 2: Decision variables x_i ∈ {0, 1} ───────────────
    # x[i] = 1 → bid i wins; x[i] = 0 → bid i loses
    x = {}
    for bid in valid_bids:
        var = pulp.LpVariable(
            f"x_{bid.bid_id.replace('-', '_').replace(' ', '_')}",
            lowBound=0,
            upBound=1,
            cat="Binary",   # Must be 0 or 1 (Integer LP)
        )
        x[bid.bid_id] = var

    # ── Step 3: Objective — maximize total value ───────────────
    prob += pulp.lpSum(bid.bid_amount * x[bid.bid_id] for bid in valid_bids), "TotalValue"

    # ── Step 4: Constraints — no resource double-allocated ──────
    # For each resource r:  sum of x_i for all bids i containing r ≤ 1
    for res_name in resources:
        bids_wanting_this_resource = [
            bid for bid in valid_bids if res_name in bid.resource_bundle
        ]
        if bids_wanting_this_resource:
            prob += (
                pulp.lpSum(x[bid.bid_id] for bid in bids_wanting_this_resource) <= 1,
                f"Resource_{res_name.replace('-', '_').replace(' ', '_')}"
            )

    # ── Step 5: Solve! ─────────────────────────────────────────
    # Use PuLP's default solver — it auto-detects CBC or any available solver.
    # Suppress verbose output with msg=0.
    solver = pulp.getSolver("PULP_CBC_CMD", msg=0, timeLimit=30)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Feasible"):
        logger.error(f"LP solver returned status: {status}")
        return [], 0.0

    # ── Step 6: Extract winners ─────────────────────────────────
    winners = [
        bid for bid in valid_bids
        if pulp.value(x[bid.bid_id]) is not None
        and round(pulp.value(x[bid.bid_id])) == 1
    ]
    total_value = pulp.value(prob.objective) or 0.0

    return winners, total_value


# ─────────────────────────────────────────────────────────────
# VCG PRICING ENGINE
# ─────────────────────────────────────────────────────────────

def compute_vcg_prices(
    winning_bids: List[Bid],
    all_bids: List[Bid],
    resources: Dict[str, Resource],
) -> Dict[str, float]:
    """
    Compute VCG (Vickrey-Clarke-Groves) payments for all winners.

    VCG Formula for winner i:
      payment_i = V*(bids - {bid_i}) - [V*(all_bids) - v_i]

    Where:
      V*(S)    = optimal total value from bid set S
      v_i      = bid_i's declared value
      bids - {bid_i} = all bids EXCEPT bid_i

    Intuition:
      You pay the EXTERNALITY you impose — the value others LOST
      because you took resources they could have used.

    Property:
      payment_i ≤ v_i always (winners never overpay their bid)
      Truthful bidding (v_i = true value) is dominant strategy.

    Returns:
        Dict mapping bid_id → VCG payment amount
    """
    if not winning_bids:
        return {}

    # Get the total value WITH all winners
    # (We already know this, but recalculate cleanly)
    total_value_with_all, _ = _solve_optimal(all_bids, resources)

    payments: Dict[str, float] = {}

    for winner in winning_bids:
        # ── Solve WITHOUT this winner's bid ──────────────────
        bids_without_winner = [b for b in all_bids if b.bid_id != winner.bid_id]
        value_without_winner, _ = _solve_optimal(bids_without_winner, resources)

        # ── Value of OTHER winners when this winner is included ──
        # = total optimal value - this winner's bid
        value_of_others_with_winner = total_value_with_all - winner.bid_amount

        # ── VCG payment ──────────────────────────────────────
        # payment = value_without_me - value_of_others_with_me
        vcg_payment = value_without_winner - value_of_others_with_winner

        # Clamp: payment ≥ 0 and ≤ bid amount (shouldn't go negative
        # in a well-formed auction, but clamp for safety)
        vcg_payment = max(0.0, min(vcg_payment, winner.bid_amount))

        payments[winner.bid_id] = round(vcg_payment, 4)

        logger.debug(
            f"VCG | {winner.agent_id} | "
            f"bid=${winner.bid_amount:.2f} | "
            f"V_without=${value_without_winner:.2f} | "
            f"V_others_with=${value_of_others_with_winner:.2f} | "
            f"payment=${vcg_payment:.2f}"
        )

    return payments


def _solve_optimal(
    bids: List[Bid],
    resources: Dict[str, Resource],
) -> Tuple[float, List[Bid]]:
    """Helper: run LP solver and return (optimal_value, winning_bids)."""
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
        resources: List[Resource],
        method: SolverMethod = SolverMethod.LP,
    ):
        self.resources: Dict[str, Resource] = {r.name: r for r in resources}
        self.method = method
        self._bids: List[Bid] = []
        self._round_counter = 0

    # ── Bid Management ──────────────────────────────────────

    def submit_bid(self, bid: Bid) -> bool:
        """
        Accept a bid. Returns True if valid, False if rejected.
        Validation: all resources in the bundle must exist.
        """
        unknown = [r for r in bid.resource_bundle if r not in self.resources]
        if unknown:
            logger.warning(f"Bid {bid.bid_id} rejected — unknown resources: {unknown}")
            return False

        # Prevent duplicate bid IDs
        if any(b.bid_id == bid.bid_id for b in self._bids):
            logger.warning(f"Bid {bid.bid_id} rejected — duplicate bid_id")
            return False

        self._bids.append(bid)
        logger.info(f"Bid accepted: {bid}")
        return True

    def get_bids(self) -> List[Bid]:
        return list(self._bids)

    def reset(self):
        """Clear bids for the next auction round."""
        self._bids.clear()

    # ── Auction Execution ───────────────────────────────────

    def run_auction(self, round_id: Optional[int] = None) -> AllocationResult:
        """
        Run one complete auction round:
          1. Solve WDP (find optimal winners)
          2. Compute VCG payments
          3. Return AllocationResult

        Args:
            round_id: If None, auto-increments internal counter.
        """
        self._round_counter += 1
        rid = round_id if round_id is not None else self._round_counter
        bids = list(self._bids)

        logger.info(f"[Round {rid}] Starting auction | {len(bids)} bids | method={self.method}")
        t_start = time.perf_counter()

        # ── Solve WDP ────────────────────────────────────────
        if self.method == SolverMethod.LP:
            winners, lp_value = solve_wdp_lp(bids, self.resources)
            _, greedy_value = solve_greedy(bids, self.resources)
        else:
            winners, greedy_value = solve_greedy(bids, self.resources)
            lp_value = greedy_value  # not computed separately in greedy mode

        # ── VCG Pricing ──────────────────────────────────────
        payments = compute_vcg_prices(winners, bids, self.resources)

        solve_ms = (time.perf_counter() - t_start) * 1000

        # ── Unallocated Resources ────────────────────────────
        allocated_resources = {r for w in winners for r in w.resource_bundle}
        unallocated = [r for r in self.resources if r not in allocated_resources]

        total_value = sum(b.bid_amount for b in winners)

        result = AllocationResult(
            round_id=rid,
            winning_bids=winners,
            payments=payments,
            total_value=total_value,
            solver_method=self.method,
            solve_time_ms=solve_ms,
            unallocated_resources=unallocated,
            greedy_value=greedy_value,
            lp_value=lp_value if self.method == SolverMethod.LP else greedy_value,
        )

        self._log_result(result)
        return result

    def _log_result(self, result: AllocationResult):
        logger.info(f"[Round {result.round_id}] Auction complete | "
                    f"winners={len(result.winning_bids)} | "
                    f"value=${result.total_value:.2f} | "
                    f"time={result.solve_time_ms:.1f}ms")
