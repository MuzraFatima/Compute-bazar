"""
=============================================================
 COMPUTE BAZAAR — Pricing Utilities (Week 2)
=============================================================

 Beyond VCG, this module provides:
   - Minimum revenue pricing (reserve prices)
   - Discount tiers (Gold/Silver/Bronze agents pay less)
   - Budget feasibility checking
   - Payment summary reporting
=============================================================
"""

from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass

from .engine import Bid, AllocationResult


# ─────────────────────────────────────────────────────────────
# TIER DISCOUNTS
# ─────────────────────────────────────────────────────────────

TIER_DISCOUNTS = {
    "gold":   0.10,   # Gold agents get 10% off VCG price
    "silver": 0.05,   # Silver agents get 5% off
    "bronze": 0.00,   # No discount
}


@dataclass
class PricedPayment:
    bid_id: str
    agent_id: str
    bid_amount: float
    vcg_payment: float          # Raw VCG price
    discount_pct: float         # Tier discount applied
    final_payment: float        # What agent actually pays
    savings: float              # bid_amount - final_payment
    savings_pct: float          # savings as % of bid

    def __repr__(self):
        return (
            f"  {self.agent_id}: bid=${self.bid_amount:.2f} | "
            f"VCG=${self.vcg_payment:.2f} | "
            f"discount={self.discount_pct*100:.0f}% | "
            f"pays=${self.final_payment:.2f} | "
            f"saves=${self.savings:.2f}"
        )


def apply_tier_pricing(
    result: AllocationResult,
    agent_tiers: Dict[str, str],     # agent_id → "gold"/"silver"/"bronze"
    reserve_prices: Optional[Dict[str, float]] = None,  # resource_name → min price
) -> List[PricedPayment]:
    """
    Apply tier discounts to VCG payments and enforce reserve prices.

    Args:
        result:         The AllocationResult from the auction engine
        agent_tiers:    Map of agent_id to tier level
        reserve_prices: Minimum payment per resource type (optional)

    Returns:
        List of PricedPayment objects (one per winner)
    """
    priced = []

    for bid in result.winning_bids:
        vcg = result.payments.get(bid.bid_id, bid.bid_amount * 0.90)
        tier = agent_tiers.get(bid.agent_id, "bronze")
        discount = TIER_DISCOUNTS.get(tier, 0.0)

        # Apply discount
        after_discount = vcg * (1 - discount)

        # Enforce reserve price: payment must cover reserve cost
        if reserve_prices:
            min_payment = sum(
                reserve_prices.get(r, 0.0)
                for r in bid.resource_bundle
            )
            after_discount = max(after_discount, min_payment)

        # Never charge more than the bid (sanity cap)
        final = min(after_discount, bid.bid_amount)
        final = max(0.0, final)  # Never negative

        priced.append(PricedPayment(
            bid_id=bid.bid_id,
            agent_id=bid.agent_id,
            bid_amount=bid.bid_amount,
            vcg_payment=vcg,
            discount_pct=discount,
            final_payment=round(final, 4),
            savings=round(bid.bid_amount - final, 4),
            savings_pct=round((bid.bid_amount - final) / bid.bid_amount * 100, 2)
                        if bid.bid_amount > 0 else 0.0,
        ))

    return priced


def format_payment_report(
    result: AllocationResult,
    priced_payments: List[PricedPayment],
) -> str:
    """
    Return a formatted string report of auction payments.
    Used in CLI output and logging.
    """
    lines = [
        "",
        "=" * 62,
        f"  PAYMENT REPORT — Round #{result.round_id}",
        "=" * 62,
        f"  Solver: {result.solver_method.value.upper()} | "
        f"Time: {result.solve_time_ms:.1f}ms",
        "",
        f"  WINNERS ({len(result.winning_bids)}):",
        "  " + "─" * 58,
    ]

    total_collected = 0.0
    for p in priced_payments:
        lines.append(str(p))
        total_collected += p.final_payment

    total_bid = sum(b.bid_amount for b in result.winning_bids)
    total_savings = total_bid - total_collected

    lines += [
        "  " + "─" * 58,
        f"  Total bid value:    ${total_bid:.2f}",
        f"  Total collected:    ${total_collected:.2f}",
        f"  Total agent savings:${total_savings:.2f}",
        "",
        f"  UNALLOCATED: {result.unallocated_resources or 'none'}",
    ]

    if result.solver_method.value == "lp" and result.greedy_value > 0:
        improvement = (result.lp_value - result.greedy_value) / result.greedy_value * 100
        lines += [
            "",
            f"  LP vs Greedy: ${result.lp_value:.2f} vs ${result.greedy_value:.2f} "
            f"({'↑' if improvement >= 0 else '↓'}{abs(improvement):.1f}%)",
        ]

    lines.append("=" * 62)
    return "\n".join(lines)


def check_budget_feasibility(
    bids: List[Bid],
    agent_budgets: Dict[str, float],  # agent_id → remaining budget
) -> List[Bid]:
    """
    Filter out bids from agents who can't afford their bid amount.
    Prevents agents from bidding more than they have.

    Returns only feasible bids.
    """
    feasible = []
    for bid in bids:
        budget = agent_budgets.get(bid.agent_id, 0.0)
        if bid.bid_amount <= budget:
            feasible.append(bid)
        else:
            from loguru import logger
            logger.warning(
                f"Bid {bid.bid_id} rejected: {bid.agent_id} bid "
                f"${bid.bid_amount:.2f} but only has ${budget:.2f}"
            )
    return feasible
