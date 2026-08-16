"""
=============================================================
 COMPUTE BAZAAR — Week 2 Unit Tests
=============================================================
 Tests the LP solver, VCG pricing, and pricing utilities.

 Run with:  python -m pytest tests/test_week2_engine.py -v
=============================================================
"""

import uuid
import pytest
from typing import List, Dict

from backend.app.auction.engine import (
    AuctionEngine, Resource, Bid, SolverMethod,
    solve_greedy, solve_wdp_lp, compute_vcg_prices,
)
from backend.app.auction.pricing import (
    apply_tier_pricing, check_budget_feasibility, PricedPayment,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

def bid(agent: str, bundle: List[str], amount: float) -> Bid:
    return Bid(
        bid_id=str(uuid.uuid4())[:8],
        agent_id=agent,
        resource_bundle=bundle,
        bid_amount=amount,
    )


def resources_3() -> Dict[str, Resource]:
    return {
        "GPU-1":   Resource("GPU-1",   "gpu",        1.0),
        "GPU-2":   Resource("GPU-2",   "gpu",        1.0),
        "API-100": Resource("API-100", "api_budget", 100.0),
    }


def engine_lp(extra_resources=None) -> AuctionEngine:
    base = [
        Resource("GPU-1",   "gpu",        1.0),
        Resource("GPU-2",   "gpu",        1.0),
        Resource("API-100", "api_budget", 100.0),
        Resource("RAM-32",  "ram",        32.0),
    ]
    return AuctionEngine(base + (extra_resources or []), method=SolverMethod.LP)


# ─────────────────────────────────────────────────────────────
# GROUP 1: LP Solver Correctness
# ─────────────────────────────────────────────────────────────

def test_lp_single_bid_wins():
    bids = [bid("A", ["GPU-1"], 50.0)]
    winners, value = solve_wdp_lp(bids, resources_3())
    assert len(winners) == 1
    assert winners[0].agent_id == "A"
    assert value == pytest.approx(50.0)


def test_lp_highest_value_wins():
    bids = [
        bid("Low",  ["GPU-1"], 20.0),
        bid("High", ["GPU-1"], 80.0),
        bid("Mid",  ["GPU-1"], 45.0),
    ]
    winners, value = solve_wdp_lp(bids, resources_3())
    assert len(winners) == 1
    assert winners[0].agent_id == "High"
    assert value == pytest.approx(80.0)


def test_lp_non_conflicting_both_win():
    bids = [
        bid("A", ["GPU-1"],   40.0),
        bid("B", ["GPU-2"],   35.0),
        bid("C", ["API-100"], 30.0),
    ]
    winners, value = solve_wdp_lp(bids, resources_3())
    winner_ids = {w.agent_id for w in winners}
    assert "A" in winner_ids
    assert "B" in winner_ids
    assert "C" in winner_ids
    assert value == pytest.approx(105.0)


def test_lp_no_resource_double_allocated():
    """Core constraint: no resource given to two winners."""
    bids = [bid(f"Agent-{i}", ["GPU-1"], float(10 + i)) for i in range(10)]
    winners, _ = solve_wdp_lp(bids, resources_3())
    # Only one agent can have GPU-1
    assert len(winners) == 1


def test_lp_beats_greedy_on_bundle_scenario():
    """
    LP strictly beats greedy on the crafted scenario:
    Bundle bid per-resource < individual bids, but bundle total > sum of individuals.
    """
    res = {
        "GPU-1":   Resource("GPU-1",   "gpu", 1.0),
        "GPU-2":   Resource("GPU-2",   "gpu", 1.0),
        "API-100": Resource("API-100", "api_budget", 100.0),
    }
    bids = [
        bid("Bundle", ["GPU-1", "GPU-2", "API-100"], 260.0),  # per-res=$86.7
        bid("B",      ["GPU-1"],                      90.0),   # per-res=$90 ← greedy picks 1st
        bid("C",      ["GPU-2"],                      85.0),
        bid("D",      ["API-100"],                    70.0),
    ]
    # Greedy picks B($90)+C($85)+D($70)=$245
    _, g_val = solve_greedy(bids, res)
    # LP finds Bundle($260) > B+C+D($245)
    lp_winners, lp_val = solve_wdp_lp(bids, res)

    assert lp_val >= g_val, "LP must be at least as good as greedy"
    assert lp_val == pytest.approx(260.0), "LP should pick the bundle"
    assert lp_winners[0].agent_id == "Bundle"


def test_lp_empty_bids():
    winners, value = solve_wdp_lp([], resources_3())
    assert winners == []
    assert value == 0.0


def test_lp_invalid_resource_skipped():
    bids = [
        bid("Valid",   ["GPU-1"],    50.0),
        bid("Invalid", ["FAKE-RES"], 999.0),  # non-existent resource
    ]
    winners, value = solve_wdp_lp(bids, resources_3())
    winner_ids = {w.agent_id for w in winners}
    assert "Valid" in winner_ids
    assert "Invalid" not in winner_ids


# ─────────────────────────────────────────────────────────────
# GROUP 2: VCG Pricing
# ─────────────────────────────────────────────────────────────

def test_vcg_payment_leq_bid():
    """VCG payment must never exceed the winner's own bid."""
    res = resources_3()
    bids = [
        bid("Finance", ["GPU-1", "API-100"], 110.0),
        bid("Gamer",   ["GPU-1"],             60.0),
        bid("Writer",  ["API-100"],            45.0),
    ]
    winners, _ = solve_wdp_lp(bids, res)
    payments = compute_vcg_prices(winners, bids, res)

    for w in winners:
        p = payments[w.bid_id]
        assert p <= w.bid_amount, (
            f"{w.agent_id}: VCG payment ${p:.2f} > bid ${w.bid_amount:.2f}"
        )


def test_vcg_payment_non_negative():
    """VCG payments must always be ≥ 0."""
    res = resources_3()
    bids = [
        bid("A", ["GPU-1"],   70.0),
        bid("B", ["GPU-2"],   55.0),
        bid("C", ["API-100"], 40.0),
    ]
    winners, _ = solve_wdp_lp(bids, res)
    payments = compute_vcg_prices(winners, bids, res)
    for bid_id, p in payments.items():
        assert p >= 0.0, f"Negative VCG payment for {bid_id}: ${p:.2f}"


def test_vcg_only_winners_get_payments():
    """VCG payments dict should only have entries for winning bids."""
    res = resources_3()
    bids = [
        bid("Winner", ["GPU-1"], 80.0),
        bid("Loser",  ["GPU-1"], 30.0),
    ]
    winners, _ = solve_wdp_lp(bids, res)
    payments = compute_vcg_prices(winners, bids, res)

    winner_ids = {w.bid_id for w in winners}
    for bid_id in payments:
        assert bid_id in winner_ids


def test_vcg_monopoly_pays_zero():
    """
    If an agent is the only one who can use a resource,
    they impose no externality → VCG payment = 0.
    """
    res = {"GPU-1": Resource("GPU-1", "gpu", 1.0)}
    bids = [bid("Solo", ["GPU-1"], 100.0)]
    winners, _ = solve_wdp_lp(bids, res)
    payments = compute_vcg_prices(winners, bids, res)

    assert payments[winners[0].bid_id] == pytest.approx(0.0, abs=0.01)


def test_vcg_competitive_externality():
    """
    When there's competition, the winner pays a meaningful price.
    Agent A wins GPU-1 ($80), displacing Agent B ($50).
    VCG payment for A should reflect B's displaced value.
    """
    res = {"GPU-1": Resource("GPU-1", "gpu", 1.0)}
    bids = [
        bid("A", ["GPU-1"], 80.0),
        bid("B", ["GPU-1"], 50.0),
    ]
    winners, _ = solve_wdp_lp(bids, res)
    payments = compute_vcg_prices(winners, bids, res)

    # Without A: B wins ($50). With A winning, others get $0.
    # A's VCG payment = $50 - $0 = $50
    a_winner = next(w for w in winners if w.agent_id == "A")
    payment = payments[a_winner.bid_id]
    assert payment == pytest.approx(50.0, abs=0.01)


# ─────────────────────────────────────────────────────────────
# GROUP 3: Auction Engine Integration
# ─────────────────────────────────────────────────────────────

def test_engine_submit_and_run():
    """End-to-end: submit bids, run auction, get results."""
    eng = engine_lp()
    eng.submit_bid(bid("A", ["GPU-1"],   60.0))
    eng.submit_bid(bid("B", ["GPU-2"],   55.0))
    eng.submit_bid(bid("C", ["API-100"], 40.0))

    result = eng.run_auction(round_id=1)
    assert len(result.winning_bids) == 3
    assert result.total_value == pytest.approx(155.0)
    assert result.round_id == 1


def test_engine_rejects_unknown_resource():
    eng = engine_lp()
    ok = eng.submit_bid(bid("X", ["UNKNOWN-RESOURCE"], 50.0))
    assert ok == False


def test_engine_rejects_duplicate_bid_id():
    eng = engine_lp()
    b = bid("A", ["GPU-1"], 50.0)
    eng.submit_bid(b)
    ok = eng.submit_bid(b)  # same bid_id
    assert ok == False


def test_engine_reset_clears_bids():
    eng = engine_lp()
    eng.submit_bid(bid("A", ["GPU-1"], 50.0))
    eng.reset()
    assert eng.get_bids() == []


def test_engine_round_increments():
    eng = engine_lp()
    eng.submit_bid(bid("A", ["GPU-1"], 50.0))
    r1 = eng.run_auction()
    eng.reset()
    eng.submit_bid(bid("B", ["GPU-2"], 40.0))
    r2 = eng.run_auction()
    assert r2.round_id == r1.round_id + 1


def test_engine_lp_vs_greedy_methods():
    """LP and greedy both run without errors; LP value ≥ greedy value."""
    bids_data = [
        bid("A", ["GPU-1"], 80.0),
        bid("B", ["GPU-2"], 75.0),
        bid("C", ["API-100"], 60.0),
    ]

    eng_lp = engine_lp()
    eng_gr = AuctionEngine(
        [Resource("GPU-1","gpu",1.0), Resource("GPU-2","gpu",1.0),
         Resource("API-100","api_budget",100.0)],
        method=SolverMethod.GREEDY
    )

    for b in bids_data:
        eng_lp.submit_bid(b)
        # Greedy engine needs fresh bids (different bid objects with same data)
        eng_gr.submit_bid(Bid(
            bid_id=b.bid_id+"g", agent_id=b.agent_id,
            resource_bundle=b.resource_bundle, bid_amount=b.bid_amount
        ))

    lp_result = eng_lp.run_auction()
    gr_result = eng_gr.run_auction()

    assert lp_result.total_value >= gr_result.total_value


# ─────────────────────────────────────────────────────────────
# GROUP 4: Pricing Utilities
# ─────────────────────────────────────────────────────────────

def test_tier_gold_discount():
    """Gold tier agents get 10% off VCG payment."""
    eng = engine_lp()
    b = bid("gold-agent", ["GPU-1"], 80.0)
    eng.submit_bid(b)
    result = eng.run_auction()

    priced = apply_tier_pricing(result, {"gold-agent": "gold"})
    assert len(priced) == 1
    p = priced[0]
    assert p.discount_pct == pytest.approx(0.10)
    assert p.final_payment <= p.vcg_payment  # discount applied


def test_tier_bronze_no_discount():
    """Bronze agents pay full VCG price."""
    eng = engine_lp()
    b = bid("bronze-agent", ["GPU-1"], 80.0)
    eng.submit_bid(b)
    result = eng.run_auction()

    priced = apply_tier_pricing(result, {"bronze-agent": "bronze"})
    p = priced[0]
    assert p.discount_pct == pytest.approx(0.0)
    assert p.final_payment == pytest.approx(p.vcg_payment, abs=0.01)


def test_budget_feasibility_filters():
    """Agents bidding over their budget should be filtered out."""
    bids_data = [
        bid("Rich",  ["GPU-1"], 80.0),
        bid("Broke", ["GPU-2"], 500.0),
    ]
    budgets = {"Rich": 1000.0, "Broke": 100.0}
    feasible = check_budget_feasibility(bids_data, budgets)
    agent_ids = [b.agent_id for b in feasible]
    assert "Rich" in agent_ids
    assert "Broke" not in agent_ids


def test_bid_dataclass_validation():
    """Bid should raise on invalid amounts."""
    with pytest.raises(ValueError):
        Bid(bid_id="x", agent_id="A", resource_bundle=["GPU-1"], bid_amount=0.0)
    with pytest.raises(ValueError):
        Bid(bid_id="x", agent_id="A", resource_bundle=[], bid_amount=50.0)
