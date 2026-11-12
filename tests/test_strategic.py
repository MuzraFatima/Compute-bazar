"""Tests for core-selecting payments and the repeated LLG game."""

import random

import pytest

from backend.app.auction.engine import Bid, Resource, compute_vcg_agent_payments, solve_exhaustive
from compute_bazaar.strategic.core import core_payments
from compute_bazaar.strategic.game import build_prompt, run_learning, run_llm, settle, settle_generic

R = {"A": Resource("A", "gpu"), "B": Resource("B", "gpu")}


def test_core_fixes_vcg_revenue_collapse_in_llg():
    bids = [Bid("l1", "L1", ["A"], 8.0), Bid("l2", "L2", ["B"], 8.0), Bid("g", "G", ["A", "B"], 10.0)]
    winners, _ = solve_exhaustive(bids, R)
    vcg = compute_vcg_agent_payments(winners, bids, R, solver=solve_exhaustive)
    core = core_payments(winners, bids, R)
    assert vcg == {"L1": pytest.approx(2.0), "L2": pytest.approx(2.0)}  # global would pay 10
    assert core == {"L1": pytest.approx(5.0), "L2": pytest.approx(5.0)}


def test_core_payments_lie_between_vcg_and_bids():
    rng = random.Random(4)
    for _ in range(15):
        bids = [
            Bid("l1", "L1", ["A"], rng.uniform(1, 100)),
            Bid("l2", "L2", ["B"], rng.uniform(1, 100)),
            Bid("g", "G", ["A", "B"], rng.uniform(1, 200)),
        ]
        winners, _ = solve_exhaustive(bids, R)
        vcg = compute_vcg_agent_payments(winners, bids, R, solver=solve_exhaustive)
        core = core_payments(winners, bids, R)
        for b in winners:
            assert vcg[b.agent_id] - 1e-6 <= core[b.agent_id] <= b.bid_amount + 1e-6


@pytest.mark.parametrize("mechanism", ["vcg", "core", "first_price"])
def test_closed_form_llg_settlement_matches_general_engine(mechanism):
    rng = random.Random(9)
    for _ in range(25):
        b = {"L1": rng.uniform(0.1, 100), "L2": rng.uniform(0.1, 100), "G": rng.uniform(0.1, 200)}
        w1, p1 = settle(mechanism, b)
        w2, p2 = settle_generic(mechanism, b)
        assert w1 == w2
        assert p1 == pytest.approx(p2, abs=1e-3)


def test_learning_bidders_go_truthful_under_vcg_and_shade_under_first_price():
    vcg = run_learning("vcg", rounds=800, seed=1)
    fp = run_learning("first_price", rounds=800, seed=1)
    assert all(0.85 <= m <= 1.25 for m in vcg["bid_value_ratio"].values())
    assert all(m < 0.7 for m in fp["bid_value_ratio"].values())


def test_mock_llm_game_runs_and_reports_ratios():
    res = run_llm("core", informed=False, chat=True, rounds=4)
    assert not res["live_llm"] and res["errors"] == 0
    assert set(res["bid_value_ratio"]) == {"L1", "L2", "G"}
    assert len(res["transcript"]) == 4


def test_prompt_mentions_dominance_only_when_informed():
    hist = [{"value": 10.0, "bid": 9.0, "won": True, "payment": 5.0, "profit": 5.0}]
    informed = build_prompt("L1", 42.0, "vcg", True, hist, {}, chat=False)
    plain = build_prompt("L1", 42.0, "vcg", False, hist, {"L2": "bid low"}, chat=True)
    assert "dominant strategy" in informed and "dominant strategy" not in plain
    assert "L2: bid low" in plain and '"message"' in plain
