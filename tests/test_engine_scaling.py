"""Tests for the solver backend selection, parallel VCG and the exact pruning shortcut."""

import random

import pytest

from backend.app.auction import engine as eng
from backend.app.auction.engine import (
    Bid,
    Resource,
    compute_vcg_agent_payments,
    solve_exhaustive,
    solve_wdp_lp,
)


def random_case(seed, n_bids=14):
    rng = random.Random(seed)
    res = {f"R{k}": Resource(f"R{k}", "gpu", units=rng.randint(1, 3)) for k in range(5)}
    bids = []
    for i in range(n_bids):
        bundle = rng.sample(list(res), rng.randint(1, 3))
        bids.append(
            Bid(
                f"b{i}",
                f"a{i // 2}",
                bundle,
                round(rng.uniform(1, 50), 2),
                quantities={r: rng.randint(1, res[r].units) for r in bundle},
            )
        )
    return bids, res


@pytest.mark.parametrize("seed", range(4))
def test_parallel_counterfactuals_match_sequential(seed):
    bids, res = random_case(seed)
    winners, value = solve_wdp_lp(bids, res)
    seq = compute_vcg_agent_payments(winners, bids, res, optimal_value=value, workers=1)
    par = compute_vcg_agent_payments(winners, bids, res, optimal_value=value, workers=4)
    assert par == pytest.approx(seq, abs=1e-4)


@pytest.mark.parametrize("seed", range(20))
def test_uncontested_shortcut_agrees_with_full_recomputation(seed):
    """Payments with the shortcut equal V*(-a) - (V* - v_a) computed by brute force for every winner."""
    bids, res = random_case(seed, n_bids=10)
    winners, value = solve_exhaustive(bids, res)
    pay = compute_vcg_agent_payments(winners, bids, res, solver=solve_exhaustive, optimal_value=value)
    won = {}
    for b in winners:
        won[b.agent_id] = won.get(b.agent_id, 0.0) + b.bid_amount
    for agent, v in won.items():
        _, v_without = solve_exhaustive([b for b in bids if b.agent_id != agent], res)
        assert pay[agent] == pytest.approx(max(0.0, v_without - (value - v)), abs=1e-6)


def test_uncontested_winner_pays_zero_without_a_solve():
    res = {"A": Resource("A", "gpu"), "B": Resource("B", "gpu")}
    bids = [Bid("x", "X", ["A"], 10.0), Bid("y", "Y", ["B"], 5.0), Bid("z", "Z", ["B"], 3.0)]
    calls = []

    def counting_solver(b, r, xor=False):
        calls.append(len(b))
        return solve_exhaustive(b, r, xor)

    winners, value = solve_exhaustive(bids, res)
    pay = compute_vcg_agent_payments(winners, bids, res, solver=counting_solver, optimal_value=value)
    assert pay == {"X": 0.0, "Y": pytest.approx(3.0)}
    assert len(calls) == 1  # only Y, whose GPU B is contested by Z, needed a counterfactual


def test_solver_selection_falls_back_to_cbc(monkeypatch):
    monkeypatch.setenv("COMPUTE_BAZAAR_SOLVER", "cbc")
    eng._SOLVER_CACHE.clear()
    assert eng.solver_name() == "PULP_CBC_CMD"
    monkeypatch.setenv("COMPUTE_BAZAAR_SOLVER", "highs")
    eng._SOLVER_CACHE.clear()
    assert eng.solver_name() in ("HiGHS", "PULP_CBC_CMD")  # HiGHS if highspy is installed
    eng._SOLVER_CACHE.clear()


def test_mip_gap_env_is_respected(monkeypatch):
    monkeypatch.setenv("COMPUTE_BAZAAR_SOLVER", "cbc")
    monkeypatch.setenv("COMPUTE_BAZAAR_MIP_GAP", "0.05")
    eng._SOLVER_CACHE.clear()
    assert eng.milp_solver().optionsDict.get("gapRel") == pytest.approx(0.05)
    eng._SOLVER_CACHE.clear()
