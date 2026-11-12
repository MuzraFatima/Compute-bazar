"""Tests for mechanisms with estimated values and needs (compute_bazaar.uncertainty)."""

import math
import random

import pytest

from backend.app.auction.engine import Bid, Resource, solve_exhaustive
from compute_bazaar.uncertainty import (
    CalibratedVCG,
    EstimatingAgent,
    ExecutionContingentVCG,
    PointVCG,
    Task,
    make_population,
)
from compute_bazaar.uncertainty.mechanisms import oracle

CAP = {"gpu": 4, "cpu": 8, "ram": 8}


def task(agent, tid, true_v, est_v, need, est_need=None, upper=None):
    est_need = est_need or dict(need)
    return Task(agent, tid, true_v, dict(need), est_v, est_need, upper or dict(est_need))


def test_point_vcg_charges_winners_even_when_they_overrun():
    tasks = [task("a", "a1", 50, 60, {"gpu": 3}, est_need={"gpu": 2}), task("b", "b1", 30, 30, {"gpu": 2})]
    out = PointVCG().run_round(tasks, CAP)
    assert {t.agent for t in out.winners} == {"a", "b"}
    assert [t.agent for t in out.failed] == ["a"]  # needed 3 GPUs, got 2
    assert out.utilities["a"] == pytest.approx(0.0 - out.payments["a"])


def test_contingent_payment_equals_vminus_a_minus_others_realized():
    # One GPU, two bidders: a wins and succeeds; V*_{-a} = 40 (b's report).
    tasks = [task("a", "a1", 70, 80, {"gpu": 4}), task("b", "b1", 35, 40, {"gpu": 4})]
    out = ExecutionContingentVCG().run_round(tasks, CAP)
    assert [t.agent for t in out.winners] == ["a"]
    assert out.payments["a"] == pytest.approx(40.0)  # 40 − (realized others = 0)
    assert out.utilities["a"] == pytest.approx(30.0)


def test_contingent_utility_is_realized_welfare_minus_a_constant():
    """Two winners: each one's utility moves one-for-one with total realized welfare."""
    tasks = [
        task("a", "a1", 50, 50, {"gpu": 2}),
        task("b", "b1", 40, 40, {"gpu": 3}, est_need={"gpu": 2}),  # will overrun
        task("c", "c1", 10, 10, {"gpu": 2}),
    ]
    out = ExecutionContingentVCG().run_round(tasks, CAP)
    assert {t.agent for t in out.winners} == {"a", "b"} and out.realized_welfare == pytest.approx(50.0)
    # V*_{-a} = b + c = 50 → u_a = 50 − 50 = 0; V*_{-b} = a + c = 60 → u_b = 50 − 60 = −10.
    assert out.utilities["a"] == pytest.approx(0.0)
    assert out.utilities["b"] == pytest.approx(-10.0)


def test_calibration_learns_an_overconfident_agents_bias():
    agent = EstimatingAgent("over", value_bias=0.5, value_noise=0.1, need_noise=0.01)
    mech = CalibratedVCG(menus=False)
    rng = random.Random(0)
    for k in range(60):
        mech.run_round([agent.draw_task(rng, k, {"gpu": 16, "cpu": 16, "ram": 16})], {"gpu": 16, "cpu": 16, "ram": 16})
    assert mech._bias("over") == pytest.approx(0.5, abs=0.12)


def test_menu_prefers_a_safe_reservation_when_capacity_is_free():
    mech = CalibratedVCG()
    t = task("a", "a1", 100, 100, {"gpu": 2}, est_need={"gpu": 1})
    out = mech.run_round([t], {"gpu": 8, "cpu": 8, "ram": 8})
    assert out.winners and not out.failed  # biggest reservation has the highest success prob


def test_menu_success_probabilities_follow_history():
    mech = CalibratedVCG(prior_strength=1.0)
    for _ in range(20):
        mech.need_ratios["a"].append({"gpu": 1.4})  # always needs ~1.4× the estimate
    assert mech.success_prob("a", 0) < 0.1  # 1.0× reservation almost always overruns
    assert mech.success_prob("a", 2) > 0.9  # 1.5× almost always fits


def test_oracle_never_overruns():
    agents = make_population(6, seed=1)
    rng = random.Random(1)
    m = oracle()
    for k in range(20):
        out = m.run_round([a.draw_task(rng, k, CAP) for a in agents], CAP)
        assert not out.failed


def test_proposition1_welfare_loss_bound_on_random_rounds():
    """W*(v) − W(x̂, v) ≤ Σ_{i ∈ x̂ ∪ x*} |v̂_i − v_i| (docs/theory.md, Proposition 1)."""
    res = {r: Resource(r, r, units=u) for r, u in CAP.items()}
    agents = make_population(8, seed=2)
    rng = random.Random(2)
    for k in range(40):
        tasks = [a.draw_task(rng, k, CAP) for a in agents]
        true_b = [Bid(t.task_id, t.agent, list(t.true_need), t.true_value, quantities=t.true_need) for t in tasks]
        est_b = [Bid(t.task_id, t.agent, list(t.true_need), t.est_value, quantities=t.true_need) for t in tasks]
        x_star, w_star = solve_exhaustive(true_b, res)
        x_hat, _ = solve_exhaustive(est_b, res)
        tv = {t.task_id: t.true_value for t in tasks}
        err = {t.task_id: abs(t.est_value - t.true_value) for t in tasks}
        loss = w_star - sum(tv[b.bid_id] for b in x_hat)
        bound = sum(err[i] for i in {b.bid_id for b in x_hat} | {b.bid_id for b in x_star})
        assert loss <= bound + 1e-6


def test_estimates_follow_the_configured_bias():
    agent = EstimatingAgent("x", value_bias=0.3, value_noise=0.05)
    rng = random.Random(3)
    logs = [math.log(t.est_value / t.true_value) for t in (agent.draw_task(rng, k, CAP) for k in range(300))]
    assert sum(logs) / len(logs) == pytest.approx(0.3, abs=0.02)
