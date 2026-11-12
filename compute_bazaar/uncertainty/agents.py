"""
Bidders with noisy, biased estimates of value and resource needs.

Each round an agent receives a task with a true value v and a true
per-resource need q (both hidden). The agent reports an estimate

    v̂ = v · exp(b_a + σ_a ε),          ε ~ N(0, 1)

where b_a is its systematic bias (an overconfident agent has b_a > 0) and
σ_a its noise. This mirrors how an LLM's self-assessed task value behaves:
correlated with the truth but off by a persistent, agent-specific factor.

Needs are estimated the same way on a log scale. Each agent also reports a
90% upper bound on its needs, as the Phase 1 predictor does. A
well-calibrated agent's bound covers the truth ~90% of the time; an
overconfident one reports bounds that are too tight (``interval_scale`` < 1).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

Z90 = 1.2816  # one-sided 90% normal quantile


@dataclass
class Task:
    agent: str
    task_id: str
    true_value: float
    true_need: dict[str, int]
    est_value: float
    est_need: dict[str, int]
    need_upper90: dict[str, int]
    meta: dict = field(default_factory=dict)


@dataclass
class EstimatingAgent:
    name: str
    value_bias: float = 0.0  # log-scale; +0.3 ≈ overstates by 35%
    value_noise: float = 0.2
    need_noise: float = 0.25
    interval_scale: float = 1.0  # <1 → overconfident intervals
    base_value: float = 50.0
    resources: tuple[str, ...] = ("gpu", "cpu", "ram")

    def draw_task(self, rng: random.Random, k: int, max_units: dict[str, int]) -> Task:
        true_value = self.base_value * math.exp(rng.gauss(0, 0.5))
        bundle = [r for r in self.resources if rng.random() < 0.7] or [self.resources[0]]
        true_need = {r: rng.randint(1, max(1, max_units[r] // 4)) for r in bundle}
        est_value = true_value * math.exp(self.value_bias + self.value_noise * rng.gauss(0, 1))
        est_need, upper = {}, {}
        for r, q in true_need.items():
            log_est = math.log(q) + self.need_noise * rng.gauss(0, 1)
            est_need[r] = max(1, round(math.exp(log_est)))
            upper[r] = max(est_need[r], math.ceil(math.exp(log_est + Z90 * self.need_noise * self.interval_scale)))
        return Task(self.name, f"{self.name}-t{k}", true_value, true_need, est_value, est_need, upper)


def make_population(n_agents: int = 12, seed: int = 0, overconfident_share: float = 0.5) -> list[EstimatingAgent]:
    """
    A mixed population: some calibrated agents, some overconfident ones that
    overstate value (bias +0.2..+0.6) and report too-tight need intervals,
    and a few under-confident ones. Roughly matches the spread of
    LLM self-estimates reported in the calibration literature.
    """
    rng = random.Random(seed)
    agents = []
    for i in range(n_agents):
        u = rng.random()
        if u < overconfident_share:
            bias, iscale = rng.uniform(0.2, 0.6), rng.uniform(0.3, 0.7)
        elif u < overconfident_share + 0.15:
            bias, iscale = rng.uniform(-0.4, -0.1), rng.uniform(1.0, 1.5)
        else:
            bias, iscale = rng.uniform(-0.05, 0.05), 1.0
        agents.append(
            EstimatingAgent(
                name=f"agent{i}",
                value_bias=bias,
                value_noise=rng.uniform(0.1, 0.35),
                interval_scale=iscale,
                base_value=math.exp(rng.gauss(math.log(50), 0.4)),
            )
        )
    return agents
