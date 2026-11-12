"""
Epoch-based, non-preemptive cluster simulator.

Each epoch the simulator (1) frees units held by jobs that finished, (2)
admits newly arrived jobs, (3) drops jobs that waited so long their value
fell below ``drop_fraction`` of the original, and (4) asks the policy which
pending jobs to start with the free GPUs/CPUs/RAM. A started job holds its
units for ``duration`` epochs.

Metrics:
  welfare       Σ value_if_started(start) over started jobs (value decays with wait)
  avg_jct       mean (finish - submit) over completed jobs, in epochs
  ftf_rho       finish-time fairness ρ = JCT / duration (1 = no waiting), as
                in Themis (Mahajan et al., NSDI 2020); mean, p95, and Jain's
                index of per-user mean 1/ρ
  utilization   busy GPU-epochs / (GPUs × makespan)
  dropped       jobs abandoned after waiting too long
"""

from __future__ import annotations

import copy
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from compute_bazaar.fairness.metrics import jains_fairness_index
from compute_bazaar.sim.traces import Job

Policy = Callable[[list[Job], dict[str, int], int, "ClusterSimulator"], list[Job]]


@dataclass
class SimResult:
    policy: str
    welfare: float
    avg_jct: float
    p95_jct: float
    ftf_mean: float
    ftf_p95: float
    ftf_jain: float
    utilization: float
    completed: int
    dropped: int
    makespan: int
    decision_ms_mean: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}
        return d


class ClusterSimulator:
    def __init__(
        self,
        jobs: list[Job],
        capacity: dict[str, int] | None = None,
        drop_fraction: float = 0.05,
        max_epochs: int = 5000,
    ):
        self.template = jobs
        self.capacity = capacity or {"gpu": 64, "cpu": 512, "ram": 2048}
        self.drop_fraction = drop_fraction
        self.max_epochs = max_epochs

    @staticmethod
    def demand(job: Job) -> dict[str, int]:
        return {"gpu": job.gpus, "cpu": job.cpus, "ram": job.ram_gb}

    def fits(self, job: Job, free: dict[str, int]) -> bool:
        return all(free[k] >= v for k, v in self.demand(job).items())

    def run(self, policy: Policy, name: str = "policy") -> SimResult:
        jobs = copy.deepcopy(self.template)
        # Jobs that could never fit are excluded up front for every policy.
        jobs = [j for j in jobs if self.fits(j, self.capacity)]
        arrivals: dict[int, list[Job]] = defaultdict(list)
        for j in jobs:
            arrivals[j.submit].append(j)

        free = dict(self.capacity)
        pending: list[Job] = []
        running: list[Job] = []
        done: list[Job] = []
        dropped: list[Job] = []
        busy_gpu_epochs = 0
        welfare = 0.0
        decision_ms: list[float] = []
        self.running = running  # exposed for policies (e.g. DRF shares)

        epoch = 0
        last_arrival = max((j.submit for j in jobs), default=0)
        while epoch <= self.max_epochs:
            # 1. completions
            still = []
            for j in running:
                if j.finish is not None and j.finish <= epoch:
                    for k, v in self.demand(j).items():
                        free[k] += v
                    done.append(j)
                else:
                    still.append(j)
            running[:] = still

            # 2. arrivals, 3. abandonment
            pending.extend(arrivals.get(epoch, []))
            keep = []
            for j in pending:
                if j.value_if_started(epoch) < self.drop_fraction * j.value:
                    dropped.append(j)
                else:
                    keep.append(j)
            pending = keep

            # 4. scheduling decision
            if pending:
                t0 = time.perf_counter()
                chosen = policy(list(pending), dict(free), epoch, self)
                decision_ms.append((time.perf_counter() - t0) * 1000)
                chosen_ids = set()
                for j in chosen:
                    if j.job_id in chosen_ids or not self.fits(j, free):
                        continue  # policies can't over-allocate
                    for k, v in self.demand(j).items():
                        free[k] -= v
                    j.start, j.finish = epoch, epoch + j.duration
                    welfare += j.value_if_started(epoch)
                    running.append(j)
                    chosen_ids.add(j.job_id)
                pending = [j for j in pending if j.job_id not in chosen_ids]

            busy_gpu_epochs += self.capacity["gpu"] - free["gpu"]
            epoch += 1
            if epoch > last_arrival and not pending and not running:
                break

        jct = [j.finish - j.submit for j in done]
        rho = [(j.finish - j.submit) / j.duration for j in done]
        per_user: dict[str, list[float]] = defaultdict(list)
        for j in done:
            per_user[j.user].append(j.duration / (j.finish - j.submit))
        user_scores = [statistics.mean(v) for v in per_user.values()]

        def p95(xs: list[float]) -> float:
            if not xs:
                return 0.0
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(0.95 * len(xs)))]

        makespan = max(epoch, 1)
        return SimResult(
            policy=name,
            welfare=welfare,
            avg_jct=statistics.mean(jct) if jct else 0.0,
            p95_jct=p95(jct),
            ftf_mean=statistics.mean(rho) if rho else 0.0,
            ftf_p95=p95(rho),
            ftf_jain=jains_fairness_index(user_scores) if user_scores else 1.0,
            utilization=busy_gpu_epochs / (self.capacity["gpu"] * makespan),
            completed=len(done),
            dropped=len(dropped),
            makespan=makespan,
            decision_ms_mean=statistics.mean(decision_ms) if decision_ms else 0.0,
        )
