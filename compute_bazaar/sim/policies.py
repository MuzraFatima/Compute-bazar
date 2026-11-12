"""
Scheduling policies for ``ClusterSimulator``.

All policies are non-preemptive and see the same information: pending jobs
(with their current value), free units and the epoch. They return the jobs
to start; the simulator rejects anything that doesn't fit.

  fifo          arrival order with head-of-line blocking (no backfill), like a
                plain Slurm/K8s queue
  fifo_backfill arrival order, skipping jobs that don't fit (EASY-style, without
                reservations)
  drf           Dominant Resource Fairness (Ghodsi et al., NSDI 2011): the user
                with the lowest dominant share goes next
  themis        Themis-style finish-time fairness (Mahajan et al., NSDI 2020):
                the worst-off half of jobs by estimated ρ compete; among them,
                highest value density wins
  greedy_value  highest value per GPU-epoch first
  bazaar        Compute Bazaar: exact multi-unit combinatorial WDP each epoch
                (ILP over GPU/CPU/RAM units), bids = jobs' current values
"""

from __future__ import annotations

from collections import defaultdict

from backend.app.auction.engine import Bid, Resource, solve_wdp_lp
from compute_bazaar.sim.traces import Job


def _pack(order: list[Job], free: dict[str, int], sim, block: bool = False) -> list[Job]:
    chosen = []
    free = dict(free)
    for j in order:
        if sim.fits(j, free):
            for k, v in sim.demand(j).items():
                free[k] -= v
            chosen.append(j)
        elif block:
            break
    return chosen


def fifo(pending, free, epoch, sim):
    return _pack(sorted(pending, key=lambda j: (j.submit, j.job_id)), free, sim, block=True)


def fifo_backfill(pending, free, epoch, sim):
    return _pack(sorted(pending, key=lambda j: (j.submit, j.job_id)), free, sim)


def greedy_value(pending, free, epoch, sim):
    return _pack(sorted(pending, key=lambda j: -j.value_if_started(epoch) / j.gpu_time), free, sim)


def drf(pending, free, epoch, sim):
    cap = sim.capacity
    usage: dict[str, dict[str, int]] = defaultdict(lambda: {"gpu": 0, "cpu": 0, "ram": 0})
    for j in sim.running:
        for k, v in sim.demand(j).items():
            usage[j.user][k] += v
    by_user: dict[str, list[Job]] = defaultdict(list)
    for j in sorted(pending, key=lambda j: (j.submit, j.job_id)):
        by_user[j.user].append(j)
    free = dict(free)
    chosen = []
    while by_user:
        user = min(by_user, key=lambda u: max(usage[u][k] / cap[k] for k in cap))
        job = next((j for j in by_user[user] if sim.fits(j, free)), None)
        if job is None:
            del by_user[user]
            continue
        by_user[user].remove(job)
        if not by_user[user]:
            del by_user[user]
        for k, v in sim.demand(job).items():
            free[k] -= v
            usage[user][k] += v
        chosen.append(job)
    return chosen


def themis(pending, free, epoch, sim, fairness_knob: float = 0.5):
    def rho(j: Job) -> float:
        return (epoch - j.submit + j.duration) / j.duration

    ranked = sorted(pending, key=rho, reverse=True)
    k = max(1, int(len(ranked) * fairness_knob))
    worst, rest = ranked[:k], ranked[k:]
    worst.sort(key=lambda j: -j.value_if_started(epoch) / j.gpu_time)
    chosen = _pack(worst, free, sim)
    left = dict(free)
    for j in chosen:
        for r, v in sim.demand(j).items():
            left[r] -= v
    rest.sort(key=lambda j: -rho(j))
    return chosen + _pack(rest, left, sim)  # work-conserving fill


def _density(j: Job, epoch: int) -> float:
    return j.value_if_started(epoch) / j.gpu_time


def bazaar(pending, free, epoch, sim, max_bids: int = 80, reserve_quantile: float | None = 0.5):
    """
    Exact multi-unit WDP over the ``max_bids`` densest pending jobs.

    ``reserve_quantile`` sets a per-GPU-epoch reserve λ at that quantile of
    the queue's value density; each bid enters the ILP as v - λ·gpus·duration.
    Without it the per-epoch optimum is myopic: it fills the cluster with long
    jobs that block denser work arriving later. A fixed reserve is the
    "seller as a phantom bidder", so VCG with it remains truthful.
    ``reserve_quantile=None`` gives the plain myopic auction.
    """
    cands = sorted(pending, key=lambda j: -_density(j, epoch))[:max_bids]
    lam = 0.0
    if reserve_quantile is not None and pending:
        dens = sorted(_density(j, epoch) for j in pending)
        lam = dens[min(len(dens) - 1, int(reserve_quantile * len(dens)))]
    resources = {k: Resource(k, k, units=v) for k, v in free.items() if v > 0}
    bids = []
    index = {}
    for j in cands:
        dem = {k: v for k, v in sim.demand(j).items() if v > 0}
        if any(k not in resources or v > resources[k].units for k, v in dem.items()):
            continue
        val = j.value_if_started(epoch) - lam * j.gpu_time
        if val <= 1e-9:
            continue
        bids.append(Bid(j.job_id, j.user, list(dem), round(val, 6), quantities=dem))
        index[j.job_id] = j
    if not bids:
        return []
    winners, _ = solve_wdp_lp(bids, resources)
    chosen = [index[b.bid_id] for b in winners]
    # Work-conserving: anything still fitting (outside the candidate cap) gets backfilled.
    left = dict(free)
    for j in chosen:
        for r, v in sim.demand(j).items():
            left[r] -= v
    rest = [j for j in pending if j.job_id not in {c.job_id for c in chosen}]
    return chosen + _pack(sorted(rest, key=lambda j: -_density(j, epoch)), left, sim)


def bazaar_myopic(pending, free, epoch, sim):
    return bazaar(pending, free, epoch, sim, reserve_quantile=None)


POLICIES = {
    "fifo": fifo,
    "fifo_backfill": fifo_backfill,
    "drf": drf,
    "themis": themis,
    "greedy_value": greedy_value,
    "bazaar_myopic": bazaar_myopic,
    "bazaar": bazaar,
}
