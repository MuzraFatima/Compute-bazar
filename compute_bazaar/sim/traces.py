"""
Workloads for the cluster simulator.

``load_pai_trace`` reads the public Alibaba PAI GPU cluster trace (Weng et
al., NSDI 2022; github.com/alibaba/clusterdata, ``cluster-trace-gpu-v2020``)
from its ``pai_task_table.csv`` (instance count, plan_gpu in % of a GPU,
plan_cpu in % of a core, plan_mem in GB, start/end timestamps).
``scripts/fetch_traces.py`` downloads it.

``synthetic_pai_like`` generates a stand-in with the trace's published
shape: most jobs need a single GPU or less, a heavy tail of multi-GPU jobs,
log-normal runtimes spanning minutes to many hours, and bursty arrivals.
It is *calibrated to* the paper's marginals, not a copy of the data, and the
experiments label which source they used.

Every job gets a private value (willingness to pay) proportional to its
GPU-hours times a per-user value rate. The value decays with queueing delay,
a standard model for time-sensitive work, so a scheduler that delays
high-value jobs loses welfare.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

PAI_TASK_COLUMNS = [
    "job_name",
    "task_name",
    "inst_num",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "plan_gpu",
    "gpu_type",
]


@dataclass
class Job:
    job_id: str
    user: str
    submit: int  # epoch at which the job arrives
    duration: int  # epochs of service needed once started
    gpus: int
    cpus: int
    ram_gb: int
    value: float  # value if started immediately
    half_life: float = 48.0  # epochs of waiting that halve the value
    # Filled in by the simulator
    start: int | None = None
    finish: int | None = None
    meta: dict = field(default_factory=dict)

    def value_if_started(self, epoch: int) -> float:
        wait = max(0, epoch - self.submit)
        return self.value * 0.5 ** (wait / self.half_life)

    @property
    def gpu_time(self) -> int:
        return self.gpus * self.duration


def synthetic_pai_like(
    n_jobs: int = 1200,
    n_users: int = 120,
    horizon: int = 600,
    seed: int = 0,
    epoch_minutes: float = 10.0,
) -> list[Job]:
    """PAI-shaped synthetic workload. One epoch = ``epoch_minutes``."""
    rng = random.Random(seed)
    # User heterogeneity: a few heavy users, many light ones (Zipf-ish), each
    # with its own value rate ($ per GPU-epoch) and urgency.
    weights = [1.0 / (k + 1) ** 0.8 for k in range(n_users)]
    rates = [math.exp(rng.gauss(math.log(4.0), 0.6)) for _ in range(n_users)]
    half_lives = [rng.choice([12.0, 24.0, 48.0, 96.0]) for _ in range(n_users)]
    gpu_choices = [1, 1, 1, 1, 1, 1, 2, 2, 4, 8]  # ~60% single-GPU, tail to 8
    jobs = []
    # Bursty arrivals: diurnal-ish intensity over the horizon
    for j in range(n_jobs):
        while True:
            t = rng.uniform(0, horizon * 0.8)
            if rng.random() < 0.6 + 0.4 * math.sin(2 * math.pi * t / 144.0) ** 2:
                break
        u = rng.choices(range(n_users), weights)[0]
        minutes = math.exp(rng.gauss(math.log(60.0), 1.3))  # median 1h, long tail
        duration = max(1, min(int(round(minutes / epoch_minutes)), 144))
        gpus = rng.choice(gpu_choices)
        value = rates[u] * gpus * duration * math.exp(rng.gauss(0, 0.3))
        jobs.append(
            Job(
                job_id=f"j{j}",
                user=f"u{u}",
                submit=int(t),
                duration=duration,
                gpus=gpus,
                cpus=gpus * rng.choice([4, 6, 8]),
                ram_gb=gpus * rng.choice([16, 32, 64]),
                value=round(value, 2),
                half_life=half_lives[u],
            )
        )
    jobs.sort(key=lambda jb: (jb.submit, jb.job_id))
    return jobs


def load_pai_trace(
    path: str | Path,
    limit: int | None = 2000,
    seed: int = 0,
    epoch_seconds: int = 600,
    max_gpus: int = 8,
) -> list[Job]:
    """
    Load jobs from ``pai_task_table.csv``. Columns used: job_name, inst_num,
    plan_gpu (100 = one GPU), plan_cpu (100 = one core), plan_mem (GB),
    start_time, end_time, user (from the job table if merged; else job_name).
    Values are synthesised as in ``synthetic_pai_like`` since the trace has
    no willingness-to-pay field.
    """
    rng = random.Random(seed)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        # The official CSV ships without a header; fall back to its documented order.
        fieldnames = None if "start_time" in first else PAI_TASK_COLUMNS
        reader = csv.DictReader(f, fieldnames=fieldnames)
        for row in reader:
            try:
                if row.get("status", "Terminated") not in ("Terminated", ""):
                    continue
                start, end = float(row["start_time"]), float(row["end_time"])
                inst = int(float(row.get("inst_num") or 1))
                gpu = float(row.get("plan_gpu") or 0) / 100.0 * inst
            except (KeyError, ValueError):
                continue
            if gpu <= 0 or end <= start:
                continue
            rows.append((start, end, row, inst, gpu))
            if limit and len(rows) >= limit * 3:
                break
    rows.sort(key=lambda r: r[0])
    rows = rows[:limit] if limit else rows
    if not rows:
        return []
    t0 = rows[0][0]
    rates: dict[str, float] = {}
    jobs = []
    for k, (start, end, row, inst, gpu) in enumerate(rows):
        user = row.get("user") or row.get("job_name", f"u{k}")[:6]
        rates.setdefault(user, math.exp(rng.gauss(math.log(4.0), 0.6)))
        gpus = max(1, min(max_gpus, int(math.ceil(gpu))))
        duration = max(1, min(144, int(round((end - start) / epoch_seconds))))
        cpus = max(1, int(math.ceil(float(row.get("plan_cpu") or 100) / 100.0 * inst)))
        ram = max(1, int(math.ceil(float(row.get("plan_mem") or 8) * inst)))
        jobs.append(
            Job(
                job_id=f"pai{k}",
                user=user,
                submit=int((start - t0) // epoch_seconds),
                duration=duration,
                gpus=gpus,
                cpus=cpus,
                ram_gb=ram,
                value=round(rates[user] * gpus * duration * math.exp(rng.gauss(0, 0.3)), 2),
                half_life=rng.choice([12.0, 24.0, 48.0, 96.0]),
                meta={"source": "alibaba-pai-2020"},
            )
        )
    return jobs
