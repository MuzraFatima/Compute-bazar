"""Tests for the trace-driven cluster simulator and scheduling policies."""

import pytest

from compute_bazaar.sim import POLICIES, ClusterSimulator, Job, load_pai_trace, synthetic_pai_like
from compute_bazaar.sim.policies import bazaar, drf, fifo, fifo_backfill

SMALL_CAP = {"gpu": 4, "cpu": 64, "ram": 512}


def job(jid, user="u", submit=0, duration=2, gpus=1, value=10.0, half_life=1e9):
    return Job(jid, user, submit, duration, gpus, gpus * 4, gpus * 16, value, half_life)


def test_value_decays_with_waiting():
    j = job("a", submit=0, value=100.0, half_life=10)
    assert j.value_if_started(0) == pytest.approx(100.0)
    assert j.value_if_started(10) == pytest.approx(50.0)


@pytest.mark.parametrize("name", list(POLICIES))
def test_every_policy_completes_a_small_trace_without_overallocation(name):
    jobs = synthetic_pai_like(n_jobs=80, n_users=10, horizon=60, seed=3)
    sim = ClusterSimulator(jobs, capacity={"gpu": 8, "cpu": 64, "ram": 256})
    r = sim.run(POLICIES[name], name)
    assert r.completed + r.dropped == len([j for j in jobs if sim.fits(j, sim.capacity)])
    assert 0.0 < r.utilization <= 1.0
    assert r.ftf_mean >= 1.0


def test_fifo_blocks_behind_a_large_head_job_but_backfill_does_not():
    jobs = [job("a_run", gpus=3, duration=5), job("b_big", gpus=4), job("c_small", gpus=1)]
    sim = ClusterSimulator(jobs, capacity=SMALL_CAP)
    free = {"gpu": 1, "cpu": 60, "ram": 500}
    assert fifo([jobs[1], jobs[2]], free, 0, sim) == []
    assert [j.job_id for j in fifo_backfill([jobs[1], jobs[2]], free, 0, sim)] == ["c_small"]


def test_drf_serves_the_user_with_the_lowest_dominant_share():
    sim = ClusterSimulator([], capacity=SMALL_CAP)
    sim.running = [job("r", user="heavy", gpus=2)]
    pending = [job("h2", user="heavy", gpus=1), job("l1", user="light", gpus=1)]
    chosen = drf(pending, {"gpu": 1, "cpu": 60, "ram": 500}, 0, sim)
    assert [j.job_id for j in chosen] == ["l1"]


def test_bazaar_beats_greedy_packing_on_a_combinatorial_instance():
    # One GPU-dense job vs two jobs that together fill the cluster better.
    sim = ClusterSimulator([], capacity=SMALL_CAP)
    pending = [job("dense", gpus=3, value=33.0), job("x", gpus=2, value=20.0), job("y", gpus=2, value=20.0)]
    chosen = bazaar(pending, dict(SMALL_CAP), 0, sim, reserve_quantile=None)
    assert sorted(j.job_id for j in chosen) == ["x", "y"]


def test_pai_loader_reads_headerless_task_table(tmp_path):
    csv_path = tmp_path / "pai_task_table.csv"
    csv_path.write_text(
        "jobA,worker,2,Terminated,1000,4600,600,30,50,V100\n"
        "jobB,worker,1,Terminated,2000,2600,400,10,100,T4\n"
        "jobC,worker,1,Failed,2000,2600,400,10,100,T4\n"
        "jobD,worker,1,Terminated,2100,2700,400,10,0,\n"
    )
    jobs = load_pai_trace(csv_path, limit=None)
    assert [j.gpus for j in jobs] == [1, 1]  # 2×0.5 GPU → 1 GPU; CPU-only and failed rows skipped
    assert jobs[0].duration == 6 and jobs[0].submit == 0
    assert jobs[1].submit == 1
