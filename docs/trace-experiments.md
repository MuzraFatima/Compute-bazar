# Trace-driven scheduling experiments

`scripts/run_trace_experiments.py` replays a PAI-shaped GPU workload through the epoch simulator in `compute_bazaar/sim/` and compares schedulers on identical job streams.

## Setup

- **Workload**: `synthetic_pai_like`, calibrated to the published shape of the Alibaba PAI 2020 GPU trace (Weng et al., NSDI 2022). About 60% of jobs use a single GPU with a tail up to 8, runtimes are log-normal (median 1 h, tail to 24 h), arrivals are bursty, and users are Zipf-like. **This is synthetic, not the trace itself.** To use the real data, run `python scripts/fetch_traces.py`, then `--pai data/pai_task_table.csv`.
- **Values**: per-user value rate × GPU-epochs × noise. Values halve after a user-specific wait (12–96 epochs), and a job is abandoned once 95% of its value has decayed.
- **Cluster**: 64 GPUs, 512 CPU cores, 2 TB RAM; 10-minute epochs; non-preemptive.
- **Scale**: 1,200 jobs from 120 users per seed, 5 seeds.

## Policies

| Policy | What it does |
|:--|:--|
| `fifo` | Arrival order with head-of-line blocking |
| `fifo_backfill` | Arrival order, skipping jobs that don't fit |
| `drf` | Dominant Resource Fairness across users (Ghodsi et al., NSDI 2011) |
| `themis` | Finish-time fairness: the worst-off half by estimated ρ compete on value density (after Mahajan et al., NSDI 2020) |
| `greedy_value` | Highest value per GPU-epoch first |
| `bazaar_myopic` | Exact multi-unit combinatorial auction each epoch on current values |
| `bazaar` | The same auction with each bid net of a per-GPU-epoch reserve at the queue's median value density |

## Results (mean ± std over 5 seeds)

| Policy | Welfare | Avg JCT (epochs) | FTF rho mean | FTF rho p95 | Jain (1/rho) | GPU util | Dropped | Decision ms |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| fifo | 65,061 ± 10,279 | 69.0 | 17.02 | 66.6 | 0.703 | 68.5% | 226 | 0.1 |
| fifo_backfill | 75,261 ± 11,993 | 54.6 | 12.55 | 48.8 | 0.774 | 74.9% | 134 | 0.3 |
| drf | 97,121 ± 14,088 | 34.6 | 7.14 | 29.8 | 0.925 | 75.5% | 89 | 1.1 |
| themis | 117,038 ± 14,902 | 21.3 | 2.36 | 4.4 | 0.974 | 77.0% | 34 | 0.1 |
| greedy_value | 149,286 ± 28,156 | 19.8 | 2.68 | 7.9 | 0.965 | 77.5% | 62 | 0.1 |
| bazaar_myopic | 145,203 ± 25,585 | 31.5 | 8.04 | 37.0 | 0.933 | 80.1% | 210 | 63.8 |
| bazaar | 151,923 ± 33,690 | 26.5 | 5.79 | 23.8 | 0.933 | 79.1% | 164 | 48.4 |

Raw per-seed data: [`results/trace_experiments.json`](../results/trace_experiments.json).

## What this shows

1. **Value-aware schedulers dominate value-agnostic queues.** FIFO loses more than half the achievable welfare to head-of-line blocking, and backfilling recovers only part of it.
2. **Running the auction myopically isn't enough.** Maximising welfare within one epoch packs long jobs that block denser work later. It ends up *below* simple greedy density on welfare and drops more jobs.
3. **A duration-aware reserve fixes most of that.** `bazaar` has the highest mean welfare and utilisation. The margin over greedy (+1.8%) is within seed variance, so the honest claim is "at least as good as the best heuristic, with a truthful payment rule attached", not a large win.
4. **Fairness is the open gap.** Themis keeps finish-time fairness near ideal (p95 ρ ≈ 4) while `bazaar` lets low-value jobs expire (p95 ρ ≈ 24, ~160 drops). That trade-off motivates the fairness-aware mechanism in Phase 4.
5. **Cost.** The exact ILP takes ~50 ms per scheduling decision, against <1 ms for the heuristics. That's fine for 10-minute epochs, and Phase 5 works on it anyway.

## Not covered

- **Gavel** (heterogeneity-aware, OSDI 2020) needs per-job throughput on several GPU types, which the simulated cluster doesn't model.
- **Tiresias** (NSDI 2019) relies on preemption, and the simulator is non-preemptive. Both are listed as future work rather than approximated badly.
