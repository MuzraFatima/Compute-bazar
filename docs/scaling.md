# Solver scaling

`scripts/benchmark_scaling.py` times exact winner determination (WDP) and full VCG pricing on random multi-unit instances: 20 resources with 2–8 units, two OR bids per agent over 1–4 resources, values per unit with noise. Near-identical value densities make these hard for branch-and-bound, so they're a pessimistic case. Medians of 2 instances per size on a Windows laptop; times are machine-dependent.

| Bids | WDP (ms) | VCG exact, sequential (s) | VCG exact, 8 threads (s) | VCG 1% gap, 8 threads (s) | Max payment drift with gap (% of V*) |
|--:|--:|--:|--:|--:|--:|
| 50 | 330 | 4.8 | 2.6 | 2.5 | 0.4% |
| 100 | 574 | 18.4 | 8.1 | 6.7 | 9.7% |
| 200 | 680 | 21.0 | 11.9 | 9.6 | 5.6% |
| 400 | 1077 | 24.9 | 12.4 | 10.0 | 2.5% |

## What helps, and what doesn't

- **WDP stays around a second up to 400 bids.** The ILP is not the bottleneck; VCG's one counterfactual solve per winning agent is.
- **Parallel counterfactuals give a ~2× speedup** (`workers=` or `COMPUTE_BAZAAR_VCG_WORKERS`). CBC runs as a subprocess, so threads parallelize for real. The speedup is capped by the slowest counterfactual and by process start-up cost.
- **The exact uncontested-winner shortcut** skips a winner's counterfactual when no losing bid asks for any resource it holds; its payment is then provably 0. It matters most in sparse markets and is covered by brute-force tests.
- **A MIP gap buys little here and costs exactness.** At 1% the gap saves 10–20% more time, but it can change *which* bids win, so individual payments drifted by up to 9.7% of V*. It is off by default and documented as a speed knob, not an ε-truthfulness guarantee.
- **HiGHS vs CBC**: HiGHS (in-process, `COMPUTE_BAZAAR_SOLVER=highs`) was not consistently faster on these instances, so CBC stays the default.

## In context

The scheduler runs one auction per 10-minute epoch, and the trace experiments bid on at most 80 jobs per epoch, so exact VCG is comfortably real-time there. Markets with thousands of simultaneous bids would need column generation or incremental counterfactuals that reuse the main solve's search tree. That is listed as future work.
