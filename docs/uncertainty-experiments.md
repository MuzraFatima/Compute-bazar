# Bidders who don't know their own value: experiments

`scripts/run_uncertainty_experiments.py` (3 seeds × 200 rounds, 12 agents, capacity 8 GPU / 16 CPU / 16 RAM units). Agents report an *estimated* value and need (see `compute_bazaar/uncertainty/agents.py`): half the population is overconfident (overstated value, too-tight intervals), some are underconfident, the rest calibrated. A winner whose true need exceeds its reservation overruns and realises nothing. Theory: [`theory.md`](theory.md).

## Results

| Mechanism | Welfare (% of oracle) | Overrun rate | Ex-post IR violations | Mean winner utility | Revenue |
|:--|--:|--:|--:|--:|--:|
| vcg_point | 72.1% ± 2.4% | 25.5% | 28.0% | 27.6 | 45,478 |
| vcg_point_upper90 | 73.7% ± 2.1% | 1.1% | 15.0% | 42.4 | 44,636 |
| vcg_contingent | 72.1% ± 2.4% | 25.5% | 93.3% | -214.6 | 435,733 |
| vcg_contingent_upper90 | 73.7% ± 2.1% | 1.1% | 67.2% | -54.4 | 155,305 |
| vcg_calibrated_fixed | 72.2% ± 1.9% | 1.1% | 31.9% | 29.8 | 58,388 |
| vcg_calibrated | 78.0% ± 1.9% | 15.5% | 30.4% | 46.7 | 30,356 |

`oracle` allocates on true values and needs and is the efficiency ceiling. Raw data: [`results/uncertainty_experiments.json`](../results/uncertainty_experiments.json).

## Reading the table

1. **Point estimates waste a quarter of the value.** Plain VCG on reports overruns on 25% of wins and reaches 72% of oracle welfare. Using the agent's own 90% bound fixes overruns but over-reserves, and gains less than two points.
2. **Contingent settlement alone is not enough, as Proposition 2(b) predicts.** It aligns incentives, but on biased reports the counterfactual term is inflated, so 93% of winners end up with negative realised utility.
3. **Calibration plus risk-adjusted menus is the best realistic mechanism**: 78.0% of oracle welfare, +4.3 points over the best baseline, with winner utility (46.7) close to the oracle's (50.4). It deliberately accepts more overruns (15%) on low-value tasks when capacity is tight, which is the expected-welfare trade-off of Proposition 3.
4. **A fixed calibrated reservation (1.5×) doesn't capture the gain.** The improvement comes from letting the market choose the reservation level per round, not from calibration alone.
5. **Proposition 1 holds on all 600 checked rounds.** Realised loss averages ~5% of the bound (max 71%), so the bound is valid but loose.

## Caveats

- The agents are a parametric model of LLM miscalibration, not LLMs. Phase 4 runs real LLM bidders.
- The menu mechanism solves an ILP with ~50 XOR bids per round, which is slower than the single-bid variants. Phase 5 covers solver speed.
- Revenue under contingent settlement is inflated by the same biased counterfactuals that break IR, so it isn't a meaningful advantage.
