# Strategic behaviour: learning and LLM bidders, core payments, fairness

## Setting

Repeated **Local-Local-Global (LLG)** auctions (`compute_bazaar/strategic/game.py`). Two GPUs, A and B: local L1 wants A, local L2 wants B, global G wants both. Values are redrawn each round (locals U(0,100), global U(0,200)). Three payment rules:

- **VCG**: agent-level Clarke pivot.
- **First-price**: winners pay their bid.
- **Core**: minimum-revenue core, VCG-nearest (`compute_bazaar/strategic/core.py`, Day & Raghavan constraint generation). In LLG it charges locals at least G's bid in total, closing VCG's revenue gap.

We report the bid/value ratio (1 = truthful, <1 shading, >1 overbidding), allocative efficiency (realised value / best possible) and seller revenue.

## A. Learning bidders (Hedge, 5 seeds × 3,000 rounds, last 20% of rounds)

| Mechanism | Efficiency | Revenue | L1 bid/value | L2 bid/value | G bid/value |
|:--|--:|--:|--:|--:|--:|
| vcg | 99.2% | 57.3 | 1.01 | 1.01 | 1.04 |
| first_price | 99.1% | 48.6 | 0.36 | 0.35 | 0.40 |
| core | 97.1% | 56.6 | 0.73 | 0.71 | 1.06 |

- **VCG**: learners converge to truthful bidding with ~99% efficiency. The dominant-strategy property survives adaptive play.
- **First-price**: everyone shades heavily (~0.35–0.40 of value), the textbook best response. Efficiency stays high here because the roles are symmetric.
- **Core**: G stays truthful, but **locals shade to ~0.72** to free-ride on each other's payment toward G's bid, costing ~2 points of efficiency. Core payments fix VCG's revenue and collusion problems at the price of incentive compatibility for locals. This is the known trade-off (Day & Milgrom 2008; Ausubel & Baranov), reproduced in the engine.

## B. LLM bidders (live Groq models, 15 rounds per condition, seed 0)

Each bidder gets the rules, its value, its last five outcomes and, in chat conditions, everyone's public messages from the previous round. Calls that still failed after two retries are run as truthful bids so the round can proceed, but are **excluded** from the bid/value statistics.

| Model | Condition | Efficiency | Revenue | L1 bid/value | L2 bid/value | G bid/value | Failed calls |
|:--|:--|--:|--:|--:|--:|--:|--:|
| openai/gpt-oss-120b | vcg_informed | 100.0% | 65.6 | 1.00 | 1.00 | 1.00 | 0 |
| openai/gpt-oss-120b | vcg_uninformed (partial) | 99.7% | 60.5 | 1.00 | 1.00 | 0.80 | 29 |
| openai/gpt-oss-20b | vcg_informed | 100.0% | 76.0 | 1.00 | 1.00 | 1.00 | 0 |
| openai/gpt-oss-20b | vcg_uninformed (partial) | 96.1% | 62.6 | 1.00 | 0.95 | 0.70 | 6 |
| openai/gpt-oss-20b | first_price (partial) | 98.8% | 137.5 | 0.59 | 1.00 | n/a | 23 |
| qwen/qwen3.8-27b | vcg_informed | 100.0% | 65.6 | 1.00 | 1.00 | 1.00 | 0 |
| qwen/qwen3.8-27b | vcg_uninformed (partial) | 100.0% | 65.6 | 1.00 | 1.00 | 1.00 | 1 |
| qwen/qwen3.8-27b | first_price | 100.0% | 132.3 | 1.00 | 0.86 | 1.04 | 0 |
| qwen/qwen3.8-27b | vcg_chat | 100.0% | 65.6 | 1.00 | 1.00 | 1.00 | 0 |
| qwen/qwen3.8-27b | core_chat | 100.0% | 85.6 | 1.00 | 1.00 | 1.00 | 0 |

**What the LLMs did**

- **Told the rule, every model is exactly truthful.** Under VCG with the dominance hint, all three models bid their value in every round (ratio 1.00).
- **Not told, the gpt-oss models' global bidder shades.** G bid ~0.80 (120b) and ~0.70 (20b) of value, costing a few points of efficiency (96.1% for 20b), while Qwen stayed truthful. Telling bidders the rule matters for LLMs in a way it doesn't for Hedge learners, who discover truthfulness on their own.
- **Qwen doesn't shade under first-price** (0.86–1.04, against ~0.35 for learners): it leaves money on the table and pushes revenue up. LLM bidders are not rational best-responders by default.
- **No collusion through chat.** With a public channel, Qwen bidders announced truthful bidding ("VCG ensures truthful bidding is dominant") and never proposed coordination, under VCG or core payments.
- **Coverage:** gpt-oss-120b and gpt-oss-20b both hit Groq's daily token cap mid-study, so their later conditions are missing or partial (flagged). llama-3.3-70b was planned but retired by Groq at run time.

Example chat messages:

- *qwen/qwen3.8-27b, vcg_chat, round 0, L1:* "L1 bids truthfully for GPU A. VCG ensures truthful bidding is dominant."
- *qwen/qwen3.8-27b, vcg_chat, round 0, L2:* "Bidding my true valuation for GPU B. VCG ensures efficiency."
- *qwen/qwen3.8-27b, vcg_chat, round 0, G:* "G is bidding the full package value. L1 and L2, if your combined bids are >= 84.11, you will win and pay 0. If your combined bids are < 84.11, G wins and pays t"
- *qwen/qwen3.8-27b, vcg_chat, round 1, L1:* "L1 bids truthfully at $25.89. VCG mechanism makes truthful bidding a dominant strategy. G's package bid must exceed my bid plus L2's bid to win, ensuring fair a"
- *qwen/qwen3.8-27b, vcg_chat, round 1, L2:* "Bidding my true valuation for GPU B. VCG ensures truthful bidding is dominant."
- *qwen/qwen3.8-27b, vcg_chat, round 1, G:* "Bidding my true package value of 80.99. VCG mechanism incentivizes truthfulness; I will only win if my value exceeds the sum of L1 and L2's bids."
- *qwen/qwen3.8-27b, vcg_chat, round 2, L1:* "Bidding my true valuation of $78.38. VCG mechanism makes truthful bidding a dominant strategy. G wins only if their package value exceeds the sum of L1 and L2's"
- *qwen/qwen3.8-27b, vcg_chat, round 2, L2:* "Bidding my true valuation of $30.33 for GPU B. VCG ensures truthful bidding is the dominant strategy to maximize long-term profit."

Full transcripts, including every prompt outcome and message, are in [`results/strategic_study.json`](../results/strategic_study.json).

## C. Fairness knob for the scheduler

The Phase 2 auction maximised welfare but trailed Themis on finish-time fairness. `bazaar_fair_<alpha>` weights each bid by rho-hat^alpha, the job's finish-time-fairness ratio if started now. The weight depends on waiting time, not on the bid, so VCG on the weighted values is an **affine maximiser** and stays truthful (Roberts 1979). Three seeds, PAI-shaped workload:

| Policy | Welfare | FTF rho p95 | Jain (1/rho) | Dropped |
|:--|--:|--:|--:|--:|
| greedy_value | 154,207 | 8.1 | 0.963 | 62 |
| themis | 116,268 | 4.3 | 0.975 | 32 |
| bazaar | 159,515 | 23.7 | 0.931 | 174 |
| bazaar_fair_0.5 | 157,867 | 20.2 | 0.929 | 186 |
| bazaar_fair_1 | 157,834 | 14.8 | 0.928 | 164 |
| bazaar_fair_2 | 155,975 | 10.3 | 0.915 | 134 |

Raising alpha trades welfare for fairness smoothly: alpha=2 cuts p95 rho from 23.7 to 10.3 for ~2% welfare. Themis is still the fairest policy and greedy density sits close to the alpha=2 point, so the claim is a truthful, tunable frontier rather than a dominating policy.

## Caveats

- The LLM study is small (one seed, 10–15 rounds per condition, three models, some conditions partial), so read it as indicative. Groq free-tier token limits (per minute and per day) were the binding constraint.
- LLM bidders see only their own history and the public chat; they never see other bids, which limits explicit coordination.
