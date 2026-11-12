# Theory: combinatorial markets for bidders who don't know their own value

This note states the formal results behind `compute_bazaar/uncertainty/` and points to the experiment that probes each one. Proofs are short and self-contained.

## Setting

- Resources $r \in R$ have integer units $u_r$. A task $i$ (owned by agent $a(i)$) needs units $q_{ir}$ and has a value $v_i$ that is realized only if it gets enough units.
- The **feasible set** $\mathcal F \subseteq \{0,1\}^n$ contains allocations $x$ with $\sum_i q_{ir} x_i \le u_r$ for all $r$, plus any XOR constraints.
- LLM agents don't observe $v_i$ or $q_{ir}$. They report estimates $\hat v_i$ and $\hat q_{ir}$, formed from a task description. These estimates are noisy and, crucially, biased in agent-specific ways.
- $W(x; v) = \sum_i x_i v_i$. $x^*(v) \in \arg\max_{x\in\mathcal F} W(x; v)$ is the efficient allocation, and $\hat x = x^*(\hat v)$ is what an exact auction picks from the reports.

## Proposition 1 (price of miscalibration)

For any true values $v$ and estimates $\hat v$, with demands known,

$$ W(x^*; v) - W(\hat x; v) \;\le\; \sum_{i \in \hat x \,\cup\, x^*} |\hat v_i - v_i| . $$

*Proof.* Write $e_i = \hat v_i - v_i$. Then
$W(\hat x; v) = W(\hat x;\hat v) - \sum_{i\in\hat x} e_i \ge W(x^*;\hat v) - \sum_{i\in\hat x} e_i = W(x^*; v) + \sum_{i\in x^*} e_i - \sum_{i\in\hat x} e_i$,
where the inequality is optimality of $\hat x$ for $\hat v$. Rearranging and bounding each sum by absolute values gives the claim. ∎

**Corollary 1a (only heterogeneous bias hurts).** If $\hat v_i = c\, v_i$ for a common $c > 0$, then $\hat x = x^*$ and the loss is zero, because the argmax is scale-invariant. Under multiplicative errors $\hat v_i = v_i e^{b_{a(i)} + \varepsilon_i}$, removing each agent's bias $b_a$ up to a *common* constant leaves only the noise term in the bound. This is why `CalibratedVCG` estimates per-agent log-biases: it doesn't need to know the absolute truth, only each agent's bias relative to the others.

*Empirical check:* `scripts/run_uncertainty_experiments.py` verifies the inequality on every round of a values-only variant (true demands, estimated values). It is never violated; the realised loss is typically a small fraction of the bound (`proposition1_check` in `results/uncertainty_experiments.json`).

## Proposition 2 (execution-contingent VCG aligns incentives, but needs calibrated reports for IR)

Let reports $r_i$ be point estimates of value. Allocate $x(r) = x^*(r)$. After execution, charge each winning agent

$$ p_a = h_a(r_{-a}) - \sum_{j:\, a(j)\neq a} x_j(r)\, v^{\text{real}}_j, \qquad h_a(r_{-a}) = \max_{x \in \mathcal F} \sum_{j:\,a(j)\ne a} x_j r_j . $$

(a) **Alignment.** Agent $a$'s utility is $u_a = \sum_j x_j(r)\,v_j^{\text{real}} - h_a(r_{-a})$: realized social welfare minus a term $a$ cannot affect. If the other agents report their true conditional expectations $r_j = \mathbb E[v^{\text{real}}_j]$, and realized values don't depend on reports given the allocation, then reporting $r_a = \mathbb E[v_a^{\text{real}} \mid \text{info}_a]$ maximizes $\mathbb E[u_a]$. Truthful reporting of expectations is an ex-interim Nash equilibrium.

*Proof.* $\mathbb E[u_a] = \sum_j x_j(r)\,\mathbb E[v_j^{\text{real}}] - h_a(r_{-a})$. With truthful others, $x(r)$ evaluated at $r_a = \mathbb E[v_a]$ maximizes the first term over $\mathcal F$, and any other report selects an allocation that is weakly worse in expectation. ∎ This is the classical two-stage argument (Mezzetti 2004; Porter, Ronen, Shoham & Tennenholtz 2008) applied to the multi-unit combinatorial setting with XOR menus.

(b) **IR fails under biased reports.** If others overstate, $h_a(r_{-a})$ is inflated by up to $\sum |r_j - \mathbb E v_j|$ over the counterfactual winners, while the subtracted realized term is not. Expected utility then drops by that amount, even for a truthful agent. Incentives are unchanged (the shift is constant in $r_a$), but individual rationality breaks.

*Empirical check:* with raw LLM-style estimates, `vcg_contingent` shows ex-post IR violations for most winners and a negative mean winner utility. Calibrating others' reports (`vcg_calibrated`) brings $h_a$ back near $\mathbb E[\cdot]$ and restores positive mean utility. This is the theory's prediction, and the reason calibration and contingent settlement belong together.

## Proposition 3 (risk-adjusted XOR menus maximize expected realized welfare)

Suppose task $i$ succeeds iff its reservation covers its true need on every resource, and that success depends only on its own reservation. For a menu of reservation levels $k \in K_i$ with success probabilities $\pi_{ik}$, bid the XOR alternatives

$$ (\text{demand } \lceil f_k \hat q_i \rceil,\ \text{value } r_i\, \pi_{ik}), \qquad k \in K_i . $$

Then the exact WDP over these bids maximizes $\mathbb E[\text{realized welfare}]$ over all allocations that give each task at most one menu level.

*Proof.* Under the assumption, expected realized welfare of a menu choice $(x_{ik})$ is $\sum_{i,k} x_{ik}\, \mathbb E[v_i]\, \pi_{ik}$ by linearity of expectation. With $r_i = \mathbb E[v_i]$ this is exactly the WDP objective, and the XOR and capacity constraints define the same feasible set. ∎

**Corollary 3a.** With estimated probabilities $\hat \pi$, Proposition 1 applied to the menu bids bounds the loss by $\sum r_i |\hat\pi_{ik} - \pi_{ik}|$ over the chosen and optimal alternatives. Calibration error in success rates costs welfare linearly, which motivates the Beta-smoothed per-agent estimates in `CalibratedVCG.success_prob`.

*Why it matters:* a fixed reservation rule (point estimate, the agent's own 90% bound, or a fixed 1.5×) either overruns or over-reserves regardless of contention. The menu lets the market decide, per round, how much overrun risk to accept on which tasks.

## What is *not* claimed

- **Dynamic incentives.** Calibration uses each agent's past outcomes only, so a current report never moves that agent's current correction. Across rounds, though, an agent could try to shape its future bias estimate. That is a repeated-game question; Phase 4 studies it empirically with adaptive and LLM bidders rather than claiming a theorem.
- **Budget balance.** VCG-family payments are not budget balanced, and contingent payments can be negative (the market pays the agent) when others' tasks over-deliver.
- **Correlated outcomes.** Proposition 3 assumes a task's success depends only on its own reservation. Shared-resource interference would need a joint model.
