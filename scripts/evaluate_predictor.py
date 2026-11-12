"""
Held-out evaluation of the resource predictor.

Synthetic tasks carry scale cues in their text ("Fine-tune a 13B model for 3
epochs", "ETL over 400GB") and their true runtime/cost scale with those cues
times log-normal noise (sigma = 0.15). That's the situation a category
average can't capture, so we compare:

  * category_mean: the previous predictor (per-category averages),
  * log_ridge:     the learned model in ``compute_bazaar/prediction/learned.py``.

Reports MAE/MAPE for runtime and cost plus the empirical coverage of the
90% predictive interval. GPU/CPU/RAM are template-driven per category and
are not scored here (they'd be zero-error by construction).

    python scripts/evaluate_predictor.py
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger  # noqa: E402

from compute_bazaar.prediction.predictor import ResourcePredictor  # noqa: E402

NOISE_SIGMA = 0.15

# (template, base_time_sec, base_cost, cue sampler -> (text kwargs, multiplier))
CATEGORIES = {
    "llm_training": (
        "Fine-tune a {p}B model for {e} epochs on task {i}",
        600.0,
        45.0,
        lambda r: (lambda p, e: ({"p": p, "e": e}, (p / 7) ** 0.8 * e / 2))(
            r.choice([1, 3, 7, 13, 34, 70]), r.randint(1, 6)
        ),
    ),
    "data_etl": (
        "Run ETL pipeline and clean a {g}GB dataset {i}",
        300.0,
        25.0,
        lambda r: (lambda g: ({"g": g}, (g / 50) ** 0.7))(r.choice([5, 20, 50, 120, 400, 900])),
    ),
    "research": (
        "Literature synthesis over {d} papers for topic {i}",
        120.0,
        15.0,
        lambda r: (lambda d: ({"d": d}, (d / 20) ** 0.6))(r.choice([5, 10, 20, 50, 120])),
    ),
    "quantitative_analysis": (
        "Run Monte Carlo risk model with {s} simulations {i}",
        240.0,
        30.0,
        lambda r: (lambda s: ({"s": s}, (s / 1000) ** 0.5))(r.choice([100, 500, 1000, 5000, 20000])),
    ),
    "general_compute": (
        "General compute backend task {i} over {g}GB",
        150.0,
        20.0,
        lambda r: (lambda g: ({"g": g}, (g / 10) ** 0.4))(r.choice([1, 5, 10, 40, 100])),
    ),
}


def make_tasks(rng: random.Random, per_cat: int, start: int) -> list[dict]:
    tasks = []
    for cat, (tmpl, t0, c0, cue) in CATEGORIES.items():
        for i in range(per_cat):
            kw, mult = cue(rng)
            noise_t = math.exp(rng.gauss(0, NOISE_SIGMA))
            noise_c = math.exp(rng.gauss(0, NOISE_SIGMA))
            tasks.append(
                {
                    "desc": tmpl.format(i=start + i, **kw),
                    "category": cat,
                    "time_sec": round(t0 * mult * noise_t, 2),
                    "cost": round(c0 * mult * noise_c, 2),
                }
            )
    return tasks


def mae(p, a):
    return statistics.mean(abs(x - y) for x, y in zip(p, a, strict=True))


def mape(p, a):
    return 100.0 * statistics.mean(abs(x - y) / abs(y) for x, y in zip(p, a, strict=True))


def evaluate(train: list[dict], test: list[dict]) -> dict:
    learned = ResourcePredictor()
    for t in train:
        learned.record_execution(t["desc"], [], t["time_sec"], t["cost"], task_category=t["category"])

    cat_means = {
        cat: (
            statistics.mean(t["time_sec"] for t in train if t["category"] == cat),
            statistics.mean(t["cost"] for t in train if t["category"] == cat),
        )
        for cat in CATEGORIES
    }

    rows = {"category_mean": {"time": [], "cost": []}, "log_ridge": {"time": [], "cost": []}}
    covered_t = covered_c = 0
    for t in test:
        pred = learned.predict(t["desc"], task_category=t["category"])
        assert pred.model == "log_ridge"
        rows["log_ridge"]["time"].append(pred.predicted_execution_time_sec)
        rows["log_ridge"]["cost"].append(pred.predicted_cost)
        rows["category_mean"]["time"].append(cat_means[t["category"]][0])
        rows["category_mean"]["cost"].append(cat_means[t["category"]][1])
        lo, hi = pred.time_interval_90
        covered_t += lo <= t["time_sec"] <= hi
        lo, hi = pred.cost_interval_90
        covered_c += lo <= t["cost"] <= hi

    actual_t = [t["time_sec"] for t in test]
    actual_c = [t["cost"] for t in test]
    out = {}
    for name, r in rows.items():
        out[name] = {
            "time_mae_sec": round(mae(r["time"], actual_t), 2),
            "time_mape_pct": round(mape(r["time"], actual_t), 2),
            "cost_mae": round(mae(r["cost"], actual_c), 2),
            "cost_mape_pct": round(mape(r["cost"], actual_c), 2),
        }
    out["log_ridge"]["time_interval90_coverage"] = round(covered_t / len(test), 3)
    out["log_ridge"]["cost_interval90_coverage"] = round(covered_c / len(test), 3)
    return out


def main() -> None:
    logger.remove()
    seeds = range(5)
    per_seed = []
    for seed in seeds:
        rng = random.Random(seed)
        train = make_tasks(rng, per_cat=40, start=0)
        test = make_tasks(rng, per_cat=20, start=1000)
        per_seed.append(evaluate(train, test))

    def agg(model: str, key: str) -> dict:
        vals = [r[model][key] for r in per_seed]
        return {"mean": round(statistics.mean(vals), 3), "std": round(statistics.stdev(vals), 3)}

    summary = {model: {key: agg(model, key) for key in per_seed[0][model]} for model in ("category_mean", "log_ridge")}
    report = {
        "setup": {
            "seeds": list(seeds),
            "train_tasks_per_seed": 200,
            "test_tasks_per_seed": 100,
            "noise_sigma_log": NOISE_SIGMA,
            "note": "Runtime/cost scale with parsed text cues; hardware needs are category templates and not scored.",
        },
        "summary": summary,
        "per_seed": per_seed,
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "predictor_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
