"""
Evaluate ResourcePredictor on held-out synthetic workloads.
Generates ground-truth synthetic tasks, trains predictor on training set,
then evaluates on held-out test set using MAE and MAPE.

Usage:
    python scripts/evaluate_predictor.py
"""

import sys
import os
import json
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compute_bazaar.prediction.predictor import ResourcePredictor, ExecutionRecord


# Synthetic ground-truth task definitions per category
CATEGORY_GROUND_TRUTH = {
    "llm_training": {
        "desc_template": "Fine-tune a language model for task {i}",
        "gpus": 2, "cpus": 8, "ram_gb": 32,
        "time_sec_range": (600, 2400),
        "cost_range": (40, 150),
        "tokens": 50000,
    },
    "data_etl": {
        "desc_template": "Run ETL pipeline and clean dataset {i}",
        "gpus": 0, "cpus": 8, "ram_gb": 32,
        "time_sec_range": (200, 500),
        "cost_range": (15, 35),
        "tokens": 10000,
    },
    "research": {
        "desc_template": "Literature synthesis and research for topic {i}",
        "gpus": 0, "cpus": 4, "ram_gb": 32,
        "time_sec_range": (80, 180),
        "cost_range": (10, 25),
        "tokens": 25000,
    },
    "quantitative_analysis": {
        "desc_template": "Run Monte Carlo simulation and risk model {i}",
        "gpus": 0, "cpus": 8, "ram_gb": 32,
        "time_sec_range": (150, 350),
        "cost_range": (20, 45),
        "tokens": 5000,
    },
    "general_compute": {
        "desc_template": "General compute backend task {i}",
        "gpus": 1, "cpus": 8, "ram_gb": 16,
        "time_sec_range": (100, 200),
        "cost_range": (15, 30),
        "tokens": 5000,
    },
}


def generate_synthetic_tasks(category, n, rng, start_idx=0):
    """Generate n synthetic tasks with ground truth values."""
    cfg = CATEGORY_GROUND_TRUTH[category]
    tasks = []
    for i in range(n):
        time_sec = rng.uniform(*cfg["time_sec_range"])
        cost = rng.uniform(*cfg["cost_range"])
        tasks.append({
            "desc": cfg["desc_template"].format(i=start_idx + i),
            "category": category,
            "gpus": cfg["gpus"],
            "cpus": cfg["cpus"],
            "ram_gb": cfg["ram_gb"],
            "time_sec": round(time_sec, 2),
            "cost": round(cost, 2),
            "tokens": cfg["tokens"],
        })
    return tasks


def mae(preds, actuals):
    n = len(preds)
    if n == 0:
        return 0.0
    return sum(abs(p - a) for p, a in zip(preds, actuals)) / n


def mape(preds, actuals):
    n = len(preds)
    if n == 0:
        return 0.0
    return 100.0 * sum(
        abs(p - a) / max(abs(a), 1e-9)
        for p, a in zip(preds, actuals)
    ) / n


def main():
    rng = random.Random(2025)
    predictor = ResourcePredictor()

    categories = list(CATEGORY_GROUND_TRUTH.keys())

    # Generate train and test sets (10 train, 10 test per category)
    train_tasks = []
    test_tasks = []
    for cat in categories:
        train_tasks.extend(generate_synthetic_tasks(cat, 10, rng, start_idx=0))
        test_tasks.extend(generate_synthetic_tasks(cat, 10, rng, start_idx=10))

    # Train predictor on training set
    for t in train_tasks:
        predictor.record_execution(
            task_description=t["desc"],
            allocated_resources=[],
            execution_time_sec=t["time_sec"],
            actual_cost=t["cost"],
            tokens_used=t["tokens"],
            task_category=t["category"],
        )

    # Evaluate on held-out test set
    results = []
    for t in test_tasks:
        pred = predictor.predict(t["desc"], task_category=t["category"])
        results.append({
            "category": t["category"],
            "actual_gpus": t["gpus"],
            "pred_gpus": pred.predicted_gpus,
            "actual_cpus": t["cpus"],
            "pred_cpus": pred.predicted_cpu_cores,
            "actual_ram_gb": t["ram_gb"],
            "pred_ram_gb": pred.predicted_ram_gb,
            "actual_time_sec": t["time_sec"],
            "pred_time_sec": pred.predicted_execution_time_sec,
            "actual_cost": t["cost"],
            "pred_cost": pred.predicted_cost,
        })

    # Compute MAE and MAPE per field
    fields = [
        ("gpu_slots",  "pred_gpus",     "actual_gpus"),
        ("cpu_cores",  "pred_cpus",     "actual_cpus"),
        ("ram_gb",     "pred_ram_gb",   "actual_ram_gb"),
        ("time_sec",   "pred_time_sec", "actual_time_sec"),
        ("cost_usd",   "pred_cost",     "actual_cost"),
    ]

    print("\n" + "=" * 70)
    print("  RESOURCE PREDICTOR EVALUATION (50 held-out synthetic tasks)")
    print("  Training set: 50 tasks (10 per category) | Test set: 50 tasks")
    print("=" * 70)
    print(f"\n{'Metric':<15} {'MAE':>10} {'MAPE':>10}")
    print("-" * 40)

    eval_results = {}
    for label, pred_key, actual_key in fields:
        preds = [r[pred_key] for r in results]
        actuals = [r[actual_key] for r in results]
        m = mae(preds, actuals)
        mp = mape(preds, actuals)
        eval_results[label] = {"mae": round(m, 4), "mape_pct": round(mp, 2)}
        print(f"{label:<15} {m:>10.4f} {mp:>9.2f}%")

    print("\nNote: Predictor uses category-conditional averaging (not neural net).")
    print("MAPE > 50% is expected for categorical/heuristic models on small datasets.")

    # Save results
    out = {
        "description": (
            "Evaluation of ResourcePredictor on 50 held-out synthetic tasks "
            "(10 per category). Predictor is trained on 50 synthetic training tasks. "
            "The predictor uses category-conditional historical averaging (no deep learning). "
            "MAPE measures relative error; MAE is in native units."
        ),
        "train_n": len(train_tasks),
        "test_n": len(test_tasks),
        "categories": categories,
        "metrics": eval_results,
    }

    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    os.makedirs(docs_dir, exist_ok=True)
    out_path = os.path.join(docs_dir, "predictor_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Predictor evaluation saved to: {out_path}")

    return eval_results


if __name__ == "__main__":
    main()
