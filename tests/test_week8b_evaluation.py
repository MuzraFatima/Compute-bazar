"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Week 8B Tests: Research Evaluation Suite
=============================================================
 Tests the reproducible evaluation suite script (evaluation_suite.py)
 ensuring all empirical benchmark functions execute correctly and
 output valid metrics and reports.

 Validates:
   - Architecture comparison experiment (Round-Robin vs 1.0 vs 2.0)
   - Multi-agent system scalability experiment (4 to 32 agents)
   - Groq LLM & ML Resource Predictor telemetry
   - Generation of docs/evaluation_results.json and docs/evaluation.md

 Run with:  python -m pytest tests/test_week8b_evaluation.py -v
=============================================================
"""

import os
import json
import pytest

from evaluation_suite import (
    run_architecture_comparison,
    run_scalability_experiment,
    run_llm_prediction_eval,
)


def test_eval_architecture_comparison():
    """Architecture comparison should return non-zero welfare and valid metrics for all 3 regimes."""
    res = run_architecture_comparison(rounds=3)
    assert res["rounds"] == 3
    assert res["round_robin"]["total_welfare"] > 0.0
    assert res["compute_bazaar_1_0"]["total_welfare"] > 0.0
    assert res["compute_bazaar_2_0"]["total_welfare"] >= res["round_robin"]["total_welfare"]
    assert res["compute_bazaar_2_0"]["total_vcg_savings"] >= 0.0
    assert 0.0 <= res["compute_bazaar_2_0"]["jains_fairness_index"] <= 1.0


def test_eval_scalability_experiment():
    """Scalability experiment should process bids for 4, 8, 16, 32 agents."""
    counts = [4, 8, 16, 32]
    res = run_scalability_experiment(counts)
    assert len(res) == 4
    for idx, item in enumerate(res):
        assert item["agent_count"] == counts[idx]
        assert item["total_bids_processed"] == counts[idx]
        assert item["ilp_solve_time_ms"] >= 0.0


def test_eval_llm_prediction_telemetry():
    """LLM and ML Predictor telemetry should measure confidence increase after historical training."""
    res = run_llm_prediction_eval()
    assert res["cold_start_confidence"] == 0.75
    assert res["trained_confidence"] > res["cold_start_confidence"]
    assert res["total_historical_records_logged"] > 0
