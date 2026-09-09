"""
=============================================================
 COMPUTE BAZAAR 2.0 -- ML Resource Predictor (Week 7)
=============================================================
 ML-based resource prediction layer that forecasts:
   - GPU, CPU, and RAM capacity requirements
   - API / LLM token consumption
   - Expected execution duration (seconds)
   - Estimated monetary cost ($)

 Utilizes historical execution data when available and falls
 back to task feature extraction models when cold-starting.
=============================================================
"""

from __future__ import annotations

import re
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class ResourcePrediction:
    """
    Structured outcome of an ML resource prediction.

    Fields:
        task_description: Original task prompt
        task_category: Categorized task domain ('llm_training', 'data_etl', 'research', etc.)
        predicted_gpus: Forecasted GPU slots (0 to 4)
        predicted_cpu_cores: Forecasted CPU cores (1 to 32)
        predicted_ram_gb: Forecasted RAM in GB (2 to 128)
        predicted_api_tokens: Forecasted LLM API tokens (0 to 1,000,000)
        predicted_execution_time_sec: Forecasted runtime in seconds
        predicted_cost: Forecasted dollar cost ($)
        recommended_bundle: Recommended list of resource names
        confidence_score: Prediction confidence from 0.0 to 1.0
    """
    task_description: str
    task_category: str
    predicted_gpus: float
    predicted_cpu_cores: float
    predicted_ram_gb: float
    predicted_api_tokens: float
    predicted_execution_time_sec: float
    predicted_cost: float
    recommended_bundle: List[str] = field(default_factory=list)
    confidence_score: float = 0.80

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_description": self.task_description,
            "task_category": self.task_category,
            "predicted_gpus": round(self.predicted_gpus, 2),
            "predicted_cpu_cores": round(self.predicted_cpu_cores, 2),
            "predicted_ram_gb": round(self.predicted_ram_gb, 2),
            "predicted_api_tokens": round(self.predicted_api_tokens, 2),
            "predicted_execution_time_sec": round(self.predicted_execution_time_sec, 2),
            "predicted_cost": round(self.predicted_cost, 2),
            "recommended_bundle": self.recommended_bundle,
            "confidence_score": round(self.confidence_score, 2),
        }


@dataclass
class ExecutionRecord:
    """Historical execution record used to train the ML predictor."""
    task_description: str
    task_category: str
    allocated_resources: List[str]
    execution_time_sec: float
    actual_cost: float
    tokens_used: float = 0.0


class ResourcePredictor:
    """
    ML-based Resource Predictor for Compute Bazaar tasks.

    Learns from historical execution data (`record_execution()`) and
    predicts compute requirements, execution duration, and costs for new tasks.
    """

    def __init__(self):
        self.execution_history: List[ExecutionRecord] = []
        
        # Base unit pricing defaults for cost prediction ($ per unit per min)
        self.unit_costs = {
            "gpu": 0.50,       # $0.50 / slot / min
            "cpu": 0.05,       # $0.05 / core / min
            "ram": 0.02,       # $0.02 / GB / min
            "api_budget": 0.10, # $0.10 / unit
            "llm_api": 0.0001, # $0.0001 / token
        }

    def record_execution(
        self,
        task_description: str,
        allocated_resources: List[str],
        execution_time_sec: float,
        actual_cost: float,
        tokens_used: float = 0.0,
        task_category: Optional[str] = None,
    ):
        """
        Record actual task execution metrics into the historical dataset.
        """
        cat = task_category or self._classify_task(task_description)
        record = ExecutionRecord(
            task_description=task_description,
            task_category=cat,
            allocated_resources=allocated_resources,
            execution_time_sec=execution_time_sec,
            actual_cost=actual_cost,
            tokens_used=tokens_used,
        )
        self.execution_history.append(record)
        logger.info(f"Predictor recorded execution | cat={cat} | duration={execution_time_sec:.1f}s | cost=${actual_cost:.2f}")

    def _classify_task(self, text: str) -> str:
        """Classify task into category using keyword heuristics."""
        t = text.lower()
        if any(w in t for w in ["fine-tune", "train", "lora", "70b", "model"]):
            return "llm_training"
        elif any(w in t for w in ["etl", "dataset", "clean", "embeddings", "dataframe"]):
            return "data_etl"
        elif any(w in t for w in ["search", "research", "scrape", "literature", "synthesis"]):
            return "research"
        elif any(w in t for w in ["monte carlo", "simulation", "risk", "financial", "quant"]):
            return "quantitative_analysis"
        elif any(w in t for w in ["api", "service", "rest", "backend", "web"]):
            return "software_dev"
        return "general_compute"

    def predict(
        self,
        task_description: str,
        task_category: Optional[str] = None,
    ) -> ResourcePrediction:
        """
        Predict compute requirements, execution time, and cost for a task.

        Args:
            task_description: Natural language task request
            task_category: Optional explicit task domain

        Returns:
            ResourcePrediction containing forecasted metrics and recommended bundle
        """
        cat = task_category or self._classify_task(task_description)
        similar_records = [r for r in self.execution_history if r.task_category == cat]

        if len(similar_records) >= 3:
            # ── ML Historical Inference (Averaging / Regression) ──
            avg_time = statistics.mean(r.execution_time_sec for r in similar_records)
            avg_cost = statistics.mean(r.actual_cost for r in similar_records)
            avg_tokens = statistics.mean(r.tokens_used for r in similar_records)
            confidence = min(0.95, 0.70 + 0.05 * len(similar_records))
        else:
            # ── Heuristic Feature Model (Cold Start) ──────────────
            avg_time, avg_cost, avg_tokens = self._heuristic_feature_prediction(task_description, cat)
            confidence = 0.75

        # Determine resource requirements and recommended bundle
        bundle, gpus, cpus, ram, tokens = self._infer_resource_bundle(task_description, cat)

        # Refine cost estimate if heuristic cost is zero
        if avg_cost <= 0.0:
            duration_min = max(0.5, avg_time / 60.0)
            avg_cost = (
                gpus * self.unit_costs["gpu"] * duration_min +
                cpus * self.unit_costs["cpu"] * duration_min +
                ram * self.unit_costs["ram"] * duration_min +
                tokens * self.unit_costs["llm_api"]
            )
            avg_cost = round(max(5.0, avg_cost), 2)

        return ResourcePrediction(
            task_description=task_description,
            task_category=cat,
            predicted_gpus=float(gpus),
            predicted_cpu_cores=float(cpus),
            predicted_ram_gb=float(ram),
            predicted_api_tokens=float(tokens),
            predicted_execution_time_sec=round(avg_time, 2),
            predicted_cost=round(avg_cost, 2),
            recommended_bundle=bundle,
            confidence_score=round(confidence, 2),
        )

    def _heuristic_feature_prediction(self, text: str, cat: str) -> Tuple[float, float, float]:
        """Cold-start feature heuristic for runtime, cost, and tokens."""
        t = text.lower()
        
        # Baseline per category
        if cat == "llm_training":
            duration = 1800.0 if "70b" in t else 600.0
            cost = 120.0 if "70b" in t else 45.0
            tokens = 50000.0
        elif cat == "data_etl":
            duration = 300.0
            cost = 25.0
            tokens = 10000.0
        elif cat == "research":
            duration = 120.0
            cost = 15.0
            tokens = 25000.0
        elif cat == "quantitative_analysis":
            duration = 240.0
            cost = 30.0
            tokens = 5000.0
        else:
            duration = 150.0
            cost = 20.0
            tokens = 5000.0

        return duration, cost, tokens

    def _infer_resource_bundle(self, text: str, cat: str) -> Tuple[List[str], int, int, int, int]:
        """Infer bundle items and capacity values based on category and prompt text."""
        t = text.lower()
        
        if cat == "llm_training":
            bundle = ["GPU-Slot-1", "GPU-Slot-2", "RAM-Block-32GB"]
            return bundle, 2, 8, 32, 50000
        elif cat == "data_etl":
            bundle = ["RAM-Block-32GB", "CPU-Cores-8"]
            return bundle, 0, 8, 32, 10000
        elif cat == "research":
            bundle = ["API-Budget-100", "RAM-Block-32GB"]
            return bundle, 0, 4, 32, 25000
        elif cat == "quantitative_analysis":
            bundle = ["CPU-Cores-8", "RAM-Block-32GB"]
            return bundle, 0, 8, 32, 5000
        else:
            bundle = ["CPU-Cores-8", "GPU-Slot-1"]
            return bundle, 1, 8, 16, 5000

    def get_model_summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics for the predictor model."""
        categories = {}
        for r in self.execution_history:
            categories[r.task_category] = categories.get(r.task_category, 0) + 1

        return {
            "total_execution_records": len(self.execution_history),
            "tracked_categories": categories,
            "unit_costs": self.unit_costs,
        }
