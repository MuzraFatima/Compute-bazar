"""
Mechanisms for bidders who don't know their own value.

LLM agents estimate a task's value and resource needs from a description.
Those estimates are noisy and often systematically biased. This package
models that setting and compares mechanisms that settle on estimates vs
realized outcomes, learn per-agent calibration, and allocate robustly under
uncertain resource needs. See docs/theory.md for the formal statements.
"""

from compute_bazaar.uncertainty.agents import EstimatingAgent, Task, make_population
from compute_bazaar.uncertainty.mechanisms import (
    MECHANISMS,
    CalibratedVCG,
    ExecutionContingentVCG,
    Mechanism,
    PointVCG,
    RoundOutcome,
)

__all__ = [
    "MECHANISMS",
    "CalibratedVCG",
    "EstimatingAgent",
    "ExecutionContingentVCG",
    "Mechanism",
    "PointVCG",
    "RoundOutcome",
    "Task",
    "make_population",
]
