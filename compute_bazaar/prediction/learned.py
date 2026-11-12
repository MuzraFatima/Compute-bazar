"""
A small learned model for task runtime and cost.

Features: hashed bag-of-words over the task text, parsed scale cues
("70B" parameters, "500GB"/"2TB" data, "2M rows", "10 epochs") on a log
scale, plus a one-hot task category. Targets are modelled in log space with
closed-form ridge regression (numpy only), which suits multiplicative
effects like "twice the data, roughly twice the time".

The residual spread on the training set gives a log-normal predictive
interval, so the market can treat a forecast as a range rather than a point.
That interval feeds the uncertainty-aware bidding in Phase 3.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass

import numpy as np

HASH_DIM = 64
CATEGORIES = (
    "llm_training",
    "data_etl",
    "research",
    "quantitative_analysis",
    "software_dev",
    "general_compute",
)

_SCALE_PATTERNS = {
    "params_b": re.compile(r"(\d+(?:\.\d+)?)\s*b\b(?!\w)", re.I),
    "data_gb": re.compile(r"(\d+(?:\.\d+)?)\s*(gb|tb)\b", re.I),
    "rows_m": re.compile(r"(\d+(?:\.\d+)?)\s*(k|m|million|billion)\s+rows", re.I),
    "epochs": re.compile(r"(\d+)\s*(epochs?|iterations?|paths|sims?|simulations)\b", re.I),
    "docs": re.compile(r"(\d+)\s*(papers|documents|docs|pages|sources)\b", re.I),
}
_UNIT_MULT = {"tb": 1000.0, "gb": 1.0, "k": 0.001, "m": 1.0, "million": 1.0, "billion": 1000.0}


def scale_cues(text: str) -> dict[str, float]:
    out = {}
    for name, pat in _SCALE_PATTERNS.items():
        m = pat.search(text)
        if not m:
            out[name] = 0.0
            continue
        val = float(m.group(1))
        if m.lastindex and m.lastindex >= 2:
            val *= _UNIT_MULT.get(m.group(2).lower(), 1.0)
        out[name] = math.log1p(val)
    return out


def featurize(text: str, category: str) -> np.ndarray:
    vec = np.zeros(HASH_DIM + len(_SCALE_PATTERNS) + len(CATEGORIES) + 1)
    for tok in re.findall(r"[a-z]+", text.lower()):
        vec[zlib.crc32(tok.encode()) % HASH_DIM] += 1.0
    norm = np.linalg.norm(vec[:HASH_DIM])
    if norm > 0:
        vec[:HASH_DIM] /= norm
    for k, v in enumerate(scale_cues(text).values()):
        vec[HASH_DIM + k] = v
    if category in CATEGORIES:
        vec[HASH_DIM + len(_SCALE_PATTERNS) + CATEGORIES.index(category)] = 1.0
    vec[-1] = 1.0  # bias
    return vec


@dataclass
class LogRidge:
    """Ridge regression on log(target) with a residual-based interval."""

    alpha: float = 1.0
    weights: np.ndarray | None = None
    sigma: float = 0.5  # residual std in log space
    n: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogRidge:
        ly = np.log(np.maximum(y, 1e-6))
        reg = self.alpha * np.eye(X.shape[1])
        reg[-1, -1] = 0.0  # don't shrink the bias
        self.weights = np.linalg.solve(X.T @ X + reg, X.T @ ly)
        resid = ly - X @ self.weights
        dof = max(1, len(y) - min(len(y) - 1, 8))
        self.sigma = float(max(0.05, math.sqrt(float(resid @ resid) / dof)))
        self.n = len(y)
        return self

    def predict(self, x: np.ndarray) -> float:
        assert self.weights is not None
        return float(math.exp(float(x @ self.weights)))

    def interval(self, x: np.ndarray, z: float = 1.645) -> tuple[float, float]:
        """Central interval (default 90%) under a log-normal predictive."""
        assert self.weights is not None
        mu = float(x @ self.weights)
        return math.exp(mu - z * self.sigma), math.exp(mu + z * self.sigma)
