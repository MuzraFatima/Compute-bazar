"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Fairness & Anti-Starvation Module (Week 6)
=============================================================
 Exports Jain's fairness index, starvation detection, and
 priority-aware anti-domination metrics.
=============================================================
"""

from compute_bazaar.fairness.metrics import (
    jains_fairness_index,
    calculate_starvation_score,
    FairnessTracker,
)

__all__ = [
    "jains_fairness_index",
    "calculate_starvation_score",
    "FairnessTracker",
]
