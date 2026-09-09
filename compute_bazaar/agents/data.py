"""
=============================================================
 COMPUTE BAZAAR 2.0 -- DataAgent (Week 3)
=============================================================
 Autonomous agent specialized in data engineering, ETL pipelines,
 dataset processing, feature extraction, and matrix operations.

 Prefers RAM + CPU + GPU resources.
=============================================================
"""

from typing import List

from compute_bazaar.agents.base import BaseAgent, AgentCapability


class DataAgent(BaseAgent):
    """
    AI agent specialized in data engineering, ETL, and dataset processing.

    Uses LLM reasoning to evaluate data processing tasks, estimate
    complexity, and determine resource requirements.

    Typical resource needs:
      - RAM for in-memory data tables (Pandas/Polars/Arrow)
      - CPU for multi-core data transformations and serialization
      - GPU for accelerated tensor ops or vector embeddings
    """

    @property
    def agent_type(self) -> str:
        return "data_processing"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability.DATA_PROCESSING

    @property
    def system_prompt(self) -> str:
        return (
            f"You are {self.name}, an expert autonomous data processing agent. "
            "You specialize in ETL pipelines, dataset cleaning, feature engineering, "
            "vector embedding generation, matrix ops, and batch transformations. "
            "You are participating in the Compute Bazaar auction platform. "
            "When analyzing tasks, consider: dataset volume, schema heterogeneity, "
            "transformation complexity, memory footprint, and parallelization potential. "
            "RAM is your primary requirement for in-memory DataFrame manipulation. "
            "CPU cores enable high-throughput parallel batch processing. "
            "GPUs are essential for hardware-accelerated tensor computations. "
            "Always respond with valid JSON when asked for structured output."
        )

    @property
    def preferred_resources(self) -> List[str]:
        return ["RAM-Block-32GB", "CPU-Cores-8", "GPU-Slot-1"]
