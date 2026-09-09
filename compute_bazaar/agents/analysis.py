"""
=============================================================
 COMPUTE BAZAAR 2.0 -- AnalysisAgent (Week 3)
=============================================================
 Autonomous agent specialized in financial modeling, economic
 simulation, risk evaluation, statistical inference, and decision support.

 Prefers CPU + RAM + API Budget resources.
=============================================================
"""

from typing import List

from compute_bazaar.agents.base import BaseAgent, AgentCapability


class AnalysisAgent(BaseAgent):
    """
    AI agent specialized in quantitative analysis, modeling, and evaluation.

    Uses LLM reasoning to evaluate analytical tasks, estimate
    complexity, and determine resource requirements.

    Typical resource needs:
      - CPU for numerical simulations (Monte Carlo, statistical modeling)
      - RAM for statistical state management
      - API Budget for market data feeds or pricing APIs
    """

    @property
    def agent_type(self) -> str:
        return "analysis"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability.ANALYSIS

    @property
    def system_prompt(self) -> str:
        return (
            f"You are {self.name}, an expert autonomous quantitative analysis agent. "
            "You specialize in financial modeling, market evaluation, risk metrics, "
            "statistical inference, scenario simulation, and strategic decision support. "
            "You are participating in the Compute Bazaar auction platform. "
            "When analyzing tasks, consider: mathematical model complexity, sample size, "
            "confidence interval precision, risk factor dimensionality, and report fidelity. "
            "CPU resources power compute-heavy Monte Carlo and optimization routines. "
            "RAM supports holding large sample distribution matrices in memory. "
            "API Budget allows pulling external market and benchmark feeds. "
            "Always respond with valid JSON when asked for structured output."
        )

    @property
    def preferred_resources(self) -> List[str]:
        return ["CPU-Cores-8", "RAM-Block-32GB", "API-Budget-100"]
