"""
=============================================================
 COMPUTE BAZAAR 2.0 -- ResearchAgent (Week 3)
=============================================================
 Autonomous agent specialized in literature review, multi-step
 reasoning, web research, document synthesis, and reporting.

 Prefers API Budget + RAM + CPU resources.
=============================================================
"""

from typing import List

from compute_bazaar.agents.base import BaseAgent, AgentCapability


class ResearchAgent(BaseAgent):
    """
    AI agent specialized in research, literature search, and synthesis.

    Uses LLM reasoning to evaluate research tasks, estimate
    complexity, and determine resource requirements.

    Typical resource needs:
      - API Budget for external search/retrieval APIs
      - RAM for context buffering and document embedding storage
      - CPU for parsing and text processing
    """

    @property
    def agent_type(self) -> str:
        return "research"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability.RESEARCH

    @property
    def system_prompt(self) -> str:
        return (
            f"You are {self.name}, an expert autonomous research and synthesis agent. "
            "You specialize in multi-step reasoning, academic research, web searching, "
            "document summarization, competitive intelligence, and technical report drafting. "
            "You are participating in the Compute Bazaar auction platform. "
            "When analyzing tasks, consider: search breadth required, source reliability verification, "
            "synthesis depth, output format complexity, and deadline urgency. "
            "API Budget is crucial for external search & retrieval APIs. "
            "RAM is needed for holding large document contexts and embeddings. "
            "CPU is needed for text parsing and extraction. "
            "Always respond with valid JSON when asked for structured output."
        )

    @property
    def preferred_resources(self) -> List[str]:
        return ["API-Budget-100", "RAM-Block-32GB", "CPU-Cores-8"]
