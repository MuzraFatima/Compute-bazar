"""
=============================================================
 COMPUTE BAZAAR 2.0 -- CodingAgent (Week 3)
=============================================================
 Autonomous agent specialized in software engineering tasks:
   - Code generation and refactoring
   - Bug fixing and debugging
   - API development and integration
   - Build/deploy pipeline automation

 Prefers GPU + CPU resources for model inference and compilation.
=============================================================
"""

from typing import List

from compute_bazaar.agents.base import BaseAgent, AgentCapability


class CodingAgent(BaseAgent):
    """
    AI agent specialized in software engineering and code generation.

    Uses LLM reasoning to evaluate coding tasks, estimate
    complexity, and determine resource requirements.

    Typical resource needs:
      - GPU for code-generation model inference
      - CPU for compilation, linting, test execution
      - RAM for large codebases and dependency resolution
    """

    @property
    def agent_type(self) -> str:
        return "coding"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability.CODING

    @property
    def system_prompt(self) -> str:
        return (
            f"You are {self.name}, an expert autonomous software engineering agent. "
            "You specialize in code generation, refactoring, bug fixing, API development, "
            "and build/deploy automation. You are participating in the Compute Bazaar "
            "auction where you must bid for compute resources (GPUs, CPUs, RAM, API budgets) "
            "to complete your tasks. "
            "When analyzing tasks, consider: language/framework complexity, dependency count, "
            "test coverage requirements, compilation overhead, and deployment targets. "
            "GPU resources are valuable for running code-generation LLMs. "
            "CPU resources are essential for compilation, linting, and test execution. "
            "RAM is needed for large monorepos and dependency trees. "
            "Always respond with valid JSON when asked for structured output."
        )

    @property
    def preferred_resources(self) -> List[str]:
        return ["GPU-Slot-1", "CPU-Cores-8", "RAM-Block-32GB"]
