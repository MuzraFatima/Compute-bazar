"""
=============================================================
 COMPUTE BAZAAR 2.0 — LLM Module
=============================================================
 Provides Groq LLM integration for Compute Bazaar agents.
 Exports the main client and configuration.
=============================================================
"""

from compute_bazaar.llm.client import (
    GroqLLMClient,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    UsageTracker,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMAPIError,
)

__all__ = [
    "GroqLLMClient",
    "LLMConfig",
    "LLMResponse",
    "LLMUsage",
    "UsageTracker",
    "LLMError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMAPIError",
]

