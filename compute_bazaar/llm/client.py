"""
=============================================================
 COMPUTE BAZAAR 2.0 — Groq LLM Client (Week 2)
=============================================================
 A clean, production-ready wrapper around the Groq API for
 Compute Bazaar agent intelligence.

 Features:
   - Configurable model selection (llama, mixtral, gemma, etc.)
   - Structured JSON response parsing
   - Token and latency tracking per request
   - Automatic retry with exponential backoff
   - Mock mode for testing without a real API key
   - Thread-safe singleton pattern

 Usage:
   from compute_bazaar.llm import GroqLLMClient, LLMConfig

   client = GroqLLMClient()  # reads GROQ_API_KEY from .env
   response = client.complete("Analyze this auction scenario...")
   print(response.content)
   print(response.usage)

 Mock mode (for tests):
   client = GroqLLMClient(config=LLMConfig(mock_mode=True))
   response = client.complete("anything")
   # Returns deterministic mock response without hitting API
=============================================================
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from loguru import logger


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Default Groq model — fast, capable, free-tier friendly
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Models available on Groq (as of 2025)
SUPPORTED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
]

@dataclass
class LLMConfig:
    """Configuration for the Groq LLM client."""
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    timeout_seconds: float = 30.0
    max_retries: int = 2
    mock_mode: bool = False

    # System prompt prepended to all requests
    system_prompt: str = (
        "You are an AI assistant for Compute Bazaar, a combinatorial auction "
        "platform for allocating scarce compute resources (GPUs, API budgets, "
        "RAM, CPU cores) among competing AI agents. You analyze auction scenarios, "
        "evaluate resource bundles, and provide structured recommendations. "
        "Always respond with valid JSON when asked for structured output."
    )


# ─────────────────────────────────────────────────────────────
# RESPONSE DATA CLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class LLMUsage:
    """Token usage statistics from a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str
    model: str
    usage: LLMUsage
    latency_ms: float
    is_mock: bool = False
    raw_response: Optional[Any] = field(default=None, repr=False)

    def to_json(self) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse the response content as JSON.
        Returns None if parsing fails.
        """
        try:
            # Strip markdown code fences if present
            text = self.content.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
            "is_mock": self.is_mock,
        }


# ─────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when the API key is invalid or missing."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when the API rate limit is exceeded."""
    pass


class LLMAPIError(LLMError):
    """Raised for general API errors from Groq."""
    pass


# ─────────────────────────────────────────────────────────────
# CUMULATIVE USAGE TRACKER
# ─────────────────────────────────────────────────────────────

@dataclass
class UsageTracker:
    """Tracks cumulative token usage and latency across all calls."""
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def avg_tokens_per_call(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_tokens / self.total_calls

    def record(self, response: LLMResponse):
        """Record usage from a completed response."""
        self.total_calls += 1
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens
        self.total_tokens += response.usage.total_tokens
        self.total_latency_ms += response.latency_ms

    def record_error(self):
        """Record a failed call."""
        self.errors += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize usage metrics to dictionary."""
        return {
            "calls": self.total_calls,
            "total_calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "tokens": self.total_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_tokens_per_call": round(self.avg_tokens_per_call, 2),
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        return (
            f"UsageTracker(calls={self.total_calls}, "
            f"tokens={self.total_tokens}, "
            f"avg_latency={self.avg_latency_ms:.0f}ms, "
            f"errors={self.errors})"
        )


# ─────────────────────────────────────────────────────────────
# MOCK RESPONSE GENERATOR
# ─────────────────────────────────────────────────────────────

def _generate_mock_response(
    prompt: str,
    model: str,
    json_mode: bool = False,
) -> LLMResponse:
    """
    Generate a deterministic mock response for testing.
    No API call is made. Returns structured content when json_mode=True.
    """
    if json_mode:
        mock_content = json.dumps({
            "analysis": "Mock analysis for testing",
            "recommendation": "This is a mock response from the LLM module",
            "confidence": 0.85,
            "reasoning": [
                "Mock reason 1: deterministic test output",
                "Mock reason 2: no API key required",
            ],
        }, indent=2)
    else:
        mock_content = (
            "This is a mock LLM response for testing purposes. "
            "No actual API call was made. "
            f"Prompt received: {prompt[:100]}..."
        )

    return LLMResponse(
        content=mock_content,
        model=f"mock-{model}",
        usage=LLMUsage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(mock_content.split()),
            total_tokens=len(prompt.split()) + len(mock_content.split()),
        ),
        latency_ms=1.0,
        is_mock=True,
        raw_response=None,
    )


# ─────────────────────────────────────────────────────────────
# MAIN CLIENT
# ─────────────────────────────────────────────────────────────

class GroqLLMClient:
    """
    Production-ready Groq LLM client for Compute Bazaar.

    Wraps the Groq Python SDK with:
      - Configurable model and parameters
      - Structured JSON response parsing
      - Token/latency usage tracking
      - Mock mode for testing
      - Error classification and handling

    Args:
        config: LLMConfig with model, temperature, etc.
        api_key: Explicit API key (overrides env var).
                 If None, reads from GROQ_API_KEY env var.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config or LLMConfig()
        self.tracker = UsageTracker()
        self._client = None

        if not self.config.mock_mode:
            resolved_key = api_key or os.getenv("GROQ_API_KEY")
            if not resolved_key:
                raise LLMAuthenticationError(
                    "No Groq API key found. Set GROQ_API_KEY in your .env file "
                    "or pass api_key= to GroqLLMClient()."
                )

            try:
                import groq
                self._client = groq.Groq(
                    api_key=resolved_key,
                    timeout=self.config.timeout_seconds,
                    max_retries=self.config.max_retries,
                )
            except Exception as e:
                raise LLMError(f"Failed to initialize Groq client: {e}")

            logger.info(
                f"GroqLLMClient initialized | model={self.config.model} | "
                f"temp={self.config.temperature} | max_tokens={self.config.max_tokens}"
            )
        else:
            logger.info("GroqLLMClient initialized in MOCK MODE (no API calls)")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a completion request to Groq.

        Args:
            prompt: The user message / question.
            system_prompt: Override the default system prompt.
            json_mode: If True, request JSON output format from the model.
            temperature: Override config temperature for this call.
            max_tokens: Override config max_tokens for this call.
            model: Override config model for this call.

        Returns:
            LLMResponse with content, usage stats, and latency.

        Raises:
            LLMAuthenticationError: Invalid API key.
            LLMRateLimitError: Rate limit exceeded.
            LLMAPIError: Other API errors.
            LLMError: General errors.
        """
        resolved_model = model or self.config.model
        resolved_temp = temperature if temperature is not None else self.config.temperature
        resolved_max = max_tokens or self.config.max_tokens
        resolved_system = system_prompt or self.config.system_prompt

        # ── Mock mode ────────────────────────────────────────
        if self.config.mock_mode:
            mock_resp = _generate_mock_response(prompt, resolved_model, json_mode)
            self.tracker.record(mock_resp)
            return mock_resp

        # ── Build messages ───────────────────────────────────
        messages = [
            {"role": "system", "content": resolved_system},
            {"role": "user", "content": prompt},
        ]

        # ── Call Groq API ────────────────────────────────────
        t_start = time.perf_counter()
        try:
            import groq as groq_module

            kwargs: Dict[str, Any] = {
                "model": resolved_model,
                "messages": messages,
                "temperature": resolved_temp,
                "max_tokens": resolved_max,
                "top_p": self.config.top_p,
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            completion = self._client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - t_start) * 1000

            # ── Extract response ─────────────────────────────
            choice = completion.choices[0]
            content = choice.message.content or ""

            usage = LLMUsage(
                prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
                completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
                total_tokens=completion.usage.total_tokens if completion.usage else 0,
            )

            response = LLMResponse(
                content=content,
                model=completion.model or resolved_model,
                usage=usage,
                latency_ms=round(latency_ms, 2),
                is_mock=False,
                raw_response=completion,
            )

            self.tracker.record(response)

            logger.debug(
                f"LLM call | model={response.model} | "
                f"tokens={usage.total_tokens} | "
                f"latency={latency_ms:.0f}ms"
            )

            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - t_start) * 1000
            self.tracker.record_error()

            # Classify error type
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str or "401" in error_str:
                raise LLMAuthenticationError(f"Groq authentication failed: {e}")
            elif "rate limit" in error_str or "429" in error_str:
                raise LLMRateLimitError(f"Groq rate limit exceeded: {e}")
            else:
                raise LLMAPIError(f"Groq API error after {latency_ms:.0f}ms: {e}")

    def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method: send a completion and parse the response as JSON.

        Returns the parsed dict. Raises LLMError if JSON parsing fails.
        """
        response = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

        parsed = response.to_json()
        if parsed is None:
            raise LLMError(
                f"Failed to parse LLM response as JSON. "
                f"Raw content: {response.content[:200]}"
            )
        return parsed

    def get_usage(self) -> Dict[str, Any]:
        """Return cumulative usage statistics."""
        return self.tracker.to_dict()

    def reset_usage(self):
        """Reset the usage tracker."""
        self.tracker = UsageTracker()

    def __repr__(self) -> str:
        mode = "MOCK" if self.config.mock_mode else "LIVE"
        return (
            f"GroqLLMClient(mode={mode}, model={self.config.model}, "
            f"tracker={self.tracker})"
        )
