"""
=============================================================
 COMPUTE BAZAAR 2.0 — Week 2 Tests: LLM Module
=============================================================
 Tests the Groq LLM client in mock mode (no API key required).
 Validates:
   - Client initialization (mock + error cases)
   - Configuration defaults and overrides
   - Mock completion responses
   - JSON mode and structured parsing
   - Token/latency usage tracking
   - Error classification
   - Usage tracker reset

 Run with:  python -m pytest tests/test_week2_llm.py -v
=============================================================
"""

import json
import pytest
from compute_bazaar.llm.client import (
    GroqLLMClient,
    LLMConfig,
    LLMResponse,
    LLMUsage,
    LLMError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMAPIError,
    UsageTracker,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    _generate_mock_response,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_client():
    """Create a GroqLLMClient in mock mode (no API key needed)."""
    config = LLMConfig(mock_mode=True)
    return GroqLLMClient(config=config)


@pytest.fixture
def custom_config():
    """Create a non-default LLM config."""
    return LLMConfig(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=512,
        top_p=0.9,
        mock_mode=True,
        system_prompt="You are a test assistant.",
    )


# ─────────────────────────────────────────────────────────────
# GROUP 1: Configuration
# ─────────────────────────────────────────────────────────────

def test_default_config():
    """Default LLMConfig should have sensible defaults."""
    config = LLMConfig()
    assert config.model == DEFAULT_MODEL
    assert config.temperature == 0.7
    assert config.max_tokens == 1024
    assert config.top_p == 1.0
    assert config.timeout_seconds == 30.0
    assert config.max_retries == 2
    assert config.mock_mode is False
    assert "Compute Bazaar" in config.system_prompt


def test_custom_config(custom_config):
    """Custom config values should be preserved."""
    assert custom_config.model == "llama-3.1-8b-instant"
    assert custom_config.temperature == 0.3
    assert custom_config.max_tokens == 512
    assert custom_config.top_p == 0.9
    assert custom_config.mock_mode is True
    assert custom_config.system_prompt == "You are a test assistant."


def test_supported_models_list():
    """SUPPORTED_MODELS should include common Groq models."""
    assert DEFAULT_MODEL in SUPPORTED_MODELS
    assert len(SUPPORTED_MODELS) >= 3


# ─────────────────────────────────────────────────────────────
# GROUP 2: Client Initialization
# ─────────────────────────────────────────────────────────────

def test_mock_client_init(mock_client):
    """Mock client should initialize without an API key."""
    assert mock_client.config.mock_mode is True
    assert mock_client._client is None
    assert mock_client.tracker.total_calls == 0


def test_mock_client_repr(mock_client):
    """Client repr should show MOCK mode."""
    r = repr(mock_client)
    assert "MOCK" in r
    assert DEFAULT_MODEL in r


def test_live_client_no_key_raises():
    """Creating a live client without API key should raise LLMAuthenticationError."""
    # Temporarily unset the env var
    import os
    original = os.environ.pop("GROQ_API_KEY", None)
    try:
        with pytest.raises(LLMAuthenticationError, match="No Groq API key found"):
            GroqLLMClient(config=LLMConfig(mock_mode=False))
    finally:
        if original is not None:
            os.environ["GROQ_API_KEY"] = original


# ─────────────────────────────────────────────────────────────
# GROUP 3: Mock Completion
# ─────────────────────────────────────────────────────────────

def test_mock_complete_returns_response(mock_client):
    """Mock complete() should return a valid LLMResponse."""
    response = mock_client.complete("What is 2+2?")
    assert isinstance(response, LLMResponse)
    assert response.is_mock is True
    assert len(response.content) > 0
    assert "mock" in response.model.lower()


def test_mock_complete_contains_prompt_echo(mock_client):
    """Mock response should echo part of the prompt."""
    prompt = "Analyze the GPU allocation for Agent-Alpha"
    response = mock_client.complete(prompt)
    assert "Analyze the GPU allocation" in response.content


def test_mock_complete_tracks_latency(mock_client):
    """Mock responses should have minimal latency."""
    response = mock_client.complete("test")
    assert response.latency_ms >= 0
    assert response.latency_ms < 100  # mock should be near-instant


def test_mock_complete_tracks_usage(mock_client):
    """Mock responses should include token usage estimates."""
    response = mock_client.complete("Hello world test prompt")
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == (
        response.usage.prompt_tokens + response.usage.completion_tokens
    )


# ─────────────────────────────────────────────────────────────
# GROUP 4: JSON Mode
# ─────────────────────────────────────────────────────────────

def test_mock_json_mode(mock_client):
    """JSON mode should return parseable JSON content."""
    response = mock_client.complete("Analyze this bid", json_mode=True)
    parsed = response.to_json()
    assert parsed is not None
    assert isinstance(parsed, dict)
    assert "analysis" in parsed
    assert "confidence" in parsed


def test_mock_complete_json_method(mock_client):
    """complete_json() should directly return a parsed dict."""
    result = mock_client.complete_json("Analyze this resource bundle")
    assert isinstance(result, dict)
    assert "analysis" in result
    assert "recommendation" in result
    assert result["confidence"] == 0.85


def test_json_parsing_with_code_fences():
    """to_json() should strip markdown code fences."""
    response = LLMResponse(
        content='```json\n{"key": "value"}\n```',
        model="test",
        usage=LLMUsage(),
        latency_ms=0,
    )
    parsed = response.to_json()
    assert parsed == {"key": "value"}


def test_json_parsing_invalid_json():
    """to_json() should return None for invalid JSON."""
    response = LLMResponse(
        content="This is not JSON at all",
        model="test",
        usage=LLMUsage(),
        latency_ms=0,
    )
    assert response.to_json() is None


# ─────────────────────────────────────────────────────────────
# GROUP 5: Usage Tracking
# ─────────────────────────────────────────────────────────────

def test_usage_tracker_records_calls(mock_client):
    """Usage tracker should count calls and tokens."""
    mock_client.complete("first call")
    mock_client.complete("second call")
    mock_client.complete("third call")

    usage = mock_client.get_usage()
    assert usage["total_calls"] == 3
    assert usage["total_tokens"] > 0
    assert usage["total_latency_ms"] > 0
    assert usage["avg_latency_ms"] > 0
    assert usage["avg_tokens_per_call"] > 0
    assert usage["errors"] == 0


def test_usage_tracker_reset(mock_client):
    """reset_usage() should zero out all counters."""
    mock_client.complete("a call")
    assert mock_client.get_usage()["total_calls"] == 1

    mock_client.reset_usage()
    usage = mock_client.get_usage()
    assert usage["total_calls"] == 0
    assert usage["total_tokens"] == 0
    assert usage["errors"] == 0


def test_usage_tracker_empty():
    """Empty tracker should return zeros and not divide by zero."""
    tracker = UsageTracker()
    assert tracker.avg_latency_ms == 0.0
    assert tracker.avg_tokens_per_call == 0.0
    assert tracker.total_calls == 0


def test_usage_tracker_record_error():
    """Error recording should increment error count."""
    tracker = UsageTracker()
    tracker.record_error()
    tracker.record_error()
    assert tracker.errors == 2
    d = tracker.to_dict()
    assert d["errors"] == 2


# ─────────────────────────────────────────────────────────────
# GROUP 6: LLMResponse
# ─────────────────────────────────────────────────────────────

def test_llm_response_to_dict(mock_client):
    """LLMResponse.to_dict() should return serializable data."""
    response = mock_client.complete("test")
    d = response.to_dict()
    assert "content" in d
    assert "model" in d
    assert "usage" in d
    assert "latency_ms" in d
    assert "is_mock" in d
    assert d["is_mock"] is True
    # Should be JSON-serializable
    json.dumps(d)


def test_llm_usage_to_dict():
    """LLMUsage.to_dict() should be serializable."""
    usage = LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    d = usage.to_dict()
    assert d["prompt_tokens"] == 10
    assert d["completion_tokens"] == 20
    assert d["total_tokens"] == 30


# ─────────────────────────────────────────────────────────────
# GROUP 7: Error Classes
# ─────────────────────────────────────────────────────────────

def test_error_hierarchy():
    """All error classes should inherit from LLMError."""
    assert issubclass(LLMAuthenticationError, LLMError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMAPIError, LLMError)


def test_error_messages():
    """Error classes should preserve messages."""
    err = LLMAuthenticationError("bad key")
    assert str(err) == "bad key"

    err2 = LLMRateLimitError("too many requests")
    assert "too many" in str(err2)


# ─────────────────────────────────────────────────────────────
# GROUP 8: Mock Response Generator
# ─────────────────────────────────────────────────────────────

def test_mock_generator_text_mode():
    """Text mode mock should include prompt echo."""
    resp = _generate_mock_response("hello world", "test-model", json_mode=False)
    assert "mock" in resp.content.lower()
    assert resp.is_mock is True
    assert resp.model == "mock-test-model"


def test_mock_generator_json_mode():
    """JSON mode mock should return valid JSON."""
    resp = _generate_mock_response("analyze bid", "test-model", json_mode=True)
    parsed = json.loads(resp.content)
    assert "analysis" in parsed
    assert isinstance(parsed["reasoning"], list)


# ─────────────────────────────────────────────────────────────
# GROUP 9: Parameter Overrides
# ─────────────────────────────────────────────────────────────

def test_complete_with_overrides(mock_client):
    """complete() should accept per-call parameter overrides."""
    response = mock_client.complete(
        prompt="test",
        temperature=0.1,
        max_tokens=256,
        model="llama-3.1-8b-instant",
        system_prompt="Custom system prompt for this call only.",
    )
    assert isinstance(response, LLMResponse)
    # Mock mode uses the resolved model in the mock model name
    assert "llama-3.1-8b-instant" in response.model


def test_custom_config_client():
    """Client with custom config should use custom model."""
    config = LLMConfig(
        model="mixtral-8x7b-32768",
        temperature=0.0,
        max_tokens=100,
        mock_mode=True,
    )
    client = GroqLLMClient(config=config)
    assert client.config.model == "mixtral-8x7b-32768"
    assert client.config.temperature == 0.0

    response = client.complete("test")
    assert "mixtral" in response.model.lower()
