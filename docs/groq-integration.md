# Compute Bazaar 2.0 -- Groq LLM Integration Guide

> Week 2 deliverable. Documents the LLM module architecture, usage patterns,
> and configuration for integrating Groq-powered intelligence into Compute Bazaar.

---

## 1. Overview

The `compute_bazaar.llm` module provides a clean wrapper around the
[Groq API](https://console.groq.com/) for adding LLM-powered intelligence
to Compute Bazaar agents. It is designed to be:

- **Isolated** -- does NOT touch the auction engine, VCG pricing, or scheduling
- **Testable** -- full mock mode for running tests without an API key
- **Observable** -- tracks token usage and latency per request
- **Configurable** -- model, temperature, and system prompts are all adjustable

```
compute_bazaar/
  llm/
    __init__.py      # Package exports
    client.py        # GroqLLMClient, LLMConfig, LLMResponse, errors
```

---

## 2. Setup

### 2.1 Install Dependencies

```bash
pip install groq python-dotenv
# or
pip install -r backend/requirements.txt
```

### 2.2 Configure API Key

1. Get a free API key at [https://console.groq.com/keys](https://console.groq.com/keys)
2. Add it to your `.env` file (never commit this file):

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

The `.env` file is already in `.gitignore`. The `.env.example` file contains
placeholder values only.

### 2.3 Verify Installation

```python
from compute_bazaar.llm import GroqLLMClient, LLMConfig

# Mock mode (no API key needed)
client = GroqLLMClient(config=LLMConfig(mock_mode=True))
response = client.complete("Hello!")
print(response.content)      # Mock response
print(response.usage.to_dict())  # Token counts

# Live mode (requires GROQ_API_KEY in .env)
client = GroqLLMClient()
response = client.complete("What is a combinatorial auction?")
print(response.content)
```

---

## 3. API Reference

### 3.1 LLMConfig

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `model` | str | `llama-3.3-70b-versatile` | Groq model to use |
| `temperature` | float | 0.7 | Sampling temperature (0.0 = deterministic) |
| `max_tokens` | int | 1024 | Maximum response tokens |
| `top_p` | float | 1.0 | Nucleus sampling |
| `timeout_seconds` | float | 30.0 | HTTP timeout |
| `max_retries` | int | 2 | Automatic retry count |
| `mock_mode` | bool | False | If True, returns deterministic mock responses |
| `system_prompt` | str | (Compute Bazaar context) | System prompt for all calls |

### 3.2 GroqLLMClient

```python
client = GroqLLMClient(
    config=LLMConfig(...),   # optional, uses defaults
    api_key="gsk_...",       # optional, overrides env var
)
```

#### `complete(prompt, **kwargs) -> LLMResponse`

Send a single completion request.

| Parameter | Type | Default | Description |
|:--|:--|:--|:--|
| `prompt` | str | (required) | User message |
| `system_prompt` | str | None | Override system prompt for this call |
| `json_mode` | bool | False | Request JSON-formatted output |
| `temperature` | float | None | Override temperature for this call |
| `max_tokens` | int | None | Override max tokens for this call |
| `model` | str | None | Override model for this call |

#### `complete_json(prompt, **kwargs) -> dict`

Convenience method: calls `complete()` with `json_mode=True` and returns the
parsed JSON dict directly. Raises `LLMError` if parsing fails.

#### `get_usage() -> dict`

Returns cumulative usage statistics:

```python
{
    "total_calls": 5,
    "total_prompt_tokens": 250,
    "total_completion_tokens": 400,
    "total_tokens": 650,
    "total_latency_ms": 1200.5,
    "avg_latency_ms": 240.1,
    "avg_tokens_per_call": 130.0,
    "errors": 0,
}
```

#### `reset_usage()`

Resets the usage tracker to zero.

### 3.3 LLMResponse

| Field | Type | Description |
|:--|:--|:--|
| `content` | str | The raw text response |
| `model` | str | Model that generated the response |
| `usage` | LLMUsage | Token counts (prompt, completion, total) |
| `latency_ms` | float | Request latency in milliseconds |
| `is_mock` | bool | Whether this was a mock response |
| `raw_response` | Any | Original Groq API response object |

Methods:
- `to_json() -> dict | None` -- Parse content as JSON (strips code fences)
- `to_dict() -> dict` -- Serializable representation

### 3.4 Exceptions

| Exception | When |
|:--|:--|
| `LLMError` | Base class for all LLM errors |
| `LLMAuthenticationError` | API key missing or invalid |
| `LLMRateLimitError` | Groq rate limit exceeded (429) |
| `LLMAPIError` | Other API errors |

---

## 4. Supported Models

| Model | Speed | Context | Best For |
|:--|:--|:--|:--|
| `llama-3.3-70b-versatile` | Fast | 128K | General reasoning (default) |
| `llama-3.1-8b-instant` | Very fast | 128K | Quick responses, low latency |
| `llama-3.2-1b-preview` | Ultra fast | 128K | Simple tasks, cost-effective |
| `llama-3.2-3b-preview` | Very fast | 128K | Balanced speed/quality |
| `mixtral-8x7b-32768` | Fast | 32K | Complex reasoning |
| `gemma2-9b-it` | Fast | 8K | Instruction following |

All models are available on the Groq free tier. The default
(`llama-3.3-70b-versatile`) offers the best balance of quality and speed.

---

## 5. Usage Patterns

### 5.1 Basic Text Completion

```python
from compute_bazaar.llm import GroqLLMClient

client = GroqLLMClient()
response = client.complete(
    "Explain VCG pricing in 2 sentences."
)
print(response.content)
print(f"Latency: {response.latency_ms:.0f}ms")
print(f"Tokens: {response.usage.total_tokens}")
```

### 5.2 Structured JSON Response

```python
result = client.complete_json(
    "Analyze this bid: Agent-Alpha wants GPU-1 + RAM-32GB for $120. "
    "Other bids: Agent-Beta wants GPU-1 for $80, Agent-Gamma wants RAM-32GB for $60. "
    "Return JSON with fields: recommended_action, reasoning, confidence."
)
print(result["recommended_action"])
print(result["confidence"])
```

### 5.3 Custom System Prompt

```python
response = client.complete(
    prompt="Should I bid higher for GPU resources?",
    system_prompt=(
        "You are a bidding advisor for a compute auction. "
        "The current market clearing price for GPUs is $75/slot. "
        "Advise conservatively."
    ),
    temperature=0.3,
)
```

### 5.4 Testing with Mock Mode

```python
from compute_bazaar.llm import GroqLLMClient, LLMConfig

# No API key needed
client = GroqLLMClient(config=LLMConfig(mock_mode=True))
response = client.complete("anything")
assert response.is_mock is True

# JSON mode also works in mock
data = client.complete_json("analyze this")
assert "analysis" in data
```

### 5.5 Usage Monitoring

```python
client = GroqLLMClient()

# Make several calls...
client.complete("first query")
client.complete("second query")

# Check cumulative usage
usage = client.get_usage()
print(f"Total calls: {usage['total_calls']}")
print(f"Total tokens: {usage['total_tokens']}")
print(f"Avg latency: {usage['avg_latency_ms']:.0f}ms")

# Reset for next session
client.reset_usage()
```

---

## 6. Architecture Decisions

| Decision | Choice | Rationale |
|:--|:--|:--|
| SDK | Official `groq` Python package | Maintained by Groq, handles auth/retry |
| Isolation | Separate `llm/` module | Does not touch auction/VCG/scheduler code |
| Testing | Mock mode built into client | Tests run without API key; deterministic |
| Config | `LLMConfig` dataclass | Composable, overridable per-call |
| Tracking | `UsageTracker` accumulator | Monitor costs/latency without external tools |
| Errors | Typed exception hierarchy | Callers can catch specific failure modes |
| JSON parsing | Auto-strip code fences | Models sometimes wrap JSON in markdown |

---

## 7. What This Module Does NOT Do

- Does NOT modify the auction engine or VCG pricing
- Does NOT implement autonomous agent decision-making (that's Week 3+)
- Does NOT expose the API key in any response or log
- Does NOT make network calls in mock mode
- Does NOT add any API endpoints to FastAPI

This module is the **foundation layer** that future autonomous agents will
build upon. It provides the LLM communication channel only.

---

## 8. Test Coverage

```
$ pytest tests/test_week2_llm.py -v
```

| Test Group | Tests | Validates |
|:--|:--|:--|
| Configuration | 3 | Default values, custom overrides, model list |
| Client Init | 3 | Mock init, repr, missing API key error |
| Mock Completion | 4 | Response structure, prompt echo, latency, usage |
| JSON Mode | 4 | JSON parsing, code fence stripping, error cases |
| Usage Tracking | 4 | Call counting, reset, empty tracker, error recording |
| LLMResponse | 2 | to_dict serialization, LLMUsage serialization |
| Error Classes | 2 | Inheritance hierarchy, message preservation |
| Mock Generator | 2 | Text mode, JSON mode deterministic output |
| Parameter Overrides | 2 | Per-call overrides, custom config client |

All tests use mock mode -- no API key required.
