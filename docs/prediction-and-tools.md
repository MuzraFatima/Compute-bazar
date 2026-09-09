# Compute Bazaar 2.0 -- Prediction Layer & Agent Safe Toolset Guide

> Week 7 deliverable. Documents the ML resource prediction layer, safe agent toolset (`check_resources`, `check_budget`, `estimate_cost`, `execute_task`), `ToolRegistry`, and task execution feedback loops.

---

## 1. Overview

Compute Bazaar 2.0 Week 7 adds an **ML-based Resource Prediction Layer** (`compute_bazaar.prediction`) and a **Safe Toolset & ToolRegistry** (`compute_bazaar.tools`).

Agents use these capabilities to forecast compute capacity (GPU, CPU, RAM, API tokens), estimate execution duration and cost, query market availability, and safely execute awarded tasks with performance feedback loops into ML predictors and Agent Memory.

```
compute_bazaar/
  prediction/
    __init__.py      # Package exports (ResourcePrediction, ResourcePredictor)
    predictor.py     # ML forecasting model & execution recorder
  tools/
    __init__.py      # Package exports (check_resources, check_budget, estimate_cost, execute_task, ToolRegistry)
    agent_tools.py   # Safe agent tools implementation
```

---

## 2. ML Resource Prediction Layer (`ResourcePredictor`)

The `ResourcePredictor` forecasts compute requirements for natural language task requests:

| Predicted Metric | Type | Description |
|:--|:--|:--|
| `predicted_gpus` | `float` | Required GPU slots (0 to 4) |
| `predicted_cpu_cores` | `float` | Required CPU cores (1 to 32) |
| `predicted_ram_gb` | `float` | Required memory in GB (2 to 128) |
| `predicted_api_tokens` | `float` | Required LLM API tokens (0 to 1,000,000) |
| `predicted_execution_time_sec` | `float` | Expected runtime in seconds |
| `predicted_cost` | `float` | Estimated monetary cost ($) |
| `recommended_bundle` | `List[str]` | Recommended list of resource item names |
| `confidence_score` | `float` | Model prediction confidence score $[0.0, 1.0]$ |

### Training & Learning Loop
When tasks complete, execution metrics are logged via `predictor.record_execution(...)`. As historical data accumulates per task category (`llm_training`, `data_etl`, `research`, `quantitative_analysis`, `software_dev`), the predictor updates its statistical averages and boosts prediction confidence.

---

## 3. Safe Agent Toolset (`ToolRegistry`)

Autonomous AI agents can query and invoke four safe tools:

1. **`check_resources(marketplace=None)`**:
   Inspects current market availability, unallocated capacity, and dynamic prices.
2. **`check_budget(agent)`**:
   Queries agent's remaining budget, total spent, surplus, and starvation state.
3. **`estimate_cost(task_description, predictor=None)`**:
   Uses `ResourcePredictor` to forecast resource requirements, duration, and cost.
4. **`execute_task(agent, task_description, allocation_result=None, predictor=None)`**:
   Safely executes tasks using awarded resources, updates agent budget, and feeds performance metrics back into `ResourcePredictor` and `AgentMemory`.

---

## 4. Usage Example

```python
from compute_bazaar.agents import CodingAgent
from compute_bazaar.prediction import ResourcePredictor
from compute_bazaar.tools import ToolRegistry
from compute_bazaar.llm import GroqLLMClient, LLMConfig

# 1. Initialize Agent
llm = GroqLLMClient(config=LLMConfig(mock_mode=True))
agent = CodingAgent("AutoDevBot", budget=500.0, llm_client=llm)

# 2. Invoke Estimate Cost Tool
prediction = agent.invoke_tool("estimate_cost", task_description="Fine-tune Llama model on 50GB code corpus")
print(f"Predicted Cost: ${prediction.predicted_cost:.2f}")
print(f"Predicted Runtime: {prediction.predicted_execution_time_sec:.0f}s")
print(f"Recommended Bundle: {prediction.recommended_bundle}")

# 3. Create Bid & Execute Task Post-Auction
bid = agent.create_bid("Fine-tune Llama model on 50GB code corpus")
# (Submit to auction engine...)

exec_result = agent.execute_awarded_task(
    "Fine-tune Llama model on 50GB code corpus",
    allocation_result={"won": True, "payment": 45.0, "resources": bid.resource_bundle},
)
print(f"Execution Status: {exec_result['status']}")
```

---

## 5. Architectural Safeguards

- **100% Backward Compatible**: Existing auction solver, VCG pricing, marketplace dynamic pricing, and fairness tracking remain fully functional.
- **Strict Budget & Safety Boundaries**: Tools execute safely in-memory without executing arbitrary system commands or risking resource exhaustion.
- **API Key Security**: Tests run strictly in `MOCK` mode using `LLMConfig(mock_mode=True)`. API keys are never exposed.
