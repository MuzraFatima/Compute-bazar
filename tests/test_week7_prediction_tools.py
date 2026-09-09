"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Week 7 Tests: Prediction & Safe Tools
=============================================================
 Tests the ML resource prediction layer, safe agent toolset, and task execution:
   - ResourcePredictor forecasting of GPUs, CPUs, RAM, API tokens, duration, and cost
   - Learning from historical execution records
   - Safe tools: check_resources, check_budget, estimate_cost, execute_task
   - ToolRegistry discovery and dynamic invocation
   - Integration with BaseAgent and end-to-end auction task execution flow

 All tests run in MOCK mode (no Groq API key required).

 Run with:  python -m pytest tests/test_week7_prediction_tools.py -v
=============================================================
"""

import pytest
from typing import List, Dict

from compute_bazaar.prediction import ResourcePredictor, ResourcePrediction
from compute_bazaar.tools import (
    check_resources,
    check_budget,
    estimate_cost,
    execute_task,
    ToolRegistry,
)
from compute_bazaar.agents import CodingAgent, ResearchAgent, DataAgent
from compute_bazaar.llm import GroqLLMClient, LLMConfig
from compute_bazaar.marketplace import DynamicMarketplace
from backend.app.auction.engine import (
    AuctionEngine,
    Resource as EngineResource,
    SolverMethod,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    """Mock Groq client for agent testing."""
    return GroqLLMClient(config=LLMConfig(mock_mode=True))


@pytest.fixture
def predictor():
    """Fresh ResourcePredictor instance."""
    return ResourcePredictor()


@pytest.fixture
def tool_registry():
    """Fresh ToolRegistry instance."""
    return ToolRegistry()


@pytest.fixture
def coding_agent(mock_llm_client):
    return CodingAgent("DevBot", budget=500.0, llm_client=mock_llm_client)


# ─────────────────────────────────────────────────────────────
# GROUP 1: ResourcePredictor Forecasting & Learning
# ─────────────────────────────────────────────────────────────

def test_resource_predictor_cold_start(predictor):
    """ResourcePredictor should generate valid predictions even with zero history (cold start)."""
    prediction = predictor.predict("LoRA fine-tuning on 70B model")

    assert isinstance(prediction, ResourcePrediction)
    assert prediction.task_category == "llm_training"
    assert prediction.predicted_gpus >= 1.0
    assert prediction.predicted_ram_gb >= 16.0
    assert prediction.predicted_execution_time_sec > 0.0
    assert prediction.predicted_cost > 0.0
    assert len(prediction.recommended_bundle) >= 1
    assert 0.0 <= prediction.confidence_score <= 1.0


def test_resource_predictor_learning_from_history(predictor):
    """Predictor should update forecasts and confidence as historical records are logged."""
    task = "Document corpus embedding batch job"
    
    # Record 4 historical executions
    for i in range(4):
        predictor.record_execution(
            task_description=task,
            allocated_resources=["RAM-Block-32GB", "CPU-Cores-8"],
            execution_time_sec=180.0,
            actual_cost=15.0,
            tokens_used=12000.0,
            task_category="data_etl",
        )

    prediction = predictor.predict(task, task_category="data_etl")

    assert prediction.predicted_execution_time_sec == pytest.approx(180.0)
    assert prediction.predicted_cost == pytest.approx(15.0)
    assert prediction.confidence_score > 0.80


def test_model_summary(predictor):
    """get_model_summary() should report historical record statistics."""
    predictor.record_execution("Train model", ["GPU-Slot-1"], 600.0, 50.0, task_category="llm_training")
    summary = predictor.get_model_summary()

    assert summary["total_execution_records"] == 1
    assert "llm_training" in summary["tracked_categories"]


# ─────────────────────────────────────────────────────────────
# GROUP 2: Safe Toolset Functions
# ─────────────────────────────────────────────────────────────

def test_tool_check_resources():
    """check_resources tool should return market inventory snapshot."""
    marketplace = DynamicMarketplace("Test Market")
    res_data = check_resources(marketplace)

    assert "supply" in res_data
    assert "resource_prices" in res_data
    assert res_data["total_resources"] >= 5


def test_tool_check_budget(coding_agent):
    """check_budget tool should return agent financial and memory status."""
    b_data = check_budget(coding_agent)

    assert b_data["agent_name"] == "DevBot"
    assert b_data["current_budget"] == 500.0
    assert b_data["is_starved"] is False


def test_tool_estimate_cost(predictor):
    """estimate_cost tool should return a ResourcePrediction."""
    pred = estimate_cost("Run Monte Carlo risk modeling", predictor=predictor)

    assert isinstance(pred, ResourcePrediction)
    assert pred.predicted_cost > 0.0


def test_tool_execute_task(coding_agent, predictor):
    """execute_task tool should simulate execution, update budget, and record performance."""
    result = execute_task(
        agent=coding_agent,
        task_description="Build REST API endpoint",
        allocation_result={"won": True, "payment": 25.0, "resources": ["CPU-Cores-8"]},
        predictor=predictor,
    )

    assert result["status"] == "completed"
    assert result["executed"] is True
    assert result["actual_cost"] == 25.0
    assert len(predictor.execution_history) == 1


# ─────────────────────────────────────────────────────────────
# GROUP 3: ToolRegistry & Agent Integration
# ─────────────────────────────────────────────────────────────

def test_tool_registry_invocation(tool_registry, coding_agent):
    """ToolRegistry should list and invoke tools by string name."""
    tools = tool_registry.list_tools()
    assert "check_resources" in tools
    assert "check_budget" in tools
    assert "estimate_cost" in tools
    assert "execute_task" in tools

    b_info = tool_registry.invoke("check_budget", agent=coding_agent)
    assert b_info["agent_name"] == "DevBot"


def test_agent_invoke_tool_and_execute_task(coding_agent):
    """BaseAgent should invoke tools natively via agent.invoke_tool() and execute_awarded_task()."""
    b_info = coding_agent.invoke_tool("check_budget")
    assert b_info["current_budget"] == 500.0

    exec_info = coding_agent.execute_awarded_task(
        "Fine-tune Llama model",
        allocation_result={"won": True, "payment": 40.0, "resources": ["GPU-Slot-1"]},
    )

    assert exec_info["status"] == "completed"
    assert exec_info["actual_cost"] == 40.0


# ─────────────────────────────────────────────────────────────
# GROUP 4: Full End-to-End Prediction -> Auction -> Task Execution
# ─────────────────────────────────────────────────────────────

def test_end_to_end_prediction_auction_execution(mock_llm_client):
    """
    End-to-End Test:
    1. Agent uses ML predictor to estimate cost & resources for a task.
    2. Agent creates adaptive bid based on task prediction.
    3. Bid is submitted to EXISTING AuctionEngine.
    4. ILP WDP solver runs and awards resources.
    5. Agent safely executes task via execute_awarded_task().
    6. Performance metrics feed back into ML predictor and AgentMemory.
    """
    agent = CodingAgent("AutoDevBot", budget=400.0, llm_client=mock_llm_client)
    engine = AuctionEngine([
        EngineResource("GPU-Slot-1", "gpu", 1.0),
        EngineResource("GPU-Slot-2", "gpu", 1.0),
        EngineResource("RAM-Block-32GB", "ram", 32.0),
        EngineResource("CPU-Cores-8", "cpu", 8.0),
        EngineResource("API-Budget-100", "api_budget", 100.0),
    ], method=SolverMethod.LP)

    task = "Deploy automated test pipeline"

    # 1. Prediction
    pred = agent.invoke_tool("estimate_cost", task_description=task)
    assert pred.predicted_cost > 0.0

    # 2. Bid Creation
    bid = agent.create_bid(task)
    agent.submit_to_engine(engine, bid)

    # 3. Auction Execution
    result = engine.run_auction(round_id=1)
    assert len(result.winning_bids) == 1

    # 4. Safe Task Execution & Feedback Loop
    winner = result.winning_bids[0]
    exec_result = agent.execute_awarded_task(
        task,
        allocation_result={
            "won": True,
            "payment": result.payments[winner.bid_id],
            "resources": winner.resource_bundle,
        },
    )

    assert exec_result["status"] == "completed"
    assert len(agent.predictor.execution_history) >= 1
