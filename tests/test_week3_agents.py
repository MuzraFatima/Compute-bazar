"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Week 3 Tests: Autonomous AI Agents
=============================================================
 Tests the autonomous agent framework (BaseAgent + 4 specialized agents).
 All tests run in MOCK mode (no Groq API key required).

 Validates:
   - Agent initialization, capabilities, and preferred resources
   - Task analysis and structured decision generation (TaskDecision)
   - Resource selection matching domain preferences
   - Budget constraints and willingness-to-pay capping
   - Budget overrides
   - Decision history logging
   - JSON response parsing & validation from Groq client
   - Error handling and fallback defaults

 Run with:  python -m pytest tests/test_week3_agents.py -v
=============================================================
"""

import json
import pytest
from compute_bazaar.agents import (
    BaseAgent,
    TaskDecision,
    AgentCapability,
    CodingAgent,
    ResearchAgent,
    DataAgent,
    AnalysisAgent,
)
from compute_bazaar.llm import GroqLLMClient, LLMConfig, LLMResponse, LLMUsage


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    """Mock Groq client for agent testing."""
    return GroqLLMClient(config=LLMConfig(mock_mode=True))


@pytest.fixture
def coding_agent(mock_llm_client):
    return CodingAgent("DevBot", budget=400.0, llm_client=mock_llm_client)


@pytest.fixture
def research_agent(mock_llm_client):
    return ResearchAgent("ScholarBot", budget=300.0, llm_client=mock_llm_client)


@pytest.fixture
def data_agent(mock_llm_client):
    return DataAgent("ETLBot", budget=500.0, llm_client=mock_llm_client)


@pytest.fixture
def analysis_agent(mock_llm_client):
    return AnalysisAgent("QuantBot", budget=600.0, llm_client=mock_llm_client)


# ─────────────────────────────────────────────────────────────
# GROUP 1: Agent Initialization & Properties
# ─────────────────────────────────────────────────────────────

def test_coding_agent_properties(coding_agent):
    """CodingAgent should declare correct type, capability, and resources."""
    assert coding_agent.name == "DevBot"
    assert coding_agent.agent_type == "coding"
    assert coding_agent.capability == AgentCapability.CODING
    assert coding_agent.budget == 400.0
    assert "GPU-Slot-1" in coding_agent.preferred_resources
    assert "CPU-Cores-8" in coding_agent.preferred_resources
    assert "code generation" in coding_agent.system_prompt.lower()


def test_research_agent_properties(research_agent):
    """ResearchAgent should declare research capability and preferred resources."""
    assert research_agent.name == "ScholarBot"
    assert research_agent.agent_type == "research"
    assert research_agent.capability == AgentCapability.RESEARCH
    assert research_agent.budget == 300.0
    assert "API-Budget-100" in research_agent.preferred_resources
    assert "research" in research_agent.system_prompt.lower()


def test_data_agent_properties(data_agent):
    """DataAgent should declare data_processing capability and preferred resources."""
    assert data_agent.name == "ETLBot"
    assert data_agent.agent_type == "data_processing"
    assert data_agent.capability == AgentCapability.DATA_PROCESSING
    assert data_agent.budget == 500.0
    assert "RAM-Block-32GB" in data_agent.preferred_resources
    assert "etl" in data_agent.system_prompt.lower()


def test_analysis_agent_properties(analysis_agent):
    """AnalysisAgent should declare analysis capability and preferred resources."""
    assert analysis_agent.name == "QuantBot"
    assert analysis_agent.agent_type == "analysis"
    assert analysis_agent.capability == AgentCapability.ANALYSIS
    assert analysis_agent.budget == 600.0
    assert "CPU-Cores-8" in analysis_agent.preferred_resources
    assert "financial modeling" in analysis_agent.system_prompt.lower()


def test_agent_repr(coding_agent):
    """Agent repr should contain name, type, budget, and mode."""
    r = repr(coding_agent)
    assert "CodingAgent" in r
    assert "DevBot" in r
    assert "$400.00" in r
    assert "MOCK" in r


# ─────────────────────────────────────────────────────────────
# GROUP 2: Task Analysis (Mock Mode)
# ─────────────────────────────────────────────────────────────

def test_coding_agent_analyze_task(coding_agent):
    """CodingAgent.analyze_task() should return a valid TaskDecision."""
    task = "Fine-tune llama model on python repo for code generation"
    decision = coding_agent.analyze_task(task)

    assert isinstance(decision, TaskDecision)
    assert decision.agent_name == "DevBot"
    assert decision.agent_type == "coding"
    assert decision.task_description == task
    assert len(decision.required_resources) > 0
    assert 1 <= decision.complexity <= 10
    assert decision.estimated_utility > 0.0
    assert decision.max_willingness_to_pay <= coding_agent.budget
    assert decision.max_willingness_to_pay <= decision.estimated_utility
    assert decision.is_mock is True
    assert len(decision.reasoning) > 0


def test_research_agent_analyze_task(research_agent):
    """ResearchAgent.analyze_task() should request research resources."""
    task = "Conduct multi-agent market research report"
    decision = research_agent.analyze_task(task)

    assert decision.agent_name == "ScholarBot"
    assert decision.agent_type == "research"
    assert "API-Budget-100" in decision.required_resources
    assert decision.max_willingness_to_pay <= research_agent.budget


def test_data_agent_analyze_task(data_agent):
    """DataAgent.analyze_task() should request data processing resources."""
    task = "Clean 10GB dataset and extract vector embeddings"
    decision = data_agent.analyze_task(task)

    assert decision.agent_name == "ETLBot"
    assert decision.agent_type == "data_processing"
    assert "RAM-Block-32GB" in decision.required_resources
    assert decision.max_willingness_to_pay <= data_agent.budget


def test_analysis_agent_analyze_task(analysis_agent):
    """AnalysisAgent.analyze_task() should request analysis resources."""
    task = "Run Monte Carlo risk analysis on portfolio"
    decision = analysis_agent.analyze_task(task)

    assert decision.agent_name == "QuantBot"
    assert decision.agent_type == "analysis"
    assert "CPU-Cores-8" in decision.required_resources
    assert decision.max_willingness_to_pay <= analysis_agent.budget


# ─────────────────────────────────────────────────────────────
# GROUP 3: Budget Constraints & Overrides
# ─────────────────────────────────────────────────────────────

def test_budget_override(coding_agent):
    """Passing budget_override to analyze_task() should cap WTP to override budget."""
    task = "Build REST API"
    decision = coding_agent.analyze_task(task, budget_override=50.0)

    assert decision.max_willingness_to_pay <= 50.0


def test_decision_history_recording(coding_agent):
    """Analyzing tasks should record decision history."""
    assert len(coding_agent.decision_history) == 0
    coding_agent.analyze_task("Task 1")
    coding_agent.analyze_task("Task 2")

    assert len(coding_agent.decision_history) == 2
    history = coding_agent.get_history()
    assert len(history) == 2
    assert history[0]["task_description"] == "Task 1"
    assert history[1]["task_description"] == "Task 2"


# ─────────────────────────────────────────────────────────────
# GROUP 4: TaskDecision Serialization & Schema
# ─────────────────────────────────────────────────────────────

def test_task_decision_to_dict(coding_agent):
    """TaskDecision.to_dict() should produce a clean JSON-serializable dict."""
    decision = coding_agent.analyze_task("Create docker pipeline")
    d = decision.to_dict()

    assert d["agent_name"] == "DevBot"
    assert d["agent_type"] == "coding"
    assert "understood_task" in d
    assert "required_resources" in d
    assert "complexity" in d
    assert "estimated_utility" in d
    assert "max_willingness_to_pay" in d
    assert "priority" in d
    assert "reasoning" in d
    assert "confidence" in d

    # Verify JSON serializability
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["agent_name"] == "DevBot"


# ─────────────────────────────────────────────────────────────
# GROUP 5: Parsing Simulated LLM Responses
# ─────────────────────────────────────────────────────────────

class StubLLMClient:
    """Stub LLM client that returns predefined JSON for testing live parsing."""
    def __init__(self, response_json: dict):
        self.config = LLMConfig(mock_mode=False)  # Pretend live mode to bypass mock generator
        self.response_json = response_json

    def complete(self, prompt, system_prompt=None, json_mode=False, temperature=None, max_tokens=None, model=None):
        return LLMResponse(
            content=json.dumps(self.response_json),
            model="stub-llama",
            usage=LLMUsage(prompt_tokens=50, completion_tokens=100, total_tokens=150),
            latency_ms=15.0,
            is_mock=False,
        )


def test_live_parsing_logic():
    """Verify BaseAgent parses structured JSON from LLM response correctly."""
    stub_json = {
        "understood_task": "Build a secure authentication service",
        "required_resources": ["GPU-Slot-1", "RAM-Block-32GB"],
        "complexity": 7,
        "estimated_utility": 150.0,
        "max_willingness_to_pay": 110.0,
        "priority": "high",
        "reasoning": [
            "Auth requires encryption overhead",
            "High concurrency demands 32GB RAM",
        ],
        "confidence": 0.90,
    }
    stub_client = StubLLMClient(stub_json)
    agent = CodingAgent("AuthBot", budget=200.0, llm_client=stub_client)

    decision = agent.analyze_task("Implement OAuth2 server")

    assert decision.understood_task == "Build a secure authentication service"
    assert decision.required_resources == ["GPU-Slot-1", "RAM-Block-32GB"]
    assert decision.complexity == 7
    assert decision.estimated_utility == 150.0
    assert decision.max_willingness_to_pay == 110.0
    assert decision.priority == "high"
    assert decision.confidence == 0.90
    assert decision.is_mock is False
    assert decision.llm_latency_ms == 15.0
    assert decision.llm_tokens_used == 150


def test_live_parsing_clamping_and_filtering():
    """Verify BaseAgent clamps invalid resource names and budget over-allocations."""
    stub_json = {
        "understood_task": "Overpriced task",
        "required_resources": ["GPU-Slot-1", "NONEXISTENT_RESOURCE"],
        "complexity": 15,  # Out of range (>10)
        "estimated_utility": 500.0,
        "max_willingness_to_pay": 999.0,  # Exceeds budget of 100.0
        "priority": "invalid_priority",
        "confidence": 1.5,  # Out of range (>1.0)
    }
    stub_client = StubLLMClient(stub_json)
    agent = CodingAgent("BudgetBot", budget=100.0, llm_client=stub_client)

    decision = agent.analyze_task("Big task")

    # NONEXISTENT_RESOURCE filtered out
    assert decision.required_resources == ["GPU-Slot-1"]
    # Complexity clamped to 10
    assert decision.complexity == 10
    # WTP clamped to budget (100.0)
    assert decision.max_willingness_to_pay == 100.0
    # Priority defaulted to normal
    assert decision.priority == "normal"
    # Confidence clamped to 1.0
    assert decision.confidence == 1.0
