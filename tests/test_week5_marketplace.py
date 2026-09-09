"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Week 5 Tests: Dynamic Marketplace
=============================================================
 Tests the dynamic marketplace framework:
   - ResourceProvider model (capacity, minimum price, reliability)
   - Multi-category resources (GPU, CPU, RAM, LLM/API tokens)
   - Dynamic resource availability toggles (online/offline/maintenance)
   - Supply and demand tracking across auction rounds
   - Scarcity-based dynamic reserve pricing
   - Interoperability with existing AuctionEngine and Week 4 AI agents

 Run with:  python -m pytest tests/test_week5_marketplace.py -v
=============================================================
"""

import pytest
from typing import List

from compute_bazaar.marketplace import (
    ResourceProvider,
    MarketResource,
    MarketResourceType,
    DynamicMarketplace,
)
from compute_bazaar.agents import CodingAgent, ResearchAgent
from compute_bazaar.llm import GroqLLMClient, LLMConfig
from backend.app.auction.engine import (
    AuctionEngine,
    Resource as EngineResource,
    SolverMethod,
)


# ─────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def marketplace():
    """Fresh DynamicMarketplace instance."""
    return DynamicMarketplace("Test Marketplace")


@pytest.fixture
def custom_provider():
    """Sample ResourceProvider with GPU, CPU, RAM, and LLM/API resources."""
    provider = ResourceProvider(
        provider_id="prov-groq-cloud",
        name="Groq Cloud Infrastructure",
        provider_type="api_vendor",
        reliability_score=0.99,
    )
    provider.add_resource(MarketResource(
        name="Groq-Llama3-API",
        resource_type=MarketResourceType.LLM_API.value,
        total_capacity=5000.0,
        available_capacity=5000.0,
        min_price=20.0,
        current_price=20.0,
        unit="tokens",
    ))
    provider.add_resource(MarketResource(
        name="Lambda-H100-GPU",
        resource_type=MarketResourceType.GPU.value,
        total_capacity=4.0,
        available_capacity=4.0,
        min_price=80.0,
        current_price=80.0,
        unit="slot",
    ))
    return provider


# ─────────────────────────────────────────────────────────────
# GROUP 1: ResourceProvider Model
# ─────────────────────────────────────────────────────────────

def test_resource_provider_initialization(custom_provider):
    """ResourceProvider should retain metadata and resource inventory."""
    assert custom_provider.provider_id == "prov-groq-cloud"
    assert custom_provider.name == "Groq Cloud Infrastructure"
    assert custom_provider.provider_type == "api_vendor"
    assert custom_provider.reliability_score == 0.99
    assert custom_provider.is_active is True
    assert len(custom_provider.resources) == 2


def test_provider_capacity_and_availability_updates(custom_provider):
    """Provider should support dynamic capacity adjustments and availability toggles."""
    # Capacity update
    custom_provider.update_resource_capacity("Lambda-H100-GPU", 8.0)
    res = custom_provider.resources["Lambda-H100-GPU"]
    assert res.total_capacity == 8.0
    assert res.available_capacity == 8.0

    # Availability toggle
    custom_provider.set_resource_availability("Groq-Llama3-API", False)
    assert res.is_available is True
    assert custom_provider.resources["Groq-Llama3-API"].is_available is False
    assert len(custom_provider.get_available_resources()) == 1


def test_provider_serialization(custom_provider):
    """ResourceProvider.to_dict() should produce a serializable state dictionary."""
    d = custom_provider.to_dict()
    assert d["provider_id"] == "prov-groq-cloud"
    assert d["resource_count"] == 2
    assert isinstance(d["resources"], list)


# ─────────────────────────────────────────────────────────────
# GROUP 2: MarketResource Categories & Capacity Logic
# ─────────────────────────────────────────────────────────────

def test_market_resource_types():
    """MarketResource should support GPU, CPU, RAM, and LLM/API categories."""
    gpu = MarketResource("GPU-1", resource_type=MarketResourceType.GPU.value, min_price=30.0)
    cpu = MarketResource("CPU-1", resource_type=MarketResourceType.CPU.value, min_price=10.0)
    ram = MarketResource("RAM-1", resource_type=MarketResourceType.RAM.value, min_price=15.0)
    llm = MarketResource("LLM-1", resource_type=MarketResourceType.LLM_API.value, min_price=25.0)

    assert gpu.resource_type == "gpu"
    assert cpu.resource_type == "cpu"
    assert ram.resource_type == "ram"
    assert llm.resource_type == "llm_api"


def test_market_resource_allocation_and_release():
    """MarketResource should allocate and release capacity correctly."""
    res = MarketResource("RAM-Block-64GB", total_capacity=64.0, available_capacity=64.0)

    assert res.allocate(16.0) is True
    assert res.available_capacity == 48.0

    # Oversubscription fails
    assert res.allocate(100.0) is False
    assert res.available_capacity == 48.0

    # Release capacity
    res.release(16.0)
    assert res.available_capacity == 64.0


def test_minimum_floor_price_enforcement():
    """Current price must never drop below minimum floor price."""
    res = MarketResource("GPU-Floor-Test", min_price=50.0, current_price=10.0)
    assert res.current_price == 50.0  # Enforced in __post_init__


# ─────────────────────────────────────────────────────────────
# GROUP 3: DynamicMarketplace & Scarcity-Based Pricing
# ─────────────────────────────────────────────────────────────

def test_marketplace_registration(marketplace, custom_provider):
    """DynamicMarketplace should register providers and catalogue all resources."""
    marketplace.register_provider(custom_provider)

    assert marketplace.get_provider("prov-groq-cloud") is not None
    assert marketplace.get_resource("Groq-Llama3-API") is not None
    assert marketplace.get_resource("Lambda-H100-GPU") is not None


def test_demand_tracking_and_scarcity_ratio(marketplace):
    """Marketplace should record demand from bids and compute scarcity ratios."""
    # Pre-seeded GPU-Slot-1 has total capacity 1.0
    bids_round1 = [["GPU-Slot-1"], ["GPU-Slot-1"], ["GPU-Slot-1"]]
    marketplace.record_demand(bids_round1)

    scarcity = marketplace.calculate_scarcity("GPU-Slot-1")
    assert scarcity == pytest.approx(3.0)  # 3 requested / 1 capacity


def test_scarcity_based_dynamic_pricing(marketplace):
    """High demand / scarcity should increase current resource price above minimum floor."""
    res = marketplace.get_resource("GPU-Slot-1")
    initial_price = res.current_price
    min_price = res.min_price

    # Simulate heavy demand (4 bids requesting GPU-Slot-1)
    marketplace.record_demand([["GPU-Slot-1"], ["GPU-Slot-1"], ["GPU-Slot-1"], ["GPU-Slot-1"]])
    marketplace.update_dynamic_pricing(alpha=0.5)

    # Scarcity = 4.0 / 1.0 = 4.0
    # Expected price = min_price * (1 + 0.5 * 4.0) = min_price * 3.0
    assert res.current_price > initial_price
    assert res.current_price == pytest.approx(min_price * 3.0)


def test_dynamic_pricing_resets_when_demand_subsides(marketplace):
    """When demand drops, dynamic price should return toward minimum floor price."""
    res = marketplace.get_resource("GPU-Slot-1")
    min_price = res.min_price

    # Round 1: High demand
    marketplace.record_demand([["GPU-Slot-1"], ["GPU-Slot-1"]])
    marketplace.update_dynamic_pricing(alpha=0.5)
    high_price = res.current_price
    assert high_price > min_price

    # Round 2: Zero demand
    marketplace.record_demand([])
    marketplace.update_dynamic_pricing(alpha=0.5)
    low_price = res.current_price
    assert low_price == pytest.approx(min_price)


def test_dynamic_resource_availability_toggle(marketplace):
    """Setting resource availability to False should take it offline."""
    assert marketplace.get_resource("GPU-Slot-1").is_available is True

    marketplace.set_resource_availability("GPU-Slot-1", False)
    assert marketplace.get_resource("GPU-Slot-1").is_available is False

    available_names = [r.name for r in marketplace.list_available_resources()]
    assert "GPU-Slot-1" not in available_names


def test_market_summary_output(marketplace, custom_provider):
    """get_market_summary() should provide complete diagnostic overview."""
    marketplace.register_provider(custom_provider)
    summary = marketplace.get_market_summary()

    assert summary["marketplace_name"] == "Test Marketplace"
    assert summary["total_providers"] >= 2
    assert summary["total_resources"] >= 7
    assert "gpu" in summary["supply"]
    assert "resource_prices" in summary


# ─────────────────────────────────────────────────────────────
# GROUP 4: Integration with Existing AuctionEngine & AI Agents
# ─────────────────────────────────────────────────────────────

def test_marketplace_interoperability_with_auction_engine(marketplace):
    """
    Marketplace resources can be converted to EngineResource objects
    and run in the existing AuctionEngine without affecting WDP or VCG.
    """
    # 1. Convert marketplace resources to EngineResource
    engine_resources = [
        EngineResource(r.name, r.resource_type, r.total_capacity)
        for r in marketplace.list_available_resources()
    ]
    engine = AuctionEngine(engine_resources, method=SolverMethod.LP)

    # 2. AI agent generates intelligent bids
    llm = GroqLLMClient(config=LLMConfig(mock_mode=True))
    agent = CodingAgent("MarketBot", budget=500.0, llm_client=llm)

    bid = agent.create_bid("Fine-tune model on GPU and RAM")
    agent.submit_to_engine(engine, bid)

    # 3. Run auction
    result = engine.run_auction(round_id=1)
    assert result.total_value > 0.0
    assert len(result.winning_bids) == 1
