"""
=============================================================
 COMPUTE BAZAAR — Integration Tests for Weeks 3-8
=============================================================
 Verifies:
   - Agent Registration REST API
   - JWT Auth token generation
   - Profile me endpoint
   - Combinatorial bid submission validation rules
   - Database model states
   - Auction scheduler resolver loop functionality
=============================================================
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker

# Setup test database path
from backend.app.db.models import Base, Agent, Resource, Auction, Bid, Allocation
from backend.app.main import app
from backend.app.api.auth import get_db


# ── Per-test in-memory DB fixture ─────────────────────────────

@pytest.fixture()
def test_client():
    """Create an isolated in-memory SQLite DB and TestClient for each test."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Seed mock resources
    db = TestingSessionLocal()
    resources = [
        Resource(name="GPU-1", resource_type="gpu", capacity=1.0, unit="slot", cost_per_unit=2.0),
        Resource(name="GPU-2", resource_type="gpu", capacity=1.0, unit="slot", cost_per_unit=2.0),
        Resource(name="API-100", resource_type="api_budget", capacity=100.0, unit="dollars", cost_per_unit=1.0),
    ]
    db.bulk_save_objects(resources)

    # Seed active auction round
    auction = Auction(round_number=1, status="open")
    db.add(auction)
    db.commit()
    db.close()

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)


# ── REST API Integration Tests ────────────────────────────────

def test_register_agent_success(test_client):
    payload = {"name": "Test-Agent-Alpha", "description": "Integrator agent", "tier": "gold"}
    res = test_client.post("/api/v1/agents/register", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["name"] == "Test-Agent-Alpha"
    assert data["tier"] == "gold"
    assert "api_key" in data


def test_register_agent_duplicate_fails(test_client):
    payload = {"name": "Duplicate-Agent", "tier": "silver"}
    res1 = test_client.post("/api/v1/agents/register", json=payload)
    assert res1.status_code == 201

    res2 = test_client.post("/api/v1/agents/register", json=payload)
    assert res2.status_code == 400
    assert "already registered" in res2.json()["detail"]


def test_get_jwt_token_auth_flow(test_client):
    # 1. Register agent
    reg = test_client.post("/api/v1/agents/register", json={"name": "Auth-Agent"})
    api_key = reg.json()["api_key"]

    # 2. Get JWT token
    res = test_client.post(f"/api/v1/auth/token?api_key={api_key}")
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_get_current_agent_profile(test_client):
    # 1. Register agent & login
    reg = test_client.post("/api/v1/agents/register", json={"name": "Profile-Agent"})
    api_key = reg.json()["api_key"]
    token = test_client.post(f"/api/v1/auth/token?api_key={api_key}").json()["access_token"]

    # 2. Check profile using JWT
    headers = {"Authorization": f"Bearer {token}"}
    res = test_client.get("/api/v1/agents/me", headers=headers)
    assert res.status_code == 200
    assert res.json()["name"] == "Profile-Agent"

    # 3. Check profile using API Key header
    headers_key = {"X-API-Key": api_key}
    res_key = test_client.get("/api/v1/agents/me", headers=headers_key)
    assert res_key.status_code == 200
    assert res_key.json()["name"] == "Profile-Agent"


def test_submit_bid_correctness(test_client):
    # 1. Register and login
    reg = test_client.post("/api/v1/agents/register", json={"name": "Bidder-Agent"})
    api_key = reg.json()["api_key"]
    headers = {"X-API-Key": api_key}

    # 2. Submit bid
    bid_payload = {
        "resource_bundle": ["GPU-1", "API-100"],
        "bid_amount": 150.0,
        "task_description": "Run model prediction batch",
        "priority": "high",
    }
    res = test_client.post("/api/v1/bids/submit", json=bid_payload, headers=headers)
    assert res.status_code == 201
    data = res.json()
    assert data["agent_name"] == "Bidder-Agent"
    assert data["resource_bundle"] == ["GPU-1", "API-100"]
    assert data["bid_amount"] == 150.0


def test_submit_bid_budget_overflow_fails(test_client):
    reg = test_client.post("/api/v1/agents/register", json={"name": "Poor-Agent"})
    api_key = reg.json()["api_key"]
    headers = {"X-API-Key": api_key}

    # Total initial budget is $1000. Let's bid $1200
    bid_payload = {
        "resource_bundle": ["GPU-1"],
        "bid_amount": 1200.0,
        "task_description": "Exceeding budget limits",
    }
    res = test_client.post("/api/v1/bids/submit", json=bid_payload, headers=headers)
    assert res.status_code == 400
    assert "Insufficient remaining budget" in res.json()["detail"]


def test_submit_bid_invalid_resource_fails(test_client):
    reg = test_client.post("/api/v1/agents/register", json={"name": "Agent-Resource-Check"})
    api_key = reg.json()["api_key"]
    headers = {"X-API-Key": api_key}

    bid_payload = {
        "resource_bundle": ["GPU-1", "INVALID-GPU-3"],
        "bid_amount": 50.0,
    }
    res = test_client.post("/api/v1/bids/submit", json=bid_payload, headers=headers)
    assert res.status_code == 400
    assert "Some resources are offline or do not exist" in res.json()["detail"]


def test_get_resources_list(test_client):
    res = test_client.get("/api/v1/resources")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 3
    resource_names = [r["name"] for r in data]
    assert "GPU-1" in resource_names
    assert "GPU-2" in resource_names
    assert "API-100" in resource_names


# ── Database Model & Scheduler solver tests ──────────────────

@pytest.mark.asyncio
async def test_scheduler_loop_runs_correctly(test_client):
    # Access the current test's overridden DB via the fixture's override
    from backend.app.api.auth import get_db as _get_db
    db_gen = app.dependency_overrides[_get_db]()
    db = next(db_gen)

    # 1. Register agents
    a1 = Agent(name="A", api_key="k1", budget_total=1000.0, budget_remaining=1000.0, tier="gold")
    a2 = Agent(name="B", api_key="k2", budget_total=1000.0, budget_remaining=1000.0, tier="silver")
    db.add_all([a1, a2])
    db.commit()

    # Get current round
    active_auction = db.query(Auction).filter(Auction.status == "open").first()

    # 2. Submit conflicting bids
    b1 = Bid(auction_id=active_auction.id, agent_id=a1.id, resource_bundle=json.dumps(["GPU-1"]), bid_amount=80.0, priority="normal")
    b2 = Bid(auction_id=active_auction.id, agent_id=a2.id, resource_bundle=json.dumps(["GPU-1"]), bid_amount=50.0, priority="normal")
    db.add_all([b1, b2])
    db.commit()

    # 3. Call run_auction_round
    from backend.app.auction.scheduler import run_auction_round
    await run_auction_round(db)

    # Verify outcomes
    closed_auction = db.query(Auction).filter(Auction.id == active_auction.id).first()
    assert closed_auction.status == "closed"
    assert closed_auction.total_value_extracted == 80.0

    win_bid = db.query(Bid).filter(Bid.id == b1.id).first()
    lost_bid = db.query(Bid).filter(Bid.id == b2.id).first()
    assert win_bid.is_winner == True
    assert lost_bid.is_winner == False

    # VCG Payment: value without A = 50; tier discount Gold 10% -> 50 * 0.90 = 45
    assert win_bid.payment_amount == pytest.approx(45.0)

    # Agent budget: $1000 - $45 = $955
    agent_a = db.query(Agent).filter(Agent.name == "A").first()
    assert agent_a.budget_remaining == pytest.approx(955.0)

    # Next round opened
    new_round = db.query(Auction).filter(Auction.status == "open").first()
    assert new_round is not None
    assert new_round.round_number == 2

    db.close()
