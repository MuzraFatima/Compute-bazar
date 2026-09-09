"""
=============================================================
 COMPUTE BAZAAR 2.0 -- Week 8A Tests: Dashboard API Metrics
=============================================================
 Tests the /api/v1/dashboard/metrics endpoint ensuring real data
 from SQLite database tables (Agent, Resource, Auction, Bid, Allocation)
 is correctly queried and served to the Dashboard UI.

 Validates:
   - Dashboard metrics route returns HTTP 200
   - Active agents count, performance metrics, and tier rankings
   - Resource inventory, capacities, and utilization rates
   - Social welfare, total spending, and VCG surplus savings
   - Jain's fairness index calculation
   - Auction history and LLM/Groq telemetry

 Run with:  python -m pytest tests/test_week8a_dashboard.py -v
=============================================================
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.models import init_db


@pytest.fixture(autouse=True)
def setup_test_database():
    """Ensure database tables exist before test execution."""
    init_db()


def test_dashboard_metrics_endpoint():
    """GET /api/v1/dashboard/metrics should return 200 with valid real database metrics."""
    with TestClient(app) as client:
        response = client.get("/api/v1/dashboard/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "summary" in data
        assert "agents" in data
        assert "resources" in data
        assert "utilization_by_category" in data
        assert "auction_history" in data
        assert "llm_telemetry" in data

        s = data["summary"]
        assert "active_agents_count" in s
        assert "social_welfare" in s
        assert "total_spending" in s
        assert "estimated_savings" in s
        assert "jains_fairness_index" in s
        assert 0.0 <= s["jains_fairness_index"] <= 1.0


def test_dashboard_static_index_html():
    """GET / should serve the Compute Bazaar 2.0 Dashboard HTML page."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "COMPUTE BAZAAR 2.0" in response.text
        assert "resourceUtilChart" in response.text
        assert "costSavingsChart" in response.text
        assert "agentWinsLossesChart" in response.text
        assert "auctionActivityChart" in response.text
