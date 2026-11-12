"""
Tests for the API performance layer: query caching with write-through
invalidation, pagination, gzip compression and static cache headers.
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import StaticPool, create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.app.api.auth import get_db
from backend.app.core.cache import VersionedTTLCache, dashboard_cache
from backend.app.db.models import Agent, Auction, Base, Bid, Resource
from backend.app.main import app


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    dashboard_cache.clear()
    yield factory, engine
    app.dependency_overrides.pop(get_db, None)
    dashboard_cache.clear()
    Base.metadata.drop_all(bind=engine)


def _seed(factory, n_resources=3, n_agents=4, bids_per_agent=3):
    db = factory()
    for i in range(n_resources):
        db.add(Resource(name=f"R-{i}", resource_type="gpu", capacity=1.0))
    auction = Auction(round_number=1, status="closed")
    db.add(auction)
    db.flush()
    for a in range(n_agents):
        agent = Agent(name=f"A-{a}", api_key=f"k-{a}")
        db.add(agent)
        db.flush()
        for b in range(bids_per_agent):
            db.add(
                Bid(
                    auction_id=auction.id,
                    agent_id=agent.id,
                    resource_bundle=json.dumps(["R-0"]),
                    bid_amount=10.0 + b,
                    is_winner=(b == 0),
                    payment_amount=5.0 if b == 0 else None,
                )
            )
    db.commit()
    db.close()


def test_versioned_cache_hits_until_version_changes():
    cache = VersionedTTLCache(ttl=60)
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    assert cache.get_or_compute("k", 1, compute) == 1
    assert cache.get_or_compute("k", 1, compute) == 1
    assert cache.get_or_compute("k", 2, compute) == 2
    assert cache.hits == 1 and cache.misses == 2


def test_versioned_cache_expires_after_ttl():
    cache = VersionedTTLCache(ttl=0)
    assert cache.get_or_compute("k", 1, lambda: "a") == "a"
    assert cache.get_or_compute("k", 1, lambda: "b") == "b"


def test_dashboard_aggregates_match_seeded_rows(session_factory):
    factory, _ = session_factory
    _seed(factory)
    data = TestClient(app).get("/api/v1/dashboard/metrics").json()
    s = data["summary"]
    assert s["total_bids_count"] == 12
    assert s["winning_bids_count"] == 4
    assert s["losing_bids_count"] == 8
    assert s["social_welfare"] == pytest.approx(40.0)
    assert s["total_spending"] == pytest.approx(20.0)
    a0 = next(a for a in data["agents"] if a["name"] == "A-0")
    assert (a0["wins"], a0["losses"], a0["total_surplus"]) == (1, 2, 5.0)
    assert data["auction_history"][0]["bids_count"] == 12


def test_dashboard_query_count_is_constant_in_agent_count(session_factory):
    factory, engine = session_factory
    _seed(factory, n_agents=12)
    statements = []

    def record(conn, cursor, statement, *args):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        TestClient(app).get("/api/v1/dashboard/metrics")
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert len(statements) < 15, f"{len(statements)} statements for 12 agents"


def test_dashboard_cache_invalidates_on_new_bid(session_factory):
    factory, _ = session_factory
    _seed(factory)
    client = TestClient(app)
    first = client.get("/api/v1/dashboard/metrics").json()["summary"]["total_bids_count"]
    db = factory()
    db.add(Bid(auction_id=1, agent_id=1, resource_bundle='["R-1"]', bid_amount=3.0))
    db.commit()
    db.close()
    second = client.get("/api/v1/dashboard/metrics").json()["summary"]["total_bids_count"]
    assert (first, second) == (12, 13)


def test_resources_are_paginated_with_total_header(session_factory):
    factory, _ = session_factory
    _seed(factory, n_resources=7)
    res = TestClient(app).get("/api/v1/resources?limit=3&offset=3")
    assert res.status_code == 200
    assert res.headers["X-Total-Count"] == "7"
    assert [r["name"] for r in res.json()] == ["R-3", "R-4", "R-5"]


def test_resources_limit_is_bounded(session_factory):
    assert TestClient(app).get("/api/v1/resources?limit=10000").status_code == 422


def test_large_json_is_gzipped(session_factory):
    factory, _ = session_factory
    _seed(factory, n_agents=30)
    res = TestClient(app).get("/api/v1/dashboard/metrics", headers={"Accept-Encoding": "gzip"})
    assert res.headers.get("content-encoding") == "gzip"


def test_static_cache_headers():
    client = TestClient(app)
    assert "max-age=604800" in client.get("/static/chart.umd.js").headers["cache-control"]
    assert client.get("/static/app.js").headers["cache-control"] == "no-cache"
