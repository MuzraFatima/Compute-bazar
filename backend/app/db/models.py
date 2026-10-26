"""
=============================================================
 COMPUTE BAZAAR — SQLAlchemy Database Models (Weeks 3-8)
=============================================================
 Defines the structure of the tables:
   - Agent: registered agents with budgets and tiers
   - Resource: available compute resources
   - Auction: auction rounds with status and timing
   - Bid: individual agent bids for combinations of resources
   - Allocation: successful resource assignments and pricing
=============================================================
"""

import os
from datetime import datetime, timezone
from typing import List

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Index, Table
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Define database location (SQLite file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "compute_bazaar.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Ensure parent directory exists for SQLite database file
if DATABASE_URL.startswith("sqlite:///"):
    sqlite_path = DATABASE_URL.replace("sqlite:///", "", 1)
    db_dir = os.path.dirname(sqlite_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    DB_PATH = sqlite_path

Base = declarative_base()

# ── Agent Model ──────────────────────────────────────────────
class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)
    api_key = Column(String, unique=True, nullable=False)
    budget_total = Column(Float, default=1000.0, nullable=False)
    budget_remaining = Column(Float, default=1000.0, nullable=False)
    tier = Column(String, default="bronze", nullable=False)  # 'gold', 'silver', 'bronze'
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(String, default=utc_now_iso, nullable=False)
    last_active_at = Column(String, nullable=True)

    # Relationships
    bids = relationship("Bid", back_populates="agent")
    allocations = relationship("Allocation", back_populates="agent")

# ── Resource Model ───────────────────────────────────────────
class Resource(Base):
    __tablename__ = "resources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    resource_type = Column(String, nullable=False)  # 'gpu', 'api_budget', 'ram', 'cpu'
    capacity = Column(Float, default=1.0, nullable=False)
    unit = Column(String, default="slot", nullable=False)
    cost_per_unit = Column(Float, default=0.0, nullable=False)
    is_available = Column(Boolean, default=True, nullable=False)
    created_at = Column(String, default=utc_now_iso, nullable=False)

# ── Auction Model ────────────────────────────────────────────
class Auction(Base):
    __tablename__ = "auctions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, unique=True, nullable=False)
    status = Column(String, default="open", nullable=False)  # 'open', 'running', 'closed'
    started_at = Column(String, default=utc_now_iso, nullable=False)
    closed_at = Column(String, nullable=True)
    total_value_extracted = Column(Float, default=0.0)
    total_agents_participated = Column(Integer, default=0)
    solver_method = Column(String, default="lp")

    # Relationships
    bids = relationship("Bid", back_populates="auction")
    allocations = relationship("Allocation", back_populates="auction")

# ── Bid Model ────────────────────────────────────────────────
class Bid(Base):
    __tablename__ = "bids"

    id = Column(Integer, primary_key=True, autoincrement=True)
    auction_id = Column(Integer, ForeignKey("auctions.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    resource_bundle = Column(String, nullable=False)  # JSON-encoded list of resource names
    bid_amount = Column(Float, nullable=False)
    task_description = Column(String, nullable=True)
    priority = Column(String, default="normal")  # 'critical', 'high', 'normal', 'low'
    submitted_at = Column(String, default=utc_now_iso, nullable=False)
    is_winner = Column(Boolean, default=False)
    payment_amount = Column(Float, nullable=True)

    # Relationships
    auction = relationship("Auction", back_populates="bids")
    agent = relationship("Agent", back_populates="bids")
    allocation = relationship("Allocation", uselist=False, back_populates="bid")

# ── Allocation Model ─────────────────────────────────────────
class Allocation(Base):
    __tablename__ = "allocations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    auction_id = Column(Integer, ForeignKey("auctions.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    bid_id = Column(Integer, ForeignKey("bids.id", ondelete="CASCADE"), nullable=False)
    resources_allocated = Column(String, nullable=False)  # JSON-encoded list of resource names
    bid_amount = Column(Float, nullable=False)
    payment_amount = Column(Float, nullable=False)
    value_to_system = Column(Float, nullable=False)
    allocated_at = Column(String, default=utc_now_iso, nullable=False)

    # Relationships
    auction = relationship("Auction", back_populates="allocations")
    agent = relationship("Agent", back_populates="allocations")
    bid = relationship("Bid", back_populates="allocation")


# ── Indexes ──────────────────────────────────────────────────
Index("idx_bids_auction", Bid.auction_id)
Index("idx_bids_agent", Bid.agent_id)
Index("idx_allocations_auction", Allocation.auction_id)


# ── Engine & Session setup ────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Safe for SQLite with multiple threads in FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)

# Auto-initialize tables on module import
init_db()
