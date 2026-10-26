"""
=============================================================
 COMPUTE BAZAAR — Database Schema (Week 1)
=============================================================
 This file defines the SQLite schema structures for Week 1.
=============================================================
"""

import sqlite3
import os
from datetime import datetime

# Database location (consistent with models.py and DATABASE_URL)
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("sqlite:///"):
    DB_PATH = DATABASE_URL.replace("sqlite:///", "", 1)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DB_PATH = os.path.join(BASE_DIR, "compute_bazaar.db")

db_dir = os.path.dirname(DB_PATH)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)


def get_connection():
    """Open a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Returns rows as dict-like objects
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key support
    return conn


def create_tables():
    """
    Create all database tables if they don't already exist.
    
    This is the 'schema' — the blueprint for our data.
    Run this once when setting up the project.
    """
    conn = get_connection()
    cursor = conn.cursor()

    print("🗄️  Creating database tables...")

    # ── Table 1: agents ────────────────────────────────────────
    # Stores information about each AI agent registered in the system
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL UNIQUE,   -- Agent's unique name
            description     TEXT,                   -- What this agent does
            api_key         TEXT NOT NULL UNIQUE,   -- Secret key for authentication
            budget_total    REAL NOT NULL DEFAULT 1000.0,  -- Total budget allocated
            budget_remaining REAL NOT NULL DEFAULT 1000.0, -- Budget left to spend
            tier            TEXT DEFAULT 'bronze',  -- 'gold', 'silver', 'bronze'
            is_active       INTEGER DEFAULT 1,      -- 1=active, 0=disabled
            created_at      TEXT NOT NULL,          -- When this agent registered
            last_active_at  TEXT                    -- Last time agent submitted a bid
        )
    """)
    print("  ✅ agents table created")

    # ── Table 2: resources ─────────────────────────────────────
    # Stores the compute resources available for auction
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT NOT NULL UNIQUE,   -- e.g., "GPU-Slot-1"
            resource_type   TEXT NOT NULL,          -- 'gpu', 'api_budget', 'ram', 'cpu'
            capacity        REAL NOT NULL,          -- Total capacity (e.g., 1.0 for one slot)
            unit            TEXT,                   -- 'slot', 'dollars', 'gb', 'cores'
            description     TEXT,                   -- Human-readable description
            cost_per_unit   REAL DEFAULT 0.0,       -- Real-world cost (for pricing reference)
            is_available    INTEGER DEFAULT 1,      -- 1=available, 0=offline/maintenance
            created_at      TEXT NOT NULL
        )
    """)
    print("  ✅ resources table created")

    # ── Table 3: auctions ──────────────────────────────────────
    # Each row represents ONE auction round
    # Auctions run periodically (e.g., every 60 seconds)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS auctions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            round_number    INTEGER NOT NULL,       -- Auction #1, #2, #3...
            status          TEXT NOT NULL DEFAULT 'open',  
                            -- 'open' → accepting bids
                            -- 'running' → solving WDP
                            -- 'closed' → results finalized
            started_at      TEXT NOT NULL,
            closed_at       TEXT,                   -- NULL until auction closes
            total_value_extracted REAL DEFAULT 0.0, -- Sum of winning bid amounts
            total_agents_participated INTEGER DEFAULT 0,
            solver_method   TEXT DEFAULT 'greedy'   -- 'greedy', 'lp', 'dp'
        )
    """)
    print("  ✅ auctions table created")

    # ── Table 4: bids ──────────────────────────────────────────
    # Stores every bid submitted by every agent in every auction
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bids (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            auction_id      INTEGER NOT NULL REFERENCES auctions(id),
            agent_id        INTEGER NOT NULL REFERENCES agents(id),
            resource_bundle TEXT NOT NULL,          -- JSON array: ["GPU-1", "API-100"]
            bid_amount      REAL NOT NULL,          -- How much the agent values this bundle
            task_description TEXT,                  -- What the agent will do with resources
            priority        TEXT DEFAULT 'normal',  -- 'critical', 'high', 'normal', 'low'
            submitted_at    TEXT NOT NULL,
            is_winner       INTEGER DEFAULT 0,      -- 1 if this bid won
            payment_amount  REAL                    -- Actual amount charged (VCG price)
        )
    """)
    print("  ✅ bids table created")

    # ── Table 5: allocations ───────────────────────────────────
    # Stores the FINAL RESULT of each auction — who got what
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS allocations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            auction_id      INTEGER NOT NULL REFERENCES auctions(id),
            agent_id        INTEGER NOT NULL REFERENCES agents(id),
            bid_id          INTEGER NOT NULL REFERENCES bids(id),
            resources_allocated TEXT NOT NULL,      -- JSON: what they received
            bid_amount      REAL NOT NULL,          -- What they bid
            payment_amount  REAL NOT NULL,          -- What they actually pay (VCG)
            value_to_system REAL NOT NULL,          -- Contribution to total system value
            allocated_at    TEXT NOT NULL
        )
    """)
    print("  ✅ allocations table created")

    # ── Index: Speed up common queries ─────────────────────────
    # Indexes make searching faster (like a book's index)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bids_auction ON bids(auction_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bids_agent ON bids(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_allocations_auction ON allocations(auction_id)")
    print("  ✅ indexes created")

    conn.commit()
    conn.close()
    print(f"\n✨ Database ready at: {DB_PATH}")


def seed_demo_data():
    """
    Insert some sample data to test the database.
    This gives us realistic data to work with during development.
    """
    conn = get_connection()
    cursor = conn.cursor()

    print("\n🌱 Seeding demo data...")

    now = datetime.now().isoformat()

    # Insert sample resources
    resources = [
        ("GPU-Slot-1",    "gpu",        1.0,   "slot",    "NVIDIA A100 GPU, 1 compute slot",      2.50),
        ("GPU-Slot-2",    "gpu",        1.0,   "slot",    "NVIDIA A100 GPU, 1 compute slot",      2.50),
        ("API-Budget-100","api_budget", 100.0, "dollars", "OpenAI API budget ($100)",              1.00),
        ("RAM-Block-32GB","ram",        32.0,  "gb",      "32GB RAM allocation",                  0.10),
        ("CPU-Cores-8",   "cpu",        8.0,   "cores",   "8 CPU cores for light compute tasks",  0.05),
    ]

    for r in resources:
        try:
            cursor.execute("""
                INSERT INTO resources (name, resource_type, capacity, unit, description, cost_per_unit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (*r, now))
        except sqlite3.IntegrityError:
            pass  # Already exists, skip

    # Insert sample agents
    import secrets
    agents = [
        ("Agent-Alpha",   "Marketing copy generator",         "gold"),
        ("Agent-Beta",    "Financial transaction processor",   "gold"),
        ("Agent-Gamma",   "Image processing pipeline",        "silver"),
        ("Agent-Delta",   "Customer support chatbot",         "silver"),
        ("Agent-Epsilon", "Fraud detection system",           "bronze"),
    ]

    for name, desc, tier in agents:
        try:
            api_key = f"cb_{secrets.token_hex(16)}"
            cursor.execute("""
                INSERT INTO agents (name, description, api_key, budget_total, budget_remaining, tier, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, desc, api_key, 1000.0, 1000.0, tier, now))
        except sqlite3.IntegrityError:
            pass  # Already exists, skip

    conn.commit()

    # Print what we inserted
    print("\n  📦 Resources in database:")
    for row in cursor.execute("SELECT name, resource_type, capacity, unit FROM resources"):
        print(f"     • {row['name']} ({row['resource_type']}, {row['capacity']} {row['unit']})")

    print("\n  🤖 Agents in database:")
    for row in cursor.execute("SELECT name, tier, budget_remaining FROM agents"):
        print(f"     • {row['name']} [{row['tier']}] — Budget: ${row['budget_remaining']:.2f}")

    conn.close()
    print("\n✅ Demo data seeded!")


if __name__ == "__main__":
    create_tables()
    seed_demo_data()
    print("\n🎉 Database setup complete! Run week1_demo.py next.")
