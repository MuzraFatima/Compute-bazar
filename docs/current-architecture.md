# Compute Bazaar 1.0 — Current Architecture (As-Built)

> This document describes the **actual** current architecture and data flow
> of Compute Bazaar 1.0 as of Week 8 completion. It is the reference baseline
> for Compute Bazaar 2.0 development.

---

## 1. System Overview

Compute Bazaar is a **combinatorial auction platform** for AI compute resources.
It allocates scarce resources (GPUs, API budgets, RAM, CPU cores) among competing
autonomous agents using **Integer Linear Programming (ILP)** and
**VCG (Vickrey-Clarke-Groves) pricing**.

```
  ┌──────────────────────┐
  │   AI Agents / Bots   │
  │  (Python SDK / HTTP) │
  └─────────┬────────────┘
            │ REST + WebSocket
            ▼
  ┌──────────────────────┐      ┌─────────────────────┐
  │   FastAPI Gateway    │      │  Static Dashboard   │
  │  /api/v1/*  routes   │─────▶│  Glassmorphism UI   │
  │  JWT + API Key Auth  │      │  Chart.js Analytics │
  └─────────┬────────────┘      └─────────────────────┘
            │
   ┌────────┴────────┐
   ▼                 ▼
┌────────────┐  ┌───────────────────┐
│ SQLite DB  │  │ Auction Engine    │
│ (SQLAlchemy│  │ ┌───────────────┐ │
│  ORM)      │◀─│ │ ILP Solver    │ │
│            │  │ │ (PuLP / CBC)  │ │
│ Tables:    │  │ └───────────────┘ │
│ - agents   │  │ ┌───────────────┐ │
│ - resources│  │ │ VCG Pricing   │ │
│ - auctions │  │ │ + Tier Disc.  │ │
│ - bids     │  │ └───────────────┘ │
│ - allocs   │  │ ┌───────────────┐ │
│            │  │ │ Scheduler     │ │
│            │  │ │ (10s rounds)  │ │
│            │  │ └───────────────┘ │
│            │  │ ┌───────────────┐ │
│            │  │ │ Simulation    │ │
│            │  │ │ (5 bot agents)│ │
│            │  │ └───────────────┘ │
└────────────┘  └───────────────────┘
```

---

## 2. Component Map

### 2.1 Auction Engine (`backend/app/auction/engine.py`)
| Element | Description |
|:--|:--|
| `AuctionEngine` class | Central orchestrator — accepts bids, runs solver, returns `AllocationResult` |
| `solve_wdp_lp()` | ILP solver using PuLP/CBC. Binary variable per bid, resource-clash constraints. |
| `solve_greedy()` | Week 1 greedy fallback — sort by value-per-resource, pick first non-conflicting. |
| `compute_vcg_prices()` | For each winner, re-solves ILP without that bidder to compute externality payment. |
| `SolverMethod` enum | `LP` (exact) or `GREEDY` (approximate) |
| Data classes | `Resource`, `Bid`, `AllocationResult` |

### 2.2 Pricing (`backend/app/auction/pricing.py`)
| Element | Description |
|:--|:--|
| `apply_tier_pricing()` | Applies tier discounts to VCG payments (Gold 10%, Silver 5%, Bronze 0%). |
| `check_budget_feasibility()` | Filters bids exceeding agent's remaining budget. |
| `format_payment_report()` | CLI-friendly text report of payments and savings. |
| Reserve prices | Optional per-resource floor prices (enforced if configured). |

### 2.3 Scheduler (`backend/app/auction/scheduler.py`)
| Element | Description |
|:--|:--|
| `run_auction_round()` | Single round lifecycle: close round → fetch bids → run ILP → compute VCG → save allocations → deduct budgets → refill low budgets → broadcast results → open next round. |
| `_scheduler_loop()` | Background `asyncio` task running rounds every 10 seconds. |
| `start_scheduler()` / `stop_scheduler()` | Lifecycle controls called from FastAPI lifespan. |
| Budget refill | Agents below $150 get topped up by $300 (simulation continuity). |

### 2.4 Simulation (`backend/app/auction/simulation.py`)
| Element | Description |
|:--|:--|
| 5 simulated agents | Alpha (ML-adaptive), Beta (truthful), Gamma (shading 70%), Delta (critical 95%), Epsilon (bargain 40%). |
| `generate_simulated_bids()` | Random bundles (1-3 resources), strategy-specific bid amounts, 85% participation rate per round. |
| `WinRateTracker` | Sliding window (last 10 rounds) win rate for ML agent bid adjustment. |

### 2.5 Database (`backend/app/db/models.py`)
| Table | Key Fields |
|:--|:--|
| `agents` | name, api_key, budget_total, budget_remaining, tier (gold/silver/bronze), is_active |
| `resources` | name, resource_type, capacity, unit, cost_per_unit, is_available |
| `auctions` | round_number, status (open/closed), total_value_extracted |
| `bids` | auction_id, agent_id, resource_bundle (JSON), bid_amount, is_winner, payment_amount |
| `allocations` | auction_id, agent_id, bid_id, resources_allocated, payment_amount |

Storage: SQLite (file-based), ORM: SQLAlchemy 2.0, session via `SessionLocal`.

### 2.6 API Layer (`backend/app/api/`)
| Route | Method | Auth | Purpose |
|:--|:--|:--|:--|
| `/api/v1/agents/register` | POST | None | Register agent, returns api_key |
| `/api/v1/auth/token` | POST | API Key | Exchange api_key for JWT |
| `/api/v1/agents/me` | GET | JWT / API Key | Agent profile & budget |
| `/api/v1/bids/submit` | POST | JWT / API Key | Submit combinatorial bid |
| `/api/v1/resources` | GET | None | List available resources |
| `/api/v1/auctions/current` | GET | None | Current/latest auction round |
| `/api/v1/simulation/start` | POST | None | Start background simulation |
| `/api/v1/simulation/stop` | POST | None | Stop background simulation |
| `/api/v1/simulation/status` | GET | None | Simulation and agent status |
| `/api/v1/analytics/comparison` | GET | None | Bazaar vs Round-Robin analytics |
| `/ws/auction-feed` | WebSocket | None | Real-time bid & round events |

### 2.7 Python SDK (`compute_bazaar/`)
| Module | Purpose |
|:--|:--|
| `client/sdk.py` | `ComputeBazaarClient` — HTTP client using stdlib `urllib` (zero-dependency). |
| `models/bundle.py` | `ResourceItem`, `CombinatorialBid`, `AllocationResult`, `AuctionRoundOutcome` data classes. |
| `generators/workload.py` | `generate_synthetic_bid()`, `generate_workload_batch()` for realistic test data. |
| `autobid/strategies.py` | ML bidding strategies: `TruthfulStrategy`, `BudgetAwareShadingStrategy`, `UrgentPriorityStrategy`, `AdaptiveQLearner`, `ContextualBanditStrategy`. |

### 2.8 Dashboard (`backend/app/static/`)
- `index.html` — Glassmorphism UI with live bid ticker, resource heatmap, analytics charts.
- `app.js` — WebSocket client, Chart.js rendering, simulation controls.
- `chart.umd.js`, `swagger-ui-bundle.js`, `swagger-ui.css` — Vendored static assets.

---

## 3. Data Flow — Single Auction Round

```
1. Round N opens (status="open")
        │
2. Agents submit bids via POST /api/v1/bids/submit
   (Or simulation loop auto-generates bids every 2.5s)
        │
3. Scheduler timer fires (10s)
        │
4. run_auction_round(db):
   ├── Close Round N (status="closed")
   ├── Fetch all bids for Round N
   ├── Convert DB Bids → Engine Bids
   ├── AuctionEngine.run_auction():
   │   ├── solve_wdp_lp() → optimal winners
   │   ├── solve_greedy() → greedy benchmark
   │   └── compute_vcg_prices() → externality payments
   ├── apply_tier_pricing() → Gold/Silver discounts
   ├── Save winners, payments, allocations to DB
   ├── Deduct agent budgets
   ├── Refill low budgets (simulation)
   ├── Broadcast "round_closed" via WebSocket
   ├── Open Round N+1 (status="open")
   └── Broadcast "new_round_opened" via WebSocket
```

---

## 4. Authentication Flow

```
1. Agent registers → POST /api/v1/agents/register
   Returns: { name, api_key, tier, budget, ... }

2. Option A: Use api_key directly
   Header: X-API-Key: cb_<hex>

3. Option B: Exchange for JWT
   POST /api/v1/auth/token?api_key=cb_<hex>
   Returns: { access_token, token_type: "bearer" }
   Header: Authorization: Bearer <jwt>
```

JWT uses HS256, 24-hour expiry, SECRET_KEY from env or hardcoded default.

---

## 5. Key Design Decisions

| Decision | Choice | Rationale |
|:--|:--|:--|
| Solver | PuLP (CBC via PULP_CBC_CMD) | Free, open-source, handles 10k+ variables |
| DB | SQLite (file-based) | Zero-config for development; PostgreSQL via docker-compose for prod |
| Auth | Dual JWT + API Key | Flexibility for both long-lived agent keys and short-lived tokens |
| SDK HTTP | stdlib `urllib` | Zero external dependencies for the client library |
| Pricing | VCG + tier discounts | Truthful mechanism with loyalty rewards |
| Simulation | Background asyncio tasks | Non-blocking; smooth 2.5s bid cadence |
| Dashboard | Static HTML/JS served by FastAPI | No separate frontend build step |

---

## 6. Known Limitations (1.0)

1. **No LLM integration** — agents use hardcoded strategies, not reasoning.
2. **No persistent agent memory** — simulation agents reset on server restart.
3. **No multi-round strategic planning** — agents bid round-by-round without history.
4. **No fairness metrics** — no Jain's index or Gini coefficient computed.
5. **SQLite in production** — single-writer lock; not suitable for high concurrency.
6. **Budget refill is artificial** — agents auto-refill below $150 for simulation continuity.
7. **No rate limiting enforced** — `RATE_LIMIT_REQUESTS_PER_MINUTE` is in .env but not wired.
8. **No setup.py/pyproject.toml** — `pip install -e .` won't work without it.
