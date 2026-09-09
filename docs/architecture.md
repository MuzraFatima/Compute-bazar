# Compute Bazaar — Architecture & System Design

This document details the architectural blueprint, optimization mathematics, auction mechanics, and runtime components of **Compute Bazaar**.

---

## 1. High-Level Architecture

Compute Bazaar is designed as a high-throughput, low-latency market infrastructure connecting competing AI agents with scarce compute resources (GPUs, TPUs, API budgets, RAM, CPU cores).

```
                      ┌─────────────────────────────────┐
                      │    AI Agents / Autonomous Bots  │
                      │ (Python SDK / HTTP / REST APIs) │
                      └───────────────┬─────────────────┘
                                      │  Bids / Auth
                                      ▼
                      ┌─────────────────────────────────┐
                      │    FastAPI Application Gateway  │
                      │  - JWT / API Key Authentication │
                      │  - Rate Limiting & Validation   │
                      │  - WebSockets Broadcast Feed    │
                      └───────────────┬─────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
┌───────────────────────┐                           ┌───────────────────────┐
│   Database Layer      │                           │  Auction Engine Loop  │
│ - SQLite / PostgreSQL │                           │ - Scheduler Worker    │
│ - Agents, Bids,       │                           │ - ILP Solver (PuLP)   │
│   Rounds, Allocations │                           │ - VCG Pricing Engine  │
└───────────────────────┘                           └───────────┬───────────┘
                                                                │
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │ Real-Time Dashboard   │
                                                    │ - Glassmorphism UI    │
                                                    │ - Heatmaps & Ticker   │
                                                    │ - Chart.js Analytics  │
                                                    └───────────────────────┘
```

---

## 2. Core Components

### 2.1 API & Web Gateway (`backend/app/api/`)
- **FastAPI Engine**: Asynchronous HTTP handling with OpenAPI schema generation.
- **Authentication (`auth.py`)**: Dual-mode security supporting long-lived **API Keys** (`X-API-Key`) and stateless **Bearer JWT Tokens**.
- **WebSocket Manager (`websocket.py`)**: Pub/Sub real-time broadcast engine streaming bid events, round closures, allocations, and market metrics directly to browser dashboards and agent listeners.

### 2.2 Auction Engine & Optimization (`backend/app/auction/`)
- **Winner Determination Problem (WDP)**: Formulated as a Binary Integer Linear Program (ILP):

$$\text{Maximize } \sum_{i \in B} v_i \cdot x_i$$

$$\text{Subject to } \sum_{i: r \in S_i} x_i \leq 1 \quad \forall r \in R$$

$$x_i \in \{0, 1\} \quad \forall i \in B$$

Where:
- $B$ = set of submitted combinatorial bids
- $R$ = set of scarce compute resources
- $S_i$ = bundle of resources requested by bid $i$
- $v_i$ = bid valuation
- $x_i$ = binary decision variable ($1$ if bid $i$ is awarded, $0$ otherwise)

- **VCG (Vickrey-Clarke-Groves) Pricing**:
Computes the Clarke Pivot Rule payment for each winning agent $i$:

$$p_i = V_{-i}^* - \sum_{j \neq i, j \in W} v_j$$

Where:
- $V_{-i}^*$ = maximum total value achievable by the auction *without* agent $i$'s participation.
- $\sum_{j \neq i, j \in W} v_j$ = actual value extracted by all *other* winners in the presence of agent $i$.
- $p_i$ represents the exact economic **externality** (opportunity cost) agent $i$ imposes on other participants.
- **Truthfulness Guarantee**: Under VCG, truthful bidding ($v_i = \text{true valuation}$) is a Dominant-Strategy Incentive-Compatible (DSIC) equilibrium.

### 2.3 Scheduler & Simulation Loop (`backend/app/auction/scheduler.py`, `simulation.py`)
- **Periodic Rounds**: Operates on a continuous $T$-second epoch (default 10s).
- **Background Worker**: Handles lifecycle events: round freeze $\to$ LP solve $\to$ VCG computation $\to$ budget deduction $\to$ allocation record $\to$ WebSocket push $\to$ next round initialization.
- **Multi-Agent Simulation**: Built-in automated simulation running Reinforcement Learning (Q-learning), truthful, strategic shading, and urgent bidders.

### 2.4 Client SDK (`compute_bazaar/client/sdk.py`)
- Standard library Python client enabling seamless agent integration without heavy third-party dependencies.

---

## 3. Data Flow Lifecycle

1. **Round Open**: The scheduler opens Round $N$ and emits a `new_round_opened` event via WebSocket.
2. **Bid Submission**: Autonomous agents evaluate workloads and submit combinatorial bids (`POST /api/v1/bids`).
3. **Round Freeze**: When the timer expires, Round $N$ freezes. No further bids are accepted.
4. **WDP Optimization**: PuLP/CBC solves the ILP problem to identify the non-conflicting subset of bids maximizing system utility.
5. **VCG Payment Computation**: For each winner, counterfactual LP problems are evaluated to determine externality costs and tier discounts (Gold 10%, Silver 5%).
6. **Settlement**: Agent remaining balances are debited, allocation records are committed to SQLite/PostgreSQL, and `round_closed` outcomes are broadcast via WebSockets.
7. **Round $N+1$**: Immediately triggered to sustain continuous scheduling.
