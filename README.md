# Compute Bazaar 2.0: Autonomous Multi-Agent Combinatorial Compute Market

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![PuLP CBC](https://img.shields.io/badge/Optimization-PuLP%20%2F%20CBC-orange.svg)](https://github.com/coin-or/pulp)
[![Tests Passing](https://img.shields.io/badge/tests-148%2F148%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An autonomous, market-driven compute allocation platform engineered for multi-agent artificial intelligence workloads. Compute Bazaar 2.0 formulates heterogeneous resource allocation as a **Combinatorial Winner Determination Problem (WDP)** solved via **Binary Integer Linear Programming (ILP)** with **Vickrey-Clarke-Groves (VCG) externality pricing**.

Autonomous AI agents powered by **Groq LLM reasoning**, cognitive memory (`AgentMemory`), machine learning resource forecasting (`ResourcePredictor`), and market anti-starvation guards (`FairnessTracker`) bid truthfully on atomic resource bundles (GPUs, TPUs, API budgets, RAM, CPU cores) without suffering from the exposure problem or queue-based blocking.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
  - [1. 4 Autonomous AI Agent Classes](#1-4-autonomous-ai-agent-classes)
  - [2. Groq LLM Cognitive Reasoning Layer](#2-groq-llm-cognitive-reasoning-layer)
  - [3. Combinatorial Auction & ILP Solver](#3-combinatorial-auction--ilp-solver)
  - [4. VCG Clarke Pivot Pricing & Tier Discounts](#4-vcg-clarke-pivot-pricing--tier-discounts)
  - [5. Intelligent & Adaptive Bidding](#5-intelligent--adaptive-bidding)
  - [6. Dynamic Marketplace & Scarcity Pricing](#6-dynamic-marketplace--scarcity-pricing)
  - [7. Cognitive Agent Memory & Jain Fairness](#7-cognitive-agent-memory--jain-fairness)
  - [8. ML Resource Prediction & Safe Toolset](#8-ml-resource-prediction--safe-toolset)
  - [9. Background Simulation Loop](#9-background-simulation-loop)
  - [10. Real-Time Glassmorphism Dashboard](#10-real-time-glassmorphism-dashboard)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [Project Structure](#project-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Example Workflow](#example-workflow)
- [Running Tests](#running-tests)
- [Future Scope](#future-scope)
- [License](#license)

---

## Overview

Modern machine learning workloads are inherently heterogeneous and co-dependent: training foundation models, executing distributed ETL pipelines, and running Monte Carlo simulations require synchronized bundles of GPU acceleration, high-capacity RAM, CPU cores, and external LLM API tokens.

Traditional schedulers treat resources as isolated silos or dispatch jobs via value-agnostic queues (First-Come First-Served or Round-Robin). Compute Bazaar 2.0 models compute allocation as an expressively combinatorial market:

1. **Agents express bundle requirements atomically**: Bids succeed or fail as complete packages ($S_i \subseteq R$), preventing stranded partial allocations (the exposure problem).
2. **Global welfare is maximized**: A binary integer program maximizes system value subject to physical capacity limits.
3. **Truthfulness is economically enforced**: VCG Clarke pivot pricing charges winners only the opportunity cost (externality) imposed on competitors, making truthful bidding a dominant strategy.
4. **Autonomous operation**: AI agents independently assess natural language task prompts, forecast hardware needs using regression models, construct bids, and adapt bidding strategies based on memory and market fairness telemetry.

---

## Problem Statement

Traditional cloud scheduling and spot resource mechanisms exhibit four major flaws:

1. **The Exposure Problem**: Standard spot markets auction compute units independently. An agent needing 2 GPUs and 32GB RAM might win only the RAM, paying for resources it cannot utilize.
2. **Head-of-Line Blocking**: In First-Come, First-Served (FCFS) schedulers (e.g., Slurm, default Kubernetes), an early low-priority batch script blocks high-value, latency-sensitive inference workloads.
3. **Value Destruction in Round-Robin**: Turn-taking queues allocate resources equally across agents regardless of true utility, forfeiting up to 38% of potential social welfare.
4. **Tactical Bidding Overhead**: Pay-as-bid auctions force agents to shade bids speculatively, resulting in market deadweight loss and complex counter-agent modeling.

---

## System Architecture

```
                      ┌─────────────────────────────────────────┐
                      │      Autonomous Multi-Agent Layer       │
                      │  Coding, Research, Data, Analysis Bots  │
                      │  - Groq LLM Inference (Reasoning)       │
                      │  - Cognitive AgentMemory (Loss streaks) │
                      │  - ML ResourcePredictor (Forecasting)   │
                      │  - Safe ToolRegistry (Execution loops)  │
                      └────────────────────┬────────────────────┘
                                           │ Combinatorial Bids
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │       FastAPI Gateway & Security        │
                      │  - Dual JWT / API Key Authentication    │
                      │  - Pydantic v2 Request Validation       │
                      │  - WebSockets Broadcast Feed            │
                      └──────────────┬───────────────────┬──────┘
                                     │                   │
                        ┌────────────┴─────┐       ┌─────┴────────────┐
                        ▼                  ▼       ▼                  ▼
              ┌──────────────────┐  ┌──────────────────────────────────┐
              │ SQLite / Postgre │  │       Auction Engine Core        │
              │ (SQLAlchemy ORM) │  │  - PuLP / CBC Binary ILP Solver  │
              │ Tables:          │  │  - VCG Clarke Pivot Pricing      │
              │ - agents         │  │  - Loyalty Tier Discounts        │
              │ - resources      │  │  - 10-Second Epoch Scheduler     │
              │ - auctions       │  │  - Background Simulation Worker  │
              │ - bids / allocs  │  │  - DynamicMarketplace Engine     │
              └──────────────────┘  └──────────────────┬───────────────┘
                                                       │ Live Events
                                                       ▼
                                    ┌──────────────────────────────────┐
                                    │    Glassmorphism Dashboard UI   │
                                    │  - Real-time Bid Ticker          │
                                    │  - Resource Heatmap & Gauges     │
                                    │  - Chart.js Allocation Analytics │
                                    └──────────────────────────────────┘
```

---

## Core Features

### 1. 4 Autonomous AI Agent Classes
Located in `compute_bazaar/agents/`, four domain-specialized agents subclass `BaseAgent`:
- **CodingAgent**: Specializes in code generation, model fine-tuning, and software compilation (`GPU-Slot-1`, `CPU-Cores-8`).
- **ResearchAgent**: Specializes in literature synthesis, arXiv search, and dense embedding retrieval (`API-Budget-100`, `RAM-Block-32GB`).
- **DataAgent**: Specializes in distributed ETL pipelines, Apache Spark batch jobs, and dataset cleaning (`RAM-Block-32GB`, `CPU-Cores-8`).
- **AnalysisAgent**: Specializes in quantitative option pricing, risk modeling, and Monte Carlo simulations (`GPU-Slot-2`, `CPU-Cores-8`).

### 2. Groq LLM Cognitive Reasoning Layer
Located in `compute_bazaar/llm/`:
- Integrates Groq's high-speed LPU inference engine (supporting `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`).
- Agents pass natural language task descriptions to the LLM to dynamically generate task complexity valuations, priority tags, and resource requirement breakdowns.
- Built-in **Mock Mode** (`LLMConfig(mock_mode=True)`) enables 100% offline, deterministic testing with full token and latency telemetry tracking.

### 3. Combinatorial Auction & ILP Solver
Located in `backend/app/auction/engine.py`:
- Formulates the Winner Determination Problem (WDP) as a **Binary Integer Linear Program**:
  $$\max \sum_{i \in B} v_i \cdot x_i \quad \text{subject to} \quad \sum_{i: r \in S_i} x_i \leq 1 \quad \forall r \in R, \quad x_i \in \{0, 1\}$$
- Solved to mathematical optimality using the Coin-or Branch and Cut (`CBC`) solver via `PuLP`.
- Guarantees zero resource double-allocation while resolving multi-resource bundle clashes.

### 4. VCG Clarke Pivot Pricing & Tier Discounts
Located in `backend/app/auction/engine.py` and `pricing.py`:
- Computes the exact externality each winner imposes on the rest of the market:
  $$p_i = V^*(B \setminus \{b_i\}) - \left[V^*(B) - v_i\right]$$
- Guarantees **Dominant-Strategy Incentive Compatibility (DSIC)**: truthful valuation reporting is mathematically dominant ($v_i = \text{true value}$).
- Incorporates configurable customer loyalty tier discounts on top of raw VCG prices: **Gold (10% off)**, **Silver (5% off)**, **Bronze (0% off)**.

### 5. Intelligent & Adaptive Bidding
Located in `compute_bazaar/agents/base.py`:
- **Truthful Bidding**: Default strategy aligned with VCG market efficiency.
- **Priority-Based Shading**: Supports strategic operational constraints: Critical ($0.95 \times v$), High ($0.85 \times v$), Normal ($0.70 \times v$), Low ($0.50 \times v$).
- **Adaptive Memory Feedback**: When agents suffer consecutive losses, bidding algorithms dynamically escalate aggressiveness toward historical resource clearing prices.

### 6. Dynamic Marketplace & Scarcity Pricing
Located in `compute_bazaar/marketplace/`:
- `DynamicMarketplace` aggregates multiple `ResourceProvider` instances.
- Tracks real-time resource demand ratios ($\text{Demand} / \text{Capacity}$).
- Dynamically scales reserve price floors during peak congestion while resetting to baseline rates when demand subsides.

### 7. Cognitive Agent Memory & Jain Fairness
Located in `compute_bazaar/agents/memory.py` and `compute_bazaar/fairness/`:
- **`AgentMemory`**: Tracks historical win rates over sliding windows, loss streaks, won resource bundles, and rolling clearing price records.
- **`jains_fairness_index`**: Evaluates system equity across participating agents:
  $$J(x) = \frac{\left(\sum_{i=1}^n x_i\right)^2}{n \sum_{i=1}^n x_i^2} \in \left[\frac{1}{n}, 1.0\right]$$
- **`FairnessTracker`**: Calculates normalized starvation risk scores ($S_r \in [0, 1]$), detects resource domination (>65% win rates), and recommends priority boosts for starved participants.

### 8. ML Resource Prediction & Safe Toolset
Located in `compute_bazaar/prediction/` and `compute_bazaar/tools/`:
- **`ResourcePredictor`**: Supervised regression model forecasting required GPU slots, CPU cores, RAM gigabytes, execution time, and dollar cost from task descriptions. Updates confidence from 50% (cold start) to 85%+ as execution records accumulate.
- **`ToolRegistry`**: Provides safe execution primitives:
  - `check_resources`: Queries live capacity and dynamic reserve prices.
  - `check_budget`: Queries agent remaining budget, spend, and starvation state.
  - `estimate_cost`: Invokes `ResourcePredictor` for capacity forecasting.
  - `execute_task`: Safely executes tasks on awarded resources and feeds actual runtime metrics back into predictor and memory.

### 9. Background Simulation Loop
Located in `backend/app/auction/simulation.py`:
- Automated simulation worker generating realistic multi-agent bidding traffic across 5 diverse personas (ML-Adaptive, Truthful, Shading, Critical, Bargain).
- Supports runtime controls via REST API (`/api/v1/simulation/start`, `/stop`, `/status`).

### 10. Real-Time Glassmorphism Dashboard
Located in `backend/app/static/`:
- Modern dark-mode UI with glassmorphism styling.
- Live WebSocket event ticker streaming bid submissions, auction round transitions, and winner announcements.
- Interactive Chart.js analytics tracking social welfare comparisons, clearing price history, and resource utilization gauges.

---

## Empirical Benchmark Results

A comprehensive benchmark was conducted across **25 reproducible rounds** with fixed seed `SEED = 42`, evaluating **ILP + VCG** against three traditional baseline schedulers (**Round-Robin**, **FCFS**, and **Random**) under the exact same multi-agent workload and resource environment.

Raw metrics are archived in [`results/benchmark_results.json`](results/benchmark_results.json); detailed analysis is documented in [`docs/benchmark.md`](docs/benchmark.md).

### 25-Round Performance Comparison

| Performance Metric | ILP + VCG (Compute Bazaar 2.0) | Round-Robin (Cyclic Priority) | FCFS (Queue-Based Arrival) | Random (Serial Dispatch) |
|:---|:---:|:---:|:---:|:---:|
| **Total Social Welfare ($)** | **$7,647.39** | $5,536.69 | $6,405.35 | $5,338.80 |
| **Mean Welfare / Round ($)** | **$305.90** | $221.47 | $256.21 | $213.55 |
| **Resource Utilization (%)** | **87.2%** | 80.8% | 80.0% | 76.8% |
| **Jain's Fairness Index ($J$)** | **0.6642** | 0.9204 | 0.4179 | 0.7733 |
| **Total Agent Cost / Payments ($)** | **$5,007.60** | $5,536.69 | $6,405.35 | $5,338.80 |
| **Agent VCG Surplus Savings ($)** | **$2,639.79 (34.5%)** | $0.00 (0.0%) | $0.00 (0.0%) | $0.00 (0.0%) |
| **Total Winners Count** | **56** | 59 | 58 | 53 |
| **Average Clearing Price ($)** | **$89.42** | $93.84 | $110.44 | $100.73 |
| **Mean Solver Latency (ms)** | **127.481 ms** | 0.031 ms | 0.007 ms | 0.018 ms |

### Key Findings
1. **Welfare Superiority**: ILP + VCG achieved **$7,647.39** in social welfare, outperforming Round-Robin by **+38.12%**, FCFS by **+19.39%**, and Random by **+43.24%**.
2. **Economic Surplus**: Under VCG pricing, winning agents received **$2,639.79 (34.5%)** in Clarke pivot price discounts relative to their truthful valuations. All pay-as-bid baselines extracted 100% of bids with zero agent surplus.
3. **FCFS Inequality**: FCFS suffered from severe head-of-line blocking ($J = 0.4179$), allowing early-arriving agents to monopolize capacity regardless of task value.
4. **Real-Time Latency**: The ILP solver resolved combinatorial bundle constraints and all counterfactual VCG subproblems in **127.48 ms** on average, well within real-time thresholds for periodic cluster scheduling epochs (10s).

---

## Project Structure

```
compute-bazaar/
├── backend/
│   ├── app/
│   │   ├── api/             # FastAPI REST endpoints & WebSocket handlers
│   │   │   ├── endpoints.py # /api/v1/ routes (bids, resources, agents)
│   │   │   └── websocket.py # Real-time WebSocket pub/sub connection manager
│   │   ├── auction/         # Optimization, pricing & simulation logic
│   │   │   ├── engine.py    # AuctionEngine, ILP solver (PuLP), VCG computation
│   │   │   ├── pricing.py   # Loyalty tier discounts & budget checking
│   │   │   ├── scheduler.py # 10-second epoch background worker
│   │   │   └── simulation.py# Multi-agent automated simulation loop
│   │   ├── db/              # SQLAlchemy models & database migrations
│   │   │   └── models.py    # Agent, Resource, Auction, Bid, Allocation tables
│   │   ├── static/          # Glassmorphism HTML/JS Dashboard & vendored assets
│   │   │   ├── index.html   # Main dashboard layout
│   │   │   └── app.js       # WebSocket feed, chart rendering, UI controls
│   │   └── main.py          # FastAPI application entrypoint & lifespan
│   └── requirements.txt     # Backend Python package dependencies
├── compute_bazaar/          # Client SDK & Agent Intelligence Package
│   ├── agents/              # Autonomous AI agent implementations
│   │   ├── base.py          # BaseAgent with adaptive bidding & tool loops
│   │   ├── coding.py        # CodingAgent (Software/Training)
│   │   ├── research.py      # ResearchAgent (Literature/Embeddings)
│   │   ├── data.py          # DataAgent (ETL/Spark)
│   │   ├── analysis.py      # AnalysisAgent (Quantitative/Monte Carlo)
│   │   └── memory.py        # Cognitive AgentMemory & BidOutcome tracking
│   ├── fairness/            # Fairness metrics & anti-starvation
│   │   └── metrics.py       # Jain's index, starvation risk, FairnessTracker
│   ├── llm/                 # Groq LLM integration layer
│   │   └── client.py        # GroqLLMClient, LLMConfig, telemetry tracking
│   ├── marketplace/         # Dynamic marketplace & provider registry
│   │   └── market.py        # DynamicMarketplace, ResourceProvider, scarcity pricing
│   ├── prediction/          # Machine learning resource forecasting
│   │   └── predictor.py     # ResourcePredictor with feedback training loop
│   └── tools/               # Safe toolset for agent operations
│       └── agent_tools.py   # check_resources, check_budget, estimate_cost, execute_task
├── docs/                    # Architecture guides, benchmarks & final report
│   ├── benchmark.md         # Comprehensive empirical benchmark report
│   ├── report.md            # Week 13 18-section final project report
│   └── ...                  # Component deep-dive guides
├── results/                 # Raw benchmark datasets
│   └── benchmark_results.json # Full per-round metrics from 25 benchmark rounds
├── tests/                   # Comprehensive automated test suite
│   ├── test_week2_engine.py # Solver & VCG unit tests
│   ├── test_week3_agents.py # Autonomous agent behavioral tests
│   ├── test_week4_bidding.py# Combinatorial bidding tests
│   ├── test_week5_marketplace.py # Scarcity pricing tests
│   ├── test_week6_fairness_memory.py # Fairness tracker & memory tests
│   ├── test_week7_prediction_tools.py # ML predictor & tool tests
│   ├── test_week8a_dashboard.py # FastAPI endpoints & UI tests
│   └── test_week8b_evaluation.py # Empirical evaluation tests
├── benchmark.py             # Reproducible 25-round benchmark script (SEED=42)
├── conftest.py              # Pytest configuration & fixtures
└── README.md                # Project documentation
```

---

## Installation & Quickstart

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/MuzraFatima/Compute-bazar.git
cd Compute-bazar

# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Environment Variables (Optional)
Copy the example configuration file:
```bash
cp .env.example .env
```
To enable live Groq LLM reasoning (offline mock mode is enabled by default for tests and benchmarks):
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```

### 4. Run the FastAPI Application Server
```bash
uvicorn backend.app.main:app --reload
```
The server will initialize the SQLite database (`test_compute_bazaar.db`), start the background auction scheduler loop (10-second epochs), and serve the dashboard:
- **Web Dashboard**: `http://localhost:8000`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`
- **WebSocket Feed**: `ws://localhost:8000/ws/auction-feed`

---

## Example Workflow

The following Python snippet demonstrates how an autonomous agent inspects market conditions, uses machine learning to forecast resource needs, constructs a truthful combinatorial bid, submits it to the engine, and executes the awarded task:

```python
from compute_bazaar.agents import CodingAgent
from compute_bazaar.llm import GroqLLMClient, LLMConfig
from compute_bazaar.tools import check_resources, estimate_cost, execute_task
from backend.app.auction.engine import AuctionEngine, Resource, SolverMethod

# 1. Initialize Infrastructure & Engine
resources = [
    Resource("GPU-Slot-1", "gpu", 1.0, "slot"),
    Resource("RAM-Block-32GB", "ram", 32.0, "gb"),
    Resource("CPU-Cores-8", "cpu", 8.0, "cores"),
]
engine = AuctionEngine(resources, method=SolverMethod.LP)

# 2. Initialize Autonomous Agent with Groq LLM Reasoning
llm = GroqLLMClient(config=LLMConfig(mock_mode=True))
agent = CodingAgent("AutoDevBot", budget=500.0, llm_client=llm)

# 3. Agent checks market availability & forecasts task needs
task_prompt = "Fine-tune 7B parameter code completion model"
prediction = estimate_cost(task_prompt)
print(f"ML Forecast: {prediction.predicted_execution_time_sec}s runtime | ${prediction.predicted_cost:.2f} cost")

# 4. Agent formulates combinatorial bundle bid and submits to engine
bid = agent.create_bid(task_prompt)
bid.resource_bundle = ["GPU-Slot-1", "CPU-Cores-8"]
bid.bid_amount = 95.0
agent.submit_to_engine(engine, bid)

# 5. Engine runs auction round (ILP solve + VCG externality pricing)
result = engine.run_auction(round_id=1)

# 6. Winner execution feedback loop
for winner in result.winning_bids:
    if winner.agent_id == agent.name:
        vcg_payment = result.payments[winner.bid_id]
        print(f"Won! Bid: ${winner.bid_amount:.2f} | VCG Payment: ${vcg_payment:.2f} | Surplus: ${winner.bid_amount - vcg_payment:.2f}")
        execute_task(agent, winner.task_description, {"won": True, "payment": vcg_payment, "resources": winner.resource_bundle})
```

---

## Running Tests

The test suite provides comprehensive verification across auction mathematics, VCG payments, multi-agent cognition, dynamic pricing, and API endpoints:

```bash
# Run all 148 mock-mode automated tests
python -m pytest tests/ --ignore=tests/test_week11_live_integration.py -v
```

### Test Suite Summary
```
tests/test_week2_engine.py ......................... PASSED
tests/test_week2_llm.py ............ PASSED
tests/test_week3_8.py .......... PASSED
tests/test_week3_agents.py ............... PASSED
tests/test_week4_bidding.py ......... PASSED
tests/test_week5_marketplace.py ............ PASSED
tests/test_week6_fairness_memory.py ............ PASSED
tests/test_week7_prediction_tools.py .......... PASSED
tests/test_week8a_dashboard.py .. PASSED
tests/test_week8b_evaluation.py ... PASSED
====================== 148 passed in 8.44s ======================
```

### Re-running the 25-Round Empirical Benchmark
To regenerate [`results/benchmark_results.json`](results/benchmark_results.json) and [`docs/benchmark.md`](docs/benchmark.md):
```bash
python benchmark.py
```

---

## Future Scope

1. **GPU-Accelerated Optimization**: Integrate cuPULP / HiGHS branch-and-price solvers to scale combinatorial winner determination to 10,000+ bids per second.
2. **Iterative Dynamic Auctions (iBundle / Ascending Clock)**: Implement ascending proxy clock auctions to reduce agent valuation elicitation overhead across multi-hour tasks.
3. **Smart Contract Settlement**: Deploy auction escrow, SLA enforcement, and VCG payments to Ethereum Layer-2 rollups (Arbitrum, Base) for trustless multi-tenant compute federations.
4. **Autonomous Reinforcement Learning Market Makers**: Deploy RL-driven liquidity agents that dynamically provision spot capacity to stabilize prices across federated clouds.
5. **Zero-Knowledge Valuation Proofs**: Integrate zk-SNARKs allowing agents to cryptographically verify budget feasibility and valuation bounds without revealing proprietary task data.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
