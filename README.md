# Compute Bazaar

An auction-based scheduling system for multi-agent compute resource allocation. AI agents bid for shared compute and API budget through a combinatorial auction, with allocation solved via integer linear programming — outperforming round-robin scheduling in resource utilization and fairness.

## Overview

Compute Bazaar models compute/API budget allocation as a market: agents submit bids for the resources they need, and an ILP solver determines the allocation that maximizes overall value while respecting capacity constraints and preventing starvation.

## Features

- **Combinatorial auction engine** — agents bid on resource bundles, not just single units
- **ILP-based solver** (via PuLP) for optimal allocation under budget/capacity constraints
- **Starvation guards** to ensure no agent is indefinitely under-served
- **REST API** built with FastAPI, backed by SQLite for persistence
- **Live dashboard** (HTML) for real-time visibility into allocations and bidding activity
- **23 automated tests** covering the auction logic, API endpoints, and edge cases

## Tech stack

| Layer | Tech |
|---|---|
| API | FastAPI |
| Data models | Pydantic |
| Optimization | PuLP (ILP solver) |
| Storage | SQLite |
| Dashboard | HTML/JS |
| Testing | pytest |

## Getting started

```bash
git clone https://github.com/MuzraFatima/Compute-bazar.git
cd Compute-bazar
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000` for the API and dashboard.

## Running tests

```bash
pytest
```

## Why this approach

Traditional round-robin scheduling treats all agents equally regardless of actual need or value of the task at hand. Compute Bazaar lets agents express how much a resource is *worth* to them, producing allocations that better reflect real priorities — while the ILP solver guarantees the allocation is provably optimal given the constraints.

## License

MIT
