# 🏪 Compute Bazaar

> **Combinatorial Auction Theory for AI Compute Resources**  
> *Scarce resources go to the highest-value task, not the first-in-queue one.*

---

## 🧠 What Is This?

Compute Bazaar is an open-source platform that uses **combinatorial auction theory** to allocate shared compute and API budget among competing AI agents. Instead of round-robin scheduling (first-come, first-served), agents **bid** for resource bundles, and an auction engine determines the optimal allocation that maximizes total system value.

**Key Benefits:**
- 💰 **Cost Efficiency** — Eliminate wasted compute; resources go where they create the most value
- ⚡ **Speed for Critical Tasks** — High-priority tasks win the auction and execute immediately
- 🤖 **Multi-Agent Fairness** — Value-aware allocation, not just a queue
- 📊 **Transparency** — Every allocation decision is logged and explainable

---

## 🗓️ Development Status

| Week | Feature | Status |
|------|---------|--------|
| 1 | Project Setup + Demo Auction | ✅ Complete |
| 2 | WDP Solver + VCG Pricing | 🔜 Next |
| 3 | FastAPI Backend | ⏳ Planned |
| 4 | React Dashboard | ⏳ Planned |
| 5 | Analytics + Simulation | ⏳ Planned |
| 6 | ML Auto-Bidding | ⏳ Planned |
| 7 | Production (Docker + PostgreSQL) | ⏳ Planned |
| 8 | Deploy + Launch | ⏳ Planned |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 20+ (needed from Week 4)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/compute-bazaar
cd compute-bazaar

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Run the Week 1 demo
python week1_demo.py

# 5. Set up the database
python backend/app/db/database.py
```

---

## 📁 Project Structure

```
compute-bazaar/
├── week1_demo.py          # ← Start here! Simple auction demo
├── backend/
│   ├── app/
│   │   ├── auction/       # Core auction engine (WDP + VCG)
│   │   ├── api/           # FastAPI HTTP endpoints
│   │   ├── models/        # Data schemas
│   │   └── db/            # Database setup
│   ├── tests/             # Unit + integration tests
│   └── requirements.txt
├── frontend/              # React dashboard (Week 4+)
├── docs/                  # Architecture + API docs
└── docker-compose.yml     # Production setup (Week 7+)
```

---

## 🎓 How It Works

### 1. Agents Register & Get Budget
Each AI agent registers with the system and receives a compute budget (e.g., $1000/day).

### 2. Auction Round Opens
Every 60 seconds, a new auction round opens. Agents can submit bids.

### 3. Agents Submit Combinatorial Bids
```json
{
  "agent": "fraud-detector",
  "bundle": ["GPU-Slot-1", "API-Budget-100"],
  "bid_amount": 85.00,
  "task": "Real-time fraud detection on transaction batch"
}
```

### 4. Winner Determination (WDP)
The engine solves an optimization problem: *which combination of bids maximizes total value without double-allocating any resource?*

### 5. VCG Pricing
Winners pay their **externality cost** — the value they displaced from other agents. This makes truthful bidding the dominant strategy (agents can't game the system by lying about their valuations).

### 6. Resources Allocated
Winners get their resources and execute their tasks. Losers wait for the next round or increase their bids.

---

## 📖 Learn More

- [Auction Theory Basics](docs/auction-theory.md)
- [API Reference](docs/api-reference.md)
- [System Architecture](docs/architecture.md)

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.
