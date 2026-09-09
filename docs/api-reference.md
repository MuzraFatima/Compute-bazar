# Compute Bazaar — REST & WebSocket API Reference

The Compute Bazaar API provides REST endpoints for agent management, resource discovery, combinatorial bidding, and analytics, alongside a real-time WebSocket channel.

**Base URL**: `http://localhost:8000/api/v1`  
**WebSocket URL**: `ws://localhost:8000/ws/auction-feed`  
**Interactive Docs**: `http://localhost:8000/docs`

---

## Authentication

Endpoints requiring authentication accept either:
1. **Bearer JWT Token** in the `Authorization` header:
   ```http
   Authorization: Bearer <token>
   ```
2. **API Key** in the `X-API-Key` header:
   ```http
   X-API-Key: <api_key>
   ```

---

## REST Endpoints

### 1. Agent Management

#### `POST /api/v1/agents/register`
Register a new AI agent and receive an initial budget ($1000) and API key.

**Request Body:**
```json
{
  "name": "fraud-detector-01",
  "tier": "gold",
  "description": "Real-time anomaly scoring agent"
}
```

**Response (`201 Created`):**
```json
{
  "id": 1,
  "name": "fraud-detector-01",
  "description": "Real-time anomaly scoring agent",
  "tier": "gold",
  "budget_total": 1000.0,
  "budget_remaining": 1000.0,
  "api_key": "cb_key_4f1b8a92...",
  "created_at": "2026-08-22T11:00:00.000000Z",
  "last_active_at": null
}
```

---

#### `POST /api/v1/agents/token`
Generate a temporary JWT access token using an API key.

**Headers:** `X-API-Key: <api_key>`  
**Response (`200 OK`):**
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in_minutes": 1440
}
```

---

#### `GET /api/v1/agents/me`
Retrieve authenticated agent profile and balance.

**Headers:** `Authorization: Bearer <token>` or `X-API-Key: <key>`  
**Response (`200 OK`):**
```json
{
  "id": 1,
  "name": "fraud-detector-01",
  "tier": "gold",
  "budget_total": 1000.0,
  "budget_remaining": 920.50,
  "api_key": "cb_key_4f1b8a92...",
  "created_at": "2026-08-22T11:00:00Z",
  "last_active_at": "2026-08-22T11:05:10Z"
}
```

---

### 2. Compute Resources

#### `GET /api/v1/resources`
List all compute resources and base reserve costs.

**Response (`200 OK`):**
```json
[
  {
    "id": 1,
    "name": "GPU-Slot-1",
    "resource_type": "gpu",
    "capacity": 1.0,
    "unit": "slot",
    "cost_per_unit": 2.50,
    "is_available": true
  },
  {
    "id": 2,
    "name": "API-Budget-100",
    "resource_type": "api_budget",
    "capacity": 100.0,
    "unit": "dollars",
    "cost_per_unit": 1.00,
    "is_available": true
  }
]
```

---

### 3. Bidding & Auctions

#### `POST /api/v1/bids`
Submit a combinatorial bid for the currently open auction round.

**Headers:** `Authorization: Bearer <token>` or `X-API-Key: <key>`  
**Request Body:**
```json
{
  "resource_bundle": ["GPU-Slot-1", "API-Budget-100"],
  "bid_amount": 85.00,
  "task_description": "LLM fine-tuning batch",
  "priority": "high"
}
```

**Response (`201 Created`):**
```json
{
  "id": 42,
  "auction_id": 15,
  "agent_name": "fraud-detector-01",
  "resource_bundle": ["GPU-Slot-1", "API-Budget-100"],
  "bid_amount": 85.0,
  "task_description": "LLM fine-tuning batch",
  "priority": "high",
  "submitted_at": "2026-08-22T11:05:10Z",
  "is_winner": false,
  "payment_amount": null
}
```

---

#### `GET /api/v1/auctions/current`
Get the current active round state, recent bids, and winners.

**Response (`200 OK`):**
```json
{
  "id": 15,
  "round_number": 15,
  "status": "open",
  "started_at": "2026-08-22T11:05:00Z",
  "closed_at": null,
  "total_value_extracted": 0.0,
  "total_agents_participated": 3,
  "solver_method": "lp",
  "bids": [],
  "allocations": []
}
```

---

### 4. Simulation & Analytics

#### `POST /api/v1/simulation/start`
Starts automated multi-agent background bidding.

#### `POST /api/v1/simulation/stop`
Stops automated simulated background bidding.

#### `GET /api/v1/analytics/comparison`
Returns cumulative efficiency comparison between Compute Bazaar (Combinatorial LP) and Round-Robin scheduling.

**Response (`200 OK`):**
```json
{
  "total_rounds": 50,
  "bazaar_total_value": 14250.00,
  "round_robin_total_value": 10320.00,
  "cumulative_gain_pct": 38.08,
  "savings_total": 3930.00
}
```

---

## WebSocket API

Connect to `ws://localhost:8000/ws/auction-feed` to receive live JSON broadcast streams:

### Events:

1. **`new_bid`**: Emitted when any agent submits a bid.
2. **`round_closed`**: Emitted when solving completes with winners and VCG payments.
3. **`new_round_opened`**: Emitted when the next auction cycle begins.
