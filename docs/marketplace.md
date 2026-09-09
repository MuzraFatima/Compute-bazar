# Compute Bazaar 2.0 -- Dynamic Marketplace & Resource Providers Guide

> Week 5 deliverable. Documents the dynamic marketplace architecture, multi-provider infrastructure, resource categories (GPU, CPU, RAM, LLM/API), operational availability toggles, demand/supply tracking, and scarcity-based dynamic pricing.

---

## 1. Overview

Compute Bazaar 2.0 introduces a multi-provider dynamic marketplace (`compute_bazaar.marketplace`) that coordinates heterogeneous compute infrastructure suppliers (Groq, Lambda Labs, AWS, on-prem clusters).

```
compute_bazaar/
  marketplace/
    __init__.py      # Package exports (ResourceProvider, MarketResource, MarketResourceType, DynamicMarketplace)
    provider.py      # ResourceProvider model & MarketResource definition
    manager.py       # DynamicMarketplace manager (supply/demand tracking, scarcity pricing)
```

---

## 2. ResourceProvider & MarketResource Models

### 2.1 ResourceProvider

A `ResourceProvider` represents an infrastructure supplier listing resources in the market:

- `provider_id`: Unique identifier (e.g. `"prov-groq-cloud"`)
- `name`: Provider display name (e.g. `"Groq Cloud Infrastructure"`)
- `provider_type`: `"cloud"`, `"api_vendor"`, `"on_prem"`, `"decentralized"`
- `reliability_score`: Operational score from `0.0` to `1.0`
- `is_active`: Global active flag

### 2.2 MarketResource Categories

`MarketResource` supports multi-category compute resources:

| Resource Category | `MarketResourceType` | Typical Unit | Description |
|:--|:--|:--|:--|
| **GPU** | `gpu` | `slot` | Hardware-accelerated GPU instances (A100, H100) |
| **CPU** | `cpu` | `cores` | Multi-core compute nodes for compilation/processing |
| **RAM** | `ram` | `gb` | Memory blocks for in-memory DataFrames and embeddings |
| **API Budget** | `api_budget` | `dollars` | Monetary credits for third-party REST services |
| **LLM/API** | `llm_api` | `tokens` | Groq API token quotas and throughput slots |
| **Storage** | `storage` | `gb` | High-speed block/object storage |
| **TPU** | `tpu` | `slot` | Tensor Processing Units |

---

## 3. Dynamic Supply, Demand, & Scarcity Pricing

### 3.1 Demand & Supply Tracking

The `DynamicMarketplace` records requested resource bundles from submitted auction bids:
- **Supply**: Total and available unallocated capacity across active providers.
- **Demand**: Total requested resource units per round.

### 3.2 Scarcity Ratio Calculation

$$\text{scarcity\_ratio} = \frac{\text{round\_demand\_requested}}{\max(1.0, \text{available\_capacity})}$$

### 3.3 Dynamic Reserve Floor Pricing

When demand for a resource spikes relative to available supply, the marketplace dynamically increases its reserve/floor price above the provider's minimum price:

$$P_{\text{dynamic}} = P_{\text{min}} \times \left(1 + \alpha \times \text{scarcity\_ratio}\right)$$

When demand subsides ($\text{scarcity\_ratio} \to 0$), the price naturally returns toward the provider's minimum floor price $P_{\text{min}}$.

---

## 4. Usage Example

```python
from compute_bazaar.marketplace import (
    DynamicMarketplace,
    ResourceProvider,
    MarketResource,
    MarketResourceType,
)

# 1. Instantiate Marketplace
market = DynamicMarketplace("Production Compute Bazaar")

# 2. Register a Cloud Provider
groq_provider = ResourceProvider(
    provider_id="prov-groq",
    name="Groq Cloud Infrastructure",
    provider_type="api_vendor",
)
groq_provider.add_resource(MarketResource(
    name="Groq-Llama3-API",
    resource_type=MarketResourceType.LLM_API.value,
    total_capacity=10000.0,
    available_capacity=10000.0,
    min_price=25.0,
    unit="tokens",
))
market.register_provider(groq_provider)

# 3. Record Demand & Update Dynamic Prices
market.record_demand([["Groq-Llama3-API"], ["Groq-Llama3-API"]])
market.update_dynamic_pricing(alpha=0.5)

# 4. View Market Summary
summary = market.get_market_summary()
print(summary["resource_prices"]["Groq-Llama3-API"])
```

---

## 5. Architectural Boundaries & Safeguards

- **Existing Logic Intact**: The core `AuctionEngine` (ILP solver and VCG pricing) and Week 4 `BaseAgent` intelligent bidding logic remain 100% untouched.
- **Minimum Floor Price Enforcement**: Dynamic prices are strictly lower-bounded by $P_{\text{min}}$.
- **API Key Security**: Tests run strictly in `MOCK` mode using `LLMConfig(mock_mode=True)`. API keys are never exposed or committed.
