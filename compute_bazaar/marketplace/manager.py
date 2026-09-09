"""
=============================================================
 COMPUTE BAZAAR 2.0 -- DynamicMarketplace Manager (Week 5)
=============================================================
 Orchestrates multi-provider compute markets:
   - Provider registration & resource cataloging
   - Demand and supply tracking per resource category
   - Scarcity ratio calculation
   - Dynamic pricing based on demand/supply imbalance
   - Operational availability toggles & market diagnostics
=============================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from loguru import logger

from compute_bazaar.marketplace.provider import (
    ResourceProvider,
    MarketResource,
    MarketResourceType,
)


class DynamicMarketplace:
    """
    Central manager for dynamic compute marketplace operations.

    Features:
      - Multi-provider registration (Groq, AWS, Lambda Labs, etc.)
      - GPU, CPU, RAM, and LLM/API resource inventory
      - Demand/supply tracking per auction round
      - Scarcity-based dynamic reserve pricing
      - Dynamic availability (node failure, maintenance, recovery)
    """

    def __init__(self, name: str = "Compute Bazaar Dynamic Marketplace"):
        self.name = name
        self.providers: Dict[str, ResourceProvider] = {}
        self.resources: Dict[str, MarketResource] = {}
        
        # Demand tracking: resource_name -> total_requested_capacity
        self.demand_history: Dict[str, float] = {}
        self.round_demand_log: List[Dict[str, float]] = []
        self.alpha_pricing_factor: float = 0.5

        # Pre-seed default provider for default resources if needed
        self._seed_default_provider()

    def _seed_default_provider(self):
        """Seed a default provider with standard Compute Bazaar 1.0 resources."""
        default_prov = ResourceProvider(
            provider_id="prov-default",
            name="Compute Bazaar Standard Supply",
            provider_type="on_prem",
            reliability_score=1.0,
        )
        
        default_resources = [
            MarketResource(name="GPU-Slot-1", resource_type=MarketResourceType.GPU.value, total_capacity=1.0, min_price=30.0, current_price=30.0, unit="slot"),
            MarketResource(name="GPU-Slot-2", resource_type=MarketResourceType.GPU.value, total_capacity=1.0, min_price=30.0, current_price=30.0, unit="slot"),
            MarketResource(name="API-Budget-100", resource_type=MarketResourceType.API_BUDGET.value, total_capacity=100.0, min_price=20.0, current_price=20.0, unit="dollars"),
            MarketResource(name="RAM-Block-32GB", resource_type=MarketResourceType.RAM.value, total_capacity=32.0, min_price=15.0, current_price=15.0, unit="gb"),
            MarketResource(name="CPU-Cores-8", resource_type=MarketResourceType.CPU.value, total_capacity=8.0, min_price=10.0, current_price=10.0, unit="cores"),
            MarketResource(name="Groq-Llama3-API", resource_type=MarketResourceType.LLM_API.value, total_capacity=1000.0, min_price=25.0, current_price=25.0, unit="tokens"),
        ]

        for res in default_resources:
            default_prov.add_resource(res)
            self.resources[res.name] = res

        self.providers[default_prov.provider_id] = default_prov

    # ── Provider Management ──────────────────────────────────

    def register_provider(self, provider: ResourceProvider):
        """Register a new resource provider and catalog its resources."""
        self.providers[provider.provider_id] = provider
        for res_name, res in provider.resources.items():
            self.resources[res_name] = res
        logger.info(f"Registered provider '{provider.name}' ({provider.provider_id}) with {len(provider.resources)} resources.")

    def get_provider(self, provider_id: str) -> Optional[ResourceProvider]:
        """Fetch provider by ID."""
        return self.providers.get(provider_id)

    def list_providers(self) -> List[ResourceProvider]:
        """List all registered providers."""
        return list(self.providers.values())

    # ── Resource Management ──────────────────────────────────

    def register_resource(self, resource: MarketResource, provider_id: str = "prov-default"):
        """Register a new individual market resource."""
        provider = self.providers.get(provider_id)
        if not provider:
            provider = ResourceProvider(provider_id=provider_id, name=f"Provider {provider_id}")
            self.register_provider(provider)
        
        provider.add_resource(resource)
        self.resources[resource.name] = resource

    def get_resource(self, name: str) -> Optional[MarketResource]:
        """Fetch resource by name."""
        return self.resources.get(name)

    def list_all_resources(self) -> List[MarketResource]:
        """List all market resources across all providers."""
        return list(self.resources.values())

    def list_available_resources(self) -> List[MarketResource]:
        """List currently operational and unallocated resources."""
        return [
            r for r in self.resources.values()
            if r.is_available and r.available_capacity > 0 and self.providers.get(r.provider_id, ResourceProvider(provider_id="x", name="x")).is_active
        ]

    def set_resource_availability(self, resource_name: str, is_available: bool):
        """Toggle dynamic operational availability of a resource."""
        if resource_name in self.resources:
            res = self.resources[resource_name]
            res.is_available = is_available
            if res.provider_id in self.providers:
                self.providers[res.provider_id].set_resource_availability(resource_name, is_available)
            logger.info(f"Resource '{resource_name}' availability set to {is_available}.")

    # ── Demand & Supply Tracking ─────────────────────────────

    def record_demand(self, bid_bundles: List[List[str]]):
        """
        Record demand requested for resources in an auction round.

        Args:
            bid_bundles: List of resource bundles requested by submitted bids
                         (e.g., [["GPU-Slot-1", "RAM-Block-32GB"], ["GPU-Slot-1"]])
        """
        current_round_demand: Dict[str, float] = {}
        for bundle in bid_bundles:
            for res_name in bundle:
                current_round_demand[res_name] = current_round_demand.get(res_name, 0.0) + 1.0
                self.demand_history[res_name] = self.demand_history.get(res_name, 0.0) + 1.0

        self.round_demand_log.append(current_round_demand)

    def get_supply_summary(self) -> Dict[str, Any]:
        """Return total and available supply capacity grouped by resource category."""
        summary: Dict[str, Dict[str, float]] = {}

        for res in self.resources.values():
            cat = res.resource_type
            if cat not in summary:
                summary[cat] = {"total_capacity": 0.0, "available_capacity": 0.0}
            if res.is_available and self.providers.get(res.provider_id, ResourceProvider(provider_id="x", name="x")).is_active:
                summary[cat]["total_capacity"] += res.total_capacity
                summary[cat]["available_capacity"] += res.available_capacity

        return summary

    def calculate_scarcity(self, resource_name: str) -> float:
        """
        Calculate scarcity ratio for a given resource:
          scarcity_ratio = round_demand / max(1.0, available_capacity)
        """
        res = self.resources.get(resource_name)
        if not res or not res.is_available or res.available_capacity <= 0:
            return 2.0  # High default scarcity if unavailable

        recent_demand = 0.0
        if self.round_demand_log:
            recent_demand = self.round_demand_log[-1].get(resource_name, 0.0)
        else:
            recent_demand = self.demand_history.get(resource_name, 0.0)

        scarcity = recent_demand / max(1.0, res.available_capacity)
        return round(scarcity, 4)

    # ── Dynamic Pricing ───────────────────────────────────────

    def update_dynamic_pricing(self, alpha: Optional[float] = None):
        """
        Recalculate dynamic reserve prices across all resources based on scarcity:
          current_price = min_price * (1 + alpha * scarcity_ratio)
        """
        pricing_alpha = alpha if alpha is not None else self.alpha_pricing_factor

        for res_name, res in self.resources.items():
            scarcity = self.calculate_scarcity(res_name)
            res.update_dynamic_price(scarcity, alpha=pricing_alpha)
            logger.debug(
                f"Dynamic price updated | resource={res_name} | "
                f"scarcity={scarcity:.2f} | min_price=${res.min_price:.2f} | "
                f"current_price=${res.current_price:.2f}"
            )

    # ── Market Summary ────────────────────────────────────────

    def get_market_summary(self) -> Dict[str, Any]:
        """Return comprehensive marketplace diagnostic state."""
        return {
            "marketplace_name": self.name,
            "total_providers": len(self.providers),
            "active_providers": sum(1 for p in self.providers.values() if p.is_active),
            "total_resources": len(self.resources),
            "available_resources": len(self.list_available_resources()),
            "supply": self.get_supply_summary(),
            "resource_prices": {
                name: {
                    "type": r.resource_type,
                    "min_price": r.min_price,
                    "current_price": r.current_price,
                    "is_available": r.is_available,
                    "capacity": f"{r.available_capacity}/{r.total_capacity}",
                }
                for name, r in self.resources.items()
            },
            "recent_demand": self.round_demand_log[-1] if self.round_demand_log else {},
        }
