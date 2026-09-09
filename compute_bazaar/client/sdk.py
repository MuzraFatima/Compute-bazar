"""
=============================================================
 COMPUTE BAZAAR — Python Client SDK (Agent Integration)
=============================================================
 Official Python client library for AI agents to connect,
 authenticate, bid for resource bundles, and query live market state.
=============================================================
"""

import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Optional, Any, Union
from compute_bazaar.models.bundle import CombinatorialBid, ResourceItem, AgentTier, PriorityLevel


class ComputeBazaarError(Exception):
    """Base exception for Compute Bazaar SDK errors."""
    pass


class ComputeBazaarClient:
    """
    Python SDK Client for Compute Bazaar.
    
    Example:
    --------
    >>> client = ComputeBazaarClient(base_url="http://127.0.0.1:8000")
    >>> agent = client.register(name="my-fraud-bot", tier="gold")
    >>> resources = client.get_resources()
    >>> bid = client.submit_bid(
    ...     bundle=["GPU-Slot-1", "API-Budget-100"],
    ...     amount=85.0,
    ...     task="Real-time transaction scoring"
    ... )
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token: Optional[str] = None
        self.agent_profile: Optional[Dict[str, Any]] = None

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Internal HTTP request dispatcher using standard library for zero external dependency runtime."""
        url = f"{self.base_url}{endpoint}"
        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ComputeBazaar-Python-SDK/0.1.0",
        }
        if self.token:
            req_headers["Authorization"] = f"Bearer {self.token}"
        elif self.api_key:
            req_headers["X-API-Key"] = self.api_key

        if headers:
            req_headers.update(headers)

        payload_bytes = None
        if data is not None:
            payload_bytes = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=payload_bytes, headers=req_headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp_bytes = resp.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
                msg = err_json.get("detail", err_body)
            except Exception:
                msg = err_body
            raise ComputeBazaarError(f"HTTP {e.code} Error on {endpoint}: {msg}")
        except urllib.error.URLError as e:
            raise ComputeBazaarError(f"Connection failed to {url}: {e.reason}")

    def register(
        self,
        name: str,
        tier: str = "bronze",
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new agent account and obtain an API key."""
        payload = {
            "name": name,
            "tier": tier,
            "description": description or f"Agent {name}",
        }
        resp = self._make_request("POST", "/api/v1/agents/register", data=payload)
        self.api_key = resp.get("api_key")
        self.agent_profile = resp
        return resp

    def get_token(self, api_key: Optional[str] = None) -> str:
        """Obtain a Bearer JWT token using an API key."""
        key = api_key or self.api_key
        if not key:
            raise ComputeBazaarError("No API key provided.")
        resp = self._make_request("POST", "/api/v1/agents/token", headers={"X-API-Key": key})
        self.token = resp.get("access_token")
        return self.token

    def get_profile(self) -> Dict[str, Any]:
        """Fetch current agent profile and remaining budget."""
        resp = self._make_request("GET", "/api/v1/agents/me")
        self.agent_profile = resp
        return resp

    def get_resources(self) -> List[Dict[str, Any]]:
        """List all available compute resources and their reserve costs."""
        return self._make_request("GET", "/api/v1/resources")

    def get_current_round(self) -> Dict[str, Any]:
        """Retrieve the currently open auction round and recent allocations."""
        return self._make_request("GET", "/api/v1/auctions/current")

    def submit_bid(
        self,
        bundle: List[str],
        amount: float,
        task: str = "",
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """
        Submit a combinatorial bid for a bundle of resources in the open auction round.
        """
        payload = {
            "resource_bundle": bundle,
            "bid_amount": float(amount),
            "task_description": task,
            "priority": priority.lower(),
        }
        return self._make_request("POST", "/api/v1/bids", data=payload)

    def get_analytics(self) -> Dict[str, Any]:
        """Retrieve aggregate efficiency analytics comparing Compute Bazaar vs Round-Robin."""
        return self._make_request("GET", "/api/v1/analytics/comparison")

    def start_simulation(self) -> Dict[str, Any]:
        """Start automated simulated background agent bidding."""
        return self._make_request("POST", "/api/v1/simulation/start")

    def stop_simulation(self) -> Dict[str, Any]:
        """Stop automated simulated background agent bidding."""
        return self._make_request("POST", "/api/v1/simulation/stop")
