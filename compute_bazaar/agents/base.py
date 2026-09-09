"""
=============================================================
 COMPUTE BAZAAR 2.0 -- BaseAgent (Week 3)
=============================================================
 Abstract base class for all autonomous AI agents.

 Each agent uses the Groq LLM to:
   - Understand a given task
   - Identify which compute resources it needs
   - Estimate task complexity (1-10)
   - Estimate the utility value of completing the task
   - Determine its maximum willingness-to-pay
   - Return a structured TaskDecision

 Subclasses define domain-specific system prompts and
 resource preferences. Bidding integration is NOT included
 here -- that is Week 4.
=============================================================
"""

from __future__ import annotations

import abc
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from compute_bazaar.llm.client import GroqLLMClient, LLMConfig, LLMResponse, LLMError


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

class AgentCapability(str, Enum):
    """Domain capabilities that agents can declare."""
    CODING = "coding"
    RESEARCH = "research"
    DATA_PROCESSING = "data_processing"
    ANALYSIS = "analysis"


@dataclass
class TaskDecision:
    """
    Structured output from an agent's task analysis.

    This is the result of the LLM reasoning about a task.
    It does NOT submit a bid -- that is a separate step (Week 4).
    """
    agent_name: str
    agent_type: str
    task_description: str

    # LLM-determined fields
    understood_task: str          # Agent's rephrasing of what the task requires
    required_resources: List[str] # Resource names needed (e.g. ["GPU-Slot-1", "RAM-Block-32GB"])
    complexity: int               # 1 (trivial) to 10 (extremely complex)
    estimated_utility: float      # Dollar value the agent assigns to completing this task
    max_willingness_to_pay: float # Maximum the agent would bid
    priority: str                 # "critical" | "high" | "normal" | "low"
    reasoning: List[str]          # Step-by-step reasoning from the LLM
    confidence: float             # 0.0 to 1.0 -- how confident the agent is

    # Metadata
    llm_latency_ms: float = 0.0
    llm_tokens_used: int = 0
    is_mock: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "task_description": self.task_description,
            "understood_task": self.understood_task,
            "required_resources": self.required_resources,
            "complexity": self.complexity,
            "estimated_utility": self.estimated_utility,
            "max_willingness_to_pay": self.max_willingness_to_pay,
            "priority": self.priority,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "llm_latency_ms": self.llm_latency_ms,
            "llm_tokens_used": self.llm_tokens_used,
            "is_mock": self.is_mock,
        }


# ─────────────────────────────────────────────────────────────
# AVAILABLE RESOURCES (mirrors the auction system)
# ─────────────────────────────────────────────────────────────

DEFAULT_AVAILABLE_RESOURCES = [
    "GPU-Slot-1",
    "GPU-Slot-2",
    "API-Budget-100",
    "RAM-Block-32GB",
    "CPU-Cores-8",
]


# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────

TASK_ANALYSIS_PROMPT = """You are {agent_name}, a {agent_type} agent in the Compute Bazaar auction platform.

Your capabilities: {capabilities}

AVAILABLE COMPUTE RESOURCES (you may only request from this list):
{resource_list}

YOUR BUDGET: ${budget:.2f}

TASK TO ANALYZE:
{task_description}

Analyze this task and return a JSON object with EXACTLY these fields:
{{
  "understood_task": "Your rephrasing of what this task requires",
  "required_resources": ["list", "of", "resource", "names", "from", "the", "available", "list"],
  "complexity": <integer 1-10>,
  "estimated_utility": <float, dollar value of completing this task>,
  "max_willingness_to_pay": <float, max you would bid, must be <= budget and <= estimated_utility>,
  "priority": "<critical|high|normal|low>",
  "reasoning": ["step 1 reasoning", "step 2 reasoning", "..."],
  "confidence": <float 0.0-1.0>
}}

RULES:
- required_resources MUST only contain names from the available resources list
- max_willingness_to_pay MUST be <= your budget (${budget:.2f})
- max_willingness_to_pay MUST be <= estimated_utility
- complexity is 1 (trivial) to 10 (extremely complex)
- confidence is 0.0 (no confidence) to 1.0 (fully confident)
- Provide clear, specific reasoning steps

Return ONLY the JSON object, no other text."""


# ─────────────────────────────────────────────────────────────
# MOCK DECISION GENERATOR
# ─────────────────────────────────────────────────────────────

def _generate_mock_decision(
    agent_name: str,
    agent_type: str,
    task_description: str,
    preferred_resources: List[str],
    budget: float,
) -> TaskDecision:
    """
    Generate a deterministic mock TaskDecision for testing.
    No LLM call is made.
    """
    # Pick resources based on agent type preferences
    resources = preferred_resources[:min(3, len(preferred_resources))]
    complexity = 5
    utility = min(budget * 0.8, 120.0)
    wtp = min(utility * 0.9, budget)

    return TaskDecision(
        agent_name=agent_name,
        agent_type=agent_type,
        task_description=task_description,
        understood_task=f"[Mock] Agent {agent_name} analyzed: {task_description[:80]}",
        required_resources=resources,
        complexity=complexity,
        estimated_utility=round(utility, 2),
        max_willingness_to_pay=round(wtp, 2),
        priority="normal",
        reasoning=[
            f"Mock reasoning: {agent_type} agent analyzed the task",
            f"Mock reasoning: identified {len(resources)} required resources",
            f"Mock reasoning: estimated complexity at {complexity}/10",
        ],
        confidence=0.75,
        llm_latency_ms=1.0,
        llm_tokens_used=0,
        is_mock=True,
    )


from compute_bazaar.models.bundle import CombinatorialBid, PriorityLevel
from backend.app.auction.engine import AuctionEngine, Bid as EngineBid
from compute_bazaar.agents.memory import AgentMemory, BidOutcome
from compute_bazaar.prediction import ResourcePredictor, ResourcePrediction
from compute_bazaar.tools import ToolRegistry, check_resources, check_budget, estimate_cost, execute_task


def to_engine_bid(bid: CombinatorialBid) -> EngineBid:
    """
    Convert a SDK/models CombinatorialBid object into an AuctionEngine Bid object.
    """
    bid_id = bid.bid_id or f"bid-{uuid.uuid4().hex[:8]}"
    return EngineBid(
        bid_id=bid_id,
        agent_id=bid.agent_name,
        resource_bundle=list(bid.resource_bundle),
        bid_amount=float(bid.bid_amount),
        task_description=bid.task_description,
        priority=bid.priority.lower() if isinstance(bid.priority, str) else bid.priority.value,
    )


# ─────────────────────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────────────────────

class BaseAgent(abc.ABC):
    """
    Abstract base class for autonomous AI agents in Compute Bazaar.

    Each agent has:
      - A name and agent type
      - A capability domain
      - A budget
      - An LLM client for reasoning
      - Cognitive AgentMemory (Week 6: historical bids, win/loss outcomes, surplus, clearing prices)
      - ML ResourcePredictor (Week 7: forecasting runtime, capacity, and cost)
      - Safe ToolRegistry (Week 7: check_resources, check_budget, estimate_cost, execute_task)
      - Preferred resources based on its domain

    Methods:
      - analyze_task(): Uses Groq LLM + ML Predictor to analyze task requirements and return TaskDecision
      - compute_intelligent_bid_amount(): Calculates adaptive bid amount using memory & market history
      - create_bid(): Generates single-resource or combinatorial bundle bid(s)
      - submit_to_engine(): Submits generated bid(s) to the existing AuctionEngine
      - record_auction_outcome(): Logs round outcomes into AgentMemory for learning
      - invoke_tool(): Executes a safe tool from the agent's tool registry
      - execute_awarded_task(): Safely executes a task and logs ML performance metrics
    """
    def __init__(
        self,
        name: str,
        budget: float = 500.0,
        llm_client: Optional[GroqLLMClient] = None,
        available_resources: Optional[List[str]] = None,
    ):
        self.name = name
        self.budget = budget
        self.available_resources = (
            available_resources or DEFAULT_AVAILABLE_RESOURCES
        )

        self.decision_history: List[TaskDecision] = []
        self.submitted_bids: List[CombinatorialBid] = []

        self.memory = AgentMemory(agent_name=name)
        self.predictor = ResourcePredictor()
        self.tools = ToolRegistry()

        # Use provided client or create a live Groq client
        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = GroqLLMClient(
                config=LLMConfig(
                    model="openai/gpt-oss-120b",
                    mock_mode=False,
                )
            )

    @property
    @abc.abstractmethod
    def agent_type(self) -> str:
        """Return the agent type name (e.g. 'coding', 'research')."""
        ...

    @property
    @abc.abstractmethod
    def capability(self) -> AgentCapability:
        """Return the agent's primary capability."""
        ...

    @property
    @abc.abstractmethod
    def system_prompt(self) -> str:
        """Return the domain-specific system prompt for this agent."""
        ...

    @property
    @abc.abstractmethod
    def preferred_resources(self) -> List[str]:
        """Return ordered list of preferred resource names for this agent type."""
        ...
    def analyze_task(
        self,
        task_description: str,
        budget_override: Optional[float] = None,
    ) -> TaskDecision:
        """
        Use the LLM to analyze a task and produce a structured decision.

        Args:
            task_description: Human-readable description of the task.
            budget_override: Override the agent's budget for this analysis.

        Returns:
            TaskDecision with all LLM-derived fields.
        """
        effective_budget = budget_override if budget_override is not None else self.budget

        # ── Mock mode shortcut ───────────────────────────────
        if self.llm.config.mock_mode:
            decision = _generate_mock_decision(
                agent_name=self.name,
                agent_type=self.agent_type,
                task_description=task_description,
                preferred_resources=self.preferred_resources,
                budget=effective_budget,
            )
            self.decision_history.append(decision)
            logger.debug(
                f"[{self.name}] Mock analysis | task='{task_description[:50]}...' | "
                f"resources={decision.required_resources} | wtp=${decision.max_willingness_to_pay}"
            )
            return decision

        # ── Build the prompt ─────────────────────────────────
        resource_list = "\n".join(f"  - {r}" for r in self.available_resources)

        prompt = TASK_ANALYSIS_PROMPT.format(
            agent_name=self.name,
            agent_type=self.agent_type,
            capabilities=self.capability.value,
            resource_list=resource_list,
            budget=effective_budget,
            task_description=task_description,
        )

        # ── Call the LLM ─────────────────────────────────────
        try:
            response = self.llm.complete(
                prompt=prompt,
                system_prompt=self.system_prompt,
                json_mode=True,
                temperature=0.4,  # Lower temp for structured output
            )

            parsed = response.to_json()
            if parsed is None:
                raise LLMError(f"Failed to parse LLM response as JSON: {response.content[:200]}")

            # ── Validate and clamp values ────────────────────
            required = parsed.get("required_resources", [])
            # Filter to only valid resources
            valid_resources = [r for r in required if r in self.available_resources]
            if not valid_resources:
                valid_resources = self.preferred_resources[:2]

            complexity = max(1, min(10, int(parsed.get("complexity", 5))))
            utility = max(0.0, float(parsed.get("estimated_utility", 50.0)))
            wtp = max(0.0, float(parsed.get("max_willingness_to_pay", utility * 0.8)))
            # Enforce budget constraint
            wtp = min(wtp, effective_budget, utility)

            priority = parsed.get("priority", "normal")
            if priority not in ("critical", "high", "normal", "low"):
                priority = "normal"

            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))

            decision = TaskDecision(
                agent_name=self.name,
                agent_type=self.agent_type,
                task_description=task_description,
                understood_task=str(parsed.get("understood_task", task_description)),
                required_resources=valid_resources,
                complexity=complexity,
                estimated_utility=round(utility, 2),
                max_willingness_to_pay=round(wtp, 2),
                priority=priority,
                reasoning=parsed.get("reasoning", ["LLM provided no reasoning"]),
                confidence=confidence,
                llm_latency_ms=response.latency_ms,
                llm_tokens_used=response.usage.total_tokens,
                is_mock=False,
            )

            self.decision_history.append(decision)

            logger.info(
                f"[{self.name}] Task analyzed | complexity={complexity} | "
                f"resources={valid_resources} | wtp=${wtp:.2f} | "
                f"latency={response.latency_ms:.0f}ms"
            )

            return decision

        except LLMError:
            raise
        except Exception as e:
            logger.error(f"[{self.name}] Task analysis failed: {e}")
            raise LLMError(f"Agent {self.name} task analysis failed: {e}")

    # ── Intelligent & Adaptive Bidding Mechanics (Weeks 4-6) ────

    def compute_intelligent_bid_amount(
        self,
        decision: TaskDecision,
        shading_factor: Optional[float] = None,
    ) -> float:
        """
        Compute an adaptive intelligent bid amount based on Groq decision metrics & AgentMemory:
          - Adaptive Starvation Boost: If agent is starved (>= 3 consecutive losses),
            boost shading multiplier to 0.95-0.98 to increase win probability.
          - High Win-Rate Margin Maximization: If win rate > 70%, shade bid to 0.70-0.75
            to increase profit surplus margin.
          - Clearing Price Alignment: Bids above historical market clearing prices.
          - Upper bounds: Capped by WTP, estimated utility, and agent's remaining budget.
        """
        if shading_factor is not None:
            multiplier = max(0.05, min(1.0, float(shading_factor)))
        elif self.memory.is_starved(threshold=3):
            # Starvation Adaptation: increase multiplier aggressively to win allocation
            multiplier = 0.96
            logger.info(f"[{self.name}] Adaptive Bidding: Starvation detected ({self.memory.consecutive_losses} losses). Boosting bid multiplier to {multiplier}.")
        elif self.memory.get_win_rate(window=10) > 0.70:
            # High win rate: shade bid to maximize surplus margin
            multiplier = 0.72
            logger.info(f"[{self.name}] Adaptive Bidding: High win rate detected ({self.memory.get_win_rate(10)*100:.0f}%). Shading bid multiplier to {multiplier} for margin maximization.")
        else:
            priority_multipliers = {
                "critical": 0.98,
                "high": 0.88,
                "normal": 0.78,
                "low": 0.55,
            }
            multiplier = priority_multipliers.get(decision.priority.lower(), 0.78)

        raw_bid = decision.max_willingness_to_pay * multiplier

        # Clearing price alignment: ensure bid is at least average clearing price if known
        if decision.required_resources:
            clearing_prices = [
                self.memory.get_average_clearing_price(r, default=0.0)
                for r in decision.required_resources
            ]
            avg_clearing = sum(clearing_prices)
            if avg_clearing > 0:
                raw_bid = max(raw_bid, avg_clearing * 1.05)

        # Upper bounds enforcement
        max_allowed = min(decision.max_willingness_to_pay, decision.estimated_utility, self.budget)
        if max_allowed <= 0:
            return 0.0

        bid_amount = max(0.01, min(raw_bid, max_allowed))
        return round(bid_amount, 2)

    def create_bid(
        self,
        task_description: str,
        decision: Optional[TaskDecision] = None,
        bid_type: str = "bundle",
        shading_factor: Optional[float] = None,
    ) -> Union[CombinatorialBid, List[CombinatorialBid]]:
        """
        Generate intelligent adaptive bid(s) using Groq-based task analysis and memory history.

        Args:
            task_description: Human-readable task description.
            decision: Optional pre-computed TaskDecision (if None, calls analyze_task()).
            bid_type: "bundle" for combinatorial bundle bid, or "single" for per-resource bids.
            shading_factor: Optional strategy multiplier (0.05 to 1.0).

        Returns:
            CombinatorialBid (for "bundle") or List[CombinatorialBid] (for "single").
        """
        if decision is None:
            decision = self.analyze_task(task_description)

        # Adaptive Priority Escalation if starved
        if self.memory.is_starved(threshold=3) and decision.priority in ("normal", "low"):
            decision.priority = "high"

        resources = decision.required_resources or self.preferred_resources[:2]
        total_bid_amount = self.compute_intelligent_bid_amount(decision, shading_factor=shading_factor)

        if bid_type.lower() == "single":
            bids: List[CombinatorialBid] = []
            num_res = len(resources)
            per_resource_amount = round(total_bid_amount / num_res, 2) if num_res > 0 else total_bid_amount
            per_resource_amount = max(0.01, per_resource_amount)
            per_resource_valuation = round(decision.estimated_utility / num_res, 2) if num_res > 0 else decision.estimated_utility

            for i, res in enumerate(resources):
                bid_id = f"{self.name[:4].lower()}-s{i+1}-{uuid.uuid4().hex[:6]}"
                bid = CombinatorialBid(
                    bid_id=bid_id,
                    agent_name=self.name,
                    resource_bundle=[res],
                    bid_amount=per_resource_amount,
                    task_description=f"{task_description} [{res}]",
                    priority=decision.priority,
                    valuation=per_resource_valuation,
                )
                bids.append(bid)
                self.submitted_bids.append(bid)
                self.memory.record_bid(bid)
            return bids
        else:
            # Combinatorial bundle bid
            bid_id = f"{self.name[:4].lower()}-b-{uuid.uuid4().hex[:6]}"
            bid = CombinatorialBid(
                bid_id=bid_id,
                agent_name=self.name,
                resource_bundle=resources,
                bid_amount=total_bid_amount,
                task_description=task_description,
                priority=decision.priority,
                valuation=decision.estimated_utility,
            )
            self.submitted_bids.append(bid)
            self.memory.record_bid(bid)
            return bid

    def record_auction_outcome(
        self,
        bid_id: str,
        round_id: int,
        won: bool,
        bid_amount: float,
        payment_amount: float = 0.0,
        resources_allocated: Optional[List[str]] = None,
    ):
        """
        Record auction round outcome in agent's AgentMemory.

        Updates win/loss counters, deducts payment from remaining budget if won,
        and logs outcome for adaptive bidding.
        """
        if won and payment_amount > 0:
            self.budget = max(0.0, self.budget - payment_amount)

        self.memory.record_outcome(
            bid_id=bid_id,
            round_id=round_id,
            won=won,
            bid_amount=bid_amount,
            payment_amount=payment_amount,
            resources_allocated=resources_allocated,
        )

    def submit_to_engine(
        self,
        engine: AuctionEngine,
        bid_or_bids: Union[CombinatorialBid, EngineBid, List[Union[CombinatorialBid, EngineBid]]],
    ) -> List[bool]:
        """
        Connect generated intelligent bid(s) to the EXISTING AuctionEngine.

        Accepts CombinatorialBid or EngineBid objects (or a list of them),
        converts if necessary, and calls engine.submit_bid().
        """
        if not isinstance(bid_or_bids, list):
            bids = [bid_or_bids]
        else:
            bids = bid_or_bids

        results = []
        for b in bids:
            if isinstance(b, CombinatorialBid):
                engine_bid = to_engine_bid(b)
            else:
                engine_bid = b

            success = engine.submit_bid(engine_bid)
            results.append(success)
            
            if success:
                logger.info(
                    f"[{self.name}] Intelligent bid connected to AuctionEngine: "
                    f"bid_id={engine_bid.bid_id} | amount=${engine_bid.bid_amount:.2f} | "
                    f"bundle={engine_bid.resource_bundle} | priority={engine_bid.priority}"
                )
        return results

    def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Invoke a safe tool from the agent's ToolRegistry.

        Available tools:
          - check_resources
          - check_budget
          - estimate_cost
          - execute_task
        """
        if tool_name == "check_budget" and "agent" not in kwargs:
            kwargs["agent"] = self
        elif tool_name == "estimate_cost" and "predictor" not in kwargs:
            kwargs["predictor"] = self.predictor
        return self.tools.invoke(tool_name, **kwargs)

    def execute_awarded_task(
        self,
        task_description: str,
        allocation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Safely execute a task using awarded compute resources.

        Simulates execution, updates agent budget, and feeds performance
        metrics back into ResourcePredictor and AgentMemory.
        """
        return self.invoke_tool(
            "execute_task",
            agent=self,
            task_description=task_description,
            allocation_result=allocation_result,
            predictor=self.predictor,
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Return serialized decision history."""
        return [d.to_dict() for d in self.decision_history]

    def __repr__(self) -> str:
        mode = "MOCK" if self.llm.config.mock_mode else "LIVE"
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"type={self.agent_type!r}, budget=${self.budget:.2f}, "
            f"mode={mode})"
        )


