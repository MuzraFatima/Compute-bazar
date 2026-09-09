"""
=============================================================
 COMPUTE BAZAAR 2.0 — Autonomous Groq LLM Agent Simulator
=============================================================
 Replaces the classic rule-based simulation bots with four
 real Groq-powered autonomous AI agents:

   - DevBot-Gold      → CodingAgent   (gold tier)
   - ResearchBot-Gold → ResearchAgent (gold tier)
   - ETLBot-Silver    → DataAgent     (silver tier)
   - QuantBot-Bronze  → AnalysisAgent (bronze tier)

 Each round:
   1. Agent calls Groq LLM to analyze a realistic task.
   2. Agent generates an intelligent combinatorial bid.
   3. Bid is saved to DB for the current auction round.
   4. Bid is broadcast to WebSocket for the live ticker.

 Errors (LLM timeout, rate limit, etc.) are caught per-agent
 so one failure never crashes the others or the scheduler.
=============================================================
"""

import json
import asyncio
import random
from datetime import datetime, timezone
from loguru import logger
from sqlalchemy.orm import Session

from backend.app.db.models import SessionLocal, Agent, Resource, Bid, Auction, init_db
from backend.app.api.websocket import manager

# ── Agent task banks ─────────────────────────────────────────

_CODING_TASKS = [
    "Develop and test a REST API for a machine learning inference service",
    "Debug and optimize a distributed Python microservice under heavy load",
    "Build an automated CI/CD pipeline with Docker and GitHub Actions",
    "Implement a real-time fraud detection algorithm in Python",
    "Refactor a legacy data pipeline to async SQLAlchemy and FastAPI",
]

_RESEARCH_TASKS = [
    "Synthesize 50 recent arXiv papers on LLM reasoning and generate a report",
    "Perform competitive intelligence sweep on top 5 AI infrastructure providers",
    "Literature review on combinatorial auction theory and mechanism design",
    "Survey benchmarking methodologies for large language model evaluation",
    "Analyze patent landscape for autonomous agent systems in cloud computing",
]

_DATA_TASKS = [
    "Clean and process a 20GB multi-modal dataset for neural search embeddings",
    "Build GPU-accelerated vector embedding pipeline for 10M product records",
    "ETL pipeline to ingest, deduplicate and transform raw transaction logs",
    "Feature engineering over 500GB sensor telemetry for anomaly detection model",
    "Batch tokenization and embedding of 2M scientific abstracts using BERT",
]

_ANALYSIS_TASKS = [
    "Monte Carlo risk simulation with 50,000 iterations for portfolio stress testing",
    "Dynamic pricing elasticity model across 12 product categories using regression",
    "Statistical analysis of A/B test results with 99% confidence intervals",
    "Multi-factor risk attribution analysis for equity portfolio rebalancing",
    "Forecast demand elasticity and optimal price point for SaaS subscription tiers",
]

# Map agent DB name → task bank
AGENT_TASK_MAP = {
    "DevBot-Gold":       _CODING_TASKS,
    "ResearchBot-Gold":  _RESEARCH_TASKS,
    "ETLBot-Silver":     _DATA_TASKS,
    "QuantBot-Bronze":   _ANALYSIS_TASKS,
}

# ── In-memory per-agent objects ──────────────────────────────
# Initialised lazily on first run to avoid import overhead at startup
_agent_objects: dict = {}
_llm_client = None

# Track total LLM calls, tokens, latency for telemetry endpoint
_llm_stats = {
    "total_calls": 0,
    "total_tokens": 0,
    "total_latency_ms": 0.0,
    "last_model": "openai/gpt-oss-120b",
}

# Simulation state
_auto_sim_task = None
_auto_sim_stop = asyncio.Event()


def get_llm_telemetry() -> dict:
    """Return accumulated LLM telemetry stats."""
    calls = _llm_stats["total_calls"]
    return {
        "calls": calls,
        "tokens": _llm_stats["total_tokens"],
        "avg_latency_ms": round(_llm_stats["total_latency_ms"] / calls, 1) if calls > 0 else 0.0,
        "is_mock": False,
        "model": _llm_stats["last_model"],
    }


def _init_agents():
    """Lazily initialise Groq LLM client and 4 autonomous agent objects."""
    global _llm_client, _agent_objects

    if _agent_objects:
        return  # Already initialised

    import os
    from dotenv import load_dotenv
    load_dotenv()

    from compute_bazaar.llm import GroqLLMClient, LLMConfig
    from compute_bazaar.agents import CodingAgent, ResearchAgent, DataAgent, AnalysisAgent

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("[AutonomousSim] GROQ_API_KEY not set — autonomous simulation disabled.")
        return

    try:
        _llm_client = GroqLLMClient(
            config=LLMConfig(model="openai/gpt-oss-120b", temperature=0.4, mock_mode=False),
            api_key=api_key,
        )
        _agent_objects = {
            "DevBot-Gold":      CodingAgent("DevBot-Gold",      budget=1000.0, llm_client=_llm_client),
            "ResearchBot-Gold": ResearchAgent("ResearchBot-Gold", budget=1000.0, llm_client=_llm_client),
            "ETLBot-Silver":    DataAgent("ETLBot-Silver",     budget=600.0,  llm_client=_llm_client),
            "QuantBot-Bronze":  AnalysisAgent("QuantBot-Bronze",  budget=400.0,  llm_client=_llm_client),
        }
        logger.info("[AutonomousSim] Groq LLM client and 4 autonomous agents initialised.")
    except Exception as e:
        logger.error(f"[AutonomousSim] Failed to initialise agents: {e}")


def ensure_autonomous_agents_in_db(db: Session):
    """Make sure the 4 autonomous agent rows exist in the DB."""
    init_db()
    agent_configs = [
        ("DevBot-Gold",      "Autonomous Groq Coding Agent",     "auto_devbot_gold",      1000.0, "gold"),
        ("ResearchBot-Gold", "Autonomous Groq Research Agent",   "auto_researchbot_gold", 1000.0, "gold"),
        ("ETLBot-Silver",    "Autonomous Groq Data ETL Agent",   "auto_etlbot_silver",    600.0,  "silver"),
        ("QuantBot-Bronze",  "Autonomous Groq Analysis Agent",   "auto_quantbot_bronze",  400.0,  "bronze"),
    ]
    for name, desc, api_key, budget, tier in agent_configs:
        existing = db.query(Agent).filter(Agent.name == name).first()
        if not existing:
            db.add(Agent(
                name=name,
                description=desc,
                api_key=api_key,
                budget_total=budget,
                budget_remaining=budget,
                tier=tier,
            ))
    db.commit()


async def _submit_agent_bid(agent_name: str, db: Session, auction_id: int):
    """
    Run one autonomous agent through the full Groq reasoning → bid pipeline.
    Saves the bid to DB and broadcasts it to WebSockets.
    Any exception is caught and logged without crashing the caller.
    """
    try:
        agent_obj = _agent_objects.get(agent_name)
        if not agent_obj:
            return

        # Pick a random task from the agent's domain
        task = random.choice(AGENT_TASK_MAP.get(agent_name, ["Process data workload"]))

        # Sync Groq LLM call — run in thread executor to keep asyncio non-blocking
        loop = asyncio.get_event_loop()
        decision = await loop.run_in_executor(
            None,
            lambda: agent_obj.analyze_task(task)
        )

        # Accumulate LLM telemetry
        _llm_stats["total_calls"] += 1
        _llm_stats["total_tokens"] += decision.llm_tokens_used
        _llm_stats["total_latency_ms"] += decision.llm_latency_ms

        # Create combinatorial bid
        combo_bid = await loop.run_in_executor(
            None,
            lambda: agent_obj.create_bid(task_description=task, decision=decision, bid_type="bundle")
        )

        # Validate resources exist in DB
        db_resources = db.query(Resource).filter(Resource.is_available == True).all()
        valid_res_names = {r.name for r in db_resources}
        bundle = [r for r in combo_bid.resource_bundle if r in valid_res_names]
        if not bundle:
            logger.warning(f"[AutonomousSim] {agent_name}: No valid resources in bundle {combo_bid.resource_bundle}")
            return

        # Clamp bid amount to agent's DB budget
        db_agent = db.query(Agent).filter(Agent.name == agent_name).first()
        if not db_agent or db_agent.budget_remaining < 5.0:
            return

        bid_amount = min(combo_bid.bid_amount, db_agent.budget_remaining)
        if bid_amount <= 0:
            return

        # Get active auction
        auction = db.query(Auction).filter(Auction.id == auction_id).first()
        if not auction or auction.status != "open":
            return

        # Persist bid to DB
        db_bid = Bid(
            auction_id=auction_id,
            agent_id=db_agent.id,
            resource_bundle=json.dumps(bundle),
            bid_amount=round(bid_amount, 2),
            task_description=task,
            priority=decision.priority,
        )
        db.add(db_bid)
        db_agent.last_active_at = datetime.now(timezone.utc).isoformat()
        db.commit()

        logger.info(
            f"[AutonomousSim] {agent_name} | task='{task[:40]}' | "
            f"complexity={decision.complexity} | WTP=${decision.max_willingness_to_pay:.2f} | "
            f"bid=${bid_amount:.2f} | bundle={bundle} | latency={decision.llm_latency_ms:.0f}ms"
        )

        # Broadcast bid to WebSocket live ticker
        await manager.broadcast_json({
            "type": "bid_submitted",
            "agent": agent_name,
            "bundle": bundle,
            "amount": round(bid_amount, 2),
            "priority": decision.priority,
            "task": task,
            "complexity": decision.complexity,
            "confidence": decision.confidence,
            "wtp": decision.max_willingness_to_pay,
            "llm_latency_ms": round(decision.llm_latency_ms, 1),
            "submitted_at": db_bid.submitted_at,
            "mode": "autonomous",
        })

    except Exception as e:
        logger.error(f"[AutonomousSim] {agent_name} bid failed: {type(e).__name__}: {e}")


async def run_autonomous_round(db: Session, auction_id: int):
    """
    Submit bids from all 4 autonomous agents concurrently for the given auction round.
    Each agent failure is isolated — others still bid.
    """
    tasks = [
        _submit_agent_bid(name, db, auction_id)
        for name in _agent_objects
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Refill agent budgets if running low to keep simulation going
    active_agents = db.query(Agent).filter(
        Agent.name.in_(list(_agent_objects.keys())),
        Agent.is_active == True
    ).all()
    for a in active_agents:
        if a.budget_remaining < 100.0:
            a.budget_remaining = min(a.budget_total, a.budget_remaining + 300.0)
    db.commit()


async def _autonomous_simulation_loop():
    """
    Background async loop: submits Groq-powered bids every 12 seconds
    (slightly longer than classic 2.5s since LLM calls take ~1-2s each).
    """
    logger.info("[AutonomousSim] Autonomous simulation loop started.")
    _init_agents()

    if not _agent_objects:
        logger.error("[AutonomousSim] No agents initialised — cannot run autonomous simulation.")
        return

    db = SessionLocal()
    try:
        ensure_autonomous_agents_in_db(db)

        while not _auto_sim_stop.is_set():
            try:
                active_auction = (
                    db.query(Auction)
                    .filter(Auction.status == "open")
                    .order_by(Auction.round_number.desc())
                    .first()
                )
                if active_auction:
                    await run_autonomous_round(db, active_auction.id)

                # Wait before next wave — LLM calls need breathing room
                await asyncio.sleep(12)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AutonomousSim] Loop error: {e}")
                await asyncio.sleep(5)
    finally:
        db.close()
        logger.info("[AutonomousSim] Autonomous simulation loop stopped.")


def start_autonomous_simulation():
    """Start the Groq-powered autonomous agent simulation background task."""
    global _auto_sim_task, _auto_sim_stop
    _auto_sim_stop.clear()
    if _auto_sim_task is None or _auto_sim_task.done():
        _auto_sim_task = asyncio.create_task(_autonomous_simulation_loop())
        logger.info("[AutonomousSim] Autonomous simulation task spawned.")


def stop_autonomous_simulation():
    """Signal the autonomous simulation loop to stop."""
    global _auto_sim_task
    if _auto_sim_task and not _auto_sim_task.done():
        _auto_sim_stop.set()
        _auto_sim_task.cancel()
        logger.info("[AutonomousSim] Autonomous simulation shutdown signalled.")


def is_autonomous_simulation_running() -> bool:
    """Return True if autonomous simulation task is active."""
    return _auto_sim_task is not None and not _auto_sim_task.done()
