"""
=============================================================
 COMPUTE BAZAAR — Workload & Bid Generators
=============================================================
 Generates realistic multi-agent compute workload bundles,
 valuations, priority distributions, and agent cohorts.
=============================================================
"""

import random
import uuid
from typing import List, Dict, Optional
from compute_bazaar.models.bundle import CombinatorialBid, ResourceItem, PriorityLevel, AgentTier

# Common AI task templates with realistic resource bundles & valuations
TASK_TEMPLATES = [
    {
        "category": "llm_fine_tuning",
        "description": "LoRA fine-tuning on 70B parameter model",
        "bundle_options": [
            ["GPU-Slot-1", "GPU-Slot-2", "RAM-Block-32GB"],
            ["GPU-Slot-1", "RAM-Block-32GB", "CPU-Cores-8"],
        ],
        "base_value_range": (80.0, 160.0),
        "priority": PriorityLevel.HIGH.value,
    },
    {
        "category": "realtime_fraud_inference",
        "description": "Sub-10ms transactional fraud detection inference",
        "bundle_options": [
            ["GPU-Slot-1", "API-Budget-100"],
            ["GPU-Slot-2", "API-Budget-100", "CPU-Cores-8"],
        ],
        "base_value_range": (110.0, 220.0),
        "priority": PriorityLevel.CRITICAL.value,
    },
    {
        "category": "batch_nlp_embedding",
        "description": "Document corpus vector embedding batch job",
        "bundle_options": [
            ["GPU-Slot-2", "CPU-Cores-8"],
            ["CPU-Cores-8", "RAM-Block-32GB"],
        ],
        "base_value_range": (35.0, 75.0),
        "priority": PriorityLevel.NORMAL.value,
    },
    {
        "category": "agentic_web_search",
        "description": "Multi-agent research and reasoning workflow",
        "bundle_options": [
            ["API-Budget-100", "CPU-Cores-8"],
            ["API-Budget-100", "RAM-Block-32GB"],
        ],
        "base_value_range": (45.0, 95.0),
        "priority": PriorityLevel.NORMAL.value,
    },
    {
        "category": "background_scraping",
        "description": "Low-priority web scraping and ETL pipeline",
        "bundle_options": [
            ["CPU-Cores-8"],
            ["RAM-Block-32GB"],
        ],
        "base_value_range": (15.0, 30.0),
        "priority": PriorityLevel.LOW.value,
    },
]


def generate_synthetic_bid(
    agent_name: str,
    available_resource_names: Optional[List[str]] = None,
    task_category: Optional[str] = None,
    noise_factor: float = 0.15,
) -> CombinatorialBid:
    """Generate a single realistic synthetic bid."""
    if task_category:
        templates = [t for t in TASK_TEMPLATES if t["category"] == task_category]
        template = templates[0] if templates else random.choice(TASK_TEMPLATES)
    else:
        template = random.choice(TASK_TEMPLATES)

    bundle = random.choice(template["bundle_options"])
    if available_resource_names:
        # Filter to only currently existing resources or fallback
        filtered = [r for r in bundle if r in available_resource_names]
        if not filtered:
            filtered = random.sample(available_resource_names, min(len(available_resource_names), random.randint(1, 2)))
        bundle = filtered

    min_val, max_val = template["base_value_range"]
    base_val = random.uniform(min_val, max_val)
    # Add noise
    val = round(base_val * (1.0 + random.uniform(-noise_factor, noise_factor)), 2)

    return CombinatorialBid(
        bid_id=str(uuid.uuid4())[:8],
        agent_name=agent_name,
        resource_bundle=bundle,
        bid_amount=val,
        task_description=template["description"],
        priority=template["priority"],
        valuation=val,
    )


def generate_workload_batch(
    num_agents: int = 10,
    available_resources: Optional[List[str]] = None,
) -> List[CombinatorialBid]:
    """Generate a batch of competitive bids from diverse simulated agents."""
    if available_resources is None:
        available_resources = [
            "GPU-Slot-1", "GPU-Slot-2", "API-Budget-100",
            "RAM-Block-32GB", "CPU-Cores-8"
        ]

    bids = []
    for i in range(num_agents):
        agent_name = f"Agent-{i+1:02d}"
        bid = generate_synthetic_bid(agent_name, available_resource_names=available_resources)
        bids.append(bid)
    return bids
