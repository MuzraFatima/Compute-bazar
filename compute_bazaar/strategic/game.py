"""
Repeated Local-Local-Global (LLG) auctions with adaptive and LLM bidders.

Two GPUs, A and B. Local L1 wants A, local L2 wants B, global G wants both.
Each round values are redrawn (locals ~ U(0, 100), global ~ U(0, 200)),
every bidder submits one bid for its bundle, and the mechanism allocates
exactly and charges:

  vcg          agent-level Clarke pivot (truthful in a one-shot auction)
  first_price  pay your bid (shading is the textbook best response)
  core         minimum-revenue core, VCG-nearest (compute_bazaar/strategic/core.py)

LLG is the standard setting where VCG's weaknesses show: payments can be far
below what the losing global bidder offered, and locals can profit by jointly
overbidding. Comparing how bidders adapt under each rule is the question.

Bidders report a *bid*; we track the bid/value ratio m. m = 1 is truthful,
m < 1 is shading, m > 1 is overbidding.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field

from backend.app.auction.engine import Bid, Resource, compute_vcg_agent_payments, solve_exhaustive
from compute_bazaar.strategic.core import core_payments

RESOURCES = {"A": Resource("A", "gpu"), "B": Resource("B", "gpu")}
ROLES = {"L1": ["A"], "L2": ["B"], "G": ["A", "B"]}
VALUE_RANGE = {"L1": 100.0, "L2": 100.0, "G": 200.0}
MULTIPLIERS = tuple(round(0.1 + 0.1 * k, 2) for k in range(15))  # 0.1 … 1.5


def settle(mechanism: str, bids: dict[str, float]) -> tuple[set[str], dict[str, float]]:
    """
    Allocate and price one LLG round in closed form. Returns (winners, payments).

    Locals win iff b1 + b2 ≥ bG. VCG: p1 = max(0, bG − b2), p2 = max(0, bG − b1);
    a winning G pays b1 + b2. Core (min revenue, VCG-nearest by equal excess):
    locals must jointly pay at least bG. ``settle_generic`` gives the same answers
    through the general engine and is used to test this function.
    """
    b1, b2, bg = bids["L1"], bids["L2"], bids["G"]
    if b1 + b2 >= bg:
        winners = {"L1", "L2"}
        if mechanism == "first_price":
            return winners, {"L1": b1, "L2": b2}
        v1, v2 = max(0.0, bg - b2), max(0.0, bg - b1)
        if mechanism == "vcg":
            return winners, {"L1": v1, "L2": v2}
        if mechanism == "core":
            gap = max(0.0, bg - v1 - v2)
            p1, p2 = v1 + gap / 2, v2 + gap / 2
            if p1 > b1:
                p1, p2 = b1, min(b2, v2 + gap - (b1 - v1))
            elif p2 > b2:
                p1, p2 = min(b1, v1 + gap - (b2 - v2)), b2
            return winners, {"L1": p1, "L2": p2}
    else:
        winners = {"G"}
        if mechanism == "first_price":
            return winners, {"G": bg}
        if mechanism in ("vcg", "core"):
            return winners, {"G": b1 + b2}
    raise ValueError(mechanism)


def settle_generic(mechanism: str, bids: dict[str, float]) -> tuple[set[str], dict[str, float]]:
    """Same as ``settle`` but through the general WDP / VCG / core code (slow)."""
    engine_bids = [Bid(f"{a}-bid", a, ROLES[a], max(b, 1e-6)) for a, b in bids.items()]
    winners, value = solve_exhaustive(engine_bids, RESOURCES)
    agents = {b.agent_id for b in winners}
    if mechanism == "first_price":
        pay = {b.agent_id: b.bid_amount for b in winners}
    elif mechanism == "vcg":
        pay = compute_vcg_agent_payments(winners, engine_bids, RESOURCES, solver=solve_exhaustive, optimal_value=value)
    elif mechanism == "core":
        pay = core_payments(winners, engine_bids, RESOURCES)
    else:
        raise ValueError(mechanism)
    return agents, pay


def utility(agent: str, value: float, winners: set[str], pay: dict[str, float]) -> float:
    return value - pay.get(agent, 0.0) if agent in winners else 0.0


@dataclass
class HedgeBidder:
    """Full-information multiplicative weights over bid/value multipliers."""

    agent: str
    eta: float = 0.08
    weights: list[float] = field(default_factory=lambda: [1.0] * len(MULTIPLIERS))
    rng: random.Random = field(default_factory=random.Random)

    def choose(self) -> float:
        return self.rng.choices(MULTIPLIERS, self.weights)[0]

    def update(self, mechanism: str, value: float, others: dict[str, float]) -> None:
        scale = VALUE_RANGE[self.agent]
        for k, m in enumerate(MULTIPLIERS):
            winners, pay = settle(mechanism, {**others, self.agent: m * value})
            self.weights[k] *= math.exp(self.eta * utility(self.agent, value, winners, pay) / scale)
        top = max(self.weights)
        self.weights = [w / top for w in self.weights]

    def mean_multiplier(self) -> float:
        z = sum(self.weights)
        return sum(m * w for m, w in zip(MULTIPLIERS, self.weights, strict=True)) / z


def run_learning(mechanism: str, rounds: int = 1500, seed: int = 0) -> dict:
    rng = random.Random(seed)
    bidders = {a: HedgeBidder(a, rng=random.Random(seed * 31 + k)) for k, a in enumerate(ROLES)}
    tail = max(1, rounds // 5)
    eff, rev, ms = [], [], {a: [] for a in ROLES}
    for t in range(rounds):
        values = {a: rng.uniform(0, VALUE_RANGE[a]) for a in ROLES}
        mult = {a: b.choose() for a, b in bidders.items()}
        bids = {a: mult[a] * values[a] for a in ROLES}
        winners, pay = settle(mechanism, bids)
        if t >= rounds - tail:
            best = max(values["L1"] + values["L2"], values["G"])
            eff.append(sum(values[a] for a in winners) / best if best > 0 else 1.0)
            rev.append(sum(pay.values()))
            for a in ROLES:
                ms[a].append(mult[a])
        for a, b in bidders.items():
            b.update(mechanism, values[a], {o: bids[o] for o in ROLES if o != a})
    return {
        "mechanism": mechanism,
        "seed": seed,
        "efficiency": statistics.mean(eff),
        "revenue": statistics.mean(rev),
        "bid_value_ratio": {a: statistics.mean(v) for a, v in ms.items()},
    }


# ── LLM bidders ─────────────────────────────────────────────────

RULES = {
    "vcg": (
        "Payment rule: if you win you pay the externality you impose on the other bidders "
        "(Vickrey-Clarke-Groves): the drop in the others' total bid value caused by your presence. "
        "You never pay more than your bid."
    ),
    "first_price": "Payment rule: if you win you pay exactly your bid.",
    "core": (
        "Payment rule: winners pay the minimum total amount such that no group of bidders, "
        "including losers, offered the seller more for the same items; individual payments "
        "are as close to the Vickrey-Clarke-Groves externality as that allows, and never above your bid."
    ),
}
TRUTHFUL_HINT = "Note: under this payment rule, bidding your true value is a dominant strategy."


def build_prompt(agent, value, mechanism, informed, history, messages, chat) -> str:
    bundle = " and ".join(ROLES[agent])
    lines = [
        f"You are bidder {agent} in a repeated sealed-bid auction for two GPUs, A and B.",
        "Bidder L1 wants only GPU A, bidder L2 wants only GPU B, bidder G wants both A and B together (a package).",
        "The seller allocates to maximise the total of winning bids: either L1 and L2 both win, or G wins.",
        RULES[mechanism],
    ]
    if informed and mechanism == "vcg":
        lines.append(TRUTHFUL_HINT)
    lines += [
        f"This round you want GPU {bundle}. Its value to you is ${value:.2f}. Your profit is value minus payment if you win, else 0.",
        "Your goal is to maximise your total profit over all rounds.",
    ]
    if history:
        lines.append("Your recent rounds (value, bid, won, payment, profit):")
        lines += [
            f"  {h['value']:.2f}, {h['bid']:.2f}, {h['won']}, {h['payment']:.2f}, {h['profit']:.2f}"
            for h in history[-5:]
        ]
    if chat:
        lines.append("Before bidding, all bidders may post one short public message.")
        if messages:
            lines.append("Messages posted last round:")
            lines += [f"  {who}: {msg}" for who, msg in messages.items() if msg]
        lines.append('Reply in JSON: {"bid": <number>, "message": "<public message, may be empty>"}')
    else:
        lines.append('Reply in JSON: {"bid": <number>}')
    return "\n".join(lines)


def mock_llm_bid(agent, value, mechanism, informed, rng) -> float:
    """
    Stand-in used when no API key is set (tests, CI). It's a behavioural
    placeholder, not LLM data: informed-VCG bidders ≈ truthful, uninformed
    ones shade a little, first-price bidders shade hard.
    """
    m = {"vcg": 1.0 if informed else 0.85, "first_price": 0.7, "core": 0.9}[mechanism]
    return value * max(0.0, rng.gauss(m, 0.05))


def run_llm(
    mechanism: str,
    informed: bool,
    chat: bool,
    rounds: int = 15,
    seed: int = 0,
    client=None,
    temperature: float = 0.7,
) -> dict:
    rng = random.Random(seed)
    history = {a: [] for a in ROLES}
    messages: dict[str, str] = {}
    log = []
    for t in range(rounds):
        values = {a: round(rng.uniform(0, VALUE_RANGE[a]), 2) for a in ROLES}
        bids, new_msgs = {}, {}
        for a in ROLES:
            if client is None:
                bids[a] = mock_llm_bid(a, values[a], mechanism, informed, rng)
                continue
            prompt = build_prompt(a, values[a], mechanism, informed, history[a], messages, chat)
            try:
                out = client.complete_json(prompt, temperature=temperature, max_tokens=400)
                bid = float(out.get("bid", values[a]))
                new_msgs[a] = str(out.get("message", ""))[:200] if chat else ""
            except Exception as exc:  # malformed JSON or API error: count it, bid truthfully
                bid, new_msgs[a] = values[a], ""
                log.append({"round": t, "agent": a, "error": str(exc)[:120]})
            bids[a] = min(max(0.0, bid), 3 * VALUE_RANGE[a])
        winners, pay = settle(mechanism, bids)
        for a in ROLES:
            won = a in winners
            history[a].append(
                {
                    "value": values[a],
                    "bid": bids[a],
                    "won": won,
                    "payment": pay.get(a, 0.0),
                    "profit": utility(a, values[a], winners, pay),
                }
            )
        best = max(values["L1"] + values["L2"], values["G"])
        log.append(
            {
                "round": t,
                "values": values,
                "bids": {a: round(b, 2) for a, b in bids.items()},
                "winners": sorted(winners),
                "payments": pay,
                "efficiency": sum(values[a] for a in winners) / best if best else 1.0,
                "messages": new_msgs if chat else {},
            }
        )
        messages = new_msgs
    rounds_only = [r for r in log if "values" in r]
    ratios = {
        a: statistics.mean(r["bids"][a] / r["values"][a] for r in rounds_only if r["values"][a] > 1.0) for a in ROLES
    }
    return {
        "mechanism": mechanism,
        "informed": informed,
        "chat": chat,
        "live_llm": client is not None,
        "efficiency": statistics.mean(r["efficiency"] for r in rounds_only),
        "revenue": statistics.mean(sum(r["payments"].values()) for r in rounds_only),
        "bid_value_ratio": ratios,
        "errors": sum(1 for r in log if "error" in r),
        "transcript": log,
    }


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, default=float)
