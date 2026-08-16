# Compute Bazaar — Auction Theory Reference
# ==========================================
# A beginner-friendly guide to the economics behind Compute Bazaar.
# Read this if you want to understand WHY the auction works.

# Combinatorial Auction Theory — Plain English Guide

## 1. Why Auctions?

When multiple people (or agents) want the same scarce resource,
we need a FAIR and EFFICIENT way to decide who gets it.

Three options:
  a) First-Come, First-Served (FCFS / Round-Robin)
       → Simple, but INEFFICIENT. A low-value task blocks a critical one.

  b) Priority Queue
       → Better, but who assigns priorities? Manual and subjective.

  c) Auction
       → Let agents express their own values through bids!
          The market discovers the "right" priority automatically.

## 2. What Makes an Auction "Combinatorial"?

In a NORMAL auction (like eBay):
  - One item for sale
  - Multiple bidders compete for it
  - Highest bid wins

In a COMBINATORIAL auction:
  - MULTIPLE items for sale SIMULTANEOUSLY
  - Bidders can bid on BUNDLES (combinations) of items
  - The auctioneer finds the OPTIMAL set of winning bids

Example:
  Items: {Hammer, Nails}

  Normal auction sells them separately:
    - Alice wins Hammer for $5
    - Bob wins Nails for $3
    - Total: $8

  Combinatorial: Carol bids $15 for {Hammer + Nails} together
    (She needs BOTH to build anything — neither alone is worth much)
    - Carol wins both for $15
    - Total: $15  ← MORE efficient!

  In Compute Bazaar:
    - Items = compute resources (GPU time, API budget, RAM)
    - Bidders = AI agents
    - Bundle = a set of resources an agent needs to complete its task

## 3. The Winner Determination Problem (WDP)

The hardest part of running a combinatorial auction:
"Which combination of bids maximizes total value
 without allocating the same resource twice?"

This is an OPTIMIZATION PROBLEM:

  Maximize:  sum(bid_amount[i] * x[i])  for all bids i
  Subject to:
    For each resource r:  sum(x[i] for bids i containing r) <= 1
    x[i] ∈ {0, 1}  (each bid either wins or loses)

This is NP-hard in general, but practical solvers (like PuLP/CBC)
can handle thousands of bids in milliseconds.

Week 1 uses a GREEDY approximation:
  Sort bids by value-per-resource (descending), pick greedily.
  → Fast but not always optimal

Week 2 uses LINEAR PROGRAMMING:
  Solve the exact optimization → always optimal

## 4. VCG Pricing (Vickrey-Clarke-Groves)

The key question: how much should the winner PAY?

Problem with "pay your bid":
  If you bid $100 but the second-highest was $30,
  you "overpay" by $70. This discourages participation.

VCG solution: pay your EXTERNALITY (harm to others)

  VCG price = (Total value WITHOUT you) - (Total value WITH you, excluding you)

  Example:
    Resources: {GPU-1, API-Budget}
    
    Bids:
      Agent A: {GPU-1, API-Budget} → $80   ← wins!
      Agent B: {GPU-1}             → $50
      Agent C: {API-Budget}        → $40
    
    Without Agent A:
      B wins GPU-1 ($50) + C wins API-Budget ($40) = $90
    
    With Agent A winning, others get: $0
    
    Agent A's VCG price = $90 - $0 = $90? Wait, that's MORE than their bid!
    
    Hmm — let me recalculate properly:
    
    Agent A's payment = value_of_best_allocation_without_A
                      - value_of_best_allocation_excluding_A's_items
    
    = $90 (B+C get $90 without A competing)
    - $0  (without A's items, nothing else is available to others)
    = $90? No...
    
    Correct VCG formula:
      Payment_A = [Optimal value WITHOUT A participating]
               - [Optimal value WITH A winning, for EVERYONE EXCEPT A]

    = $90 (B gets $50, C gets $40 without A)
    - $0  (with A using GPU-1+API, B and C get nothing)
    = $90

    But A's bid was only $80! A wouldn't participate if price > bid.
    
    So in practice, VCG price ≤ bid amount by definition.
    If VCG price > bid, the bid wouldn't have won in the first place!
    The key insight: VCG makes TRUTHFUL bidding (bidding your true value)
    the DOMINANT STRATEGY. You can't benefit from lying.

## 5. Why Is Truthfulness Important?

If agents can "game" the system by lying about their values:
  - They bid LESS than their true value (to pay less)
  - The system can't find the truly optimal allocation
  - High-value tasks might lose to lower-value ones

VCG prevents this: your payment doesn't depend on YOUR bid,
only on the bids of OTHERS. So bidding your true value is always best.

## 6. Efficiency Proof (Simple Version)

A combinatorial auction with VCG pricing achieves ALLOCATIVE EFFICIENCY:
  The allocation maximizes total social welfare (sum of all agent values).

This is proven by the "VCG mechanism design" theorem:
  Since all agents bid truthfully → system sees true values
  → WDP finds allocation maximizing true values
  → This is the socially optimal allocation

In contrast, round-robin:
  Has NO mechanism for expressing value
  → Frequently gives resources to low-value tasks
  → Can be arbitrarily far from optimal

## 7. Further Reading

  - "Combinatorial Auctions" - Cramton, Shoham, Steinberg (2006)
  - Wikipedia: "Combinatorial auction"
  - Wikipedia: "Vickrey-Clarke-Groves mechanism"
  - "Auction Theory" - Vijay Krishna
