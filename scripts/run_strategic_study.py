"""
How do adaptive and LLM bidders behave under VCG, first-price and core payments?

Part A (learning): Hedge bidders over bid/value multipliers in repeated LLG
auctions, several seeds per mechanism.
Part B (LLM): LLM bidders through Groq under five conditions:
  vcg_informed     VCG, told truthful bidding is dominant
  vcg_uninformed   VCG, rules only
  first_price      pay-your-bid
  vcg_chat         VCG with a public message channel (collusion opportunity)
  core_chat        core-selecting payments with the same channel

    python scripts/run_strategic_study.py                       # learning + live LLMs if GROQ_API_KEY is set
    python scripts/run_strategic_study.py --llm mock            # behavioural stand-in, no API calls
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402
from loguru import logger  # noqa: E402

from compute_bazaar.strategic.game import run_learning, run_llm  # noqa: E402

CONDITIONS = {
    "vcg_informed": ("vcg", True, False),
    "vcg_uninformed": ("vcg", False, False),
    "first_price": ("first_price", False, False),
    "vcg_chat": ("vcg", False, True),
    "core_chat": ("core", False, True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learning-seeds", type=int, default=5)
    ap.add_argument("--learning-rounds", type=int, default=3000)
    ap.add_argument("--llm", choices=["live", "mock", "skip"], default="live")
    ap.add_argument("--models", default="openai/gpt-oss-120b,qwen/qwen3.8-27b")
    ap.add_argument("--llm-rounds", type=int, default=15)
    ap.add_argument("--out", default="results/strategic_study.json")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--skip-learning", action="store_true", help="only run the LLM part (e.g. one model per process)")
    args = ap.parse_args()
    logger.remove()
    load_dotenv()

    out: dict = {"learning": {}, "llm": {}}

    def save() -> None:  # after every condition, so a long live run is never lost
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))

    for mech in () if args.skip_learning else ("vcg", "first_price", "core"):
        rs = [run_learning(mech, rounds=args.learning_rounds, seed=s) for s in range(args.learning_seeds)]
        out["learning"][mech] = {
            "efficiency": statistics.mean(r["efficiency"] for r in rs),
            "revenue": statistics.mean(r["revenue"] for r in rs),
            "bid_value_ratio": {a: statistics.mean(r["bid_value_ratio"][a] for r in rs) for a in ("L1", "L2", "G")},
            "seeds": args.learning_seeds,
            "rounds": args.learning_rounds,
        }
        print(f"learning {mech}: {out['learning'][mech]}", flush=True)

    if args.llm != "skip":
        models = args.models.split(",") if args.llm == "live" else ["mock"]
        for model in models:
            client = None
            if args.llm == "live":
                from compute_bazaar.llm import GroqLLMClient, LLMConfig

                client = GroqLLMClient(config=LLMConfig(model=model, max_tokens=1200))
            out["llm"][model] = {}
            for cond in args.conditions.split(","):
                mech, informed, chat = CONDITIONS[cond]
                t0 = time.perf_counter()
                res = run_llm(mech, informed, chat, rounds=args.llm_rounds, seed=0, client=client)
                out["llm"][model][cond] = res
                save()
                print(
                    f"{model} {cond}: eff={res['efficiency']:.3f} rev={res['revenue']:.1f} "
                    f"ratios={ {k: round(v, 2) for k, v in res['bid_value_ratio'].items()} } "
                    f"errors={res['errors']} ({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
            if client is not None:
                out["llm"][model]["_usage"] = client.get_usage()

    save()


if __name__ == "__main__":
    main()
