"""
Regenerate every result in the repository with one command.

    python scripts/reproduce.py            # full runs, overwrite results/ (≈1-2 h on a laptop)
    python scripts/reproduce.py --quick    # small versions into results/quick/ (≈5 min), a smoke test

Live LLM runs are only included with --live-llm (they need GROQ_API_KEY and
cost API credits); otherwise the strategic study uses the labelled mock bidders.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def steps(quick: bool, live_llm: bool) -> list[tuple[str, list[str]]]:
    out = "results/quick" if quick else "results"
    llm = "live" if live_llm else "mock"
    if quick:
        return [
            (
                "mechanism verification",
                ["scripts/verify_mechanism.py", "--instances", "300", "--out", f"{out}/mechanism_verification.json"],
            ),
            ("predictor evaluation", ["scripts/evaluate_predictor.py"]),
            (
                "trace experiments",
                [
                    "scripts/run_trace_experiments.py",
                    "--seeds",
                    "1",
                    "--jobs",
                    "300",
                    "--users",
                    "40",
                    "--out",
                    f"{out}/trace_experiments.json",
                ],
            ),
            (
                "uncertainty experiments",
                [
                    "scripts/run_uncertainty_experiments.py",
                    "--seeds",
                    "1",
                    "--rounds",
                    "30",
                    "--out",
                    f"{out}/uncertainty_experiments.json",
                ],
            ),
            (
                "strategic study",
                [
                    "scripts/run_strategic_study.py",
                    "--llm",
                    llm,
                    "--learning-seeds",
                    "1",
                    "--learning-rounds",
                    "500",
                    "--llm-rounds",
                    "3",
                    "--out",
                    f"{out}/strategic_study.json",
                ],
            ),
            (
                "scaling benchmark",
                [
                    "scripts/benchmark_scaling.py",
                    "--sizes",
                    "25,50",
                    "--repeats",
                    "1",
                    "--configs",
                    "exact-8",
                    "--out",
                    f"{out}/scaling_benchmark.json",
                ],
            ),
        ]
    return [
        ("multi-seed micro-benchmark", ["benchmark.py"]),
        ("mechanism verification", ["scripts/verify_mechanism.py", "--instances", "10000"]),
        ("predictor evaluation", ["scripts/evaluate_predictor.py"]),
        ("trace experiments", ["scripts/run_trace_experiments.py", "--seeds", "5"]),
        (
            "fairness frontier",
            [
                "scripts/run_trace_experiments.py",
                "--seeds",
                "3",
                "--policies",
                "greedy_value,themis,bazaar,bazaar_fair_0.5,bazaar_fair_1,bazaar_fair_2",
                "--out",
                "results/fairness_frontier.json",
            ],
        ),
        ("uncertainty experiments", ["scripts/run_uncertainty_experiments.py", "--seeds", "3", "--rounds", "200"]),
        ("strategic study", ["scripts/run_strategic_study.py", "--llm", llm]),
        ("scaling benchmark", ["scripts/benchmark_scaling.py"]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--live-llm", action="store_true")
    args = ap.parse_args()
    (ROOT / ("results/quick" if args.quick else "results")).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for name, cmd in steps(args.quick, args.live_llm):
        print(f"\n=== {name} ===", flush=True)
        t = time.perf_counter()
        subprocess.run([PY, *cmd], cwd=ROOT, check=True)
        print(f"--- {name}: {time.perf_counter() - t:.0f}s", flush=True)
    print(f"\nAll done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
