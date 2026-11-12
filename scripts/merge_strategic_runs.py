"""
Merge per-model strategic-study runs into results/strategic_study.json.

Live LLM runs are split by model (each Groq model has its own rate limits),
so they run as separate processes and are merged here. A run can be marked
partial: Groq's daily token cap for openai/gpt-oss-120b was hit mid-study,
so only its first two conditions have data, and the second is flagged.

    python scripts/merge_strategic_runs.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def load(name: str) -> dict:
    p = R / name
    return json.loads(p.read_text()) if p.exists() else {"learning": {}, "llm": {}}


def main() -> None:
    p1 = load("_llm_gptoss_part1.json")
    qwen = load("_llm_qwen.json")
    small = load("_llm_gptoss20b.json")

    big = p1["llm"].get("openai/gpt-oss-120b", {})
    big_kept = {}
    if "vcg_informed" in big:
        big_kept["vcg_informed"] = big["vcg_informed"]
    if "vcg_uninformed" in big:
        r = dict(big["vcg_uninformed"])
        r["partial_reason"] = (
            f"{r['errors']} of 45 bids hit Groq's per-minute token limit; ratios use the remaining bids"
        )
        big_kept["vcg_uninformed"] = r
    big_kept["_note"] = {
        "omitted": ["first_price", "vcg_chat", "core_chat"],
        "reason": "Groq daily token cap (200k tokens/day) for openai/gpt-oss-120b was reached",
    }

    out = {
        "learning": p1["learning"],
        "llm": {"openai/gpt-oss-120b": big_kept, **qwen["llm"], **small["llm"]},
        "notes": {
            "qwen/qwen3.8-27b": "15 rounds per condition",
            "openai/gpt-oss-20b": "10 rounds per condition (to fit its daily token cap)",
            "openai/gpt-oss-120b": "15 rounds per condition; partial, see _note",
            "llama-3.3-70b-versatile": "planned, but retired by Groq (404) at run time",
        },
    }
    (R / "strategic_study.json").write_text(json.dumps(out, indent=2, default=float))
    for model, conds in out["llm"].items():
        print(model, [c for c in conds if not c.startswith("_")])


if __name__ == "__main__":
    main()
