"""
Merge per-model strategic-study runs into results/strategic_study.json.

Live LLM runs go one model per process (each Groq model has its own rate
limits) and are saved raw in results/llm_runs/. The free tier also has a
daily token cap that two models reached mid-study, so conditions are
classified automatically:
  * every LLM call failed           -> omitted
  * some calls failed after retries -> kept, flagged partial (failed bids are
                                       already excluded from the ratios)

    python scripts/merge_strategic_runs.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
RUNS = R / "llm_runs"


def main() -> None:
    learning = {}
    llm: dict = {}
    for f in sorted(RUNS.glob("*.json")):
        data = json.loads(f.read_text())
        learning = learning or data.get("learning", {})
        for model, conds in data["llm"].items():
            kept, omitted = {}, []
            for cond, r in conds.items():
                if cond.startswith("_"):
                    continue
                calls = len(r["transcript"]) and 3 * sum(1 for t in r["transcript"] if "values" in t)
                if r["errors"] >= calls:
                    omitted.append(cond)
                    continue
                if r["errors"]:
                    r = {**r, "partial_reason": f"{r['errors']} of {calls} LLM calls failed (Groq token limits)"}
                kept[cond] = r
            if omitted:
                kept["_omitted"] = {"conditions": omitted, "reason": "Groq daily token cap reached; every call failed"}
            llm[model] = kept
    out = {"learning": learning, "llm": llm, "raw_runs": "results/llm_runs/"}
    (R / "strategic_study.json").write_text(json.dumps(out, indent=2, default=float))
    for model, conds in llm.items():
        print(
            model,
            {c: ("partial" if r.get("partial_reason") else "ok") for c, r in conds.items() if not c.startswith("_")},
        )


if __name__ == "__main__":
    main()
