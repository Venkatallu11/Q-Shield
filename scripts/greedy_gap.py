"""How much is the exponential exact planner actually worth?

`exhaustive` is O(2^n) in migration candidates. If a tractable greedy recovers
the same plan almost always, the exact planner is a reference implementation
rather than something to run on a real inventory. Reported in
docs/FINDINGS.md section 6.
"""

from __future__ import annotations

import json
import statistics
import sys

from qshield.experiments.generator import make_instances
from qshield.optimizer import exhaustive, greedy_marginal
from qshield.uncertainty import summarize

INSTANCES = 300
SEED = 20260907


def main() -> int:
    identical = 0
    gaps: list[float] = []
    for problem in make_instances(INSTANCES, SEED):
        if not problem.candidates:
            continue
        best, _ = exhaustive(problem)
        greedy = greedy_marginal(problem)
        if set(best.selected) == set(greedy.selected):
            identical += 1
        gaps.append(greedy.objective - best.objective)

    report = {
        "experiment": "greedy_optimality_gap",
        "question": "Does the tractable greedy planner recover the exact optimum?",
        "design": {"instances": len(gaps), "seed": SEED},
        "identical_plan_chosen": identical,
        "identical_plan_rate": round(identical / len(gaps), 6),
        "optimality_gap": summarize(gaps),
        "max_gap": round(max(gaps), 6),
        "note": (
            "gap is greedy_marginal objective minus the exhaustive optimum, so it "
            "is non-negative by construction; the question is its magnitude"
        ),
    }
    print(json.dumps(report, indent=2))
    with open("results/greedy_gap.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    assert min(gaps) >= -1e-9, "greedy beat the exhaustive optimum: optimizer is wrong"
    print(
        f"\n{identical}/{len(gaps)} identical ({identical / len(gaps):.1%}); "
        f"mean gap {statistics.fmean(gaps):.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
