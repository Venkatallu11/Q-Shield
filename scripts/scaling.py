"""Where the exact planner stops being usable.

Produces the table quoted in `qshield/planner.py`, so the threshold that decides
between the exact and greedy planners is a measurement rather than a claim in a
docstring. Budgets are widened so the feasible set approaches the full power set,
which is the worst case the planner has to survive.
"""

from __future__ import annotations

import json
import random
import sys
import time

from qshield.experiments.generator import GeneratorConfig, make_instance
from qshield.optimizer import MigrationProblem, exhaustive, greedy_marginal
from qshield.planner import MAX_EXACT_CANDIDATES

NODE_COUNTS = (6, 8, 10, 12, 14, 16, 18, 20)
SEED = 7


def main() -> int:
    rows = []
    for nodes in NODE_COUNTS:
        base = make_instance(random.Random(SEED), GeneratorConfig(nodes=nodes,
                                                                 cross_links=nodes))
        problem = MigrationProblem(
            base.assets, base.edges, base.entrypoints, base.targets,
            base.replacements, budget=1e6,
        )
        feasible = len(problem.feasible_plans())

        start = time.perf_counter()
        exhaustive(problem)
        exact_seconds = time.perf_counter() - start

        start = time.perf_counter()
        greedy_marginal(problem)
        greedy_seconds = time.perf_counter() - start

        rows.append({
            "nodes": nodes,
            "migration_candidates": len(problem.candidates),
            "attack_paths": len(problem.path_set),
            "feasible_plans": feasible,
            "exhaustive_seconds": round(exact_seconds, 4),
            "greedy_seconds": round(greedy_seconds, 4),
            "speedup": round(exact_seconds / greedy_seconds, 1) if greedy_seconds else None,
        })
        print(
            f"{nodes:>3} nodes  {len(problem.candidates):>3} candidates  "
            f"{feasible:>7} plans  exact {exact_seconds:7.3f}s  "
            f"greedy {greedy_seconds:6.3f}s",
            file=sys.stderr,
        )

    report = {
        "experiment": "planner_scaling",
        "question": "At what size does exhaustive planning stop finishing?",
        "design": {"seed": SEED, "budget": "unbounded, so the feasible set is the "
                                            "full power set", "node_counts": list(NODE_COUNTS)},
        "threshold_in_use": MAX_EXACT_CANDIDATES,
        "reproducibility": (
            "the plan counts and candidate counts are deterministic; the timings "
            "are wall-clock and vary with the machine, so this is the one report "
            "in results/ that does not reproduce byte-identically"
        ),
        "rows": rows,
        "reading": (
            "Each additional migration candidate roughly doubles the exact "
            "planner's cost while the greedy planner is flat. The threshold in "
            "qshield/planner.py is set from this table; above it the planner "
            "downgrades and says so."
        ),
    }
    print(json.dumps(report, indent=2))
    with open("results/planner_scaling.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
