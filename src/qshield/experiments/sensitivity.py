"""How much does the recommended plan depend on the objective weights?

If a small change in an uncalibrated weight reorders the recommendation, the
recommendation is a property of the analyst's prior rather than of the system,
and no amount of Monte Carlo on the *other* parameters will reveal that.

0.3's ``weight_sensitivity`` swept a 3x3x3 grid and printed the number of
distinct plans. Two problems: the grid contained a scale-duplicate (the
objective normalises by the weight sum, so ``(0.2,0.2,0.1)`` and
``(0.6,0.6,0.3)`` are one weighting), and a plan count alone does not say whether
the variation is a genuine reordering or two plans that score within rounding of
each other. This version de-duplicates on the simplex and reports, for each
weighting, how much worse the *globally most common* plan would have been —
the practical question, since an analyst has to commit to one plan.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from ..model import DEFAULT_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem, exhaustive
from ..uncertainty import summarize
from .generalization import weight_grid


def run(
    problem: MigrationProblem,
    *,
    eval_weights: Iterable[ObjectiveWeights] | None = None,
    reference: ObjectiveWeights = DEFAULT_WEIGHTS,
) -> dict[str, object]:
    grid = list(eval_weights) if eval_weights is not None else weight_grid()
    rows: list[dict[str, object]] = []
    plans: list[tuple[str, ...]] = []

    for w in grid:
        best, _ = exhaustive(problem, weights=w)
        plans.append(best.selected)
        rows.append(
            {
                "weights": w.as_tuple(),
                "normalized": w.key(),
                "selected": list(best.selected),
                "cost": round(best.cost, 6),
                "objective": round(best.objective, 6),
            }
        )

    counts = Counter(plans)
    modal_plan, modal_count = counts.most_common(1)[0]
    reference_plan, _ = exhaustive(problem, weights=reference)

    # Regret of committing to one plan across the whole grid.
    regrets: list[float] = []
    reference_regrets: list[float] = []
    for w, row in zip(grid, rows, strict=True):
        optimal = float(row["objective"])
        regrets.append(problem.evaluate(modal_plan, weights=w).value - optimal)
        reference_regrets.append(
            problem.evaluate(reference_plan.selected, weights=w).value - optimal
        )

    return {
        "experiment": "weight_sensitivity",
        "question": "Is the recommended plan stable under the uncalibrated weights?",
        "design": {
            "weightings_evaluated": len(grid),
            "grid_deduplicated_on_simplex": True,
            "reference_weights": reference.as_tuple(),
        },
        "distinct_plans": len(counts),
        "modal_plan": {
            "selected": list(modal_plan),
            "chosen_under": modal_count,
            "share": round(modal_count / len(grid), 6),
            "regret_vs_per_weighting_optimum": summarize(regrets),
        },
        "reference_plan": {
            "selected": list(reference_plan.selected),
            "regret_vs_per_weighting_optimum": summarize(reference_regrets),
            "max_regret": round(max(reference_regrets), 6) if reference_regrets else 0.0,
        },
        "plan_frequency": [
            {"selected": list(plan), "weightings": n} for plan, n in counts.most_common()
        ],
        "per_weighting": rows,
    }
