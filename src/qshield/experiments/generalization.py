"""Does the plan survive being wrong about the weights?

The 0.3 README states plainly that "the quantum factors and objective weights
are hypotheses" awaiting calibration. If that is true, the operative question is
not how a plan scores under the weights used to choose it — it is how the plan
scores under the weights that turn out to be *right*.

This experiment makes that explicit. Each arm plans once under ``train_weights``,
then every plan is scored under a grid of ``eval_weights`` covering the simplex.
The training weights are one point in that grid; the rest are held out. An arm
whose advantage exists only near its training point is fitting the analyst's
prior, not the system.

This is the sharpest falsification test in the suite, because unlike the
in-objective comparison there is no structural guarantee anywhere: under
mismatched weights the exhaustive planner is minimising the wrong function and
can, and does, lose to a heuristic.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from itertools import product

from ..model import DEFAULT_WEIGHTS, NODE_ONLY_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem, Plan, exhaustive, greedy_flat
from ..uncertainty import bootstrap_ci, paired_comparison, summarize
from .generator import GeneratorConfig, make_instances

ARMS = ("flat_priority", "node_only", "path_aware")


def weight_grid(
    node: Sequence[float] = (0.2, 0.4, 0.6),
    path: Sequence[float] = (0.2, 0.4, 0.6),
    rotation: Sequence[float] = (0.1, 0.3, 0.5),
) -> list[ObjectiveWeights]:
    """Distinct weightings on the normalised simplex.

    The objective divides by the sum of weights, so grid points that differ only
    by a scale factor are the same weighting. 0.3's 3x3x3 sweep contained such a
    duplicate — ``(0.2, 0.2, 0.1)`` and ``(0.6, 0.6, 0.3)`` are one point — and
    reported it as two independent settings. De-duplicating on ``key()`` keeps
    the sweep honest about how much of the simplex it actually covers.
    """
    seen = set()
    out: list[ObjectiveWeights] = []
    for n, p, r in product(node, path, rotation):
        w = ObjectiveWeights(n, p, r)
        if w.key() in seen:
            continue
        seen.add(w.key())
        out.append(w)
    return out


def _plan_arms(problem: MigrationProblem, train: ObjectiveWeights) -> dict[str, Plan]:
    path_aware, _ = exhaustive(problem, weights=train, strategy="path_aware")
    node_only, _ = exhaustive(problem, weights=NODE_ONLY_WEIGHTS, strategy="node_only")
    return {
        "flat_priority": greedy_flat(problem, weights=train),
        "node_only": node_only,
        "path_aware": path_aware,
    }


def run(
    *,
    instances: int = 300,
    seed: int = 20260907,
    train_weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    eval_weights: Iterable[ObjectiveWeights] | None = None,
    config: GeneratorConfig = GeneratorConfig(),
) -> dict[str, object]:
    grid = list(eval_weights) if eval_weights is not None else weight_grid()
    problems = [p for p in make_instances(instances, seed, config) if p.candidates]

    # instance -> arm -> plan, chosen once under the training weights
    plans = [_plan_arms(p, train_weights) for p in problems]

    # (eval weighting) -> arm -> per-instance objective
    per_weighting: list[dict[str, object]] = []
    # Pooled across the whole grid, for the headline number.
    pooled: dict[tuple[str, str], list[float]] = {}

    for w in grid:
        scores = {
            arm: [
                problem.evaluate(plan[arm].selected, weights=w).value
                for problem, plan in zip(problems, plans, strict=True)
            ]
            for arm in ARMS
        }
        entry: dict[str, object] = {
            "eval_weights": w.as_tuple(),
            "normalized": w.key(),
            "is_training_point": w.key() == train_weights.key(),
            "mean_objective": {arm: round(statistics.fmean(scores[arm]), 6) for arm in ARMS},
        }
        for a, b in (("node_only", "path_aware"), ("flat_priority", "path_aware")):
            deltas = [x - y for x, y in zip(scores[a], scores[b], strict=True)]
            entry[f"{a}_vs_{b}"] = paired_comparison(deltas, label_a=a, label_b=b)
            pooled.setdefault((a, b), []).extend(deltas)
        per_weighting.append(entry)

    return {
        "experiment": "generalization",
        "question": (
            "Do plans chosen under the default objective weights remain better "
            "when the weights are misspecified?"
        ),
        "design": {
            "instances": len(problems),
            "train_weights": train_weights.as_tuple(),
            "eval_weightings": len(grid),
            "grid_deduplicated_on_simplex": True,
            "structural_guarantee": (
                "none: under eval weights different from the training weights, the "
                "exhaustive planner minimises a different function than it is scored "
                "on, so every arm can lose"
            ),
            "generator": config.as_dict(),
            "seed": seed,
        },
        "pooled": {
            f"{a}_vs_{b}": {
                **paired_comparison(deltas, label_a=a, label_b=b),
                "mean_delta_ci95_pooled": [
                    round(v, 6) for v in bootstrap_ci(deltas, seed=seed)
                ],
            }
            for (a, b), deltas in pooled.items()
        },
        "robustness": _robustness(per_weighting),
        "per_weighting": per_weighting,
    }


def _robustness(per_weighting: Sequence[dict[str, object]]) -> dict[str, object]:
    """Across how much of the weight simplex does each advantage hold?

    A method that helps at its training point and nowhere else is not robust,
    however large the effect at that point.
    """
    out: dict[str, object] = {}
    for pair in ("node_only_vs_path_aware", "flat_priority_vs_path_aware"):
        favourable = 0
        adverse = 0
        means: list[float] = []
        for entry in per_weighting:
            comparison = entry[pair]
            mean = float(comparison["mean_delta"])
            means.append(mean)
            lo, hi = comparison["mean_delta_ci95"]
            if lo > 0:
                favourable += 1   # path_aware better, CI excludes zero
            elif hi < 0:
                adverse += 1      # path_aware worse, CI excludes zero
        out[pair] = {
            "weightings": len(per_weighting),
            "path_aware_better_ci_excludes_zero": favourable,
            "path_aware_worse_ci_excludes_zero": adverse,
            "inconclusive": len(per_weighting) - favourable - adverse,
            "fraction_favourable": round(favourable / max(1, len(per_weighting)), 6),
            "mean_delta_across_weightings": summarize(means),
        }
    return out
