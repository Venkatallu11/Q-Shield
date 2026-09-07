"""Can a tail-aware path term repair the worst-case regression?

The 0.4 ablation produced a clear negative result: minimising the mean of the
path risks made *worst-case* path exposure significantly worse than a graph-blind
planner (mean -0.736, 95% CI [-0.971, -0.516], losing 86 of 125 decisive
instances). The mechanism is not subtle — the cheapest way to lower an average is
to improve the many moderate paths and leave the worst one standing.

This experiment asks whether replacing the mean with a conditional value at risk
over the worst ``alpha`` fraction of paths fixes it, and what it costs. Three
outcomes are possible and all are informative:

* some ``alpha`` beats the graph-blind control on **both** worst-case and mean
  path risk — the mean was simply the wrong aggregate;
* worst-case improves only by giving up mean performance — a real trade-off, and
  the frontier is the deliverable rather than a single recommended value;
* no ``alpha`` helps — the regression is not about the aggregate at all, and the
  path term itself is suspect.

Falsifiable throughout: every arm is scored on ``path_max`` and ``path_mean``,
and an arm optimising CVaR at one ``alpha`` has no guarantee on either.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..model import DEFAULT_WEIGHTS, NODE_ONLY_WEIGHTS, ObjectiveWeights
from ..optimizer import MigrationProblem, exhaustive
from ..uncertainty import paired_comparison, summarize
from .generator import GeneratorConfig, make_instances

# 1.0 reproduces the 0.4 mean-based objective and anchors the sweep.
DEFAULT_ALPHAS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.05)

# Scored on every arm; none of them minimises these directly.
METRICS = ("path_max", "path_mean", "node_mean")


def run(
    *,
    instances: int = 300,
    seed: int = 20260907,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    base_weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    config: GeneratorConfig = GeneratorConfig(),
) -> dict[str, object]:
    problems = [p for p in make_instances(instances, seed, config) if p.candidates]

    # Control: exhaustive search, no path term at all. Same search power, so a
    # difference against it is attributable to the path term and its aggregate.
    control = [exhaustive(p, weights=NODE_ONLY_WEIGHTS)[0].selected for p in problems]
    control_metrics = _score(problems, control)

    rows: list[dict[str, object]] = []
    for alpha in alphas:
        weights = ObjectiveWeights(
            base_weights.node, base_weights.path, base_weights.rotation, alpha
        )
        plans = [exhaustive(p, weights=weights)[0].selected for p in problems]
        metrics = _score(problems, plans)

        entry: dict[str, object] = {
            "path_alpha": alpha,
            "is_0_4_behaviour": alpha >= 1.0,
            "mean_worst_paths_averaged": _paths_averaged(problems, alpha),
            "plan_differs_from_control": round(
                sum(1 for a, b in zip(plans, control, strict=True) if set(a) != set(b))
                / len(plans),
                6,
            ),
            "mean_cost": summarize([p.cost_of(plan) for p, plan in
                                    zip(problems, plans, strict=True)])["mean"],
        }
        for metric in METRICS:
            deltas = [c - t for c, t in zip(control_metrics[metric], metrics[metric],
                                            strict=True)]
            entry[metric] = paired_comparison(
                deltas, label_a="node_only", label_b=f"cvar_{alpha}"
            )
        rows.append(entry)

    return {
        "experiment": "tail_risk",
        "question": (
            "Does aggregating path risk over the worst alpha-fraction of paths, "
            "instead of over all of them, repair the worst-case regression the "
            "0.4 ablation measured?"
        ),
        "design": {
            "instances": len(problems),
            "alphas": list(alphas),
            "control": "exhaustive search minimising node risk only (path-blind)",
            "held_out_metrics": list(METRICS),
            "structural_guarantee": (
                "none on these metrics: an arm minimising CVaR at one alpha is not "
                "minimising path_max, path_mean or node_mean, so it can lose on any"
            ),
            "base_weights": base_weights.as_tuple(),
            "seed": seed,
            "generator": config.as_dict(),
        },
        "sweep": rows,
        "verdict": _verdict(rows),
    }


def _score(
    problems: Sequence[MigrationProblem], plans: Sequence[Sequence[str]]
) -> dict[str, list[float]]:
    """Score each plan on every held-out metric.

    Metrics are read off the objective result rather than recomputed, and the
    weighting used here is irrelevant to them: ``path_max``, ``path_mean`` and
    ``node_mean`` are components, not the weighted value.
    """
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for problem, plan in zip(problems, plans, strict=True):
        result = problem.evaluate(plan)
        for metric in METRICS:
            out[metric].append(getattr(result, metric))
    return out


def _paths_averaged(problems: Sequence[MigrationProblem], alpha: float) -> float:
    """How many paths the aggregate actually averages, on average.

    With few paths per instance, small alphas all collapse to "the single worst
    path" and the sweep stops distinguishing them; this makes that visible rather
    than leaving it to be inferred from flat results.
    """
    import math

    counts = [
        max(1, math.ceil(alpha * len(p.path_set))) if len(p.path_set) else 0
        for p in problems
    ]
    return round(sum(counts) / len(counts), 4) if counts else 0.0


def _verdict(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Which alphas beat the path-blind control on worst-case *and* mean.

    A confidence interval that excludes zero in the favourable direction is the
    bar; overlapping zero counts as no evidence, not as a small win.
    """
    both: list[float] = []
    worst_only: list[float] = []
    neither: list[float] = []
    for row in rows:
        worst_lo = row["path_max"]["mean_delta_ci95"][0]
        mean_lo = row["path_mean"]["mean_delta_ci95"][0]
        alpha = row["path_alpha"]
        if worst_lo > 0 and mean_lo > 0:
            both.append(alpha)
        elif worst_lo > 0:
            worst_only.append(alpha)
        elif worst_lo >= 0 or mean_lo > 0:
            neither.append(alpha)
        else:
            neither.append(alpha)
    return {
        "alphas_beating_control_on_worst_case_and_mean": both,
        "alphas_beating_control_on_worst_case_only": worst_only,
        "alphas_not_beating_control_on_worst_case": neither,
        "criterion": "95% bootstrap CI on the paired mean difference excludes zero",
    }
